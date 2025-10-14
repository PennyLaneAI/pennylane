# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the QNode class and qnode decorator.
"""

from __future__ import annotations

import copy
import functools
import inspect
import logging
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Literal, get_args

from cachetools import Cache, LRUCache

import pennylane as qml
from pennylane import math, pytrees
from pennylane.exceptions import PennyLaneDeprecationWarning, QuantumFunctionError
from pennylane.logging import debug_logger
from pennylane.math import Interface
from pennylane.measurements import MidMeasureMP, Shots, ShotsLike
from pennylane.queuing import AnnotatedQueue
from pennylane.tape import QuantumScript
from pennylane.transforms.core import TransformDispatcher, TransformProgram
from pennylane.typing import TensorLike

from .execution import execute
from .resolution import SupportedDiffMethods, _validate_jax_version

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if TYPE_CHECKING:
    from typing import TypeAlias

    from pennylane.concurrency.executors import ExecBackends
    from pennylane.devices import Device, LegacyDevice
    from pennylane.transforms.core import TransformContainer
    from pennylane.typing import Result
    from pennylane.workflow.resolution import SupportedDiffMethods

    SupportedDeviceAPIs: TypeAlias = LegacyDevice | Device


def _convert_to_interface(result: Result, interface: Interface) -> Result:
    """
    Recursively convert a result to the given interface.
    """

    if interface == Interface.NUMPY:
        return result

    if isinstance(result, (list, tuple)):
        return type(result)(_convert_to_interface(r, interface) for r in result)

    if isinstance(result, dict):
        return {k: _convert_to_interface(v, interface) for k, v in result.items()}

    return math.asarray(result, like=interface.get_like())


def _make_execution_config(
    circuit: QNode | None,
    diff_method: str | None = None,
    mcm_config: qml.devices.MCMConfig | None = None,
) -> qml.devices.ExecutionConfig:
    circuit_interface = getattr(circuit, "interface", Interface.NUMPY.value)
    execute_kwargs = getattr(circuit, "execute_kwargs", {})
    gradient_kwargs = getattr(circuit, "gradient_kwargs", {})
    grad_on_execution = execute_kwargs.get("grad_on_execution")
    if circuit_interface in {Interface.JAX.value, Interface.JAX_JIT.value}:
        grad_on_execution = False
    elif grad_on_execution == "best":
        grad_on_execution = None

    return qml.devices.ExecutionConfig(
        interface=circuit_interface,
        gradient_keyword_arguments=gradient_kwargs,
        gradient_method=diff_method,
        grad_on_execution=grad_on_execution,
        use_device_jacobian_product=execute_kwargs.get("device_vjp", False),
        mcm_config=mcm_config or qml.devices.MCMConfig(),
    )


def _to_qfunc_output_type(results: Result, qfunc_output, has_partitioned_shots: bool) -> Result:
    if has_partitioned_shots:
        return tuple(_to_qfunc_output_type(r, qfunc_output, False) for r in results)

    qfunc_output_leaves, qfunc_output_structure = pytrees.flatten(
        qfunc_output, is_leaf=lambda obj: isinstance(obj, (qml.measurements.MeasurementProcess))
    )

    # counts results are treated as a leaf
    results_leaves = pytrees.flatten(results, is_leaf=lambda obj: isinstance(obj, dict))[0]

    # patch for transforms that change the number of results like metric_tensor
    if len(results_leaves) != len(qfunc_output_leaves):
        if isinstance(qfunc_output, (Sequence, qml.measurements.MeasurementProcess)):
            return results
        return type(qfunc_output)(results)

    # result spec squeezes out dim for single measurement value
    # we need to add it back in
    if len(qfunc_output_leaves) == 1:
        results = (results,)

    return pytrees.unflatten(results, qfunc_output_structure)


def _validate_mcm_config(
    postselect_mode: Literal["hw-like", "fill-shots"] | None,
    mcm_method: Literal["deferred", "one-shot", "tree-traversal"] | None,
) -> None:
    qml.devices.MCMConfig(postselect_mode=postselect_mode, mcm_method=mcm_method)


def _validate_qfunc_output(qfunc_output, measurements) -> None:
    measurement_processes = pytrees.flatten(
        qfunc_output,
        is_leaf=lambda obj: isinstance(obj, qml.measurements.MeasurementProcess),
    )[0]

    # user provides no measurements or non-measurements
    if len(measurement_processes) == 0:
        measurement_processes = None
    else:
        # patch for tensor measurement objects, e.g., qml.math.hstack <-> [tensor([tensor(...), tensor(...)])]
        if isinstance(measurement_processes[0], Iterable) and any(
            isinstance(m, TensorLike) for m in measurement_processes[0]
        ):
            measurement_processes = [
                m.base.item()
                for m in measurement_processes[0]
                if isinstance(m.base.item(), qml.measurements.MeasurementProcess)
            ]

    if not measurement_processes or not all(
        isinstance(m, qml.measurements.MeasurementProcess) for m in measurement_processes
    ):
        raise QuantumFunctionError(
            "A quantum function must return either a single measurement, "
            "or a nonempty sequence of measurements."
        )

    terminal_measurements = [m for m in measurements if not isinstance(m, MidMeasureMP)]

    if any(
        ret is not m for ret, m in zip(measurement_processes, terminal_measurements, strict=True)
    ):
        raise QuantumFunctionError(
            "All measurements must be returned in the order they are measured."
        )


def _validate_diff_method(
    device: SupportedDeviceAPIs, diff_method: str | TransformDispatcher
) -> None:
    if diff_method is None:
        return

    # performs type validation
    config = _make_execution_config(None, diff_method)

    if device.supports_derivatives(config):
        return
    if diff_method in {"backprop", "adjoint", "device"}:  # device-only derivatives
        raise QuantumFunctionError(
            f"Device {device} does not support {diff_method} with requested circuit."
        )
    if isinstance(diff_method, str) and diff_method in tuple(get_args(SupportedDiffMethods)):
        return
    if isinstance(diff_method, TransformDispatcher):
        return

    raise QuantumFunctionError(
        f"Differentiation method {diff_method} not recognized. Allowed "
        f"options are {tuple(get_args(SupportedDiffMethods))}."
    )


# pylint: disable=too-many-instance-attributes
class QNode:
    r"""Represents a quantum node in the hybrid computational graph.

    A *quantum node* contains a :ref:`quantum function <intro_vcirc_qfunc>` (corresponding to
    a `variational circuit <https://pennylane.ai/qml/glossary/variational_circuit>`__)
    and the computational device it is executed on.

    The QNode calls the quantum function to construct a :class:`~.QuantumTape` instance representing
    the quantum circuit.

    Args:
        func (Callable): a quantum function
        device (~.Device): a PennyLane-compatible device
        interface (str): The interface that will be used for classical backpropagation.
            This affects the types of objects that can be passed to/returned from the QNode. See
            ``qml.math.SUPPORTED_INTERFACE_USER_INPUT`` for a list of all accepted strings.

            * ``"autograd"``: Allows autograd to backpropagate
              through the QNode. The QNode accepts default Python types
              (floats, ints, lists, tuples, dicts) as well as NumPy array arguments,
              and returns NumPy arrays.

            * ``"torch"``: Allows PyTorch to backpropagate
              through the QNode. The QNode accepts and returns Torch tensors.

            * ``"tf"``: Allows TensorFlow in eager mode to backpropagate
              through the QNode. The QNode accepts and returns
              TensorFlow ``tf.Variable`` and ``tf.tensor`` objects.

            * ``"jax"``: Allows JAX to backpropagate
              through the QNode. The QNode accepts and returns
              JAX ``Array`` objects.

            * ``None``: The QNode accepts default Python types
              (floats, ints, lists, tuples, dicts) as well as NumPy array arguments,
              and returns NumPy arrays. It does not connect to any
              machine learning library automatically for backpropagation.

            * ``"auto"``: The QNode automatically detects the interface from the input values of
              the quantum function.

        diff_method (str or .TransformDispatcher): The method of differentiation to use in
            the created QNode. Can either be a :class:`~.TransformDispatcher`, which includes all
            quantum gradient transforms in the :mod:`qml.gradients <.gradients>` module, or a string. The following
            strings are allowed:

            * ``"best"``: Best available method. Uses classical backpropagation or the
              device directly to compute the gradient if supported, otherwise will use
              the analytic parameter-shift rule where possible with finite-difference as a fallback.

            * ``"device"``: Queries the device directly for the gradient.
              Only allowed on devices that provide their own gradient computation.

            * ``"backprop"``: Use classical backpropagation. Only allowed on
              simulator devices that are classically end-to-end differentiable,
              for example :class:`default.qubit <~.DefaultQubit>`. Note that
              the returned QNode can only be used with the machine-learning
              framework supported by the device.

            * ``"adjoint"``: Uses an `adjoint method <https://arxiv.org/abs/2009.02823>`__ that
              reverses through the circuit after a forward pass by iteratively applying the inverse
              (adjoint) gate. Only allowed on supported simulator devices such as
              :class:`default.qubit <~.DefaultQubit>`.

            * ``"parameter-shift"``: Use the analytic parameter-shift
              rule for all supported quantum operation arguments, with finite-difference
              as a fallback.

            * ``"hadamard"``: Use the standard analytic hadamard gradient test rule for
              all supported quantum operation arguments. More info is in the documentation
              for :func:`qml.gradients.hadamard_grad <.gradients.hadamard_grad>`. Reversed,
              direct, and reversed-direct modes can be selected via a ``"mode"`` in ``gradient_kwargs``.

            * ``"finite-diff"``: Uses numerical finite-differences for all quantum operation
              arguments.

            * ``"spsa"``: Uses a simultaneous perturbation of all operation arguments to approximate
              the derivative.

            * ``None``: QNode cannot be differentiated. Works the same as ``interface=None``.

        grad_on_execution (bool, str): Whether the gradients should be computed on the execution or not.
            Only applies if the device is queried for the gradient; gradient transform
            functions available in ``qml.gradients`` are only supported on the backward
            pass. The 'best' option chooses automatically between the two options and is default.
        cache="auto" (str or bool or dict or Cache): Whether to cache evalulations.
            ``"auto"`` indicates to cache only when ``max_diff > 1``. This can result in
            a reduction in quantum evaluations during higher order gradient computations.
            If ``True``, a cache with corresponding ``cachesize`` is created for each batch
            execution. If ``False``, no caching is used. You may also pass your own cache
            to be used; this can be any object that implements the special methods
            ``__getitem__()``, ``__setitem__()``, and ``__delitem__()``, such as a dictionary.
        cachesize (int): The size of any auto-created caches. Only applies when ``cache=True``.
        max_diff (int): If ``diff_method`` is a gradient transform, this option specifies
            the maximum number of derivatives to support. Increasing this value allows
            for higher order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backwards pass.
        device_vjp (bool): Whether or not to use the device-provided Vector Jacobian Product (VJP).
            A value of ``None`` indicates to use it if the device provides it, but use the full jacobian otherwise.
        postselect_mode (str): Configuration for handling shots with mid-circuit measurement postselection. If
            ``"hw-like"``, invalid shots will be discarded and only results for valid shots will be returned.
            If ``"fill-shots"``, results corresponding to the original number of shots will be returned. The
            default is ``None``, in which case the device will automatically choose the best configuration. For
            usage details, please refer to the :doc:`dynamic quantum circuits page </introduction/dynamic_quantum_circuits>`.
        mcm_method (str): The strategy for applying mid-circuit measurements.
            Available methods include ``"deferred"`` (to use the deferred
            measurement principle), ``"one-shot"`` (to execute the circuit
            for each shot separately when using finite shots), and
            ``"tree-traversal"`` (visits the tree of possible MCM sequences,
            only supported on ``default.qubit`` and ``lightning.qubit``).
            If not provided, the device will select the method automatically.
            For usage details, refer to the :doc:`dynamic quantum circuits page </introduction/dynamic_quantum_circuits>`.
        gradient_kwargs (dict): A dictionary of keyword arguments that are passed to the differentiation
            method. Please refer to the :mod:`qml.gradients <.gradients>` module for details
            on supported options for your chosen gradient transform.
        static_argnums (int | Sequence[int]): *Only applicable when the experimental capture mode is enabled.*
            An ``int`` or collection of ``int``\ s that specify which positional arguments to treat as static.
        executor_backend (ExecBackends | str): The backend executor for concurrent function execution. This argument
            allows for selective control of how to run data-parallel/task-based parallel functions via a defined execution
            environment. All supported options can be queried using
            :func:`~qml.concurrency.executors.get_supported_backends.
            The default value is :class:`qml.concurrency.executors.native.MP_PoolExec`.

    **Example**

    QNodes can be created by decorating a quantum function:

    >>> dev = qml.device("default.qubit", wires=1)
    >>> @qml.qnode(dev)
    ... def circuit(x):
    ...     qml.RX(x, wires=0)
    ...     return qml.expval(qml.Z(0))

    or by instantiating the class directly:

    >>> def circuit(x):
    ...     qml.RX(x, wires=0)
    ...     return qml.expval(qml.Z(0))
    >>> dev = qml.device("default.qubit", wires=1)
    >>> qnode = qml.QNode(circuit, dev)

    .. details::
        :title: Parameter broadcasting
        :href: parameter-broadcasting

        QNodes can be executed simultaneously for multiple parameter settings, which is called
        *parameter broadcasting* or *parameter batching*.
        We start with a simple example and briefly look at the scenarios in which broadcasting is
        possible and useful. Finally we give rules and conventions regarding the usage of
        broadcasting, together with some more complex examples.
        Also see the :class:`~.pennylane.operation.Operator` documentation for implementation
        details.

        **Example**

        Again consider the following ``circuit``:

        >>> dev = qml.device("default.qubit", wires=1)
        >>> @qml.qnode(dev)
        ... def circuit(x):
        ...     qml.RX(x, wires=0)
        ...     return qml.expval(qml.Z(0))

        If we want to execute it at multiple values ``x``,
        we may pass those as a one-dimensional array to the QNode:

        >>> x = np.array([np.pi / 6, np.pi * 3 / 4, np.pi * 7 / 6])
        >>> circuit(x)
        tensor([ 0.8660254 , -0.70710678, -0.8660254 ], requires_grad=True)

        The resulting array contains the QNode evaluations at the single values:

        >>> [circuit(x_val) for x_val in x]
        [tensor(0.8660254, requires_grad=True),
         tensor(-0.70710678, requires_grad=True),
         tensor(-0.8660254, requires_grad=True)]

        In addition to the results being stacked into one ``tensor`` already, the broadcasted
        execution actually is performed in one simulation of the quantum circuit, instead of
        three sequential simulations.

        **Benefits & Supported QNodes**

        Parameter broadcasting can be useful to simplify the execution syntax with QNodes. More
        importantly though, the simultaneous execution via broadcasting can be significantly
        faster than iterating over parameters manually. If we compare the execution time for the
        above QNode ``circuit`` between broadcasting and manual iteration for an input size of
        ``100``, we find a speedup factor of about :math:`30`.
        This speedup is a feature of classical simulators, but broadcasting may reduce
        the communication overhead for quantum hardware devices as well.

        A QNode supports broadcasting if all operators that receive broadcasted parameters do so.
        (Operators that are used in the circuit but do not receive broadcasted inputs do not need
        to support it.) A list of supporting operators is available in
        :obj:`~.pennylane.ops.qubit.attributes.supports_broadcasting`.
        Whether or not broadcasting delivers an increased performance will depend on whether the
        used device is a classical simulator and natively supports this.

        If a device does not natively support broadcasting, it will execute broadcasted QNode calls
        by expanding the input arguments into separate executions. That is, every device can
        execute QNodes with broadcasting, but only supporting devices will benefit from it.

        **Usage**

        The first example above is rather simple. Broadcasting is possible in more complex
        scenarios as well, for which it is useful to understand the concept in more detail.
        The following rules and conventions apply:

        *There is at most one broadcasting axis*

        The broadcasted input has (exactly) one more axis than the operator(s) which receive(s)
        it would usually expect. For example, most operators expect a single scalar input and the
        *broadcasted* input correspondingly is a 1D array:

        >>> x = np.array([1., 2., 3.])
        >>> op = qml.RX(x, wires=0) # Additional axis of size 3.

        An operator ``op`` that supports broadcasting indicates the expected number of
        axes--or dimensions--in its attribute ``op.ndim_params``. This attribute is a tuple with
        one integer per argument of ``op``. The batch size of a broadcasted operator is stored
        in ``op.batch_size``:

        >>> op.ndim_params # RX takes one scalar input.
        (0,)
        >>> op.batch_size # The broadcasting axis has size 3.
        3

        The broadcasting axis is always the leading axis of an argument passed to an operator:

        >>> from scipy.stats import unitary_group
        >>> U = np.stack([unitary_group.rvs(4) for _ in range(3)])
        >>> U.shape # U stores three two-qubit unitaries, each of shape 4x4
        (3, 4, 4)
        >>> op = qml.QubitUnitary(U, wires=[0, 1])
        >>> op.batch_size
        3

        Stacking multiple broadcasting axes is *not* supported.

        *Multiple operators are broadcasted simultaneously*

        It is possible to broadcast multiple parameters simultaneously. In this case, the batch
        size of the broadcasting axes must match, and the parameters are combined like in Python's
        ``zip`` function. Non-broadcasted parameters do not need
        to be augmented manually but can simply be used as one would in individual QNode
        executions:

        .. code-block:: python

            dev = qml.device("default.qubit", wires=4)
            @qml.qnode(dev)
            def circuit(x, y, U):
                qml.QubitUnitary(U, wires=[0, 1, 2, 3])
                qml.RX(x, wires=0)
                qml.RY(y, wires=1)
                qml.RX(x, wires=2)
                qml.RY(y, wires=3)
                return qml.expval(qml.Z(0) @ qml.X(1) @ qml.Z(2) @ qml.Z(3))


            x = np.array([0.4, 2.1, -1.3])
            y = 2.71
            U = np.stack([unitary_group.rvs(16) for _ in range(3)])

        This circuit takes three arguments, and the first two are used twice each. ``x`` and
        ``U`` will lead to a batch size of ``3`` for the ``RX`` rotations and the multi-qubit
        unitary, respectively. The input ``y`` is a ``float`` value and will be used together with
        all three values in ``x`` and ``U``. We obtain three output values:

        >>> circuit(x, y, U)
        tensor([-0.06939911,  0.26051235, -0.20361048], requires_grad=True)

        This is equivalent to iterating over all broadcasted arguments using ``zip``:

        >>> [circuit(x_val, y, U_val) for x_val, U_val in zip(x, U)]
        [tensor(-0.06939911, requires_grad=True),
         tensor(0.26051235, requires_grad=True),
         tensor(-0.20361048, requires_grad=True)]

        In the same way it is possible to broadcast multiple arguments of a single operator,
        for example:

        >>> qml.Rot.ndim_params # Rot takes three scalar arguments
        (0, 0, 0)
        >>> x = np.array([0.4, 2.3, -0.1]) # Broadcast the first argument with size 3
        >>> y = 1.6 # Do not broadcast the second argument
        >>> z = np.array([1.2, -0.5, 2.5]) # Broadcast the third argument with size 3
        >>> op = qml.Rot(x, y, z, wires=0)
        >>> op.batch_size
        3

        *Broadcasting does not modify classical processing*

        Note that classical processing in QNodes will happen *before* broadcasting is taken into
        account. This means, that while *operators* always interpret the first axis as the
        broadcasting axis, QNodes do not necessarily do so:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit_unpacking(x):
                qml.RX(x[0], wires=0)
                qml.RY(x[1], wires=1)
                qml.RZ(x[2], wires=1)
                return qml.expval(qml.Z(0) @ qml.X(1))

            x = np.array([[1, 2], [3, 4], [5, 6]])

        The prepared parameter ``x`` has shape ``(3, 2)``, corresponding to the three operations
        and a batch size of ``2``:

        >>> circuit_unpacking(x)
        tensor([0.02162852, 0.30239696], requires_grad=True)

        If we were to iterate manually over the parameter settings, we probably would put the
        batching axis in ``x`` first. This is not the behaviour with parameter broadcasting
        because it does not modify the unpacking step within the QNode, so that ``x`` is
        unpacked *first* and the unpacked elements are expected to contain the
        broadcasted parameters for each operator individually;
        if we attempted to put the broadcasting axis of size ``2`` first, the
        indexing of ``x`` would fail in the ``RZ`` rotation within the QNode.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        func: Callable,
        device: SupportedDeviceAPIs,
        interface: str | Interface = Interface.AUTO,
        diff_method: TransformDispatcher | SupportedDiffMethods = "best",
        *,
        shots: ShotsLike | Literal["unset"] = "unset",
        grad_on_execution: bool | Literal["best"] = "best",
        cache: Cache | dict | Literal["auto"] | bool = "auto",
        cachesize: int = 10000,
        max_diff: int = 1,
        device_vjp: bool | None = False,
        postselect_mode: Literal["hw-like", "fill-shots"] | None = None,
        mcm_method: Literal["deferred", "one-shot", "tree-traversal"] | None = None,
        gradient_kwargs: dict | None = None,
        static_argnums: int | Iterable[int] = (),
        executor_backend: ExecBackends | str | None = None,
    ) -> None:
        self._init_args = locals()
        del self._init_args["self"]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                """Creating QNode(func=%s, device=%s, interface=%s, diff_method=%s, grad_on_execution=%s, cache=%s, cachesize=%s, max_diff=%s, gradient_kwargs=%s""",
                (
                    func
                    if not (logger.isEnabledFor(qml.logging.TRACE) and inspect.isfunction(func))
                    else "\n" + inspect.getsource(func)
                ),
                repr(device),
                interface,
                diff_method,
                grad_on_execution,
                cache,
                cachesize,
                max_diff,
                gradient_kwargs,
            )

        if not isinstance(device, (qml.devices.LegacyDevice, qml.devices.Device)):
            raise QuantumFunctionError("Invalid device. Device must be a valid PennyLane device.")

        if not isinstance(device, qml.devices.Device):
            device = qml.devices.LegacyDeviceFacade(device)

        gradient_kwargs = gradient_kwargs or {}

        if "shots" in inspect.signature(func).parameters:
            warnings.warn(
                "Detected 'shots' as an argument to the given quantum function. "
                "The 'shots' argument name is reserved for overriding the number of shots "
                "taken by the device. Its use outside of this context should be avoided.",
                UserWarning,
            )
            self._qfunc_uses_shots_arg = True
        else:
            self._qfunc_uses_shots_arg = False

        # input arguments
        self.func = func
        self.device: Device = device
        self._interface = Interface(interface)
        if self._interface in (Interface.JAX, Interface.JAX_JIT):
            _validate_jax_version()
        self.diff_method = diff_method
        _validate_diff_method(self.device, self.diff_method)

        self.capture_cache: LRUCache = LRUCache(maxsize=1000)
        if isinstance(static_argnums, int):
            static_argnums = (static_argnums,)
        self.static_argnums = sorted(static_argnums)

        # execution keyword arguments
        _validate_mcm_config(postselect_mode, mcm_method)
        self.execute_kwargs = {
            "grad_on_execution": grad_on_execution,
            "cache": cache,
            "cachesize": cachesize,
            "max_diff": max_diff,
            "device_vjp": device_vjp,
            "postselect_mode": postselect_mode,
            "mcm_method": mcm_method,
            "executor_backend": executor_backend,
        }

        # internal data attributes
        self._tape = None
        self._qfunc_output = None
        self._gradient_fn = None
        self.gradient_kwargs = gradient_kwargs

        self._shots: Shots = device.shots if shots == "unset" else Shots(shots)
        self._shots_override_device: bool = shots != "unset"
        self._transform_program = TransformProgram()
        functools.update_wrapper(self, func)

    def __copy__(self) -> QNode:
        copied_qnode = QNode.__new__(QNode)
        for attr, value in vars(self).items():
            if attr not in {"execute_kwargs", "_transform_program", "gradient_kwargs"}:
                setattr(copied_qnode, attr, value)

        copied_qnode.execute_kwargs = dict(self.execute_kwargs)
        copied_qnode._transform_program = qml.transforms.core.TransformProgram(
            self.transform_program
        )
        copied_qnode.gradient_kwargs = dict(self.gradient_kwargs)
        return copied_qnode

    def __repr__(self) -> str:
        """String representation."""
        if not isinstance(self.device, qml.devices.LegacyDeviceFacade):
            return f"<QNode: device='{self.device}', interface='{self.interface}', diff_method='{self.diff_method}', shots='{self.shots}'>"

        detail = "<QNode: wires={}, device='{}', interface='{}', diff_method='{}', shots='{}'>"
        return detail.format(
            self.device.num_wires,
            self.device.short_name,
            self.interface,
            self.diff_method,
            self.shots,
        )

    @property
    def shots(self) -> Shots:
        """Default shots for execution workflows.

        Note that this property is not able to be set directly; only `set_shots` can modify it.

        """
        return self._shots

    @shots.setter
    def shots(self, _):
        raise AttributeError(
            "Shots cannot be set on a qnode instance. You can set shots with `qml.set_shots`."
        )

    @property
    def interface(self) -> str:
        """The interface used by the QNode"""
        return "jax" if qml.capture.enabled() else self._interface.value

    @interface.setter
    def interface(self, value: str):
        self._interface = Interface(value)

    @property
    def transform_program(self) -> TransformProgram:
        """The transform program used by the QNode."""
        return self._transform_program

    @debug_logger
    def add_transform(self, transform_container: TransformContainer):
        """Add a transform (container) to the transform program.

        .. warning::

            This method is deprecated and will be removed in v0.44. Instead, please use :meth:`~.TransformProgram.push_back` on
            the ``QNode.transform_program`` property to add transforms to the transform program.

        .. warning:: This is a developer facing feature and is called when a transform is applied on a QNode.
        """
        warnings.warn(
            "The `qml.QNode.add_transform` method is deprecated and will be removed in v0.44. "
            "Instead, please use `QNode.transform_program.push_back(transform_container=transform_container)`.",
            PennyLaneDeprecationWarning,
        )
        self._transform_program.push_back(transform_container=transform_container)

    def update(self, **kwargs) -> QNode:
        """Returns a new QNode instance but with updated settings (e.g., a different `diff_method`). Any settings not specified will retain their original value.

        .. note::
            The QNode`s transform program cannot be updated using this method.

        Keyword Args:
            **kwargs: The provided keyword arguments must match that of :meth:`QNode.__init__`.
                The list of supported gradient keyword arguments can be found at ``qml.gradients.SUPPORTED_GRADIENT_KWARGS``.

        Returns:
            qnode (QNode): new QNode with updated settings


        Raises:
            ValueError: if provided keyword arguments are invalid

        **Example**

        Let's begin by defining a ``QNode`` object,

        .. code-block:: python

            dev = qml.device("default.qubit")

            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(x):
                qml.RZ(x, wires=0)
                qml.CNOT(wires=[0, 1])
                qml.RY(x, wires=1)
                return qml.expval(qml.PauliZ(1))

        If we wish to try out a new configuration without having to repeat the
        boilerplate above, we can use the ``QNode.update`` method. For example,
        we can update the differentiation method and execution arguments,

        >>> new_circuit = circuit.update(diff_method="adjoint", device_vjp=True)
        >>> print(new_circuit.diff_method)
        adjoint
        >>> print(new_circuit.execute_kwargs["device_vjp"])
        True

        Similarly, if we wish to re-configure the interface used for execution,

        >>> new_circuit= circuit.update(interface="torch")
        >>> new_circuit(1)
        tensor(0.5403, dtype=torch.float64)
        """
        if not kwargs:
            valid_params = set(self._init_args.copy()) | qml.gradients.SUPPORTED_GRADIENT_KWARGS
            raise ValueError(
                f"Must specify at least one configuration property to update. Valid properties are: {valid_params}."
            )

        original_init_args = self._init_args.copy()
        # gradient_kwargs defaults to None
        original_init_args["gradient_kwargs"] = original_init_args["gradient_kwargs"] or {}
        # nested dictionary update
        new_gradient_kwargs = kwargs.pop("gradient_kwargs", {})
        old_gradient_kwargs = (original_init_args.get("gradient_kwargs", {})).copy()
        old_gradient_kwargs.update(new_gradient_kwargs)
        kwargs["gradient_kwargs"] = old_gradient_kwargs

        old_shots = self.shots
        # set shots issue
        if (
            not self._shots_override_device
            and "device" in kwargs
            and old_shots != kwargs["device"].shots
        ):
            warnings.warn(
                "The device's shots value does not match the QNode's shots value. "
                "This may lead to unexpected behaviour. Use `set_shots` to update the QNode's shots.",
                UserWarning,
            )

        original_init_args.update(kwargs)
        updated_qn = QNode(**original_init_args)
        # pylint: disable=protected-access
        if updated_qn.shots != old_shots:
            updated_qn._set_shots(old_shots)
        if self._shots_override_device:
            updated_qn._shots_override_device = True

        # pylint: disable=protected-access
        updated_qn._transform_program = qml.transforms.core.TransformProgram(self.transform_program)
        return updated_qn

    def update_shots(self, shots: int | Shots) -> QNode:
        """Update the number of shots used by the QNode.

        Args:
            shots (int or Shots): The new number of shots to use.

        Returns:
            qnode (QNode): new QNode with updated shots
        """

        # Create a copy of the current QNode
        updated_qn = copy.copy(self)

        # Update the shots attribute directly
        # pylint: disable=protected-access
        updated_qn._set_shots(shots)

        return updated_qn

    def _set_shots(self, shots: int | Shots) -> None:
        """Set the number of shots used by the QNode.

        Args:
            shots (int or Shots): The new number of shots to use.
        """

        self._shots = Shots(shots)
        self._shots_override_device = True

    def _get_shots(self, kwargs: dict) -> Shots:
        """
        Note that this mutates kwargs to remove shots from it.
        """
        if self._qfunc_uses_shots_arg:
            return self.shots
        if "shots" in kwargs:
            # NOTE: at removal, remember to remove the userwarning below as well
            warnings.warn(
                "Specifying 'shots' when executing a QNode is deprecated and will be removed in "
                "v0.44. Please set shots on QNode initialization, or use qml.set_shots instead.",
                PennyLaneDeprecationWarning,
                stacklevel=2,
            )
            if self._shots_override_device:
                _kwargs_shots = kwargs.pop("shots")
                warnings.warn(
                    "Both 'shots=' parameter and 'set_shots' transform are specified. "
                    f"The transform will take precedence over 'shots={_kwargs_shots}.'",
                    UserWarning,
                    stacklevel=2,
                )

        if self._shots_override_device:  # QNode.shots precedency:
            return self.shots
        return kwargs.pop("shots", self.shots)

    @debug_logger
    def construct(self, args, kwargs) -> qml.tape.QuantumScript:
        """Call the quantum function with a tape context, ensuring the operations get queued."""
        kwargs = copy.copy(kwargs)
        shots = self._get_shots(kwargs)

        # Before constructing the tape, we pass the device to the
        # debugger to ensure they are compatible if there are any
        # breakpoints in the circuit
        # pylint: disable=import-outside-toplevel
        from pennylane.debugging import pldb_device_manager

        with pldb_device_manager(self.device):
            with AnnotatedQueue() as q:
                self._qfunc_output = self.func(*args, **kwargs)

        tape = QuantumScript.from_queue(q, shots)

        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = math.get_trainable_indices(params)

        _validate_qfunc_output(self._qfunc_output, tape.measurements)
        self._tape = tape
        return tape

    def _impl_call(self, *args, **kwargs) -> Result:
        # construct the tape
        tape = self.construct(args, kwargs)

        # Calculate the classical jacobians if necessary
        self._transform_program.set_classical_component(self, args, kwargs)

        res = execute(
            (tape,),
            device=self.device,
            diff_method=self.diff_method,
            interface=self.interface,
            transform_program=self._transform_program,
            gradient_kwargs=self.gradient_kwargs,
            **self.execute_kwargs,
        )
        res = res[0]

        # convert result to the interface in case the qfunc has no parameters

        if (
            len(tape.get_parameters(trainable_only=False)) == 0
            and not self._transform_program.is_informative
            and self.interface != "auto"
        ):
            res = _convert_to_interface(res, math.Interface(self.interface))

        return _to_qfunc_output_type(res, self._qfunc_output, tape.shots.has_partitioned_shots)

    def __call__(self, *args, **kwargs) -> Result:
        if qml.capture.enabled():
            from ._capture_qnode import capture_qnode  # pylint: disable=import-outside-toplevel

            return capture_qnode(self, *args, **kwargs)
        return self._impl_call(*args, **kwargs)


def qnode(device, **kwargs) -> Callable[[Callable], QNode]:
    """Docstring will be updated below."""
    return functools.partial(QNode, device=device, **kwargs)


qnode.__doc__ = QNode.__doc__
qnode.__signature__ = inspect.signature(QNode)
