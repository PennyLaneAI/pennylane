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
import copy
import functools
import inspect
import logging
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, Optional, Union, get_args

from cachetools import Cache

import pennylane as qml
from pennylane.debugging import pldb_device_manager
from pennylane.logging import debug_logger
from pennylane.math import Interface, SupportedInterfaceUserInput, get_canonical_interface_name
from pennylane.measurements import MidMeasureMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms.core import TransformContainer, TransformDispatcher, TransformProgram

from ._capture_qnode import capture_qnode
from ._setup_transform_program import _setup_transform_program
from .resolution import SupportedDiffMethods, _resolve_execution_config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SupportedDeviceAPIs = Union["qml.devices.LegacyDevice", "qml.devices.Device"]


def _convert_to_interface(result, interface: Interface):
    """
    Recursively convert a result to the given interface.
    """

    if interface == Interface.NUMPY:
        return result

    if isinstance(result, (list, tuple)):
        return type(result)(_convert_to_interface(r, interface) for r in result)

    if isinstance(result, dict):
        return {k: _convert_to_interface(v, interface) for k, v in result.items()}

    return qml.math.asarray(result, like=interface.get_like())


def _make_execution_config(
    circuit: Optional["QNode"], diff_method=None, mcm_config=None
) -> "qml.devices.ExecutionConfig":
    circuit_interface = getattr(circuit, "interface", Interface.NUMPY)
    execute_kwargs = getattr(circuit, "execute_kwargs", {})
    gradient_kwargs = getattr(circuit, "gradient_kwargs", {})
    grad_on_execution = execute_kwargs.get("grad_on_execution")
    if circuit_interface == Interface.JAX:
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


def _to_qfunc_output_type(
    results: qml.typing.Result, qfunc_output, has_partitioned_shots
) -> qml.typing.Result:

    if has_partitioned_shots:
        return tuple(_to_qfunc_output_type(r, qfunc_output, False) for r in results)

    qfunc_output_leaves, qfunc_output_structure = qml.pytrees.flatten(
        qfunc_output, is_leaf=lambda obj: isinstance(obj, (qml.measurements.MeasurementProcess))
    )

    # counts results are treated as a leaf
    results_leaves = qml.pytrees.flatten(results, is_leaf=lambda obj: isinstance(obj, dict))[0]

    # patch for transforms that change the number of results like metric_tensor
    if len(results_leaves) != len(qfunc_output_leaves):
        if isinstance(qfunc_output, (Sequence, qml.measurements.MeasurementProcess)):
            return results
        return type(qfunc_output)(results)

    # result spec squeezes out dim for single measurement value
    # we need to add it back in
    if len(qfunc_output_leaves) == 1:
        results = (results,)

    return qml.pytrees.unflatten(results, qfunc_output_structure)


def _validate_gradient_kwargs(gradient_kwargs: dict) -> None:
    for kwarg in gradient_kwargs:
        if kwarg == "expansion_strategy":
            raise ValueError(
                "'expansion_strategy' is no longer a valid keyword argument to QNode."
                " To inspect the circuit at a given stage in the transform program, please"
                " use qml.workflow.construct_batch instead."
            )

        if kwarg == "max_expansion":
            raise ValueError("'max_expansion' is no longer a valid keyword argument to QNode.")
        if kwarg in ["gradient_fn", "grad_method"]:
            warnings.warn(
                "It appears you may be trying to set the method of differentiation via the "
                f"keyword argument {kwarg}. This is not supported in qnode and will default to "
                "backpropogation. Use diff_method instead."
            )
        elif kwarg == "shots":
            raise ValueError(
                "'shots' is not a valid gradient_kwarg. If your quantum function takes the "
                "argument 'shots' or if you want to set the number of shots with which the "
                "QNode is executed, pass it to the QNode call, not its definition."
            )
        elif kwarg not in qml.gradients.SUPPORTED_GRADIENT_KWARGS:
            warnings.warn(
                f"Received gradient_kwarg {kwarg}, which is not included in the list of "
                "standard qnode gradient kwargs."
            )


def _validate_qfunc_output(qfunc_output, measurements) -> None:
    measurement_processes = qml.pytrees.flatten(
        qfunc_output,
        is_leaf=lambda obj: isinstance(obj, qml.measurements.MeasurementProcess),
    )[0]

    # user provides no measurements or non-measurements
    if len(measurement_processes) == 0:
        measurement_processes = None
    else:
        # patch for tensor measurement objects, e.g., qml.math.hstack <-> [tensor([tensor(...), tensor(...)])]
        if isinstance(measurement_processes[0], Iterable) and any(
            isinstance(m, qml.typing.TensorLike) for m in measurement_processes[0]
        ):
            measurement_processes = [
                m.base.item()
                for m in measurement_processes[0]
                if isinstance(m.base.item(), qml.measurements.MeasurementProcess)
            ]

    if not measurement_processes or not all(
        isinstance(m, qml.measurements.MeasurementProcess) for m in measurement_processes
    ):
        raise qml.QuantumFunctionError(
            "A quantum function must return either a single measurement, "
            "or a nonempty sequence of measurements."
        )

    terminal_measurements = [m for m in measurements if not isinstance(m, MidMeasureMP)]

    if any(ret is not m for ret, m in zip(measurement_processes, terminal_measurements)):
        raise qml.QuantumFunctionError(
            "All measurements must be returned in the order they are measured."
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
        func (callable): a quantum function
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

            * ``"hadamard"``: Use the analytic hadamard gradient test
              rule for all supported quantum operation arguments. More info is in the documentation
              :func:`qml.gradients.hadamard_grad <.gradients.hadamard_grad>`.

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
        mcm_method (str): Strategy to use when executing circuits with mid-circuit measurements. Use ``"deferred"``
            to apply the deferred measurements principle (using the :func:`~pennylane.defer_measurements` transform),
            or ``"one-shot"`` if using finite shots to execute the circuit for each shot separately.
            ``default.qubit`` also supports ``"tree-traversal"`` which visits the tree of possible MCM sequences
            as the name suggests. If not provided,
            the device will determine the best choice automatically. For usage details, please refer to the
            :doc:`dynamic quantum circuits page </introduction/dynamic_quantum_circuits>`.

    Keyword Args:
        **kwargs: Any additional keyword arguments provided are passed to the differentiation
            method. Please refer to the :mod:`qml.gradients <.gradients>` module for details
            on supported options for your chosen gradient transform.

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
        used device is a classical simulator and natively supports this. The latter can be checked
        with the capabilities of the device:

        >>> dev.capabilities()["supports_broadcasting"]
        True

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
        interface: SupportedInterfaceUserInput = Interface.AUTO,
        diff_method: Union[TransformDispatcher, SupportedDiffMethods] = "best",
        *,
        grad_on_execution: Literal[True, False, "best"] = "best",
        cache: Union[Cache, Literal["auto", True, False]] = "auto",
        cachesize: int = 10000,
        max_diff: int = 1,
        device_vjp: Union[None, bool] = False,
        postselect_mode: Literal[None, "hw-like", "fill-shots"] = None,
        mcm_method: Literal[None, "deferred", "one-shot", "tree-traversal"] = None,
        **gradient_kwargs,
    ):

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
            raise qml.QuantumFunctionError(
                "Invalid device. Device must be a valid PennyLane device."
            )

        if not isinstance(device, qml.devices.Device):
            device = qml.devices.LegacyDeviceFacade(device)

        _validate_gradient_kwargs(gradient_kwargs)
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
        self.device = device
        self._interface = (
            Interface.NUMPY if diff_method is None else get_canonical_interface_name(interface)
        )
        self.diff_method = diff_method
        mcm_config = qml.devices.MCMConfig(mcm_method=mcm_method, postselect_mode=postselect_mode)
        cache = (max_diff > 1) if cache == "auto" else cache

        # execution keyword arguments
        self.execute_kwargs = {
            "grad_on_execution": grad_on_execution,
            "cache": cache,
            "cachesize": cachesize,
            "max_diff": max_diff,
            "device_vjp": device_vjp,
            "mcm_config": mcm_config,
        }

        # internal data attributes
        self._tape = None
        self._qfunc_output = None
        self._gradient_fn = None
        self.gradient_kwargs = gradient_kwargs

        self._transform_program = TransformProgram()
        functools.update_wrapper(self, func)

        # validation check.  Will raise error if bad diff_method
        if diff_method is not None:
            QNode.get_gradient_fn(self.device, self.interface, self.diff_method)

    def __copy__(self) -> "QNode":
        copied_qnode = QNode.__new__(QNode)
        for attr, value in vars(self).items():
            if attr not in {"execute_kwargs", "_transform_program", "gradient_kwargs"}:
                setattr(copied_qnode, attr, value)

        copied_qnode.execute_kwargs = dict(self.execute_kwargs)
        copied_qnode._transform_program = qml.transforms.core.TransformProgram(
            self.transform_program
        )  # pylint: disable=protected-access
        copied_qnode.gradient_kwargs = dict(self.gradient_kwargs)
        return copied_qnode

    def __repr__(self) -> str:
        """String representation."""
        if not isinstance(self.device, qml.devices.LegacyDeviceFacade):
            return f"<QNode: device='{self.device}', interface='{self.interface}', diff_method='{self.diff_method}'>"

        detail = "<QNode: wires={}, device='{}', interface='{}', diff_method='{}'>"
        return detail.format(
            self.device.num_wires,
            self.device.short_name,
            self.interface,
            self.diff_method,
        )

    @property
    def interface(self) -> str:
        """The interface used by the QNode"""
        return self._interface.value

    @interface.setter
    def interface(self, value: SupportedInterfaceUserInput):
        self._interface = get_canonical_interface_name(value)

    @property
    def transform_program(self) -> TransformProgram:
        """The transform program used by the QNode."""
        return self._transform_program

    @debug_logger
    def add_transform(self, transform_container: TransformContainer):
        """Add a transform (container) to the transform program.

        .. warning:: This is a developer facing feature and is called when a transform is applied on a QNode.
        """
        self._transform_program.push_back(transform_container=transform_container)

    # pylint: disable=too-many-return-statements, unused-argument
    @staticmethod
    @debug_logger
    def get_gradient_fn(
        device: SupportedDeviceAPIs,
        interface,
        diff_method: Union[TransformDispatcher, SupportedDiffMethods] = "best",
        tape: Optional["qml.tape.QuantumTape"] = None,
    ):
        """Determine the best differentiation method, interface, and device
        for a requested device, interface, and diff method.

        Args:
            device (.device.Device): PennyLane device
            interface (str): name of the requested interface
            diff_method (str or .TransformDispatcher): The requested method of differentiation.
                If a string, allowed options are ``"best"``, ``"backprop"``, ``"adjoint"``,
                ``"device"``, ``"parameter-shift"``, ``"hadamard"``, ``"finite-diff"``, or ``"spsa"``.
                A gradient transform may also be passed here.
            tape (Optional[.QuantumTape]): the circuit that will be differentiated. Should include shots information.

        Returns:
            tuple[str or .TransformDispatcher, dict, .device.Device: Tuple containing the ``gradient_fn``,
            ``gradient_kwargs``, and the device to use when calling the execute function.
        """
        if diff_method is None:
            return None, {}, device

        config = _make_execution_config(None, diff_method)

        if device.supports_derivatives(config, circuit=tape):
            new_config = device.setup_execution_config(config)
            return new_config.gradient_method, {}, device

        if diff_method in {"backprop", "adjoint", "device"}:  # device-only derivatives
            raise qml.QuantumFunctionError(
                f"Device {device} does not support {diff_method} with requested circuit."
            )

        if diff_method == "best":
            if tape and any(isinstance(o, qml.operation.CV) for o in tape):
                return qml.gradients.param_shift_cv, {"dev": device}, device

            return qml.gradients.param_shift, {}, device

        if diff_method == "parameter-shift":
            if tape and any(isinstance(o, qml.operation.CV) and o.name != "Identity" for o in tape):
                return qml.gradients.param_shift_cv, {"dev": device}, device
            return qml.gradients.param_shift, {}, device

        if diff_method == "finite-diff":
            return qml.gradients.finite_diff, {}, device

        if diff_method == "spsa":
            return qml.gradients.spsa_grad, {}, device

        if diff_method == "hadamard":
            return qml.gradients.hadamard_grad, {}, device

        if isinstance(diff_method, qml.transforms.core.TransformDispatcher):
            return diff_method, {}, device

        raise qml.QuantumFunctionError(
            f"Differentiation method {diff_method} not recognized. Allowed "
            f"options are {tuple(get_args(SupportedDiffMethods))}."
        )

    # pylint: disable=unused-argument
    @staticmethod
    @debug_logger
    def get_best_method(
        device: SupportedDeviceAPIs,
        interface: SupportedInterfaceUserInput,
        tape: Optional["qml.tape.QuantumTape"] = None,
    ) -> tuple[
        Union[TransformDispatcher, Literal["device", "backprop", "parameter-shift", "finite-diff"]],
        dict[str, Any],
        SupportedDeviceAPIs,
    ]:
        """
        .. warning::

            This method is deprecated in v0.40 and will be removed in v0.41.
            Instead, use the :func:`qml.workflow.get_best_diff_method <.workflow.get_best_diff_method>` function.

        Returns the 'best' differentiation method
        for a particular device and interface combination.

        This method attempts to determine support for differentiation
        methods using the following order:

        * ``"device"``
        * ``"backprop"``
        * ``"parameter-shift"``
        * ``"finite-diff"``

        The first differentiation method that is supported (going from
        top to bottom) will be returned. Note that the SPSA-based and Hadamard-based gradients
        are not included here.

        Args:
            device (.devices.Device): PennyLane device
            interface (str): name of the requested interface
            shots

        Returns:
            tuple[str or .TransformDispatcher, dict, .device.Device: Tuple containing the ``gradient_fn``,
            ``gradient_kwargs``, and the device to use when calling the execute function.
        """

        warnings.warn(
            "QNode.get_best_method is deprecated and will be removed in v0.41. "
            "Instead, use the qml.workflow.get_best_diff_method function.",
            qml.PennyLaneDeprecationWarning,
        )

        if not isinstance(device, qml.devices.Device):
            device = qml.devices.LegacyDeviceFacade(device)

        config = _make_execution_config(None, "best")

        if device.supports_derivatives(config, circuit=tape):
            new_config = device.setup_execution_config(config)
            return new_config.gradient_method, {}, device

        if tape and any(isinstance(o, qml.operation.CV) for o in tape):
            return qml.gradients.param_shift_cv, {"dev": device}, device

        return qml.gradients.param_shift, {}, device

    @staticmethod
    @debug_logger
    def best_method_str(device: SupportedDeviceAPIs, interface: SupportedInterfaceUserInput) -> str:
        """
        .. warning::

            This method is deprecated in v0.40 and will be removed in v0.41.
            Instead, use the :func:`qml.workflow.get_best_diff_method <.workflow.get_best_diff_method>` function.


        Similar to :meth:`~.get_best_method`, except return the
        'best' differentiation method in human-readable format.

        This method attempts to determine support for differentiation
        methods using the following order:

        * ``"device"``
        * ``"backprop"``
        * ``"parameter-shift"``
        * ``"finite-diff"``

        The first differentiation method that is supported (going from
        top to bottom) will be returned. Note that the SPSA-based and Hadamard-based gradient
        are not included here.

        This method is intended only for debugging purposes. Otherwise,
        :meth:`~.get_best_method` should be used instead.

        Args:
            device (.devices.Device): PennyLane device
            interface (str): name of the requested interface

        Returns:
            str: The gradient function to use in human-readable format.
        """

        warnings.warn(
            "QNode.best_method_str is deprecated and will be removed in v0.41. "
            "Instead, use the qml.workflow.get_best_diff_method function.",
            qml.PennyLaneDeprecationWarning,
        )

        if not isinstance(device, qml.devices.Device):
            device = qml.devices.LegacyDeviceFacade(device)

        warnings.filterwarnings(
            "ignore", "QNode.get_best_method is deprecated", qml.PennyLaneDeprecationWarning
        )
        transform = QNode.get_best_method(device, interface)[0]

        if transform is qml.gradients.finite_diff:
            return "finite-diff"

        if transform in (qml.gradients.param_shift, qml.gradients.param_shift_cv):
            return "parameter-shift"

        # only other options at this point are "backprop" or "device"
        return transform

    @property
    def tape(self) -> QuantumTape:
        """The quantum tape

        .. warning::

            This property is deprecated in v0.40 and will be removed in v0.41.
            Instead, use the :func:`qml.workflow.construct_tape <.workflow.construct_tape>` function.
        """

        warnings.warn(
            "The tape/qtape property is deprecated and will be removed in v0.41. "
            "Instead, use the qml.workflow.construct_tape function.",
            qml.PennyLaneDeprecationWarning,
        )
        return self._tape

    qtape = tape  # for backwards compatibility

    @debug_logger
    def construct(self, args, kwargs):
        """Call the quantum function with a tape context, ensuring the operations get queued."""
        kwargs = copy.copy(kwargs)

        if self._qfunc_uses_shots_arg:
            shots = self.device.shots
        else:
            shots = kwargs.pop("shots", self.device.shots)

        # Before constructing the tape, we pass the device to the
        # debugger to ensure they are compatible if there are any
        # breakpoints in the circuit
        with pldb_device_manager(self.device):
            with qml.queuing.AnnotatedQueue() as q:
                self._qfunc_output = self.func(*args, **kwargs)

        self._tape = QuantumScript.from_queue(q, shots)

        params = self._tape.get_parameters(trainable_only=False)
        self._tape.trainable_params = qml.math.get_trainable_indices(params)

        _validate_qfunc_output(self._qfunc_output, self._tape.measurements)

    def _execution_component(self, args: tuple, kwargs: dict) -> qml.typing.Result:
        """Construct the transform program and execute the tapes. Helper function for ``__call__``

        Args:
            args (tuple): the arguments the QNode is called with
            kwargs (dict): the keyword arguments the QNode is called with

        Returns:
            Result

        """

        execute_kwargs = copy.copy(self.execute_kwargs)
        mcm_config = copy.copy(execute_kwargs["mcm_config"])

        config = _make_execution_config(self, self.diff_method, mcm_config=mcm_config)
        config = _resolve_execution_config(
            config, self.device, (self._tape,), self.transform_program
        )

        outer_transform_program, inner_transform_program = _setup_transform_program(
            self.transform_program,
            self.device,
            config,
            execute_kwargs["cache"],
            execute_kwargs["cachesize"],
        )

        # Calculate the classical jacobians if necessary
        outer_transform_program.set_classical_component(self, args, kwargs)

        res = qml.execute(
            (self._tape,),
            device=self.device,
            diff_method=config.gradient_method,
            interface=config.interface,
            transform_program=outer_transform_program,
            inner_transform=inner_transform_program,
            config=config,
            gradient_kwargs=config.gradient_keyword_arguments,
            **execute_kwargs,
        )
        res = res[0]

        # convert result to the interface in case the qfunc has no parameters

        if (
            len(self._tape.get_parameters(trainable_only=False)) == 0
            and not self.transform_program.is_informative
        ):
            res = _convert_to_interface(res, config.interface)

        return _to_qfunc_output_type(
            res, self._qfunc_output, self._tape.shots.has_partitioned_shots
        )

    def _impl_call(self, *args, **kwargs) -> qml.typing.Result:

        # construct the tape
        self.construct(args, kwargs)

        old_interface = self.interface
        if old_interface == "auto":
            interface = (
                Interface.JAX
                if qml.capture.enabled()
                else qml.math.get_interface(*args, *list(kwargs.values()))
            )
            try:
                interface = get_canonical_interface_name(interface)
            except ValueError:
                interface = Interface.NUMPY
            self._interface = interface

        try:
            res = self._execution_component(args, kwargs)
        finally:
            if old_interface == "auto":
                self._interface = Interface.AUTO

        return res

    def __call__(self, *args, **kwargs) -> qml.typing.Result:
        if qml.capture.enabled():
            return capture_qnode(self, *args, **kwargs)
        return self._impl_call(*args, **kwargs)


def qnode(device, **kwargs):
    """Docstring will be updated below."""
    return functools.partial(QNode, device=device, **kwargs)


qnode.__doc__ = QNode.__doc__
qnode.__signature__ = inspect.signature(QNode)
