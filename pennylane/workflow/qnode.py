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
# pylint: disable=too-many-instance-attributes,too-many-arguments,protected-access,unnecessary-lambda-assignment, too-many-branches, too-many-statements
import copy
import functools
import inspect
import logging
import warnings
from collections.abc import Sequence
from typing import Optional, Union

import pennylane as qml
from pennylane import Device
from pennylane.debugging import pldb_device_manager
from pennylane.logging import debug_logger
from pennylane.measurements import CountsMP, MidMeasureMP, Shots
from pennylane.tape import QuantumScript, QuantumTape

from .execution import INTERFACE_MAP, SUPPORTED_INTERFACES

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _convert_to_interface(res, interface):
    """
    Recursively convert res to the given interface.
    """
    interface = INTERFACE_MAP[interface]

    if interface in ["Numpy"]:
        return res

    if isinstance(res, (list, tuple)):
        return type(res)(_convert_to_interface(r, interface) for r in res)

    if isinstance(res, dict):
        return {k: _convert_to_interface(v, interface) for k, v in res.items()}

    return qml.math.asarray(res, like=interface if interface != "tf" else "tensorflow")


# pylint: disable=protected-access
def _get_device_shots(device) -> Shots:
    if isinstance(device, qml.devices.LegacyDevice):
        if device._shot_vector:
            return Shots(device._raw_shot_sequence)
        return Shots(device.shots)
    return device.shots


def _make_execution_config(
    circuit: Optional["QNode"], diff_method=None
) -> "qml.devices.ExecutionConfig":
    if diff_method is None or isinstance(diff_method, str):
        _gradient_method = diff_method
    else:
        _gradient_method = "gradient-transform"
    execute_kwargs = getattr(circuit, "execute_kwargs", {})
    grad_on_execution = execute_kwargs.get("grad_on_execution")
    if getattr(circuit, "interface", "") == "jax":
        grad_on_execution = False
    elif grad_on_execution == "best":
        grad_on_execution = None
    mcm_config = execute_kwargs.get("mcm_config", {})

    return qml.devices.ExecutionConfig(
        interface=getattr(circuit, "interface", None),
        gradient_method=_gradient_method,
        grad_on_execution=grad_on_execution,
        use_device_jacobian_product=execute_kwargs.get("device_vjp", False),
        mcm_config=mcm_config,
    )


def _to_qfunc_output_type(
    results: qml.typing.Result, qfunc_output, has_partitioned_shots
) -> qml.typing.Result:

    if has_partitioned_shots:
        return tuple(_to_qfunc_output_type(r, qfunc_output, False) for r in results)

    # Special case of single Measurement in a list
    if isinstance(qfunc_output, list) and len(qfunc_output) == 1:
        results = [results]

    # If the return type is not tuple (list or ndarray) (Autograd and TF backprop removed)
    if isinstance(qfunc_output, (tuple, qml.measurements.MeasurementProcess)):
        return results

    return type(qfunc_output)(results)


class QNode:
    """Represents a quantum node in the hybrid computational graph.

    A *quantum node* contains a :ref:`quantum function <intro_vcirc_qfunc>` (corresponding to
    a `variational circuit <https://pennylane.ai/qml/glossary/variational_circuit>`)
    and the computational device it is executed on.

    The QNode calls the quantum function to construct a :class:`~.QuantumTape` instance representing
    the quantum circuit.

    Args:
        func (callable): a quantum function
        device (~.Device): a PennyLane-compatible device
        interface (str): The interface that will be used for classical backpropagation.
            This affects the types of objects that can be passed to/returned from the QNode. See
            ``qml.workflow.SUPPORTED_INTERFACES`` for a list of all accepted strings.

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

        expansion_strategy (str): The strategy to use when circuit expansions or decompositions
            are required.

            - ``gradient``: The QNode will attempt to decompose
              the internal circuit such that all circuit operations are supported by the gradient
              method. Further decompositions required for device execution are performed by the
              device prior to circuit execution.

            - ``device``: The QNode will attempt to decompose the internal circuit
              such that all circuit operations are natively supported by the device.

            The ``gradient`` strategy typically results in a reduction in quantum device evaluations
            required during optimization, at the expense of an increase in classical preprocessing.
        max_expansion (int): The number of times the internal circuit should be expanded when
            executed on a device. Expansion occurs when an operation or measurement is not
            supported, and results in a gate decomposition. If any operations in the decomposition
            remain unsupported by the device, another expansion occurs.
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
            usage details, please refer to the :doc:`main measurements page </introduction/measurements>`.
        mcm_method (str): Strategy to use when executing circuits with mid-circuit measurements. Use ``"deferred"``
            to apply the deferred measurements principle (using the :func:`~pennylane.defer_measurements` transform),
            or ``"one-shot"`` if using finite shots to execute the circuit for each shot separately.
            ``default.qubit`` also supports ``"tree-traversal"`` which visits the tree of possible MCM sequences
            as the name suggests. If not provided,
            the device will determine the best choice automatically. For usage details, please refer to the
            :doc:`main measurements page </introduction/measurements>`.

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

    def __init__(
        self,
        func,
        device: Union[Device, "qml.devices.Device"],
        interface="auto",
        diff_method="best",
        expansion_strategy="gradient",
        max_expansion=10,
        grad_on_execution="best",
        cache="auto",
        cachesize=10000,
        max_diff=1,
        device_vjp=False,
        postselect_mode=None,
        mcm_method=None,
        **gradient_kwargs,
    ):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                """Creating QNode(func=%s, device=%s, interface=%s, diff_method=%s, expansion_strategy=%s, max_expansion=%s, grad_on_execution=%s, cache=%s, cachesize=%s, max_diff=%s, gradient_kwargs=%s""",
                (
                    func
                    if not (logger.isEnabledFor(qml.logging.TRACE) and inspect.isfunction(func))
                    else "\n" + inspect.getsource(func)
                ),
                repr(device),
                interface,
                diff_method,
                expansion_strategy,
                max_expansion,
                grad_on_execution,
                cache,
                cachesize,
                max_diff,
                gradient_kwargs,
            )

        if interface not in SUPPORTED_INTERFACES:
            raise qml.QuantumFunctionError(
                f"Unknown interface {interface}. Interface must be "
                f"one of {SUPPORTED_INTERFACES}."
            )

        if not isinstance(device, (qml.devices.LegacyDevice, qml.devices.Device)):
            raise qml.QuantumFunctionError(
                "Invalid device. Device must be a valid PennyLane device."
            )

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

        for kwarg in gradient_kwargs:
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

        # input arguments
        self.func = func
        self.device = device
        self._interface = interface
        self.diff_method = diff_method
        self.expansion_strategy = expansion_strategy
        self.max_expansion = max_expansion
        cache = (max_diff > 1) if cache == "auto" else cache

        mcm_config = qml.devices.MCMConfig(mcm_method=mcm_method, postselect_mode=postselect_mode)

        # execution keyword arguments
        self.execute_kwargs = {
            "grad_on_execution": grad_on_execution,
            "cache": cache,
            "cachesize": cachesize,
            "max_diff": max_diff,
            "max_expansion": max_expansion,
            "device_vjp": device_vjp,
            "mcm_config": mcm_config,
        }

        if self.expansion_strategy == "device":
            self.execute_kwargs["expand_fn"] = None

        # internal data attributes
        self._tape = None
        self._qfunc_output = None
        self._user_gradient_kwargs = gradient_kwargs
        self._original_device = device
        self.gradient_fn = None
        self.gradient_kwargs = {}
        self._tape_cached = False

        self._transform_program = qml.transforms.core.TransformProgram()
        self._update_gradient_fn()
        functools.update_wrapper(self, func)

    def __copy__(self):
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

    def __repr__(self):
        """String representation."""
        if isinstance(self.device, qml.devices.Device):
            return f"<QNode: device='{self.device}', interface='{self.interface}', diff_method='{self.diff_method}'>"

        detail = "<QNode: wires={}, device='{}', interface='{}', diff_method='{}'>"
        return detail.format(
            self.device.num_wires,
            self.device.short_name,
            self.interface,
            self.diff_method,
        )

    @property
    def interface(self):
        """The interface used by the QNode"""
        return self._interface

    @interface.setter
    def interface(self, value):
        if value not in SUPPORTED_INTERFACES:
            raise qml.QuantumFunctionError(
                f"Unknown interface {value}. Interface must be one of {SUPPORTED_INTERFACES}."
            )

        self._interface = INTERFACE_MAP[value]
        self._update_gradient_fn(shots=self.device.shots)

    @property
    def transform_program(self):
        """The transform program used by the QNode."""
        return self._transform_program

    @debug_logger
    def add_transform(self, transform_container):
        """Add a transform (container) to the transform program.

        .. warning:: This is a developer facing feature and is called when a transform is applied on a QNode.
        """
        self._transform_program.push_back(transform_container=transform_container)

    def _update_gradient_fn(self, shots=None, tape=None):
        if self.diff_method is None:
            self._interface = None
            self.gradient_fn = None
            self.gradient_kwargs = {}
            return
        if tape is None and shots:
            tape = qml.tape.QuantumScript([], [], shots=shots)

        diff_method = self.diff_method
        if (
            self.device.name == "lightning.qubit"
            and qml.metric_tensor in self.transform_program
            and self.diff_method == "best"
        ):
            diff_method = "parameter-shift"

        self.gradient_fn, self.gradient_kwargs, self.device = QNode.get_gradient_fn(
            self._original_device, self.interface, diff_method, tape=tape
        )
        self.gradient_kwargs.update(self._user_gradient_kwargs or {})

    def _update_original_device(self):
        # FIX: If the qnode swapped the device, increase the num_execution value on the original device.
        # In the long run, we should make sure that the user's device is the one
        # actually run so she has full control. This could be done by changing the class
        # of the user's device before and after executing the tape.
        if self.device is not self._original_device:
            if not self._tape_cached:
                self._original_device._num_executions += 1  # pylint: disable=protected-access

            # Update for state vector simulators that have the _pre_rotated_state attribute
            if hasattr(self._original_device, "_pre_rotated_state"):
                self._original_device._pre_rotated_state = self.device._pre_rotated_state

            # Update for state vector simulators that have the _state attribute
            if hasattr(self._original_device, "_state"):
                self._original_device._state = self.device._state

    # pylint: disable=too-many-return-statements
    @staticmethod
    @debug_logger
    def get_gradient_fn(
        device, interface, diff_method="best", tape: Optional["qml.tape.QuantumTape"] = None
    ):
        """Determine the best differentiation method, interface, and device
        for a requested device, interface, and diff method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface
            diff_method (str or .TransformDispatcher): The requested method of differentiation.
                If a string, allowed options are ``"best"``, ``"backprop"``, ``"adjoint"``,
                ``"device"``, ``"parameter-shift"``, ``"hadamard"``, ``"finite-diff"``, or ``"spsa"``.
                A gradient transform may also be passed here.
            tape (Optional[.QuantumTape]): the circuit that will be differentiated. Should include shots information.

        Returns:
            tuple[str or .TransformDispatcher, dict, .Device: Tuple containing the ``gradient_fn``,
            ``gradient_kwargs``, and the device to use when calling the execute function.
        """

        config = _make_execution_config(None, diff_method)
        if isinstance(device, qml.devices.Device):
            if device.supports_derivatives(config, circuit=tape):
                new_config = device.preprocess(config)[1]
                return new_config.gradient_method, {}, device
            if diff_method in {"backprop", "adjoint", "device"}:  # device-only derivatives
                raise qml.QuantumFunctionError(
                    f"Device {device} does not support {diff_method} with requested circuit."
                )

        if diff_method == "best":
            return QNode.get_best_method(device, interface, tape=tape)

        if isinstance(device, qml.devices.LegacyDevice):
            # handled by device.supports_derivatives with new device interface
            if diff_method == "backprop":
                return QNode._validate_backprop_method(device, interface, tape=tape)

            if diff_method == "adjoint":
                return QNode._validate_adjoint_method(device)

            if diff_method == "device":
                return QNode._validate_device_method(device)

        if diff_method == "parameter-shift":
            return QNode._validate_parameter_shift(device)

        if diff_method == "finite-diff":
            return qml.gradients.finite_diff, {}, device

        if diff_method == "spsa":
            return qml.gradients.spsa_grad, {}, device

        if diff_method == "hadamard":
            return qml.gradients.hadamard_grad, {}, device

        if isinstance(diff_method, str):
            raise qml.QuantumFunctionError(
                f"Differentiation method {diff_method} not recognized. Allowed "
                "options are ('best', 'parameter-shift', 'backprop', 'finite-diff', "
                "'device', 'adjoint', 'spsa', 'hadamard')."
            )

        if isinstance(diff_method, qml.transforms.core.TransformDispatcher):
            return diff_method, {}, device

        raise qml.QuantumFunctionError(
            f"Differentiation method {diff_method} must be a gradient transform or a string."
        )

    @staticmethod
    @debug_logger
    def get_best_method(device, interface, tape=None):
        """Returns the 'best' differentiation method
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
            device (.Device): PennyLane device
            interface (str): name of the requested interface
            shots

        Returns:
            tuple[str or .TransformDispatcher, dict, .Device: Tuple containing the ``gradient_fn``,
            ``gradient_kwargs``, and the device to use when calling the execute function.
        """
        config = _make_execution_config(None, "best")
        if isinstance(device, qml.devices.Device):

            if device.supports_derivatives(config, circuit=tape):
                new_config = device.preprocess(config)[1]
                return new_config.gradient_method, {}, device

            return QNode._validate_parameter_shift(device)

        try:
            return QNode._validate_device_method(device)
        except qml.QuantumFunctionError:
            try:
                return QNode._validate_backprop_method(device, interface, tape=tape)
            except qml.QuantumFunctionError:
                try:
                    return QNode._validate_parameter_shift(device)
                except qml.QuantumFunctionError:
                    return qml.gradients.finite_diff, {}, device

    @staticmethod
    @debug_logger
    def best_method_str(device, interface):
        """Similar to :meth:`~.get_best_method`, except return the
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
            device (.Device): PennyLane device
            interface (str): name of the requested interface

        Returns:
            str: The gradient function to use in human-readable format.
        """
        transform = QNode.get_best_method(device, interface)[0]

        if transform is qml.gradients.finite_diff:
            return "finite-diff"

        if transform in (qml.gradients.param_shift, qml.gradients.param_shift_cv):
            return "parameter-shift"

        # only other options at this point are "backprop" or "device"
        return transform

    @staticmethod
    @debug_logger
    def _validate_backprop_method(device, interface, tape=None):
        if isinstance(device, qml.devices.Device):
            raise ValueError(
                "QNode._validate_backprop_method only applies to the qml.Device interface."
            )
        if tape.shots if tape else _get_device_shots(device):
            raise qml.QuantumFunctionError("Backpropagation is only supported when shots=None.")

        if tape and any(isinstance(m.obs, qml.SparseHamiltonian) for m in tape.measurements):
            raise qml.QuantumFunctionError("backprop cannot differentiate a qml.SparseHamiltonian.")
        mapped_interface = INTERFACE_MAP.get(interface, interface)

        # determine if the device supports backpropagation
        backprop_interface = device.capabilities().get("passthru_interface", None)

        if backprop_interface is not None:
            # device supports backpropagation natively
            if mapped_interface == backprop_interface or interface == "auto":
                return "backprop", {}, device

            raise qml.QuantumFunctionError(
                f"Device {device.short_name} only supports diff_method='backprop' when using the "
                f"{backprop_interface} interface."
            )

        # determine if the device has any child devices that support backpropagation
        backprop_devices = device.capabilities().get("passthru_devices", None)

        if backprop_devices is not None:
            # device is analytic and has child devices that support backpropagation natively
            if interface == "auto":
                return "backprop", {}, device

            if mapped_interface in backprop_devices:
                # no need to create another device if the child device is the same (e.g., default.mixed)
                if backprop_devices[mapped_interface] == device.short_name:
                    return "backprop", {}, device

                # TODO: need a better way of passing existing device init options
                # to a new device?
                expand_fn = device.expand_fn
                batch_transform = device.batch_transform
                debugger = device._debugger
                tracker = device.tracker

                new_device = qml.device(
                    backprop_devices[mapped_interface], wires=device.wires, shots=device.shots
                )
                new_device.expand_fn = expand_fn
                new_device.batch_transform = batch_transform
                new_device._debugger = debugger
                new_device.tracker = tracker

                return "backprop", {}, new_device

            raise qml.QuantumFunctionError(
                f"Device {device.short_name} only supports diff_method='backprop' when using the "
                f"{list(backprop_devices.keys())} interfaces."
            )

        raise qml.QuantumFunctionError(
            f"The {device.short_name} device does not support native computations with "
            "autodifferentiation frameworks."
        )

    @staticmethod
    def _validate_adjoint_method(device):
        # The conditions below provide a minimal set of requirements that we can likely improve upon in
        # future, or alternatively summarize within a single device capability. Moreover, we also
        # need to inspect the circuit measurements to ensure only expectation values are taken. This
        # cannot be done here since we don't yet know the composition of the circuit.

        if isinstance(device, qml.devices.Device):
            raise ValueError(
                "QNode._validate_adjoint_method only applies to the qml.Device interface."
            )
        required_attrs = ["_apply_operation", "_apply_unitary", "adjoint_jacobian"]
        supported_device = all(hasattr(device, attr) for attr in required_attrs)
        supported_device = supported_device and device.capabilities().get("returns_state")

        if not supported_device:
            raise ValueError(
                f"The {device.short_name} device does not support adjoint differentiation."
            )

        if device.shots is not None:
            warnings.warn(
                "Requested adjoint differentiation to be computed with finite shots. "
                "Adjoint differentiation always calculated exactly.",
                UserWarning,
            )
        return "device", {"use_device_state": True, "method": "adjoint_jacobian"}, device

    @staticmethod
    def _validate_device_method(device):
        if isinstance(device, qml.devices.Device):
            raise ValueError(
                "QNode._validate_device_method only applies to the qml.Device interface."
            )
        # determine if the device provides its own jacobian method
        if device.capabilities().get("provides_jacobian", False):
            return "device", {}, device

        raise qml.QuantumFunctionError(
            f"The {device.short_name} device does not provide a native "
            "method for computing the jacobian."
        )

    @staticmethod
    def _validate_parameter_shift(device):
        if isinstance(device, qml.devices.Device):
            return qml.gradients.param_shift, {}, device
        model = device.capabilities().get("model", None)

        if model in {"qubit", "qutrit"}:
            return qml.gradients.param_shift, {}, device

        if model == "cv":
            return qml.gradients.param_shift_cv, {"dev": device}, device

        raise qml.QuantumFunctionError(
            f"Device {device.short_name} uses an unknown model ('{model}') "
            "that does not support the parameter-shift rule."
        )

    @property
    def tape(self) -> QuantumTape:
        """The quantum tape"""
        return self._tape

    qtape = tape  # for backwards compatibility

    @debug_logger
    def construct(self, args, kwargs):  # pylint: disable=too-many-branches
        """Call the quantum function with a tape context, ensuring the operations get queued."""
        kwargs = copy.copy(kwargs)
        old_interface = self.interface

        if self._qfunc_uses_shots_arg:
            shots = _get_device_shots(self._original_device)
        else:
            shots = kwargs.pop("shots", _get_device_shots(self._original_device))

        if old_interface == "auto":
            self.interface = qml.math.get_interface(*args, *list(kwargs.values()))

        # Before constructing the tape, we pass the device to the
        # debugger to ensure they are compatible if there are any
        # breakpoints in the circuit
        with pldb_device_manager(self.device):
            with qml.queuing.AnnotatedQueue() as q:
                self._qfunc_output = self.func(*args, **kwargs)

        self._tape = QuantumScript.from_queue(q, shots)

        params = self.tape.get_parameters(trainable_only=False)
        self.tape.trainable_params = qml.math.get_trainable_indices(params)

        if any(isinstance(m, CountsMP) for m in self.tape.measurements) and any(
            qml.math.is_abstract(a) for a in args
        ):
            raise qml.QuantumFunctionError("Can't JIT a quantum function that returns counts.")

        if isinstance(self._qfunc_output, qml.numpy.ndarray):
            measurement_processes = tuple(self.tape.measurements)
        elif not isinstance(self._qfunc_output, Sequence):
            measurement_processes = (self._qfunc_output,)
        else:
            measurement_processes = self._qfunc_output

        if not measurement_processes or not all(
            isinstance(m, qml.measurements.MeasurementProcess) for m in measurement_processes
        ):
            raise qml.QuantumFunctionError(
                "A quantum function must return either a single measurement, "
                "or a nonempty sequence of measurements."
            )

        terminal_measurements = [
            m for m in self.tape.measurements if not isinstance(m, MidMeasureMP)
        ]

        if any(ret is not m for ret, m in zip(measurement_processes, terminal_measurements)):
            raise qml.QuantumFunctionError(
                "All measurements must be returned in the order they are measured."
            )

        num_wires = len(self.tape.wires) if not self.device.wires else len(self.device.wires)
        for obj in self.tape.operations + self.tape.observables:
            if (
                getattr(obj, "num_wires", None) is qml.operation.WiresEnum.AllWires
                and obj.wires
                and len(obj.wires) != num_wires
            ):
                # check here only if enough wires
                raise qml.QuantumFunctionError(f"Operator {obj.name} must act on all wires")

        if self.expansion_strategy == "device":
            if isinstance(self.device, qml.devices.Device):
                tape, _ = self.device.preprocess()[0]([self.tape])
                if len(tape) != 1:
                    raise ValueError(
                        "Using 'device' for the `expansion_strategy` is not supported for batches of tapes"
                    )
                self._tape = tape[0]
            else:
                self._tape = self.device.expand_fn(self.tape, max_expansion=self.max_expansion)

        if old_interface == "auto":
            self.interface = "auto"

    def _execution_component(self, args: tuple, kwargs: dict, override_shots) -> qml.typing.Result:
        """Construct the transform program and execute the tapes. Helper function for ``__call__``

        Args:
            args (tuple): the arguments the QNode is called with
            kwargs (dict): the keyword arguments the QNode is called with
            override_shots : the shots to use for the execution.

        Returns:
            Result

        """

        cache = self.execute_kwargs.get("cache", False)
        using_custom_cache = (
            hasattr(cache, "__getitem__")
            and hasattr(cache, "__setitem__")
            and hasattr(cache, "__delitem__")
        )
        self._tape_cached = using_custom_cache and self.tape.hash in cache

        mcm_config = copy.copy(self.execute_kwargs["mcm_config"])
        finite_shots = _get_device_shots if override_shots is False else override_shots
        if not finite_shots:
            mcm_config.postselect_mode = None
            if mcm_config.mcm_method in ("one-shot", "tree-traversal"):
                raise ValueError(
                    f"Cannot use the '{mcm_config.mcm_method}' method for mid-circuit measurements with analytic mode."
                )
        if mcm_config.mcm_method == "single-branch-statistics":
            raise ValueError("Cannot use mcm_method='single-branch-statistics' without qml.qjit.")

        full_transform_program = qml.transforms.core.TransformProgram(self.transform_program)
        inner_transform_program = qml.transforms.core.TransformProgram()
        config = None

        if isinstance(self.device, qml.devices.Device):

            config = _make_execution_config(self, self.gradient_fn)
            device_transform_program, config = self.device.preprocess(execution_config=config)

            if config.use_device_gradient:
                full_transform_program += device_transform_program
            else:
                inner_transform_program += device_transform_program

        has_mcm_support = (
            any(isinstance(op, MidMeasureMP) for op in self._tape)
            and hasattr(self.device, "capabilities")
            and self.device.capabilities().get("supports_mid_measure", False)
        )
        if has_mcm_support:
            inner_transform_program.add_transform(
                qml.devices.preprocess.mid_circuit_measurements,
                device=self.device,
                mcm_config=mcm_config,
                interface=self.interface,
            )
            override_shots = 1
        elif hasattr(self.device, "capabilities"):
            inner_transform_program.add_transform(
                qml.defer_measurements,
                device=self.device,
            )

        # Add the gradient expand to the program if necessary
        if getattr(self.gradient_fn, "expand_transform", False):
            full_transform_program.insert_front_transform(
                qml.transform(self.gradient_fn.expand_transform),
                **self.gradient_kwargs,
            )

        # Calculate the classical jacobians if necessary
        full_transform_program.set_classical_component(self, args, kwargs)
        _prune_dynamic_transform(full_transform_program, inner_transform_program)

        # pylint: disable=unexpected-keyword-arg
        res = qml.execute(
            (self._tape,),
            device=self.device,
            gradient_fn=self.gradient_fn,
            interface=self.interface,
            transform_program=full_transform_program,
            inner_transform=inner_transform_program,
            config=config,
            gradient_kwargs=self.gradient_kwargs,
            override_shots=override_shots,
            **self.execute_kwargs,
        )
        res = res[0]

        # convert result to the interface in case the qfunc has no parameters

        if (
            len(self.tape.get_parameters(trainable_only=False)) == 0
            and not self.transform_program.is_informative
        ):
            res = _convert_to_interface(res, self.interface)

        return _to_qfunc_output_type(
            res, self._qfunc_output, self._tape.shots.has_partitioned_shots
        )

    def _impl_call(self, *args, **kwargs) -> qml.typing.Result:

        old_interface = self.interface
        if old_interface == "auto":
            interface = qml.math.get_interface(*args, *list(kwargs.values()))
            self._interface = INTERFACE_MAP[interface]

        if self._qfunc_uses_shots_arg:
            override_shots = False
        else:
            if "shots" not in kwargs:
                kwargs["shots"] = _get_device_shots(self._original_device)
            override_shots = kwargs["shots"]

        # construct the tape
        self.construct(args, kwargs)

        original_grad_fn = [self.gradient_fn, self.gradient_kwargs, self.device]
        self._update_gradient_fn(shots=override_shots, tape=self._tape)

        try:
            res = self._execution_component(args, kwargs, override_shots=override_shots)
        finally:
            if old_interface == "auto":
                self._interface = "auto"

            self._update_original_device()

            _, self.gradient_kwargs, self.device = original_grad_fn

        return res

    def __call__(self, *args, **kwargs) -> qml.typing.Result:
        if qml.capture.enabled():
            return qml.capture.qnode_call(self, *args, **kwargs)
        return self._impl_call(*args, **kwargs)


qnode = lambda device, **kwargs: functools.partial(QNode, device=device, **kwargs)
qnode.__doc__ = QNode.__doc__
qnode.__signature__ = inspect.signature(QNode)


def _prune_dynamic_transform(outer_transform, inner_transform):
    """Ensure a single ``dynamic_one_shot`` transform is applied.

    Sometimes device preprocess contains a ``mid_circuit_measurements`` transform, which will
    be added to the inner transform program. If the user then applies a ``dynamic_one_shot``
    manually, it will duplicate the ``mid_circuit_measurements`` transform. This function ensures
    that there is only one ``dynamic_one_shot`` transform in the outer and inner transform
    programs combined.

    """

    all_transforms = outer_transform + inner_transform
    type_to_keep = 0
    if any("mid_circuit_measurements" in str(t) for t in all_transforms):
        type_to_keep = 2
    elif any("dynamic_one_shot" in str(t) for t in all_transforms):
        type_to_keep = 1

    if type_to_keep == 0:
        return

    dynamic_transform_found = inner_transform.prune_dynamic_transform(type_to_keep)
    if dynamic_transform_found:
        type_to_keep = 0
    outer_transform.prune_dynamic_transform(type_to_keep)
