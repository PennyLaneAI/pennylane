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
from collections.abc import Sequence
from typing import MutableMapping, Optional, Union
from dataclasses import replace

from cachetools import LRUCache

import pennylane as qml
from pennylane.logging import debug_logger
from pennylane.measurements import CountsMP, MidMeasureMP, Shots
from pennylane.measurements.measurements import MeasurementProcess
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms.core.transform_program import TransformProgram
from pennylane.typing import TensorLike

from .execution import INTERFACE_MAP, SUPPORTED_INTERFACES

from .beta_execution import execute, resolve_execution_config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _convert_to_interface(res: qml.typing.Result, interface: Optional[str])-> qml.typing.Result:
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


def _validate_qfunc_output(tape: qml.tape.QuantumScript, qfunc_output) -> None:
    if isinstance(qfunc_output, qml.numpy.ndarray):
        measurement_processes = tuple(qfunc_output)
    elif not isinstance(qfunc_output, Sequence):
        measurement_processes = (qfunc_output,)
    else:
        measurement_processes = qfunc_output

    if not measurement_processes or not all(
        isinstance(m, qml.measurements.MeasurementProcess) for m in measurement_processes
    ):
        raise qml.QuantumFunctionError(
            "A quantum function must return either a single measurement, "
            "or a nonempty sequence of measurements."
        )
    if any(ret is not m for ret, m in zip(measurement_processes, tape.measurements)):
        raise qml.QuantumFunctionError(
            "All measurements must be returned in the order they are measured."
        )


def _to_qfunc_output_type(
    results: qml.typing.Result, qfunc_output, has_partitioned_shots: bool
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


def _resolve_cache(cache, cachesize: int, derivative_order: int) -> Optional[MutableMapping]:
    if cache is True:
        return LRUCache(cachesize)
    if cache == "auto":
        return LRUCache(cachesize) if derivative_order > 1 else None
    if cache is False:
        return None
    if isinstance(cache, MutableMapping):
        return cache
    raise ValueError(f"got cache {cache} which is invalid")


class QNode:
    """Represents a quantum node in the hybrid computational graph.

    A *quantum node* contains a :ref:`quantum function <intro_vcirc_qfunc>`
    (corresponding to a :ref:`variational circuit <glossary_variational_circuit>`)
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
        device: "qml.devices.Device",
        interface="auto",
        diff_method="best",
        expansion_strategy="gradient",
        max_expansion=10,
        grad_on_execution="best",
        cache="auto",
        cachesize=10000,
        max_diff=1,
        device_vjp=False,
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

        if isinstance(device, qml.devices.LegacyDevice):
            device = qml.deviecs.LegacyDeviceFacade(device)

        if interface not in SUPPORTED_INTERFACES:
            raise qml.QuantumFunctionError(
                f"Unknown interface {interface}. Interface must be "
                f"one of {SUPPORTED_INTERFACES}."
            )

        if not isinstance(device, qml.devices.Device):
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
                    f"It appears you may be trying to set the method of differentiation via the kwarg "
                    f"{kwarg}. This is not supported in qnode and will default to backpropogation. Use "
                    f"diff_method instead."
                )
            elif kwarg not in qml.gradients.SUPPORTED_GRADIENT_KWARGS:
                warnings.warn(
                    f"Received gradient_kwarg {kwarg}, which is not included in the list of standard qnode "
                    f"gradient kwargs."
                )

        # input arguments
        self.func = func
        self._device = device

        self._default_execution_config = qml.devices.ExecutionConfig(
            gradient_method=diff_method,
            use_device_jacobian_product=device_vjp,
            grad_on_execution=None if grad_on_execution=="best" else grad_on_execution,
            gradient_keyword_arguments=gradient_kwargs,
            interface=interface,
            derivative_order=max_diff
        )
        dummy_tape = qml.tape.QuantumScript([],[],shots=device.shots)
        resolve_execution_config((dummy_tape,), device, self.default_execution_config)

        self._cache = cache
        self._cachesize = cachesize

        self._transform_program = qml.transforms.core.TransformProgram()
        functools.update_wrapper(self, func)

    _tape : Optional[qml.tape.QuantumScript] = None
    _qfunc_output = Union[tuple, TensorLike, MeasurementProcess]

    @property
    def device(self) -> qml.devices.Device:
        return self._device

    @property
    def default_execution_config(self) -> qml.devices.ExecutionConfig:
        return self._default_execution_config

    @property
    def interface(self):
        return self.default_execution_config.interface

    @property
    def diff_method(self):
        return self.default_execution_config.gradient_method

    @property
    def gradient_kwargs(self) -> dict:
        return self.default_execution_config.gradient_keyword_arguments

    @property
    def max_diff(self) -> int:
        return self.default_execution_config.derivative_order

    @property
    def execute_kwargs(self) -> dict:
        return {
            "grad_on_execution": self.default_execution_config.grad_on_execution,
            "cache": self._cache,
            "cachesize": self._cachesize,
            "max_diff": self.default_execution_config.derivative_order,
            "max_expansion": None,
            "device_vjp": self.default_execution_config.use_device_jacobian_product,
        }

    @property
    def gradient_fn(self) -> Union[None, str, qml.transforms.core.TransformDispatcher]:
        return QNode.get_gradient_fn(self.device,  self.interface, self.diff_method, self.tape)
        

    def __copy__(self):
        copied_qnode = QNode.__new__(QNode)
        for attr, value in vars(self).items():
            if attr not in {"execute_kwargs", "_transform_program", "gradient_kwargs"}:
                setattr(copied_qnode, attr, value)
        copied_qnode._transform_program = qml.transforms.core.TransformProgram(
            self.transform_program
        )  # pylint: disable=protected-access
        copied_qnode._default_execution_config = copy.deepcopy(self._default_execution_config)
        return copied_qnode

    def __repr__(self):
        """String representation."""
        return f"<QNode: device='{self.device}', interface='{self.interface}', diff_method='{self.diff_method}'>"

    @interface.setter
    def interface(self, value):
        if value not in SUPPORTED_INTERFACES:
            raise qml.QuantumFunctionError(
                f"Unknown interface {value}. Interface must be one of {SUPPORTED_INTERFACES}."
            )

        self._interface = INTERFACE_MAP[value]
        self._default_execution_config = replace(self._default_execution_config, interface=value)

    @property
    def transform_program(self) -> TransformProgram:
        """The transform program used by the QNode."""
        return self._transform_program

    @property
    def tape(self) -> QuantumTape:
        """The quantum tape"""
        return self._tape

    qtape = tape  # for backwards compatibility

    @staticmethod
    def get_gradient_fn(
        device, interface, diff_method="best", tape: Optional["qml.tape.QuantumTape"] = None
    ):
        if tape is None:
            tape = qml.tape.QuantumScript([], [], shots=device.shots)
        dummy_execution_config = qml.devices.ExecutionConfig(interface=interface, gradient_method=diff_method)
        new_config = resolve_execution_config((tape,), device, dummy_execution_config)
        return new_config.gradient_method

    @staticmethod
    @debug_logger
    def get_best_method(device, interface, tape=None):
        if tape is None:
            tape = qml.tape.QuantumScript([], [], shots=device.shots)
        dummy_execution_config = qml.devices.ExecutionConfig(interface=interface, gradient_method="best")
        new_config = resolve_execution_config((tape,), device, dummy_execution_config)
        return new_config.gradient_method, {}, device

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


    @debug_logger
    def construct(self, args, kwargs) -> qml.tape.QuantumScript:
        """Call the quantum function with a tape context, ensuring the operations get queued."""
        kwargs = copy.copy(kwargs)

        if self._qfunc_uses_shots_arg:
            shots = self.device.shots
        else:
            shots = kwargs.pop("shots", self.device.shots)

        with qml.queuing.AnnotatedQueue() as q:
            self._qfunc_output = self.func(*args, **kwargs)

        self._tape = QuantumScript.from_queue(q, shots)
        params = self.tape.get_parameters(trainable_only=False)
        self.tape.trainable_params = qml.math.get_trainable_indices(params)
        _validate_qfunc_output(self.tape, self._qfunc_output)

        return self._tape

    def _impl_call(self, *args, **kwargs) -> qml.typing.Result:
        # construct the tape
        tape = self.construct(args, kwargs)

        # Calculate the classical jacobians if necessary
        #full_transform_program.set_classical_component(self, args, kwargs)
        
        cache = _resolve_cache(self._cache, self._cachesize, self.max_diff)        

        result_batch = execute(
            (tape, ),
            self.device,
            self.default_execution_config,
            user_transform_program= self.transform_program,
            cache = cache
        )
        
        res = result_batch[0]

        # convert result to the interface in case the qfunc has no parameters

        if self.interface != "auto" and (
            len(tape.get_parameters(trainable_only=False)) == 0
            and not self.transform_program.is_informative
        ):
            res = _convert_to_interface(res, self.interface)

        return _to_qfunc_output_type(
            res, self._qfunc_output, tape.shots.has_partitioned_shots
        )


    def __call__(self, *args, **kwargs) -> qml.typing.Result:
        if qml.capture.enabled():
            return qml.capture.qnode_call(self, *args, **kwargs)
        return self._impl_call(*args, **kwargs)


qnode = lambda device, **kwargs: functools.partial(QNode, device=device, **kwargs)
qnode.__doc__ = QNode.__doc__
qnode.__signature__ = inspect.signature(QNode)
