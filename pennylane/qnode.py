# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
import functools
import inspect
import warnings
from collections.abc import Sequence
from typing import Union


import autograd

import pennylane as qml
from pennylane import Device
from pennylane.interfaces import INTERFACE_MAP, SUPPORTED_INTERFACES, set_shots
from pennylane.measurements import ClassicalShadowMP, CountsMP, MidMeasureMP
from pennylane.tape import QuantumTape, make_qscript


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
            ``qml.interfaces.SUPPORTED_INTERFACES`` for a list of all accepted strings.

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

        diff_method (str or .gradient_transform): The method of differentiation to use in the created QNode.
            Can either be a :class:`~.gradient_transform`, which includes all quantum gradient
            transforms in the :mod:`qml.gradients <.gradients>` module, or a string. The following
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
        mode (str): Deprecated kwarg indicating whether the gradients should be computed on the forward
            pass (``forward``) or the backward pass (``backward``). Only applies
            if the device is queried for the gradient; gradient transform
            functions available in ``qml.gradients`` are only supported on the backward
            pass. This argument does nothing with the new return system, and users should
            instead set ``grad_on_execution`` to indicate their desired behaviour.
        cache (bool or dict or Cache): Whether to cache evaluations. This can result in
            a significant reduction in quantum evaluations during gradient computations.
            If ``True``, a cache with corresponding ``cachesize`` is created for each batch
            execution. If ``False``, no caching is used. You may also pass your own cache
            to be used; this can be any object that implements the special methods
            ``__getitem__()``, ``__setitem__()``, and ``__delitem__()``, such as a dictionary.
        cachesize (int): The size of any auto-created caches. Only applies when ``cache=True``.
        max_diff (int): If ``diff_method`` is a gradient transform, this option specifies
            the maximum number of derivatives to support. Increasing this value allows
            for higher order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backwards pass.

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
    ...     return qml.expval(qml.PauliZ(0))

    or by instantiating the class directly:

    >>> def circuit(x):
    ...     qml.RX(x, wires=0)
    ...     return qml.expval(qml.PauliZ(0))
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
        ...     return qml.expval(qml.PauliZ(0))

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
                return qml.expval(qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliZ(2) @ qml.PauliZ(3))


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
                return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

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
        device: Union[Device, "qml.devices.experimental.Device"],
        interface="auto",
        diff_method="best",
        shots=None,
        expansion_strategy="gradient",
        max_expansion=10,
        grad_on_execution="best",
        mode=None,
        cache=True,
        cachesize=10000,
        max_diff=1,
        **gradient_kwargs,
    ):
        if interface not in SUPPORTED_INTERFACES:
            raise qml.QuantumFunctionError(
                f"Unknown interface {interface}. Interface must be "
                f"one of {SUPPORTED_INTERFACES}."
            )

        if not isinstance(device, (Device, qml.devices.experimental.Device)):
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

        if mode is None:
            mode = "best"
        elif qml.active_return():
            warnings.warn(
                "The `mode` keyword argument is deprecated and does nothing with the new return system. "
                "Please use `grad_on_execution` instead.",
                UserWarning,
            )
        else:
            warnings.warn(
                "The `mode` keyword argument is deprecated, along with the old return system. In "
                "the new return system, you should set the `grad_on_execution` boolean instead.",
                UserWarning,
            )

        if (
            hasattr(device, "shots")
            and device.shots != shots
            and device.shots is not None
            and shots is None
        ):
            warnings.warn(
                "Shots should now be specified on the qnode instead of on the device."
                "Using shots from the device. QNode specified shots will be used in v0.33."
            )
            shots = device.shots

        # input arguments
        self.func = func
        self.device = device
        self._interface = interface
        self.diff_method = diff_method
        self.expansion_strategy = expansion_strategy
        self.max_expansion = max_expansion
        self._shots = shots

        # execution keyword arguments
        self.execute_kwargs = {
            "mode": mode,
            "grad_on_execution": grad_on_execution,
            "cache": cache,
            "cachesize": cachesize,
            "max_diff": max_diff,
            "max_expansion": max_expansion,
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

        self._update_gradient_fn()
        functools.update_wrapper(self, func)
        self._transform_program = qml.transforms.core.TransformProgram()

    def __repr__(self):
        """String representation."""
        if isinstance(self.device, qml.devices.experimental.Device):
            return f"<QNode: device='{self.device}', interface='{self.interface}', diff_method='{self.diff_method}'>"

        detail = "<QNode: wires={}, device='{}', interface='{}', diff_method='{}'>"
        return detail.format(
            self.device.num_wires,
            self.device.short_name,
            self.interface,
            self.diff_method,
        )

    @property
    def default_shots(self) -> qml.measurements.Shots:
        """The default shots to use for an execution.

        Can be overridden on a per-call basis using the ``qnode(*args, shots=new_shots, **kwargs)`` syntax.

        """
        return qml.measurements.Shots(self._shots)

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
        self._update_gradient_fn()

    @property
    def transform_program(self):
        """The transform program used by the QNode.

        .. warning:: This is an experimental feature.
        """
        return self._transform_program

    def add_transform(self, transform_container):
        """Add a transform container to the transform program.

        .. warning:: This is an experimental feature.
        """
        self._transform_program.push_back(transform_container=transform_container)

    def _update_gradient_fn(self, shots=None):
        if self.diff_method is None:
            self._interface = None
            self.gradient_fn = None
            self.gradient_kwargs = {}
            return
        if self.interface == "auto" and self.diff_method in ["backprop", "best"]:
            if self.diff_method == "backprop":
                # Check that the device has the capabilities to support backprop
                if isinstance(self.device, Device):
                    backprop_devices = self.device.capabilities().get("passthru_devices", None)
                    if backprop_devices is None:
                        raise qml.QuantumFunctionError(
                            f"The {self.device.short_name} device does not support native computations with "
                            "autodifferentiation frameworks."
                        )
            return

        self.gradient_fn, self.gradient_kwargs, self.device = self.get_gradient_fn(
            self._original_device, self.interface, self.diff_method, shots=shots
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
    def get_gradient_fn(device, interface, diff_method="best", shots=None):
        """Determine the best differentiation method, interface, and device
        for a requested device, interface, and diff method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface
            diff_method (str or .gradient_transform): The requested method of differentiation.
                If a string, allowed options are ``"best"``, ``"backprop"``, ``"adjoint"``,
                ``"device"``, ``"parameter-shift"``, ``"hadamard"``, ``"finite-diff"``, or ``"spsa"``.
                A gradient transform may also be passed here.

        Returns:
            tuple[str or .gradient_transform, dict, .Device: Tuple containing the ``gradient_fn``,
            ``gradient_kwargs``, and the device to use when calling the execute function.
        """
        if diff_method == "best":
            return QNode.get_best_method(device, interface, shots=shots)

        if diff_method == "backprop":
            return QNode._validate_backprop_method(device, interface, shots=shots)

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

        if isinstance(diff_method, qml.gradients.gradient_transform):
            return diff_method, {}, device

        raise qml.QuantumFunctionError(
            f"Differentiation method {diff_method} must be a gradient transform or a string."
        )

    @staticmethod
    def get_best_method(device, interface, shots=None):
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

        Returns:
            tuple[str or .gradient_transform, dict, .Device: Tuple containing the ``gradient_fn``,
            ``gradient_kwargs``, and the device to use when calling the execute function.
        """
        try:
            return QNode._validate_device_method(device)
        except qml.QuantumFunctionError:
            try:
                return QNode._validate_backprop_method(device, interface, shots=shots)
            except qml.QuantumFunctionError:
                try:
                    return QNode._validate_parameter_shift(device)
                except qml.QuantumFunctionError:
                    return qml.gradients.finite_diff, {}, device

    @staticmethod
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
    def _validate_backprop_method(device, interface, shots=None):
        if shots is not None or getattr(device, "shots", None) is not None:
            raise qml.QuantumFunctionError("Backpropagation is only supported when shots=None.")

        if isinstance(device, qml.devices.experimental.Device):
            config = qml.devices.experimental.ExecutionConfig(
                gradient_method="backprop", interface=interface
            )
            if device.supports_derivatives(config):
                return "backprop", {}, device
            raise qml.QuantumFunctionError(f"Device {device.name} does not support backprop")

        mapped_interface = INTERFACE_MAP.get(interface, interface)

        # determine if the device supports backpropagation
        backprop_interface = device.capabilities().get("passthru_interface", None)

        if backprop_interface is not None:
            # device supports backpropagation natively

            if mapped_interface == backprop_interface:
                return "backprop", {}, device

            raise qml.QuantumFunctionError(
                f"Device {device.short_name} only supports diff_method='backprop' when using the "
                f"{backprop_interface} interface."
            )

        # determine if the device has any child devices that support backpropagation
        backprop_devices = device.capabilities().get("passthru_devices", None)

        if backprop_devices is not None:
            # device is analytic and has child devices that support backpropagation natively

            if mapped_interface in backprop_devices:
                # no need to create another device if the child device is the same (e.g., default.mixed)
                if backprop_devices[mapped_interface] == device.short_name:
                    return "backprop", {}, device

                # TODO: need a better way of passing existing device init options
                # to a new device?
                expand_fn = device.expand_fn
                batch_transform = device.batch_transform

                device = qml.device(backprop_devices[mapped_interface], wires=device.wires)
                device.expand_fn = expand_fn
                device.batch_transform = batch_transform

                return "backprop", {}, device

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

        if isinstance(device, qml.devices.experimental.Device):
            config = qml.devices.experimental.ExecutionConfig(
                gradient_method="adjoint", use_device_gradient=True
            )
            if device.supports_derivatives(config):
                return "adjoint", {}, device
            raise ValueError(f"The {device} device does not support adjoint differentiation.")
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
        if isinstance(device, Device):
            # determine if the device provides its own jacobian method
            if device.capabilities().get("provides_jacobian", False):
                return "device", {}, device
            name = device.short_name
        else:
            config = qml.devices.experimental.ExecutionConfig(gradient_method="device")
            if device.supports_derivatives(config):
                return "device", {}, device
            name = device.name

        raise qml.QuantumFunctionError(
            f"The {name} device does not provide a native " "method for computing the jacobian."
        )

    @staticmethod
    def _validate_parameter_shift(device):
        if isinstance(device, qml.devices.experimental.Device):
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

    def construct(self, args, kwargs, override_shots="unset"):  # pylint: disable=too-many-branches
        """Call the quantum function with a tape context, ensuring the operations get queued."""
        old_interface = self.interface

        override_shots = self._shots if isinstance(override_shots, str) else override_shots

        if old_interface == "auto":
            self.interface = qml.math.get_interface(*args, *list(kwargs.values()))

        self._tape = make_qscript(self.func, override_shots)(*args, **kwargs)
        self._qfunc_output = self.tape._qfunc_output

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
        if any(ret != m for ret, m in zip(measurement_processes, terminal_measurements)):
            raise qml.QuantumFunctionError(
                "All measurements must be returned in the order they are measured."
            )

        for obj in self.tape.operations + self.tape.observables:
            if (
                getattr(obj, "num_wires", None) is qml.operation.WiresEnum.AllWires
                and len(obj.wires) != self.device.num_wires
            ):
                # check here only if enough wires
                raise qml.QuantumFunctionError(f"Operator {obj.name} must act on all wires")

            # pylint: disable=no-member
            if isinstance(obj, qml.ops.qubit.SparseHamiltonian) and self.gradient_fn == "backprop":
                raise qml.QuantumFunctionError(
                    "SparseHamiltonian observable must be used with the parameter-shift "
                    "differentiation method"
                )

        # Apply the deferred measurement principle if the device doesn't
        # support mid-circuit measurements natively
        # TODO:
        # 1. Change once mid-circuit measurements are not considered as tape
        # operations
        # 2. Move this expansion to Device (e.g., default_expand_fn or
        # batch_transform method)
        if any(isinstance(m, MidMeasureMP) for m in self.tape.operations):
            self._tape = qml.defer_measurements(self._tape)

        if self.expansion_strategy == "device":
            self._tape = self.device.expand_fn(self.tape, max_expansion=self.max_expansion)

        # If the gradient function is a transform, expand the tape so that
        # all operations are supported by the transform.
        if isinstance(self.gradient_fn, qml.gradients.gradient_transform):
            self._tape = self.gradient_fn.expand_fn(self._tape)

        if old_interface == "auto":
            self.interface = "auto"

    def __call__(self, *args, **kwargs) -> qml.typing.Result:
        override_shots = False
        old_interface = self.interface

        if old_interface == "auto":
            self.interface = qml.math.get_interface(*args, *list(kwargs.values()))
            self.device.tracker = self._original_device.tracker

        original_grad_fn = [self.gradient_fn, self.gradient_kwargs, self.device]
        override_shots = self._shots
        if not self._qfunc_uses_shots_arg:
            if "shots" in kwargs:
                override_shots = kwargs.pop("shots")
                # Since shots has changed, we need to update the preferred gradient function.
                # This is because the gradient function chosen at initialization may
                # no longer be applicable.

                # pylint: disable=not-callable
                # update the gradient function
                if isinstance(self._original_device, Device):
                    set_shots(self._original_device, override_shots)(self._update_gradient_fn)()
                else:
                    self._update_gradient_fn(shots=override_shots)

        # construct the tape
        self.construct(args, kwargs, override_shots=override_shots)

        cache = self.execute_kwargs.get("cache", False)
        using_custom_cache = (
            hasattr(cache, "__getitem__")
            and hasattr(cache, "__setitem__")
            and hasattr(cache, "__delitem__")
        )
        self._tape_cached = using_custom_cache and self.tape.hash in cache

        if qml.active_return():
            if "mode" in self.execute_kwargs:
                self.execute_kwargs.pop("mode")
            # pylint: disable=unexpected-keyword-arg
            res = qml.execute(
                [self.tape],
                device=self.device,
                gradient_fn=self.gradient_fn,
                interface=self.interface,
                gradient_kwargs=self.gradient_kwargs,
                override_shots=override_shots,
                **self.execute_kwargs,
            )

            res = res[0]

            # convert result to the interface in case the qfunc has no parameters

            if len(self.tape.get_parameters(trainable_only=False)) == 0:
                res = _convert_to_interface(res, self.interface)

            if old_interface == "auto":
                self.interface = "auto"

            # Special case of single Measurement in a list
            if isinstance(self._qfunc_output, list) and len(self._qfunc_output) == 1:
                return [res]

            # If the return type is not tuple (list or ndarray) (Autograd and TF backprop removed)
            if not isinstance(self._qfunc_output, (tuple, qml.measurements.MeasurementProcess)):
                has_partitioned_shots = (
                    self.tape.shots.has_partitioned_shots
                    if isinstance(self.device, qml.devices.experimental.Device)
                    else self.device._shot_vector
                )
                if has_partitioned_shots:
                    res = [type(self.tape._qfunc_output)(r) for r in res]
                    res = tuple(res)
                else:
                    res = type(self.tape._qfunc_output)(res)

            # restore the initialization gradient function
            self.gradient_fn, self.gradient_kwargs, self.device = original_grad_fn

            self._update_original_device()

            return res
        if "mode" in self.execute_kwargs:
            mode = self.execute_kwargs.pop("mode")
            if mode == "forward":
                grad_on_execution = True
            elif mode == "backward":
                grad_on_execution = False
            else:
                grad_on_execution = "best"
            self.execute_kwargs["grad_on_execution"] = grad_on_execution
        # pylint: disable=unexpected-keyword-arg
        res = qml.execute(
            [self.tape],
            device=self.device,
            gradient_fn=self.gradient_fn,
            interface=self.interface,
            gradient_kwargs=self.gradient_kwargs,
            override_shots=override_shots,
            **self.execute_kwargs,
        )

        if old_interface == "auto":
            self.interface = "auto"

        if autograd.isinstance(res, (tuple, list)) and len(res) == 1:
            # If a device batch transform was applied, we need to 'unpack'
            # the returned tuple/list to a float.
            #
            # Note that we use autograd.isinstance, because on the backwards pass
            # with Autograd, lists and tuples are converted to autograd.box.SequenceBox.
            # autograd.isinstance is a 'safer' isinstance check that supports
            # autograd backwards passes.
            #
            # TODO: find a more explicit way of determining that a batch transform
            # was applied.

            res = res[0]

        if not isinstance(self._qfunc_output, Sequence) and isinstance(
            self._qfunc_output, CountsMP
        ):
            if self.device._has_partitioned_shots():
                return tuple(res)

            # return a dictionary with counts not as a single-element array
            return res[0]

        if isinstance(self._qfunc_output, Sequence) and any(
            isinstance(m, CountsMP) for m in self._qfunc_output
        ):
            # If Counts was returned with other measurements, then apply the
            # data structure used in the qfunc
            qfunc_output_type = type(self._qfunc_output)
            return qfunc_output_type(res)

        self.gradient_fn, self.gradient_kwargs, self.device = original_grad_fn

        self._update_original_device()

        if isinstance(self._qfunc_output, Sequence) or (
            self.tape.is_sampled and self.device._has_partitioned_shots()
        ):
            return res

        if isinstance(self._qfunc_output, ClassicalShadowMP):
            # if classical shadows is returned, then don't squeeze the
            # last axis corresponding to the number of qubits
            return qml.math.squeeze(res, axis=0)

        # Squeeze arraylike outputs
        return qml.math.squeeze(res)


qnode = lambda device, **kwargs: functools.partial(QNode, device=device, **kwargs)
qnode.__doc__ = QNode.__doc__
qnode.__signature__ = inspect.signature(QNode)
