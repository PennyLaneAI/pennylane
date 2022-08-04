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
# pylint: disable=too-many-instance-attributes,too-many-arguments,protected-access
from collections.abc import Sequence
import functools
import inspect
import warnings

import autograd

import pennylane as qml
from pennylane import Device
from pennylane.interfaces import set_shots, SUPPORTED_INTERFACES, INTERFACE_MAP


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
              JAX ``DeviceArray`` objects.

            * ``None``: The QNode accepts default Python types
              (floats, ints, lists, tuples, dicts) as well as NumPy array arguments,
              and returns NumPy arrays. It does not connect to any
              machine learning library automatically for backpropagation.

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

            * ``"finite-diff"``: Uses numerical finite-differences for all quantum operation
              arguments.

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
        mode (str): Whether the gradients should be computed on the forward
            pass (``forward``) or the backward pass (``backward``). Only applies
            if the device is queried for the gradient; gradient transform
            functions available in ``qml.gradients`` are only supported on the backward
            pass.
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
    ...     return expval(qml.PauliZ(0))

    or by instantiating the class directly:

    >>> def circuit(x):
    ...     qml.RX(x, wires=0)
    ...     return expval(qml.PauliZ(0))
    >>> dev = qml.device("default.qubit", wires=1)
    >>> qnode = qml.QNode(circuit, dev)
    """

    def __init__(
        self,
        func,
        device,
        interface="autograd",
        diff_method="best",
        expansion_strategy="gradient",
        max_expansion=10,
        mode="best",
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

        if not isinstance(device, Device):
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

        # input arguments
        self.func = func
        self.device = device
        self._interface = interface
        self.diff_method = diff_method
        self.expansion_strategy = expansion_strategy
        self.max_expansion = max_expansion

        # execution keyword arguments
        self.execute_kwargs = {
            "mode": mode,
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
        self.gradient_kwargs = None
        self._tape_cached = False

        self._update_gradient_fn()
        functools.update_wrapper(self, func)

    def __repr__(self):
        """String representation."""
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

        self._interface = value
        self._update_gradient_fn()

    def _update_gradient_fn(self):
        if self.diff_method is None:
            self._interface = None
            self.gradient_fn = None
            self.gradient_kwargs = {}
            return

        self.gradient_fn, self.gradient_kwargs, self.device = self.get_gradient_fn(
            self._original_device, self.interface, self.diff_method
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
    def get_gradient_fn(device, interface, diff_method="best"):
        """Determine the best differentiation method, interface, and device
        for a requested device, interface, and diff method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface
            diff_method (str or .gradient_transform): The requested method of differentiation.
                If a string, allowed options are ``"best"``, ``"backprop"``, ``"adjoint"``, ``"device"``,
                ``"parameter-shift"``, or ``"finite-diff"``. A gradient transform may
                also be passed here.

        Returns:
            tuple[str or .gradient_transform, dict, .Device: Tuple containing the ``gradient_fn``,
            ``gradient_kwargs``, and the device to use when calling the execute function.
        """

        if diff_method == "best":
            return QNode.get_best_method(device, interface)

        if diff_method == "backprop":
            return QNode._validate_backprop_method(device, interface)

        if diff_method == "adjoint":
            return QNode._validate_adjoint_method(device)

        if diff_method == "device":
            return QNode._validate_device_method(device)

        if diff_method == "parameter-shift":
            return QNode._validate_parameter_shift(device)

        if diff_method == "finite-diff":
            return qml.gradients.finite_diff, {}, device

        if isinstance(diff_method, str):
            raise qml.QuantumFunctionError(
                f"Differentiation method {diff_method} not recognized. Allowed "
                "options are ('best', 'parameter-shift', 'backprop', 'finite-diff', 'device', 'adjoint')."
            )

        if isinstance(diff_method, qml.gradients.gradient_transform):
            return diff_method, {}, device

        raise qml.QuantumFunctionError(
            f"Differentiation method {diff_method} must be a gradient transform or a string."
        )

    @staticmethod
    def get_best_method(device, interface):
        """Returns the 'best' differentiation method
        for a particular device and interface combination.

        This method attempts to determine support for differentiation
        methods using the following order:

        * ``"device"``
        * ``"backprop"``
        * ``"parameter-shift"``
        * ``"finite-diff"``

        The first differentiation method that is supported (going from
        top to bottom) will be returned.

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
                return QNode._validate_backprop_method(device, interface)
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
        top to bottom) will be returned.

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
    def _validate_backprop_method(device, interface):
        if device.shots is not None:
            raise qml.QuantumFunctionError("Backpropagation is only supported when shots=None.")

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

                device = qml.device(
                    backprop_devices[mapped_interface], wires=device.wires, shots=device.shots
                )
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

        required_attrs = ["_apply_operation", "_apply_unitary", "adjoint_jacobian"]
        supported_device = all(hasattr(device, attr) for attr in required_attrs)
        supported_device = supported_device and device.capabilities().get("returns_state")

        if not supported_device:
            raise ValueError(
                f"The {device.short_name} device does not support adjoint differentiation."
            )

        if device.shots is not None:
            warnings.warn(
                "Requested adjoint differentiation to be computed with finite shots."
                " Adjoint differentiation always calculated exactly.",
                UserWarning,
            )

        return "device", {"use_device_state": True, "method": "adjoint_jacobian"}, device

    @staticmethod
    def _validate_device_method(device):
        # determine if the device provides its own jacobian method
        provides_jacobian = device.capabilities().get("provides_jacobian", False)

        if not provides_jacobian:
            raise qml.QuantumFunctionError(
                f"The {device.short_name} device does not provide a native "
                "method for computing the jacobian."
            )

        return "device", {}, device

    @staticmethod
    def _validate_parameter_shift(device):
        model = device.capabilities().get("model", None)

        if model == "qubit":
            return qml.gradients.param_shift, {}, device

        if model == "cv":
            return qml.gradients.param_shift_cv, {"dev": device}, device

        raise qml.QuantumFunctionError(
            f"Device {device.short_name} uses an unknown model ('{model}') "
            "that does not support the parameter-shift rule."
        )

    @property
    def tape(self):
        """The quantum tape"""
        return self._tape

    qtape = tape  # for backwards compatibility

    def construct(self, args, kwargs):
        """Call the quantum function with a tape context, ensuring the operations get queued."""

        self._tape = qml.tape.QuantumTape()

        with self.tape:
            self._qfunc_output = self.func(*args, **kwargs)
        self._tape._qfunc_output = self._qfunc_output

        params = self.tape.get_parameters(trainable_only=False)
        self.tape.trainable_params = qml.math.get_trainable_indices(params)

        if not isinstance(self._qfunc_output, Sequence):
            measurement_processes = (self._qfunc_output,)
        else:
            measurement_processes = self._qfunc_output

        if not all(
            isinstance(m, qml.measurements.MeasurementProcess) for m in measurement_processes
        ):
            raise qml.QuantumFunctionError(
                "A quantum function must return either a single measurement, "
                "or a nonempty sequence of measurements."
            )

        terminal_measurements = [
            m for m in self.tape.measurements if m.return_type != qml.measurements.MidMeasure
        ]
        if not all(ret == m for ret, m in zip(measurement_processes, terminal_measurements)):
            raise qml.QuantumFunctionError(
                "All measurements must be returned in the order they are measured."
            )

        for obj in self.tape.operations + self.tape.observables:

            if getattr(obj, "num_wires", None) is qml.operation.WiresEnum.AllWires:
                # check here only if enough wires
                if len(obj.wires) != self.device.num_wires:
                    raise qml.QuantumFunctionError(f"Operator {obj.name} must act on all wires")

            # pylint: disable=no-member
            if isinstance(obj, qml.ops.qubit.SparseHamiltonian) and self.gradient_fn == "backprop":
                raise qml.QuantumFunctionError(
                    "SparseHamiltonian observable must be used with the parameter-shift"
                    " differentiation method"
                )

        # Apply the deferred measurement principle if the device doesn't
        # support mid-circuit measurements natively
        # TODO:
        # 1. Change once mid-circuit measurements are not considered as tape
        # operations
        # 2. Move this expansion to Device (e.g., default_expand_fn or
        # batch_transform method)
        if any(
            getattr(obs, "return_type", None) == qml.measurements.MidMeasure
            for obs in self.tape.operations
        ):
            self._tape = qml.defer_measurements(self._tape)

        if self.expansion_strategy == "device":
            self._tape = self.device.expand_fn(self.tape, max_expansion=self.max_expansion)

        # If the gradient function is a transform, expand the tape so that
        # all operations are supported by the transform.
        if isinstance(self.gradient_fn, qml.gradients.gradient_transform):
            self._tape = self.gradient_fn.expand_fn(self._tape)

    def __call__(self, *args, **kwargs):
        override_shots = False

        if not self._qfunc_uses_shots_arg:
            # If shots specified in call but not in qfunc signature,
            # interpret it as device shots value for this call.
            override_shots = kwargs.pop("shots", False)

            if override_shots is not False:
                # Since shots has changed, we need to update the preferred gradient function.
                # This is because the gradient function chosen at initialization may
                # no longer be applicable.

                # store the initialization gradient function
                original_grad_fn = [self.gradient_fn, self.gradient_kwargs, self.device]

                # pylint: disable=not-callable
                # update the gradient function
                set_shots(self._original_device, override_shots)(self._update_gradient_fn)()

        # construct the tape
        self.construct(args, kwargs)

        cache = self.execute_kwargs.get("cache", False)
        using_custom_cache = (
            hasattr(cache, "__getitem__")
            and hasattr(cache, "__setitem__")
            and hasattr(cache, "__delitem__")
        )
        self._tape_cached = using_custom_cache and self.tape.hash in cache

        res = qml.execute(
            [self.tape],
            device=self.device,
            gradient_fn=self.gradient_fn,
            interface=self.interface,
            gradient_kwargs=self.gradient_kwargs,
            override_shots=override_shots,
            **self.execute_kwargs,
        )

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

        if (
            not isinstance(self._qfunc_output, Sequence)
            and self._qfunc_output.return_type is qml.measurements.Counts
        ):

            if not self.device._has_partitioned_shots():
                # return a dictionary with counts not as a single-element array
                return res[0]

            return tuple(res)

        if isinstance(self._qfunc_output, Sequence) and any(
            m.return_type is qml.measurements.Counts for m in self._qfunc_output
        ):

            # If Counts was returned with other measurements, then apply the
            # data structure used in the qfunc
            qfunc_output_type = type(self._qfunc_output)
            return qfunc_output_type(res)

        if override_shots is not False:
            # restore the initialization gradient function
            self.gradient_fn, self.gradient_kwargs, self.device = original_grad_fn

        self._update_original_device()

        if isinstance(self._qfunc_output, Sequence) or (
            self.tape.is_sampled and self.device._has_partitioned_shots()
        ):
            return res

        if self._qfunc_output.return_type is qml.measurements.Shadow:
            # if classical shadows is returned, then don't squeeze the
            # last axis corresponding to the number of qubits
            return qml.math.squeeze(res, axis=0)

        # Squeeze arraylike outputs
        return qml.math.squeeze(res)


qnode = lambda device, **kwargs: functools.partial(QNode, device=device, **kwargs)
qnode.__doc__ = QNode.__doc__
qnode.__signature__ = inspect.signature(QNode)
