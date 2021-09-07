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

import pennylane as qml
from pennylane import Device
from pennylane.interfaces.batch import set_shots, SUPPORTED_INTERFACES
from pennylane.operation import State


class QNode:
    """New QNode"""

    def __init__(
        self,
        func,
        device,
        interface="autograd",
        diff_method="best",
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
        self.max_expansion = max_expansion

        # execution keyword arguments
        self.execute_kwargs = {
            "mode": mode,
            "cache": cache,
            "cachesize": cachesize,
            "max_diff": max_diff,
        }

        # internal data attributes
        self._tape = None
        self._qfunc_output = None
        self._gradient_kwargs = gradient_kwargs
        self._original_device = device
        self.gradient_fn = None
        self.gradient_kwargs = None

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
                f"Unknown interface {value}. Interface must be " f"one of {SUPPORTED_INTERFACES}."
            )

        self._interface = value
        self._update_gradient_fn()

    def _update_gradient_fn(self):
        self.gradient_fn, self.gradient_kwargs, self.device = self.get_gradient_fn(
            self._original_device, self.interface, self.diff_method
        )
        self.gradient_kwargs.update(self._gradient_kwargs or {})

    def _update_original_device(self):
        # FIX: If the qnode swapped the device, increase the num_execution value on the original device.
        # In the long run, we should make sure that the user's device is the one
        # actually run so she has full control. This could be done by changing the class
        # of the user's device before and after executing the tape.

        if self.device is not self._original_device:
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
                If a string, one of ``"best"``, ``"backprop"``, ``"adjoint"``, ``"device"``,
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
                "options are ('best', 'parameter-shift', 'backprop', 'finite-diff', 'device', 'reversible', 'adjoint')."
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
    def _validate_backprop_method(device, interface):
        # determine if the device supports backpropagation
        backprop_interface = device.capabilities().get("passthru_interface", None)

        # determine if the device has any child devices that support backpropagation
        backprop_devices = device.capabilities().get("passthru_devices", None)

        if backprop_interface is not None:
            # device supports backpropagation natively

            if interface == backprop_interface:
                return "backprop", {}, device

            raise qml.QuantumFunctionError(
                f"Device {device.short_name} only supports diff_method='backprop' when using the "
                f"{backprop_interface} interface."
            )

        if device.shots is None and backprop_devices is not None:

            # device is analytic and has child devices that support backpropagation natively

            if interface in backprop_devices:
                # TODO: need a better way of passing existing device init options
                # to a new device?
                device = qml.device(
                    backprop_devices[interface],
                    wires=device.wires,
                    shots=device.shots,
                )
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

        supported_device = hasattr(device, "_apply_operation")
        supported_device = supported_device and hasattr(device, "_apply_unitary")
        supported_device = supported_device and device.capabilities().get("returns_state")
        supported_device = supported_device and hasattr(device, "adjoint_jacobian")

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

        if self.interface == "autograd":
            # HOTFIX: to maintain compatibility with core, here we treat
            # all inputs that do not explicitly specify `requires_grad=False`
            # as trainable. This should be removed at some point, forcing users
            # to specify `requires_grad=True` for trainable parameters.
            args = [
                qml.numpy.array(a, requires_grad=True) if not hasattr(a, "requires_grad") else a
                for a in args
            ]

        self._tape = qml.tape.JacobianTape()

        with self.tape:
            self._qfunc_output = self.func(*args, **kwargs)

        params = self.tape.get_parameters(trainable_only=False)
        self.tape.trainable_params = qml.math.get_trainable_indices(params)

        if not isinstance(self._qfunc_output, Sequence):
            measurement_processes = (self._qfunc_output,)
        else:
            measurement_processes = self._qfunc_output

        if not all(isinstance(m, qml.measure.MeasurementProcess) for m in measurement_processes):
            raise qml.QuantumFunctionError(
                "A quantum function must return either a single measurement, "
                "or a nonempty sequence of measurements."
            )

        state_returns = any(m.return_type is State for m in measurement_processes)

        # TODO: pass complex128 to Torch/TF if state is being returned. Move this to the
        # execute() function.

        if not all(ret == m for ret, m in zip(measurement_processes, self.tape.measurements)):
            raise qml.QuantumFunctionError(
                "All measurements must be returned in the order they are measured."
            )

        for obj in self.tape.operations + self.tape.observables:

            if getattr(obj, "num_wires", None) is qml.operation.WiresEnum.AllWires:
                # check here only if enough wires
                if len(obj.wires) != self.device.num_wires:
                    raise qml.QuantumFunctionError(
                        "Operator {} must act on all wires".format(obj.name)
                    )

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

                # update the gradient function
                set_shots(self._original_device, override_shots)(self._update_gradient_fn)()

        # construct the tape
        self.construct(args, kwargs)

        res = qml.execute(
            [self.tape],
            device=self.device,
            gradient_fn=self.gradient_fn,
            interface=self.interface,
            gradient_kwargs=self.gradient_kwargs,
            override_shots=override_shots,
            **self.execute_kwargs,
        )[0]

        if override_shots is not False:
            # restore the initialization gradient function
            self.gradient_fn, self.gradient_kwargs, self.device = original_grad_fn

        self._update_original_device()

        if isinstance(self._qfunc_output, Sequence) or (
            self.tape.is_sampled and self.device._has_partitioned_shots()
        ):
            return res

        return qml.math.squeeze(res)


qnode = lambda dev, **kwargs: functools.partial(QNode, device=dev, **kwargs)
qnode = functools.update_wrapper(qnode, QNode)
