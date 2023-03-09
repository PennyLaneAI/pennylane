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
# pylint: disable=too-many-instance-attributes,too-many-arguments,protected-access,unnecessary-lambda-assignment
import functools
import inspect
import warnings
from collections.abc import Sequence

import autograd

import pennylane as qml
from pennylane import Device
from pennylane.interfaces import INTERFACE_MAP, SUPPORTED_INTERFACES, set_shots
from pennylane.measurements import ClassicalShadowMP, CountsMP, MidMeasureMP
from pennylane.tape import QuantumTape, make_qscript


class QNodeExperimental:
    """Represents a quantum node in the hybrid computational graph."""

    def __init__(
        self,
        func,
        device,
        interface="auto",
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

        for kwarg in gradient_kwargs:
            if kwarg in ["gradient_fn", "grad_method"]:
                warnings.warn(
                    f"It appears you may be trying to set the method of differentiation via the kwarg "
                    f"{kwarg}. This is not supported in qnode and will defualt to backpropogation. Use "
                    f"diff_method instead."
                )
            elif kwarg not in qml.gradients.SUPPORTED_GRADIENT_KWARGS:
                warnings.warn(
                    f"Received gradient_kwarg {kwarg}, which is not included in the list of standard qnode "
                    f"gradient kwargs."
                )

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
        self.gradient_kwargs = {}
        self._tape_cached = False

        self._update_gradient_fn()
        functools.update_wrapper(self, func)

        self.transform_program = qml.transforms.experimental.TransformProgram()

    def __repr__(self):
        """String representation."""
        detail = "<QNodeExperimental: wires={}, device='{}', interface='{}', diff_method='{}'>"
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
        self._update_gradient_fn()

    def _update_gradient_fn(self):
        if self.diff_method is None:
            self._interface = None
            self.gradient_fn = None
            self.gradient_kwargs = {}
            return
        if self.interface == "auto" and self.diff_method in ["backprop", "best"]:
            if self.diff_method == "backprop":
                # Check that the device has the capabilities to support backprop
                backprop_devices = self.device.capabilities().get("passthru_devices", None)
                if backprop_devices is None:
                    raise qml.QuantumFunctionError(
                        f"The {self.device.short_name} device does not support native computations with "
                        "autodifferentiation frameworks."
                    )
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
                If a string, allowed options are ``"best"``, ``"backprop"``, ``"adjoint"``,
                ``"device"``, ``"parameter-shift"``, ``"hadamard"``, ``"finite-diff"``, or ``"spsa"``.
                A gradient transform may also be passed here.

        Returns:
            tuple[str or .gradient_transform, dict, .Device: Tuple containing the ``gradient_fn``,
            ``gradient_kwargs``, and the device to use when calling the execute function.
        """
        if diff_method == "best":
            return QNodeExperimental.get_best_method(device, interface)

        if diff_method == "backprop":
            return QNodeExperimental._validate_backprop_method(device, interface)

        if diff_method == "adjoint":
            return QNodeExperimental._validate_adjoint_method(device)

        if diff_method == "device":
            return QNodeExperimental._validate_device_method(device)

        if diff_method == "parameter-shift":
            return QNodeExperimental._validate_parameter_shift(device)

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
            return QNodeExperimental._validate_device_method(device)
        except qml.QuantumFunctionError:
            try:
                return QNodeExperimental._validate_backprop_method(device, interface)
            except qml.QuantumFunctionError:
                try:
                    return QNodeExperimental._validate_parameter_shift(device)
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
        transform = QNodeExperimental.get_best_method(device, interface)[0]

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
                "Requested adjoint differentiation to be computed with finite shots. "
                "Adjoint differentiation always calculated exactly.",
                UserWarning,
            )
        return "device", {"use_device_state": True, "method": "adjoint_jacobian"}, device

    @staticmethod
    def _validate_device_method(device):
        # determine if the device provides its own jacobian method
        if device.capabilities().get("provides_jacobian", False):
            return "device", {}, device

        raise qml.QuantumFunctionError(
            f"The {device.short_name} device does not provide a native "
            "method for computing the jacobian."
        )

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
    def tape(self) -> QuantumTape:
        """The quantum tape"""
        return self._tape

    qtape = tape  # for backwards compatibility

    def construct(self, args, kwargs):  # pylint: disable=too-many-branches
        """Call the quantum function with a tape context, ensuring the operations get queued."""
        old_interface = self.interface

        if old_interface == "auto":
            self.interface = qml.math.get_interface(*args, *list(kwargs.values()))

        self._tape = make_qscript(self.func)(*args, **kwargs)
        self._qfunc_output = self.tape._qfunc_output

        params = self.tape.get_parameters(trainable_only=False)
        self.tape.trainable_params = qml.math.get_trainable_indices(params)

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

    def __call__(self, *args, **kwargs):  # pylint: disable=too-many-branches, too-many-statements
        override_shots = False
        old_interface = self.interface

        if old_interface == "auto":
            self.interface = qml.math.get_interface(*args, *list(kwargs.values()))
            self.device.tracker = self._original_device.tracker

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

        res = qml.interfaces.execute_experimental(
            [self.tape], device=self.device, transforms_program=self.transform_program
        )
        res = res[0]

        if old_interface == "auto":
            self.interface = "auto"

        # Special case of single Measurement in a list
        if isinstance(self._qfunc_output, list) and len(self._qfunc_output) == 1:
            return [res]

        # If the return type is not tuple (list or ndarray) (Autograd and TF backprop removed)
        if not isinstance(self._qfunc_output, (tuple, qml.measurements.MeasurementProcess)):
            if self.device._shot_vector:
                res = [type(self.tape._qfunc_output)(r) for r in res]
                res = tuple(res)

            else:
                res = type(self.tape._qfunc_output)(res)

        if override_shots is not False:
            # restore the initialization gradient function
            self.gradient_fn, self.gradient_kwargs, self.device = original_grad_fn

        self._update_original_device()

        # TODO 3: Apply post QNode processing from transform

        return res


qnode_experimental = lambda device, **kwargs: functools.partial(
    QNodeExperimental, device=device, **kwargs
)
qnode_experimental.__doc__ = QNodeExperimental.__doc__
qnode_experimental.__signature__ = inspect.signature(QNodeExperimental)
