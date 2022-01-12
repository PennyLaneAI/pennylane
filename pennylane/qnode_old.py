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
# pylint: disable=import-outside-toplevel
# pylint:disable=too-many-branches
from collections.abc import Sequence
from functools import lru_cache, update_wrapper
import warnings
import inspect

import numpy as np

import pennylane as qml
from pennylane import Device

from pennylane.operation import State

from pennylane.interfaces.autograd import AutogradInterface, np as anp
from pennylane.tape import (
    JacobianTape,
    QubitParamShiftTape,
    CVParamShiftTape,
    ReversibleTape,
)


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
            This affects the types of objects that can be passed to/returned from the QNode:

            * ``"autograd"``: Allows autograd to backpropagate
              through the QNode. The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays.

            * ``"jax"``:  Allows JAX to backpropogate through the QNode.
              The QNode accepts and returns a single expectation value or variance.

            * ``"torch"``: Allows PyTorch to backpropogate
              through the QNode. The QNode accepts and returns Torch tensors.

            * ``"tf"``: Allows TensorFlow in eager mode to backpropogate
              through the QNode. The QNode accepts and returns
              TensorFlow ``tf.Variable`` and ``tf.tensor`` objects.

            * ``None``: The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays. It does not connect to any
              machine learning library automatically for backpropagation.

        diff_method (str): the method of differentiation to use in the created QNode

            * ``"best"``: Best available method. Uses classical backpropagation or the
              device directly to compute the gradient if supported, otherwise will use
              the analytic parameter-shift rule where possible with finite-difference as a fallback.

            * ``"device"``: Queries the device directly for the gradient.
              Only allowed on devices that provide their own gradient computation.

            * ``"backprop"``: Use classical backpropagation. Only allowed on simulator
              devices that are classically end-to-end differentiable, for example
              :class:`default.tensor.tf <~.DefaultTensorTF>`. Note that the returned
              QNode can only be used with the machine-learning framework supported
              by the device.

            * ``"reversible"``: Uses a reversible method for computing the gradient.
              This method is similar to ``"backprop"``, but trades off increased
              runtime with significantly lower memory usage. Compared to the
              parameter-shift rule, the reversible method can be faster or slower,
              depending on the density and location of parametrized gates in a circuit.
              Only allowed on (simulator) devices with the "reversible" capability,
              for example :class:`default.qubit <~.DefaultQubit>`.

            * ``"adjoint"``: Uses an `adjoint method <https://arxiv.org/abs/2009.02823>`__ that
              reverses through the circuit after a forward pass by iteratively applying the inverse
              (adjoint) gate. This method is similar to the reversible method, but has a lower time
              overhead and a similar memory overhead. Only allowed on simulator devices such as
              :class:`default.qubit <~.DefaultQubit>`.

            * ``"parameter-shift"``: Use the analytic parameter-shift
              rule for all supported quantum operation arguments, with finite-difference
              as a fallback.

            * ``"finite-diff"``: Uses numerical finite-differences for all quantum operation
              arguments.

            * ``None``: QNode cannot be differentiated. Works the same as ``interface=None``.

        mutable (bool): If True, the underlying quantum circuit is re-constructed with
            every evaluation. This is the recommended approach, as it allows the underlying
            quantum structure to depend on (potentially trainable) QNode input arguments,
            however may add some overhead at evaluation time. If this is set to False, the
            quantum structure will only be constructed on the *first* evaluation of the QNode,
            and is stored and re-used for further quantum evaluations. Only set this to False
            if it is known that the underlying quantum structure is **independent of QNode input**.
        max_expansion (int): The number of times the internal circuit should be expanded when
            executed on a device. Expansion occurs when an operation or measurement is not
            supported, and results in a gate decomposition. If any operations in the decomposition
            remain unsupported by the device, another expansion occurs.
        h (float): step size for the finite difference method
        order (int): The order of the finite difference method to use. ``1`` corresponds
            to forward finite differences, ``2`` to centered finite differences.
        shift (float): the size of the shift for two-term parameter-shift gradient computations
        adjoint_cache (bool): For TensorFlow and PyTorch interfaces and adjoint differentiation,
            this indicates whether to save the device state after the forward pass.  Doing so saves a
            forward execution. Device state automatically reused with autograd and JAX interfaces.
        argnum (int, list(int), None): Which argument(s) to compute the Jacobian
            with respect to. When there are fewer parameters specified than the
            total number of trainable parameters, the jacobian is being estimated. Note
            that this option is only applicable for the following differentiation methods:
            ``"parameter-shift"``, ``"finite-diff"`` and ``"reversible"``.
        kwargs: used to catch all unrecognized keyword arguments and provide a user warning
            about them

    **Example**

    >>> def circuit(x):
    ...     qml.RX(x, wires=0)
    ...     return expval(qml.PauliZ(0))
    >>> dev = qml.device("default.qubit", wires=1)
    >>> qnode = qml.QNode(circuit, dev)
    """

    # pylint:disable=too-many-instance-attributes,too-many-arguments

    def __init__(
        self,
        func,
        device,
        interface="autograd",
        diff_method="best",
        mutable=True,
        max_expansion=10,
        h=1e-7,
        order=1,
        shift=np.pi / 2,
        adjoint_cache=True,
        argnum=None,
        **kwargs,
    ):

        if diff_method is None:
            # TODO: update this behaviour once the new differentiable pipeline is the default
            # We set "best" to allow internals to work and leverage the interface = None
            # feature to restrict differentiability
            diff_method = "best"
            interface = None

        if interface is not None and interface not in self.INTERFACE_MAP:
            raise qml.QuantumFunctionError(
                f"Unknown interface {interface}. Interface must be "
                f"one of {list(self.INTERFACE_MAP.keys())}."
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

        if kwargs:
            for key in kwargs:
                warnings.warn(
                    f"'{key}' is unrecognized, and will not be included in your computation. "
                    "Please review the QNode class or qnode decorator for the list of available "
                    "keyword variables.",
                    UserWarning,
                )

        diff_options = {
            "h": h,
            "order": order,
            "shift": shift,
            "adjoint_cache": adjoint_cache,
            "argnum": argnum,
        }

        self.mutable = mutable
        self.func = func
        self._original_device = device
        self.qtape = None
        self.qfunc_output = None
        # store the user-specified differentiation method
        self.diff_method = diff_method
        self.diff_method_change = False

        self._tape, self.interface, self.device, tape_diff_options = self.get_tape(
            device, interface, diff_method
        )
        # if diff_method is best, then set it to the actual diff method being used
        if self.diff_method == "best":
            self.diff_method_change = True
            self.diff_method = self._get_best_diff_method(tape_diff_options)

        # The arguments to be passed to JacobianTape.jacobian
        self.diff_options = diff_options
        self.diff_options.update(tape_diff_options)

        self.dtype = np.float64
        self.max_expansion = max_expansion

    def __repr__(self):
        """String representation."""
        detail = "<QNode: wires={}, device='{}', interface='{}', diff_method='{}'>"
        return detail.format(
            self.device.num_wires,
            self.device.short_name,
            self.interface,
            self.diff_method,
        )

    @staticmethod
    def _get_best_diff_method(tape_diff_options):
        """Update diff_method to reflect which method has been selected"""
        if tape_diff_options["method"] == "device":
            method = "device"
        elif tape_diff_options["method"] == "backprop":
            method = "backprop"
        elif tape_diff_options["method"] == "best":
            method = "parameter-shift"
        elif tape_diff_options["method"] == "numeric":
            method = "finite-diff"
        return method

    # pylint: disable=too-many-return-statements
    @staticmethod
    def get_tape(device, interface, diff_method="best"):
        """Determine the best JacobianTape, differentiation method, interface, and device
        for a requested device, interface, and diff method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface
            diff_method (str): The requested method of differentiation. One of
                ``"best"``, ``"backprop"``, ``"reversible"``, ``"adjoint"``, ``"device"``,
                ``"parameter-shift"``, or ``"finite-diff"``.

        Returns:
            tuple[.JacobianTape, str, .Device, dict[str, str]]: Tuple containing the compatible
            JacobianTape, the interface to apply, the device to use, and the method argument
            to pass to the ``JacobianTape.jacobian`` method.
        """

        if diff_method == "best":
            return QNode.get_best_method(device, interface)

        if diff_method == "backprop":
            return QNode._validate_backprop_method(device, interface)

        if diff_method == "reversible":
            return QNode._validate_reversible_method(device, interface)

        if diff_method == "adjoint":
            return QNode._validate_adjoint_method(device, interface)

        if diff_method == "device":
            return QNode._validate_device_method(device, interface)

        if diff_method == "parameter-shift":
            return (
                QNode._get_parameter_shift_tape(device),
                interface,
                device,
                {"method": "analytic"},
            )

        if diff_method == "finite-diff":
            return JacobianTape, interface, device, {"method": "numeric"}

        raise qml.QuantumFunctionError(
            f"Differentiation method {diff_method} not recognized. Allowed "
            "options are ('best', 'parameter-shift', 'backprop', 'finite-diff', 'device', 'reversible', 'adjoint')."
        )

    @staticmethod
    def get_best_method(device, interface):
        """Returns the 'best' JacobianTape and differentiation method
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
            tuple[.JacobianTape, str, .Device, dict[str, str]]: Tuple containing the compatible
            JacobianTape, the interface to apply, the device to use, and the method argument
            to pass to the ``JacobianTape.jacobian`` method.
        """
        try:
            return QNode._validate_device_method(device, interface)
        except qml.QuantumFunctionError:
            try:
                return QNode._validate_backprop_method(device, interface)
            except qml.QuantumFunctionError:
                try:
                    return (
                        QNode._get_parameter_shift_tape(device),
                        interface,
                        device,
                        {"method": "best"},
                    )
                except qml.QuantumFunctionError:
                    return JacobianTape, interface, device, {"method": "numeric"}

    @staticmethod
    def _validate_backprop_method(device, interface):
        """Validates whether a particular device and JacobianTape interface
        supports the ``"backprop"`` differentiation method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface

        Returns:
            tuple[.JacobianTape, str, .Device, dict[str, str]]: Tuple containing the compatible
            JacobianTape, the interface to apply, the device to use, and the method argument
            to pass to the ``JacobianTape.jacobian`` method.

        Raises:
            qml.QuantumFunctionError: if the device does not support backpropagation, or the
            interface provided is not compatible with the device
        """
        if device.shots is not None:
            raise qml.QuantumFunctionError(
                "Devices with finite shots are incompatible with backpropogation. "
                "Please set shots=None or choose a different diff_method."
            )

        # determine if the device supports backpropagation
        backprop_interface = device.capabilities().get("passthru_interface", None)

        # determine if the device has any child devices that support backpropagation
        backprop_devices = device.capabilities().get("passthru_devices", None)

        if getattr(device, "cache", 0):
            raise qml.QuantumFunctionError(
                "Device caching is incompatible with the backprop diff_method"
            )

        if backprop_interface is not None:
            # device supports backpropagation natively

            if interface == backprop_interface:
                return JacobianTape, interface, device, {"method": "backprop"}

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
                return JacobianTape, interface, device, {"method": "backprop"}

            raise qml.QuantumFunctionError(
                f"Device {device.short_name} only supports diff_method='backprop' when using the "
                f"{list(backprop_devices.keys())} interfaces."
            )

        raise qml.QuantumFunctionError(
            f"The {device.short_name} device does not support native computations with "
            "autodifferentiation frameworks."
        )

    @staticmethod
    def _validate_reversible_method(device, interface):
        """Validates whether a particular device and JacobianTape interface
        supports the ``"reversible"`` differentiation method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface

        Returns:
            tuple[.JacobianTape, str, .Device, dict[str, str]]: Tuple containing the compatible
            JacobianTape, the interface to apply, the device to use, and the method argument
            to pass to the ``JacobianTape.jacobian`` method.

        Raises:
            qml.QuantumFunctionError: if the device does not support reversible backprop
        """
        # TODO: update when all capabilities keys changed to "supports_reversible_diff"
        supports_reverse = device.capabilities().get("supports_reversible_diff", False)
        supports_reverse = supports_reverse or device.capabilities().get("reversible_diff", False)

        if not supports_reverse:
            raise ValueError(
                f"The {device.short_name} device does not support reversible differentiation."
            )

        if device.shots is not None:
            warnings.warn(
                "Requested reversible differentiation to be computed with finite shots."
                " Reversible differentiation always calculated exactly.",
                UserWarning,
            )

        return ReversibleTape, interface, device, {"method": "analytic"}

    @staticmethod
    def _validate_adjoint_method(device, interface):
        """Validates whether a particular device and JacobianTape interface
        supports the ``"adjoint"`` differentiation method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface

        Returns:
            tuple[.JacobianTape, str, .Device, dict[str, str]]: Tuple containing the compatible
            JacobianTape, the interface to apply, the device to use, and the method argument
            to pass to the ``JacobianTape.jacobian`` method.

        Raises:
            qml.QuantumFunctionError: if the device does not support adjoint backprop
        """
        supported_device = hasattr(device, "_apply_operation")
        supported_device = supported_device and hasattr(device, "_apply_unitary")
        supported_device = supported_device and device.capabilities().get("returns_state")
        supported_device = supported_device and hasattr(device, "adjoint_jacobian")
        # The above provides a minimal set of requirements that we can likely improve upon in
        # future, or alternatively summarize within a single device capability. Moreover, we also
        # need to inspect the circuit measurements to ensure only expectation values are taken. This
        # cannot be done here since we don't yet know the composition of the circuit.

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

        jac_options = {"method": "device", "jacobian_method": "adjoint_jacobian"}
        # reuse the forward pass
        # torch and tensorflow can cache the state
        if interface in {"autograd", "jax"}:
            jac_options["device_pd_options"] = {"use_device_state": True}

        return (
            JacobianTape,
            interface,
            device,
            jac_options,
        )

    @staticmethod
    def _validate_device_method(device, interface):
        """Validates whether a particular device and JacobianTape interface
        supports the ``"device"`` differentiation method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface

        Returns:
            tuple[.JacobianTape, str, .Device, dict[str, str]]: Tuple containing the compatible
            JacobianTape, the interface to apply, the device to use, and the method argument
            to pass to the ``JacobianTape.jacobian`` method.

        Raises:
            qml.QuantumFunctionError: if the device does not provide a native method for computing
            the Jacobian
        """
        # determine if the device provides its own jacobian method
        provides_jacobian = device.capabilities().get("provides_jacobian", False)

        if not provides_jacobian:
            raise qml.QuantumFunctionError(
                f"The {device.short_name} device does not provide a native "
                "method for computing the jacobian."
            )

        return JacobianTape, interface, device, {"method": "device"}

    @staticmethod
    def _get_parameter_shift_tape(device):
        """Validates whether a particular device
        supports the parameter-shift differentiation method, and returns
        the correct tape.

        Args:
            device (.Device): PennyLane device

        Returns:
            .JacobianTape: the compatible JacobianTape

        Raises:
            qml.QuantumFunctionError: if the device model does not have a corresponding
            parameter-shift rule
        """
        # determine if the device provides its own jacobian method
        model = device.capabilities().get("model", None)

        if model == "qubit":
            return QubitParamShiftTape

        if model == "cv":
            return CVParamShiftTape

        raise qml.QuantumFunctionError(
            f"Device {device.short_name} uses an unknown model ('{model}') "
            "that does not support the parameter-shift rule."
        )

    def construct(self, args, kwargs):
        """Call the quantum function with a tape context, ensuring the operations get queued."""

        if self.interface == "autograd":
            # HOTFIX: to maintain compatibility with core, here we treat
            # all inputs that do not explicitly specify `requires_grad=False`
            # as trainable. This should be removed at some point, forcing users
            # to specify `requires_grad=True` for trainable parameters.
            args = [
                anp.array(a, requires_grad=True) if not hasattr(a, "requires_grad") else a
                for a in args
            ]

        self.qtape = self._tape()

        with self.qtape:
            self.qfunc_output = self.func(*args, **kwargs)

        if not isinstance(self.qfunc_output, Sequence):
            measurement_processes = (self.qfunc_output,)
        else:
            measurement_processes = self.qfunc_output

        if not all(isinstance(m, qml.measure.MeasurementProcess) for m in measurement_processes):
            raise qml.QuantumFunctionError(
                "A quantum function must return either a single measurement, "
                "or a nonempty sequence of measurements."
            )

        state_returns = any(m.return_type is State for m in measurement_processes)

        # apply the interface (if any)

        explicit_backprop = self.diff_options["method"] == "backprop"
        best_and_passthru = (
            self.diff_options["method"] == "best"
            and "passthru_interface" in self.device.capabilities()
        )
        backprop_diff = explicit_backprop or best_and_passthru
        if not backprop_diff and self.interface is not None:
            # pylint: disable=protected-access
            if state_returns and self.interface in ["torch", "tf"]:
                # The state is complex and we need to indicate this in the to_torch or to_tf
                # functions
                self.INTERFACE_MAP[self.interface](self, dtype=np.complex128)
            else:
                self.INTERFACE_MAP[self.interface](self)

        if not all(ret == m for ret, m in zip(measurement_processes, self.qtape.measurements)):
            raise qml.QuantumFunctionError(
                "All measurements must be returned in the order they are measured."
            )

        for obj in self.qtape.operations + self.qtape.observables:

            if getattr(obj, "num_wires", None) is qml.operation.WiresEnum.AllWires:
                # check here only if enough wires
                if len(obj.wires) != self.device.num_wires:
                    raise qml.QuantumFunctionError(f"Operator {obj.name} must act on all wires")

            if (
                isinstance(obj, qml.ops.qubit.SparseHamiltonian)
                and self.diff_method != "parameter-shift"
            ):
                raise qml.QuantumFunctionError(
                    "SparseHamiltonian observable must be used with the parameter-shift"
                    " differentiation method"
                )

        # pylint: disable=protected-access
        obs_on_same_wire = len(self.qtape._obs_sharing_wires) > 0
        ops_not_supported = any(
            isinstance(op, qml.tape.QuantumTape)  # nested tapes must be expanded
            or not self.device.supports_operation(op.name)  # unsupported ops must be expanded
            for op in self.qtape.operations
        )

        # expand out the tape, if nested tapes are present, any operations are not supported on the
        # device, or multiple observables are measured on the same wire
        if ops_not_supported or obs_on_same_wire:
            self.qtape = self.qtape.expand(
                depth=self.max_expansion,
                stop_at=lambda obj: not isinstance(obj, qml.tape.QuantumTape)
                and self.device.supports_operation(obj.name),
            )

        # provide the jacobian options
        self.qtape.jacobian_options = self.diff_options

        if self.diff_options["method"] == "backprop":
            params = self.qtape.get_parameters(trainable_only=False)
            self.qtape.trainable_params = qml.math.get_trainable_indices(params)

    def __call__(self, *args, **kwargs):

        # If shots specified in call but not in qfunc signature,
        # interpret it as device shots value for this call.
        # TODO: make this more functional by passing shots as qtape.execute(.., shots=shots).
        original_shots = -1
        if "shots" in kwargs and not self._qfunc_uses_shots_arg:
            original_shots = self.device.shots  # remember device shots
            # remove shots from kwargs and temporarily change on device
            self.device.shots = kwargs.pop("shots", None)

        if self.mutable or self.qtape is None:
            # construct the tape
            self.construct(args, kwargs)

        # Under certain conditions, split tape into multiple tapes and recombine them.
        # Else just execute the tape, and let the device take care of things.
        hamiltonian_in_obs = "Hamiltonian" in [obs.name for obs in self.qtape.observables]
        # if the device does not support Hamiltonians, we split them
        supports_hamiltonian = self.device.supports_observable("Hamiltonian")
        # if the user wants a finite-shots computation we always split Hamiltonians
        finite_shots = self.device.shots is not None
        # if a grouping has been computed for all Hamiltonians we assume that they should be split
        grouping_known = all(
            obs.grouping_indices is not None
            for obs in self.qtape.observables
            if obs.name == "Hamiltonian"
        )
        if hamiltonian_in_obs and ((not supports_hamiltonian or finite_shots) or grouping_known):
            try:
                tapes, fn = qml.transforms.hamiltonian_expand(self.qtape, group=False)
            except ValueError as e:
                raise ValueError(
                    "Can only return the expectation of a single Hamiltonian observable"
                ) from e
            results = [tape.execute(device=self.device) for tape in tapes]
            res = fn(results)
        else:
            res = self.qtape.execute(device=self.device)

        finite_diff = any(
            getattr(x["op"], "grad_method", None) == "F" for x in self.qtape._par_info.values()
        )
        if finite_diff and self.diff_method_change:
            self.diff_method = "finite-diff"

        # if shots was changed
        if original_shots != -1:
            # reinstate default on device
            self.device.shots = original_shots

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

        if isinstance(self.qfunc_output, Sequence) or (
            self.qtape.is_sampled and self.device._has_partitioned_shots()
        ):
            return res

        return qml.math.squeeze(res)

    def draw(
        self, charset="unicode", wire_order=None, show_all_wires=False, max_length=None
    ):  # pylint: disable=unused-argument
        """Draw the quantum tape as a circuit diagram.

        Args:
            charset (str, optional): The charset that should be used. Currently, "unicode" and
                "ascii" are supported.
            wire_order (Sequence[Any]): The order (from top to bottom) to print the wires of the circuit.
                If not provided, this defaults to the wire order of the device.
            show_all_wires (bool): If True, all wires, including empty wires, are printed.
            max_length (int, optional): Maximum string width (columns) when printing the circuit to the CLI.

        Raises:
            ValueError: if the given charset is not supported
            .QuantumFunctionError: drawing is impossible because the underlying
                quantum tape has not yet been constructed

        Returns:
            str: the circuit representation of the tape

        **Example**

        Consider the following circuit as an example:

        .. code-block:: python3

            @qml.qnode(dev)
            def circuit(a, w):
                qml.Hadamard(0)
                qml.CRX(a, wires=[0, 1])
                qml.Rot(*w, wires=[1])
                qml.CRX(-a, wires=[0, 1])
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        We can draw the QNode after execution:

        >>> result = circuit(2.3, [1.2, 3.2, 0.7])
        >>> print(circuit.draw())
        0: ──H──╭C────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
        1: ─────╰RX(2.3)──Rot(1.2, 3.2, 0.7)──╰RX(-2.3)──╰┤ ⟨Z ⊗ Z⟩
        >>> print(circuit.draw(charset="ascii"))
        0: --H--+C----------------------------+C---------+| <Z @ Z>
        1: -----+RX(2.3)--Rot(1.2, 3.2, 0.7)--+RX(-2.3)--+| <Z @ Z>

        Circuit drawing works with devices with custom wire labels:

        .. code-block:: python3

            dev = qml.device('default.qubit', wires=["a", -1, "q2"])

            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(wires=-1)
                qml.CNOT(wires=["a", "q2"])
                qml.RX(0.2, wires="a")
                return qml.expval(qml.PauliX(wires="q2"))

        When printed, the wire order matches the order defined on the device:

        >>> print(circuit.draw())
          a: ─────╭C──RX(0.2)──┤
         -1: ──H──│────────────┤
         q2: ─────╰X───────────┤ ⟨X⟩

        We can use the ``wire_order`` argument to change the wire order:

        >>> print(circuit.draw(wire_order=["q2", "a", -1]))
         q2: ──╭X───────────┤ ⟨X⟩
          a: ──╰C──RX(0.2)──┤
         -1: ───H───────────┤
        """
        warnings.warn(
            "The QNode.draw method has been deprecated. "
            "Please use the qml.draw(qnode)(*args) function instead.",
            UserWarning,
        )

        if self.qtape is None:
            raise qml.QuantumFunctionError(
                "The QNode can only be drawn after its quantum tape has been constructed."
            )

        wire_order = wire_order or self.device.wires
        wire_order = qml.wires.Wires(wire_order)

        if show_all_wires and len(wire_order) < self.device.num_wires:
            raise ValueError(
                "When show_all_wires is enabled, the provided wire order must contain all wires on the device."
            )

        if not self.device.wires.contains_wires(wire_order):
            raise ValueError(
                f"Provided wire order {wire_order.labels} contains wires not contained on the device: {self.device.wires}."
            )

        return self.qtape.draw(
            charset=charset,
            wire_order=wire_order,
            show_all_wires=show_all_wires,
            max_length=max_length,
        )

    @property
    def specs(self):
        """Resource information about a quantum circuit.

        Returns:
        dict[str, Union[defaultdict,int]]: dictionaries that contain QNode specifications

        **Example**

        .. code-block:: python3

            dev = qml.device('default.qubit', wires=2)
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(x[0], wires=0)
                qml.RY(x[1], wires=1)
                qml.CNOT(wires=(0,1))
                return qml.probs(wires=(0,1))

            x = np.array([0.1, 0.2])
            res = circuit(x)

        >>> circuit.specs
        {'gate_sizes': defaultdict(int, {1: 2, 2: 1}),
        'gate_types': defaultdict(int, {'RX': 1, 'RY': 1, 'CNOT': 1}),
        'num_operations': 3,
        'num_observables': 1,
        'num_diagonalizing_gates': 0,
        'num_used_wires': 2,
        'depth': 2,
        'num_device_wires': 2,
        'device_name': 'default.qubit.autograd',
        'diff_method': 'backprop'}

        """
        if self.qtape is None:
            raise qml.QuantumFunctionError(
                "The QNode specifications can only be calculated after its quantum tape has been constructed."
            )

        info = self.qtape.specs.copy()

        info["num_device_wires"] = self.device.num_wires
        info["device_name"] = self.device.short_name
        info["diff_method"] = self.diff_method

        # tapes do not accurately track parameters for backprop
        # TODO: calculate number of trainable parameters in backprop
        # find better syntax for determining if backprop
        if info["diff_method"] == "backprop":
            del info["num_trainable_params"]

        return info

    def to_tf(self, dtype=None):
        """Apply the TensorFlow interface to the internal quantum tape.

        Args:
            dtype (tf.dtype): The dtype that the TensorFlow QNode should
                output. If not provided, the default is ``tf.float64``.

        Raises:
            .QuantumFunctionError: if TensorFlow >= 2.1 is not installed
        """
        # pylint: disable=import-outside-toplevel
        try:
            import tensorflow as tf
            from pennylane.interfaces.tf import TFInterface

            if self.interface != "tf" and self.interface is not None:
                # Since the interface is changing, need to re-validate the tape class.
                # if method was changed from "best", set it back to best
                if self.diff_method_change:
                    diff_method = "best"
                else:
                    diff_method = self.diff_method
                self._tape, interface, self.device, diff_options = self.get_tape(
                    self._original_device, "tf", diff_method
                )
                if self.diff_method_change:
                    self.diff_method = self._get_best_diff_method(diff_options)
                self.interface = interface
                self.diff_options.update(diff_options)
            else:
                self.interface = "tf"

            if not isinstance(self.dtype, tf.DType):
                self.dtype = None

            self.dtype = dtype or self.dtype or TFInterface.dtype

            if self.qtape is not None:
                TFInterface.apply(self.qtape, dtype=tf.as_dtype(self.dtype))

        except ImportError as e:
            raise qml.QuantumFunctionError(
                "TensorFlow not found. Please install the latest "
                "version of TensorFlow to enable the 'tf' interface."
            ) from e

    def to_torch(self, dtype=None):
        """Apply the Torch interface to the internal quantum tape.

        Args:
            dtype (tf.dtype): The dtype that the Torch QNode should
                output. If not provided, the default is ``torch.float64``.

        Raises:
            .QuantumFunctionError: if PyTorch >= 1.3 is not installed
        """
        # pylint: disable=import-outside-toplevel
        try:
            import torch
            from pennylane.interfaces.torch import TorchInterface

            if self.interface != "torch" and self.interface is not None:
                # Since the interface is changing, need to re-validate the tape class.
                # if method was changed from "best", set it back to best
                if self.diff_method_change:
                    diff_method = "best"
                else:
                    diff_method = self.diff_method
                self._tape, interface, self.device, diff_options = self.get_tape(
                    self._original_device, "torch", diff_method
                )
                if self.diff_method_change:
                    self.diff_method = self._get_best_diff_method(diff_options)
                self.interface = interface
                self.diff_options.update(diff_options)
            else:
                self.interface = "torch"

            if not isinstance(self.dtype, torch.dtype):
                self.dtype = None

            self.dtype = dtype or self.dtype or TorchInterface.dtype

            if self.dtype is np.complex128:
                self.dtype = torch.complex128

            if self.qtape is not None:
                TorchInterface.apply(self.qtape, dtype=self.dtype)

        except ImportError as e:
            raise qml.QuantumFunctionError(
                "PyTorch not found. Please install the latest "
                "version of PyTorch to enable the 'torch' interface."
            ) from e

    def to_autograd(self):
        """Apply the Autograd interface to the internal quantum tape."""
        self.dtype = AutogradInterface.dtype

        if self.interface != "autograd" and self.interface is not None:
            # Since the interface is changing, need to re-validate the tape class.
            # if method was changed from "best", set it back to best
            if self.diff_method_change:
                diff_method = "best"
            else:
                diff_method = self.diff_method
            self._tape, interface, self.device, diff_options = self.get_tape(
                self._original_device, "autograd", diff_method
            )
            if self.diff_method_change:
                self.diff_method = self._get_best_diff_method(diff_options)
            self.interface = interface
            self.diff_options.update(diff_options)
        else:
            self.interface = "autograd"

        if self.qtape is not None:
            AutogradInterface.apply(self.qtape)

    def to_jax(self):
        """Apply the JAX interface to the internal quantum tape.

        Args:
            dtype (tf.dtype): The dtype that the JAX QNode should
                output. If not provided, the default is ``jnp.float64``.

        Raises:
            .QuantumFunctionError: if TensorFlow >= 2.1 is not installed
        """
        # pylint: disable=import-outside-toplevel
        try:
            from pennylane.interfaces.jax import JAXInterface

            if self.interface != "jax" and self.interface is not None:
                # Since the interface is changing, need to re-validate the tape class.
                # if method was changed from "best", set it back to best
                if self.diff_method_change:
                    diff_method = "best"
                else:
                    diff_method = self.diff_method
                self._tape, interface, self.device, diff_options = self.get_tape(
                    self._original_device, "jax", diff_method
                )
                if self.diff_method_change:
                    self.diff_method = self._get_best_diff_method(diff_options)
                self.interface = interface
                self.diff_options.update(diff_options)
            else:
                self.interface = "jax"

            if self.qtape is not None:
                JAXInterface.apply(self.qtape)

        except ImportError as e:
            raise qml.QuantumFunctionError(
                "JAX not found. Please install the latest "
                "version of JAX to enable the 'jax' interface."
            ) from e

    INTERFACE_MAP = {
        "autograd": to_autograd,
        "torch": to_torch,
        "tf": to_tf,
        "jax": to_jax,
    }


# pylint:disable=too-many-arguments
def qnode(
    device,
    interface="autograd",
    diff_method="best",
    mutable=True,
    max_expansion=10,
    h=1e-7,
    order=1,
    shift=np.pi / 2,
    adjoint_cache=True,
    argnum=None,
    **kwargs,
):
    """Decorator for creating QNodes.

    This decorator is used to indicate to PennyLane that the decorated function contains a
    :ref:`quantum variational circuit <glossary_variational_circuit>` that should be bound to a
    compatible device.

    The QNode calls the quantum function to construct a :class:`~.QuantumTape` instance representing
    the quantum circuit.

    Args:
        func (callable): a quantum function
        device (~.Device): a PennyLane-compatible device
        interface (str): The interface that will be used for classical backpropagation.
            This affects the types of objects that can be passed to/returned from the QNode:

            * ``"autograd"``: Allows autograd to backpropogate
              through the QNode. The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays.

            * ``"torch"``: Allows PyTorch to backpropogate
              through the QNode. The QNode accepts and returns Torch tensors.

            * ``"tf"``: Allows TensorFlow in eager mode to backpropogate
              through the QNode. The QNode accepts and returns
              TensorFlow ``tf.Variable`` and ``tf.tensor`` objects.

            * ``None``: The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays. It does not connect to any
              machine learning library automatically for backpropagation.

        diff_method (str): the method of differentiation to use in the created QNode.

            * ``"best"``: Best available method. Uses classical backpropagation or the
              device directly to compute the gradient if supported, otherwise will use
              the analytic parameter-shift rule where possible with finite-difference as a fallback.

            * ``"backprop"``: Use classical backpropagation. Only allowed on simulator
              devices that are classically end-to-end differentiable, for example
              :class:`default.tensor.tf <~.DefaultTensorTF>`. Note that the returned
              QNode can only be used with the machine-learning framework supported
              by the device; a separate ``interface`` argument should not be passed.

            * ``"reversible"``: Uses a reversible method for computing the gradient.
              This method is similar to ``"backprop"``, but trades off increased
              runtime with significantly lower memory usage. Compared to the
              parameter-shift rule, the reversible method can be faster or slower,
              depending on the density and location of parametrized gates in a circuit.
              Only allowed on (simulator) devices with the "reversible" capability,
              for example :class:`default.qubit <~.DefaultQubit>`.

            * ``"adjoint"``: Uses an adjoint `method <https://arxiv.org/abs/2009.02823>`__ that
              reverses through the circuit after a forward pass by iteratively applying the inverse
              (adjoint) gate. This method is similar to the reversible method, but has a lower time
              overhead and a similar memory overhead. Only allowed on simulator devices such as
              :class:`default.qubit <~.DefaultQubit>`.

            * ``"device"``: Queries the device directly for the gradient.
              Only allowed on devices that provide their own gradient rules.

            * ``"parameter-shift"``: Use the analytic parameter-shift
              rule for all supported quantum operation arguments, with finite-difference
              as a fallback.

            * ``"finite-diff"``: Uses numerical finite-differences for all quantum
              operation arguments.

        mutable (bool): If True, the underlying quantum circuit is re-constructed with
            every evaluation. This is the recommended approach, as it allows the underlying
            quantum structure to depend on (potentially trainable) QNode input arguments,
            however may add some overhead at evaluation time. If this is set to False, the
            quantum structure will only be constructed on the *first* evaluation of the QNode,
            and is stored and re-used for further quantum evaluations. Only set this to False
            if it is known that the underlying quantum structure is **independent of QNode input**.
        max_expansion (int): The number of times the internal circuit should be expanded when
            executed on a device. Expansion occurs when an operation or measurement is not
            supported, and results in a gate decomposition. If any operations in the decomposition
            remain unsupported by the device, another expansion occurs.
        h (float): step size for the finite difference method
        order (int): The order of the finite difference method to use. ``1`` corresponds
            to forward finite differences, ``2`` to centered finite differences.
        shift (float): the size of the shift for two-term parameter-shift gradient computations
        adjoint_cache (bool): For TensorFlow and PyTorch interfaces and adjoint differentiation,
            this indicates whether to save the device state after the forward pass.  Doing so saves a
            forward execution. Device state automatically reused with autograd and JAX interfaces.
        argnum (int, list(int), None): Which argument(s) to compute the Jacobian
            with respect to. When there are fewer parameters specified than the
            total number of trainable parameters, the jacobian is being estimated. Note
            that this option is only applicable for the following differentiation methods:
            ``"parameter-shift"``, ``"finite-diff"`` and ``"reversible"``.
        kwargs: used to catch all unrecognized keyword arguments and provide a user warning
            about them

    **Example**

    >>> dev = qml.device("default.qubit", wires=1)
    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=0)
    >>>     return expval(qml.PauliZ(0))
    """

    @lru_cache()
    def qfunc_decorator(func):
        """The actual decorator"""
        qn = QNode(
            func,
            device,
            interface=interface,
            diff_method=diff_method,
            mutable=mutable,
            max_expansion=max_expansion,
            h=h,
            order=order,
            shift=shift,
            adjoint_cache=adjoint_cache,
            argnum=argnum,
            **kwargs,
        )
        return update_wrapper(qn, func)

    return qfunc_decorator
