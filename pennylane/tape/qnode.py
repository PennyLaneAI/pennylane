# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
from collections.abc import Sequence
from functools import lru_cache, update_wrapper, wraps
import warnings

import numpy as np

import pennylane as qml
from pennylane import Device

from pennylane.operation import State

from pennylane.tape.interfaces.autograd import AutogradInterface, np as anp
from pennylane.tape.tapes import JacobianTape, QubitParamShiftTape, CVParamShiftTape, ReversibleTape


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

            * ``interface='autograd'``: Allows autograd to backpropagate
              through the QNode. The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays.

            * ``interface='torch'``: Allows PyTorch to backpropogate
              through the QNode. The QNode accepts and returns Torch tensors.

            * ``interface='tf'``: Allows TensorFlow in eager mode to backpropogate
              through the QNode. The QNode accepts and returns
              TensorFlow ``tf.Variable`` and ``tf.tensor`` objects.

            * ``None``: The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays. It does not connect to any
              machine learning library automatically for backpropagation.

        diff_method (str, None): the method of differentiation to use in the created QNode

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

    Keyword Args:
        h=1e-7 (float): step size for the finite difference method
        order=1 (int): The order of the finite difference method to use. ``1`` corresponds
            to forward finite differences, ``2`` to centered finite differences.
        shift=pi/2 (float): the size of the shift for two-term parameter-shift gradient computations

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
        **diff_options,
    ):

        if interface is not None and interface not in self.INTERFACE_MAP:
            raise qml.QuantumFunctionError(
                f"Unknown interface {interface}. Interface must be "
                f"one of {list(self.INTERFACE_MAP.keys())}."
            )

        if not isinstance(device, Device):
            raise qml.QuantumFunctionError(
                "Invalid device. Device must be a valid PennyLane device."
            )

        self.mutable = mutable
        self.func = func
        self._original_device = device
        self.qtape = None
        self.qfunc_output = None
        # store the user-specified differentiation method
        self.diff_method = diff_method

        self._tape, self.interface, self.device, tape_diff_options = self.get_tape(
            device, interface, diff_method
        )

        # The arguments to be passed to JacobianTape.jacobian
        self.diff_options = diff_options or {}
        self.diff_options.update(tape_diff_options)

        self.dtype = np.float64
        self.max_expansion = max_expansion

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

        * ``"backprop"``
        * ``"device"``
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

        if getattr(device, "analytic", False) and backprop_devices is not None:
            # device is analytic and has child devices that support backpropagation natively

            if interface in backprop_devices:
                # TODO: need a better way of passing existing device init options
                # to a new device?
                device = qml.device(
                    backprop_devices[interface],
                    wires=device.wires,
                    shots=device.shots,
                    analytic=True,
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

        return (
            JacobianTape,
            interface,
            device,
            {"method": "device", "jacobian_method": "adjoint_jacobian"},
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

        if not all(isinstance(m, qml.tape.MeasurementProcess) for m in measurement_processes):
            raise qml.QuantumFunctionError(
                "A quantum function must return either a single measurement, "
                "or a nonempty sequence of measurements."
            )

        state_returns = any([m.return_type is State for m in measurement_processes])

        # apply the interface (if any)
        if self.diff_options["method"] != "backprop" and self.interface is not None:
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
                    raise qml.QuantumFunctionError(
                        "Operator {} must act on all wires".format(obj.name)
                    )

        # provide the jacobian options
        self.qtape.jacobian_options = self.diff_options

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

    def __call__(self, *args, **kwargs):
        if self.mutable or self.qtape is None:
            # construct the tape
            self.construct(args, kwargs)

        # execute the tape
        res = self.qtape.execute(device=self.device)

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

        if isinstance(self.qfunc_output, Sequence):
            return res

        return qml.math.squeeze(res)

    def metric_tensor(self, *args, diag_approx=False, only_construct=False, **kwargs):
        """Evaluate the value of the metric tensor.

        Args:
            args (tuple[Any]): positional arguments
            kwargs (dict[str, Any]): auxiliary arguments
            diag_approx (bool): iff True, use the diagonal approximation
            only_construct (bool): Iff True, construct the circuits used for computing
                the metric tensor but do not execute them, and return the tapes.

        Returns:
            array[float]: metric tensor
        """
        return metric_tensor(self, diag_approx=diag_approx, only_construct=only_construct)(
            *args, **kwargs
        )

    def draw(
        self, charset="unicode", wire_order=None, show_all_wires=False, **kwargs
    ):  # pylint: disable=unused-argument
        """Draw the quantum tape as a circuit diagram.

        Args:
            charset (str, optional): The charset that should be used. Currently, "unicode" and
                "ascii" are supported.
            wire_order (Sequence[Any]): The order (from top to bottom) to print the wires of the circuit.
                If not provided, this defaults to the wire order of the device.
            show_all_wires (bool): If True, all wires, including empty wires, are printed.

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
        # TODO: remove 'kwargs' when tape mode is default.
        # Currently it only exists to match the signature of non-tape mode draw.
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
            charset=charset, wire_order=wire_order, show_all_wires=show_all_wires
        )

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
            from pennylane.tape.interfaces.tf import TFInterface

            if self.interface != "tf" and self.interface is not None:
                # Since the interface is changing, need to re-validate the tape class.
                self._tape, interface, self.device, diff_options = self.get_tape(
                    self._original_device, "tf", self.diff_method
                )

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
            from pennylane.tape.interfaces.torch import TorchInterface

            if self.interface != "torch" and self.interface is not None:
                # Since the interface is changing, need to re-validate the tape class.
                self._tape, interface, self.device, diff_options = self.get_tape(
                    self._original_device, "torch", self.diff_method
                )

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
            self._tape, interface, self.device, diff_options = self.get_tape(
                self._original_device, "autograd", self.diff_method
            )

            self.interface = interface
            self.diff_options.update(diff_options)
        else:
            self.interface = "autograd"

        if self.qtape is not None:
            AutogradInterface.apply(self.qtape)

    def to_jax(self):
        """Validation checks when a user expects to use the JAX interface."""
        if self.diff_method != "backprop":
            raise qml.QuantumFunctionError(
                "The JAX interface can only be used with "
                "diff_method='backprop' on supported devices"
            )
        self.interface = "jax"

    INTERFACE_MAP = {"autograd": to_autograd, "torch": to_torch, "tf": to_tf, "jax": to_jax}


def qnode(
    device, interface="autograd", diff_method="best", mutable=True, max_expansion=10, **diff_options
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

            * ``interface='autograd'``: Allows autograd to backpropogate
              through the QNode. The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays.

            * ``interface='torch'``: Allows PyTorch to backpropogate
              through the QNode. The QNode accepts and returns Torch tensors.

            * ``interface='tf'``: Allows TensorFlow in eager mode to backpropogate
              through the QNode. The QNode accepts and returns
              TensorFlow ``tf.Variable`` and ``tf.tensor`` objects.

            * ``None``: The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays. It does not connect to any
              machine learning library automatically for backpropagation.

        diff_method (str, None): the method of differentiation to use in the created QNode.

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

    Keyword Args:
        h=1e-7 (float): Step size for the finite difference method.
        order=1 (int): The order of the finite difference method to use. ``1`` corresponds
            to forward finite differences, ``2`` to centered finite differences.

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
            **diff_options,
        )
        return update_wrapper(qn, func)

    return qfunc_decorator


def _get_classical_jacobian(_qnode):
    """Helper function to extract the Jacobian
    matrix of the classical part of a QNode"""

    def classical_preprocessing(*args, **kwargs):
        """Returns the trainable gate parameters for
        a given QNode input"""
        _qnode.construct(args, kwargs)
        return qml.math.stack(_qnode.qtape.get_parameters())

    if _qnode.interface == "autograd":
        return qml.jacobian(classical_preprocessing)

    if _qnode.interface == "torch":
        import torch

        def _jacobian(*args, **kwargs):  # pylint: disable=unused-argument
            return torch.autograd.functional.jacobian(classical_preprocessing, args)

        return _jacobian

    if _qnode.interface == "jax":
        import jax

        return jax.jacobian(classical_preprocessing)

    if _qnode.interface == "tf":
        import tensorflow as tf

        def _jacobian(*args, **kwargs):
            with tf.GradientTape() as tape:
                tape.watch(args)
                gate_params = classical_preprocessing(*args, **kwargs)

            return tape.jacobian(gate_params, args)

        return _jacobian


def metric_tensor(_qnode, diag_approx=False, only_construct=False):
    """metric_tensor(qnode, diag_approx=False, only_construct=False)
    Returns a function that returns the value of the metric tensor
    of a given QNode.

    .. note::

        Currently, only the :class:`~.RX`, :class:`~.RY`, :class:`~.RZ`, and
        :class:`~.PhaseShift` parametrized gates are supported.
        All other parametrized gates will be decomposed if possible.

    Args:
        qnode (.QNode or .ExpvalCost): QNode(s) to compute the metric tensor of
        diag_approx (bool): iff True, use the diagonal approximation
        only_construct (bool): Iff True, construct the circuits used for computing
            the metric tensor but do not execute them, and return the tapes.

    Returns:
        func: Function which accepts the same arguments as the QNode. When called, this
        function will return the metric tensor.

    **Example**

    Consider the following QNode:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="autograd")
        def circuit(weights):
            # layer 1
            qml.RX(weights[0, 0], wires=0)
            qml.RX(weights[0, 1], wires=1)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])

            # layer 2
            qml.RZ(weights[1, 0], wires=0)
            qml.RZ(weights[1, 1], wires=2)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.expval(qml.PauliY(2))

    We can use the ``metric_tensor`` function to generate a new function, that returns the
    metric tensor of this QNode:

    >>> met_fn = qml.metric_tensor(circuit)
    >>> weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
    >>> met_fn(weights)
    tensor([[0.25  , 0.    , 0.    , 0.    ],
            [0.    , 0.25  , 0.    , 0.    ],
            [0.    , 0.    , 0.0025, 0.0024],
            [0.    , 0.    , 0.0024, 0.0123]], requires_grad=True)

    The returned metric tensor is also fully differentiable, in all interfaces.
    For example, differentiating the ``(3, 2)`` element:

    >>> grad_fn = qml.grad(lambda x: met_fn(x)[3, 2])
    >>> grad_fn(weights)
    array([[ 0.04867729, -0.00049502,  0.        ],
           [ 0.        ,  0.        ,  0.        ]])
    """
    if _qnode.__class__.__name__ == "ExpvalCost":
        if _qnode._multiple_devices:  # pylint: disable=protected-access
            warnings.warn(
                "ExpvalCost was instantiated with multiple devices. Only the first device "
                "will be used to evaluate the metric tensor."
            )

        _qnode = _qnode.qnodes.qnodes[0]

    if not isinstance(_qnode, QNode):
        # non-tape mode QNode
        return lambda *args, **kwargs: _qnode.metric_tensor(
            args, kwargs, diag_approx=diag_approx, only_construct=only_construct
        )

    def _metric_tensor_fn(*args, **kwargs):
        jac = qml.math.stack(_get_classical_jacobian(_qnode)(*args, **kwargs))
        jac = qml.math.reshape(jac, [_qnode.qtape.num_params, -1])

        wrt, perm = np.nonzero(qml.math.toarray(jac))
        perm = np.argsort(np.argsort(perm))

        _qnode.construct(args, kwargs)

        metric_tensor_tapes, processing_fn = qml.tape.transforms.metric_tensor(
            _qnode.qtape,
            diag_approx=diag_approx,
            wrt=wrt.tolist() if _qnode.diff_options["method"] == "backprop" else None,
        )

        if only_construct:
            return metric_tensor_tapes

        res = [t.execute(device=_qnode.device) for t in metric_tensor_tapes]
        mt = processing_fn(res)

        # permute rows ad columns
        mt = qml.math.gather(mt, perm)
        mt = qml.math.gather(qml.math.T(mt), perm)
        return mt

    return _metric_tensor_fn


def draw(_qnode, charset="unicode", wire_order=None, show_all_wires=False):
    """draw(qnode, charset="unicode", wire_order=None, show_all_wires=False)
    Create a function that draws the given _qnode.

    Args:
        qnode (.QNode): the input QNode that is to be drawn.
        charset (str, optional): The charset that should be used. Currently, "unicode" and
            "ascii" are supported.
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
        show_all_wires (bool): If True, all wires, including empty wires, are printed.

    Returns:
        A function that has the same arguement signature as ``qnode``. When called,
        the function will draw the QNode.

    **Example**

    Given the following definition of a QNode,

    .. code-block:: python3

        qml.enable_tape()

        @qml.qnode(dev)
        def circuit(a, w):
            qml.Hadamard(0)
            qml.CRX(a, wires=[0, 1])
            qml.Rot(*w, wires=[1])
            qml.CRX(-a, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    We can draw the it like such:

    >>> drawer = qml.draw(circuit)
    >>> drawer(a=2.3, w=[1.2, 3.2, 0.7])
    0: ──H──╭C────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
    1: ─────╰RX(2.3)──Rot(1.2, 3.2, 0.7)──╰RX(-2.3)──╰┤ ⟨Z ⊗ Z⟩

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

    >>> drawer = qml.draw(circuit)
    >>> drawer()
      a: ─────╭C──RX(0.2)──┤
     -1: ──H──│────────────┤
     q2: ─────╰X───────────┤ ⟨X⟩

    We can use the ``wire_order`` argument to change the wire order:

    >>> drawer = qml.draw(circuit, wire_order=["q2", "a", -1])
    >>> drawer()
     q2: ──╭X───────────┤ ⟨X⟩
      a: ──╰C──RX(0.2)──┤
     -1: ───H───────────┤
    """
    if not hasattr(_qnode, "qtape"):
        raise ValueError(
            "qml.draw only works when tape mode is enabled. "
            "You can enable tape mode with qml.enable_tape()."
        )

    @wraps(_qnode)
    def wrapper(*args, **kwargs):
        _qnode.construct(args, kwargs)
        _wire_order = wire_order or _qnode.device.wires
        _wire_order = qml.wires.Wires(_wire_order)
        return _qnode.qtape.draw(charset, wire_order=_wire_order, show_all_wires=show_all_wires)

    return wrapper
