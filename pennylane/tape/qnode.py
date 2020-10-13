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
from collections.abc import Sequence
from collections import OrderedDict
from functools import lru_cache, update_wrapper

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

    The QNode calls the quantum function to construct a :class:`~.JacobianTape` instance representing
    the quantum circuit.

    .. note::

        The quantum tape is an *experimental* feature. QNodes that use the quantum
        tape have access to advanced features, such as in-QNode classical processing,
        but do not yet have feature parity with the standard PennyLane QNode.

        This quantum tape-comaptible QNode can either be created directly,

        >>> import pennylane as qml
        >>> qml.tape.QNode(qfunc, dev)

        or enabled globally via :func:`~.enable_tape` without changing your PennyLane code:

        >>> qml.enable_tape()

        For more details, see :mod:`pennylane.tape`.

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

            * ``"device"``: Queries the device directly for the gradient.
              Only allowed on devices that provide their own gradient computation.

            * ``"parameter-shift"``: Use the analytic parameter-shift
              rule for all supported quantum operation arguments, with finite-difference
              as a fallback.

            * ``"finite-diff"``: Uses numerical finite-differences for all quantum operation
              arguments.

        caching (int): Number of device executions to store in a cache to speed up subsequent
            executions. A value of ``0`` indicates that no caching will take place. Once filled,
            older elements of the cache are removed and replaced with the most recent device
            executions to keep the cache up to date.

    Keyword Args:
        h=1e-7 (float): step size for the finite difference method
        order=1 (int): The order of the finite difference method to use. ``1`` corresponds
            to forward finite differences, ``2`` to centered finite differences.

    **Example**

    >>> qml.enable_tape()
    >>> def circuit(x):
    ...     qml.RX(x, wires=0)
    ...     return expval(qml.PauliZ(0))
    >>> dev = qml.device("default.qubit", wires=1)
    >>> qnode = qml.QNode(circuit, dev)
    """

    # pylint:disable=too-many-instance-attributes,too-many-arguments

    def __init__(
        self, func, device, interface="autograd", diff_method="best", caching=0, **diff_options
    ):

        if interface is not None and interface not in self.INTERFACE_MAP:
            raise qml.QuantumFunctionError(
                f"Unknown interface {interface}. Interface must be "
                f"one of {self.INTERFACE_MAP.values()}."
            )

        if not isinstance(device, Device):
            raise qml.QuantumFunctionError(
                "Invalid device. Device must be a valid PennyLane device."
            )

        self.func = func
        self.device = device
        self.qtape = None

        self._tape, self.interface, self.diff_method = self.get_tape(device, interface, diff_method)
        self.diff_options = diff_options or {}
        self.diff_options["method"] = self.diff_method

        self.dtype = np.float64
        self.max_expansion = 2

        self._caching = caching
        """float: number of device executions to store in a cache to speed up subsequent
        executions. If set to zero, no caching occurs."""

        if caching != 0 and self.diff_method == "backprop":
            raise ValueError('Caching mode is incompatible with the "backprop" diff_method')

        self._cache_execute = OrderedDict()
        """OrderedDict[int: Any]: A copy of the ``_cache_execute`` dictionary from the quantum
        tape"""

    @staticmethod
    def get_tape(device, interface, diff_method="best"):
        """Determine the best JacobianTape, differentiation method, and interface
        for a requested device, interface, and diff method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface
            diff_method (str): The requested method of differentiation. One of
                ``"best"``, ``"backprop"``, ``"reversible"``, ``"device"``,
                ``"parameter-shift"``, or ``"finite-diff"``.

        Returns:
            tuple[.JacobianTape, str, str]: tuple containing the compatible
            JacobianTape, the interface to apply, and the method argument
            to pass to the ``JacobianTape.jacobian`` method
        """

        if diff_method == "best":
            return QNode.get_best_method(device, interface)

        if diff_method == "backprop":
            return QNode._validate_backprop_method(device, interface)

        if diff_method == "reversible":
            return QNode._validate_reversible_method(device, interface)

        if diff_method == "device":
            return QNode._validate_device_method(device, interface)

        if diff_method == "parameter-shift":
            return QNode._get_parameter_shift_tape(device), interface, "analytic"

        if diff_method == "finite-diff":
            return JacobianTape, interface, "numeric"

        raise qml.QuantumFunctionError(
            f"Differentiation method {diff_method} not recognized. Allowed "
            "options are ('best', 'parameter-shift', 'backprop', 'finite-diff', 'device', 'reversible')."
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
            tuple[.JacobianTape, str, str]: tuple containing the compatible
            JacobianTape, the interface to apply, and the method argument
            to pass to the ``JacobianTape.jacobian`` method
        """
        try:
            return QNode._validate_backprop_method(device, interface)
        except qml.QuantumFunctionError:
            try:
                return QNode._validate_device_method(device, interface)
            except qml.QuantumFunctionError:
                try:
                    return QNode._get_parameter_shift_tape(device), interface, "best"
                except qml.QuantumFunctionError:
                    return JacobianTape, interface, "numeric"

    @staticmethod
    def _validate_backprop_method(device, interface):
        """Validates whether a particular device and JacobianTape interface
        supports the ``"backprop"`` differentiation method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface

        Returns:
            tuple[.JacobianTape, str, str]: tuple containing the compatible
            JacobianTape, the interface to apply, and the method argument
            to pass to the ``JacobianTape.jacobian`` method

        Raises:
            qml.QuantumFunctionError: if the device does not support backpropagation, or the
            interface provided is not compatible with the device
        """
        # determine if the device supports backpropagation
        backprop_interface = device.capabilities().get("passthru_interface", None)

        if backprop_interface is not None:

            if interface == backprop_interface:
                return JacobianTape, None, "backprop"

            raise qml.QuantumFunctionError(
                f"Device {device.short_name} only supports diff_method='backprop' when using the "
                f"{backprop_interface} interface."
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
            tuple[.JacobianTape, str, str]: tuple containing the compatible
            JacobianTape, the interface to apply, and the method argument
            to pass to the ``JacobianTape.jacobian`` method

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

        return ReversibleTape, interface, "analytic"

    @staticmethod
    def _validate_device_method(device, interface):
        """Validates whether a particular device and JacobianTape interface
        supports the ``"device"`` differentiation method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface

        Returns:
            tuple[.JacobianTape, str, str]: tuple containing the compatible
            JacobianTape, the interface to apply, and the method argument
            to pass to the ``JacobianTape.jacobian`` method

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

        return JacobianTape, interface, "device"

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

        self.qtape = self._tape(caching=self._caching)

        with self.qtape:
            measurement_processes = self.func(*args, **kwargs)

        if not isinstance(measurement_processes, Sequence):
            measurement_processes = (measurement_processes,)

        if not all(isinstance(m, qml.tape.MeasurementProcess) for m in measurement_processes):
            raise qml.QuantumFunctionError(
                "A quantum function must return either a single measurement, "
                "or a nonempty sequence of measurements."
            )

        state_returns = any([m.return_type is State for m in measurement_processes])

        # apply the interface (if any)
        if self.interface is not None:
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

        # provide the jacobian options
        self.qtape.jacobian_options = self.diff_options

        stop_at = self.device.operations

        # Hotfix that allows controlled rotations to return the correct gradients
        # when using the parameter shift rule.
        if isinstance(self.qtape, QubitParamShiftTape):
            # controlled rotations aren't supported by the parameter-shift rule
            stop_at = set(self.device.operations) - {"CRX", "CRZ", "CRY", "CRot"}

        # expand out the tape, if any operations are not supported on the device
        if not {op.name for op in self.qtape.operations}.issubset(stop_at):
            self.qtape = self.qtape.expand(
                depth=self.max_expansion, stop_at=lambda obj: obj.name in stop_at
            )

    def __call__(self, *args, **kwargs):

        if self.interface == "autograd":
            # HOTFIX: to maintain compatibility with core, here we treat
            # all inputs that do not explicitly specify `requires_grad=False`
            # as trainable. This should be removed at some point, forcing users
            # to specify `requires_grad=True` for trainable parameters.
            args = [
                anp.array(a, requires_grad=True) if not hasattr(a, "requires_grad") else a
                for a in args
            ]

        # construct the tape
        self.construct(args, kwargs)

        if self._caching:
            # Every time the QNode is called, it creates a new tape. We want the tape cache to
            # persist over multiple tapes, so hence keep track of it as a QNode attribute and
            # load it into the new tape
            self.qtape._cache_execute = self._cache_execute

        # execute the tape
        res = self.qtape.execute(device=self.device)

        # HOTFIX: to maintain compatibility with core, we squeeze
        # all outputs.

        # Get the namespace associated with the return type
        res_type_namespace = res.__class__.__module__.split(".")[0]

        if res_type_namespace in ("pennylane", "autograd"):
            # For PennyLane and autograd we must branch, since
            # 'squeeze' does not exist in the top-level of the namespace
            return anp.squeeze(res)

        if self._caching:
            self._cache_execute = self.qtape._cache_execute

        return __import__(res_type_namespace).squeeze(res)

    def draw(self, charset="unicode"):
        """Draw the quantum tape as a circuit diagram.

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

        Args:
            charset (str, optional): The charset that should be used. Currently, "unicode" and
                "ascii" are supported.

        Raises:
            ValueError: if the given charset is not supported
            .QuantumFunctionError: drawing is impossible because the underlying
                quantum tape has not yet been constructed

        Returns:
            str: the circuit representation of the tape

        """
        if self.qtape is None:
            raise qml.QuantumFunctionError(
                "The QNode can only be drawn after its quantum tape has been constructed."
            )

        return self.qtape.draw(charset=charset)

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
        self.interface = "autograd"
        self.dtype = AutogradInterface.dtype

        if self.qtape is not None:
            AutogradInterface.apply(self.qtape)

    @property
    def caching(self):
        """float: number of device executions to store in a cache to speed up subsequent
        executions. If set to zero, no caching occurs."""
        return self._caching

    INTERFACE_MAP = {"autograd": to_autograd, "torch": to_torch, "tf": to_tf}


def qnode(device, interface="autograd", diff_method="best", caching=0, **diff_options):
    """Decorator for creating QNodes.

    This decorator is used to indicate to PennyLane that the decorated function contains a
    :ref:`quantum variational circuit <glossary_variational_circuit>` that should be bound to a
    compatible device.

    The QNode calls the quantum function to construct a :class:`~.JacobianTape` instance representing
    the quantum circuit.

    .. note::

        The quantum tape is an *experimental* feature. QNodes that use the quantum
        tape have access to advanced features, such as in-QNode classical processing,
        but do not yet have feature parity with the standard PennyLane QNode.

        This quantum tape-comaptible QNode can either be created directly,

        >>> import pennylane as qml
        >>> @qml.tape.qnode(dev)

        or enabled globally via :func:`~.enable_tape` without changing your PennyLane code:

        >>> qml.enable_tape()

        For more details, see :mod:`pennylane.tape`.

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

            * ``"device"``: Queries the device directly for the gradient.
              Only allowed on devices that provide their own gradient rules.

            * ``"parameter-shift"``: Use the analytic parameter-shift
              rule for all supported quantum operation arguments, with finite-difference
              as a fallback.

            * ``"finite-diff"``: Uses numerical finite-differences for all quantum
              operation arguments.

        caching (int): Number of device executions to store in a cache to speed up subsequent
            executions. A value of ``0`` indicates that no caching will take place. Once filled,
            older elements of the cache are removed and replaced with the most recent device
            executions to keep the cache up to date.

    Keyword Args:
        h=1e-7 (float): Step size for the finite difference method.
        order=1 (int): The order of the finite difference method to use. ``1`` corresponds
            to forward finite differences, ``2`` to centered finite differences.

    **Example**

    >>> qml.enable_tape()
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
            caching=caching,
            **diff_options,
        )
        return update_wrapper(qn, func)

    return qfunc_decorator
