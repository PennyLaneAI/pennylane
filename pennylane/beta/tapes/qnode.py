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
from functools import lru_cache, update_wrapper

import numpy as np

from pennylane import Device
from pennylane.beta.tapes import QuantumTape
from pennylane.beta.queuing import MeasurementProcess
from pennylane.beta.interfaces.autograd import AutogradInterface


class QuantumFunctionError(Exception):
    """Exception raised when an illegal operation is defined in a quantum function."""


class QNode:
    """Base class for quantum nodes in the hybrid computational graph.

    A *quantum node* encapsulates a :ref:`quantum function <intro_vcirc_qfunc>`
    (corresponding to a :ref:`variational circuit <glossary_variational_circuit>`)
    and the computational device it is executed on.

    The QNode calls the quantum function to construct a :class:`~.QuantumTape` instance representing
    the quantum circuit.

    .. note::

        As the quantum tape is a *beta* feature, the standard PennyLane
        measurement functions cannot be used. You will need to instead
        import modified measurement functions within the quantum tape:

        >>> from pennylane.beta.queuing import expval, var, sample, probs

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
              rule where possible, with finite-difference as a fallback.

            * ``"finite-diff"``: Uses numerical finite-differences for all parameters.

    Keyword Args:
        h=1e-7 (float): Step size for the finite difference method.
        order=1 (int): The order of the finite difference method to use. ``1`` corresponds
            to forward finite differences, ``2`` to centered finite differences.

    **Example**

    >>> from pennylane.beta.queuing import expval, var, sample, probs
    >>> from pennylane.beta.tapes import QNode
    >>> def circuit(x):
    >>>     qml.RX(x, wires=0)
    >>>     return expval(qml.PauliZ(0))
    >>> dev = qml.device("default.qubit", wires=1)
    >>> qnode = QNode(circuit, dev)
    """

    # pylint:disable=too-many-instance-attributes

    def __init__(self, func, device, interface="autograd", diff_method="best", **diff_options):

        if interface is not None and interface not in self.INTERFACE_MAP:
            raise QuantumFunctionError(
                f"Unknown interface {interface}. Interface must be "
                f"one of {self.INTERFACE_MAP.values()}."
            )

        if not isinstance(device, Device):
            raise QuantumFunctionError("Invalid device. Device must be a valid PennyLane device.")

        self.func = func
        self.device = device
        self.qtape = None

        self._tape, self.interface, self.diff_method = self._get_tape(
            device, interface, diff_method
        )
        self.diff_options = diff_options or {}
        self.diff_options["method"] = self.diff_method

        self.dtype = np.float64
        self.max_expansion = 2

    @staticmethod
    def _get_tape(device, interface, diff_method="best"):
        """Determine the best QuantumTape, differentiation method, and interface
        for a requested device, interface, and diff method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface
            diff_method (str): The requested method of differentiation. One of
                ``"best"``, ``"backprop"``, ``"reversible"``, ``"device"``,
                ``"parameter-shift"``, or ``"finite-diff"``.

        Returns:
            tuple[.QuantumTape, str, str]: tuple containing the compatible
            QuantumTape, the interface to apply, and the method argument
            to pass to the ``QuantumTape.jacobian`` method.
        """

        if diff_method == "best":
            return QNode._get_best_tape(device, interface)

        if diff_method == "backprop":
            return QNode._get_backprop_tape(device, interface)

        if diff_method == "device":
            return QNode._get_device_tape(device, interface)

        if diff_method == "finite-diff":
            return QuantumTape, interface, "numeric"

        raise QuantumFunctionError(
            f"Differentiation method {diff_method} not recognized. Allowed "
            "options are ('best', 'parameter-shift', 'backprop', 'finite-diff', 'device', 'reversible')."
        )

    @staticmethod
    def _get_best_tape(device, interface):
        """Returns the 'best' QuantumTape and differentiation method
        for a particular device and interface combination.

        This method attempts to determine support for differentiation
        methods using the following order:

        * ``"backprop"``
        * ``"device"``
        * ``"parameter-shift"``
        * ``"reversible"``
        * ``"finite-diff"``

        The first differentiation method that is supported (going from
        top to bottom) will be returned.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface

        Returns:
            tuple[.QuantumTape, str, str]: tuple containing the compatible
            QuantumTape, the interface to apply, and the method argument
            to pass to the ``QuantumTape.jacobian`` method.
        """
        try:
            return QNode._get_backprop_tape(device, interface)
        except QuantumFunctionError:
            try:
                return QNode._get_device_tape(device, interface)
            except QuantumFunctionError:
                # add parameter shift tapes here when available
                return QuantumTape, interface, "numeric"

    @staticmethod
    def _get_backprop_tape(device, interface):
        """Validates whether a particular device and QuantumTape interface
        supports the ``"backprop"`` differentiation method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface

        Returns:
            tuple[.QuantumTape, str, str]: tuple containing the compatible
            QuantumTape, the interface to apply, and the method argument
            to pass to the ``QuantumTape.jacobian`` method.

        Raises:
            QuantumFunctionError: If the device does not support backpropagation, or the
            interface provided is not compatible with the device.
        """
        # determine if the device supports backpropagation
        backprop_interface = device.capabilities().get("passthru_interface", None)

        if backprop_interface is not None:

            if interface == backprop_interface:
                return QuantumTape, None, "backprop"

            raise QuantumFunctionError(
                f"Device {device.short_name} only supports diff_method='backprop' when using the "
                f"{backprop_interface} interface."
            )

        raise QuantumFunctionError(
            f"The {device.short_name} device does not support native computations with "
            "autodifferentiation frameworks."
        )

    @staticmethod
    def _get_device_tape(device, interface):
        """Validates whether a particular device and QuantumTape interface
        supports the ``"device"`` differentiation method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface

        Returns:
            tuple[.QuantumTape, str, str]: tuple containing the compatible
            QuantumTape, the interface to apply, and the method argument
            to pass to the ``QuantumTape.jacobian`` method.

        Raises:
            QuantumFunctionError: if the device does not provide a native method for computing
            the Jacobian.
        """
        # determine if the device provides its own jacobian method
        provides_jacobian = device.capabilities().get("provides_jacobian", False)

        if not provides_jacobian:
            raise QuantumFunctionError(
                f"The {device.short_name} device does not provide a native "
                "method for computing the jacobian."
            )

        return QuantumTape, interface, "device"

    def construct(self, args, kwargs):
        """Call the quantum function with a tape context,
        ensuring the operations get queued."""
        self.qtape = self._tape()

        # apply the interface (if any)
        if self.interface is not None:
            self.INTERFACE_MAP[self.interface](self)

        with self.qtape:
            measurements = self.func(*args, **kwargs)

        if not isinstance(measurements, Sequence):
            measurements = (measurements,)

        if not all(isinstance(m, MeasurementProcess) for m in measurements):
            raise QuantumFunctionError(
                "A quantum function must return either a single measured observable "
                "or a nonempty sequence of measured observables."
            )

        if not all(ret == tape[0] for ret, tape in zip(measurements, self.qtape._obs)):
            raise QuantumFunctionError(
                "All measurements must be returned in the order they are measured."
            )

        # provide the jacobian options
        self.qtape.jacobian_options = self.diff_options

        # expand out the tape, if any operations are not supported on the device
        if not {op.name for op in self.qtape.operations}.issubset(self.device.operations):
            self.qtape = self.qtape.expand(depth=self.max_expansion, stop_at=self.device.operations)

    def __call__(self, *args, **kwargs):
        # construct the tape
        self.construct(args, kwargs)

        # execute the tape
        return self.qtape.execute(device=self.device)

    def to_tf(self, dtype=None):
        """Apply the TensorFlow interface to the internal quantum tape.

        Args:
            dtype (tf.dtype): The dtype that the TensorFlow QNode should
                output. If not provided, the default is ``tf.float64``.

        Raises:
            QuantumFunctionError: if TensorFlow >= 2.1 is not installed
        """
        # pylint: disable=import-outside-toplevel
        try:
            import tensorflow as tf
            from pennylane.beta.interfaces.tf import TFInterface

            self.interface = "tf"

            if not isinstance(self.dtype, tf.DType):
                self.dtype = None

            self.dtype = dtype or self.dtype or TFInterface.dtype

            if self.qtape is not None:
                TFInterface.apply(self.qtape, dtype=self.dtype)

        except ImportError:
            raise QuantumFunctionError(
                "TensorFlow not found. Please install the latest "
                "version of TensorFlow to enable the 'tf' interface."
            )

    def to_torch(self, dtype=None):
        """Apply the Torch interface to the internal quantum tape.

        Args:
            dtype (tf.dtype): The dtype that the Torch QNode should
                output. If not provided, the default is ``torch.float64``.

        Raises:
            QuantumFunctionError: if PyTorch >= 1.3 is not installed
        """
        # pylint: disable=import-outside-toplevel
        try:
            import torch
            from pennylane.beta.interfaces.torch import TorchInterface

            self.interface = "torch"

            if not isinstance(self.dtype, torch.dtype):
                self.dtype = None

            self.dtype = dtype or self.dtype or TorchInterface.dtype

            if self.qtape is not None:
                TorchInterface.apply(self.qtape, dtype=self.dtype)

        except ImportError:
            raise QuantumFunctionError(
                "PyTorch not found. Please install the latest "
                "version of PyTorch to enable the 'torch' interface."
            )

    def to_autograd(self):
        """Apply the Autograd interface to the internal quantum tape."""
        self.interface = "autograd"
        self.dtype = AutogradInterface.dtype

        if self.qtape is not None:
            AutogradInterface.apply(self.qtape)

    INTERFACE_MAP = {"autograd": to_autograd, "torch": to_torch, "tf": to_tf}


def qnode(device, interface="autograd", diff_method="best", **diff_options):
    """Decorator for creating QNodes.

    When applied to a quantum function, this decorator converts it into
    a :class:`QNode` instance.

    .. note::

        As the quantum tape is a *beta* feature, the standard PennyLane
        measurement functions cannot be used. You will need to instead
        import modified measurement functions within the quantum tape:

        >>> from pennylane.beta.queuing import expval, var, sample, probs

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
              rule where possible, with finite-difference as a fallback.

            * ``"finite-diff"``: Uses numerical finite-differences for all parameters.=

    Keyword Args:
        h=1e-7 (float): Step size for the finite difference method.
        order=1 (int): The order of the finite difference method to use. ``1`` corresponds
            to forward finite differences, ``2`` to centered finite differences.

    **Example**

    >>> from pennylane.beta.queuing import expval, var, sample, probs
    >>> from pennylane.beta.tapes import qnode
    >>> dev = qml.device("default.qubit", wires=1)
    >>> @qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=0)
    >>>     return expval(qml.PauliZ(0))
    """

    @lru_cache()
    def qfunc_decorator(func):
        """The actual decorator"""
        qn = QNode(func, device, interface=interface, diff_method=diff_method, **diff_options)
        return update_wrapper(qn, func)

    return qfunc_decorator
