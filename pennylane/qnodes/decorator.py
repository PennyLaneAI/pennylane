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
QNode decorator.
"""
from functools import lru_cache

from .base import BaseQNode
from .cv import CVQNode
from .device_jacobian import DeviceJacobianQNode
from .jacobian import JacobianQNode
from .qubit import QubitQNode
from .passthru import PassthruQNode


PARAMETER_SHIFT_QNODES = {"qubit": QubitQNode, "cv": CVQNode}
ALLOWED_DIFF_METHODS = ("best", "backprop", "device", "parameter-shift", "finite-diff")
ALLOWED_INTERFACES = ("autograd", "numpy", "torch", "tf")


def _get_qnode_class(device, interface, diff_method):
    """Returns the class for the specified QNode.

    Args:
        device (~.Device): a PennyLane-compatible device
        interface (str): the interface that will be used for classical backpropagation
        diff_method (str, None): the method of differentiation to use in the created QNode

    Raises:
        ValueError: if an unrecognized ``diff_method`` is provided

    Returns:
        ~.BaseQNode: the QNode class object that is compatible with the provided device and
        differentiation method
    """
    # pylint: disable=too-many-return-statements,too-many-branches
    model = device.capabilities().get("model", "qubit")
    passthru_interface = device.capabilities().get("passthru_interface", None)
    device_provides_jacobian = device.capabilities().get("provides_jacobian", False)

    allows_passthru = passthru_interface is not None

    if diff_method is None:
        # QNode is not differentiable
        return BaseQNode

    if diff_method == "best":

        if allows_passthru and interface == passthru_interface:
            # hand off differentiation to the device without type conversion
            return PassthruQNode

        if device_provides_jacobian:
            # hand off differentiation to the device
            return DeviceJacobianQNode

        if model in PARAMETER_SHIFT_QNODES:
            # parameter-shift analytic differentiation
            return PARAMETER_SHIFT_QNODES[model]

    if diff_method == "backprop":
        if allows_passthru:
            if interface != passthru_interface:
                raise ValueError(
                    "Device {} only supports the {} interface when "
                    "diff_method='backprop'".format(device.short_name, passthru_interface)
                )
            return PassthruQNode

        raise ValueError(
            "The {} device does not support native computations with "
            "autodifferentiation frameworks.".format(device.short_name)
        )

    if diff_method == "device":
        if device_provides_jacobian:
            return DeviceJacobianQNode

        raise ValueError(
            "The {} device does not provide a native method "
            "for computing the jacobian.".format(device.short_name)
        )

    if diff_method == "parameter-shift":
        if model in PARAMETER_SHIFT_QNODES:
            # parameter-shift analytic differentiation
            return PARAMETER_SHIFT_QNODES[model]

        raise ValueError(
            "The parameter shift rule is not available for devices with model {}.".format(model)
        )

    if diff_method in ALLOWED_DIFF_METHODS:
        # finite differences
        return JacobianQNode

    raise ValueError(
        "Differentiation method {} not recognized. Allowed "
        "options are {}".format(diff_method, ALLOWED_DIFF_METHODS)
    )


def _apply_interface(qnode_, interface, diff_method):
    """Applies an interface to a specified QNode.

    Args:
        qnode_ (~.BaseQNode): the QNode to which the interface is applied
        interface (str): the interface that will be used for classical backpropagation
        diff_method (str, None): the method of differentiation to use in the created QNode

    Raises:
        ValueError: if an unrecognized or invalid ``interface`` is provided

    Returns:
        callable: the QNode method that creates the interface
    """
    if interface is None:
        # if no interface is specified, return the 'bare' QNode
        return qnode_

    if interface == "torch":
        return qnode_.to_torch()

    if interface == "tf":
        return qnode_.to_tf()

    if interface in ("autograd", "numpy"):
        # keep "numpy" for backwards compatibility
        return qnode_.to_autograd()

    raise ValueError(
        "Interface {} not recognized. Allowed "
        "interfaces are {}".format(diff_method, ALLOWED_INTERFACES)
    )


def QNode(func, device, *, interface="autograd", mutable=True, diff_method="best", **kwargs):
    """QNode constructor for creating QNodes.

    When applied to a quantum function and device, converts it into
    a :class:`QNode` instance.

    **Example**

    >>> def circuit(x):
    >>>     qml.RX(x, wires=0)
    >>>     return qml.expval(qml.PauliZ(0))
    >>> dev = qml.device("default.qubit", wires=1)
    >>> qnode = QNode(circuit, dev)

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

        mutable (bool): whether the QNode circuit is mutable
        diff_method (str, None): the method of differentiation to use in the created QNode.

            * ``"best"``: Best available method. Uses classical backpropagation or the
              device directly to compute the gradient if supported, otherwise will use
              the analytic parameter-shift rule where possible with finite-difference as a fallback.

            * ``"backprop"``: Use classical backpropagation. Only allowed on simulator
              devices that are classically end-to-end differentiable, for example
              :class:`default.tensor.tf <~.DefaultTensorTF>`. Note that the returned
              QNode can only be used with the machine learning framework supported
              by the device; a separate ``interface`` argument should not be passed.

            * ``"device"``: Queries the device directly for the gradient.
              Only allowed on devices that provide their own gradient rules.

            * ``"parameter-shift"``: Use the analytic parameter-shift
              rule where possible, with finite-difference as a fallback.

            * ``"finite-diff"``: Uses numerical finite-differences for all parameters.

            * ``None``: a non-differentiable QNode is returned.

    Keyword Args:
        h (float): Step size for the finite difference method. Default is ``1e-7`` for analytic devices, or
            ``0.3`` for non-analytic devices (those that estimate expectation values with a finite number of shots).
        order (int): order for the finite-difference method, must be 1 (default) or 2
    """
    qnode_class = _get_qnode_class(device, interface, diff_method)
    qnode_ = qnode_class(func, device, mutable=mutable, **kwargs)

    if not isinstance(qnode_, PassthruQNode):
        # PassthruQNode's do not support interface conversions
        qnode_ = _apply_interface(qnode_, interface, diff_method)

    return qnode_


def qnode(device, *, interface="autograd", mutable=True, diff_method="best", **kwargs):
    """Decorator for creating QNodes.

    When applied to a quantum function, this decorator converts it into
    a :class:`QNode` instance.

    **Example**

    >>> dev = qml.device("default.qubit", wires=1)
    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=0)
    >>>     return qml.expval(qml.PauliZ(0))

    Args:
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

        mutable (bool): whether the QNode circuit is mutable
        diff_method (str, None): the method of differentiation to use in the created QNode.

            * ``"best"``: Best available method. Uses classical backpropagation or the
              device directly to compute the gradient if supported, otherwise will use
              the analytic parameter-shift rule where possible with finite-difference as a fallback.

            * ``"backprop"``: Use classical backpropagation. Only allowed on simulator
              devices that are classically end-to-end differentiable, for example
              :class:`default.tensor.tf <~.DefaultTensorTF>`. Note that the returned
              QNode can only be used with the machine learning framework supported
              by the device; a separate ``interface`` argument should not be passed.

            * ``"device"``: Queries the device directly for the gradient.
              Only allowed on devices that provide their own gradient rules.

            * ``"parameter-shift"``: Use the analytic parameter-shift
              rule where possible, with finite-difference as a fallback.

            * ``"finite-diff"``: Uses numerical finite-differences for all parameters.

            * ``None``: a non-differentiable QNode is returned.

    Keyword Args:
        h (float): Step size for the finite difference method. Default is ``1e-7`` for analytic devices, or
            ``0.3`` for non-analytic devices (those that estimate expectation values with a finite number of shots).
        order (int): order for the finite-difference method, must be 1 (default) or 2
   """

    @lru_cache()
    def qfunc_decorator(func):
        """The actual decorator"""
        return QNode(
            func, device, interface=interface, mutable=mutable, diff_method=diff_method, **kwargs
        )

    return qfunc_decorator
