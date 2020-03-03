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


PARAMETER_SHIFT_QNODES = {"qubit": QubitQNode, "cv": CVQNode}
ALLOWED_DIFF_METHODS = ("best", "parameter-shift", "finite-diff")
ALLOWED_INTERFACES = ("autograd", "numpy", "torch", "tf")


def QNode(func, device, *, interface="autograd", mutable=True, diff_method="best", h=None, properties=None):
    """QNode constructor for creating QNodes.

    When applied to a quantum function and device, converts it into
    a :class:`QNode` instance.

    **Example:**

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
              through the QNode.The QNode accepts and returns Torch tensors.

            * ``interface='tf'``: Allows TensorFlow in eager mode to backpropogate
              through the QNode.The QNode accepts and returns
              TensorFlow ``tf.Variable`` and ``tf.tensor`` objects.

            * ``None``: The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays. It does not connect to any
              machine learning library automatically for backpropagation.

        mutable (bool): whether the QNode circuit is mutable
        diff_method (str, None): the method of differentiation to use in the created QNode.

            * ``"best"``: Best available method. Uses the device directly to compute
              the gradient if supported, otherwise will use the analytic parameter-shift
              rule where possible with finite-difference as a fallback.

            * ``"parameter-shift"``: Use the analytic parameter-shift
              rule where possible with finite-difference as a fallback.

            * ``"finite-diff"``: Uses numerical finite-differences.

            * ``None``: a non-differentiable QNode is returned.

        h (float): step size for parameter shift or the finite
            difference method

        properties (dict[str->Any]): additional keyword properties passed to the QNode
    """
    if diff_method is None:
        # QNode is not differentiable
        return BaseQNode(func, device, mutable=mutable, properties=properties)

    if diff_method not in ALLOWED_DIFF_METHODS:
        raise ValueError(
            "Differentiation method {} not recognized. Allowed "
            "options are {}".format(diff_method, ALLOWED_DIFF_METHODS)
        )

    # Set the default model to qubit, for backwards compatability with existing plugins
    # TODO: once all plugins have been updated to add `model` to their
    # capabilities dictionary, update the logic here
    model = device.capabilities().get("model", "qubit")
    device_jacobian = device.capabilities().get("provides_jacobian", False)

    if device_jacobian and (diff_method == "best"):
        # hand off differentiation to the device
        node = DeviceJacobianQNode(func, device, mutable=mutable, properties=properties)

    elif model in PARAMETER_SHIFT_QNODES and diff_method in ("best", "parameter-shift"):
        # parameter-shift analytic differentiation
        node = PARAMETER_SHIFT_QNODES[model](func, device, mutable=mutable, properties=properties)

    else:
        # finite differences
        node = JacobianQNode(func, device, mutable=mutable, h=h, properties=properties)

    if interface == "torch":
        return node.to_torch()

    if interface == "tf":
        return node.to_tf()

    if interface in ("autograd", "numpy"):
        # keep "numpy" for backwards compatibility
        return node.to_autograd()

    if interface is None:
        # if no interface is specified, return the 'bare' QNode
        return node

    raise ValueError(
        "Interface {} not recognized. Allowed "
        "interfaces are {}".format(diff_method, ALLOWED_INTERFACES)
    )


def qnode(device, *, interface="autograd", mutable=True, diff_method="best", h=None, properties=None):
    """Decorator for creating QNodes.

    When applied to a quantum function, this decorator converts it into
    a :class:`QNode` instance.

    **Example:**

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
              through the QNode.The QNode accepts and returns Torch tensors.

            * ``interface='tf'``: Allows TensorFlow in eager mode to backpropogate
              through the QNode.The QNode accepts and returns
              TensorFlow ``tf.Variable`` and ``tf.tensor`` objects.

            * ``None``: The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays. It does not connect to any
              machine learning library automatically for backpropagation.

        mutable (bool): whether the QNode circuit is mutable
        diff_method (str, None): the method of differentiation to use in the created QNode.

            * ``"best"``: Best available method. Uses the device directly to compute
              the gradient if supported, otherwise will use the analytic parameter-shift
              rule where possible with finite-difference as a fallback.

            * ``"parameter-shift"``: Use the analytic parameter-shift
              rule where possible with finite-difference as a fallback.

            * ``"finite-diff"``: Uses numerical finite-differences.

            * ``None``: a non-differentiable QNode is returned.
        h (float): step size for parameter shift or the finite
            difference method

        properties (dict[str->Any]): additional keyword properties passed to the QNode
    """

    @lru_cache()
    def qfunc_decorator(func):
        """The actual decorator"""
        return QNode(
            func,
            device,
            interface=interface,
            mutable=mutable,
            diff_method=diff_method,
            h=h,
            properties=properties,
        )

    return qfunc_decorator
