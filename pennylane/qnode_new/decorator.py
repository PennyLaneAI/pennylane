# Copyright 2019 Xanadu Quantum Technologies Inc.

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

from .cv import CVQNode
from .jacobian import JacobianQNode
from .qnode import QNode as BaseQNode
from .qubit import QubitQNode


def QNode(
    func, device, *, interface="autograd", mutable=True, differentiable=True, properties=None
):
    """QNode constructor for creating QNodes.

    When applied to a quantum function and device, converts it into
    a :class:`QNode` instance.

    Args:
        func (callable): a quantum function
        device (~.Device): a PennyLane-compatible device
        interface (str): The interface that will be used for classical processing
            and automatic differentiation. This affects
            the types of objects that can be passed to/returned from the QNode:

            * ``interface='autograd'``: The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays.

            * ``interface='torch'``: The QNode accepts and returns Torch tensors.

            * ``interface='tfe'``: The QNode accepts and returns eager execution
              TensorFlow ``tfe.Variable`` objects.

        mutable (bool): whether the QNode circuit is mutable
        differentiable (bool): whether the QNode is differentiable
        properties (dict[str->Any]): additional keyword properties passed to the QNode
    """
    if not differentiable:
        # QNode is not differentiable
        return BaseQNode(func, device, mutable=mutable, properties=properties)

    # Query the device to determine what analytic Jacobian QNodes
    # are supported.

    # TODO: should the device declare supported QNode
    # classes, as well as the default QNode to use?

    # Set the default model to qubit, for backwards compatability with existing plugins
    # TODO: once all plugins have been updated to add `model` to their
    # capabilities plugin, change the logic here.
    model = device.capabilities().get("model", "qubit")

    if model == "qubit":
        node = QubitQNode(func, device, mutable=mutable, properties=properties)
    elif model == "cv":
        node = CVQNode(func, device, mutable=mutable, properties=properties)
    else:
        # unknown circuit type, default to finite differences
        node = JacobianQNode(func, device, mutable=mutable, properties=properties)

    if interface == "torch":
        return node.to_torch()

    if interface == "tf":
        return node.to_tf()

    if interface in ("autograd", "numpy"):
        # keep "numpy" for backwards compatibility
        return node.to_autograd()

    # if no interface is specified, return the 'bare' QNode
    return node


def qnode(device, *, interface="autograd", mutable=True, differentiable=True, properties=None):
    """Decorator for creating QNodes.

    When applied to a quantum function, this decorator converts it into
    a :class:`QNode` instance.

    Args:
        device (~.Device): a PennyLane-compatible device
        interface (str): The interface that will be used for classical processing
            and automatic differentiation. This affects
            the types of objects that can be passed to/returned from the QNode:

            * ``interface='autograd'``: The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays.

            * ``interface='torch'``: The QNode accepts and returns Torch tensors.

            * ``interface='tfe'``: The QNode accepts and returns eager execution
              TensorFlow ``tfe.Variable`` objects.

        mutable (bool): whether the QNode circuit is mutable
        differentiable (bool): whether the QNode is differentiable
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
            differentiable=differentiable,
            properties=properties,
        )

    return qfunc_decorator
