# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Class and functions for activating, deactivating and checking the new return types system
"""
# pylint: disable=global-statement
__activated = True


def enable_return():
    """Function that turns on the experimental return type system that prefers the use of sequences over arrays.

    The new system guarantees that a sequence (e.g., list or tuple) is returned based on the ``return`` statement of the
    quantum function. This system avoids the creation of ragged arrays, where multiple measurements are stacked
    together.

    **Example**

    The following example shows that for multiple measurements the current PennyLane system is creating a ragged tensor.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
              qml.Hadamard(wires=[0])
              qml.CRX(x, wires=[0, 1])
              return qml.probs(wires=[0]), qml.vn_entropy(wires=[0]), qml.probs(wires=1), qml.expval(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)

    >>> qnode(0.5)
    tensor([0.5       , 0.5       , 0.08014815, 0.96939564, 0.03060436,
        0.93879128], requires_grad=True)

    when you activate the new return type the result is simply a tuple containing each measurement.

    .. code-block:: python

        qml.enable_return()

        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
              qml.Hadamard(wires=[0])
              qml.CRX(x, wires=[0, 1])
              return qml.probs(wires=[0]), qml.vn_entropy(wires=[0]), qml.probs(wires=1), qml.expval(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)

    >>> qnode(0.5)
    (tensor([0.5, 0.5], requires_grad=True), tensor(0.08014815, requires_grad=True), tensor([0.96939564, 0.03060436], requires_grad=True), tensor(0.93879128, requires_grad=True))
    """

    global __activated
    __activated = True


def disable_return():
    """Function that turns off the new return type system.

    **Example**

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
              qml.Hadamard(wires=[0])
              qml.CRX(x, wires=[0, 1])
              return qml.probs(wires=[0]), qml.vn_entropy(wires=[0]), qml.probs(wires=1), qml.expval(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)


    >>> qml.enable_return()
    >>> res = qnode(0.5)
    >>> res
    (tensor([0.5, 0.5], requires_grad=True), tensor(0.08014815, requires_grad=True), tensor([0.96939564, 0.03060436], requires_grad=True), tensor(0.93879128, requires_grad=True))
    >>> qml.disable_return()
    >>> res = qnode(0.5)
    >>> res
    tensor([0.5       , 0.5       , 0.08014815, 0.96939564, 0.03060436, 0.93879128], requires_grad=True)

    """
    global __activated
    __activated = False  # pragma: no cover


def active_return():
    """Function that checks if the new return types system is activated.

    Returns:
        bool: Returns ``True`` if the new return types system is activated.

    **Example**

    By default, the new return types system is turned off:

    >>> active_return()
    False

    It can be activated:

    >>> enable_return()
    >>> active_return()
    True

    And it can also be deactivated:

    >>> enable_return()
    >>> active_return()
    True
    >>> disable_return()
    >>> active_return()
    False
    """
    return __activated
