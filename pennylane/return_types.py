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
__activated = False


def enable_return():
    """Function that turns on the new return type system. The new system guarantees intuitive return types such that a
    sequence (e.g., list or tuple) is returned based on the `return` statement of the quantum function. This system
    avoids the creation of ragged arrays, where multiple measurements are stacked together.

    **Example**

    The following example shows that for multiple measurements the current PennyLane system is creating a ragged tensor.
    ```python
    dev = qml.device("default.qubit", wires=2)

    def circuit(x):
          qml.Hadamard(wires=[0])
          qml.CRX(x, wires=[0, 1])
          return qml.probs(wires=[0]), qml.vn_entropy(wires=[0]), qml.probs(wires=1), qml.expval(qml.PauliZ(wires=1))

    qnode = qml.QNode(circuit, dev)
    ```

    ```pycon
    >>> res = qnode(0.5)
    >>> res
    tensor([0.5       , 0.5       , 0.08014815, 0.96939564, 0.03060436,
        0.93879128], requires_grad=True)
    ```

    when you activate the new return type the results are simply a tuple containing each measurement.
    ```python
    qml.enable_return()

    dev = qml.device("default.qubit", wires=2)

    def circuit(x):
          qml.Hadamard(wires=[0])
          qml.CRX(x, wires=[0, 1])
          return qml.probs(wires=[0]), qml.vn_entropy(wires=[0]), qml.probs(wires=1), qml.expval(qml.PauliZ(wires=1))

    qnode = qml.QNode(circuit, dev)
    ```

    ```pycon
    >>> res = qnode(0.5)
    >>> res
    (tensor([0.5, 0.5], requires_grad=True), tensor(0.08014815, requires_grad=True), tensor([0.96939564, 0.03060436], requires_grad=True), tensor(0.93879128, requires_grad=True))
    ```

    The new return types unlocks the use of `probs` mixed with diffrent measurements in backpropagation with Jax:

    ```
    import jax

    qml.enable_return()

    dev = qml.device("default.qubit", wires=2)
    qml.enable_return()

    @qml.qnode(dev, interface="jax")
    def circuit(a):
      qml.RX(a[0], wires=0)
      qml.CNOT(wires=(0, 1))
      qml.RY(a[1], wires=1)
      qml.RZ(a[2], wires=1)
      return qml.expval(qml.PauliZ(wires=0)), qml.probs(wires=[0, 1]), qml.vn_entropy(wires=1)

    x = jax.numpy.array([0.1, 0.2, 0.3])
    ```

    ```pycon
    res = jax.jacobian(circuit)(x)
    >>> res
    (DeviceArray([-9.9833414e-02, -7.4505806e-09, -3.9932679e-10], dtype=float32),
    DeviceArray([[-4.9419206e-02, -9.9086545e-02,  3.4938008e-09],
               [-4.9750542e-04,  9.9086538e-02,  1.2768372e-10],
               [ 4.9750548e-04,  2.4812977e-04,  4.8371929e-13],
               [ 4.9419202e-02, -2.4812980e-04,  2.6696912e-11]],            dtype=float32),
    DeviceArray([ 2.9899091e-01, -4.4703484e-08,  9.5104014e-10], dtype=float32))
    ```

    where before the following error was raised:
    ```ValueError: All input arrays must have the same shape.```
    """

    global __activated
    __activated = True


def disable_return():
    """Function that turns off the new return type system.

    **Example**
    dev = qml.device("default.qubit", wires=2)

    def circuit(x):
          qml.Hadamard(wires=[0])
          qml.CRX(x, wires=[0, 1])
          return qml.probs(wires=[0]), qml.vn_entropy(wires=[0]), qml.probs(wires=1), qml.expval(qml.PauliZ(wires=1))

    qnode = qml.QNode(circuit, dev)
    ```

    ```pycon
    >>> qml.enable_return()
    >>> res = qnode(0.5)
    >>> res
    (tensor([0.5, 0.5], requires_grad=True), tensor(0.08014815, requires_grad=True), tensor([0.96939564, 0.03060436], requires_grad=True), tensor(0.93879128, requires_grad=True))

    >>> qml.disable_return()
    >>> res = qnode(0.5)
    >>> res
    tensor([0.5       , 0.5       , 0.08014815, 0.96939564, 0.03060436, 0.93879128], requires_grad=True)
    ```

    """
    global __activated
    __activated = False  # pragma: no cover


def active_return():
    """Function that checks if the new return types system is activated."""
    return __activated
