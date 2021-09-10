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
import pytest
import numpy as np

import pennylane as qml
from pennylane.transforms.classical_jacobian import classical_jacobian

a = -2.1
b = 0.71
x = np.array([0.3, 2.3, 0.1])
y = np.array([[1, 2],[4, 5]], dtype=float)

circuits = []

def circuit_0(a):
    [qml.RX(a, wires=0) for i in range(4)]
    return qml.expval(qml.PauliZ(0))
circuits.append(circuit_0)

def circuit_1(a, b):
    qml.RX(qml.math.sin(a), wires=0)
    qml.RZ(a/3, wires=0)
    qml.CNOT(wires=[0,1])
    qml.RY(b**2, wires=1)
    qml.RZ(1/b, wires=1)
    return qml.expval(qml.PauliZ(0))
circuits.append(circuit_1)

def circuit_2(x):
    for _x in x:
        qml.RX(_x, wires=0)
    return qml.expval(qml.PauliZ(0))
circuits.append(circuit_2)

def circuit_3(x, y):
    for _x in x:
        qml.RX(_x, wires=0)
    for i in range(len(y)):
        [qml.RY(_y, wires=1) for _y in y[i]]
    return qml.expval(qml.PauliZ(0))
circuits.append(circuit_3)

perm_3 = ([2,0,1], [1,2,0,3])
def circuit_4(x, y):
    for i in perm_3[0]:
        qml.RX(x[i], wires=0)
    for j in perm_3[1]:
        qml.RY(y[j//2, j%2], wires=1)
    return qml.expval(qml.PauliZ(0))
circuits.append(circuit_4)

args = [(a,), (a,b), (x,), (x, y), (x, y),]

interfaces = ["jax", "autograd", "tf", "torch"]
class_jacs = [
    (np.ones(4),),
    (np.array([np.cos(a), 1/3, 0., 0.,]), np.array([0., 0., 2*b, -1/(b**2)]),),
    (np.eye(len(x)),),
    (
        np.vstack([np.eye(len(x)),np.zeros((4,3))]),
        np.vstack([np.zeros((3,)+y.shape), np.eye(np.prod(y.shape)).reshape(-1, *y.shape)])
    ),
    (
        np.vstack([np.eye(len(x)),np.zeros((4,3))])[perm_3[0]+[3,4,5,6]],
        np.vstack([np.zeros((3,)+y.shape), np.eye(np.prod(y.shape))[perm_3[1]].reshape(-1, *y.shape)])
    ),
]

expected_outputs = {
    "jax": [_jac[0] for _jac in class_jacs],
    "autograd": class_jacs,#[(_jac[0],) for _jac in class_jacs],
    "tf": class_jacs,
    "torch": class_jacs,
}


@pytest.mark.parametrize("i, circuit_args", enumerate(zip(circuits, args)))
@pytest.mark.parametrize("interface", interfaces)
def test_without_argnums(i, circuit_args, interface):
    circuit, args = circuit_args
    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface)
    if interface=="tf":
        import tensorflow as tf
        args = tuple((tf.constant(arg, dtype=tf.double) for arg in args))
    elif interface=="torch":
        import torch
        args = tuple((torch.tensor(arg) for arg in args))

    jac = classical_jacobian(qnode)(*args)
    expected_jac = expected_outputs[interface][i]
    if interface=="autograd" and all((np.isscalar(arg) for arg in args)):
        expected_jac = qml.math.stack(expected_jac).T

    print(jac)
    print(expected_jac)
    if isinstance(jac, tuple):
        for i in range(len(jac)):
            print(jac[i], expected_jac[i])
            assert np.allclose(jac[i], expected_jac[i])
    else:
        assert np.allclose(jac, expected_jac)





