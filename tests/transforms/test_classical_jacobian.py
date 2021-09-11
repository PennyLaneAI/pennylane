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
y = np.array([[1.0, 2.0], [4.0, 5.0]])
z = np.array([2.1, -0.3, 0.62, 0.89])


def circuit_0(a):
    [qml.RX(a, wires=0) for i in range(4)]
    return qml.expval(qml.PauliZ(0))


def circuit_1(a, b):
    qml.RX(qml.math.sin(a), wires=0)
    qml.RZ(a / 3, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(b ** 2, wires=1)
    qml.RZ(1 / b, wires=1)
    return qml.expval(qml.PauliZ(0))


def circuit_2(x):
    for _x in x:
        qml.RX(_x, wires=0)
    return qml.expval(qml.PauliZ(0))


def circuit_3(x, y):
    for _x in x:
        qml.RX(_x, wires=0)
    for i in range(len(y)):
        [qml.RY(_y, wires=1) for _y in y[i]]
    return qml.expval(qml.PauliZ(0))


perm_3 = ([2, 0, 1], [1, 2, 0, 3])


def circuit_4(x, y):
    for i in perm_3[0]:
        qml.RX(x[i], wires=0)
    for j in perm_3[1]:
        qml.RY(y[j // 2, j % 2], wires=1)
    return qml.expval(qml.PauliZ(0))


def circuit_5(x, y, z):
    for _x in x:
        qml.RX(_x, wires=0)
    qml.RZ(y[0, 1] * y[1, 0], wires=1)
    qml.RY(z[0] + 0.2 * z[1] ** 2, wires=1)
    return qml.expval(qml.PauliZ(0))


circuits = [circuit_0, circuit_1, circuit_2, circuit_3, circuit_4, circuit_5]
args = [(a,), (a, b), (x,), (x, y), (x, y), (x, y, z)]
interfaces = ["jax", "autograd", "tf", "torch"]

class_jacs = [
    (np.ones(4),),
    (
        np.array(
            [
                np.cos(a),
                1 / 3,
                0.0,
                0.0,
            ]
        ),
        np.array([0.0, 0.0, 2 * b, -1 / (b ** 2)]),
    ),
    (np.eye(len(x)),),
    (
        np.vstack([np.eye(len(x)), np.zeros((4, 3))]),
        np.vstack([np.zeros((3,) + y.shape), np.eye(np.prod(y.shape)).reshape(-1, *y.shape)]),
    ),
    (
        np.vstack([np.eye(len(x)), np.zeros((4, 3))])[perm_3[0] + [3, 4, 5, 6]],
        np.vstack(
            [np.zeros((3,) + y.shape), np.eye(np.prod(y.shape))[perm_3[1]].reshape(-1, *y.shape)]
        ),
    ),
    (
        np.vstack([np.eye(len(x)), np.zeros((2, 3))]),
        np.vstack(
            [
                np.zeros((3,) + y.shape),
                np.array([[[0.0, y[1, 0]], [y[0, 1], 0.0]]]),
                np.zeros((1,) + y.shape),
            ]
        ),
        np.vstack([np.zeros((4, 4)), np.array([1, 0.4 * z[1], 0.0, 0.0])]),
    ),
]


expected_outputs_without_argnums = {
    "jax": [_jac[0] for _jac in class_jacs],
    "autograd": class_jacs,
    "tf": class_jacs,
    "torch": class_jacs,
}


@pytest.mark.parametrize("i, circuit_args", enumerate(zip(circuits, args)))
@pytest.mark.parametrize("interface", interfaces)
def test_without_argnums(i, circuit_args, interface):
    circuit, args = circuit_args
    if interface == "tf":
        tf = pytest.importorskip("tensorflow")
        args = tuple((tf.constant(arg, dtype=tf.double) for arg in args))
    elif interface == "torch":
        torch = pytest.importorskip("torch")
        args = tuple((torch.tensor(arg) for arg in args))
    elif interface == "jax":
        # Do not need the package but skip if JAX device not available
        pytest.importorskip("jax")

    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface)
    jac = classical_jacobian(qnode)(*args)
    expected_jac = expected_outputs_without_argnums[interface][i]
    if interface == "autograd" and all((np.isscalar(arg) for arg in args)):
        expected_jac = qml.math.stack(expected_jac).T

    if isinstance(jac, tuple):
        for i in range(len(jac)):
            assert np.allclose(jac[i], expected_jac[i])
    else:
        assert np.allclose(jac, expected_jac)


scalar_argnums = [0, 1, 0, 1, 0, 1]
expected_outputs_with_scalar_argnums = {
    "jax": [_jac[argnum] for _jac, argnum in zip(class_jacs, scalar_argnums)],
    "autograd": [_jac[argnum] for _jac, argnum in zip(class_jacs, scalar_argnums)],
    "tf": [_jac[argnum] for _jac, argnum in zip(class_jacs, scalar_argnums)],
    "torch": [_jac[argnum] for _jac, argnum in zip(class_jacs, scalar_argnums)],
}


@pytest.mark.parametrize("i, circuit_args_argnums", enumerate(zip(circuits, args, scalar_argnums)))
@pytest.mark.parametrize("interface", interfaces)
def test_with_scalar_argnums(i, circuit_args_argnums, interface):
    circuit, args, argnums = circuit_args_argnums
    if interface == "tf":
        tf = pytest.importorskip("tensorflow")
        args = tuple((tf.constant(arg, dtype=tf.double) for arg in args))
    elif interface == "torch":
        torch = pytest.importorskip("torch")
        args = tuple((torch.tensor(arg) for arg in args))
    elif interface == "jax":
        # Do not need the package but skip if JAX device not available
        pytest.importorskip("jax")

    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface)
    jac = classical_jacobian(qnode, argnums=argnums)(*args)
    expected_jac = expected_outputs_with_scalar_argnums[interface][i]
    # NOTE: For Autograd we use stacking to replicate qml.jacobian behaviour for scalar-only inputs
    if interface == "autograd" and all((np.isscalar(arg) for arg in args)):
        expected_jac = qml.math.stack(expected_jac).T

    if isinstance(jac, tuple):
        for i in range(len(jac)):
            assert np.allclose(jac[i], expected_jac[i])
    else:
        assert np.allclose(jac, expected_jac)


single_list_argnums = [[0], [1], [0], [1], [0], [2]]
expected_outputs_with_single_list_argnums = {
    "jax": [(_jac[argnum[0]],) for _jac, argnum in zip(class_jacs, single_list_argnums)],
    "autograd": [(_jac[argnum[0]],) for _jac, argnum in zip(class_jacs, single_list_argnums)],
    "tf": [(_jac[argnum[0]],) for _jac, argnum in zip(class_jacs, single_list_argnums)],
    "torch": [(_jac[argnum[0]],) for _jac, argnum in zip(class_jacs, single_list_argnums)],
}


@pytest.mark.parametrize(
    "i, circuit_args_argnums", enumerate(zip(circuits, args, single_list_argnums))
)
@pytest.mark.parametrize("interface", interfaces)
def test_with_single_list_argnums(i, circuit_args_argnums, interface):
    circuit, args, argnums = circuit_args_argnums
    if interface == "tf":
        tf = pytest.importorskip("tensorflow")
        args = tuple((tf.constant(arg, dtype=tf.double) for arg in args))
    elif interface == "torch":
        torch = pytest.importorskip("torch")
        args = tuple((torch.tensor(arg) for arg in args))
    elif interface == "jax":
        # Do not need the package but skip if JAX device not available
        pytest.importorskip("jax")

    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface)
    jac = classical_jacobian(qnode, argnums=argnums)(*args)
    expected_jac = expected_outputs_with_single_list_argnums[interface][i]
    # NOTE: Here we skip the stacking part for Autograd as a tuple is expected if argnums is an iterable

    if isinstance(jac, tuple):
        for i in range(len(jac)):
            assert np.allclose(jac[i], expected_jac[i])
    else:
        assert np.allclose(jac, expected_jac)


list_argnums = [[0], [0, 1], [0], [0, 1], [0, 1], [0, 2]]
expected_outputs_with_list_argnums = {
    "jax": [
        tuple((_jac[_num] for _num in argnum)) for _jac, argnum in zip(class_jacs, list_argnums)
    ],
    "autograd": [
        tuple((_jac[_num] for _num in argnum)) for _jac, argnum in zip(class_jacs, list_argnums)
    ],
    "torch": [
        tuple((_jac[_num] for _num in argnum)) for _jac, argnum in zip(class_jacs, list_argnums)
    ],
    "tf": [
        tuple((_jac[_num] for _num in argnum)) for _jac, argnum in zip(class_jacs, list_argnums)
    ],
}


@pytest.mark.parametrize("i, circuit_args_argnums", enumerate(zip(circuits, args, list_argnums)))
@pytest.mark.parametrize("interface", interfaces)
def test_with_list_argnums(i, circuit_args_argnums, interface):
    circuit, args, argnums = circuit_args_argnums
    if interface == "tf":
        tf = pytest.importorskip("tensorflow")
        args = tuple((tf.constant(arg, dtype=tf.double) for arg in args))
    elif interface == "torch":
        torch = pytest.importorskip("torch")
        args = tuple((torch.tensor(arg) for arg in args))
    elif interface == "jax":
        # Do not need the package but skip if JAX device not available
        pytest.importorskip("jax")

    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface)
    jac = classical_jacobian(qnode, argnums=argnums)(*args)
    expected_jac = expected_outputs_with_list_argnums[interface][i]
    # NOTE: Here we skip the stacking part for Autograd as a tuple is expected if argnums is an iterable

    if isinstance(jac, tuple):
        for i in range(len(jac)):
            assert np.allclose(jac[i], expected_jac[i])
    else:
        assert np.allclose(jac, expected_jac)
