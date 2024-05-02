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
"""Tests for the qml.gradients.classical_jacobian function."""
import numpy as np

# pylint: disable=too-many-arguments
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.gradients.classical_jacobian import classical_jacobian

a = pnp.array(-2.1, requires_grad=True)
b = pnp.array(0.71, requires_grad=True)
w = pnp.array([0.3, 2.3, 0.1], requires_grad=True)
x = pnp.array([0.3, 2.3, 0.1], requires_grad=True)
y = pnp.array([[1.0, 2.0], [4.0, 5.0]], requires_grad=True)
z = pnp.array([2.1, -0.3, 0.62, 0.89], requires_grad=True)


def circuit_0(val):
    qml.RZ(0.2, wires=0)
    _ = [qml.RX(val, wires=0) for i in range(4)]
    return qml.expval(qml.PauliZ(0))


def circuit_1(a_, b_):
    qml.RX(qml.math.sin(a_), wires=0)
    qml.RZ(a_ / 3, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(b_**2, wires=1)
    qml.RZ(1 / b_, wires=1)
    return qml.expval(qml.PauliZ(0))


def circuit_2(par):
    _ = [qml.RX(par[i], wires=0) for i in range(3)]
    return qml.expval(qml.PauliZ(0))


def circuit_3(par0, par1):
    _ = [qml.RX(p, wires=0) for p in par0]
    _ = [qml.RY(p, wires=1) for par in par1 for p in par]
    return qml.expval(qml.PauliZ(0))


perm_3 = ([2, 0, 1], [1, 2, 0, 3])


def circuit_4(x_, y_):
    for i in perm_3[0]:
        qml.RX(x_[i], wires=0)
    for j in perm_3[1]:
        qml.RY(y_[j // 2, j % 2], wires=1)
    return qml.expval(qml.PauliZ(0))


def circuit_5(par0, par1, par2):
    _ = [qml.RX(_x, wires=0) for _x in par0]
    qml.RZ(par1[0, 1] * par1[1, 0], wires=1)
    qml.RY(par2[0] + 0.2 * par2[1] ** 2, wires=1)
    return qml.expval(qml.PauliZ(0))


def circuit_6(par0, par1):
    _ = [qml.RX(p, wires=0) for p in par0]
    _ = [qml.RX(p, wires=0) for p in par1]
    return qml.expval(qml.PauliZ(0))


circuits = [circuit_0, circuit_1, circuit_2, circuit_3, circuit_4, circuit_5, circuit_6]
all_args = [(a,), (a, b), (x,), (x, y), (x, y), (x, y, z), (w, x)]

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
        np.array([0.0, 0.0, 2 * b, -1 / (b**2)]),
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
    tuple(np.eye(len(x) + len(w)).reshape((2, len(x), len(x) + len(w))).transpose([0, 2, 1])),
]

interfaces = ["auto", "autograd"]


@pytest.mark.autograd
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize("circuit, args, expected_jac", zip(circuits, all_args, class_jacs))
@pytest.mark.parametrize("interface", interfaces)
def test_autograd_without_argnum(circuit, args, expected_jac, diff_method, interface):
    r"""Test ``classical_jacobian`` with ``argnum=None`` and Autograd."""
    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode)(*args)

    if len(args) == 1:
        # For a single argument, the Jacobian is unpacked
        assert np.allclose(jac, expected_jac[0])
    else:
        assert len(jac) == len(expected_jac)
        for _jac, _expected_jac in zip(jac, expected_jac):
            assert np.allclose(_jac, _expected_jac)


interfaces = ["tf"]


@pytest.mark.tf
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize("circuit, args, expected_jac", zip(circuits, all_args, class_jacs))
@pytest.mark.parametrize("interface", interfaces)
def test_tf_without_argnum(circuit, args, expected_jac, diff_method, interface):
    r"""Test ``classical_jacobian`` with ``argnum=None`` and Tensorflow."""
    import tensorflow as tf

    args = tuple((tf.Variable(arg, dtype=tf.double) for arg in args))
    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode)(*args)

    assert len(jac) == len(expected_jac)
    for _jac, _expected_jac in zip(jac, expected_jac):
        assert np.allclose(_jac, _expected_jac)


interfaces = ["torch"]


@pytest.mark.torch
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize("circuit, args, expected_jac", zip(circuits, all_args, class_jacs))
@pytest.mark.parametrize("interface", interfaces)
def test_torch_without_argnum(circuit, args, expected_jac, diff_method, interface):
    r"""Test ``classical_jacobian`` with ``argnum=None`` and Torch."""
    import torch

    args = tuple((torch.tensor(arg, requires_grad=True) for arg in args))
    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode)(*args)

    assert len(jac) == len(expected_jac)
    for _jac, _expected_jac in zip(jac, expected_jac):
        assert np.allclose(_jac, _expected_jac)

    # also test with an untrainable argument
    args[0].requires_grad = False
    jac = classical_jacobian(qnode)(*args)

    assert len(jac) == len(expected_jac) - 1
    for _jac, _expected_jac in zip(jac, expected_jac[1:]):
        assert np.allclose(_jac, _expected_jac)


scalar_argnum = [0, 1, 0, 1, 0, 1, 0]

interfaces = ["auto", "autograd"]


@pytest.mark.autograd
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize(
    "circuit, args, expected_jac, argnum", zip(circuits, all_args, class_jacs, scalar_argnum)
)
@pytest.mark.parametrize("interface", interfaces)
def test_autograd_with_scalar_argnum(circuit, args, expected_jac, argnum, diff_method, interface):
    r"""Test ``classical_jacobian`` with ``argnum=<int>`` and Autograd."""
    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode, argnum=argnum)(*args)
    if interface == "auto":
        assert qnode.interface == "auto"
    expected_jac = expected_jac[argnum]
    assert np.allclose(jac, expected_jac)


interfaces = ["tf"]


@pytest.mark.tf
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize(
    "circuit, args, expected_jac, argnum", zip(circuits, all_args, class_jacs, scalar_argnum)
)
@pytest.mark.parametrize("interface", interfaces)
def test_tf_with_scalar_argnum(circuit, args, expected_jac, argnum, diff_method, interface):
    r"""Test ``classical_jacobian`` with ``argnum=<int>`` and TensorFlow."""
    import tensorflow as tf

    args = tuple((tf.Variable(arg, dtype=tf.double) for arg in args))
    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode, argnum=argnum)(*args)
    expected_jac = expected_jac[argnum]
    assert np.allclose(jac, expected_jac)


interfaces = ["torch"]


@pytest.mark.torch
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize(
    "circuit, args, expected_jac, argnum", zip(circuits, all_args, class_jacs, scalar_argnum)
)
@pytest.mark.parametrize("interface", interfaces)
def test_torch_with_scalar_argnum(circuit, args, expected_jac, argnum, diff_method, interface):
    r"""Test ``classical_jacobian`` with ``argnum=<int>`` and Torch."""
    import torch

    args = tuple((torch.tensor(arg) for arg in args))
    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode, argnum=argnum)(*args)
    expected_jac = expected_jac[argnum]
    assert np.allclose(jac, expected_jac)


single_list_argnum = [[0], [1], [0], [1], [0], [2], [0]]

interfaces = ["auto", "autograd"]


@pytest.mark.autograd
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize(
    "circuit, args, expected_jac, argnum", zip(circuits, all_args, class_jacs, single_list_argnum)
)
@pytest.mark.parametrize("interface", interfaces)
def test_autograd_with_single_list_argnum(
    circuit, args, expected_jac, argnum, diff_method, interface
):
    r"""Test ``classical_jacobian`` with ``argnum=Sequence[int]`` of length 1 and Autograd."""
    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode, argnum=argnum)(*args)
    expected_jac = (expected_jac[argnum[0]],)

    assert len(jac) == 1
    assert np.allclose(jac[0], expected_jac[0])


interfaces = ["tf"]


@pytest.mark.tf
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize(
    "circuit, args, expected_jac, argnum", zip(circuits, all_args, class_jacs, single_list_argnum)
)
@pytest.mark.parametrize("interface", interfaces)
def test_tf_with_single_list_argnum(circuit, args, expected_jac, argnum, diff_method, interface):
    r"""Test ``classical_jacobian`` with ``argnum=Sequence[int]`` of length 1 and TensorFlow."""
    import tensorflow as tf

    args = tuple((tf.Variable(arg, dtype=tf.double) for arg in args))
    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode, argnum=argnum)(*args)
    expected_jac = (expected_jac[argnum[0]],)
    assert len(jac) == 1
    assert np.allclose(jac[0], expected_jac[0])


interfaces = ["torch"]


@pytest.mark.torch
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize(
    "circuit, args, expected_jac, argnum", zip(circuits, all_args, class_jacs, single_list_argnum)
)
@pytest.mark.parametrize("interface", interfaces)
def test_torch_with_single_list_argnum(circuit, args, expected_jac, argnum, diff_method, interface):
    r"""Test ``classical_jacobian`` with ``argnum=Sequence[int]`` of length 1 and Torch."""
    import torch

    args = tuple((torch.tensor(arg) for arg in args))
    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode, argnum=argnum)(*args)
    expected_jac = (expected_jac[argnum[0]],)
    assert len(jac) == 1
    assert np.allclose(jac[0], expected_jac[0])


sequence_argnum = [[0], [0, 1], (0,), [0, 1], (0, 1), {0, 2}, [0, 1]]

interfaces = ["auto", "autograd"]


@pytest.mark.autograd
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize(
    "circuit, args, expected_jac, argnum", zip(circuits, all_args, class_jacs, sequence_argnum)
)
@pytest.mark.parametrize("interface", interfaces)
def test_autograd_with_sequence_argnum(circuit, args, expected_jac, argnum, diff_method, interface):
    r"""Test ``classical_jacobian`` with ``argnum=Sequence[int]`` and Autograd."""
    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode, argnum=argnum)(*args)
    expected_jac = tuple((expected_jac[num] for num in argnum))
    assert len(jac) == len(expected_jac)
    for _jac, _expected_jac in zip(jac, expected_jac):
        assert np.allclose(_jac, _expected_jac)


interfaces = ["tf"]


@pytest.mark.tf
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize(
    "circuit, args, expected_jac, argnum", zip(circuits, all_args, class_jacs, sequence_argnum)
)
@pytest.mark.parametrize("interface", interfaces)
def test_tf_with_sequence_argnum(circuit, args, expected_jac, argnum, diff_method, interface):
    r"""Test ``classical_jacobian`` with ``argnum=Sequence[int]`` and TensorFlow."""
    import tensorflow as tf

    args = tuple((tf.Variable(arg, dtype=tf.double) for arg in args))
    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode, argnum=argnum)(*args)
    expected_jac = tuple((expected_jac[num] for num in argnum))
    assert len(jac) == len(expected_jac)
    for _jac, _expected_jac in zip(jac, expected_jac):
        assert np.allclose(_jac, _expected_jac)


interfaces = ["torch"]


@pytest.mark.torch
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize(
    "circuit, args, expected_jac, argnum", zip(circuits, all_args, class_jacs, sequence_argnum)
)
@pytest.mark.parametrize("interface", interfaces)
def test_torch_with_sequence_argnum(circuit, args, expected_jac, argnum, diff_method, interface):
    r"""Test ``classical_jacobian`` with ``argnum=Sequence[int]`` and Torch."""
    import torch

    args = tuple((torch.tensor(arg) for arg in args))
    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode, argnum=argnum)(*args)
    expected_jac = tuple((expected_jac[num] for num in argnum))
    assert len(jac) == len(expected_jac)
    for _jac, _expected_jac in zip(jac, expected_jac):
        assert np.allclose(_jac, _expected_jac)


expected_jac_not_trainable_only = np.array([0.0, 1.0, 1.0, 1.0, 1.0])

interfaces = ["auto", "autograd"]


@pytest.mark.autograd
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize("interface", interfaces)
def test_autograd_not_trainable_only(diff_method, interface):
    r"""Test ``classical_jacobian`` with ``argnum=<int>`` and Autograd
    with ``trainable_only=False`` ."""
    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit_0, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode, argnum=0, trainable_only=False)(a)
    assert np.allclose(jac, expected_jac_not_trainable_only)


interfaces = ["tf"]


@pytest.mark.tf
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize("interface", interfaces)
def test_tf_not_trainable_only(diff_method, interface):
    r"""Test ``classical_jacobian`` with ``argnum=<int>`` and Tensorflow
    with ``trainable_only=False`` ."""
    import tensorflow as tf

    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit_0, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode, argnum=0, trainable_only=False)(tf.Variable(a))
    assert np.allclose(jac, expected_jac_not_trainable_only)


interfaces = ["torch"]


@pytest.mark.torch
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize("interface", interfaces)
def test_torch_not_trainable_only(diff_method, interface):
    r"""Test ``classical_jacobian`` with ``argnum=<int>`` and Torch
    with ``trainable_only=False`` ."""
    import torch

    dev = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(circuit_0, dev, interface=interface, diff_method=diff_method)
    jac = classical_jacobian(qnode, argnum=0, trainable_only=False)(
        torch.tensor(a, requires_grad=True)
    )
    assert np.allclose(jac, expected_jac_not_trainable_only)


class_jacs = [
    (np.ones(4),),
    (
        np.array(
            [
                np.cos(a),
                1 / 3,
            ]
        ),
        np.array([2 * b, -1 / (b**2)]),
        tuple(
            [
                np.array(
                    [
                        np.cos(a),
                        1 / 3,
                        0.0,
                        0.0,
                    ]
                ),
                np.array([0.0, 0.0, 2 * b, -1 / (b**2)]),
            ]
        ),
    ),
    (np.eye(len(x)),),
    (
        np.vstack([np.eye(len(x))]),
        np.vstack([np.eye(np.prod(y.shape)).reshape(-1, *y.shape)]),
        tuple(
            [
                np.vstack([np.eye(len(x)), np.zeros((4, 3))]),
                np.vstack(
                    [np.zeros((3,) + y.shape), np.eye(np.prod(y.shape)).reshape(-1, *y.shape)]
                ),
            ]
        ),
    ),
    (
        np.vstack([np.eye(len(x))])[perm_3[0]],
        np.vstack([np.eye(np.prod(y.shape))[perm_3[1]].reshape(-1, *y.shape)]),
        tuple(
            [
                np.vstack([np.eye(len(x)), np.zeros((4, 3))])[perm_3[0] + [3, 4, 5, 6]],
                np.vstack(
                    [
                        np.zeros((3,) + y.shape),
                        np.eye(np.prod(y.shape))[perm_3[1]].reshape(-1, *y.shape),
                    ]
                ),
            ]
        ),
    ),
    (
        np.vstack([np.eye(len(x))]),
        np.vstack(
            [
                np.array([[[0.0, y[1, 0]], [y[0, 1], 0.0]]]),
            ]
        ),
        np.vstack([np.array([1, 0.4 * z[1], 0.0, 0.0])]),
        tuple(
            [
                np.vstack([np.eye(len(x)), np.zeros((1, 3))]),
                np.vstack([np.zeros((3, 4)), np.array([1, 0.4 * z[1], 0.0, 0.0])]),
            ]
        ),
    ),
    [
        np.eye(len(x)),
        np.eye(len(w)),
        tuple(np.eye(len(x) + len(w)).reshape((2, len(x), len(x) + len(w))).transpose([0, 2, 1])),
    ],
]


@pytest.mark.jax
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
class TestJax:
    """Class to test Jax integration with classical Jacobian."""

    interfaces = ["jax"]

    @pytest.mark.parametrize("circuit, args, expected_jac", zip(circuits, all_args, class_jacs))
    @pytest.mark.parametrize("interface", interfaces)
    def test_jax_without_argnum(self, circuit, args, expected_jac, diff_method, interface):
        r"""Test ``classical_jacobian`` with ``argnum=None`` and JAX."""
        import jax.numpy as jnp

        args = tuple((jnp.array(arg) for arg in args))
        # JAX behaviour: argnum=None yields only the Jacobian with respect to the first arg.
        expected_jac = expected_jac[0]
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
        jac = classical_jacobian(qnode)(*args)
        assert np.allclose(jac, expected_jac)

    @pytest.mark.parametrize(
        "circuit, args, expected_jac, argnum",
        zip(circuits, all_args, class_jacs, single_list_argnum),
    )
    @pytest.mark.parametrize("interface", interfaces)
    def test_jax_with_single_list_argnum(
        self, circuit, args, expected_jac, argnum, diff_method, interface
    ):
        r"""Test ``classical_jacobian`` with ``argnum=Sequence[int]`` of length 1 and JAX."""
        print(argnum)
        import jax.numpy as jnp

        args = tuple((jnp.array(arg) for arg in args))
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
        jac = classical_jacobian(qnode, argnum=argnum)(*args)
        expected_jac = (expected_jac[argnum[0]],)
        assert len(jac) == 1
        assert np.allclose(jac[0], expected_jac[0])

    @pytest.mark.parametrize(
        "circuit, args, expected_jac, argnum", zip(circuits, all_args, class_jacs, scalar_argnum)
    )
    @pytest.mark.parametrize("interface", interfaces)
    def test_jax_with_scalar_argnum(
        self, circuit, args, expected_jac, argnum, diff_method, interface
    ):
        r"""Test ``classical_jacobian`` with ``argnum=<int>`` and JAX."""
        import jax.numpy as jnp

        args = tuple((jnp.array(arg) for arg in args))
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
        jac = classical_jacobian(qnode, argnum=argnum)(*args)
        expected_jac = expected_jac[argnum]
        assert np.allclose(jac, expected_jac)

    @pytest.mark.parametrize(
        "circuit, args, expected_jac, argnum", zip(circuits, all_args, class_jacs, sequence_argnum)
    )
    @pytest.mark.parametrize("interface", interfaces)
    def test_jax_with_sequence_argnum(
        self, circuit, args, expected_jac, argnum, diff_method, interface
    ):
        r"""Test ``classical_jacobian`` with ``argnum=Sequence[int]`` and JAX."""
        import jax.numpy as jnp

        args = tuple((jnp.array(arg) for arg in args))
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
        jac = classical_jacobian(qnode, argnum=argnum)(*args)

        if len(argnum) > 1:
            expected_jac = expected_jac[-1]
        print(argnum)
        print(jac, expected_jac)
        assert len(jac) == len(expected_jac)
        for _jac, _expected_jac in zip(jac, expected_jac):
            assert np.allclose(_jac, _expected_jac)

    @pytest.mark.parametrize("interface", interfaces)
    def test_jax_not_trainable_only(self, diff_method, interface):
        r"""Test ``classical_jacobian`` with ``argnum=<int>`` and JAX
        with ``trainable_only=False`` ."""
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit_0, dev, interface=interface, diff_method=diff_method)
        jac = classical_jacobian(qnode, argnum=0, trainable_only=False)(a)
        assert np.allclose(jac, expected_jac_not_trainable_only)
