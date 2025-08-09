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
"""
Tests for the LinearCombination class.
"""

# pylint: disable=too-many-public-methods, too-few-public-methods
from collections.abc import Iterable
from copy import copy

import numpy as np
import pytest
import scipy

import pennylane as qml
from pennylane import X, Y, Z
from pennylane import numpy as pnp
from pennylane.exceptions import DeviceError
from pennylane.ops import LinearCombination
from pennylane.pauli import PauliSentence, PauliWord
from pennylane.wires import Wires

# Make test data in different interfaces, if installed
COEFFS_PARAM_INTERFACE = [
    ([-0.05, 0.17], 1.7, "autograd"),
    (np.array([-0.05, 0.17]), np.array(1.7), "autograd"),
    (pnp.array([-0.05, 0.17], requires_grad=True), pnp.array(1.7, requires_grad=True), "autograd"),
]

try:
    import jax
    from jax import numpy as jnp

    COEFFS_PARAM_INTERFACE.append((jnp.array([-0.05, 0.17]), jnp.array(1.7), "jax"))
except ImportError:
    pass

try:
    import tensorflow as tf

    COEFFS_PARAM_INTERFACE.append(
        (tf.Variable([-0.05, 0.17], dtype=tf.double), tf.Variable(1.7, dtype=tf.double), "tf")
    )
except ImportError:
    pass

try:
    import torch

    COEFFS_PARAM_INTERFACE.append((torch.tensor([-0.05, 0.17]), torch.tensor(1.7), "torch"))
except ImportError:
    pass

H_ONE_QUBIT = np.array([[1.0, 0.5j], [-0.5j, 2.5]])

H_TWO_QUBITS = np.array(
    [[0.5, 1.0j, 0.0, -3j], [-1.0j, -1.1, 0.0, -0.1], [0.0, 0.0, -0.9, 12.0], [3j, -0.1, 12.0, 0.0]]
)

COEFFS = [(0.5, 1.2, -0.7), (2.2, -0.2, 0.0), (0.33,)]

OBSERVABLES = [
    (Z(0), Y(0), Z(1)),
    (X(0) @ Z(1), Y(0) @ Z(1), Z(1)),
    (qml.Hermitian(H_TWO_QUBITS, [0, 1]),),
]

valid_LinearCombinations = [
    ((1.0,), (qml.Hermitian(H_TWO_QUBITS, [0, 1]),)),
    ((-0.8,), (Z(0),)),
    ((0.6,), (X(0) @ X(1),)),
    ((0.5, -1.6), (X(0), Y(1))),
    ((0.5, -1.6), (X(1), Y(1))),
    ((0.5, -1.6), (X("a"), Y("b"))),
    ((1.1, -0.4, 0.333), (X(0), qml.Hermitian(H_ONE_QUBIT, 2), Z(2))),
    ((-0.4, 0.15), (qml.Hermitian(H_TWO_QUBITS, [0, 2]), Z(1))),
    ([1.5, 2.0], [Z(0), Y(2)]),
    (np.array([-0.1, 0.5]), [qml.Hermitian(H_TWO_QUBITS, [0, 1]), Y(0)]),
    ((0.5, 1.2), (X(0), X(0) @ X(1))),
    ((0.5 + 1.2j, 1.2 + 0.5j), (X(0), Y(1))),
    ((0.7 + 0j, 0 + 1.3j), (X(0), Y(1))),
]

invalid_LinearCombinations = [
    ((), (Z(0),)),
    ((), (Z(0), Y(1))),
    ((3.5,), ()),
    ((1.2, -0.4), ()),
    ((0.5, 1.2), (Z(0),)),
    ((1.0,), (Z(0), Y(0))),
]

simplify_LinearCombinations = [
    (
        qml.ops.LinearCombination([], []),
        qml.ops.LinearCombination([], []),
    ),
    (
        qml.ops.LinearCombination([1, 1, 1], [X(0) @ qml.Identity(1), X(0), X(1)]),
        qml.ops.LinearCombination([2, 1], [X(0), X(1)]),
    ),
    (
        qml.ops.LinearCombination([-1, 1, 1], [X(0) @ qml.Identity(1), X(0), X(1)]),
        qml.ops.LinearCombination([0, 1], [qml.X(0), X(1)]),
    ),
    (
        qml.ops.LinearCombination(
            [1, 0.5],
            [X(0) @ Y(1), Y(1) @ qml.Identity(2) @ X(0)],
        ),
        qml.ops.LinearCombination([1.5], [X(0) @ Y(1)]),
    ),
    (
        qml.ops.LinearCombination(
            [1, 1, 0.5],
            [
                qml.Hermitian(np.array([[1, 0], [0, -1]]), "a"),
                X("b") @ Y(1.3),
                Y(1.3) @ qml.Identity(-0.9) @ X("b"),
            ],
        ),
        qml.ops.LinearCombination(
            [1, 1.5],
            [qml.Hermitian(np.array([[1, 0], [0, -1]]), "a"), X("b") @ Y(1.3)],
        ),
    ),
    # Simplifies to zero LinearCombination
    (
        qml.ops.LinearCombination([1, -0.5, -0.5], [X(0) @ qml.Identity(1), X(0), X(0)]),
        qml.ops.LinearCombination([0.0], [qml.X(0)]),
    ),
    (
        qml.ops.LinearCombination(
            [1, -1],
            [X(4) @ qml.Identity(0) @ X(1), X(4) @ X(1)],
        ),
        qml.ops.LinearCombination([0.0], [qml.X(4) @ qml.X(1)]),
    ),
    (
        qml.ops.LinearCombination([0], [qml.Identity(0)]),
        qml.ops.LinearCombination([0], [qml.I(0)]),
    ),
]

equal_LinearCombinations = [
    (
        qml.ops.LinearCombination([1, 1], [X(0) @ qml.Identity(1), Z(0)]),
        qml.ops.LinearCombination([1, 1], [X(0), Z(0)]),
        True,
    ),
    (
        qml.ops.LinearCombination([1, 1], [X(0) @ qml.Identity(1), Y(2) @ Z(0)]),
        qml.ops.LinearCombination([1, 1], [X(0), Z(0) @ Y(2) @ qml.Identity(1)]),
        True,
    ),
    (
        qml.ops.LinearCombination([1, 1, 1], [X(0) @ qml.Identity(1), Z(0), qml.Identity(1)]),
        qml.ops.LinearCombination([1, 1], [X(0), Z(0)]),
        False,
    ),
    (
        qml.ops.LinearCombination([1], [Z(0) @ X(1)]),
        Z(0) @ X(1),
        True,
    ),
    (qml.ops.LinearCombination([1], [Z(0)]), Z(0), True),
    (
        qml.ops.LinearCombination(
            [1, 1, 1],
            [
                qml.Hermitian(np.array([[1, 0], [0, -1]]), "b") @ qml.Identity(7),
                Z(3),
                qml.Identity(1.2),
            ],
        ),
        qml.ops.LinearCombination(
            [1, 1, 1],
            [qml.Hermitian(np.array([[1, 0], [0, -1]]), "b"), Z(3), qml.Identity(1.2)],
        ),
        True,
    ),
    (
        qml.ops.LinearCombination([1, 1], [Z(3) @ qml.Identity(1.2), Z(3)]),
        qml.ops.LinearCombination([2], [Z(3)]),
        True,
    ),
]

add_LinearCombinations = [
    (
        qml.ops.LinearCombination([1, 1.2, 0.1], [X(0), Z(1), X(2)]),
        qml.ops.LinearCombination([0.5, 0.3, 1], [X(0), X(1), X(2)]),
        qml.ops.LinearCombination([1.5, 1.2, 1.1, 0.3], [X(0), Z(1), X(2), X(1)]),
    ),
    (
        qml.ops.LinearCombination([1.3, 0.2, 0.7], [X(0) @ X(1), qml.Hadamard(1), X(2)]),
        qml.ops.LinearCombination([0.5, 0.3, 1.6], [X(0), X(1) @ X(0), X(2)]),
        qml.ops.LinearCombination(
            [1.6, 0.2, 2.3, 0.5],
            [X(0) @ X(1), qml.Hadamard(1), X(2), X(0)],
        ),
    ),
    (
        qml.ops.LinearCombination([1, 1], [X(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
        qml.ops.LinearCombination(
            [0.5, 0.5], [X(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]
        ),
        qml.ops.LinearCombination(
            [1.5, 1.5], [X(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]
        ),
    ),
    (
        qml.ops.LinearCombination([1, 1.2, 0.1], [X(0), Z(1), X(2)]),
        X(0) @ qml.Identity(1),
        qml.ops.LinearCombination([2, 1.2, 0.1], [X(0), Z(1), X(2)]),
    ),
    (
        qml.ops.LinearCombination([1.3, 0.2, 0.7], [X(0) @ X(1), qml.Hadamard(1), X(2)]),
        qml.Hadamard(1),
        qml.ops.LinearCombination([1.3, 1.2, 0.7], [X(0) @ X(1), qml.Hadamard(1), X(2)]),
    ),
    (
        qml.ops.LinearCombination([1, 1.2, 0.1], [X("b"), Z(3.1), X(1.6)]),
        X("b") @ qml.Identity(5),
        qml.ops.LinearCombination([2, 1.2, 0.1], [X("b"), Z(3.1), X(1.6)]),
    ),
    # Case where arguments coeffs and ops to the LinearCombination are iterables other than lists
    (
        qml.ops.LinearCombination((1, 1.2, 0.1), (X(0), Z(1), X(2))),
        qml.ops.LinearCombination(np.array([0.5, 0.3, 1]), np.array([X(0), X(1), X(2)])),
        qml.ops.LinearCombination(
            (1.5, 1.2, 1.1, 0.3),
            np.array([X(0), Z(1), X(2), X(1)]),
        ),
    ),
    # Case where the 1st LinearCombination doesn't contain all wires
    (
        qml.ops.LinearCombination([1.23, -3.45], [X(0), Y(1)]),
        qml.ops.LinearCombination([6.78], [Z(2)]),
        qml.ops.LinearCombination([1.23, -3.45, 6.78], [X(0), Y(1), Z(2)]),
    ),
]

add_zero_LinearCombinations = [
    qml.ops.LinearCombination([1, 1.2, 0.1], [X(0), Z(1), X(2)]),
    qml.ops.LinearCombination([1, 1], [X(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
    qml.ops.LinearCombination([1.5, 1.2, 1.1, 0.3], [X(0), Z(1), X(2), X(1)]),
]

iadd_zero_LinearCombinations = [
    # identical LinearCombinations
    (
        qml.ops.LinearCombination([1, 1.2, 0.1], [X(0), Z(1), X(2)]),
        qml.ops.LinearCombination([1, 1.2, 0.1], [X(0), Z(1), X(2)]),
    ),
    (
        qml.ops.LinearCombination([1, 1], [X(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
        qml.ops.LinearCombination([1, 1], [X(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
    ),
    (
        qml.ops.LinearCombination([1.5, 1.2, 1.1, 0.3], [X(0), Z(1), X(2), X(1)]),
        qml.ops.LinearCombination([1.5, 1.2, 1.1, 0.3], [X(0), Z(1), X(2), X(1)]),
    ),
]

sub_LinearCombinations = [
    (
        qml.ops.LinearCombination([1, 1.2, 0.1], [X(0), Z(1), X(2)]),
        qml.ops.LinearCombination([0.5, 0.3, 1.6], [X(0), X(1), X(2)]),
        qml.ops.LinearCombination([0.5, 1.2, -1.5, -0.3], [X(0), Z(1), X(2), X(1)]),
    ),
    (
        qml.ops.LinearCombination([1, 1.2, 0.1], [X(0), Z(1), X(2)]),
        X(0) @ qml.Identity(1),
        qml.ops.LinearCombination(
            [1, 1.2, 0.1, 1.0], [qml.X(0), Z(1), qml.X(2), -1 * (qml.X(0) @ qml.I(1))]
        ),
    ),
    (
        qml.ops.LinearCombination([1, 1.2, 0.1], [X("b"), Z(3.1), X(1.6)]),
        X("b") @ qml.Identity(1),
        qml.ops.LinearCombination(
            [1, 1.2, 0.1, 1], [X("b"), Z(3.1), X(1.6), -1 * (qml.X("b") @ qml.I(1))]
        ),
    ),
    # The result is the zero LinearCombination
    (
        qml.ops.LinearCombination([1, 1.2, 0.1], [X(0), Z(1), X(2)]),
        qml.ops.LinearCombination([1, 1.2, 0.1], [X(0), Z(1), X(2)]),
        qml.ops.LinearCombination(
            [1, 1.2, 0.1, 1],
            [X(0), Z(1), X(2), -1 * qml.ops.LinearCombination([1, 1.2, 0.1], [X(0), Z(1), X(2)])],
        ),
    ),
    (
        qml.ops.LinearCombination([1.0, 2.0], [X(4), Z(2)]),
        qml.ops.LinearCombination([1.0, 2.0], [X(4), Z(2)]),
        qml.ops.LinearCombination(
            [1.0, 2.0, 1.0],
            [qml.X(4), qml.Z(2), -1 * qml.ops.LinearCombination([1.0, 2.0], [X(4), Z(2)])],
        ),
    ),
    # Case where arguments coeffs and ops to the LinearCombination are iterables other than lists
    (
        qml.ops.LinearCombination((1, 1.2, 0.1), (X(0), Z(1), X(2))),
        qml.ops.LinearCombination(np.array([0.5, 0.3, 1.6]), np.array([X(0), X(1), X(2)])),
        qml.ops.LinearCombination(
            (0.5, 1.2, -1.5, -0.3),
            np.array([X(0), Z(1), X(2), X(1)]),
        ),
    ),
    # Case where the 1st LinearCombination doesn't contain all wires
    (
        qml.ops.LinearCombination([1.23, -3.45], [X(0), Y(1)]),
        qml.ops.LinearCombination([6.78], [Z(2)]),
        qml.ops.LinearCombination([1.23, -3.45, -6.78], [X(0), Y(1), Z(2)]),
    ),
]

mul_LinearCombinations = [
    (
        0.5,
        qml.ops.LinearCombination(
            [1, 2], [X(0), Z(1)]
        ),  # Case where the types of the coefficient and the scalar differ
        qml.ops.LinearCombination([0.5, 1.0], [X(0), Z(1)]),
    ),
    (
        3.0,
        qml.ops.LinearCombination([1.5, 0.5], [X(0), Z(1)]),
        qml.ops.LinearCombination([4.5, 1.5], [X(0), Z(1)]),
    ),
    (
        -1.3,
        qml.ops.LinearCombination([1, -0.3], [X(0), Z(1) @ Z(2)]),
        qml.ops.LinearCombination([-1.3, 0.39], [X(0), Z(1) @ Z(2)]),
    ),
    (
        -1.3,
        qml.ops.LinearCombination(
            [1, -0.3],
            [qml.Hermitian(np.array([[1, 0], [0, -1]]), "b"), Z(23) @ Z(0)],
        ),
        qml.ops.LinearCombination(
            [-1.3, 0.39],
            [qml.Hermitian(np.array([[1, 0], [0, -1]]), "b"), Z(23) @ Z(0)],
        ),
    ),
    # The result is the zero LinearCombination
    (
        0.0,
        qml.ops.LinearCombination([1], [X(0)]),
        qml.ops.LinearCombination([0], [X(0)]),
    ),
    (
        0.0,
        qml.ops.LinearCombination([1.0, 1.2, 0.1], [X(0), Z(1), X(2)]),
        qml.ops.LinearCombination([0.0, 0.0, 0.0], [X(0), Z(1), X(2)]),
    ),
    # Case where arguments coeffs and ops to the LinearCombination are iterables other than lists
    (
        3.0,
        qml.ops.LinearCombination((1.5, 0.5), (X(0), Z(1))),
        qml.ops.LinearCombination(np.array([4.5, 1.5]), np.array([X(0), Z(1)])),
    ),
]

matmul_LinearCombinations = [
    (
        qml.ops.LinearCombination([1, 1], [X(0), Z(1)]),
        qml.ops.LinearCombination([0.5, 0.5], [Z(2), Z(3)]),
        qml.ops.LinearCombination(
            [0.5, 0.5, 0.5, 0.5],
            [
                X(0) @ Z(2),
                X(0) @ Z(3),
                Z(1) @ Z(2),
                Z(1) @ Z(3),
            ],
        ),
    ),
    (
        qml.ops.LinearCombination([0.5, 0.25], [X(0) @ X(1), Z(0)]),
        qml.ops.LinearCombination([1, 1], [X(3) @ Z(2), Z(2)]),
        qml.ops.LinearCombination(
            [0.5, 0.5, 0.25, 0.25],
            [
                X(0) @ X(1) @ X(3) @ Z(2),
                X(0) @ X(1) @ Z(2),
                Z(0) @ X(3) @ Z(2),
                Z(0) @ Z(2),
            ],
        ),
    ),
    (
        qml.ops.LinearCombination([1, 1], [X(0), Z(1)]),
        X(2),
        qml.ops.LinearCombination([1, 1], [X(0) @ X(2), Z(1) @ X(2)]),
    ),
]

rmatmul_LinearCombinations = [
    (
        qml.ops.LinearCombination([0.5, 0.5], [Z(2), Z(3)]),
        qml.ops.LinearCombination([1, 1], [X(0), Z(1)]),
        qml.ops.LinearCombination(
            [0.5, 0.5, 0.5, 0.5],
            [
                X(0) @ Z(2),
                X(0) @ Z(3),
                Z(1) @ Z(2),
                Z(1) @ Z(3),
            ],
        ),
    ),
    (
        qml.ops.LinearCombination([1, 1], [X(3) @ Z(2), Z(2)]),
        qml.ops.LinearCombination([0.5, 0.25], [X(0) @ X(1), Z(0)]),
        qml.ops.LinearCombination(
            [0.5, 0.5, 0.25, 0.25],
            [
                X(0) @ X(1) @ X(3) @ Z(2),
                X(0) @ X(1) @ Z(2),
                Z(0) @ X(3) @ Z(2),
                Z(0) @ Z(2),
            ],
        ),
    ),
    (
        qml.ops.LinearCombination([1, 1], [X(0), Z(1)]),
        X(2),
        qml.ops.LinearCombination([1, 1], [X(2) @ X(0), X(2) @ Z(1)]),
    ),
]

big_LinearCombination_coeffs = np.array(
    [
        -0.04207898,
        0.17771287,
        0.17771287,
        -0.24274281,
        -0.24274281,
        0.17059738,
        0.04475014,
        -0.04475014,
        -0.04475014,
        0.04475014,
        0.12293305,
        0.16768319,
        0.16768319,
        0.12293305,
        0.17627641,
    ]
)

big_LinearCombination_ops = [
    qml.Identity(wires=[0]),
    Z(wires=[0]),
    Z(wires=[1]),
    Z(wires=[2]),
    Z(wires=[3]),
    Z(wires=[0]) @ Z(wires=[1]),
    Y(wires=[0]) @ X(wires=[1]) @ X(wires=[2]) @ Y(wires=[3]),
    Y(wires=[0]) @ Y(wires=[1]) @ X(wires=[2]) @ X(wires=[3]),
    X(wires=[0]) @ X(wires=[1]) @ Y(wires=[2]) @ Y(wires=[3]),
    X(wires=[0]) @ Y(wires=[1]) @ Y(wires=[2]) @ X(wires=[3]),
    Z(wires=[0]) @ Z(wires=[2]),
    Z(wires=[0]) @ Z(wires=[3]),
    Z(wires=[1]) @ Z(wires=[2]),
    Z(wires=[1]) @ Z(wires=[3]),
    Z(wires=[2]) @ Z(wires=[3]),
]

big_LinearCombination = qml.ops.LinearCombination(
    big_LinearCombination_coeffs, big_LinearCombination_ops
)

big_LinearCombination_grad = (
    np.array(
        [
            [
                [6.52084595e-18, -2.11464420e-02, -1.16576858e-02],
                [-8.22589330e-18, -5.20597922e-02, -1.85365365e-02],
                [-2.73850768e-17, 1.14202988e-01, -5.45041403e-03],
                [-1.27514307e-17, -1.10465531e-01, 5.19489457e-02],
            ],
            [
                [-2.45428288e-02, 8.38921555e-02, -2.00641818e-17],
                [-2.21085973e-02, 7.39332741e-04, -1.25580654e-17],
                [9.62058625e-03, -1.51398765e-01, 2.02129847e-03],
                [1.10020832e-03, -3.49066271e-01, 2.13669117e-03],
            ],
        ]
    ),
)


def circuit1(param):
    """First Pauli subcircuit"""
    qml.RX(param, wires=0)
    qml.RY(param, wires=0)
    return qml.expval(X(0))


def circuit2(param):
    """Second Pauli subcircuit"""
    qml.RX(param, wires=0)
    qml.RY(param, wires=0)
    return qml.expval(Z(0))


dev = qml.device("default.qubit", wires=2)


class TestLinearCombination:
    """Test the LinearCombination class"""

    def test_error_if_observables_operator(self):
        """Test thatt an error is raised if an operator is provided to observables."""

        with pytest.raises(ValueError, match=r"observables must be an Iterable of Operator's"):
            qml.ops.LinearCombination([1, 1], qml.X(0) @ qml.Y(1))

    PAULI_REPS = (
        ([], [], PauliSentence({})),
        (
            list(range(3)),
            [X(i) for i in range(3)],
            PauliSentence({PauliWord({i: "X"}): 1.0 * i for i in range(3)}),
        ),
        (
            list(range(3)),
            [qml.s_prod(i, X(i)) for i in range(3)],
            PauliSentence({PauliWord({i: "X"}): 1.0 * i * i for i in range(3)}),
        ),
    )

    @pytest.mark.parametrize("simplify", [None, True])
    @pytest.mark.parametrize("coeffs, ops, true_pauli", PAULI_REPS)
    def test_pauli_rep(self, coeffs, ops, true_pauli, simplify):
        """Test the pauli rep is correctly constructed"""
        if simplify:
            H = qml.ops.LinearCombination(coeffs, ops).simplify()
        else:
            H = qml.ops.LinearCombination(coeffs, ops)
        pr = H.pauli_rep
        if simplify:
            pr.simplify()
            true_pauli.simplify()
        assert pr is not None
        assert pr == true_pauli

    def test_is_hermitian_trivial(self):
        """Test that an empty LinearCombination is trivially hermitian"""
        op = qml.ops.LinearCombination([], [])
        assert op.is_hermitian

    IS_HERMITIAN_TEST = (
        (qml.ops.LinearCombination([0.5, 0.5], [X(0), X(1) @ X(2)]), True),
        (qml.ops.LinearCombination([0.5, 0.5j], [X(0), X(1) @ X(2)]), False),
        (qml.ops.LinearCombination([0.5, 0.5], [X(0), qml.Hadamard(0)]), True),
    )

    @pytest.mark.parametrize("op, res", IS_HERMITIAN_TEST)
    def test_is_hermitian(self, op, res):
        assert op.is_hermitian is res

    @pytest.mark.parametrize("coeffs, ops", valid_LinearCombinations)
    def test_LinearCombination_valid_init(self, coeffs, ops):
        """Tests that the LinearCombination object is created with
        the correct attributes"""
        H = qml.ops.LinearCombination(coeffs, ops)
        assert np.allclose(H.terms()[0], coeffs)
        assert H.terms()[1] == list(ops)

    @pytest.mark.parametrize("coeffs, ops", invalid_LinearCombinations)
    def test_LinearCombination_invalid_init_exception(self, coeffs, ops):
        """Tests that an exception is raised when giving an invalid
        combination of coefficients and ops"""
        with pytest.raises(ValueError, match="number of coefficients and operators does not match"):
            qml.ops.LinearCombination(coeffs, ops)

    def test_integer_coefficients(self):
        """Test that handling integers is not a problem"""
        H1, H2, true_res = (
            qml.ops.LinearCombination([1, 2], [X(4), Z(2)]),  # not failing with float coeffs
            qml.ops.LinearCombination([1, 2], [X(4), Z(2)]),
            qml.ops.LinearCombination([0, 0], [qml.X(4), qml.Z(2)]),
        )
        res = H1 - H2
        qml.assert_equal(qml.simplify(res), true_res)

    # pylint: disable=protected-access
    @pytest.mark.parametrize("coeffs, ops", valid_LinearCombinations)
    @pytest.mark.parametrize("grouping_type", (None, "qwc"))
    def test_flatten_unflatten(self, coeffs, ops, grouping_type):
        """Test the flatten and unflatten methods for LinearCombinations"""

        if any(not qml.pauli.is_pauli_word(t) for t in ops) and grouping_type:
            pytest.skip("grouping type must be none if a term is not a pauli word.")

        H = LinearCombination(coeffs, ops, grouping_type=grouping_type)
        data, metadata = H._flatten()
        assert metadata[0] == H.grouping_indices
        assert hash(metadata)
        assert len(data) == 2
        assert qml.math.allequal(
            data[0], H._coeffs
        )  # Previously checking "is" instead of "==", problem?
        assert data[1] == H._ops

        new_H = LinearCombination._unflatten(*H._flatten())
        qml.assert_equal(H, new_H)
        assert new_H.grouping_indices == H.grouping_indices

    @pytest.mark.parametrize("coeffs, ops", valid_LinearCombinations)
    def test_LinearCombination_wires(self, coeffs, ops):
        """Tests that the LinearCombination object has correct wires."""
        H = qml.ops.LinearCombination(coeffs, ops)
        assert set(H.wires) == {w for op in H.ops for w in op.wires}

    def test_label(self):
        """Tests the label method of LinearCombination when <=3 coefficients."""
        H = qml.ops.LinearCombination((-0.8,), (Z(0),))
        assert H.label() == "𝓗"
        assert H.label(decimals=2) == "𝓗\n(-0.80)"

    def test_label_many_coefficients(self):
        """Tests the label method of LinearCombination when >3 coefficients."""
        H = LinearCombination([0.1] * 5, [X(i) for i in range(5)])
        assert H.label() == "𝓗"
        assert H.label(decimals=2) == "𝓗"

    LINEARCOMBINATION_STR = (
        (qml.ops.LinearCombination([0.5, 0.5], [X(0), X(1)]), "0.5 * X(0) + 0.5 * X(1)"),
        (
            qml.ops.LinearCombination([0.5, 0.5], [qml.prod(X(0), X(1)), qml.prod(X(1), X(2))]),
            "0.5 * (X(0) @ X(1)) + 0.5 * (X(1) @ X(2))",
        ),
    )

    @pytest.mark.parametrize("op, string", LINEARCOMBINATION_STR)
    def test_LinearCombination_str(self, op, string):
        """Tests that the __str__ function for printing is correct"""
        assert str(op) == string

    LINEARCOMBINATION_REPR = (
        (qml.ops.LinearCombination([0.5, 0.5], [X(0), X(1)]), "0.5 * X(0) + 0.5 * X(1)"),
        (
            qml.ops.LinearCombination([0.5, 0.5], [qml.prod(X(0), X(1)), qml.prod(X(1), X(2))]),
            "0.5 * (X(0) @ X(1)) + 0.5 * (X(1) @ X(2))",
        ),
        (
            qml.ops.LinearCombination(range(15), [qml.prod(X(i), X(i + 1)) for i in range(15)]),
            "(\n    0 * (X(0) @ X(1))\n  + 1 * (X(1) @ X(2))\n  + 2 * (X(2) @ X(3))\n  + 3 * (X(3) @ X(4))\n  + 4 * (X(4) @ X(5))\n  + 5 * (X(5) @ X(6))\n  + 6 * (X(6) @ X(7))\n  + 7 * (X(7) @ X(8))\n  + 8 * (X(8) @ X(9))\n  + 9 * (X(9) @ X(10))\n  + 10 * (X(10) @ X(11))\n  + 11 * (X(11) @ X(12))\n  + 12 * (X(12) @ X(13))\n  + 13 * (X(13) @ X(14))\n  + 14 * (X(14) @ X(15))\n)",
        ),
    )

    @pytest.mark.parametrize("op, string", LINEARCOMBINATION_REPR)
    def test_LinearCombination_repr(self, op, string):
        """Tests that the __repr__ function for printing is correct"""
        assert repr(op) == string

    def test_LinearCombination_name(self):
        """Tests the name property of the LinearCombination class"""
        H = qml.ops.LinearCombination([], [])
        assert H.name == "LinearCombination"

    @pytest.mark.parametrize(("old_H", "new_H"), simplify_LinearCombinations)
    def test_simplify(self, old_H, new_H):
        """Tests the simplify method"""
        old_H = old_H.simplify()
        qml.assert_equal(old_H, new_H)

    def test_simplify_while_queueing(self):
        """Tests that simplifying a LinearCombination in a tape context
        queues the simplified LinearCombination."""

        with qml.queuing.AnnotatedQueue() as q:
            a = X(wires=0)
            b = Y(wires=1)
            c = qml.Identity(wires=2)
            d = b @ c
            H = qml.ops.LinearCombination([1.0, 2.0], [a, d])
            H = H.simplify()

        # check that H is simplified
        assert H.ops == [a, b]
        # check that the simplified LinearCombination is in the queue
        assert q.get_info(H) is not None

    COMPARE_WITH_OPS = (
        (qml.ops.LinearCombination([0.5], [X(0) @ X(1)]), qml.s_prod(0.5, X(0) @ X(1))),
        (qml.ops.LinearCombination([0.5], [X(0) + X(1)]), qml.s_prod(0.5, qml.sum(X(0), X(1)))),
        (qml.ops.LinearCombination([1.0], [X(0)]), X(0)),
        # (qml.ops.LinearCombination([1.0], [qml.Hadamard(0)]), qml.Hadamard(0)), # TODO fix qml.equal check for Observables having to be the same type
        (qml.ops.LinearCombination([1.0], [X(0) @ X(1)]), X(0) @ X(1)),
    )

    @pytest.mark.parametrize(("H1", "H2", "H"), add_LinearCombinations)
    def test_LinearCombination_add(self, H1, H2, H):
        """Tests that LinearCombinations are added correctly"""
        res = H1 + H2
        assert isinstance(res, LinearCombination)
        qml.assert_equal(H, qml.simplify(res))

    @pytest.mark.parametrize("H", add_zero_LinearCombinations)
    def test_LinearCombination_add_zero(self, H):
        """Tests that LinearCombinations can be added to zero"""
        assert H == (H + 0)
        assert H == (0 + H)
        assert H == (H + 0.0)
        assert H == (0.0 + H)
        assert H == (H + 0e1)
        assert H == (0e1 + H)

    @pytest.mark.parametrize(("coeff", "H", "res"), mul_LinearCombinations)
    def test_LinearCombination_mul(self, coeff, H, res):
        """Tests that scalars and LinearCombinations are multiplied correctly"""
        assert res == (coeff * H)
        assert res == (H * coeff)

    def test_LinearCombination_mul_coeff_cast(self):
        """Test that the coefficients are correct when the type of the existing
        and the new coefficients differ."""
        h = qml.ops.LinearCombination([0.5, 0.5], [X(0) @ X(0), Y(0) @ Y(1)])
        assert np.all(h.coeffs == np.array([0.5, 0.5]))

    @pytest.mark.parametrize(("H1", "H2", "H"), sub_LinearCombinations)
    def test_LinearCombination_sub(self, H1, H2, H):
        """Tests that LinearCombinations are subtracted correctly"""
        qml.assert_equal(H, H1 - H2)

    def test_LinearCombination_tensor_matmul(self):
        """Tests that a LinearCombination can be multiplied by a tensor."""
        H = qml.ops.LinearCombination([1.0, 1.0], [X(0), Y(0)])
        t = Z(1) @ Z(2)
        out = H @ t

        qml.assert_equal(qml.prod(H, t), out)

    def test_LinearCombination_matmul_overlapping_wires_raises_error(self):
        """Test that an error is raised when attempting to multiply two
        LinearCombination operators with overlapping wires"""
        op1 = qml.ops.LinearCombination([1.0], [X(0)])
        op2 = qml.ops.LinearCombination([1.0], [Y(0)])
        with pytest.raises(ValueError, match="LinearCombinations can only be multiplied together"):
            _ = op1 @ op2

    def test_matmul_with_non_pauli_op(self):
        """Test multiplication with another operator that does not have a pauli rep"""
        H = qml.ops.LinearCombination([0.5], [X(0)])
        assert H.pauli_rep == PauliSentence({PauliWord({0: "X"}): 0.5})
        op = qml.Hadamard(0)

        res = H @ op
        assert res.pauli_rep is None
        assert res == (qml.ops.LinearCombination([0.5], [X(0) @ qml.Hadamard(0)]))

    @pytest.mark.parametrize(("H1", "H2", "H"), matmul_LinearCombinations)
    def test_LinearCombination_matmul(self, H1, H2, H):
        """Tests that LinearCombinations are tensored correctly"""
        assert H == (H1 @ H2)

    @pytest.mark.parametrize(("H1", "H2", "H"), rmatmul_LinearCombinations)
    def test_LinearCombination_rmatmul(self, H1, H2, H):
        """Tests that LinearCombinations are tensored correctly when using __rmatmul__"""
        assert H == (H1 @ H2)

    def test_arithmetic_errors(self):
        """Tests that the arithmetic operations thrown the correct errors"""
        H = qml.ops.LinearCombination([1], [Z(0)])
        A = [[1, 0], [0, -1]]
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = H @ A
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = A @ H
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = H + A
        with pytest.raises(TypeError, match="can't multiply sequence by non-int"):
            _ = H * A
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = H - A
        with pytest.raises(TypeError, match="unsupported operand type"):
            H += A
        with pytest.raises(TypeError, match="unsupported operand type"):
            H *= A
        with pytest.raises(TypeError, match="unsupported operand type"):
            H -= A

    def test_LinearCombination_queue_outside(self):
        """Tests that LinearCombination are queued correctly when components are defined outside the recording context."""

        H = X(1) + 3 * Z(0) @ Z(2) + Z(1)

        with qml.queuing.AnnotatedQueue() as q:
            qml.Hadamard(wires=1)
            X(wires=0)
            qml.expval(H)

        assert len(q.queue) == 3
        assert isinstance(q.queue[0], qml.Hadamard)
        assert isinstance(q.queue[1], qml.PauliX)
        assert isinstance(q.queue[2], qml.measurements.MeasurementProcess)
        queue_op = q.queue[2].obs
        assert H.pauli_rep == queue_op.pauli_rep

    def test_LinearCombination_queue_inside(self):
        """Tests that LinearCombination are queued correctly when components are instantiated inside the recording context."""
        with qml.queuing.AnnotatedQueue() as q:
            m = qml.expval(qml.ops.LinearCombination([1, 3, 1], [X(1), Z(0) @ Z(2), Z(1)]))

        assert len(q.queue) == 1
        assert q.queue[0] is m

    def test_terms(self):
        """Tests that the terms representation is returned correctly."""
        coeffs = pnp.array([1.0, 2.0], requires_grad=True)
        ops = [X(0), Z(1)]
        h = qml.ops.LinearCombination(coeffs, ops)
        c, o = h.terms()
        assert isinstance(c, Iterable)
        assert isinstance(o, list)
        assert all(isinstance(item, np.ndarray) for item in c)
        assert all(item.requires_grad for item in c)
        assert all(isinstance(item, qml.operation.Operator) for item in o)

    def test_LinearCombination_no_empty_wire_list_error(self):
        """Test that empty LinearCombination does not raise an empty wire error."""
        lincomb = qml.ops.LinearCombination([], [])
        assert isinstance(lincomb, qml.ops.LinearCombination)

    def test_map_wires_no_grouping(self):
        """Test the map_wires method."""
        coeffs = pnp.array([1.0, 2.0, -3.0], requires_grad=True)
        ops = [X(0), Z(1), Y(2)]
        h = qml.ops.LinearCombination(coeffs, ops)
        wire_map = {0: 10, 1: 11, 2: 12}
        mapped_h = h.map_wires(wire_map=wire_map)
        final_obs = [X(10), Z(11), Y(12)]
        assert h is not mapped_h
        assert h.wires == Wires([0, 1, 2])
        assert mapped_h.wires == Wires([10, 11, 12])
        for obs1, obs2 in zip(mapped_h.ops, final_obs):
            qml.assert_equal(obs1, obs2)
        for coeff1, coeff2 in zip(mapped_h.coeffs, h.coeffs):
            assert coeff1 == coeff2

    def test_map_wires_grouping(self):
        """Test the map_wires method."""
        coeffs = pnp.array([1.0, 2.0, -3.0], requires_grad=True)
        ops = [X(0), Z(1), Y(2)]
        h = qml.ops.LinearCombination(coeffs, ops, grouping_type="qwc")
        group_indices_before = copy(h.grouping_indices)
        wire_map = {0: 10, 1: 11, 2: 12}
        mapped_h = h.map_wires(wire_map=wire_map)
        final_obs = [X(10), Z(11), Y(12)]
        assert h is not mapped_h
        assert h.wires == Wires([0, 1, 2])
        assert mapped_h.wires == Wires([10, 11, 12])
        for obs1, obs2 in zip(mapped_h.ops, final_obs):
            qml.assert_equal(obs1, obs2)
        for coeff1, coeff2 in zip(mapped_h.coeffs, h.coeffs):
            assert coeff1 == coeff2
        assert group_indices_before == mapped_h.grouping_indices

    def test_hermitian_tensor_prod(self):
        """Test that the tensor product of a LinearCombination with Hermitian observable works."""
        tensor = X(0) @ X(1)
        herm = qml.Hermitian([[1, 0], [0, 1]], wires=4)

        ham = qml.ops.LinearCombination([1.0, 1.0], [tensor, X(2)]) @ qml.ops.LinearCombination(
            [1.0], [herm]
        )
        assert isinstance(ham, qml.ops.LinearCombination)

    def test_diagonalizing_gates(self):
        """Test that LinearCombination has valid diagonalizing gates"""
        LC = qml.ops.LinearCombination([1.1, 2.2], [qml.X(0), qml.Z(0)])
        SUM = qml.sum(qml.s_prod(1.1, qml.X(0)), qml.s_prod(2.2, qml.Z(0)))

        assert LC.diagonalizing_gates() == SUM.diagonalizing_gates()

    def test_eigvals(self):
        """Test that LinearCombination has valid eigvals"""
        LC = qml.ops.LinearCombination([1.1, 2.2, 3.3], [qml.X(0), qml.Z(0), qml.Y(1)])

        assert len(LC.overlapping_ops[0]) > 1  # will use one branch
        assert len(LC.overlapping_ops[1]) == 1  # will use the other branch

        SUM = qml.sum(
            qml.s_prod(1.1, qml.X(0)), qml.s_prod(2.2, qml.Z(0)), qml.s_prod(3.3, qml.Y(1))
        )

        assert np.all(LC.eigvals() == SUM.eigvals())


class TestLinearCombinationCoefficients:
    """Test the creation of a LinearCombination"""

    @pytest.mark.parametrize("coeffs", [el[0] for el in COEFFS_PARAM_INTERFACE])
    def test_creation_different_coeff_types(self, coeffs):
        """Check that LinearCombination's coefficients and data attributes are set correctly."""
        H = qml.ops.LinearCombination(coeffs, [X(0), Z(0)])
        assert np.allclose(coeffs, H.coeffs)
        assert np.allclose([coeffs[i] for i in range(qml.math.shape(coeffs)[0])], H.data)

    @pytest.mark.parametrize("coeffs", [el[0] for el in COEFFS_PARAM_INTERFACE])
    def test_simplify(self, coeffs):
        """Test that simplify works with different coefficient types."""
        H1 = qml.ops.LinearCombination(coeffs, [X(0), Z(1)])
        H2 = qml.ops.LinearCombination(coeffs, [X(0), qml.Identity(0) @ Z(1)])
        H2 = H2.simplify()
        assert H1 == (H2)
        assert qml.math.allclose(H1.data, H2.data)

    # TODO: increase coverage
    def test_operands(self):
        op = qml.ops.LinearCombination([1.1, 2.2], [X(0), Z(0)])
        assert op.operands == (qml.s_prod(1.1, X(0)), qml.s_prod(2.2, Z(0)))


@pytest.mark.tf
class TestLinearCombinationArithmeticTF:
    """Tests creation of LinearCombinations using arithmetic
    operations with TensorFlow tensor coefficients."""

    def test_LinearCombination_equal(self):
        """Tests equality"""
        coeffs = tf.Variable([0.5, -1.6])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = tf.Variable([-1.6, 0.5])
        obs2 = [Y(1), X(0)]
        H2 = qml.ops.LinearCombination(coeffs2, obs2)

        assert H1 == (H2)

    def test_LinearCombination_add(self):
        """Tests that LinearCombinations are added correctly"""
        coeffs = tf.Variable([0.5, -1.5])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = tf.Variable([0.5, -0.5])
        H2 = qml.ops.LinearCombination(coeffs2, obs)

        coeffs_expected = tf.Variable([1.0, -2.0])
        H = qml.ops.LinearCombination(coeffs_expected, obs)

        assert H == (H1 + H2)

    def test_LinearCombination_sub(self):
        """Tests that LinearCombinations are subtracted correctly"""
        coeffs = tf.constant([1.0, -2.0])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = tf.constant([0.5, -0.5])
        H2 = qml.ops.LinearCombination(coeffs2, obs)

        coeffs_expected = tf.constant([0.5, -1.5])
        H = qml.ops.LinearCombination(coeffs_expected, obs)

        assert H == (H1 - H2)

    def test_LinearCombination_matmul(self):
        """Tests that LinearCombinations are tensored correctly"""

        coeffs = tf.Variable([1.0, 2.0])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = tf.Variable([-1.0, -2.0])
        obs2 = [X(2), Y(3)]
        H2 = qml.ops.LinearCombination(coeffs2, obs2)

        coeffs_expected = tf.Variable([-4.0, -2.0, -2.0, -1.0])
        obs_expected = [
            qml.prod(Y(1), Y(3)),
            qml.prod(X(0), Y(3)),
            qml.prod(X(2), Y(1)),
            qml.prod(X(0), X(2)),
        ]
        H = qml.ops.LinearCombination(coeffs_expected, obs_expected)

        assert H == (H1 @ H2)


@pytest.mark.torch
class TestLinearCombinationArithmeticTorch:
    """Tests creation of LinearCombinations using arithmetic
    operations with torch tensor coefficients."""

    def test_LinearCombination_equal(self):
        """Tests equality"""
        coeffs = torch.tensor([0.5, -1.6])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = torch.tensor([-1.6, 0.5])
        obs2 = [Y(1), X(0)]
        H2 = qml.ops.LinearCombination(coeffs2, obs2)

        assert H1 == (H2)

    def test_LinearCombination_add(self):
        """Tests that LinearCombinations are added correctly"""
        coeffs = torch.tensor([0.5, -1.6])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = torch.tensor([0.5, -0.4])
        H2 = qml.ops.LinearCombination(coeffs2, obs)

        coeffs_expected = torch.tensor([1.0, -2.0])
        H = qml.ops.LinearCombination(coeffs_expected, obs)

        assert H == (H1 + H2)

    def test_LinearCombination_sub(self):
        """Tests that LinearCombinations are subtracted correctly"""
        coeffs = torch.tensor([1.0, -2.0])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = torch.tensor([0.5, -0.4])
        H2 = qml.ops.LinearCombination(coeffs2, obs)

        coeffs_expected = torch.tensor([0.5, -1.6])
        H = qml.ops.LinearCombination(coeffs_expected, obs)

        assert H == (H1 - H2)

        H1 -= H2
        assert H == (H1)

    def test_LinearCombination_matmul(self):
        """Tests that LinearCombinations are tensored correctly"""

        coeffs = torch.tensor([1.0, 2.0])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = torch.tensor([-1.0, -2.0])
        obs2 = [X(2), Y(3)]
        H2 = qml.ops.LinearCombination(coeffs2, obs2)

        coeffs_expected = torch.tensor([-4.0, -2.0, -2.0, -1.0])
        obs_expected = [
            qml.prod(Y(1), Y(3)),
            qml.prod(X(0), Y(3)),
            qml.prod(X(2), Y(1)),
            qml.prod(X(0), X(2)),
        ]
        H = qml.ops.LinearCombination(coeffs_expected, obs_expected)

        assert H == (H1 @ H2)


@pytest.mark.autograd
class TestLinearCombinationArithmeticAutograd:
    """Tests creation of LinearCombinations using arithmetic
    operations with autograd tensor coefficients."""

    def test_LinearCombination_equal(self):
        """Tests equality"""
        coeffs = pnp.array([0.5, -1.6])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = pnp.array([-1.6, 0.5])
        obs2 = [Y(1), X(0)]
        H2 = qml.ops.LinearCombination(coeffs2, obs2)

        assert H1 == (H2)

    def test_LinearCombination_add(self):
        """Tests that LinearCombinations are added correctly"""
        coeffs = pnp.array([0.5, -1.5])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = pnp.array([0.5, -0.5])
        H2 = qml.ops.LinearCombination(coeffs2, obs)

        coeffs_expected = pnp.array([1.0, -2.0])
        H = qml.ops.LinearCombination(coeffs_expected, obs)

        assert H == (H1 + H2)

    def test_LinearCombination_sub(self):
        """Tests that LinearCombinations are subtracted correctly"""
        coeffs = pnp.array([1.0, -2.0])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = pnp.array([0.5, -0.5])
        H2 = qml.ops.LinearCombination(coeffs2, obs)

        coeffs_expected = pnp.array([0.5, -1.5])
        H = qml.ops.LinearCombination(coeffs_expected, obs)

        assert H == (H1 - H2)

    def test_LinearCombination_matmul(self):
        """Tests that LinearCombinations are tensored correctly"""
        coeffs = pnp.array([1.0, 2.0])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = pnp.array([-1.0, -2.0])
        obs2 = [X(2), Y(3)]
        H2 = qml.ops.LinearCombination(coeffs2, obs2)

        coeffs_expected = pnp.array([-4.0, -2.0, -2.0, -1.0])
        obs_expected = [
            qml.prod(Y(1), Y(3)),
            qml.prod(X(0), Y(3)),
            qml.prod(X(2), Y(1)),
            qml.prod(X(0), X(2)),
        ]
        H = qml.ops.LinearCombination(coeffs_expected, obs_expected)

        assert H == (H1 @ H2)


class TestLinearCombinationSparseMatrix:
    """Tests for sparse matrix representation."""

    @pytest.mark.parametrize(
        ("coeffs", "obs", "wires", "ref_matrix"),
        [
            (
                [1, -0.45],
                [qml.prod(Z(0), Z(1)), qml.prod(Y(0), Z(1))],
                None,
                np.array(
                    [
                        [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.45j, 0.0 + 0.0j],
                        [0.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j, 0.0 - 0.45j],
                        [0.0 - 0.45j, 0.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.45j, 0.0 + 0.0j, 1.0 + 0.0j],
                    ]
                ),
            ),
            (
                [0.1],
                [qml.prod(Z("b"), X("a"))],
                ["a", "c", "b"],
                np.array(
                    [
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.1 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            -0.1 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.1 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            -0.1 + 0.0j,
                        ],
                        [
                            0.1 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            -0.1 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.1 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            -0.1 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                    ]
                ),
            ),
            (
                [0.21, -0.78, 0.52],
                [
                    qml.prod(Z(0), Z(1)),
                    qml.prod(X(0), Z(1)),
                    qml.prod(Y(0), Z(1)),
                ],
                None,
                np.array(
                    [
                        [0.21 + 0.0j, 0.0 + 0.0j, -0.78 - 0.52j, 0.0 + 0.0j],
                        [0.0 + 0.0j, -0.21 + 0.0j, 0.0 + 0.0j, 0.78 + 0.52j],
                        [-0.78 + 0.52j, 0.0 + 0.0j, -0.21 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.78 - 0.52j, 0.0 + 0.0j, 0.21 + 0.0j],
                    ]
                ),
            ),
        ],
    )
    def test_sparse_matrix(self, coeffs, obs, wires, ref_matrix):
        """Tests that sparse_LinearCombination returns a correct sparse matrix"""
        H = qml.ops.LinearCombination(coeffs, obs)

        sparse_matrix = H.sparse_matrix(wire_order=wires)

        assert np.allclose(sparse_matrix.toarray(), ref_matrix)

    def test_sparse_format(self):
        """Tests that sparse_LinearCombination returns a scipy.sparse.csr_matrix object"""

        coeffs = [-0.25, 0.75]
        obs = [
            X(wires=[0]) @ Z(wires=[1]),
            Y(wires=[0]) @ Z(wires=[1]),
        ]
        H = qml.ops.LinearCombination(coeffs, obs)

        sparse_matrix = H.sparse_matrix()

        assert isinstance(sparse_matrix, scipy.sparse.csr_matrix)


@pytest.mark.jax
class TestLinearCombinationArithmeticJax:
    """Tests creation of LinearCombinations using arithmetic
    operations with jax tensor coefficients."""

    def test_LinearCombination_equal(self):
        """Tests equality"""
        coeffs = jnp.array([0.5, -1.6])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = jnp.array([-1.6, 0.5])
        obs2 = [Y(1), X(0)]
        H2 = qml.ops.LinearCombination(coeffs2, obs2)

        assert H1 == (H2)

    def test_LinearCombination_add(self):
        """Tests that LinearCombinations are added correctly"""
        coeffs = jnp.array([0.5, -1.5])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = jnp.array([0.5, -0.5])
        H2 = qml.ops.LinearCombination(coeffs2, obs)

        coeffs_expected = jnp.array([1.0, -2.0])
        H = qml.ops.LinearCombination(coeffs_expected, obs)

        assert H == (H1 + H2)

    def test_LinearCombination_sub(self):
        """Tests that LinearCombinations are subtracted correctly"""

        coeffs = jnp.array([1.0, -2.0])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = jnp.array([0.5, -0.4])
        H2 = qml.ops.LinearCombination(coeffs2, obs)

        coeffs_expected = jnp.array([0.5, -1.6])
        H = qml.ops.LinearCombination(coeffs_expected, obs)

        assert H == (H1 - H2)

        H1 -= H2
        assert H == (H1)

    def test_LinearCombination_matmul(self):
        """Tests that LinearCombinations are tensored correctly"""

        coeffs = jnp.array([1.0, 2.0])
        obs = [X(0), Y(1)]
        H1 = qml.ops.LinearCombination(coeffs, obs)

        coeffs2 = jnp.array([-1.0, -2.0])
        obs2 = [X(2), Y(3)]
        H2 = qml.ops.LinearCombination(coeffs2, obs2)

        coeffs_expected = jnp.array([-4.0, -2.0, -2.0, -1.0])
        obs_expected = [
            qml.prod(Y(1), Y(3)),
            qml.prod(X(0), Y(3)),
            qml.prod(X(2), Y(1)),
            qml.prod(X(0), X(2)),
        ]
        H = qml.ops.LinearCombination(coeffs_expected, obs_expected)

        assert H == (H1 @ H2)


class TestGrouping:
    """Tests for the grouping functionality"""

    def test_set_on_initialization(self):
        """Test that grouping indices can be set on initialization."""

        op = qml.ops.LinearCombination([1, 1], [qml.X(0), qml.Y(1)], _grouping_indices=[[0, 1]])
        assert op.grouping_indices == [[0, 1]]

    def test_indentities_preserved(self):
        """Tests that the grouping indices do not drop identity terms when the wire order is nonstandard."""

        obs = [Z(1), Z(0), qml.Identity(0)]

        H = qml.ops.LinearCombination([1.0, 1.0, 1.0], obs, grouping_type="qwc")
        assert H.grouping_indices == ((0, 1, 2),)

    def test_grouping_is_correct_kwarg(self):
        """Basic test checking that grouping with a kwarg works as expected"""
        a = X(0)
        b = X(1)
        c = Z(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        H = qml.ops.LinearCombination(coeffs, obs, grouping_type="qwc")
        assert H.grouping_indices == ((0, 1), (2,))

    def test_grouping_is_correct_compute_grouping(self):
        """Basic test checking that grouping with compute_grouping works as expected"""
        a = X(0)
        b = X(1)
        c = Z(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        H = qml.ops.LinearCombination(coeffs, obs, grouping_type="qwc")
        H.compute_grouping()
        assert H.grouping_indices == ((0, 1), (2,))

    def test_grouping_raises_error(self):
        """Check that compute_grouping raises an error when
        attempting to compute groups for non-Pauli operators"""
        a = qml.Hadamard(0)
        b = X(1)
        c = Z(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        with pytest.raises(ValueError, match="Cannot compute grouping"):
            H = qml.ops.LinearCombination(coeffs, obs, grouping_type="qwc")
            H.compute_grouping()

    def test_set_grouping(self):
        """Test that we can set grouping indices."""
        H = qml.ops.LinearCombination([1.0, 2.0, 3.0], [X(0), X(1), Z(0)])
        H.grouping_indices = [[0, 1], [2]]

        assert H.grouping_indices == ((0, 1), (2,))

    def test_set_grouping_error(self):
        """Test that grouping indices are validated."""
        H = qml.ops.LinearCombination([1.0, 2.0, 3.0], [X(0), X(1), Z(0)])

        with pytest.raises(ValueError, match="The grouped index value"):
            H.grouping_indices = [[3, 1], [2]]

        with pytest.raises(ValueError, match="The grouped index value"):
            H.grouping_indices = "a"

    def test_grouping_for_non_groupable_LinearCombinations(self):
        """Test that grouping is computed correctly, even if no observables commute"""
        a = X(0)
        b = Y(0)
        c = Z(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        H = qml.ops.LinearCombination(coeffs, obs, grouping_type="qwc")
        assert H.grouping_indices == ((0,), (1,), (2,))

    def test_grouping_is_reset_when_simplifying(self):
        """Tests that calling simplify() resets the grouping"""
        obs = [X(0), X(1), Z(0)]
        coeffs = [1.0, 2.0, 3.0]

        H = qml.ops.LinearCombination(coeffs, obs, grouping_type="qwc")
        assert H.grouping_indices is not None

        H = H.simplify()
        assert H.grouping_indices is None

    def test_grouping_does_not_alter_queue(self):
        """Tests that grouping is invisible to the queue."""
        a = X(0)
        b = X(1)
        c = Z(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        with qml.queuing.AnnotatedQueue() as q:
            H = qml.ops.LinearCombination(coeffs, obs, grouping_type="qwc")

        assert q.queue == [H]

    def test_grouping_method_can_be_set(self):
        r"""Tests that the grouping method can be controlled by kwargs.
        This is done by changing from default to 'lf' and checking the result."""
        # Create a graph with unique solution so that test does not depend on solver/implementation
        a = X(0)
        b = X(0)
        c = Z(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        # compute grouping during construction
        H2 = qml.ops.LinearCombination(coeffs, obs, grouping_type="qwc", method="lf")
        assert set(H2.grouping_indices) == {(0, 1), (2,)}

        # compute grouping separately
        H3 = qml.ops.LinearCombination(coeffs, obs, grouping_type=None)
        H3.compute_grouping(method="lf")
        assert set(H3.grouping_indices) == {(0, 1), (2,)}

    def test_grouping_with_duplicate_terms(self):
        """Test that the grouping indices are correct when the LinearCombination has duplicate
        operators."""
        a = X(0)
        b = X(1)
        c = Z(0)
        d = X(0)
        e = Z(0)
        obs = [a, b, c, d, e]
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]

        # compute grouping during construction
        H2 = qml.ops.LinearCombination(coeffs, obs, grouping_type="qwc")

        assert H2.grouping_indices == ((0, 1, 3), (2, 4))
        # Following assertions are to check that grouping does not mutate the list of ops/coeffs
        assert H2.coeffs == coeffs
        assert H2.ops == obs


class TestLinearCombinationEvaluation:
    """Test the usage of a LinearCombination as an observable"""

    @pytest.mark.parametrize("coeffs, param, interface", COEFFS_PARAM_INTERFACE)
    def test_vqe_forward_different_coeff_types(self, coeffs, param, interface):
        """Check that manually splitting a LinearCombination expectation has the same
        result as passing the LinearCombination as an observable"""
        device = qml.device("default.qubit", wires=2)
        H = qml.ops.LinearCombination(coeffs, [X(0), Z(0)])

        @qml.qnode(device, interface=interface)
        def circuit():
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(H)

        @qml.qnode(device, interface=interface)
        def node1():
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(X(0))

        @qml.qnode(device, interface=interface)
        def node2():
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(Z(0))

        res = circuit()
        res_expected = coeffs[0] * node1() + coeffs[1] * node2()
        assert np.isclose(res, res_expected)

    def test_simplify_reduces_tape_parameters(self):
        """Test that simplifying a LinearCombination reduces the number of parameters on a tape"""
        device = qml.device("default.qubit", wires=2)

        @qml.qnode(device)
        def circuit():
            qml.RY(0.1, wires=0)
            return qml.expval(qml.simplify(qml.ops.LinearCombination([1.0, 2.0], [X(1), X(1)])))

        tape = qml.workflow.construct_tape(circuit)()
        pars = tape.get_parameters(trainable_only=False)
        # simplify worked and added 1. and 2.
        assert pars == [0.1, 3.0]


class TestLinearCombinationDifferentiation:
    """Test that the LinearCombination coefficients are differentiable"""

    @pytest.mark.parametrize("simplify", [True, False])
    @pytest.mark.parametrize("group", [None, "qwc"])
    def test_trainable_coeffs_paramshift(self, simplify, group):
        """Test the parameter-shift method by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a LinearCombination expectation"""
        coeffs = pnp.array([-0.05, 0.17], requires_grad=True)
        param = pnp.array(1.7, requires_grad=True)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.simplify(qml.ops.LinearCombination(coeffs, [X(0), Z(0)], grouping_type=group))
                if simplify
                else qml.ops.LinearCombination(coeffs, [X(0), Z(0)], grouping_type=group)
            )

        grad_fn = qml.grad(circuit)
        grad = grad_fn(coeffs, param)

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)
        half1 = qml.QNode(circuit1, dev, diff_method="parameter-shift")
        half2 = qml.QNode(circuit2, dev, diff_method="parameter-shift")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        grad_fn_expected = qml.grad(combine)
        grad_expected = grad_fn_expected(coeffs, param)

        assert np.allclose(grad[0], grad_expected[0])
        assert np.allclose(grad[1], grad_expected[1])

    def test_nontrainable_coeffs_paramshift(self):
        """Test the parameter-shift method if the coefficients are explicitly set non-trainable
        by not passing them to the qnode."""
        coeffs = np.array([-0.05, 0.17])
        param = pnp.array(1.7, requires_grad=True)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.ops.LinearCombination(
                    coeffs,
                    [X(0), Z(0)],
                )
            )

        grad_fn = qml.grad(circuit)
        grad = grad_fn(param)

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)
        half1 = qml.QNode(circuit1, dev, diff_method="parameter-shift")
        half2 = qml.QNode(circuit2, dev, diff_method="parameter-shift")

        def combine(param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        grad_fn_expected = qml.grad(combine)
        grad_expected = grad_fn_expected(param)

        assert np.allclose(grad, grad_expected)

    @pytest.mark.autograd
    @pytest.mark.parametrize("simplify", [True, False])
    @pytest.mark.parametrize("group", [None, "qwc"])
    def test_trainable_coeffs_autograd(self, simplify, group):
        """Test the autograd interface by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a LinearCombination expectation"""
        coeffs = pnp.array([-0.05, 0.17], requires_grad=True)
        param = pnp.array(1.7, requires_grad=True)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="autograd")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.simplify(qml.ops.LinearCombination(coeffs, [X(0), Z(0)], grouping_type=group))
                if simplify
                else qml.ops.LinearCombination(coeffs, [X(0), Z(0)], grouping_type=group)
            )

        grad_fn = qml.grad(circuit)
        grad = grad_fn(coeffs, param)

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)
        half1 = qml.QNode(circuit1, dev, interface="autograd")
        half2 = qml.QNode(circuit2, dev, interface="autograd")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        grad_fn_expected = qml.grad(combine)
        grad_expected = grad_fn_expected(coeffs, param)

        assert np.allclose(grad[0], grad_expected[0])
        assert np.allclose(grad[1], grad_expected[1])

    @pytest.mark.autograd
    def test_nontrainable_coeffs_autograd(self):
        """Test the autograd interface if the coefficients are explicitly set non-trainable"""
        coeffs = pnp.array([-0.05, 0.17], requires_grad=False)
        param = pnp.array(1.7, requires_grad=True)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="autograd")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.ops.LinearCombination(coeffs, [X(0), Z(0)]))

        grad_fn = qml.grad(circuit)
        grad = grad_fn(coeffs, param)

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)
        half1 = qml.QNode(circuit1, dev, interface="autograd")
        half2 = qml.QNode(circuit2, dev, interface="autograd")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        grad_fn_expected = qml.grad(combine)
        grad_expected = grad_fn_expected(coeffs, param)

        assert np.allclose(grad, grad_expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("simplify", [True, False])
    @pytest.mark.parametrize("group", [None, "qwc"])
    def test_trainable_coeffs_jax(self, simplify, group):
        """Test the jax interface by comparing the differentiation of linearly
        combined subcircuits with the differentiation of a LinearCombination expectation"""

        coeffs = jnp.array([-0.05, 0.17])
        param = jnp.array(1.7)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.simplify(qml.ops.LinearCombination(coeffs, [X(0), Z(0)], grouping_type=group))
                if simplify
                else qml.ops.LinearCombination(coeffs, [X(0), Z(0)], grouping_type=group)
            )

        grad_fn = jax.grad(circuit)
        grad = grad_fn(coeffs, param)

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)
        half1 = qml.QNode(circuit1, dev, interface="jax", diff_method="backprop")
        half2 = qml.QNode(circuit2, dev, interface="jax", diff_method="backprop")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        grad_fn_expected = jax.grad(combine)
        grad_expected = grad_fn_expected(coeffs, param)

        assert np.allclose(grad[0], grad_expected[0])
        assert np.allclose(grad[1], grad_expected[1])

    # pylint: disable=superfluous-parens
    @pytest.mark.jax
    def test_nontrainable_coeffs_jax(self):
        """Test the jax interface if the coefficients are explicitly set non-trainable"""
        coeffs = np.array([-0.05, 0.17])
        param = jnp.array(1.7)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.ops.LinearCombination(coeffs, [X(0), Z(0)]))

        grad_fn = jax.grad(circuit, argnums=1)
        grad = grad_fn(coeffs, param)

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)
        half1 = qml.QNode(circuit1, dev, interface="jax", diff_method="backprop")
        half2 = qml.QNode(circuit2, dev, interface="jax", diff_method="backprop")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        grad_fn_expected = jax.grad(combine, argnums=1)
        grad_expected = grad_fn_expected(coeffs, param)

        assert np.allclose(grad, grad_expected)

    @pytest.mark.torch
    @pytest.mark.parametrize("simplify", [True, False])
    @pytest.mark.parametrize("group", [None, "qwc"])
    def test_trainable_coeffs_torch_simplify(self, group, simplify):
        """Test the torch interface by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a LinearCombination expectation"""
        coeffs = torch.tensor([-0.05, 0.17], requires_grad=True)
        param = torch.tensor(1.7, requires_grad=True)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.simplify(qml.ops.LinearCombination(coeffs, [X(0), Z(0)], grouping_type=group))
                if simplify
                else qml.ops.LinearCombination(coeffs, [X(0), Z(0)], grouping_type=group)
            )

        res = circuit(coeffs, param)
        res.backward()  # pylint:disable=no-member
        grad = (coeffs.grad, param.grad)

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)

        # we need to create new tensors here
        coeffs2 = torch.tensor([-0.05, 0.17], requires_grad=True)
        param2 = torch.tensor(1.7, requires_grad=True)

        half1 = qml.QNode(circuit1, dev, interface="torch", diff_method="backprop")
        half2 = qml.QNode(circuit2, dev, interface="torch", diff_method="backprop")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        res_expected = combine(coeffs2, param2)
        res_expected.backward()
        grad_expected = (coeffs2.grad, param2.grad)

        assert qml.math.allclose(grad[0], grad_expected[0])
        assert qml.math.allclose(grad[1], grad_expected[1])

    @pytest.mark.torch
    def test_nontrainable_coeffs_torch(self):
        """Test the torch interface if the coefficients are explicitly set non-trainable"""
        coeffs = torch.tensor([-0.05, 0.17], requires_grad=False)
        param = torch.tensor(1.7, requires_grad=True)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.ops.LinearCombination(
                    coeffs,
                    [X(0), Z(0)],
                )
            )

        res = circuit(coeffs, param)
        res.backward()  # pylint:disable=no-member

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)

        # we need to create new tensors here
        coeffs2 = torch.tensor([-0.05, 0.17], requires_grad=False)
        param2 = torch.tensor(1.7, requires_grad=True)

        half1 = qml.QNode(circuit1, dev, interface="torch", diff_method="backprop")
        half2 = qml.QNode(circuit2, dev, interface="torch", diff_method="backprop")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        res_expected = combine(coeffs2, param2)
        res_expected.backward()

        assert coeffs.grad is None
        assert np.allclose(param.grad, param2.grad)

    @pytest.mark.tf
    @pytest.mark.parametrize("simplify", [True, False])
    @pytest.mark.parametrize("group", [None, "qwc"])
    def test_trainable_coeffs_tf(self, simplify, group):
        """Test the tf interface by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a LinearCombination expectation"""
        coeffs = tf.Variable([-0.05, 0.17], dtype=tf.double)
        param = tf.Variable(1.7, dtype=tf.double)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.simplify(qml.ops.LinearCombination(coeffs, [X(0), Z(0)], grouping_type=group))
                if simplify
                else qml.ops.LinearCombination(coeffs, [X(0), Z(0)], grouping_type=group)
            )

        with tf.GradientTape() as tape:
            res = circuit(coeffs, param)
        grad = tape.gradient(res, [coeffs, param])

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)

        # we need to create new tensors here
        coeffs2 = tf.Variable([-0.05, 0.17], dtype=tf.double)
        param2 = tf.Variable(1.7, dtype=tf.double)
        half1 = qml.QNode(circuit1, dev, interface="tf", diff_method="backprop")
        half2 = qml.QNode(circuit2, dev, interface="tf", diff_method="backprop")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        with tf.GradientTape() as tape2:
            res_expected = combine(coeffs2, param2)
        grad_expected = tape2.gradient(res_expected, [coeffs2, param2])

        assert np.allclose(grad[0], grad_expected[0])
        assert np.allclose(grad[1], grad_expected[1])

    @pytest.mark.tf
    def test_nontrainable_coeffs_tf(self):
        """Test the tf interface if the coefficients are explicitly set non-trainable"""

        coeffs = tf.constant([-0.05, 0.17], dtype=tf.double)
        param = tf.Variable(1.7, dtype=tf.double)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.ops.LinearCombination(
                    coeffs,
                    [X(0), Z(0)],
                )
            )

        with tf.GradientTape() as tape:
            res = circuit(coeffs, param)
        grad = tape.gradient(res, [coeffs, param])

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)

        # we need to create new tensors here
        coeffs2 = tf.constant([-0.05, 0.17], dtype=tf.double)
        param2 = tf.Variable(1.7, dtype=tf.double)
        half1 = qml.QNode(circuit1, dev, interface="tf", diff_method="backprop")
        half2 = qml.QNode(circuit2, dev, interface="tf", diff_method="backprop")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        with tf.GradientTape() as tape2:
            res_expected = combine(coeffs2, param2)
        grad_expected = tape2.gradient(res_expected, [coeffs2, param2])

        assert grad[0] is None
        assert np.allclose(grad[1], grad_expected[1])

    # TODO: update logic of adjoint differentiation to catch attempt to differentiate lincomb coeffs
    @pytest.mark.xfail
    def test_not_supported_by_adjoint_differentiation(self):
        """Test that error is raised when attempting the adjoint differentiation method."""
        device = qml.device("default.qubit", wires=2)

        coeffs = pnp.array([-0.05, 0.17], requires_grad=True)
        param = pnp.array(1.7, requires_grad=True)

        @qml.qnode(device, diff_method="adjoint")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.ops.LinearCombination(
                    coeffs,
                    [X(0), Z(0)],
                )
            )

        grad_fn = qml.grad(circuit)
        with pytest.raises(
            DeviceError,
            match="not supported on adjoint",
        ):
            grad_fn(coeffs, param)


# pylint: disable=protected-access
@pytest.mark.capture
def test_create_instance_while_tracing():
    """Test that a LinearCombination instance can be created while tracing."""

    def f(a, b):
        op1 = qml.X._primitive.impl(0, n_wires=1)
        op2 = qml.Y._primitive.impl(0, n_wires=1)
        op = qml.ops.LinearCombination._primitive.impl(a, b, op1, op2, n_obs=2)
        assert isinstance(op, qml.ops.LinearCombination)

    jax.make_jaxpr(f)(1, 2)
