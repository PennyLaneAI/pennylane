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
Tests for the Hamiltonian class.
"""
from unittest.mock import patch

import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.wires import Wires
from collections.abc import Iterable

# Make test data in different interfaces, if installed
COEFFS_PARAM_INTERFACE = [
    ([-0.05, 0.17], 1.7, "autograd"),
    (np.array([-0.05, 0.17]), np.array(1.7), "autograd"),
    (pnp.array([-0.05, 0.17], requires_grad=True), pnp.array(1.7, requires_grad=True), "autograd"),
]

try:
    from jax import numpy as jnp

    COEFFS_PARAM_INTERFACE.append((jnp.array([-0.05, 0.17]), jnp.array(1.7), "jax"))
except ImportError:
    pass

try:
    import tf

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
    (qml.PauliZ(0), qml.PauliY(0), qml.PauliZ(1)),
    (qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(0) @ qml.PauliZ(1), qml.PauliZ(1)),
    (qml.Hermitian(H_TWO_QUBITS, [0, 1]),),
]

valid_hamiltonians = [
    ((1.0,), (qml.Hermitian(H_TWO_QUBITS, [0, 1]),)),
    ((-0.8,), (qml.PauliZ(0),)),
    ((0.6,), (qml.PauliX(0) @ qml.PauliX(1),)),
    ((0.5, -1.6), (qml.PauliX(0), qml.PauliY(1))),
    ((0.5, -1.6), (qml.PauliX(1), qml.PauliY(1))),
    ((0.5, -1.6), (qml.PauliX("a"), qml.PauliY("b"))),
    ((1.1, -0.4, 0.333), (qml.PauliX(0), qml.Hermitian(H_ONE_QUBIT, 2), qml.PauliZ(2))),
    ((-0.4, 0.15), (qml.Hermitian(H_TWO_QUBITS, [0, 2]), qml.PauliZ(1))),
    ([1.5, 2.0], [qml.PauliZ(0), qml.PauliY(2)]),
    (np.array([-0.1, 0.5]), [qml.Hermitian(H_TWO_QUBITS, [0, 1]), qml.PauliY(0)]),
    ((0.5, 1.2), (qml.PauliX(0), qml.PauliX(0) @ qml.PauliX(1))),
    ((0.5 + 1.2j, 1.2 + 0.5j), (qml.PauliX(0), qml.PauliY(1))),
    ((0.7 + 0j, 0 + 1.3j), (qml.PauliX(0), qml.PauliY(1))),
]

valid_hamiltonians_str = [
    "  (1.0) [Hermitian0,1]",
    "  (-0.8) [Z0]",
    "  (0.6) [X0 X1]",
    "  (-1.6) [Y1]\n+ (0.5) [X0]",
    "  (-1.6) [Y1]\n+ (0.5) [X1]",
    "  (-1.6) [Yb]\n+ (0.5) [Xa]",
    "  (-0.4) [Hermitian2]\n+ (0.333) [Z2]\n+ (1.1) [X0]",
    "  (0.15) [Z1]\n+ (-0.4) [Hermitian0,2]",
    "  (1.5) [Z0]\n+ (2.0) [Y2]",
    "  (0.5) [Y0]\n+ (-0.1) [Hermitian0,1]",
    "  (0.5) [X0]\n+ (1.2) [X0 X1]",
    "  ((0.5+1.2j)) [X0]\n+ ((1.2+0.5j)) [Y1]",
    "  (1.3j) [Y1]\n+ ((0.7+0j)) [X0]",
]

valid_hamiltonians_repr = [
    "<Hamiltonian: terms=1, wires=[0, 1]>",
    "<Hamiltonian: terms=1, wires=[0]>",
    "<Hamiltonian: terms=1, wires=[0, 1]>",
    "<Hamiltonian: terms=2, wires=[0, 1]>",
    "<Hamiltonian: terms=2, wires=[1]>",
    "<Hamiltonian: terms=2, wires=['a', 'b']>",
    "<Hamiltonian: terms=3, wires=[0, 2]>",
    "<Hamiltonian: terms=2, wires=[0, 1, 2]>",
    "<Hamiltonian: terms=2, wires=[0, 2]>",
    "<Hamiltonian: terms=2, wires=[0, 1]>",
    "<Hamiltonian: terms=2, wires=[0, 1]>",
    "<Hamiltonian: terms=2, wires=[0, 1]>",
    "<Hamiltonian: terms=2, wires=[0, 1]>",
]

invalid_hamiltonians = [
    ((), (qml.PauliZ(0),)),
    ((), (qml.PauliZ(0), qml.PauliY(1))),
    ((3.5,), ()),
    ((1.2, -0.4), ()),
    ((0.5, 1.2), (qml.PauliZ(0),)),
    ((1.0,), (qml.PauliZ(0), qml.PauliY(0))),
]

simplify_hamiltonians = [
    (
        qml.Hamiltonian([1, 1, 1], [qml.PauliX(0) @ qml.Identity(1), qml.PauliX(0), qml.PauliX(1)]),
        qml.Hamiltonian([2, 1], [qml.PauliX(0), qml.PauliX(1)]),
    ),
    (
        qml.Hamiltonian(
            [-1, 1, 1], [qml.PauliX(0) @ qml.Identity(1), qml.PauliX(0), qml.PauliX(1)]
        ),
        qml.Hamiltonian([1], [qml.PauliX(1)]),
    ),
    (
        qml.Hamiltonian(
            [1, 0.5],
            [qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(1) @ qml.Identity(2) @ qml.PauliX(0)],
        ),
        qml.Hamiltonian([1.5], [qml.PauliX(0) @ qml.PauliY(1)]),
    ),
    (
        qml.Hamiltonian(
            [1, 1, 0.5],
            [
                qml.Hermitian(np.array([[1, 0], [0, -1]]), "a"),
                qml.PauliX("b") @ qml.PauliY(1.3),
                qml.PauliY(1.3) @ qml.Identity(-0.9) @ qml.PauliX("b"),
            ],
        ),
        qml.Hamiltonian(
            [1, 1.5],
            [qml.Hermitian(np.array([[1, 0], [0, -1]]), "a"), qml.PauliX("b") @ qml.PauliY(1.3)],
        ),
    ),
    # Simplifies to zero Hamiltonian
    (
        qml.Hamiltonian(
            [1, -0.5, -0.5], [qml.PauliX(0) @ qml.Identity(1), qml.PauliX(0), qml.PauliX(0)]
        ),
        qml.Hamiltonian([], []),
    ),
    (
        qml.Hamiltonian(
            [1, -1],
            [qml.PauliX(4) @ qml.Identity(0) @ qml.PauliX(1), qml.PauliX(4) @ qml.PauliX(1)],
        ),
        qml.Hamiltonian([], []),
    ),
    (
        qml.Hamiltonian([0], [qml.Identity(0)]),
        qml.Hamiltonian([0], [qml.Identity(0)]),
    ),
]

equal_hamiltonians = [
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0) @ qml.Identity(1), qml.PauliZ(0)]),
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(0)]),
        True,
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0) @ qml.Identity(1), qml.PauliY(2) @ qml.PauliZ(0)]),
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliY(2) @ qml.Identity(1)]),
        True,
    ),
    (
        qml.Hamiltonian(
            [1, 1, 1], [qml.PauliX(0) @ qml.Identity(1), qml.PauliZ(0), qml.Identity(1)]
        ),
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(0)]),
        False,
    ),
    (qml.Hamiltonian([1], [qml.PauliZ(0) @ qml.PauliX(1)]), qml.PauliZ(0) @ qml.PauliX(1), True),
    (qml.Hamiltonian([1], [qml.PauliZ(0)]), qml.PauliZ(0), True),
    (
        qml.Hamiltonian(
            [1, 1, 1],
            [
                qml.Hermitian(np.array([[1, 0], [0, -1]]), "b") @ qml.Identity(7),
                qml.PauliZ(3),
                qml.Identity(1.2),
            ],
        ),
        qml.Hamiltonian(
            [1, 1, 1],
            [qml.Hermitian(np.array([[1, 0], [0, -1]]), "b"), qml.PauliZ(3), qml.Identity(1.2)],
        ),
        True,
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliZ(3) @ qml.Identity(1.2), qml.PauliZ(3)]),
        qml.Hamiltonian([2], [qml.PauliZ(3)]),
        True,
    ),
]

add_hamiltonians = [
    (
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
        qml.Hamiltonian([0.5, 0.3, 1], [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)]),
        qml.Hamiltonian(
            [1.5, 1.2, 1.1, 0.3], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2), qml.PauliX(1)]
        ),
    ),
    (
        qml.Hamiltonian(
            [1.3, 0.2, 0.7], [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(2)]
        ),
        qml.Hamiltonian(
            [0.5, 0.3, 1.6], [qml.PauliX(0), qml.PauliX(1) @ qml.PauliX(0), qml.PauliX(2)]
        ),
        qml.Hamiltonian(
            [1.6, 0.2, 2.3, 0.5],
            [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(2), qml.PauliX(0)],
        ),
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
        qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
        qml.Hamiltonian([1.5, 1.5], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
    ),
    (
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
        qml.PauliX(0) @ qml.Identity(1),
        qml.Hamiltonian([2, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
    ),
    (
        qml.Hamiltonian(
            [1.3, 0.2, 0.7], [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(2)]
        ),
        qml.Hadamard(1),
        qml.Hamiltonian(
            [1.3, 1.2, 0.7], [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(2)]
        ),
    ),
    (
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX("b"), qml.PauliZ(3.1), qml.PauliX(1.6)]),
        qml.PauliX("b") @ qml.Identity(5),
        qml.Hamiltonian([2, 1.2, 0.1], [qml.PauliX("b"), qml.PauliZ(3.1), qml.PauliX(1.6)]),
    ),
    # Case where arguments coeffs and ops to the Hamiltonian are iterables other than lists
    (
        qml.Hamiltonian((1, 1.2, 0.1), (qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2))),
        qml.Hamiltonian(
            np.array([0.5, 0.3, 1]), np.array([qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)])
        ),
        qml.Hamiltonian(
            (1.5, 1.2, 1.1, 0.3),
            np.array([qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2), qml.PauliX(1)]),
        ),
    ),
    # Case where the 1st hamiltonian doesn't contain all wires
    (
        qml.Hamiltonian([1.23, -3.45], [qml.PauliX(0), qml.PauliY(1)]),
        qml.Hamiltonian([6.78], [qml.PauliZ(2)]),
        qml.Hamiltonian([1.23, -3.45, 6.78], [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]),
    ),
]

add_zero_hamiltonians = [
    qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
    qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
    qml.Hamiltonian(
        [1.5, 1.2, 1.1, 0.3], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2), qml.PauliX(1)]
    ),
]

iadd_zero_hamiltonians = [
    # identical hamiltonians
    (
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
    ),
    (
        qml.Hamiltonian(
            [1.5, 1.2, 1.1, 0.3], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2), qml.PauliX(1)]
        ),
        qml.Hamiltonian(
            [1.5, 1.2, 1.1, 0.3], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2), qml.PauliX(1)]
        ),
    ),
]

sub_hamiltonians = [
    (
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
        qml.Hamiltonian([0.5, 0.3, 1.6], [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)]),
        qml.Hamiltonian(
            [0.5, 1.2, -1.5, -0.3], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2), qml.PauliX(1)]
        ),
    ),
    (
        qml.Hamiltonian(
            [1.3, 0.2, 1], [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(2)]
        ),
        qml.Hamiltonian(
            [0.5, 0.3, 1], [qml.PauliX(0), qml.PauliX(1) @ qml.PauliX(0), qml.PauliX(2)]
        ),
        qml.Hamiltonian(
            [1, 0.2, -0.5], [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(0)]
        ),
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
        qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
        qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
    ),
    (
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
        qml.PauliX(0) @ qml.Identity(1),
        qml.Hamiltonian([1.2, 0.1], [qml.PauliZ(1), qml.PauliX(2)]),
    ),
    (
        qml.Hamiltonian(
            [1.3, 0.2, 0.7], [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(2)]
        ),
        qml.Hadamard(1),
        qml.Hamiltonian(
            [1.3, -0.8, 0.7], [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(2)]
        ),
    ),
    (
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX("b"), qml.PauliZ(3.1), qml.PauliX(1.6)]),
        qml.PauliX("b") @ qml.Identity(1),
        qml.Hamiltonian([1.2, 0.1], [qml.PauliZ(3.1), qml.PauliX(1.6)]),
    ),
    # The result is the zero Hamiltonian
    (
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
        qml.Hamiltonian([], []),
    ),
    (
        qml.Hamiltonian([1, 2], [qml.PauliX(4), qml.PauliZ(2)]),
        qml.Hamiltonian([1, 2], [qml.PauliX(4), qml.PauliZ(2)]),
        qml.Hamiltonian([], []),
    ),
    # Case where arguments coeffs and ops to the Hamiltonian are iterables other than lists
    (
        qml.Hamiltonian((1, 1.2, 0.1), (qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2))),
        qml.Hamiltonian(
            np.array([0.5, 0.3, 1.6]), np.array([qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)])
        ),
        qml.Hamiltonian(
            (0.5, 1.2, -1.5, -0.3),
            np.array([qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2), qml.PauliX(1)]),
        ),
    ),
    # Case where the 1st hamiltonian doesn't contain all wires
    (
        qml.Hamiltonian([1.23, -3.45], [qml.PauliX(0), qml.PauliY(1)]),
        qml.Hamiltonian([6.78], [qml.PauliZ(2)]),
        qml.Hamiltonian([1.23, -3.45, -6.78], [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]),
    ),
]

mul_hamiltonians = [
    (
        0.5,
        qml.Hamiltonian(
            [1, 2], [qml.PauliX(0), qml.PauliZ(1)]
        ),  # Case where the types of the coefficient and the scalar differ
        qml.Hamiltonian([0.5, 1.0], [qml.PauliX(0), qml.PauliZ(1)]),
    ),
    (
        3,
        qml.Hamiltonian([1.5, 0.5], [qml.PauliX(0), qml.PauliZ(1)]),
        qml.Hamiltonian([4.5, 1.5], [qml.PauliX(0), qml.PauliZ(1)]),
    ),
    (
        -1.3,
        qml.Hamiltonian([1, -0.3], [qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]),
        qml.Hamiltonian([-1.3, 0.39], [qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ),
    (
        -1.3,
        qml.Hamiltonian(
            [1, -0.3],
            [qml.Hermitian(np.array([[1, 0], [0, -1]]), "b"), qml.PauliZ(23) @ qml.PauliZ(0)],
        ),
        qml.Hamiltonian(
            [-1.3, 0.39],
            [qml.Hermitian(np.array([[1, 0], [0, -1]]), "b"), qml.PauliZ(23) @ qml.PauliZ(0)],
        ),
    ),
    # The result is the zero Hamiltonian
    (
        0,
        qml.Hamiltonian([1], [qml.PauliX(0)]),
        qml.Hamiltonian([0], [qml.PauliX(0)]),
    ),
    (
        0,
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
        qml.Hamiltonian([0, 0, 0], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
    ),
    # Case where arguments coeffs and ops to the Hamiltonian are iterables other than lists
    (
        3,
        qml.Hamiltonian((1.5, 0.5), (qml.PauliX(0), qml.PauliZ(1))),
        qml.Hamiltonian(np.array([4.5, 1.5]), np.array([qml.PauliX(0), qml.PauliZ(1)])),
    ),
]

matmul_hamiltonians = [
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)]),
        qml.Hamiltonian([0.5, 0.5], [qml.PauliZ(2), qml.PauliZ(3)]),
        qml.Hamiltonian(
            [0.5, 0.5, 0.5, 0.5],
            [
                qml.PauliX(0) @ qml.PauliZ(2),
                qml.PauliX(0) @ qml.PauliZ(3),
                qml.PauliZ(1) @ qml.PauliZ(2),
                qml.PauliZ(1) @ qml.PauliZ(3),
            ],
        ),
    ),
    (
        qml.Hamiltonian([0.5, 0.25], [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0)]),
        qml.Hamiltonian([1, 1], [qml.PauliX(3) @ qml.PauliZ(2), qml.PauliZ(2)]),
        qml.Hamiltonian(
            [0.5, 0.5, 0.25, 0.25],
            [
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(3) @ qml.PauliZ(2),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliZ(2),
                qml.PauliZ(0) @ qml.PauliX(3) @ qml.PauliZ(2),
                qml.PauliZ(0) @ qml.PauliZ(2),
            ],
        ),
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX("b"), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
        qml.Hamiltonian([2, 2], [qml.PauliZ(1.2), qml.PauliY("c")]),
        qml.Hamiltonian(
            [2, 2, 2, 2],
            [
                qml.PauliX("b") @ qml.PauliZ(1.2),
                qml.PauliX("b") @ qml.PauliY("c"),
                qml.Hermitian(np.array([[1, 0], [0, -1]]), 0) @ qml.PauliZ(1.2),
                qml.Hermitian(np.array([[1, 0], [0, -1]]), 0) @ qml.PauliY("c"),
            ],
        ),
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)]),
        qml.PauliX(2),
        qml.Hamiltonian([1, 1], [qml.PauliX(0) @ qml.PauliX(2), qml.PauliZ(1) @ qml.PauliX(2)]),
    ),
    # Case where arguments coeffs and ops to the Hamiltonian are iterables other than lists
    (
        qml.Hamiltonian((1, 1), (qml.PauliX(0), qml.PauliZ(1))),
        qml.Hamiltonian(np.array([0.5, 0.5]), np.array([qml.PauliZ(2), qml.PauliZ(3)])),
        qml.Hamiltonian(
            (0.5, 0.5, 0.5, 0.5),
            np.array(
                [
                    qml.PauliX(0) @ qml.PauliZ(2),
                    qml.PauliX(0) @ qml.PauliZ(3),
                    qml.PauliZ(1) @ qml.PauliZ(2),
                    qml.PauliZ(1) @ qml.PauliZ(3),
                ]
            ),
        ),
    ),
]

rmatmul_hamiltonians = [
    (
        qml.Hamiltonian([0.5, 0.5], [qml.PauliZ(2), qml.PauliZ(3)]),
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)]),
        qml.Hamiltonian(
            [0.5, 0.5, 0.5, 0.5],
            [
                qml.PauliX(0) @ qml.PauliZ(2),
                qml.PauliX(0) @ qml.PauliZ(3),
                qml.PauliZ(1) @ qml.PauliZ(2),
                qml.PauliZ(1) @ qml.PauliZ(3),
            ],
        ),
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(3) @ qml.PauliZ(2), qml.PauliZ(2)]),
        qml.Hamiltonian([0.5, 0.25], [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0)]),
        qml.Hamiltonian(
            [0.5, 0.5, 0.25, 0.25],
            [
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(3) @ qml.PauliZ(2),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliZ(2),
                qml.PauliZ(0) @ qml.PauliX(3) @ qml.PauliZ(2),
                qml.PauliZ(0) @ qml.PauliZ(2),
            ],
        ),
    ),
    (
        qml.Hamiltonian([2, 2], [qml.PauliZ(1.2), qml.PauliY("c")]),
        qml.Hamiltonian([1, 1], [qml.PauliX("b"), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
        qml.Hamiltonian(
            [2, 2, 2, 2],
            [
                qml.PauliX("b") @ qml.PauliZ(1.2),
                qml.PauliX("b") @ qml.PauliY("c"),
                qml.Hermitian(np.array([[1, 0], [0, -1]]), 0) @ qml.PauliZ(1.2),
                qml.Hermitian(np.array([[1, 0], [0, -1]]), 0) @ qml.PauliY("c"),
            ],
        ),
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)]),
        qml.PauliX(2),
        qml.Hamiltonian([1, 1], [qml.PauliX(2) @ qml.PauliX(0), qml.PauliX(2) @ qml.PauliZ(1)]),
    ),
    # Case where arguments coeffs and ops to the Hamiltonian are iterables other than lists
    (
        qml.Hamiltonian(np.array([0.5, 0.5]), np.array([qml.PauliZ(2), qml.PauliZ(3)])),
        qml.Hamiltonian((1, 1), (qml.PauliX(0), qml.PauliZ(1))),
        qml.Hamiltonian(
            (0.5, 0.5, 0.5, 0.5),
            np.array(
                [
                    qml.PauliX(0) @ qml.PauliZ(2),
                    qml.PauliX(0) @ qml.PauliZ(3),
                    qml.PauliZ(1) @ qml.PauliZ(2),
                    qml.PauliZ(1) @ qml.PauliZ(3),
                ]
            ),
        ),
    ),
]

big_hamiltonian_coeffs = np.array(
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

big_hamiltonian_ops = [
    qml.Identity(wires=[0]),
    qml.PauliZ(wires=[0]),
    qml.PauliZ(wires=[1]),
    qml.PauliZ(wires=[2]),
    qml.PauliZ(wires=[3]),
    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]),
    qml.PauliY(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliX(wires=[2]) @ qml.PauliY(wires=[3]),
    qml.PauliY(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliX(wires=[2]) @ qml.PauliX(wires=[3]),
    qml.PauliX(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliY(wires=[2]) @ qml.PauliY(wires=[3]),
    qml.PauliX(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliY(wires=[2]) @ qml.PauliX(wires=[3]),
    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[2]),
    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[3]),
    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[3]),
    qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[3]),
]

big_hamiltonian = qml.Hamiltonian(big_hamiltonian_coeffs, big_hamiltonian_ops)

big_hamiltonian_grad = (
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
    return qml.expval(qml.PauliX(0))


def circuit2(param):
    """Second Pauli subcircuit"""
    qml.RX(param, wires=0)
    qml.RY(param, wires=0)
    return qml.expval(qml.PauliZ(0))


dev = qml.device("default.qubit", wires=2)


class TestHamiltonian:
    """Test the Hamiltonian class"""

    @pytest.mark.parametrize("coeffs, ops", valid_hamiltonians)
    def test_hamiltonian_valid_init(self, coeffs, ops):
        """Tests that the Hamiltonian object is created with
        the correct attributes"""
        H = qml.Hamiltonian(coeffs, ops)
        assert np.allclose(H.terms()[0], coeffs)
        assert H.terms()[1] == list(ops)

    @pytest.mark.parametrize("coeffs, ops", invalid_hamiltonians)
    def test_hamiltonian_invalid_init_exception(self, coeffs, ops):
        """Tests that an exception is raised when giving an invalid
        combination of coefficients and ops"""
        with pytest.raises(ValueError, match="number of coefficients and operators does not match"):
            H = qml.Hamiltonian(coeffs, ops)

    @pytest.mark.parametrize(
        "obs", [[qml.PauliX(0), qml.CNOT(wires=[0, 1])], [qml.PauliZ, qml.PauliZ(0)]]
    )
    def test_hamiltonian_invalid_observables(self, obs):
        """Tests that an exception is raised when
        a complex Hamiltonian is given"""
        coeffs = [0.1, 0.2]

        with pytest.raises(ValueError, match="observables are not valid"):
            qml.Hamiltonian(coeffs, obs)

    @pytest.mark.parametrize("coeffs, ops", valid_hamiltonians)
    def test_hamiltonian_wires(self, coeffs, ops):
        """Tests that the Hamiltonian object has correct wires."""
        H = qml.Hamiltonian(coeffs, ops)
        assert set(H.wires) == set([w for op in H.ops for w in op.wires])

    def test_label(self):
        """Tests the label method of Hamiltonian when <=3 coefficients."""
        H = qml.Hamiltonian((-0.8,), (qml.PauliZ(0),))
        assert H.label() == "𝓗"
        assert H.label(decimals=2) == "𝓗\n(-0.80)"

    def test_label_many_coefficients(self):
        """Tests the label method of Hamiltonian when >3 coefficients."""
        H = (
            0.1 * qml.PauliX(0)
            + 0.1 * qml.PauliY(1)
            + 0.3 * qml.PauliZ(0) @ qml.PauliX(1)
            + 0.4 * qml.PauliX(3)
        )
        assert H.label() == "𝓗"
        assert H.label(decimals=2) == "𝓗"

    @pytest.mark.parametrize("terms, string", zip(valid_hamiltonians, valid_hamiltonians_str))
    def test_hamiltonian_str(self, terms, string):
        """Tests that the __str__ function for printing is correct"""
        H = qml.Hamiltonian(*terms)
        assert H.__str__() == string

    @patch("builtins.print")
    def test_small_hamiltonian_ipython_display(self, mock_print):
        """Test that the ipython_dipslay method prints __str__."""
        H = 1.0 * qml.PauliX(0)
        H._ipython_display_()
        mock_print.assert_called_with(str(H))

    @patch("builtins.print")
    def test_big_hamiltonian_ipython_display(self, mock_print):
        """Test that the ipython_display method prints __repr__ when H has more than 15 terms."""
        H = qml.Hamiltonian([1] * 16, [qml.PauliX(i) for i in range(16)])
        H._ipython_display_()
        mock_print.assert_called_with(repr(H))

    @pytest.mark.parametrize("terms, string", zip(valid_hamiltonians, valid_hamiltonians_repr))
    def test_hamiltonian_repr(self, terms, string):
        """Tests that the __repr__ function for printing is correct"""
        H = qml.Hamiltonian(*terms)
        assert H.__repr__() == string

    def test_hamiltonian_name(self):
        """Tests the name property of the Hamiltonian class"""
        H = qml.Hamiltonian([], [])
        assert H.name == "Hamiltonian"

    @pytest.mark.parametrize(("old_H", "new_H"), simplify_hamiltonians)
    def test_simplify(self, old_H, new_H):
        """Tests the simplify method"""
        old_H.simplify()
        assert old_H.compare(new_H)

    def test_simplify_while_queueing(self):
        """Tests that simplifying a Hamiltonian in a tape context
        queues the simplified Hamiltonian."""

        with qml.tape.QuantumTape() as tape:
            a = qml.PauliX(wires=0)
            b = qml.PauliY(wires=1)
            c = qml.Identity(wires=2)
            d = b @ c
            H = qml.Hamiltonian([1.0, 2.0], [a, d])
            H.simplify()

        # check that H is simplified
        assert H.ops == [a, b]
        # check that the simplified Hamiltonian is in the queue
        assert H in tape._queue

    def test_data(self):
        """Tests the obs_data method"""

        H = qml.Hamiltonian(
            [1, 1, 0.5],
            [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliX(2) @ qml.Identity(1)],
        )
        data = H._obs_data()

        assert data == {
            (1, frozenset([("PauliZ", qml.wires.Wires(0), ())])),
            (
                1,
                frozenset([("PauliZ", qml.wires.Wires(0), ()), ("PauliX", qml.wires.Wires(1), ())]),
            ),
            (0.5, frozenset([("PauliX", qml.wires.Wires(2), ())])),
        }

    def test_hamiltonian_equal_error(self):
        """Tests that the correct error is raised when compare() is called on invalid type"""

        H = qml.Hamiltonian([1], [qml.PauliZ(0)])
        with pytest.raises(
            ValueError,
            match=r"Can only compare a Hamiltonian, and a Hamiltonian/Observable/Tensor.",
        ):
            H.compare([[1, 0], [0, -1]])

    @pytest.mark.parametrize(("H1", "H2", "res"), equal_hamiltonians)
    def test_hamiltonian_equal(self, H1, H2, res):
        """Tests that equality can be checked between Hamiltonians"""
        assert H1.compare(H2) == res

    @pytest.mark.parametrize(("H1", "H2", "H"), add_hamiltonians)
    def test_hamiltonian_add(self, H1, H2, H):
        """Tests that Hamiltonians are added correctly"""
        assert H.compare(H1 + H2)

    @pytest.mark.parametrize("H", add_zero_hamiltonians)
    def test_hamiltonian_add_zero(self, H):
        """Tests that Hamiltonians can be added to zero"""
        assert H.compare(H + 0)
        assert H.compare(0 + H)
        assert H.compare(H + 0.0)
        assert H.compare(0.0 + H)
        assert H.compare(H + 0e1)
        assert H.compare(0e1 + H)

    @pytest.mark.parametrize(("coeff", "H", "res"), mul_hamiltonians)
    def test_hamiltonian_mul(self, coeff, H, res):
        """Tests that scalars and Hamiltonians are multiplied correctly"""
        assert res.compare(coeff * H)
        assert res.compare(H * coeff)

    def test_hamiltonian_mul_coeff_cast(self):
        """Test that the coefficients are correct when the type of the existing
        and the new coefficients differ."""
        h = 0.5 * (qml.PauliX(0) @ qml.PauliX(0) + qml.PauliY(0) @ qml.PauliY(1))
        assert np.all(h.coeffs == np.array([0.5, 0.5]))

    @pytest.mark.parametrize(("H1", "H2", "H"), sub_hamiltonians)
    def test_hamiltonian_sub(self, H1, H2, H):
        """Tests that Hamiltonians are subtracted correctly"""
        assert H.compare(H1 - H2)

    @pytest.mark.parametrize(("H1", "H2", "H"), matmul_hamiltonians)
    def test_hamiltonian_matmul(self, H1, H2, H):
        """Tests that Hamiltonians are tensored correctly"""
        assert H.compare(H1 @ H2)

    @pytest.mark.parametrize(("H1", "H2", "H"), rmatmul_hamiltonians)
    def test_hamiltonian_matmul(self, H1, H2, H):
        """Tests that Hamiltonians are tensored correctly when using __rmatmul__"""
        assert H.compare(H1.__rmatmul__(H2))

    def test_hamiltonian_same_wires(self):
        """Test if a ValueError is raised when multiplication between Hamiltonians acting on the
        same wires is attempted"""
        h1 = qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliZ(1)])

        with pytest.raises(ValueError, match="Hamiltonians can only be multiplied together if"):
            h1 @ h1

    @pytest.mark.parametrize(("H1", "H2", "H"), add_hamiltonians)
    def test_hamiltonian_iadd(self, H1, H2, H):
        """Tests that Hamiltonians are added inline correctly"""
        H1 += H2
        assert H.compare(H1)
        assert H.wires == H1.wires

    @pytest.mark.parametrize(("H1", "H2"), iadd_zero_hamiltonians)
    def test_hamiltonian_iadd_zero(self, H1, H2):
        """Tests in-place addition between Hamiltonians and zero"""
        H1 += 0
        assert H1.compare(H2)
        H1 += 0.0
        assert H1.compare(H2)
        H1 += 0e1
        assert H1.compare(H2)

    @pytest.mark.parametrize(("coeff", "H", "res"), mul_hamiltonians)
    def test_hamiltonian_imul(self, coeff, H, res):
        """Tests that scalars and Hamiltonians are multiplied inline correctly"""
        H *= coeff
        assert res.compare(H)

    @pytest.mark.parametrize(("H1", "H2", "H"), sub_hamiltonians)
    def test_hamiltonian_isub(self, H1, H2, H):
        """Tests that Hamiltonians are subtracted inline correctly"""
        H1 -= H2
        assert H.compare(H1)
        assert H.wires == H1.wires

    def test_arithmetic_errors(self):
        """Tests that the arithmetic operations thrown the correct errors"""
        H = qml.Hamiltonian([1], [qml.PauliZ(0)])
        A = [[1, 0], [0, -1]]
        with pytest.raises(ValueError, match="Cannot tensor product Hamiltonian"):
            H @ A
        with pytest.raises(ValueError, match="Cannot tensor product Hamiltonian"):
            H.__rmatmul__(A)
        with pytest.raises(ValueError, match="Cannot add Hamiltonian"):
            H + A
        with pytest.raises(ValueError, match="Cannot multiply Hamiltonian"):
            H * A
        with pytest.raises(ValueError, match="Cannot subtract"):
            H - A
        with pytest.raises(ValueError, match="Cannot add Hamiltonian"):
            H += A
        with pytest.raises(ValueError, match="Cannot multiply Hamiltonian"):
            H *= A
        with pytest.raises(ValueError, match="Cannot subtract"):
            H -= A

    def test_hamiltonian_queue_outside(self):
        """Tests that Hamiltonian are queued correctly when components are defined outside the recording context."""

        queue = [
            qml.Hadamard(wires=1),
            qml.PauliX(wires=0),
            qml.Hamiltonian(
                [1, 3, 1], [qml.PauliX(1), qml.PauliZ(0) @ qml.PauliZ(2), qml.PauliZ(1)]
            ),
        ]

        H = qml.PauliX(1) + 3 * qml.PauliZ(0) @ qml.PauliZ(2) + qml.PauliZ(1)

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=1)
            qml.PauliX(wires=0)
            qml.expval(H)

        assert len(tape.queue) == 3
        assert isinstance(tape.queue[0], qml.Hadamard)
        assert isinstance(tape.queue[1], qml.PauliX)
        assert isinstance(tape.queue[2], qml.measurements.MeasurementProcess)
        assert H.compare(tape.queue[2].obs)

    def test_hamiltonian_queue_inside(self):
        """Tests that Hamiltonian are queued correctly when components are instantiated inside the recording context."""

        queue = [
            qml.Hadamard(wires=1),
            qml.PauliX(wires=0),
            qml.PauliX(1),
            qml.PauliZ(0),
            qml.PauliZ(2),
            qml.PauliZ(0) @ qml.PauliZ(2),
            qml.PauliZ(1),
            qml.Hamiltonian(
                [1, 3, 1], [qml.PauliX(1), qml.PauliZ(0) @ qml.PauliZ(2), qml.PauliZ(1)]
            ),
        ]

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=1)
            qml.PauliX(wires=0)
            qml.expval(
                qml.Hamiltonian(
                    [1, 3, 1], [qml.PauliX(1), qml.PauliZ(0) @ qml.PauliZ(2), qml.PauliZ(1)]
                )
            )

        assert np.all([q1.compare(q2) for q1, q2 in zip(tape.queue, queue)])

    def test_terms(self):
        """Tests that the terms representation is returned correctly."""
        coeffs = pnp.array([1.0, 2.0], requires_grad=True)
        ops = [qml.PauliX(0), qml.PauliZ(1)]
        h = qml.Hamiltonian(coeffs, ops)
        c, o = h.terms()
        assert isinstance(c, Iterable)
        assert isinstance(o, list)
        assert all(isinstance(item, np.ndarray) for item in c)
        assert all(item.requires_grad for item in c)
        assert all(isinstance(item, qml.operation.Operator) for item in o)

    def test_hamiltonian_no_empty_wire_list_error(self):
        """Test that empty Hamiltonian does not raise an empty wire error."""
        hamiltonian = qml.Hamiltonian([], [])
        assert isinstance(hamiltonian, qml.Hamiltonian)

    def test_map_wires(self):
        """Test the map_wires method."""
        coeffs = pnp.array([1.0, 2.0, -3.0], requires_grad=True)
        ops = [qml.PauliX(0), qml.PauliZ(1), qml.PauliY(2)]
        h = qml.Hamiltonian(coeffs, ops)
        wire_map = {0: 10, 1: 11, 2: 12}
        mapped_h = h.map_wires(wire_map=wire_map)
        final_obs = [qml.PauliX(10), qml.PauliZ(11), qml.PauliY(12)]
        assert h is not mapped_h
        assert h.wires == Wires([0, 1, 2])
        assert mapped_h.wires == Wires([10, 11, 12])
        for obs1, obs2 in zip(mapped_h.ops, final_obs):
            assert qml.equal(obs1, obs2)
        for coeff1, coeff2 in zip(mapped_h.coeffs, h.coeffs):
            assert coeff1 == coeff2

    def test_hermitian_tensor_prod(self):
        """Test that the tensor product of a Hamiltonian with Hermitian observable works."""
        tensor = qml.PauliX(0) @ qml.PauliX(1)
        herm = qml.Hermitian([[1, 0], [0, 1]], wires=4)

        ham = qml.Hamiltonian([1.0, 1.0], [tensor, qml.PauliX(2)]) @ qml.Hamiltonian([1.0], [herm])
        assert isinstance(ham, qml.Hamiltonian)


class TestHamiltonianCoefficients:
    """Test the creation of a Hamiltonian"""

    @pytest.mark.parametrize("coeffs", [el[0] for el in COEFFS_PARAM_INTERFACE])
    def test_creation_different_coeff_types(self, coeffs):
        """Check that Hamiltonian's coefficients and data attributes are set correctly."""
        H = qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(0)])
        assert np.allclose(coeffs, H.coeffs)
        assert np.allclose([coeffs[i] for i in range(qml.math.shape(coeffs)[0])], H.data)

    @pytest.mark.parametrize("coeffs", [el[0] for el in COEFFS_PARAM_INTERFACE])
    def test_simplify(self, coeffs):
        """Test that simplify works with different coefficient types."""
        H1 = qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(1)])
        H2 = qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.Identity(0) @ qml.PauliZ(1)])
        H2.simplify()
        assert H1.compare(H2)
        assert H1.data == H2.data


class TestHamiltonianArithmeticTF:
    """Tests creation of Hamiltonians using arithmetic
    operations with TensorFlow tensor coefficients."""

    @pytest.mark.tf
    def test_hamiltonian_equal(self):
        """Tests equality"""
        import tensorflow as tf

        coeffs = tf.Variable([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = tf.Variable([-1.6, 0.5])
        obs2 = [qml.PauliY(1), qml.PauliX(0)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        assert H1.compare(H2)

    @pytest.mark.tf
    def test_hamiltonian_add(self):
        """Tests that Hamiltonians are added correctly"""
        import tensorflow as tf

        coeffs = tf.Variable([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = tf.Variable([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = tf.Variable([1.0, -2.0])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 + H2)

        H1 += H2
        assert H.compare(H1)

    @pytest.mark.tf
    def test_hamiltonian_sub(self):
        """Tests that Hamiltonians are subtracted correctly"""
        import tensorflow as tf

        coeffs = tf.Variable([1.0, -2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = tf.Variable([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = tf.Variable([0.5, -1.6])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 - H2)

        H1 -= H2
        assert H.compare(H1)

    @pytest.mark.tf
    def test_hamiltonian_matmul(self):
        """Tests that Hamiltonians are tensored correctly"""
        import tensorflow as tf

        coeffs = tf.Variable([1.0, 2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = tf.Variable([-1.0, -2.0])
        obs2 = [qml.PauliX(2), qml.PauliY(3)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        coeffs_expected = tf.Variable([-4.0, -2.0, -2.0, -1.0])
        obs_expected = [
            qml.PauliY(1) @ qml.PauliY(3),
            qml.PauliX(0) @ qml.PauliY(3),
            qml.PauliX(2) @ qml.PauliY(1),
            qml.PauliX(0) @ qml.PauliX(2),
        ]
        H = qml.Hamiltonian(coeffs_expected, obs_expected)

        assert H.compare(H1 @ H2)


class TestHamiltonianArithmeticTorch:
    """Tests creation of Hamiltonians using arithmetic
    operations with torch tensor coefficients."""

    @pytest.mark.torch
    def test_hamiltonian_equal(self):
        """Tests equality"""
        import torch

        coeffs = torch.tensor([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = torch.tensor([-1.6, 0.5])
        obs2 = [qml.PauliY(1), qml.PauliX(0)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        assert H1.compare(H2)

    @pytest.mark.torch
    def test_hamiltonian_add(self):
        """Tests that Hamiltonians are added correctly"""
        import torch

        coeffs = torch.tensor([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = torch.tensor([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = torch.tensor([1.0, -2.0])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 + H2)

        H1 += H2
        assert H.compare(H1)

    @pytest.mark.torch
    def test_hamiltonian_sub(self):
        """Tests that Hamiltonians are subtracted correctly"""
        import torch

        coeffs = torch.tensor([1.0, -2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = torch.tensor([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = torch.tensor([0.5, -1.6])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 - H2)

        H1 -= H2
        assert H.compare(H1)

    @pytest.mark.torch
    def test_hamiltonian_matmul(self):
        """Tests that Hamiltonians are tensored correctly"""
        import torch

        coeffs = torch.tensor([1.0, 2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = torch.tensor([-1.0, -2.0])
        obs2 = [qml.PauliX(2), qml.PauliY(3)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        coeffs_expected = torch.tensor([-4.0, -2.0, -2.0, -1.0])
        obs_expected = [
            qml.PauliY(1) @ qml.PauliY(3),
            qml.PauliX(0) @ qml.PauliY(3),
            qml.PauliX(2) @ qml.PauliY(1),
            qml.PauliX(0) @ qml.PauliX(2),
        ]
        H = qml.Hamiltonian(coeffs_expected, obs_expected)

        assert H.compare(H1 @ H2)


class TestHamiltonianArithmeticAutograd:
    """Tests creation of Hamiltonians using arithmetic
    operations with autograd tensor coefficients."""

    @pytest.mark.autograd
    def test_hamiltonian_equal(self):
        """Tests equality"""
        coeffs = pnp.array([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = pnp.array([-1.6, 0.5])
        obs2 = [qml.PauliY(1), qml.PauliX(0)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        assert H1.compare(H2)

    @pytest.mark.autograd
    def test_hamiltonian_add(self):
        """Tests that Hamiltonians are added correctly"""
        coeffs = pnp.array([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = pnp.array([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = pnp.array([1.0, -2.0])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 + H2)

        H1 += H2
        assert H.compare(H1)

    @pytest.mark.autograd
    def test_hamiltonian_sub(self):
        """Tests that Hamiltonians are subtracted correctly"""
        coeffs = pnp.array([1.0, -2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = pnp.array([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = pnp.array([0.5, -1.6])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 - H2)

        H1 -= H2
        assert H.compare(H1)

    @pytest.mark.autograd
    def test_hamiltonian_matmul(self):
        """Tests that Hamiltonians are tensored correctly"""
        coeffs = pnp.array([1.0, 2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = pnp.array([-1.0, -2.0])
        obs2 = [qml.PauliX(2), qml.PauliY(3)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        coeffs_expected = pnp.array([-4.0, -2.0, -2.0, -1.0])
        obs_expected = [
            qml.PauliY(1) @ qml.PauliY(3),
            qml.PauliX(0) @ qml.PauliY(3),
            qml.PauliX(2) @ qml.PauliY(1),
            qml.PauliX(0) @ qml.PauliX(2),
        ]
        H = qml.Hamiltonian(coeffs_expected, obs_expected)

        assert H.compare(H1 @ H2)


class TestHamiltonianArithmeticJax:
    """Tests creation of Hamiltonians using arithmetic
    operations with jax tensor coefficients."""

    @pytest.mark.jax
    def test_hamiltonian_equal(self):
        """Tests equality"""
        from jax import numpy as jnp

        coeffs = jnp.array([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = jnp.array([-1.6, 0.5])
        obs2 = [qml.PauliY(1), qml.PauliX(0)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        assert H1.compare(H2)

    @pytest.mark.jax
    def test_hamiltonian_add(self):
        """Tests that Hamiltonians are added correctly"""
        from jax import numpy as jnp

        coeffs = jnp.array([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = jnp.array([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = jnp.array([1.0, -2.0])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 + H2)

        H1 += H2
        assert H.compare(H1)

    @pytest.mark.jax
    def test_hamiltonian_sub(self):
        """Tests that Hamiltonians are subtracted correctly"""
        from jax import numpy as jnp

        coeffs = jnp.array([1.0, -2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = jnp.array([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = jnp.array([0.5, -1.6])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 - H2)

        H1 -= H2
        assert H.compare(H1)

    @pytest.mark.jax
    def test_hamiltonian_matmul(self):
        """Tests that Hamiltonians are tensored correctly"""
        from jax import numpy as jnp

        coeffs = jnp.array([1.0, 2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = jnp.array([-1.0, -2.0])
        obs2 = [qml.PauliX(2), qml.PauliY(3)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        coeffs_expected = jnp.array([-4.0, -2.0, -2.0, -1.0])
        obs_expected = [
            qml.PauliY(1) @ qml.PauliY(3),
            qml.PauliX(0) @ qml.PauliY(3),
            qml.PauliX(2) @ qml.PauliY(1),
            qml.PauliX(0) @ qml.PauliX(2),
        ]
        H = qml.Hamiltonian(coeffs_expected, obs_expected)

        assert H.compare(H1 @ H2)


class TestGrouping:
    """Tests for the grouping functionality"""

    def test_grouping_is_correct_kwarg(self):
        """Basic test checking that grouping with a kwarg works as expected"""
        a = qml.PauliX(0)
        b = qml.PauliX(1)
        c = qml.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        H = qml.Hamiltonian(coeffs, obs, grouping_type="qwc")
        assert H.grouping_indices == [[0, 1], [2]]

    def test_grouping_is_correct_compute_grouping(self):
        """Basic test checking that grouping with compute_grouping works as expected"""
        a = qml.PauliX(0)
        b = qml.PauliX(1)
        c = qml.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        H = qml.Hamiltonian(coeffs, obs, grouping_type="qwc")
        H.compute_grouping()
        assert H.grouping_indices == [[0, 1], [2]]

    def test_set_grouping(self):
        """Test that we can set grouping indices."""
        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliX(0), qml.PauliX(1), qml.PauliZ(0)])
        H.grouping_indices = [[0, 1], [2]]

        assert H.grouping_indices == [[0, 1], [2]]

    def test_set_grouping_error(self):
        """Test that grouping indices are validated."""
        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliX(0), qml.PauliX(1), qml.PauliZ(0)])

        with pytest.raises(ValueError, match="The grouped index value"):
            H.grouping_indices = [[3, 1], [2]]

        with pytest.raises(ValueError, match="The grouped index value"):
            H.grouping_indices = "a"

    def test_grouping_for_non_groupable_hamiltonians(self):
        """Test that grouping is computed correctly, even if no observables commute"""
        a = qml.PauliX(0)
        b = qml.PauliY(0)
        c = qml.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        H = qml.Hamiltonian(coeffs, obs, grouping_type="qwc")
        assert H.grouping_indices == [[0], [1], [2]]

    def test_grouping_is_reset_when_simplifying(self):
        """Tests that calling simplify() resets the grouping"""
        obs = [qml.PauliX(0), qml.PauliX(1), qml.PauliZ(0)]
        coeffs = [1.0, 2.0, 3.0]

        H = qml.Hamiltonian(coeffs, obs, grouping_type="qwc")
        assert H.grouping_indices is not None

        H.simplify()
        assert H.grouping_indices is None

    def test_grouping_does_not_alter_queue(self):
        """Tests that grouping is invisible to the queue."""
        a = qml.PauliX(0)
        b = qml.PauliX(1)
        c = qml.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        with qml.tape.QuantumTape() as tape:
            H = qml.Hamiltonian(coeffs, obs, grouping_type="qwc")

        assert tape.queue == [H]

    def test_grouping_method_can_be_set(self):
        r"""Tests that the grouping method can be controlled by kwargs.
        This is done by changing from default to 'rlf' and checking the result."""
        a = qml.PauliX(0)
        b = qml.PauliX(1)
        c = qml.PauliZ(0)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        # compute grouping during construction
        H2 = qml.Hamiltonian(coeffs, obs, grouping_type="qwc", method="lf")
        assert H2.grouping_indices == [[2, 1], [0]]

        # compute grouping separately
        H3 = qml.Hamiltonian(coeffs, obs, grouping_type=None)
        H3.compute_grouping(method="lf")
        assert H3.grouping_indices == [[2, 1], [0]]


class TestHamiltonianEvaluation:
    """Test the usage of a Hamiltonian as an observable"""

    @pytest.mark.parametrize("coeffs, param, interface", COEFFS_PARAM_INTERFACE)
    def test_vqe_forward_different_coeff_types(self, coeffs, param, interface):
        """Check that manually splitting a Hamiltonian expectation has the same
        result as passing the Hamiltonian as an observable"""
        dev = qml.device("default.qubit", wires=2)
        H = qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(0)])

        @qml.qnode(dev, interface=interface)
        def circuit():
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(H)

        @qml.qnode(dev, interface=interface)
        def circuit1():
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.PauliX(0))

        @qml.qnode(dev, interface=interface)
        def circuit2():
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit()
        res_expected = coeffs[0] * circuit1() + coeffs[1] * circuit2()
        assert np.isclose(res, res_expected)

    def test_simplify_reduces_tape_parameters(self):
        """Test that simplifying a Hamiltonian reduces the number of parameters on a tape"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RY(0.1, wires=0)
            return qml.expval(
                qml.Hamiltonian([1.0, 2.0], [qml.PauliX(1), qml.PauliX(1)], simplify=True)
            )

        circuit()
        pars = circuit.qtape.get_parameters(trainable_only=False)
        # simplify worked and added 1. and 2.
        assert pars == [0.1, 3.0]


class TestHamiltonianDifferentiation:
    """Test that the Hamiltonian coefficients are differentiable"""

    @pytest.mark.parametrize("simplify", [True, False])
    @pytest.mark.parametrize("group", [None, "qwc"])
    def test_trainable_coeffs_paramshift(self, simplify, group):
        """Test the parameter-shift method by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a Hamiltonian expectation"""
        coeffs = pnp.array([-0.05, 0.17], requires_grad=True)
        param = pnp.array(1.7, requires_grad=True)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.Hamiltonian(
                    coeffs,
                    [qml.PauliX(0), qml.PauliZ(0)],
                    simplify=simplify,
                    grouping_type=group,
                )
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
        coeffs = pnp.array([-0.05, 0.17], requires_grad=False)
        param = pnp.array(1.7, requires_grad=True)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.Hamiltonian(
                    coeffs,
                    [qml.PauliX(0), qml.PauliZ(0)],
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
        with the differentiation of a Hamiltonian expectation"""
        coeffs = pnp.array([-0.05, 0.17], requires_grad=True)
        param = pnp.array(1.7, requires_grad=True)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="autograd")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.Hamiltonian(
                    coeffs,
                    [qml.PauliX(0), qml.PauliZ(0)],
                    simplify=simplify,
                    grouping_type=group,
                )
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
            return qml.expval(qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(0)]))

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
        """Test the jax interface by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a Hamiltonian expectation"""

        import jax
        import jax.numpy as jnp

        coeffs = jnp.array([-0.05, 0.17])
        param = jnp.array(1.7)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.Hamiltonian(
                    coeffs,
                    [qml.PauliX(0), qml.PauliZ(0)],
                    simplify=simplify,
                    grouping_type=group,
                )
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

    @pytest.mark.jax
    def test_nontrainable_coeffs_jax(self):
        """Test the jax interface if the coefficients are explicitly set non-trainable"""

        import jax
        import jax.numpy as jnp

        coeffs = np.array([-0.05, 0.17])
        param = jnp.array(1.7)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(0)]))

        grad_fn = jax.grad(circuit, argnums=(1))
        grad = grad_fn(coeffs, param)

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)
        half1 = qml.QNode(circuit1, dev, interface="jax", diff_method="backprop")
        half2 = qml.QNode(circuit2, dev, interface="jax", diff_method="backprop")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        grad_fn_expected = jax.grad(combine, argnums=(1))
        grad_expected = grad_fn_expected(coeffs, param)

        assert np.allclose(grad, grad_expected)

    @pytest.mark.torch
    @pytest.mark.parametrize("simplify", [True, False])
    @pytest.mark.parametrize("group", [None, "qwc"])
    def test_trainable_coeffs_torch(self, simplify, group):
        """Test the torch interface by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a Hamiltonian expectation"""

        import torch

        coeffs = torch.tensor([-0.05, 0.17], requires_grad=True)
        param = torch.tensor(1.7, requires_grad=True)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.Hamiltonian(
                    coeffs,
                    [qml.PauliX(0), qml.PauliZ(0)],
                    simplify=simplify,
                    grouping_type=group,
                )
            )

        res = circuit(coeffs, param)
        res.backward()
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

        assert np.allclose(grad[0], grad_expected[0])
        assert np.allclose(grad[1], grad_expected[1])

    @pytest.mark.torch
    def test_nontrainable_coeffs_torch(self):
        """Test the torch interface if the coefficients are explicitly set non-trainable"""

        import torch

        coeffs = torch.tensor([-0.05, 0.17], requires_grad=False)
        param = torch.tensor(1.7, requires_grad=True)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.Hamiltonian(
                    coeffs,
                    [qml.PauliX(0), qml.PauliZ(0)],
                )
            )

        res = circuit(coeffs, param)
        res.backward()

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
        with the differentiation of a Hamiltonian expectation"""

        import tensorflow as tf

        coeffs = tf.Variable([-0.05, 0.17], dtype=tf.double)
        param = tf.Variable(1.7, dtype=tf.double)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.Hamiltonian(
                    coeffs,
                    [qml.PauliX(0), qml.PauliZ(0)],
                    simplify=simplify,
                    grouping_type=group,
                )
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

        import tensorflow as tf

        coeffs = tf.constant([-0.05, 0.17], dtype=tf.double)
        param = tf.Variable(1.7, dtype=tf.double)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.Hamiltonian(
                    coeffs,
                    [qml.PauliX(0), qml.PauliZ(0)],
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

    def test_not_supported_by_adjoint_differentiation(self):
        """Test that error is raised when attempting the adjoint differentiation method."""
        dev = qml.device("default.qubit", wires=2)

        coeffs = pnp.array([-0.05, 0.17], requires_grad=True)
        param = pnp.array(1.7, requires_grad=True)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.Hamiltonian(
                    coeffs,
                    [qml.PauliX(0), qml.PauliZ(0)],
                )
            )

        grad_fn = qml.grad(circuit)
        with pytest.raises(
            qml.QuantumFunctionError,
            match="Adjoint differentiation method does not support Hamiltonian observables",
        ):
            grad_fn(coeffs, param)
