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
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp

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

    COEFFS_PARAM_INTERFACE.append((torch.tensor([-0.05, 0.17]), torch.tensor([1.7]), "torch"))
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
]

mul_hamiltonians = [
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
        assert np.allclose(H.terms[0], coeffs)
        assert H.terms[1] == list(ops)

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

    @pytest.mark.parametrize("terms, string", zip(valid_hamiltonians, valid_hamiltonians_str))
    def test_hamiltonian_str(self, terms, string):
        """Tests that the __str__ function for printing is correct"""
        H = qml.Hamiltonian(*terms)
        assert H.__str__() == string

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

    @pytest.mark.parametrize(("coeff", "H", "res"), mul_hamiltonians)
    def test_hamiltonian_mul(self, coeff, H, res):
        """Tests that scalars and Hamiltonians are multiplied correctly"""
        assert res.compare(coeff * H)
        assert res.compare(H * coeff)

    @pytest.mark.parametrize(("H1", "H2", "H"), sub_hamiltonians)
    def test_hamiltonian_sub(self, H1, H2, H):
        """Tests that Hamiltonians are subtracted correctly"""
        assert H.compare(H1 - H2)

    @pytest.mark.parametrize(("H1", "H2", "H"), matmul_hamiltonians)
    def test_hamiltonian_matmul(self, H1, H2, H):
        """Tests that Hamiltonians are tensored correctly"""
        assert H.compare(H1 @ H2)

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

    def test_arithmetic_errors(self):
        """Tests that the arithmetic operations thrown the correct errors"""
        H = qml.Hamiltonian([1], [qml.PauliZ(0)])
        A = [[1, 0], [0, -1]]
        with pytest.raises(ValueError, match="Cannot tensor product Hamiltonian"):
            H @ A
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

    def test_hamiltonian_queue(self):
        """Tests that Hamiltonian are queued correctly"""

        # Outside of tape

        queue = [
            qml.Hadamard(wires=1),
            qml.PauliX(wires=0),
            qml.PauliZ(0),
            qml.PauliZ(2),
            qml.PauliZ(0) @ qml.PauliZ(2),
            qml.PauliX(1),
            qml.PauliZ(1),
            qml.Hamiltonian(
                [1, 3, 1], [qml.PauliX(1), qml.PauliZ(0) @ qml.PauliZ(2), qml.PauliZ(1)]
            ),
        ]

        H = qml.PauliX(1) + 3 * qml.PauliZ(0) @ qml.PauliZ(2) + qml.PauliZ(1)

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=1)
            qml.PauliX(wires=0)
            qml.expval(H)

        assert np.all([q1.compare(q2) for q1, q2 in zip(tape.queue, queue)])

        # Inside of tape

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

    def test_hamiltonian_equal(self):
        """Tests equality"""
        tf = pytest.importorskip("tensorflow")

        coeffs = tf.Variable([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = tf.Variable([-1.6, 0.5])
        obs2 = [qml.PauliY(1), qml.PauliX(0)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        assert H1.compare(H2)

    def test_hamiltonian_add(self):
        """Tests that Hamiltonians are added correctly"""
        tf = pytest.importorskip("tensorflow")

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

    def test_hamiltonian_sub(self):
        """Tests that Hamiltonians are subtracted correctly"""
        tf = pytest.importorskip("tensorflow")

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

    def test_hamiltonian_matmul(self):
        """Tests that Hamiltonians are tensored correctly"""
        tf = pytest.importorskip("tensorflow")

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

    def test_hamiltonian_equal(self):
        """Tests equality"""
        torch = pytest.importorskip("torch")

        coeffs = torch.tensor([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = torch.tensor([-1.6, 0.5])
        obs2 = [qml.PauliY(1), qml.PauliX(0)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        assert H1.compare(H2)

    def test_hamiltonian_add(self):
        """Tests that Hamiltonians are added correctly"""
        torch = pytest.importorskip("torch")

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

    def test_hamiltonian_sub(self):
        """Tests that Hamiltonians are subtracted correctly"""
        torch = pytest.importorskip("torch")

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

    def test_hamiltonian_matmul(self):
        """Tests that Hamiltonians are tensored correctly"""
        torch = pytest.importorskip("torch")

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

    def test_hamiltonian_equal(self):
        """Tests equality"""
        coeffs = pnp.array([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = pnp.array([-1.6, 0.5])
        obs2 = [qml.PauliY(1), qml.PauliX(0)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        assert H1.compare(H2)

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

    def test_hamiltonian_equal(self):
        """Tests equality"""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        coeffs = jnp.array([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = jnp.array([-1.6, 0.5])
        obs2 = [qml.PauliY(1), qml.PauliX(0)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        assert H1.compare(H2)

    def test_hamiltonian_add(self):
        """Tests that Hamiltonians are added correctly"""
        jax = pytest.importorskip("jax")
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

    def test_hamiltonian_sub(self):
        """Tests that Hamiltonians are subtracted correctly"""
        jax = pytest.importorskip("jax")
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

    def test_hamiltonian_matmul(self):
        """Tests that Hamiltonians are tensored correctly"""
        jax = pytest.importorskip("jax")
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

        assert tape.queue == [a, b, c, H]

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


@pytest.mark.parametrize("simplify", [True, False])
@pytest.mark.parametrize("group", [None, "qwc"])
class TestHamiltonianDifferentiation:
    """Test that the Hamiltonian coefficients are differentiable"""

    def test_vqe_differentiation_paramshift(self, simplify, group):
        """Test the parameter-shift method by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a Hamiltonian expectation"""
        coeffs = np.array([-0.05, 0.17])
        param = np.array(1.7)

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

    def test_vqe_differentiation_autograd(self, simplify, group):
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

    def test_vqe_differentiation_jax(self, simplify, group):
        """Test the jax interface by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a Hamiltonian expectation"""

        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
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

    def test_vqe_differentiation_torch(self, simplify, group):
        """Test the torch interface by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a Hamiltonian expectation"""

        torch = pytest.importorskip("torch")
        coeffs = torch.tensor([-0.05, 0.17], requires_grad=True)
        param = torch.tensor(1.7, requires_grad=True)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="torch")
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

        half1 = qml.QNode(circuit1, dev, interface="torch")
        half2 = qml.QNode(circuit2, dev, interface="torch")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        res_expected = combine(coeffs2, param2)
        res_expected.backward()
        grad_expected = (coeffs2.grad, param2.grad)

        assert np.allclose(grad[0], grad_expected[0])
        assert np.allclose(grad[1], grad_expected[1])

    def test_vqe_differentiation_tf(self, simplify, group):
        """Test the tf interface by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a Hamiltonian expectation"""

        tf = pytest.importorskip("tf")
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
