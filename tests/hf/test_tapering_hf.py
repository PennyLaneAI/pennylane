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
Unit tests for functions needed for qubit tapering.
"""
import functools

import pennylane as qml
import pytest
import scipy
from pennylane import numpy as np
from pennylane.hf.tapering import (
    _binary_matrix,
    _kernel,
    _observable_mult,
    _reduced_row_echelon,
    clifford,
    generate_paulis,
    generate_symmetries,
    get_generators,
    optimal_sector,
    transform_hamiltonian,
    transform_hf,
)


@pytest.mark.parametrize(
    ("terms", "num_qubits", "result"),
    [
        (
            [
                qml.Identity(wires=[0]),
                qml.PauliZ(wires=[0]),
                qml.PauliZ(wires=[1]),
                qml.PauliZ(wires=[2]),
                qml.PauliZ(wires=[3]),
                qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]),
                qml.PauliY(wires=[0])
                @ qml.PauliX(wires=[1])
                @ qml.PauliX(wires=[2])
                @ qml.PauliY(wires=[3]),
                qml.PauliY(wires=[0])
                @ qml.PauliY(wires=[1])
                @ qml.PauliX(wires=[2])
                @ qml.PauliX(wires=[3]),
                qml.PauliX(wires=[0])
                @ qml.PauliX(wires=[1])
                @ qml.PauliY(wires=[2])
                @ qml.PauliY(wires=[3]),
                qml.PauliX(wires=[0])
                @ qml.PauliY(wires=[1])
                @ qml.PauliY(wires=[2])
                @ qml.PauliX(wires=[3]),
                qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[2]),
                qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[3]),
                qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
                qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[3]),
                qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[3]),
            ],
            4,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 0, 1, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            [
                qml.PauliZ(wires=["a"]) @ qml.PauliX(wires=["b"]),
                qml.PauliZ(wires=["a"]) @ qml.PauliY(wires=["c"]),
                qml.PauliX(wires=["a"]) @ qml.PauliY(wires=["d"]),
            ],
            4,
            np.array(
                [[1, 0, 0, 0, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 1, 1, 0, 0, 1]]
            ),
        ),
    ],
)
def test_binary_matrix(terms, num_qubits, result):
    r"""Test that _binary_matrix returns the correct result."""
    binary_matrix = _binary_matrix(terms, num_qubits)
    assert (binary_matrix == result).all()


@pytest.mark.parametrize(
    ("binary_matrix", "result"),
    [
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 0, 1, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
    ],
)
def test_reduced_row_echelon(binary_matrix, result):
    r"""Test that _reduced_row_echelon returns the correct result."""

    # build row echelon form of the matrix
    shape = binary_matrix.shape
    for irow in range(shape[0]):
        pivot_index = 0
        if np.count_nonzero(binary_matrix[irow, :]):
            pivot_index = np.nonzero(binary_matrix[irow, :])[0][0]

        for jrow in range(shape[0]):
            if jrow != irow and binary_matrix[jrow, pivot_index]:
                binary_matrix[jrow, :] = (binary_matrix[jrow, :] + binary_matrix[irow, :]) % 2

    indices = [
        irow
        for irow in range(shape[0] - 1)
        if np.array_equal(binary_matrix[irow, :], np.zeros(shape[1]))
    ]

    temp_row_echelon_matrix = binary_matrix.copy()
    for row in indices[::-1]:
        temp_row_echelon_matrix = np.delete(temp_row_echelon_matrix, row, axis=0)

    row_echelon_matrix = np.zeros(shape, dtype=int)
    row_echelon_matrix[: shape[0] - len(indices), :] = temp_row_echelon_matrix

    # build reduced row echelon form of the matrix from row echelon form
    for idx in range(len(row_echelon_matrix))[:0:-1]:
        nonzeros = np.nonzero(row_echelon_matrix[idx])[0]
        if len(nonzeros) > 0:
            redrow = (row_echelon_matrix[idx, :] % 2).reshape(1, -1)
            coeffs = (
                (-row_echelon_matrix[:idx, nonzeros[0]] / row_echelon_matrix[idx, nonzeros[0]]) % 2
            ).reshape(1, -1)
            row_echelon_matrix[:idx, :] = (
                row_echelon_matrix[:idx, :] + (coeffs.T * redrow) % 2
            ) % 2

    # get reduced row echelon form from the _reduced_row_echelon function
    rref_bin_mat = _reduced_row_echelon(binary_matrix)

    assert (rref_bin_mat == row_echelon_matrix).all()
    assert (rref_bin_mat == result).all()


@pytest.mark.parametrize(
    ("binary_matrix", "result"),
    [
        (
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [[0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 1]]
            ),
        ),
    ],
)
def test_kernel(binary_matrix, result):
    r"""Test that _kernel returns the correct result."""

    # get the kernel from the gaussian elimination.
    pivots = (binary_matrix.T != 0).argmax(axis=0)
    nonpivots = np.setdiff1d(range(len(binary_matrix[0])), pivots)

    kernel = []
    for col in nonpivots:
        col_vector = binary_matrix[:, col]
        null_vector = np.zeros((binary_matrix.shape[1]), dtype=int)
        null_vector[col] = 1
        for i in pivots:
            first_entry = np.where(binary_matrix[:, i] == 1)[0][0]
            if col_vector[first_entry] == 1:
                null_vector[i] = 1
        kernel.append(null_vector.tolist())

    # get the nullspace from the _kernel function.
    nullspace = _kernel(binary_matrix)

    for nullvec in kernel:
        assert nullvec in nullspace.tolist()

    assert (nullspace == result).all()


@pytest.mark.parametrize(
    ("nullspace", "num_qubits", "result"),
    [
        (
            np.array(
                [[0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 1]]
            ),
            4,
            [  # Correct generators as given by Bravyi et al. (arXiv:1701.08213).
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(3)]),
            ],
        ),
    ],
)
def test_get_generators(nullspace, num_qubits, result):
    r"""Test that get_generators returns the correct result."""

    generators = get_generators(nullspace, num_qubits)
    for g1, g2 in zip(generators, result):
        assert g1.compare(g2)


@pytest.mark.parametrize(
    ("generators", "num_qubits", "result"),
    [
        (
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(3)]),
            ],
            4,
            [qml.PauliX(wires=[1]), qml.PauliX(wires=[2]), qml.PauliX(wires=[3])],
        ),
    ],
)
def test_generate_paulis(generators, num_qubits, result):
    r"""Test that generate_paulis returns the correct result."""
    pauli_ops = generate_paulis(generators, num_qubits)
    for p1, p2 in zip(pauli_ops, result):
        assert p1.compare(p2)


@pytest.mark.parametrize(
    ("symbols", "geometry", "num_qubits", "res_generators", "res_pauli_ops"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.40104295]], requires_grad=False),
            4,
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(3)]),
            ],
            [qml.PauliX(wires=[1]), qml.PauliX(wires=[2]), qml.PauliX(wires=[3])],
        ),
    ],
)
def test_generate_symmetries(symbols, geometry, num_qubits, res_generators, res_pauli_ops):
    r"""Test that generate_symmetries returns the correct result."""

    mol = qml.hf.Molecule(symbols, geometry)
    hamiltonian = qml.hf.generate_hamiltonian(mol)()
    generators, pauli_ops = generate_symmetries(hamiltonian, num_qubits)

    for g1, g2 in zip(generators, res_generators):
        assert g1.compare(g2)

    for p1, p2 in zip(pauli_ops, res_pauli_ops):
        assert p1.compare(p2)


@pytest.mark.parametrize(
    ("obs_a", "obs_b", "result"),
    [
        (
            qml.Hamiltonian(np.array([-1.0]), [qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliX(2)]),
            qml.Hamiltonian(np.array([-1.0]), [qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliX(2)]),
            qml.Hamiltonian(np.array([1.0]), [qml.Identity(0)]),
        ),
        (
            qml.Hamiltonian(
                np.array([0.5, 0.5]), [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliZ(1)]
            ),
            qml.Hamiltonian(
                np.array([0.5, 0.5]), [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0) @ qml.PauliZ(1)]
            ),
            qml.Hamiltonian(
                np.array([-0.25j, 0.25j, -0.25j, 0.25]),
                [qml.PauliY(0), qml.PauliY(1), qml.PauliZ(1), qml.PauliY(0) @ qml.PauliX(1)],
            ),
        ),
    ],
)
def test_observable_mult(obs_a, obs_b, result):
    r"""Test that observable_mult returns the correct result."""
    o = _observable_mult(obs_a, obs_b)
    assert o.compare(result)


@pytest.mark.parametrize(
    ("generator", "paulix_ops", "result"),
    [
        (
            [
                qml.Hamiltonian(np.array([1.0]), [qml.PauliZ(0)]),
                qml.Hamiltonian(np.array([1.0]), [qml.PauliZ(1)]),
            ],
            [qml.PauliX(0), qml.PauliX(1)],
            qml.Hamiltonian(
                np.array(
                    [
                        (1 / np.sqrt(2)) ** 2,
                        (1 / np.sqrt(2)) ** 2,
                        (1 / np.sqrt(2)) ** 2,
                        (1 / np.sqrt(2)) ** 2,
                    ]
                ),
                [
                    qml.PauliZ(0) @ qml.PauliZ(1),
                    qml.PauliZ(0) @ qml.PauliX(1),
                    qml.PauliX(0) @ qml.PauliZ(1),
                    qml.PauliX(0) @ qml.PauliX(1),
                ],
            ),
        ),
    ],
)
def test_cliford(generator, paulix_ops, result):
    r"""Test that clifford returns the correct operator."""
    u = clifford(generator, paulix_ops)
    assert u.compare(result)


@pytest.mark.parametrize(
    ("symbols", "geometry", "generator", "paulix_ops", "paulix_sector", "ham_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, -0.69440367], [0.0, 0.0, 0.69440367]], requires_grad=False),
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(3)]),
            ],
            [qml.PauliX(1), qml.PauliX(2), qml.PauliX(3)],
            [1, -1, -1],
            qml.Hamiltonian(
                np.array([-0.3210344, 0.18092703, 0.79596785]),
                [qml.Identity(0), qml.PauliX(0), qml.PauliZ(0)],
            ),
        ),
    ],
)
def test_transform_hamiltonian(symbols, geometry, generator, paulix_ops, paulix_sector, ham_ref):
    r"""Test that transform_hamiltonian returns the correct hamiltonian."""
    mol = qml.hf.Molecule(symbols, geometry)
    h = qml.hf.generate_hamiltonian(mol)()
    ham_calc = transform_hamiltonian(h, generator, paulix_ops, paulix_sector)

    # sort Hamiltonian terms and then compare with reference
    sorted_terms = list(sorted(zip(ham_calc.terms()[0], ham_calc.terms()[1])))
    for i, term in enumerate(sorted_terms):
        assert np.allclose(term[0], ham_ref.terms()[0][i])
        assert term[1].compare(ham_ref.terms()[1][i])


@pytest.mark.parametrize(
    ("symbols", "geometry", "charge", "generators", "num_electrons", "result"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.40104295]], requires_grad=False),
            0,
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(3)]),
            ],
            2,
            [1, -1, -1],
        ),
        (
            ["He", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4588684632]], requires_grad=False),
            1,
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(1) @ qml.PauliZ(3)]),
            ],
            1,
            [-1, 1],
        ),
        (
            ["H", "H", "H"],
            np.array(
                [[-0.84586466, 0.0, 0.0], [0.84586466, 0.0, 0.0], [0.0, 1.46508057, 0.0]],
                requires_grad=False,
            ),
            1,
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2) @ qml.PauliZ(4)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(1) @ qml.PauliZ(3) @ qml.PauliZ(5)]),
            ],
            2,
            [-1, -1],
        ),
    ],
)
def test_optimal_sector(symbols, geometry, charge, generators, num_electrons, result):
    r"""Test that find_optimal_sector returns the correct result."""
    mol = qml.hf.Molecule(symbols, geometry, charge)
    hamiltonian = qml.hf.generate_hamiltonian(mol)()

    perm = optimal_sector(hamiltonian, generators, num_electrons)

    assert perm == result


@pytest.mark.parametrize(
    ("symbols", "geometry", "generators"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.40104295]], requires_grad=False),
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(3)]),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ("num_electrons", "msg_match"),
    [
        (0, f"The number of active electrons must be greater than zero"),
        (5, f"Number of active orbitals cannot be smaller than number of active electrons"),
    ],
)
def test_exceptions_optimal_sector(symbols, geometry, generators, num_electrons, msg_match):
    r"""Test that find_optimal_sector returns the correct result."""
    mol = qml.hf.Molecule(symbols, geometry)
    hamiltonian = qml.hf.generate_hamiltonian(mol)()

    with pytest.raises(ValueError, match=msg_match):
        optimal_sector(hamiltonian, generators, num_electrons)


@pytest.mark.parametrize(
    ("generators", "paulix_ops", "paulix_sector", "num_electrons", "num_wires", "result"),
    [
        (
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(3)]),
            ],
            [qml.PauliX(wires=[1]), qml.PauliX(wires=[2]), qml.PauliX(wires=[3])],
            (1, -1, -1),
            2,
            4,
            np.array([1]),
        ),
        (
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2) @ qml.PauliZ(4)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(1) @ qml.PauliZ(3) @ qml.PauliZ(5)]),
            ],
            [qml.PauliX(wires=[2]), qml.PauliX(wires=[3])],
            (-1, -1),
            2,
            6,
            np.array([1, 1, 0, 0]),
        ),
        (
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(6) @ qml.PauliZ(7)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(8) @ qml.PauliZ(9)]),
                qml.Hamiltonian(
                    [1.0],
                    [
                        qml.PauliZ(wires=[0])
                        @ qml.PauliZ(wires=[2])
                        @ qml.PauliZ(wires=[4])
                        @ qml.PauliZ(wires=[6])
                        @ qml.PauliZ(wires=[8])
                        @ qml.PauliZ(wires=[10])
                    ],
                ),
                qml.Hamiltonian(
                    [1.0],
                    [
                        qml.PauliZ(wires=[1])
                        @ qml.PauliZ(wires=[3])
                        @ qml.PauliZ(wires=[5])
                        @ qml.PauliZ(wires=[6])
                        @ qml.PauliZ(wires=[8])
                        @ qml.PauliZ(wires=[11])
                    ],
                ),
            ],
            [
                qml.PauliX(wires=[7]),
                qml.PauliX(wires=[9]),
                qml.PauliX(wires=[0]),
                qml.PauliX(wires=[1]),
            ],
            (1, 1, 1, 1),
            4,
            12,
            np.array([1, 1, 0, 0, 0, 0, 0, 0]),
        ),
    ],
)
def test_transform_hf(generators, paulix_ops, paulix_sector, num_electrons, num_wires, result):
    r"""Test that transform_hf returns the correct result."""

    tapered_hf_state = transform_hf(
        generators,
        paulix_ops,
        paulix_sector,
        num_electrons,
        num_wires,
    )
    assert np.all(tapered_hf_state == result)


@pytest.mark.parametrize(
    ("symbols", "geometry", "charge"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.40104295]], requires_grad=True),
            0,
        ),
        (
            ["He", "H"],
            np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4588684632]],
                requires_grad=True,
            ),
            1,
        ),
        (
            ["H", "H", "H"],
            np.array(
                [[-0.84586466, 0.0, 0.0], [0.84586466, 0.0, 0.0], [0.0, 1.46508057, 0.0]],
                requires_grad=True,
            ),
            1,
        ),
        (
            ["H", "H", "H", "H"],
            np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
                requires_grad=True,
            ),
            0,
        ),
    ],
)
def test_hf_energy(symbols, geometry, charge):
    r"""Test that HF energy obtained from the tapered Hamiltonian and tapered Hartree Fock state is consistent."""
    mol = qml.hf.Molecule(symbols, geometry, charge)
    hamiltonian = qml.hf.generate_hamiltonian(mol)(geometry)
    hf_state = np.where(np.arange(len(hamiltonian.wires)) < mol.n_electrons, 1, 0)
    generators, paulix_ops = generate_symmetries(hamiltonian, len(hamiltonian.wires))
    paulix_sector = optimal_sector(hamiltonian, generators, mol.n_electrons)

    hamiltonian_tapered = transform_hamiltonian(hamiltonian, generators, paulix_ops, paulix_sector)
    hf_state_tapered = transform_hf(
        generators, paulix_ops, paulix_sector, mol.n_electrons, len(hamiltonian.wires)
    )

    # calculate the HF energy <\psi_{HF}| H |\psi_{HF}> for tapered and untapered Hamiltonian
    o = np.array([1, 0])
    l = np.array([0, 1])
    state = functools.reduce(lambda i, j: np.kron(i, j), [l if s else o for s in hf_state])
    state_tapered = functools.reduce(
        lambda i, j: np.kron(i, j), [l if s else o for s in hf_state_tapered]
    )

    energy = (
        scipy.sparse.coo_matrix(state)
        @ qml.utils.sparse_hamiltonian(hamiltonian)
        @ scipy.sparse.coo_matrix(state).T
    ).toarray()
    energy_tapered = (
        scipy.sparse.coo_matrix(state_tapered)
        @ qml.utils.sparse_hamiltonian(hamiltonian_tapered)
        @ scipy.sparse.coo_matrix(state_tapered).T
    ).toarray()

    assert np.isclose(energy, energy_tapered)
