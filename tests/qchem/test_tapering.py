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
from lib2to3.pytree import convert

import pytest
import scipy

import pennylane as qml
from pennylane import numpy as np
from pennylane.qchem.tapering import (
    _binary_matrix,
    _kernel,
    _observable_mult,
    _reduced_row_echelon,
    clifford,
    optimal_sector,
    taper_hf,
    taper_operation,
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
    pauli_ops = qml.paulix_ops(generators, num_qubits)
    for p1, p2 in zip(pauli_ops, result):
        assert p1.compare(p2)


@pytest.mark.parametrize(
    ("symbols", "geometry", "num_qubits", "res_generators"),
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
        ),
    ],
)
def test_symmetry_generators(symbols, geometry, num_qubits, res_generators):
    r"""Test that symmetry_generators returns the correct result."""

    mol = qml.qchem.Molecule(symbols, geometry)
    hamiltonian = qml.qchem.diff_hamiltonian(mol)()
    generators = qml.symmetry_generators(hamiltonian)

    for g1, g2 in zip(generators, res_generators):
        assert g1.compare(g2)


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
    ("generator", "paulixops", "result"),
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
def test_cliford(generator, paulixops, result):
    r"""Test that clifford returns the correct operator."""
    u = clifford(generator, paulixops)
    assert u.compare(result)


@pytest.mark.parametrize(
    ("symbols", "geometry", "generator", "paulixops", "paulix_sector", "ham_ref"),
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
def test_transform_hamiltonian(symbols, geometry, generator, paulixops, paulix_sector, ham_ref):
    r"""Test that transform_hamiltonian returns the correct hamiltonian."""
    mol = qml.qchem.Molecule(symbols, geometry)
    h = qml.qchem.diff_hamiltonian(mol)()
    ham_calc = qml.taper(h, generator, paulixops, paulix_sector)

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
    mol = qml.qchem.Molecule(symbols, geometry, charge)
    hamiltonian = qml.qchem.diff_hamiltonian(mol)()

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
    mol = qml.qchem.Molecule(symbols, geometry)
    hamiltonian = qml.qchem.diff_hamiltonian(mol)()

    with pytest.raises(ValueError, match=msg_match):
        optimal_sector(hamiltonian, generators, num_electrons)


@pytest.mark.parametrize(
    ("generators", "paulixops", "paulix_sector", "num_electrons", "num_wires", "result"),
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
def test_transform_hf(generators, paulixops, paulix_sector, num_electrons, num_wires, result):
    r"""Test that transform_hf returns the correct result."""

    tapered_hf_state = taper_hf(
        generators,
        paulixops,
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
def test_taper_obs(symbols, geometry, charge):
    r"""Test that the expectation values of tapered observables with respect to the
    tapered Hartree-Fock state (:math:`\langle HF|obs|HF \rangle`) are consistent."""
    mol = qml.qchem.Molecule(symbols, geometry, charge)
    hamiltonian = qml.qchem.diff_hamiltonian(mol)(geometry)
    hf_state = np.where(np.arange(len(hamiltonian.wires)) < mol.n_electrons, 1, 0)
    generators = qml.symmetry_generators(hamiltonian)
    paulixops = qml.paulix_ops(generators, len(hamiltonian.wires))
    paulix_sector = optimal_sector(hamiltonian, generators, mol.n_electrons)
    hf_state_tapered = taper_hf(
        generators, paulixops, paulix_sector, mol.n_electrons, len(hamiltonian.wires)
    )

    # calculate the HF energy <\psi_{HF}| H |\psi_{HF}> for tapered and untapered Hamiltonian
    o = np.array([1, 0])
    l = np.array([0, 1])
    state = functools.reduce(lambda i, j: np.kron(i, j), [l if s else o for s in hf_state])
    state_tapered = functools.reduce(
        lambda i, j: np.kron(i, j), [l if s else o for s in hf_state_tapered]
    )

    observables = [
        hamiltonian,
        qml.qchem.particle_number(len(hamiltonian.wires)),
        qml.qchem.spin2(mol.n_electrons, len(hamiltonian.wires)),
        qml.qchem.spinz(len(hamiltonian.wires)),
    ]
    for observable in observables:
        tapered_obs = qml.taper(observable, generators, paulixops, paulix_sector)
        obs_val = (
            scipy.sparse.coo_matrix(state)
            @ qml.utils.sparse_hamiltonian(observable)
            @ scipy.sparse.coo_matrix(state).T
        ).toarray()
        obs_val_tapered = (
            scipy.sparse.coo_matrix(state_tapered)
            @ qml.utils.sparse_hamiltonian(tapered_obs)
            @ scipy.sparse.coo_matrix(state_tapered).T
        ).toarray()
        assert np.isclose(obs_val, obs_val_tapered)


@pytest.mark.parametrize(
    ("generators", "paulixops", "paulix_sector", "num_electrons", "num_wires", "num_commuting"),
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
            (0, 1),
        ),
        (
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(1) @ qml.PauliZ(3)]),
            ],
            [qml.PauliX(wires=[2]), qml.PauliX(wires=[3])],
            (-1, -1),
            2,
            4,
            (2, 1),
        ),
        (
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2) @ qml.PauliZ(4)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(1) @ qml.PauliZ(3) @ qml.PauliZ(5)]),
            ],
            [qml.PauliX(wires=[4]), qml.PauliX(wires=[5])],
            (-1, -1),
            2,
            6,
            (4, 4),
        ),
        (
            [
                qml.Hamiltonian(
                    [1.0], [qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(4) @ qml.PauliZ(5)]
                ),
                qml.Hamiltonian(
                    [1.0], [qml.PauliZ(0) @ qml.PauliZ(2) @ qml.PauliZ(4) @ qml.PauliZ(6)]
                ),
                qml.Hamiltonian(
                    [1.0], [qml.PauliZ(0) @ qml.PauliZ(3) @ qml.PauliZ(4) @ qml.PauliZ(7)]
                ),
            ],
            [qml.PauliX(wires=[5]), qml.PauliX(wires=[6]), qml.PauliX(wires=[7])],
            (1, 1, 1),
            4,
            8,
            (4, 10),
        ),
    ],
)
def test_taper_excitations(
    generators, paulixops, paulix_sector, num_electrons, num_wires, num_commuting
):
    r"""Test that the tapered excitation operators using :func:`~.taper_operation`
    are consistent with the tapered Hartree-Fock state."""

    hf_state = np.where(np.arange(num_wires) < num_electrons, 1, 0)
    hf_tapered = taper_hf(generators, paulixops, paulix_sector, num_electrons, num_wires)
    particle_num = qml.qchem.particle_number(num_wires)
    particle_num_tapered = qml.taper(particle_num, generators, paulixops, paulix_sector)

    o = np.array([1, 0])
    l = np.array([0, 1])
    state = functools.reduce(lambda i, j: np.kron(i, j), [l if s else o for s in hf_state])
    state_tapered = functools.reduce(
        lambda i, j: np.kron(i, j), [l if s else o for s in hf_tapered]
    )

    singles, doubles = qml.qchem.excitations(num_electrons, num_wires)
    exc_fnc = [qml.SingleExcitation, qml.DoubleExcitation]
    exc_obs, exc_tap = [[], []], [[], []]
    for idx, exc in enumerate([singles, doubles]):
        exc_obs[idx] = [exc_fnc[idx](1.0, wires=wire) for wire in exc]
        exc_tap[idx] = [
            taper_operation(op, generators, paulixops, paulix_sector, range(num_wires))
            for op in exc_obs[idx]
        ]
        exc_tap[idx] = [x for x in exc_tap[idx] if x]
        assert len(exc_tap[idx]) == num_commuting[idx]

    obs_all, obs_tap = exc_obs[0] + exc_obs[1], exc_tap[0] + exc_tap[1]
    for op_all, op_tap in zip(obs_all, obs_tap):
        if op_tap:

            excited_state = np.matmul(qml.matrix(op_all, wire_order=range(len(hf_state))), state)
            ob_tap_mat = functools.reduce(
                lambda i, j: np.matmul(i, j),
                [qml.matrix(op, wire_order=range(len(hf_tapered))) for op in op_tap],
            )
            excited_state_tapered = np.matmul(ob_tap_mat, state_tapered)

            # check if tapered excitation gate remains particle number conserving
            pnum_val = (
                scipy.sparse.coo_matrix(excited_state)
                @ qml.utils.sparse_hamiltonian(particle_num)
                @ scipy.sparse.coo_matrix(excited_state).T
            ).toarray()
            pnum_val_tapered = (
                scipy.sparse.coo_matrix(excited_state_tapered)
                @ qml.utils.sparse_hamiltonian(particle_num_tapered)
                @ scipy.sparse.coo_matrix(excited_state_tapered).T
            ).toarray()
            assert np.isclose(pnum_val, pnum_val_tapered)


@pytest.mark.parametrize(
    ("operation", "gen_op", "message_match"),
    [
        (qml.U2(1, 1, 2), None, "is not implemented, please provide it with 'gen_op' args"),
        (
            qml.U2(1, 1, 2),
            np.identity(16),
            "Generator for the operation needs to a qml.Hamiltonian",
        ),
        (
            qml.U2(1, 1, 2),
            qml.Hamiltonian([1], [qml.Identity(2)]),
            "doesn't seem to be the correct generator for the",
        ),
    ],
)
def test_inconsistent_taper_ops(operation, gen_op, message_match):
    r"""Test that an error is raised if a set of inconsistent arguments is input"""

    symbols, geometry, charge = (
        ["He", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4588684632]]),
        1,
    )
    mol = qml.qchem.Molecule(symbols, geometry, charge)
    hamiltonian = qml.qchem.diff_hamiltonian(mol)(geometry)

    generators = qml.symmetry_generators(hamiltonian)
    paulixops = qml.paulix_ops(generators, len(hamiltonian.wires))
    paulix_sector = optimal_sector(hamiltonian, generators, mol.n_electrons)
    wire_order = hamiltonian.wires

    with pytest.raises(Exception, match=message_match):
        taper_operation(operation, generators, paulixops, paulix_sector, wire_order, gen_op)


@pytest.mark.parametrize(
    ("operation", "gen_op"),
    [
        (qml.PauliX(1), qml.Hamiltonian((-np.pi / 2,), [qml.PauliX(wires=[1])])),
        (qml.PauliZ(3), qml.Hamiltonian((-np.pi / 2,), [qml.PauliZ(wires=[3])])),
        (
            qml.OrbitalRotation(1, wires=[0, 1, 2, 3]),
            qml.Hamiltonian(
                (0.25, -0.25, 0.25, -0.25),
                [
                    qml.PauliX(wires=[0]) @ qml.PauliY(wires=[2]),
                    qml.PauliY(wires=[0]) @ qml.PauliX(wires=[2]),
                    qml.PauliX(wires=[1]) @ qml.PauliY(wires=[3]),
                    qml.PauliY(wires=[1]) @ qml.PauliX(wires=[3]),
                ],
            ),
        ),
    ],
)
def test_consistent_taper_ops(operation, gen_op):
    r"""Test that operations are tapered consistently when their generators are provided manually and when they are constructed internally"""

    # HeH+ tapering data for [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4588684632]]
    generators = [
        qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
        qml.Hamiltonian([1.0], [qml.PauliZ(1) @ qml.PauliZ(3)]),
    ]
    paulixops = [qml.PauliX(wires=[2]), qml.PauliX(wires=[3])]
    paulix_sector = (-1, -1)
    wire_order = range(4)

    taper_op1 = taper_operation(operation, generators, paulixops, paulix_sector, wire_order, None)
    taper_op2 = taper_operation(operation, generators, paulixops, paulix_sector, wire_order, gen_op)
    assert np.all([qml.equal(op1, op2) for op1, op2 in zip(taper_op1, taper_op2)])

    tape1, tape2 = qml.tape.QuantumTape(), qml.tape.QuantumTape()
    with tape1:
        taper_operation(operation, generators, paulixops, paulix_sector, wire_order, None)
    with tape2:
        taper_operation(operation, generators, paulixops, paulix_sector, wire_order, gen_op)

    taper_circuit1 = [x for x in tape1.circuit if x.label() != "I"]
    taper_circuit2 = [x for x in tape2.circuit if x.label() != "I"]

    assert len(taper_op1) == len(taper_circuit1) and len(taper_op2) == len(taper_circuit2)
    assert np.all([qml.equal(op1, op2) for op1, op2 in zip(taper_circuit1, taper_op1)])
    assert np.all([qml.equal(op1, op2) for op1, op2 in zip(taper_circuit2, taper_op2)])


@pytest.mark.parametrize(
    ("symbols", "geometry", "charge", "fci_energy"),
    [
        (
            ["He", "H"],
            np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4588684632]],
                requires_grad=True,
            ),
            1,
            -2.85102402,
        ),
    ],
)
def test_tapered_vqe(symbols, geometry, charge, fci_energy):
    r"""Test that the vqe routines with tapered operation can converge to ground-state energy"""
    num_electrons, num_wires = 2, 4
    mol = qml.qchem.Molecule(symbols, geometry, charge)
    hamiltonian = qml.qchem.diff_hamiltonian(mol)(geometry)

    generators = qml.symmetry_generators(hamiltonian)
    paulixops = qml.paulix_ops(generators, len(hamiltonian.wires))
    paulix_sector = optimal_sector(hamiltonian, generators, mol.n_electrons)

    tapered_ham = qml.taper(hamiltonian, generators, paulixops, paulix_sector)
    hf_state_tapered = taper_hf(
        generators, paulixops, paulix_sector, mol.n_electrons, len(hamiltonian.wires)
    )

    singles, doubles = qml.qchem.excitations(num_electrons, num_wires)

    dev = qml.device("default.qubit", wires=tapered_ham.wires)

    @qml.qnode(dev)
    def circuit(params):
        qml.BasisState(hf_state_tapered, wires=tapered_ham.wires)
        for idx, double in enumerate(doubles):
            qml.qchem.taper_operation(
                qml.DoubleExcitation(params[idx], wires=double),
                generators,
                paulixops,
                paulix_sector,
                range(num_wires),
            )
        for idx, single in enumerate(singles):
            qml.qchem.taper_operation(
                qml.SingleExcitation(params[len(doubles) + idx], wires=single),
                generators,
                paulixops,
                paulix_sector,
                range(num_wires),
            )
        return qml.expval(tapered_ham)

    optimizer = qml.GradientDescentOptimizer(stepsize=0.5)
    params = np.zeros(len(doubles) + len(singles), requires_grad=True)

    converge = False
    for _ in range(50):
        params, vqe_energy = optimizer.step_and_cost(circuit, params)
        if np.abs(fci_energy - vqe_energy) <= 1.6e-3:
            converge = True
    assert converge
