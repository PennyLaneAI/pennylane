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
# pylint: disable=too-many-arguments
import functools

import pytest
import scipy

import pennylane as qml
from pennylane import numpy as np
from pennylane.math.utils import binary_finite_reduced_row_echelon
from pennylane.pauli import pauli_sentence
from pennylane.qchem.tapering import (
    _kernel,
    _split_pauli_sentence,
    _taper_pauli_sentence,
    clifford,
    optimal_sector,
    taper_hf,
    taper_operation,
)


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
    rref_bin_mat = binary_finite_reduced_row_echelon(binary_matrix)

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
        qml.assert_equal(p1, p2)

    # test arithmetic op compatibility:
    generators_as_ops = [pauli_sentence(g).operation() for g in generators]
    assert not any(isinstance(g, qml.Hamiltonian) for g in generators_as_ops)

    for p1, p2 in zip(pauli_ops, result):
        qml.assert_equal(p1, p2)


@pytest.mark.parametrize(
    ("symbols", "geometry", "res_generators"),
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
def test_symmetry_generators(symbols, geometry, res_generators):
    r"""Test that symmetry_generators returns the correct result."""

    mol = qml.qchem.Molecule(symbols, geometry)
    hamiltonian = qml.qchem.diff_hamiltonian(mol)()
    generators = qml.symmetry_generators(hamiltonian)
    for g1, g2 in zip(generators, res_generators):
        assert pauli_sentence(g1) == pauli_sentence(g2)


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
def test_clifford(generator, paulixops, result):
    r"""Test that clifford returns the correct operator."""
    u = clifford(generator, paulixops)
    assert pauli_sentence(u) == pauli_sentence(result)

    # test arithmetic op compatibility:
    result_as_op = pauli_sentence(result).operation()

    assert pauli_sentence(result_as_op) == pauli_sentence(u)


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
    hamref_terms = list(zip(*ham_ref.terms()))

    for term, ref_term in zip(sorted_terms, hamref_terms):
        assert np.allclose(term[0], ref_term[0])
        assert pauli_sentence(term[1]) == pauli_sentence(ref_term[1])


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
        (0, "The number of active electrons must be greater than zero"),
        (5, "Number of active orbitals cannot be smaller than number of active electrons"),
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
    state = functools.reduce(np.kron, [l if s else o for s in hf_state])
    state_tapered = functools.reduce(np.kron, [l if s else o for s in hf_state_tapered])

    observables = [
        hamiltonian,
        qml.qchem.particle_number(len(hamiltonian.wires)),
        qml.qchem.spin2(mol.n_electrons, len(hamiltonian.wires)),
        qml.qchem.spinz(len(hamiltonian.wires)),
    ]
    for observable in observables:

        obs_ps = qml.pauli.pauli_sentence(observable)
        tapered_obs = qml.taper(observable, generators, paulixops, paulix_sector)
        tapered_ps = _taper_pauli_sentence(obs_ps, generators, paulixops, paulix_sector)

        obs_val = (
            scipy.sparse.coo_matrix(state)
            @ observable.sparse_matrix(wire_order=range(len(observable.wires)))
            @ scipy.sparse.coo_matrix(state).T
        ).toarray()
        obs_val_tapered = (
            scipy.sparse.coo_matrix(state_tapered)
            @ tapered_obs.sparse_matrix(wire_order=range(len(tapered_obs.wires)))
            @ scipy.sparse.coo_matrix(state_tapered).T
        ).toarray()

        assert np.isclose(obs_val, obs_val_tapered)
        qml.assert_equal(tapered_obs, tapered_ps)


@pytest.mark.parametrize(
    ("symbols", "geometry", "charge", "generators", "paulixops", "paulix_sector", "num_commuting"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.40104295]], requires_grad=True),
            0,
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(3)]),
            ],
            [qml.PauliX(wires=[1]), qml.PauliX(wires=[2]), qml.PauliX(wires=[3])],
            (1, -1, -1),
            (0, 1),
        ),
        (
            ["He", "H"],
            np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4588684632]],
                requires_grad=True,
            ),
            1,
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(1) @ qml.PauliZ(3)]),
            ],
            [qml.PauliX(wires=[2]), qml.PauliX(wires=[3])],
            (-1, -1),
            (2, 1),
        ),
        (
            ["H", "H", "H"],
            np.array(
                [[-0.84586466, 0.0, 0.0], [0.84586466, 0.0, 0.0], [0.0, 1.46508057, 0.0]],
                requires_grad=True,
            ),
            1,
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2) @ qml.PauliZ(4)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(1) @ qml.PauliZ(3) @ qml.PauliZ(5)]),
            ],
            [qml.PauliX(wires=[4]), qml.PauliX(wires=[5])],
            (-1, -1),
            (4, 4),
        ),
    ],
)
def test_taper_excitations(
    symbols, geometry, charge, generators, paulixops, paulix_sector, num_commuting
):
    r"""Test that the tapered excitation operators using :func:`~.taper_operation`
    are consistent with the tapered Hartree-Fock state."""

    mol = qml.qchem.Molecule(symbols, geometry, charge)
    num_electrons, num_wires = mol.n_electrons, 2 * mol.n_orbitals
    hf_state = np.where(np.arange(num_wires) < num_electrons, 1, 0)
    hf_tapered = taper_hf(generators, paulixops, paulix_sector, num_electrons, num_wires)

    observables = [
        qml.qchem.diff_hamiltonian(mol)(geometry),
        qml.qchem.particle_number(num_wires),
        qml.qchem.spin2(num_electrons, num_wires),
        qml.qchem.spinz(num_wires),
    ]

    tapered_obs = [
        qml.taper(observale, generators, paulixops, paulix_sector) for observale in observables
    ]

    o = np.array([1, 0])
    l = np.array([0, 1])
    state = functools.reduce(np.kron, [l if s else o for s in hf_state])
    state_tapered = functools.reduce(np.kron, [l if s else o for s in hf_tapered])

    singles, doubles = qml.qchem.excitations(num_electrons, num_wires)
    exc_fnc = [qml.SingleExcitation, qml.DoubleExcitation]
    exc_obs, exc_tap = [[], []], [[], []]
    for idx, exc in enumerate([singles, doubles]):
        exc_obs[idx] = [exc_fnc[idx](np.pi, wires=wire) for wire in exc]
        exc_tap[idx] = [
            taper_operation(op, generators, paulixops, paulix_sector, range(num_wires))
            for op in exc_obs[idx]
        ]
        exc_obs[idx] = [x for i, x in enumerate(exc_obs[idx]) if exc_tap[idx][i]]
        exc_tap[idx] = [x for x in exc_tap[idx] if x]
        assert len(exc_tap[idx]) == num_commuting[idx]

    obs_all, obs_tap = exc_obs[0] + exc_obs[1], exc_tap[0] + exc_tap[1]
    for op_all, op_tap in zip(obs_all, obs_tap):
        if op_tap:
            excited_state = np.matmul(qml.matrix(op_all, wire_order=range(len(hf_state))), state)
            ob_tap_mat = functools.reduce(
                np.matmul,
                [qml.matrix(op, wire_order=range(len(hf_tapered))) for op in op_tap],
            )
            excited_state_tapered = np.matmul(ob_tap_mat, state_tapered)
            # check if tapered excitation gates remains spin and particle-number conserving,
            # and also evolves the tapered-state to have consistent energy values
            for obs, tap_obs in zip(observables, tapered_obs):
                expec_val = (
                    scipy.sparse.coo_matrix(excited_state)
                    @ obs.sparse_matrix(wire_order=range(len(hf_state)))
                    @ scipy.sparse.coo_matrix(excited_state).getH()
                ).toarray()
                expec_val_tapered = (
                    scipy.sparse.coo_matrix(excited_state_tapered)
                    @ tap_obs.sparse_matrix(wire_order=range(len(hf_tapered)))
                    @ scipy.sparse.coo_matrix(excited_state_tapered).getH()
                ).toarray()
                assert np.isclose(expec_val, expec_val_tapered)


@pytest.mark.parametrize(
    ("operation", "op_gen", "message_match"),
    [
        (qml.U2(1, 1, 2), None, "is not implemented, please provide it with 'op_gen' args"),
        (
            qml.U2(1, 1, 2),
            np.identity(16),
            "Generator for the operation needs to be a valid operator",
        ),
        (
            qml.U2(1, 1, 2),
            qml.Hamiltonian([1], [qml.Identity(2)]),
            "doesn't seem to be the correct generator for the",
        ),
    ],
)
def test_inconsistent_taper_ops(operation, op_gen, message_match):
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
        taper_operation(operation, generators, paulixops, paulix_sector, wire_order, op_gen=op_gen)


@pytest.mark.parametrize(
    ("operation", "op_gen"),
    [
        (qml.PauliX(1), qml.Hamiltonian((np.pi / 2,), [qml.PauliX(wires=[1])])),
        (qml.PauliY(2), qml.Hamiltonian((np.pi / 2,), [qml.PauliY(wires=[2])])),
        (qml.PauliZ(3), qml.Hamiltonian((np.pi / 2,), [qml.PauliZ(wires=[3])])),
        (
            qml.OrbitalRotation(np.pi, wires=[0, 1, 2, 3]),
            qml.Hamiltonian(
                (0.25, -0.25, 0.25, -0.25),
                [
                    qml.PauliX(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliY(wires=[2]),
                    qml.PauliY(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliX(wires=[2]),
                    qml.PauliX(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliY(wires=[3]),
                    qml.PauliY(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliX(wires=[3]),
                ],
            ),
        ),
        (
            qml.FermionicSWAP(np.pi, wires=[0, 1]),
            qml.Hamiltonian(
                (0.5, -0.25, -0.25, -0.25, -0.25),
                [
                    qml.Identity(wires=[0]) @ qml.Identity(wires=[1]),
                    qml.Identity(wires=[0]) @ qml.PauliZ(wires=[1]),
                    qml.PauliZ(wires=[0]) @ qml.Identity(wires=[1]),
                    qml.PauliX(wires=[0]) @ qml.PauliX(wires=[1]),
                    qml.PauliY(wires=[0]) @ qml.PauliY(wires=[1]),
                ],
            ),
        ),
    ],
)
def test_consistent_taper_ops(operation, op_gen):
    r"""Test that operations are tapered consistently when their generators are provided manually and when they are constructed internally"""

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

    taper_op1 = taper_operation(
        operation, generators, paulixops, paulix_sector, wire_order, op_gen=None
    )
    taper_op2 = taper_operation(
        operation, generators, paulixops, paulix_sector, wire_order, op_gen=op_gen
    )
    for op1, op2 in zip(taper_op1, taper_op2):
        qml.assert_equal(op1.base, op2.base)

    with qml.queuing.AnnotatedQueue() as q_tape1:
        taper_operation(operation, generators, paulixops, paulix_sector, wire_order, op_gen=None)
    with qml.queuing.AnnotatedQueue() as q_tape2:
        taper_operation(operation, generators, paulixops, paulix_sector, wire_order, op_gen=op_gen)
    tape1 = qml.tape.QuantumScript.from_queue(q_tape1)
    tape2 = qml.tape.QuantumScript.from_queue(q_tape2)
    taper_circuit1 = [x for x in tape1.circuit if x.label() != "I"]
    taper_circuit2 = [x for x in tape2.circuit if x.label() != "I"]

    assert len(taper_op1) == len(taper_circuit1)
    assert len(taper_op2) == len(taper_circuit2)
    for op1, op2 in zip(taper_circuit1, taper_op1):
        qml.assert_equal(op1.base, op2.base)
    for op1, op2 in zip(taper_circuit2, taper_op2):
        qml.assert_equal(op1.base, op2.base)

    if taper_op1:
        observables = [
            hamiltonian,  # for energy
            sum(qml.PauliZ(wire) for wire in hamiltonian.wires),  # for local-cost 1
            # for local-cost 2
            sum(qml.PauliZ(wire) @ qml.PauliZ(wire + 1) for wire in hamiltonian.wires[:-1]),
        ]
        tapered_obs = [
            qml.taper(observale, generators, paulixops, paulix_sector) for observale in observables
        ]

        hf_state = np.where(np.arange(len(hamiltonian.wires)) < mol.n_electrons, 1, 0)
        hf_tapered = taper_hf(
            generators, paulixops, paulix_sector, mol.n_electrons, len(hamiltonian.wires)
        )
        o, l = np.array([1, 0]), np.array([0, 1])
        state = functools.reduce(np.kron, [l if s else o for s in hf_state])
        state_tapered = functools.reduce(np.kron, [l if s else o for s in hf_tapered])
        evolved_state = np.matmul(qml.matrix(operation, wire_order=range(len(hf_state))), state)
        ob_tap_mat = functools.reduce(
            np.matmul,
            [qml.matrix(op, wire_order=range(len(hf_tapered))) for op in taper_op1],
        )
        evolved_state_tapered = np.matmul(ob_tap_mat, state_tapered)

        for obs, tap_obs in zip(observables, tapered_obs):
            expec_val = (
                scipy.sparse.coo_matrix(evolved_state)
                @ scipy.sparse.coo_matrix(qml.matrix(obs, wire_order=range(len(hf_state))))
                @ scipy.sparse.coo_matrix(evolved_state).getH()
            ).toarray()
            expec_val_tapered = (
                scipy.sparse.coo_matrix(evolved_state_tapered)
                @ scipy.sparse.coo_matrix(qml.matrix(tap_obs, wire_order=range(len(hf_tapered))))
                @ scipy.sparse.coo_matrix(evolved_state_tapered).getH()
            ).toarray()
            assert np.isclose(expec_val, expec_val_tapered)


@pytest.mark.parametrize(
    ("operation", "op_wires", "op_gen"),
    [
        (qml.RZ, [3], None),
        (qml.RY, [2], qml.Hamiltonian([-0.5], [qml.PauliY(wires=[2])])),
        (qml.SingleExcitation, [0, 2], None),
        (
            qml.OrbitalRotation,
            [0, 1, 2, 3],
            lambda wires: qml.Hamiltonian(
                (0.25, -0.25, 0.25, -0.25),
                [
                    qml.PauliX(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliY(wires=[2]),
                    qml.PauliY(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliX(wires=[2]),
                    qml.PauliX(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliY(wires=[3]),
                    qml.PauliY(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliX(wires=[3]),
                ],
            ),
        ),
    ],
)
def test_taper_callable_ops(operation, op_wires, op_gen):
    """Test that operation callables can be used to obtain their consistent taperings"""

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

    taper_op_fn = taper_operation(
        operation, generators, paulixops, paulix_sector, wire_order, op_wires, op_gen
    )
    assert callable(taper_op_fn)

    for params in [0.0, 1.3, np.pi / 2, 2.37, np.pi]:
        if callable(op_gen):
            op_gen = op_gen(op_wires)
        taper_op = taper_operation(
            operation(params, wires=op_wires),
            generators,
            paulixops,
            paulix_sector,
            wire_order,
            op_wires=op_wires,
            op_gen=op_gen,
        )
        for op1, op2 in zip(taper_op_fn(params), taper_op):
            qml.assert_equal(op1.base, op2.base)


@pytest.mark.parametrize(
    ("operation", "op_wires", "op_gen"),
    [
        (
            lambda phi, wires: qml.QubitUnitary(
                qml.math.array(
                    [
                        [qml.math.cos(phi / 2), 0, 0, -1j * qml.math.sin(phi / 2)],
                        [0, qml.math.cos(phi / 2), -1j * qml.math.sin(phi / 2), 0],
                        [0, -1j * qml.math.sin(phi / 2), qml.math.cos(phi / 2), 0],
                        [-1j * qml.math.sin(phi / 2), 0, 0, qml.math.cos(phi / 2)],
                    ]
                ),
                wires=wires,
            ),
            [0, 2],
            lambda phi, wires: -0.5 * phi * qml.PauliX(wires=wires[0]) @ qml.PauliX(wires=wires[1]),
        ),
    ],
)
def test_taper_matrix_ops(operation, op_wires, op_gen):
    """Test that taper_operation can be used with gate operation built using matrices"""

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

    taper_op1 = taper_operation(
        qml.IsingXX,
        generators,
        paulixops,
        paulix_sector,
        wire_order,
        op_wires=op_wires,
    )
    assert callable(taper_op1)

    for params in [0.0, 1.3, np.pi / 2, 2.37, np.pi]:
        taper_op2 = taper_operation(
            operation(params, wires=op_wires),
            generators,
            paulixops,
            paulix_sector,
            wire_order,
            op_wires=op_wires,
            op_gen=functools.partial(op_gen, phi=params),
        )
        for op1, op2 in zip(taper_op1(params), taper_op2):
            qml.assert_equal(op1.base, op2.base)


@pytest.mark.parametrize(
    ("operation", "op_wires", "op_gen", "message_match"),
    [
        (
            qml.SingleExcitation,
            None,
            None,
            "Wires for the operation must be provided with 'op_wires' args if either of 'operation' or 'op_gen' is a callable",
        ),
        (
            qml.OrbitalRotation,
            [0, 1, 2, 3],
            lambda: qml.Hamiltonian(
                (0.25, -0.25, 0.25, -0.25),
                [
                    qml.PauliX(wires=[0]) @ qml.PauliY(wires=[2]),
                    qml.PauliY(wires=[0]) @ qml.PauliX(wires=[2]),
                    qml.PauliX(wires=[1]) @ qml.PauliY(wires=[3]),
                    qml.PauliY(wires=[1]) @ qml.PauliX(wires=[3]),
                ],
            ),
            "Generator function provided with 'op_gen' should have 'wires' as its only required keyword argument.",
        ),
    ],
)
def test_inconsistent_callable_ops(operation, op_wires, op_gen, message_match):
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
        taper_operation(
            operation, generators, paulixops, paulix_sector, wire_order, op_wires, op_gen
        )


@pytest.mark.parametrize(("ps_size", "max_size"), [(190, 49), (40, 13)])
def test_split_pauli_sentence(ps_size, max_size):
    """Test that _split_pauli_sentence splits the PauliSentence objects into correct chunks."""

    sentence = qml.pauli.PauliSentence(
        {qml.pauli.PauliWord({i: "X", i + 1: "Y", i + 2: "Z"}): i for i in range(ps_size)}
    )

    split_sentence = {}
    for ps in _split_pauli_sentence(sentence, max_size=max_size):
        assert len(ps) <= max_size
        split_sentence = {**split_sentence, **ps}

    assert sentence == qml.pauli.PauliSentence(split_sentence)


@pytest.mark.parametrize(
    ("symbols", "geometry"),
    [(["Li", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.13]]))],
)
def test_taper_wire_order(symbols, geometry):
    r"""Test that a tapering workflow results in correct order of wires."""

    molecule = qml.qchem.Molecule(symbols, geometry)
    hamiltonian, num_wires = qml.qchem.molecular_hamiltonian(molecule)

    generators = qml.symmetry_generators(hamiltonian)
    paulixops = qml.paulix_ops(generators, num_wires)
    paulix_sector = optimal_sector(hamiltonian, generators, molecule.n_electrons)

    tapered_ham = qml.taper(hamiltonian, generators, paulixops, paulix_sector)
    assert tapered_ham.wires.tolist() == list(sorted(tapered_ham.wires))


@pytest.mark.jax
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
    ],
)
def test_taper_jax_jit(symbols, geometry, charge):
    r"""Test that an observable can be tapred within a jax-jit workflow."""

    import jax

    molecule = qml.qchem.Molecule(symbols, jax.numpy.array(geometry), charge)
    hamiltonian, num_wires = qml.qchem.molecular_hamiltonian(molecule)

    generators = qml.symmetry_generators(hamiltonian)
    paulixops = qml.paulix_ops(generators, num_wires)
    paulix_sector = tuple(optimal_sector(hamiltonian, generators, molecule.n_electrons))

    tapered_ham1 = qml.simplify(qml.taper(hamiltonian, generators, paulixops, paulix_sector))
    tapered_ham2 = qml.simplify(
        jax.jit(qml.taper, static_argnums=[3])(hamiltonian, generators, paulixops, paulix_sector)
    )

    assert qml.math.get_deep_interface(tapered_ham1.terms()[0]) == "jax"
    assert qml.math.get_deep_interface(tapered_ham2.terms()[0]) == "jax"
    qml.assert_equal(tapered_ham1, tapered_ham2)
