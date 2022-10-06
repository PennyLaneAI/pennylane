# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for functions needed for two-electron integral tensor factorization.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np


@pytest.mark.parametrize(
    ("two_tensor", "tol_f", "tol_s", "factors_ref"),
    [
        # two-electron tensor computed as
        # symbols  = ['H', 'H']
        # geometry = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], requires_grad = False) / 0.529177
        # mol = qml.qchem.Molecule(symbols, geometry, basis_name='sto-3g')
        # core, one, two = qml.qchem.electron_integrals(mol)()
        # two = np.swapaxes(two, 1, 3) # convert to chemist notation
        (
            np.array(
                [
                    [
                        [[6.74755872e-01, -2.85826918e-13], [-2.85799162e-13, 6.63711349e-01]],
                        [[-2.85965696e-13, 1.81210478e-01], [1.81210478e-01, -2.63900013e-13]],
                    ],
                    [
                        [[-2.85854673e-13, 1.81210478e-01], [1.81210478e-01, -2.63900013e-13]],
                        [[6.63711349e-01, -2.63677968e-13], [-2.63788991e-13, 6.97651447e-01]],
                    ],
                ]
            ),
            1.0e-5,
            1.0e-5,
            # factors computed with openfermion (rearranged)
            np.array(
                [
                    [[1.06723441e-01, 6.58493593e-17], [6.58493593e-17, -1.04898533e-01]],
                    [[-1.11022302e-16, -4.25688222e-01], [-4.25688222e-01, -1.11022302e-16]],
                    [[-8.14472857e-01, 1.40518540e-16], [1.40518540e-16, -8.28642144e-01]],
                ]
            ),
        ),
        (
            np.array(
                [
                    [
                        [[6.74755872e-01, -2.85826918e-13], [-2.85799162e-13, 6.63711349e-01]],
                        [[-2.85965696e-13, 1.81210478e-01], [1.81210478e-01, -2.63900013e-13]],
                    ],
                    [
                        [[-2.85854673e-13, 1.81210478e-01], [1.81210478e-01, -2.63900013e-13]],
                        [[6.63711349e-01, -2.63677968e-13], [-2.63788991e-13, 6.97651447e-01]],
                    ],
                ]
            ),
            1.0e-1,
            1.0e-1,
            np.array(
                [
                    [[-1.11022302e-16, -4.25688222e-01], [-4.25688222e-01, -1.11022302e-16]],
                    [[-8.14472857e-01, 1.40518540e-16], [1.40518540e-16, -8.28642144e-01]],
                ]
            ),
        ),
    ],
)
def test_factorize(two_tensor, tol_f, tol_s, factors_ref):
    r"""Test that factorize function returns the correct values."""
    factors, eigvals, eigvecs = qml.qchem.factorize(two_tensor, tol_f, tol_s)

    eigvals_ref, eigvecs_ref = np.linalg.eigh(factors_ref)

    assert np.allclose(factors, factors_ref)
    assert np.allclose(eigvals, eigvals_ref)
    assert np.allclose(eigvecs, eigvecs_ref)


@pytest.mark.parametrize(
    "two_tensor",
    [
        # two-electron tensor computed as
        # symbols  = ['H', 'H']
        # geometry = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], requires_grad = False) / 0.529177
        # mol = qml.qchem.Molecule(symbols, geometry, basis_name='sto-3g')
        # core, one, two = qml.qchem.electron_integrals(mol)()
        # two = np.swapaxes(two, 1, 3) # convert to chemist notation
        np.array(
            [
                [
                    [[6.74755872e-01, -2.85826918e-13], [-2.85799162e-13, 6.63711349e-01]],
                    [[-2.85965696e-13, 1.81210478e-01], [1.81210478e-01, -2.63900013e-13]],
                ],
                [
                    [[-2.85854673e-13, 1.81210478e-01], [1.81210478e-01, -2.63900013e-13]],
                    [[6.63711349e-01, -2.63677968e-13], [-2.63788991e-13, 6.97651447e-01]],
                ],
            ]
        ),
    ],
)
def test_factorize_reproduce(two_tensor):
    r"""Test that factors returned by the factorize function reproduce the two-electron tensor."""
    factors, _, _ = qml.qchem.factorize(two_tensor, 1e-5, 1e-5)
    two_computed = np.zeros(two_tensor.shape)
    for mat in factors:
        two_computed += np.einsum("ij, lk", mat, mat)

    assert np.allclose(two_computed, two_tensor)


@pytest.mark.parametrize(
    "two_tensor",
    [
        np.array(
            [
                [6.74755872e-01, -2.85826918e-13, -2.85799162e-13, 6.63711349e-01],
                [-2.85965696e-13, 1.81210478e-01, 1.81210478e-01, -2.63900013e-13],
                [-2.85854673e-13, 1.81210478e-01, 1.81210478e-01, -2.63900013e-13],
                [6.63711349e-01, -2.63677968e-13, -2.63788991e-13, 6.97651447e-01],
            ]
        ),
    ],
)
def test_shape_error(two_tensor):
    r"""Test that the factorize function raises an error when the two-electron integral tensor does
    not have the correct shape."""
    with pytest.raises(ValueError, match="The two-electron repulsion tensor must have"):
        qml.qchem.factorize(two_tensor, 1e-5, 1e-5)


@pytest.mark.parametrize(
    "two_tensor",
    [
        np.array(
            [
                [
                    [[6.74755872e-01, -2.85826918e-13], [-2.85799162e-13, 6.63711349e-01]],
                    [[-2.85965696e-13, 1.81210478e-01], [1.81210478e-01, -2.63900013e-13]],
                ],
                [
                    [[-2.85854673e-13, 1.81210478e-01], [1.81210478e-01, -2.63900013e-13]],
                    [[6.63711349e-01, -2.63677968e-13], [-2.63788991e-13, 6.97651447e-01]],
                ],
            ]
        ),
    ],
)
def test_empty_error(two_tensor):
    r"""Test that the factorize function raises an error when all factors or their eigenvectors are
    discarded."""
    with pytest.raises(ValueError, match="All factors are discarded."):
        qml.qchem.factorize(two_tensor, 1e1, 1e-5)

    with pytest.raises(ValueError, match="All eigenvectors are discarded."):
        qml.qchem.factorize(two_tensor, 1e-5, 1e1)


@pytest.mark.parametrize(
    ("one_matrix", "two_tensor", "tol_factor", "coeffs_ref", "ops_ref", "eigvecs_ref"),
    [
        (  # one_matrix and two_tensor are obtained with:
            # symbols  = ['H', 'H']
            # geometry = np.array([[0.0, 0.0, 0.0], [1.39839789, 0.0, 0.0]], requires_grad = False)
            # mol = qml.qchem.Molecule(symbols, geometry)
            # core, one_matrix, two_tensor = qml.qchem.electron_integrals(mol)()
            np.array([[-1.25330961e00, -2.07722728e-13], [-2.07611706e-13, -4.75069041e-01]]),
            np.array(  # two-electron integral tensor in physicist notation
                [
                    [
                        [[6.74755872e-01, 2.39697151e-13], [2.39780418e-13, 1.81210478e-01]],
                        [[2.39808173e-13, 1.81210478e-01], [6.63711349e-01, 2.21378471e-13]],
                    ],
                    [
                        [[2.39808173e-13, 6.63711349e-01], [1.81210478e-01, 2.21822560e-13]],
                        [[1.81210478e-01, 2.21489493e-13], [2.21267449e-13, 6.97651447e-01]],
                    ],
                ]
            ),
            1.0e-5,
            [  # computed manually, multiplied by 2 to account for spin
                np.array([-3.07829375, 1.92254344, 1.15575031]),
                np.array([-9.73801723e-05, 5.59923133e-03, 9.57150297e-05, -5.59756619e-03]),
                np.array([-0.09060523, 0.09060523]),
                np.array([-0.68077716, 1.01246018, -0.66913628, 0.33745327]),
            ],
            [  # computed manually
                [qml.Identity(0), qml.PauliZ(0), qml.PauliZ(1)],
                [qml.PauliZ(0), qml.Identity(0), qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)],
                [qml.PauliZ(0) @ qml.PauliZ(1), qml.Identity(1)],
                [qml.PauliZ(0), qml.Identity(0), qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)],
            ],
            [  # computed manually
                np.array([[-1.00000000e00, 4.67809646e-13], [-4.67809646e-13, -1.00000000e00]]),
                np.array([[-1.00000000e00, -2.27711575e-14], [2.27711575e-14, -1.00000000e00]]),
                np.array([[-0.70710678, -0.70710678], [-0.70710678, 0.70710678]]),
                np.array([[2.21447776e-11, 1.00000000e00], [-1.00000000e00, 2.21447776e-11]]),
            ],
        ),
    ],
)
def test_basis_rotation_output(
    one_matrix, two_tensor, tol_factor, coeffs_ref, ops_ref, eigvecs_ref
):
    r"""Test that basis_rotation function returns the correct values."""
    coeffs, ops, eigvecs = qml.qchem.basis_rotation(one_matrix, two_tensor, tol_factor)

    for i, coeff in enumerate(coeffs):
        assert np.allclose(np.sort(coeff), np.sort(coeffs_ref[i]))

    for j, op in enumerate(ops):
        ops_ref_str = [qml.grouping.pauli_word_to_string(t) for t in ops_ref[i]]
        for o in op:
            assert qml.grouping.pauli_word_to_string(o) in ops_ref_str

    for i, vec in enumerate(eigvecs):
        assert np.allclose(vec, eigvecs_ref[i])


@pytest.mark.parametrize(
    ("core", "one_electron", "two_electron"),
    [
        (
            np.array([0.71510405]),
            np.array([[-1.25330961e00, 4.13891144e-13], [4.14002166e-13, -4.75069041e-01]]),
            np.array(
                [
                    [
                        [[6.74755872e-01, -4.78089790e-13], [-4.77978768e-13, 1.81210478e-01]],
                        [[-4.77978768e-13, 1.81210478e-01], [6.63711349e-01, -4.41091608e-13]],
                    ],
                    [
                        [[-4.77978768e-13, 6.63711349e-01], [1.81210478e-01, -4.40869563e-13]],
                        [[1.81210478e-01, -4.40869563e-13], [-4.40980585e-13, 6.97651447e-01]],
                    ],
                ]
            ),
        )
    ],
)
def test_basis_rotation_utransform(core, one_electron, two_electron):
    r"""Test that basis_rotation function returns the correct transformation matrices. This test
    constructs the matrix representation of a factorized Hamiltonian and then applies the
    transformation matrices to generate a new set of fermionic creation and annihilation operators.
    A new Hamiltonian is generated from these operators and is compared with the original
    Hamiltonian.
    """
    c_group, o_group, u_transform = qml.qchem.basis_rotation(one_electron, two_electron)

    a_cr = [  # fermionic creation operators
        np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ),
        np.array(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        ),
    ]

    a_an = [  # fermionic annihilation operators
        np.array(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        ),
        np.array(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
        ),
    ]

    # compute matrix representation of the factorized Hamiltonian. Only the one-body part is
    # included to keep the test simple.

    t_matrix = one_electron - 0.5 * np.einsum("illj", two_electron)

    h1 = 0.0
    for i, ac in enumerate(a_cr):
        for j, aa in enumerate(a_an):
            h1 += t_matrix[i, j] * ac @ aa

    h1 += np.identity(len(h1)) * core

    # compute matrix representation of the basis-rotated Hamiltonian. Only the one-body part is
    # included to keep the test simple.

    u = u_transform[0]
    ac_rot = []
    aa_rot = []

    for u_ in u.T:  # construct the rotated creation operators
        ac_new = 0.0
        for i, ac in enumerate(a_cr):
            ac_new += u_[i] * ac
        ac_rot.append(ac_new)

    for u_ in u.T:  # construct the rotated annihilation operators
        aa_new = 0.0
        for i, aa in enumerate(a_an):
            aa_new += u_[i] * aa
        aa_rot.append(aa_new)

    val, _ = np.linalg.eigh(t_matrix)
    h2 = 0.0
    for i, v in enumerate(val):
        h2 += v * ac_rot[i] @ aa_rot[i]

    h2 += np.identity(len(h2)) * core

    assert np.allclose(h1, h2)
