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
# pylint: disable=too-many-arguments
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
    assert np.allclose(eigvals, np.einsum("ti,tj->tij", eigvals_ref, eigvals_ref))
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
    factors1, _, _ = qml.qchem.factorize(two_tensor, 1e-5, 1e-5, cholesky=False)
    factors2, _, _ = qml.qchem.factorize(two_tensor, 1e-5, 1e-5, cholesky=True)

    assert qml.math.allclose(np.tensordot(factors1, factors1, axes=([0], [0])), two_tensor)
    assert qml.math.allclose(np.tensordot(factors2, factors2, axes=([0], [0])), two_tensor)


@pytest.mark.external
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
@pytest.mark.parametrize("cholesky", [False, True])
@pytest.mark.parametrize("regularization", [None, "L1", "L2"])
def test_factorize_compressed_reproduce(two_tensor, cholesky, regularization):
    r"""Test that factors returned by the factorize function reproduce the two-electron tensor."""
    optax = pytest.importorskip("optax")

    factors, cores, leaves = qml.qchem.factorize(
        two_tensor,
        cholesky=cholesky,
        compressed=True,
        regularization=regularization,
        optimizer=optax.adam(learning_rate=0.001),
    )

    assert qml.math.allclose(np.einsum("tpqi,trsi->pqrs", factors, factors), two_tensor, atol=1e-3)
    assert qml.math.allclose(
        qml.math.einsum("tpk,tqk,tkl,trl,tsl->pqrs", leaves, leaves, cores, leaves, leaves),
        two_tensor,
        atol=1e-3,
    )


@pytest.mark.external
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
def test_regularization_error(two_tensor):
    r"""Test that the factorize function raises an error when incorrect regularization is provided."""
    _ = pytest.importorskip("optax")

    with pytest.raises(ValueError, match="Supported regularization types include"):
        qml.qchem.factorize(two_tensor, compressed=True, regularization=True)


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
            np.array([[-1.25330978e00, -2.11164419e-13], [-2.10831352e-13, -4.75068865e-01]]),
            np.array(  # two-electron integral tensor in physicist notation
                [
                    [
                        [[6.74755925e-01, 2.43333131e-13], [2.43333131e-13, 1.81210462e-01]],
                        [[2.43333131e-13, 1.81210462e-01], [6.63711398e-01, 2.24598118e-13]],
                    ],
                    [
                        [[2.43166598e-13, 6.63711398e-01], [1.81210462e-01, 2.24598118e-13]],
                        [[1.81210462e-01, 2.24598118e-13], [2.24820162e-13, 6.97651499e-01]],
                    ],
                ]
            ),
            1.0e-5,
            [  # computed manually, multiplied by 2 to account for spin
                np.array([0.84064649, -2.59579282, 0.84064649, 0.45724992, 0.45724992]),
                np.array(
                    [
                        -9.73801723e-05,
                        5.60006390e-03,
                        -9.73801723e-05,
                        2.84747318e-03,
                        9.57150297e-05,
                        -2.79878310e-03,
                        9.57150297e-05,
                        -2.79878310e-03,
                        -2.79878310e-03,
                        -2.79878310e-03,
                        2.75092558e-03,
                    ]
                ),
                np.array(
                    [
                        0.04530262,
                        -0.04530262,
                        -0.04530262,
                        -0.04530262,
                        -0.04530262,
                        0.09060523,
                        0.04530262,
                    ]
                ),
                np.array(
                    [
                        -0.68077716,
                        1.6874169,
                        -0.68077716,
                        0.17166195,
                        -0.66913628,
                        0.16872663,
                        -0.66913628,
                        0.16872663,
                        0.16872663,
                        0.16872663,
                        0.16584151,
                    ]
                ),
            ],
            [  # computed manually
                [
                    qml.PauliZ(wires=[0]),
                    qml.Identity(wires=[0]),
                    qml.PauliZ(wires=[1]),
                    qml.PauliZ(wires=[2]),
                    qml.PauliZ(wires=[3]),
                ],
                [
                    qml.PauliZ(wires=[0]),
                    qml.Identity(wires=[0]),
                    qml.PauliZ(wires=[1]),
                    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]),
                    qml.PauliZ(wires=[2]),
                    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[2]),
                    qml.PauliZ(wires=[3]),
                    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[3]),
                    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
                    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[3]),
                    qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[3]),
                ],
                [
                    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]),
                    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[2]),
                    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[3]),
                    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
                    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[3]),
                    qml.Identity(wires=[2]),
                    qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[3]),
                ],
                [
                    qml.PauliZ(wires=[0]),
                    qml.Identity(wires=[0]),
                    qml.PauliZ(wires=[1]),
                    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]),
                    qml.PauliZ(wires=[2]),
                    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[2]),
                    qml.PauliZ(wires=[3]),
                    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[3]),
                    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
                    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[3]),
                    qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[3]),
                ],
            ],
            [  # computed manually
                np.array(
                    [
                        [-1.00000000e00, 0.00000000e00, -5.71767345e-13, -0.00000000e00],
                        [0.00000000e00, 1.00000000e00, 0.00000000e00, 5.71767345e-13],
                        [-5.71767345e-13, 0.00000000e00, 1.00000000e00, 0.00000000e00],
                        [0.00000000e00, 5.71767345e-13, -0.00000000e00, -1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [-0.0, -0.0, 0.0, -1.0],
                        [-0.0, 0.0, -1.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [-1.0, -0.0, 0.0, 0.0],
                    ]
                ),
                np.array(
                    [
                        [-0.70710678, 0.0, -0.0, -0.70710678],
                        [0.0, -0.70710678, -0.70710678, 0.0],
                        [0.70710678, 0.0, 0.0, -0.70710678],
                        [0.0, 0.70710678, -0.70710678, -0.0],
                    ]
                ),
                np.array(
                    [
                        [-0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, -0.0],
                        [0.0, -0.0, 0.0, 1.0],
                        [-0.0, -0.0, 1.0, 0.0],
                    ]
                ),
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
        ops_ref_str = [qml.pauli.pauli_word_to_string(t) for t in ops_ref[j]]
        for o in op:
            assert (qml.pauli.pauli_word_to_string(o) or "I") in ops_ref_str

    for i, vecs in enumerate(eigvecs):
        checks = []
        for vec in vecs.T:
            check = False
            for rf_vec in eigvecs_ref[i].T:
                if np.allclose(rf_vec, vec) or np.allclose(-rf_vec, vec):
                    check = True
                    break
            checks.append(check)
        assert np.all(checks)


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
    *_, u_transform = qml.qchem.basis_rotation(one_electron, two_electron)

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


@pytest.mark.parametrize(
    ("two_body_tensor", "spatial_basis", "one_body_correction", "chemist_two_body_coeffs"),
    [
        (
            np.array(
                [
                    [
                        [[6.74755925e-01, 2.43333131e-13], [2.43333131e-13, 1.81210462e-01]],
                        [[2.43333131e-13, 1.81210462e-01], [6.63711398e-01, 2.24598118e-13]],
                    ],
                    [
                        [[2.43166598e-13, 6.63711398e-01], [1.81210462e-01, 2.24598118e-13]],
                        [[1.81210462e-01, 2.24598118e-13], [2.24820162e-13, 6.97651499e-01]],
                    ],
                ]
            ),
            True,
            np.array(
                [
                    [-4.27983194e-01, -2.33965625e-13],
                    [-2.33882358e-13, -4.39430980e-01],
                ]
            ),
            np.array(
                [
                    [
                        [[3.37377963e-01, 1.21666566e-13], [1.21666566e-13, 3.31855699e-01]],
                        [[1.21666566e-13, 9.06052311e-02], [9.06052311e-02, 1.12299059e-13]],
                    ],
                    [
                        [[1.21583299e-13, 9.06052311e-02], [9.06052311e-02, 1.12410081e-13]],
                        [[3.31855699e-01, 1.12299059e-13], [1.12299059e-13, 3.48825749e-01]],
                    ],
                ]
            ),
        ),
        (
            np.array(
                [
                    [
                        [
                            [3.37377963e-01, 0.00000000e00, 1.21666566e-13, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [1.21666566e-13, 0.00000000e00, 9.06052311e-02, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [3.37377963e-01, 0.00000000e00, 1.21666566e-13, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [1.21666566e-13, 0.00000000e00, 9.06052311e-02, 0.00000000e00],
                        ],
                        [
                            [1.21666566e-13, 0.00000000e00, 9.06052311e-02, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [3.31855699e-01, 0.00000000e00, 1.12299059e-13, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [1.21666566e-13, 0.00000000e00, 9.06052311e-02, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [3.31855699e-01, 0.00000000e00, 1.12299059e-13, 0.00000000e00],
                        ],
                    ],
                    [
                        [
                            [0.00000000e00, 3.37377963e-01, 0.00000000e00, 1.21666566e-13],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 1.21666566e-13, 0.00000000e00, 9.06052311e-02],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 3.37377963e-01, 0.00000000e00, 1.21666566e-13],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 1.21666566e-13, 0.00000000e00, 9.06052311e-02],
                        ],
                        [
                            [0.00000000e00, 1.21666566e-13, 0.00000000e00, 9.06052311e-02],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 3.31855699e-01, 0.00000000e00, 1.12299059e-13],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 1.21666566e-13, 0.00000000e00, 9.06052311e-02],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 3.31855699e-01, 0.00000000e00, 1.12299059e-13],
                        ],
                    ],
                    [
                        [
                            [1.21583299e-13, 0.00000000e00, 3.31855699e-01, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [9.06052311e-02, 0.00000000e00, 1.12299059e-13, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [1.21583299e-13, 0.00000000e00, 3.31855699e-01, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [9.06052311e-02, 0.00000000e00, 1.12299059e-13, 0.00000000e00],
                        ],
                        [
                            [9.06052311e-02, 0.00000000e00, 1.12299059e-13, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [1.12410081e-13, 0.00000000e00, 3.48825749e-01, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [9.06052311e-02, 0.00000000e00, 1.12299059e-13, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [1.12410081e-13, 0.00000000e00, 3.48825749e-01, 0.00000000e00],
                        ],
                    ],
                    [
                        [
                            [0.00000000e00, 1.21583299e-13, 0.00000000e00, 3.31855699e-01],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 9.06052311e-02, 0.00000000e00, 1.12299059e-13],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 1.21583299e-13, 0.00000000e00, 3.31855699e-01],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 9.06052311e-02, 0.00000000e00, 1.12299059e-13],
                        ],
                        [
                            [0.00000000e00, 9.06052311e-02, 0.00000000e00, 1.12299059e-13],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 1.12410081e-13, 0.00000000e00, 3.48825749e-01],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 9.06052311e-02, 0.00000000e00, 1.12299059e-13],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 1.12410081e-13, 0.00000000e00, 3.48825749e-01],
                        ],
                    ],
                ]
            ),
            False,
            np.array(
                [
                    [-4.27983194e-01, -0.00000000e00, -2.33965625e-13, -0.00000000e00],
                    [-0.00000000e00, -4.27983194e-01, -0.00000000e00, -2.33965625e-13],
                    [-2.33882358e-13, -0.00000000e00, -4.39430980e-01, -0.00000000e00],
                    [-0.00000000e00, -2.33882358e-13, -0.00000000e00, -4.39430980e-01],
                ]
            ),
            np.array(
                [
                    [
                        [
                            [3.37377963e-01, 0.00000000e00, 1.21666566e-13, 0.00000000e00],
                            [0.00000000e00, 3.37377963e-01, 0.00000000e00, 1.21666566e-13],
                            [1.21666566e-13, 0.00000000e00, 3.31855699e-01, 0.00000000e00],
                            [0.00000000e00, 1.21666566e-13, 0.00000000e00, 3.31855699e-01],
                        ],
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                        [
                            [1.21666566e-13, 0.00000000e00, 9.06052311e-02, 0.00000000e00],
                            [0.00000000e00, 1.21666566e-13, 0.00000000e00, 9.06052311e-02],
                            [9.06052311e-02, 0.00000000e00, 1.12299059e-13, 0.00000000e00],
                            [0.00000000e00, 9.06052311e-02, 0.00000000e00, 1.12299059e-13],
                        ],
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                    ],
                    [
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                        [
                            [3.37377963e-01, 0.00000000e00, 1.21666566e-13, 0.00000000e00],
                            [0.00000000e00, 3.37377963e-01, 0.00000000e00, 1.21666566e-13],
                            [1.21666566e-13, 0.00000000e00, 3.31855699e-01, 0.00000000e00],
                            [0.00000000e00, 1.21666566e-13, 0.00000000e00, 3.31855699e-01],
                        ],
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                        [
                            [1.21666566e-13, 0.00000000e00, 9.06052311e-02, 0.00000000e00],
                            [0.00000000e00, 1.21666566e-13, 0.00000000e00, 9.06052311e-02],
                            [9.06052311e-02, 0.00000000e00, 1.12299059e-13, 0.00000000e00],
                            [0.00000000e00, 9.06052311e-02, 0.00000000e00, 1.12299059e-13],
                        ],
                    ],
                    [
                        [
                            [1.21583299e-13, 0.00000000e00, 9.06052311e-02, 0.00000000e00],
                            [0.00000000e00, 1.21583299e-13, 0.00000000e00, 9.06052311e-02],
                            [9.06052311e-02, 0.00000000e00, 1.12410081e-13, 0.00000000e00],
                            [0.00000000e00, 9.06052311e-02, 0.00000000e00, 1.12410081e-13],
                        ],
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                        [
                            [3.31855699e-01, 0.00000000e00, 1.12299059e-13, 0.00000000e00],
                            [0.00000000e00, 3.31855699e-01, 0.00000000e00, 1.12299059e-13],
                            [1.12299059e-13, 0.00000000e00, 3.48825749e-01, 0.00000000e00],
                            [0.00000000e00, 1.12299059e-13, 0.00000000e00, 3.48825749e-01],
                        ],
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                    ],
                    [
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                        [
                            [1.21583299e-13, 0.00000000e00, 9.06052311e-02, 0.00000000e00],
                            [0.00000000e00, 1.21583299e-13, 0.00000000e00, 9.06052311e-02],
                            [9.06052311e-02, 0.00000000e00, 1.12410081e-13, 0.00000000e00],
                            [0.00000000e00, 9.06052311e-02, 0.00000000e00, 1.12410081e-13],
                        ],
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        ],
                        [
                            [3.31855699e-01, 0.00000000e00, 1.12299059e-13, 0.00000000e00],
                            [0.00000000e00, 3.31855699e-01, 0.00000000e00, 1.12299059e-13],
                            [1.12299059e-13, 0.00000000e00, 3.48825749e-01, 0.00000000e00],
                            [0.00000000e00, 1.12299059e-13, 0.00000000e00, 3.48825749e-01],
                        ],
                    ],
                ]
            ),
        ),
    ],
)
def test_chemist_transform(
    two_body_tensor, spatial_basis, one_body_correction, chemist_two_body_coeffs
):
    r"""Test that `_chemist_transform` builds correct two-body tensors in
    chemist notation with correct one-body corrections"""
    # pylint: disable=protected-access
    one_body_corr, chemist_two_body = qml.qchem.factorization._chemist_transform(
        two_body_tensor=two_body_tensor, spatial_basis=spatial_basis
    )
    assert np.allclose(one_body_corr, one_body_correction)
    assert np.allclose(chemist_two_body, chemist_two_body_coeffs)

    (chemist_one_body,) = qml.qchem.factorization._chemist_transform(
        one_body_tensor=one_body_correction, spatial_basis=spatial_basis
    )
    assert np.allclose(chemist_one_body, one_body_correction)

    chemist_one_body, chemist_two_body = qml.qchem.factorization._chemist_transform(
        one_body_tensor=one_body_correction,
        two_body_tensor=two_body_tensor,
        spatial_basis=spatial_basis,
    )

    assert np.allclose(one_body_corr, one_body_correction)


@pytest.mark.parametrize(
    ("core_shifted", "one_body_shifted", "two_body_shifted"),
    [
        # Following shifted terms have been computed manually for HeH+ moelcule.
        # Their correctness has been verified by computing the chemist Hamiltonian
        # and observing its eigenspectrum for the lowest eigenvalue with same
        # number of electrons.
        #
        # >>> f_chemist = chemist_fermionic_observable(core_shifted, one_body_shifted, two_body_shifted)
        # >>> H_chemist = qml.jordan_wigner(f_chemist)
        # >>> eigvals, eigvecs = np.linalg.eigh(H_chemist.matrix())
        # >>> for eigval, eigvec in zip(eigvals, eigvecs.T):
        # ...    if (eigvec @ qml.matrix(qml.qchem.particle_number(4)) @ eigvec.conj().T) == 2:
        # ...        print(eigval)
        # ...        break
        # -2.688647053431185
        (
            np.array([0.14782753]),
            np.array([[-1.55435269, 0.08134727], [0.08134727, -0.0890333]]),
            np.array(
                [
                    [
                        [[0.02932015, -0.04067343], [-0.04067343, -0.02931994]],
                        [[-0.04067343, 0.08211742], [0.08211742, 0.04067303]],
                    ],
                    [
                        [[-0.04067343, 0.08211742], [0.08211742, 0.04067303]],
                        [[-0.02931994, 0.04067303], [0.04067303, 0.02932037]],
                    ],
                ]
            ),
        ),
    ],
)
def test_symmetry_shift(core_shifted, one_body_shifted, two_body_shifted):
    """Test that `symmetry_shift` builds correct two-body tensors with accurate correction terms"""
    symbols = ["He", "H"]
    geometry = qml.numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False)
    mol = qml.qchem.Molecule(symbols, geometry, charge=1, basis_name="STO-3G")
    core, one, two = qml.qchem.electron_integrals(mol)()

    # pylint: disable=protected-access
    cone, ctwo = qml.qchem.factorization._chemist_transform(one, two, spatial_basis=True)
    score, sone, stwo = qml.qchem.symmetry_shift(core, cone, ctwo, n_elec=mol.n_electrons)

    assert np.allclose(score, core_shifted)
    assert np.allclose(sone, one_body_shifted)
    assert np.allclose(stwo, two_body_shifted)
