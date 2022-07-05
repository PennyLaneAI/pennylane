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
