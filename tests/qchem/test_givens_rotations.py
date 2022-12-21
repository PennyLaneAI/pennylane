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
Unit tests for functions needed for performing givens decomposition of a unitary.
"""

import pytest

from scipy.stats import unitary_group

import pennylane as qml
from pennylane import numpy as np
from pennylane.qchem.givens_rotations import givens_matrix, givens_rotate, givens_decomposition


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize(
    ("a", "b"),
    [
        (1.2, 2.3),
        (1.2j, 2.3j),
        (1.5 + 2.3j, 2.1 - 3.7j),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.2, 2.3j),
    ],
)
def test_givens_matrix(a, b, left):
    r"""Test that Givens rotation matrix are build correctly."""

    grot_mat = givens_matrix(a, b, left)
    assert np.isreal(grot_mat[0, 1]) and np.isreal(grot_mat[1, 1])

    rotated_vector = grot_mat @ np.array([a, b]).T
    result_element = b / np.abs(b) * np.hypot(np.abs(a), np.abs(b)) if b else 1.0
    rvec = np.array([0.0, result_element]).T if left else np.array([result_element, 0.0]).T
    assert np.allclose(rotated_vector, rvec)

    res1 = np.round(grot_mat @ grot_mat.conj().T, 5)
    res2 = np.round(grot_mat.conj().T @ grot_mat, 5)
    assert np.all(res1 == res2) and np.all(res1 == np.eye(2))


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("row", [True, False])
@pytest.mark.parametrize("indices", [[0, 1], [2, 3], [1, 4], [0, 3]])
@pytest.mark.parametrize("shape", [(5, 5), (6, 6)])
def test_givens_rotate(shape, indices, row, left):
    r"""Test that Givens rotation is performed correctly."""
    matrix = np.random.rand(*shape) * 1j + np.random.rand(*shape)
    unitary, (i, j) = matrix.copy(), indices
    if row:
        a, b = matrix[indices, j - 1]
        grot_mat = givens_matrix(a, b, left)
        givens_rotate(unitary, grot_mat, indices, row)
        res = b / np.abs(b) * np.hypot(np.abs(a), np.abs(b)) if b else 1.0
        if left:
            assert np.isclose(unitary[indices[0], j - 1], 0.0) and np.isclose(
                unitary[indices[1], j - 1], res
            )
        else:
            assert np.isclose(unitary[indices[0], j - 1], res) and np.isclose(
                unitary[indices[1], j - 1], 0.0
            )
    else:
        a, b = matrix[j - 1, indices].T
        grot_mat = givens_matrix(a, b, left)
        givens_rotate(unitary, grot_mat, indices, row)
        res = b / np.abs(b) * np.hypot(np.abs(a), np.abs(b)) if b else 1.0
        if left:
            assert np.isclose(unitary[j - 1, indices[0]], 0.0) and np.isclose(
                unitary[j - 1, indices[1]], res
            )
        else:
            assert np.isclose(unitary[j - 1, indices[0]], res) and np.isclose(
                unitary[j - 1, indices[1]], 0.0
            )


@pytest.mark.parametrize("shape", [2, 3, 7, 8, 15, 16])
def test_givens_decomposition(shape):
    r"""Test that Givens decomposition is performed correctly."""

    matrix = unitary_group.rvs(shape)
    phase_mat, ordered_rotations = givens_decomposition(matrix)
    decomposed_matrix = np.diag(phase_mat)

    for grot_mat, (i, j) in ordered_rotations:
        rotation_matrix = np.eye(shape, dtype=complex)
        rotation_matrix[i, i], rotation_matrix[j, j] = grot_mat[0, 0], grot_mat[1, 1]
        rotation_matrix[i, j], rotation_matrix[j, i] = grot_mat[0, 1], grot_mat[1, 0]
        decomposed_matrix = decomposed_matrix @ rotation_matrix

    assert np.allclose(matrix, decomposed_matrix)
