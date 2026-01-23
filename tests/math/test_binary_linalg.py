# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for binary linear algebra functions in qml.math"""
import numpy as np
import pytest

from pennylane import math as fn


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

    # get reduced row echelon form from the implemented function
    rref_bin_mat = fn.binary_finite_reduced_row_echelon(binary_matrix)

    assert (rref_bin_mat == row_echelon_matrix).all()
    assert (rref_bin_mat == result).all()


class TestBinaryRank:

    @pytest.mark.parametrize(
        "binary_matrix, expected",
        [
            (np.eye(2, dtype=int), 2),
            (np.eye(3, dtype=int), 3),
            (np.eye(17, dtype=int), 17),
            (np.array([[0, 1], [1, 1]]), 2),
            (np.array([[0, 1], [0, 1]]), 1),
            (np.array([[0, 0], [1, 1]]), 1),
            (np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0]]), 2),
            (np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]]), 2),
            (np.eye(100, dtype=int)[:63], 63),
            (np.eye(100, dtype=int)[:, :63], 63),
        ],
    )
    def test_square_matrix(self, binary_matrix, expected):
        """Test that the binary_rank function correctly computes
        the rank over Z_2 for a square matrix."""
        rk = fn.binary_rank(binary_matrix)
        assert rk == expected

    @pytest.mark.parametrize(
        "binary_matrix, expected",
        [
            (np.concatenate([np.eye(2, dtype=int), [[0, 1]]]), 2),
            (np.concatenate([np.eye(2, dtype=int), [[0], [1]]], axis=1), 2),
            (np.array([[0, 1], [1, 1], [1, 0], [0, 1]]), 2),
            (np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0], [0, 1, 1]]), 2),
            (np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0], [0, 1, 1]]).T, 2),
            (np.zeros((4, 9), dtype=int), 0),
        ],
    )
    def test_rectangular_matrix(self, binary_matrix, expected):
        """Test that the binary_rank function correctly computes
        the rank over Z_2 for a rectangular (non-square) matrix."""
        rk = fn.binary_rank(binary_matrix)
        assert rk == expected

    @pytest.mark.parametrize("shape", [(2, 3), (4, 7), (5, 93), (14, 4), (100, 7)])
    def test_with_random_matrix_against_rref(self, shape, seed):
        """Test that the rank computed with ``binary_rank`` matches the
        rank inferred from the reduced row echelon form of a random matrix."""
        rng = np.random.default_rng(seed)
        binary_matrix = rng.choice(2, size=shape, replace=True)
        rk_direct = fn.binary_rank(binary_matrix)

        rref = fn.binary_finite_reduced_row_echelon(binary_matrix)
        rk_from_rref = np.sum(np.any(rref, axis=1))
        assert rk_direct == rk_from_rref
