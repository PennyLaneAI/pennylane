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


def _make_random_regular_matrix(n, random_ops, seed):
    """Create a random regular (=non-singular) binary matrix.
    This is done by performing random row additions on the identity matrix, preserving
    the regularity of the identity matrix itself.

    In the picture of quantum circuits, we are computing the parity matrix of a random CNOT
    circuit.
    """
    rng = np.random.default_rng(seed)
    P = np.eye(n, dtype=int)
    for _ in range(random_ops):
        i, j = rng.choice(n, size=2, replace=False)  # Random pair of rows
        P[i] += P[j]  # Add second sampled row to first sampled row
    return P % 2  # Make into binary matrix


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

    assert rref_bin_mat.shape == binary_matrix.shape
    assert set(rref_bin_mat.flat).issubset({0, 1})
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


class TestSolveBinaryLinearSystem:
    """Tests for the helper method ``binary_solve_linear_system``."""

    @pytest.mark.parametrize(
        "A, b, expected",
        [
            (np.eye(2, dtype=int), np.array([0, 0]), np.array([0, 0])),
            (np.eye(4, dtype=int), np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0])),
            (np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]]), np.array([0, 0, 1]), np.array([1, 1, 1])),
            (np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0]]), np.array([1, 1, 1]), np.array([1, 1, 1])),
            (np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0]]), np.array([0, 1, 0]), np.array([1, 0, 1])),
            (
                np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 1, 1]]),
                np.array([1, 0, 0, 0]),
                np.array([1, 0, 0, 0]),
            ),
            (
                np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 1, 1]]),
                np.array([1, 0, 1, 1]),
                np.array([0, 1, 0, 0]),
            ),
            (
                np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 1, 1]]),
                np.array([1, 1, 1, 1]),
                np.array([0, 0, 1, 0]),
            ),
            (
                np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 1, 1]]),
                np.array([1, 1, 0, 1]),
                np.array([0, 1, 1, 1]),
            ),
        ],
    )
    def test_with_regular_matrix(self, A, b, expected):
        """Test that the system is solved for a regular matrix and coefficient vector b."""
        # Input validation
        assert fn.binary_rank(A) == len(A)  # Regular matrix
        assert np.allclose((A @ expected) % 2, b)  # expected is a solution

        x_sol = fn.binary_solve_linear_system(A, b)
        assert x_sol.shape == b.shape and x_sol.dtype == np.int64
        assert set(x_sol).issubset({0, 1})
        assert np.allclose(x_sol, expected)  # Solution unique due to regularity, x_sol must match

    @pytest.mark.parametrize(
        "A, b",
        [
            (np.array([[1, 0], [0, 0]]), np.array([0, 1])),
            (np.array([[1, 0], [1, 0]]), np.array([1, 0])),
        ],
    )
    def test_with_singular_matrix_error(self, A, b):
        """Test that an error is raised if a linear system has a singular matrix."""
        with pytest.raises(np.linalg.LinAlgError, match="Singular binary matrix"):
            _ = fn.binary_solve_linear_system(A, b)

    @pytest.mark.parametrize("n", [1, 4, 5, 10, 13, 28, 102])
    def test_with_random_regular_matrix(self, n, seed):
        """Test that the system is solved correctly for a random regular matrix
        of given dimension and reverse-engineered coefficient vector b."""
        random_ops = n**2 if n > 1 else 0  # Can't do row operations in 1D.
        A = _make_random_regular_matrix(n, random_ops, seed)
        x = np.random.default_rng(seed + 1).integers(0, 2, size=(n,))
        b = (A @ x) % 2
        x_sol = fn.binary_solve_linear_system(A, b)
        assert x_sol.shape == (n,) and x_sol.dtype == np.int64
        assert set(x_sol).issubset({0, 1})
        assert np.allclose((A @ x_sol) % 2, b)
        assert np.allclose(x_sol, x)


class TestBinaryIsIndependent:

    def test_error_shape_mismatch(self):
        """Test that for mismatched shapes there is an error raised."""
        vector = np.array([1, 0])
        basis = np.eye(3, dtype=int)
        with pytest.raises(ValueError, match="columns of `basis` should have the same length"):
            fn.binary_is_independent(vector, basis)

    @pytest.mark.parametrize(
        "vector, basis, expected",
        [
            (np.array([0, 1]), np.array([[1], [0]]), True),
            (np.array([1, 0]), np.array([[1], [0]]), False),
            (np.array([1, 0]), np.eye(2, dtype=int), False),
            (np.array([0, 0, 1]), np.array([[1, 0], [0, 1], [1, 0]]), True),
            (np.array([0, 0, 1]), np.array([[1, 1], [0, 0], [1, 0]]), False),
            (np.array([0, 0, 1]), np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1]]), False),
            (np.array([0, 0, 1, 0]), np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0]]), True),
            (np.array([0, 0, 1, 1]), np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0]]), False),
            (
                np.array([0, 0, 1, 0]),
                np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 0], [0, 1, 0, 1]]),
                False,
            ),
        ],
    )
    def test_independence(self, vector, basis, expected):
        """Test that linear independence and dependence are recognized correctly."""
        # input validation. This equality is an assumption of ``binary_is_independent``.
        assert fn.binary_rank(basis) == min(basis.shape)

        is_indep = fn.binary_is_independent(vector, basis)
        assert is_indep is expected
