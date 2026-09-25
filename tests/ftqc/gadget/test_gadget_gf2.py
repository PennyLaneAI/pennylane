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

"""Unit tests for the GF(2) linear algebra used by ``pennylane.ftqc.gadget``."""

import numpy as np
import pytest

from pennylane.ftqc.gadget import _gf2

STEANE_H = np.array(
    [
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1],
    ],
    dtype=np.uint8,
)

REP3 = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)


class TestAsBits:
    """Tests for coercion to GF(2) matrices."""

    def test_vector_becomes_row(self):
        """Test that a 1-D input is promoted to a single row."""
        out = _gf2.as_bits([1, 0, 1])
        assert out.shape == (1, 3)
        assert out.dtype == np.uint8

    def test_empty_with_column_count(self):
        """Test that an empty input is allowed when the column count is given."""
        assert _gf2.as_bits([], 4).shape == (0, 4)

    def test_empty_needs_column_count(self):
        """Test that an empty input without a column count is rejected."""
        with pytest.raises(ValueError, match="explicit column count"):
            _gf2.as_bits([])

    def test_rejects_non_binary_entries(self):
        """Test that entries outside {0, 1} are rejected."""
        with pytest.raises(ValueError, match="must be 0 or 1"):
            _gf2.as_bits([[0, 2]])

    def test_rejects_higher_rank_arrays(self):
        """Test that inputs with more than two dimensions are rejected."""
        with pytest.raises(ValueError, match="expected a 2-D matrix"):
            _gf2.as_bits(np.zeros((2, 2, 2)))


class TestRank:
    """Tests for row reduction and rank."""

    def test_row_reduce_pivots(self):
        """Test the reduced form and pivots of the distance-3 repetition checks."""
        rref, pivots = _gf2.row_reduce(REP3)
        assert pivots == [0, 1]
        assert np.array_equal(rref, np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8))

    @pytest.mark.parametrize(
        "matrix, expected",
        [
            (STEANE_H, 3),
            (REP3, 2),
            (np.vstack([REP3, REP3[0] ^ REP3[1]]), 2),
            (np.zeros((0, 5), dtype=np.uint8), 0),
        ],
    )
    def test_rank(self, matrix, expected):
        """Test the GF(2) rank, including dependent rows and empty matrices."""
        assert _gf2.rank(matrix) == expected


class TestRowSpaceAndSolve:
    """Tests for row-space membership and linear solves."""

    def test_zero_vector_always_in_row_space(self):
        """Test that the zero vector is in the row space of any matrix, even an empty one."""
        assert _gf2.in_row_space(np.zeros(3, dtype=np.uint8), np.zeros((0, 3), dtype=np.uint8))

    def test_nonzero_vector_not_in_empty_row_space(self):
        """Test that no nonzero vector is in the row space of an empty matrix."""
        assert not _gf2.in_row_space([1, 0, 0], np.zeros((0, 3), dtype=np.uint8))

    def test_in_row_space(self):
        """Test membership for a combination of rows and for a vector outside the span."""
        assert _gf2.in_row_space([1, 0, 1], REP3)
        assert not _gf2.in_row_space([1, 0, 0], REP3)

    def test_solve_recovers_coefficients(self):
        """Test that the returned coefficients reproduce the target."""
        target = np.array([1, 0, 1], dtype=np.uint8)
        coeffs = _gf2.solve(REP3, target)
        assert np.array_equal(coeffs @ REP3 % 2, target)

    def test_solve_unsolvable(self):
        """Test that a target outside the row space has no solution."""
        assert _gf2.solve(REP3, [1, 0, 0]) is None

    def test_solve_with_no_rows(self):
        """Test solving against an empty matrix: only the zero target is reachable."""
        empty = np.zeros((0, 3), dtype=np.uint8)
        assert _gf2.solve(empty, [0, 0, 0]).shape == (0,)
        assert _gf2.solve(empty, [0, 1, 0]) is None


class TestNullSpace:
    """Tests for the null space."""

    @pytest.mark.parametrize("matrix", [STEANE_H, REP3])
    def test_null_space_is_annihilated(self, matrix):
        """Test that the null space has the right dimension and is annihilated."""
        basis = _gf2.null_space(matrix)
        assert basis.shape == (matrix.shape[1] - _gf2.rank(matrix), matrix.shape[1])
        assert not (matrix.astype(int) @ basis.T.astype(int) % 2).any()

    def test_null_space_of_no_rows(self):
        """Test that the null space of an empty matrix is the whole space."""
        assert np.array_equal(
            _gf2.null_space(np.zeros((0, 3), dtype=np.uint8)), np.eye(3, dtype=np.uint8)
        )


class TestSupportedCombinations:
    """Tests for the combinations of rows supported inside a mask."""

    def test_combination_inside_mask(self):
        """Test that only products with support inside the mask are returned."""
        mask = np.array([True, True, False])
        out = _gf2.supported_combinations(REP3, mask)
        assert np.array_equal(out, np.array([[1, 1, 0]], dtype=np.uint8))

    def test_full_mask_spans_rows(self):
        """Test that a full mask returns a basis of the row space."""
        out = _gf2.supported_combinations(REP3, np.ones(3, dtype=bool))
        assert _gf2.rank(out) == 2

    def test_no_combination(self):
        """Test that a mask no product fits inside gives an empty result."""
        out = _gf2.supported_combinations(REP3, np.array([True, False, False]))
        assert out.shape == (0, 3)

    def test_empty_rows(self):
        """Test that no rows give no combinations."""
        out = _gf2.supported_combinations(np.zeros((0, 3), dtype=np.uint8), np.ones(3, bool))
        assert out.shape == (0, 3)


class TestCommutesAndWeights:
    """Tests for the CSS commutation product and weights."""

    def test_steane_is_css(self):
        """Test that the Steane checks satisfy Hx Hz^T = 0."""
        assert not _gf2.commutes(STEANE_H, STEANE_H).any()

    def test_anticommuting_pair(self):
        """Test that X0 and Z0 anticommute."""
        assert _gf2.commutes(np.array([[1, 0]], np.uint8), np.array([[1, 0]], np.uint8)) == 1

    def test_commutes_with_empty(self):
        """Test the shape of the product when one side is empty."""
        out = _gf2.commutes(np.zeros((0, 3), np.uint8), REP3)
        assert out.shape == (0, 2)

    def test_weights(self):
        """Test row and column weights, including the empty case."""
        assert _gf2.row_weights(STEANE_H).tolist() == [4, 4, 4]
        assert _gf2.col_weights(STEANE_H).tolist() == [1, 1, 2, 1, 2, 2, 3]
        assert _gf2.row_weights(np.zeros((0, 3), np.uint8)).shape == (0,)
        assert _gf2.col_weights(np.zeros((0, 3), np.uint8)).tolist() == [0, 0, 0]
