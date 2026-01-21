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
Unit tests for the SumOfSlatersStatePreparation template.
"""
from itertools import product

import numpy as np
import pytest

from pennylane.math import ceil_log2

# import pennylane as qml
# from pennylane import numpy as pnp
from pennylane.templates.state_preparations.sum_of_slaters import (
    _columns_differ,
    _find_ell,
    _get_bits_basis,
    _rank_over_z2,
    _select_rows,
    compute_sos_encoding,
)


def _is_binary(x: np.ndarray) -> bool:
    """Return whether all entries of a numpy array are binary."""
    return set(x.flat).issubset({np.int64(0), np.int64(1)})


def _random_regular_matrix(n, random_ops, seed: int):
    """Create a random regular (=non-singular) binary matrix.
    This is done by performing random row additions on the identity matrix, preserving
    the regularity of the identity matrix itself.

    In the picture of quantum circuits, we are computing the parity matrix of a random CNOT
    circuit.
    """
    rs = np.random.RandomState(seed)
    P = np.eye(n, dtype=int)
    for _ in range(random_ops):
        i, j = rs.choice(n, size=2, replace=False)  # Random pair of rows
        P[i] += P[j]  # Add second sampled row to first sampled row
    return P % 2  # Make into binary matrix


def _random_distinct_bitstrings(num_bits, num_strings, seed):
    """Create a numpy array of ``num_strings`` distinct bit strings of
    length ``num_bits``. The output size is ``(num_bits, num_strings)``,
    i.e. the bit strings are stored as columns."""
    rs = np.random.RandomState(seed)
    # Sample random integers
    ints = rs.choice(2**num_bits, size=num_strings, replace=False)
    # Convert integers to bitstrings
    bitstrings = ((ints[:, None] >> np.arange(num_bits - 1, -1, -1)[None, :]) % 2).T
    assert _columns_differ(bitstrings)  # Validate that columns are distinct
    return bitstrings


class TestHelperFunctions:

    @pytest.mark.parametrize(
        "bits, expected",
        [
            (np.eye(2, dtype=int), True),
            (np.array([[0, 1], [1, 1]]), True),
            (np.array([[0, 1], [0, 1]]), True),
            (np.array([[0, 0], [1, 1]]), False),
            (np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]]), True),
        ],
    )
    def test_columns_differ(self, bits, expected):
        """Test the _columns_differ helper function."""
        assert _columns_differ(bits) is expected

    @pytest.mark.parametrize(
        "bits ",
        [
            np.array([[0, 0, 1, 1], [0, 1, 0, 1]]),
            np.array([[0, 0, 1], [0, 1, 0]]),
            np.array([[0, 0, 1, 1], [0, 0, 1, 0], [0, 1, 1, 1]]),
        ],
    )
    def test_select_rows_need_all_rows(self, bits):
        """Test that _select_rows correctly selects all rows if they are all needed to
        discrimnate the columns of the input"""
        selectors, new_bits = _select_rows(bits)
        assert set(selectors) == set(range(len(bits)))  # all rows are needed
        assert np.allclose(new_bits, bits)

    @pytest.mark.parametrize(
        "bits, skip_rows",
        [
            (np.array([[0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 0, 1]]), [1]),
            (np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), [2]),
            (np.array([[0, 0, 1, 1], [0, 0, 1, 0], [0, 1, 1, 0]]), [1]),
            (
                np.array(
                    [
                        [0, 0, 0, 0, 1, 1, 1, 1],
                        [0, 0, 1, 1, 0, 0, 1, 1],
                        [0, 1, 0, 1, 0, 1, 0, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1],
                    ]
                ),
                [3],
            ),
            (
                np.array(
                    [
                        [0, 0, 1, 1],
                        [0, 1, 1, 1],
                        [1, 1, 1, 1],
                        [0, 0, 0, 1],
                        [1, 0, 1, 0],
                        [0, 0, 1, 1],
                    ]
                ),
                [1, 2, 3, 5],
            ),
            (
                np.array(
                    [
                        [0, 0, 1, 1],
                        [1, 0, 1, 0],
                        [0, 1, 1, 1],
                        [1, 1, 1, 1],
                        [0, 0, 0, 1],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1],
                    ]
                ),
                [2, 3, 4, 5, 6],
            ),
            (
                np.array(
                    [
                        [0, 1, 1, 1],
                        [1, 1, 1, 1],
                        [0, 0, 0, 1],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [1, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                ),
                [0, 1, 2, 4, 6],
            ),
        ],
    )
    def test_select_rows_need_only_few_rows(self, bits, skip_rows):
        """Test that _select_rows correctly selects a subset of rows if they are not all needed to
        discrimnate the columns of the input"""
        selectors, new_bits = _select_rows(bits)
        assert set(selectors) == set(range(len(bits))) - set(skip_rows)
        assert np.allclose(new_bits, bits[np.array(selectors)])

    @pytest.mark.parametrize(
        "bits, expected",
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
    def test_rank(self, bits, expected):
        """Test that the _rank helper function correctly computes the rank over Z_2."""
        rk = _rank_over_z2(bits)
        assert rk == expected

    def test_rank_over_z2_error(self):
        """Test that an error is raised if more than 64 bits are required to represent the bit
        strings in the input to _rank_over_z2."""
        with pytest.raises(NotImplementedError, match="smaller than or equal to 64"):
            _ = _rank_over_z2(np.eye(65, dtype=int))

    @pytest.mark.parametrize(
        "bits",
        [
            np.array([[0, 1, 0], [1, 0, 1]]),
            np.array([[0, 1, 1], [1, 0, 1]]),
            np.array([[1, 1, 1], [1, 0, 1]]),
            _random_regular_matrix(4, 14, seed=5214),
            _random_regular_matrix(15, 174, seed=514),
            _random_regular_matrix(55, 1074, seed=14),
            np.concatenate(
                [
                    _random_regular_matrix(15, 615, seed=9185),
                    np.random.default_rng(852).choice(2, size=(15, 85)),
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    np.random.default_rng(52).choice(2, size=(32, 64)),
                    _random_regular_matrix(32, 5615, seed=985),
                ],
                axis=1,
            ),
        ],
    )
    def test_get_bits_basis(self, bits):
        """Test that _get_bits_basis can be used to compute a basis over
        Z_2 out of overcomplete bit strings."""
        r, D = bits.shape
        basis, bits_other_than_last_basis_vec = _get_bits_basis(bits)
        assert basis.shape == (r, r)
        assert _is_binary(basis)
        assert _rank_over_z2(basis) == r

        assert bits_other_than_last_basis_vec.shape == (r, D - 1)
        assert _is_binary(bits_other_than_last_basis_vec)

    @pytest.mark.parametrize(
        "r, len_N, len_M, seed",
        [
            (5, 2, 4, 9214),
            (9, 10, 13, 214),
            (16, 8, 2, 1),
            (16, 0, 2, 10),
            (16, 31, 0, 12),
        ],
    )
    def test_find_ell(self, r, len_N, len_M, seed):
        """Test that _find_ell can find a vector spanned by a given basis that is not colliding
        with any vector in a combination of two existing sets of bitstrings, set_M and set_N.

        set_M is supposed to consists of any distinct bitstrings that can be spanned by all but
        the last basis vector. set_N in turn consists of bitstrings that require the last basis
        vector in their decomposition.

        For example, we might have
        basis =
        | 1 0 0 0 |
        | 0 1 0 1 |
        | 0 0 0 1 |
        | 0 0 1 0 |

        as well as
        set_M =        set_N =
        | 1 1 0 1 |      | 1 |
        | 1 0 1 0 |      | 0 |
        | 0 0 0 0 |      | 1 |
        | 0 0 0 1 |      | 0 |

        We see that set_M consists of columns composed from the first two columns of basis.
        set_N however requires the last column of basis, which we call v_r.
        The task is then to produce a column composed of the first three basis vectors that does
        not match anything in set_M, any column in set_N + v_r (add v_r to each column) or any
        column in set_M + set_N + v_r (add each combination of set elements), as well as the
        zeros-column. That is, we need to dodge
        avoid =
        | 1 1 0 1   1    0 0 1 0  0 |    | 1 1 0 1 0 0 |
        | 1 0 1 0   1    0 1 0 1  0 | ≡  | 1 0 1 0 0 1 |
        | 0 0 0 0   0    0 0 0 0  0 |    | 0 0 0 0 0 0 |
        | 0 0 0 1   0    0 0 0 1  0 |    | 0 0 0 1 0 1 |
          ┕--M--┙ N+v_r ┕M+N+v_r┙ zeros

        A solution to this is the column
        ell =
        | 1 |
        | 1 |
        | 0 |
        | 1 |
        """
        # The following is a necessary condition for a vector ell with desired properties to exist
        # in the first place
        avoid_size = 1 + len_N + len_M + len_M * len_N
        assert avoid_size < 2 ** (r - 1), f"{avoid_size=}, {2**(r-1)=}"

        basis = _random_regular_matrix(r, r**2, seed=seed)
        set_M = _random_distinct_bitstrings(r - 1, len_M, seed=seed + 8512)
        # Convert random bits to basis
        set_M = (basis[:, :-1] @ set_M) % 2

        set_N = _random_distinct_bitstrings(r - 1, len_N, seed=seed + 52)
        # Convert random bits to basis and add last basis vector.
        set_N = (basis[:, :-1] @ set_N + basis[:, -1:]) % 2

        ell = _find_ell(set_M, set_N, basis)
        # Bitstrings that need to be avoided
        shifted_set_N = (set_N + basis[:, -1:]) % 2

        combs = np.array([a + b for a, b in product(set_M.T, shifted_set_N.T)]).T
        if len(combs) == 0:
            combs = combs.reshape((r, 0))
        avoid = (
            np.concatenate([set_M, shifted_set_N, np.zeros((r, 1), dtype=int), combs], axis=1) % 2
        )
        matches = avoid == ell[:, None]
        # Assert that the found vector ell is indeed different from all vectors to be avoided.
        assert not np.any(np.all(matches, axis=0))


class TestComputeSosEncoding:

    @pytest.mark.parametrize("r, D", [(3, 3), (4, 5), (9, 17), (8, 32)])
    def test_trivial_case(self, r, D):
        """Test ``compute_sos_encoding`` for cases where the number ``D`` of
        input bit strings and their length ``r`` satisfies ``r<=2⌈log_2(D)⌉-1``.
        In this case, the encoding is the identity matrix.
        """
        assert r <= 2 * ceil_log2(D) - 1  # Test case input validation

        bits = _random_distinct_bitstrings(r, D, 2519)
        U, b = compute_sos_encoding(bits)
        assert np.allclose(U, np.eye(r))
        assert np.allclose(b, bits)

    @pytest.mark.parametrize("r, D", [(8, 16), (10, 17), (19, 20), (23, 1052), (22, 1521)])
    def test_nontrivial_case(self, r, D):
        """Test ``compute_sos_encoding`` for cases where the number ``D`` of
        input bit strings and their length ``r`` satisfies ``r>2⌈log_2(D)⌉-1``.
        In this case, the encoding is the identity matrix.
        """
        m = 2 * ceil_log2(D) - 1
        assert r > m  # Test case input validation
        # The algorithm assumes that we do not use more bits than there are linearly
        # independent bitstrings. We re-sample if this is not satisfied by the random bits.
        rk = r - 1
        while rk != r:
            bits = _random_distinct_bitstrings(r, D, 519)
            rk = _rank_over_z2(bits)

        U, b = compute_sos_encoding(bits)
        assert U.shape == (m, r)
        assert _is_binary(U)
        assert b.shape == (m, D)
        assert _is_binary(b)
        assert np.allclose((U @ bits) % 2, b)
        assert _columns_differ(b)
