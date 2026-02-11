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
"""
Unit tests for the SumOfSlatersPrep template.
"""
from itertools import combinations, product

import numpy as np
import pytest

import pennylane as qml
from pennylane.decomposition import list_decomps
from pennylane.math import binary_matrix_rank, ceil_log2
from pennylane.ops.functions import assert_valid
from pennylane.templates.state_preparations.sum_of_slaters import (
    SumOfSlatersPrep,
    _columns_differ,
    _find_ell,
    _find_single_w,
    _int_to_binary,
    _sos_state_prep,
    compute_sos_encoding,
    select_sos_rows,
)


def _is_binary(x: np.ndarray) -> bool:
    """Return whether all entries of a numpy array are binary."""
    return set(x.flat).issubset({0, 1})


def random_distinct_integers(high, size, rng):

    if high < 2**25:
        return rng.choice(high, size=size, replace=False)

    samples = set()
    while len(samples) < size:
        samples.add(int(rng.integers(high)))
    return np.array(list(samples), dtype=int)


def _random_regular_matrix(n, random_ops, seed: int):
    """Create a random regular (=non-singular) binary matrix.
    This is done by performing random row additions on the identity matrix, preserving
    the regularity of the identity matrix itself.

    In the picture of quantum circuits, we are computing the parity matrix of a random CNOT
    circuit.
    """
    rng = np.random.default_rng(seed)
    P = np.eye(n, dtype=int)
    for _ in range(random_ops):
        i, j = random_distinct_integers(n, 2, rng)  # Random pair of rows
        P[i] += P[j]  # Add second sampled row to first sampled row
    return P % 2  # Make into binary matrix


def random_distinct_bitstrings(num_bits, num_strings, seed, full_rank=False):
    """Create a numpy array of ``num_strings`` distinct bit strings of
    length ``num_bits``. The output size is ``(num_bits, num_strings)``,
    i.e. the bit strings are stored as columns.
    If ``full_rank=True`` is specified in addition, make sure that the bits span the full
    space of ``num_bits`` bit strings.

    """
    rng = np.random.default_rng(seed)

    # Sample fewer unconstrained bit strings if we want full rank. We will insert a regular random
    # matrix to ensure the full rank.
    num_samples = num_strings - num_bits if full_rank else num_strings
    # Sample random integers
    ints = random_distinct_integers(2**num_bits, num_samples, rng)
    # Convert integers to bitstrings
    bitstrings = ((ints[:, None] >> np.arange(num_bits - 1, -1, -1)[None, :]) % 2).T

    if full_rank:
        # If we want full rank, we sample a random regular matrix and shuffle it into the
        # unconstrained random samples from above
        assert num_strings >= num_bits
        regular_part = _random_regular_matrix(num_bits, random_ops=num_bits**2, seed=seed)
        bitstrings = np.concatenate([regular_part, bitstrings], axis=1)
        rng.shuffle(bitstrings, axis=1)

    assert _columns_differ(bitstrings)  # Validate that columns are distinct

    if full_rank:
        assert binary_matrix_rank(bitstrings) == num_bits
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
            (np.eye(64, dtype=int), True),
        ],
    )
    def test_columns_differ(self, bits, expected):
        """Test the _columns_differ helper function."""
        assert _columns_differ(bits) is expected

    @pytest.mark.parametrize("size", [65, 100])
    def test_columns_differ_error(self, size):
        """Test that an error is raised for bitstrings that are too large."""
        bits = np.ones((size, 4), dtype=int)
        bits[:4, :] = np.eye(4, dtype=int)
        with pytest.raises(ValueError, match="Column comparison uses 64-bit integers internally."):
            _columns_differ(bits)

    @pytest.mark.parametrize(
        "bits ",
        [
            np.array([[0, 0, 1, 1], [0, 1, 0, 1]]),
            np.array([[0, 0, 1], [0, 1, 0]]),
            np.array([[0, 0, 1, 1], [0, 0, 1, 0], [0, 1, 1, 1]]),
        ],
    )
    def test_select_sos_rows_need_all_rows(self, bits):
        """Test that select_sos_rows correctly selects all rows if they are all needed to
        discrimnate the columns of the input"""
        selectors, new_bits = select_sos_rows(bits)
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
    def test_select_sos_rows_need_only_few_rows(self, bits, skip_rows):
        """Test that select_sos_rows correctly selects a subset of rows if they are not all needed to
        discrimnate the columns of the input"""
        selectors, new_bits = select_sos_rows(bits)
        assert set(selectors) == set(range(len(bits))) - set(skip_rows)
        assert np.allclose(new_bits, bits[np.array(selectors)])

    @pytest.mark.parametrize("num_bits", [1, 2, 5])
    def test_select_sos_rows_single_column(self, num_bits):
        """Test that the edge case of a single bitstring is handled correctly."""
        bits = np.array([[0], [1], [0], [1], [1]])[:num_bits]
        selectors, new_bits = select_sos_rows(bits)
        assert selectors == [0]
        assert np.array_equal(new_bits, np.array([[0]]))

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
        set_M = random_distinct_bitstrings(r - 1, len_M, seed=seed + 8512)
        # Convert random bits to basis
        set_M = (basis[:, :-1] @ set_M) % 2

        set_N = random_distinct_bitstrings(r - 1, len_N, seed=seed + 52)
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

    @pytest.mark.parametrize(
        "bits",
        [
            np.array([[1], [0]]),
            np.array([[1, 0], [0, 1], [1, 1]]),
            np.array([[1, 0], [0, 1], [1, 1]]),
            np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]]),
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        ],
    )
    def test_find_single_w(self, bits):
        """Test _find_single_w."""
        copy = bits.copy()
        W = _find_single_w(bits)
        assert np.allclose(bits, copy)  # Input not altered
        assert isinstance(W, np.ndarray) and W.shape == (len(bits), 1)
        assert _is_binary(W)
        # Assert that the newly found vector indeed differs from the inputs
        assert _columns_differ(np.concatenate([bits, W], axis=1))
        # Assert that the newly found vector indeed differs from the differences of inputs
        assert not any(np.allclose((b0 + b1) % 2, W) for b0, b1 in combinations(bits.T, r=2))
        # Assert that the newly found vector is not the zero vector
        assert not np.allclose(W, 0)

    def test_find_single_w_too_many_bits(self, seed):
        """Test _find_single_w raises an error for bitstrings longer than 64 bits."""
        bits = np.concatenate(
            [random_distinct_bitstrings(30, 13, seed), random_distinct_bitstrings(35, 13, seed + 1)]
        )
        with pytest.raises(ValueError, match="Bitstring search _find_single_w"):
            _find_single_w(bits)


class TestComputeSosEncoding:
    """Tests for ``compute_sos_encoding``."""

    @pytest.mark.parametrize("r, D", [(3, 3), (4, 5), (9, 17), (8, 32)])
    def test_trivial_case(self, r, D):
        """Test ``compute_sos_encoding`` for cases where the number ``D`` of
        input bit strings and their length ``r`` satisfies ``r<=2⌈log_2(D)⌉-1``.
        In this case, the encoding is the identity matrix.
        """
        assert r <= 2 * ceil_log2(D) - 1  # Test case input validation

        bits = random_distinct_bitstrings(r, D, 2519)
        U, b = compute_sos_encoding(bits)
        assert np.allclose(U, np.eye(r))
        assert np.allclose(b, bits)

    @pytest.mark.parametrize(
        "r, D",
        [
            (4, 4),
            (6, 6),
            (8, 16),
            (10, 17),
            (11, 17),
            (12, 18),
            (19, 20),
            (53, 53),
            (23, 1052),
            (22, 1521),
        ],
    )
    def test_nontrivial_case(self, r, D):
        """Test ``compute_sos_encoding`` for cases where the number ``D`` of
        input bit strings and their length ``r`` satisfies ``r>2⌈log_2(D)⌉-1``.
        """

        m = 2 * ceil_log2(D) - 1
        # Test case input validation: Want nontrivial case and need D>=r for full rank input
        assert D >= r > m
        # The algorithm assumes that we do not use more bits than there are linearly
        # independent bitstrings. We make sure this is true by passing full_rank=True
        bits = random_distinct_bitstrings(r, D, 519, full_rank=True)

        U, b = compute_sos_encoding(bits)
        assert U.shape == (m, r)
        assert _is_binary(U)
        assert b.shape == (m, D)
        assert _is_binary(b)
        assert np.allclose((U @ bits) % 2, b)
        assert _columns_differ(b)

    @pytest.mark.parametrize(
        "r, D",
        [
            (4, 4),
            (6, 6),
            (8, 16),
            (10, 17),
            (11, 17),
            (12, 18),
            (19, 20),
            (53, 53),
            (23, 1052),
            (22, 1521),
        ],
    )
    def test_integration_with_select_sos_rows(self, r, D, seed):
        """Test that compute_sos_encoding integrates with select_sos_rows as intended."""

        bits = random_distinct_bitstrings(r, D, seed)
        selector_ids, new_bits = select_sos_rows(bits)

        new_r = len(new_bits)
        assert len(selector_ids) == new_r

        U, b = compute_sos_encoding(new_bits)
        m = min(2 * ceil_log2(D) - 1, new_r)
        assert U.shape == (m, new_r)
        assert _is_binary(U)
        assert b.shape == (m, D)
        assert _is_binary(b)
        assert np.allclose((U @ new_bits) % 2, b)
        assert _columns_differ(b)


class TestSumOfSlatersPrep:
    """Test the quantum template ``SumOfSlatersPrep``."""

    def make_random_data(self, num_wires, num_entries, seed):
        """Produce some random input data for ``SumOfSlatersPrep`` with given specs."""
        rng = np.random.default_rng(seed)
        coefficients = rng.random(num_entries)
        coefficients /= np.linalg.norm(coefficients)
        indices = tuple(rng.choice(2**num_wires, size=num_entries, replace=False))
        return coefficients, indices

    @pytest.mark.parametrize(
        "num_wires, num_entries",
        [(2, 1), (2, 2), (2, 4), (4, 3), (4, 6), (10, 3), (10, 137), (17, 1421)],
    )
    def test_standard_validity(self, num_wires, num_entries, seed):
        """Test that SumOfSlatersPrep is a valid PennyLane operator."""
        coefficients, indices = self.make_random_data(num_wires, num_entries, seed)
        wires = list(range(num_wires))
        op = SumOfSlatersPrep(coefficients, wires, indices=indices)
        assert_valid(op, skip_differentiation=True)

    def test_old_decomposition_system_disabled(self):
        """We are using ``qml.allocate`` in the decomposition, so the validation for
        decomposition in the old system breaks. Hence we manually deactivated the fallback
        of compute_decomposition to the new decomp system."""
        num_wires = 5
        coefficients, indices = self.make_random_data(num_wires, 13, seed=141)
        wires = list(range(num_wires))
        op = SumOfSlatersPrep(coefficients, wires, indices=indices)
        # In this case, assert_valid actually asserts that compute_decomposition raises an error.
        assert op.has_decomposition is False

    @pytest.mark.parametrize("num_wires", [3, 4, 5])
    @pytest.mark.parametrize("num_entries", [1, 4, 5, 6])
    @pytest.mark.parametrize("num_bits", [4, 5, 6])
    def test_register_sizes(self, num_wires, num_entries, num_bits, seed):
        """Test for ``SumOfSlatersPrep.required_register_sizes`` and work wire spec of
        ``_sos_state_prep``."""

        coefficients, _ = self.make_random_data(num_wires, num_entries, seed=seed)

        # In this test we do a bit of gymnastics: We first create random distinct bitstrings,
        # then make them less redundant via select_sos_rows, and then create the `indices` to
        # be passed into SumOfSlatersPrep. The effective `num_bits` is then given by the length
        # of the less redundant bitstrings, rather than the test case input `num_bits`.
        bits = random_distinct_bitstrings(min(num_bits, num_wires), num_entries, seed)
        reduced_bits = select_sos_rows(bits)[1]
        num_bits = len(reduced_bits)
        indices = 2 ** np.arange(num_bits - 1, -1, -1) @ reduced_bits

        sizes = SumOfSlatersPrep.required_register_sizes(num_entries, num_bits, num_wires)
        d = ceil_log2(num_entries)
        assert sizes["wires"] == num_wires
        assert sizes["enumeration_wires"] == d
        assert sizes["identification_wires"] == max((num_bits > 2 * d - 1) * (2 * d - 1), 0)
        assert sizes["qrom_work_wires"] == max(d - 1, 0)
        assert sizes["mcx_work_wires"] == max(min(num_bits, 2 * d - 1), 0)

        op = SumOfSlatersPrep(coefficients, range(num_wires), indices)
        exp_resource_params = {"D": num_entries, "num_bits": num_bits, "num_wires": num_wires}
        assert exp_resource_params == op.resource_params

        registered_work_wires = _sos_state_prep.get_work_wire_spec(**exp_resource_params)
        assert sum(sizes.values()) - num_wires == registered_work_wires.total

    def test_int_to_binary(self, seed):
        """Test for ``_int_to_binary`` used in SumOfSlatersPrep decomposition."""
        rng = np.random.default_rng(seed)
        x = rng.choice(2**12, size=174, replace=False)
        out_full_width = _int_to_binary(x, 12)
        powers = 2 ** np.arange(11, -1, -1)
        assert np.allclose(powers @ out_full_width, x)

        out_slightly_too_small_width = _int_to_binary(x, 11)
        powers = 2 ** np.arange(10, -1, -1)
        assert np.allclose(powers @ out_slightly_too_small_width, x % 2**11)

        out_way_too_small_width = _int_to_binary(x, 6)
        powers = 2 ** np.arange(5, -1, -1)
        assert np.allclose(powers @ out_way_too_small_width, x % 2**6)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize(
        "num_wires,num_entries",
        [(3, 1), (3, 2), (3, 3), (4, 3), (4, 15), (5, 4), (5, 21), (10, 63), (10, 123)],
    )
    def test_decomposition_prepares_state(self, num_wires, num_entries, seed):
        """Test that the decomposition of SumOfSlatersPrep actually prepares the desired state."""

        coefficients, indices = self.make_random_data(num_wires, num_entries, seed=seed)

        for rule in list_decomps(SumOfSlatersPrep):

            @qml.qnode(qml.device("default.qubit"))
            def func():
                # pylint: disable=cell-var-from-loop
                # Make sure that the output state length is at least 2**num_wires
                qml.Identity(range(num_wires))
                rule(coefficients, wires=range(num_wires), indices=indices)
                return qml.state()

            out_state = func()

            # We infer the total and aux wire counts from the state shape, because small-scale
            # edge cases often have fewer work wires than the general case.
            num_all_wires = ceil_log2(out_state.shape[0])
            num_aux_wires = num_all_wires - num_wires
            for _ in range(num_aux_wires):
                assert np.allclose(out_state[1::2], 0.0), "\n".join(
                    [
                        f"{a} : {b}"
                        for a, b in zip(np.where(out_state)[0], out_state[np.where(out_state)])
                    ]
                )
                out_state = out_state[::2]
            assert np.allclose([out_state[key] for key in indices], coefficients)
