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
Unit tests for the PartialUnaryStatePreparation template.
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane.decomposition import list_decomps
from pennylane.math import binary_matrix_rank, ceil_log2
from pennylane.ops.functions import assert_valid
from pennylane.templates.state_preparations.partial_unary import PartialUnaryStatePreparation


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

    if full_rank:
        assert binary_matrix_rank(bitstrings) == num_bits
    return bitstrings


class TestPartialUnaryStatePreparation:
    """Test the quantum template ``PartialUnaryStatePreparation``."""

    def make_random_data(self, num_wires, num_entries, seed):
        """Produce some random input data for ``PartialUnaryStatePreparation`` with given specs."""
        rng = np.random.default_rng(seed)
        coefficients = rng.random(num_entries)
        coefficients /= np.linalg.norm(coefficients)
        indices = tuple(rng.choice(2**num_wires, size=num_entries, replace=False))
        return coefficients, indices

    @pytest.mark.jax
    @pytest.mark.parametrize("provide_work_wires", [False, True])
    @pytest.mark.parametrize(
        "num_wires, num_entries",
        [(2, 1), (2, 2), (2, 4), (4, 3), (4, 6), (10, 3), (10, 10), (10, 137), (13, 1421)],
    )
    def test_standard_validity(self, num_wires, num_entries, seed, provide_work_wires):
        """Test that PartialUnaryStatePreparation is a valid PennyLane operator."""
        coefficients, indices = self.make_random_data(num_wires, num_entries, seed)
        wires = list(range(num_wires))
        if provide_work_wires:
            num_work_wires = max(qp.math.ceil_log2(num_entries) - 1, 1)
            work_wires = tuple(range(num_wires, num_wires + num_work_wires))
        else:
            work_wires = ()

        op = PartialUnaryStatePreparation(
            coefficients, wires, indices=indices, work_wires=work_wires
        )
        assert_valid(op, skip_differentiation=True)

    def test_old_decomposition_system_disabled(self):
        """We are using ``qp.allocate`` in the decomposition, so the validation for
        decomposition in the old system breaks. Hence we manually deactivated the fallback
        of compute_decomposition to the new decomp system."""
        num_wires = 5
        coefficients, indices = self.make_random_data(num_wires, 13, seed=141)
        wires = list(range(num_wires))
        op = PartialUnaryStatePreparation(coefficients, wires, indices=indices, work_wires=None)
        # In this case, assert_valid actually asserts that compute_decomposition raises an error.
        assert op.has_decomposition is False

    @pytest.mark.parametrize("provide_work_wires", [False, True])
    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize(
        "num_wires,num_entries",
        [(3, 1), (3, 2), (3, 3), (4, 3), (4, 15), (5, 4), (5, 21), (7, 63)],
    )
    def test_decomposition_prepares_state(self, num_wires, num_entries, seed, provide_work_wires):
        """Test that the decomposition of PartialUnaryStatePreparation actually prepares the desired state."""

        coefficients, indices = self.make_random_data(num_wires, num_entries, seed=seed)
        needed_work_wires = max(qp.math.ceil_log2(num_entries) - 1, 1)
        if provide_work_wires:
            num_work_wires = needed_work_wires
        else:
            num_work_wires = 0

        work_wires = list(range(num_wires, num_wires + num_work_wires))

        for j, rule in enumerate(list_decomps(PartialUnaryStatePreparation)):
            applicable = rule.is_applicable(num_entries, num_wires, num_work_wires)
            # If provide_work_wires=False/True (=> cast to 0/1), we expect the decomposition
            # rule with index 0/1 to be applicable.
            assert applicable is (j == int(provide_work_wires))
            if not applicable:
                continue

            @qp.qnode(qp.device("lightning.qubit", wires=num_wires + needed_work_wires))
            @qp.transforms.resolve_dynamic_wires(min_int=num_wires + num_work_wires)
            def func():
                # pylint: disable=cell-var-from-loop
                # Make sure that the output state length is at least 2**num_wires
                rule(coefficients, wires=range(num_wires), indices=indices, work_wires=work_wires)
                return qp.state()

            out_state = func()

            # We infer the total and aux wire counts from the state shape, because small-scale
            # edge cases often have fewer work wires than the general case.
            num_all_wires = ceil_log2(out_state.shape[0])
            num_aux_wires = num_all_wires - num_wires
            for _ in range(num_aux_wires):
                assert np.allclose(out_state[1::2], 0.0)
                out_state = out_state[::2]
            assert np.allclose([out_state[key] for key in indices], coefficients)
