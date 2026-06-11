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
Unit tests for the SumOfSlatersPrep2 template.
"""

# pylint: disable=missing-kwoa

import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.templates import SumOfSlatersPrep2
from pennylane.ops.functions import assert_valid


def make_registers(indices, num_wires):
    """Create wire registers required for a given set of indices and target
    wires for ``SumOfSlatersPrep2``. The size of the required registers is computed with
    ``SumOfSlatersPrep.required_register_sizes`` and ``qp.registers`` is used to produce the
    registers themselves. This function assumes consecutive integer wire labels."""
    return qp.registers(SumOfSlatersPrep2.required_register_sizes(indices, num_wires))


class TestSumOfSlatersPrep2:
    """Test the quantum template ``SumOfSlatersPrep2``."""

    def make_random_data(self, num_wires, num_entries, seed):
        """Produce some random input data for ``SumOfSlatersPrep2`` with given specs."""
        rng = np.random.default_rng(seed)
        coefficients = rng.random(num_entries)
        coefficients /= np.linalg.norm(coefficients)
        indices = tuple(rng.choice(2**num_wires, size=num_entries, replace=False))
        return coefficients, indices

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "num_wires, num_entries",
        [(2, 1), (2, 2), (2, 4), (4, 3), (4, 6), (10, 3), (10, 10), (10, 137), (13, 1421)],
    )
    def test_standard_validity(self, num_wires, num_entries, seed):
        """Test that SumOfSlatersPrep2 is a valid PennyLane operator."""
        coefficients, indices = self.make_random_data(num_wires, num_entries, seed)
        all_wires = make_registers(indices, num_wires)
        op = SumOfSlatersPrep2(coefficients, **all_wires, indices=indices)
        assert_valid(op, skip_differentiation=True)

    @pytest.mark.jax
    @pytest.mark.parametrize("n", [7, 9, 15, 16, 17])
    def test_standard_validity_non_id_encoding(self, n, seed):
        """Test that SumOfSlatersPrep2 is a valid PennyLane operator for non-identity
        encoding scenario."""
        coefficients, _ = self.make_random_data(n, n, seed=seed)
        # Create bits that force a non-identity encoding
        bits = np.eye(n, dtype=int)[: n - 1]
        np.random.seed(seed)
        np.random.shuffle(bits)
        num_bits = n - 1
        indices = tuple(2 ** np.arange(num_bits - 1, -1, -1) @ bits)
        all_wires = make_registers(indices, n)
        op = SumOfSlatersPrep2(coefficients, **all_wires, indices=indices)
        assert_valid(op, skip_differentiation=True)

    def test_old_decomposition_system_disabled(self):
        """We are using ``qp.allocate`` in the decomposition, so the validation for
        decomposition in the old system breaks. Hence we manually deactivated the fallback
        of compute_decomposition to the new decomp system."""
        num_wires = 5
        coefficients, indices = self.make_random_data(num_wires, 13, seed=141)
        all_wires = make_registers(indices, num_wires)
        op = SumOfSlatersPrep2(coefficients, **all_wires, indices=indices)
        # In this case, assert_valid actually asserts that compute_decomposition raises an error.
        assert op.has_decomposition is False

    @pytest.mark.parametrize("use_qjit", [False, True])
    @pytest.mark.parametrize(
        "num_wires,num_entries",
        [(3, 1), (3, 2), (3, 3), (4, 3), (4, 15), (5, 4), (5, 21), (7, 63)],
    )
    def test_decomposition_prepares_state(self, num_wires, num_entries, seed, use_qjit):
        """Test that the decomposition of SumOfSlatersPrep2 actually prepares the desired state."""
        # pylint: disable=unsubscriptable-object
        if num_entries == 63 and use_qjit:
            pytest.skip(
                reason="This test case takes over a minute and does not provide unique value"
            )

        coefficients, indices = self.make_random_data(num_wires, num_entries, seed=seed)
        wires = list(range(num_wires))
        all_wires = make_registers(indices, num_wires)

        with qp.decomposition.toggle_graph_ctx(
            True
        ):  # safe alternative to avoid enabling graph globally on the labs test runner

            for rule in qp.list_decomps(SumOfSlatersPrep2):

                @qp.qnode(qp.device("lightning.qubit"))
                def func(coefficients):
                    # pylint: disable=cell-var-from-loop
                    # Make sure that the output state length is at least 2**num_wires
                    qp.Identity(wires)
                    rule(coefficients, **all_wires, indices=indices)
                    return qp.state()

                if use_qjit:
                    func = qp.qjit(func)

                out_state = func(coefficients)

                # We infer the total and aux wire counts from the state shape, because small-scale
                # edge cases often have fewer work wires than the general case.
                num_all_wires = qp.math.ceil_log2(out_state.shape[0])
                num_aux_wires = num_all_wires - num_wires
                for _ in range(num_aux_wires):
                    assert np.allclose(out_state[1::2], 0.0), "\n".join(
                        [
                            f"{a} : {b}"
                            for a, b in zip(
                                np.where(out_state)[0], out_state[np.where(out_state)], strict=True
                            )
                        ]
                    )
                    out_state = out_state[::2]
                assert np.allclose([out_state[key] for key in indices], coefficients)

    @staticmethod
    def force_powers_of_two(indices: tuple, num_wires: int) -> tuple:
        """Force a set of indices to contain all powers of two from 1 to 2**num_wires."""
        # This implementation feels somewhat complicated, but it works. It likely scales terribly.
        powers = {2**i for i in range(num_wires)}
        power_ids = []
        for i, idx in enumerate(indices):
            if idx in powers:
                power_ids.append(i)
                powers.remove(idx)

        new_indices = list(indices)  # Make mutable, indices is a tuple
        k = 0
        # Iterate over powers of two that are not yet in indices
        for power in powers:
            # Search for an index where there is no power of two yet
            while k in power_ids:
                k += 1
            # Insert the power of two in a spot where there isn't one yet.
            new_indices[k] = power
            # Increment k so we don't overwrite the just inserted value in the next iteration
            k += 1

        return tuple(new_indices)

    @pytest.mark.parametrize("num_wires,num_entries", [(7, 7), (8, 10)])
    def test_decomposition_prepares_state_non_id_encoding(self, num_wires, num_entries, seed):
        """Test that the decomposition of SumOfSlatersPrep2 actually prepares the desired state."""
        # pylint: disable=unsubscriptable-object

        coefficients, indices = self.make_random_data(num_wires, num_entries, seed=seed)
        # Add indices (powers of two) that force many bits to be required,
        # avoiding the identity encoding case
        indices = self.force_powers_of_two(indices, num_wires)
        wires = list(range(num_wires))
        all_wires = make_registers(indices, num_wires)

        with qp.decomposition.toggle_graph_ctx(
            True
        ):  # safe alternative to avoid enabling graph globally on the labs test runner
            # Currently just one rule is implemented, but this test should pass for all decompositions
            for rule in qp.list_decomps(SumOfSlatersPrep2):

                @qp.qnode(qp.device("lightning.qubit"))
                def func():
                    # pylint: disable=cell-var-from-loop
                    # Make sure that the output state length is at least 2**num_wires
                    qp.Identity(wires)
                    rule(coefficients, **all_wires, indices=indices)
                    return qp.state()

                out_state = func()

                # We infer the total and aux wire counts from the state shape, because small-scale
                # edge cases often have fewer work wires than the general case.
                num_all_wires = qp.math.ceil_log2(out_state.shape[0])
                num_aux_wires = num_all_wires - num_wires
                for _ in range(num_aux_wires):
                    assert np.allclose(out_state[1::2], 0.0), "\n".join(
                        [
                            f"{a} : {b}"
                            for a, b in zip(
                                np.where(out_state)[0], out_state[np.where(out_state)], strict=True
                            )
                        ]
                    )
                    out_state = out_state[::2]
                assert np.allclose([out_state[key] for key in indices], coefficients)
