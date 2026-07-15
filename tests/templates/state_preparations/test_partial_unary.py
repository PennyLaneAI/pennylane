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
from pennylane.templates.state_preparations.partial_unary import (
    PartialUnaryStatePreparation,
    PUIsometryFinder,
)

# pylint: disable=protected-access


def random_distinct_integers(high, size, rng):
    if high < 2**25:
        return rng.choice(high, size=size, replace=False)

    samples = set()
    if high < 2**64:
        while len(samples) < size:
            samples.add(int(rng.integers(high)))
    else:
        # This works but it sacrifices uniformity of the distribution and only work for powers of 2
        assert high.bit_count() == 1
        split = 2 ** (high.bit_length() // 2)
        smaller_0 = random_distinct_integers(split, size=size, rng=rng).astype(object)
        smaller_1 = random_distinct_integers(high // split, size=size, rng=rng).astype(object)
        return smaller_0 + smaller_1 * split

    return np.array(list(samples), dtype=int)


class TestPUIsometryFinder:
    """Tests for the isometry finding algorithm in PUIsometryFinder."""

    def test_error_for_duplicate_basis_states(self):
        """Test that an error is raised if there are duplicate basis states."""
        match = "must be unique, got 3 basis states but just 2 distinct"
        with pytest.raises(ValueError, match=match):
            PUIsometryFinder([125012, 9251, 9251], 100)

    def test_error_for_too_few_states(self):
        """Test that an error is raised if there are less than two basis states."""
        match = "At least two basis states are required"
        with pytest.raises(ValueError, match=match):
            PUIsometryFinder([125012], 100)

    def test_error_for_too_few_qubits(self):
        """Test that an error is raised if there are zero or less qubits."""
        match = "n_qubits must be a positive integer"
        with pytest.raises(ValueError, match=match):
            PUIsometryFinder([125012, 2, 9, 9251], 0)
        with pytest.raises(ValueError, match=match):
            PUIsometryFinder([125012, 2, 9, 9251], -20)
        with pytest.raises(ValueError, match=match):
            PUIsometryFinder([125012, 2, 9, 9251], 20.0)

    @pytest.mark.parametrize(
        "num_entries, n, expected",
        [
            (2, 2, [2, 1, 1, 1, 2, np.uint64, np.uint64]),
            (2, 8, [8, 1, 7, 4, 2, np.uint64, np.uint64]),
            (2, 65, [65, 1, 64, 64, 2, object, int]),
            (3, 3, [3, 2, 1, 1, 3, np.uint64, np.uint64]),
            (4, 8, [8, 2, 6, 4, 4, np.uint64, np.uint64]),
            (15, 5, [5, 4, 1, 1, 15, np.uint64, np.uint64]),
            (23, 29, [29, 5, 24, 16, 23, np.uint64, np.uint64]),
            (7, 65, [65, 3, 62, 32, 7, object, int]),
            (112563, 100, [100, 17, 83, 64, 112563, object, int]),
        ],
    )
    def test_sizes(self, num_entries, n, expected, seed):
        """Test that the qubit count, subspace register size, remainder register size,
        target/max batch size, tableau size and data types are all initialized correctly."""
        rng = np.random.default_rng(seed)
        states = random_distinct_integers(2**n, num_entries, rng)
        iso_finder = PUIsometryFinder(states, n)
        specs = [
            getattr(iso_finder, attr)
            for attr in ["n", "n_subspace", "n_r", "m", "_packed_dtype", "_word"]
        ]
        specs.insert(-2, len(iso_finder.tableau))
        assert specs == expected

    @pytest.mark.parametrize("num_entries, n", [(2, 1), (3, 2), (4, 2), (7, 3), (4097, 13)])
    def test_sizes_many_states(self, num_entries, n, seed):
        """Test that the qubit count, subspace register size, remainder register size,
        target/max batch size, tableau size and data types are all initialized correctly."""
        rng = np.random.default_rng(seed)
        states = np.arange(2**n)
        rng.shuffle(states)
        states = states[:num_entries]
        iso_finder = PUIsometryFinder(states, n)
        specs = [
            getattr(iso_finder, attr)
            for attr in ["n", "n_subspace", "n_r", "m", "_packed_dtype", "_word"]
        ]
        specs.insert(-2, len(iso_finder.tableau))
        assert specs == [n, n, 0, 0, num_entries, np.uint64, np.uint64]

    def _validate_circuit_structure(self, circuit, iso_finder, num_entries):
        """Validate that the structure of a circuit returned by ``find_isometry`` is correct."""
        n_subspace, n, m = iso_finder.n_subspace, iso_finder.n, iso_finder.m
        batch_size = 0
        seen_fanouts = 0
        for _type, *data in circuit["structure"]:

            if _type == 0:
                assert len(data) == 4
                assert all(isinstance(d, int) for d in data)
                assert data[2:] == [0, 0]  # Dummy values
                k_start, k = data[:2]
                assert 0 <= k_start < k <= num_entries
                assert k - k_start == batch_size
                batch_size = 0

            elif _type == 1:
                assert len(data) == 4
                assert all(isinstance(d, int) for d in data)
                assert data[1] == seen_fanouts
                seen_fanouts += 1
                assert data[2:] == [0, 0]  # Dummy values
                assert n_subspace <= data[0] < n
                batch_size += 1
                assert batch_size <= m

            elif _type == 2:
                assert len(data) == 4
                assert all(isinstance(d, int) for d in data)
                assert all(n_subspace <= d < n for d in data[:2])
                assert data[2:] == [0, 0]  # Dummy values

            elif _type == 3:
                assert len(data) == 4
                assert all(isinstance(d, int) for d in data)
                assert all(n_subspace <= d < n for d in data[:3])
                assert 0 <= data[3] <= 1

            else:
                raise AssertionError(
                    "Expected the first entry in each circuit structure object to be an integer"
                    f"between 0 and 3 (incl.), but got {_type}"
                )

        assert np.shape(circuit["fanout_bits"]) == (seen_fanouts, n - 1)

    def _validate_circuit_ops(self, circuit, iso_finder, basis_states):
        """Validate that the a circuit returned by ``find_isometry`` implements the right
        isometry."""
        n_subspace = iso_finder.n_subspace

        # Load the final states
        final_states = list(map(int, iso_finder.tableau))
        states = np.array(
            [[(val >> s) & iso_finder._one for s in iso_finder._shifts] for val in final_states]
        ).astype(np.int8)
        # Transform the final states back
        for _type, *data in reversed(circuit["structure"]):
            if _type == 0:
                k_start, k = data[:2]
                batch = k - k_start
                control_bits = qp.math.int_to_binary(np.arange(k_start, k), n_subspace)
                # Broadcasted version of `apply_multi_controlled_x`.
                # A row is flipped iff all control bits match control_values
                match = np.all(states[None, :, :n_subspace] == control_bits[:, None, :], axis=2)
                states[:, np.arange(n_subspace, batch + n_subspace)] ^= match.astype(np.int8).T
            elif _type == 1:
                control, bit_pointer = data[:2]
                bits = circuit["fanout_bits"][bit_pointer]
                ctrl_bits = states[:, control]  # rows where the control is active
                # Bit indices that need to be flipped. Need to take into account that ``bits`` does
                # not contain the control bit itself.
                target_bits = np.concatenate(
                    [np.where(bits[:control])[0], np.where(bits[control:])[0] + (control + 1)]
                )
                states[:, target_bits] ^= ctrl_bits[:, None]
            elif _type == 2:
                w0, w1 = data[:2]
                states[:, [w0, w1]] = states[:, [w1, w0]]

            elif _type == 3:
                *wires, second_ctrl_val = data
                control, target = np.array(wires[:2]), wires[2]
                # A row is flipped iff all control bits match control_values
                match = np.all(states[:, control] == np.array([1, second_ctrl_val]), axis=1)
                states[:, target] ^= match.astype(np.int8)

        # Compute target state bit tableau
        target_states = np.array(
            [
                [(int(val) >> s) & iso_finder._one for s in iso_finder._shifts]
                for val in basis_states
            ]
        ).astype(np.int8)
        assert np.allclose(target_states, states)

    @pytest.mark.parametrize(
        "num_entries, n",
        [(2, 2), (2, 8), (2, 65), (3, 3), (4, 8), (15, 5), (23, 29), (7, 65), (1563, 100)],
    )
    def test_find_isometry(self, num_entries, n, seed):
        """Test the main method ``find_isometry``."""
        rng = np.random.default_rng(seed)
        states = random_distinct_integers(2**n, num_entries, rng)
        iso_finder = PUIsometryFinder(states, n)
        circuit, bijection = iso_finder.find_isometry()

        # Validate the internal tableau state:
        # All remainder qubits are zeroed everywhere
        assert np.all((iso_finder.tableau & iso_finder.rem_mask) == iso_finder._zero)
        # The cached version of this also is correct
        assert np.all(iso_finder._in_subspace)
        assert iso_finder._n_not_subspace == 0
        # The subspace qubits are enumerating the num_entries integers specified in the bijection
        assert np.allclose(
            (iso_finder.tableau >> iso_finder._nr_shift)[np.array(list(bijection.keys()))],
            np.array(list(bijection.values())),
        )

        # Validate circuit structure:
        self._validate_circuit_structure(circuit, iso_finder, num_entries)
        self._validate_circuit_ops(circuit, iso_finder, states)


def _is_binary(x: np.ndarray) -> bool:
    """Return whether all entries of a numpy array are binary."""
    return set(x.flat).issubset({0, 1})


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


def assert_pui_correctness(rule, coefficients, indices, wire_specs):
    """Run a correctness test for PartialUnaryStatePreparation that checks that the correct
    state is being prepared."""
    wires, work_wires, num_device_wires = wire_specs
    num_wires = len(wires)
    num_work_wires = len(work_wires)

    @qp.qnode(qp.device("lightning.qubit", wires=num_device_wires))
    @qp.transforms.resolve_dynamic_wires(min_int=num_wires + num_work_wires)
    def func():
        # pylint: disable=cell-var-from-loop
        # Make sure that the output state length is at least 2**num_wires
        rule(coefficients, wires=wires, indices=indices, work_wires=work_wires)
        return qp.state()

    out_state = func()

    # We infer the total and aux wire counts from the state shape, because small-scale
    # edge cases often have fewer work wires than the general case.
    num_all_used_wires = ceil_log2(out_state.shape[0])
    num_aux_wires = num_all_used_wires - num_wires
    for _ in range(num_aux_wires):
        assert np.allclose(out_state[1::2], 0.0)
        out_state = out_state[::2]
    # Arrange state vector for the custom randomized target wire ordering
    out_state = qp.math.expand_vector(out_state, range(num_wires), wires)
    assert np.allclose([out_state[key] for key in indices], coefficients)


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

    @pytest.mark.parametrize("provide_work_wires", [False, True])
    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize(
        "num_wires,num_entries",
        [
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 3),
            (4, 15),
            (5, 4),
            (5, 21),
            (7, 63),
            (8, 32),
            (12, 128),
            (14, 16),
        ],
    )
    def test_decomposition_prepares_state(self, num_wires, num_entries, seed, provide_work_wires):
        """Test that the decomposition of PartialUnaryStatePreparation actually prepares the desired state."""

        coefficients, indices = self.make_random_data(num_wires, num_entries, seed=seed)
        needed_work_wires = max(qp.math.ceil_log2(num_entries) - 1, 1)
        if provide_work_wires:
            num_work_wires = needed_work_wires
        else:
            num_work_wires = 0

        wires = list(range(num_wires))
        rng = np.random.default_rng(seed)
        rng.shuffle(wires)

        work_wires = list(range(num_wires, num_wires + num_work_wires))
        rng.shuffle(work_wires)
        # If provide_work_wires=False/True (=> cast to 0/1), we expect the decomposition
        # rule with index 0/1 to be applicable. Exception: For num_entries=1, the rule with
        # index 1 should be applicable
        applicable_rule = int(provide_work_wires) if num_entries > 1 else 1

        for j, rule in enumerate(list_decomps(PartialUnaryStatePreparation)):
            applicable = rule.is_applicable(num_entries, num_wires, num_work_wires)
            assert applicable is (j == applicable_rule)
            if not applicable:
                continue

            wire_specs = wires, work_wires, num_wires + needed_work_wires
            assert_pui_correctness(rule, coefficients, indices, wire_specs)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize(
        "num_wires, num_entries, num_work_wires", [(7, 5, 15), (3, 2, 6), (4, 14, 8)]
    )
    def test_decomposition_correct_many_work(self, num_wires, num_entries, num_work_wires, seed):
        """Test that the decomposition of PartialUnaryStatePreparation actually
        prepares the desired state, also for too many work wires."""

        coefficients, indices = self.make_random_data(num_wires, num_entries, seed=seed)
        wires = list(range(num_wires))
        rng = np.random.default_rng(seed)
        rng.shuffle(wires)

        work_wires = list(range(num_wires, num_wires + num_work_wires))
        rng.shuffle(work_wires)

        for j, rule in enumerate(list_decomps(PartialUnaryStatePreparation)):
            applicable = rule.is_applicable(num_entries, num_wires, num_work_wires)
            assert applicable is (j == 1)
            if not applicable:
                continue

            wire_specs = wires, work_wires, num_wires + num_work_wires
            assert_pui_correctness(rule, coefficients, indices, wire_specs)

    def test_decomposition_error_flawed_circuit_object(self, monkeypatch):
        """Test that the decomposition function raises an error if the circuit structure
        data contains an invalid entry."""

        def mocked_find_isometry(self):
            """Mocked version of find_isometry that does nothing but creating an invalid circuit."""
            # Invalid _type: 4
            circuit = {
                "structure": [(4, 0, 1, 2, 3)],
                "fanout_bits": np.zeros((0, self.n - 1), dtype=np.int8),
            }
            return circuit, {i: int(val) for i, val in enumerate(self.tableau)}

        match = "Expected _type ids between 0 and 3 (incl), got 4"

        with monkeypatch.context() as m:
            m.setattr(PUIsometryFinder, "find_isometry", mocked_find_isometry)
            iso_finder = PUIsometryFinder([1, 4, 925, 1250], 11)
            print(iso_finder.find_isometry)
            with pytest.raises(ValueError, match=match):
                iso_finder.find_isometry()

    def test_input_validation(self):
        """Test that validation errors are raise for invalid inputs."""
        non_unique_indices = (0, 4, 1, 2, 0, 6, 4)
        coeffs = np.ones(len(non_unique_indices))
        wires = [0, 1, 2, 3]
        with pytest.raises(ValueError, match="must be unique"):
            PartialUnaryStatePreparation(coeffs, wires, non_unique_indices, [])

        unique_indices = (0, 4, 1, 2, 3, 6, 8)
        too_many_coeffs = np.ones(len(unique_indices) + 1)
        with pytest.raises(ValueError, match="number of coefficients and the number of state"):
            PartialUnaryStatePreparation(too_many_coeffs, wires, unique_indices, [])

        unique_indices = (0, 4, 1, 2, 3, 6, 63)
        with pytest.raises(ValueError, match=r"must be smaller than 2\*\*len\(wires\)=16"):
            PartialUnaryStatePreparation(coeffs, wires, unique_indices, [])

        unique_indices = (0, -4, 1, 2, 3, 6, 10)
        with pytest.raises(ValueError, match=r"must be positive"):
            PartialUnaryStatePreparation(coeffs, wires, unique_indices, [])
