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
    _find_affine_subspace_isometry,
    _pui_state_prep_core,
    _pui_state_prep_resources,
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
            # num_entries, n, [n, n_subspace, n_r, m, len(tableau), _packed_dtype, _word]
            (2, 2, [2, 1, 1, 1, 2, np.uint64, np.uint64]),
            (2, 8, [8, 1, 7, 2, 2, np.uint64, np.uint64]),
            (2, 65, [65, 1, 64, 2, 2, object, int]),
            (100, 71, [71, 7, 64, 64, 100, object, int]),
            (3, 3, [3, 2, 1, 1, 3, np.uint64, np.uint64]),
            (4, 8, [8, 2, 6, 4, 4, np.uint64, np.uint64]),
            (15, 5, [5, 4, 1, 1, 15, np.uint64, np.uint64]),
            (23, 29, [29, 5, 24, 16, 23, np.uint64, np.uint64]),
            (7, 65, [65, 3, 62, 7, 7, object, int]),
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

    def _validate_circuit_structure(self, circuit, fanout_bits, iso_finder, num_entries):
        """Validate that the structure of a circuit returned by ``find_isometry`` is correct."""
        n_subspace, n, m = iso_finder.n_subspace, iso_finder.n, iso_finder.m
        batch_size = 0
        seen_fanouts = 0
        for _type, *data in circuit:

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

        assert np.shape(fanout_bits) == (seen_fanouts, n - 1)

    def _validate_circuit_ops(self, circuit, fanout_bits, iso_finder, basis_states):
        """Validate that the a circuit returned by ``find_isometry`` implements the right
        isometry."""
        n_subspace = iso_finder.n_subspace

        # Load the final states
        final_states = list(map(int, iso_finder.tableau))
        states = np.array(
            [[(val >> s) & iso_finder._one for s in iso_finder._shifts] for val in final_states]
        ).astype(np.int8)
        # Transform the final states back
        for _type, *data in reversed(circuit):
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
                bits = fanout_bits[bit_pointer]
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
        circuit, fanout_bits, bijection = iso_finder.find_isometry()

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
        self._validate_circuit_structure(circuit, fanout_bits, iso_finder, num_entries)
        self._validate_circuit_ops(circuit, fanout_bits, iso_finder, states)

    def test_find_isometry_without_remainder_register(self):
        """The identity isometry is returned when the subspace occupies the full register."""
        iso_finder = PUIsometryFinder([2, 0, 1], 2)

        circuit, fanout_bits, bijection = iso_finder.find_isometry()

        assert circuit == []
        assert fanout_bits == []
        assert bijection == {0: 2, 1: 0, 2: 1}

    def test_toffoli_may_use_subspace_control(self):
        """A fallback Toffoli may distinguish rows using a subspace qubit."""
        iso_finder = PUIsometryFinder([0b0010, 0b0110, 0b1000], 4)
        circuit, _, _ = iso_finder.find_isometry()
        toffolis = [data for op_type, *data in circuit if op_type == 3]

        assert toffolis
        assert any(data[1] < iso_finder.n_subspace for data in toffolis)


class TestAffineSubspaceIsometry:
    """Tests for the Clifford-only affine-support fast path."""

    @staticmethod
    def _apply_circuit(states, circuit, n):
        states = list(map(int, states))
        for op_type, *data in circuit:
            if op_type == "X":
                mask = 1 << (n - 1 - data[0])
                states = [state ^ mask for state in states]
            elif op_type == "CNOT":
                control, target = data
                control_mask = 1 << (n - 1 - control)
                target_mask = 1 << (n - 1 - target)
                states = [
                    state ^ target_mask if state & control_mask else state for state in states
                ]
            else:
                w0, w1 = data[0]
                mask0, mask1 = 1 << (n - 1 - w0), 1 << (n - 1 - w1)
                states = [
                    state ^ mask0 ^ mask1 if bool(state & mask0) != bool(state & mask1) else state
                    for state in states
                ]
        return states

    def test_affine_support_maps_to_subspace(self):
        """An affine support of minimal dimension maps into the subspace using Clifford gates."""
        n, n_subspace = 8, 3
        anchor = 0b10110110
        basis = (0b11001001, 0b00110111, 0b01011100)
        states = []
        for mask in range(2**n_subspace):
            state = anchor
            for j, vector in enumerate(basis):
                if (mask >> j) & 1:
                    state ^= vector
            states.append(state)

        circuit, bijection = _find_affine_subspace_isometry(states, n, n_subspace)
        assert all(op_type in {"X", "CNOT", "SWAP"} for op_type, *_ in circuit)

        mapped = self._apply_circuit(states, circuit, n)
        assert all((state & ((1 << (n - n_subspace)) - 1)) == 0 for state in mapped)
        assert [state >> (n - n_subspace) for state in mapped] == [
            bijection[i] for i in range(len(states))
        ]

    def test_non_affine_support_falls_back(self):
        """Support with affine rank above the subspace width does not use the fast path."""
        assert _find_affine_subspace_isometry([0, 1, 2, 4], 4, 2) is None

    def test_affine_decomposition_omits_qrom_and_toffoli(self):
        """The decomposition emits no non-Clifford isometry operations for affine support."""
        coefficients = np.ones(4) / 2
        indices = (0b1010, 0b0110, 0b1001, 0b0101)

        with qp.queuing.AnnotatedQueue() as queue:
            _pui_state_prep_core(coefficients, range(4), indices, work_wires=[4])

        ops = [wrapped.obj for wrapped in queue]
        assert not any(isinstance(op, (qp.QROM, qp.MultiControlledX)) for op in ops)
        assert any(isinstance(op, qp.CNOT) for op in ops)

    def test_resource_model_caps_batches_and_accounts_for_excess_wires(self):
        """Resource heuristics cap QROM widths and account for the enlarged register."""
        base = _pui_state_prep_resources(
            num_entries=3, num_wires=4, num_work_wires=1, is_affine=False
        )
        excess = _pui_state_prep_resources(
            num_entries=3, num_wires=4, num_work_wires=5, is_affine=False
        )

        assert len(base) == 8
        assert len(excess) == 9
        assert {len(rep.target_wires) for rep in base if isinstance(rep, qp.QROM)} == {1, 2}
        assert {len(rep.target_wires) for rep in excess if isinstance(rep, qp.QROM)} == {1, 2, 3}
        assert base[qp.SWAP] == 4
        assert excess[qp.SWAP] == 8

    def test_affine_resource_params_avoid_dynamic_allocation(self):
        """Affine support uses the decomposition rule that does not allocate work wires."""
        op = PartialUnaryStatePreparation(
            np.ones(4) / 2,
            wires=range(4),
            indices=(0b0000, 0b0011, 0b1100, 0b1111),
            work_wires=(),
        )
        dynamic_rule, provided_rule = list_decomps(PartialUnaryStatePreparation)

        assert op.resource_params["is_affine"]
        assert not dynamic_rule.is_applicable(**op.resource_params)
        assert provided_rule.is_applicable(**op.resource_params)
        assert dynamic_rule.get_work_wire_spec(**op.resource_params).total == 0


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

    # run test once with qjit, once without
    for _qjit in [False, True]:
        if _qjit:
            from catalyst.device.decomposition import catalyst_decompose

            gate_set = {
                "QROM",
                "MultiplexerStatePreparation",
                "ForLoop",
                "Cond",
                "CNOT",
                "PauliX",
                "MultiControlledX",
            }
            func = qp.qjit(catalyst_decompose(func, capabilities=None, target_gates=gate_set))

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

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_complex_coefficients_on_non_affine_support(self):
        """The generic PUI path preserves arbitrary relative phases."""
        indices = (0, 3, 7, 17, 25)
        coefficients = np.array([1, 1j, -2, -2j, 3], dtype=complex)
        coefficients /= np.linalg.norm(coefficients)
        assert _find_affine_subspace_isometry(indices, 5, 3) is None

        dev = qp.device("default.qubit", wires=7)

        @qp.qnode(dev)
        def circuit():
            PartialUnaryStatePreparation(coefficients, range(5), indices, [5, 6])
            return qp.state()

        target = np.zeros(32, dtype=complex)
        target[list(indices)] = coefficients
        assert np.allclose(circuit()[::4], target)

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

    @pytest.mark.catalyst
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
        op = PartialUnaryStatePreparation(coefficients, wires, indices, work_wires)
        applicable_rule = int(provide_work_wires or op.resource_params["is_affine"])

        for j, rule in enumerate(list_decomps(PartialUnaryStatePreparation)):
            applicable = rule.is_applicable(**op.resource_params)
            assert applicable is (j == applicable_rule)
            if not applicable:
                continue

            wire_specs = wires, work_wires, num_wires + needed_work_wires
            assert_pui_correctness(rule, coefficients, indices, wire_specs)

    @pytest.mark.catalyst
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
        op = PartialUnaryStatePreparation(coefficients, wires, indices, work_wires)

        for j, rule in enumerate(list_decomps(PartialUnaryStatePreparation)):
            applicable = rule.is_applicable(**op.resource_params)
            assert applicable is (j == 1)
            if not applicable:
                continue

            wire_specs = wires, work_wires, num_wires + num_work_wires
            assert_pui_correctness(rule, coefficients, indices, wire_specs)

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

        with pytest.raises(ValueError, match="At least one state index"):
            PartialUnaryStatePreparation(np.array([]), wires, (), [])

        with pytest.raises(TypeError, match="must be integers"):
            PartialUnaryStatePreparation(np.ones(2) / np.sqrt(2), wires, (0, 1.5), [])

        with pytest.raises(ValueError, match="must be disjoint"):
            PartialUnaryStatePreparation(np.ones(2) / np.sqrt(2), wires, (0, 1), [3, 4])
