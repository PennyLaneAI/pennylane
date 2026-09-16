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
"""Tests for UniformPrep, AliasSampling, and alias_sampling_wires."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.decomposition import list_decomps
from pennylane.ops.functions.assert_valid import _test_decomposition_rule, assert_valid
from pennylane.templates.subroutines.alias_sampling import _build_alias_tables
from pennylane.typing import AbstractWires


def _wire_layout(n_states):
    """Return (target_wires, work_wires, n_wires) for a given n_states."""
    k = (n_states & -n_states).bit_length() - 1
    L = n_states >> k
    logL = (L - 1).bit_length()
    n_tgt = k + logL
    n_work = max(logL - 1, 1)

    target_wires = list(range(n_tgt))
    work_wires = list(range(n_tgt, n_tgt + 1 + n_work))
    n_wires = n_tgt + 1 + n_work
    return target_wires, work_wires, n_wires


def _target_probs(n_states):
    """Run the circuit and return the probability on the target register."""
    target_wires, work_wires, n_wires = _wire_layout(n_states)
    dev = qp.device("default.qubit", wires=n_wires)

    @qp.qnode(dev)
    def circuit():
        qp.UniformPrep(n_states, target_wires, work_wires)
        return qp.probs(wires=target_wires)

    return np.asarray(circuit())


class TestUniformPrep:
    """Test UniformPrep state preparation."""

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize("n_states", [4, 5])
    def test_assert_valid_and_decomposition(self, n_states):
        """Test that UniformPrep is a valid Operator2 and decomposes, with and without capture."""
        target_wires, work_wires, _ = _wire_layout(n_states)
        op = qp.UniformPrep(n_states, target_wires, work_wires)
        assert_valid(op, skip_differentiation=True)
        for rule in list_decomps(qp.UniformPrep):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize("n_states", [3, 5, 10, 11, 20, 24])
    def test_uniform_distribution(self, n_states):
        """Test that the first n_states basis states are equally likely; the rest are zero."""
        probs = _target_probs(n_states)
        assert np.allclose(probs[:n_states], 1 / n_states)
        assert np.allclose(probs[n_states:], 0.0)

    @pytest.mark.parametrize("n_states", [3, 4, 10, 11])
    def test_state_amplitudes(self, n_states):
        """Test that amplitudes on the target register have equal magnitude sqrt(1/n_states)."""
        target_wires, work_wires, n_wires = _wire_layout(n_states)
        dev = qp.device("default.qubit", wires=n_wires)

        @qp.qnode(dev)
        def circuit():
            qp.UniformPrep(n_states, target_wires, work_wires)
            return qp.state()

        state = np.asarray(circuit())
        block = 2 ** (n_wires - len(target_wires))
        target_amps = state[::block][: 2 ** len(target_wires)]
        assert np.allclose(target_amps[:n_states], np.sqrt(1 / n_states))
        assert np.allclose(target_amps[n_states:], 0.0)

    @pytest.mark.parametrize("n_states", [2, 4, 8, 16])
    def test_power_of_two(self, n_states):
        """Test that powers of two reduce to plain Hadamards over the whole register."""
        probs = _target_probs(n_states)
        assert np.allclose(probs, 1 / n_states)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_single_state(self):
        """Test that n_states = 1 uses zero target wires and leaves the register in |0>."""
        with qp.queuing.AnnotatedQueue() as q:
            qp.UniformPrep(1, [], work_wires=[1])
        assert len(q.queue) == 1
        tape = qp.tape.QuantumScript.from_queue(q)
        assert tape.operations[0].name == "UniformPrep"

        op = qp.UniformPrep(1, [], work_wires=[])
        for rule in list_decomps(qp.UniformPrep):
            _test_decomposition_rule(op, rule)

    def test_wrong_target_wire_count_raises(self):
        """Test that a target register of the wrong size raises a clear error."""
        with pytest.raises(ValueError, match="target_wires must have 3 wires"):
            qp.UniformPrep(5, [0, 1], work_wires=[3, 4])

    def test_insufficient_work_wires_raises(self):
        """Test that too few work wires raise a clear error."""
        with pytest.raises(ValueError, match="work_wires must have at least 4 wires"):
            qp.UniformPrep(9, [0, 1, 2, 3], work_wires=[4, 5])

    def test_non_positive_n_states_raises(self):
        """Test that an error is raised when n_states is not a positive integer."""
        with pytest.raises(ValueError, match="n_states must be at least 1"):
            qp.UniformPrep(n_states=0, target_wires=[0, 1, 2], work_wires=[3, 4, 5])

    def test_overlapping_wires_raise_for_power_of_two(self):
        """Test that target/work overlap is rejected even when n_states is a power of two."""
        with pytest.raises(ValueError, match="must not overlap"):
            qp.UniformPrep(4, [0, 1], work_wires=[0])

    def test_abstract_wires_length_is_validated(self):
        """Test that register sizes are checked for AbstractWires, which still expose a length."""
        with pytest.raises(ValueError, match="target_wires must have 3 wires"):
            qp.UniformPrep(5, AbstractWires(2), AbstractWires(3))
        with pytest.raises(ValueError, match="work_wires must have at least 3 wires"):
            qp.UniformPrep(5, AbstractWires(3), AbstractWires(1))
        op = qp.UniformPrep(5, AbstractWires(3), AbstractWires(3))
        assert isinstance(op.target_wires, AbstractWires)
        assert isinstance(op.work_wires, AbstractWires)

    def test_mixed_concrete_and_abstract_wires(self):
        """Test that length checks run per register, including mixed concrete/abstract inputs."""
        with pytest.raises(ValueError, match="work_wires must have at least 3 wires"):
            qp.UniformPrep(5, range(3), AbstractWires(1))
        op = qp.UniformPrep(5, range(3), AbstractWires(3))
        assert list(op.target_wires) == [0, 1, 2]
        assert isinstance(op.work_wires, AbstractWires)


def _reconstruct_amplitudes(alt, keep, mu):
    """Exact ground-truth distribution from the integer alias tables (Eq. 29 from arXiv:1805.03662)."""
    L, n = len(alt), 2**mu
    rho = np.zeros(L)
    for l in range(L):
        rho[l] += keep[l]
        for k in range(L):
            if alt[k] == l:
                rho[l] += n - keep[k]
    return rho / (n * L)


class TestBuildAliasTables:
    """Test the classical alias-table construction."""

    @pytest.mark.parametrize("L", [2, 4, 7])
    @pytest.mark.parametrize("mu", [4, 5, 8])
    def test_ranges(self, L, mu):
        """Test that alt is in range [0, L), and keep is in [0, 2**mu)."""
        rng = np.random.default_rng(L * 100 + mu)
        w = rng.random(L) + 0.05
        alt, keep = _build_alias_tables(w, mu)
        assert len(alt) == L and len(keep) == L
        assert all(0 <= a < L for a in alt)
        assert all(0 <= k < 2**mu for k in keep)

    @pytest.mark.parametrize("L", [2, 3, 5, 8])
    @pytest.mark.parametrize("mu", [6, 8])
    def test_normalization_constraint(self, L, mu):
        """Test that the reconstruction matches the target within the mu-bit bound L/2**mu."""
        rng = np.random.default_rng(L + 7 * mu)
        w = rng.random(L) + 0.05
        target = w / w.sum()
        alt, keep = _build_alias_tables(w, mu)
        recon = _reconstruct_amplitudes(alt, keep, mu)
        assert np.max(np.abs(recon - target)) <= float(L) / 2**mu

    def test_negative_probs_raise(self):
        """Test that negative probabilities raise a ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            _build_alias_tables([0.5, -0.1, 0.6], 4)

    def test_zero_sum_raises(self):
        """Test that a ValueError is raised when the probabilities sum to a non-positive integer."""
        with pytest.raises(ValueError, match="positive value"):
            _build_alias_tables([0.0, 0.0], 4)


@pytest.mark.parametrize(
    "L, mu, expected_target, expected_temp, expected_work",
    [
        (1, 4, 0, 12, 0),
        (2, 4, 1, 13, 0),
        (3, 5, 2, 17, 2),
        (4, 6, 2, 20, 0),
        (8, 5, 3, 18, 0),
        (16, 7, 4, 25, 0),
    ],
)
def test_alias_sampling_wires(L, mu, expected_target, expected_temp, expected_work):
    """Test that alias_sampling_wires returns correct wire allocations for given L and mu."""
    req = qp.alias_sampling_wires(L, mu)

    assert req["target_wires"] == expected_target
    assert req["temp_wires"] == expected_temp
    assert req["work_wires"] == expected_work


@pytest.mark.parametrize("mu", [True, 0])
def test_alias_sampling_wires_invalid_mu_raises(mu):
    """Test that alias_sampling_wires rejects an invalid precision."""
    with pytest.raises(ValueError, match="mu must be a positive integer"):
        qp.alias_sampling_wires(2, mu)


def test_alias_sampling_wires_invalid_n_states_raises():
    """Test that alias_sampling_wires rejects an empty coefficient register."""
    with pytest.raises(ValueError, match="n_states must be at least 1"):
        qp.alias_sampling_wires(0, 1)


def _alias_registers(L, mu, w=None):
    if w is None:
        w = np.random.default_rng(L).random(L) + 0.05
    req = qp.alias_sampling_wires(L, mu)
    n = sum(req.values())
    wires, temp, work = np.split(np.arange(n), np.cumsum([req["target_wires"], req["temp_wires"]]))
    return w, wires, temp, work, n


class TestAliasSampling:
    """Test the alias sampling circuit."""

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_assert_valid_and_decomposition(self):
        """Test that AliasSampling is a valid Operator2 and decomposes, with and without capture."""
        w, wires, temp, work, _ = _alias_registers(4, 3)
        op = qp.AliasSampling(w, 3, wires, temp, work)
        assert_valid(op, skip_differentiation=True)
        for rule in list_decomps(qp.AliasSampling):
            _test_decomposition_rule(op, rule)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_adjoint_decomposition(self):
        """Test that the adjoint decomposition is capture compatible."""
        w, wires, temp, work, _ = _alias_registers(4, 3)
        op = qp.adjoint(qp.AliasSampling(w, 3, wires, temp, work))
        for rule in list_decomps("Adjoint(AliasSampling)"):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize("L", [2, 3, 4, 5, 6])
    def test_marginal_matches_reconstruction(self, L):
        """Test that the target marginal matches the classical tables within the mu-bit
        bound, and that no probability leaks onto indices >= L."""
        mu = 3
        rng = np.random.default_rng(L * 13 + 1)
        w = rng.random(L) + 0.05
        recon = _reconstruct_amplitudes(*_build_alias_tables(w, mu), mu)
        w, wires, temp, work, n = _alias_registers(L, mu, w)

        @qp.qnode(qp.device("default.qubit", wires=n))
        def circuit():
            qp.AliasSampling(w, mu, wires, temp, work)
            return qp.probs(wires=wires)

        probs = np.asarray(circuit())
        target = w / w.sum()
        assert np.allclose(probs[:L], target, atol=1 / 2**mu)
        assert np.allclose(probs[:L], recon, atol=1e-9)
        assert np.isclose(probs[:L].sum(), 1.0, atol=1e-6)
        assert np.allclose(probs[L:], 0.0, atol=1e-9)

    @pytest.mark.parametrize("L", [3, 5])
    def test_work_wires_clean_temp_wires_entangled(self, L):
        """Test that work_wires return to |0> while temp_wires stay entangled."""
        mu = 2
        w, wires, temp, work, n = _alias_registers(L, mu)

        @qp.qnode(qp.device("default.qubit", wires=n))
        def circuit():
            qp.AliasSampling(w, mu, wires, temp, work)
            return qp.probs(wires=work), qp.probs(wires=temp)

        work_probs, temp_probs = circuit()
        assert np.isclose(np.asarray(work_probs)[0], 1.0)
        assert not np.isclose(np.asarray(temp_probs)[0], 1.0)

    def test_adjoint_uncomputes(self):
        """Test that prepare-dagger returns every register, temp_wires included, to |0>."""
        L, mu = 3, 2
        w, wires, temp, work, n = _alias_registers(L, mu, np.random.default_rng(0).random(L) + 0.05)

        @qp.qnode(qp.device("default.qubit", wires=n))
        def circuit():
            qp.AliasSampling(w, mu, wires, temp, work)
            qp.adjoint(qp.AliasSampling(w, mu, wires, temp, work))
            return qp.probs()

        assert np.isclose(np.asarray(circuit())[0], 1.0)

    @pytest.mark.parametrize("mu", [True, 0])
    def test_invalid_mu_raises(self, mu):
        """Test that mu must be a positive integer."""
        with pytest.raises(ValueError, match="mu must be a positive integer"):
            qp.AliasSampling([1.0], mu, [], [0, 1, 2], [])

    def test_empty_probs_raises(self):
        """Test that probs must contain at least one entry."""
        with pytest.raises(ValueError, match="probs must have at least one entry"):
            qp.AliasSampling([], 1, [], [0, 1, 2], [])

    def test_zero_sum_probs_raise(self):
        """Test that probs must have a positive sum."""
        with pytest.raises(ValueError, match="probs must sum to a positive value"):
            qp.AliasSampling([0.0, 0.0], 1, [0], list(range(1, 5)), [])

    def test_2d_probs_raise(self):
        """Test that a 2-D probs array is rejected instead of being flattened."""
        with pytest.raises(ValueError, match="1-D sequence"):
            qp.AliasSampling([[0.5, 0.5]], 1, [0], list(range(1, 5)), [])

    @pytest.mark.parametrize("probs", [[0.5, -0.1, 0.6], [0.5, np.nan], [0.5, np.inf]])
    def test_invalid_probs_raise(self, probs):
        """Test that negative or non-finite probs are rejected in the constructor."""
        with pytest.raises(ValueError, match="non-negative and finite"):
            qp.AliasSampling(probs, 1, [0, 1], list(range(2, 6)), [])

    @pytest.mark.parametrize(
        ("target_wires", "temp_wires", "work_wires", "match"),
        [
            ([0], list(range(1, 9)), [9, 10], "target_wires must have 2 entries"),
            ([0, 1], list(range(2, 9)), [9, 10], "temp_wires must have 8 entries"),
            ([0, 1], list(range(2, 10)), [10], "work_wires must have at least 2 entries"),
        ],
    )
    def test_invalid_register_sizes_raise(self, target_wires, temp_wires, work_wires, match):
        """Test that each register size is validated."""
        with pytest.raises(ValueError, match=match):
            qp.AliasSampling([0.2, 0.3, 0.5], 2, target_wires, temp_wires, work_wires)

    def test_abstract_wires_length_is_validated(self):
        """Test that register sizes are checked for AbstractWires, which still expose a length."""
        with pytest.raises(ValueError, match="target_wires must have 2 entries"):
            qp.AliasSampling(
                [0.2, 0.3, 0.5], 2, AbstractWires(1), AbstractWires(8), AbstractWires(2)
            )
        op = qp.AliasSampling(
            [0.2, 0.3, 0.5], 2, AbstractWires(2), AbstractWires(8), AbstractWires(2)
        )
        assert isinstance(op.target_wires, AbstractWires)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_single_coefficient_decomposition(self):
        """Test that the zero-target-wire decomposition is valid."""
        op = qp.AliasSampling([1.0], 1, [], [0, 1, 2], [])
        for rule in list_decomps(qp.AliasSampling):
            _test_decomposition_rule(op, rule)
