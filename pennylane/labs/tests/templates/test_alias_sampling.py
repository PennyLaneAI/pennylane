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
Tests for the alias sampling uniform-state-preparation template.
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane import wires
from pennylane.labs.templates import uniform_prep_ops
from pennylane.labs.templates.alias_sampling import _build_alias_tables
from pennylane.labs.templates import alias_sampling_wires, alias_sampling

def _wire_layout(n_states):
    """Return (target_wires, flag, work_wires, n_wires) for a given n_states."""
    k = (n_states & -n_states).bit_length() - 1
    L = n_states >> k
    logL = (L - 1).bit_length()
    n_tgt = k + logL
    n_work = max(logL - 1, 1)

    target_wires = list(range(n_tgt))
    flag = n_tgt
    work_wires = list(range(n_tgt + 1, n_tgt + 1 + n_work))
    n_wires = n_tgt + 1 + n_work
    return target_wires, flag, work_wires, n_wires


def _target_probs(n_states):
    """Run the circuit and return the probability marginal on the target register."""
    target_wires, flag, work_wires, n_wires = _wire_layout(n_states)
    dev = qp.device("default.qubit", wires=n_wires)

    @qp.qnode(dev)
    def circuit():
        uniform_prep_ops(n_states, target_wires, flag, work_wires)
        return qp.probs(wires=target_wires)

    return circuit()


class TestUniformPrepOps:
    """Test the uniform_prep_ops alias-sampling state preparation."""

    @pytest.mark.parametrize("n_states", [3, 5, 10, 11, 20, 24])
    def test_uniform_distribution(self, n_states):
        """Tests that the first n_states basis states are equally likely; the rest are zero."""
        probs = _target_probs(n_states)

        assert np.allclose(probs[:n_states], 1 / n_states)
        assert np.allclose(probs[n_states:], 0.0)

    @pytest.mark.parametrize("n_states", [3, 4, 10, 11])
    def test_state_amplitudes(self, n_states):
        """Amplitudes on the target register have equal magnitude sqrt(1/n_states)."""
        target_wires, flag, work_wires, n_wires = _wire_layout(n_states)
        dev = qp.device("default.qubit", wires=n_wires)

        @qp.qnode(dev)
        def circuit():
            uniform_prep_ops(n_states, target_wires, flag, work_wires)
            return qp.state()

        state = circuit()
        block = 2 ** (n_wires - len(target_wires))
        target_amps = state[::block][: 2 ** len(target_wires)]
        assert np.allclose(np.abs(target_amps[:n_states]), np.sqrt(1 / n_states))
        assert np.allclose(target_amps[n_states:], 0.0)

    @pytest.mark.parametrize("n_states", [2, 4, 8, 16])
    def test_power_of_two(self, n_states):
        """Test that powers of two reduce to plain Hadamards over the whole register."""
        probs = _target_probs(n_states)
        assert np.allclose(probs, 1 / n_states)

    def test_single_state(self):
        """Test that n_states = 1 uses zero target wires and leaves the register in |0>."""
        with qp.queuing.AnnotatedQueue() as q:
            uniform_prep_ops(1, [], flag=0, work_wires=[1])
        assert len(q.queue) == 0

    def test_wrong_target_wire_count_raises(self):
        """Test that  target register of the wrong size raises a clear error."""
        # n_states = 5 needs 3 target wires; give it 2.
        with pytest.raises(ValueError, match="target_wires must have 3 wires"):
            with qp.queuing.AnnotatedQueue():
                uniform_prep_ops(5, [0, 1], flag=2, work_wires=[3, 4])

    def test_non_positive_n_states_raises(self):
        """Test that an error is raised when n_states is not a positive integer."""
        with pytest.raises(ValueError, match="n_states must be at least 1"):
            with qp.queuing.AnnotatedQueue():
                uniform_prep_ops(n_states=0, target_wires=[0, 1, 2], flag=3, work_wires=[4, 5])

def _reconstruct_amplitudes(alt, keep, mu):
    """Exact ground-truth distribution from the integer alias tables (Eq. 29 from 	arXiv:1805.03662)."""
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


@pytest.mark.parametrize("L, mu", [(2, 4), (3, 5), (4, 6), (8, 5), (16, 7)])
def test_alias_sampling_wires(L, mu):
    """Test that alias_sampling_wires returns the correct number of wires for a given L and mu."""
    logL = max((L - 1).bit_length(), 1)
    req = alias_sampling_wires(L, mu)
    assert req["target_wires"] == logL
    # sigma(mu) + alt(logL) + keep(mu) + flag(1) + comparator scratch(mu-1)
    assert req["temp_wires"] == mu + logL + mu + 1 + max(mu - 1, 0)
    assert req["work_wires"] == 1 + (max(logL, mu, 2) + 4)


class TestAliasSampling:
    """Test the alias sampling circuit."""

    @pytest.mark.parametrize("L", [2, 3, 4, 5, 6])
    def test_marginal_matches_reconstruction(self, L):
        """Test that the probability distribution matches the classical reconstruction through alias sampling."""
        mu = 4
        rng = np.random.default_rng(L * 13 + 1)
        w = rng.random(L) + 0.05
        recon = _reconstruct_amplitudes(*_build_alias_tables(w, mu), mu)

        req = alias_sampling_wires(L, mu)
        n = req["target_wires"] + req["temp_wires"] + req["work_wires"]
        wires, temp, work = np.split(np.arange(n), np.cumsum([req["target_wires"], req["temp_wires"]]))

        dev = qp.device("lightning.qubit", wires=n)
        @qp.qnode(dev)
        def circuit():
            alias_sampling(w, mu, wires, temp, work)
            return qp.probs(wires=wires)
        probs = np.asarray(circuit())

        assert np.allclose(probs[:L], recon, atol=1e-9)

    @pytest.mark.parametrize("L", [2, 3, 4, 5, 6])
    def test_no_leakage(self, L):
        """Test that the probability value on indices >= L is negligible."""
        mu = 4
        rng = np.random.default_rng(L * 13 + 1)
        w = rng.random(L) + 0.05

        req = alias_sampling_wires(L, mu)
        n = req["target_wires"] + req["temp_wires"] + req["work_wires"]
        wires, temp, work = np.split(np.arange(n), np.cumsum([req["target_wires"], req["temp_wires"]]))

        dev = qp.device("lightning.qubit", wires=n)
        @qp.qnode(dev)
        def circuit():
            alias_sampling(w, mu, wires, temp, work)
            return qp.probs(wires=wires)
        probs = np.asarray(circuit())

        assert np.isclose(probs[:L].sum(), 1.0, atol=1e-6)
        assert np.allclose(probs[L:], 0.0, atol=1e-9)
