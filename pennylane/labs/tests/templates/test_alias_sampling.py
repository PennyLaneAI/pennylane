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
from pennylane.labs.templates import uniform_prep_ops


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

    @pytest.mark.parametrize("n_states", [0, -1, -5])
    def test_non_positive_n_states_raises(self, n_states):
        """Test that an error is raised when n_states is not a positive integer."""
        with pytest.raises(ValueError, match="n_states must be at least 1"):
            with qp.queuing.AnnotatedQueue():
                uniform_prep_ops(n_states, [0, 1, 2], flag=3, work_wires=[4, 5])
