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

"""Unit tests for syndrome-extraction scheduling, ``pennylane.ftqc.gadget.schedule``."""

import numpy as np
import pytest

from pennylane.ftqc import gadget
from pennylane.ftqc.gadget.library import rep_code_zz_merge, repetition_code, steane_code


def _phases():
    _, (base, merged), _ = rep_code_zz_merge(d=3)
    _, (base5, merged5), _ = rep_code_zz_merge(d=5)
    return [
        gadget.Phase.from_code("steane", steane_code()),
        gadget.Phase.from_code("rep5", repetition_code(5)),
        base,
        merged,
        base5,
        merged5,
    ]


class TestSchedulePhase:
    """Tests for the edge colouring of one phase."""

    @pytest.mark.parametrize("phase", _phases(), ids=lambda p: p.name)
    def test_collision_free_and_complete(self, phase):
        """Test that every check-qubit interaction is scheduled exactly once and that no
        check or qubit takes part in two interactions in the same layer."""
        sched = gadget.schedule_phase(phase)
        scheduled = [edge for layer in sched.layers for edge in layer]
        expected = {
            (i, int(q))
            for i in range(phase.checks.shape[0])
            for q in np.nonzero(phase.checks[i])[0]
        }
        assert sorted(scheduled) == sorted(expected)
        for layer in sched.layers:
            checks, qubits = zip(*layer)
            assert len(set(checks)) == len(checks)
            assert len(set(qubits)) == len(qubits)

    @pytest.mark.parametrize("phase", _phases(), ids=lambda p: p.name)
    def test_meets_konig_bound(self, phase):
        """Test that the depth equals max(check weight, qubit degree)."""
        sched = gadget.schedule_phase(phase)
        assert sched.bound == max(phase.max_check_weight, phase.max_qubit_degree)
        assert sched.optimal

    def test_steane(self):
        """Test the Steane extraction schedule."""
        sched = gadget.schedule_phase(gadget.Phase.from_code("steane", steane_code()))
        assert (sched.depth, sched.bound, sched.n_interactions) == (6, 6, 24)
        assert sched.summary() == "phase steane: depth 6 (optimal), 24 interactions per round"

    def test_empty_phase(self):
        """Test that a phase with no checks has an empty schedule."""
        phase = gadget.Phase(
            "idle", np.zeros((0, 2), np.uint8), np.zeros((0, 2), np.uint8), np.ones(2, bool)
        )
        sched = gadget.schedule_phase(phase)
        assert (sched.depth, sched.bound, sched.optimal) == (0, 0, True)

    def test_summary_above_bound(self):
        """Test that a schedule above the bound says so."""
        sched = gadget.Schedule("p", layers=(((0, 0),), ((0, 1),), ((1, 0),)), bound=2)
        assert not sched.optimal
        assert "ABOVE the bound of 2" in sched.summary()


class TestScheduleGadget:
    """Tests for scheduling every phase a gadget measures."""

    def test_surgery(self, rep_zz):
        """Test the schedule of the ZZ merge: each used phase once, in order of first use."""
        sched = gadget.schedule_gadget(rep_zz[2].program)
        assert [s.phase for s in sched.per_phase] == ["base", "merged"]
        assert [(s.depth, s.n_interactions) for s in sched.per_phase] == [(2, 8), (2, 10)]
        assert sched.rounds_per_phase == (("base", 1), ("merged", 3), ("base", 1))
        assert (sched.total_layers, sched.max_depth) == (10, 2)

    def test_memory(self, steane_mem):
        """Test the schedule of the memory gadget."""
        sched = gadget.schedule_gadget(steane_mem[2].program)
        assert (sched.total_layers, sched.max_depth) == (18, 6)
        assert sched.summary().splitlines() == [
            "schedule for steane_memory",
            "  phase steane: depth 6 (optimal), 24 interactions per round",
            "  total 18 interaction layers; deepest round 6 layers",
        ]
