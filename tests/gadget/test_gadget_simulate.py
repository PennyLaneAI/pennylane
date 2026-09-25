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

"""Tests for the stim-backed checks in ``pennylane.gadget.simulate`` and ``verify``."""

from dataclasses import replace

import numpy as np
import pytest

from pennylane import gadget
from pennylane.gadget.library import rep_code_zz_merge

stim = pytest.importorskip("stim")
simulate = pytest.importorskip("pennylane.gadget.simulate")

pytestmark = pytest.mark.external


def _statuses(receipt):
    return {c.name: c.status for c in receipt.checks}


def _detail(receipt, name):
    (check,) = [c for c in receipt.checks if c.name == name]
    return check.detail


class TestBuildCircuit:
    """Tests for the phenomenological circuit built from a gadget and its layout."""

    @pytest.mark.parametrize(
        "fixture, n_meas, n_det, n_obs",
        [("steane_mem", 6 + 18, 18, 0), ("rep_zz", 4 + 23, 22, 1), ("aux_merge", 4 + 27, 25, 1)],
    )
    def test_sizes(self, fixture, n_meas, n_det, n_obs, request):
        """Test that the circuit has an ideal entry round plus every gadget record, and one
        stim detector and observable per layout entry."""
        defn = request.getfixturevalue(fixture)
        prog = (defn[2] if isinstance(defn, tuple) else defn).program
        layout = gadget.derive_detectors(prog)
        circuit, rec, vsyn, count = simulate.build_circuit(prog, layout)
        assert count == n_meas == circuit.num_measurements
        assert len(rec) + len(vsyn) == n_meas
        assert circuit.num_detectors == n_det
        assert circuit.num_observables == n_obs

    def test_noiseless_detectors_do_not_fire(self, rep_zz):
        """Test that every derived detector is silent without noise."""
        prog = rep_zz[2].program
        circuit, _, _, _ = simulate.build_circuit(prog, gadget.derive_detectors(prog), p=0.0)
        dets = circuit.compile_detector_sampler().sample(shots=64)
        assert not np.any(dets)

    def test_empty_parity_is_unsupported(self, rep_zz):
        """Test that an empty detector parity cannot be emitted."""
        prog = rep_zz[2].program
        layout = gadget.derive_detectors(prog)
        empty = gadget.Detector("empty", gadget.RecordExpr(), "z", "repeat")
        bad = replace(layout, detectors=(empty,))
        with pytest.raises(simulate.SimulationUnsupported, match="empty parity"):
            simulate.build_circuit(prog, bad)


class TestCheck:
    """Tests for the determinism and distance search."""

    def test_memory_has_no_outcome_to_corrupt(self, steane_mem):
        """Test that a memory gadget is deterministic but has no outcome distance."""
        prog = steane_mem[2].program
        result = simulate.check(prog, gadget.derive_detectors(prog))
        assert result.deterministic
        assert result.distance is None
        assert result.detail == "no observables declared, so there is no outcome to corrupt"

    def test_surgery_distance(self, rep_zz):
        """Test the phenomenological distance of the ZZ merge outcome."""
        prog = rep_zz[2].program
        result = simulate.check(prog, gadget.derive_detectors(prog))
        assert result.deterministic
        assert result.distance == 3
        assert isinstance(result.dem, stim.DetectorErrorModel)

    def test_detector_missing_entry_reference_is_rejected(self, steane_mem):
        """Test that a first-round X detector that does not close against the input syndrome,
        and is therefore random, is caught as non-deterministic."""
        prog = steane_mem[2].program
        layout = gadget.derive_detectors(prog)
        wrong = gadget.Detector("mem/r0/c0-open", prog.record("mem").at(0, 0), "x", "entry")
        bad = replace(layout, detectors=layout.detectors + (wrong,))
        assert not simulate.check(prog, bad).deterministic

        receipt, _ = gadget.verify(prog, bad)
        assert _statuses(receipt)["simulation.determinism"] == "fail"


class TestVerifyWithSimulation:
    """Tests that simulation turns asserted claims into certified ones, or refutes them."""

    def test_surgery_is_certified(self, rep_zz):
        """Test that the ZZ merge verifies and gains a certified phenomenological claim."""
        receipt, _ = gadget.verify(rep_zz[2].program)
        assert receipt.ok
        assert _statuses(receipt)["simulation.determinism"] == "pass"
        assert _statuses(receipt)["simulation.distance"] == "pass"
        asserted, certified = receipt.claims
        assert not asserted.certified
        assert (certified.value, certified.regime, certified.certified) == (
            3,
            "phenomenological",
            True,
        )
        assert certified.method.startswith("stim search_for_undetectable_logical_errors")

    def test_memory(self, steane_mem):
        """Test that the memory gadget is deterministic and its distance check is skipped."""
        receipt, _ = gadget.verify(steane_mem[2].program)
        assert receipt.ok
        assert _statuses(receipt)["simulation.determinism"] == "pass"
        assert _statuses(receipt)["simulation.distance"] == "skip"
        assert not any(c.certified for c in receipt.claims)

    @pytest.mark.parametrize("merged_rounds, distance", [(1, 1), (2, 2), (3, 3), (4, 3)])
    def test_round_budget_sweep(self, merged_rounds, distance):
        """Test that the certified distance is min(merged rounds, d): the new merged check
        has no first-round detector, so flipping it in every round is undetectable until the
        round count reaches the code distance."""
        _, _, defn = rep_code_zz_merge(d=3, merged_rounds=merged_rounds)
        receipt, _ = gadget.verify(defn.program)
        assert receipt.ok
        (certified,) = [c for c in receipt.claims if c.certified]
        assert certified.value == distance

    def test_overclaim_fails(self):
        """Test that claiming the code distance for a single merged round fails twice: on
        the round budget and on the simulated distance."""
        thin = rep_code_zz_merge(d=3, merged_rounds=1)[2].program
        over = thin.with_claims(gadget.DistanceClaim(3, "phenomenological"))
        receipt, _ = gadget.verify(over)
        assert not receipt.ok
        assert [c.name for c in receipt.failures] == ["rounds.budget", "claims.contradiction"]
        assert "simulation found an undetectable logical error of weight 1" in _detail(
            receipt, "claims.contradiction"
        )

    def test_aux_merge(self, aux_merge):
        """Test that a merge through a detached auxiliary qubit is deterministic with distance 3."""
        receipt, _ = gadget.verify(aux_merge.program)
        assert receipt.ok
        assert "has weight 3" in _detail(receipt, "simulation.distance")

    def test_repeated_measurement(self, rep_zz):
        """Test that measuring the same logical twice in an enclosing gadget keeps every detector
        deterministic, including the join check closing against the first merge."""
        code, phases, measure_zz = rep_zz

        @gadget.define(
            action=gadget.Action.measure(("z", (0, 1)), ("z", (0, 1))),
            code=code,
            phases=phases,
        )
        def twice(handle):
            handle, (first,) = measure_zz(handle)
            handle, (second,) = measure_zz(handle)
            return handle, gadget.observe(first, index=0), gadget.observe(second, index=1)

        receipt, layout = gadget.verify(twice.program)
        assert _statuses(receipt)["simulation.determinism"] == "pass"
        assert _statuses(receipt)["logical.action"] == "pass"
        assert [name for name, _ in layout.undetermined] == ["measure_zz#0/merged[r0,c4]"]
        assert "has weight 3" in _detail(receipt, "simulation.distance")
