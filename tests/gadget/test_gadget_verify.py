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

"""Unit tests for the algebraic checks in ``pennylane.gadget.verify``.

Every test here runs with ``simulate=False``; the stim-backed checks are covered in
``test_gadget_simulate.py``.
"""

from dataclasses import replace

import numpy as np
import pytest

from pennylane import gadget
from pennylane.gadget.library import rep_chain, rep_code_zz_merge


def _statuses(receipt):
    return {c.name: c.status for c in receipt.checks}


def _detail(receipt, name):
    (check,) = [c for c in receipt.checks if c.name == name]
    return check.detail


class TestReceipt:
    """Tests for receipts."""

    def test_unknown_status(self):
        """Test that only the four known statuses can be recorded."""
        with pytest.raises(ValueError, match="unknown status 'maybe'"):
            gadget.Receipt("g", "f").add("c", "maybe", "")

    def test_report_orders_worst_first(self):
        """Test that the report lists failures before warnings, skips and passes."""
        receipt = gadget.Receipt("g", "abc")
        for status in ("pass", "skip", "warn", "fail"):
            receipt.add(f"check.{status}", status, status)
        lines = receipt.report().splitlines()
        assert lines[0] == "verification of g [abc]: 1 FAILURE(S)"
        assert [line.split()[0] for line in lines[1:]] == ["[FAIL]", "[WARN]", "[SKIP]", "[PASS]"]
        assert not receipt.ok
        assert [c.name for c in receipt.failures] == ["check.fail"]

    def test_check_str(self):
        """Test the one-line form of a check."""
        assert str(gadget.Check("a.b", "pass", "fine")) == "[PASS] a.b: fine"


class TestLibraryGadgets:
    """Tests that the library gadgets verify, with the statuses the design documents."""

    def test_surgery(self, rep_zz):
        """Test the receipt of the repetition-code ZZ merge."""
        receipt, layout = gadget.verify(rep_zz[2].program, simulate=False)
        assert receipt.ok
        assert _statuses(receipt) == {
            "layout.freshness": "pass",
            "frame.shared": "pass",
            "phase.entry_matches_code": "pass",
            "phase.css": "pass",
            "transition.consistency": "pass",
            "k.entry": "pass",
            "k.gauged": "pass",
            "logical.action": "pass",
            "detector.coverage": "warn",
            "rounds.budget": "pass",
            "claims.discipline": "warn",
            "simulation": "skip",
        }
        assert _detail(receipt, "simulation") == "disabled by caller"
        assert "1 record(s) have no first-round detector" in _detail(receipt, "detector.coverage")
        assert layout.n_detectors == 22

    def test_memory(self, steane_mem):
        """Test the receipt of the Steane memory gadget."""
        receipt, _ = gadget.verify(steane_mem[2].program, simulate=False)
        assert receipt.ok
        assert _statuses(receipt) == {
            "layout.freshness": "pass",
            "frame.shared": "pass",
            "phase.entry_matches_code": "pass",
            "phase.css": "pass",
            "transition.consistency": "pass",
            "k.entry": "pass",
            "k.gauged": "skip",
            "logical.action": "skip",
            "detector.coverage": "pass",
            "rounds.budget": "skip",
            "claims.discipline": "warn",
            "simulation": "skip",
        }
        assert _detail(receipt, "rounds.budget") == "no merged phase"

    def test_aux_merge(self, aux_merge):
        """Test that a merge through an initialized, then detached, auxiliary qubit verifies."""
        receipt, _ = gadget.verify(aux_merge.program, simulate=False)
        assert receipt.ok
        assert _detail(receipt, "frame.shared") == "6 data + 1 auxiliary in one frame of 7"

    def test_claims_are_carried_uncertified(self, rep_zz):
        """Test that author claims are carried forward unchanged and flagged."""
        prog = rep_zz[2].program
        receipt, _ = gadget.verify(prog, simulate=False)
        assert receipt.claims == list(prog.claims)
        assert "author-asserted, not established here" in _detail(receipt, "claims.discipline")
        assert "claims carried forward:" in receipt.report()

    def test_supplied_layout_is_returned(self, rep_zz):
        """Test that a supplied layout is used and returned rather than re-derived."""
        prog = rep_zz[2].program
        layout = gadget.derive_detectors(prog)
        _, used = gadget.verify(prog, layout, simulate=False)
        assert used is layout


class TestFailures:
    """Tests that each check fails on the mistake it exists to catch."""

    def test_stale_layout(self):
        """Test that a layout derived from a different gadget body is rejected."""
        old = rep_code_zz_merge(d=3, merged_rounds=2)[2].program
        new = rep_code_zz_merge(d=3, merged_rounds=3)[2].program
        receipt, _ = gadget.verify(new, gadget.derive_detectors(old), simulate=False)
        assert _statuses(receipt)["layout.freshness"] == "fail"
        assert "re-derive it" in _detail(receipt, "layout.freshness")

    def test_entry_phase_must_accept_code(self, rep_zz):
        """Test that the entry phase must contain the declared code's checks and protect the
        declared number of logical qubits."""
        code, _, _ = rep_zz
        hz_a, _, _ = rep_chain(3, offset=0, n_frame=6)
        partial = gadget.Phase("partial", np.zeros((0, 6), np.uint8), hz_a, np.ones(6, bool))

        @gadget.define(action=gadget.Action.idle(), code=code, phases=(partial,))
        def half(handle):
            handle, _ = gadget.rounds(handle, 1, record="r")
            return handle

        receipt, _ = gadget.verify(half.program, simulate=False)
        assert _statuses(receipt)["phase.entry_matches_code"] == "fail"
        assert "Z checks of code rep3x2 are not in the row space" in _detail(
            receipt, "phase.entry_matches_code"
        )
        assert _detail(receipt, "k.entry") == (
            "entry phase partial protects k=4 but code rep3x2 has k=2"
        )

    def test_ungauged_aux(self, rep_zz):
        """Test that a merged phase leaving auxiliary qubits unchecked fails both the gauging and
        the transition checks: the auxiliary qubits are undetected logical degrees of freedom."""
        code, (base6, merged6), _ = rep_zz
        pad = np.zeros((base6.hz.shape[0], 2), dtype=np.uint8)
        active = np.array([True] * 6 + [False] * 2)
        base = gadget.Phase("base", np.zeros((0, 8), np.uint8), np.hstack([base6.hz, pad]), active)
        loose = gadget.Phase(
            "loose",
            np.zeros((0, 8), np.uint8),
            np.hstack([merged6.hz, np.zeros((5, 2), np.uint8)]),
            np.ones(8, dtype=bool),
        )

        @gadget.define(
            action=gadget.Action.measure(("z", (0, 1))),
            code=code,
            phases=(base, loose),
            n_data=6,
        )
        def ungauged(handle):
            handle, _ = gadget.rounds(handle, 1, record="pre")
            handle = gadget.deform(handle, to="loose", init={6: "z", 7: "z"})
            handle, checks = gadget.rounds(handle, 3, record="merged")
            return handle, gadget.observe(checks.product((4,)), index=0)

        receipt, _ = gadget.verify(ungauged.program, simulate=False)
        assert {c.name for c in receipt.failures} == {"k.gauged", "transition.consistency"}
        assert "expected k=1 in the merged phase but got loose: k=3" in _detail(receipt, "k.gauged")
        assert "qubit 7 initialized in Z but no Z check of that phase touches it" in _detail(
            receipt, "transition.consistency"
        )

    def test_detached_qubit_must_leave_frame(self, aux_pair):
        """Test that measuring out a qubit that stays active is a transition failure."""
        code, base, merged = aux_pair

        @gadget.define(action=gadget.Action.idle(), code=code, phases=(base, merged), n_data=6)
        def leaky(handle):
            handle = gadget.deform(handle, to="merged", init={6: "z"})
            handle, _ = gadget.rounds(handle, 1, record="merged")
            handle, _ = gadget.detach(handle, to="base", measure_out={0: "z", 6: "z"}, record="out")
            return handle

        receipt, _ = gadget.verify(leaky.program, simulate=False)
        assert _statuses(receipt)["transition.consistency"] == "fail"
        assert "qubit 0 is measured out but stays active" in _detail(
            receipt, "transition.consistency"
        )

    def test_logical_action_uses_completed_parity(self, rep_zz):
        """Test that a supplied layout reporting the uncompleted parity fails: the parity
        as written measures the join check, not the declared logical."""
        prog = rep_zz[2].program
        layout = gadget.derive_detectors(prog)
        (obs,) = layout.observables
        uncompleted = replace(layout, observables=(replace(obs, expr=obs.author_expr),))
        receipt, _ = gadget.verify(prog, uncompleted, simulate=False)
        assert _detail(receipt, "logical.action") == (
            "outcome 0: the completed parity measures Z on qubits [2, 3] but the declared "
            "logical Z on logical qubits [0, 1] has support [0, 3]"
        )

    def test_logical_action_missing_observable(self, rep_zz):
        """Test that a declared outcome must have an observable in the layout."""
        prog = rep_zz[2].program
        layout = replace(gadget.derive_detectors(prog), observables=())
        receipt, _ = gadget.verify(prog, layout, simulate=False)
        assert "outcome 0 is declared but the detector layout has no observable" in _detail(
            receipt, "logical.action"
        )

    def test_round_budget(self):
        """Test that a fault-distance claim above the merged round count fails."""
        thin = rep_code_zz_merge(d=3, merged_rounds=1)[2].program
        over = thin.with_claims(gadget.DistanceClaim(3, "phenomenological"))
        receipt, _ = gadget.verify(over, simulate=False)
        assert [c.name for c in receipt.failures] == ["rounds.budget"]
        assert _detail(receipt, "rounds.budget") == (
            "a fault distance of 3 needs at least 3 rounds in the merged phase to protect "
            "against measurement errors, but the body schedules 1"
        )

    def test_round_budget_ignores_static_claims(self, rep_zz):
        """Test that a static code distance is not checked against the round budget."""
        prog = replace(rep_zz[2].program, claims=(gadget.DistanceClaim(3, "static"),))
        receipt, _ = gadget.verify(prog, simulate=False)
        assert _statuses(receipt)["rounds.budget"] == "skip"

    @pytest.mark.parametrize(
        "claims, status",
        [((), "skip"), ((gadget.DistanceClaim(3, "static", True, "known"),), "pass")],
    )
    def test_claims_discipline(self, rep_zz, claims, status):
        """Test that the claims check passes only when every claim is certified."""
        prog = replace(rep_zz[2].program, claims=claims)
        receipt, _ = gadget.verify(prog, simulate=False)
        assert _statuses(receipt)["claims.discipline"] == status

    def test_simulation_size_limit(self, steane_mem):
        """Test that simulation is skipped, not passed, above the size limit."""
        receipt, _ = gadget.verify(steane_mem[2].program, max_sim_qubits=4)
        assert _statuses(receipt)["simulation"] == "skip"
        assert _detail(receipt, "simulation") == "frame of 7 qubits exceeds max_sim_qubits=4"
