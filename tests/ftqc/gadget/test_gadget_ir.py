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

"""Unit tests for the traced form of a gadget, ``pennylane.ftqc.gadget.ir``."""

import numpy as np
import pytest

from pennylane.ftqc import gadget
from pennylane.ftqc.gadget.ir import Deform, Frame, Observe, RecordTerm, Rounds
from pennylane.ftqc.gadget.library import rep_code_zz_merge, steane_code


class TestPhase:
    """Tests for phases: stabilizer groups over a shared qubit frame."""

    def test_from_code_pads_frame(self):
        """Test that lifting a code into a wider frame pads with inactive qubits."""
        phase = gadget.Phase.from_code("s", steane_code(), n_frame=9)
        assert phase.n_frame == 9
        assert phase.n_active == 7
        assert phase.active.tolist() == [True] * 7 + [False] * 2
        assert phase.k == 1
        assert not phase.checks[:, 7:].any()

    def test_from_code_frame_too_small(self):
        """Test that the frame cannot be smaller than the code."""
        with pytest.raises(gadget.GadgetError, match="frame of 5 is smaller than the code's 7"):
            gadget.Phase.from_code("s", steane_code(), n_frame=5)

    def test_checks_and_axes(self):
        """Test that X checks are stacked above Z checks in syndrome order."""
        phase = gadget.Phase.from_code("s", steane_code())
        assert phase.checks.shape == (6, 7)
        assert phase.check_axes == ("x",) * 3 + ("z",) * 3
        assert phase.syndrome_width == 6
        assert (phase.max_check_weight, phase.max_qubit_degree) == (4, 6)

    def test_empty_phase(self):
        """Test a phase that measures nothing."""
        phase = gadget.Phase(
            "idle", np.zeros((0, 2), np.uint8), np.zeros((0, 2), np.uint8), np.ones(2, bool)
        )
        assert phase.checks.shape == (0, 2)
        assert phase.k == 2

    def test_shape_mismatch(self):
        """Test that check matrices must span the whole frame."""
        with pytest.raises(gadget.GadgetError, match="hz has shape"):
            gadget.Phase(
                "p", np.zeros((0, 3), np.uint8), np.ones((1, 2), np.uint8), np.ones(3, bool)
            )

    def test_css_condition(self):
        """Test that a phase must satisfy the CSS condition."""
        with pytest.raises(gadget.GadgetError, match="CSS condition violated"):
            gadget.Phase(
                "p", np.array([[1, 0]], np.uint8), np.array([[1, 1]], np.uint8), np.ones(2, bool)
            )

    def test_support_on_inactive_qubit(self):
        """Test that a check cannot touch a qubit that is not live in the phase."""
        with pytest.raises(gadget.GadgetError, match=r"hz has support on 1 inactive qubit\(s\)"):
            gadget.Phase(
                "p",
                np.zeros((0, 3), np.uint8),
                np.array([[0, 1, 1]], np.uint8),
                np.array([True, True, False]),
            )


class TestRecords:
    """Tests for record blocks and record parities."""

    @pytest.fixture
    def block(self):
        """A 3-round, 5-check block."""
        return gadget.RecordBlock("m", 0, "merged", rounds=3, width=5, axes=("z",) * 5)

    def test_at(self, block):
        """Test that a single record is named by block, round and check."""
        expr = block.at(1, 4)
        assert expr.terms == frozenset({RecordTerm("m", 1, 4)})
        assert expr.describe() == "m[r1,c4]"

    @pytest.mark.parametrize(
        "r, c, what", [(3, 0, "round 3"), (-1, 0, "round -1"), (0, 5, "check 5")]
    )
    def test_at_out_of_range(self, block, r, c, what):
        """Test that records outside the block are rejected."""
        with pytest.raises(gadget.GadgetError, match=f"record m: {what} out of range"):
            block.at(r, c)

    def test_round_and_check_slices(self, block):
        """Test that a round has one record per check and a check one record per round."""
        assert len(block.round(0)) == 5
        assert len(block.check(2)) == 3

    def test_product_uses_final_round(self, block):
        """Test that ``product`` takes the parity of checks in the final round."""
        assert block.product((4,)).describe() == "m[r2,c4]"
        assert len(block.product().terms) == 5

    def test_parity_is_gf2(self, block):
        """Test that XOR-ing a record with itself cancels."""
        a, b = block.at(0, 0), block.at(0, 1)
        assert not a ^ a
        assert (a ^ b ^ a) == b

    def test_describe_is_sorted(self, block):
        """Test that descriptions are stable, with entry-syndrome terms last."""
        expr = block.at(2, 0) ^ gadget.entry_syndrome((3, 1)) ^ block.at(0, 4)
        assert expr.describe() == "m[r0,c4] ^ m[r2,c0] ^ entry[1] ^ entry[3]"
        assert gadget.RecordExpr().describe() == "0"


class TestAction:
    """Tests for logical action declarations."""

    def test_idle(self):
        """Test that idle has no outcomes."""
        action = gadget.Action.idle()
        assert str(action) == "idle"
        assert action.n_outcomes == 0

    def test_prepare(self):
        """Test that preparation is named after its state and has no outcomes."""
        action = gadget.Action.prepare("plus")
        assert str(action) == "prepare(plus)"
        assert action.n_outcomes == 0

    def test_measure(self):
        """Test that a measurement has one outcome per Pauli product, in argument order."""
        action = gadget.Action.measure(("Z", (0, 1)), ("x", [2]))
        assert action.paulis == (("z", (0, 1)), ("x", (2,)))
        assert action.n_outcomes == 2
        assert str(action) == "measure(Z_0_1, X_2)"

    def test_measure_rejects_non_css_axis(self):
        """Test that only X and Z products can be declared."""
        with pytest.raises(gadget.GadgetError, match="axis must be 'x' or 'z', got 'y'"):
            gadget.Action.measure(("y", (0,)))


class TestGadgetProgram:
    """Tests for the traced program of the repetition-code ZZ merge."""

    def test_traced_ops(self, rep_zz):
        """Test the op sequence and that each op consumes the handle the previous one made."""
        _, _, defn = rep_zz
        prog = defn.program
        assert [type(op) for op in prog.ops] == [
            Rounds,
            Deform,
            Rounds,
            Observe,
            Deform,
            Rounds,
            Frame,
        ]
        threaded = [op for op in prog.ops if op.handle_in >= 0]
        assert threaded[0].handle_in == prog.inputs[0]
        for prev, nxt in zip(threaded, threaded[1:]):
            assert nxt.handle_in == prev.handle_out
        assert threaded[-1].handle_out == prog.outputs[0]

    def test_resources(self, rep_zz, steane_mem):
        """Test round, volume and syndrome-width accounting."""
        prog = rep_zz[2].program
        assert prog.total_rounds == 5
        assert prog.spacetime_volume == 30
        assert prog.max_syndrome_width == 5
        assert (prog.n_data, prog.n_aux, prog.n_frame) == (6, 0, 6)
        mem = steane_mem[2].program
        assert (mem.total_rounds, mem.spacetime_volume) == (3, 21)

    def test_lookups(self, rep_zz):
        """Test phase and record lookup by name."""
        prog = rep_zz[2].program
        assert prog.phase("merged").syndrome_width == 5
        assert prog.record("merged").rounds == 3
        with pytest.raises(gadget.GadgetError, match="no phase named 'nope'"):
            prog.phase("nope")
        with pytest.raises(gadget.GadgetError, match="no record block named 'nope'"):
            prog.record("nope")

    def test_summary(self, rep_zz):
        """Test the diagnostic summary."""
        summary = rep_zz[2].program.summary()
        assert "action       : measure(Z_0_1)" in summary
        assert "phases       : base(k=2, m=4), merged(k=1, m=5)" in summary
        assert "records      : pre(1x4), merged(3x5), post(1x4)" in summary
        assert "notes        : Measure logical Z tensor Z" in summary

    def test_fingerprint_is_deterministic(self):
        """Test that retracing gives the same fingerprint and a different schedule does not."""
        a = rep_code_zz_merge(d=3)[2].program.fingerprint()
        b = rep_code_zz_merge(d=3)[2].program.fingerprint()
        c = rep_code_zz_merge(d=3, merged_rounds=2)[2].program.fingerprint()
        assert a == b
        assert a != c

    def test_with_claims(self, rep_zz):
        """Test that adding a claim returns a copy and leaves the fingerprint unchanged, so a
        layout derived before the claim stays fresh."""
        prog = rep_zz[2].program
        claim = gadget.DistanceClaim(2, "phenomenological")
        out = prog.with_claims(claim)
        assert out.claims == prog.claims + (claim,)
        assert len(prog.claims) == 1
        assert out.fingerprint() == prog.fingerprint()
