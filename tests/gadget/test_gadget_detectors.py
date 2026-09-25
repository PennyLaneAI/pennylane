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

"""Unit tests for detector derivation, ``pennylane.gadget.detectors``."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from pennylane import gadget
from pennylane.gadget.ir import RecordTerm
from pennylane.gadget.library import steane_code


def _by_name(layout):
    return {d.name: d for d in layout.detectors}


class TestSurgeryContract:
    """Tests for the layout derived from the repetition-code ZZ merge."""

    @pytest.fixture
    def layout(self, rep_zz):
        """The derived layout."""
        return gadget.derive_detectors(rep_zz[2].program)

    def test_counts(self, layout, rep_zz):
        """Test the detector counts and provenance of the layout."""
        assert layout.n_detectors == 22
        assert Counter(d.kind for d in layout.detectors) == {"entry": 4, "repeat": 18}
        assert layout.entry_width == 4
        assert layout.regime == "phenomenological"
        assert layout.fingerprint == rep_zz[2].program.fingerprint()
        assert layout.gadget == "measure_zz"

    @pytest.mark.parametrize("c", range(4))
    def test_entry_closes_against_entry_syndrome(self, layout, c):
        """Test that the first round of the entry phase is deterministic relative to the
        syndrome the encoded qubits arrived with."""
        det = _by_name(layout)[f"pre/r0/c{c}"]
        assert det.kind == "entry"
        assert det.expr.describe() == f"pre[r0,c{c}] ^ entry[{c}]"

    def test_repeat_across_phase_windows(self, layout):
        """Test that a check re-measured in a new phase closes against its last value."""
        dets = _by_name(layout)
        assert dets["merged/r0/c0"].expr.describe() == "merged[r0,c0] ^ pre[r0,c0]"
        assert dets["merged/r1/c0"].expr.describe() == "merged[r0,c0] ^ merged[r1,c0]"
        assert dets["post/r0/c3"].expr.describe() == "merged[r2,c3] ^ post[r0,c3]"

    def test_new_check_has_no_first_round_detector(self, layout):
        """Test that a check measured for the first time yields no detector and is reported
        as undetermined, while later rounds of it still carry repeat detectors."""
        dets = _by_name(layout)
        assert "merged/r0/c4" not in dets
        assert dets["merged/r1/c4"].kind == "repeat"
        ((name, why),) = layout.undetermined
        assert name == "merged[r0,c4]"
        assert "Z check on qubits [2, 3] is measured for the first time" in why

    def test_observable_completion(self, layout):
        """Test that the author's parity is completed so it measures the declared logical
        Z tensor Z exactly."""
        (obs,) = layout.observables
        assert obs.index == 0
        assert obs.axis == "z"
        assert obs.author_expr.describe() == "merged[r2,c4]"
        assert obs.expr.describe() == "merged[r2,c0] ^ merged[r2,c1] ^ merged[r2,c4]"
        assert np.nonzero(obs.operator)[0].tolist() == [0, 3]

    def test_detector_matrix(self, layout, rep_zz):
        """Test the detector matrix over the flat record index."""
        prog = rep_zz[2].program
        mat, labels = layout.detector_matrix(prog)
        assert mat.shape == (22, 23)
        assert len(labels) == 23
        index, _ = layout.record_index(prog)
        # The undetermined record only enters through the repeat detector of the next round.
        assert index[RecordTerm("merged", 0, 4)] == 8
        assert mat[:, 8].sum() == 1

    def test_exit_records(self, layout):
        """Test that the exit records map each exit-phase check to its last record."""
        assert sorted(layout.exit_records) == [0, 1, 2, 3]
        assert layout.exit_records[2].describe() == "post[r0,c2]"

    def test_regime_label(self, rep_zz):
        """Test that the regime is a label supplied by the caller."""
        assert gadget.derive_detectors(rep_zz[2].program, regime="circuit").regime == "circuit"

    def test_summary(self, layout):
        """Test the diagnostic summary."""
        summary = layout.summary()
        assert "detectors    : 22 (entry=4, repeat=18)" in summary
        assert "outcome 0: Z on qubits [0, 3]" in summary
        assert "undetermined : 1 record(s)" in summary


def test_memory_layout(steane_mem):
    """Test that a memory gadget has a detector on every record and no observables."""
    prog = steane_mem[2].program
    layout = gadget.derive_detectors(prog)
    assert layout.n_detectors == 18
    assert Counter(d.kind for d in layout.detectors) == {"entry": 6, "repeat": 12}
    assert layout.entry_width == 6
    assert layout.observables == ()
    assert layout.undetermined == ()
    mat, _ = layout.detector_matrix(prog)
    assert mat.shape == (18, 18)
    assert mat.any(axis=0).all()


class TestObservableCompletion:
    """Tests that completion accepts any coset representative and rejects the rest."""

    @pytest.mark.parametrize("checks", [(4,), (0,), (1,), (0, 1, 4)])
    def test_representative_does_not_matter(self, rep_zz, checks):
        """Test that every representative of the right coset completes to the same parity."""
        code, phases, _ = rep_zz

        @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases)
        def other_representative(handle):
            handle, _ = gadget.rounds(handle, 1, record="pre")
            handle = gadget.deform(handle, to="merged")
            handle, merged = gadget.rounds(handle, 3, record="merged")
            return handle, gadget.observe(merged.product(checks), index=0)

        (obs,) = gadget.derive_detectors(other_representative.program).observables
        assert obs.expr.describe() == "merged[r2,c0] ^ merged[r2,c1] ^ merged[r2,c4]"

    def test_too_early(self, rep_zz):
        """Test that a parity whose residue is not known yet is rejected with the reason."""
        code, phases, _ = rep_zz

        @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases)
        def too_early(handle):
            handle, pre = gadget.rounds(handle, 1, record="pre")
            outcome = gadget.observe(pre.at(0, 0), index=0)
            handle = gadget.deform(handle, to="merged")
            handle, _ = gadget.rounds(handle, 3, record="merged")
            return handle, outcome

        with pytest.raises(
            gadget.GadgetError,
            match=(
                r"too_early: outcome 0 measures Z on qubits \[0, 1\], which differs from the "
                r"declared logical operator \(support \[0, 3\]\) by an operator on qubits "
                r"\[1, 3\] whose value is not known at that point"
            ),
        ):
            gadget.derive_detectors(too_early.program)

    def test_mixed_axes(self):
        """Test that an outcome built from both X and Z checks is rejected."""
        code = steane_code()

        @gadget.define(
            action=gadget.Action.measure(("z", (0,))),
            code=code,
            phases=(gadget.Phase.from_code("s", code),),
        )
        def mixed(handle):
            handle, r = gadget.rounds(handle, 1, record="r")
            return handle, gadget.observe(r.at(0, 0) ^ r.at(0, 3), index=0)

        with pytest.raises(gadget.GadgetError, match=r"mixes \['x', 'z'\] checks"):
            gadget.derive_detectors(mixed.program)

    def test_declared_axis_mismatch(self, rep_zz):
        """Test that an outcome declared as X cannot be read off Z checks."""
        code, phases, _ = rep_zz

        @gadget.define(action=gadget.Action.measure(("x", (0, 1))), code=code, phases=phases)
        def wrong_axis(handle):
            handle = gadget.deform(handle, to="merged")
            handle, r = gadget.rounds(handle, 1, record="r")
            return handle, gadget.observe(r.product((4,)), index=0)

        with pytest.raises(gadget.GadgetError, match="declared as X but its record parity"):
            gadget.derive_detectors(wrong_axis.program)


class TestDetachContract:
    """Tests for readout detectors produced when qubits leave the frame."""

    @pytest.fixture
    def layout(self, aux_merge):
        """The derived layout of the auxiliary-qubit merge."""
        return gadget.derive_detectors(aux_merge.program)

    def test_readout_detector(self, layout):
        """Test that reading out an auxiliary qubit prepared in Z yields a readout detector."""
        (det,) = [d for d in layout.detectors if d.kind == "readout"]
        assert det.name == "out/z/0"
        assert det.expr.describe() == "out[r0,c0]"

    def test_counts(self, layout):
        """Test the detector counts, with both auxiliary-qubit checks undetermined in round 0."""
        assert Counter(d.kind for d in layout.detectors) == {
            "entry": 4,
            "readout": 1,
            "repeat": 20,
        }
        assert [name for name, _ in layout.undetermined] == ["merged[r0,c4]", "merged[r0,c5]"]

    def test_completion_through_aux(self, layout):
        """Test that the product of the auxiliary-qubit checks completes to the logical Z tensor Z."""
        (obs,) = layout.observables
        assert np.nonzero(obs.operator)[0].tolist() == [0, 3]
        assert obs.expr.describe() == (
            "merged[r2,c0] ^ merged[r2,c1] ^ merged[r2,c4] ^ merged[r2,c5]"
        )

    def test_frame_after_detach(self, layout):
        """Test that checks after the detach close against the merged-phase records."""
        assert _by_name(layout)["post/r0/c0"].expr.describe() == "merged[r2,c0] ^ post[r0,c0]"


def test_detector_matrix_rejects_unknown_records(rep_zz):
    """Test that a supplied layout referring to records the gadget never produced is
    rejected when it is laid out as a matrix."""
    prog = rep_zz[2].program
    layout = gadget.derive_detectors(prog)
    stray = gadget.Detector(
        name="stray",
        expr=gadget.RecordExpr(terms=frozenset({RecordTerm("nope", 0, 0)})),
        axis="z",
        kind="repeat",
    )
    bad = replace(layout, detectors=layout.detectors + (stray,))
    with pytest.raises(gadget.GadgetError, match="detector stray refers to unknown record"):
        bad.detector_matrix(prog)
