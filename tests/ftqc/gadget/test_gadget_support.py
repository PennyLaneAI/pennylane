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

"""Unit tests for toolchain support checks, ``pennylane.ftqc.gadget.support``."""

from dataclasses import replace

import pytest

from pennylane.ftqc import gadget
from pennylane.ftqc.gadget.library import rep_code_zz_merge, steane_memory
from pennylane.ftqc.gadget.support import CATALYST_EVIDENCE


def _gaps(report):
    return [(g.capability, g.blocking) for g in report.gaps]


class TestCatalystToday:
    """Tests against the Catalyst pipeline as it exists today."""

    def test_memory(self, steane_mem):
        """Test that a multi-round memory gadget is blocked only on multi-round decoding."""
        report = gadget.check_support(steane_mem[2].program, toolchain=gadget.CATALYST_CURRENT)
        assert not report.lowerable
        assert _gaps(report) == [
            ("multi-round decoding", True),
            ("fault-tolerant syndrome extraction", False),
        ]
        assert report.satisfied == [
            "k=1 within the toolchain's limit of 1",
            "single-phase gadget, no deformation op needed",
            "syndrome width 6 within the decoder's 64 checks",
            "6 syndrome bits fit the 8-byte payload",
        ]

    def test_single_round_memory_is_lowerable(self):
        """Test that non-blocking gaps are reported but do not block lowering."""
        report = gadget.check_support(steane_memory(rounds=1)[2].program)
        assert report.lowerable
        assert _gaps(report) == [("fault-tolerant syndrome extraction", False)]
        assert "round structure within the toolchain's decode capability" in report.satisfied

    def test_surgery(self, rep_zz):
        """Test every gap reported for the ZZ merge, in order."""
        report = gadget.check_support(rep_zz[2].program)
        assert report.toolchain == gadget.CATALYST_CURRENT.name
        assert _gaps(report) == [
            ("codeblocks with k > 1", True),
            ("code deformation in the IR", True),
            ("per-logical-qubit measurement indexing", True),
            ("multi-round decoding", True),
            ("code family rep3x2", False),
            ("fault-tolerant syndrome extraction", False),
        ]
        assert report.gaps[1].required_by == "the gadget deforms into phase(s) merged, base"
        assert report.report().startswith(
            "support: measure_zz on catalyst-qecl + backline realtime: 6 gap(s)"
        )

    def test_every_gap_cites_evidence(self, rep_zz):
        """Test that every gap is backed by one of the recorded file-and-line citations."""
        report = gadget.check_support(rep_code_zz_merge(d=6)[2].program)
        assert report.gaps
        assert all(g.evidence in CATALYST_EVIDENCE.values() for g in report.gaps)
        assert all(
            g.evidence in CATALYST_EVIDENCE.values()
            for g in gadget.check_support(rep_zz[2].program).gaps
        )

    def test_x_measurement(self, rep_zz):
        """Test that an X measurement is reported against a Z-only toolchain."""
        prog = replace(rep_zz[2].program, action=gadget.Action.measure(("x", (0, 1))))
        report = gadget.check_support(prog)
        assert ("logical X measurement", True) in _gaps(report)


def test_size_limits():
    """Test the decoder-size and transport-payload gaps."""
    prog = rep_code_zz_merge(d=6)[2].program
    tiny = gadget.Toolchain(
        name="tiny", max_decoder_checks=10, max_decoder_qubits=8, payload_bytes=1
    )
    report = gadget.check_support(prog, toolchain=tiny)
    by_name = {g.capability: g for g in report.gaps}
    assert by_name["decoder with more than 10 checks"].required_by == (
        "widest phase produces 11 syndrome bits per round"
    )
    assert by_name["decoder with more than 8 qubits"].required_by == "the gadget frame is 12 qubits"
    assert "a round needs 2 messages" in by_name["realtime transport payload"].workaround


@pytest.mark.parametrize("fixture", ["steane_mem", "rep_zz"])
def test_catalyst_target_is_lowerable(fixture, request):
    """Test that both library gadgets lower against the proposed toolchain."""
    prog = request.getfixturevalue(fixture)[2].program
    report = gadget.check_support(prog, toolchain=gadget.CATALYST_PROPOSED)
    assert report.lowerable
    assert not report.gaps
    assert report.report().splitlines()[0].endswith(": lowerable")


def test_optional_inputs_are_reported(rep_zz):
    """Test that a supplied schedule and layout are summarised."""
    prog = rep_zz[2].program
    report = gadget.check_support(
        prog,
        toolchain=gadget.CATALYST_PROPOSED,
        layout=gadget.derive_detectors(prog),
        schedule=gadget.schedule_gadget(prog),
    )
    assert "extraction depth 2 layers per round (Koenig bound met: True)" in report.satisfied
    assert "22 detectors, 4 entry-syndrome bits" in report.satisfied


def test_gap_str():
    """Test the multi-line form of a gap."""
    gap = gadget.Gap("thing", "a gadget", "file.py:1", "do it", blocking=False)
    assert str(gap).splitlines() == [
        "[degrades] thing",
        "    needed for : a gadget",
        "    evidence   : file.py:1",
        "    workaround : do it",
    ]
