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
This module contains descriptions of compilation toolchains, and :func:`check_support`,
which reports what a toolchain is missing to compile a gadget.

Each limit of :data:`CATALYST_CURRENT` is backed by an entry of
:data:`CATALYST_EVIDENCE` naming the file where it lives. :func:`check_support` only
reports; it never changes a gadget to fit a toolchain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detectors import DetectorLayout
from .ir import Deform, GadgetProgram, Rounds
from .schedule import GadgetSchedule


@dataclass(frozen=True)
class Gap:
    """One thing a toolchain is missing to compile a gadget.

    Args:
        capability (str): short name of what is missing
        required_by (str): what in the gadget needs it
        evidence (str): where the limit is found
        workaround (str): what would have to change, or what can be done instead
        blocking (bool): ``False`` if the gap reduces fidelity or analysis but does not
            prevent compilation
    """

    capability: str
    required_by: str
    evidence: str
    workaround: str
    blocking: bool = True

    def __str__(self) -> str:
        mark = "BLOCKING" if self.blocking else "degrades"
        return (
            f"[{mark}] {self.capability}\n"
            f"    needed for : {self.required_by}\n"
            f"    evidence   : {self.evidence}\n"
            f"    workaround : {self.workaround}"
        )


@dataclass(frozen=True)
class Toolchain:
    """A compilation toolchain, described by the limits it has.

    Args:
        name (str): name of the toolchain
        codes (tuple[str] or None): Names of the codes the toolchain can instantiate, or
            ``None`` if codes can be supplied as data.
        max_k (int): largest number of logical qubits per code block
        supports_deformation (bool): whether the IR can change the measured stabilizer group
        supports_multi_round_decode (bool): whether decoding accepts more than one round
        measurable_axes (tuple[str]): logical measurement axes that can be compiled
        max_decoder_checks (int): number of checks the real-time decoder accepts
        max_decoder_qubits (int): number of qubits the real-time decoder accepts
        payload_bytes (int): bytes of syndrome per real-time transport message
        fault_tolerant_cycle (bool): whether the generated check-measurement circuit is
            fault tolerant
        notes (str): free-text description
    """

    name: str
    codes: tuple[str, ...] | None = ()
    max_k: int = 1
    supports_deformation: bool = False
    supports_multi_round_decode: bool = False
    measurable_axes: tuple[str, ...] = ("z",)
    max_decoder_checks: int = 64
    max_decoder_qubits: int = 64
    payload_bytes: int = 8
    fault_tolerant_cycle: bool = False
    notes: str = ""


CATALYST_CURRENT = Toolchain(
    name="catalyst-qecl + backline realtime",
    codes=("Steane", "Shor913"),
    max_k=1,
    supports_deformation=False,
    supports_multi_round_decode=False,
    measurable_axes=("z",),
    max_decoder_checks=64,
    max_decoder_qubits=64,
    payload_bytes=8,
    fault_tolerant_cycle=False,
    notes="mirrors the limits found in the qecl/qecp lowering and the Triton decoder",
)
"""~.Toolchain: the Catalyst QEC pipeline and the Backline real-time path, with the limits
recorded in :data:`CATALYST_EVIDENCE`"""

CATALYST_PROPOSED = Toolchain(
    name="catalyst-qecl (gadget-capable, proposed)",
    codes=None,
    max_k=64,
    supports_deformation=True,
    supports_multi_round_decode=True,
    measurable_axes=("x", "z"),
    max_decoder_checks=4096,
    max_decoder_qubits=4096,
    payload_bytes=512,
    fault_tolerant_cycle=True,
    notes="the Catalyst QEC pipeline with the limits recorded in CATALYST_EVIDENCE lifted",
)
"""~.Toolchain: a proposed Catalyst pipeline with the limits of :data:`CATALYST_CURRENT`
lifted"""

_QECP = "catalyst frontend/catalyst/python_interface/transforms/qecp"

CATALYST_EVIDENCE = {
    "max_k": f"{_QECP}/convert_qecl_to_qecp.py:412 raises NotImplementedError for k > 1; "
    "stated as a known limitation at line 25",
    "measure_index": f"{_QECP}/convert_qecl_to_qecp.py:313 -- the measure lowering cannot "
    "address an individual logical qubit inside a codeblock, although !qecl.measure does "
    "carry an idx operand",
    "measurable_axes": f"{_QECP}/convert_qecl_to_qecp.py:28 -- only logical Pauli Z "
    "observables are supported for lowering qecl.measure",
    "fault_tolerant_cycle": f"{_QECP}/convert_qecl_to_qecp.py:30 -- the generated QEC cycles "
    "are not fault tolerant, they do not account for syndrome measurement errors",
    "deformation": "no operation in the qecl or qecp dialects changes the measured stabilizer "
    "group; qecl exposes alloc/extract/insert/fabricate/encode/noise/qec/gates/measure only "
    "(17 ops in frontend/catalyst/python_interface/dialects/qecl.py)",
    "multi_round_decode": "qecp.decode_esm_css takes a single syndrome operand and a single "
    "Tanner graph, so one call covers one round "
    "(frontend/catalyst/python_interface/dialects/qecp.py)",
    "decoder_size": "pennylane/backline/decoders/triton/decoder_frontend.py:265-268 caps H at "
    "64 checks and 64 qubits; H is unrolled at Triton compile time in bp_iters.py, so the cap "
    "is a code-generation property rather than a runtime parameter",
    "payload": "pennylane/backline/placement.py:31 sets DEFAULT_MESSAGE_BYTES = 8, matching "
    "the 8-byte payload in the wire protocol",
    "codes": f"{_QECP}/qec_code_lib.py:81,133 defines Steane and Shor913 only",
}
"""dict[str, str]: where each limit of :data:`CATALYST_CURRENT` is found"""


@dataclass
class SupportReport:
    """The result of :func:`check_support`.

    Args:
        gadget (str): name of the gadget
        toolchain (str): name of the toolchain
        gaps (list[~.Gap]): what the toolchain is missing
        satisfied (list[str]): requirements the toolchain meets
    """

    gadget: str
    toolchain: str
    gaps: list[Gap] = field(default_factory=list)
    satisfied: list[str] = field(default_factory=list)

    @property
    def lowerable(self) -> bool:
        """Whether no gap is blocking."""
        return not any(g.blocking for g in self.gaps)

    def report(self) -> str:
        """A multi-line report.

        Returns:
            str: the report
        """
        head = f"support: {self.gadget} on {self.toolchain}: " + (
            "lowerable" if self.lowerable else f"{len(self.gaps)} gap(s)"
        )
        lines = [head]
        for s in self.satisfied:
            lines.append(f"  ok: {s}")
        for g in self.gaps:
            lines.append("  " + str(g).replace("\n", "\n  "))
        return "\n".join(lines)


def check_support(
    program: GadgetProgram,
    toolchain: Toolchain = CATALYST_CURRENT,
    layout: DetectorLayout | None = None,
    schedule: GadgetSchedule | None = None,
) -> SupportReport:
    """Report what a toolchain is missing to compile a gadget.

    Args:
        program (~.GadgetProgram): the traced gadget
        toolchain (~.Toolchain): the toolchain to check against
        layout (~.DetectorLayout or None): if given, its size is included in the report
        schedule (~.GadgetSchedule or None): if given, its depth is included in the report

    Returns:
        ~.SupportReport: the gaps found and the requirements met

    **Example**

    >>> from pennylane.ftqc import gadget
    >>> from pennylane.ftqc.gadget.library import steane_memory
    >>> _, _, memory = steane_memory(rounds=3)
    >>> report = gadget.check_support(memory.program)
    >>> report.lowerable
    False
    >>> [(g.capability, g.blocking) for g in report.gaps]
    [('multi-round decoding', True), ('fault-tolerant syndrome extraction', False)]
    >>> gadget.check_support(memory.program, gadget.CATALYST_PROPOSED).lowerable
    True
    """
    out = SupportReport(gadget=program.name, toolchain=toolchain.name)
    code = program.code

    if code.k > toolchain.max_k:
        out.gaps.append(
            Gap(
                capability=f"codeblocks with k > {toolchain.max_k}",
                required_by=f"code {code.name} has k={code.k}",
                evidence=CATALYST_EVIDENCE["max_k"],
                workaround="lower a k=1 gadget today; multi-logical codeblocks need the "
                "qecl-to-qecp conversion to index logical qubits within a block",
            )
        )
    else:
        out.satisfied.append(f"k={code.k} within the toolchain's limit of {toolchain.max_k}")

    has_deform = any(isinstance(op, Deform) for op in program.ops)
    if has_deform and not toolchain.supports_deformation:
        phases = ", ".join(op.to_phase for op in program.ops if isinstance(op, Deform))
        out.gaps.append(
            Gap(
                capability="code deformation in the IR",
                required_by=f"the gadget deforms into phase(s) {phases}",
                evidence=CATALYST_EVIDENCE["deformation"],
                workaround="emit the proposed gadget.deform / gadget.rounds ops and add a "
                "qecl-level pass that expands them; until then only single-phase gadgets "
                "lower end to end",
            )
        )
    elif not has_deform:
        out.satisfied.append("single-phase gadget, no deformation op needed")
    else:
        out.satisfied.append("toolchain supports deformation")

    for axis, _ in program.action.paulis:
        if axis not in toolchain.measurable_axes:
            out.gaps.append(
                Gap(
                    capability=f"logical {axis.upper()} measurement",
                    required_by=f"declared action {program.action}",
                    evidence=CATALYST_EVIDENCE["measurable_axes"],
                    workaround="conjugate the measurement by a logical basis change, or "
                    "extend the measure lowering to both axes",
                )
            )
            break
    else:
        if program.action.paulis:
            out.satisfied.append(
                f"measurement axes {sorted({a for a, _ in program.action.paulis})} supported"
            )

    if program.action.kind == "measure" and program.action.paulis:
        multi_qubit = any(len(qubits) > 1 for _, qubits in program.action.paulis)
        if multi_qubit and toolchain.max_k <= 1:
            out.gaps.append(
                Gap(
                    capability="per-logical-qubit measurement indexing",
                    required_by="the action measures a product across more than one "
                    "logical qubit",
                    evidence=CATALYST_EVIDENCE["measure_index"],
                    workaround="the idx operand already exists on qecl.measure; the "
                    "conversion pass needs to honour it",
                )
            )

    rounds = [op for op in program.ops if isinstance(op, Rounds)]
    if any(op.count > 1 for op in rounds) and not toolchain.supports_multi_round_decode:
        out.gaps.append(
            Gap(
                capability="multi-round decoding",
                required_by=f"the gadget schedules {program.total_rounds} rounds and its "
                "detectors span consecutive rounds",
                evidence=CATALYST_EVIDENCE["multi_round_decode"],
                workaround="pass the detector matrix rather than a single-round Tanner "
                "graph, and give the decode op a window operand",
            )
        )
    else:
        out.satisfied.append("round structure within the toolchain's decode capability")

    width = program.max_syndrome_width
    if width > toolchain.max_decoder_checks:
        out.gaps.append(
            Gap(
                capability=f"decoder with more than {toolchain.max_decoder_checks} checks",
                required_by=f"widest phase produces {width} syndrome bits per round",
                evidence=CATALYST_EVIDENCE["decoder_size"],
                workaround="the parity-check matrix is unrolled at Triton compile time, so "
                "raising the cap is a code-generation change, not a parameter",
            )
        )
    else:
        out.satisfied.append(
            f"syndrome width {width} within the decoder's {toolchain.max_decoder_checks} checks"
        )

    if program.n_frame > toolchain.max_decoder_qubits:
        out.gaps.append(
            Gap(
                capability=f"decoder with more than {toolchain.max_decoder_qubits} qubits",
                required_by=f"the gadget frame is {program.n_frame} qubits",
                evidence=CATALYST_EVIDENCE["decoder_size"],
                workaround="same code-generation change as the check cap",
            )
        )

    bits_per_round = width
    if bits_per_round > toolchain.payload_bytes * 8:
        out.gaps.append(
            Gap(
                capability="realtime transport payload",
                required_by=f"{bits_per_round} syndrome bits per round",
                evidence=CATALYST_EVIDENCE["payload"],
                workaround=f"{toolchain.payload_bytes * 8} bits fit in one slot today, so a "
                f"round needs {-(-bits_per_round // (toolchain.payload_bytes * 8))} messages "
                "or a wider slot",
            )
        )
    else:
        out.satisfied.append(
            f"{bits_per_round} syndrome bits fit the {toolchain.payload_bytes}-byte payload"
        )

    if toolchain.codes is not None and code.name not in toolchain.codes:
        out.gaps.append(
            Gap(
                capability=f"code family {code.name}",
                required_by="the code the gadget is written against",
                evidence=CATALYST_EVIDENCE["codes"],
                workaround="codes must enter as typed data from the frontend rather than "
                "being selected by name from a fixed library",
                blocking=False,
            )
        )

    if not toolchain.fault_tolerant_cycle:
        out.gaps.append(
            Gap(
                capability="fault-tolerant syndrome extraction",
                required_by="any phenomenological or circuit-level distance claim about "
                "the lowered circuit",
                evidence=CATALYST_EVIDENCE["fault_tolerant_cycle"],
                workaround="the frontend's distance claims stay phenomenological until the "
                "emitted cycle is fault tolerant; do not report them as circuit level",
                blocking=False,
            )
        )

    if schedule is not None:
        out.satisfied.append(
            f"extraction depth {schedule.max_depth} layers per round (Koenig bound met: "
            f"{all(s.optimal for s in schedule.per_phase)})"
        )
    if layout is not None:
        out.satisfied.append(
            f"{layout.n_detectors} detectors, {layout.entry_width} entry-syndrome bits"
        )
    return out


__all__ = [
    "Gap",
    "Toolchain",
    "SupportReport",
    "check_support",
    "CATALYST_CURRENT",
    "CATALYST_PROPOSED",
    "CATALYST_EVIDENCE",
]
