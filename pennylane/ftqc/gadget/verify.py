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
This module contains :func:`verify`, which checks a traced gadget and reports the evidence
for each result.

The algebraic checks always run. When `Stim <https://github.com/quantumlib/Stim>`__ is
installed, :func:`verify` also builds a phenomenological noise simulation of the gadget
(see :mod:`pennylane.ftqc.gadget.simulate`), which confirms the derived detectors are
deterministic and certifies a fault distance. A check that cannot run is reported as
skipped, never as passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _gf2
from .codes import DistanceClaim
from .detectors import DetectorLayout, _logical_operator, _operator_of, derive_detectors
from .ir import Deform, Detach, GadgetProgram, Rounds

_STATUS_ORDER = {"fail": 0, "warn": 1, "skip": 2, "pass": 3}


@dataclass(frozen=True)
class Check:
    """The result of one verification check.

    Args:
        name (str): name of the check
        status (str): ``"pass"``, ``"warn"``, ``"skip"`` or ``"fail"``
        detail (str): what was checked and what was found
    """

    name: str
    status: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.status.upper():4}] {self.name}: {self.detail}"


@dataclass
class Receipt:
    """The results of verifying one gadget, and the distance claims it may carry.

    Args:
        gadget (str): name of the gadget
        fingerprint (str): fingerprint of the verified program
        checks (list[~.Check]): check results
        claims (list[~.DistanceClaim]): The author's claims, followed by any claims
            certified during verification.

    **Example**

    >>> from pennylane.ftqc import gadget
    >>> receipt = gadget.Receipt("g", "0123")
    >>> receipt.add("k.entry", "pass", "entry phase protects k=1 as declared")
    >>> receipt.add("rounds.budget", "fail", "too few merged rounds")
    >>> receipt.ok
    False
    >>> print(receipt.report())
    verification of g [0123]: 1 FAILURE(S)
      [FAIL] rounds.budget: too few merged rounds
      [PASS] k.entry: entry phase protects k=1 as declared
    """

    gadget: str
    fingerprint: str
    checks: list[Check] = field(default_factory=list)
    claims: list[DistanceClaim] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        """Record the result of a check.

        Args:
            name (str): name of the check
            status (str): ``"pass"``, ``"warn"``, ``"skip"`` or ``"fail"``
            detail (str): what was checked and what was found

        Raises:
            ValueError: if the status is not one of the four above
        """
        if status not in _STATUS_ORDER:
            raise ValueError(f"unknown status {status!r}")
        self.checks.append(Check(name, status, detail))

    @property
    def ok(self) -> bool:
        """Whether no check failed."""
        return not any(c.status == "fail" for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        """The failed checks."""
        return [c for c in self.checks if c.status == "fail"]

    def report(self) -> str:
        """A multi-line report, with the most severe results first.

        Returns:
            str: the report
        """
        ordered = sorted(self.checks, key=lambda c: (_STATUS_ORDER[c.status], c.name))
        head = f"verification of {self.gadget} [{self.fingerprint}]: " + (
            "OK" if self.ok else f"{len(self.failures)} FAILURE(S)"
        )
        lines = [head] + [f"  {c}" for c in ordered]
        if self.claims:
            lines.append("  claims carried forward:")
            lines += [f"    {c}" for c in self.claims]
        return "\n".join(lines)


def verify(
    program: GadgetProgram,
    layout: DetectorLayout | None = None,
    *,
    simulate: bool = True,
    max_sim_qubits: int = 64,
) -> tuple[Receipt, DetectorLayout]:
    """Check a traced gadget and its detector layout.

    The checks are:

    * ``layout.freshness``: the layout was derived from this program.
    * ``frame.shared``: every phase uses one frame, which contains the data qubits.
    * ``phase.entry_matches_code`` and ``phase.css``: the entry phase measures the code's
      checks, and every phase satisfies the CSS condition.
    * ``transition.consistency``: each qubit prepared by a deformation is checked in the
      new phase, and each qubit read out leaves the frame.
    * ``k.entry`` and ``k.gauged``: the entry phase has the code's ``k``, and a merged
      phase removes exactly one logical degree of freedom per declared outcome.
    * ``logical.action``: each completed outcome parity measures exactly the declared
      logical operator.
    * ``logical.revealed``: no logical Pauli product beyond those the declared action
      measures becomes determined, since determining one destroys that logical information.
    * ``detector.coverage``: records without a detector are reported.
    * ``rounds.budget``: a claimed fault distance is not larger than the number of rounds
      spent in merged phases.
    * ``claims.discipline``: uncertified claims are reported.
    * ``simulation.determinism``, ``simulation.distance`` and ``claims.contradiction``:
      with Stim installed, the detectors are deterministic without noise, the minimum
      undetectable outcome error is found, and no phenomenological claim exceeds it.

    Args:
        program (~.GadgetProgram): the traced gadget
        layout (~.DetectorLayout or None): Layout to check. Derived from ``program`` if
            not given.
        simulate (bool): whether to run the Stim-based checks
        max_sim_qubits (int): frame size above which the Stim-based checks are skipped

    Returns:
        tuple[~.Receipt, ~.DetectorLayout]: the receipt and the layout that was checked

    Raises:
        GadgetError: if ``layout`` is not given and :func:`~.derive_detectors` rejects the
            program

    **Example**

    >>> from pennylane.ftqc import gadget
    >>> from pennylane.ftqc.gadget.library import rep_code_zz_merge
    >>> _, _, measure_zz = rep_code_zz_merge(d=3)
    >>> receipt, layout = gadget.verify(measure_zz.program, simulate=False)
    >>> receipt.ok
    True
    >>> [c.name for c in receipt.checks if c.status == "warn"]
    ['detector.coverage', 'claims.discipline']

    Claiming more distance than the merged rounds can protect fails:

    >>> _, _, thin = rep_code_zz_merge(d=3, merged_rounds=1)
    >>> overclaimed = thin.program.with_claims(gadget.DistanceClaim(3, "phenomenological"))
    >>> receipt, _ = gadget.verify(overclaimed, simulate=False)
    >>> [c.name for c in receipt.failures]
    ['rounds.budget']
    """
    receipt = Receipt(gadget=program.name, fingerprint=program.fingerprint())
    layout = layout or derive_detectors(program)

    _check_freshness(program, layout, receipt)
    _check_frame(program, receipt)
    _check_phase_algebra(program, receipt)
    _check_transitions(program, receipt)
    _check_k_accounting(program, receipt)
    _check_logical_action(program, layout, receipt)
    _check_revealed(program, layout, receipt)
    _check_detector_coverage(program, layout, receipt)
    _check_round_budget(program, receipt)
    _carry_claims(program, receipt)

    if simulate:
        _run_simulation(program, layout, receipt, max_sim_qubits)
    else:
        receipt.add("simulation", "skip", "disabled by caller")

    return receipt, layout


# --------------------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------------------


def _check_freshness(program: GadgetProgram, layout: DetectorLayout, receipt: Receipt) -> None:
    if layout.fingerprint != program.fingerprint():
        receipt.add(
            "layout.freshness",
            "fail",
            f"the detector layout was derived from {layout.fingerprint} but the gadget is "
            f"{program.fingerprint()}; re-derive it",
        )
    else:
        receipt.add("layout.freshness", "pass", "detector layout matches the gadget fingerprint")


def _check_frame(program: GadgetProgram, receipt: Receipt) -> None:
    sizes = {p.n_frame for p in program.phases}
    if len(sizes) != 1:
        receipt.add("frame.shared", "fail", f"phases disagree on frame size: {sorted(sizes)}")
        return
    if program.n_data > program.n_frame:
        receipt.add(
            "frame.shared",
            "fail",
            f"n_data={program.n_data} exceeds the frame of {program.n_frame}",
        )
        return
    receipt.add(
        "frame.shared",
        "pass",
        f"{program.n_data} data + {program.n_aux} auxiliary in one frame of {program.n_frame}",
    )


def _check_phase_algebra(program: GadgetProgram, receipt: Receipt) -> None:
    # Phases enforce the CSS condition when they are created; restate it as evidence and
    # check the entry phase against the code.
    entry = program.phases[0]
    code = program.code
    pad = program.n_frame - code.n
    hx = np.hstack([code.hx, np.zeros((code.hx.shape[0], pad), np.uint8)]) if code.hx.size else None
    hz = np.hstack([code.hz, np.zeros((code.hz.shape[0], pad), np.uint8)]) if code.hz.size else None
    mismatch = []
    if hx is not None and _gf2.rank(np.vstack([entry.hx, hx])) != _gf2.rank(entry.hx):
        mismatch.append("X")
    if hz is not None and _gf2.rank(np.vstack([entry.hz, hz])) != _gf2.rank(entry.hz):
        mismatch.append("Z")
    if mismatch:
        receipt.add(
            "phase.entry_matches_code",
            "fail",
            f"{'/'.join(mismatch)} checks of code {code.name} are not in the row space of "
            f"entry phase {entry.name}; the gadget does not accept the code it declares",
        )
    else:
        receipt.add(
            "phase.entry_matches_code",
            "pass",
            f"entry phase {entry.name} contains the checks of {code.name}",
        )
    receipt.add(
        "phase.css",
        "pass",
        "every phase satisfies Hx Hz^T = 0 (enforced at construction) for phases "
        + ", ".join(p.name for p in program.phases),
    )


def _check_transitions(program: GadgetProgram, receipt: Receipt) -> None:
    problems = []
    for op in program.ops:
        if isinstance(op, Deform):
            target = program.phase(op.to_phase)
            for qubit, basis in op.init:
                mat = target.hz if basis == "z" else target.hx
                if not mat.size or not mat[:, qubit].any():
                    problems.append(
                        f"deform to {op.to_phase}: qubit {qubit} initialized in "
                        f"{basis.upper()} but no {basis.upper()} check of that phase "
                        "touches it, so the preparation is never checked"
                    )
        if isinstance(op, Detach):
            target = program.phase(op.to_phase)
            for qubit, _ in op.measure_out:
                if target.active[qubit]:
                    problems.append(
                        f"detach to {op.to_phase}: qubit {qubit} is measured out but "
                        "stays active in the target phase"
                    )
    if problems:
        receipt.add("transition.consistency", "fail", "; ".join(problems))
    else:
        receipt.add(
            "transition.consistency",
            "pass",
            "every activated qubit is checked and every measured-out qubit leaves the frame",
        )


def _check_k_accounting(program: GadgetProgram, receipt: Receipt) -> None:
    entry = program.phases[0]
    k_entry = entry.k
    t = program.action.n_outcomes
    detail = ", ".join(f"{p.name}: k={p.k}" for p in program.phases)
    if k_entry != program.code.k:
        receipt.add(
            "k.entry",
            "fail",
            f"entry phase {entry.name} protects k={k_entry} but code {program.code.name} "
            f"has k={program.code.k}",
        )
    else:
        receipt.add("k.entry", "pass", f"entry phase protects k={k_entry} as declared")

    targets = [program.phase(op.to_phase) for op in program.ops if isinstance(op, Deform)]
    merged = [p for p in targets if p.k != k_entry]
    if not merged or t == 0:
        receipt.add("k.gauged", "skip", f"no gauging deformation or no outcomes ({detail})")
        return
    bad = [p for p in merged if p.k != k_entry - t]
    if bad:
        receipt.add(
            "k.gauged",
            "fail",
            f"a gadget measuring {t} logical operator(s) must gauge away exactly {t} "
            f"degrees of freedom, expected k={k_entry - t} in the merged phase but got "
            + ", ".join(f"{p.name}: k={p.k}" for p in bad)
            + ". A merged phase with too large k has left an auxiliary degree of freedom "
            "ungauged, which is an undetected logical error channel.",
        )
    else:
        receipt.add(
            "k.gauged",
            "pass",
            f"merged phase(s) protect k={k_entry - t}, consistent with measuring {t} "
            f"logical operator(s) ({detail})",
        )


def _check_logical_action(program: GadgetProgram, layout: DetectorLayout, receipt: Receipt) -> None:
    # The operator is recomputed from the completed parity, independently of the
    # derivation, and must equal the declared logical exactly: a parity that differs by a
    # stabilizer measures a different operator with different byproducts.
    if program.action.kind != "measure":
        receipt.add("logical.action", "skip", f"action {program.action} has no outcome to check")
        return
    by_index = {obs.index: obs for obs in layout.observables}
    problems = []
    for i, (axis, qubits) in enumerate(program.action.paulis):
        obs = by_index.get(i)
        if obs is None:
            problems.append(
                f"outcome {i} is declared but the detector layout has no observable for it"
            )
            continue
        if obs.axis != axis:
            problems.append(
                f"outcome {i}: declared axis {axis.upper()} but the record parity is built "
                f"from {obs.axis.upper()} checks"
            )
            continue
        target = _logical_operator(program, axis, qubits)
        operator, axes = _operator_of(program, obs.expr)
        if len(axes) > 1:
            problems.append(f"outcome {i}: completed parity mixes {sorted(axes)} checks")
            continue
        if not np.array_equal(operator, target):
            problems.append(
                f"outcome {i}: the completed parity measures {axis.upper()} on qubits "
                f"{np.nonzero(operator)[0].tolist()} but the declared logical "
                f"{axis.upper()} on logical qubits {list(qubits)} has support "
                f"{np.nonzero(target)[0].tolist()}"
            )
    if problems:
        receipt.add("logical.action", "fail", "; ".join(problems))
    else:
        receipt.add(
            "logical.action",
            "pass",
            "every completed outcome parity measures exactly the declared logical "
            f"operator of {program.action}",
        )


def _check_revealed(program: GadgetProgram, layout: DetectorLayout, receipt: Receipt) -> None:
    if program.action.kind == "prepare":
        receipt.add(
            "logical.revealed", "skip", f"action {program.action} fixes logical values by design"
        )
        return

    def as_vector(qubits):
        v = np.zeros(program.code.k, dtype=np.uint8)
        for q in qubits:
            v[q] ^= 1
        return v

    def label(axis, qubits):
        return axis.upper() + "".join(f"_{q}" for q in qubits)

    allowed = {
        axis: np.array(
            [as_vector(q) for a, q in program.action.paulis if a == axis], dtype=np.uint8
        ).reshape(-1, program.code.k)
        for axis in ("x", "z")
    }
    extra = [
        label(axis, qubits)
        for axis, qubits in layout.revealed
        if not _gf2.in_row_space(as_vector(qubits), allowed[axis])
    ]
    if extra:
        receipt.add(
            "logical.revealed",
            "fail",
            f"the gadget also determines {', '.join(extra)}, which its declared action "
            f"{program.action} does not measure; measuring them destroys that logical "
            "information",
        )
    else:
        measured = ", ".join(label(a, q) for a, q in layout.revealed) or "no logical product"
        receipt.add(
            "logical.revealed",
            "pass",
            f"the gadget determines only what its declared action measures ({measured})",
        )


def _check_detector_coverage(
    program: GadgetProgram, layout: DetectorLayout, receipt: Receipt
) -> None:
    expected_records = sum(b.rounds * b.width for b in program.records)
    mat, _ = layout.detector_matrix(program)
    touched = int((mat.any(axis=0)).sum()) if mat.size else 0
    status = "pass"
    detail = (
        f"{layout.n_detectors} detectors over {expected_records} records, "
        f"{touched} records participate"
    )
    if layout.undetermined:
        status = "warn"
        detail += (
            f"; {len(layout.undetermined)} record(s) have no first-round detector "
            "because the check is newly introduced -- correct, and the decoder must be "
            "told not to expect determinism there"
        )
    receipt.add("detector.coverage", status, detail)


def _check_round_budget(program: GadgetProgram, receipt: Receipt) -> None:
    claims = [c for c in program.claims if c.regime in ("phenomenological", "circuit")]
    if not claims:
        receipt.add("rounds.budget", "skip", "no fault-distance claim to check rounds against")
        return
    d = max(c.value for c in claims)
    k_entry = program.phases[0].k
    merged_names = {
        op.to_phase
        for op in program.ops
        if isinstance(op, Deform) and program.phase(op.to_phase).k != k_entry
    }
    rounds_in_merged = sum(
        op.count for op in program.ops if isinstance(op, Rounds) and op.phase in merged_names
    )
    if not merged_names:
        receipt.add("rounds.budget", "skip", "no merged phase")
        return
    if rounds_in_merged < d:
        receipt.add(
            "rounds.budget",
            "fail",
            f"a fault distance of {d} needs at least {d} rounds in the merged phase to "
            f"protect against measurement errors, but the body schedules "
            f"{rounds_in_merged}",
        )
    else:
        receipt.add(
            "rounds.budget",
            "pass",
            f"{rounds_in_merged} merged-phase rounds cover a claimed fault distance of {d}",
        )


def _carry_claims(program: GadgetProgram, receipt: Receipt) -> None:
    receipt.claims = list(program.claims)
    uncertified = [c for c in program.claims if not c.certified]
    if uncertified:
        receipt.add(
            "claims.discipline",
            "warn",
            f"{len(uncertified)} claim(s) are author-asserted, not established here: "
            + "; ".join(str(c) for c in uncertified),
        )
    elif program.claims:
        receipt.add("claims.discipline", "pass", "every claim is certified")
    else:
        receipt.add("claims.discipline", "skip", "the gadget makes no distance claim")


def _run_simulation(
    program: GadgetProgram,
    layout: DetectorLayout,
    receipt: Receipt,
    max_sim_qubits: int,
) -> None:
    if program.n_frame > max_sim_qubits:
        receipt.add(
            "simulation",
            "skip",
            f"frame of {program.n_frame} qubits exceeds max_sim_qubits={max_sim_qubits}",
        )
        return
    try:
        from . import simulate as _sim
    except Exception as exc:  # pragma: no cover - import guard
        receipt.add("simulation", "skip", f"stim unavailable: {exc}")
        return
    try:
        result = _sim.check(program, layout)
    except _sim.SimulationUnsupported as exc:
        receipt.add("simulation", "skip", str(exc))
        return

    if result.deterministic:
        receipt.add(
            "simulation.determinism",
            "pass",
            f"stim confirms all {layout.n_detectors} detectors are deterministic under "
            "noiseless phenomenological simulation",
        )
    else:
        receipt.add(
            "simulation.determinism",
            "fail",
            f"stim reports non-deterministic detector(s): {result.detail}",
        )
        return

    if result.distance is None:
        receipt.add("simulation.distance", "skip", result.detail)
        return
    receipt.add(
        "simulation.distance",
        "pass",
        f"minimum undetectable logical error found by stim has weight {result.distance} "
        f"in the phenomenological model ({result.detail})",
    )
    certified = DistanceClaim(
        value=result.distance,
        regime="phenomenological",
        certified=True,
        method="stim search_for_undetectable_logical_errors on the derived detectors",
    )
    receipt.claims.append(certified)
    for claim in program.claims:
        if claim.regime == "phenomenological" and claim.value > result.distance:
            receipt.add(
                "claims.contradiction",
                "fail",
                f"the gadget claims {claim} but simulation found an undetectable logical "
                f"error of weight {result.distance}",
            )


__all__ = ["Check", "Receipt", "verify"]
