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
This module contains the derivation of detectors and logical observables from a traced
gadget.

:func:`derive_detectors` walks the traced program while keeping, for every operator whose
value is known at that point, the measurement parity that last determined it. Each check
outcome whose operator is known becomes a detector against that parity. A check measured
for the first time whose operator is not known has a random first outcome; it gets no
detector and is reported as undetermined. The same bookkeeping completes each declared
outcome parity, so the observable measures the declared logical operator exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _gf2
from .ir import (
    Deform,
    Detach,
    GadgetError,
    GadgetProgram,
    Observe,
    RecordExpr,
    RecordTerm,
    Rounds,
    entry_syndrome,
)

# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Detector:
    """A parity of measurement outcomes that is deterministic in the absence of faults.

    Args:
        name (str): identifier, derived from the record block, round and check
        expr (~.RecordExpr): the parity
        axis (str): Pauli axis of the checks involved, ``"x"`` or ``"z"``
        kind (str): Why the parity is deterministic. ``"entry"``: the first round of the
            entry phase, against the entry syndrome. ``"repeat"``: a check measured again.
            ``"init"``: a check fixed by preparing auxiliary qubits. ``"readout"``: a
            product of readouts equal to a known operator.
    """

    name: str
    expr: RecordExpr
    axis: str
    kind: str


@dataclass(frozen=True)
class LogicalObservable:
    """The parity that reveals one declared logical measurement outcome.

    Args:
        index (int): outcome index in the declared action
        expr (~.RecordExpr): The completed parity. This is the parity a decoder must
            report as the outcome.
        author_expr (~.RecordExpr): the parity as declared in the gadget body
        operator (array[int]): support of the Pauli operator measured, over the frame
        axis (str): Pauli axis, ``"x"`` or ``"z"``
    """

    index: int
    expr: RecordExpr
    author_expr: RecordExpr
    operator: np.ndarray
    axis: str


@dataclass(frozen=True)
class DetectorLayout:
    """The detectors and observables of one traced gadget.

    Args:
        gadget (str): name of the gadget
        fingerprint (str): Fingerprint of the program the layout was derived from. A
            mismatch with the program means the layout is stale.
        regime (str): noise regime the layout is valid for
        detectors (tuple[~.Detector]): deterministic parities
        observables (tuple[~.LogicalObservable]): outcome parities, one per declared outcome
        entry_width (int): number of entry-syndrome bits the first detectors refer to
        exit_records (dict[int, ~.RecordExpr]): For each check of the exit phase, the
            record holding its latest outcome. A gadget that follows closes its first
            detectors against these.
        undetermined (tuple[tuple[str, str]]): Records with no detector, each with the
            reason. A decoder must not treat these outcomes as deterministic.
        revealed (tuple[tuple[str, tuple[int]]]): A basis, per axis and in reduced row
            echelon form over GF(2), of the logical Pauli products whose value becomes
            determined at some point in the gadget, which means the gadget measures them.
            Each product is ``(axis, logical_qubits)``; every product of basis elements of
            one axis is determined as well.

    **Example**

    >>> from pennylane.ftqc import gadget
    >>> from pennylane.ftqc.gadget.library import rep_code_zz_merge
    >>> _, _, measure_zz = rep_code_zz_merge(d=3)
    >>> layout = gadget.derive_detectors(measure_zz.program)
    >>> layout.n_detectors, layout.entry_width
    (22, 4)
    >>> [name for name, _ in layout.undetermined]
    ['merged[r0,c4]']
    >>> layout.detector_matrix(measure_zz.program)[0].shape
    (22, 23)
    """

    gadget: str
    fingerprint: str
    regime: str
    detectors: tuple[Detector, ...]
    observables: tuple[LogicalObservable, ...]
    entry_width: int = 0
    exit_records: dict[int, RecordExpr] = field(default_factory=dict)
    undetermined: tuple[tuple[str, str], ...] = ()
    revealed: tuple[tuple[str, tuple[int, ...]], ...] = ()

    @property
    def n_detectors(self) -> int:
        """Number of detectors."""
        return len(self.detectors)

    def record_index(self, program: GadgetProgram) -> tuple[dict[RecordTerm, int], list[str]]:
        """A flat index over every measurement outcome of the gadget.

        Outcomes are numbered block by block, round by round, check by check.

        Args:
            program (~.GadgetProgram): the program the layout was derived from

        Returns:
            tuple[dict[RecordTerm, int], list[str]]: the index of each outcome, and a label
            for each index
        """
        labels: list[str] = []
        index: dict[RecordTerm, int] = {}
        for block in program.records:
            for r in range(block.rounds):
                for c in range(block.width):
                    index[RecordTerm(block.name, r, c)] = len(labels)
                    labels.append(f"{block.name}[r{r},c{c}]")
        return index, labels

    def detector_matrix(self, program: GadgetProgram) -> tuple[np.ndarray, list[str]]:
        """The detectors as a binary matrix over the flat outcome index.

        Entry-syndrome bits are not columns of the matrix.

        Args:
            program (~.GadgetProgram): the program the layout was derived from

        Returns:
            tuple[array[int], list[str]]: the ``(n_detectors, n_outcomes)`` matrix, and the
            label of each column

        Raises:
            GadgetError: if a detector refers to an outcome the program does not produce
        """
        index, labels = self.record_index(program)
        mat = np.zeros((len(self.detectors), len(labels)), dtype=np.uint8)
        for i, det in enumerate(self.detectors):
            for term in det.expr.terms:
                if term not in index:
                    raise GadgetError(
                        f"detector layout for {self.gadget}: detector {det.name} refers to "
                        f"unknown record {term}"
                    )
                mat[i, index[term]] ^= 1
        return mat, labels

    def summary(self) -> str:
        """A multi-line description of the layout.

        Returns:
            str: the description
        """
        by_kind: dict[str, int] = {}
        for d in self.detectors:
            by_kind[d.kind] = by_kind.get(d.kind, 0) + 1
        lines = [
            f"detector layout for {self.gadget}  [{self.fingerprint}] regime={self.regime}",
            f"  detectors    : {len(self.detectors)} ("
            + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items()))
            + ")",
            f"  entry syndrome: {self.entry_width} bits",
            f"  observables  : {len(self.observables)}",
        ]
        for obs in self.observables:
            support = np.nonzero(obs.operator)[0].tolist()
            lines.append(f"    outcome {obs.index}: {obs.axis.upper()} on qubits {support}")
            lines.append(f"      written  : {obs.author_expr.describe()}")
            lines.append(f"      completed: {obs.expr.describe()}")
        if self.undetermined:
            lines.append(f"  undetermined : {len(self.undetermined)} record(s)")
            for name, why in self.undetermined[:8]:
                lines.append(f"    {name}: {why}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Known-operator bookkeeping
# --------------------------------------------------------------------------------------


class _Tracker:
    """Operators known to be deterministic, each with the parity that last determined it."""

    def __init__(self, n_frame: int):
        self.n_frame = n_frame
        self.vecs: dict[str, list[np.ndarray]] = {"x": [], "z": []}
        self.exprs: dict[str, list[RecordExpr]] = {"x": [], "z": []}

    def assign(self, axis: str, vec: np.ndarray, expr: RecordExpr) -> None:
        """Record that ``vec`` is deterministic, with ``expr`` holding its current value."""
        vec = np.asarray(vec, dtype=np.uint8).reshape(-1)
        for i, existing in enumerate(self.vecs[axis]):
            if np.array_equal(existing, vec):
                self.exprs[axis][i] = expr
                return
        self.vecs[axis].append(vec)
        self.exprs[axis].append(expr)

    def stack(self, axis: str) -> np.ndarray:
        """Known operators of one axis, newest first."""
        if not self.vecs[axis]:
            return np.zeros((0, self.n_frame), dtype=np.uint8)
        return np.array(list(reversed(self.vecs[axis])), dtype=np.uint8)

    def reference(self, axis: str, target: np.ndarray) -> RecordExpr | None:
        """The parity holding the current value of ``target``, or ``None`` if unknown.

        Newer assignments are preferred, so a check measured again in a later phase is
        compared with its latest outcome.
        """
        target = np.asarray(target, dtype=np.uint8).reshape(-1)
        if not target.any():
            return RecordExpr()
        rows = self.stack(axis)
        coeffs = _gf2.solve(rows, target)
        if coeffs is None:
            return None
        exprs = list(reversed(self.exprs[axis]))
        out = RecordExpr()
        for i in np.nonzero(coeffs)[0]:
            out = out ^ exprs[int(i)]
        return out

    def measure(self, axis: str, vec: np.ndarray) -> None:
        """Update the known operators for a measurement or preparation of ``vec`` on ``axis``.

        Known operators of the other axis that anticommute with ``vec`` stop being
        deterministic. Each of them but one is multiplied by that one, which leaves products
        that commute with ``vec`` and stay known; the remaining one is dropped.
        """
        other = "z" if axis == "x" else "x"
        vec = np.asarray(vec, dtype=np.uint8).reshape(-1)
        hits = [
            i
            for i, known in enumerate(self.vecs[other])
            if int(known.astype(np.int64) @ vec.astype(np.int64)) % 2
        ]
        if not hits:
            return
        pivot, rest = hits[0], hits[1:]
        for i in rest:
            self.vecs[other][i] = self.vecs[other][i] ^ self.vecs[other][pivot]
            self.exprs[other][i] = self.exprs[other][i] ^ self.exprs[other][pivot]
        del self.vecs[other][pivot]
        del self.exprs[other][pivot]

    def restrict(self, live: np.ndarray) -> None:
        """Keep only operators supported on the live qubits.

        Known single-qubit operators on qubits leaving the frame, such as their readouts, are
        first multiplied into the other known operators, so that an operator whose value
        follows from them and from live-qubit support stays known.
        """
        dead = ~live
        for axis in ("x", "z"):
            singles = {
                int(np.flatnonzero(v)[0]): i
                for i, v in enumerate(self.vecs[axis])
                if v.sum() == 1 and dead[np.flatnonzero(v)[0]]
            }
            for i, vec in enumerate(self.vecs[axis]):
                if i in singles.values():
                    continue
                for q in np.flatnonzero(vec & dead):
                    j = singles.get(int(q))
                    if j is not None:
                        self.vecs[axis][i] = self.vecs[axis][i] ^ self.vecs[axis][j]
                        self.exprs[axis][i] = self.exprs[axis][i] ^ self.exprs[axis][j]
            keep = [i for i, v in enumerate(self.vecs[axis]) if not v[dead].any()]
            self.vecs[axis] = [self.vecs[axis][i] for i in keep]
            self.exprs[axis] = [self.exprs[axis][i] for i in keep]


def _single(n_frame: int, qubit: int) -> np.ndarray:
    v = np.zeros(n_frame, dtype=np.uint8)
    v[qubit] = 1
    return v


# --------------------------------------------------------------------------------------
# Derivation
# --------------------------------------------------------------------------------------


def derive_detectors(program: GadgetProgram, regime: str = "phenomenological") -> DetectorLayout:
    """Derive the detectors and logical observables of a traced gadget.

    The first round of the entry phase is compared with the entry syndrome (see
    :func:`~.entry_syndrome`). Later rounds are compared with the latest outcome of the
    same operator, including across phase changes. A check whose operator is not known
    when it is first measured has a random outcome, so it gets no first-round detector and
    is listed in :attr:`~.DetectorLayout.undetermined`.

    Measuring a check makes the known operators it anticommutes with random, except for
    their products that commute with it, which stay known. When qubits leave the frame,
    operators whose values follow from their readouts and from checks on the remaining
    qubits stay known. A basis of the logical products whose values become known is
    recorded in :attr:`~.DetectorLayout.revealed`.

    Each declared outcome is completed: if its parity differs from the declared logical
    operator by operators whose values are known, the parities holding those values are
    added. Any parity in the right coset gives the same completed observable.

    Args:
        program (~.GadgetProgram): the traced gadget
        regime (str): Noise regime to label the layout with. The derivation is exact for
            the phenomenological regime.

    Returns:
        ~.DetectorLayout: the derived layout

    Raises:
        GadgetError: If a declared outcome mixes X and Z checks, uses a different axis
            from the declared one, or differs from the declared logical operator by an
            operator whose value is not known at that point.

    **Example**

    The ZZ merge of :func:`~.library.rep_code_zz_merge` declares the outcome of the join
    check alone. Completion adds the outcomes of the base checks that turn it into logical
    Z on both blocks:

    >>> from pennylane.ftqc import gadget
    >>> from pennylane.ftqc.gadget.library import rep_code_zz_merge
    >>> _, _, measure_zz = rep_code_zz_merge(d=3)
    >>> (obs,) = gadget.derive_detectors(measure_zz.program).observables
    >>> obs.author_expr.describe()
    'merged[r2,c4]'
    >>> obs.expr.describe()
    'merged[r2,c0] ^ merged[r2,c1] ^ merged[r2,c4]'
    """
    entry_name = _first_phase(program)
    entry = program.phase(entry_name)
    track = _Tracker(program.n_frame)

    # On entry, each entry-phase check is known relative to the entry syndrome.
    for i, axis in enumerate(entry.check_axes):
        track.assign(axis, entry.checks[i], entry_syndrome((i,)))

    detectors: list[Detector] = []
    undetermined: list[tuple[str, str]] = []
    observables: list[LogicalObservable] = []
    revealed: dict[str, list[np.ndarray]] = {"x": [], "z": []}
    last_seen: dict[int, RecordExpr] = {}
    current = entry_name

    for op in program.ops:
        if isinstance(op, Deform):
            for qubit, basis in op.init:
                track.measure(basis, _single(program.n_frame, qubit))
                track.assign(basis, _single(program.n_frame, qubit), RecordExpr())
            current = op.to_phase
            continue

        if isinstance(op, Rounds):
            phase = program.phase(op.phase)
            block = program.record(op.record)
            for c in range(block.width):
                row = phase.checks[c]
                axis = phase.check_axes[c]
                ref = track.reference(axis, row)
                if ref is None:
                    undetermined.append(
                        (
                            f"{op.record}[r0,c{c}]",
                            f"{axis.upper()} check on qubits "
                            f"{np.nonzero(row)[0].tolist()} is measured for the first "
                            "time and is not implied by anything known, so its first "
                            "outcome is random and carries no detector",
                        )
                    )
                else:
                    kind = "entry" if ref.entry else ("repeat" if ref.terms else "init")
                    detectors.append(
                        Detector(
                            name=f"{op.record}/r0/c{c}",
                            expr=block.at(0, c) ^ ref,
                            axis=axis,
                            kind=kind,
                        )
                    )
                for r in range(1, op.count):
                    detectors.append(
                        Detector(
                            name=f"{op.record}/r{r}/c{c}",
                            expr=block.at(r, c) ^ block.at(r - 1, c),
                            axis=axis,
                            kind="repeat",
                        )
                    )
                track.measure(axis, row)
                track.assign(axis, row, block.at(op.count - 1, c))
                last_seen[c] = block.at(op.count - 1, c)
            _record_revealed(program, track, revealed)
            current = op.phase
            continue

        if isinstance(op, Detach):
            block = program.record(op.record)
            index_of = {q: i for i, (q, _) in enumerate(op.measure_out)}
            for axis in ("x", "z"):
                in_axis = [q for q, b in op.measure_out if b == axis]
                if not in_axis:
                    continue
                mask = np.zeros(program.n_frame, dtype=bool)
                mask[in_axis] = True
                combos = _gf2.supported_combinations(track.stack(axis), mask)
                for j, vec in enumerate(combos):
                    readout = RecordExpr()
                    for q in np.nonzero(vec)[0]:
                        readout = readout ^ block.at(0, index_of[int(q)])
                    ref = track.reference(axis, vec)
                    if ref is None or not readout:
                        continue
                    detectors.append(
                        Detector(
                            name=f"{op.record}/{axis}/{j}",
                            expr=readout ^ ref,
                            axis=axis,
                            kind="readout",
                        )
                    )
            for q, basis in op.measure_out:
                track.measure(basis, _single(program.n_frame, q))
                track.assign(basis, _single(program.n_frame, q), block.at(0, index_of[q]))
            _record_revealed(program, track, revealed)
            track.restrict(program.phase(op.to_phase).active)
            current = op.to_phase
            continue

        if isinstance(op, Observe) and op.observable_index is not None:
            observables.append(_observable(program, op, track))

    out_phase = program.phase(current)
    exit_records = {i: last_seen[i] for i in range(out_phase.syndrome_width) if i in last_seen}
    return DetectorLayout(
        gadget=program.name,
        fingerprint=program.fingerprint(),
        regime=regime,
        detectors=tuple(detectors),
        observables=tuple(observables),
        entry_width=entry.syndrome_width,
        exit_records=exit_records,
        undetermined=tuple(undetermined),
        revealed=_revealed_basis(revealed),
    )


def _record_revealed(
    program: GadgetProgram, track: _Tracker, revealed: dict[str, list[np.ndarray]]
) -> None:
    """Add, per axis, generators of the logical products whose value is currently determined.

    A product ``c`` of logical operators ``L`` is determined when ``c @ L`` is a combination
    ``e @ K`` of the known operators ``K``. The pairs ``(c, e)`` with ``c @ L + e @ K = 0``
    are the null space of ``[L; K]`` transposed, and their ``c`` parts span the determined
    products.
    """
    k = program.code.k
    if not k:
        return
    for axis in ("x", "z"):
        logicals = np.array(
            [_logical_operator(program, axis, (q,)) for q in range(k)], dtype=np.uint8
        )
        stacked = np.vstack([logicals, track.stack(axis)])
        for combination in _gf2.null_space(stacked.T.astype(np.uint8)):
            if combination[:k].any():
                revealed[axis].append(combination[:k])


def _revealed_basis(
    revealed: dict[str, list[np.ndarray]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """A canonical basis, per axis, of the recorded logical products."""
    basis = []
    for axis in ("x", "z"):
        if revealed[axis]:
            rows, _ = _gf2.row_reduce(np.array(revealed[axis], dtype=np.uint8))
            basis += [(axis, tuple(int(q) for q in np.flatnonzero(row))) for row in rows]
    return tuple(basis)


def _first_phase(program: GadgetProgram) -> str:
    for op in program.ops:
        if isinstance(op, Rounds):
            return op.phase
        if isinstance(op, (Deform, Detach)):
            break
    return program.phases[0].name


def _operator_of(program: GadgetProgram, expr: RecordExpr) -> tuple[np.ndarray, set[str]]:
    """The Pauli operator a parity measures, and the axes of the outcomes in it."""
    operator = np.zeros(program.n_frame, dtype=np.uint8)
    axes: set[str] = set()
    for term in expr.terms:
        block = program.record(term.block)
        detach = _detach_for(program, block.name)
        if detach is not None:
            qubit, basis = detach.measure_out[term.check]
            operator[qubit] ^= 1
            axes.add(basis)
        else:
            phase = program.phase(block.phase)
            operator ^= phase.checks[term.check]
            axes.add(phase.check_axes[term.check])
    return operator, axes


def _observable(program: GadgetProgram, op: Observe, track: _Tracker) -> LogicalObservable:
    """Complete a declared outcome parity so that it measures the declared logical."""
    operator, axes = _operator_of(program, op.expr)
    if len(axes) > 1:
        raise GadgetError(
            f"{program.name}: outcome {op.observable_index} mixes {sorted(axes)} checks; "
            "a CSS logical measurement outcome must come from a single axis"
        )
    axis = axes.pop() if axes else "z"

    target = _declared_operator(program, op.observable_index, axis)
    if target is None or not (operator ^ target).any():
        return LogicalObservable(
            index=op.observable_index,
            expr=op.expr,
            author_expr=op.expr,
            operator=operator,
            axis=axis,
        )
    residue = operator ^ target
    ref = track.reference(axis, residue)
    if ref is None:
        raise GadgetError(
            f"{program.name}: outcome {op.observable_index} measures {axis.upper()} on "
            f"qubits {np.nonzero(operator)[0].tolist()}, which differs from the declared "
            f"logical operator (support {np.nonzero(target)[0].tolist()}) by an operator "
            f"on qubits {np.nonzero(residue)[0].tolist()} whose value is not known at "
            "that point. The parity does not reveal the declared logical measurement."
        )
    return LogicalObservable(
        index=op.observable_index,
        expr=op.expr ^ ref,
        author_expr=op.expr,
        operator=target,
        axis=axis,
    )


def _declared_operator(program: GadgetProgram, index: int, axis: str) -> np.ndarray | None:
    """Support, over the frame, of the logical operator outcome ``index`` must measure."""
    action = program.action
    if action.kind != "measure" or index >= len(action.paulis):
        return None
    declared_axis, qubits = action.paulis[index]
    if declared_axis != axis:
        raise GadgetError(
            f"{program.name}: outcome {index} is declared as {declared_axis.upper()} but "
            f"its record parity is built from {axis.upper()} checks"
        )
    return _logical_operator(program, axis, qubits)


def _logical_operator(program: GadgetProgram, axis: str, qubits: tuple[int, ...]) -> np.ndarray:
    """Support, over the frame, of the product of logical ``axis`` on logical ``qubits``."""
    code = program.code
    reps = code.lx if axis == "x" else code.lz
    out = np.zeros(program.n_frame, dtype=np.uint8)
    for q in qubits:
        out[: code.n] ^= reps[q]
    return out


def _detach_for(program: GadgetProgram, record: str) -> Detach | None:
    for op in program.ops:
        if isinstance(op, Detach) and op.record == record:
            return op
    return None


__all__ = ["Detector", "LogicalObservable", "DetectorLayout", "derive_detectors"]
