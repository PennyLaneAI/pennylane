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
This module contains the traced representation of a gadget: phases, handles, records,
operations and the declared logical action.

A gadget body is ordinary Python. Tracing it with :func:`~pennylane.gadget.define` produces a
:class:`GadgetProgram`, a frozen record of the phases it measures, the operations it applies
and the measurement records those operations produce. Every later stage (detector
derivation, verification, scheduling, support checks and emission) consumes the program,
never the Python function.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from . import _gf2
from .codes import CSSCode, DistanceClaim

Basis = Literal["x", "z"]


class GadgetError(ValueError):
    """Raised when a gadget body or its traced program is inconsistent."""


class OwnershipError(GadgetError):
    """Raised when a handle is used after it has been consumed."""


# --------------------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase:
    """A stabilizer group measured over one or more rounds, on a shared qubit frame.

    All phases of a gadget share one qubit frame: data qubits first, then any auxiliary
    qubits. A qubit that is not live in a phase is excluded by ``active`` and must not be
    touched by any of its checks. Moving between phases is how a gadget changes the
    measured stabilizer group, for example to merge two blocks.

    Args:
        name (str): identifier, used in diagnostics and as the IR symbol name
        hx (array[int]): X checks over the frame, shape ``(mx, n_frame)``
        hz (array[int]): Z checks over the frame, shape ``(mz, n_frame)``
        active (array[bool]): mask of the qubits that are live in this phase

    Raises:
        GadgetError: if the check matrices do not span the frame, if they violate the CSS
            condition, or if a check touches an inactive qubit

    **Example**

    >>> from pennylane import gadget
    >>> from pennylane.gadget.library import steane_code
    >>> phase = gadget.Phase.from_code("steane", steane_code())
    >>> phase.syndrome_width, phase.k
    (6, 1)
    >>> phase.check_axes
    ('x', 'x', 'x', 'z', 'z', 'z')
    """

    name: str
    hx: np.ndarray
    hz: np.ndarray
    active: np.ndarray

    def __post_init__(self) -> None:
        n = int(self.active.shape[0])
        for label, mat in (("hx", self.hx), ("hz", self.hz)):
            if mat.ndim != 2 or mat.shape[1] != n:
                raise GadgetError(
                    f"phase {self.name}: {label} has shape {mat.shape}, expected (*, {n})"
                )
        violations = _gf2.commutes(self.hx, self.hz)
        if violations.any():
            raise GadgetError(
                f"phase {self.name}: CSS condition violated, Hx Hz^T != 0 mod 2 "
                f"({int(violations.sum())} anticommuting check pairs)"
            )
        dead = ~self.active
        for label, mat in (("hx", self.hx), ("hz", self.hz)):
            if mat.size and (mat[:, dead].any()):
                bad = int(np.unique(np.nonzero(mat[:, dead])[1]).size)
                raise GadgetError(
                    f"phase {self.name}: {label} has support on {bad} inactive qubit(s)"
                )

    @property
    def n_frame(self) -> int:
        """Size of the shared qubit frame."""
        return int(self.active.shape[0])

    @property
    def n_active(self) -> int:
        """Number of live qubits in this phase."""
        return int(self.active.sum())

    @property
    def k(self) -> int:
        """Number of logical degrees of freedom the phase's stabilizer group leaves free."""
        return self.n_active - _gf2.rank(self.hx) - _gf2.rank(self.hz)

    @property
    def checks(self) -> np.ndarray:
        """X checks stacked above Z checks, in the order their outcomes are recorded."""
        if not self.hx.size and not self.hz.size:
            return np.zeros((0, self.n_frame), dtype=np.uint8)
        return np.vstack([self.hx, self.hz])

    @property
    def check_axes(self) -> tuple[str, ...]:
        """Pauli axis of each row of :attr:`checks`."""
        return ("x",) * self.hx.shape[0] + ("z",) * self.hz.shape[0]

    @property
    def syndrome_width(self) -> int:
        """Number of check outcomes recorded per round."""
        return int(self.hx.shape[0] + self.hz.shape[0])

    @property
    def max_check_weight(self) -> int:
        """Largest check weight in this phase."""
        return int(
            max(
                _gf2.row_weights(self.hx).max(initial=0),
                _gf2.row_weights(self.hz).max(initial=0),
            )
        )

    @property
    def max_qubit_degree(self) -> int:
        """Largest number of checks acting on any single qubit."""
        return int((_gf2.col_weights(self.hx) + _gf2.col_weights(self.hz)).max(initial=0))

    @staticmethod
    def from_code(name: str, code: CSSCode, n_frame: int | None = None) -> Phase:
        """Create a phase measuring the checks of a code.

        Args:
            name (str): name of the phase
            code (~.CSSCode): code whose checks are measured
            n_frame (int): Size of the qubit frame. Qubits beyond ``code.n`` are inactive.
                Defaults to ``code.n``.

        Returns:
            ~.Phase: the phase

        Raises:
            GadgetError: if the frame is smaller than the code
        """
        n = code.n if n_frame is None else n_frame
        if n < code.n:
            raise GadgetError(f"phase {name}: frame of {n} is smaller than the code's {code.n}")
        pad = n - code.n
        active = np.zeros(n, dtype=bool)
        active[: code.n] = True

        def widen(mat: np.ndarray) -> np.ndarray:
            if not mat.size:
                return np.zeros((0, n), dtype=np.uint8)
            return np.hstack([mat, np.zeros((mat.shape[0], pad), dtype=np.uint8)])

        return Phase(name=name, hx=widen(code.hx), hz=widen(code.hz), active=active)


# --------------------------------------------------------------------------------------
# Handles
# --------------------------------------------------------------------------------------


@dataclass(eq=False)
class Handle:
    """A single-use reference to the encoded qubits at one point in a gadget body.

    Each traced operation that acts on the qubits consumes the handle it is given and
    returns a new one. Using a consumed handle raises :class:`OwnershipError`, so a body
    cannot act twice on the same encoded state. Handles are created by the tracer and are
    not constructed directly.

    Args:
        value_id (int): identity of the handle within the traced program
        code (~.CSSCode): code of the encoded qubits
        phase (str): name of the phase currently being measured
        alive (bool): ``False`` once the handle has been consumed
        consumed_by (str or None): description of the operation that consumed it
    """

    value_id: int
    code: CSSCode
    phase: str
    alive: bool = True
    consumed_by: str | None = None

    def _check_live(self, op: str) -> None:
        if not self.alive:
            raise OwnershipError(
                f"{op}: handle %{self.value_id} was already consumed by "
                f"{self.consumed_by}. Encoded qubits cannot be used twice; pass on the "
                "handle returned by that operation instead."
            )

    def __repr__(self) -> str:
        state = "live" if self.alive else f"consumed by {self.consumed_by}"
        return f"Handle(%{self.value_id}, {self.code.name}, phase={self.phase}, {state})"


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordTerm:
    """One measurement outcome, addressed by record block, round and check.

    Args:
        block (str): name of the record block
        round (int): round within the block
        check (int): check within the round
    """

    block: str
    round: int
    check: int


@dataclass(frozen=True)
class RecordExpr:
    """A parity of measurement outcomes, optionally including entry-syndrome bits.

    Parities are combined with ``^``. Entry-syndrome bits stand for the check outcomes the
    encoded qubits carried when the gadget started (see :func:`entry_syndrome`), so a
    parity that spans the start of the gadget does not need to refer to anything outside
    it.

    Args:
        terms (frozenset[RecordTerm]): measurement outcomes in the parity
        entry (frozenset[int]): entry-syndrome bits in the parity

    **Example**

    >>> from pennylane import gadget
    >>> block = gadget.RecordBlock("m", 0, "merged", rounds=2, width=3, axes=("z",) * 3)
    >>> (block.at(1, 0) ^ gadget.entry_syndrome((2,))).describe()
    'm[r1,c0] ^ entry[2]'
    >>> bool(block.at(0, 1) ^ block.at(0, 1))
    False
    """

    terms: frozenset[RecordTerm] = frozenset()
    entry: frozenset[int] = frozenset()

    def __xor__(self, other: RecordExpr) -> RecordExpr:
        return RecordExpr(terms=self.terms ^ other.terms, entry=self.entry ^ other.entry)

    def __bool__(self) -> bool:
        return bool(self.terms or self.entry)

    def describe(self) -> str:
        """A readable form of the parity, sorted for stable output.

        Returns:
            str: the terms joined by ``" ^ "``, or ``"0"`` for the empty parity
        """
        parts = [
            f"{t.block}[r{t.round},c{t.check}]"
            for t in sorted(self.terms, key=lambda t: (t.block, t.round, t.check))
        ]
        parts += [f"entry[{i}]" for i in sorted(self.entry)]
        return " ^ ".join(parts) if parts else "0"


@dataclass(frozen=True)
class RecordBlock:
    """The measurement outcomes produced by one traced operation.

    A block holds ``rounds`` rounds of ``width`` outcomes. Its name identifies it within
    the gadget, and outcomes are addressed by position rather than by the Python variable
    they were assigned to, so they keep their identity through inlining and emission.
    Blocks are created by :func:`~pennylane.gadget.rounds` and
    :func:`~pennylane.gadget.detach`.

    Args:
        name (str): name, unique within the gadget
        op_index (int): position of the producing operation in the traced program
        phase (str): phase whose checks were measured
        rounds (int): number of rounds recorded
        width (int): number of outcomes per round
        axes (tuple[str]): Pauli axis of each outcome in a round

    **Example**

    >>> from pennylane import gadget
    >>> block = gadget.RecordBlock("m", 0, "merged", rounds=3, width=5, axes=("z",) * 5)
    >>> block.at(2, 4).describe()
    'm[r2,c4]'
    >>> block.product((0, 4)).describe()
    'm[r2,c0] ^ m[r2,c4]'
    """

    name: str
    op_index: int
    phase: str
    rounds: int
    width: int
    axes: tuple[str, ...]

    def at(self, round_index: int, check_index: int) -> RecordExpr:
        """The outcome of one check in one round.

        Args:
            round_index (int): round within the block
            check_index (int): check within the round

        Returns:
            ~.RecordExpr: the single-outcome parity

        Raises:
            GadgetError: if either index is out of range
        """
        if not 0 <= round_index < self.rounds:
            raise GadgetError(f"record {self.name}: round {round_index} out of range")
        if not 0 <= check_index < self.width:
            raise GadgetError(f"record {self.name}: check {check_index} out of range")
        return RecordExpr(terms=frozenset({RecordTerm(self.name, round_index, check_index)}))

    def round(self, round_index: int) -> tuple[RecordExpr, ...]:
        """Every outcome of one round, in check order."""
        return tuple(self.at(round_index, c) for c in range(self.width))

    def check(self, check_index: int) -> tuple[RecordExpr, ...]:
        """Every outcome of one check, in round order."""
        return tuple(self.at(r, check_index) for r in range(self.rounds))

    def product(self, indices: tuple[int, ...] | None = None) -> RecordExpr:
        """The parity of a set of checks in the final round.

        Args:
            indices (tuple[int]): Checks to include. Defaults to every check.

        Returns:
            ~.RecordExpr: the parity
        """
        idx = tuple(range(self.width)) if indices is None else indices
        out = RecordExpr()
        for c in idx:
            out = out ^ self.at(self.rounds - 1, c)
        return out


def entry_syndrome(indices: tuple[int, ...]) -> RecordExpr:
    """The parity of bits of the syndrome the encoded qubits carry at the start of a gadget.

    Bit ``i`` is the most recent outcome of check ``i`` of the entry phase before the
    gadget started. Detectors on the first round of a gadget are closed against these
    bits, and a caller that composes gadgets supplies them from its own records.

    Args:
        indices (tuple[int]): entry-phase checks in the parity

    Returns:
        ~.RecordExpr: the parity

    **Example**

    >>> from pennylane import gadget
    >>> gadget.entry_syndrome((0, 3)).describe()
    'entry[0] ^ entry[3]'
    """
    return RecordExpr(entry=frozenset(indices))


# --------------------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Op:
    """Base class for traced operations.

    Args:
        index (int): position of the operation in the traced program
        handle_in (int): value id of the consumed handle, or ``-1``
        handle_out (int): value id of the produced handle, or ``-1``
    """

    index: int
    handle_in: int
    handle_out: int


@dataclass(frozen=True)
class Deform(Op):
    """Switch to another phase over the same frame, optionally activating qubits.

    Args:
        to_phase (str): name of the phase entered
        init (tuple[tuple[int, str]]): activated qubits and the basis each is prepared in
    """

    to_phase: str = ""
    init: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class Rounds(Op):
    """Measure every check of the current phase for a fixed number of rounds.

    Args:
        phase (str): name of the phase measured
        count (int): number of rounds
        record (str): name of the record block produced
    """

    phase: str = ""
    count: int = 1
    record: str = ""


@dataclass(frozen=True)
class Detach(Op):
    """Switch to another phase, reading out the qubits it deactivates.

    Args:
        to_phase (str): name of the phase entered
        measure_out (tuple[tuple[int, str]]): deactivated qubits and their readout basis
        record (str): name of the record block holding the readouts
    """

    to_phase: str = ""
    measure_out: tuple[tuple[int, str], ...] = ()
    record: str = ""


@dataclass(frozen=True)
class Frame(Op):
    """Apply one row of the declared Pauli frame updates, conditioned on a parity.

    Args:
        expr (~.RecordExpr): parity the update is conditioned on
        update_index (int): row of the program's ``frame_update`` matrix
    """

    expr: RecordExpr = field(default_factory=RecordExpr)
    update_index: int = 0


@dataclass(frozen=True)
class Observe(Op):
    """Declare a parity as one of the gadget's measurement outcomes.

    Args:
        expr (~.RecordExpr): the parity as written in the gadget body
        observable_index (int or None): Outcome index. ``None`` marks an outcome of a
            called gadget that the enclosing body has not exposed; it is not an outcome of
            the enclosing gadget.
    """

    expr: RecordExpr = field(default_factory=RecordExpr)
    observable_index: int | None = 0


# --------------------------------------------------------------------------------------
# Declared logical behaviour
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """The logical operation a gadget is declared to perform.

    The declared action is checked against the traced body by
    :func:`~pennylane.gadget.verify`: for a measurement, each outcome parity must measure
    exactly the declared logical Pauli product. Create actions with :meth:`idle`,
    :meth:`measure` or :meth:`prepare`.

    Args:
        kind (str): ``"idle"``, ``"measure"`` or ``"prepare"``
        paulis (tuple[tuple[str, tuple[int]]]): For a measurement, each measured product as
            ``(axis, logical_qubits)``, in outcome order.
        name (str): display name

    **Example**

    >>> from pennylane import gadget
    >>> action = gadget.Action.measure(("z", (0, 1)), ("x", (1,)))
    >>> print(action)
    measure(Z_0_1, X_1)
    >>> action.n_outcomes
    2
    """

    kind: str
    paulis: tuple[tuple[str, tuple[int, ...]], ...] = ()
    name: str = ""

    @classmethod
    def idle(cls) -> Action:
        """Preserve the logical state without producing an outcome.

        Returns:
            ~.Action: the idle action
        """
        return cls(kind="idle", name="idle")

    @classmethod
    def measure(cls, *paulis: tuple[str, tuple[int, ...]]) -> Action:
        """Measure one or more logical Pauli products without destroying the encoding.

        Args:
            *paulis (tuple[str, Sequence[int]]): Each product as ``(axis, logical_qubits)``;
                for example ``("x", (0, 3))`` is logical X on logical qubits 0 and 3. The
                axis is ``"x"`` or ``"z"``. Outcomes are indexed in argument order.

        Returns:
            ~.Action: the measurement action

        Raises:
            GadgetError: if an axis is not ``"x"`` or ``"z"``
        """
        norm = tuple((axis.lower(), tuple(int(p) for p in qubits)) for axis, qubits in paulis)
        for axis, _ in norm:
            if axis not in ("x", "z"):
                raise GadgetError(f"Action.measure: axis must be 'x' or 'z', got {axis!r}")
        return cls(kind="measure", paulis=norm)

    @classmethod
    def prepare(cls, state: str = "zero") -> Action:
        """Prepare the logical qubits in a named state.

        Args:
            state (str): name of the prepared state

        Returns:
            ~.Action: the preparation action
        """
        return cls(kind="prepare", name=f"prepare({state})")

    @property
    def n_outcomes(self) -> int:
        """Number of measurement outcomes the action produces."""
        return len(self.paulis) if self.kind == "measure" else 0

    def __str__(self) -> str:
        if self.kind != "measure":
            return self.name or self.kind
        terms = ", ".join(
            axis.upper() + "".join(f"_{q}" for q in qubits) for axis, qubits in self.paulis
        )
        return f"measure({terms})"


# --------------------------------------------------------------------------------------
# The traced program
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GadgetProgram:
    """The traced form of one gadget.

    Programs are produced by :func:`~pennylane.gadget.define` and consumed by every later
    stage. They contain no Python callables, and :meth:`fingerprint` identifies the code,
    phases and operation sequence, so results derived from a program can be checked
    against it later.

    Args:
        name (str): gadget name
        code (~.CSSCode): code of the encoded qubits on entry
        action (~.Action): declared logical action
        n_frame (int): size of the shared qubit frame
        n_data (int): number of data qubits at the start of the frame
        phases (tuple[~.Phase]): every phase the gadget may measure
        ops (tuple[Op]): traced operations, in order
        records (tuple[~.RecordBlock]): record blocks, in the order they were produced
        frame_update (array[int]): Declared Pauli frame updates, one row per update over
            the data qubits. The first rows belong to the gadget's own outcomes; rows of
            called gadgets follow.
        inputs (tuple[int]): value ids of the input handles
        outputs (tuple[int]): value ids of the output handles
        claims (tuple[~.DistanceClaim]): distance claims made by the author
        notes (str): free-text description

    **Example**

    >>> from pennylane.gadget.library import rep_code_zz_merge
    >>> _, _, measure_zz = rep_code_zz_merge(d=3)
    >>> program = measure_zz.program
    >>> print(program.action)
    measure(Z_0_1)
    >>> [r.name for r in program.records]
    ['pre', 'merged', 'post']
    >>> program.total_rounds, program.max_syndrome_width
    (5, 5)
    """

    name: str
    code: CSSCode
    action: Action
    n_frame: int
    n_data: int
    phases: tuple[Phase, ...]
    ops: tuple[Op, ...]
    records: tuple[RecordBlock, ...]
    frame_update: np.ndarray
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    claims: tuple[DistanceClaim, ...] = ()
    notes: str = ""

    def phase(self, name: str) -> Phase:
        """Look up a phase by name.

        Raises:
            GadgetError: if there is no such phase
        """
        for p in self.phases:
            if p.name == name:
                return p
        raise GadgetError(f"{self.name}: no phase named {name!r}")

    def record(self, name: str) -> RecordBlock:
        """Look up a record block by name.

        Raises:
            GadgetError: if there is no such block
        """
        for r in self.records:
            if r.name == name:
                return r
        raise GadgetError(f"{self.name}: no record block named {name!r}")

    @property
    def n_aux(self) -> int:
        """Number of auxiliary qubits in the frame beyond the data qubits."""
        return self.n_frame - self.n_data

    @property
    def total_rounds(self) -> int:
        """Total number of measurement rounds."""
        return sum(op.count for op in self.ops if isinstance(op, Rounds))

    @property
    def spacetime_volume(self) -> int:
        """Sum over measurement rounds of the number of live qubits."""
        total = 0
        for op in self.ops:
            if isinstance(op, Rounds):
                total += op.count * self.phase(op.phase).n_active
        return total

    @property
    def max_syndrome_width(self) -> int:
        """Largest number of check outcomes produced in one round by any phase."""
        return max((p.syndrome_width for p in self.phases), default=0)

    @property
    def observables(self) -> tuple[Observe, ...]:
        """The operations declaring the gadget's own outcomes, in program order."""
        return tuple(
            op for op in self.ops if isinstance(op, Observe) and op.observable_index is not None
        )

    def fingerprint(self) -> str:
        """A content hash of the code, phases and operation sequence.

        Distance claims and notes are not included, so adding a claim does not invalidate
        detectors derived from the program.

        Returns:
            str: 16 hexadecimal characters
        """
        h = hashlib.sha256()
        h.update(self.code.fingerprint().encode())
        for p in self.phases:
            h.update(p.name.encode())
            h.update(np.ascontiguousarray(p.hx).tobytes())
            h.update(np.ascontiguousarray(p.hz).tobytes())
            h.update(np.ascontiguousarray(p.active).tobytes())
        for op in self.ops:
            h.update(repr(op).encode())
        return h.hexdigest()[:16]

    def with_claims(self, *claims: DistanceClaim) -> GadgetProgram:
        """A copy of the program carrying additional distance claims.

        Args:
            *claims (~.DistanceClaim): claims to add

        Returns:
            ~.GadgetProgram: the new program
        """
        return replace(self, claims=self.claims + claims)

    def summary(self) -> str:
        """A multi-line description of the program.

        Returns:
            str: the description
        """
        lines = [
            f"gadget {self.name}  [{self.fingerprint()}]",
            f"  action       : {self.action}",
            f"  code         : {self.code}",
            f"  frame        : {self.n_data} data + {self.n_aux} auxiliary = {self.n_frame}",
            "  phases       : "
            + ", ".join(f"{p.name}(k={p.k}, m={p.syndrome_width})" for p in self.phases),
            f"  rounds       : {self.total_rounds}   spacetime volume: {self.spacetime_volume}",
            f"  syndrome/rnd : {self.max_syndrome_width} bits (widest phase)",
            f"  check weight : {max((p.max_check_weight for p in self.phases), default=0)}"
            f"   qubit degree: {max((p.max_qubit_degree for p in self.phases), default=0)}",
            "  records      : "
            + ", ".join(f"{r.name}({r.rounds}x{r.width})" for r in self.records),
        ]
        for claim in self.claims:
            lines.append(f"  claim        : {claim}")
        if self.notes:
            lines.append(f"  notes        : {self.notes}")
        return "\n".join(lines)
