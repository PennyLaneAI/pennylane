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
This module contains the :func:`define` decorator and the operations used inside a gadget
body.

The stabilizer groups a gadget measures are declared up front as :class:`~.Phase` objects.
The body only schedules them: how many rounds are measured in which phase, when the phase
changes, which measurement parities are outcomes and where the declared Pauli frame updates
apply. A body cannot introduce a check that was not declared, so every later analysis works
on a fixed set of matrices.
"""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

from .codes import CSSCode, DistanceClaim
from .ir import (
    Action,
    Deform,
    Detach,
    Frame,
    GadgetError,
    GadgetProgram,
    Handle,
    Observe,
    Op,
    OwnershipError,
    Phase,
    RecordBlock,
    RecordExpr,
    Rounds,
)

_TRACER: ContextVar["_Tracer | None"] = ContextVar("gadget_tracer", default=None)


# --------------------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """A measurement outcome of a gadget, identified by its index and its parity.

    Outcomes are returned by :func:`observe`, and by calling a :class:`TracedGadget` inside
    another gadget's body. In the second case the outcome is not yet an outcome of the
    enclosing gadget, and its ``index`` is ``None`` until the body passes it to
    :func:`observe`.

    Args:
        index (int or None): outcome index in the enclosing gadget, or ``None``
        expr (~.RecordExpr): the parity as written
        op_index (int or None): position of the producing observe operation in the body
    """

    index: int | None
    expr: RecordExpr
    op_index: int | None = None

    def __xor__(self, other: "Outcome | RecordExpr") -> RecordExpr:
        return self.expr ^ (other.expr if isinstance(other, Outcome) else other)


# --------------------------------------------------------------------------------------
# The tracer
# --------------------------------------------------------------------------------------


class _Tracer:
    """Collects the operations of one gadget body while it is traced."""

    def __init__(self, name: str, code: CSSCode, phases: Sequence[Phase], frame_update: np.ndarray):
        self.name = name
        self.code = code
        self.phases = tuple(phases)
        if not self.phases:
            raise GadgetError(f"{name}: at least one phase must be declared")
        frames = {p.n_frame for p in self.phases}
        if len(frames) != 1:
            raise GadgetError(
                f"{name}: every phase must share one qubit frame, got sizes {sorted(frames)}"
            )
        self.n_frame = self.phases[0].n_frame
        self.ops: list[Op] = []
        self.records: list[RecordBlock] = []
        self._next_value = 0
        self._record_names: set[str] = set()
        self.calls: dict[str, int] = {}
        # The body's own rows, one per declared outcome, then rows carried by inlined calls.
        self.frame_update = frame_update

    def new_handle(self, phase: str) -> Handle:
        """Create a live handle in ``phase``."""
        handle = Handle(value_id=self._next_value, code=self.code, phase=phase)
        self._next_value += 1
        return handle

    def consume(self, handle: Handle, op_name: str) -> int:
        """Mark ``handle`` as consumed and return its value id."""
        if not isinstance(handle, Handle):
            raise GadgetError(f"{op_name}: expected a Handle, got {type(handle).__name__}")
        handle._check_live(op_name)
        handle.alive = False
        handle.consumed_by = f"{op_name} (op {len(self.ops)})"
        return handle.value_id

    def phase_named(self, name: str) -> Phase:
        """Look up a declared phase by name."""
        for p in self.phases:
            if p.name == name:
                return p
        raise GadgetError(
            f"{self.name}: no phase named {name!r}; declared phases are "
            + ", ".join(p.name for p in self.phases)
        )

    def add_rounds_block(self, name: str, phase: Phase, count: int) -> RecordBlock:
        """Register the record block of ``count`` rounds of ``phase``."""
        return self.add_record_block(
            name, phase.name, count, phase.syndrome_width, phase.check_axes
        )

    def add_record_block(
        self, name: str, phase: str, count: int, width: int, axes: tuple[str, ...]
    ) -> RecordBlock:
        """Register a record block under a name that is unique within the gadget."""
        if name in self._record_names:
            raise GadgetError(f"{self.name}: duplicate record block name {name!r}")
        self._record_names.add(name)
        block = RecordBlock(
            name=name, op_index=len(self.ops), phase=phase, rounds=count, width=width, axes=axes
        )
        self.records.append(block)
        return block


def _tracer(op_name: str) -> _Tracer:
    tr = _TRACER.get()
    if tr is None:
        raise GadgetError(f"{op_name} can only be called inside a @gadget.define body")
    return tr


# --------------------------------------------------------------------------------------
# Traced operations
# --------------------------------------------------------------------------------------


def rounds(handle: Handle, count: int, *, record: str) -> tuple[Handle, RecordBlock]:
    """Measure every check of the current phase for a fixed number of rounds.

    Args:
        handle (~.Handle): handle on the encoded qubits, consumed by this operation
        count (int): Number of rounds. This must be a Python ``int``, because the number of
            rounds determines the detector structure.
        record (str): name of the record block holding the outcomes

    Returns:
        tuple[~.Handle, ~.RecordBlock]: the new handle, and the ``count`` by
        ``syndrome_width`` block of outcomes

    Raises:
        GadgetError: if ``count`` is not a positive ``int``, or ``record`` is already used

    **Example**

    >>> from pennylane.ftqc import gadget
    >>> from pennylane.ftqc.gadget.library import steane_code
    >>> code = steane_code()
    >>> @gadget.define(
    ...     action=gadget.Action.idle(), code=code, phases=(gadget.Phase.from_code("s", code),)
    ... )
    ... def memory(handle):
    ...     handle, outcomes = gadget.rounds(handle, 3, record="mem")
    ...     return handle
    >>> block = memory.program.record("mem")
    >>> block.rounds, block.width
    (3, 6)
    """
    tr = _tracer("rounds")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise GadgetError(
            f"rounds: count must be a positive Python int (got {count!r}). Round counts "
            "fix the detector structure and cannot be runtime values."
        )
    phase = tr.phase_named(handle.phase)
    block = tr.add_rounds_block(record, phase, count)
    src = tr.consume(handle, "rounds")
    out = tr.new_handle(handle.phase)
    tr.ops.append(
        Rounds(
            index=len(tr.ops),
            handle_in=src,
            handle_out=out.value_id,
            phase=phase.name,
            count=count,
            record=record,
        )
    )
    return out, block


def deform(handle: Handle, *, to: str, init: dict[int, str] | None = None) -> Handle:
    """Switch to another declared phase over the same qubit frame.

    Qubits that are inactive in the current phase and active in the new one must be given
    the basis they are prepared in, so that the first measurement of the checks that act on
    them has a known outcome.

    Args:
        handle (~.Handle): handle on the encoded qubits, consumed by this operation
        to (str): name of the phase to enter
        init (dict[int, str]): each activated qubit and its preparation basis, ``"x"`` or
            ``"z"``

    Returns:
        ~.Handle: the new handle

    Raises:
        GadgetError: if the phase is not declared, an activated qubit has no basis, or a
            qubit given a basis is not active in the new phase

    **Example**

    >>> from pennylane.ftqc import gadget
    >>> from pennylane.ftqc.gadget.library import rep_code_zz_merge
    >>> code, phases, _ = rep_code_zz_merge(d=3)
    >>> @gadget.define(action=gadget.Action.idle(), code=code, phases=phases)
    ... def merge_and_split(handle):
    ...     handle = gadget.deform(handle, to="merged")
    ...     handle, _ = gadget.rounds(handle, 3, record="merged")
    ...     return gadget.deform(handle, to="base")
    >>> [type(op).__name__ for op in merge_and_split.program.ops]
    ['Deform', 'Rounds', 'Deform']
    """
    tr = _tracer("deform")
    target = tr.phase_named(to)
    src_phase = tr.phase_named(handle.phase)
    newly = np.nonzero(target.active & ~src_phase.active)[0]
    declared = {} if init is None else {int(q): b.lower() for q, b in init.items()}
    for q in declared:
        if not target.active[q]:
            raise GadgetError(
                f"deform to {to}: qubit {q} is initialized but is not active in that phase"
            )
    missing = sorted(int(q) for q in newly if int(q) not in declared)
    if missing:
        raise GadgetError(
            f"deform to {to}: qubits {missing} become active but no init basis was given. "
            "An auxiliary qubit with no stated initial basis has no deterministic "
            "first-round detector."
        )
    src = tr.consume(handle, "deform")
    out = tr.new_handle(to)
    tr.ops.append(
        Deform(
            index=len(tr.ops),
            handle_in=src,
            handle_out=out.value_id,
            to_phase=to,
            init=tuple(sorted(declared.items())),
        )
    )
    return out


def detach(
    handle: Handle,
    *,
    to: str,
    measure_out: dict[int, str] | None = None,
    record: str,
) -> tuple[Handle, RecordBlock]:
    """Switch to another declared phase, reading out the qubits it deactivates.

    Args:
        handle (~.Handle): handle on the encoded qubits, consumed by this operation
        to (str): name of the phase to enter
        measure_out (dict[int, str]): Each qubit to read out and its readout basis, ``"x"``
            or ``"z"``. Every qubit that is active now and inactive in the new phase must
            be included.
        record (str): name of the record block holding the readouts

    Returns:
        tuple[~.Handle, ~.RecordBlock]: the new handle, and a one-round block with one
        readout per qubit in ``measure_out``, in qubit order

    Raises:
        GadgetError: if the phase is not declared, or a deactivated qubit has no basis

    **Example**

    An auxiliary qubit 1 is activated with a Z check, then read out:

    .. code-block:: python

        import numpy as np
        from pennylane.ftqc import gadget
        from pennylane.ftqc.gadget.library import repetition_code

        code = repetition_code(1)
        no_x = np.zeros((0, 2), dtype=np.uint8)
        base = gadget.Phase("base", no_x, np.zeros((0, 2), np.uint8), np.array([True, False]))
        wide = gadget.Phase("wide", no_x, np.array([[0, 1]], np.uint8), np.array([True, True]))

        @gadget.define(action=gadget.Action.idle(), code=code, phases=(base, wide))
        def borrow(handle):
            handle = gadget.deform(handle, to="wide", init={1: "z"})
            handle, _ = gadget.rounds(handle, 1, record="aux")
            handle, _ = gadget.detach(handle, to="base", measure_out={1: "z"}, record="out")
            return handle

    >>> borrow.program.record("out").width
    1
    """
    tr = _tracer("detach")
    target = tr.phase_named(to)
    src_phase = tr.phase_named(handle.phase)
    leaving = [int(q) for q in np.nonzero(src_phase.active & ~target.active)[0]]
    declared = {} if measure_out is None else {int(q): b.lower() for q, b in measure_out.items()}
    missing = sorted(q for q in leaving if q not in declared)
    if missing:
        raise GadgetError(
            f"detach to {to}: qubits {missing} leave the frame but no readout basis was "
            "given. Their outcomes carry the gadget result and cannot be dropped."
        )
    ordered = tuple(sorted(declared.items()))
    block = tr.add_record_block(
        record,
        src_phase.name,
        count=1,
        width=len(ordered),
        axes=tuple(basis for _, basis in ordered),
    )
    src = tr.consume(handle, "detach")
    out = tr.new_handle(to)
    tr.ops.append(
        Detach(
            index=len(tr.ops),
            handle_in=src,
            handle_out=out.value_id,
            to_phase=to,
            measure_out=ordered,
            record=record,
        )
    )
    return out, block


def observe(expr: RecordExpr | Outcome, *, index: int) -> Outcome:
    """Declare a measurement parity as outcome ``index`` of the gadget.

    The parity does not have to be the product of checks that measures the declared
    logical operator exactly: :func:`~.derive_detectors` adds the outcomes of checks whose
    values are already known. It is rejected if no such completion exists.

    Given an outcome returned by a gadget called in this body, the outcome is declared at
    the point where that gadget produced it, so it is treated exactly as in the called
    gadget. Any other parity, including a parity of several outcomes, is declared at this
    point in the body.

    Args:
        expr (~.RecordExpr or ~.Outcome): the parity, or an outcome of a called gadget
        index (int): outcome index in the gadget's declared action

    Returns:
        ~.Outcome: the declared outcome

    Raises:
        GadgetError: if the parity is empty, or an outcome of a called gadget has already
            been declared or does not belong to this body

    **Example**

    >>> from pennylane.ftqc import gadget
    >>> from pennylane.ftqc.gadget.library import rep_code_zz_merge
    >>> code, phases, _ = rep_code_zz_merge(d=3)
    >>> @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases)
    ... def measure_zz(handle):
    ...     handle, _ = gadget.rounds(handle, 1, record="pre")
    ...     handle = gadget.deform(handle, to="merged")
    ...     handle, merged = gadget.rounds(handle, 3, record="merged")
    ...     return handle, gadget.observe(merged.product((4,)), index=0)
    >>> measure_zz.program.observables[0].expr.describe()
    'merged[r2,c4]'
    """
    tr = _tracer("observe")
    if isinstance(expr, Outcome) and expr.op_index is not None:
        return _expose(tr, expr, index)
    e = expr.expr if isinstance(expr, Outcome) else expr
    if not isinstance(e, RecordExpr) or not e:
        raise GadgetError("observe: expected a non-empty record parity")
    tr.ops.append(
        Observe(index=len(tr.ops), handle_in=-1, handle_out=-1, expr=e, observable_index=index)
    )
    return Outcome(index=index, expr=e, op_index=len(tr.ops) - 1)


def _expose(tr: _Tracer, outcome: Outcome, index: int) -> Outcome:
    pos = outcome.op_index
    op = tr.ops[pos] if 0 <= pos < len(tr.ops) else None
    if not isinstance(op, Observe) or op.expr != outcome.expr:
        raise GadgetError("observe: the outcome was not produced in this body")
    if op.observable_index is not None:
        raise GadgetError(
            f"observe: the outcome is already exposed as outcome {op.observable_index}"
        )
    tr.ops[pos] = replace(op, observable_index=index)
    return Outcome(index=index, expr=op.expr, op_index=pos)


def frame(handle: Handle, expr: RecordExpr | Outcome, *, update_index: int = 0) -> Handle:
    """Apply a declared Pauli frame update, conditioned on a measurement parity.

    The Pauli operators themselves are declared with the ``frame_update`` argument of
    :func:`define`. The body only chooses which declared row is applied and on which
    parity it is conditioned.

    Args:
        handle (~.Handle): handle on the encoded qubits, consumed by this operation
        expr (~.RecordExpr or ~.Outcome): the parity the update is conditioned on
        update_index (int): row of the declared ``frame_update`` matrix

    Returns:
        ~.Handle: the new handle

    **Example**

    >>> from pennylane.ftqc import gadget
    >>> from pennylane.ftqc.gadget.library import rep_code_zz_merge
    >>> _, _, measure_zz = rep_code_zz_merge(d=3)
    >>> [op.update_index for op in measure_zz.program.ops if type(op).__name__ == "Frame"]
    [0]
    """
    tr = _tracer("frame")
    e = expr.expr if isinstance(expr, Outcome) else expr
    src = tr.consume(handle, "frame")
    out = tr.new_handle(handle.phase)
    tr.ops.append(
        Frame(
            index=len(tr.ops),
            handle_in=src,
            handle_out=out.value_id,
            expr=e,
            update_index=update_index,
        )
    )
    return out


# --------------------------------------------------------------------------------------
# The decorator
# --------------------------------------------------------------------------------------


@dataclass
class TracedGadget:
    """A gadget body traced by :func:`define`.

    Calling a traced gadget inside another gadget's body inlines its operations there.
    Record block names are prefixed with ``name#n``, where ``n`` counts the calls to that
    gadget in the enclosing body, so two calls never share records.

    Args:
        program (~.GadgetProgram): the traced program
        fn (Callable): the decorated function

    **Example**

    >>> from pennylane.ftqc.gadget.library import steane_memory
    >>> _, _, memory = steane_memory(rounds=3)
    >>> memory
    <gadget steane_memory action=idle>
    >>> [r.name for r in memory.records]
    ['mem']
    """

    program: GadgetProgram
    fn: Callable

    @property
    def name(self) -> str:
        """Name of the gadget."""
        return self.program.name

    @property
    def records(self) -> tuple[RecordBlock, ...]:
        """Record blocks produced by the traced body."""
        return self.program.records

    def __call__(self, handle: Handle) -> tuple[Handle, tuple[Outcome, ...]]:
        """Inline this gadget into the body being traced.

        Args:
            handle (~.Handle): handle on the encoded qubits, consumed by the call

        Returns:
            tuple[~.Handle, tuple[~.Outcome]]: The new handle, and this gadget's outcomes in
            its declared order. They are not outcomes of the enclosing gadget until passed
            to :func:`observe`.
        """
        return _inline(self, handle)

    def __repr__(self) -> str:
        return f"<gadget {self.name} action={self.program.action}>"


def define(
    *,
    action: Action,
    code: CSSCode,
    phases: Sequence[Phase],
    frame_update=None,
    entry_phase: str | None = None,
    n_data: int | None = None,
    claims: Sequence[DistanceClaim] = (),
    notes: str = "",
):
    """Decorator that traces a Python function into a gadget.

    The decorated function receives a :class:`~.Handle` on the encoded qubits and must
    return the final handle, optionally followed by the declared outcomes. It is traced
    once, when it is decorated, so ownership and phase errors are raised immediately.

    Args:
        action (~.Action): the logical operation the gadget performs
        code (~.CSSCode): code of the encoded qubits on entry
        phases (Sequence[~.Phase]): every phase the body may enter, on one shared frame
        frame_update (array_like): Pauli frame updates over the data qubits, one row per
            declared outcome. Defaults to no update. Rows declared by gadgets called in the
            body are appended to the traced program's matrix.
        entry_phase (str): Phase of the encoded qubits on entry. Defaults to the first
            declared phase.
        n_data (int): Number of data qubits at the start of the frame. Defaults to
            ``code.n``.
        claims (Sequence[~.DistanceClaim]): distance claims to record in the program
        notes (str): Free-text description. Defaults to the first line of the function's
            docstring.

    Returns:
        Callable[[Callable], ~.TracedGadget]: the decorator

    Raises:
        GadgetError: if the phases, the frame updates, the declared outcomes or the body
            are inconsistent
        OwnershipError: if the body uses a consumed handle, or does not pass a handle on

    .. seealso:: :func:`~.verify`, :func:`~.derive_detectors`

    **Example**

    A memory gadget that measures the Steane code's checks for three rounds:

    >>> from pennylane.ftqc import gadget
    >>> from pennylane.ftqc.gadget.library import steane_code
    >>> code = steane_code()
    >>> @gadget.define(
    ...     action=gadget.Action.idle(), code=code, phases=(gadget.Phase.from_code("s", code),)
    ... )
    ... def memory(handle):
    ...     \"\"\"Hold a Steane block.\"\"\"
    ...     handle, _ = gadget.rounds(handle, 3, record="mem")
    ...     return handle
    >>> memory.program.total_rounds, memory.program.notes
    (3, 'Hold a Steane block.')

    .. details::
        :title: Usage Details

        **Composition**

        A traced gadget can be called inside another gadget's body. Its outcomes are not
        outcomes of the caller until they are passed to :func:`observe`, so the caller
        chooses which outcomes to declare and in which order. Frame updates declared by
        the called gadget are carried into the caller's ``frame_update`` matrix below its
        own rows.

        .. code-block:: python

            from pennylane.ftqc import gadget
            from pennylane.ftqc.gadget.library import rep_code_zz_merge

            code, phases, measure_zz = rep_code_zz_merge(d=3)

            @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases)
            def keep_second(handle):
                handle, _ = measure_zz(handle)
                handle, (second,) = measure_zz(handle)
                return handle, gadget.observe(second, index=0)

        >>> [r.name for r in keep_second.records][:3]
        ['measure_zz#0/pre', 'measure_zz#0/merged', 'measure_zz#0/post']
        >>> keep_second.program.frame_update.shape
        (3, 6)
    """

    def decorator(fn: Callable) -> TracedGadget:
        name = fn.__name__
        for _, qubits in action.paulis:
            bad = [q for q in qubits if not 0 <= q < code.k]
            if bad:
                raise GadgetError(
                    f"{name}: logical qubit(s) {bad} out of range for code {code.name} "
                    f"with k={code.k}"
                )
        n_d = code.n if n_data is None else int(n_data)
        t = action.n_outcomes
        fu = (
            np.zeros((t, n_d), dtype=np.uint8)
            if frame_update is None
            else np.asarray(frame_update, dtype=np.uint8).reshape(-1, n_d)
        )
        if fu.shape[0] != t:
            raise GadgetError(f"{name}: frame_update has {fu.shape[0]} rows for {t} outcome(s)")

        tr = _Tracer(name, code, phases, fu)
        start = entry_phase or tr.phases[0].name
        tr.phase_named(start)
        handle = tr.new_handle(start)
        inputs = (handle.value_id,)

        token = _TRACER.set(tr)
        try:
            result = fn(handle)
        finally:
            _TRACER.reset(token)

        out, _ = _unpack_result(name, result)
        if out.alive is False:
            raise OwnershipError(
                f"{name}: the returned handle %{out.value_id} was already consumed"
            )
        produced = [op.handle_out for op in tr.ops if op.handle_out >= 0]
        consumed = {op.handle_in for op in tr.ops if op.handle_in >= 0}
        dangling = sorted(set(produced) - consumed - {out.value_id})
        if dangling:
            raise OwnershipError(
                f"{name}: handle value(s) {dangling} are produced but never consumed or "
                "returned. Every handle must be passed on to the end of the body."
            )

        doc = (inspect.getdoc(fn) or "").strip().splitlines()
        program = GadgetProgram(
            name=name,
            code=code,
            action=action,
            n_frame=tr.n_frame,
            n_data=n_d,
            phases=tr.phases,
            ops=tuple(tr.ops),
            records=tuple(tr.records),
            frame_update=tr.frame_update,
            inputs=inputs,
            outputs=(out.value_id,),
            claims=tuple(claims),
            notes=notes or (doc[0] if doc else ""),
        )
        _check_outcome_count(program)
        return TracedGadget(program=program, fn=fn)

    return decorator


def _unpack_result(name: str, result) -> tuple[Handle, tuple[Outcome, ...]]:
    if isinstance(result, Handle):
        return result, ()
    if isinstance(result, tuple) and result and isinstance(result[0], Handle):
        outs = []
        for item in result[1:]:
            if isinstance(item, Outcome):
                outs.append(item)
            elif isinstance(item, (tuple, list)):
                outs.extend(o for o in item if isinstance(o, Outcome))
        return result[0], tuple(outs)
    raise GadgetError(
        f"{name}: a gadget body must return the handle, optionally followed by outcomes; "
        f"got {type(result).__name__}"
    )


def _check_outcome_count(program: GadgetProgram) -> None:
    declared = program.action.n_outcomes
    observed = len(program.observables)
    if declared != observed:
        raise GadgetError(
            f"{program.name}: declared action {program.action} has {declared} outcome(s) "
            f"but the body exposes {observed} via observe()"
        )
    indices = sorted(op.observable_index for op in program.observables)
    if indices != list(range(declared)):
        raise GadgetError(
            f"{program.name}: observable indices must be exactly 0..{declared - 1}, got {indices}"
        )


# --------------------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------------------


def _qualify(expr: RecordExpr, prefix: str) -> RecordExpr:
    return RecordExpr(
        terms=frozenset(replace(t, block=f"{prefix}/{t.block}") for t in expr.terms),
        entry=expr.entry,
    )


def _inline(traced: TracedGadget, handle: Handle) -> tuple[Handle, tuple[Outcome, ...]]:
    """Copy a traced gadget's operations into the body being traced."""
    tr = _tracer(f"call {traced.name}")
    sub = traced.program
    if sub.code.fingerprint() != tr.code.fingerprint():
        raise GadgetError(
            f"call {traced.name}: code mismatch, gadget expects {sub.code.name} "
            f"but the enclosing gadget uses {tr.code.name}"
        )
    entry = sub.phases[0].name if not sub.ops else _entry_phase(sub)
    if handle.phase != entry:
        raise GadgetError(
            f"call {traced.name}: gadget expects a handle in phase {entry!r}, "
            f"got {handle.phase!r}"
        )
    for p in sub.phases:
        try:
            existing = tr.phase_named(p.name)
        except GadgetError:
            tr.phases = tr.phases + (p,)
            continue
        if not (
            np.array_equal(existing.hx, p.hx)
            and np.array_equal(existing.hz, p.hz)
            and np.array_equal(existing.active, p.active)
        ):
            raise GadgetError(
                f"call {traced.name}: phase name {p.name!r} already refers to different checks"
            )

    n_calls = tr.calls.get(traced.name, 0)
    tr.calls[traced.name] = n_calls + 1
    prefix = f"{traced.name}#{n_calls}"
    row_base = _carry_frame_update(tr, traced)

    value_map: dict[int, int] = {sub.inputs[0]: tr.consume(handle, f"call {traced.name}")}
    exposed: dict[int, Outcome] = {}
    last: Handle = handle

    for op in sub.ops:
        if isinstance(op, Observe):
            _inline_observe(tr, op, prefix, exposed)
            continue

        out = tr.new_handle(_phase_after(op, last.phase))
        value_map[op.handle_out] = out.value_id
        common = {
            "index": len(tr.ops),
            "handle_in": value_map[op.handle_in],
            "handle_out": out.value_id,
        }
        if isinstance(op, Rounds):
            block_name = f"{prefix}/{op.record}"
            tr.add_rounds_block(block_name, tr.phase_named(op.phase), op.count)
            new = Rounds(**common, phase=op.phase, count=op.count, record=block_name)
        elif isinstance(op, Deform):
            new = Deform(**common, to_phase=op.to_phase, init=op.init)
        elif isinstance(op, Detach):
            block_name = f"{prefix}/{op.record}"
            tr.add_record_block(
                block_name,
                last.phase,
                count=1,
                width=len(op.measure_out),
                axes=tuple(basis for _, basis in op.measure_out),
            )
            new = Detach(
                **common, to_phase=op.to_phase, measure_out=op.measure_out, record=block_name
            )
        elif isinstance(op, Frame):
            new = Frame(
                **common,
                expr=_qualify(op.expr, prefix),
                update_index=row_base + op.update_index,
            )
        else:  # pragma: no cover - defensive
            raise GadgetError(f"call {traced.name}: cannot inline op {type(op).__name__}")
        tr.ops.append(new)
        last = out

    return last, tuple(exposed[i] for i in sorted(exposed))


def _carry_frame_update(tr: _Tracer, traced: TracedGadget) -> int:
    """Append the called gadget's declared frame updates, returning the row they start at.

    The rows are carried rather than re-declared by the caller, so the caller cannot drop
    or mistype a called gadget's byproduct correction.
    """
    rows = traced.program.frame_update
    if rows.shape[1] != tr.frame_update.shape[1]:
        raise GadgetError(
            f"call {traced.name}: frame updates are over {rows.shape[1]} data qubits but the "
            f"enclosing gadget uses {tr.frame_update.shape[1]}"
        )
    base = tr.frame_update.shape[0]
    tr.frame_update = np.vstack([tr.frame_update, rows])
    return base


def _inline_observe(tr: _Tracer, op: Observe, prefix: str, exposed: dict[int, Outcome]) -> None:
    """Inline an outcome as undeclared; the caller declares it with :func:`observe`."""
    expr = _qualify(op.expr, prefix)
    tr.ops.append(
        Observe(index=len(tr.ops), handle_in=-1, handle_out=-1, expr=expr, observable_index=None)
    )
    if op.observable_index is not None:
        exposed[op.observable_index] = Outcome(index=None, expr=expr, op_index=len(tr.ops) - 1)


def _entry_phase(program: GadgetProgram) -> str:
    for op in program.ops:
        if isinstance(op, Rounds):
            return op.phase
        if isinstance(op, (Deform, Detach)):
            return program.phases[0].name
    return program.phases[0].name


def _phase_after(op: Op, current: str) -> str:
    if isinstance(op, (Deform, Detach)):
        return op.to_phase
    return current


def unroll(count: int, handle: Handle, body: Callable[[Handle], Handle]) -> Handle:
    """Apply a function to a handle a fixed number of times.

    The loop is unrolled while tracing, so each iteration produces its own operations and
    record blocks, and ``body`` must use a different record name in each iteration.

    Args:
        count (int): number of iterations
        handle (~.Handle): handle on the encoded qubits
        body (Callable[[~.Handle], ~.Handle]): function applied in each iteration

    Returns:
        ~.Handle: the handle returned by the last iteration

    Raises:
        GadgetError: if ``count`` is not a positive ``int``

    **Example**

    >>> from pennylane.ftqc import gadget
    >>> from pennylane.ftqc.gadget.library import steane_memory
    >>> code, phases, memory = steane_memory(rounds=2)
    >>> @gadget.define(action=gadget.Action.idle(), code=code, phases=phases)
    ... def hold(handle):
    ...     return gadget.unroll(3, handle, lambda h: memory(h)[0])
    >>> [r.name for r in hold.records]
    ['steane_memory#0/mem', 'steane_memory#1/mem', 'steane_memory#2/mem']
    """
    if not isinstance(count, int) or count < 1:
        raise GadgetError("unroll: count must be a positive Python int")
    for _ in range(count):
        handle = body(handle)
    return handle
