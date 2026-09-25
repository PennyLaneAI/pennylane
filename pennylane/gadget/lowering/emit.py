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
This module contains :func:`emit`, which writes a traced gadget as an xDSL module in the
``gadget`` dialect over Catalyst's ``!qecl.codeblock`` type.

The module contains one ``gadget.phase`` per phase, one ``func.func`` holding the traced
operations, and one ``gadget.detectors`` operation referring to the function. The function
carries ``gadget.action``, ``gadget.code`` and ``gadget.fingerprint`` attributes, so the
detectors can be checked against the gadget they were derived from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import builtin, func
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, ModuleOp, StringAttr, i64
from xdsl.ir import Block, Region
from xdsl.rewriter import InsertPoint

from ..detectors import DetectorLayout, derive_detectors
from ..ir import Deform, Detach, Frame, GadgetProgram, Observe, RecordExpr, RecordTerm, Rounds
from . import dialect as gd
from .catalyst_dialects import load_dialect_module, source_of

_qecl = load_dialect_module("qecl")
QECL_SOURCE = source_of(_qecl)
"""str: the file the ``qecl`` dialect definitions were loaded from"""


@dataclass
class Emission:
    """An emitted module, and the program and detector layout it was built from.

    Args:
        module (ModuleOp): the emitted module
        program (~.GadgetProgram): the traced gadget
        layout (~.DetectorLayout): the detectors attached to the module
        qecl_source (str): the file the ``qecl`` dialect definitions were loaded from
    """

    module: ModuleOp
    program: GadgetProgram
    layout: DetectorLayout
    qecl_source: str = QECL_SOURCE

    def verify(self) -> None:
        """Run the xDSL verifier over the module.

        Raises:
            VerifyException: if the module is invalid
        """
        self.module.verify()

    def text(self) -> str:
        """The module as IR text.

        Returns:
            str: the IR
        """
        return str(self.module)


def context() -> Context:
    """An xDSL context with every dialect used by emitted modules loaded.

    Returns:
        Context: the context
    """
    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(_qecl.QecLogical)
    ctx.load_dialect(gd.GadgetDialect)
    return ctx


def codeblock_type(program: GadgetProgram):
    """The ``!qecl.codeblock<k>`` type of a gadget's encoded qubits.

    Args:
        program (~.GadgetProgram): the traced gadget

    Returns:
        LogicalCodeblockType: the type
    """
    return _qecl.LogicalCodeblockType(program.code.k)


def emit(
    program: GadgetProgram,
    layout: DetectorLayout | None = None,
    verify: bool = True,
) -> Emission:
    """Write a traced gadget as an xDSL module.

    Each declared outcome is emitted with its completed parity from the detector layout,
    since that is the parity a decoder must report. Outcomes of called gadgets that the
    enclosing body did not declare are not emitted.

    Args:
        program (~.GadgetProgram): the traced gadget
        layout (~.DetectorLayout or None): detectors to attach, derived if not given
        verify (bool): whether to run the xDSL verifier on the result

    Returns:
        ~.Emission: the module

    Raises:
        NotImplementedError: if an outcome or frame-update parity spans more than one
            record block, which the single-operand ``gadget.observable`` and
            ``gadget.frame_update`` operations cannot express

    **Example**

    .. code-block:: python

        from pennylane.gadget.library import steane_memory
        from pennylane.gadget.lowering import emit

        _, _, memory = steane_memory(rounds=3)
        emission = emit(memory.program)

    >>> [op.name for op in emission.module.body.block.ops]
    ['gadget.phase', 'func.func', 'gadget.detectors']
    """
    layout = layout or derive_detectors(program)
    module = ModuleOp([])
    top = Builder(InsertPoint.at_end(module.body.block))

    for phase in program.phases:
        top.insert(
            gd.PhaseOp(
                sym_name=phase.name,
                hx=phase.hx,
                hz=phase.hz,
                active=phase.active.astype(np.uint8),
            )
        )

    cb = codeblock_type(program)
    block = Block(arg_types=[cb])
    body = Builder(InsertPoint.at_end(block))
    handle = block.args[0]

    record_values: dict[str, object] = {}
    for op in program.ops:
        if isinstance(op, Deform):
            new = gd.DeformOp(handle, op.to_phase, op.init)
            body.insert(new)
            handle = new.out_block
        elif isinstance(op, Rounds):
            rec = program.record(op.record)
            new = gd.RoundsOp(handle, op.phase, op.count, rec.width)
            body.insert(new)
            handle = new.out_block
            record_values[op.record] = new.records
        elif isinstance(op, Detach):
            new = gd.DetachOp(handle, op.to_phase, op.measure_out)
            body.insert(new)
            handle = new.out_block
            record_values[op.record] = new.records
        elif isinstance(op, Observe):
            if op.observable_index is None:
                continue
            expr = _completed(layout, op.observable_index, op.expr)
            value, terms = _single_block(program, expr, record_values, "observable")
            body.insert(
                gd.ObservableOp(value, op.observable_index, terms, tuple(sorted(expr.entry)))
            )
        elif isinstance(op, Frame):
            if not op.expr:
                continue
            value, terms = _single_block(program, op.expr, record_values, "frame update")
            new = gd.FrameUpdateOp(
                handle, value, op.update_index, terms, tuple(sorted(op.expr.entry))
            )
            body.insert(new)
            handle = new.out_block
        else:  # pragma: no cover - defensive
            raise NotImplementedError(f"no emission for {type(op).__name__}")

    body.insert(func.ReturnOp(handle))
    fn = func.FuncOp(program.name, ((cb,), (cb,)), Region([block]))
    fn.attributes["gadget.action"] = StringAttr(str(program.action))
    fn.attributes["gadget.code"] = StringAttr(program.code.name)
    fn.attributes["gadget.fingerprint"] = StringAttr(program.fingerprint())
    fn.attributes["gadget.frame_update"] = gd.bits_attr(program.frame_update)
    fn.attributes["gadget.claims"] = ArrayAttr([StringAttr(str(c)) for c in program.claims])
    fn.attributes["gadget.n_frame"] = IntegerAttr(program.n_frame, i64)
    top.insert(fn)

    det, labels = layout.detector_matrix(program)
    obs = _observable_matrix(program, layout, len(labels))
    index, _ = layout.record_index(program)
    undetermined = tuple(
        index[_term_from_label(name)]
        for name, _ in layout.undetermined
        if _term_from_label(name) in index
    )
    top.insert(
        gd.DetectorsOp(
            of=program.name,
            regime=layout.regime,
            detectors=det,
            observables=obs,
            entry_width=layout.entry_width,
            undetermined=undetermined,
        )
    )

    emission = Emission(module=module, program=program, layout=layout)
    if verify:
        emission.verify()
    return emission


def _single_block(
    program: GadgetProgram,
    expr: RecordExpr,
    record_values: dict[str, object],
    what: str,
):
    """Resolve a parity over one record block into its SSA value and index triples.

    The emitted operations take a single record operand, so a parity spanning several
    blocks is rejected rather than truncated.
    """
    blocks = {t.block for t in expr.terms}
    if len(blocks) > 1:
        raise NotImplementedError(
            f"{program.name}: the {what} parity spans record blocks {sorted(blocks)}; "
            "gadget.observable and gadget.frame_update take one record operand in this "
            "starting implementation. Widening them to variadic operands is a dialect "
            "change, not a frontend change."
        )
    if not blocks:
        raise NotImplementedError(f"{program.name}: the {what} parity has no record terms")
    name = blocks.pop()
    if name not in record_values:
        raise NotImplementedError(
            f"{program.name}: the {what} parity refers to record block {name!r}, which "
            "no emitted op produced"
        )
    terms = tuple(
        (0, t.round, t.check) for t in sorted(expr.terms, key=lambda t: (t.round, t.check))
    )
    return record_values[name], terms


def _completed(layout: DetectorLayout, index: int, fallback: RecordExpr) -> RecordExpr:
    """The completed parity of outcome ``index``, or the declared parity if there is none."""
    for obs in layout.observables:
        if obs.index == index:
            return obs.expr
    return fallback


def _observable_matrix(
    program: GadgetProgram, layout: DetectorLayout, n_records: int
) -> np.ndarray:
    index, _ = layout.record_index(program)
    mat = np.zeros((max(len(layout.observables), 1), n_records), dtype=np.uint8)
    for obs in layout.observables:
        for term in obs.expr.terms:
            mat[obs.index, index[term]] ^= 1
    return mat


def _term_from_label(label: str) -> RecordTerm:
    name, rest = label.rsplit("[", 1)
    rounds, checks = rest.rstrip("]").split(",")
    return RecordTerm(name, int(rounds[1:]), int(checks[1:]))


__all__ = ["Emission", "emit", "context", "codeblock_type", "QECL_SOURCE"]
