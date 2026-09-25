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
This module contains :func:`lower_gadget_to_qecl`, which lowers the subset of the
``gadget`` dialect that ``qecl`` can express.

Only single-phase gadgets on ``k=1`` codeblocks whose records are not used by later
operations are lowered: each round becomes a ``qecl.qec`` cycle. Anything else raises
:class:`LoweringGap` naming the missing capability and where the limit is found. The pass
never approximates an operation it cannot express; lowering a deformation to plain QEC
cycles, for instance, would never measure the merged checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp

from ..support import CATALYST_EVIDENCE
from . import dialect as gd
from .catalyst_dialects import load_dialect_module

_qecl = load_dialect_module("qecl")


class LoweringGap(NotImplementedError):
    """Raised when a gadget needs a capability that ``qecl`` does not have.

    Args:
        capability (str): the missing capability
        op (str): the operation or function that needs it
        evidence (str): where the limit is found
        workaround (str): what would have to change
    """

    def __init__(self, capability: str, op: str, evidence: str, workaround: str):
        self.capability = capability
        self.op = op
        self.evidence = evidence
        self.workaround = workaround
        super().__init__(
            f"cannot lower {op} to qecl: {capability}\n"
            f"  evidence  : {evidence}\n"
            f"  workaround: {workaround}"
        )


@dataclass
class LoweringResult:
    """The result of :func:`lower_gadget_to_qecl`.

    Args:
        module (ModuleOp): the rewritten module
        qec_cycles (int): number of ``qecl.qec`` operations inserted
        measurements (int): number of ``qecl.measure`` operations inserted
        notes (list[str]): additional remarks
    """

    module: ModuleOp
    qec_cycles: int = 0
    measurements: int = 0
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        """The rewritten module as IR text.

        Returns:
            str: the IR
        """
        return str(self.module)

    def summary(self) -> str:
        """A one-line description of what was lowered.

        Returns:
            str: the description
        """
        return f"lowered to qecl: {self.qec_cycles} qecl.qec, {self.measurements} qecl.measure" + (
            "; " + "; ".join(self.notes) if self.notes else ""
        )


def lower_gadget_to_qecl(module: ModuleOp) -> LoweringResult:
    """Rewrite ``gadget`` operations into ``qecl`` operations, in place.

    Functions without a ``gadget.action`` attribute are left unchanged.

    Args:
        module (ModuleOp): a module produced by :func:`~.lowering.emit`

    Returns:
        ~.LoweringResult: what was lowered

    Raises:
        LoweringGap: if a gadget uses a codeblock with ``k > 1``, a deformation or detach,
            more than one phase, records used by later operations, or a frame update

    **Example**

    .. code-block:: python

        from pennylane.gadget.library import steane_memory
        from pennylane.gadget.lowering import emit, lower_gadget_to_qecl

        _, _, memory = steane_memory(rounds=3)
        result = lower_gadget_to_qecl(emit(memory.program).module)

    >>> print(result.summary())
    lowered to qecl: 3 qecl.qec, 0 qecl.measure
    """
    result = LoweringResult(module=module)

    for fn in [op for op in module.body.block.ops if isinstance(op, func.FuncOp)]:
        if "gadget.action" not in fn.attributes:
            continue
        _check_k(fn)
        ops = list(fn.body.block.ops)
        phases = {op.phase.string_value() for op in ops if isinstance(op, gd.RoundsOp)}
        for op in ops:
            if isinstance(op, gd.DeformOp):
                raise LoweringGap(
                    capability="an operation that changes the measured stabilizer group",
                    op=f"gadget.deform to @{op.to_phase.string_value()}",
                    evidence=CATALYST_EVIDENCE["deformation"],
                    workaround="add a qecl operation carrying the target phase's check "
                    "matrices, or expand the deformation into qecp before qecl loses the "
                    "stabilizer information",
                )
            if isinstance(op, gd.DetachOp):
                raise LoweringGap(
                    capability="mid-circuit readout of a subset of a codeblock's qubits",
                    op=f"gadget.detach to @{op.to_phase.string_value()}",
                    evidence=CATALYST_EVIDENCE["deformation"],
                    workaround="qecl.measure addresses logical qubits, not physical ones; "
                    "detach has to be expanded at the qecp level where physical qubits exist",
                )
        if len(phases) > 1:
            raise LoweringGap(
                capability="more than one stabilizer group per program",
                op=f"func @{fn.sym_name.data}",
                evidence=CATALYST_EVIDENCE["deformation"],
                workaround="split the gadget so each lowered function measures one phase",
            )

        for op in ops:
            if isinstance(op, gd.RoundsOp):
                _lower_rounds(op, result)
            elif isinstance(op, gd.ObservableOp):
                _lower_observable(fn, op, result)
            elif isinstance(op, gd.FrameUpdateOp):
                raise LoweringGap(
                    capability="classically conditioned Pauli frame update",
                    op="gadget.frame_update",
                    evidence="the PauliFrame dialect exists in Catalyst but qecl has no op "
                    "that consumes a measurement record to condition a frame update",
                    workaround="lower the frame update into the PauliFrame dialect rather "
                    "than qecl, and keep the record identity attached to the producing op",
                )
    return result


def _check_k(fn: func.FuncOp) -> None:
    arg_type = fn.body.block.args[0].type
    k = arg_type.k.value.data if hasattr(arg_type, "k") else None
    if k is not None and k > 1:
        raise LoweringGap(
            capability=f"codeblocks with k > 1 (this gadget uses k={k})",
            op=f"func @{fn.sym_name.data}",
            evidence=CATALYST_EVIDENCE["max_k"],
            workaround="lower a k=1 code, or teach convert-qecl-to-qecp to index "
            "logical qubits within a codeblock",
        )


def _lower_rounds(op: gd.RoundsOp, result: LoweringResult) -> None:
    """Replace a round window by one ``qecl.qec`` per round."""
    if op.records.uses:
        raise LoweringGap(
            capability="measurement records as IR values",
            op="gadget.rounds",
            evidence="qecl.qec returns only a codeblock; the syndrome it produces is not an "
            "IR value, so no later op can refer to it",
            workaround="give qecl.qec a syndrome result, or lower rounds to qecp where "
            "decode_esm_css consumes an explicit syndrome",
        )
    current = op.in_block
    inserted = []
    for _ in range(op.count.value.data):
        cycle = _qecl.QecCycleOp(current)
        inserted.append(cycle)
        current = cycle.out_codeblock
    parent = op.parent_block()
    assert parent is not None
    for cycle in inserted:
        parent.insert_op_before(cycle, op)
    op.out_block.replace_by(current)
    op.detach()
    op.erase()
    result.qec_cycles += len(inserted)


def _lower_observable(fn: func.FuncOp, op: gd.ObservableOp, result: LoweringResult) -> None:
    """Reject an outcome, since ``qecl.measure`` is a destructive single-qubit projection
    rather than the parity of records a gadget outcome is."""
    raise LoweringGap(
        capability="non-destructive logical product measurement",
        op="gadget.observable",
        evidence=CATALYST_EVIDENCE["measurable_axes"]
        + "; qecl.measure is a destructive single-logical-qubit projection",
        workaround="a surgery outcome is a parity of syndrome records, not a projection; "
        "qecl needs a measurement op whose result is derived from records, or the outcome "
        "has to be assembled at the qecp level",
    )


__all__ = ["LoweringGap", "LoweringResult", "lower_gadget_to_qecl"]
