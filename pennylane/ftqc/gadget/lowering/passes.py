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

import numpy as np
from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Operation, SSAValue
from xdsl.parser import Parser

from .. import _gf2
from ..ir import GadgetError
from ..support import CATALYST_EVIDENCE
from . import dialect as gd
from .catalyst_dialects import load_dialect_module
from .emit import context

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
        LoweringGap: if a gadget uses a codeblock with ``k > 1``, a deformation or detach, an
            outcome, a frame update, more than one phase, or records used by later operations

    **Example**

    .. code-block:: python

        from pennylane.ftqc.gadget.library import steane_memory
        from pennylane.ftqc.gadget.lowering import emit, lower_gadget_to_qecl

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
            if isinstance(op, gd.ObservableOp):
                raise _observable_gap()
            if isinstance(op, gd.FrameUpdateOp):
                raise LoweringGap(
                    capability="classically conditioned Pauli frame update",
                    op="gadget.frame_update",
                    evidence="the PauliFrame dialect exists in Catalyst but qecl has no op "
                    "that consumes a measurement record to condition a frame update",
                    workaround="lower the frame update into the PauliFrame dialect rather "
                    "than qecl, and keep the record identity attached to the producing op",
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


def _observable_gap() -> LoweringGap:
    """The gap for an outcome: ``qecl.measure`` is a destructive single-qubit projection, not
    the parity of records a gadget outcome is."""
    return LoweringGap(
        capability="non-destructive logical product measurement",
        op="gadget.observable",
        evidence=CATALYST_EVIDENCE["measurable_axes"]
        + "; qecl.measure is a destructive single-logical-qubit projection",
        workaround="a surgery outcome is a parity of syndrome records, not a projection; "
        "qecl needs a measurement op whose result is derived from records, or the outcome "
        "has to be assembled at the qecp level",
    )


def inline_gadget_call(payload: str, codeblock: SSAValue) -> tuple[list[Operation], SSAValue]:
    """Lower an emitted gadget to ``qecl`` operations acting on a given codeblock.

    This is the hook Catalyst's ``convert-quantum-to-qecl`` pass uses for a call recorded by
    :func:`~pennylane.ftqc.gadget.apply`: the payload is the text of the gadget's emitted module,
    and the returned operations replace the call. Each ``qecl.qec`` operation is tagged with
    the gadget's name and its code's check matrices, so that :func:`check_pipeline_code` can
    later compare them with the code the pipeline lowers to.

    Args:
        payload (str): IR text produced by :func:`~.lowering.emit`
        codeblock (SSAValue): the ``!qecl.codeblock`` value the gadget acts on

    Returns:
        tuple[list[Operation], SSAValue]: the operations to insert, in order, and the
        codeblock value after the gadget

    Raises:
        LoweringGap: if the gadget cannot be expressed in ``qecl``
        GadgetError: if the payload holds no gadget, or the gadget acts on a different
            codeblock type
    """
    module = Parser(context(), payload).parse_module()
    lower_gadget_to_qecl(module)
    fns = [
        op
        for op in module.body.block.ops
        if isinstance(op, func.FuncOp) and "gadget.action" in op.attributes
    ]
    if len(fns) != 1:
        raise GadgetError(f"gadget payload must hold exactly one gadget, found {len(fns)}")
    fn = fns[0]
    arg = fn.body.block.args[0]
    if arg.type != codeblock.type:
        raise GadgetError(
            f"gadget {fn.sym_name.data} acts on {arg.type} but is applied to {codeblock.type}"
        )

    tags = {
        "gadget.name": StringAttr(fn.sym_name.data),
        "gadget.code": fn.attributes["gadget.code"],
        "gadget.code_hx": fn.attributes["gadget.code_hx"],
        "gadget.code_hz": fn.attributes["gadget.code_hz"],
    }
    mapping: dict[SSAValue, SSAValue] = {arg: codeblock}
    inserted: list[Operation] = []
    for op in fn.body.block.ops:
        if isinstance(op, func.ReturnOp):
            return inserted, mapping.get(op.operands[0], op.operands[0])
        new = op.clone(value_mapper=mapping)
        if isinstance(new, _qecl.QecCycleOp):
            new.attributes.update(tags)
        inserted.append(new)
    raise GadgetError(f"gadget {fn.sym_name.data} has no return")  # pragma: no cover


def check_pipeline_code(module: ModuleOp, x_checks, z_checks, code: str) -> None:
    """Check that every inlined gadget was written for the code the pipeline lowers to.

    ``qecl.qec`` does not name a code; the code is chosen when ``qecl`` is lowered to
    ``qecp``. Catalyst's ``convert-qecl-to-qecp`` pass calls this function, so that a gadget
    written for one code is not silently compiled with another. Codes are compared by the
    stabilizer groups their checks generate, on the same qubit order.

    Args:
        module (ModuleOp): the module being lowered
        x_checks (array_like): X checks of the pipeline code
        z_checks (array_like): Z checks of the pipeline code
        code (str): name of the pipeline code, for error messages

    Raises:
        GadgetError: if a gadget's code differs from the pipeline code
    """
    x = np.asarray(x_checks, dtype=np.uint8)
    z = np.asarray(z_checks, dtype=np.uint8)
    for op in module.walk():
        if not isinstance(op, _qecl.QecCycleOp) or "gadget.code_hx" not in op.attributes:
            continue
        hx = _matrix(op.attributes["gadget.code_hx"])
        hz = _matrix(op.attributes["gadget.code_hz"])
        if not (_same_group(hx, x) and _same_group(hz, z)):
            raise GadgetError(
                f"gadget {op.attributes['gadget.name'].data} was written for code "
                f"{op.attributes['gadget.code'].data}, whose checks differ from those of the "
                f"pipeline code {code}. Compile it with a pipeline code that has the same "
                "checks on the same qubits."
            )


def _matrix(attr) -> np.ndarray:
    shape = attr.get_type().get_shape()
    return np.array(attr.get_values(), dtype=np.uint8).reshape(shape)


def _same_group(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape[1:] != b.shape[1:]:
        return False
    rank = _gf2.rank(np.vstack([a, b]))
    return rank == _gf2.rank(a) == _gf2.rank(b)


__all__ = [
    "LoweringGap",
    "LoweringResult",
    "lower_gadget_to_qecl",
    "inline_gadget_call",
    "check_pipeline_code",
]
