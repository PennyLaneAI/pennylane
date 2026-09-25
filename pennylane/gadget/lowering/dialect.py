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
This module contains a proposed ``gadget`` xDSL dialect, the IR that traced gadgets are
emitted into.

The ``qecl`` dialect has no operation that changes the measured stabilizer group, no value
for measurement records, and nowhere to attach detectors, so gadgets are first emitted into
this dialect:

* The encoded qubits are a ``!qecl.codeblock<k>`` value, so the IR interoperates with the
  existing ``qecl`` pipeline. The type does not record the current phase.
* Phases are module-level ``gadget.phase`` symbols; ``gadget.deform`` and
  ``gadget.rounds`` refer to them by name, so each check matrix appears once per module.
* Measurement records are SSA values of type ``!gadget.records<rounds x width>``. Parities
  are ``[operand, round, check]`` triples into them, so record identity belongs to the
  producing operation.
* Detectors are a module-level ``gadget.detectors`` operation referring to the gadget
  function by symbol, holding binary detector and observable matrices over the gadget's
  flat record index.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseIntOrFPElementsAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
    TensorType,
    i8,
    i64,
)
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    VarConstraint,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer


@irdl_attr_definition
class RecordsType(ParametrizedAttribute, TypeAttribute):
    """The type of a block of measurement records: ``rounds`` rounds of ``width`` outcomes.

    Printed as ``!gadget.records<rounds x width>``.

    Args:
        rounds (int or IntegerAttr): number of rounds
        width (int or IntegerAttr): number of outcomes per round
    """

    name = "gadget.records"

    rounds: IntegerAttr[IntegerType]
    width: IntegerAttr[IntegerType]

    def __init__(self, rounds: int | IntegerAttr, width: int | IntegerAttr):
        if isinstance(rounds, int):
            rounds = IntegerAttr(rounds, 64)
        if isinstance(width, int):
            width = IntegerAttr(width, 64)
        super().__init__(rounds, width)

    def print_parameters(self, printer: Printer) -> None:
        """Print ``<rounds x width>``."""
        with printer.in_angle_brackets():
            printer.print_int(self.rounds.value.data)
            printer.print_string(" x ")
            printer.print_int(self.width.value.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser):
        """Parse ``<rounds x width>``."""
        with parser.in_angle_brackets():
            rounds = parser.parse_integer()
            parser.parse_characters("x")
            width = parser.parse_integer()
        return [IntegerAttr(rounds, 64), IntegerAttr(width, 64)]

    @property
    def n_rounds(self) -> int:
        """Number of rounds."""
        return self.rounds.value.data

    @property
    def n_width(self) -> int:
        """Number of outcomes per round."""
        return self.width.value.data


def bits_attr(matrix) -> DenseIntOrFPElementsAttr:
    """Pack a binary matrix or vector into a dense ``i8`` elements attribute.

    Args:
        matrix (array_like): the binary matrix or vector

    Returns:
        DenseIntOrFPElementsAttr: the attribute
    """
    arr = np.asarray(matrix, dtype=np.uint8)
    return DenseIntOrFPElementsAttr.from_list(
        TensorType(i8, list(arr.shape)), [int(v) for v in arr.reshape(-1)]
    )


@irdl_op_definition
class PhaseOp(IRDLOperation):
    """Declare a phase: a stabilizer group some part of the module measures.

    ``active`` is stored explicitly, because a qubit can be live in a phase without any
    check acting on it.

    Args:
        sym_name (str): symbol name of the phase
        hx (array_like): X checks
        hz (array_like): Z checks
        active (array_like): mask of live qubits
    """

    name = "gadget.phase"

    sym_name = prop_def(StringAttr)
    hx = prop_def(DenseIntOrFPElementsAttr)
    hz = prop_def(DenseIntOrFPElementsAttr)
    active = prop_def(DenseIntOrFPElementsAttr)

    def __init__(self, sym_name: str, hx, hz, active):
        super().__init__(
            properties={
                "sym_name": StringAttr(sym_name),
                "hx": bits_attr(hx),
                "hz": bits_attr(hz),
                "active": bits_attr(active),
            }
        )


@irdl_op_definition
class DeformOp(IRDLOperation):
    """Switch a codeblock to another phase over the same qubit frame.

    ``init`` holds ``[qubit, axis]`` pairs for the qubits the new phase activates, with axis
    ``0`` for X and ``1`` for Z.

    Args:
        in_block (SSAValue): the codeblock
        to_phase (str): symbol name of the phase entered
        init (tuple[tuple[int, str]]): activated qubits and their preparation basis
    """

    T: ClassVar = VarConstraint("T", base(Attribute))

    name = "gadget.deform"

    in_block = operand_def(T)
    out_block = result_def(T)

    to_phase = prop_def(SymbolRefAttr)
    init = opt_prop_def(ArrayAttr)

    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "$in_block attr-dict `:` type($in_block)"

    def __init__(self, in_block, to_phase: str, init: tuple[tuple[int, str], ...] = ()):
        pairs = ArrayAttr(
            [
                ArrayAttr([IntegerAttr(int(q), i64), IntegerAttr(0 if a == "x" else 1, i64)])
                for q, a in init
            ]
        )
        super().__init__(
            operands=(in_block,),
            result_types=(in_block.type,),
            properties={"to_phase": SymbolRefAttr(to_phase), "init": pairs},
        )


@irdl_op_definition
class RoundsOp(IRDLOperation):
    """Measure the current phase's checks for a fixed number of rounds.

    The round count is an attribute rather than an operand, so the detector structure is
    fixed at compile time.

    Args:
        in_block (SSAValue): the codeblock
        phase (str): symbol name of the phase measured
        count (int): number of rounds
        width (int): number of checks in the phase
    """

    T: ClassVar = VarConstraint("T", base(Attribute))

    name = "gadget.rounds"

    in_block = operand_def(T)
    out_block = result_def(T)
    records = result_def(RecordsType)

    phase = prop_def(SymbolRefAttr)
    count = prop_def(IntegerAttr[IntegerType])

    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "$in_block attr-dict `:` type($in_block) `,` type($records)"

    def __init__(self, in_block, phase: str, count: int, width: int):
        super().__init__(
            operands=(in_block,),
            result_types=(in_block.type, RecordsType(count, width)),
            properties={
                "phase": SymbolRefAttr(phase),
                "count": IntegerAttr(count, i64),
            },
        )


@irdl_op_definition
class DetachOp(IRDLOperation):
    """Switch a codeblock to another phase, reading out the qubits it deactivates.

    The readouts are a one-round record block whose check index enumerates
    ``measure_out`` in order.

    Args:
        in_block (SSAValue): the codeblock
        to_phase (str): symbol name of the phase entered
        measure_out (tuple[tuple[int, str]]): qubits read out and their readout basis
    """

    T: ClassVar = VarConstraint("T", base(Attribute))

    name = "gadget.detach"

    in_block = operand_def(T)
    out_block = result_def(T)
    records = result_def(RecordsType)

    to_phase = prop_def(SymbolRefAttr)
    measure_out = prop_def(ArrayAttr)

    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "$in_block attr-dict `:` type($in_block) `,` type($records)"

    def __init__(self, in_block, to_phase: str, measure_out: tuple[tuple[int, str], ...]):
        pairs = ArrayAttr(
            [
                ArrayAttr([IntegerAttr(int(q), i64), IntegerAttr(0 if a == "x" else 1, i64)])
                for q, a in measure_out
            ]
        )
        super().__init__(
            operands=(in_block,),
            result_types=(in_block.type, RecordsType(1, len(measure_out))),
            properties={"to_phase": SymbolRefAttr(to_phase), "measure_out": pairs},
        )


def parity_attr(terms: tuple[tuple[int, int, int], ...], entry: tuple[int, ...]) -> ArrayAttr:
    """Encode a parity as ``[[operand, round, check], ...]`` triples and entry-syndrome bits.

    Args:
        terms (tuple[tuple[int, int, int]]): record operand, round and check of each outcome
        entry (tuple[int]): entry-syndrome bits

    Returns:
        ArrayAttr: ``[terms, entry]``
    """
    return ArrayAttr(
        [
            ArrayAttr(
                [
                    ArrayAttr(
                        [
                            IntegerAttr(int(o), i64),
                            IntegerAttr(int(r), i64),
                            IntegerAttr(int(c), i64),
                        ]
                    )
                    for o, r, c in terms
                ]
            ),
            ArrayAttr([IntegerAttr(int(s), i64) for s in entry]),
        ]
    )


@irdl_op_definition
class ObservableOp(IRDLOperation):
    """Declare that a parity of records is outcome ``index`` of the gadget.

    Args:
        records (SSAValue): the record block the parity refers to
        index (int): outcome index
        terms (Sequence[tuple[int, int, int]]): the parity, as index triples
        entry (Sequence[int]): entry-syndrome bits in the parity
    """

    name = "gadget.observable"

    records = operand_def(RecordsType)
    index = prop_def(IntegerAttr[IntegerType])
    parity = prop_def(ArrayAttr)

    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "$records attr-dict `:` type($records)"

    def __init__(self, records, index: int, terms, entry=()):
        super().__init__(
            operands=(records,),
            result_types=(),
            properties={
                "index": IntegerAttr(int(index), i64),
                "parity": parity_attr(tuple(terms), tuple(entry)),
            },
        )


@irdl_op_definition
class FrameUpdateOp(IRDLOperation):
    """Apply one row of the gadget's declared Pauli frame updates, conditioned on a parity.

    The update is a row index into the ``gadget.frame_update`` matrix of the enclosing
    function; the operation cannot name an arbitrary Pauli operator.

    Args:
        in_block (SSAValue): the codeblock
        records (SSAValue): the record block the parity refers to
        update_index (int): row of the declared frame-update matrix
        terms (Sequence[tuple[int, int, int]]): the parity, as index triples
        entry (Sequence[int]): entry-syndrome bits in the parity
    """

    T: ClassVar = VarConstraint("T", base(Attribute))

    name = "gadget.frame_update"

    in_block = operand_def(T)
    records = operand_def(RecordsType)
    out_block = result_def(T)

    update_index = prop_def(IntegerAttr[IntegerType])
    parity = prop_def(ArrayAttr)

    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "$in_block `,` $records attr-dict `:` type($in_block) `,` type($records)"

    def __init__(self, in_block, records, update_index: int, terms, entry=()):
        super().__init__(
            operands=(in_block, records),
            result_types=(in_block.type,),
            properties={
                "update_index": IntegerAttr(int(update_index), i64),
                "parity": parity_attr(tuple(terms), tuple(entry)),
            },
        )


@irdl_op_definition
class DetectorsOp(IRDLOperation):
    """The detectors and observables of one gadget function.

    Both are binary matrices over the function's flat record index. ``undetermined`` lists
    the flat indices of records that have no detector because their outcome is random.

    Args:
        of (str): symbol name of the gadget function
        regime (str): noise regime the detectors are valid for
        detectors (array_like): ``(n_detectors, n_records)`` matrix
        observables (array_like): ``(n_outcomes, n_records)`` matrix
        entry_width (int): number of entry-syndrome bits
        undetermined (tuple[int]): flat indices of records without a detector
    """

    name = "gadget.detectors"

    of = prop_def(SymbolRefAttr)
    regime = prop_def(StringAttr)
    detectors = prop_def(DenseIntOrFPElementsAttr)
    observables = prop_def(DenseIntOrFPElementsAttr)
    entry_width = prop_def(IntegerAttr[IntegerType])
    undetermined = prop_def(ArrayAttr)

    def __init__(
        self,
        of: str,
        regime: str,
        detectors,
        observables,
        entry_width: int,
        undetermined: tuple[int, ...] = (),
    ):
        super().__init__(
            properties={
                "of": SymbolRefAttr(of),
                "regime": StringAttr(regime),
                "detectors": bits_attr(detectors),
                "observables": bits_attr(observables),
                "entry_width": IntegerAttr(int(entry_width), i64),
                "undetermined": ArrayAttr([IntegerAttr(int(i), i64) for i in undetermined]),
            }
        )


GadgetDialect = Dialect(
    "gadget",
    [
        PhaseOp,
        DeformOp,
        RoundsOp,
        DetachOp,
        ObservableOp,
        FrameUpdateOp,
        DetectorsOp,
    ],
    [RecordsType],
)
"""xdsl.ir.Dialect: the ``gadget`` dialect"""

__all__ = [
    "RecordsType",
    "PhaseOp",
    "DeformOp",
    "RoundsOp",
    "DetachOp",
    "ObservableOp",
    "FrameUpdateOp",
    "DetectorsOp",
    "GadgetDialect",
    "bits_attr",
    "parity_attr",
]
