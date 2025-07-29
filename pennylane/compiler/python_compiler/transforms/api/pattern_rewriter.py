# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pattern rewriter API for quantum compilation passes."""

from collections.abc import Sequence
from numbers import Number

from xdsl.dialects import arith, builtin, tensor
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriterListener, PatternRewriteWalker
from xdsl.rewriter import InsertPoint

from ...dialects import quantum


class StateManagement:
    """A container class for managing state."""

    wire_to_qubit_map: dict[int, SSAValue]
    qubit_to_wire_map: dict[SSAValue, int]

    def __init__(self, wires: Sequence[int]):
        self._wires = tuple(wires)
        self.wire_to_qubit_map = {}
        self.qubit_to_wire_map = {}

    @property
    def wires(self) -> tuple[int]:
        """Wire labels."""
        return self._wires

    def update_qubit(self, old_qubit: SSAValue, new_qubit: SSAValue):
        """Update a qubit."""
        wire = self.qubit_to_wire_map[old_qubit]
        self.wire_to_qubit_map[wire] = new_qubit
        self.qubit_to_wire_map[new_qubit] = wire
        self.qubit_to_wire_map.pop(old_qubit, None)

    def __getitem__(self, val: int | SSAValue) -> int | SSAValue:
        if isinstance(val, SSAValue):
            return self.qubit_to_wire_map[val]
        return self.wire_to_qubit_map[val]

    def __setitem__(self, key: int | SSAValue, item: SSAValue | int):
        if isinstance(key, SSAValue):
            old_wire = self.qubit_to_wire_map.pop(key, None)
            self.wire_to_qubit_map.pop(old_wire, None)
            self.qubit_to_wire_map[key] = item
            self.wire_to_qubit_map[item] = key
        else:
            old_qubit = self.wire_to_qubit_map.pop(key, None)
            self.qubit_to_wire_map.pop(old_qubit, None)
            self.wire_to_qubit_map[key] = item
            self.qubit_to_wire_map[item] = key

    def get_static_wire(self, op: quantum.ExtractOp, update=True) -> int | None:
        """Get the wire label to which a qubit extraction corresponds."""
        if (idx_attr := getattr(op, "idx_attr", None)) is not None:
            wire = idx_attr.value.data
            if update:
                self[wire] = op.qubit
            return wire

        return None


class PLPatternRewriter(PatternRewriter):
    """A ``PatternRewriter`` with abstractions for quantum compilation passes.

    This is a subclass of ``xdsl.pattern_rewriter.PatternRewriter`` that exposes
    methods to abstract away low-level pattern-rewriting details relevant to
    quantum compilation passes.
    """

    def erase_quantum_gate_op(self, op: Operation) -> None:
        """Erase a quantum gate.

        Safely erase a quantum gate from the module being transformed. This method automatically
        handles and pre-processing required before safely erasing an operation. To erase quantum
        gates, which include ``CustomOp``, ``MultiRZOp``, and ``QubitUnitaryOp``, it is recommended
        to use this method instead of ``erase_op``.

        Args:
            op (xdsl.ir.Operation): The operation to erase
        """
        if not isinstance(op, (quantum.CustomOp, quantum.MultiRZOp, quantum.QubitUnitaryOp)):
            return

        # We can also use the following code to perform the same task:
        # self.replace_op(op, (), op.in_qubits + op.in_ctrl_qubits)

        for iq, oq in zip(op.in_qubits, op.out_qubits, strict=True):
            self.replace_all_uses_with(oq, iq)
        for icq, ocq in zip(op.in_ctrl_qubits, op.out_ctrl_qubits, strict=True):
            self.replace_all_uses_with(ocq, icq)

        self.erase_op(op)

    def create_constant_op(self, cst: Number, insert_point: InsertPoint = None):
        """Create a constant and insert it into the IR. The corresponding SSA value is returned."""
        data = [cst]
        if isinstance(cst, float):
            elem_type = builtin.Float64Type()
        elif isinstance(cst, complex):
            elem_type = builtin.ComplexType()
        else:
            elem_type = builtin.IntegerType(64)

        type_ = builtin.TensorType(elem_type, [1])
        constAttr = builtin.DenseIntOrFPElementsAttr.from_list(type_, data)
        constantOp = arith.ConstantOp(constAttr)
        indexOp = arith.ConstantOp.from_int_and_width(0, 64)
        extractOp = tensor.ExtractOp(
            tensor=constantOp.result, indices=indexOp.result, result_type=elem_type
        )

        if insert_point is None:
            self.insert_op_before_matched_op(constantOp)
            self.insert_op_before_matched_op(indexOp)
            self.insert_op_before_matched_op(extractOp)

        else:
            self.insert_op(constantOp, insert_point)
            self.insert_op(indexOp, insert_point)
            self.insert_op(extractOp, insert_point)

        return extractOp.result


class PLPatternRewriteWalker(PatternRewriteWalker):
    """A ``PatternRewriteWalker`` for traversing and rewriting modules.

    This is a subclass of ``xdsl.pattern_rewriter.PatternRewriteWalker that uses a custom
    rewriter that contains abstractions for quantum compilation passes."""

    def _process_worklist(self, listener: PatternRewriterListener) -> bool:
        """
        Process the worklist until it is empty.
        Returns true if any modification was done.
        """
        rewriter_has_done_action = False

        # Handle empty worklist
        op = self._worklist.pop()
        if op is None:
            return rewriter_has_done_action

        # Create a rewriter on the first operation
        # Here, we use our custom rewriter instead of the default PatternRewriter.
        rewriter = PLPatternRewriter(op)
        rewriter.extend_from_listener(listener)

        # do/while loop
        while True:
            # Reset the rewriter on `op`
            rewriter.has_done_action = False
            rewriter.current_operation = op
            rewriter.insertion_point = InsertPoint.before(op)

            # Apply the pattern on the operation
            try:
                self.pattern.match_and_rewrite(op, rewriter)
            except Exception as err:  # pylint: disable=broad-exception-caught
                op.emit_error(
                    f"Error while applying pattern: {err}",
                    underlying_error=err,
                )
            rewriter_has_done_action |= rewriter.has_done_action

            # If the worklist is empty, we are done
            op = self._worklist.pop()
            if op is None:
                return rewriter_has_done_action
