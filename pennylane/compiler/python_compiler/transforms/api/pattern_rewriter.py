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

from xdsl.dialects import arith, builtin, stablehlo, tensor
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriterListener, PatternRewriteWalker
from xdsl.rewriter import InsertPoint

from pennylane import measurements, ops
from pennylane.operation import Operator

from ...dialects import mbqc, quantum

# Tuple of all operations that return qubits
_ops_returning_qubits = (
    quantum.CustomOp,
    quantum.AllocQubitOp,
    quantum.ExtractOp,
    quantum.GlobalPhaseOp,
    quantum.MeasureOp,
    quantum.MultiRZOp,
    quantum.QubitUnitaryOp,
    quantum.SetBasisStateOp,
    quantum.SetStateOp,
    mbqc.MeasureInBasisOp,
)

# Tuple of all operations that return "out_qubits"
_out_qubits_ops = (
    quantum.CustomOp,
    quantum.MultiRZOp,
    quantum.QubitUnitaryOp,
    quantum.SetBasisStateOp,
    quantum.SetStateOp,
)

# Tuple of all operations that return "out_ctrl_qubits"
_out_ctrl_qubits_ops = (
    quantum.CustomOp,
    quantum.GlobalPhaseOp,
    quantum.MultiRZOp,
    quantum.QubitUnitaryOp,
)

# Tuple of all operations that return "out_qubit"
_out_qubit_ops = (quantum.MeasureOp, mbqc.MeasureInBasisOp)

# Tuple of all operations that return "qubit"
_qubit_ops = (quantum.AllocQubitOp, quantum.ExtractOp)


class StateManagement:
    """A container class for managing wire mapping."""

    wire_to_qubit_map: dict[int, SSAValue]
    qubit_to_wire_map: dict[SSAValue, int]

    def __init__(self, wires: Sequence[int] | None = None):
        self._wires = tuple(wires) if wires else ()
        self.wire_to_qubit_map = {}
        self.qubit_to_wire_map = {}

    @property
    def wires(self) -> tuple[int]:
        """Wire labels."""
        return self._wires

    def update_qubit(self, old_qubit: SSAValue, new_qubit: SSAValue) -> None:
        """Update a qubit."""
        wire = self.qubit_to_wire_map[old_qubit]
        self.wire_to_qubit_map[wire] = new_qubit
        self.qubit_to_wire_map[new_qubit] = wire
        self.qubit_to_wire_map.pop(old_qubit, None)

    def __getitem__(self, val: int | SSAValue) -> int | SSAValue | None:
        if isinstance(val, SSAValue):
            return self.qubit_to_wire_map[val]

        if self._wires and val not in self._wires:
            raise ValueError(f"{val} is not an available wire.")
        return self.wire_to_qubit_map.get(val, None)

    def __setitem__(self, key: int | SSAValue, item: SSAValue | int) -> None:
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
        wire = None
        if (idx_attr := getattr(op, "idx_attr", None)) is not None:
            wire = idx_attr.value.data

        else:
            idx = op.idx
            if isinstance(idx.owner, arith.ConstantOp):
                wire = idx.owner.properties["value"].data
            elif isinstance(idx.owner, tensor.ExtractOp):
                operand = idx.owner.operands[0]
                if isinstance(operand.owner, stablehlo.ConstantOp):
                    wire = operand.owner.properties["value"].get_values()[0]

        if wire is not None and update:
            self[wire] = op.qubit
        return wire

    def update_from_op(self, op: Operation):
        """Update the wire mapping from an operation's outputs"""
        if isinstance(op, quantum.ExtractOp):
            _ = self.get_static_wire(op, update=True)

        if isinstance(op, _out_qubit_ops):
            self.update_qubit(op.in_qubit, op.out_qubit)

        if isinstance(op, _out_qubits_ops):
            for iq, oq in zip(op.in_qubits, op.out_qubits, strict=True):
                self.update_qubit(iq, oq)

        if isinstance(op, _out_ctrl_qubits_ops):
            for iq, oq in zip(op.in_ctrl_qubits, op.out_ctrl_qubits, strict=True):
                self.update_qubit(iq, oq)

        if isinstance(op, (quantum.InsertOp, quantum.DeallocQubitOp)):
            qubit = op.qubit
            wire = self.qubit_to_wire_map.pop(qubit, None)
            _ = self.wire_to_qubit_map.pop(wire, None)


def _get_bfs_out_qubits(op):
    out_qubits = ()
    if not isinstance(op, _ops_returning_qubits):
        return out_qubits

    if isinstance(op, _out_qubits_ops):
        out_qubits += tuple(op.out_qubits)
    if isinstance(op, _out_ctrl_qubits_ops):
        out_qubits += tuple(op.out_ctrl_qubits)
    if isinstance(op, _out_qubit_ops):
        out_qubits += (op.out_qubit,)
    if isinstance(op, _qubit_ops):
        out_qubits += (op.qubit,)

    return out_qubits


# TODO: Integration StateManagement with rewriting


class PLPatternRewriter(PatternRewriter):
    """A ``PatternRewriter`` with abstractions for quantum compilation passes.

    This is a subclass of ``xdsl.pattern_rewriter.PatternRewriter`` that exposes
    methods to abstract away low-level pattern-rewriting details relevant to
    quantum compilation passes.
    """

    def __init__(self, current_operation: Operation):
        super().__init__(current_operation)
        self.wire_manager = StateManagement()

    def erase_quantum_gate_op(self, op: Operation, update_qubits: bool = True) -> None:
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

        for iq, oq in zip(
            op.in_qubits + op.in_ctrl_qubits, op.out_qubits + op.out_ctrl_qubits, strict=True
        ):
            self.replace_all_uses_with(oq, iq)
            if update_qubits:
                self.wire_manager.update_qubit(oq, iq)

        self.erase_op(op)

    def create_constant(self, cst: Number, insert_point: InsertPoint):
        """Create a ConstantOp and insert it into the IR. The corresponding SSA value is returned."""
        data = [cst]
        if isinstance(cst, float):
            elem_type = builtin.Float64Type()
        elif isinstance(cst, complex):
            elem_type = builtin.ComplexType()
        elif isinstance(cst, bool):
            elem_type = builtin.IntegerType(1)
        else:
            elem_type = builtin.IntegerType(64)

        type_ = builtin.TensorType(elem_type, [1])
        constAttr = builtin.DenseIntOrFPElementsAttr.from_list(type_, data)
        constantOp = arith.ConstantOp(constAttr)
        indexOp = arith.ConstantOp.from_int_and_width(0, 64)
        extractOp = tensor.ExtractOp(
            tensor=constantOp.result, indices=indexOp.result, result_type=elem_type
        )

        self.insert_op(constantOp, insert_point)
        self.insert_op(indexOp, insert_point)
        self.insert_op(extractOp, insert_point)

        return extractOp.result

    def iter_qubit_successors(self, op: Operation, traversal_type="bfs"):
        """Iterator function to do a breadth-first traversal over the output qubits
        of an operation. First returned value is the original operation."""
        if traversal_type not in ("bfs", "dfs"):
            raise ValueError(
                f"Unrecognized traversal type {traversal_type}. Valid types are 'bfs' and 'dfs'"
            )
        pop_idx = 0 if traversal_type == "bfs" else -1
        op_queue = [op]

        while op_queue:
            cur_op = op_queue.pop(pop_idx)
            out_qubits = _get_bfs_out_qubits(op)
            for q in out_qubits:
                for use in q.uses:
                    use_op = use.operation
                    if use_op not in op_queue:
                        op_queue.append(use_op)

            yield cur_op

    def _get_gate_base(self, gate: Operator):
        """Get the base op of a gate."""
        if not isinstance(gate, ops.SymbolicOp):
            return gate, (), (), False

        if isinstance(gate, ops.Controlled):
            base_gate, ctrl_wires, ctrl_vals, adjoint = self._get_gate_base(gate.base)
            ctrl_wires = tuple(gate.control_wires) + tuple(ctrl_wires)
            ctrl_vals = tuple(gate.control_values) + tuple(ctrl_vals)
            return base_gate, ctrl_wires, ctrl_vals, adjoint

        if isinstance(gate, ops.Adjoint):
            base_gate, ctrl_wires, ctrl_vals, adjoint = self._get_gate_base(gate.base)
            adjoint = adjoint ^ True
            return base_gate, tuple(ctrl_wires), tuple(ctrl_vals), adjoint

        raise RuntimeError("reeeeeeeee")

    def insert_gate(
        self, gate: Operator, insert_point: InsertPoint, params: SSAValue | None = None
    ):
        """Insert a PL gate into the IR at the provided insertion point."""
        gate, ctrl_wires, ctrl_vals, adjoint = self._get_gate_base(gate)

        # TODO: Add branches for MultiRZ, GlobalPhase, and QubitUnitary
        params = params or tuple(
            self.create_constant(d, insert_point=insert_point) for d in gate.data
        )
        in_qubits = tuple(self.wire_manager[w] for w in gate.wires)
        in_ctrl_qubits = tuple(self.wire_manager[w] for w in ctrl_wires) if ctrl_wires else None
        in_ctrl_values = None

        if ctrl_vals:
            true_cst = None
            false_cst = None
            if any(ctrl_vals):
                true_cst = self.create_constant(True, insert_point=insert_point)
            if not all(ctrl_vals):
                false_cst = self.create_constant(False, insert_point=insert_point)
            in_ctrl_values = tuple(true_cst if v else false_cst for v in ctrl_vals)

        customOp = quantum.CustomOp(
            in_qubits=in_qubits,
            gate_name=gate.name,
            params=params,
            in_ctrl_qubits=in_ctrl_qubits,
            in_ctrl_values=in_ctrl_values,
            adjoint=adjoint,
        )
        self.insert_op(customOp, insertion_point=insert_point)

        for iq, oq in zip(
            customOp.in_qubits + customOp.in_ctrl_qubits,
            customOp.out_qubits + customOp.out_ctrl_qubits,
            strict=True,
        ):
            iq.replace_by_if(oq, lambda use: use.operation != customOp)
            self.notify_op_modified(customOp)
            self.wire_manager.update_qubit(iq, oq)

    def insert_observable(self, obs: Operator, insert_point: InsertPoint):
        """Insert a PL observable into the IR at the provided insertion point."""

    def insert_measurement(self, mp: measurements.MeasurementProcess, insert_point: InsertPoint):
        """Insert a PL measurement into the IR at the provided insertion point."""

    def insert_mid_measure(self, mcm: measurements.MidMeasureMP, insert_point: InsertPoint):
        """Insert a PL measurement into the IR at the provided insertion point."""


# pylint: disable=too-few-public-methods
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
            # pylint: disable=attribute-defined-outside-init
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
