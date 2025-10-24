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

"""This file contains the implementation of the outline_state_evolution transform.

Known limitations
-----------------

    *   If the current pass is applied multiple times, the transform will fail as it would redefined the `state_evolution` func. This is
        caused by the way we define the terminal_boundary_op. Each time the pass is applied to the IR, it would insert a new
        terminal_boundary_op into the IR. TODOs: Instead of inserting a new `terminal_boundary_op` op to the IR when applying the pass, it
        would be better to: 1. define a quantum.terminator op before this pass and use it as a delineation of quantum gate operation;
        2. move the `simplify_io` to a separate pass.
"""

from dataclasses import dataclass
from itertools import chain

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func
from xdsl.ir import Operation, SSAValue
from xdsl.rewriter import InsertPoint

from pennylane.compiler.python_compiler import compiler_transform
from pennylane.compiler.python_compiler.dialects import quantum


@dataclass(frozen=True)
class OutlineStateEvolutionPass(passes.ModulePass):
    """Pass that puts gate operations into an outline_state_evolution callable."""

    name = "outline-state-evolution"

    # pylint: disable=no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the outline-state-evolution pass."""
        for op in module.ops:
            if isinstance(op, func.FuncOp) and "qnode" in op.attributes:
                rewriter = pattern_rewriter.PatternRewriter(op)
                OutlineStateEvolutionPattern().match_and_rewrite(op, rewriter)


outline_state_evolution_pass = compiler_transform(OutlineStateEvolutionPass)


class OutlineStateEvolutionPattern(pattern_rewriter.RewritePattern):
    """RewritePattern for outlined state evolution regions in a quantum function."""

    # pylint: disable=too-few-public-methods
    def _get_parent_module(self, op: func.FuncOp) -> builtin.ModuleOp:
        """Get the first ancestral builtin.ModuleOp op of a given func.func op."""
        while (op := op.parent_op()) and not isinstance(op, builtin.ModuleOp):
            pass
        if op is None:
            raise RuntimeError(
                "The given qnode func is not nested within a builtin.module. Please ensure the qnode func is defined in a builtin.module."
            )
        return op

    def __init__(self):
        self.module: builtin.ModuleOp = None
        self.original_func_op: func.FuncOp = None

        # To determine the boundary of quantum gate operations in the IR
        self.alloc_op: quantum.AllocOp = None
        self.terminal_boundary_op: Operation = None

        # Input and outputs of the state evolution func
        self.required_inputs: list[SSAValue] = None
        self.required_outputs: list[SSAValue] = None

        # State evolution function region
        self.state_evolution_func: func.FuncOp = None

    # pylint: disable=too-many-arguments
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        """Transform a quantum function (qnode) to outline state evolution regions.
        This implementation assumes that there is only one `quantum.alloc` operation in
        the func operations with a "qnode" attribute and all quantum operations are between
        the unique `quantum.alloc` operation and the terminal_boundary_op. All operations in between
        are to be moved to the newly created outline-state-evolution function operation."""
        if "qnode" not in func_op.attributes:
            return

        self.original_func_op = func_op

        self.module = self._get_parent_module(func_op)

        # Simplify the quantum I/O to use only registers at boundaries
        self._simplify_quantum_io(func_op, rewriter)

        # Create a new function op for the state evolution region and insert it
        # into the parent scope of the original func with qnode attribute
        self._create_state_evolution_function(rewriter)

        # Replace the original region with a call to the state evolution function
        # by inserting the corresponding callOp and update the rest of operations
        # in the qnode func.
        self._finalize_transformation()

    # pylint: disable=no-else-return
    def _get_qubit_idx(self, op: Operation) -> int | None:
        """Get the index of qubit that an ExtractOp op extracts."""
        if getattr(op, "idx", None):
            return op.idx
        return getattr(op, "idx_attr", None)

    def _set_up_terminal_boundary_op(
        self,
        current_reg: quantum.QuregType,
        terminal_boundary_op: Operation | None,
        qubit_to_reg_idx: dict,
        op: Operation,
        rewriter: pattern_rewriter.PatternRewriter,
    ):
        """Set up the terminal boundary operation. This terminal_boundary_op is set as the last
        quantum.insert operations added to the IR."""
        insert_ops = set()

        # Insert all qubits recorded in the qubit_to_reg_idx dict before the
        # pre-assumed terminal operations.
        insertion_point = InsertPoint.before(op)
        for qb, idx in qubit_to_reg_idx.items():
            insert_op = quantum.InsertOp(current_reg, idx, qb)
            rewriter.insert_op(insert_op, insertion_point=insertion_point)
            insert_ops.add(insert_op)
            terminal_boundary_op = insert_op
            current_reg = insert_op.out_qreg

        # Add the `"terminal_boundary"` attribute to the last newly added
        # `quantum.insert` operation.
        if terminal_boundary_op is None:
            raise RuntimeError("A terminal_boundary_op op is not found in the circuit.")
        terminal_boundary_op.attributes["terminal_boundary"] = builtin.UnitAttr()
        prev_qreg = terminal_boundary_op.in_qreg

        # extract ops
        insertion_point = InsertPoint.before(op)
        for qb, idx in list(qubit_to_reg_idx.items()):
            extract_op = quantum.ExtractOp(current_reg, idx)
            rewriter.insert_op(extract_op, insertion_point=insertion_point)
            qb.replace_by_if(extract_op.qubit, lambda use: use.operation not in insert_ops)
            for use in qb.uses:
                rewriter.notify_op_modified(use.operation)
            # update the qubit_to_reg_idx dict
            qubit_to_reg_idx[extract_op.qubit] = idx
            # pop out qb from the dict
            del qubit_to_reg_idx[qb]
        return current_reg, prev_qreg, terminal_boundary_op

    # pylint: disable=cell-var-from-loop, too-many-branches
    def _simplify_quantum_io(
        self, func_op: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter
    ) -> func.FuncOp:
        """Simplify quantum I/O to use only registers at segment boundaries.

        This ensures that state evolution regions only take registers as input/output,
        not individual qubits.
        """
        current_reg = None
        # Note that all qubits recorded in the `qubit_to_reg_idx` will be inserted into
        # the IR and the last insert_op will be set as the `terminal_boundary_op`.`
        qubit_to_reg_idx = {}
        terminal_boundary_op = None
        terminal_op_in_reg = None

        for op in func_op.body.ops:
            match op:
                case quantum.AllocOp():
                    current_reg = op.qreg
                case quantum.ExtractOp():
                    # Update register mapping
                    extract_idx = self._get_qubit_idx(op)
                    qubit_to_reg_idx[op.qubit] = extract_idx
                    # branch to update extract_op with new qreg
                    if op.qreg is terminal_op_in_reg:
                        insertion_point = InsertPoint.before(op)
                        extract_op = quantum.ExtractOp(current_reg, extract_idx)
                        rewriter.insert_op(extract_op, insertion_point=insertion_point)
                        rewriter.replace_all_uses_with(op.results[0], extract_op.results[0])
                        rewriter.erase_op(op)

                case quantum.MeasureOp():
                    # TODOs: what if the qubit that quantum.measure target at is reset?
                    # Not a concern by EOY 2025
                    qubit_to_reg_idx[op.out_qubit] = qubit_to_reg_idx[op.in_qubit]
                    del qubit_to_reg_idx[op.in_qubit]
                case quantum.CustomOp():
                    # To update the qubit_to_reg_idx map for the return type.
                    for i, qb in enumerate(chain(op.in_qubits, op.in_ctrl_qubits)):
                        qubit_to_reg_idx[op.results[i]] = qubit_to_reg_idx[qb]
                        del qubit_to_reg_idx[qb]
                case quantum.InsertOp():
                    if not terminal_op_in_reg:
                        if op.idx_attr and qubit_to_reg_idx[op.qubit] is not op.idx_attr:
                            raise ValueError("op.qubit should be op.idx_attr.")
                        del qubit_to_reg_idx[op.qubit]
                        current_reg = op.out_qreg
                    # branch to update insert_op with new qreg
                    if op.in_qreg is terminal_op_in_reg:
                        insertion_point = InsertPoint.before(op)
                        index = op.idx if op.idx else op.idx_attr
                        insert_op = quantum.InsertOp(current_reg, index, op.qubit)
                        rewriter.insert_op(insert_op, insertion_point=insertion_point)
                        rewriter.replace_all_uses_with(op.out_qreg, insert_op.out_qreg)
                        rewriter.erase_op(op)

                case _ if (
                    isinstance(
                        op,
                        (
                            quantum.ComputationalBasisOp,
                            quantum.NamedObsOp,
                            quantum.HamiltonianOp,
                            quantum.TensorOp,
                        ),
                    )
                    and not terminal_boundary_op
                ):
                    current_reg, terminal_op_in_reg, terminal_boundary_op = (
                        self._set_up_terminal_boundary_op(
                            current_reg, terminal_boundary_op, qubit_to_reg_idx, op, rewriter
                        )
                    )
                case _:
                    # Handle other operations that might has qreg result
                    # Note that this branch might not be tested so far as adjoint op is not
                    # tested so far.
                    if reg := next(
                        (reg for reg in op.results if isinstance(reg.type, quantum.QuregType)), None
                    ):
                        current_reg = reg

    def _create_state_evolution_function(self, rewriter: pattern_rewriter.PatternRewriter):
        """Create a new func.func for the state evolution region using clone approach."""

        alloc_op, terminal_boundary_op = self._find_evolution_range()

        # collect operation from alloc_op to terminal_boundary_op
        ops_to_clone = self._collect_operations_in_range(alloc_op, terminal_boundary_op)

        # collect required inputs for the state evolution funcOp
        required_inputs = self._collect_required_inputs_for_state_evolution_func(ops_to_clone)

        # collect required outputs for the state evolution funcOp
        required_outputs = self._collect_required_outputs(ops_to_clone, terminal_boundary_op)

        register_inputs = []
        other_inputs = []
        for val in required_inputs:
            if isinstance(val.type, quantum.QuregType):
                register_inputs.append(val)
            else:
                other_inputs.append(val)

        register_outputs = []
        other_outputs = []
        for val in required_outputs:
            if isinstance(val.type, quantum.QuregType):
                register_outputs.append(val)
            else:
                other_outputs.append(val)

        ordered_inputs = register_inputs + other_inputs
        ordered_outputs = register_outputs + other_outputs

        input_types = [val.type for val in ordered_inputs]
        output_types = [val.type for val in ordered_outputs]
        fun_type = builtin.FunctionType.from_lists(input_types, output_types)

        # create a new func.func op and insert it into the IR
        state_evolution_func = func.FuncOp(
            self.original_func_op.sym_name.data + ".state_evolution", fun_type, visibility="public"
        )
        rewriter.insert_op(state_evolution_func, InsertPoint.at_end(self.module.body.block))

        # TODOs: how to define the `value_mapper` arg is not stated in the xdl.core module [here](https://github.com/xdslproject/xdsl/blob/e1301e0204bcf6ea5ed433e7da00bee57d07e695/xdsl/ir/core.py#L1429)_.
        # It looks like storing ssa value to be cloned would maintain the dependency relationship required to build the new DAG for the new ops.
        block = state_evolution_func.regions[0].block
        value_mapper = {}  # only args ssavlue is required
        for input, block_arg in zip(ordered_inputs, block.args):
            value_mapper[input] = block_arg

        self._clone_operations_to_block(ops_to_clone, block, value_mapper)
        self._add_return_statement(block, ordered_outputs, value_mapper)

        self.required_inputs = ordered_inputs
        self.required_outputs = ordered_outputs
        self.alloc_op = alloc_op
        self.terminal_boundary_op = terminal_boundary_op
        self.state_evolution_func = state_evolution_func

    def _find_evolution_range(self):
        """find alloc_op and terminal_boundary_op"""
        alloc_op = None
        terminal_boundary_op = None

        for op in self.original_func_op.body.walk():
            if isinstance(op, quantum.AllocOp):
                alloc_op = op
            elif hasattr(op, "attributes") and "terminal_boundary" in op.attributes:
                terminal_boundary_op = op

        if not (alloc_op and terminal_boundary_op):
            raise RuntimeError("Could not find both alloc_op and terminal_boundary_op")

        return alloc_op, terminal_boundary_op

    def _collect_operations_in_range(self, begin_op, end_op):
        """collect top-level operations in range, let op.clone() handle nesting"""
        ops_to_clone = []

        if begin_op.parent_block() != end_op.parent_block():
            raise ValueError("begin_op and end_op are not in the same block")

        block = begin_op.parent_block()

        # skip until begin_op
        op_iter = iter(block.ops)
        while (op := next(op_iter, None)) != begin_op:
            pass

        # collect top-level operations until end_op
        while (op := next(op_iter, None)) != end_op:
            ops_to_clone.append(op)

        # collect the terminal_boundary_op
        if op is not None:
            ops_to_clone.append(op)

        return ops_to_clone

    def _collect_required_inputs_for_state_evolution_func(
        self, ops: list[Operation]
    ) -> list[SSAValue]:
        """Collect required inputs for the state evolution funcOp with a given list of operations.
        Note that this method does not intent to keep the order of required input SSAValues.
        """
        ops_walk = list(chain(*[op.walk() for op in ops]))

        # a set records the ssa values defined by the ops list
        ops_defined_values = set()
        # a set records all the ssa values required for all operations in the ops list
        all_operands = set()

        for nested_op in ops_walk:
            ops_defined_values.update(nested_op.results)
            all_operands.update(nested_op.operands)

            if hasattr(nested_op, "regions") and nested_op.regions:
                for region in nested_op.regions:
                    for block in region.blocks:
                        ops_defined_values.update(block.args)

        # the ssa values not defined by the operations in the ops list
        missing_defs = list(all_operands - ops_defined_values)
        required_inputs = [v for v in missing_defs if v is not None]

        return required_inputs

    def _collect_required_outputs(
        self, ops: list[Operation], terminal_op: Operation
    ) -> list[SSAValue]:
        """Get required outputs for the state evolution funcOp with a given list of operations.
        Note: It only considers the values that are defined in the operations and required by
        the operations after the terminal operation.
        """
        ops_walk = list(chain(*[op.walk() for op in ops]))

        ops_defined_values = set()

        for op_walk in ops_walk:
            ops_defined_values.update(op_walk.results)

        # use list here to maintain the order of required outputs
        required_outputs = []
        found_terminal = False
        for op in self.original_func_op.body.walk():
            # branch for the operations before the terminal_op
            if op == terminal_op:
                found_terminal = True
                continue

            # branch for the operations after the terminal_op
            if found_terminal:
                for operand in op.operands:
                    # a required output is an operand defined by a result of op in the ops
                    if operand in ops_defined_values and operand not in required_outputs:
                        required_outputs.append(operand)

        return required_outputs

    def _add_return_statement(self, target_block, required_outputs, value_mapper):
        """add a func.return op to function"""
        return_values = []
        for output_val in required_outputs:
            if output_val not in value_mapper:
                raise ValueError(f"output_val {output_val} not in value_mapper")
            return_values.append(value_mapper[output_val])

        return_op = func.ReturnOp(*return_values)
        target_block.add_op(return_op)

    def _clone_operations_to_block(self, ops_to_clone, target_block, value_mapper):
        """Clone operations to target block, use value_mapper to update references"""
        for op in ops_to_clone:
            cloned_op = op.clone(value_mapper)
            target_block.add_op(cloned_op)

    def _finalize_transformation(self):
        """Replace the original function with a call to the state evolution function."""
        original_block = self.original_func_op.body.block
        ops_list = list(original_block.ops)

        begin_idx = None
        end_idx = None
        for i, op in enumerate(ops_list):
            if op == self.alloc_op:
                begin_idx = i + 1
            elif op == self.terminal_boundary_op:
                end_idx = i + 1
                break

        if begin_idx is None:
            raise RuntimeError("A quantum.alloc operation is not found in original function.")
        if end_idx is None:
            raise RuntimeError(
                "A terminal_boundary_op operation is not found in original function."
            )
        if begin_idx > end_idx:
            raise RuntimeError(
                "A quantum.alloc operation should come before the terminal_boundary_op."
            )

        post_ops = ops_list[end_idx:]

        call_args = list(self.required_inputs)
        result_types = [val.type for val in self.required_outputs]

        call_op = func.CallOp(self.state_evolution_func.sym_name.data, call_args, result_types)

        # TODO: I just removed all ops and add them again to update with value_mapper.
        # It's not efficient, just because it's easy to implement. Should using replace use method
        # instead.
        # De-attach all ops of the original function
        call_result_mapper = {}
        for i, required_output in enumerate(self.required_outputs):
            call_result_mapper[required_output] = call_op.results[i]

        value_mapper = call_result_mapper.copy()
        original_block.add_op(call_op)
        for op in post_ops:
            cloned_op = op.clone(value_mapper)
            original_block.add_op(cloned_op)

        # replace ops_list with call_op
        for op in chain(reversed(post_ops), reversed(ops_list[begin_idx:end_idx])):
            op.detach()
            op.erase()
