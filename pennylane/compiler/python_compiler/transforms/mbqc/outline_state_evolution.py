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

"""This file contains the implementation of the outline_state_evolution transform."""

from dataclasses import dataclass
from itertools import chain
from typing import Type, TypeVar

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func
from xdsl.ir import Operation, SSAValue
from xdsl.rewriter import InsertPoint

from pennylane.compiler.python_compiler import compiler_transform
from pennylane.compiler.python_compiler.dialects import quantum


@dataclass(frozen=True)
class OutlineStateEvolutionPass(passes.ModulePass):
    """Pass that puts gate operations into a private outline_state_evolution callable."""

    name = "outline-state-evolution"

    # pylint: disable=no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the outline-state-evolution pass."""
        qnode = None
        for op in module.ops:
            if isinstance(op, func.FuncOp) and "qnode" in op.attributes:
                qnode = op
                break
        if qnode is None:
            raise RuntimeError("There is no funcOp with qnode attribute in the module")
        rewriter = pattern_rewriter.PatternRewriter(module)
        OutlineStateEvolutionPattern().match_and_rewrite(qnode, rewriter)


outline_state_evolution_pass = compiler_transform(OutlineStateEvolutionPass)


class OutlineStateEvolutionPattern(pattern_rewriter.RewritePattern):
    """RewritePattern for outlined state evolution regions in a quantum function."""

    def _get_parent_module(self, op: Operation) -> builtin.ModuleOp:
        """Walk up the parent tree until a builtin.ModuleOp op is found."""
        while (op := op.parent_op()) and not isinstance(op, builtin.ModuleOp):
            pass
        if op is None:
            raise RuntimeError("Got orphaned qnode function")
        return op

    def __init__(self):
        self.module: builtin.ModuleOp = None
        self.original_func_op: func.FuncOp = None

        # To determine the boundary of quantum gate operations in the IR
        self.alloc_op: quantum.AllocOp = None
        self.terminal_boundary_op: Operation = None

        # Input and outputs of the state evolution func
        self.missing_inputs: list[SSAValue] = None
        self.required_outputs: list[SSAValue] = None

        # State evolution function region
        self.state_evolution_func: func.FuncOp = None

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
        self._finalize_transformation(rewriter)

    def _get_extract_idx(self, op: Operation) -> int | None:
        """Get the extract index from an ExtractOp op."""
        if hasattr(op, "idx") and op.idx:
            return op.idx
        elif hasattr(op, "idx_attr"):
            return op.idx_attr
        return None

    def _simplify_quantum_io(
        self, func_op: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter
    ) -> func.FuncOp:
        """Simplify quantum I/O to use only registers at segment boundaries.

        This ensures that state evolution regions only take registers as input/output,
        not individual qubits.
        """
        # Note that there are three operations in quantum dialects that return a qreg
        # 1, quantum.alloc; 2, quantum.insert; 3, quantum.adjoint. mbqc.graph_state_prep
        # also returns qreg but mbqc related operations should be inserted into the IR after
        # this pass.
        current_reg = None
        qubit_to_reg_idx = {}
        terminal_boundary_op = None

        for op in func_op.body.ops:
            match op:
                case quantum.AllocOp():
                    current_reg = op.qreg
                case quantum.ExtractOp():
                    # Update register mapping
                    extract_idx = self._get_extract_idx(op)
                    qubit_to_reg_idx[op.qubit] = extract_idx
                    op.operands = (current_reg, extract_idx)
                case quantum.MeasureOp():
                    # TODOs: what if the qubit that quantum.measure target at is reset?
                    qubit_to_reg_idx[op.out_qubit] = qubit_to_reg_idx[op.in_qubit]
                    del qubit_to_reg_idx[op.in_qubit]
                case quantum.CustomOp():
                    # To update the qubit_to_reg_idx map for the return type.
                    for i, qb in enumerate(chain(op.in_qubits, op.in_ctrl_qubits)):
                        qubit_to_reg_idx[op.results[i]] = qubit_to_reg_idx[qb]
                        del qubit_to_reg_idx[qb]
                case quantum.InsertOp():
                    assert qubit_to_reg_idx[op.qubit] is op.idx_attr if op.idx_attr else True
                    del qubit_to_reg_idx[op.qubit]
                    # update register since it might have changed
                    op.operands = (current_reg, op.idx, op.qubit)
                    current_reg = op.out_qreg

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
                    insert_ops = set()

                    # Insert all qubits recorded in the qubit_to_reg_idx dict before the
                    # pre-assumed terminal opeartions.
                    rewriter.insertion_point = InsertPoint.before(op)
                    for qb, idx in qubit_to_reg_idx.items():
                        insert_op = quantum.InsertOp(current_reg, idx, qb)
                        rewriter.insert(insert_op)
                        insert_ops.add(insert_op)
                        current_reg = insert_op.out_qreg

                    # Add the `"terminal_boundary"` attribute to the last newly added
                    # `quantum.insert` operation.
                    list(insert_ops)[-1].attributes["terminal_boundary"] = builtin.UnitAttr()
                    # Now a terminal boundary operation is created and terminal_boundary_op
                    # should be updated.
                    terminal_boundary_op = list(insert_ops)[-1]

                    # extract ops
                    rewriter.insertion_point = InsertPoint.before(op)
                    for qb, idx in list(qubit_to_reg_idx.items()):
                        extract_op = quantum.ExtractOp(current_reg, idx)
                        rewriter.insert(extract_op)
                        qb.replace_by_if(
                            extract_op.qubit, lambda use: use.operation not in insert_ops
                        )
                        # update the qubit_to_reg_idx dict
                        qubit_to_reg_idx[extract_op.qubit] = idx
                        del qubit_to_reg_idx[qb]

                case _:
                    # Handle other operations that might has qreg result
                    # Note that this branch might not be tested so far as adjoint op is not
                    # tested so far.
                    if reg := next(
                        (reg for reg in op.results if isinstance(reg.type, quantum.QuregType)), None
                    ):
                        current_reg = reg

    def _create_state_evolution_function(self, rewriter: pattern_rewriter.PatternRewriter):
        """Create a new function for the state evolution region using clone approach."""

        alloc_op, terminal_boundary_op = self._find_evolution_range()

        # collect operation from alloc_op to terminal_boundary_op
        ops_to_clone = self._collect_operations_in_range(alloc_op, terminal_boundary_op)

        # analyze missing values for ops
        missing_inputs = self._analyze_missing_values_for_ops(ops_to_clone)

        # analyze required outputs for ops
        required_outputs = self._analyze_required_outputs(ops_to_clone, terminal_boundary_op)

        register_inputs = []
        other_inputs = []
        for val in missing_inputs:
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

        state_evolution_func = func.FuncOp(
            self.original_func_op.sym_name.data + ".state_evolution", fun_type, visibility="private"
        )
        rewriter.insert_op(state_evolution_func, InsertPoint.at_end(self.module.body.block))

        block = state_evolution_func.regions[0].block
        value_mapper = {}
        for missing_val, block_arg in zip(ordered_inputs, block.args):
            value_mapper[missing_val] = block_arg

        self._clone_operations_to_block(ops_to_clone, block, value_mapper)
        self._add_return_statement(block, ordered_outputs, value_mapper)

        self.missing_inputs = ordered_inputs
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

        if not alloc_op or not terminal_boundary_op:
            raise RuntimeError("Could not find alloc_op or terminal_boundary_op")

        if alloc_op.parent_block() != terminal_boundary_op.parent_block():
            raise RuntimeError("alloc_op and terminal_boundary_op are not in the same block")

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

    def _analyze_missing_values_for_ops(self, ops: list[Operation]) -> list[SSAValue]:
        """Get missing values for ops. Given a list of operations, return the values
        that are missing from the operations.
        """
        ops_walk = list(chain(*[op.walk() for op in ops]))

        ops_defined_values = set()
        all_operands = set()

        for nested_op in ops_walk:
            ops_defined_values.update(nested_op.results)
            all_operands.update(nested_op.operands)

            if hasattr(nested_op, "regions") and nested_op.regions:
                for region in nested_op.regions:
                    for block in region.blocks:
                        ops_defined_values.update(block.args)

        missing_values = list(all_operands - ops_defined_values)
        missing_values = [v for v in missing_values if v is not None]

        return missing_values

    def _analyze_required_outputs(
        self, ops: list[Operation], terminal_op: Operation
    ) -> list[SSAValue]:
        """Get required outputs for ops.
        Given a list of operations and a terminal operation, return the values that are
        required by the operations after the terminal operation. Noted: It's only consdider
        the values that are defined in the operations and required by the operations after
        the terminal operation!
        """
        ops_walk = list(chain(*[op.walk() for op in ops]))

        ops_defined_values = set()

        for nested_op in ops_walk:
            ops_defined_values.update(nested_op.results)

        required_outputs = set()
        found_terminal = False
        for op in self.original_func_op.body.walk():
            if op == terminal_op:
                found_terminal = True
                continue

            if found_terminal:
                for operand in op.operands:
                    if operand in ops_defined_values:
                        required_outputs.add(operand)

        return list(required_outputs)

    def _add_return_statement(self, target_block, required_outputs, value_mapper):
        """add return statement to function"""
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

            self._update_value_mapper_recursively(op, cloned_op, value_mapper)

    def _update_value_mapper_recursively(self, orig_op, cloned_op, value_mapper):
        """update value_mapper for all operations in operation"""
        for orig_result, new_result in zip(orig_op.results, cloned_op.results):
            value_mapper[orig_result] = new_result

        for orig_region, cloned_region in zip(orig_op.regions, cloned_op.regions):
            self._update_region_value_mapper(orig_region, cloned_region, value_mapper)

    def _update_region_value_mapper(self, orig_region, cloned_region, value_mapper):
        """update value_mapper for all operations in region"""
        for orig_block, cloned_block in zip(orig_region.blocks, cloned_region.blocks):
            for orig_arg, cloned_arg in zip(orig_block.args, cloned_block.args):
                value_mapper[orig_arg] = cloned_arg

            for orig_nested_op, cloned_nested_op in zip(orig_block.ops, cloned_block.ops):
                self._update_value_mapper_recursively(
                    orig_nested_op, cloned_nested_op, value_mapper
                )

    def _finalize_transformation(self, rewriter: pattern_rewriter.PatternRewriter):
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

        assert begin_idx is not None, "alloc_op not found in original function"
        assert end_idx is not None, "terminal_boundary_op not found in original function"
        assert begin_idx <= end_idx, "alloc_op should come before terminal_boundary_op"

        pre_ops = ops_list[:begin_idx]
        post_ops = ops_list[end_idx:]

        call_args = list(self.missing_inputs)
        result_types = [val.type for val in self.required_outputs]

        call_op = func.CallOp(self.state_evolution_func.sym_name.data, call_args, result_types)

        # call_result_mapper is required for the clone operation. See more [here](https://github.com/xdslproject/xdsl/blob/e1301e0204bcf6ea5ed433e7da00bee57d07e695/xdsl/ir/core.py#L1429)_.
        call_result_mapper = {}
        for i, required_output in enumerate(self.required_outputs):
            if i < len(call_op.results):
                call_result_mapper[required_output] = call_op.results[i]

        # TODO: I just removed all ops and add them again to update with value_mapper.
        # It's not efficient, just because it's easy to implement. Should using replace use method
        # instead.
        # Deattach all ops of the orginal function
        for op in reversed(ops_list):
            op.detach()

        # Add a list of ops before the quantum.alloc ops in the original function
        new_ops = []
        for op in pre_ops:
            new_ops.append(op)

        # Add a callOp to the box
        new_ops.append(call_op)

        value_mapper = call_result_mapper.copy()

        for i, op in enumerate(post_ops):
            cloned_op = op.clone(value_mapper)

            for orig_result, new_result in zip(op.results, cloned_op.results):
                value_mapper[orig_result] = new_result

            new_ops.append(cloned_op)

        for op in new_ops:
            original_block.add_op(op)
