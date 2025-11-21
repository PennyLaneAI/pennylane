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
# pylint: disable=no-member  # False positives with xDSL region.block access

"""Implementation of the Tree-Traversal MCM simulation method as an xDSL transform in Catalyst."""

from dataclasses import dataclass, field
from itertools import chain
from typing import List, Tuple, Type, TypeVar

from xdsl import context
from xdsl.dialects import arith, builtin, func, memref, scf, tensor
from xdsl.ir import Block, BlockArgument, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import BlockInsertPoint, InsertPoint

from pennylane.compiler.python_compiler import compiler_transform
from pennylane.compiler.python_compiler.dialects import quantum, stablehlo


##############################################################################
# Some useful utils
##############################################################################
def initialize_memref_with_value(dest: SSAValue, value: SSAValue, size: int | SSAValue):
    """Initialize a memref with value"""
    # lb
    c0_index = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())

    # step
    c1_index = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())

    loop_body = Region()
    loop_block = Block(arg_types=[builtin.IndexType()])
    loop_body.add_block(loop_block)

    store_op = memref.StoreOp.get(value, dest, (loop_block.args[0],))
    yield_op = scf.YieldOp()

    for op in [store_op, yield_op]:
        loop_block.add_op(op)

    # Create the for loop
    for_op = scf.ForOp(
        lb=c0_index.results[0], ub=size, step=c1_index.results[0], iter_args=[], body=loop_body
    )
    return (c0_index, c1_index, for_op)

@dataclass
class ProgramSegment:  # pylint: disable=too-many-instance-attributes
    """A program segment and associated data."""

    ops: list[Operation] = field(default_factory=list)
    mcm: quantum.MeasureOp = None
    # This is the assumption that only one register is used in the segment
    reg_in: SSAValue = None
    reg_out: SSAValue = None
    inputs: set[SSAValue] = None
    outputs: set[SSAValue] = None
    fun: func.FuncOp = None
    depth: int = 0


@dataclass(frozen=True)
class TreeTraversalPass(ModulePass):
    """Pass that transforms a quantum function (qnode) to perform tree-traversal simulation."""

    name = "tree-traversal"

    def apply(self, _ctx: context.Context, module_op: builtin.ModuleOp) -> None:
        """Apply the tree-traversal pass to all QNode functions in the module."""

        for op in module_op.ops:
            if isinstance(op, func.FuncOp) and "qnode" in op.attributes:
                rewriter = PatternRewriter(op)

                unroll_pattern = UnrollLoopPattern()
                unroll_pattern.match_and_rewrite(op, rewriter)

                IfOperatorPartitioningPass().match_and_rewrite(op, rewriter)


                TreeTraversalPattern().match_and_rewrite(op, rewriter)


tree_traversal_pass = compiler_transform(TreeTraversalPass)


# pylint: disable=too-many-instance-attributes
class TreeTraversalPattern(RewritePattern):
    """Tree-Traversal MCM simulation method as an xDSL transform in Catalyst."""

    T = TypeVar("T")

    def get_parent_of_type(self, op: Operation, kind: Type[T]) -> T | None:
        """Walk up the parent tree until an op of the specified type is found."""
        while (op := op.parent_op()) and not isinstance(op, kind):
            pass
        return op

    def __init__(self):
        self.module: builtin.ModuleOp = None
        self.quantum_segments: list[ProgramSegment] = []

        # the original function op
        self.original_func_op: func.FuncOp = None

        # the simple io function, which should be removed after the transformation
        self.simple_io_func: func.FuncOp = None

        # the main traversal function entry
        self.tt_op: func.FuncOp = None

        # alloc op
        self.alloc_op: quantum.AllocOp = None

        # Attributes that may be defined later
        self.terminal_segment: ProgramSegment = None
        self.state_transition_func: func.FuncOp = None
        self.all_segment_io: list = None
        self.values_as_io_index: dict = None
        self.tree_depth: SSAValue = None
        self.statevec_size: SSAValue = None
        self.statevec_stack: SSAValue = None
        self.probs_stack: SSAValue = None
        self.visited_stack: SSAValue = None
        self.folded_result: SSAValue = None
        self.traversal_op: Operation = None

        # Type infos:
        self.probs_stack_type = builtin.MemRefType(builtin.f64, (builtin.DYNAMIC_INDEX,))
        self.visited_stack_type = builtin.MemRefType(builtin.i8, (builtin.DYNAMIC_INDEX,))
        self.statevec_stack_type = builtin.MemRefType(
            builtin.ComplexType(builtin.f64),
            [builtin.DYNAMIC_INDEX, builtin.DYNAMIC_INDEX],  # [depth, 2^n]
        )
        self.folded_result_type = builtin.MemRefType(builtin.f64, (builtin.DYNAMIC_INDEX,))

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, func_op: func.FuncOp, rewriter: PatternRewriter
    ):  # pylint: disable=arguments-differ
        """Transform a quantum function (qnode) to perform tree-traversal simulation."""
        self.original_func_op = func_op

        if "qnode" not in func_op.attributes:
            return

        self.module = self.get_parent_of_type(func_op, builtin.ModuleOp)
        assert self.module is not None, "got orphaned qnode function"

        # If no measurements found, no need to apply tree-traversal
        if not self.check_if_qnode_have_mcm(func_op):
            return

        # Start with creating a new QNode function that will perform the tree traversal simulation.
        # We prep the original QNode by ensuring measure boundaries are also register boundaries.
        self.simple_io_func = self.simplify_quantum_io(func_op, rewriter)

        self.setup_traversal_function(self.simple_io_func, rewriter)

        self.split_traversal_segments(self.simple_io_func, rewriter)

        self.initialize_data_structures(rewriter)

        self.generate_traversal_code(rewriter)

        self.finalize_traversal_function(rewriter)

    def get_idx(self, op: Operation) -> int | None:
        """Get the index of the operation."""
        return op.idx if op.idx else op.idx_attr

    def check_if_qnode_have_mcm(self, func_op: func.FuncOp) -> bool:
        """Check if the QNode function contains any measurement operations."""
        op_walker = func_op.body.walk()
        for op in op_walker:
            if isinstance(op, quantum.MeasureOp):
                return True
        return False

    def simplify_quantum_io(self, func_op: func.FuncOp, rewriter: PatternRewriter) -> func.FuncOp:
        """In order to facilitate quantum value handling, we will reinsert all extracted qubits
        into the register at the end of each segment, and only allow the register as quantum
        input and output of segments.

        This pass guarantees that each measure op is preceded by exactly 1 ExtractOp, and whose
        input is a fully "reassembled" register ready to be passed across difficult program
        boundaries (e.g. control flow, function calls).
        """
        cloned_fun = func_op.clone()
        cloned_fun.sym_name = builtin.StringAttr(func_op.sym_name.data + ".simple_io")
        rewriter.insert_op(cloned_fun, InsertPoint.after(func_op))

        current_reg = None
        qubit_to_reg_idx = {}

        for op in cloned_fun.body.ops:
            match op:
                case quantum.AllocOp():
                    current_reg = op.qreg
                case quantum.ExtractOp():
                    # update register since it might have changed
                    extract_idx = self.get_idx(op)
                    qubit_to_reg_idx[op.qubit] = extract_idx
                    op.operands = (current_reg, extract_idx)
                case quantum.CustomOp():
                    for i, qb in enumerate(chain(op.in_qubits, op.in_ctrl_qubits)):
                        qubit_to_reg_idx[op.out_qubits[i]] = i
                        qubit_to_reg_idx[op.results[i]] = qubit_to_reg_idx[qb]
                        del qubit_to_reg_idx[qb]
                case quantum.InsertOp():
                    assert qubit_to_reg_idx[op.qubit] is op.idx_attr if op.idx_attr else True
                    del qubit_to_reg_idx[op.qubit]
                    # update register since it might have changed
                    op.operands = (current_reg, op.idx, op.qubit)
                    current_reg = op.out_qreg
                case quantum.MeasureOp():
                    # find the qubit to be measured and its index
                    mcm_qubit, mcm_idx = next(
                        ((qb, idx) for qb, idx in qubit_to_reg_idx.items() if qb == op.in_qubit),
                        (None, None),
                    )
                    if mcm_qubit is None:
                        raise RuntimeError(
                            f"Could not find qubit {op.in_qubit} in register mapping"
                        )

                    insert_ops = set()

                    # create a register boundary before the measure
                    rewriter.insertion_point = InsertPoint.before(op)
                    for qb, idx in qubit_to_reg_idx.items():
                        insert_op = quantum.InsertOp(current_reg, idx, qb)
                        rewriter.insert(insert_op)
                        insert_ops.add(insert_op)
                        current_reg = insert_op.out_qreg

                    # extract the qubit that will be measured from the register
                    extract_op = quantum.ExtractOp(current_reg, mcm_idx)
                    extract_op.attributes["meas_boundary"] = builtin.UnitAttr()
                    rewriter.insert(extract_op)
                    # update the measure operation to use the extracted qubit
                    op.operands = (extract_op.qubit,)

                    # update the map to process the measure qubit - remove the old qubit
                    del qubit_to_reg_idx[mcm_qubit]

                    # restore qubit values from before the register boundary
                    rewriter.insertion_point = InsertPoint.after(op)

                    def _create_replace_filter(excluded_operations):
                        """Create a filter function to avoid cell-var-from-loop issues."""
                        return lambda use: use.operation not in excluded_operations
                    for qb, idx in list(qubit_to_reg_idx.items()):
                        extract_op = quantum.ExtractOp(current_reg, idx)
                        rewriter.insert(extract_op)
                        qb.replace_by_if(
                            extract_op.qubit, _create_replace_filter(insert_ops)
                        )
                        qubit_to_reg_idx[extract_op.qubit] = idx
                        del qubit_to_reg_idx[qb]

                    qubit_to_reg_idx[op.out_qubit] = mcm_idx
                    # break
                case _:
                    # Handle other operations that might has qreg result
                    # Note that this branch might not be tested so far as adjoint op is not
                    # tested so far.
                    if reg := next(
                        (reg for reg in op.results if isinstance(reg.type, quantum.QuregType)), None
                    ):
                        current_reg = reg

        return cloned_fun

    def setup_traversal_function(self, func_op: func.FuncOp, rewriter: PatternRewriter):
        """Setup a clone of the original QNode function, which will instead perform TT."""

        tt_op = func_op.clone_without_regions()
        tt_op.sym_name = builtin.StringAttr(func_op.sym_name.data + ".tree_traversal")
        rewriter.create_block(BlockInsertPoint.at_start(tt_op.body), tt_op.function_type.inputs)
        rewriter.insert_op(tt_op, InsertPoint.at_end(self.module.body.block))

        self.tt_op = tt_op

    def finalize_traversal_function(self, rewriter: PatternRewriter):
        """Complete the function and ensure it's correctly formed, e.g. returning proper results."""
        rewriter.insertion_point = InsertPoint.at_end(self.tt_op.body.block)

        while_op = self.traversal_op
        assert while_op is not None, "Could not find while loop in traversal function"

        # Clone and insert terminal operations, using the final qreg from while loop
        final_qreg = while_op.results[1]

        value_mapper = {}

        if hasattr(self, "terminal_segment") and self.terminal_segment.ops:
            for op in self.terminal_segment.ops:
                if isinstance(op, func.ReturnOp):
                    # TODO: For return operation, we need to get the actual result from the
                    # traversal.
                    c0 = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
                    load_op = memref.LoadOp.get(self.folded_result, (c0,))
                    tensor_op = tensor.FromElementsOp.create(
                        operands=[load_op.results[0]],
                        result_types=[builtin.TensorType(builtin.Float64Type(), [])],
                    )
                    return_op = func.ReturnOp(*tensor_op.results)
                    for insert_op in (c0, load_op, tensor_op, return_op):
                        rewriter.insert_op(insert_op, InsertPoint.at_end(self.tt_op.body.block))

                elif isinstance(op, quantum.DeallocOp):
                    # Use the final qreg from while loop for dealloc
                    dealloc_op = op.clone(value_mapper)
                    dealloc_op.operands = (final_qreg,)
                    rewriter.insert(dealloc_op)
                else:
                    cloned_op = op.clone(value_mapper)
                    rewriter.insert(cloned_op)
        else:
            result_vals = []
            for resType in self.tt_op.function_type.outputs:
                assert isinstance(resType, builtin.TensorType)
                result = tensor.EmptyOp((), tensor_type=resType)
                result_vals.append(rewriter.insert(result))
            rewriter.insert(func.ReturnOp(*result_vals))

        for op in reversed(self.original_func_op.body.ops):
            op.detach()
            op.erase()

        call_op = func.CallOp(
            self.tt_op.sym_name.data,
            self.original_func_op.args,
            [builtin.TensorType(builtin.Float64Type(), [])],  # Single tensor<f64> output
        )

        return_op = func.ReturnOp(*call_op.results)

        for op in (call_op, return_op):
            rewriter.insert_op(op, InsertPoint.at_end(self.original_func_op.body.block))

        # remove simple_io function
        self.simple_io_func.detach()
        self.simple_io_func.erase()

    def split_traversal_segments(self, func_op: func.FuncOp, rewriter: PatternRewriter):
        """Split the quantum function into segments separated by measure operations.

        Due to the pre-processing of the QNode (result in simple_io_func), we can assume the register is the only
        quantum value going between segments.
        """
        rewriter.insertion_point = InsertPoint.at_start(self.tt_op.body.block)

        # Ideally try to iterate over the function only once.
        op_iter = iter(func_op.body.ops)

        # Skip to the start of the first simulation segment.
        value_mapper = {}
        while (op := next(op_iter, None)) and not isinstance(op, quantum.AllocOp):
            rewriter.insert(op.clone(value_mapper))
        assert op is not None, "didn't find an alloc op"

        # clone the alloc op
        self.alloc_op = rewriter.insert(op.clone(value_mapper))

        # Split ops into segments divided by measurements.
        quantum_segments = [ProgramSegment(reg_in=op.qreg)]
        while (op := next(op_iter, None)) and not isinstance(op, quantum.DeallocOp):
            if hasattr(op, "attributes") and "meas_boundary" in op.attributes:
                del op.attributes["meas_boundary"]
                last_op = quantum_segments[-1].ops[-1]
                assert isinstance(last_op, quantum.InsertOp)
                quantum_segments[-1].reg_out = last_op.out_qreg
                quantum_segments.append(ProgramSegment(reg_in=last_op.out_qreg))
                quantum_segments[-1].depth = quantum_segments[-2].depth + 1
            elif isinstance(op, quantum.MeasureOp):
                quantum_segments[-1].mcm = op
            quantum_segments[-1].ops.append(op)

        assert op is not None, "didn't find a dealloc op"
        quantum_segments[-1].reg_out = op.qreg
        self.quantum_segments = quantum_segments

        # Go through the rest of the function to initialize the missing input values set.
        terminal_segment = ProgramSegment(ops=[op])  # dealloc op
        while op := next(op_iter, None):
            terminal_segment.ops.append(op)

        # TODO:
        # Copy terminal segment operations into the traversal function
        # These operations (dealloc, device_release, return) need to be executed
        # at the end of the tree traversal simulation
        self.terminal_segment = terminal_segment

        # Generate new functions for each segment separated by a measure op.
        # We traverse them bottom up first to correctly determine the I/O of each segment.
        missing_input_values = set()
        self.populate_segment_io(terminal_segment, missing_input_values)
        for segment in reversed(quantum_segments):
            self.populate_segment_io(segment, missing_input_values)  # inplace

        # Create the traversal handling function
        # It is control the state transition of the tree traversal
        self.state_transition_func = self.create_state_transition_function(rewriter)

        # contains the inputs to the first segment + all MCM results
        func_args = set(func_op.args)
        missing_input_values = missing_input_values - func_args
        all_segment_io = [quantum_segments[0].reg_in, *func_args, *missing_input_values]
        values_as_io_index = {v: k for k, v in enumerate(all_segment_io)}
        for idx, segment in enumerate(quantum_segments):
            new_segment = self.clone_ops_into_func(segment, idx, rewriter)
            new_segment = self.additional_transform_for_segment(new_segment, segment, rewriter)
            segment.fun = new_segment
            values_as_io_index.update(
                (x, i + len(all_segment_io)) for i, x in enumerate(segment.outputs)
            )
            all_segment_io.extend(segment.outputs)

        self.generate_function_table(all_segment_io, values_as_io_index, rewriter)

        # store some useful values for later
        self.all_segment_io = all_segment_io
        self.values_as_io_index = values_as_io_index

    def additional_transform_for_segment(  # pylint: disable=too-many-branches
        self, new_segment: func.FuncOp, segment: ProgramSegment, rewriter: PatternRewriter
    ) -> func.FuncOp:
        """Apply additional transformations to each segment function as needed."""

        real_mcm_op = None
        current_if_op = None

        # Extract the IfOps containing measure operations
        op_walker = new_segment.body.walk()
        for op in op_walker:
            if isinstance(op, scf.IfOp):
                if real_mcm_op is None:
                    real_mcm_op = op
                    continue
                contain_mcm = "contain_mcm" in op.attributes
                if contain_mcm:
                    current_if_op = op

        if (
            current_if_op is not None
            and real_mcm_op is not None
            and current_if_op is not real_mcm_op
        ):
            where_to_move_real_mcm = None

            # Find the measure operation inside the true branch of the inner IfOp
            for inner_op in current_if_op.true_region.ops:
                if isinstance(inner_op, quantum.MeasureOp):
                    where_to_move_real_mcm = inner_op
                    break

            if where_to_move_real_mcm is None:
                where_to_move_real_mcm = current_if_op.true_region.ops.first

            # Update the qreg input in the inner IfOp
            q_bit_mcm = real_mcm_op.results[1]
            real_mcm_op_mcm_q_bit = real_mcm_op.true_region.ops.first

            q_bit_mcm.replace_by_if(
                real_mcm_op_mcm_q_bit.operands[0], lambda use: use.operation is not real_mcm_op
            )

            # Move the real measure IfOp before the mcm inside the inner IfOp
            real_mcm_op.detach()

            if isinstance(where_to_move_real_mcm, quantum.MeasureOp):
                rewriter.insert_op(real_mcm_op, InsertPoint.before(where_to_move_real_mcm))
                for new_mcm_use, old_mcm_use in zip(
                    real_mcm_op.results, where_to_move_real_mcm.results
                ):
                    old_mcm_use.replace_by_if(
                        new_mcm_use, lambda use: use.operation is not where_to_move_real_mcm
                    )

                old_mcm_operands = where_to_move_real_mcm.operands[0]
                real_mcm_op_walks = real_mcm_op.walk()
                for op in real_mcm_op_walks:
                    if isinstance(op, quantum.MeasureOp):
                        op.operands = (old_mcm_operands,)
                        rewriter.notify_op_modified(op)
                where_to_move_real_mcm.detach()
                where_to_move_real_mcm.erase()
            else:
                real_mcm_op.erase()


        return new_segment

    def generate_function_table(  # pylint: disable=no-member
        self,
        all_segment_io: list[SSAValue],
        values_as_io_index: dict[SSAValue, int],
        rewriter: PatternRewriter,
    ):
        """Create a program segment dispatcher via a large function table switch statement.

        The dispatcher needs the entire segment IO as arguments/results in order to properly
        wire the input & output of each segment together, since the dispatcher is invoked
        dynamically but call arguments & results are static SSA values. An alternative would be
        to pass around segment IO via memory, but this requires additional IR operations & types
        not available in builtin dialects.
        """

        # function op
        all_io_types = [val.type for val in all_segment_io]
        fun_type = builtin.FunctionType.from_lists(
            [
                builtin.IndexType(),  # function id is the depth of the tree, index for segment function
                builtin.IndexType(),  # branch number, 0, 1, or 2
                self.visited_stack_type,  # visited stack
                self.statevec_stack_type,  # state vector
                self.probs_stack_type,  # probs stack
                self.folded_result_type,  # folded result
                builtin.IndexType(),  # statevec size
                *all_io_types,
            ],
            all_io_types,
        )
        funcTableOp = func.FuncOp("segment_table", fun_type)
        rewriter.insert_op(funcTableOp, InsertPoint.at_end(self.module.body.block))

        # function body
        fun_index = funcTableOp.args[0]
        branch_type = funcTableOp.args[1]
        visited_stack = funcTableOp.args[2]
        statevec_stack = funcTableOp.args[3]
        probs_stack = funcTableOp.args[4]
        folded_result = funcTableOp.args[5]
        statevec_size = funcTableOp.args[6]
        io_args = funcTableOp.args[7:]

        cases = builtin.DenseArrayBase.from_list(builtin.i64, range(len(self.quantum_segments)))
        switchOp = scf.IndexSwitchOp(
            fun_index,
            cases,
            Region(Block()),
            [Region(Block()) for _ in range(len(cases))],
            all_io_types,
        )
        returnOp = func.ReturnOp(*switchOp.results)

        for op in (switchOp, returnOp):
            rewriter.insert_op(op, InsertPoint.at_end(funcTableOp.body.block))

        # switch op base case
        rewriter.insert_op(scf.YieldOp(*io_args), InsertPoint.at_end(switchOp.default_region.block))

        # switch op match cases
        for case, segment in enumerate(self.quantum_segments):
            args = (
                [branch_type]  # branch_type for postselect logic
                + [visited_stack]  # visited_stack for visited status
                + [statevec_stack]  # statevector stack for storing and restoring statevector
                + [probs_stack]  # probs stack for recording the probability before each segment
                + [folded_result]  # folded result
                + [statevec_size]  # statevec size for storing and restoring statevector
                + [io_args[0]]  # quantum register
                + [io_args[values_as_io_index[value]] for value in segment.inputs]
            )
            res_types = [quantum.QuregType()] + [res.type for res in segment.outputs]
            callOp = func.CallOp(self.quantum_segments[case].fun.sym_name.data, args, res_types)

            updated_results = list(io_args)
            updated_results[0] = callOp.results[0]
            for new_res, ref in zip(callOp.results[1:], segment.outputs):
                updated_results[values_as_io_index[ref]] = new_res

            yieldOp = scf.YieldOp(*updated_results)

            for op in (callOp, yieldOp):
                rewriter.insert_op(op, InsertPoint.at_end(switchOp.case_regions[case].block))

    def clone_ops_into_func(self, segment: ProgramSegment, counter: int, rewriter: PatternRewriter):  # pylint: disable=no-member
        """Clone a set of ops into a new function."""
        op_list, input_vals, output_vals = segment.ops, segment.inputs, segment.outputs
        input_vals = [segment.reg_in] + input_vals
        output_vals = [segment.reg_out] + output_vals
        if not op_list:
            return None

        fun_type = builtin.FunctionType.from_lists(
            [builtin.IndexType()]  #
            + [self.visited_stack_type]
            + [self.statevec_stack_type]
            + [self.probs_stack_type]
            + [self.folded_result_type]
            + [builtin.IndexType()]
            + [arg.type for arg in input_vals],
            [res.type for res in output_vals],
        )
        new_func = func.FuncOp(f"quantum_segment_{counter}", fun_type)

        # branch_type is args[0], actual inputs start from args[5]
        branch_type = new_func.args[0]
        visited_stack = new_func.args[1]
        statevec_stack = new_func.args[2]
        probs_stack = new_func.args[3]
        folded_result = new_func.args[4]
        statevec_size = new_func.args[5]
        value_mapper = dict(zip(input_vals, new_func.args[6:]))

        if counter == 0:
            # visited[depth] = 2
            # Convert depth integer to SSAValue
            depth_ssa = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
            c2_visited = arith.ConstantOp.from_int_and_width(2, builtin.i8)
            store_op = memref.StoreOp.get(c2_visited, visited_stack, (depth_ssa,))
            for op in (c2_visited, depth_ssa, store_op):
                rewriter.insert_op(op, InsertPoint.at_end(new_func.body.block))

        measurement_ops = []
        for op in op_list:
            if isinstance(op, quantum.MeasureOp):
                self.clone_measure_op_with_postselect(
                    op,
                    segment.depth,
                    branch_type,
                    visited_stack,
                    statevec_stack,
                    probs_stack,
                    statevec_size,
                    value_mapper,
                    new_func,
                    rewriter,
                )
            else:
                new_op = op.clone(value_mapper)
                if isinstance(new_op, quantum.ExpvalOp):
                    measurement_ops.append(new_op)
                rewriter.insert_op(new_op, InsertPoint.at_end(new_func.body.block))

        assert len(measurement_ops) <= 1, "expected at most one measurement op in the segment"
        if measurement_ops:
            for measurement_op in measurement_ops:
                # measurement_op = measurement_ops[0]
                # folded_result[counter + 1] += measurement_op.results[0] * probs_stack[counter + 1]
                c1 = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
                depth_ssa = arith.ConstantOp.from_int_and_width(counter, builtin.IndexType())
                folded_result_idx = arith.AddiOp(depth_ssa, c1)

                current_folded_op = memref.LoadOp.get(folded_result, (folded_result_idx,))
                current_prob_op = memref.LoadOp.get(probs_stack, (folded_result_idx,))
                measurement_value = measurement_op.results[0]
                mul_op = arith.MulfOp(measurement_value, current_prob_op.results[0])
                new_folded_op = arith.AddfOp(current_folded_op.results[0], mul_op.results[0])
                store_folded = memref.StoreOp.get(
                    new_folded_op.results[0], folded_result, (folded_result_idx,)
                )

                for op in [
                    c1,
                    depth_ssa,
                    folded_result_idx,
                    current_folded_op,
                    current_prob_op,
                    mul_op,
                    new_folded_op,
                    store_folded,
                ]:
                    rewriter.insert_op(op, InsertPoint.at_end(new_func.body.block))

        return_op = func.ReturnOp(*(value_mapper[res] for res in output_vals))
        rewriter.insert_op(return_op, InsertPoint.at_end(new_func.body.block))

        # inser the new segment function into the module
        rewriter.insert_op(new_func, InsertPoint.at_end(self.module.body.block))
        return new_func

    def create_state_transition_function(self, rewriter: PatternRewriter):  # pylint: disable=too-many-statements,no-member
        """The created function is used to handle the state transition of the tree traversal.
        [Done] update the visited stack
        [Done] store the state from caling quantum.state
        [Done] restore the state from calling quantum.set_state
        [TODO] store the prob to the probs stack

        The behaviour of the generated IR as below:
        ```python
        def scfIf(qubit, depth, branch) -> qreg, postselect, visited_status:
            if branch == 0:
                prob = quantum.probs(qubit)
                if prob[0] == 1.0:
                    return qreg, 0, 2
                elif prob[0] == 0.0:
                    return qreg, 1, 2
                else:
                    state = quantum.state
                    statevec_stack[depth] = state
                    return qreg, 0, 1
            else:
                state = statevec_stack[depth]
                updated_qreg = quantum.set_state(state)
                return updated_qreg, 1, 2

        qreg, postelect, visited_state = scfIf(qubit, depth, branch)

        # updat the visited stack
        visited[depth] = visited_state
        return qreg, postelect
        ```
        """

        func_type = builtin.FunctionType.from_lists(
            [
                quantum.QubitType(),
                quantum.QuregType(),  # qreg
                builtin.IndexType(),  # depth
                builtin.IndexType(),  # branch_type
                self.visited_stack_type,
                self.statevec_stack_type,
                self.probs_stack_type,
                builtin.IndexType(),
            ],  # statevec_size
            [quantum.QuregType(), builtin.i8],  # updated_qreg, postselect
        )

        new_func = func.FuncOp("state_transition", func_type)
        rewriter.insert_op(new_func, InsertPoint.at_end(self.module.body.block))

        # Get function arguments
        qubit = new_func.args[0]
        qreg = new_func.args[1]
        depth = new_func.args[2]
        branch_type = new_func.args[3]
        visited_stack = new_func.args[4]
        statevec_stack = new_func.args[5]
        probs_stack = new_func.args[6]
        statevec_size = new_func.args[7]

        # Test
        # return qreg, 0
        # mark visited[depth] = 2
        # c2_visited = arith.ConstantOp.from_int_and_width(2, builtin.i8)
        # storeOp = memref.StoreOp.get(c2_visited, visited_stack, (depth,))
        # for op in (c2_visited, storeOp):
        #     rewriter.insert_op(op, InsertPoint.at_end(new_func.body.block))

        # c0_postselect = arith.ConstantOp.from_int_and_width(0, builtin.i8)
        # returnOp = func.ReturnOp(qreg, c0_postselect)
        # for op in (c0_postselect, returnOp):
        #     rewriter.insert_op(op, InsertPoint.at_end(new_func.body.block))
        # return new_func

        # Check if branch_type == 0
        c0_branch = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        is_left_branch = arith.CmpiOp(branch_type, c0_branch, "eq")

        # Create if-else structure
        if_op = scf.IfOp(
            is_left_branch,
            [
                quantum.QuregType(),
                builtin.i8,
                builtin.i8,
            ],  # updated_qreg, postselect, visited_status
            Region(Block()),
            Region(Block()),
        )

        true_block = if_op.true_region.block
        false_block = if_op.false_region.block

        # True branch (branch_type == 0): compute probabilities and determine postselect/visited_status
        # Create computational basis observable
        def create_prob_ops(qubit: SSAValue):
            comp_basis_op = quantum.ComputationalBasisOp(
                operands=[[qubit], []], result_types=[quantum.ObservableType()]
            )

            # Get probabilities
            probs_op = quantum.ProbsOp(
                operands=[comp_basis_op.results[0], [], []],
                result_types=[builtin.TensorType(builtin.Float64Type(), [2])],
            )

            # Extract probability values
            c0_idx = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
            c1_idx = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
            prob_0 = tensor.ExtractOp(
                probs_op.results[0], c0_idx, result_type=builtin.Float64Type()
            )
            prob_1 = tensor.ExtractOp(
                probs_op.results[0], c1_idx, result_type=builtin.Float64Type()
            )
            return (comp_basis_op, probs_op, c0_idx, c1_idx, prob_0, prob_1)

        prob_ops = create_prob_ops(qubit)
        prob_0, prob_1 = prob_ops[-2:]

        c0_0 = arith.ConstantOp(builtin.FloatAttr(0.0, builtin.Float64Type()))
        # Check if prob_0 == 0.0
        prob_0_eq_0 = arith.CmpfOp(prob_0.results[0], c0_0, "oeq")
        # Check if prob_1 == 0.0
        prob_1_eq_0 = arith.CmpfOp(prob_1.results[0], c0_0, "oeq")

        for op in (*prob_ops, c0_0, prob_0_eq_0, prob_1_eq_0):
            rewriter.insert_op(op, InsertPoint.at_end(true_block))

        # Nested if-else for purning segment based on probability
        inner_if = scf.IfOp(
            prob_0_eq_0,
            [quantum.QuregType(), builtin.i8, builtin.i8],
            Region(Block()),
            Region(Block()),
        )

        inner_true_block = inner_if.true_region.block
        inner_false_block = inner_if.false_region.block

        # Inner true: result[0] == 0.0 -> return qreg, 1, 2
        # saved probs[depth + 1] = prob[1]
        c1 = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
        probs_stack_idx = arith.AddiOp(depth, c1)
        store_op = memref.StoreOp.get(prob_1.results[0], probs_stack, (probs_stack_idx,))
        for op in (c1, probs_stack_idx, store_op):
            rewriter.insert_op(op, InsertPoint.at_end(inner_true_block))

        # yield qreg, 1, 2
        c1_postselect = arith.ConstantOp(builtin.IntegerAttr(1, builtin.i8))
        c2_visited = arith.ConstantOp(builtin.IntegerAttr(2, builtin.i8))
        inner_true_yield = scf.YieldOp(qreg, c1_postselect, c2_visited)

        for op in (c1_postselect, c2_visited, inner_true_yield):
            rewriter.insert_op(op, InsertPoint.at_end(inner_true_block))

        # Inner false: check if prob_1 == 0.0 (meaning prob_0 == 1.0, so postselect 1)
        inner_inner_if = scf.IfOp(
            prob_1_eq_0,
            [quantum.QuregType(), builtin.i8, builtin.i8],
            Region(Block()),
            Region(Block()),
        )

        inner_inner_true_block = inner_inner_if.true_region.block
        inner_inner_false_block = inner_inner_if.false_region.block

        # Inner inner true: result[1] == 0.0 -> return qreg, 0, 2
        # saved probs[depth + 1] = prob[0]
        c1 = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
        probs_stack_idx = arith.AddiOp(depth, c1)
        store_op = memref.StoreOp.get(prob_0.results[0], probs_stack, (probs_stack_idx,))
        for op in (c1, probs_stack_idx, store_op):
            rewriter.insert_op(op, InsertPoint.at_end(inner_inner_true_block))

        c0_postselect = arith.ConstantOp(builtin.IntegerAttr(0, builtin.i8))
        c2_visited_2 = arith.ConstantOp(builtin.IntegerAttr(2, builtin.i8))
        inner_inner_true_yield = scf.YieldOp(qreg, c0_postselect, c2_visited_2)
        for op in (c0_postselect, c2_visited_2, inner_inner_true_yield):
            rewriter.insert_op(op, InsertPoint.at_end(inner_inner_true_block))

        # Inner inner false: neither 1.0 nor 0.0 -> return 0, 1
        # saved probs[depth + 1] = prob[0]
        c1 = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
        probs_stack_idx = arith.AddiOp(depth, c1)
        store_op = memref.StoreOp.get(prob_0.results[0], probs_stack, (probs_stack_idx,))
        for op in (c1, probs_stack_idx, store_op):
            rewriter.insert_op(op, InsertPoint.at_end(inner_inner_false_block))

        self.handle_store_state(
            qreg, depth, statevec_stack, statevec_size, rewriter, inner_inner_false_block
        )

        c0_postselect_2 = arith.ConstantOp(builtin.IntegerAttr(0, builtin.i8))
        c1_visited = arith.ConstantOp(builtin.IntegerAttr(1, builtin.i8))
        inner_inner_false_yield = scf.YieldOp(qreg, c0_postselect_2, c1_visited)
        for op in (c0_postselect_2, c1_visited, inner_inner_false_yield):
            rewriter.insert_op(op, InsertPoint.at_end(inner_inner_false_block))

        # Insert inner inner if into inner false block
        inner_false_yield = scf.YieldOp(*inner_inner_if.results)
        for op in (inner_inner_if, inner_false_yield):
            rewriter.insert_op(op, InsertPoint.at_end(inner_false_block))

        # Insert inner if into true block
        true_yield = scf.YieldOp(*inner_if.results)
        for op in (inner_if, true_yield):
            rewriter.insert_op(op, InsertPoint.at_end(true_block))

        # False branch (branch_type != 0): restore state and return 1, 2
        # saved probs[depth + 1] = 1 - saved_probs[depth + 1]
        c1 = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
        probs_stack_idx = arith.AddiOp(depth, c1)
        c1_f64 = arith.ConstantOp(builtin.FloatAttr(1.0, type=builtin.f64))

        # load saved_probs[depth + 1]
        load_op = memref.LoadOp.get(probs_stack, (probs_stack_idx,))
        sub_op = arith.SubfOp(c1_f64.results[0], load_op.results[0])
        store_op = memref.StoreOp.get(sub_op.results[0], probs_stack, (probs_stack_idx,))
        for op in (c1, probs_stack_idx, c1_f64, load_op, sub_op, store_op):
            rewriter.insert_op(op, InsertPoint.at_end(false_block))

        restore_reg = self.handle_restore(
            qreg, depth, statevec_stack, statevec_size, rewriter, false_block
        )

        c1_postselect_false = arith.ConstantOp(builtin.IntegerAttr(1, builtin.i8))
        c2_visited_false = arith.ConstantOp(builtin.IntegerAttr(2, builtin.i8))
        false_yield = scf.YieldOp(restore_reg, c1_postselect_false, c2_visited_false)
        for op in (c1_postselect_false, c2_visited_false, false_yield):
            rewriter.insert_op(op, InsertPoint.at_end(false_block))

        # Update visited stack
        visited_value = if_op.results[2]
        store_op = memref.StoreOp.get(visited_value, visited_stack, (depth,))
        return_op = func.ReturnOp(*if_op.results[0:2])

        for op in (c0_branch, is_left_branch, if_op, store_op, return_op):
            rewriter.insert_op(op, InsertPoint.at_end(new_func.body.block))

        return new_func

    def clone_measure_op_with_postselect(  # pylint: disable=too-many-arguments,no-member
        self,
        measure_op: quantum.MeasureOp,
        depth: int,
        branch_type: SSAValue,
        visited_stack: SSAValue,
        statevec_stack: SSAValue,
        probs_stack: SSAValue,
        statevec_size: SSAValue,
        value_mapper: dict,
        new_func: func.FuncOp,
        rewriter: PatternRewriter,
    ):
        """Clone a MeasureOp with postselect based on branch_type."""
        extract_op = measure_op.in_qubit.owner
        assert isinstance(extract_op, quantum.ExtractOp), "in_qubit must be an ExtractOp"

        mapped_in_qubit = value_mapper[measure_op.in_qubit]
        mapped_in_qreg = value_mapper[extract_op.qreg]

        # Convert depth integer to SSAValue
        depth_ssa = arith.ConstantOp.from_int_and_width(depth, builtin.IndexType())
        rewriter.insert_op(depth_ssa, InsertPoint.at_end(new_func.body.block))

        state_transition_func_call = func.CallOp(
            "state_transition",
            [
                mapped_in_qubit,
                mapped_in_qreg,
                depth_ssa.results[0],
                branch_type,
                visited_stack,
                statevec_stack,
                probs_stack,
                statevec_size,
            ],
            return_types=[quantum.QuregType(), builtin.i8],
        )
        rewriter.insert_op(state_transition_func_call, InsertPoint.at_end(new_func.body.block))

        call_result = state_transition_func_call.results
        updated_qreg = call_result[0]
        postselect = call_result[1]

        # Create condition: branch_type == 0
        c0_branch = arith.ConstantOp.from_int_and_width(0, postselect.type)
        is_left_branch = arith.CmpiOp(postselect, c0_branch, "eq")

        result_types = [measure_op.mres.type, measure_op.out_qubit.type]
        if_op = scf.IfOp(is_left_branch, result_types, Region(Block()), Region(Block()))

        # Extract attributes
        measure_op_attr = measure_op.attributes.copy()

        # True branch: postselect = 0 (left branch)
        true_block = if_op.true_region.block
        measure_op_left = quantum.MeasureOp(mapped_in_qubit, postselect=0)
        measure_op_left.attributes.update(measure_op_attr)
        rewriter.insert_op(measure_op_left, InsertPoint.at_end(true_block))
        rewriter.insert_op(
            scf.YieldOp(measure_op_left.mres, measure_op_left.out_qubit),
            InsertPoint.at_end(true_block),
        )

        # False branch: postselect = 1 (right branch)
        false_block = if_op.false_region.block
        measure_op_right = quantum.MeasureOp(mapped_in_qubit, postselect=1)
        measure_op_right.attributes.update(measure_op_attr)
        rewriter.insert_op(measure_op_right, InsertPoint.at_end(false_block))
        rewriter.insert_op(
            scf.YieldOp(measure_op_right.mres, measure_op_right.out_qubit),
            InsertPoint.at_end(false_block),
        )

        for op in (c0_branch, is_left_branch, if_op):
            rewriter.insert_op(op, InsertPoint.at_end(new_func.body.blocks[0]))

        value_mapper[measure_op.mres] = if_op.results[0]
        value_mapper[measure_op.out_qubit] = if_op.results[1]
        value_mapper[extract_op.qreg] = updated_qreg

    @staticmethod
    def populate_segment_io(
        segment: ProgramSegment, missing_inputs: set[SSAValue]
    ) -> tuple[set[SSAValue], set[SSAValue]]:
        """Gather SSA values that need to be passed in and out of the segment to be outlined."""
        inputs = set()
        outputs = set()

        # The segment only needs to return values produced here (i.e. in all op.results) and
        # required by segments further down (i.e. in missing_inputs).
        # The inputs are determined straightforwardly by all operands not defined in this segment.
        # TODO: We might need to be more careful with qubit/register values in the future.
        for op in reversed(segment.ops):
            inputs.update(op.operands)
            inputs.difference_update(op.results)

            outputs.update(op.results)
        outputs.intersection_update(missing_inputs)

        # Update the information used in subsequent calls.
        segment.inputs = [inp for inp in inputs if not isinstance(inp.type, quantum.QuregType)]
        segment.outputs = [out for out in outputs if not isinstance(out.type, quantum.QuregType)]

        missing_inputs.difference_update(segment.outputs)
        missing_inputs.update(segment.inputs)

    def initialize_data_structures(self, rewriter: PatternRewriter):
        """Create data structures in the IR required for the dynamic tree traversal."""
        rewriter.insertion_point = InsertPoint.at_end(self.tt_op.body.block)

        # get the qubit count
        if self.alloc_op.nqubits:
            qubit_count = arith.IndexCastOp(self.alloc_op.nqubits, builtin.IndexType())
        else:
            qubit_count = arith.ConstantOp.from_int_and_width(
                self.alloc_op.nqubits_attr.value, builtin.IndexType()
            )

        # get the tree depth (for now just the segment count)
        tree_depth = arith.ConstantOp.from_int_and_width(
            len(self.quantum_segments), builtin.IndexType()
        )
        c1_index = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
        updated_tree_depth = arith.AddiOp(tree_depth.results[0], c1_index.results[0])

        # initialize stack variables #

        # statevector storage to allow for rollback
        c1 = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
        statevec_size = arith.ShLIOp(c1, qubit_count)

        # Define statevec_stack as memref<depth x (2^n) x complex<f64>>
        statevec_stack = memref.AllocOp((tree_depth, statevec_size), (), self.statevec_stack_type)

        # probabilities for each branch are tracked here
        probs_stack = memref.AllocOp((updated_tree_depth,), (), self.probs_stack_type)
        c1_f64 = arith.ConstantOp(builtin.FloatAttr(1.0, type=builtin.f64))
        length_probs_stack = arith.AddiOp(tree_depth.results[0], c1)
        init_probs_stack_ops = initialize_memref_with_value(
            probs_stack, c1_f64.results[0], length_probs_stack
        )

        # For the current path, we track whether a node is:
        #  - unvisited: 0
        #  - visited down the left branch: 1
        #  - finished: 2
        visited_stack = memref.AllocOp((tree_depth,), (), self.visited_stack_type)

        # initialize with 0
        c0_i8 = arith.ConstantOp.from_int_and_width(0, builtin.i8)
        init_visited_stack_ops = initialize_memref_with_value(
            visited_stack, c0_i8.results[0], tree_depth.results[0]
        )

        c0_f64 = arith.ConstantOp(builtin.FloatAttr(0.0, type=builtin.f64))
        length_folded_result = arith.AddiOp(tree_depth.results[0], c1)
        folded_result = memref.AllocOp((length_folded_result,), (), self.folded_result_type)
        init_folded_result_ops = initialize_memref_with_value(
            folded_result, c0_f64.results[0], length_folded_result
        )

        for op in (
            qubit_count,
            tree_depth,
            c1_index,
            updated_tree_depth,
            c1,
            statevec_size,
            statevec_stack,
            probs_stack,
            c1_f64,
            length_probs_stack,
            *init_probs_stack_ops,
            visited_stack,
            c0_i8,
            *init_visited_stack_ops,
            c0_f64,
            length_folded_result,
            folded_result,
            *init_folded_result_ops,
        ):
            rewriter.insert(op)

        # store some useful values for later
        self.tree_depth = tree_depth.result
        self.statevec_size = statevec_size.result
        self.statevec_stack = statevec_stack.results[0]
        self.probs_stack = probs_stack.results[0]
        self.visited_stack = visited_stack.results[0]
        self.folded_result = folded_result.results[0]

    def generate_traversal_code(self, rewriter: PatternRewriter):  # pylint: disable=too-many-branches,too-many-statements,no-member
        """Create the traversal code of the quantum simulation tree."""
        rewriter.insertion_point = InsertPoint.at_end(self.tt_op.body.block)

        func_args = list(self.simple_io_func.args)
        tt_args = list(self.tt_op.args)
        func_to_tt_mapping = dict(zip(func_args, tt_args))

        # loop instruction
        depth_init = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        segment_io_types = [val.type for val in self.all_segment_io]
        segment_io_inits = []
        for val in self.all_segment_io:
            if val in func_args:  # original function arguments
                tt_arg = func_to_tt_mapping[val]
                segment_io_inits.append(tt_arg)
            else:  # other values, initialize from types
                init_vals = self.initialize_values_from_types([val.type], rewriter)
                segment_io_inits.extend(init_vals)

        # segment_io_inits = self.initialize_values_from_types(segment_io_types, rewriter)
        iter_arg_types = [
            depth_init.result.type,
            *segment_io_types,
        ]
        iter_arg_inits = [depth_init, *segment_io_inits]
        conditionBlock = Block(arg_types=iter_arg_types)
        bodyBlock = Block(arg_types=iter_arg_types)
        traversalOp = scf.WhileOp(iter_arg_inits, iter_arg_types, (conditionBlock,), (bodyBlock,))

        self.traversal_op = traversalOp

        for op in (depth_init, traversalOp):
            if isinstance(op, Operation):  # bypass "ops" that are already SSA values
                rewriter.insert(op)

        # condition block of the while loop
        current_depth = conditionBlock.args[0]
        current_depth = self.check_if_leaf(current_depth, rewriter)
        segment_iter_args = conditionBlock.args[1:]

        # if (depth > 0 or visited[depth] != 2) ...
        c0_ = arith.ConstantOp.from_int_and_width(0, current_depth.type)
        depth_gt_zero = arith.CmpiOp(current_depth, c0_, "sge")  # depth > 0

        # # load visited[depth]
        # load_visited = memref.LoadOp.get(self.visited_stack, (current_depth,))
        # visited_status = arith.IndexCastOp(load_visited.results[0], builtin.IndexType())
        # c2 = arith.ConstantOp.from_int_and_width(2, builtin.IndexType())
        # visited_not_finished = arith.CmpiOp(visited_status, c2, "ne")  # visited[depth] != 2

        # or_op = arith.OrIOp(depth_gt_zero, visited_not_finished)

        condOp = scf.ConditionOp(depth_gt_zero, current_depth, *segment_iter_args)
        # condOp = scf.ConditionOp(or_op, current_depth, *segment_iter_args)

        # for op in (c0_, depth_gt_zero, load_visited, visited_status, c2, visited_not_finished, or_op, condOp):
        #     rewriter.insert_op(op, InsertPoint.at_end(conditionBlock))
        for op in (c0_, depth_gt_zero, condOp):
            rewriter.insert_op(op, InsertPoint.at_end(conditionBlock))

        # body block of the while
        current_depth = bodyBlock.args[0]
        segment_iter_args = bodyBlock.args[1:]

        visited_status = memref.LoadOp.get(self.visited_stack, (current_depth,))
        branch = arith.IndexCastOp(visited_status, builtin.IndexType())

        for op in (visited_status, branch):
            rewriter.insert_op(op, InsertPoint.at_end(bodyBlock))

        # The main logic of the traversal code
        # if casted_status < 2:
        #     call segment_table
        #     depth += 1
        # elif case == 2:
        #     visited[depth] = 0
        #     depth -= 1
        # else:
        #     depth = -1
        #     error
        # return depth, segment_iter_args

        # Check if  < 2 (unvisited or left visited)
        c2 = arith.ConstantOp.from_int_and_width(2, builtin.IndexType())
        not_finished = arith.CmpiOp(branch, c2, "slt")

        if_op = scf.IfOp(
            not_finished,
            (current_depth.type, *segment_io_types),
            Region(Block()),
            Region(Block()),
        )
        true_block = if_op.true_region.block
        false_block = if_op.false_region.block

        # True branch: casted_status < 2 (unvisited or left visited)
        # Call segment_table and increment depth
        call_op = func.CallOp(
            "segment_table",
            [
                current_depth,
                branch,
                self.visited_stack,
                self.statevec_stack,
                self.probs_stack,
                self.folded_result,
                self.statevec_size,
                *segment_iter_args,
            ],
            [val.type for val in segment_iter_args],
        )

        # Increment depth
        c1 = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        updated_depth = arith.AddiOp(current_depth, c1)
        true_yield = scf.YieldOp(updated_depth, *call_op.results)

        for op in (call_op, c1, updated_depth, true_yield):
            rewriter.insert_op(op, InsertPoint.at_end(true_block))

        # Check if casted_status == 2
        finished = arith.CmpiOp(branch, c2, "eq")

        inner_if = scf.IfOp(
            finished, (current_depth.type, *segment_io_types), Region(Block()), Region(Block())
        )
        inner_true_block = inner_if.true_region.block
        inner_false_block = inner_if.false_region.block

        # Inner true: casted_status == 2
        # folded_result[depth] += saved_probs[depth] * folded_result[depth + 1]
        c1 = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
        probs_stack_idx = arith.AddiOp(current_depth, c1)

        load_probs_op = memref.LoadOp.get(self.probs_stack, (current_depth,))
        load_parent_folded_op = memref.LoadOp.get(self.folded_result, (current_depth,))
        load_folded_op = memref.LoadOp.get(self.folded_result, (probs_stack_idx,))

        mulf_op = arith.MulfOp(load_probs_op.results[0], load_folded_op.results[0])
        addf_op = arith.AddfOp(mulf_op.results[0], load_parent_folded_op.results[0])
        store_op = memref.StoreOp.get(addf_op.results[0], self.folded_result, (current_depth,))

        # folded_result[depth + 1] = 0
        f0 = arith.ConstantOp(builtin.FloatAttr(0.0, type=builtin.f64))
        store_op_2 = memref.StoreOp.get(f0, self.folded_result, (probs_stack_idx,))

        for op in (
            c1,
            probs_stack_idx,
            load_probs_op,
            load_parent_folded_op,
            load_folded_op,
            mulf_op,
            addf_op,
            store_op,
            f0,
            store_op_2,
        ):
            rewriter.insert_op(op, InsertPoint.at_end(inner_true_block))

        # visited[depth] = 0
        # depth -= 1
        c0 = arith.ConstantOp.from_int_and_width(0, self.visited_stack.type.element_type)
        update_visited = memref.StoreOp.get(c0, self.visited_stack, (current_depth,))

        c1_dec = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        decremented_depth = arith.SubiOp(current_depth, c1_dec)

        inner_true_yield = scf.YieldOp(decremented_depth, *segment_iter_args)

        for op in (c0, update_visited, c1_dec, decremented_depth, inner_true_yield):
            rewriter.insert_op(op, InsertPoint.at_end(inner_true_block))

        # Inner false: casted_status > 2 (error case)
        # depth = -1
        c_neg_1 = arith.ConstantOp.from_int_and_width(-1, current_depth.type)
        inner_false_yield = scf.YieldOp(c_neg_1, *segment_iter_args)

        for op in (c_neg_1, inner_false_yield):
            rewriter.insert_op(op, InsertPoint.at_end(inner_false_block))

        # Insert inner if into false block
        false_yield = scf.YieldOp(*inner_if.results)
        for op in (finished, inner_if, false_yield):
            rewriter.insert_op(op, InsertPoint.at_end(false_block))

        # Yield op for the while loop
        yield_op = scf.YieldOp(*if_op.results)

        # Insert all operations into the body block in the correct order
        # First insert the basic operations
        for op in (c2, not_finished, if_op, yield_op):
            rewriter.insert_op(op, InsertPoint.at_end(bodyBlock))

    def initialize_values_from_types(self, types, rewriter: PatternRewriter):  # pylint: disable=too-many-branches
        """Generate dummy values for the provided types. Quantum types are treated specially and
        will make use of the quantum.AllocOp reference collected at an earlier stage."""

        # TODO: handling quantum dummy values can be tricky, let's try this for now
        qreg_stub = self.alloc_op.results[0]
        qubit_stub = quantum.ExtractOp(qreg_stub, 0)
        rewriter.insert_op(qubit_stub, InsertPoint.at_end(self.tt_op.body.block))
        qubit_stub_used = False

        ops = []
        need_insert_ops = []
        for ty in types:
            match ty:
                case builtin.IndexType() | builtin.IntegerType():
                    const_op = arith.ConstantOp(builtin.IntegerAttr(0, ty))
                    need_insert_ops.append(const_op)
                    ops.append(const_op)
                case ty if isinstance(ty, builtin._FloatType):  # pylint: disable=protected-access
                    const_op = arith.ConstantOp(builtin.FloatAttr(0.0, ty))
                    need_insert_ops.append(const_op)
                    ops.append(const_op)
                case builtin.ComplexType():
                    assert False, "Complex type unsupported"
                case builtin.TensorType():
                    if ty == builtin.TensorType(builtin.i64, []):
                        const_op = arith.ConstantOp(builtin.IntegerAttr(0, builtin.i64))
                        tensor_op = tensor.FromElementsOp.create(
                            operands=[const_op.results[0]], result_types=[ty]
                        )
                        need_insert_ops.extend([const_op, tensor_op])
                        ops.append(tensor_op)
                    elif ty == builtin.TensorType(builtin.f64, []):
                        const_op = arith.ConstantOp(builtin.FloatAttr(0.0, builtin.f64))
                        tensor_op = tensor.FromElementsOp.create(
                            operands=[const_op.results[0]], result_types=[ty]
                        )
                        need_insert_ops.extend([const_op, tensor_op])
                        ops.append(tensor_op)
                    else:
                        empty_op = tensor.EmptyOp((), ty)
                        need_insert_ops.append(empty_op)
                        ops.append(empty_op)
                case builtin.MemRefType():
                    alloca_op = memref.AllocaOp.get(ty)  # assume this is not called in a loop
                    need_insert_ops.append(alloca_op)
                    ops.append(alloca_op)
                case quantum.QubitType():
                    if qubit_stub_used:
                        # Create a new qubit extraction for each additional qubit needed
                        new_idx = arith.ConstantOp(builtin.IntegerAttr(0, builtin.i64))
                        new_extract = quantum.ExtractOp(qreg_stub, new_idx)
                        need_insert_ops.extend([new_idx, new_extract])
                        ops.append(new_extract.results[0])
                    else:
                        ops.append(qubit_stub.results[0])
                        qubit_stub_used = True
                case quantum.QuregType():
                    ops.append(self.alloc_op.results[0])

        for op in need_insert_ops:
            rewriter.insert_op(op, InsertPoint.at_end(self.tt_op.body.block))

        return ops

    def check_if_leaf(self, current_depth: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        """Verify whether we've hit the bottom of the tree, and perform update actions."""
        assert isinstance(current_depth.owner, Block)
        ip_backup = rewriter.insertion_point
        rewriter.insertion_point = InsertPoint.at_start(current_depth.owner)

        # if instruction
        hit_leaf = arith.CmpiOp(current_depth, self.tree_depth, "eq")
        trueBlock, falseBlock = Block(), Block()
        ifOp = scf.IfOp(hit_leaf, (current_depth.type,), (trueBlock,), (falseBlock,))

        for op in (hit_leaf, ifOp):
            rewriter.insert(op)

        # true branch body - hit leaf, just go back up
        c1 = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        updated_depth = arith.SubiOp(current_depth, c1)

        yieldOp = scf.YieldOp(updated_depth)

        for op in (c1, updated_depth, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(trueBlock))

        # false branch body
        yieldOp = scf.YieldOp(current_depth)
        rewriter.insert_op(yieldOp, InsertPoint.at_end(falseBlock))

        rewriter.insertion_point = ip_backup
        return ifOp.results[0]

    def handle_restore(  # pylint: disable=too-many-arguments
        self,
        qreg: SSAValue,
        current_depth: SSAValue,
        statevec_stack: SSAValue,
        statevec_size: SSAValue,
        rewriter: PatternRewriter,
        insert_block: Block,
    ) -> SSAValue:
        """This function is used to restore the state of the quantum register.
        The behaviour of the generated IR as below:
        ```python
        def restore(qreg, depth):
            state = statevec_stack[depth]
            qreg = quantum.set_state(state)
            return qreg
        ```
        """
        state_memref_type = builtin.MemRefType(
            builtin.ComplexType(builtin.f64),
            (builtin.DYNAMIC_INDEX,),
            builtin.StridedLayoutAttr([1], None),
        )

        all_ops = []

        statevec = memref.SubviewOp.get(
            statevec_stack, (current_depth, 0), (1, statevec_size), (1, 1), state_memref_type
        )
        all_ops.append(statevec)

        qubits = []
        assert self.alloc_op.nqubits_attr is not None, "nqubits of alloc should be a constant"
        for i in range(self.alloc_op.nqubits_attr.value.data):
            extract_op = quantum.ExtractOp(qreg, i)
            qubits.append(extract_op.results[0])
            all_ops.append(extract_op)

        set_state_op = quantum.SetStateOp.create(
            operands=[statevec.results[0], *qubits],
            result_types=[quantum.QubitType() for _ in qubits],
        )
        all_ops.append(set_state_op)

        last_insert_op = None
        for i in range(self.alloc_op.nqubits_attr.value.data):
            insert = quantum.InsertOp(qreg, i, set_state_op.results[i])
            qreg = insert.results[0]
            last_insert_op = insert
            all_ops.append(insert)

        assert last_insert_op is not None

        for op in all_ops:
            rewriter.insert_op(op, InsertPoint.at_end(insert_block))

        return last_insert_op.results[0]

    def handle_store_state(  # pylint: disable=too-many-arguments
        self,
        qreg: SSAValue,
        depth: SSAValue,
        statevec_stack: SSAValue,
        statevec_size: SSAValue,
        rewriter: PatternRewriter,
        insert_block: Block,
    ):
        """This function is used to store the state of the quantum register.
        The behaviour of the generated IR as below:
        ```python
        def store_state(qreg, depth):
            state = quantum.state(qreg)
            statevec_stack[depth] = state
        ```
        """
        # store the state
        # cast_op = arith.IndexCastOp(self.statevec_size, builtin.i64)
        # dynamic_shape = cast_op.results[0]
        state_comp_basis_op = quantum.ComputationalBasisOp(
            operands=([], [qreg]), result_types=[quantum.ObservableType()]  # (qubits, qreg)
        )

        cast_op = arith.IndexCastOp(statevec_size, builtin.i64)

        state_op = quantum.StateOp(
            operands=[state_comp_basis_op.results[0], cast_op.results[0], None],
            result_types=[
                builtin.TensorType(
                    builtin.ComplexType(builtin.Float64Type()), [builtin.DYNAMIC_INDEX]
                )
            ],
        )

        loop_body = Region()
        loop_block = Block(arg_types=[builtin.IndexType()])  # i parameter
        loop_body.add_block(loop_block)

        element = tensor.ExtractOp(
            state_op.results[0],
            loop_block.args[0],  # i parameter
            result_type=builtin.ComplexType(builtin.f64),
        )

        store_op = memref.StoreOp.get(
            element.results[0], statevec_stack, (depth, loop_block.args[0])
        )

        yield_op = scf.YieldOp()

        for op in [element, store_op, yield_op]:
            loop_block.add_op(op)

        c0 = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        c1 = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())

        for_op = scf.ForOp(
            lb=c0.results[0],
            ub=statevec_size,
            step=c1.results[0],  # step = 1
            iter_args=[],
            body=loop_body,
        )

        for op in (state_comp_basis_op, cast_op, state_op, c0, c1, for_op):
            rewriter.insert_op(op, InsertPoint.at_end(insert_block))


class IfOperatorPartitioningPass(RewritePattern):
    """A rewrite pattern that partitions scf.IfOps containing measurement-controlled
    operations into separate branches for each operator.
    ️"""

    IfOpWithDepth = Tuple[scf.IfOp, int]

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: func.FuncOp, rewriter: PatternRewriter
    ) -> None:  # pylint: disable=arguments-differ
        """Partition the if operation into separate branches for each operator."""

        self.original_func_op = op
        # Detect mcm inside If statement
        flat_if = self.detect_mcm_in_if_ops(op)

        if not flat_if:
            return

        # Split IfOps into only true branches
        self.split_nested_if_ops(op, rewriter)


        # Flatten nested IfOps
        self.flatten_nested_IfOps(op, rewriter)

        # Adding fake MeasureOp before if Op with the attribute contain_mcm = "true"
        self.adding_fake_measureOp(op, rewriter)

    def __init__(self):
        self.module: builtin.ModuleOp = None
        self.original_func_op: func.FuncOp = None
        self.holder_returns: dict[scf.IfOp, scf.IfOp] = {}

    def adding_fake_measureOp(self, op: func.FuncOp, rewriter: PatternRewriter) -> None:
        """Add fake MeasureOp before IfOps that contain measurement-controlled operations."""
        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp):
                contain_mcm = "contain_mcm" in current_op.attributes

                if not contain_mcm:
                    continue

                # Get quantum register from missing values
                missing_values = self.analyze_missing_values_for_ops([current_op])
                qreg_if_op = [mv for mv in missing_values if isinstance(mv.type, quantum.QuregType)]

                # Extract a quantum.bit from the current qreg
                # Define a constant index 0
                if not qreg_if_op:
                    continue

                # Extract a quantum.bit from the current qreg
                # Define a constant index 0
                c0 = arith.ConstantOp.from_int_and_width(0, builtin.i64)
                q_extract = quantum.ExtractOp(qreg_if_op[0], c0.results[0])

                rewriter.insert_op(c0, InsertPoint.before(current_op))
                rewriter.insert_op(q_extract, InsertPoint.before(current_op))

                # Adding a fake operation MeasureOP before the if operation using the extracted qubit
                measure_op = quantum.MeasureOp(q_extract.results[0])
                # Add the attribute fake_measure to identify it later
                measure_op.attributes["fake_measure"] = builtin.StringAttr("true")
                rewriter.insert_op(measure_op, InsertPoint.before(current_op))

                q_insert = quantum.InsertOp(qreg_if_op[0], c0.results[0], measure_op.out_qubit)
                rewriter.insert_op(q_insert, InsertPoint.before(current_op))

                # Replace the old q_reg with the output of q_insert
                def _create_exclusion_filter(exclude_extract, exclude_insert):
                    return lambda use: use.operation not in [exclude_extract, exclude_insert]
                qreg_if_op[0].replace_by_if(
                    q_insert.results[0], _create_exclusion_filter(q_extract, q_insert)
                )

    def detect_mcm_in_if_ops(self, op: func.FuncOp) -> bool:
        """Detect if there are measurement-controlled operations inside IfOps."""
        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp):
                # Check if there are measurement-controlled operations inside the IfOp
                for inner_op in current_op.true_region.ops:
                    if isinstance(inner_op, quantum.MeasureOp):
                        return True
                for inner_op in current_op.false_region.ops:
                    if isinstance(inner_op, quantum.MeasureOp):
                        return True
        return False

    def flatten_nested_IfOps(self, main_op: func.FuncOp, rewriter: PatternRewriter) -> None:
        """Flatten nested scf.IfOps into a single level scf.IfOp."""

        # Check for deepest nested IfOps
        nested_IfOp = self.get_deepest_nested_ifs(main_op)

        depth = nested_IfOp[0][1] if nested_IfOp else 0
        target_if_op = nested_IfOp[0][0] if nested_IfOp else None

        if depth > 1:
            self.flatten_if_ops_deep(target_if_op.parent_op(), rewriter)
            self.flatten_nested_IfOps(main_op, rewriter)

    def get_deepest_nested_ifs(self, parent_if_op: scf.IfOp) -> IfOpWithDepth:
        """Finds the scf.if operation(s) nested at the maximum depth inside the parent_if_op."""
        # The parent IfOp A is at depth 0, so its immediate children (B, D) are at depth 1.
        # We initialize the search list.
        deepest_ops_with_depth: List[IfOperatorPartitioningPass.IfOpWithDepth] = [(None, 0)]

        # Start the recursion. We look *inside* the regions of the parent_if_op.
        self._find_deepest_if_recursive(parent_if_op, 0, deepest_ops_with_depth)

        # Extract only the IfOp objects from the list of (IfOp, depth) tuples.
        return deepest_ops_with_depth

    def _find_deepest_if_recursive(
        self, op: Operation, current_depth: int, max_depth_ops: List[IfOpWithDepth]
    ) -> None:
        """
        Helper function to recursively traverse the IR, tracking the max depth
        of scf.If operations found so far.
        """
        # Iterate over all nested regions (then_region, else_region, etc.)
        for region in op.regions:
            for block in region.blocks:
                for child_op in block.ops:

                    new_depth = current_depth

                    if isinstance(child_op, scf.IfOp):
                        # the if should have the attribute  contain_mcm = "true"

                        contain_mcm = "contain_mcm" in child_op.attributes

                        if not contain_mcm:
                            continue

                        # Found an IfOp, increase the depth for the ops *inside* its regions.
                        # This IfOp itself is at 'current_depth + 1'.
                        new_depth = current_depth + 1

                        # --- Check and Update Max Depth List ---

                        # 1. Is this deeper than the current max? (First find or deeper op)
                        if not max_depth_ops or new_depth > max_depth_ops[0][1]:
                            # It's a new maximum depth! Clear the old list and start fresh.
                            max_depth_ops.clear()
                            max_depth_ops.append((child_op, new_depth))

                        # 2. Is this at the same depth as the current max? (A tie)
                        elif new_depth == max_depth_ops[0][1]:
                            # Add it to the list of winners.
                            max_depth_ops.append((child_op, new_depth))

                    # Recursively search inside this child op (regardless of its type)
                    # We pass the potentially *increased* new_depth.
                    self._find_deepest_if_recursive(child_op, new_depth, max_depth_ops)

    def flatten_if_ops_deep(self, main_op: scf.IfOp, rewriter: PatternRewriter) -> None:
        """Flatten nested scf.IfOps into a single level scf.IfOp."""

        if isinstance(main_op, scf.IfOp):

            outer_if_op = main_op

            new_outer_if_op_output = list(outer_if_op.results)
            new_outer_if_op_output_types = [out.type for out in outer_if_op.results]

            _, nested_if_ops = self.get_nested_if_ops(outer_if_op)
            where_to_insert = outer_if_op

            # Holder for IfOps that are kept for updating SSA values later
            self.holder_returns = {}

            for inner_op in nested_if_ops:

                where_to_insert, outer_if_op = self.move_inner_if_op_2_outer(
                    inner_op,
                    outer_if_op,
                    new_outer_if_op_output,
                    new_outer_if_op_output_types,
                    where_to_insert,
                    rewriter,
                )

            # detach and erase old outer if op
            for hold_op in self.holder_returns:
                hold_op.detach()
                hold_op.erase()

    def move_inner_if_op_2_outer(  # pylint: disable=too-many-branches,too-many-arguments,too-many-statements,no-member
        self,
        inner_op: scf.IfOp,
        outer_if_op: scf.IfOp,
        new_outer_if_op_output: list[SSAValue],
        new_outer_if_op_output_types: list[Type],
        where_to_insert: scf.IfOp,
        rewriter: PatternRewriter,
    ) -> None:
        """Move inner IfOp after the outer IfOp."""

        definition_outer = self.analyze_definitions_for_ops([outer_if_op])
        missing_values_inner = self.analyze_missing_values_for_ops([inner_op])

        ssa_needed_from_outer = set(missing_values_inner).intersection(set(definition_outer))

        # Select only definition outer. Use list to preserve order
        missing_values_inner = [mv for mv in missing_values_inner if mv in definition_outer]

        for mv in ssa_needed_from_outer:
            if not isinstance(mv.type, quantum.QuregType):
                new_outer_if_op_output.append(mv)
                new_outer_if_op_output_types.append(mv.type)

        inner_results = inner_op.results

        # Replace the qreg from the inner IfOp with the immediate outer IfOp qreg
        # This dont affect the inner IfOp since its qreg is only used in quantum ops inside its regions
        qreg_if_op_inner = [
            mv for mv in missing_values_inner if isinstance(mv.type, quantum.QuregType)
        ]

        for result in inner_results:
            if isinstance(result.type, quantum.QuregType):
                result.replace_by(qreg_if_op_inner[0])

        qreg_if_op_outer = [
            output
            for output in where_to_insert.results
            if isinstance(output.type, quantum.QuregType)
        ]

        assert (
            len(qreg_if_op_outer) == 1
        ), "Expected exactly one quantum register in outer IfOp results."

        # Detach inner_op from its parent before modifying
        if len(inner_results) == 1:
            inner_op.detach()
        else:
            # Add a new attribute to mark it as flattened
            inner_op.attributes["old_return"] = builtin.StringAttr("true")

        # expand the current attr_dict
        attr_dict = inner_op.attributes.copy()
        attr_dict.update({"flattened": builtin.StringAttr("true")})

        ############################################################################################
        # Create new inner IfOp with updated regions

        # ------------------------------------------------------------------------------------------
        # Inner true region

        # Create comprehensive value mapping for all values used in both regions
        value_mapper = {}
        value_mapper[qreg_if_op_inner[0]] = qreg_if_op_outer[0]

        inner_true_region = inner_op.true_region

        true_ops = list(inner_true_region.blocks[0].ops)

        new_true_block = Block()

        self.clone_operations_to_block(true_ops, new_true_block, value_mapper)

        # ------------------------------------------------------------------------------------------
        # Inner false region

        false_inner_ops = list(inner_op.false_region.blocks[0].ops)

        new_false_block = None

        if len(false_inner_ops) == 1 and isinstance(false_inner_ops[0], scf.YieldOp):
            # If the false region only contains a yield operation, we can create an empty block

            # Create a new empty block for false region
            new_false_block = Block()

            # Create a yield operation for false region using the same return types as the original IfOp
            yield_false = scf.YieldOp(where_to_insert.results[0])

            # Create a new empty block for false region
            new_false_block.add_op(yield_false)

        else:
            # If the false region contains other operations, clone them as usual
            false_block_inner = inner_op.false_region.detach_block(0)
            false_ops = list(false_block_inner.ops)

            new_false_block = Block()

            value_mapper = {qreg_if_op_inner[0]: qreg_if_op_outer[0]}
            self.clone_operations_to_block(false_ops, new_false_block, value_mapper)

        new_if_op_attrs = where_to_insert.attributes.copy()
        new_if_op_attrs.update(attr_dict or {})
        # ------------------------------------------------------------------------------------------
        # Create new IfOp with cloned regions

        # Check if we need to update the conditional, if the conditional not depends on previous IfOp results
        # that have been removed, then we need to update it
        needs_to_update_conditional = True

        if inner_op.cond.owner.attributes.get("old_return", None) is not None and isinstance(
            inner_op.cond.owner, scf.IfOp
        ):
            hold_return = inner_op.cond.owner
            return_index = list(hold_return.results).index(inner_op.cond)
            conditional = self.holder_returns[hold_return].results[return_index]
            needs_to_update_conditional = False

            for res in hold_return.results:
                if res in missing_values_inner:
                    remove_index = missing_values_inner.index(res)
                    missing_values_inner.pop(remove_index)

        else:
            conditional = inner_op.cond

        new_inner_op = scf.IfOp(
            conditional, inner_op.result_types, [new_true_block], [new_false_block], new_if_op_attrs
        )
        rewriter.insert_op(new_inner_op, InsertPoint.after(where_to_insert))

        # Update uses of old inner IfOp results to new inner IfOp results
        new_inner_op_ops = list(chain(*[op.walk() for op in [new_inner_op]]))
        where_to_insert.results[0].replace_by_if(
            new_inner_op.results[0], lambda use: use.operation not in new_inner_op_ops
        )

        where_to_insert = new_inner_op

        # Detach and erase old inner IfOp
        if len(inner_results) == 1:
            inner_op.erase()
        else:
            self.holder_returns[inner_op] = new_inner_op
            update_unused_cond = False
            unused_op = None
            for op in self.holder_returns:
                for res in op.results:
                    if inner_op.cond == res:
                        update_unused_cond = True
                        unused_op = op
            if update_unused_cond:
                inner_op.cond.replace_by(unused_op.cond)
        ############################################################################################
        # Create a new outer IfOp that includes the new outputs needed from the inner IfOp

        # ------------------------------------------------------------------------------------------
        # Outer true block

        true_block = outer_if_op.true_region.detach_block(0)

        true_yield_op = [op for op in true_block.ops if isinstance(op, scf.YieldOp)][-1]

        # Merge the existing true yield operands with the missing values from inner IfOp
        new_res = list(true_yield_op.operands) + [
            ssa for ssa in missing_values_inner if not isinstance(ssa.type, quantum.QuregType)
        ]
        return_types = [new_r.type for new_r in new_res]

        new_true_yield_op = scf.YieldOp(*new_res)

        rewriter.replace_op(true_yield_op, new_true_yield_op)

        # ------------------------------------------------------------------------------------------
        # Outer false block

        # Detach the false block to preserve SSA dependencies
        false_block = outer_if_op.false_region.detach_block(0)

        false_op_res = []

        if needs_to_update_conditional:
            false_op = arith.ConstantOp(builtin.IntegerAttr(0, builtin.IntegerType(1)))
            false_op_res.append(false_op.result)
            rewriter.insert_op(false_op, InsertPoint.at_start(false_block))

        false_yield_op = [op for op in false_block.ops if isinstance(op, scf.YieldOp)][-1]

        new_res = list(false_yield_op.operands) + false_op_res

        new_false_yield_op = scf.YieldOp(*new_res)

        rewriter.replace_op(false_yield_op, new_false_yield_op)

        # ------------------------------------------------------------------------------------------
        # Create new IfOp with cloned regions
        new_outer_if_op = scf.IfOp(
            outer_if_op.cond,
            return_types,
            [true_block],
            [false_block],
            outer_if_op.attributes.copy(),
        )

        # Add it at the top of the block
        rewriter.insert_op(new_outer_if_op, InsertPoint.before(outer_if_op))

        for old_result, new_result in zip(outer_if_op.results, new_outer_if_op.results):
            old_result.replace_by(new_result)

        outer_if_op.detach()
        outer_if_op.erase()

        outer_if_op = new_outer_if_op

        if needs_to_update_conditional:
            new_cond = new_inner_op.cond
            new_cond.replace_by_if(
                outer_if_op.results[-1], lambda use: use.operation in [new_inner_op]
            )

        return where_to_insert, outer_if_op

    def get_nested_if_ops(self, op: scf.IfOp) -> tuple[bool, list[scf.IfOp]]:
        """Get nested IfOps from the given IfOp, checking immediate operations only."""
        nested_if_ops = []
        # Only check the immediate operations in the true region (not nested deeper)
        for inner_op in op.true_region.block.ops:
            if isinstance(inner_op, scf.IfOp):
                nested_if_ops.append(inner_op)
        # Only check the immediate operations in the false region (not nested deeper)
        for inner_op in op.false_region.block.ops:
            if isinstance(inner_op, scf.IfOp):
                nested_if_ops.append(inner_op)
        return len(nested_if_ops) > 0, nested_if_ops

    def split_nested_if_ops(
        self, op: func.FuncOp, rewriter: PatternRewriter, go_deeper: bool = False
    ) -> None:
        """Recursively split nested scf.IfOps into separate branches for true and false regions."""

        if go_deeper and isinstance(op, scf.IfOp):

            # Process true region
            true_region = op.true_region
            have_mcm_nested_if_op = False
            for inner_op in true_region.ops:

                if isinstance(inner_op, scf.IfOp):

                    have_mcm_nested_if_op = self.detect_mcm_in_if_ops(inner_op)
                    self.look_and_split_if_ops(inner_op, rewriter)

                if have_mcm_nested_if_op and isinstance(
                    inner_op, (quantum.CustomOp, quantum.MeasureOp)
                ):
                    raise ValueError("Not supported: CustomOp after MCM nested IfOp.")

            # Process false region
            false_region = op.false_region
            have_mcm_nested_if_op = False
            for inner_op in false_region.ops:
                if isinstance(inner_op, scf.IfOp):

                    have_mcm_nested_if_op = self.detect_mcm_in_if_ops(inner_op)

                    self.look_and_split_if_ops(inner_op, rewriter)

                if have_mcm_nested_if_op and isinstance(
                    inner_op, (quantum.CustomOp, quantum.MeasureOp)
                ):
                    raise ValueError(
                        "Not supported: CustomOp after nested IfOp with mid-circuit measurement."
                    )
            return

        # Initial call to split nested IfOps in the function
        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp) and self.detect_mcm_in_if_ops(current_op):
                self.look_and_split_if_ops(current_op, rewriter)

    def look_and_split_if_ops(self, current_op: func.FuncOp, rewriter: PatternRewriter) -> None:
        """Look for scf.IfOps and split them if they contain measurement-controlled operations."""

        mcm_counts = self.count_mcm_in_if_op(current_op)
        assert mcm_counts[0] < 2 and mcm_counts[1] < 2, "Not support IfOp with more than 2 mcm"

        have_nested_if_ops = self.looking_for_nested_if_ops(current_op)

        # Recursively split deeper nested IfOps first
        if have_nested_if_ops:
            self.split_nested_if_ops(current_op, rewriter, go_deeper=True)
            self.split_if_op(current_op, rewriter)
        # Deepest level, split directly
        if not have_nested_if_ops:
            self.split_if_op(current_op, rewriter)

    def count_mcm_in_if_op(self, op: scf.IfOp) -> list[int]:
        """Count mid-circuit measurements in true and false regions of an IfOp."""
        count_true = 0
        for inner_op in op.true_region.ops:
            if isinstance(inner_op, quantum.MeasureOp):
                count_true += 1
        count_false = 0
        for inner_op in op.false_region.ops:
            if isinstance(inner_op, quantum.MeasureOp):
                count_false += 1
        return [count_true, count_false]

    def looking_for_nested_if_ops(self, op: scf.IfOp) -> bool:
        """Look for nested IfOps within the given IfOp's regions."""
        for inner_op in op.true_region.ops:
            if isinstance(inner_op, scf.IfOp):
                return True
        for inner_op in op.false_region.ops:
            if isinstance(inner_op, scf.IfOp):
                return True
        return False

    def split_if_op(self, op: func.FuncOp, rewriter: PatternRewriter) -> None:
        """Split an scf.IfOp into separate branches for true and false regions."""

        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp):

                # Analyze missing values for the IfOp
                missing_values = self.analyze_missing_values_for_ops([current_op])

                # Get quantum register from missing values
                qreg_if_op = [mv for mv in missing_values if isinstance(mv.type, quantum.QuregType)]

                # True and False regions
                true_region = current_op.true_region
                false_region = current_op.false_region

                # --------------------------------------------------------------------------
                # New partitioning logic for True region
                # --------------------------------------------------------------------------

                value_mapper = {}

                attr_dict = {
                    "contain_mcm": builtin.StringAttr("true"),
                    "partition": builtin.StringAttr("true_branch"),
                }

                new_if_op_4_true = self.create_if_op_partition(
                    rewriter, true_region, current_op, value_mapper, current_op, attr_dict=attr_dict
                )

                # --------------------------------------------------------------------------
                # New partitioning logic for False region
                # --------------------------------------------------------------------------
                # Add the negation of the condition to the false branch if needed

                true_op = arith.ConstantOp(builtin.IntegerAttr(1, builtin.IntegerType(1)))
                not_op = arith.XOrIOp(current_op.cond, true_op.result)

                # Insert not_op after new_if_op
                for new_op in [not_op, true_op]:
                    rewriter.insert_op(new_op, InsertPoint.after(new_if_op_4_true))

                # --------------------------------------------------------------------------
                # Create the new IfOp for the false region
                # --------------------------------------------------------------------------

                value_mapper = {qreg_if_op[0]: new_if_op_4_true.results[0]}

                attr_dict = {
                    "contain_mcm": builtin.StringAttr("true"),
                    "partition": builtin.StringAttr("false_branch"),
                }

                _ = self.create_if_op_partition(
                    rewriter,
                    false_region,
                    new_if_op_4_true,
                    value_mapper,
                    not_op,
                    conditional=not_op.result,
                    attr_dict=attr_dict,
                )

                # --------------------------------------------------------------------------

                original_if_op_results = current_op.results[0]
                original_if_op_results.replace_by(qreg_if_op[0])

                list_op_if = list(current_op.walk())

                # Remove the ops in the original IfOp
                for if_op in list_op_if[::-1]:
                    if_op.detach()
                    if_op.erase()

    def create_if_op_partition(  # pylint: disable=too-many-arguments
        self,
        rewriter: PatternRewriter,
        if_region: Region,
        previous_IfOp: scf.IfOp,
        value_mapper: dict[SSAValue, SSAValue],
        op_where_insert_after: Operation,
        conditional: SSAValue = None,
        attr_dict: dict[str, builtin.Attribute] = None,
    ) -> scf.IfOp:
        """Create a new IfOp partition with cloned regions and updated value mapping."""

        true_ops = list(if_region.blocks[0].ops)

        new_true_block = Block()

        self.clone_operations_to_block(true_ops, new_true_block, value_mapper)

        # --------------------------------------------------------------------------
        # Create a new empty block for false region
        new_false_block = Block()

        # Create a yield operation for false region using the same return types as the original IfOp
        yield_false = scf.YieldOp(previous_IfOp.results[0])

        # Create a new empty block for false region
        new_false_block.add_op(yield_false)

        new_if_op_attrs = previous_IfOp.attributes.copy()
        new_if_op_attrs.update(attr_dict or {})
        # --------------------------------------------------------------------------
        # Create new IfOp with cloned regions
        # scf.IfOp (
        # cond: SSAValue | Operation,
        # return_types: Sequence[Attribute],
        # true_region: Region | Sequence[Block] | Sequence[Operation],
        # false_region: Region | Sequence[Block] | Sequence[Operation] | None = None,
        # attr_dict: dict[str, Attribute] | None = None,
        # )

        if conditional is None:
            conditional = previous_IfOp.cond

        new_if_op_4_true = scf.IfOp(
            conditional,
            previous_IfOp.result_types,
            [new_true_block],
            [new_false_block],
            new_if_op_attrs,
        )
        rewriter.insert_op(new_if_op_4_true, InsertPoint.after(op_where_insert_after))

        new_if_op_4_true_ops = list(chain(*[op.walk() for op in [new_if_op_4_true]]))

        previous_IfOp.results[0].replace_by_if(
            new_if_op_4_true.results[0], lambda use: use.operation not in new_if_op_4_true_ops
        )

        return new_if_op_4_true

    def analyze_missing_values_for_ops(self, ops: list[Operation]) -> list[SSAValue]:
        """get missing values for ops
        Given a list of operations, return the values that are missing from the operations.
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

    def analyze_definitions_for_ops(self, ops: list[Operation]) -> list[SSAValue]:
        """get defined values for ops
        Given a list of operations, return the values that are defined by the operations.
        """
        # ops_walk = list(chain(*[op.walk() for op in ops]))
        ops_walk = []

        for op in ops:
            for region in op.regions:
                for block in region.blocks:
                    for child_op in block.ops:
                        ops_walk.append(child_op)

        ops_defined_values = set()

        for nested_op in ops_walk:
            ops_defined_values.update(nested_op.results)

            if hasattr(nested_op, "regions") and nested_op.regions:
                for region in nested_op.regions:
                    for block in region.blocks:
                        ops_defined_values.update(block.args)

        return list(ops_defined_values)

    def analyze_required_outputs(
        self, ops: list[Operation], terminal_op: Operation, new_original_func_op: func.FuncOp = None
    ) -> list[SSAValue]:
        """get required outputs for ops
        Given a list of operations and a terminal operation, return the values that are
        required by the operations after the terminal operation.
        Noted: It's only consdider the values that are defined in the operations and required by
        the operations after the terminal operation!
        """
        ops_walk = list(chain(*[op.walk() for op in ops]))

        ops_defined_values = set()

        for nested_op in ops_walk:
            ops_defined_values.update(nested_op.results)

        required_outputs = set()
        found_terminal = False

        body_walk = self.original_func_op.body.walk()

        if new_original_func_op is not None:
            body_walk = new_original_func_op.body.walk()

        for op in body_walk:
            if op == terminal_op:
                found_terminal = True
                continue

            if found_terminal:
                for operand in op.operands:
                    if operand in ops_defined_values:
                        required_outputs.add(operand)

        return list(required_outputs)

    def clone_operations_to_block(self, ops_to_clone, target_block, value_mapper):
        """Clone operations to target block, use value_mapper to update references"""
        for op in ops_to_clone:
            cloned_op = op.clone(value_mapper)
            target_block.add_op(cloned_op)

            self.update_value_mapper_recursively(op, cloned_op, value_mapper)

    def update_value_mapper_recursively(self, orig_op, cloned_op, value_mapper):
        """update value_mapper for all operations in operation"""
        for orig_result, new_result in zip(orig_op.results, cloned_op.results):
            value_mapper[orig_result] = new_result

        for orig_region, cloned_region in zip(orig_op.regions, cloned_op.regions):
            self.update_region_value_mapper(orig_region, cloned_region, value_mapper)

    def update_region_value_mapper(self, orig_region, cloned_region, value_mapper):
        """update value_mapper for all operations in region"""
        for orig_block, cloned_block in zip(orig_region.blocks, cloned_region.blocks):
            for orig_arg, cloned_arg in zip(orig_block.args, cloned_block.args):
                value_mapper[orig_arg] = cloned_arg

            for orig_nested_op, cloned_nested_op in zip(orig_block.ops, cloned_block.ops):
                self.update_value_mapper_recursively(orig_nested_op, cloned_nested_op, value_mapper)


class UnrollLoopPattern(RewritePattern):
    """A rewrite pattern that unrolls scf.ForOps containing measurement-controlled
    operations into separate branches for each operator."""

    def __init__(self):
        """Initialize UnrollLoopPattern."""
        self.needs_unroll: bool = False

    def match_and_rewrite(
        self, op: scf.ForOp, rewriter: PatternRewriter
    ) -> None:  # pylint: disable=arguments-differ
        """Unroll nested scf.ForOps into separate branches for each operator."""

        self.needs_unroll = self.detect_mcm_in_loop_ops(op)

        if not self.needs_unroll:
            return

        self.unroll_nested_loops(op, rewriter)
        op_walk = op.walk()
        for nested_op in op_walk:
            if isinstance(nested_op, scf.ForOp):
                if not self.detect_mcm_in_loop_ops(nested_op):
                    continue
                self.unroll_loop(nested_op, rewriter)

    def unrolling_applied(self) -> bool:
        """Check if unrolling was applied."""
        return self.needs_unroll

    def detect_mcm_in_loop_ops(self, op: scf.ForOp) -> bool:
        """Detect if there are mid-circuit measurement operations inside ForOps."""
        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.ForOp):
                for inner_op in current_op.body.ops:
                    if isinstance(inner_op, quantum.MeasureOp):
                        return True
        return False

    def unroll_nested_loops(self, main_op: scf.ForOp, rewriter: PatternRewriter) -> None:
        """Unroll nested scf.ForOps into separate branches for each operator."""

        # Check for deepest nested ForOps
        nested_ForOp = self.get_deepest_for_loops(main_op)

        depth = nested_ForOp[0][1] if nested_ForOp else 0
        target_for_op = nested_ForOp[0][0] if nested_ForOp else None

        if depth > 1:
            self.unroll_loop(target_for_op.parent_op(), rewriter)
            self.unroll_nested_loops(main_op, rewriter)

    def get_deepest_for_loops(self, parent_op: scf.ForOp) -> list[tuple[scf.ForOp, int]]:
        """Finds the scf.for operation(s) nested at the maximum depth inside the parent_op."""

        deepest_ops_with_depth: List[tuple[scf.ForOp, int]] = [(None, 0)]

        # Start the recursion. We look *inside* the regions of the parent_op.
        self._find_deepest_for_recursive(parent_op, 0, deepest_ops_with_depth)

        # Extract only the ForOp objects from the list of (ForOp, depth) tuples.
        return deepest_ops_with_depth

    def _find_deepest_for_recursive(
        self, op: Operation, current_depth: int, max_depth_ops: List[tuple[scf.ForOp, int]]
    ) -> None:
        """
        Helper function to recursively traverse the IR, tracking the max depth
        of scf.For operations found so far.
        """
        # Iterate over all nested regions (then_region, else_region, etc.)
        for region in op.regions:
            for block in region.blocks:
                for child_op in block.ops:

                    new_depth = current_depth

                    if isinstance(child_op, scf.ForOp):
                        # Found an ForOp, increase the depth for the ops *inside* its regions.
                        # This ForOp itself is at 'current_depth + 1'.
                        new_depth = current_depth + 1

                        # --- Check and Update Max Depth List ---

                        # 1. Is this deeper than the current max? (First find or deeper op)
                        if not max_depth_ops or new_depth > max_depth_ops[0][1]:
                            # It's a new maximum depth! Clear the old list and start fresh.
                            max_depth_ops.clear()
                            max_depth_ops.append((child_op, new_depth))

                        # 2. Is this at the same depth as the current max? (A tie)
                        elif new_depth == max_depth_ops[0][1]:
                            # Add it to the list of winners.
                            max_depth_ops.append((child_op, new_depth))

                    # Recursively search inside this child op (regardless of its type)
                    # We pass the potentially *increased* new_depth.
                    self._find_deepest_for_recursive(child_op, new_depth, max_depth_ops)

    def unroll_loop(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:
        """Unroll an scf.ForOp into separate branches for each operator."""

        def find_constant_bound(bound: SSAValue) -> tuple[bool, Operation | None]:
            """Find the constant value of a SSA in ForOp bounds"""

            check_bound = bound
            while True:
                if isinstance(check_bound.owner, arith.ConstantOp):
                    return True, check_bound.owner
                if isinstance(check_bound.owner, stablehlo.ConstantOp):
                    return True, check_bound.owner
                if isinstance(check_bound, BlockArgument):
                    return False, None
                if len(check_bound.owner.operands) == 0:
                    return False, None

                check_bound = check_bound.owner.operands[0]

        ub_found, ub_op = find_constant_bound(op.ub)
        lb_found, lb_op = find_constant_bound(op.lb)
        step_found, step_op = find_constant_bound(op.step)

        assert (
            lb_found and ub_found and step_found
        ), "UnrollLoopPattern: Cannot unroll loop, bounds or step are not constant."

        def check_extract_value(bound: Operation) -> int:

            is_IntegerAttr: bool = isinstance(bound.value, builtin.IntegerAttr)
            is_ElementsAttr: bool = isinstance(bound.value, builtin.DenseIntOrFPElementsAttr)

            assert (
                is_IntegerAttr or is_ElementsAttr
            ), "UnrollLoopPattern: The ForOp bound should come from arith.ConstantOp or stablehlo.ConstantOp"

            if is_IntegerAttr:
                return bound.value.value.data

            # is_ElementsAttr
            return bound.value.data.data[0]

        lb = check_extract_value(lb_op)
        ub = check_extract_value(ub_op)
        step = check_extract_value(step_op)

        iter_args: tuple[SSAValue, ...] = op.iter_args

        i_arg, *block_iter_args = op.body.block.args

        inner_op_clone: Operation | None = None

        for i in range(lb, ub, step):
            i_op = rewriter.insert_op(
                arith.ConstantOp(builtin.IntegerAttr(i, builtin.IndexType())),
                InsertPoint.before(op),
            )
            i_op.result.name_hint = i_arg.name_hint

            value_mapper: dict[SSAValue, SSAValue] = dict(
                zip(block_iter_args, iter_args, strict=True)
            )
            value_mapper[i_arg] = i_op.result

            for inner_op in op.body.block.ops:
                if isinstance(inner_op, scf.YieldOp):
                    iter_args = tuple(value_mapper.get(val, val) for val in inner_op.arguments)
                else:
                    inner_op_clone = inner_op.clone(value_mapper)
                    rewriter.insert_op(inner_op_clone, InsertPoint.before(op))

        if inner_op_clone is not None:
            op.results[0].replace_by(inner_op_clone.results[0])

            op.detach()
            op.erase()
