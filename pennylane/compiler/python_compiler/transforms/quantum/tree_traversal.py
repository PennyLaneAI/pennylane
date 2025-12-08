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

from xdsl import context
from xdsl.dialects import arith, builtin, func, memref, scf, tensor
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import BlockInsertPoint, InsertPoint

from pennylane.compiler.python_compiler import compiler_transform
from pennylane.compiler.python_compiler.dialects import quantum
from pennylane.compiler.python_compiler.utils import get_parent_of_type

from .tree_traversal_if_statements import IfOperatorPartitioningPattern
from .tree_traversal_unroll_static_loops import UnrollLoopPattern


##############################################################################
# Some useful utils
##############################################################################
def initialize_memref_with_value(dest: SSAValue, value: SSAValue, size: SSAValue):
    """Initialize a memref with value"""
    # lower bound
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


@dataclass
class StackAttributes: # pylint: disable=too-many-instance-attributes
    """Stack-related attributes and their types for tree traversal."""

    # Stack values
    statevec_stack: SSAValue = None
    probs_stack: SSAValue = None
    visited_stack: SSAValue = None
    folded_result: SSAValue = None

    # Stack types
    probs_stack_type: builtin.MemRefType = field(
        default_factory=lambda: builtin.MemRefType(builtin.f64, (builtin.DYNAMIC_INDEX,))
    )
    visited_stack_type: builtin.MemRefType = field(
        default_factory=lambda: builtin.MemRefType(builtin.i8, (builtin.DYNAMIC_INDEX,))
    )
    statevec_stack_type: builtin.MemRefType = field(
        default_factory=lambda: builtin.MemRefType(
            builtin.ComplexType(builtin.f64),
            [builtin.DYNAMIC_INDEX, builtin.DYNAMIC_INDEX],  # [depth, 2^n]
        )
    )
    folded_result_type: builtin.MemRefType = field(
        default_factory=lambda: builtin.MemRefType(builtin.f64, (builtin.DYNAMIC_INDEX,))
    )


@dataclass
class FunctionOps:
    """Function operations used in tree traversal transformation."""

    # The original function op
    original_func_op: func.FuncOp = None
    # The simple io function, which should be removed after the transformation
    simple_io_func: func.FuncOp = None
    # The main traversal function entry
    tt_op: func.FuncOp = None
    # The state transition function
    state_transition_func: func.FuncOp = None


@dataclass
class SegmentInfo:
    """Information about program segments and their I/O."""

    terminal_segment: ProgramSegment = None
    all_segment_io: list = None
    values_as_io_index: dict = None


@dataclass
class TraversalState:
    """State information for the tree traversal."""

    tree_depth: SSAValue = None
    statevec_size: SSAValue = None
    traversal_op: Operation = None
    alloc_op: quantum.AllocOp = None


@dataclass(frozen=True)
class TreeTraversalPass(ModulePass):
    """Pass that transforms a quantum function (qnode) to perform tree-traversal simulation."""

    name = "tree-traversal"

    def apply(self, _ctx: context.Context, module_op: builtin.ModuleOp) -> None:
        """Apply the tree-traversal pass to all QNode functions in the module."""

        for op in module_op.ops:
            if isinstance(op, func.FuncOp) and "qnode" in op.attributes:
                rewriter = PatternRewriter(op)

                UnrollLoopPattern().match_and_rewrite(op, rewriter)

                IfOperatorPartitioningPattern().match_and_rewrite(op, rewriter)

                TreeTraversalPattern().match_and_rewrite(op, rewriter)
                break


tree_traversal_pass = compiler_transform(TreeTraversalPass)


class TreeTraversalPattern(RewritePattern):
    """Tree-Traversal MCM simulation method as an xDSL transform in Catalyst."""

    def __init__(self):
        self.module: builtin.ModuleOp = None
        self.quantum_segments: list[ProgramSegment] = []

        # Grouped attributes
        self.stacks = StackAttributes()
        self.funcs = FunctionOps()
        self.segments = SegmentInfo()
        self.state = TraversalState()

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, func_op: func.FuncOp, rewriter: PatternRewriter
    ):  # pylint: disable=arguments-differ
        """Transform a quantum function (qnode) to perform tree-traversal simulation."""
        self.funcs.original_func_op = func_op

        if "qnode" not in func_op.attributes:
            return

        self.module = get_parent_of_type(func_op, builtin.ModuleOp)

        # If no measurements found, no need to apply tree-traversal
        if not self.check_if_qnode_has_mcms(func_op):
            return

        # Start with creating a new QNode function that will perform the tree traversal simulation.
        # We prep the original QNode by ensuring measure boundaries are also quantum register boundaries.
        self.funcs.simple_io_func = self.simplify_quantum_io(func_op, rewriter)

        self.setup_traversal_function(self.funcs.simple_io_func, rewriter)

        self.split_traversal_segments(self.funcs.simple_io_func, rewriter)

        self.initialize_traversal_attrs(rewriter)

        self.generate_traversal_code(rewriter)

        self.finalize_traversal_function(rewriter)

    def get_qubit_idx(self, op: quantum.ExtractOp) -> builtin.IntegerAttr | SSAValue | None:
        """Get the index of the operation."""
        return op.idx if op.idx else op.idx_attr

    def check_if_qnode_has_mcms(self, func_op: func.FuncOp) -> bool:
        """Check if the QNode function contains any measurement operations."""
        for op in func_op.body.walk():
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

        current_reg: SSAValue = None
        qubit_to_reg_idx: dict[SSAValue, SSAValue | builtin.IntegerAttr] = {}

        for op in cloned_fun.body.ops:
            match op:
                case quantum.AllocOp():
                    current_reg = op.qreg
                case quantum.ExtractOp():
                    # update register since it might have changed
                    extract_idx = self.get_qubit_idx(op)
                    qubit_to_reg_idx[op.qubit] = extract_idx
                    op.operands = (current_reg, extract_idx)
                case quantum.CustomOp():
                    for i, qb in enumerate(chain(op.in_qubits, op.in_ctrl_qubits)):
                        qubit_to_reg_idx[op.results[i]] = qubit_to_reg_idx[qb]
                        del qubit_to_reg_idx[qb]
                case quantum.InsertOp():
                    assert qubit_to_reg_idx[op.qubit] == op.idx_attr if op.idx_attr else True
                    del qubit_to_reg_idx[op.qubit]
                    # update register since it might have changed
                    op.operands = (current_reg, op.idx, op.qubit)
                    current_reg = op.out_qreg
                case quantum.MeasureOp():
                    # find the qubit to be measured and its index
                    mcm_idx = qubit_to_reg_idx.get(op.in_qubit, None)
                    if mcm_idx is None:
                        raise RuntimeError(
                            f"Could not find qubit {op.in_qubit} in register mapping"
                        )

                    # store the old qubit before processing the measure operation
                    mcm_qubit = op.in_qubit
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

                    for qb, idx in list(qubit_to_reg_idx.items()):
                        extract_op = quantum.ExtractOp(current_reg, idx)
                        rewriter.insert(extract_op)
                        qb.replace_by_if(
                            extract_op.qubit, lambda use: use.operation not in insert_ops
                        )
                        qubit_to_reg_idx[extract_op.qubit] = idx
                        del qubit_to_reg_idx[qb]

                    qubit_to_reg_idx[op.out_qubit] = mcm_idx
                    # break
                case _:
                    # Handle other operations that might have qreg result
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

        self.funcs.tt_op = tt_op

    def finalize_traversal_function(self, rewriter: PatternRewriter):
        """Complete the function and ensure it's correctly formed, e.g. returning proper results."""
        rewriter.insertion_point = InsertPoint.at_end(self.funcs.tt_op.body.block)

        while_op = self.state.traversal_op
        assert while_op is not None, "Could not find while loop in traversal function"

        # Clone and insert terminal operations, using the final qreg from while loop
        final_qreg = while_op.results[1]

        value_mapper = {}

        if hasattr(self.segments, "terminal_segment") and self.segments.terminal_segment.ops:
            for op in self.segments.terminal_segment.ops:
                if isinstance(op, func.ReturnOp):
                    # TODO: For return operation, we need to get the actual result from the
                    # traversal.
                    c0 = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
                    load_op = memref.LoadOp.get(self.stacks.folded_result, (c0,))
                    tensor_op = tensor.FromElementsOp.create(
                        operands=[load_op.results[0]],
                        result_types=[builtin.TensorType(builtin.Float64Type(), [])],
                    )
                    return_op = func.ReturnOp(*tensor_op.results)
                    for insert_op in (c0, load_op, tensor_op, return_op):
                        rewriter.insert_op(
                            insert_op, InsertPoint.at_end(self.funcs.tt_op.body.block)
                        )

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
            for resType in self.funcs.tt_op.function_type.outputs:
                assert isinstance(resType, builtin.TensorType)
                result = tensor.EmptyOp((), tensor_type=resType)
                result_vals.append(rewriter.insert(result))
            rewriter.insert(func.ReturnOp(*result_vals))

        for op in reversed(self.funcs.original_func_op.body.ops):
            rewriter.erase_op(op)

        call_op = func.CallOp(
            self.funcs.tt_op.sym_name.data,
            self.funcs.original_func_op.args,
            [builtin.TensorType(builtin.Float64Type(), [])],  # Single tensor<f64> output
        )

        return_op = func.ReturnOp(*call_op.results)

        for op in (call_op, return_op):
            rewriter.insert_op(op, InsertPoint.at_end(self.funcs.original_func_op.body.block))

        # remove simple_io function
        rewriter.erase_op(self.funcs.simple_io_func)

    def split_traversal_segments(self, func_op: func.FuncOp, rewriter: PatternRewriter):
        """Split the quantum function into segments separated by measure operations.

        Due to the pre-processing of the QNode (result in simple_io_func), we can assume the register is the only
        quantum value going between segments.
        """
        rewriter.insertion_point = InsertPoint.at_start(self.funcs.tt_op.body.block)

        # Ideally try to iterate over the function only once.
        op_iter = iter(func_op.body.ops)

        # Skip to the start of the first simulation segment.
        value_mapper = {}
        while (op := next(op_iter, None)) and not isinstance(op, quantum.AllocOp):
            rewriter.insert(op.clone(value_mapper))
        assert op is not None, "didn't find an alloc op"

        # clone the alloc op
        self.state.alloc_op = rewriter.insert(op.clone(value_mapper))

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
        self.segments.terminal_segment = terminal_segment

        # Generate new functions for each segment separated by a measure op.
        # We traverse them bottom up first to correctly determine the I/O of each segment.
        missing_input_values = set()
        self.populate_segment_io(terminal_segment, missing_input_values)
        for segment in reversed(quantum_segments):
            self.populate_segment_io(segment, missing_input_values)  # inplace

        # Create the traversal handling function
        # It is control the state transition of the tree traversal
        self.funcs.state_transition_func = self.create_state_transition_function(rewriter)

        # contains the inputs to the first segment + all MCM results
        func_args = set(func_op.args)
        missing_input_values = missing_input_values - func_args
        all_segment_io = [quantum_segments[0].reg_in, *func_args, *missing_input_values]
        values_as_io_index = {v: k for k, v in enumerate(all_segment_io)}
        for idx, segment in enumerate(quantum_segments):
            new_segment = self.clone_ops_into_func(segment, idx, rewriter)
            new_segment = self.additional_transform_for_segment(new_segment, rewriter)
            segment.fun = new_segment
            values_as_io_index.update(
                (x, i + len(all_segment_io)) for i, x in enumerate(segment.outputs)
            )
            all_segment_io.extend(segment.outputs)

        self.generate_function_table(all_segment_io, values_as_io_index, rewriter)

        # store some useful values for later
        self.segments.all_segment_io = all_segment_io
        self.segments.values_as_io_index = values_as_io_index

    def additional_transform_for_segment(
        self, new_segment: func.FuncOp, rewriter: PatternRewriter
    ) -> func.FuncOp:
        """Apply additional transformations to each segment function as needed."""

        # Placeholder for additional transformations per segment
        new_segment = self.segment_transformation_if_statement(new_segment, rewriter)

        return new_segment

    def segment_transformation_if_statement(
        self, new_segment: func.FuncOp, rewriter: PatternRewriter
    ) -> func.FuncOp:
        """Transform nested if-statements containing mid-circuit measurements (MCMs).

        This method handles the case where measurement operations are nested within conditional
        blocks. It identifies and relocates the "real" measurement operation (the actual quantum
        measurement) to ensure proper execution order within the tree-traversal simulation.

        The transformation addresses scenarios where:
        1. A measurement operation is wrapped in an outer if-statement (real_mcm_op)
        2. Another conditional if-statement contains MCM-related operations (current_if_op)
        3. These nested structures need to be reorganized for correct tree-traversal behavior

        Returns:
            The modified function operation with transformed if-statement structure. If no
            transformation is needed (e.g., no nested MCM structure found), returns the
            original segment unchanged.

        Implementation Details:
            - Searches for two distinct IfOps: one containing the real measurement and another
              marked with "contain_mcm" attribute
            - Moves the real measurement operation to the appropriate location within the
              inner if-statement's true branch
            - Updates quantum register and qubit references to maintain correct data flow
            - Removes the old measurement operation after replacement to avoid duplication
        """

        real_mcm_op = None
        current_if_op = None

        # Extract the IfOps containing measure operations
        for op in new_segment.body.walk():
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
                rewriter.erase_op(where_to_move_real_mcm)
            else:
                rewriter.erase_op(real_mcm_op)

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
                self.stacks.visited_stack_type,  # visited stack
                self.stacks.statevec_stack_type,  # state vector
                self.stacks.probs_stack_type,  # probs stack
                self.stacks.folded_result_type,  # folded result
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
            args = [
                branch_type,  # branch_type for postselect logic
                visited_stack,  # visited_stack for visited status
                statevec_stack,  # statevector stack for storing and restoring statevector
                probs_stack,  # probs stack for recording the probability before each segment
                folded_result,  # folded result
                statevec_size,  # statevec size for storing and restoring statevector
                io_args[0],  # quantum register
                *[io_args[values_as_io_index[value]] for value in segment.inputs],
            ]
            res_types = [quantum.QuregType()] + [res.type for res in segment.outputs]
            callOp = func.CallOp(self.quantum_segments[case].fun.sym_name.data, args, res_types)

            updated_results = list(io_args)
            updated_results[0] = callOp.results[0]
            for new_res, ref in zip(callOp.results[1:], segment.outputs):
                updated_results[values_as_io_index[ref]] = new_res

            yieldOp = scf.YieldOp(*updated_results)

            for op in (callOp, yieldOp):
                rewriter.insert_op(op, InsertPoint.at_end(switchOp.case_regions[case].block))

    def clone_ops_into_func(
        self, segment: ProgramSegment, counter: int, rewriter: PatternRewriter
    ):  # pylint: disable=no-member
        """Clone a set of ops into a new function."""
        op_list, input_vals, output_vals = segment.ops, segment.inputs, segment.outputs
        input_vals = [segment.reg_in] + input_vals
        output_vals = [segment.reg_out] + output_vals
        if not op_list:
            return None

        fun_type = builtin.FunctionType.from_lists(
            [
                builtin.IndexType(),  # branch_type
                self.stacks.visited_stack_type,
                self.stacks.statevec_stack_type,
                self.stacks.probs_stack_type,
                self.stacks.folded_result_type,
                builtin.IndexType(),  # statevec_size
                *[arg.type for arg in input_vals],
            ],
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

    def create_state_transition_function(
        self, rewriter: PatternRewriter
    ):  # pylint: disable=too-many-statements,no-member
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
                self.stacks.visited_stack_type,
                self.stacks.statevec_stack_type,
                self.stacks.probs_stack_type,
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

    def initialize_traversal_attrs(self, rewriter: PatternRewriter):
        """Create data structures in the IR required for the dynamic tree traversal."""
        rewriter.insertion_point = InsertPoint.at_end(self.funcs.tt_op.body.block)

        # get the qubit count
        if self.state.alloc_op.nqubits:
            qubit_count = arith.IndexCastOp(self.state.alloc_op.nqubits, builtin.IndexType())
        else:
            qubit_count = arith.ConstantOp.from_int_and_width(
                self.state.alloc_op.nqubits_attr.value, builtin.IndexType()
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
        statevec_stack = memref.AllocOp(
            (tree_depth, statevec_size), (), self.stacks.statevec_stack_type
        )

        # probabilities for each branch are tracked here
        probs_stack = memref.AllocOp((updated_tree_depth,), (), self.stacks.probs_stack_type)
        c1_f64 = arith.ConstantOp(builtin.FloatAttr(1.0, type=builtin.f64))
        length_probs_stack = arith.AddiOp(tree_depth.results[0], c1)
        init_probs_stack_ops = initialize_memref_with_value(
            probs_stack, c1_f64.results[0], length_probs_stack
        )

        # For the current path, we track whether a node is:
        #  - unvisited: 0
        #  - visited down the left branch: 1
        #  - finished: 2
        visited_stack = memref.AllocOp((tree_depth,), (), self.stacks.visited_stack_type)

        # initialize with 0
        c0_i8 = arith.ConstantOp.from_int_and_width(0, builtin.i8)
        init_visited_stack_ops = initialize_memref_with_value(
            visited_stack, c0_i8.results[0], tree_depth.results[0]
        )

        c0_f64 = arith.ConstantOp(builtin.FloatAttr(0.0, type=builtin.f64))
        length_folded_result = arith.AddiOp(tree_depth.results[0], c1)
        folded_result = memref.AllocOp((length_folded_result,), (), self.stacks.folded_result_type)
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
        self.state.tree_depth = tree_depth.result
        self.state.statevec_size = statevec_size.result
        self.stacks.statevec_stack = statevec_stack.results[0]
        self.stacks.probs_stack = probs_stack.results[0]
        self.stacks.visited_stack = visited_stack.results[0]
        self.stacks.folded_result = folded_result.results[0]

    def generate_traversal_code(
        self, rewriter: PatternRewriter
    ):  # pylint: disable=too-many-branches,too-many-statements,no-member
        """Create the traversal code of the quantum simulation tree."""
        rewriter.insertion_point = InsertPoint.at_end(self.funcs.tt_op.body.block)

        func_args = list(self.funcs.simple_io_func.args)
        tt_args = list(self.funcs.tt_op.args)
        func_to_tt_mapping = dict(zip(func_args, tt_args))

        # loop instruction
        depth_init = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        segment_io_types = [val.type for val in self.segments.all_segment_io]
        segment_io_inits = []
        for val in self.segments.all_segment_io:
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

        self.state.traversal_op = traversalOp

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

        visited_status = memref.LoadOp.get(self.stacks.visited_stack, (current_depth,))
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
                self.stacks.visited_stack,
                self.stacks.statevec_stack,
                self.stacks.probs_stack,
                self.stacks.folded_result,
                self.state.statevec_size,
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

        load_probs_op = memref.LoadOp.get(self.stacks.probs_stack, (current_depth,))
        load_parent_folded_op = memref.LoadOp.get(self.stacks.folded_result, (current_depth,))
        load_folded_op = memref.LoadOp.get(self.stacks.folded_result, (probs_stack_idx,))

        mulf_op = arith.MulfOp(load_probs_op.results[0], load_folded_op.results[0])
        addf_op = arith.AddfOp(mulf_op.results[0], load_parent_folded_op.results[0])
        store_op = memref.StoreOp.get(
            addf_op.results[0], self.stacks.folded_result, (current_depth,)
        )

        # folded_result[depth + 1] = 0
        f0 = arith.ConstantOp(builtin.FloatAttr(0.0, type=builtin.f64))
        store_op_2 = memref.StoreOp.get(f0, self.stacks.folded_result, (probs_stack_idx,))

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
        c0 = arith.ConstantOp.from_int_and_width(0, self.stacks.visited_stack.type.element_type)
        update_visited = memref.StoreOp.get(c0, self.stacks.visited_stack, (current_depth,))

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

    def initialize_values_from_types(
        self, types, rewriter: PatternRewriter
    ):  # pylint: disable=too-many-branches
        """Generate dummy values for the provided types. Quantum types are treated specially and
        will make use of the quantum.AllocOp reference collected at an earlier stage."""

        # TODO: handling quantum dummy values can be tricky, let's try this for now
        qreg_stub = self.state.alloc_op.results[0]
        qubit_stub = quantum.ExtractOp(qreg_stub, 0)
        rewriter.insert_op(qubit_stub, InsertPoint.at_end(self.funcs.tt_op.body.block))
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
                    ops.append(self.state.alloc_op.results[0])

        for op in need_insert_ops:
            rewriter.insert_op(op, InsertPoint.at_end(self.funcs.tt_op.body.block))

        return ops

    def check_if_leaf(self, current_depth: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        """Verify whether we've hit the bottom of the tree, and perform update actions."""
        assert isinstance(current_depth.owner, Block)
        ip_backup = rewriter.insertion_point
        rewriter.insertion_point = InsertPoint.at_start(current_depth.owner)

        # if instruction
        hit_leaf = arith.CmpiOp(current_depth, self.state.tree_depth, "eq")
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
        assert self.state.alloc_op.nqubits_attr is not None, "nqubits of alloc should be a constant"
        for i in range(self.state.alloc_op.nqubits_attr.value.data):
            extract_op = quantum.ExtractOp(qreg, i)
            qubits.append(extract_op.results[0])
            all_ops.append(extract_op)

        set_state_op = quantum.SetStateOp.create(
            operands=[statevec.results[0], *qubits],
            result_types=[quantum.QubitType() for _ in qubits],
        )
        all_ops.append(set_state_op)

        last_insert_op = None
        for i in range(self.state.alloc_op.nqubits_attr.value.data):
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
            subview = statevec_stack[depth, :]  # Create 1D view
            quantum.state(qreg, state_in=subview)
        ```
        """
        # Create a 1D subview of statevec_stack[depth, :]
        state_memref_type = builtin.MemRefType(
            builtin.ComplexType(builtin.f64),
            (builtin.DYNAMIC_INDEX,),
            builtin.StridedLayoutAttr([1], None),
        )

        statevec_subview = memref.SubviewOp.get(
            statevec_stack,
            (depth, 0),  # offsets: start at [depth, 0]
            (1, statevec_size),  # sizes: 1 row, statevec_size columns
            (1, 1),  # strides
            state_memref_type,  # Result is 1D memref
        )

        state_comp_basis_op = quantum.ComputationalBasisOp(
            operands=([], [qreg]), result_types=[quantum.ObservableType()]  # (qubits, qreg)
        )

        cast_op = arith.IndexCastOp(statevec_size, builtin.i64)

        # StateOp with state_in writes directly to the memref
        state_op = quantum.StateOp(
            operands=[
                state_comp_basis_op.results[0],
                cast_op.results[0],
                statevec_subview.results[0],
            ],
            result_types=[None],
        )

        for op in (state_comp_basis_op, cast_op, statevec_subview, state_op):
            rewriter.insert_op(op, InsertPoint.at_end(insert_block))
