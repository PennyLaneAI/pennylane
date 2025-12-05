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

"""
Implementation of the Tree-Traversal MCM simulation method as an xDSL transform in Catalyst.
This module contains a rewrite pattern that partitions if statements containing mid-circuit measurements.
"""

from itertools import chain
from typing import List, Tuple, Type

from xdsl.dialects import arith, builtin, func, scf
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint

from pennylane.compiler.python_compiler.dialects import quantum

from .tree_traversal_utils_tmp import print_mlir, print_ssa_values


class IfOperatorPartitioningPattern(RewritePattern):
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
        flat_if = self.detect_mcm_in_if_ops_2(op)

        if not flat_if:
            return

        print_mlir(op, "Before IfOp Partitioning:")
        # Split IfOps into only true branches
        self.split_if_ops(op, rewriter)

        print_mlir(op, "After IfOp Splitting:")

        # Flatten nested IfOps
        self.flatten_nested_IfOps(op, rewriter)

        # Adding fake MeasureOp before if Op with the attribute contain_mcm = "true"
        self.adding_fake_measureOp(op, rewriter)

    def __init__(self):
        self.module: builtin.ModuleOp = None
        self.original_func_op: func.FuncOp = None
        self.if_op_with_mcm: List[scf.IfOp] = []

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
                qreg_if_op[0].replace_by_if(
                    q_insert.results[0], lambda use: use.operation not in [q_extract, q_insert]
                )

    def detect_mcm_in_if_ops(self, op: scf.IfOp) -> bool:
        """Detect if there are measurement-controlled operations inside IfOps."""
        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp):
                #  Check if there are mid-circuit measurements inside the IfOp
                for inner_op in current_op.true_region.ops:
                    if isinstance(inner_op, quantum.MeasureOp):
                        return True
                for inner_op in current_op.false_region.ops:
                    if isinstance(inner_op, quantum.MeasureOp):
                        return True
        return False

    def detect_mcm_in_if_ops_2(self, op: scf.IfOp) -> bool:
        """Detect if there are measurement-controlled operations inside IfOps."""
        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, quantum.MeasureOp):
                self.collect_all_parent_ifs(current_op, stop=op)

        return len(self.if_op_with_mcm) > 0


    def collect_all_parent_ifs(self, op: Operation, stop: Operation) -> None:
        """Collect all parent scf.IfOps of a given operation up to a stop operation."""
        while (op := op.parent_op()) and op != stop:
            if isinstance(op, scf.IfOp):
                if op in self.if_op_with_mcm:
                    self.if_op_with_mcm.remove(op)
                self.if_op_with_mcm.append(op)

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
        deepest_ops_with_depth: List[IfOperatorPartitioningPattern.IfOpWithDepth] = [(None, 0)]

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
            holder_returns: dict[scf.IfOp, scf.IfOp] = {}

            for inner_op in nested_if_ops:

                where_to_insert, outer_if_op = self.move_inner_if_op_2_outer(
                    inner_op,
                    outer_if_op,
                    new_outer_if_op_output,
                    new_outer_if_op_output_types,
                    where_to_insert,
                    holder_returns,
                    rewriter,
                )

            # detach and erase old outer if op
            for hold_op in holder_returns:
                rewriter.erase_op(hold_op)

    def move_inner_if_op_2_outer(  # pylint: disable=too-many-branches,too-many-arguments,too-many-statements,no-member
        self,
        inner_op: scf.IfOp,
        outer_if_op: scf.IfOp,
        new_outer_if_op_output: list[SSAValue],
        new_outer_if_op_output_types: list[Type],
        where_to_insert: scf.IfOp,
        holder_returns: dict[scf.IfOp, scf.IfOp],
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
            conditional = holder_returns[hold_return].results[return_index]
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
            rewriter.erase_op(inner_op)
        else:
            holder_returns[inner_op] = new_inner_op
            update_unused_cond = False
            unused_op = None
            for op in holder_returns:
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

        rewriter.erase_op(outer_if_op)

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

    def split_if_ops(self, op: func.FuncOp, rewriter: PatternRewriter) -> None:
        """Split all scf.IfOps containing mid-circuit measurement operations into separate branches."""

        for _if_op in self.if_op_with_mcm:
            self.split_if_op(_if_op, rewriter)


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
                    rewriter.erase_op(if_op)

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
