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
This module contains a rewrite pattern that unrolls static loops containing mid-circuit measurements.
"""

from xdsl.dialects import arith, builtin, scf
from xdsl.ir import BlockArgument, Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.rewriter import InsertPoint

from pennylane.compiler.python_compiler.dialects import quantum, stablehlo
from pennylane.exceptions import CompileError


class UnrollLoopPattern(RewritePattern):
    """A rewrite pattern that unrolls scf.ForOps containing measurement-controlled
    operations into separate branches for each operator."""

    def __init__(self):
        """Initialize UnrollLoopPattern."""
        self.needs_unroll: bool = False
        self.for_loop_to_unroll: list[scf.ForOp] = []

    def match_and_rewrite(
        self, op: Operation, rewriter: PatternRewriter
    ) -> None:  # pylint: disable=arguments-differ
        """Unroll nested scf.ForOps into separate branches for each operator."""

        self.needs_unroll = self.detect_mcm_in_loop_ops(op)

        if not self.needs_unroll:
            return

        # for _for_loop in self.for_loop_to_unroll:
        #     print_mlir(_for_loop, "ForOp to unroll:")

        neasted_loop_to_unroll = []

        # Looking for parent loops of the loops to unroll
        for _for_op in self.for_loop_to_unroll:

            neasted_loop_to_unroll.append(_for_op)

            parent_op = _for_op.parent_op()
            while parent_op != op:
                if isinstance(parent_op, scf.ForOp):

                    # Keep the most outer loop only once and later unroll from outer to inner
                    if parent_op in neasted_loop_to_unroll:
                        neasted_loop_to_unroll.remove(parent_op)

                    neasted_loop_to_unroll.append(parent_op)

                parent_op = parent_op.parent_op()

        for _for_op in neasted_loop_to_unroll:
            # print_mlir(_for_op, "Last ForOp to unroll:")
            self.unroll_loop(_for_op, rewriter)

    def detect_mcm_in_loop_ops(self, op: Operation) -> bool:
        """Detect if there are mid-circuit measurement operations inside ForOps."""
        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.ForOp):
                for inner_op in current_op.body.ops:
                    if isinstance(inner_op, quantum.MeasureOp):
                        self.for_loop_to_unroll.append(current_op)

        return len(self.for_loop_to_unroll) > 0

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

        if not (lb_found and ub_found and step_found):
            raise CompileError(
                "Tree Traversal requires loops containing mid-circuit measurements to have "
                "constant bounds and step values known at compile time. "
                "The loop being compiled has dynamic bounds that cannot be determined statically. "
                "To resolve this issue, ensure that loop bounds are literals or compile-time constants "
                "(e.g., use `for i in range(5)` instead of `for i in range(n)` where n is a runtime variable)."
            )

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

        if inner_op_clone is not None or len(range(lb, ub, step)) == 0:
            for op_res, iter_ssa in zip(op.results, iter_args):
                op_res.replace_by(iter_ssa)

            op.detach()
            op.erase()
