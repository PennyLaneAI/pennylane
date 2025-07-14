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

"""This file contains the implementation of the convert_to_mbqc_formalism transform,
written using xDSL."""

from dataclasses import dataclass

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin, func, memref
from xdsl.dialects.scf import ForOp, IfOp, WhileOp
from xdsl.rewriter import InsertPoint

from ..quantum_dialect import AllocOp
from .api import compiler_transform


@dataclass(frozen=True)
class ConvertToMBQCFormalismPass(passes.ModulePass):
    """Pass that converts gates in the MBQC gate set to the MBQC formalism."""

    name = "convert-to-mbqc-formalism"

    # pylint: disable=arguments-renamed,no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the convert-to-mbqc-formalism pass."""
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier(
                [
                    PreAllocateAuxWiresPattern(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(module)


convert_to_mbqc_formalism = compiler_transform(ConvertToMBQCFormalismPass)


class PreAllocateAuxWiresPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for pre-allocate 13 more aux wires."""

    # pylint: disable=no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, root: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter
    ):  # pylint: disable=arguments-differ, cell-var-from-loop
        """Match and rewrite for pre-allocate 13 more aux wires."""

        # Replace AllocOp with 13 more qubits
        for region in root.regions:
            for op in region.ops:
                if isinstance(op, AllocOp):
                    # Note that as of 25.07.14, generic assembly format is used
                    # the number of qubits is assigned to the nqubits_attr.
                    num_wires_int = op.properties["nqubits_attr"].value.data
                    # Note that 13 more qubits is added and this is subject to change
                    # once the dynamic qubit allocate and deallocate is supported.
                    total_num_wires_int = num_wires_int + 13
                    new_op = AllocOp(total_num_wires_int)
                    rewriter.replace_op(op, new_op)

                    qubit_mgr = memref.AllocOp.get(builtin.i64, shape=[total_num_wires_int])
                    rewriter.insert_op(qubit_mgr, insertion_point=InsertPoint.after(new_op))
                    prev_op = qubit_mgr
                    for i in range(total_num_wires_int):
                        const_op = arith.ConstantOp.from_int_and_width(i, builtin.IndexType())
                        rewriter.insert_op(const_op, insertion_point=InsertPoint.after(prev_op))
                        # index_op = arith.ConstantOp.from_int_and_width(i, builtin.IndexType())
                        # rewriter.insert_op(index_op, insertion_point=InsertPoint.after(const_op))
                        store_op = memref.StoreOp.get(const_op, qubit_mgr, const_op)
                        rewriter.insert_op(store_op, insertion_point=InsertPoint.after(const_op))
                        prev_op = store_op


# class AddQubitMapPattern(
#     pattern_rewriter.RewritePattern
# ):  # pylint: disable=too-few-public-methods
#     """Pass for adding the QubitMgr logic."""

#     # pylint: disable=no-self-use
#     @pattern_rewriter.op_type_rewrite_pattern
#     def match_and_rewrite(
#         self, root: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter
#     ):  # pylint: disable=arguments-differ, cell-var-from-loop
#         """Match and rewrite for pre-allocate 13 more aux wires."""

#         # Replace AllocOp with 13 more qubits
#         for region in root.regions:
#             for op in region.ops:
#                 if isinstance(op, AllocOp):
#                     # Initialize
#                     num_wires_int = op.properties["nqubits_attr"].value.data
#                     qubit_mgr = memref.AllocOp.get(builtin.IntegerAttr(), 0, (num_wires_int))
#                     rewriter.insert_op(qubit_mgr, insertion_point=InsertPoint.after(op))
#                     prev_op = qubit_mgr
#                     for i in range(num_wires_int):
#                         const_op = arith.ConstantOp.from_int_and_width(i, builtin.IndexType())
#                         rewriter.insert_op(const_op, insertion_point=InsertPoint.after(prev_op))
#                         index_op = arith.ConstantOp.from_int_and_width(i, builtin.IndexType())
#                         rewriter.insert_op(index_op, insertion_point=InsertPoint.after(const_op))
#                         store_op = memref.StoreOp.get(const_op, qubit_mgr, index_op)
#                         prev_op = store_op
