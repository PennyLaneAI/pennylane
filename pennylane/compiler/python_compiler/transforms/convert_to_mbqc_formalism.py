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
from xdsl.dialects import arith, builtin, func, memref, vector
from xdsl.dialects.scf import ForOp, IfOp, WhileOp
from xdsl.rewriter import InsertPoint

from ..quantum_dialect import AllocOp, CustomOp, ExtractOp, InsertOp
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
                    ConvertToMBQCFormalismPattern(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(module)


convert_to_mbqc_formalism = compiler_transform(ConvertToMBQCFormalismPass)


class ConvertToMBQCFormalismPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for pre-allocate 13 more aux wires."""

    # pylint: disable=no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, root: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter
    ):  # pylint: disable=arguments-differ, cell-var-from-loop
        """Match and rewrite for pre-allocate 13 more aux wires."""

        num_wires_int = 0
        registers_state = None
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
                    registers_state = new_op.results[0]
                elif isinstance(op, CustomOp) and op.gate_name.data in [
                    "PauliX",
                ]:
                    current_op = op
                    # TODO
                    # Convert a gate operation in the standard circuit model

                    # NOTE: The following logic mimic the QubitMgr class defined in the `ftqc.utils` module
                    # 1. Extract the target wire and the result wires in the auxiliary registers
                    target_qubits_index = op.results[0].index
                    target_qubit = ExtractOp(registers_state, target_qubits_index)
                    rewriter.insert_op(target_qubit, insertion_point=InsertPoint.after(current_op))
                    current_op = target_qubit
                    result_qubits_index = target_qubits_index + num_wires_int
                    res_qubit = ExtractOp(registers_state, result_qubits_index)
                    rewriter.insert_op(res_qubit, insertion_point=InsertPoint.after(current_op))
                    current_op = res_qubit

                    # 2. Swap the target register and the result register
                    new_target_qubit = InsertOp(registers_state, result_qubits_index, target_qubit)
                    rewriter.insert_op(
                        new_target_qubit, insertion_point=InsertPoint.after(current_op)
                    )
                    registers_state = new_target_qubit.results[0]

                    current_op = new_target_qubit

                    new_res_qubit = InsertOp(registers_state, target_qubits_index, res_qubit)
                    rewriter.insert_op(new_res_qubit, insertion_point=InsertPoint.after(current_op))
