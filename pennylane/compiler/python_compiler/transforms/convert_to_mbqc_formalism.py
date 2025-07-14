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
from xdsl.dialects import builtin, func
from xdsl.dialects.scf import ForOp, IfOp, WhileOp

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
            ConvertToMBQCFormalismPattern(),
            apply_recursively=False,
        ).rewrite_module(module)


convert_to_mbqc_formalism = compiler_transform(ConvertToMBQCFormalismPass)


class ConvertToMBQCFormalismPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for converting gates defined in the MBQC gate set to the MBQC formalism, while keeping the program structure."""

    # pylint: disable=no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, root: func.FuncOp | IfOp | ForOp | WhileOp, rewriter: pattern_rewriter.PatternRewriter
    ):  # pylint: disable=arguments-differ, cell-var-from-loop
        """Match and rewrite for converting to the MBQC formalism."""

        # Replace AllocOp with 13 more qubits
        for region in root.regions:
            for op in region.ops:
                if isinstance(op, AllocOp):
                    # Note that as of 25.07.14, generic assembly format is used
                    # the number of qubits is assigned to the nqubits_attr.
                    num_wires_int = op.properties["nqubits_attr"].value.data
                    # Note that 13 more qubits is added and this is subject to change
                    # once the dynamic qubit allocate and deallocate is supported.
                    total_num_wires = num_wires_int + 13
                    new_op = AllocOp(total_num_wires)
                    rewriter.replace_op(op, new_op)

        # Mimic the wire/aux_wire to physical register mapping with memref

        # for region in root.regions:
        #     phi = None
        #     global_phases = []
        #     for op in region.ops:
        #         if isinstance(op, GlobalPhaseOp):
        #             global_phases.append(op)

        #     if len(global_phases) < 2:
        #         continue

        #     prev = global_phases[0]
        #     phi_sum = prev.operands[0]
        #     for current in global_phases[1:]:
        #         phi = current.operands[0]
        #         addOp = arith.AddfOp(phi, phi_sum)
        #         rewriter.insert_op(addOp, InsertPoint.before(current))
        #         phi_sum = addOp.result

        #         rewriter.erase_op(prev)
        #         prev = current

        #     prev.operands[0].replace_by_if(phi_sum, lambda use: use.operation == prev)
        #     rewriter.notify_op_modified(prev)
