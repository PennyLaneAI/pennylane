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

"""This file contains the implementation of the combine_global_phases transform,
written using xDSL."""

from dataclasses import dataclass

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin, func
from xdsl.dialects.scf import ForOp, IfOp, WhileOp
from xdsl.rewriter import InsertPoint

from ...dialects.quantum import GlobalPhaseOp
from ...pass_api import compiler_transform


class CombineGlobalPhasesPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for combining all :class:`~pennylane.GlobalPhase` gates within the same region
    at the last global phase gate."""

    # pylint: disable=no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, root: func.FuncOp | IfOp | ForOp | WhileOp, rewriter: pattern_rewriter.PatternRewriter
    ):  # pylint: disable=cell-var-from-loop
        """Match and rewrite for the combine-global-phases pattern acting on functions or
        control-flow blocks containing GlobalPhase operations.
        """

        for region in root.regions:
            phi = None
            global_phases = []
            for op in region.ops:
                if isinstance(op, GlobalPhaseOp):
                    global_phases.append(op)

            if len(global_phases) < 2:
                continue

            prev = global_phases[0]
            phi_sum = prev.operands[0]
            for current in global_phases[1:]:
                phi = current.operands[0]
                addOp = arith.AddfOp(phi, phi_sum)
                rewriter.insert_op(addOp, InsertPoint.before(current))
                phi_sum = addOp.result

                rewriter.erase_op(prev)
                prev = current

            prev.operands[0].replace_by_if(phi_sum, lambda use: use.operation == prev)
            rewriter.notify_op_modified(prev)


@dataclass(frozen=True)
class CombineGlobalPhasesPass(passes.ModulePass):
    """Pass that combines all global phases within a region into the last global phase operation
    within the region.
    """

    name = "combine-global-phases"

    # pylint: disable=no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the combine-global-phases pass."""
        pattern_rewriter.PatternRewriteWalker(
            CombineGlobalPhasesPattern(),
            apply_recursively=False,
        ).rewrite_module(module)


combine_global_phases_pass = compiler_transform(CombineGlobalPhasesPass)
