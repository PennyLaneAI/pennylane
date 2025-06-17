# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
from xdsl.rewriter import InsertPoint

from ..quantum_dialect import GlobalPhaseOp
from .utils import xdsl_transform


class CombineGlobalPhasesPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for removing ~:class:`~pennylane.GlobalPhase` gates in the circuit (if exists) and adding
    at the end of list of operations with its phase being a total global phase computed as the algebraic sum of
    all global phases in the original circuit."""

    # pylint: disable=no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter
    ):  # pylint: disable=arguments-differ
        """Implementation of rewriting FuncOps that may contain operations corresponding to
        consecutive composable rotations."""
        phi = None
        for op in funcOp.body.walk():
                if not isinstance(op, GlobalPhaseOp):
                    continue
                if phi is None:
                    phi = op.operands[0]
                else:
                    addOp = arith.AddfOp(op.operands[0], phi)
                    phi = addOp.result
                rewriter.erase_op(op)
        if phi:
            new_op = GlobalPhaseOp(params=phi)
            rewriter.insert_op(new_op, InsertPoint.at_end(funcOp.body.parent_block()))


@dataclass(frozen=True)
class CombineGlobalPhasesPass(passes.ModulePass):
    """Pass for combining global phase gates."""

    name = "combine-global-phases"

    # pylint: disable=arguments-renamed,no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the combination of global phase gates pass."""
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([CombineGlobalPhasesPattern()])
        ).rewrite_module(module)


combine_global_phases_pass = xdsl_transform(CombineGlobalPhasesPass)
