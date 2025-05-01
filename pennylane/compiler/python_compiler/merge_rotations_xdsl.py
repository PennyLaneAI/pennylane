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

"""This file contains the implementation of the merge_rotations transform,
written using xDSL."""

from dataclasses import dataclass

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func

from .quantum_dialect import CustomOp


class MergeRotationsSingleQubitPattern(pattern_rewriter.RewritePattern):
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):

        pass


@dataclass(frozen=True)
class MergeRotationsSingleQubitPass(passes.ModulePass):
    name = "merge-rotations-single-qubit"

    def apply(self, ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([MergeRotationsSingleQubitPattern()])
        ).rewrite_module(module)
