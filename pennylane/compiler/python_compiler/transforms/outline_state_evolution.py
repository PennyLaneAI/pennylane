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

"""This module contains the implementation of the outline_state_evolution transform,
written using xDSL.
"""

from dataclasses import dataclass

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func

from .api import compiler_transform


@dataclass(frozen=True)
class OutlineStateEvolutionPass(passes.ModulePass):
    """Pass that outlines the operations that evolve the quantum state of a device.

    "Outlining" refers to the process of moving sequences operations into a separate function and
    replacing the original sequence of operations with a call to this function.
    """

    name = "outline-state-evolution"

    # pylint: disable=arguments-renamed,no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the outline-state-evolution pass."""
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([OutlineStateEvolutionPattern()])
        ).rewrite_module(module)


outline_state_evolution_pass = compiler_transform(OutlineStateEvolutionPass)


class OutlineStateEvolutionPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """Rewrite pattern for the ``outline-state-evolution`` transform, outlines the operations that
    evolve the quantum state of a device.."""

    # pylint: disable=no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter
    ):  # pylint: disable=arguments-differ
        """TODO"""
        raise NotImplementedError("OutlineStateEvolutionPattern not yet implemented!")
