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

TODO
~~~~

This file is under heavy development and many parts are incomplete.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func
from xdsl.ir import Attribute, Region

from ..dialects import quantum
from .api import compiler_transform

_OBSERVABLE_OPS = (
    quantum.ComputationalBasisOp,
    quantum.HamiltonianOp,
    quantum.HermitianOp,
    quantum.NamedObsOp,
    quantum.TensorOp,
)


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
        walker = pattern_rewriter.PatternRewriteWalker(OutlineStateEvolutionPattern())
        walker.rewrite_module(module)


outline_state_evolution_pass = compiler_transform(OutlineStateEvolutionPass)


class OutlineStateEvolutionPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """Rewrite pattern for the ``outline-state-evolution`` transform, outlines the operations that
    evolve the quantum state of a device.."""

    # pylint: disable=no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, func_op: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter
    ):  # pylint: disable=arguments-differ
        """TODO"""
        state_evolution_region = Region()

        for op in reversed(func_op.body.ops):
            if any(isinstance(result_type, quantum.QubitType) for result_type in op.result_types):
                # This op evolves the state
                op._outline_flag = True

        breakpoint()

        evolve_state_subroutine_op = func.FuncOp(
            name="evolve_quantum_state",
            function_type=(
                (quantum.QuregType, *func_op.function_type.inputs),  # Input types
                (quantum.QuregType),  # Return types
            ),
            region=state_evolution_region,
            visibility="private",
        )
