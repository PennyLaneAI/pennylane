# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file contains the implementation of the cancel_inverses transform,
written using xDSL."""

from dataclasses import dataclass

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func

from .quantum_dialect import CustomOp

self_inverses = ("PauliZ", "PauliX", "PauliY", "Hadamard", "Identity")


class DeepCancelInversesSingleQubitPattern(pattern_rewriter.RewritePattern):
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        """Deep Cancel for Self Inverses"""
        for op in funcOp.body.walk():

            while isinstance(op, CustomOp) and op.gate_name.data in self_inverses:

                next_user = None
                for use in op.results[0].uses:
                    user = use.operation
                    if isinstance(user, CustomOp) and user.gate_name.data == op.gate_name.data:
                        next_user = user
                        break

                if next_user is None:
                    break

                rewriter._replace_all_uses_with(next_user.results[0], op.in_qubits[0])
                rewriter.erase_op(next_user)
                rewriter.erase_op(op)
                op = op.in_qubits[0].owner


@dataclass(frozen=True)
class DeepCancelInversesSingleQubitPass(passes.ModulePass):
    name = "deep-cancel-inverses-single-qubit"

    def apply(self, ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([DeepCancelInversesSingleQubitPattern()])
        ).rewrite_module(module)
