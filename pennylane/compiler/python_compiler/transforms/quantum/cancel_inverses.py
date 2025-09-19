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
"""This file contains the implementation of the cancel_inverses transform,
written using xDSL."""

from dataclasses import dataclass

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func
from xdsl.ir import Operation

from ...dialects.quantum import CustomOp
from ...pass_api import compiler_transform

self_inverses = [
    "Identity",
    "Hadamard",
    "PauliX",
    "PauliY",
    "PauliZ",
    "CNOT",
    "CZ",
    "CY",
    "CH",
    "SWAP",
    "Toffoli",
    "CCZ",
]


def _can_cancel(op: CustomOp, next_op: Operation) -> bool:
    if isinstance(next_op, CustomOp):
        if op.gate_name.data == next_op.gate_name.data:
            if (
                op.out_qubits == next_op.in_qubits
                and op.out_ctrl_qubits == next_op.in_ctrl_qubits
                and op.in_ctrl_values == next_op.in_ctrl_values
            ):
                return True

    return False


class IterativeCancelInversesPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for iteratively cancelling consecutive self-inverse gates."""

    # pylint: disable=no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        """Implementation of rewriting FuncOps that may contain operations corresponding to
        self-inverse gates."""
        for op in funcOp.body.walk():

            while isinstance(op, CustomOp) and op.gate_name.data in self_inverses:

                next_user = None
                for use in op.results[0].uses:
                    user = use.operation
                    if _can_cancel(op, user):
                        next_user: CustomOp = user
                        break

                if next_user is None:
                    break

                for q1, q2 in zip(op.in_qubits, next_user.out_qubits, strict=True):
                    rewriter.replace_all_uses_with(q2, q1)
                for cq1, cq2 in zip(op.in_ctrl_qubits, next_user.out_ctrl_qubits, strict=True):
                    rewriter.replace_all_uses_with(cq2, cq1)
                rewriter.erase_op(next_user)
                rewriter.erase_op(op)
                op = op.in_qubits[0].owner


@dataclass(frozen=True)
class IterativeCancelInversesPass(passes.ModulePass):
    """Pass for iteratively cancelling consecutive self-inverse gates."""

    name = "xdsl-cancel-inverses"

    # pylint: disable=no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the iterative cancel inverses pass."""
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([IterativeCancelInversesPattern()])
        ).rewrite_module(module)


iterative_cancel_inverses_pass = compiler_transform(IterativeCancelInversesPass)
