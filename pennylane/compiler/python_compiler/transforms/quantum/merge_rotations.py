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
from xdsl.dialects import arith, builtin, func
from xdsl.ir import Operation
from xdsl.rewriter import InsertPoint

from ...dialects.quantum import CustomOp
from ...pass_api import compiler_transform

# Can handle all composible rotations except Rot... for now
composable_rotations = [
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
    "CRX",
    "CRY",
    "CRZ",
    "ControlledPhaseShift",
    "IsingXX",
    "IsingYY",
    "IsingXY",
    "IsingZZ",
    "MultiRZ",
    "SingleExcitation",
    "DoubleExcitation",
    "SingleExcitationMinus",
    "SingleExcitationPlus",
    "DoubleExcitationMinus",
    "DoubleExcitationPlus",
    "OrbitalRotation",
]


def _can_merge(op: CustomOp, next_op: Operation) -> bool:
    if isinstance(next_op, CustomOp):
        if op.gate_name.data == next_op.gate_name.data:
            if (
                op.out_qubits == next_op.in_qubits
                and op.out_ctrl_qubits == next_op.in_ctrl_qubits
                and op.in_ctrl_values == next_op.in_ctrl_values
            ):
                return True

    return False


class MergeRotationsPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for merging consecutive composable rotations."""

    # pylint: disable=no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        """Implementation of rewriting FuncOps that may contain operations corresponding to
        consecutive composable rotations."""
        for op in funcOp.body.walk():
            if not isinstance(op, CustomOp):
                continue

            gate_name = op.gate_name.data
            if gate_name not in composable_rotations:
                continue

            param = op.operands[0]
            while True:
                for use in op.results[0].uses:
                    user = use.operation
                    if _can_merge(op, user):
                        next_user = user
                        break
                else:
                    # No next_user was set because no user could be merged. Go to next op
                    break

                for q1, q2 in zip(op.in_qubits, op.out_qubits, strict=True):
                    rewriter.replace_all_uses_with(q2, q1)
                for cq1, cq2 in zip(op.in_ctrl_qubits, op.out_ctrl_qubits, strict=True):
                    rewriter.replace_all_uses_with(cq2, cq1)

                # Whether op is adjoint will determine adjoint of new_op, and how to combine angles
                is_adjoint = getattr(op, "adjoint", False)
                rewriter.erase_op(op)

                next_param = next_user.operands[0]
                next_is_adjoint = getattr(next_user, "adjoint", False)
                if is_adjoint == next_is_adjoint:
                    # If both op and next_user are adjoint, or neither is, we add angles
                    combOp = arith.AddfOp(param, next_param)
                else:
                    # If one of op and next_user is adjoint, we subtract angles
                    combOp = arith.SubfOp(param, next_param)
                rewriter.insert_op(combOp, InsertPoint.before(next_user))
                param = combOp.result

                new_op = CustomOp(
                    in_qubits=next_user.in_qubits,
                    gate_name=next_user.gate_name,
                    params=(param,),
                    in_ctrl_qubits=next_user.in_ctrl_qubits,
                    in_ctrl_values=next_user.in_ctrl_values,
                    adjoint=is_adjoint,
                )
                rewriter.replace_op(next_user, new_op)
                op = new_op


@dataclass(frozen=True)
class MergeRotationsPass(passes.ModulePass):
    """Pass for merging consecutive composable rotation gates."""

    name = "xdsl-merge-rotations"

    # pylint: disable=no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the merge rotations pass."""
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([MergeRotationsPattern()])
        ).rewrite_module(module)


merge_rotations_pass = compiler_transform(MergeRotationsPass)
