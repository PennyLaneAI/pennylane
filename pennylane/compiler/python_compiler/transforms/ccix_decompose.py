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
from xdsl.rewriter import InsertPoint

from pennylane import ops

from ..dialects.quantum import CustomOp
from .api import compiler_transform
from .api.pattern_rewriter import PLPatternRewriter, PLPatternRewriteWalker


class CCiXDecomposePattern(pattern_rewriter.RewritePattern):
    """CiX decompose pattern."""

    def find_match(self, op: Operation):
        if isinstance(op, CustomOp):
            if op.gate_name.data == "Toffoli" and len(op.in_ctrl_qubits) == 0:
                ctrl_qubits = op.in_qubits[:2]
                if ctrl_qubits[0].owner == ctrl_qubits[1].owner and isinstance(
                    ctrl_qubits[0].owner, CustomOp
                ):
                    prev_op = ctrl_qubits[0].owner
                    if prev_op.gate_name.data == "S" and len(prev_op.in_ctrl_qubits) == 1:
                        return prev_op, op

        return None

    def replace_with_subroutine(self, s, toffoli, rewriter: PLPatternRewriter):
        in_wires = [rewriter.wire_manager[q] for q in toffoli.in_qubits]

        insertion_point = InsertPoint.after(toffoli)
        rewriter.insert_gate(ops.Hadamard(in_wires[2]), insertion_point)
        rewriter.insert_gate(ops.adjoint(ops.T(in_wires[2])), insertion_point)
        rewriter.insert_gate(ops.CNOT(in_wires[1:]), insertion_point)
        rewriter.insert_gate(ops.T(in_wires[2]), insertion_point)
        rewriter.insert_gate(ops.CNOT([in_wires[0], in_wires[2]]), insertion_point)
        rewriter.insert_gate(ops.adjoint(ops.T(in_wires[2])), insertion_point)
        rewriter.insert_gate(ops.CNOT(in_wires[1:]), insertion_point)
        rewriter.insert_gate(ops.T(in_wires[2]), insertion_point)
        rewriter.insert_gate(ops.CNOT([in_wires[0], in_wires[2]]), insertion_point)
        rewriter.insert_gate(ops.Hadamard(in_wires[2]), insertion_point)

        rewriter.erase_quantum_gate_op(s, update_qubits=False)
        rewriter.erase_quantum_gate_op(toffoli, update_qubits=False)

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: PLPatternRewriter):
        """Match and rewrite."""
        for op in funcOp.body.walk():

            match = self.find_match(op)
            if match is None:
                print(op)
                # Update wire manager
                rewriter.wire_manager.update_from_op(op)
                continue

            self.replace_with_subroutine(*match, rewriter)


class CCiXDecomposePass(passes.ModulePass):
    """Pass impl"""

    name = "ccix-decompose"

    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply pass"""
        PLPatternRewriteWalker(CCiXDecomposePattern()).rewrite_module(module)


ccix_decompose_pass = compiler_transform(CCiXDecomposePass)
