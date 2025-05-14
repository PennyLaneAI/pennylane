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

"""This file contains the implementation of the unitary_to_rot transform,
written using xDSL."""

from dataclasses import dataclass
from xdsl import context, passes, pattern_rewriter

from xdsl.dialects import builtin, func
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import Float64Type, FloatAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.rewriter import InsertPoint

import pennylane as qml
from pennylane.ops.op_math.decompositions import one_qubit_decomposition, two_qubit_decomposition

from pennylane.compiler.python_compiler.quantum_dialect import CustomOp, QubitUnitaryOp, QubitType

# TODO: this implementation is not complete yet.


# pylint: disable=unused-argument
def extract_matrix_value(op: QubitUnitaryOp):
    """Extract the matrix value from a QubitUnitaryOp."""

    # TODO: This function doesn't work yet.
    # To move on, let's just return a random matrix
    return qml.numpy.random.rand(2, 2)


class UnitaryToRotPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """Pattern to match QubitUnitaryOp and rewrite it to CustomOp."""

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter
    ):  # pylint: disable=arguments-differ
        """Match QubitUnitaryOp and rewrite it to CustomOp."""
        for op in funcOp.body.walk():
            if not isinstance(op, QubitUnitaryOp):
                continue
            matrix_value = extract_matrix_value(op)
            extract_op = op.in_qubits[0].owner
            wire_index = extract_op.idx_attr.parameters[0].data
            matrix_shape = qml.math.shape(matrix_value)
            if matrix_shape == (2, 2):
                ops = one_qubit_decomposition(matrix_value, wire_index)
            elif matrix_shape == (4, 4):
                ops = two_qubit_decomposition(matrix_value, wire_index)
            else:
                ops = [op]

            for qml_op in ops:
                angle = qml_op.parameters[0].item()
                wire = qml_op.wires[0]

                # (right now know this is a float)
                angle_attr = FloatAttr(angle, Float64Type())
                angle_const_op = ConstantOp(value=angle_attr)
                rewriter.insert_op(angle_const_op, InsertPoint.before(op))

                # And we do the same for the wire
                wire_attr = IntegerAttr(wire, IntegerType(64))
                wire_const_op = ConstantOp(value=wire_attr)
                rewriter.insert_op(wire_const_op, InsertPoint.before(op))

                # TODO: here I am simply parsing the properties and successors of the QubitUnitaryOp
                # and passing them to the custom op. But it would be probably better to it differently.
                custom_op = CustomOp(
                    operands=(angle_const_op.result, wire_const_op.result, None, None),
                    properties={
                        "gate_name": StringAttr(qml_op.name),
                        **op.properties,
                    },
                    attributes={},
                    successors=op.successors,
                    regions=(),
                    result_types=(QubitType(), []),
                )

                rewriter.insert_op(custom_op, InsertPoint.before(op))

                for old_res, new_res in zip(op.results, custom_op.results):
                    old_res.replace_by(new_res)

            # We finally erase the QubitUnitaryOp
            rewriter.erase_op(op)


@dataclass(frozen=True)
class UnitaryToRotPass(passes.ModulePass):
    """Pass to apply the unitary_to_rot transform."""

    name = "unitary-to-rot"

    # pylint: disable=arguments-differ
    def apply(self, ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        """Apply the unitary_to_rot transform to the module."""
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([UnitaryToRotPattern(module)])
        ).rewrite_module(module)
