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

"""This file contains the implementation of the convert_to_mbqc_formalism transform,
written using xDSL."""

import math
from dataclasses import dataclass

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin, func, scf, tensor
from xdsl.dialects.scf import ForOp, IfOp, WhileOp
from xdsl.ir import Operation, SSAValue
from xdsl.ir.core import OpResult
from xdsl.rewriter import InsertPoint

from ...dialects.mbqc import (
    GraphStatePrepOp,
    MeasureInBasisOp,
    MeasurementPlaneAttr,
    MeasurementPlaneEnum,
)
from ...dialects.quantum import (
    AllocOp,
    CustomOp,
    DeallocQubitOp,
    ExtractOp,
    GlobalPhaseOp,
    InsertOp,
    QubitType,
)
from ...pass_api import compiler_transform
from ...visualization.xdsl_conversion import ssa_to_qml_wires

_PAULIS = {
    "PauliX",
    "PauliY",
    "PauliZ",
    "Identity",
}

_MBQC_CLIFFORD_GATES = {
    "Hadamard",
    "S",
    "CNOT",
}

_MBQC_NON_CLIFFORD_GATES = {
    "RZ",
    "RotXZX",
}

_MBQC_TWO_QUBIT_GATES = {
    "CNOT",
}


@dataclass(frozen=True)
class AddPauliTrackerPass(passes.ModulePass):
    """Pass that add Pauli Tracker to the IR."""

    name = "add-pauli-tracker"

    # pylint: disable=no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the add-pauli-tracker pass."""
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier(
                [
                    AddPauliTrackerPattern(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(module)


add_pauli_tracker_pass = compiler_transform(AddPauliTrackerPass)


class AddPauliTrackerPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods,no-self-use
    """RewritePattern for adding Pauli Tracker to the IR."""

    # pylint: disable=no-self-use

    def _insert_xz_record_tensors(
        self, num_qubits: int, op, rewriter: pattern_rewriter.PatternRewriter
    ):
        prev_op = op
        num_qubits = op.nqubits_attr.value.data
        xz_record_tensor_type = tensor.TensorType(builtin.i1, [num_qubits])
        const_zero = arith.ConstantOp.from_int_and_width(0, 1)
        rewriter.insert_op(const_zero, InsertPoint.after(prev_op))

        prev_op = const_zero
        x_record = tensor.SplatOp(
            input=const_zero.result,
            dynamicSizes=(),
            result_type=xz_record_tensor_type,
        )
        x_record.attributes["x_record"] = builtin.UnitAttr()
        rewriter.insert_op(x_record, InsertPoint.after(prev_op))

        prev_op = x_record
        z_record = tensor.SplatOp(
            input=const_zero.result,
            dynamicSizes=(),
            result_type=xz_record_tensor_type,
        )
        z_record.attributes["z_record"] = builtin.UnitAttr()
        rewriter.insert_op(z_record, InsertPoint.after(prev_op))
        return x_record.result, z_record.result

    def _extract_xz_records(self, wires, x_record, z_record, op, rewriter):
        xz = []
        wires_ssa = []
        for wire in wires:
            target_wire_index = arith.ConstantOp(builtin.IntegerAttr(wire, builtin.IndexType()))
            rewriter.insert_op(target_wire_index, InsertPoint.before(op))

            wires_ssa.append(target_wire_index)

            target_x_record = tensor.ExtractOp(x_record, target_wire_index, builtin.i1)
            rewriter.insert_op(target_x_record, InsertPoint.before(op))
            xz.append(target_x_record.result)
            target_z_record = tensor.ExtractOp(z_record, target_wire_index, builtin.i1)
            rewriter.insert_op(target_z_record, InsertPoint.before(op))
            xz.append(target_z_record.result)
        return xz, wires_ssa

    def _insert_xz_records(self, wires, xz, x_record, z_record, op, rewriter):
        for i, wire in enumerate(wires):
            x, z = xz[i * 2 + 0], xz[i * 2 + 1]
            update_x_record = tensor.InsertOp(x, x_record, wire)
            rewriter.insert_op(update_x_record, InsertPoint.before(op))

            prev_op = update_x_record
            update_z_record = tensor.InsertOp(z, z_record, wire)
            rewriter.insert_op(update_z_record, InsertPoint.before(prev_op))

    def _commute_h(self, xz, op, rewriter):
        x, z = xz
        return [z, x]

    def _commute_s(self, xz, op, rewriter):
        x, z = xz
        new_z = arith.XOrIOp(x, z)
        rewriter.insert_op(new_z, InsertPoint.before(op))
        return [x, new_z.result]

    def _commute_cnot(self, xz, op, rewriter):
        xc, zc, xt, zt = xz
        new_zc = arith.XOrIOp(zc, zt)
        rewriter.insert_op(new_zc, InsertPoint.before(op))
        new_xt = arith.XOrIOp(xc, xt)
        rewriter.insert_op(new_xt, InsertPoint.before(op))

        return [xc, new_zc.result, new_xt.result, zt]

    def _apply_clifford_commute_rule(self, xz, op, rewriter):
        if op.gate_name.data == "Hadamard":
            return self._commute_h(xz, op, rewriter)
        elif op.gate_name.data == "S":
            return self._commute_s(xz, op, rewriter)
        elif op.gate_name.data == "CNOT":
            return self._commute_cnot(xz, op, rewriter)
        else:
            raise ValueError(f"No commute rule for {op.gate_name.data}")

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, root: func.FuncOp | IfOp | WhileOp | ForOp, rewriter: pattern_rewriter.PatternRewriter
    ):
        qreg = None
        x_record = None
        z_record = None
        for region in root.regions:
            for op in region.ops:
                if isinstance(op, AllocOp):
                    qreg = op.results
                    num_qubits = op.nqubits_attr.value.data
                    x_record, z_record = self._insert_xz_record_tensors(num_qubits, op, rewriter)
                elif isinstance(op, CustomOp) and op.gate_name.data in _MBQC_CLIFFORD_GATES:
                    wires_int = ssa_to_qml_wires(op)
                    # Extract xz record information
                    xz, wires_ssa = self._extract_xz_records(
                        wires_int, x_record, z_record, op, rewriter
                    )
                    # Apply commute rules
                    new_xz = self._apply_clifford_commute_rule(xz, op, rewriter)
                    # Insert new_xz to x_record and z_record
                    self._insert_xz_records(wires_ssa, new_xz, x_record, z_record, op, rewriter)
                elif isinstance(op, CustomOp) and op.gate_name.data in _PAULIS:
                    if op.gate_name.data != "Identity":
                        wires_int = ssa_to_qml_wires(op)
                        # Extract xz record information
                        xz, wires_ssa = self._extract_xz_records(
                            wires_int, x_record, z_record, op, rewriter
                        )

                        cst_one = arith.ConstantOp.from_int_and_width(1, builtin.i1)
                        rewriter.insert_op(cst_one, InsertPoint.before(op))

                        x, z = xz
                        new_xz = []
                        if op.gate_name.data == "PauliX":
                            new_x = arith.XOrIOp(x, cst_one)
                            rewriter.insert_op(new_x, InsertPoint.before(op))
                            new_xz = [new_x, z]
                        elif op.gate_name.data == "PauliY":
                            new_x = arith.XOrIOp(x, cst_one)
                            rewriter.insert_op(new_x, InsertPoint.before(op))
                            new_z = arith.XOrIOp(z, cst_one)
                            rewriter.insert_op(new_z, InsertPoint.before(op))
                            new_xz = [new_x, new_z]
                        else:
                            new_z = arith.XOrIOp(z, cst_one)
                            rewriter.insert_op(new_z, InsertPoint.before(op))
                            new_xz = [x, new_z]

                    rewriter.replace_all_uses_with(op.results[0], op.operands[0])
                    rewriter.erase_op(op)
