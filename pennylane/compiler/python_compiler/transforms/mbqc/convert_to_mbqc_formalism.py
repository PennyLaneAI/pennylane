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
from xdsl.dialects import arith, builtin, func, scf
from xdsl.dialects.scf import ForOp, IfOp, WhileOp
from xdsl.ir import SSAValue
from xdsl.ir.core import OpResult
from xdsl.rewriter import InsertPoint

from ...dialects.mbqc import (
    GraphStatePrepOp,
    MeasureInBasisOp,
    MeasurementPlaneAttr,
    MeasurementPlaneEnum,
)
from ...dialects.quantum import CustomOp, DeallocQubitOp, ExtractOp, GlobalPhaseOp, QubitType
from ...pass_api import compiler_transform
from .graph_state_utils import generate_adj_matrix, get_num_aux_wires

_PAULIS = {
    "PauliX",
    "PauliY",
    "PauliZ",
    "Identity",
}

_MBQC_ONE_QUBIT_GATES = {
    "Hadamard",
    "S",
    "RZ",
    "RotXZX",
}

_MBQC_TWO_QUBIT_GATES = {
    "CNOT",
}


@dataclass(frozen=True)
class ConvertToMBQCFormalismPass(passes.ModulePass):
    """Pass that converts gates in the MBQC gate set to the MBQC formalism."""

    name = "convert-to-mbqc-formalism"

    # pylint: disable=no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the convert-to-mbqc-formalism pass."""
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier(
                [
                    ConvertToMBQCFormalismPattern(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(module)


convert_to_mbqc_formalism_pass = compiler_transform(ConvertToMBQCFormalismPass)


class ConvertToMBQCFormalismPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods,no-self-use
    """RewritePattern for converting to the MBQC formalism."""

    def _prep_graph_state(
        self,
        adj_matrix_op: arith.ConstantOp,
        op: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):
        """Insert a graph state prep operation into the IR for each gate and extract and return auxiliary qubits
        in the graph state.

        Args:
            adj_matrix_op (arith.ConstantOp) : An `arith.ConstantOp` object stores the connectivity
            of auxiliary qubits in the graph state.
            op (CustomOp) : A `CustomOp` object. Note that op here is a quantum.customop object
                 instead of a qml.ops.
            rewriter (pattern_rewriter.PatternRewriter): A PatternRewriter object.

        Return:
            graph_qubits_dict : A dictionary of auxiliary qubits in the graph state. The keys represents
            the indices of qubits described in the [`arXiv:quant-ph/0301052 <https://arxiv.org/abs/quant-ph/0301052>`_].
        """
        num_aux_wres = get_num_aux_wires(op.gate_name.data)

        graph_state_prep_op = GraphStatePrepOp(adj_matrix_op.result, "Hadamard", "CZ")
        rewriter.insert_op(graph_state_prep_op, InsertPoint.before(op))
        graph_state_reg = graph_state_prep_op.results[0]

        graph_qubit_dict = {}
        # Extract qubit from the graph state reg
        for i in range(num_aux_wres):
            extract_op = ExtractOp(graph_state_reg, i)
            rewriter.insert_op(extract_op, InsertPoint.before(op))

            # Note the following line maps the aux qubit index in the register to the
            # standard context book MBQC representation. Note that auxiliary qubits in
            # the graph state for one qubit gates only hit the `if` branch as `i` is
            # always less than `4`, while the auxiliary qubits in graph state for a `CNOT`
            # gate with an index >= 7 would hit the `else` branch.
            key = i + 2 if i < 7 else i + 3

            graph_qubit_dict[key] = extract_op.results[0]

        return graph_qubit_dict

    def _insert_xy_basis_measure_op(
        self,
        angle: float,
        qubit: QubitType,
        insert_before: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):
        """Insert an arbitrary basis measure related operations to the IR.
        Args:
            angle (float) : The angle of measurement basis.
            qubit (QubitType) : The target qubit to be measured.
            insert_before (CustomOp) : A `CustomOp` object to be used as the insertion point for the `measure` op.
            rewriter (pattern_rewriter.PatternRewriter): A PatternRewriter object.

        Returns:
            The results include: 1. a measurement result; 2, a result qubit.
        """
        plane_op = MeasurementPlaneAttr(MeasurementPlaneEnum("XY"))
        # Create a constant op from a float64 variable
        const_angle_op = arith.ConstantOp(builtin.FloatAttr(data=angle, type=builtin.Float64Type()))
        # Insert the constant op into the IR
        rewriter.insert_op(const_angle_op, InsertPoint.before(insert_before))
        # Create a MeasureInBasisOp op
        measure_op = MeasureInBasisOp(in_qubit=qubit, plane=plane_op, angle=const_angle_op)
        # Insert the newly created measure_op into the IR
        rewriter.insert_op(measure_op, InsertPoint.before(insert_before))
        # Returns the results of the newly created measure_op.
        # The results include: 1, a measurement result; 2, a result qubit.
        return measure_op.results

    def _insert_measure_x_op(self, qubit, op, rewriter: pattern_rewriter.PatternRewriter):
        """Insert a X-basis measure op related operations to the IR."""
        return self._insert_xy_basis_measure_op(0.0, qubit, op, rewriter)

    def _insert_measure_y_op(self, qubit, op, rewriter: pattern_rewriter.PatternRewriter):
        """Insert a Y-basis measure op related operations to the IR."""
        return self._insert_xy_basis_measure_op(math.pi / 2, qubit, op, rewriter)

    def _insert_cond_arbitrary_basis_measure_op(
        self,
        meas_parity: builtin.IntegerType,
        angle: SSAValue[builtin.Float64Type],
        plane: str,
        qubit: QubitType,
        insert_before: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """
        Insert a conditional arbitrary basis measurement operation based on a previous measurement result.
        Args:
            meas_parity (builtin.IntegerType) : A parity of previous measurements.
            angle (SSAValue[builtin.Float64Type]) : An angle SSAValue from a parametric gate operation.
            plane (str): Plane of the measurement basis.
            qubit (QubitType) : The target qubit to be measured.
            insert_before (CustomOp) : A `CustomOp` object.
            rewriter (pattern_rewriter.PatternRewriter): A PatternRewriter object.

        Returns:
            The results include: 1. a measurement result; 2, a result qubit.
        """

        plane_op = MeasurementPlaneAttr(MeasurementPlaneEnum(plane))

        # Create a MeasureInBasisOp op for the true region
        measure_op = MeasureInBasisOp(in_qubit=qubit, plane=plane_op, angle=angle)

        # Create a const object hold the `-angle` value
        const_neg_angle_op = arith.NegfOp(angle)

        # Create a MeasureInBasisOp op for the false region
        measure_neg_op = MeasureInBasisOp(
            in_qubit=qubit, plane=plane_op, angle=const_neg_angle_op.result
        )
        # Create the true region
        true_region = [measure_op, scf.YieldOp(measure_op.results[0], measure_op.results[1])]
        # Create the false region
        false_region = [
            const_neg_angle_op,
            measure_neg_op,
            scf.YieldOp(measure_neg_op.results[0], measure_neg_op.results[1]),
        ]

        # Create a If control flow Op
        cond_op = IfOp(
            meas_parity,
            (builtin.IntegerType(1), QubitType()),
            true_region,
            false_region,
        )
        # Insert cond_op to the IR
        rewriter.insert_op(cond_op, InsertPoint.before(insert_before))

        # Return the result of if control flow
        return cond_op.results

    def _hadamard_measurements(
        self, graph_qubits_dict, op, rewriter: pattern_rewriter.PatternRewriter
    ):
        """Insert measurement ops for a Hadamard gate and return measurement results and the result graph qubits"""
        m1, graph_qubits_dict[1] = self._insert_measure_x_op(graph_qubits_dict[1], op, rewriter)
        m2, graph_qubits_dict[2] = self._insert_measure_y_op(graph_qubits_dict[2], op, rewriter)
        m3, graph_qubits_dict[3] = self._insert_measure_y_op(graph_qubits_dict[3], op, rewriter)
        m4, graph_qubits_dict[4] = self._insert_measure_y_op(graph_qubits_dict[4], op, rewriter)
        return [m1, m2, m3, m4], graph_qubits_dict

    def _s_measurements(self, graph_qubits_dict, op, rewriter: pattern_rewriter.PatternRewriter):
        """Insert measurement ops for a S gate and return measurement results and the result graph qubits"""
        m1, graph_qubits_dict[1] = self._insert_measure_x_op(graph_qubits_dict[1], op, rewriter)
        m2, graph_qubits_dict[2] = self._insert_measure_x_op(graph_qubits_dict[2], op, rewriter)
        m3, graph_qubits_dict[3] = self._insert_measure_y_op(graph_qubits_dict[3], op, rewriter)
        m4, graph_qubits_dict[4] = self._insert_measure_x_op(graph_qubits_dict[4], op, rewriter)
        return [m1, m2, m3, m4], graph_qubits_dict

    def _cnot_measurements(self, graph_qubits_dict, op, rewriter: pattern_rewriter.PatternRewriter):
        """Insert measurement ops for a CNOT gate and return measurement results and the result graph qubits"""
        m1, graph_qubits_dict[1] = self._insert_measure_x_op(graph_qubits_dict[1], op, rewriter)
        m2, graph_qubits_dict[2] = self._insert_measure_y_op(graph_qubits_dict[2], op, rewriter)
        m3, graph_qubits_dict[3] = self._insert_measure_y_op(graph_qubits_dict[3], op, rewriter)
        m4, graph_qubits_dict[4] = self._insert_measure_y_op(graph_qubits_dict[4], op, rewriter)
        m5, graph_qubits_dict[5] = self._insert_measure_y_op(graph_qubits_dict[5], op, rewriter)
        m6, graph_qubits_dict[6] = self._insert_measure_y_op(graph_qubits_dict[6], op, rewriter)
        m8, graph_qubits_dict[8] = self._insert_measure_y_op(graph_qubits_dict[8], op, rewriter)
        m9, graph_qubits_dict[9] = self._insert_measure_x_op(graph_qubits_dict[9], op, rewriter)
        m10, graph_qubits_dict[10] = self._insert_measure_x_op(graph_qubits_dict[10], op, rewriter)
        m11, graph_qubits_dict[11] = self._insert_measure_x_op(graph_qubits_dict[11], op, rewriter)
        m12, graph_qubits_dict[12] = self._insert_measure_y_op(graph_qubits_dict[12], op, rewriter)
        m13, graph_qubits_dict[13] = self._insert_measure_x_op(graph_qubits_dict[13], op, rewriter)
        m14, graph_qubits_dict[14] = self._insert_measure_x_op(graph_qubits_dict[14], op, rewriter)

        return [m1, m2, m3, m4, m5, m6, m8, m9, m10, m11, m12, m13, m14], graph_qubits_dict

    def _rz_measurements(self, graph_qubits_dict, op, rewriter: pattern_rewriter.PatternRewriter):
        """Insert measurement ops for a RZ gate and return measurement results and the result graph qubits"""
        m1, graph_qubits_dict[1] = self._insert_measure_x_op(graph_qubits_dict[1], op, rewriter)
        m2, graph_qubits_dict[2] = self._insert_measure_x_op(graph_qubits_dict[2], op, rewriter)
        m3, graph_qubits_dict[3] = self._insert_cond_arbitrary_basis_measure_op(
            m2, op.params[0], "XY", graph_qubits_dict[3], op, rewriter
        )
        m4, graph_qubits_dict[4] = self._insert_measure_x_op(graph_qubits_dict[4], op, rewriter)
        return [m1, m2, m3, m4], graph_qubits_dict

    def _rotxzx_measurements(
        self, graph_qubits_dict, op, rewriter: pattern_rewriter.PatternRewriter
    ):
        """Insert measurement ops for a RotXZX gate and return measurement results and the result graph qubits"""
        m1, graph_qubits_dict[1] = self._insert_measure_x_op(graph_qubits_dict[1], op, rewriter)
        m2, graph_qubits_dict[2] = self._insert_cond_arbitrary_basis_measure_op(
            m1, op.params[0], "XY", graph_qubits_dict[2], op, rewriter
        )
        m3, graph_qubits_dict[3] = self._insert_cond_arbitrary_basis_measure_op(
            m2, op.params[1], "XY", graph_qubits_dict[3], op, rewriter
        )

        m1_xor_m3 = arith.XOrIOp(m1, m3)
        rewriter.insert_op(m1_xor_m3, InsertPoint.before(op))

        m4, graph_qubits_dict[4] = self._insert_cond_arbitrary_basis_measure_op(
            m1_xor_m3.result, op.params[2], "XY", graph_qubits_dict[4], op, rewriter
        )
        return [m1, m2, m3, m4], graph_qubits_dict

    def _queue_measurements(
        self, graph_qubits_dict: dict, op: CustomOp, rewriter: pattern_rewriter.PatternRewriter
    ):
        """Add mid-measurement ops to the IR and return the measurement results and the updated graph qubit dict.
        Args:
            graph_qubits_dict (dict) : A dict stores all qubit info in a graph state.
            op (CustomOp) : A gate operation object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.

        Returns:
            A list of mid-measurement results and the updated graph qubit dict.
        """
        match op.gate_name.data:
            case "Hadamard":
                return self._hadamard_measurements(graph_qubits_dict, op, rewriter)
            case "S":
                return self._s_measurements(graph_qubits_dict, op, rewriter)
            case "RZ":
                return self._rz_measurements(graph_qubits_dict, op, rewriter)
            case "RotXZX":
                return self._rotxzx_measurements(graph_qubits_dict, op, rewriter)
            case "CNOT":
                return self._cnot_measurements(graph_qubits_dict, op, rewriter)
            case _:
                raise ValueError(
                    f"{op.gate_name.data} is not supported in the MBQC formalism. Please decompose it into the MBQC gate set."
                )

    def _insert_cond_byproduct_op(
        self,
        parity_res: OpResult,
        gate_name: str,
        qubit: QubitType,
        op: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Insert a byproduct op related operations to the IR.
        Args:
            parity_res (OpResult) : Parity check result.
            gate_name (str) : The name of the gate to be corrected.
            qubit (QubitType) : The result auxiliary qubit to be corrected.
            op (CustomOp) : A gate operation object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.

        Return:
            The result auxiliary qubit.
        """
        # Create a byproduct_op object
        byproduct_op = CustomOp(in_qubits=qubit, gate_name=gate_name)
        true_region = [byproduct_op, scf.YieldOp(byproduct_op.results[0])]

        false_region = [scf.YieldOp(qubit)]
        cond_op = IfOp(parity_res, (QubitType(),), true_region, false_region)
        # Insert cond_op to the IR
        rewriter.insert_op(cond_op, InsertPoint.before(op))
        return cond_op.results[0]

    def _parity_check(
        self,
        mres: list[builtin.IntegerType],
        op: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
        additional_const_one: bool = False,
    ):
        """Insert parity check related operations to the IR.
        Args:
            mres (list[builtin.IntegerType]): A list of the mid-measurement results.
            op (CustomOp) : A gate operation object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.
            additional_const_one (bool) : Whether we need to add an additional const one to get the
                parity or not. Defaults to False.
        Returns:
            The result of parity check.
        """
        prev_res = mres[0]
        xor_op = None
        # Create xor ops to iterate all elements in the mres and insert them to the IR
        for i in range(1, len(mres)):
            xor_op = arith.XOrIOp(prev_res, mres[i])
            rewriter.insert_op(xor_op, InsertPoint.before(op))
            prev_res = xor_op.result

        # Create an xor op for an additional const one and insert ops to the IR
        if additional_const_one:
            constant_one_op = arith.ConstantOp.from_int_and_width(1, builtin.i1)
            rewriter.insert_op(constant_one_op, InsertPoint.before(op))
            xor_op = arith.XOrIOp(prev_res, constant_one_op)
            rewriter.insert_op(xor_op, InsertPoint.before(op))
            prev_res = xor_op.result

        return prev_res

    def _hadamard_corrections(
        self,
        mres: list[builtin.IntegerType],
        qubit: QubitType,
        op: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):
        """Insert correction ops of a Hadamard gate to the IR.
        Args:
            mres (list[builtin.IntegerType]): A list of the mid-measurement results.
            qubit (QubitType) : An auxiliary result qubit.
            op (CustomOp) : A gate operation object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.

        Returns:
            The result auxiliary qubit.
        """
        m1, m2, m3, m4 = mres

        # X correction
        x_parity = self._parity_check([m1, m3, m4], op, rewriter)
        res_aux_qubit = self._insert_cond_byproduct_op(x_parity, "PauliX", qubit, op, rewriter)

        # Z correction
        z_parity = self._parity_check([m2, m3], op, rewriter)
        res_aux_qubit = self._insert_cond_byproduct_op(
            z_parity, "PauliZ", res_aux_qubit, op, rewriter
        )

        return res_aux_qubit

    def _s_corrections(
        self,
        mres: list[builtin.IntegerType],
        qubit: QubitType,
        op: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):
        """Insert correction ops of a S gate to the IR.
        Args:
            mres (list[builtin.IntegerType]): A list of the mid-measurement results.
            qubit (QubitType) : An auxiliary result qubit.
            op (CustomOp) : A gate operation object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.

        Returns:
            The result auxiliary qubit.
        """
        m1, m2, m3, m4 = mres

        # X correction
        x_parity = self._parity_check([m2, m4], op, rewriter)
        res_aux_qubit = self._insert_cond_byproduct_op(x_parity, "PauliX", qubit, op, rewriter)

        # Z correction
        z_parity = self._parity_check([m1, m2, m3], op, rewriter, additional_const_one=True)
        res_aux_qubit = self._insert_cond_byproduct_op(
            z_parity, "PauliZ", res_aux_qubit, op, rewriter
        )
        return res_aux_qubit

    def _rot_corrections(
        self,
        mres: list[builtin.IntegerType],
        qubit: QubitType,
        op: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):
        """Insert correction ops of a RotXZX or RZ gate to the IR.
        Args:
            mres (list[builtin.IntegerType]): A list of the mid-measurement results.
            qubit (QubitType) : An auxiliary result qubit.
            op (CustomOp) : A gate operation object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.

        Returns:
            The result auxiliary qubit.
        """
        m1, m2, m3, m4 = mres
        # X correction
        x_parity = self._parity_check([m2, m4], op, rewriter)
        res_aux_qubit = self._insert_cond_byproduct_op(x_parity, "PauliX", qubit, op, rewriter)

        # Z correction
        z_parity = self._parity_check([m1, m3], op, rewriter)
        res_aux_qubit = self._insert_cond_byproduct_op(
            z_parity, "PauliZ", res_aux_qubit, op, rewriter
        )
        return res_aux_qubit

    def _cnot_corrections(
        self,
        mres: list[builtin.IntegerType],
        qubits: list[QubitType],
        op: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):
        """Insert correction ops of a CNOT gate to the IR.
        Args:
            mres (list[builtin.IntegerType]): A list of the mid-measurement results.
            qubits (list[QubitType]) : A list of auxiliary result qubits.
            op (CustomOp) : A gate operation object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.

        Returns:
            The result auxiliary qubits.
        """
        m1, m2, m3, m4, m5, m6, m8, m9, m10, m11, m12, m13, m14 = mres
        # Corrections for the control qubit
        x_parity = self._parity_check([m2, m3, m5, m6], op, rewriter)
        ctrl_aux_qubit = self._insert_cond_byproduct_op(x_parity, "PauliX", qubits[0], op, rewriter)
        z_parity = self._parity_check(
            [m1, m3, m4, m5, m8, m9, m11], op, rewriter, additional_const_one=True
        )
        ctrl_aux_qubit = self._insert_cond_byproduct_op(
            z_parity, "PauliZ", ctrl_aux_qubit, op, rewriter
        )

        # Corrections for the target qubit
        x_parity = self._parity_check([m2, m3, m8, m10, m12, m14], op, rewriter)
        tgt_aux_qubit = self._insert_cond_byproduct_op(x_parity, "PauliX", qubits[1], op, rewriter)
        z_parity = self._parity_check([m9, m11, m13], op, rewriter)
        tgt_aux_qubit = self._insert_cond_byproduct_op(
            x_parity, "PauliZ", tgt_aux_qubit, op, rewriter
        )

        return ctrl_aux_qubit, tgt_aux_qubit

    def _insert_byprod_corrections(
        self,
        mres: list[builtin.IntegerType],
        qubits: QubitType | list[QubitType],
        op: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):
        """Insert correction ops for the result auxiliary qubit/s to the IR.
        Args:
            mres (list[builtin.IntegerType]): A list of the mid-measurement results.
            qubits (QubitType | list[QubitType]) : An or a list of auxiliary result qubit.
            op (CustomOp) : A gate operation object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.

        Returns:
            The result auxiliary qubits.
        """
        match op.gate_name.data:
            case "Hadamard":
                return self._hadamard_corrections(mres, qubits, op, rewriter)
            case "S":
                return self._s_corrections(mres, qubits, op, rewriter)
            case "RotXZX":
                return self._rot_corrections(mres, qubits, op, rewriter)
            case "RZ":
                return self._rot_corrections(mres, qubits, op, rewriter)
            case "CNOT":
                return self._cnot_corrections(mres, qubits, op, rewriter)
            case _:
                raise ValueError(
                    f"{op.gate_name.data} is not supported in the MBQC formalism. Please decompose it into the MBQC gate set."
                )

    def _deallocate_aux_qubits(
        self,
        graph_qubits_dict: dict,
        res_aux_qb: list[int],
        insert_before: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):
        """Deallocate non-result qubits in the graph qubit dict.
        Args:
            graph_qubits_dict (dict) : A dict stores all qubits in a graph state.
            res_aux_qb (list[int]) : A list of keys of result auxiliary qubits.
            insert_before (CustomOp) : A gate operation object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.
        """
        # Deallocate non result aux_qubits
        for node in graph_qubits_dict:
            if node not in res_aux_qb:
                dealloc_qubit_op = DeallocQubitOp(graph_qubits_dict[node])
                rewriter.insert_op(dealloc_qubit_op, InsertPoint.before(insert_before))

    # pylint: disable=no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, root: func.FuncOp | IfOp | WhileOp | ForOp, rewriter: pattern_rewriter.PatternRewriter
    ):
        """Match and rewrite for converting to the MBQC formalism."""

        for region in root.regions:
            # TODOs: Current implementation ensures only one type adj matrix op is inserted into one region of an IR.
            # We can further optimize it by ensure only one type adj matrix op is inserted into the IR. We can
            # come back to this later.
            one_wire_adj_matrix_op = None
            two_wire_adj_matrix_op = None
            for op in region.ops:
                # TODOs: Refactor the code below in this loop into separate functions
                if isinstance(op, CustomOp) and op.gate_name.data in _MBQC_ONE_QUBIT_GATES:
                    # Allocate auxiliary qubits and entangle them
                    if one_wire_adj_matrix_op is None:
                        adj_matrix = generate_adj_matrix(op.gate_name.data)
                        one_wire_adj_matrix_op = arith.ConstantOp(
                            builtin.DenseIntOrFPElementsAttr.from_list(
                                type=builtin.TensorType(
                                    builtin.IntegerType(1), shape=(len(adj_matrix),)
                                ),
                                data=adj_matrix,
                            )
                        )
                        rewriter.insert_op(one_wire_adj_matrix_op, InsertPoint.before(op))

                    graph_qubits_dict = self._prep_graph_state(one_wire_adj_matrix_op, op, rewriter)

                    # Entangle the op.in_qubits[0] with the graph_qubits_dict[2]
                    cz_op = CustomOp(
                        in_qubits=[op.in_qubits[0], graph_qubits_dict[2]], gate_name="CZ"
                    )
                    rewriter.insert_op(cz_op, InsertPoint.before(op))

                    # Update the graph qubit dict
                    graph_qubits_dict[1], graph_qubits_dict[2] = cz_op.results

                    # Insert measurement ops to the IR
                    mres, graph_qubits_dict = self._queue_measurements(
                        graph_qubits_dict, op, rewriter
                    )

                    # Insert byproduct ops to the IR
                    graph_qubits_dict[5] = self._insert_byprod_corrections(
                        mres, graph_qubits_dict[5], op, rewriter
                    )

                    # Deallocate the non-result auxiliary qubits and target qubit in the qreg
                    self._deallocate_aux_qubits(graph_qubits_dict, [5], op, rewriter)

                    # Replace all uses of output qubit of op with the result auxiliary qubit
                    rewriter.replace_all_uses_with(op.results[0], graph_qubits_dict[5])
                    # Remove op operation
                    rewriter.erase_op(op)
                elif isinstance(op, CustomOp) and op.gate_name.data in _MBQC_TWO_QUBIT_GATES:
                    # Allocate auxiliary qubits and entangle them
                    if two_wire_adj_matrix_op is None:
                        adj_matrix = generate_adj_matrix(op.gate_name.data)
                        two_wire_adj_matrix_op = arith.ConstantOp(
                            builtin.DenseIntOrFPElementsAttr.from_list(
                                type=builtin.TensorType(
                                    builtin.IntegerType(1), shape=(len(adj_matrix),)
                                ),
                                data=adj_matrix,
                            )
                        )
                        rewriter.insert_op(two_wire_adj_matrix_op, InsertPoint.before(op))
                    graph_qubits_dict = self._prep_graph_state(two_wire_adj_matrix_op, op, rewriter)

                    # Entangle the op.in_qubits[0] with the graph_qubits_dict[2]
                    cz_op = CustomOp(
                        in_qubits=[op.in_qubits[0], graph_qubits_dict[2]], gate_name="CZ"
                    )
                    rewriter.insert_op(cz_op, InsertPoint.before(op))
                    graph_qubits_dict[1], graph_qubits_dict[2] = cz_op.results

                    # Entangle op.in_qubits[1] with with the graph_qubits_dict[10] for a CNOT gate
                    cz_op = CustomOp(
                        in_qubits=[op.in_qubits[1], graph_qubits_dict[10]], gate_name="CZ"
                    )
                    rewriter.insert_op(cz_op, InsertPoint.before(op))
                    graph_qubits_dict[9], graph_qubits_dict[10] = cz_op.results

                    # Insert measurement ops to the IR
                    mres, graph_qubits_dict = self._queue_measurements(
                        graph_qubits_dict, op, rewriter
                    )

                    # Insert byproduct ops to the IR
                    graph_qubits_dict[7], graph_qubits_dict[15] = self._insert_byprod_corrections(
                        mres, [graph_qubits_dict[7], graph_qubits_dict[15]], op, rewriter
                    )

                    # Deallocate non-result aux_qubits and the target/control qubits in the qreg
                    self._deallocate_aux_qubits(graph_qubits_dict, [7, 15], op, rewriter)

                    # Replace all uses of output qubit of op with the result auxiliary qubit
                    rewriter.replace_all_uses_with(op.results[0], graph_qubits_dict[7])
                    rewriter.replace_all_uses_with(op.results[1], graph_qubits_dict[15])
                    # Remove op operation
                    rewriter.erase_op(op)
                elif isinstance(op, GlobalPhaseOp) or (
                    isinstance(op, CustomOp) and op.gate_name.data in _PAULIS
                ):
                    continue
                elif isinstance(op, CustomOp):
                    raise NotImplementedError(
                        f"{op.gate_name.data} cannot be converted to the MBQC formalism."
                    )
