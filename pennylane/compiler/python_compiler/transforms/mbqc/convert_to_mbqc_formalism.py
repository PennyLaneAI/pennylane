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

from xdsl import builder, context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin, func, scf
from xdsl.dialects.scf import ForOp, IfOp, IndexSwitchOp, WhileOp
from xdsl.ir import SSAValue
from xdsl.ir.core import Block, OpResult, Region
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

_MBQC_GATES = _MBQC_ONE_QUBIT_GATES | _MBQC_TWO_QUBIT_GATES


@dataclass(frozen=True)
class ConvertToMBQCFormalismPass(passes.ModulePass):
    """Pass that converts gates in the MBQC gate set to the MBQC formalism."""

    name = "convert-to-mbqc-formalism"

    def _prep_graph_state(self, gate_name: str):
        """Add a graph state prep operation into the subroutine for each gate and extract and return auxiliary qubits
        in the graph state.

        Args:
            gate_name[str]: Name of gate operation.

        Return:
            graph_qubit_dict : A dictionary of qubits in the graph
        """
        num_aux_wres = get_num_aux_wires(gate_name)

        adj_matrix_op = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(
                type=builtin.TensorType(
                    builtin.IntegerType(1), shape=(len(generate_adj_matrix(gate_name)),)
                ),
                data=generate_adj_matrix(gate_name),
            )
        )

        graph_state_prep_op = GraphStatePrepOp(adj_matrix_op.result, "Hadamard", "CZ")
        graph_state_reg = graph_state_prep_op.results[0]

        graph_qubit_dict = {}
        # Extract qubit from the graph state reg
        for i in range(num_aux_wres):
            extract_op = ExtractOp(graph_state_reg, i)

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
        const_angle_op: arith.ConstantOp,
        qubit: QubitType,
    ):
        """Add an arbitrary basis measure related operations to the subroutine.
        Args:
            const_angle_op (arith.ConstantOp) : The angle of measurement basis.
            qubit (QubitType) : The target qubit to be measured.

        Returns:
            The results include: 1. a measurement result; 2, a result qubit.
        """
        plane_op = MeasurementPlaneAttr(MeasurementPlaneEnum("XY"))
        # Create a MeasureInBasisOp op
        measure_op = MeasureInBasisOp(in_qubit=qubit, plane=plane_op, angle=const_angle_op)
        # Returns the results of the newly created measure_op.
        # The results include: 1, a measurement result; 2, a result qubit.
        return measure_op.results

    def _insert_cond_arbitrary_basis_measure_op(
        self,
        meas_parity: builtin.IntegerType,
        angle: SSAValue[builtin.Float64Type],
        plane: str,
        qubit: QubitType,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """
        Add a conditional arbitrary basis measurement operation based on a previous measurement result.
        Args:
            meas_parity (builtin.IntegerType) : A parity of previous measurements.
            angle (SSAValue[builtin.Float64Type]) : An angle SSAValue from a parametric gate operation.
            plane (str): Plane of the measurement basis.
            qubit (QubitType) : The target qubit to be measured.

        Returns:
            The results include: 1. a measurement result; 2, a result qubit.
        """
        constant_one_op = arith.ConstantOp.from_int_and_width(1, builtin.i1)
        cond = arith.CmpiOp(meas_parity, constant_one_op, "eq")
        branch = scf.IfOp(
            cond,
            (
                builtin.IntegerType(1),
                QubitType(),
            ),
            Region(Block()),
            Region(Block()),
        )

        plane_op = MeasurementPlaneAttr(MeasurementPlaneEnum(plane))

        with builder.ImplicitBuilder(branch.true_region):
            measure_op = MeasureInBasisOp(in_qubit=qubit, plane=plane_op, angle=angle)
            scf.YieldOp(measure_op.results[0], measure_op.results[1])
        with builder.ImplicitBuilder(branch.false_region):
            const_neg_angle_op = arith.NegfOp(angle)
            measure_neg_op = MeasureInBasisOp(
                in_qubit=qubit, plane=plane_op, angle=const_neg_angle_op.result
            )
            scf.YieldOp(measure_neg_op.results[0], measure_neg_op.results[1])

        return branch.results

    def _hadamard_measurements(self, graph_qubit_dict):
        """Add measurement ops for a Hadamard gate to the subroutine"""
        const_x_angle = arith.ConstantOp(
            builtin.FloatAttr(data=0.0, type=builtin.Float64Type())
        )  # measure_x
        const_y_angle = arith.ConstantOp(
            builtin.FloatAttr(data=math.pi / 2, type=builtin.Float64Type())
        )  # measure_y
        m1, graph_qubit_dict[1] = self._insert_xy_basis_measure_op(
            const_x_angle, graph_qubit_dict[1]
        )
        m2, graph_qubit_dict[2] = self._insert_xy_basis_measure_op(
            const_y_angle, graph_qubit_dict[2]
        )
        m3, graph_qubit_dict[3] = self._insert_xy_basis_measure_op(
            const_y_angle, graph_qubit_dict[3]
        )
        m4, graph_qubit_dict[4] = self._insert_xy_basis_measure_op(
            const_y_angle, graph_qubit_dict[4]
        )
        return [m1, m2, m3, m4], graph_qubit_dict

    def _s_measurements(self, graph_qubit_dict):
        """Add measurement ops for a S gate to the subroutine"""
        const_x_angle = arith.ConstantOp(
            builtin.FloatAttr(data=0.0, type=builtin.Float64Type())
        )  # measure_x
        const_y_angle = arith.ConstantOp(
            builtin.FloatAttr(data=math.pi / 2, type=builtin.Float64Type())
        )  # measure_y
        m1, graph_qubit_dict[1] = self._insert_xy_basis_measure_op(
            const_x_angle, graph_qubit_dict[1]
        )
        m2, graph_qubit_dict[2] = self._insert_xy_basis_measure_op(
            const_x_angle, graph_qubit_dict[2]
        )
        m3, graph_qubit_dict[3] = self._insert_xy_basis_measure_op(
            const_y_angle, graph_qubit_dict[3]
        )
        m4, graph_qubit_dict[4] = self._insert_xy_basis_measure_op(
            const_x_angle, graph_qubit_dict[4]
        )
        return [m1, m2, m3, m4], graph_qubit_dict

    def _rz_measurements(self, graph_qubit_dict, params):
        """Add measurement ops for a RZ gate to the subroutine"""
        const_x_angle = arith.ConstantOp(
            builtin.FloatAttr(data=0.0, type=builtin.Float64Type())
        )  # measure_x
        m1, graph_qubit_dict[1] = self._insert_xy_basis_measure_op(
            const_x_angle, graph_qubit_dict[1]
        )
        m2, graph_qubit_dict[2] = self._insert_xy_basis_measure_op(
            const_x_angle, graph_qubit_dict[2]
        )
        m3, graph_qubit_dict[3] = self._insert_cond_arbitrary_basis_measure_op(
            m2, params[0], "XY", graph_qubit_dict[3]
        )
        m4, graph_qubit_dict[4] = self._insert_xy_basis_measure_op(
            const_x_angle, graph_qubit_dict[4]
        )
        return [m1, m2, m3, m4], graph_qubit_dict

    def _rotxzx_measurements(self, graph_qubit_dict, params):
        """Add measurement ops for a RotXZX gate to the subroutine"""
        const_x_angle = arith.ConstantOp(
            builtin.FloatAttr(data=0.0, type=builtin.Float64Type())
        )  # measure_x
        m1, graph_qubit_dict[1] = self._insert_xy_basis_measure_op(
            const_x_angle, graph_qubit_dict[1]
        )
        m2, graph_qubit_dict[2] = self._insert_cond_arbitrary_basis_measure_op(
            m1, params[0], "XY", graph_qubit_dict[2]
        )
        m3, graph_qubit_dict[3] = self._insert_cond_arbitrary_basis_measure_op(
            m2, params[1], "XY", graph_qubit_dict[3]
        )

        m1_xor_m3 = arith.XOrIOp(m1, m3)

        m4, graph_qubit_dict[4] = self._insert_cond_arbitrary_basis_measure_op(
            m1_xor_m3.result, params[2], "XY", graph_qubit_dict[4]
        )
        return [m1, m2, m3, m4], graph_qubit_dict

    def _cnot_measurements(self, graph_qubit_dict):
        """Add measurement ops for a CNOT gate to the subroutine"""
        const_x_angle = arith.ConstantOp(
            builtin.FloatAttr(data=0.0, type=builtin.Float64Type())
        )  # measure_x
        const_y_angle = arith.ConstantOp(
            builtin.FloatAttr(data=math.pi / 2, type=builtin.Float64Type())
        )  # measure_y
        m1, graph_qubit_dict[1] = self._insert_xy_basis_measure_op(
            const_x_angle, graph_qubit_dict[1]
        )
        m2, graph_qubit_dict[2] = self._insert_xy_basis_measure_op(
            const_y_angle, graph_qubit_dict[2]
        )
        m3, graph_qubit_dict[3] = self._insert_xy_basis_measure_op(
            const_y_angle, graph_qubit_dict[3]
        )
        m4, graph_qubit_dict[4] = self._insert_xy_basis_measure_op(
            const_y_angle, graph_qubit_dict[4]
        )
        m5, graph_qubit_dict[5] = self._insert_xy_basis_measure_op(
            const_y_angle, graph_qubit_dict[5]
        )
        m6, graph_qubit_dict[6] = self._insert_xy_basis_measure_op(
            const_y_angle, graph_qubit_dict[6]
        )
        m8, graph_qubit_dict[8] = self._insert_xy_basis_measure_op(
            const_y_angle, graph_qubit_dict[8]
        )
        m9, graph_qubit_dict[9] = self._insert_xy_basis_measure_op(
            const_x_angle, graph_qubit_dict[9]
        )
        m10, graph_qubit_dict[10] = self._insert_xy_basis_measure_op(
            const_x_angle, graph_qubit_dict[10]
        )
        m11, graph_qubit_dict[11] = self._insert_xy_basis_measure_op(
            const_x_angle, graph_qubit_dict[11]
        )
        m12, graph_qubit_dict[12] = self._insert_xy_basis_measure_op(
            const_y_angle, graph_qubit_dict[12]
        )
        m13, graph_qubit_dict[13] = self._insert_xy_basis_measure_op(
            const_x_angle, graph_qubit_dict[13]
        )
        m14, graph_qubit_dict[14] = self._insert_xy_basis_measure_op(
            const_x_angle, graph_qubit_dict[14]
        )

        return [m1, m2, m3, m4, m5, m6, m8, m9, m10, m11, m12, m13, m14], graph_qubit_dict

    def _parity_check(
        self,
        mres: list[builtin.IntegerType],
        additional_const_one: bool = False,
    ):
        """Add parity check related operations to the subroutine.
        Args:
            mres (list[builtin.IntegerType]): A list of the mid-measurement results.
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
            prev_res = xor_op.result

        # Create an xor op for an additional const one and insert ops to the IR
        if additional_const_one:
            constant_one_op = arith.ConstantOp.from_int_and_width(1, builtin.i1)
            xor_op = arith.XOrIOp(prev_res, constant_one_op)
            prev_res = xor_op.result

        return prev_res

    def _insert_cond_byproduct_op(
        self,
        parity_res: OpResult,
        gate_name: str,
        qubit: QubitType,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Add a byproduct op related operations to the subroutine.
        Args:
            parity_res (OpResult) : Parity check result.
            gate_name (str) : The name of the gate to be corrected.
            qubit (QubitType) : The result auxiliary qubit to be corrected.

        Return:
            The result auxiliary qubit.
        """
        constant_one_op = arith.ConstantOp.from_int_and_width(1, builtin.i1)
        cond = arith.CmpiOp(parity_res, constant_one_op, "eq")
        branch = scf.IfOp(cond, (QubitType(),), Region(Block()), Region(Block()))

        with builder.ImplicitBuilder(branch.true_region):
            byproduct_op = CustomOp(in_qubits=qubit, gate_name=gate_name)
            scf.YieldOp(byproduct_op.results[0])
        with builder.ImplicitBuilder(branch.false_region):
            scf.YieldOp(qubit)
        return branch.results[0]

    def _hadamard_corrections(
        self,
        mres: list[builtin.IntegerType],
        qubit: QubitType,
    ):
        """Add correction ops of a Hadamard gate to the subroutine.
        Args:
            mres (list[builtin.IntegerType]): A list of the mid-measurement results.
            qubit (QubitType) : An auxiliary result qubit.

        Returns:
            The result auxiliary qubit.
        """
        m1, m2, m3, m4 = mres

        # X correction
        x_parity = self._parity_check([m1, m3, m4])
        res_aux_qubit = self._insert_cond_byproduct_op(x_parity, "PauliX", qubit)

        # Z correction
        z_parity = self._parity_check([m2, m3])
        res_aux_qubit = self._insert_cond_byproduct_op(z_parity, "PauliZ", res_aux_qubit)

        return res_aux_qubit

    def _s_corrections(
        self,
        mres: list[builtin.IntegerType],
        qubit: QubitType,
    ):
        """Add correction ops of a S gate to the subroutine.
        Args:
            mres (list[builtin.IntegerType]): A list of the mid-measurement results.
            qubit (QubitType) : An auxiliary result qubit.

        Returns:
            The result auxiliary qubit.
        """
        m1, m2, m3, m4 = mres

        # X correction
        x_parity = self._parity_check([m2, m4])
        res_aux_qubit = self._insert_cond_byproduct_op(x_parity, "PauliX", qubit)

        # Z correction
        z_parity = self._parity_check([m1, m2, m3], additional_const_one=True)
        res_aux_qubit = self._insert_cond_byproduct_op(z_parity, "PauliZ", res_aux_qubit)
        return res_aux_qubit

    def _rot_corrections(
        self,
        mres: list[builtin.IntegerType],
        qubit: QubitType,
    ):
        """Add correction ops of a RotXZX or RZ gate to the subroutine.
        Args:
            mres (list[builtin.IntegerType]): A list of the mid-measurement results.
            qubit (QubitType) : An auxiliary result qubit.

        Returns:
            The result auxiliary qubit.
        """
        m1, m2, m3, m4 = mres
        # X correction
        x_parity = self._parity_check([m2, m4])
        res_aux_qubit = self._insert_cond_byproduct_op(x_parity, "PauliX", qubit)

        # Z correction
        z_parity = self._parity_check([m1, m3])
        res_aux_qubit = self._insert_cond_byproduct_op(z_parity, "PauliZ", res_aux_qubit)
        return res_aux_qubit

    def _cnot_corrections(
        self,
        mres: list[builtin.IntegerType],
        qubits: list[QubitType],
    ):
        """Add correction ops of a CNOT gate to the subroutine.
        Args:
            mres (list[builtin.IntegerType]): A list of the mid-measurement results.
            qubits (list[QubitType]) : A list of auxiliary result qubits.
        Returns:
            The result auxiliary qubits.
        """
        m1, m2, m3, m4, m5, m6, m8, m9, m10, m11, m12, m13, m14 = mres
        # Corrections for the control qubit
        x_parity = self._parity_check([m2, m3, m5, m6])
        ctrl_aux_qubit = self._insert_cond_byproduct_op(x_parity, "PauliX", qubits[0])
        z_parity = self._parity_check([m1, m3, m4, m5, m8, m9, m11], additional_const_one=True)
        ctrl_aux_qubit = self._insert_cond_byproduct_op(z_parity, "PauliZ", ctrl_aux_qubit)

        # Corrections for the target qubit
        x_parity = self._parity_check([m2, m3, m8, m10, m12, m14])
        tgt_aux_qubit = self._insert_cond_byproduct_op(x_parity, "PauliX", qubits[1])
        z_parity = self._parity_check([m9, m11, m13])
        tgt_aux_qubit = self._insert_cond_byproduct_op(z_parity, "PauliZ", tgt_aux_qubit)

        return ctrl_aux_qubit, tgt_aux_qubit

    def _queue_measurements(
        self, gate_name: str, graph_qubit_dict, params: None | list[builtin.Float64Type] = None
    ):
        """Add measurement ops to the subroutine.
        Args:
            gate_name (str): Gate name.
            graph_qubit_dict (list[builtin.IntegerType]): A list of the mid-measurement results.
            params (None | list[builtin.Float64Type]) : Parameters of the gate.

        Returns:
            The measurement results and updated graph_qubit_dict.

        """
        match gate_name:
            case "Hadamard":
                return self._hadamard_measurements(graph_qubit_dict)
            case "S":
                return self._s_measurements(graph_qubit_dict)
            case "RZ":
                return self._rz_measurements(graph_qubit_dict, params)
            case "RotXZX":
                return self._rotxzx_measurements(graph_qubit_dict, params)
            case "CNOT":
                return self._cnot_measurements(graph_qubit_dict)
            case _:
                raise ValueError(
                    f"{gate_name} is not supported in the MBQC formalism. Please decompose it into the MBQC gate set."
                )

    def _insert_byprod_corrections(
        self,
        gate_name: str,
        mres: list[builtin.IntegerType],
        qubits: QubitType | list[QubitType],
    ):
        """Add correction ops for the result auxiliary qubit/s to the subroutine.
        Args:
            gate_name (str): Gate name.
            mres (list[builtin.IntegerType]): A list of the mid-measurement results.
            qubits (QubitType | list[QubitType]) : An or a list of auxiliary result qubit.

        Returns:
            The result auxiliary qubits.
        """
        match gate_name:
            case "Hadamard":
                return self._hadamard_corrections(mres, qubits)
            case "S":
                return self._s_corrections(mres, qubits)
            case "RotXZX":
                return self._rot_corrections(mres, qubits)
            case "RZ":
                return self._rot_corrections(mres, qubits)
            case "CNOT":
                return self._cnot_corrections(mres, qubits)
            case _:
                raise ValueError(
                    f"{gate_name} is not supported in the MBQC formalism. Please decompose it into the MBQC gate set."
                )

    def _create_single_qubit_gate_subroutine(self, gate_name: str):
        """Create a subroutine for a single qubit gate based on the given name.
        Args:
            gate_name (str): Name of the gate.

        Returns:
            The corresponding subroutine (func.FuncOp).
        """
        if gate_name not in _MBQC_ONE_QUBIT_GATES:
            raise NotImplementedError(f"Subroutine for the {gate_name} gate is not supported.")
        # ensure the order of parameters are aligned with customOp
        input_types = ()
        if gate_name == "RZ":
            input_types += (builtin.Float64Type(),)
        if gate_name == "RotXZX":
            input_types += (builtin.Float64Type(),) * 3
        input_types += (QubitType(),)

        output_types = (QubitType(),)
        block = Block(arg_types=input_types)

        with builder.ImplicitBuilder(block):
            in_qubits = [block.args[-1]]
            params = None
            if gate_name == "RZ":
                params = [block.args[0]]
            if gate_name == "RotXZX":
                params = [
                    block.args[0],
                    block.args[1],
                    block.args[2],
                ]

            graph_qubit_dict = self._prep_graph_state(gate_name=gate_name)

            cz_op = CustomOp(in_qubits=[in_qubits[0], graph_qubit_dict[2]], gate_name="CZ")

            graph_qubit_dict[1], graph_qubit_dict[2] = cz_op.results

            mres, graph_qubit_dict = self._queue_measurements(gate_name, graph_qubit_dict, params)

            # The following could be removed to support Pauli tracker
            by_product_correction = self._insert_byprod_corrections(
                gate_name, mres, graph_qubit_dict[5]
            )

            graph_qubit_dict[5] = by_product_correction

            for node in graph_qubit_dict:
                if node not in [5]:
                    _ = DeallocQubitOp(graph_qubit_dict[node])

            func.ReturnOp(graph_qubit_dict[5])

        region = Region([block])
        # Note that visibility is set as private to ensure the subroutines that are
        # not called (dead code) can be eliminated as the ["symbol-dce"](https://github.com/PennyLaneAI/catalyst/blob/372c376eb821e830da778fdc8af423eeb487eab6/frontend/catalyst/pipelines.py#L248)_
        # pass was added to the pipeline.
        funcOp = func.FuncOp(
            gate_name.lower() + "_in_mbqc",
            (input_types, output_types),
            visibility="private",
            region=region,
        )
        # Add an attribute to the mbqc transform subroutine
        funcOp.attributes["mbqc_transform"] = builtin.NoneAttr()
        return funcOp

    def _create_cnot_gate_subroutine(self):
        """Create a subroutine for a CNOT gate."""
        gate_name = "CNOT"
        input_types = (
            QubitType(),
            QubitType(),
        )
        output_types = (
            QubitType(),
            QubitType(),
        )
        block = Block(arg_types=input_types)

        with builder.ImplicitBuilder(block):
            in_qubits = [block.args[0], block.args[1]]

            graph_qubit_dict = self._prep_graph_state(gate_name=gate_name)

            # Entangle the op.in_qubits[0] with the graph_qubits_dict[2]
            cz_op = CustomOp(in_qubits=[in_qubits[0], graph_qubit_dict[2]], gate_name="CZ")
            graph_qubit_dict[1], graph_qubit_dict[2] = cz_op.results

            # Entangle op.in_qubits[1] with with the graph_qubits_dict[10] for a CNOT gate
            cz_op = CustomOp(in_qubits=[in_qubits[1], graph_qubit_dict[10]], gate_name="CZ")
            graph_qubit_dict[9], graph_qubit_dict[10] = cz_op.results

            mres, graph_qubit_dict = self._queue_measurements(gate_name, graph_qubit_dict)

            # The following could be removed to support Pauli tracker
            graph_qubit_dict[7], graph_qubit_dict[15] = self._insert_byprod_corrections(
                gate_name, mres, [graph_qubit_dict[7], graph_qubit_dict[15]]
            )

            for node in graph_qubit_dict:
                if node not in [7, 15]:
                    _ = DeallocQubitOp(graph_qubit_dict[node])

            func.ReturnOp(
                *(
                    graph_qubit_dict[7],
                    graph_qubit_dict[15],
                )
            )

        region = Region([block])
        # Note that visibility is set as private to ensure the subroutines that are
        # not called (dead code) can be eliminated as the ["symbol-dce"](https://github.com/PennyLaneAI/catalyst/blob/372c376eb821e830da778fdc8af423eeb487eab6/frontend/catalyst/pipelines.py#L248)_
        # pass was added to the pipeline.
        funcOp = func.FuncOp(
            gate_name.lower() + "_in_mbqc",
            (input_types, output_types),
            visibility="private",
            region=region,
        )
        # Add an attribute to the mbqc transform subroutine
        funcOp.attributes["mbqc_transform"] = builtin.NoneAttr()
        return funcOp

    # pylint: disable=no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the convert-to-mbqc-formalism pass."""
        # Insert subroutines for all gates in the MBQC gate set to the module.
        # Note that the visibility of those subroutines are set as private, which ensure
        # the ["symbol-dce"](https://github.com/PennyLaneAI/catalyst/blob/372c376eb821e830da778fdc8af423eeb487eab6/frontend/catalyst/pipelines.py#L248)_
        # pass could eliminate the unreferenced subroutines.
        subroutine_dict = {}

        for gate_name in _MBQC_ONE_QUBIT_GATES:
            funcOp = self._create_single_qubit_gate_subroutine(gate_name)
            module.regions[0].blocks.first.add_op(funcOp)
            subroutine_dict[gate_name] = funcOp

        cnot_funcOp = self._create_cnot_gate_subroutine()
        module.regions[0].blocks.first.add_op(cnot_funcOp)
        subroutine_dict["CNOT"] = cnot_funcOp

        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier(
                [
                    ConvertToMBQCFormalismPattern(subroutine_dict),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(module)


convert_to_mbqc_formalism_pass = compiler_transform(ConvertToMBQCFormalismPass)


class ConvertToMBQCFormalismPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods,no-self-use
    """RewritePattern for converting to the MBQC formalism."""

    def __init__(self, subroutines_dict):
        self.subroutine_dict = subroutines_dict

    # pylint: disable=no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        root: func.FuncOp | IfOp | WhileOp | ForOp | IndexSwitchOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):
        """Match and rewrite for converting to the MBQC formalism."""

        # Ensure that "Hadamard"/"CZ" gates in mbqc_transform subroutines are not converted.
        if isinstance(root, func.FuncOp) and "mbqc_transform" in root.attributes:
            return

        for region in root.regions:
            # Continue if the region has no block (i.e., function that has no body, and the body is
            # defined in runtime.)
            if not region.blocks:
                continue

            for op in region.ops:
                if isinstance(op, CustomOp) and op.gate_name.data in _MBQC_GATES:
                    callee = builtin.SymbolRefAttr(op.gate_name.data.lower() + "_in_mbqc")
                    arguments = []
                    for param in op.params:
                        arguments.append(param)
                    for qubit in op.in_qubits:
                        arguments.append(qubit)

                    return_types = self.subroutine_dict[
                        op.gate_name.data
                    ].function_type.outputs.data
                    callOp = func.CallOp(callee, arguments, return_types)
                    rewriter.insert_op(callOp, InsertPoint.before(op))
                    for i, out_qubit in enumerate(op.out_qubits):
                        rewriter.replace_all_uses_with(out_qubit, callOp.results[i])
                    rewriter.erase_op(op)

                elif isinstance(op, GlobalPhaseOp) or (
                    isinstance(op, CustomOp) and op.gate_name.data in _PAULIS
                ):
                    continue
                elif isinstance(op, CustomOp):
                    raise NotImplementedError(
                        f"{op.gate_name.data} cannot be converted to the MBQC formalism."
                    )
