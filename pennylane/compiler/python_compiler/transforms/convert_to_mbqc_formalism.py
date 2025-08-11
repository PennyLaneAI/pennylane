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

import networkx as nx
from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin, func, scf
from xdsl.dialects.scf import ForOp, IfOp, WhileOp
from xdsl.ir import SSAValue
from xdsl.ir.core import OpResult
from xdsl.rewriter import InsertPoint

from ..dialects.mbqc import MeasureInBasisOp, MeasurementPlaneAttr, MeasurementPlaneEnum
from ..dialects.quantum import AllocQubitOp, CustomOp, DeallocQubitOp, QubitType
from .api import compiler_transform


def _generate_graph(op_name: str):
    """Generate a network graph to represent the connectivity of auxiliary qubits of
    a gate.
    Args:
        op_name (str): Gate name.
    Returns:
        A graph represents the connectivity of auxiliary qubits.
    """
    if op_name in ["RotXZX", "RZ", "Hadamard", "S"]:
        return _generate_one_wire_op_lattice()
    if op_name == "CNOT":
        return _generate_cnot_graph()
    raise NotImplementedError(f"{op_name} is not supported in the MBQC formalism.")


def _generate_cnot_graph():
    """Generate a networkx graph to represent the connectivity of auxiliary qubits of
    a CNOT gate based on the textbook MBQC formalism. Note that wire 1 is the control
    wire and wire 9 is the target wire as described in the Fig.2 of
    [`arXiv:quant-ph/0301052 <https://arxiv.org/abs/quant-ph/0301052>`_].

    Returns:
        A graph represents the connectivity of auxiliary qubits of a CNOT gate.
    """
    g = nx.Graph()
    wires = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]
    edges = [
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (4, 8),
        (8, 12),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
    ]
    g.add_nodes_from(wires)
    g.add_edges_from(edges)
    return g


def _generate_one_wire_op_lattice():
    """Generate lattice graph for a one-wire gate based on the textbook MBQC formalism.
    Note wire 1 is the target wire in the Fig. 2 of [`arXiv:quant-ph/0301052 <https://arxiv.org/abs/quant-ph/0301052>`_].

    Returns:
        A graph represents the connectivity of auxiliary qubits of a one-wire gate.
    """
    g = nx.Graph()
    wires = [2, 3, 4, 5]
    edges = [
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    g.add_nodes_from(wires)
    g.add_edges_from(edges)
    return g


@dataclass(frozen=True)
class ConvertToMBQCFormalismPass(passes.ModulePass):
    """Pass that converts gates in the MBQC gate set to the MBQC formalism."""

    name = "convert-to-mbqc-formalism"

    # pylint: disable=arguments-renamed,no-self-use
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
):  # pylint: disable=too-few-public-methods, no-self-use, unpacking-non-sequence
    """RewritePattern for converting to the MBQC formalism."""

    def _prep_graph_state(self, op: CustomOp, rewriter: pattern_rewriter.PatternRewriter):
        """Allocate auxiliary qubits and prepare a graph state for auxiliary qubits. Auxiliary qubits are
        entangled in the way described in the textbook MBQC form. **Note that all ops
        inserted into the IR in this function could be abstracted into a primitive that
        return a graph state that can be measured directly.**

        Args:
            op (CustomOp) : A `CustomOp` object. Note that op here is a quantum.customop object
                 instead of a qml.ops.
            rewriter (pattern_rewriter.PatternRewriter): A PatternRewriter object.

        Return:
            graph_qubits_dict : A dictionary of auxiliary qubits in the graph state. The keys represents
            the indices of qubits described in the [`arXiv:quant-ph/0301052 <https://arxiv.org/abs/quant-ph/0301052>`_].
        """
        graph = _generate_graph(op.gate_name.data)

        graph_qubits_dict = {}

        nodes = graph.nodes
        edges = graph.edges

        # Create auxiliary qubit allocation objects
        for node in nodes:
            graph_qubits_dict[node] = AllocQubitOp()

        # Insert auxiliary qubit allocation objects into the IR
        for _, qubit in graph_qubits_dict.items():
            rewriter.insert_op(qubit, InsertPoint.before(op))

        # Prepare auxiliary qubits by applying Hadamard ops before entanglement
        for node in nodes:
            in_qubits = graph_qubits_dict[node]
            # Create a Hadamard gate object for each auxiliary qubit
            HadamardOp = CustomOp(in_qubits=in_qubits, gate_name="Hadamard")
            # Insert the newly created Hadamard gate object to the IR
            rewriter.insert_op(HadamardOp, InsertPoint.before(op))
            # Ensure the qubits in the graph_qubits_dict are updated
            graph_qubits_dict[node] = HadamardOp.results[0]

        # Apply CZ gate to entangle each nearest auxiliary qubit pair
        for edge in edges:
            in_qubits = [graph_qubits_dict[node] for node in edge]
            # Create a CZ gate object for each auxiliary qubit pair
            CZOp = CustomOp(in_qubits=in_qubits, gate_name="CZ")
            # Insert the newly created CZ gate object to the IR
            rewriter.insert_op(CZOp, InsertPoint.before(op))
            # Ensure the qubits in the graph_qubits_dict are updated
            graph_qubits_dict[edge[0]], graph_qubits_dict[edge[1]] = (
                CZOp.results[0],
                CZOp.results[1],
            )

        return graph_qubits_dict

    def _insert_arbitrary_basis_measure_op(
        self,
        angle: float,
        plane: str,
        qubit: QubitType,
        insert_before: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Insert an arbitrary basis measure related operations to the IR.
        Args:
            angle (float) : Angle of the measurement basis.
            plane (str): Plane of the measurement basis.
            qubit (QubitType) : The target qubit to be measured.
            insert_before (CustomOp) : A `CustomOp` object to be used as the insertion point for the `measure` op.
            rewriter (pattern_rewriter.PatternRewriter): A PatternRewriter object.

        Returns:
            The measurement results include the result qubit and the measurement result.
        """
        in_qubit = qubit
        planeOp = MeasurementPlaneAttr(MeasurementPlaneEnum(plane))
        # Create a constant op from a float64 variable
        constAngleOp = arith.ConstantOp(builtin.FloatAttr(data=angle, type=builtin.Float64Type()))
        # Insert the constant op into the IR
        rewriter.insert_op(constAngleOp, InsertPoint.before(insert_before))
        # Create a MeasureInBasisOp op
        measureOp = MeasureInBasisOp(in_qubit=in_qubit, plane=planeOp, angle=constAngleOp)
        # Insert the newly created measureOp into the IR
        rewriter.insert_op(measureOp, InsertPoint.before(insert_before))
        # Returns the results of the newly created measureOp.
        # The results include: 1, a measurement result; 2, a result qubit.
        return measureOp.results

    def _measure_x_op(self, qubit, op, rewriter: pattern_rewriter.PatternRewriter):
        """Insert a X-basis measure op related operations to the IR."""
        return self._insert_arbitrary_basis_measure_op(0.0, "XY", qubit, op, rewriter)

    def _measure_y_op(self, qubit, op, rewriter: pattern_rewriter.PatternRewriter):
        """Insert a Y-basis measure op related operations to the IR."""
        return self._insert_arbitrary_basis_measure_op(math.pi / 2, "XY", qubit, op, rewriter)

    def _cond_insert_arbitrary_basis_measure_op(
        self,
        prev_mres: builtin.IntegerType,
        angle: SSAValue[builtin.Float64Type],
        plane: str,
        qubit: QubitType,
        op: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """
        Insert a conditional arbitrary basis measurement operation based on a previous measurement result.
        Args:
            pre_mres (builtin.IntegerType) : A previous measurement result.
            angle (SSAValue[builtin.Float64Type]) : An angle SSAValue from a parametric gate operation. Note that
                `_insert_arbitrary_basis_measure_op` accepts a float object instead.
            plane (str): Plane of the measurement basis.
            qubit (QubitType) : The target qubit to be measured.
            op (CustomOp) : A `CustomOp` object.
            rewriter (pattern_rewriter.PatternRewriter): A PatternRewriter object.

        Returns:
            The results include: 1. a measurement result; 2, a result qubit.
        """
        # Create a const op, which hold the integer `1`.
        constantOneOp = arith.ConstantOp.from_int_and_width(1, builtin.i1)
        # Insert the const op into the IR
        rewriter.insert_op(constantOneOp, InsertPoint.before(op))
        # Create a CmpiOp object which compares a measurement result with the const one object
        cmpOp = arith.CmpiOp(prev_mres, constantOneOp, "eq")
        # Insert the newly created CmpiOp object into the IR
        rewriter.insert_op(cmpOp, InsertPoint.before(op))

        in_qubit = qubit
        planeOp = MeasurementPlaneAttr(MeasurementPlaneEnum(plane))

        # Create a MeasureInBasisOp op for the true region
        measureOp = MeasureInBasisOp(in_qubit=in_qubit, plane=planeOp, angle=angle)

        # Create a const object hold the `-angle` value
        constNegAngleOp = arith.NegfOp(angle)

        # Create a MeasureInBasisOp op for the false region
        # TODOs: Need confirmation on if we have to insert ops created for the false and true regions to the IR.
        # It seems that the code here [https://github.com/xdslproject/xdsl/blob/37fceab602d98efbb2ba7ecd5548aa657eed558d/tests/interpreters/test_scf_interpreter.py#L64-L74]
        # does not explicitly insert the Ops created to the IR. Checking the result IR after applying current implementation
        # of this pass, all Ops created in this block are inserted to the IR. Need some experts to confirm if it's the best practice.
        measureNegOp = MeasureInBasisOp(
            in_qubit=in_qubit, plane=planeOp, angle=constNegAngleOp.result
        )
        # Create the true region
        true_region = [measureOp, scf.YieldOp(measureOp.results[0], measureOp.results[1])]
        # Create the false region
        false_region = [
            constNegAngleOp,
            measureNegOp,
            scf.YieldOp(measureNegOp.results[0], measureNegOp.results[1]),
        ]

        # Create a If control flow Op
        condOp = IfOp(
            cmpOp,
            (builtin.IntegerType(1), QubitType()),
            true_region,
            false_region,
        )
        # Insert condOp to the IR
        rewriter.insert_op(condOp, InsertPoint.before(op))

        # Return the result of if control flow
        return condOp.results

    def _hadamard_measurements(
        self, graph_qubits_dict, op, rewriter: pattern_rewriter.PatternRewriter
    ):
        """Insert measurement ops for a Hadamard gate and return measurement results and the result graph qubits"""
        m1, graph_qubits_dict[1] = self._measure_x_op(graph_qubits_dict[1], op, rewriter)
        m2, graph_qubits_dict[2] = self._measure_y_op(graph_qubits_dict[2], op, rewriter)
        m3, graph_qubits_dict[3] = self._measure_y_op(graph_qubits_dict[3], op, rewriter)
        m4, graph_qubits_dict[4] = self._measure_y_op(graph_qubits_dict[4], op, rewriter)
        return [m1, m2, m3, m4], graph_qubits_dict

    def _s_measurements(self, graph_qubits_dict, op, rewriter: pattern_rewriter.PatternRewriter):
        """Insert measurement ops for a S gate and return measurement results and the result graph qubits"""
        m1, graph_qubits_dict[1] = self._measure_x_op(graph_qubits_dict[1], op, rewriter)
        m2, graph_qubits_dict[2] = self._measure_x_op(graph_qubits_dict[2], op, rewriter)
        m3, graph_qubits_dict[3] = self._measure_y_op(graph_qubits_dict[3], op, rewriter)
        m4, graph_qubits_dict[4] = self._measure_x_op(graph_qubits_dict[4], op, rewriter)
        return [m1, m2, m3, m4], graph_qubits_dict

    def _cnot_measurements(self, graph_qubits_dict, op, rewriter: pattern_rewriter.PatternRewriter):
        """Insert measurement ops for a CNOT gate and return measurement results and the result graph qubits"""
        m1, graph_qubits_dict[1] = self._measure_x_op(graph_qubits_dict[1], op, rewriter)
        m2, graph_qubits_dict[2] = self._measure_y_op(graph_qubits_dict[2], op, rewriter)
        m3, graph_qubits_dict[3] = self._measure_y_op(graph_qubits_dict[3], op, rewriter)
        m4, graph_qubits_dict[4] = self._measure_y_op(graph_qubits_dict[4], op, rewriter)
        m5, graph_qubits_dict[5] = self._measure_y_op(graph_qubits_dict[5], op, rewriter)
        m6, graph_qubits_dict[6] = self._measure_y_op(graph_qubits_dict[6], op, rewriter)
        m8, graph_qubits_dict[8] = self._measure_y_op(graph_qubits_dict[8], op, rewriter)
        m9, graph_qubits_dict[9] = self._measure_x_op(graph_qubits_dict[9], op, rewriter)
        m10, graph_qubits_dict[10] = self._measure_x_op(graph_qubits_dict[10], op, rewriter)
        m11, graph_qubits_dict[11] = self._measure_x_op(graph_qubits_dict[11], op, rewriter)
        m12, graph_qubits_dict[12] = self._measure_y_op(graph_qubits_dict[12], op, rewriter)
        m13, graph_qubits_dict[13] = self._measure_x_op(graph_qubits_dict[13], op, rewriter)
        m14, graph_qubits_dict[14] = self._measure_x_op(graph_qubits_dict[14], op, rewriter)

        return [m1, m2, m3, m4, m5, m6, m8, m9, m10, m11, m12, m13, m14], graph_qubits_dict

    def _rz_measurements(self, graph_qubits_dict, op, rewriter: pattern_rewriter.PatternRewriter):
        """Insert measurement ops for a RZ gate and return measurement results and the result graph qubits"""
        m1, graph_qubits_dict[1] = self._measure_x_op(graph_qubits_dict[1], op, rewriter)
        m2, graph_qubits_dict[2] = self._measure_x_op(graph_qubits_dict[2], op, rewriter)
        m3, graph_qubits_dict[3] = self._cond_insert_arbitrary_basis_measure_op(
            m2, op.params[0], "XY", graph_qubits_dict[3], op, rewriter
        )
        m4, graph_qubits_dict[4] = self._measure_x_op(graph_qubits_dict[4], op, rewriter)
        return [m1, m2, m3, m4], graph_qubits_dict

    def _rotxzx_measurements(
        self, graph_qubits_dict, op, rewriter: pattern_rewriter.PatternRewriter
    ):
        """Insert measurement ops for a RotXZX gate and return measurement results and the result graph qubits"""
        m1, graph_qubits_dict[1] = self._measure_x_op(graph_qubits_dict[1], op, rewriter)
        m2, graph_qubits_dict[2] = self._cond_insert_arbitrary_basis_measure_op(
            m1, op.params[0], "XY", graph_qubits_dict[2], op, rewriter
        )
        m3, graph_qubits_dict[3] = self._cond_insert_arbitrary_basis_measure_op(
            m2, op.params[1], "XY", graph_qubits_dict[3], op, rewriter
        )

        m1_xor_m3 = arith.XOrIOp(m1, m3)
        rewriter.insert_op(m1_xor_m3, InsertPoint.before(op))

        m4, graph_qubits_dict[4] = self._cond_insert_arbitrary_basis_measure_op(
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
            gate_name (str) : The name of gate to be corrected.
            qubit (QubitType) : The result auxiliary qubit to be corrected.
            op (CustomOp) : A gate operation object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.

        Return:
            The result auxiliary qubit.
        """
        # Create a const op, which hold the integer `1`.
        constantOneOp = arith.ConstantOp.from_int_and_width(1, builtin.i1)
        # Insert the const op into the IR
        rewriter.insert_op(constantOneOp, InsertPoint.before(op))
        # Crate a CmpiOp object which compares a parity check result with the const one object
        cmpOp = arith.CmpiOp(parity_res, constantOneOp, "eq")
        # Insert the newly created CmpiOp object into the IR
        rewriter.insert_op(cmpOp, InsertPoint.before(op))
        in_qubit = qubit

        # Create a byproductOp object
        byproductOp = CustomOp(in_qubits=in_qubit, gate_name=gate_name)
        true_region = [byproductOp, scf.YieldOp(byproductOp.results[0])]

        # Create an `Identity` gate object
        identityOp = CustomOp(in_qubits=in_qubit, gate_name="Identity")
        # TODOs check if we can set false_region = [scf.YieldOp(in_qubit)]
        # The answer is no as xDSL==0.46.0, the error msg is:
        # "'NoneType' object has no attribute 'add_use'""
        false_region = [identityOp, scf.YieldOp(identityOp.results[0])]
        condOp = IfOp(cmpOp, QubitType(), true_region, false_region)
        # Insert condOp to the IR
        rewriter.insert_op(condOp, InsertPoint.before(op))
        return condOp.results

    def _parity_check(
        self,
        mres: list[builtin.IntegerType],
        op: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
        add_const_one: bool = False,
    ):
        """Insert parity check related operations to the IR.
        Args:
            mres (list[builtin.IntegerType]): A list of the mid-measurement results.
            op (CustomOp) : A gate operation object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.
            add_const_one (bool) : Whether we need to add a const one to get the parity or not. Defaults to False.

        Returns:
            The result of parity check.
        """
        prev_res = mres[0]
        addOp = None
        # Create add ops to sum up the mres and insert them to the IR
        for i in range(1, len(mres)):
            addOp = arith.AddiOp(prev_res, mres[i])
            rewriter.insert_op(addOp, InsertPoint.before(op))
            prev_res = addOp.result

        # Create add const one ops and insert them to the IR
        constantOneOp = arith.ConstantOp.from_int_and_width(1, builtin.i1)
        rewriter.insert_op(constantOneOp, InsertPoint.before(op))

        # Create an add op to add an additional const one and insert ops to the IR
        if add_const_one:
            addOp = arith.AddiOp(prev_res, constantOneOp)
            rewriter.insert_op(addOp, InsertPoint.before(op))
            prev_res = addOp.result

        # Add a xor op for parity check and insert it to the IR
        xorOp = arith.XOrIOp(prev_res, constantOneOp)
        rewriter.insert_op(xorOp, InsertPoint.before(op))

        # Return the parity check result
        return xorOp.result

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
            op (CustomOp) : A gate operatio object.
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
            op (CustomOp) : A gate operatio object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.

        Returns:
            The result auxiliary qubit.
        """
        m1, m2, m3, m4 = mres

        # X correction
        x_parity = self._parity_check([m2, m4], op, rewriter)
        res_aux_qubit = self._insert_cond_byproduct_op(x_parity, "PauliX", qubit, op, rewriter)

        # Z correction
        z_parity = self._parity_check([m1, m2, m3], op, rewriter, add_const_one=True)
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
            op (CustomOp) : A gate operatio object.
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
            op (CustomOp) : A gate operatio object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.

        Returns:
            The result auxiliary qubits.
        """
        m1, m2, m3, m4, m5, m6, m8, m9, m10, m11, m12, m13, m14 = mres
        # Corrections for the control qubit
        x_parity = self._parity_check([m2, m3, m5, m6], op, rewriter)
        ctrl_aux_qubit = self._insert_cond_byproduct_op(x_parity, "PauliX", qubits[0], op, rewriter)
        z_parity = self._parity_check(
            [m1, m3, m4, m5, m8, m9, m11], op, rewriter, add_const_one=True
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

    def _queue_byprod_corrections(
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
            op (CustomOp) : A gate operatio object.
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

    def _swap_qb_in_reg_aux_res_qb(
        self,
        qb_in_reg: QubitType,
        aux_res_qubit: QubitType,
        op: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):
        """Swap the target qubit in the global register with the corresonding result auxiliary qubit.
        Args:
            qb_in_reg (QubitType): A qubit in the global qubit register.
            aux_res_qubit (QubitType): The result auxiliary qubit.
            op (CustomOp) : A gate operatio object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.

        Return:
            The result qubits, which are the swapping result of the qubit in the global register and the auxiliary result qubit.
        """
        # NOTE: IdentityOp inserted here is a temporal solution to fix the issue that SWAPOp can't accept
        # `res_aux_qubit` as an in_qubits variable, i.e. SWAPOp = CustomOp(in_qubits=(target_qubit, res_aux_qubit), gate_name="SWAP").
        # The error message would be "| Error while applying pattern: 'NoneType' object has no attribute 'add_use'". Is it a upstream
        # issue or caused by the way how we define the `CustomOp` operation? Not sure why.
        # NOTE: It is also worth noting here. The target qubit in the global register is measured and I assume that
        # physically the target qubit can not be swapped with the result auxiliary qubit, right?
        # TODOS: We need more clarifications for the questions above in the following steps.
        IdentityOp = CustomOp(in_qubits=(aux_res_qubit), gate_name="Identity")
        rewriter.insert_op(IdentityOp, InsertPoint.before(op))

        return (IdentityOp.results[0], qb_in_reg)

    def _deallocate_aux_qubits(
        self,
        graph_qubits_dict: dict,
        qb_in_reg_key: list[int],
        op: CustomOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):
        """Deallocate the auxiliary qubits in the graph qubit dict.
        Args:
            graph_qubits_dict (dict) : A dict stores all qubits in a graph state.
            qb_in_reg_key (list[int]) : A list of keys to qubits in the global register.
            op (CustomOp) : A gate operatio object.
            rewriter (pattern_rewriter.PatternRewriter): A pattern rewriter.
        """
        # Deallocate aux_qubits
        for node in graph_qubits_dict:
            if node not in qb_in_reg_key:
                deallocQubitOp = DeallocQubitOp(graph_qubits_dict[node])
                rewriter.insert_op(deallocQubitOp, InsertPoint.before(op))

    # pylint: disable=no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, root: func.FuncOp | IfOp | WhileOp | ForOp, rewriter: pattern_rewriter.PatternRewriter
    ):  # pylint: disable=arguments-differ, cell-var-from-loop
        """Match and rewrite for converting to the MBQC formalism."""

        for region in root.regions:
            for op in region.ops:
                if isinstance(op, CustomOp) and op.gate_name.data in [
                    "Hadamard",
                    "S",
                    "RZ",
                    "RotXZX",
                ]:
                    # Allocate auxiliary qubits and entangle them
                    graph_qubits_dict = self._prep_graph_state(op, rewriter)

                    # Entangle the op.in_qubits[0] with the graph_qubits_dict[2]
                    CZOp = CustomOp(
                        in_qubits=[op.in_qubits[0], graph_qubits_dict[2]], gate_name="CZ"
                    )
                    rewriter.insert_op(CZOp, InsertPoint.before(op))

                    # Update the graph qubit dict
                    graph_qubits_dict[1], graph_qubits_dict[2] = CZOp.results

                    # Insert measurement ops to the IR
                    mres, graph_qubits_dict = self._queue_measurements(
                        graph_qubits_dict, op, rewriter
                    )

                    # Insert byproduct ops to the IR
                    graph_qubits_dict[5] = self._queue_byprod_corrections(
                        mres, graph_qubits_dict[5], op, rewriter
                    )

                    # Swap the target qubit in the global register with the result auxiliary qubit
                    graph_qubits_dict[1], graph_qubits_dict[5] = self._swap_qb_in_reg_aux_res_qb(
                        graph_qubits_dict[1], graph_qubits_dict[5], op, rewriter
                    )

                    # Deallocate the auxiliary qubits
                    self._deallocate_aux_qubits(graph_qubits_dict, [1, 5], op, rewriter)

                    # Replace all uses of output qubit of op with the result_qubit
                    rewriter.replace_all_uses_with(op.results[0], graph_qubits_dict[1])
                    # Remove op operation
                    rewriter.erase_op(op)
                elif isinstance(op, CustomOp) and op.gate_name.data == "CNOT":
                    # Allocate auxiliary qubits and entangle them
                    graph_qubits_dict = self._prep_graph_state(op, rewriter)

                    # Entangle the op.in_qubits[0] with the graph_qubits_dict[2]
                    CZOp = CustomOp(
                        in_qubits=[op.in_qubits[0], graph_qubits_dict[2]], gate_name="CZ"
                    )
                    rewriter.insert_op(CZOp, InsertPoint.before(op))
                    graph_qubits_dict[1], graph_qubits_dict[2] = CZOp.results

                    # Entangle op.in_qubits[1] with with the graph_qubits_dict[10] for a CNOT gate
                    CZOp = CustomOp(
                        in_qubits=[op.in_qubits[1], graph_qubits_dict[10]], gate_name="CZ"
                    )
                    rewriter.insert_op(CZOp, InsertPoint.before(op))
                    graph_qubits_dict[9], graph_qubits_dict[10] = CZOp.results

                    # Insert measurement ops to the IR
                    mres, graph_qubits_dict = self._queue_measurements(
                        graph_qubits_dict, op, rewriter
                    )

                    # Insert byproduct ops to the IR
                    graph_qubits_dict[7], graph_qubits_dict[15] = self._queue_byprod_corrections(
                        mres, [graph_qubits_dict[7], graph_qubits_dict[15]], op, rewriter
                    )

                    # Swap the ctrl qubit in the global register with the result auxiliary qubit
                    graph_qubits_dict[1], graph_qubits_dict[7] = self._swap_qb_in_reg_aux_res_qb(
                        graph_qubits_dict[1], graph_qubits_dict[7], op, rewriter
                    )

                    # Swap the target qubit in the global register with the result auxiliary qubit
                    graph_qubits_dict[9], graph_qubits_dict[15] = self._swap_qb_in_reg_aux_res_qb(
                        graph_qubits_dict[9], graph_qubits_dict[15], op, rewriter
                    )

                    # Deallocate aux_qubits
                    self._deallocate_aux_qubits(graph_qubits_dict, [1, 7, 9, 15], op, rewriter)

                    # Replace all uses of output qubit of op with the result_qubit
                    rewriter.replace_all_uses_with(op.results[0], graph_qubits_dict[1])
                    rewriter.replace_all_uses_with(op.results[1], graph_qubits_dict[9])
                    # Remove op operation
                    rewriter.erase_op(op)
