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

from dataclasses import dataclass

import networkx as nx
from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin, func, memref, tensor, vector
from xdsl.dialects.scf import ForOp, IfOp, WhileOp, YieldOp
from xdsl.rewriter import InsertPoint

from pennylane.ftqc import generate_lattice
from pennylane.ops import CZ, H

from ..dialects.mbqc import MeasureInBasisOp, MeasurementPlaneAttr, MeasurementPlaneEnum
from ..dialects.quantum import AllocQubitOp, CustomOp, DeallocQubitOp
from .api import compiler_transform


def _generate_cnot_lattice():
    """Generate lattice graph for a CNOT gate based on the textbook MBQC formalism."""
    wires = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]
    g = nx.Graph()
    g.add_nodes_from(wires)
    g.add_edges_from(
        [
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
    )
    return g


def _generate_one_wire_op_lattice():
    """Generate lattice graph for a one-wire gate based on the textbook MBQC formalism."""
    wires = [2, 3, 4, 5]
    g = nx.Graph()
    g.add_nodes_from(wires)
    g.add_edges_from(
        [
            (2, 3),
            (3, 4),
            (4, 5),
        ]
    )
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
):  # pylint: disable=too-few-public-methods
    """RewritePattern for converting to the MBQC formalism."""

    def _make_graph_state(self, op, rewriter):
        """Make a graph state and return the allocated qubits"""
        graph = None
        if op.gate_name.data == "CNOT":
            graph = _generate_cnot_lattice()
        elif op.gate_name.data in [
            "Hadamard",
            "S",
            "RZ",
            "RotXZX",
        ]:
            graph = _generate_one_wire_op_lattice()
        else:
            raise NotImplementedError(f"{op.gate_name.data} gate is not implemented.")

        nodes = graph.nodes
        edges = graph.edges

        # Allocate qubits
        aux_qubits_dict = {}
        for node in nodes:
            aux_qubits_dict[node] = AllocQubitOp()
        # Insert qubit allocation ops before the op
        for aux_qubit in aux_qubits_dict.values:
            rewriter.insert_op(aux_qubit, insert_point=InsertPoint.before(op))

        # Apply Hadamard gate to each auxiliary qubit
        for node in nodes:
            in_qubits = aux_qubits_dict[node]
            gate_name = "Hadamard"
            HadamardOp = CustomOp(in_qubits=in_qubits, gate_name=gate_name)
            rewriter.insert_op(HadamardOp, insert_point=InsertPoint.before(op))

        # Apply CZ gate to entangle each nearest auxiliary qubit pair
        for edge in edges:
            in_qubits = [aux_qubits_dict[node] for node in edge]
            gate_name = "CZ"
            CZOp = CustomOp(in_qubits=in_qubits, gate_name=gate_name)
            rewriter.insert_op(CZOp, insert_point=InsertPoint.before(op))

        return aux_qubits_dict

    def _insert_arbitary_basis_measure_op(self, angle, plane, qubit, op, rewriter):
        """Insert arbitary basis measure related operations before the op operation."""
        in_qubit = qubit
        planeOp = MeasurementPlaneAttr(MeasurementPlaneEnum(plane))
        constAngleOp = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(
                type=builtin.TensorType(builtin.Float64Type(), shape=()), data=(angle,)
            )
        )
        # Insert the constant angleOP
        rewriter.insert_op(constAngleOp, insert_point=InsertPoint.before(op))
        measureOp = MeasureInBasisOp(in_qubit=in_qubit, plane=planeOp, angle=constAngleOp)
        # Insert measureOp
        rewriter.insert_op(measureOp, insert_point=InsertPoint.before(op))
        return measureOp.results

    def _cond_insert_arbitary_basis_measure_op(self, cond, angle, plane, qubit, op, rewriter):
        in_qubit = qubit
        planeOp = MeasurementPlaneAttr(MeasurementPlaneEnum(plane))
        constAngleOp = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(
                type=builtin.TensorType(builtin.Float64Type(), shape=()), data=(angle,)
            )
        )
        measureOp = MeasureInBasisOp(in_qubit=in_qubit, plane=planeOp, angle=constAngleOp)
        constNegAngleOp = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(
                type=builtin.TensorType(builtin.Float64Type(), shape=()), data=(-angle,)
            )
        )
        measureNegOp = MeasureInBasisOp(in_qubit=in_qubit, plane=planeOp, angle=constNegAngleOp)
        ture_region = [planeOp, constAngleOp, measureOp, YieldOp()]
        false_region = [planeOp, constNegAngleOp, measureNegOp, YieldOp()]
        condOp = IfOp(
            cond=cond, return_types=[], ture_region=ture_region, false_region=false_region
        )
        # Insert condOp before op operations
        rewriter.insert_op(condOp, insert_point=InsertPoint.before(op))
        return condOp.results

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
                    aux_qubits_dict = self._make_graph_state(op, rewriter)
                    # Entangle the target wire to qubit 2 in the aux_qubits_dict
                    in_qubits = [op.in_qubits[0], aux_qubits_dict[2]]
                    gate_name = "CZ"
                    CZOp = CustomOp(in_qubits=in_qubits, gate_name=gate_name)
                    rewriter.insert_op(CZOp, insert_point=InsertPoint.before(op))

                    # Insert measurement Op before the op operation

                    # Swap the target qubit with the output qubit
                    in_qubits = [op.in_qubits[0], aux_qubits_dict[5]]
                    gate_name = "SWAP"
                    SWAPOp = CustomOp(in_qubits=in_qubits, gate_name=gate_name)
                    rewriter.insert_op(SWAPOp, insert_point=InsertPoint.before(op))

                    # Deallocate aux_qubits except for the last qubit
                    for _, aux_qubit in aux_qubits_dict:
                        deallocQubitOp = DeallocQubitOp(aux_qubit)
                        rewriter.insert_op(deallocQubitOp, insertion_point=InsertPoint.after(op))

                    # Erase the current operation
                    rewriter.erase_op(op)
