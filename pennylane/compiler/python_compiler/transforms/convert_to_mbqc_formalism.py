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
from xdsl.rewriter import InsertPoint

from ..dialects.mbqc import MeasureInBasisOp, MeasurementPlaneAttr, MeasurementPlaneEnum
from ..dialects.quantum import AllocQubitOp, CustomOp, DeallocQubitOp, QubitType
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
        for node in aux_qubits_dict:
            rewriter.insert_op(aux_qubits_dict[node], InsertPoint.before(op))

        # Apply Hadamard gate to each auxiliary qubit
        for node in nodes:
            in_qubits = aux_qubits_dict[node]
            gate_name = "Hadamard"
            HadamardOp = CustomOp(in_qubits=in_qubits, gate_name=gate_name)
            rewriter.insert_op(HadamardOp, InsertPoint.before(op))
            aux_qubits_dict[node] = HadamardOp.results[0]

        # Apply CZ gate to entangle each nearest auxiliary qubit pair
        for edge in edges:
            in_qubits = [aux_qubits_dict[node] for node in edge]
            gate_name = "CZ"
            CZOp = CustomOp(in_qubits=in_qubits, gate_name=gate_name)
            rewriter.insert_op(CZOp, InsertPoint.before(op))
            aux_qubits_dict[edge[0]], aux_qubits_dict[edge[1]] = CZOp.results[0], CZOp.results[1]

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
        rewriter.insert_op(constAngleOp, InsertPoint.before(op))
        measureOp = MeasureInBasisOp(in_qubit=in_qubit, plane=planeOp, angle=constAngleOp)
        # Insert measureOp
        rewriter.insert_op(measureOp, InsertPoint.before(op))
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
        ture_region = [planeOp, constAngleOp, measureOp, scf.YieldOp(measureOp.results)]
        false_region = [planeOp, constNegAngleOp, measureNegOp, scf.YieldOp(measureNegOp.results)]
        condOp = IfOp(
            cond=cond, return_types=[], ture_region=ture_region, false_region=false_region
        )
        # Insert condOp before op operations
        rewriter.insert_op(condOp, InsertPoint.before(op))
        return condOp.results

    def _insert_byproduct_op(self, cond, gate_name, qubit, op, rewriter):
        in_qubit = qubit
        byproductOp = CustomOp(in_qubits=in_qubit, gate_name=gate_name)
        ture_region = [byproductOp, scf.YieldOp(byproductOp.results[0])]
        identityOp = CustomOp(in_qubits=in_qubit, gate_name="Identity")
        false_region = [identityOp, scf.YieldOp(identityOp.results[0])]
        condOp = IfOp(cond, QubitType(), ture_region, false_region)
        # Insert condOp before op operations
        rewriter.insert_op(condOp, InsertPoint.before(op))
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
                    rewriter.insert_op(CZOp, InsertPoint.before(op))

                    target_qubit, aux_qubits_dict[2] = CZOp.results
                    res_aux_qubit = aux_qubits_dict[5]

                    # Insert measurement Op before the op operation
                    if op.gate_name.data == "Hadamard":
                        m1, qubit1 = self._insert_arbitary_basis_measure_op(
                            angle=0.0, plane="XY", qubit=target_qubit, op=op, rewriter=rewriter
                        )
                        target_qubit = qubit1
                        m2, qubit2 = self._insert_arbitary_basis_measure_op(
                            angle=math.pi / 2,
                            plane="XY",
                            qubit=aux_qubits_dict[2],
                            op=op,
                            rewriter=rewriter,
                        )
                        aux_qubits_dict[2] = qubit2
                        m3, qubit3 = self._insert_arbitary_basis_measure_op(
                            angle=math.pi / 2,
                            plane="XY",
                            qubit=aux_qubits_dict[3],
                            op=op,
                            rewriter=rewriter,
                        )
                        aux_qubits_dict[3] = qubit3
                        m4, qubit4 = self._insert_arbitary_basis_measure_op(
                            angle=math.pi / 2,
                            plane="XY",
                            qubit=aux_qubits_dict[4],
                            op=op,
                            rewriter=rewriter,
                        )
                        aux_qubits_dict[4] = qubit4

                        # Insert by-product corrections
                        # x correction: m1, m3, m4
                        m13_sum_x = arith.AddiOp(m1, m3)
                        rewriter.insert_op(m13_sum_x, InsertPoint.before(op))
                        m134_sum_x = arith.AddiOp(m13_sum_x.result, m4)
                        rewriter.insert_op(m134_sum_x, InsertPoint.before(op))
                        # # Insert a conditional IfOp
                        constantOneOp = arith.ConstantOp.from_int_and_width(1, builtin.i1)
                        rewriter.insert_op(constantOneOp, InsertPoint.before(op))
                        xorOp = arith.XOrIOp(m134_sum_x.result, constantOneOp)
                        rewriter.insert_op(xorOp, InsertPoint.before(op))
                        cmpOp = arith.CmpiOp(xorOp.result, constantOneOp, "eq")
                        rewriter.insert_op(cmpOp, InsertPoint.before(op))

                        res_aux_qubit = self._insert_byproduct_op(
                            cmpOp.result, "PauliX", res_aux_qubit, op, rewriter
                        )

                        # z correction: m2, m3
                        m23_sum_z = arith.AddiOp(m2, m3)
                        rewriter.insert_op(m23_sum_z, InsertPoint.before(op))

                        xorOp = arith.XOrIOp(m23_sum_z.result, constantOneOp)
                        rewriter.insert_op(xorOp, InsertPoint.before(op))

                        cmpOp = arith.CmpiOp(xorOp.result, constantOneOp, "eq")
                        rewriter.insert_op(cmpOp, InsertPoint.before(op))

                        res_aux_qubit = self._insert_byproduct_op(
                            cmpOp.result, "PauliZ", res_aux_qubit, op, rewriter
                        )

                    # NOTE: IdentityOp inserted here is a temporal solution to fix the issue that SWAPOp can't accept
                    # `res_aux_qubit` as an in_qubits variable, i.e. SWAPOp = CustomOp(in_qubits=(target_qubit, res_aux_qubit), gate_name="SWAP").
                    # The error message would be "| Error while applying pattern: 'NoneType' object has no attribute 'add_use'". Is it a upstream
                    # issue or caused by the way how we define the `CustomOp` operation? Not sure why.
                    IdentityOp = CustomOp(in_qubits=(res_aux_qubit), gate_name="Identity")
                    rewriter.insert_op(IdentityOp, InsertPoint.before(op))
                    in_qubits = (target_qubit, IdentityOp.results[0])
                    SWAPOp = CustomOp(in_qubits=in_qubits, gate_name="SWAP")
                    rewriter.insert_op(SWAPOp, InsertPoint.before(op))
                    result_qubit, aux_qubits_dict[5] = SWAPOp.results

                    # Deallocate aux_qubits
                    for node in aux_qubits_dict:
                        deallocQubitOp = DeallocQubitOp(aux_qubits_dict[node])
                        rewriter.insert_op(deallocQubitOp, InsertPoint.before(op))

                    # Replace all uses of output qubit of op with the result_qubit
                    rewriter.replace_all_uses_with(op.results[0], result_qubit)
                    # Remove op operation
                    rewriter.erase_op(op)
