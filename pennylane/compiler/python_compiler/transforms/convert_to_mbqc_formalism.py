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
        constAngleOp = arith.ConstantOp(builtin.FloatAttr(data=angle, type=builtin.Float64Type()))
        # Insert the constant angleOP
        rewriter.insert_op(constAngleOp, InsertPoint.before(op))
        measureOp = MeasureInBasisOp(in_qubit=in_qubit, plane=planeOp, angle=constAngleOp)
        # Insert measureOp
        rewriter.insert_op(measureOp, InsertPoint.before(op))
        return measureOp.results

    def _cond_insert_arbitary_basis_measure_op(self, prev_mres, angle, plane, qubit, op, rewriter):
        constantOneOp = arith.ConstantOp.from_int_and_width(1, builtin.i1)
        rewriter.insert_op(constantOneOp, InsertPoint.before(op))
        cmpOp = arith.CmpiOp(prev_mres, constantOneOp, "eq")
        rewriter.insert_op(cmpOp, InsertPoint.before(op))
        in_qubit = qubit
        planeOp = MeasurementPlaneAttr(MeasurementPlaneEnum(plane))
        measureOp = MeasureInBasisOp(in_qubit=in_qubit, plane=planeOp, angle=angle)

        constNegAngleOp = arith.NegfOp(angle)

        measureNegOp = MeasureInBasisOp(
            in_qubit=in_qubit, plane=planeOp, angle=constNegAngleOp.result
        )
        ture_region = [measureOp, scf.YieldOp(measureOp.results[0], measureOp.results[1])]
        false_region = [
            constNegAngleOp,
            measureNegOp,
            scf.YieldOp(measureNegOp.results[0], measureNegOp.results[1]),
        ]
        condOp = IfOp(
            cmpOp,
            (builtin.IntegerType(1), QubitType()),
            ture_region,
            false_region,
        )
        # Insert condOp before op operations
        rewriter.insert_op(condOp, InsertPoint.before(op))
        return condOp.results

    def _insert_byprod_exp_op(self, mres, insert_before_op, rewriter, add_one_op=False):
        prev_res = mres[0]
        addOp = None
        for i in range(1, len(mres)):
            addOp = arith.AddiOp(prev_res, mres[i])
            rewriter.insert_op(addOp, InsertPoint.before(insert_before_op))
            prev_res = addOp.result

        constantOneOp = arith.ConstantOp.from_int_and_width(1, builtin.i1)
        rewriter.insert_op(constantOneOp, InsertPoint.before(insert_before_op))

        if add_one_op:
            addOp = arith.AddiOp(prev_res, constantOneOp)
            rewriter.insert_op(addOp, InsertPoint.before(insert_before_op))
            prev_res = addOp.result

        xorOp = arith.XOrIOp(prev_res, constantOneOp)
        rewriter.insert_op(xorOp, InsertPoint.before(insert_before_op))
        return xorOp.result

    def _insert_cond_byproduct_op(self, exp_index, gate_name, qubit, op, rewriter):
        constantOneOp = arith.ConstantOp.from_int_and_width(1, builtin.i1)
        rewriter.insert_op(constantOneOp, InsertPoint.before(op))
        cmpOp = arith.CmpiOp(exp_index, constantOneOp, "eq")
        rewriter.insert_op(cmpOp, InsertPoint.before(op))
        in_qubit = qubit
        byproductOp = CustomOp(in_qubits=in_qubit, gate_name=gate_name)
        ture_region = [byproductOp, scf.YieldOp(byproductOp.results[0])]
        identityOp = CustomOp(in_qubits=in_qubit, gate_name="Identity")
        false_region = [identityOp, scf.YieldOp(identityOp.results[0])]
        condOp = IfOp(cmpOp, QubitType(), ture_region, false_region)
        # Insert condOp before op operations
        rewriter.insert_op(condOp, InsertPoint.before(op))
        return condOp.results

    def _get_measurement_param_with_gatename(self, gatename):
        match gatename:
            case "Hadamard":
                xmres_add_one = False
                zmres_add_one = False
                x_mres_idx = [1, 3, 4]
                z_mres_idx = [2, 3]
                angles = {1: 0.0, 2: math.pi / 2, 3: math.pi / 2, 4: math.pi / 2}
                planes = {1: "XY", 2: "XY", 3: "XY", 4: "XY"}
                return x_mres_idx, z_mres_idx, angles, planes, xmres_add_one, zmres_add_one

            case "S":
                x_mres_idx = [2, 4]
                z_mres_idx = [1, 2, 3]
                angles = {1: 0.0, 2: 0.0, 3: math.pi / 2, 4: 0.0}
                planes = {1: "XY", 2: "XY", 3: "XY", 4: "XY"}
                xmres_add_one = False
                zmres_add_one = True
                return x_mres_idx, z_mres_idx, angles, planes, xmres_add_one, zmres_add_one

            case "CNOT":
                xc_mres_idx = [2, 3, 5, 6]
                xt_mres_idx = [2, 3, 8, 10, 12, 14]
                zc_mres_idx = [1, 3, 4, 5, 8, 9, 11]
                zt_mres_idx = [9, 11, 13]

                xcmres_add_one = False
                xtmres_add_one = False
                zcmres_add_one = True
                ztmres_add_one = True

                angles = {
                    1: 0.0,
                    2: math.pi / 2,
                    3: math.pi / 2,
                    4: math.pi / 2,
                    5: math.pi / 2,
                    6: math.pi / 2,
                    8: math.pi / 2,
                    9: 0.0,
                    10: 0.0,
                    11: 0.0,
                    12: math.pi / 2,
                    13: 0.0,
                    14: 0.0,
                }
                planes = {
                    1: "XY",
                    2: "XY",
                    3: "XY",
                    4: "XY",
                    5: "XY",
                    6: "XY",
                    8: "XY",
                    9: "XY",
                    10: "XY",
                    11: "XY",
                    12: "XY",
                    13: "XY",
                    14: "XY",
                }
                return (
                    xc_mres_idx,
                    xt_mres_idx,
                    zc_mres_idx,
                    zt_mres_idx,
                    xcmres_add_one,
                    xtmres_add_one,
                    zcmres_add_one,
                    ztmres_add_one,
                    angles,
                    planes,
                )

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

                    # TODOs: the following line assign the target qubit to aux_qubits_dict[1] could be confusing, it's subject to be refactored.
                    aux_qubits_dict[1], aux_qubits_dict[2] = CZOp.results
                    res_aux_qubit = aux_qubits_dict[5]

                    # Get measurement params with gate name

                    x_mres_idx, z_mres_idx, angles, planes, xmres_add_one, zmres_add_one = (
                        (self._get_measurement_param_with_gatename(op.gate_name.data))
                        if op.gate_name.data in ["Hadamard", "S"]
                        else [None] * 6
                    )

                    x_mres = []
                    z_mres = []
                    if op.gate_name.data in ["Hadamard", "S"]:
                        for key in angles:
                            mres, aux_qubits_dict[key] = self._insert_arbitary_basis_measure_op(
                                angle=angles[key],
                                plane=planes[key],
                                qubit=aux_qubits_dict[key],
                                op=op,
                                rewriter=rewriter,
                            )
                            if key in x_mres_idx:
                                x_mres.append(mres)
                            if key in z_mres_idx:
                                z_mres.append(mres)
                    if op.gate_name.data in ["RZ"]:
                        m1, aux_qubits_dict[1] = self._insert_arbitary_basis_measure_op(
                            angle=0.0,
                            plane="XY",
                            qubit=aux_qubits_dict[1],
                            op=op,
                            rewriter=rewriter,
                        )
                        m2, aux_qubits_dict[2] = self._insert_arbitary_basis_measure_op(
                            angle=0.0,
                            plane="XY",
                            qubit=aux_qubits_dict[2],
                            op=op,
                            rewriter=rewriter,
                        )
                        m3, aux_qubits_dict[3] = self._cond_insert_arbitary_basis_measure_op(
                            prev_mres=m2,
                            angle=op.params[0],
                            plane="XY",
                            qubit=aux_qubits_dict[3],
                            op=op,
                            rewriter=rewriter,
                        )
                        m4, aux_qubits_dict[4] = self._insert_arbitary_basis_measure_op(
                            angle=0.0,
                            plane="XY",
                            qubit=aux_qubits_dict[4],
                            op=op,
                            rewriter=rewriter,
                        )
                        x_mres = [m2, m4]
                        z_mres = [m1, m3]

                    if op.gate_name.data in ["RotXZX"]:
                        m1, aux_qubits_dict[1] = self._insert_arbitary_basis_measure_op(
                            angle=0.0,
                            plane="XY",
                            qubit=aux_qubits_dict[1],
                            op=op,
                            rewriter=rewriter,
                        )
                        m2, aux_qubits_dict[2] = self._cond_insert_arbitary_basis_measure_op(
                            prev_mres=m1,
                            angle=op.params[0],
                            plane="XY",
                            qubit=aux_qubits_dict[2],
                            op=op,
                            rewriter=rewriter,
                        )
                        m3, aux_qubits_dict[3] = self._cond_insert_arbitary_basis_measure_op(
                            prev_mres=m2,
                            angle=op.params[1],
                            plane="XY",
                            qubit=aux_qubits_dict[3],
                            op=op,
                            rewriter=rewriter,
                        )
                        m4, aux_qubits_dict[4] = self._cond_insert_arbitary_basis_measure_op(
                            prev_mres=m3,
                            angle=op.params[2],
                            plane="XY",
                            qubit=aux_qubits_dict[4],
                            op=op,
                            rewriter=rewriter,
                        )
                        x_mres = [m2, m4]
                        z_mres = [m1, m3]
                    # Apply corrections
                    x_exp = self._insert_byprod_exp_op(x_mres, op, rewriter, xmres_add_one)
                    z_exp = self._insert_byprod_exp_op(z_mres, op, rewriter, zmres_add_one)

                    res_aux_qubit = self._insert_cond_byproduct_op(
                        x_exp, "PauliX", res_aux_qubit, op, rewriter
                    )

                    res_aux_qubit = self._insert_cond_byproduct_op(
                        z_exp, "PauliZ", res_aux_qubit, op, rewriter
                    )

                    # NOTE: IdentityOp inserted here is a temporal solution to fix the issue that SWAPOp can't accept
                    # `res_aux_qubit` as an in_qubits variable, i.e. SWAPOp = CustomOp(in_qubits=(target_qubit, res_aux_qubit), gate_name="SWAP").
                    # The error message would be "| Error while applying pattern: 'NoneType' object has no attribute 'add_use'". Is it a upstream
                    # issue or caused by the way how we define the `CustomOp` operation? Not sure why.
                    IdentityOp = CustomOp(in_qubits=(res_aux_qubit), gate_name="Identity")
                    rewriter.insert_op(IdentityOp, InsertPoint.before(op))
                    in_qubits = (aux_qubits_dict[1], IdentityOp.results[0])
                    SWAPOp = CustomOp(in_qubits=in_qubits, gate_name="SWAP")
                    rewriter.insert_op(SWAPOp, InsertPoint.before(op))
                    result_qubit, aux_qubits_dict[5] = SWAPOp.results

                    # Deallocate aux_qubits
                    for node in aux_qubits_dict:
                        if node != 1:  # 1st qubit is the target qubit in the global register
                            deallocQubitOp = DeallocQubitOp(aux_qubits_dict[node])
                            rewriter.insert_op(deallocQubitOp, InsertPoint.before(op))

                    # Replace all uses of output qubit of op with the result_qubit
                    rewriter.replace_all_uses_with(op.results[0], result_qubit)
                    # Remove op operation
                    rewriter.erase_op(op)
                elif isinstance(op, CustomOp) and op.gate_name.data == "CNOT":
                    aux_qubits_dict = self._make_graph_state(op, rewriter)
                    # Entangle the target wire to qubit 2 in the aux_qubits_dict
                    in_qubits_ctrl = [op.in_qubits[0], aux_qubits_dict[2]]
                    CZ_ctrl_Op = CustomOp(in_qubits=in_qubits_ctrl, gate_name="CZ")
                    rewriter.insert_op(CZ_ctrl_Op, InsertPoint.before(op))
                    aux_qubits_dict[1], aux_qubits_dict[2] = CZ_ctrl_Op.results

                    in_qubits_target = [op.in_qubits[1], aux_qubits_dict[10]]
                    CZ_tgt_Op = CustomOp(in_qubits=in_qubits_target, gate_name="CZ")
                    rewriter.insert_op(CZ_tgt_Op, InsertPoint.before(op))
                    aux_qubits_dict[9], aux_qubits_dict[10] = CZ_tgt_Op.results

                    ctrl_qubit = aux_qubits_dict[1]
                    tgt_qubit = aux_qubits_dict[9]

                    (
                        xc_mres_idx,
                        xt_mres_idx,
                        zc_mres_idx,
                        zt_mres_idx,
                        xcmres_add_one,
                        xtmres_add_one,
                        zcmres_add_one,
                        ztmres_add_one,
                        angles,
                        planes,
                    ) = self._get_measurement_param_with_gatename(op.gate_name.data)
                    xt_mres = []
                    zt_mres = []
                    xc_mres = []
                    zc_mres = []
                    for key in angles:
                        mres, aux_qubits_dict[key] = self._insert_arbitary_basis_measure_op(
                            angle=angles[key],
                            plane=planes[key],
                            qubit=aux_qubits_dict[key],
                            op=op,
                            rewriter=rewriter,
                        )
                        if key in xc_mres_idx:
                            xc_mres.append(mres)
                        if key in zc_mres_idx:
                            zc_mres.append(mres)
                        if key in xt_mres_idx:
                            xt_mres.append(mres)
                        if key in zt_mres_idx:
                            zt_mres.append(mres)

                    # Apply corrections
                    xc_exp = self._insert_byprod_exp_op(xc_mres, op, rewriter, xcmres_add_one)
                    zc_exp = self._insert_byprod_exp_op(zc_mres, op, rewriter, zcmres_add_one)

                    xt_exp = self._insert_byprod_exp_op(xt_mres, op, rewriter, xtmres_add_one)
                    zt_exp = self._insert_byprod_exp_op(zt_mres, op, rewriter, ztmres_add_one)

                    res_c_qubit = self._insert_cond_byproduct_op(
                        xc_exp, "PauliX", ctrl_qubit, op, rewriter
                    )

                    res_c_qubit = self._insert_cond_byproduct_op(
                        zc_exp, "PauliZ", res_c_qubit, op, rewriter
                    )

                    res_t_qubit = self._insert_cond_byproduct_op(
                        xt_exp, "PauliX", tgt_qubit, op, rewriter
                    )

                    res_t_qubit = self._insert_cond_byproduct_op(
                        zt_exp, "PauliZ", res_t_qubit, op, rewriter
                    )

                    # NOTE: IdentityOp inserted here is a temporal solution to fix the issue that SWAPOp can't accept
                    # `res_aux_qubit` as an in_qubits variable, i.e. SWAPOp = CustomOp(in_qubits=(target_qubit, res_aux_qubit), gate_name="SWAP").
                    # The error message would be "| Error while applying pattern: 'NoneType' object has no attribute 'add_use'". Is it a upstream
                    # issue or caused by the way how we define the `CustomOp` operation? Not sure why.
                    IdentityOp = CustomOp(in_qubits=(res_c_qubit), gate_name="Identity")
                    rewriter.insert_op(IdentityOp, InsertPoint.before(op))

                    in_qubits = (aux_qubits_dict[1], IdentityOp.results[0])
                    SWAPOp = CustomOp(in_qubits=in_qubits, gate_name="SWAP")
                    rewriter.insert_op(SWAPOp, InsertPoint.before(op))
                    result_c_qubit, aux_qubits_dict[7] = SWAPOp.results

                    IdentityOp = CustomOp(in_qubits=(res_t_qubit), gate_name="Identity")
                    rewriter.insert_op(IdentityOp, InsertPoint.before(op))

                    in_qubits = (aux_qubits_dict[9], IdentityOp.results[0])
                    SWAPOp = CustomOp(in_qubits=in_qubits, gate_name="SWAP")
                    rewriter.insert_op(SWAPOp, InsertPoint.before(op))
                    result_t_qubit, aux_qubits_dict[15] = SWAPOp.results

                    # Deallocate aux_qubits
                    for node in aux_qubits_dict:
                        if (
                            node != 1 and node != 9
                        ):  # 1st qubit is the target qubit in the global register
                            deallocQubitOp = DeallocQubitOp(aux_qubits_dict[node])
                            rewriter.insert_op(deallocQubitOp, InsertPoint.before(op))

                    # Replace all uses of output qubit of op with the result_qubit
                    rewriter.replace_all_uses_with(op.results[0], result_c_qubit)
                    rewriter.replace_all_uses_with(op.results[1], result_t_qubit)
                    # Remove op operation
                    rewriter.erase_op(op)
