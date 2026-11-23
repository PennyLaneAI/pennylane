# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Built-in decomposition rules for op_math.controlled_ops."""

# pylint: disable=unused-argument


from collections import defaultdict
from functools import partial

import numpy as np
from scipy import sparse

import pennylane as qp
from pennylane import math
from pennylane.allocation import allocate
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition.decomposition_rule import (
    DecompCollection,
    _list_decomps,
    get_fixed_decomp,
)
from pennylane.decomposition.resources import resolve_work_wire_type
from pennylane.decomposition_rules.op_math.adjoint2 import (
    _adjoint_negate_angle as adjoint_rotation2,
)
from pennylane.decomposition_rules.op_math.decompositions.controlled_decompositions import (
    _controlled_unitary_bisect,
    _controlled_unitary_multi_control_zyz,
    _controlled_unitary_single_control_zyz,
    _controlled_unitary_two_qubit,
    _mcx_many_work_wires,
    _mcx_no_work_wires,
    _mcx_one_work_wire,
    _mcx_to_cnot_or_toffoli,
    _mcx_two_controls_elbows,
    _mcx_two_work_wires,
    _wrap_mcx_rule_w_alloc,
)
from pennylane.decomposition_rules.op_math.pow2 import _pow_scale_angle as pow_rotation2
from pennylane.decomposition_rules.op_math.pow2 import pow_involutory as pow_involutory2
from pennylane.decomposition_rules.op_math.symbolicop2 import self_adjoint
from pennylane.ops.identity import GlobalPhase
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure, pauli_measure
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.op_math.controlled import _resolve_ctrl_values
from pennylane.ops.op_math.controlled2 import ControlledOp2
from pennylane.ops.op_math.controlled_ops import (
    CCZ,
    CH,
    CNOT,
    CRX,
    CRY,
    CRZ,
    CSWAP,
    CY,
    CZ,
    ControlledPhaseShift,
    ControlledQubitUnitary,
    CRot,
    MultiControlledX,
    Toffoli,
)
from pennylane.ops.qubit import X, Y, Z
from pennylane.typing import AbstractWires, Bool, Complex, Float, TensorLike, Wire
from pennylane.wires import WiresLike, concatenate_wires


def _to_general_c_qu_standard_resources(U, wires, control_values, work_wires, work_wire_type, **_):
    num_target_wires = int(math.log2(qp.math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    num_work_wires = len(work_wires)
    u_size = 2**num_target_wires
    return {
        ControlledOp2(
            qp.QubitUnitary(Complex[u_size, u_size], wires=Wire[num_target_wires]),
            control_wires=Wire[num_control_wires],
            control_values=Bool[num_control_wires],
            work_wires=Wire[num_work_wires],
            work_wire_type=work_wire_type,
        ): 1
    }


@qp.register_condition(lambda U, *_, **__: math.ceil_log2(qp.math.shape(U)[0]) > 2)
@qp.register_resources(_to_general_c_qu_standard_resources, name="to_controlled_qubit_unitary")
def _to_general_c_qu_standard(U, wires, control_values, work_wires, work_wire_type, **_):
    """Convert a ControlledQubitUnitary to a general Controlled(QubitUnitary) so that
    the graph finds the general decomposition rule of applying control to the decomposition
    of the base QubitUnitary."""
    num_target_wires = len(wires) - len(control_values)
    ControlledOp2(
        qp.QubitUnitary(U, wires=wires[-num_target_wires:]),
        control_wires=wires[:-num_target_wires],
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps(
    ControlledQubitUnitary,
    _controlled_unitary_bisect,
    _controlled_unitary_single_control_zyz,
    _controlled_unitary_multi_control_zyz,
    _controlled_unitary_two_qubit,
    _to_general_c_qu_standard,
)


def _conj_transpose(U):
    return U.conjugate().transpose() if sparse.issparse(U) else math.moveaxis(math.conj(U), -2, -1)


@register_resources(lambda base: {abstractify(base): 1}, name="conjugate_transpose")
def _adjoint_controlled_qubit_unitary_standard(base):
    ControlledQubitUnitary(
        _conj_transpose(base.U),
        wires=base.wires,
        control_values=base.control_values,
        work_wires=base.work_wires,
        work_wire_type=base.work_wire_type,
    )


add_decomps("Adjoint(ControlledQubitUnitary)", _adjoint_controlled_qubit_unitary_standard)


def _ch_to_ry_cz_ry_resources(wires: WiresLike = None):
    return {qp.RY: 2, qp.CZ: 1}


@register_resources(_ch_to_ry_cz_ry_resources, name="to_ry_cz_ry")
def _ch_to_ry_cz_ry(wires: WiresLike, **__):
    qp.RY(-np.pi / 4, wires=wires[1])
    qp.CZ(wires=wires)
    qp.RY(+np.pi / 4, wires=wires[1])


add_decomps(CH, _ch_to_ry_cz_ry)

add_decomps("Adjoint(CH)", self_adjoint)

add_decomps("Pow(CH)", pow_involutory2)


def _cy_to_ry_cnot_resources(wires: AbstractWires):
    return {qp.CRY: 1, qp.S: 1}


@register_resources(_cy_to_ry_cnot_resources, name="to_ry_cnot")
def _cy_to_ry_cnot(wires: WiresLike, **__):
    qp.CRY(np.pi, wires=wires)
    qp.S(wires=wires[0])


def _pauli_ctrl_pauli_ppr_resources(wires: AbstractWires, pauli0, pauli1):
    """Resources for _pauli_ctrl_pauli_ppr."""
    resources = defaultdict(int)
    p0, p1 = pauli0.__name__[-1], pauli1.__name__[-1]
    resources[qp.PPR_4(-1, pauli_word=p0, wires=Wire[len(p0)])] += 1
    resources[qp.PPR_4(-1, pauli_word=p1, wires=Wire[len(p1)])] += 1
    resources[qp.PPR_4(1, pauli_word=p0 + p1, wires=Wire[len(p0 + p1)])] += 1
    resources[qp.GlobalPhase] += 1
    return dict(resources)


def _pauli_ctrl_pauli_ppr(wires: AbstractWires, pauli0, pauli1):
    """Generalized two-qubit Pauli-controlled Pauli gate decomposition to PPRs."""
    p0, p1 = pauli0.__name__[-1], pauli1.__name__[-1]
    qp.PPR_4(-1, p0, wires=wires[0])
    qp.PPR_4(-1, p1, wires=wires[1])
    qp.PPR_4(1, p0 + p1, wires=wires)
    qp.GlobalPhase(np.pi / 4)


@register_resources(partial(_pauli_ctrl_pauli_ppr_resources, pauli0=Z, pauli1=Y), name="to_ppr")
def _cy_to_ppr(wires: AbstractWires):
    _pauli_ctrl_pauli_ppr(wires, qp.Z, qp.Y)


def _pauli_ctrl_pauli_ppm(wires, pauli0, pauli1):
    """Generalized two-qubit Pauli-controlled Pauli gate decomposition to PPMs."""
    with qp.allocate(1, state="zero", restored=False) as work_wires:
        m0 = pauli_measure(pauli0.__name__[-1] + "X", [wires[0], work_wires[0]])
        m1 = pauli_measure("Z" + pauli1.__name__[-1], [work_wires[0], wires[1]])
        m2 = pauli_measure("X", work_wires)
        qp.cond(m1, pauli0)(wires[0])
        qp.cond(m0 != m2, pauli1)(wires[1])
        qp.cond(m1 & (m0 != m2), qp.GlobalPhase)(np.pi)
        qp.cond(m2, qp.Z)(work_wires[0])  # Reset work wire (to |+>), achieving pure state


@qp.register_resources(
    {
        PauliMeasure("ZX", wires=Wire[2]): 1,
        PauliMeasure("ZY", wires=Wire[2]): 1,
        PauliMeasure("X", wires=Wire[1]): 1,
        Z: 2,
        Y: 1,
        GlobalPhase: 1,
    },
    work_wires={"burnable": 1},
    name="lattice_surgery_ppm",
)
def _cy_lattice_surgery_ppm(wires: AbstractWires):
    _pauli_ctrl_pauli_ppm(wires, qp.Z, qp.Y)


add_decomps(CY, _cy_to_ry_cnot, _cy_to_ppr, _cy_lattice_surgery_ppm)

add_decomps("Adjoint(CY)", self_adjoint)

add_decomps("Pow(CY)", pow_involutory2)


def _cz_to_cps_resources(wires: AbstractWires):
    return {qp.ControlledPhaseShift: 1}


@register_resources(_cz_to_cps_resources, name="to_cps")
def _cz_to_cps(wires: WiresLike):
    qp.ControlledPhaseShift(np.pi, wires=wires)


def _cz_to_cnot_resources(wires: AbstractWires):
    return {qp.H: 2, qp.CNOT: 1}


@register_resources(_cz_to_cnot_resources, name="to_cnot")
def _cz_to_cnot(wires: WiresLike):
    qp.H(wires=wires[1])
    qp.CNOT(wires=wires)
    qp.H(wires=wires[1])


@register_resources(partial(_pauli_ctrl_pauli_ppr_resources, pauli0=Z, pauli1=Z), name="to_ppr")
def _cz_to_ppr(wires: AbstractWires):
    _pauli_ctrl_pauli_ppr(wires, qp.Z, qp.Z)


def _cz_lattice_surgery_ppm_resources(wires: AbstractWires):
    return {
        PauliMeasure("ZX", wires=Wire[2]): 1,
        PauliMeasure("ZZ", wires=Wire[2]): 1,
        PauliMeasure("X", wires=Wire[1]): 1,
        qp.Z: 3,
        qp.GlobalPhase: 1,
    }


@qp.register_resources(
    _cz_lattice_surgery_ppm_resources, work_wires={"burnable": 1}, name="lattice_surgery_ppm"
)
def _cz_lattice_surgery_ppm(wires: WiresLike):
    _pauli_ctrl_pauli_ppm(wires, qp.Z, qp.Z)


add_decomps(CZ, _cz_to_cps, _cz_to_cnot, _cz_to_ppr, _cz_lattice_surgery_ppm)

add_decomps("Adjoint(CZ)", self_adjoint)

add_decomps("Pow(CZ)", pow_involutory2)


def _cswap_to_cnot_toffoli_resources(wires: WiresLike = None):
    return {qp.CNOT: 2, qp.Toffoli: 1}


@register_resources(_cswap_to_cnot_toffoli_resources, name="to_cnot_toffoli")
def _cswap_to_cnot_toffoli(wires: WiresLike, **__):
    qp.CNOT([wires[2], wires[1]])
    qp.Toffoli(wires=[wires[0], wires[1], wires[2]])
    qp.CNOT([wires[2], wires[1]])


def _cswap_to_ppr_resources(wires: WiresLike = None):
    return {
        qp.PPR_8(-1, pauli_word="ZZZ", wires=Wire[3]): 1,
        qp.PPR_8(-1, pauli_word="ZYY", wires=Wire[3]): 1,
        qp.PPR_8(-1, pauli_word="ZXX", wires=Wire[3]): 1,
        qp.PPR_8(1, pauli_word="ZZ", wires=Wire[2]): 1,
        qp.PPR_8(1, pauli_word="YY", wires=Wire[2]): 1,
        qp.PPR_8(1, pauli_word="XX", wires=Wire[2]): 1,
        qp.PPR_8(1, pauli_word="Z", wires=Wire[1]): 1,
        qp.GlobalPhase: 1,
    }


@register_resources(_cswap_to_ppr_resources, name="to_ppr")
def _cswap_to_ppr(wires: WiresLike, **_):
    qp.PPR_8(-1, "ZZZ", wires=wires)
    qp.PPR_8(-1, "ZYY", wires=wires)
    qp.PPR_8(-1, "ZXX", wires=wires)
    qp.PPR_8(1, "ZZ", wires=wires[1:])
    qp.PPR_8(1, "YY", wires=wires[1:])
    qp.PPR_8(1, "XX", wires=wires[1:])
    qp.PPR_8(1, "Z", wires=wires[0])
    qp.GlobalPhase(-np.pi / 8)


add_decomps(CSWAP, _cswap_to_cnot_toffoli, _cswap_to_ppr)

add_decomps("Adjoint(CSWAP)", self_adjoint)

add_decomps("Pow(CSWAP)", pow_involutory2)


def _ccz_to_clifford_t_resources(wires: AbstractWires):
    return {
        qp.CNOT: 6,
        _adjoint_abstract(qp.T): 3,
        qp.T: 4,
        qp.Hadamard: 2,
    }


@register_resources(_ccz_to_clifford_t_resources, name="to_clifford_t")
def _ccz_to_clifford_t(wires: WiresLike, **__):
    qp.CNOT(wires=[wires[1], wires[2]])
    qp.adjoint(qp.T(wires=wires[2]))
    qp.CNOT(wires=[wires[0], wires[2]])
    qp.T(wires=wires[2])
    qp.CNOT(wires=[wires[1], wires[2]])
    qp.adjoint(qp.T(wires=wires[2]))
    qp.CNOT(wires=[wires[0], wires[2]])
    qp.T(wires=wires[2])
    qp.T(wires=wires[1])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.Hadamard(wires=wires[2])
    qp.T(wires=wires[0])
    qp.adjoint(qp.T(wires=wires[1]))
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.Hadamard(wires=wires[2])


def _ccz_to_toffoli_resources(wires: AbstractWires):
    return {qp.Hadamard: 2, qp.Toffoli: 1}


@register_resources(_ccz_to_toffoli_resources, name="to_toffoli")
def _ccz_to_toffoli(wires: WiresLike, **__):
    qp.Hadamard(wires[2])
    qp.Toffoli(wires)
    qp.Hadamard(wires[2])


add_decomps(CCZ, _ccz_to_clifford_t, _ccz_to_toffoli)

add_decomps("Adjoint(CCZ)", self_adjoint)

add_decomps("Pow(CCZ)", pow_involutory2)


def _cnot_to_cz_h_resources(wires: WiresLike):
    return {qp.H: 2, qp.CZ: 1}


@register_resources(_cnot_to_cz_h_resources, name="to_cz_h")
def _cnot_to_cz_h(wires: WiresLike):
    qp.H(wires[1])
    qp.CZ(wires=wires)
    qp.H(wires[1])


@register_resources(partial(_pauli_ctrl_pauli_ppr_resources, pauli0=Z, pauli1=X), name="to_ppr")
def _cnot_to_ppr(wires: AbstractWires):
    _pauli_ctrl_pauli_ppr(wires, qp.Z, qp.X)


def _cnot_lattice_surgery_ppm_resources(wires: WiresLike):
    return {
        PauliMeasure("ZX", wires=Wire[2]): 2,
        PauliMeasure("X", wires=Wire[1]): 1,
        qp.Z: 2,
        qp.X: 1,
        qp.GlobalPhase: 1,
    }


@qp.register_resources(
    _cnot_lattice_surgery_ppm_resources, work_wires={"burnable": 1}, name="lattice_surgery_ppm"
)
def _cnot_lattice_surgery_ppm(wires: WiresLike):
    _pauli_ctrl_pauli_ppm(wires, qp.Z, qp.X)


add_decomps(CNOT, _cnot_to_cz_h, _cnot_to_ppr, _cnot_lattice_surgery_ppm)

add_decomps("Adjoint(CNOT)", self_adjoint)

add_decomps("Pow(CNOT)", pow_involutory2)


def _ctrl_cnot_to_mcx_resources(base, control_wires, control_values, work_wires, work_wire_type):
    ctrl_values = _resolve_ctrl_values(control_values, [True], len(control_wires))
    return {
        qp.MultiControlledX(
            Wire[len(control_wires) + 2],
            control_values=ctrl_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@qp.register_resources(_ctrl_cnot_to_mcx_resources, name="to_mcx")
def _ctrl_cnot_to_mcx(base, control_wires, control_values, work_wires, work_wire_type):
    ctrl_values = _resolve_ctrl_values(control_values, [True], len(control_wires))
    qp.MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        control_values=ctrl_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(CNOT)", _ctrl_cnot_to_mcx)


def _toffoli_to_clifford_t_resources(wires: WiresLike):
    return {
        qp.Hadamard: 2,
        qp.CNOT: 6,
        qp.T: 4,
        _adjoint_abstract(qp.T): 3,
    }


@register_resources(_toffoli_to_clifford_t_resources, name="to_clifford_t")
def _toffoli_to_clifford_t(wires: WiresLike):
    qp.Hadamard(wires=wires[2])
    CNOT(wires=[wires[1], wires[2]])
    qp.adjoint(qp.T(wires=wires[2]))
    CNOT(wires=[wires[0], wires[2]])
    qp.T(wires=wires[2])
    CNOT(wires=[wires[1], wires[2]])
    qp.adjoint(qp.T(wires=wires[2]))
    CNOT(wires=[wires[0], wires[2]])
    qp.T(wires=wires[2])
    qp.T(wires=wires[1])
    CNOT(wires=[wires[0], wires[1]])
    qp.Hadamard(wires=wires[2])
    qp.T(wires=wires[0])
    qp.adjoint(qp.T(wires=wires[1]))
    CNOT(wires=[wires[0], wires[1]])


def _toffoli_to_ppr_resources(wires: WiresLike):
    return {
        qp.PPR_8(-1, pauli_word="ZZ", wires=Wire[2]): 1,
        qp.PPR_8(-1, pauli_word="ZX", wires=Wire[2]): 2,
        qp.PPR_8(1, pauli_word="ZZX", wires=Wire[3]): 1,
        qp.PPR_8(1, pauli_word="X", wires=Wire[1]): 1,
        qp.PPR_8(1, pauli_word="Z", wires=Wire[1]): 2,
        qp.GlobalPhase: 1,
    }


@register_resources(_toffoli_to_ppr_resources, name="to_ppr")
def _toffoli_to_ppr(wires: WiresLike):
    qp.PPR_8(-1, "ZZ", wires=wires[:2])
    qp.PPR_8(-1, "ZX", wires=[wires[0], wires[2]])
    qp.PPR_8(-1, "ZX", wires=wires[1:])
    qp.PPR_8(1, "ZZX", wires=wires)
    qp.PPR_8(1, "X", wires=wires[2])
    qp.PPR_8(1, "Z", wires=wires[1])
    qp.PPR_8(1, "Z", wires=wires[0])
    qp.GlobalPhase(-np.pi / 8)


def _toffoli_elbow_resources(*_, **__):
    return {_change_op_basis_abstract(qp.Elbow, qp.CNOT, qp.adjoint(qp.Elbow(Wire[3]))): 1}


@register_resources(_toffoli_elbow_resources, work_wires={"zeroed": 1}, name="elbow")
def _toffoli_elbow(wires: WiresLike):
    with allocate(1, qp.allocation.AllocateState.ZERO, restored=True) as work_wires:
        qp.change_op_basis(
            qp.Elbow([wires[0], wires[1], work_wires[0]]),
            qp.CNOT([work_wires[0], wires[2]]),
        )


add_decomps(Toffoli, _toffoli_to_clifford_t, _toffoli_to_ppr, _toffoli_elbow)

add_decomps("Adjoint(Toffoli)", self_adjoint)

add_decomps("Pow(Toffoli)", pow_involutory2)


def _ctrl_toffoli_to_mcx_resources(base, control_wires, control_values, work_wires, work_wire_type):
    ctrl_values = _resolve_ctrl_values(control_values, [True, True], len(control_wires))
    return {
        qp.MultiControlledX(
            Wire[len(control_wires) + 3],
            control_values=ctrl_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@qp.register_resources(_ctrl_toffoli_to_mcx_resources, name="to_mcx")
def _ctrl_toffoli_to_mcx(base, control_wires, control_values, work_wires, work_wire_type):
    ctrl_values = _resolve_ctrl_values(control_values, [True, True], len(control_wires))
    qp.MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        control_values=ctrl_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(Toffoli)", _ctrl_toffoli_to_mcx)


@_list_decomps.register
def _list_mcx_decomps(op: MultiControlledX):
    # fixed_decomps should override everything
    if fixed_rule := get_fixed_decomp(op):
        return DecompCollection([fixed_rule])
    if not op.work_wires:
        return DecompCollection(_list_mcx_no_work_wire_decomps(op))
    if len(op.wires) == 2:
        return [_mcx_to_cnot_or_toffoli]
    if len(op.wires) == 3:
        return [_mcx_to_cnot_or_toffoli, _mcx_two_controls_elbows]
    return [
        _mcx_many_work_wires,
        _mcx_two_work_wires,
        _mcx_one_work_wire,
        _mcx_no_work_wires,
    ]


def _list_mcx_no_work_wire_decomps(op: MultiControlledX):

    if len(op.wires) == 2:
        return [_mcx_to_cnot_or_toffoli]

    if len(op.wires) == 3:
        elbow_rule = _wrap_mcx_rule_w_alloc(
            _mcx_two_controls_elbows, 1, "zeroed", name="two_controls_elbows_allocated"
        )
        return [_mcx_to_cnot_or_toffoli, elbow_rule]

    capture_compatible_rules = [
        _wrap_mcx_rule_w_alloc(_mcx_two_work_wires, 2, "zeroed", "two_zeroed_work_wires"),
        _wrap_mcx_rule_w_alloc(_mcx_two_work_wires, 2, "borrowed", "two_borrowed_work_wires"),
        _wrap_mcx_rule_w_alloc(_mcx_one_work_wire, 1, "zeroed", "one_zeroed_work_wire"),
        _wrap_mcx_rule_w_alloc(_mcx_one_work_wire, 1, "borrowed", "one_borrowed_work_wire"),
        _mcx_no_work_wires,
    ]
    if qp.capture.enabled():
        return capture_compatible_rules

    # TODO: the following decomposition rules are not capture compatible [sc-129521]
    return [
        _wrap_mcx_rule_w_alloc(
            _mcx_many_work_wires,
            len(op.control_wires) - 2,
            "zeroed",
            "many_zeroed_work_wires",
        ),
        _wrap_mcx_rule_w_alloc(
            _mcx_many_work_wires,
            len(op.control_wires) - 2,
            "borrowed",
            "many_borrowed_work_wires",
        ),
    ] + capture_compatible_rules


add_decomps(
    "MultiControlledX",
    _mcx_to_cnot_or_toffoli,
    _mcx_two_controls_elbows,
    _mcx_many_work_wires,
    _mcx_two_work_wires,
    _mcx_one_work_wire,
    _mcx_no_work_wires,
    _wrap_mcx_rule_w_alloc(
        _mcx_two_controls_elbows, 1, "zeroed", name="two_controls_elbows_allocated"
    ),
    _wrap_mcx_rule_w_alloc(_mcx_two_work_wires, 2, "zeroed", "two_zeroed_work_wires"),
    _wrap_mcx_rule_w_alloc(_mcx_two_work_wires, 2, "borrowed", "two_borrowed_work_wires"),
    _wrap_mcx_rule_w_alloc(_mcx_one_work_wire, 1, "zeroed", "one_zeroed_work_wire"),
    _wrap_mcx_rule_w_alloc(_mcx_one_work_wire, 1, "borrowed", "one_borrowed_work_wire"),
    # TODO: include the allocation-wrapped versions of _mcx_many_work_wires
    # after we support tracer indexing into allocated wires. [sc-129521]
)

add_decomps("Adjoint(MultiControlledX)", self_adjoint)

add_decomps("Pow(MultiControlledX)", pow_involutory2)


def _ctrl_mcx_to_mcx_resources(base, control_wires, control_values, work_wires, work_wire_type):
    work_wire_type = resolve_work_wire_type(
        base.work_wires,
        base.work_wire_type,
        work_wires,
        work_wire_type,
    )
    return {
        qp.MultiControlledX(
            control_wires + base.wires,
            control_values=Bool[len(control_values) + len(base.control_values)],
            work_wires=work_wires + base.work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@qp.register_resources(_ctrl_mcx_to_mcx_resources, name="to_mcx")
def _ctrl_mcx_to_mcx(base, control_wires, control_values, work_wires, work_wire_type):
    n_ctrl_wires = len(control_wires)
    work_wire_type = resolve_work_wire_type(
        base.work_wires,
        base.work_wire_type,
        work_wires,
        work_wire_type,
    )
    qp.MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        control_values=_resolve_ctrl_values(control_values, base.control_values, n_ctrl_wires),
        work_wires=concatenate_wires(work_wires, base.work_wires),
        work_wire_type=work_wire_type,
    )


add_decomps("C(MultiControlledX)", _ctrl_mcx_to_mcx)


def _crx_to_rz_ry_resources(phi, wires):
    return {qp.RZ: 2, qp.RY: 2, qp.CNOT: 2}


@register_resources(_crx_to_rz_ry_resources, name="to_rz_ry")
def _crx_to_rz_ry(phi: TensorLike, wires: WiresLike):
    qp.RZ(np.pi / 2, wires=wires[1])
    qp.RY(phi / 2, wires=wires[1])
    qp.CNOT(wires=wires)
    qp.RY(-phi / 2, wires=wires[1])
    qp.CNOT(wires=wires)
    qp.RZ(-np.pi / 2, wires=wires[1])


def _crx_to_rx_cz_resources(phi, wires):
    return {qp.RX: 2, qp.CZ: 2}


@register_resources(_crx_to_rx_cz_resources, name="to_rx_cz")
def _crx_to_rx_cz(phi: TensorLike, wires: WiresLike):
    qp.RX(phi / 2, wires=wires[1])
    qp.CZ(wires=wires)
    qp.RX(-phi / 2, wires=wires[1])
    qp.CZ(wires=wires)


def _crx_to_h_crz_resources(phi, wires):
    return {qp.Hadamard: 2, qp.CRZ: 1}


@register_resources(_crx_to_h_crz_resources, name="to_h_crz")
def _crx_to_h_crz(phi: TensorLike, wires: WiresLike):
    qp.Hadamard(wires=wires[1])
    qp.CRZ(phi, wires=wires)
    qp.Hadamard(wires=wires[1])


def _crx_to_ppr_resources(phi, wires):
    return {
        qp.PauliRot(Float, pauli_word="ZX", wires=Wire[2]): 1,
        qp.PauliRot(Float, pauli_word="X", wires=Wire[1]): 1,
    }


@register_resources(_crx_to_ppr_resources, name="to_paulirot")
def _crx_to_ppr(phi: TensorLike, wires: WiresLike):
    qp.PauliRot(phi / 2, "X", wires=wires[1])
    qp.PauliRot(-phi / 2, "ZX", wires=wires)


add_decomps(CRX, _crx_to_rx_cz, _crx_to_rz_ry, _crx_to_h_crz, _crx_to_ppr)

add_decomps("Adjoint(CRX)", adjoint_rotation2)

add_decomps("Pow(CRX)", pow_rotation2)


def _cry_to_ry_cnot_resources(phi, wires):
    return {qp.RY: 2, qp.CNOT: 2}


@register_resources(_cry_to_ry_cnot_resources, name="to_ry_cnot")
def _cry_to_ry_cnot(phi: TensorLike, wires: WiresLike):
    qp.RY(phi / 2, wires=wires[1])
    qp.CNOT(wires=wires)
    qp.RY(-phi / 2, wires=wires[1])
    qp.CNOT(wires=wires)


def _cry_to_ppr_resources(phi, wires):
    return {
        qp.PauliRot(Float, pauli_word="ZY", wires=Wire[2]): 1,
        qp.PauliRot(Float, pauli_word="Y", wires=Wire[1]): 1,
    }


@register_resources(_cry_to_ppr_resources, name="to_paulirot")
def _cry_to_ppr(phi: TensorLike, wires: WiresLike):
    qp.PauliRot(phi / 2, "Y", wires=wires[1])
    qp.PauliRot(-phi / 2, "ZY", wires=wires)


add_decomps(CRY, _cry_to_ry_cnot, _cry_to_ppr)

add_decomps("Adjoint(CRY)", adjoint_rotation2)

add_decomps("Pow(CRY)", pow_rotation2)


def _crz_to_rz_cnot_resources(phi, wires):
    return {qp.RZ: 2, qp.CNOT: 2}


@register_resources(_crz_to_rz_cnot_resources, name="to_rz_cnot")
def _crz_to_rz_cnot(phi: TensorLike, wires: WiresLike):
    qp.RZ(phi / 2, wires=wires[1])
    qp.CNOT(wires=wires)
    qp.RZ(-phi / 2, wires=wires[1])
    qp.CNOT(wires=wires)


def _crz_to_ppr_resources(phi, wires):
    return {
        qp.PauliRot(Float, pauli_word="ZZ", wires=Wire[2]): 1,
        qp.PauliRot(Float, pauli_word="Z", wires=Wire[1]): 1,
    }


@register_resources(_crz_to_ppr_resources, name="to_paulirot")
def _crz_to_ppr(phi: TensorLike, wires: WiresLike):
    qp.PauliRot(phi / 2, "Z", wires=wires[1])
    qp.PauliRot(-phi / 2, "ZZ", wires=wires)


add_decomps(CRZ, _crz_to_rz_cnot, _crz_to_ppr)

add_decomps("Adjoint(CRZ)", adjoint_rotation2)

add_decomps("Pow(CRZ)", pow_rotation2)


def _crot_to_rot_cnot_resources(*_, **__):
    return {qp.RZ: 3, qp.CNOT: 2, qp.RY: 2}


@register_resources(_crot_to_rot_cnot_resources, name="to_rot_cnot")
def _crot_to_rot_cnot(phi: TensorLike, theta: TensorLike, omega: TensorLike, wires: WiresLike):
    qp.RZ((phi - omega) / 2, wires=wires[1])
    qp.CNOT(wires=wires)
    qp.RZ(-(phi + omega) / 2, wires=wires[1])
    qp.RY(-theta / 2, wires=wires[1])
    qp.CNOT(wires=wires)
    qp.RY(theta / 2, wires=wires[1])
    qp.RZ(omega, wires=wires[1])


add_decomps(CRot, _crot_to_rot_cnot)


@register_resources({CRot: 1}, name="reverse_angles")
def _adjoint_crot_standard(base):
    CRot(-base.omega, -base.theta, -base.phi, wires=base.wires)


add_decomps("Adjoint(CRot)", _adjoint_crot_standard)


def _cphase_to_rz_cnot_resources(phi, wires):
    return {qp.RZ: 3, qp.CNOT: 2, qp.GlobalPhase: 1}


@register_resources(_cphase_to_rz_cnot_resources, name="to_rz_cnot")
def _cphase_to_rz_cnot(phi: TensorLike, wires: WiresLike, **__):
    qp.RZ(phi / 2, wires=wires[0])
    qp.CNOT(wires=wires)
    qp.RZ(-phi / 2, wires=wires[1])
    qp.CNOT(wires=wires)
    qp.RZ(phi / 2, wires=wires[1])
    qp.GlobalPhase(-phi / 4)


def _cphase_to_ppr_resources(phi, wires):
    return {
        qp.GlobalPhase: 1,
        qp.PauliRot(Float, pauli_word="Z", wires=Wire[1]): 2,
        qp.PauliRot(Float, pauli_word="ZZ", wires=Wire[2]): 1,
    }


@register_resources(_cphase_to_ppr_resources, name="to_paulirot")
def _cphase_to_ppr(phi: TensorLike, wires: WiresLike, **__):
    qp.PauliRot(-phi / 2, pauli_word="ZZ", wires=wires)
    qp.PauliRot(phi / 2, pauli_word="Z", wires=wires[1])
    qp.PauliRot(phi / 2, pauli_word="Z", wires=wires[0])
    qp.GlobalPhase(-phi / 4)


def _cphase_to_single_phase_resources(phi, wires):
    elbow = qp.TemporaryAND(wires=Wire[3])
    return {_change_op_basis_abstract(elbow, qp.PhaseShift, qp.adjoint(elbow)): 1}


@register_resources(
    _cphase_to_single_phase_resources, work_wires={"zeroed": 1}, name="to_single_phase"
)
def _cphase_to_single_phase(phi: TensorLike, wires: WiresLike, **__):
    with qp.allocate(1, state="zero", restored=True) as work_wire:
        elbow = qp.TemporaryAND([wires[0], wires[1], work_wire[0]])
        qp.change_op_basis(elbow, qp.PhaseShift(phi, work_wire))


add_decomps(ControlledPhaseShift, _cphase_to_rz_cnot, _cphase_to_ppr, _cphase_to_single_phase)

add_decomps("Adjoint(ControlledPhaseShift)", adjoint_rotation2)

add_decomps("Pow(ControlledPhaseShift)", pow_rotation2)
