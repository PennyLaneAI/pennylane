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


"""Built-in decomposition rules for qubit.non_parametric_ops."""

# pylint: disable=unused-argument


from collections import Counter
from functools import partial

import numpy as np

import pennylane as qp
from pennylane import math
from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
)
from pennylane.decomposition_rules.op_math.controlled2 import (
    flip_zero_control as flip_zero_control2,
)
from pennylane.decomposition_rules.op_math.pow2 import (
    make_pow_decomp_with_period as make_pow_decomp_with_period2,
)
from pennylane.decomposition_rules.op_math.pow2 import pow_involutory as pow_involutory2
from pennylane.decomposition_rules.op_math.symbolicop2 import self_adjoint
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure, pauli_measure
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.qubit.non_parametric_ops import (
    ECR,
    ISWAP,
    PPR_2,
    PPR_4,
    PPR_8,
    SISWAP,
    SWAP,
    SX,
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    S,
    T,
    Z,
)
from pennylane.ops.subcircuit import subcircuit
from pennylane.typing import AbstractWires, Wire
from pennylane.wires import WiresLike, concatenate_wires


def _hadamard_to_rz_rx_resources(wires: WiresLike):
    return {qp.RZ: 2, qp.RX: 1, qp.GlobalPhase: 1}


@register_resources(_hadamard_to_rz_rx_resources, name="to_rz_rx")
def _hadamard_to_rz_rx(wires: WiresLike, **__):
    qp.RZ(np.pi / 2, wires=wires)
    qp.RX(np.pi / 2, wires=wires)
    qp.RZ(np.pi / 2, wires=wires)
    qp.GlobalPhase(-np.pi / 2)


def _hadamard_to_rz_ry_resources(wires: WiresLike):
    return {qp.RZ: 1, qp.RY: 1, qp.GlobalPhase: 1}


@register_resources(_hadamard_to_rz_ry_resources, name="to_rz_ry")
def _hadamard_to_rz_ry(wires: WiresLike, **__):
    qp.RZ(np.pi, wires=wires)
    qp.RY(np.pi / 2, wires=wires)
    qp.GlobalPhase(-np.pi / 2)


def _hadamard_to_ppm_resources(wires: WiresLike):
    return {
        PauliMeasure("YY", wires=Wire[2]): 1,
        PauliMeasure("X", wires=Wire[1]): 1,
        qp.Y: 1,
        qp.Z: 2,
        qp.GlobalPhase: 1,
    }


@qp.register_resources(_hadamard_to_ppm_resources, work_wires={"burnable": 1}, name="to_ppm")
def _hadamard_to_ppm(wires, **__):
    with qp.allocate(1, state="zero", restored=False) as work_wires:
        qp.Z(wires)
        m0 = pauli_measure("YY", [wires[0], work_wires[0]])
        m1 = pauli_measure("X", work_wires)
        qp.cond(m0 == m1, qp.Y)(wires)
        qp.cond(m0 == m1, qp.GlobalPhase)(np.pi / 2)
        qp.cond(m1, qp.Z)(work_wires[0])  # Reset work wire to |+>


def _hadamard_to_ppr_resources(wires: WiresLike):
    return {qp.PPR_2(1, "Z", Wire[1]): 1, qp.PPR_4(1, "Y", Wire[1]): 1, qp.GlobalPhase: 1}


@register_resources(_hadamard_to_ppr_resources, name="to_ppr")
def _hadamard_to_ppr(wires: WiresLike, **__):
    qp.PPR_2(1, "Z", wires=wires)
    qp.PPR_4(1, "Y", wires=wires)
    qp.GlobalPhase(-np.pi / 2)


add_decomps(Hadamard, _hadamard_to_rz_rx, _hadamard_to_rz_ry, _hadamard_to_ppm, _hadamard_to_ppr)

add_decomps("Adjoint(Hadamard)", self_adjoint)

add_decomps("Pow(Hadamard)", pow_involutory2)


def _controlled_hadamard_standard_resources(
    base, control_wires, control_values, work_wires, work_wire_type
):
    if len(control_wires) == 1:
        return {qp.CH: 1}
    return {
        qp.H: 2,
        qp.RY: 2,
        qp.MultiControlledX(
            Wire[len(control_wires) + 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 1,
    }


@register_resources(_controlled_hadamard_standard_resources, name="to_mcx")
def _controlled_hadamard_standard(base, control_wires, control_values, work_wires, work_wire_type):
    wires = concatenate_wires(control_wires, base.wires)
    if len(control_wires) == 1:
        qp.CH(wires)
        return

    qp.RY(-np.pi / 4, wires=wires[-1])
    qp.H(wires=wires[-1])
    qp.MultiControlledX(
        wires,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    qp.H(wires=wires[-1])
    qp.RY(np.pi / 4, wires=wires[-1])


add_decomps("C(Hadamard)", flip_zero_control2(_controlled_hadamard_standard, name="to_mcx"))


def _paulix_to_rx_resources(wires: AbstractWires):
    return {qp.GlobalPhase: 1, qp.RX: 1}


@register_resources(_paulix_to_rx_resources, name="to_rx")
def _paulix_to_rx(wires: WiresLike):
    qp.RX(np.pi, wires=wires)
    qp.GlobalPhase(-np.pi / 2)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 2, 0.5))
@register_resources(lambda *_, **__: {qp.SX: 1}, name="to_sx")
def _pow_x_to_sx(base, z):
    qp.SX(wires=base.wires)


@register_resources(lambda *_, **__: {qp.RX: 1, qp.GlobalPhase: 1}, name="to_rx")
def _pow_x_to_rx(base, z):
    z_mod2 = qp.math.array(z) % 2
    qp.RX(np.pi * z_mod2, wires=base.wires)
    qp.GlobalPhase(-np.pi / 2 * z_mod2)


def _paulix_to_ppr_resources(wires: AbstractWires):
    return {qp.PPR_2(1, "X", Wire[1]): 1, qp.GlobalPhase: 1}


@register_resources(_paulix_to_ppr_resources, name="to_ppr")
def _paulix_to_ppr(wires: WiresLike):
    qp.PPR_2(1, "X", wires=wires)
    qp.GlobalPhase(-np.pi / 2)


add_decomps(PauliX, _paulix_to_rx, _paulix_to_ppr)

add_decomps("Adjoint(PauliX)", self_adjoint)

add_decomps("Pow(PauliX)", pow_involutory2, _pow_x_to_rx, _pow_x_to_sx)


def _ctrl_x_to_mcx_resources(base, control_wires, control_values, work_wires, work_wire_type):
    return {
        qp.MultiControlledX(
            Wire[len(control_wires) + 1],
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@qp.register_resources(_ctrl_x_to_mcx_resources, name="to_mcx")
def _ctrl_x_to_mcx(base, control_wires, control_values, work_wires, work_wire_type):
    qp.MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(PauliX)", _ctrl_x_to_mcx)


def _pauliy_to_ry_gp_resources(wires: AbstractWires):
    return {qp.GlobalPhase: 1, qp.RY: 1}


@register_resources(_pauliy_to_ry_gp_resources, name="to_ry_gp")
def _pauliy_to_ry_gp(wires: WiresLike):
    qp.RY(np.pi, wires=wires)
    qp.GlobalPhase(-np.pi / 2)


@register_resources(lambda *_, **__: {qp.RY: 1, qp.GlobalPhase: 1}, name="to_ry")
def _pow_y_standard(base, z):
    z_mod2 = qp.math.array(z) % 2
    qp.RY(np.pi * z_mod2, wires=base.wires)
    qp.GlobalPhase(-np.pi / 2 * z_mod2)


def _pauliy_to_ppr_resources(wires: AbstractWires):
    return {qp.PPR_2(1, "Y", Wire[1]): 1, qp.GlobalPhase: 1}


@register_resources(_pauliy_to_ppr_resources, name="to_ppr")
def _pauliy_to_ppr(wires: WiresLike):
    qp.PPR_2(1, "Y", wires=wires)
    qp.GlobalPhase(-np.pi / 2)


add_decomps(PauliY, _pauliy_to_ry_gp, _pauliy_to_ppr)

add_decomps("Adjoint(PauliY)", self_adjoint)

add_decomps("Pow(PauliY)", pow_involutory2, _pow_y_standard)


def _controlled_y_standard_resources(
    base, control_wires, control_values, work_wires, work_wire_type
):
    if len(control_wires) == 1:
        return {qp.CY: 1}
    return {
        qp.S: 1,
        _adjoint_abstract(qp.S): 1,
        qp.MultiControlledX(
            Wire[len(control_wires) + 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 1,
    }


@register_resources(_controlled_y_standard_resources, name="to_mcx")
def _controlled_y_standard(base, control_wires, control_values, work_wires, work_wire_type):
    wires = concatenate_wires(control_wires, base.wires)

    if len(control_wires) == 1:
        qp.CY(wires=wires)
        return

    qp.adjoint(qp.S(wires[-1]))
    qp.MultiControlledX(wires, work_wires=work_wires, work_wire_type=work_wire_type)
    qp.S(wires=wires[-1])


add_decomps("C(PauliY)", flip_zero_control2(_controlled_y_standard, name="to_mcx"))


def _pauliz_to_ps_resources(wires: AbstractWires):
    return {qp.PhaseShift: 1}


@register_resources(_pauliz_to_ps_resources, name="to_ps")
def _pauliz_to_ps(wires: WiresLike):
    qp.PhaseShift(np.pi, wires=wires)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 2, 0.5))
@register_resources(lambda *_, **__: {qp.S: 1}, name="to_s")
def _pow_z_to_s(base, z):
    qp.S(wires=base.wires)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 2, 0.25))
@register_resources(lambda *_, **__: {qp.T: 1}, name="to_t")
def _pow_z_to_t(base, z):
    qp.T(wires=base.wires)


@register_resources(lambda *_, **__: {qp.PhaseShift: 1}, name="to_phaseshift")
def _pow_z_standard(base, z):
    z_mod2 = qp.math.array(z) % 2
    qp.PhaseShift(np.pi * z_mod2, wires=base.wires)


def _pauliz_to_ppr_resources(wires: AbstractWires):
    return {qp.PPR_2(1, "Z", Wire[1]): 1, qp.GlobalPhase: 1}


@register_resources(_pauliz_to_ppr_resources, name="to_ppr")
def _pauliz_to_ppr(wires: WiresLike):
    qp.PPR_2(1, "Z", wires=wires)
    qp.GlobalPhase(-np.pi / 2)


add_decomps(PauliZ, _pauliz_to_ps, _pauliz_to_ppr)

add_decomps("Adjoint(PauliZ)", self_adjoint)

add_decomps("Pow(PauliZ)", pow_involutory2, _pow_z_standard, _pow_z_to_s, _pow_z_to_t)


def _controlled_z_standard_resources(
    base, control_wires, control_values, work_wires, work_wire_type
):
    if len(control_wires) == 1:
        return {qp.CZ: 1}
    return {
        qp.H: 2,
        qp.MultiControlledX(
            Wire[len(control_wires) + 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 1,
    }


@register_resources(_controlled_z_standard_resources, name="to_mcx")
def _controlled_z_standard(base, control_wires, control_values, work_wires, work_wire_type):
    wires = concatenate_wires(control_wires, base.wires)

    if len(control_wires) == 1:
        qp.CZ(wires=wires)
        return

    qp.H(wires=wires[-1])
    qp.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)
    qp.H(wires=wires[-1])


@register_condition(lambda control_wires, **_: len(control_wires) == 2)
@register_resources(lambda *_, **__: {qp.CCZ: 1}, name="to_ccz")
def _controlled_z_to_ccz(base, control_wires, **_):
    qp.CCZ(wires=concatenate_wires(control_wires, base.wires))


add_decomps(
    "C(PauliZ)",
    flip_zero_control2(_controlled_z_to_ccz, name="to_ccz"),
    flip_zero_control2(_controlled_z_standard, name="to_mcx"),
)


def _s_to_phaseshift_resources(wires: WiresLike = None):
    return {qp.PhaseShift: 1}


@register_resources(_s_to_phaseshift_resources, name="to_phaseshift")
def _s_to_phaseshift(wires: WiresLike | None = None):
    qp.PhaseShift(np.pi / 2, wires=wires)


def _s_to_ppr_resources(wires: AbstractWires):
    return {qp.PPR_4(1, "Z", Wire[1]): 1, qp.GlobalPhase: 1}


@register_resources(_s_to_ppr_resources, name="to_ppr")
def _s_to_ppr(wires: WiresLike):
    qp.PPR_4(1, "Z", wires=wires)
    qp.GlobalPhase(-np.pi / 4)


add_decomps(S, _s_to_phaseshift, _s_to_ppr)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 4, 0.5))
@register_resources(lambda *_, **__: {qp.T: 1}, name="to_t")
def _pow_s_to_t(base, z):
    qp.T(wires=base.wires)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 4, 2))
@register_resources(lambda *_, **__: {qp.Z: 1}, name="to_z")
def _pow_s_to_z(base, z):
    qp.Z(wires=base.wires)


@register_resources(lambda *_, **__: {qp.PhaseShift: 1}, name="to_phaseshift")
def _pow_s_standard(base, z):
    z_mod4 = qp.math.array(z) % 4
    qp.PhaseShift(np.pi * z_mod4 / 2, wires=base.wires)


add_decomps("Pow(S)", make_pow_decomp_with_period2(4), _pow_s_standard, _pow_s_to_t, _pow_s_to_z)


def _t_to_phaseshift_resources(wires=None):
    return {qp.PhaseShift: 1}


@register_resources(_t_to_phaseshift_resources, name="to_phaseshift")
def _t_to_phaseshift(wires=None):
    qp.PhaseShift(np.pi / 4, wires=wires)


def _t_to_ppr_resources(wires: AbstractWires):
    return {qp.PPR_8(1, "Z", Wire[1]): 1, qp.GlobalPhase: 1}


@register_resources(_t_to_ppr_resources, name="to_ppr")
def _t_to_ppr(wires: WiresLike):
    qp.PPR_8(1, "Z", wires=wires)
    qp.GlobalPhase(-np.pi / 8)


@register_resources(lambda *_, **__: {qp.PhaseShift: 1}, name="to_phaseshift")
def _pow_t_standard(base, z):
    z_mod8 = qp.math.array(z) % 8
    qp.PhaseShift(np.pi * z_mod8 / 4, wires=base.wires)


add_decomps(T, _t_to_phaseshift, _t_to_ppr)

add_decomps("Pow(T)", make_pow_decomp_with_period2(8), _pow_t_standard)


def _sx_to_rx_resources(wires: WiresLike = None):
    return {qp.RX: 1, qp.GlobalPhase: 1}


@register_resources(_sx_to_rx_resources, name="to_rx")
def _sx_to_rx(wires: WiresLike | None = None):
    qp.RX(np.pi / 2, wires=wires)
    qp.GlobalPhase(-np.pi / 4)


add_decomps(SX, _sx_to_rx)


@register_condition(lambda base, z: math.shape(z) == () and z % 4 == 2)
@register_resources(lambda *_, **__: {qp.X: 1}, name="to_x")
def _pow_sx_to_x(base, z):
    qp.X(base.wires)


@register_resources(lambda *_, **__: {qp.RX: 1, qp.GlobalPhase: 1}, name="to_rx")
def _pow_sx_standard(base, z):
    z_mod4 = qp.math.array(z) % 4
    qp.RX(np.pi / 2 * z_mod4, wires=base.wires)
    qp.GlobalPhase(-np.pi / 4 * z_mod4)


add_decomps("Pow(SX)", make_pow_decomp_with_period2(4), _pow_sx_to_x, _pow_sx_standard)


def _swap_to_cnot_resources(wires: WiresLike):
    return {qp.CNOT: 3}


@register_resources(_swap_to_cnot_resources, name="to_cnot")
def _swap_to_cnot(wires, **_):
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.CNOT(wires=[wires[1], wires[0]])
    qp.CNOT(wires=[wires[0], wires[1]])


def _swap_to_ppr_resources(wires: WiresLike):
    return {
        qp.PPR_4(1, pauli_word="XX", wires=Wire[2]): 1,
        qp.PPR_4(1, pauli_word="YY", wires=Wire[2]): 1,
        qp.PPR_4(1, pauli_word="ZZ", wires=Wire[2]): 1,
        qp.GlobalPhase: 1,
    }


@register_resources(_swap_to_ppr_resources, name="to_ppr")
def _swap_to_ppr(wires, **_):
    qp.PPR_4(1, "YY", wires=wires)
    qp.PPR_4(1, "XX", wires=wires)
    qp.PPR_4(1, "ZZ", wires=wires)
    qp.GlobalPhase(-np.pi / 4)


add_decomps(SWAP, _swap_to_cnot, _swap_to_ppr)

add_decomps("Adjoint(SWAP)", self_adjoint)

add_decomps("Pow(SWAP)", pow_involutory2)


def _controlled_swap_standard_resources(
    base, control_wires, control_values, work_wires, work_wire_type
):
    return {
        qp.CNOT: 2,
        qp.MultiControlledX(
            Wire[len(control_wires) + 2],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 1,
    }


@register_resources(_controlled_swap_standard_resources, name="to_cnot_mcx")
def _controlled_swap_standard(base, control_wires, control_values, work_wires, work_wire_type):
    wires = concatenate_wires(control_wires, base.wires)
    qp.CNOT(wires=[wires[-2], wires[-1]])
    qp.MultiControlledX(
        wires=concatenate_wires(wires[:-2], [wires[-1], wires[-2]]),
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    qp.CNOT(wires=[wires[-2], wires[-1]])


@register_condition(lambda control_wires, **_: len(control_wires) == 1)
@register_resources(lambda *_, **__: {qp.CSWAP: 1}, name="to_cswap")
def _controlled_swap_to_cswap(base, control_wires, **_):
    qp.CSWAP(wires=concatenate_wires(control_wires, base.wires))


add_decomps(
    "C(SWAP)",
    flip_zero_control2(_controlled_swap_to_cswap, name="to_cswap"),
    flip_zero_control2(_controlled_swap_standard, name="to_cnot_mcx"),
)


def _ecr_standard_resources(wires: WiresLike):
    return {Z: 1, qp.CNOT: 1, SX: 1, qp.RX: 2, qp.RY: 1}


@register_resources(_ecr_standard_resources, name="to_cnot_single_qubit")
def _ecr_standard(wires, **_):
    Z(wires=[wires[0]])
    qp.CNOT(wires=[wires[0], wires[1]])
    SX(wires=[wires[1]])
    qp.RX(np.pi / 2, wires=[wires[0]])
    qp.RY(np.pi / 2, wires=[wires[0]])
    qp.RX(np.pi / 2, wires=[wires[0]])


add_decomps(ECR, _ecr_standard)

add_decomps("Adjoint(ECR)", self_adjoint)

add_decomps("Pow(ECR)", pow_involutory2)


def _iswap_standard_resources(wires: WiresLike):
    return {qp.S: 2, qp.Hadamard: 2, qp.CNOT: 2}


@register_resources(_iswap_standard_resources, name="to_s_h_cnot")
def _iswap_standard(wires):
    S(wires=wires[0])
    S(wires=wires[1])
    Hadamard(wires=wires[0])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.CNOT(wires=[wires[1], wires[0]])
    Hadamard(wires=wires[1])


def _iswap_to_ppr_resources(wires: WiresLike):
    return {
        qp.PPR_4(-1, pauli_word="XX", wires=Wire[2]): 1,
        qp.PPR_4(-1, pauli_word="YY", wires=Wire[2]): 1,
    }


@register_resources(_iswap_to_ppr_resources, name="to_ppr")
def _iswap_to_ppr(wires):
    qp.PPR_4(-1, "YY", wires=wires)
    qp.PPR_4(-1, "XX", wires=wires)


add_decomps(ISWAP, _iswap_standard, _iswap_to_ppr)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 4, 0.5))
@register_resources(lambda *_, **__: {qp.SISWAP: 1}, name="to_siswap")
def _pow_iswap_to_siswap(base, z):
    qp.SISWAP(wires=base.wires)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 4, 2))
@register_resources(lambda *_, **__: {qp.Z: 2}, name="to_zz")
def _pow_iswap_to_zz(base, z):
    qp.Z(wires=base.wires[0])
    qp.Z(wires=base.wires[1])


add_decomps("Pow(ISWAP)", make_pow_decomp_with_period2(4), _pow_iswap_to_zz, _pow_iswap_to_siswap)


def _siswap_standard_resources(wires: WiresLike):
    return {SX: 6, qp.RZ: 4, qp.CNOT: 2}


@register_resources(_siswap_standard_resources, name="to_sx_rz_cnot")
def _siswap_standard(wires):
    SX(wires=wires[0])
    qp.RZ(np.pi / 2, wires=wires[0])
    qp.CNOT(wires=[wires[0], wires[1]])
    SX(wires=wires[0])
    qp.RZ(7 * np.pi / 4, wires=wires[0])
    SX(wires=wires[0])
    qp.RZ(np.pi / 2, wires=wires[0])
    SX(wires=wires[1])
    qp.RZ(7 * np.pi / 4, wires=wires[1])
    qp.CNOT(wires=[wires[0], wires[1]])
    SX(wires=wires[0])
    SX(wires=wires[1])


def _siswap_to_ppr_resources(wires: WiresLike):
    return {
        qp.PPR_8(-1, pauli_word="XX", wires=Wire[2]): 1,
        qp.PPR_8(-1, pauli_word="YY", wires=Wire[2]): 1,
    }


@register_resources(_siswap_to_ppr_resources, name="to_ppr")
def _siswap_to_ppr(wires, **_):
    qp.PPR_8(-1, "YY", wires=wires)
    qp.PPR_8(-1, "XX", wires=wires)


add_decomps(SISWAP, _siswap_standard, _siswap_to_ppr)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 8, 2))
@register_resources(lambda *_, **__: {qp.ISWAP: 1}, name="to_iswap")
def _pow_siswap_to_iswap(base, z):
    qp.ISWAP(base.wires)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 8, 4))
@register_resources(lambda *_, **__: {qp.Z: 2}, name="to_zz")
def _pow_siswap_to_zz(base, z):
    qp.Z(wires=base.wires[0])
    qp.Z(wires=base.wires[1])


add_decomps("Pow(SISWAP)", make_pow_decomp_with_period2(8), _pow_siswap_to_zz, _pow_siswap_to_iswap)


_PAULI_OPS = {"X": PauliX, "Y": PauliY, "Z": PauliZ}


def _ppr2_to_paulis_resources(pauli_word, **_):
    return {qp.GlobalPhase: 1, **Counter(_PAULI_OPS[p] for p in pauli_word if p != "I")}


@register_resources(_ppr2_to_paulis_resources, name="to_paulis")
def _ppr2_to_paulis(sign, pauli_word, wires):
    qp.GlobalPhase(sign * np.pi / 2)
    for wire, pauli in zip(wires, pauli_word, strict=True):
        if pauli != "I":
            _PAULI_OPS[pauli](wires=wire)


def _z_gate(sign, gate, wires):
    """``gate`` (``S`` or ``T``) for ``sign=1`` and its adjoint for ``sign=-1``."""
    return gate(wires) if sign > 0 else qp.adjoint(gate(wires))


def _ppr_basis_resources(pauli_word, reverse, **_):
    resources = {}
    if len(pauli_word) > 1:
        resources[qp.CNOT] = len(pauli_word) - 1
    num_hadamards = pauli_word.count("X") + pauli_word.count("Y")
    if num_hadamards:
        resources[Hadamard] = num_hadamards
    if pauli_word.count("Y"):
        resources[S if reverse else _adjoint_abstract(S)] = pauli_word.count("Y")
    return resources


@subcircuit(static_argnames=("pauli_word", "reverse"))
@register_resources(_ppr_basis_resources)
def _PPRBasis(pauli_word: str, wires: WiresLike, reverse: bool):
    if reverse:
        for index in range(1, len(wires)):
            qp.CNOT(wires=(wires[index], wires[index - 1]))
    for wire, pauli in zip(wires, pauli_word, strict=True):
        if pauli == "X":
            Hadamard(wire)
        elif pauli == "Y":
            if not reverse:
                qp.adjoint(S(wire))
            Hadamard(wire)
            if reverse:
                S(wire)
    if not reverse:
        for index in range(len(wires) - 1, 0, -1):
            qp.CNOT(wires=(wires[index], wires[index - 1]))


def _ppr_via_z_gate_resources(gate, sign, pauli_word, wires):
    """Resources of ``_ppr_via_z_gate`` and a global phase."""
    active_word = pauli_word.replace("I", "")
    if not active_word:
        return {qp.GlobalPhase: 1}

    z_gate = _z_gate(sign, gate, Wire[1])
    if active_word == "Z":
        return {qp.GlobalPhase: 1, z_gate: 1}

    compute = _PPRBasis(active_word, wires=Wire[len(active_word)], reverse=False)
    uncompute = _PPRBasis(active_word, wires=Wire[len(active_word)], reverse=True)
    return {qp.GlobalPhase: 1, _change_op_basis_abstract(compute, z_gate, uncompute): 1}


def _ppr_via_z_gate(sign, pauli_word, wires, gate):
    """Apply the Pauli product rotation that is equal to ``gate`` (``S`` or ``T``) or its adjoint
    for the Pauli word ``"Z"``, up to a global phase, by changing the basis of ``pauli_word`` to
    a single Pauli ``Z``.

    Examples:

    For ``pauli_word="XYZ"``, ``gate=T`` and ``sign=1``:

    .. code-block:: text

        0: ──H────────╭X──T─╭X──H───────
        1: ──S†──H─╭X─╰●────╰●─╭X──H──S─
        2: ────────╰●──────────╰●───────

    For ``pauli_word="IYZIX"``, ``gate=S`` and ``sign=-1``, the identities are skipped, the
    central gate acts on the first non-identity wire, and the central gate is adjoint.

    .. code-block:: text

        0: ────────────────────────
        1: ──S†──H─╭X──S†─╭X──H──S─
        2: ─────╭X─╰●─────╰●─╭X────
        3: ─────│────────────│─────
        4: ──H──╰●───────────╰●──H─
    """
    active = [(wire, pauli) for wire, pauli in zip(wires, pauli_word, strict=True) if pauli != "I"]
    if not active:
        return

    active_wires = [wire for wire, _ in active]
    central_op = _z_gate(sign, gate, active_wires[0])
    if pauli_word.replace("I", "") == "Z":
        return

    active_word = pauli_word.replace("I", "")
    qp.change_op_basis(
        _PPRBasis(active_word, wires=active_wires, reverse=False),
        central_op,
        _PPRBasis(active_word, wires=active_wires, reverse=True),
    )


@register_resources(partial(_ppr_via_z_gate_resources, gate=S), name="to_cliffords")
def _ppr4_to_cliffords(sign, pauli_word, wires):
    qp.GlobalPhase(sign * np.pi / 4)
    _ppr_via_z_gate(sign, pauli_word, wires, S)


@register_resources(partial(_ppr_via_z_gate_resources, gate=T), name="to_clifford_t")
def _ppr8_to_clifford_t(sign, pauli_word, wires):
    qp.GlobalPhase(sign * np.pi / 8)
    _ppr_via_z_gate(sign, pauli_word, wires, T)


add_decomps(PPR_2, _ppr2_to_paulis)

add_decomps(PPR_4, _ppr4_to_cliffords)

add_decomps(PPR_8, _ppr8_to_clifford_t)


def _adjoint_ppr_flip_sign_standard_resources(base):
    return {type(base)(-base.sign, pauli_word=base.pauli_word, wires=Wire[len(base.wires)]): 1}


@register_resources(_adjoint_ppr_flip_sign_standard_resources, name="flip_sign")
def _adjoint_ppr_flip_sign_standard(base):
    type(base)(-base.sign, pauli_word=base.pauli_word, wires=base.wires)


add_decomps("Adjoint(PPR_2)", _adjoint_ppr_flip_sign_standard)

add_decomps("Adjoint(PPR_4)", _adjoint_ppr_flip_sign_standard)

add_decomps("Adjoint(PPR_8)", _adjoint_ppr_flip_sign_standard)
