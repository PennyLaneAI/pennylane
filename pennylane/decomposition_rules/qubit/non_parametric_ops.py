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

from pennylane import allocation, math, ops
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.ops.op_math.controlled2 import flip_zero_control
from pennylane.ops.op_math.pow2 import make_pow_decomp_with_period, pow_involutory
from pennylane.decomposition.symbolic_decomposition import self_adjoint
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


@register_resources({ops.RZ: 2, ops.RX: 1, ops.GlobalPhase: 1}, name="to_rz_rx")
def hadamard_to_rz_rx(wires: WiresLike, **__):
    """Decompose ``Hadamard`` using the ``to_rz_rx`` strategy."""
    ops.RZ(np.pi / 2, wires=wires)
    ops.RX(np.pi / 2, wires=wires)
    ops.RZ(np.pi / 2, wires=wires)
    ops.GlobalPhase(-np.pi / 2)


@register_resources({ops.RZ: 1, ops.RY: 1, ops.GlobalPhase: 1}, name="to_rz_ry")
def hadamard_to_rz_ry(wires: WiresLike, **__):
    """Decompose ``Hadamard`` using the ``to_rz_ry`` strategy."""
    ops.RZ(np.pi, wires=wires)
    ops.RY(np.pi / 2, wires=wires)
    ops.GlobalPhase(-np.pi / 2)


@register_resources(
    {
        PauliMeasure("YY", wires=Wire[2]): 1,
        PauliMeasure("X", wires=Wire[1]): 1,
        PauliY: 1,
        PauliZ: 2,
        ops.GlobalPhase: 1,
    },
    work_wires={"burnable": 1},
    name="to_ppm",
)
def hadamard_to_ppm(wires, **__):
    """Decompose ``Hadamard`` using the ``to_ppm`` strategy."""
    with allocation.allocate(1, state="zero", restored=False) as work_wires:
        PauliZ(wires)
        m0 = pauli_measure("YY", [wires[0], work_wires[0]])
        m1 = pauli_measure("X", work_wires)
        ops.cond(m0 == m1, PauliY)(wires)
        ops.cond(m0 == m1, ops.GlobalPhase)(np.pi / 2)
        ops.cond(m1, PauliZ)(work_wires[0])  # Reset work wire to |+>


@register_resources(
    {PPR_2(1, "Z", Wire[1]): 1, PPR_4(1, "Y", Wire[1]): 1, ops.GlobalPhase: 1},
    name="to_ppr",
)
def hadamard_to_ppr(wires: WiresLike, **__):
    """Decompose ``Hadamard`` using the ``to_ppr`` strategy."""
    PPR_2(1, "Z", wires=wires)
    PPR_4(1, "Y", wires=wires)
    ops.GlobalPhase(-np.pi / 2)


add_decomps(Hadamard, hadamard_to_rz_rx, hadamard_to_rz_ry, hadamard_to_ppm, hadamard_to_ppr)

add_decomps("Adjoint(Hadamard)", self_adjoint)

add_decomps("Pow(Hadamard)", pow_involutory)


def _controlled_hadamard_standard_resources(
    base, control_wires, control_values, work_wires, work_wire_type
):
    if len(control_wires) == 1:
        return {ops.CH: 1}
    return {
        Hadamard: 2,
        ops.RY: 2,
        ops.MultiControlledX(
            Wire[len(control_wires) + 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 1,
    }


@register_resources(_controlled_hadamard_standard_resources, name="to_mcx")
def controlled_hadamard_standard(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose ``C(Hadamard)`` using the ``to_mcx`` strategy."""
    wires = concatenate_wires(control_wires, base.wires)
    if len(control_wires) == 1:
        ops.CH(wires)
        return

    ops.RY(-np.pi / 4, wires=wires[-1])
    Hadamard(wires=wires[-1])
    ops.MultiControlledX(wires, work_wires=work_wires, work_wire_type=work_wire_type)
    Hadamard(wires=wires[-1])
    ops.RY(np.pi / 4, wires=wires[-1])


add_decomps("C(Hadamard)", flip_zero_control(controlled_hadamard_standard, name="to_mcx"))


@register_resources({ops.GlobalPhase: 1, ops.RX: 1}, name="to_rx")
def paulix_to_rx(wires: WiresLike):
    """Decompose ``PauliX`` using the ``to_rx`` strategy."""
    ops.RX(np.pi, wires=wires)
    ops.GlobalPhase(-np.pi / 2)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 2, 0.5))
@register_resources({SX: 1}, name="to_sx")
def pow_x_to_sx(base, z):
    """Decompose ``Pow(PauliX)`` using the ``to_sx`` strategy."""
    SX(wires=base.wires)


@register_resources({ops.RX: 1, ops.GlobalPhase: 1}, name="to_rx")
def pow_x_to_rx(base, z):
    """Decompose ``Pow(PauliX)`` using the ``to_rx`` strategy."""
    z_mod2 = math.array(z) % 2
    ops.RX(np.pi * z_mod2, wires=base.wires)
    ops.GlobalPhase(-np.pi / 2 * z_mod2)


@register_resources({PPR_2(1, "X", Wire[1]): 1, ops.GlobalPhase: 1}, name="to_ppr")
def paulix_to_ppr(wires: WiresLike):
    """Decompose ``PauliX`` using the ``to_ppr`` strategy."""
    PPR_2(1, "X", wires=wires)
    ops.GlobalPhase(-np.pi / 2)


add_decomps(PauliX, paulix_to_rx, paulix_to_ppr)

add_decomps("Adjoint(PauliX)", self_adjoint)

add_decomps("Pow(PauliX)", pow_involutory, pow_x_to_rx, pow_x_to_sx)


def _ctrl_x_to_mcx_resources(base, control_wires, control_values, work_wires, work_wire_type):
    return {
        ops.MultiControlledX(
            Wire[len(control_wires) + 1],
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@register_resources(_ctrl_x_to_mcx_resources, name="to_mcx")
def ctrl_x_to_mcx(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose ``C(PauliX)`` using the ``to_mcx`` strategy."""
    ops.MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(PauliX)", ctrl_x_to_mcx)


@register_resources({ops.GlobalPhase: 1, ops.RY: 1}, name="to_ry_gp")
def pauliy_to_ry_gp(wires: WiresLike):
    """Decompose ``PauliY`` using the ``to_ry_gp`` strategy."""
    ops.RY(np.pi, wires=wires)
    ops.GlobalPhase(-np.pi / 2)


@register_resources({ops.RY: 1, ops.GlobalPhase: 1}, name="to_ry")
def pow_y_standard(base, z):
    """Decompose ``Pow(PauliY)`` using the ``to_ry`` strategy."""
    z_mod2 = math.array(z) % 2
    ops.RY(np.pi * z_mod2, wires=base.wires)
    ops.GlobalPhase(-np.pi / 2 * z_mod2)


@register_resources({PPR_2(1, "Y", Wire[1]): 1, ops.GlobalPhase: 1}, name="to_ppr")
def pauliy_to_ppr(wires: WiresLike):
    """Decompose ``PauliY`` using the ``to_ppr`` strategy."""
    PPR_2(1, "Y", wires=wires)
    ops.GlobalPhase(-np.pi / 2)


add_decomps(PauliY, pauliy_to_ry_gp, pauliy_to_ppr)

add_decomps("Adjoint(PauliY)", self_adjoint)

add_decomps("Pow(PauliY)", pow_involutory, pow_y_standard)


def _controlled_y_standard_resources(
    base, control_wires, control_values, work_wires, work_wire_type
):
    if len(control_wires) == 1:
        return {ops.CY: 1}
    return {
        S: 1,
        _adjoint_abstract(S): 1,
        ops.MultiControlledX(
            Wire[len(control_wires) + 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 1,
    }


@register_resources(_controlled_y_standard_resources, name="to_mcx")
def controlled_y_standard(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose ``C(PauliY)`` using the ``to_mcx`` strategy."""
    wires = concatenate_wires(control_wires, base.wires)

    if len(control_wires) == 1:
        ops.CY(wires=wires)
        return

    ops.adjoint(S(wires[-1]))
    ops.MultiControlledX(wires, work_wires=work_wires, work_wire_type=work_wire_type)
    S(wires=wires[-1])


add_decomps("C(PauliY)", flip_zero_control(controlled_y_standard, name="to_mcx"))


@register_resources({ops.PhaseShift: 1}, name="to_ps")
def pauliz_to_ps(wires: WiresLike):
    """Decompose ``PauliZ`` using the ``to_ps`` strategy."""
    ops.PhaseShift(np.pi, wires=wires)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 2, 0.5))
@register_resources({S: 1}, name="to_s")
def pow_z_to_s(base, z):
    """Decompose ``Pow(PauliZ)`` using the ``to_s`` strategy."""
    S(wires=base.wires)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 2, 0.25))
@register_resources({T: 1}, name="to_t")
def pow_z_to_t(base, z):
    """Decompose ``Pow(PauliZ)`` using the ``to_t`` strategy."""
    T(wires=base.wires)


@register_resources({ops.PhaseShift: 1}, name="to_phaseshift")
def pow_z_standard(base, z):
    """Decompose ``Pow(PauliZ)`` using the ``to_phaseshift`` strategy."""
    z_mod2 = math.array(z) % 2
    ops.PhaseShift(np.pi * z_mod2, wires=base.wires)


@register_resources({PPR_2(1, "Z", Wire[1]): 1, ops.GlobalPhase: 1}, name="to_ppr")
def pauliz_to_ppr(wires: WiresLike):
    """Decompose ``PauliZ`` using the ``to_ppr`` strategy."""
    PPR_2(1, "Z", wires=wires)
    ops.GlobalPhase(-np.pi / 2)


add_decomps(PauliZ, pauliz_to_ps, pauliz_to_ppr)

add_decomps("Adjoint(PauliZ)", self_adjoint)

add_decomps("Pow(PauliZ)", pow_involutory, pow_z_standard, pow_z_to_s, pow_z_to_t)


def _controlled_z_standard_resources(
    base, control_wires, control_values, work_wires, work_wire_type
):
    if len(control_wires) == 1:
        return {ops.CZ: 1}
    return {
        Hadamard: 2,
        ops.MultiControlledX(
            Wire[len(control_wires) + 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 1,
    }


@register_resources(_controlled_z_standard_resources, name="to_mcx")
def controlled_z_standard(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose ``C(PauliZ)`` using the ``to_mcx`` strategy."""
    wires = concatenate_wires(control_wires, base.wires)

    if len(control_wires) == 1:
        ops.CZ(wires=wires)
        return

    Hadamard(wires=wires[-1])
    ops.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)
    Hadamard(wires=wires[-1])


@register_condition(lambda control_wires, **_: len(control_wires) == 2)
@register_resources({ops.CCZ: 1}, name="to_ccz")
def controlled_z_to_ccz(base, control_wires, **_):
    """Decompose ``C(PauliZ)`` using the ``to_ccz`` strategy."""
    ops.CCZ(wires=concatenate_wires(control_wires, base.wires))


add_decomps(
    "C(PauliZ)",
    flip_zero_control(controlled_z_to_ccz, name="to_ccz"),
    flip_zero_control(controlled_z_standard, name="to_mcx"),
)


@register_resources({ops.PhaseShift: 1}, name="to_phaseshift")
def s_to_phaseshift(wires: WiresLike | None = None):
    """Decompose ``S`` using the ``to_phaseshift`` strategy."""
    ops.PhaseShift(np.pi / 2, wires=wires)


@register_resources({PPR_4(1, "Z", Wire[1]): 1, ops.GlobalPhase: 1}, name="to_ppr")
def s_to_ppr(wires: WiresLike):
    """Decompose ``S`` using the ``to_ppr`` strategy."""
    PPR_4(1, "Z", wires=wires)
    ops.GlobalPhase(-np.pi / 4)


add_decomps(S, s_to_phaseshift, s_to_ppr)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 4, 0.5))
@register_resources({T: 1}, name="to_t")
def pow_s_to_t(base, z):
    """Decompose ``Pow(S)`` using the ``to_t`` strategy."""
    T(wires=base.wires)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 4, 2))
@register_resources({PauliZ: 1}, name="to_z")
def pow_s_to_z(base, z):
    """Decompose ``Pow(S)`` using the ``to_z`` strategy."""
    PauliZ(wires=base.wires)


@register_resources({ops.PhaseShift: 1}, name="to_phaseshift")
def pow_s_standard(base, z):
    """Decompose ``Pow(S)`` using the ``to_phaseshift`` strategy."""
    z_mod4 = math.array(z) % 4
    ops.PhaseShift(np.pi * z_mod4 / 2, wires=base.wires)


add_decomps("Pow(S)", make_pow_decomp_with_period(4), pow_s_standard, pow_s_to_t, pow_s_to_z)


@register_resources({ops.PhaseShift: 1}, name="to_phaseshift")
def t_to_phaseshift(wires=None):
    """Decompose ``T`` using the ``to_phaseshift`` strategy."""
    ops.PhaseShift(np.pi / 4, wires=wires)


@register_resources({PPR_8(1, "Z", Wire[1]): 1, ops.GlobalPhase: 1}, name="to_ppr")
def t_to_ppr(wires: WiresLike):
    """Decompose ``T`` using the ``to_ppr`` strategy."""
    PPR_8(1, "Z", wires=wires)
    ops.GlobalPhase(-np.pi / 8)


@register_resources({ops.PhaseShift: 1}, name="to_phaseshift")
def pow_t_standard(base, z):
    """Decompose ``Pow(T)`` using the ``to_phaseshift`` strategy."""
    z_mod8 = math.array(z) % 8
    ops.PhaseShift(np.pi * z_mod8 / 4, wires=base.wires)


add_decomps(T, t_to_phaseshift, t_to_ppr)

add_decomps("Pow(T)", make_pow_decomp_with_period(8), pow_t_standard)


@register_resources({ops.RX: 1, ops.GlobalPhase: 1}, name="to_rx")
def sx_to_rx(wires: WiresLike | None = None):
    """Decompose ``SX`` using the ``to_rx`` strategy."""
    ops.RX(np.pi / 2, wires=wires)
    ops.GlobalPhase(-np.pi / 4)


add_decomps(SX, sx_to_rx)


@register_condition(lambda base, z: math.shape(z) == () and z % 4 == 2)
@register_resources({PauliX: 1}, name="to_x")
def pow_sx_to_x(base, z):
    """Decompose ``Pow(SX)`` using the ``to_x`` strategy."""
    PauliX(base.wires)


@register_resources({ops.RX: 1, ops.GlobalPhase: 1}, name="to_rx")
def pow_sx_standard(base, z):
    """Decompose ``Pow(SX)`` using the ``to_rx`` strategy."""
    z_mod4 = math.array(z) % 4
    ops.RX(np.pi / 2 * z_mod4, wires=base.wires)
    ops.GlobalPhase(-np.pi / 4 * z_mod4)


add_decomps("Pow(SX)", make_pow_decomp_with_period(4), pow_sx_to_x, pow_sx_standard)


@register_resources({ops.CNOT: 3}, name="to_cnot")
def swap_to_cnot(wires, **_):
    """Decompose ``SWAP`` using the ``to_cnot`` strategy."""
    ops.CNOT(wires=[wires[0], wires[1]])
    ops.CNOT(wires=[wires[1], wires[0]])
    ops.CNOT(wires=[wires[0], wires[1]])


@register_resources(
    {
        PPR_4(1, pauli_word="XX", wires=Wire[2]): 1,
        PPR_4(1, pauli_word="YY", wires=Wire[2]): 1,
        PPR_4(1, pauli_word="ZZ", wires=Wire[2]): 1,
        ops.GlobalPhase: 1,
    },
    name="to_ppr",
)
def swap_to_ppr(wires, **_):
    """Decompose ``SWAP`` using the ``to_ppr`` strategy."""
    PPR_4(1, "YY", wires=wires)
    PPR_4(1, "XX", wires=wires)
    PPR_4(1, "ZZ", wires=wires)
    ops.GlobalPhase(-np.pi / 4)


add_decomps(SWAP, swap_to_cnot, swap_to_ppr)

add_decomps("Adjoint(SWAP)", self_adjoint)

add_decomps("Pow(SWAP)", pow_involutory)


def _controlled_swap_standard_resources(
    base, control_wires, control_values, work_wires, work_wire_type
):
    return {
        ops.CNOT: 2,
        ops.MultiControlledX(
            Wire[len(control_wires) + 2],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 1,
    }


@register_resources(_controlled_swap_standard_resources, name="to_cnot_mcx")
def controlled_swap_standard(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose ``C(SWAP)`` using the ``to_cnot_mcx`` strategy."""
    wires = concatenate_wires(control_wires, base.wires)
    ops.CNOT(wires=[wires[-2], wires[-1]])
    ops.MultiControlledX(
        wires=concatenate_wires(wires[:-2], [wires[-1], wires[-2]]),
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    ops.CNOT(wires=[wires[-2], wires[-1]])


@register_condition(lambda control_wires, **_: len(control_wires) == 1)
@register_resources({ops.CSWAP: 1}, name="to_cswap")
def controlled_swap_to_cswap(base, control_wires, **_):
    """Decompose ``C(SWAP)`` using the ``to_cswap`` strategy."""
    ops.CSWAP(wires=concatenate_wires(control_wires, base.wires))


add_decomps(
    "C(SWAP)",
    flip_zero_control(controlled_swap_to_cswap, name="to_cswap"),
    flip_zero_control(controlled_swap_standard, name="to_cnot_mcx"),
)


@register_resources({Z: 1, ops.CNOT: 1, SX: 1, ops.RX: 2, ops.RY: 1}, name="to_cnot_single_qubit")
def ecr_standard(wires, **_):
    """Decompose ``ECR`` using the ``to_cnot_single_qubit`` strategy."""
    Z(wires=[wires[0]])
    ops.CNOT(wires=[wires[0], wires[1]])
    SX(wires=[wires[1]])
    ops.RX(np.pi / 2, wires=[wires[0]])
    ops.RY(np.pi / 2, wires=[wires[0]])
    ops.RX(np.pi / 2, wires=[wires[0]])


add_decomps(ECR, ecr_standard)

add_decomps("Adjoint(ECR)", self_adjoint)

add_decomps("Pow(ECR)", pow_involutory)


@register_resources({S: 2, Hadamard: 2, ops.CNOT: 2}, name="to_s_h_cnot")
def iswap_standard(wires):
    """Decompose ``ISWAP`` using the ``to_s_h_cnot`` strategy."""
    S(wires=wires[0])
    S(wires=wires[1])
    Hadamard(wires=wires[0])
    ops.CNOT(wires=[wires[0], wires[1]])
    ops.CNOT(wires=[wires[1], wires[0]])
    Hadamard(wires=wires[1])


@register_resources(
    {PPR_4(-1, pauli_word="XX", wires=Wire[2]): 1, PPR_4(-1, pauli_word="YY", wires=Wire[2]): 1},
    name="to_ppr",
)
def iswap_to_ppr(wires):
    """Decompose ``ISWAP`` using the ``to_ppr`` strategy."""
    PPR_4(-1, "YY", wires=wires)
    PPR_4(-1, "XX", wires=wires)


add_decomps(ISWAP, iswap_standard, iswap_to_ppr)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 4, 0.5))
@register_resources({SISWAP: 1}, name="to_siswap")
def pow_iswap_to_siswap(base, z):
    """Decompose ``Pow(ISWAP)`` using the ``to_siswap`` strategy."""
    SISWAP(wires=base.wires)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 4, 2))
@register_resources({PauliZ: 2}, name="to_zz")
def pow_iswap_to_zz(base, z):
    """Decompose ``Pow(ISWAP)`` using the ``to_zz`` strategy."""
    PauliZ(wires=base.wires[0])
    PauliZ(wires=base.wires[1])


add_decomps("Pow(ISWAP)", make_pow_decomp_with_period(4), pow_iswap_to_zz, pow_iswap_to_siswap)


@register_resources({SX: 6, ops.RZ: 4, ops.CNOT: 2}, name="to_sx_rz_cnot")
def siswap_standard(wires):
    """Decompose ``SISWAP`` using the ``to_sx_rz_cnot`` strategy."""
    SX(wires=wires[0])
    ops.RZ(np.pi / 2, wires=wires[0])
    ops.CNOT(wires=[wires[0], wires[1]])
    SX(wires=wires[0])
    ops.RZ(7 * np.pi / 4, wires=wires[0])
    SX(wires=wires[0])
    ops.RZ(np.pi / 2, wires=wires[0])
    SX(wires=wires[1])
    ops.RZ(7 * np.pi / 4, wires=wires[1])
    ops.CNOT(wires=[wires[0], wires[1]])
    SX(wires=wires[0])
    SX(wires=wires[1])


@register_resources(
    {PPR_8(-1, pauli_word="XX", wires=Wire[2]): 1, PPR_8(-1, pauli_word="YY", wires=Wire[2]): 1},
    name="to_ppr",
)
def siswap_to_ppr(wires, **_):
    """Decompose ``SISWAP`` using the ``to_ppr`` strategy."""
    PPR_8(-1, "YY", wires=wires)
    PPR_8(-1, "XX", wires=wires)


add_decomps(SISWAP, siswap_standard, siswap_to_ppr)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 8, 2))
@register_resources({ISWAP: 1}, name="to_iswap")
def pow_siswap_to_iswap(base, z):
    """Decompose ``Pow(SISWAP)`` using the ``to_iswap`` strategy."""
    ISWAP(base.wires)


@register_condition(lambda base, z: math.shape(z) == () and math.allclose(z % 8, 4))
@register_resources({PauliZ: 2}, name="to_zz")
def pow_siswap_to_zz(base, z):
    """Decompose ``Pow(SISWAP)`` using the ``to_zz`` strategy."""
    PauliZ(wires=base.wires[0])
    PauliZ(wires=base.wires[1])


add_decomps("Pow(SISWAP)", make_pow_decomp_with_period(8), pow_siswap_to_zz, pow_siswap_to_iswap)


_PAULI_OPS = {"X": PauliX, "Y": PauliY, "Z": PauliZ}


def _ppr2_to_paulis_resources(pauli_word, **_):
    return {ops.GlobalPhase: 1, **Counter(_PAULI_OPS[p] for p in pauli_word if p != "I")}


@register_resources(_ppr2_to_paulis_resources, name="to_paulis")
def ppr2_to_paulis(sign, pauli_word, wires):
    """Decompose ``PPR_2`` using the ``to_paulis`` strategy."""
    ops.GlobalPhase(sign * np.pi / 2)
    for wire, pauli in zip(wires, pauli_word, strict=True):
        if pauli != "I":
            _PAULI_OPS[pauli](wires=wire)


def _z_gate(sign, gate, wires):
    """``gate`` (``S`` or ``T``) for ``sign=1`` and its adjoint for ``sign=-1``."""
    return gate(wires) if sign > 0 else ops.adjoint(gate(wires))


def _ppr_basis_resources(pauli_word, reverse, **_):
    resources = {}
    if len(pauli_word) > 1:
        resources[ops.CNOT] = len(pauli_word) - 1
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
            ops.CNOT(wires=(wires[index], wires[index - 1]))
    for wire, pauli in zip(wires, pauli_word, strict=True):
        if pauli == "X":
            Hadamard(wire)
        elif pauli == "Y":
            if not reverse:
                ops.adjoint(S(wire))
            Hadamard(wire)
            if reverse:
                S(wire)
    if not reverse:
        for index in range(len(wires) - 1, 0, -1):
            ops.CNOT(wires=(wires[index], wires[index - 1]))


def _ppr_via_z_gate_resources(gate, sign, pauli_word, wires):
    """Resources of ``_ppr_via_z_gate`` and a global phase."""
    active_word = pauli_word.replace("I", "")
    if not active_word:
        return {ops.GlobalPhase: 1}

    z_gate = _z_gate(sign, gate, Wire[1])
    if active_word == "Z":
        return {ops.GlobalPhase: 1, z_gate: 1}

    compute = _PPRBasis(active_word, wires=Wire[len(active_word)], reverse=False)
    uncompute = _PPRBasis(active_word, wires=Wire[len(active_word)], reverse=True)
    return {ops.GlobalPhase: 1, _change_op_basis_abstract(compute, z_gate, uncompute): 1}


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
    ops.change_op_basis(
        _PPRBasis(active_word, wires=active_wires, reverse=False),
        central_op,
        _PPRBasis(active_word, wires=active_wires, reverse=True),
    )


@register_resources(partial(_ppr_via_z_gate_resources, gate=S), name="to_cliffords")
def ppr4_to_cliffords(sign, pauli_word, wires):
    """Decompose ``PPR_4`` using the ``to_cliffords`` strategy."""
    ops.GlobalPhase(sign * np.pi / 4)
    _ppr_via_z_gate(sign, pauli_word, wires, S)


@register_resources(partial(_ppr_via_z_gate_resources, gate=T), name="to_clifford_t")
def ppr8_to_clifford_t(sign, pauli_word, wires):
    """Decompose ``PPR_8`` using the ``to_clifford_t`` strategy."""
    ops.GlobalPhase(sign * np.pi / 8)
    _ppr_via_z_gate(sign, pauli_word, wires, T)


add_decomps(PPR_2, ppr2_to_paulis)

add_decomps(PPR_4, ppr4_to_cliffords)

add_decomps(PPR_8, ppr8_to_clifford_t)


def _adjoint_ppr_flip_sign_standard_resources(base):
    return {type(base)(-base.sign, pauli_word=base.pauli_word, wires=Wire[len(base.wires)]): 1}


@register_resources(_adjoint_ppr_flip_sign_standard_resources, name="flip_sign")
def adjoint_ppr_flip_sign_standard(base):
    """Apply the ``flip_sign`` decomposition strategy to the operator."""
    type(base)(-base.sign, pauli_word=base.pauli_word, wires=base.wires)


add_decomps("Adjoint(PPR_2)", adjoint_ppr_flip_sign_standard)

add_decomps("Adjoint(PPR_4)", adjoint_ppr_flip_sign_standard)

add_decomps("Adjoint(PPR_8)", adjoint_ppr_flip_sign_standard)
