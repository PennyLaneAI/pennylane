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

"""Decomposition rules for controlled ops."""

from collections import defaultdict
from functools import partial

import numpy as np

from pennylane import allocation, ops, templates
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition.symbolic_decomposition import self_adjoint
from pennylane.ops.op_math.adjoint2 import _adjoint_negate_angle
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.op_math.pow2 import _pow_scale_angle, pow_involutory
from pennylane.ops.op_math.controlled_ops import (
    CCZ,
    CH,
    CNOT,
    CRX,
    CRY,
    CRZ,
    CRot,
    CSWAP,
    CY,
    CZ,
    ControlledPhaseShift,
    Toffoli,
)
from pennylane.typing import AbstractWires, Float, TensorLike, Wire
from pennylane.wires import WiresLike, concatenate_wires


@register_resources({ops.RY: 2, CZ: 1}, name="to_ry_cz_ry")
def ch_to_ry_cz_ry(wires: WiresLike, **__):
    """Decompose ``CH`` using the ``to_ry_cz_ry`` strategy."""
    ops.RY(-np.pi / 4, wires=wires[1])
    CZ(wires=wires)
    ops.RY(+np.pi / 4, wires=wires[1])


add_decomps(CH, ch_to_ry_cz_ry)
add_decomps("Adjoint(CH)", self_adjoint)
add_decomps("Pow(CH)", pow_involutory)


@register_resources({CRY: 1, ops.S: 1}, name="to_ry_cnot")
def cy_to_ry_cnot(wires: WiresLike, **__):
    """Decompose ``CY`` using the ``to_ry_cnot`` strategy."""
    CRY(np.pi, wires=wires)
    ops.S(wires=wires[0])


# pylint: disable-next=unused-argument
def _pauli_ctrl_pauli_ppr_resources(wires: AbstractWires, pauli0, pauli1):
    """Resources for _pauli_ctrl_pauli_ppr."""
    resources = defaultdict(int)
    p0, p1 = pauli0.__name__[-1], pauli1.__name__[-1]
    resources[ops.PPR_4(-1, pauli_word=p0, wires=Wire[len(p0)])] += 1
    resources[ops.PPR_4(-1, pauli_word=p1, wires=Wire[len(p1)])] += 1
    resources[ops.PPR_4(1, pauli_word=p0 + p1, wires=Wire[len(p0 + p1)])] += 1
    resources[ops.GlobalPhase] += 1
    return dict(resources)


def _pauli_ctrl_pauli_ppr(wires: AbstractWires, pauli0, pauli1):
    """Generalized two-qubit Pauli-controlled Pauli gate decomposition to PPRs."""
    p0, p1 = pauli0.__name__[-1], pauli1.__name__[-1]
    ops.PPR_4(-1, p0, wires=wires[0])
    ops.PPR_4(-1, p1, wires=wires[1])
    ops.PPR_4(1, p0 + p1, wires=wires)
    ops.GlobalPhase(np.pi / 4)


@register_resources(partial(_pauli_ctrl_pauli_ppr_resources, pauli0=ops.Z, pauli1=ops.Y), name="to_ppr")
def cy_to_ppr(wires: AbstractWires):
    """Decompose ``CY`` using the ``to_ppr`` strategy."""
    _pauli_ctrl_pauli_ppr(wires, ops.Z, ops.Y)


def _pauli_ctrl_pauli_ppm(wires, pauli0, pauli1):
    """Generalized two-qubit Pauli-controlled Pauli gate decomposition to PPMs."""
    with allocation.allocate(1, state="zero", restored=False) as work_wires:
        m0 = ops.pauli_measure(pauli0.__name__[-1] + "X", [wires[0], work_wires[0]])
        m1 = ops.pauli_measure("Z" + pauli1.__name__[-1], [work_wires[0], wires[1]])
        m2 = ops.pauli_measure("X", work_wires)
        ops.cond(m1, pauli0)(wires[0])
        ops.cond(m0 != m2, pauli1)(wires[1])
        ops.cond(m1 & (m0 != m2), ops.GlobalPhase)(np.pi)
        ops.cond(m2, ops.Z)(work_wires[0])  # Reset work wire (to |+>), achieving pure state


@register_resources(
    {
        ops.PauliMeasure('ZX', wires=Wire[2]): 1,
        ops.PauliMeasure('ZY', wires=Wire[2]): 1,
        ops.PauliMeasure('X', wires=Wire[1]): 1,
        ops.Z: 2,
        ops.Y: 1,
        ops.GlobalPhase: 1,
    },
    work_wires={'burnable': 1},
    name='lattice_surgery_ppm',
)
def cy_lattice_surgery_ppm(wires: AbstractWires):
    """Decompose ``CY`` using the ``lattice_surgery_ppm`` strategy."""
    _pauli_ctrl_pauli_ppm(wires, ops.Z, ops.Y)


add_decomps(CY, cy_to_ry_cnot, cy_to_ppr, cy_lattice_surgery_ppm)
add_decomps("Adjoint(CY)", self_adjoint)
add_decomps("Pow(CY)", pow_involutory)


@register_resources({ControlledPhaseShift: 1}, name="to_cps")
def cz_to_cps(wires: WiresLike):
    """Decompose ``CZ`` using the ``to_cps`` strategy."""
    ControlledPhaseShift(np.pi, wires=wires)


@register_resources({ops.H: 2, CNOT: 1}, name="to_cnot")
def cz_to_cnot(wires: WiresLike):
    """Decompose ``CZ`` using the ``to_cnot`` strategy."""
    ops.H(wires=wires[1])
    CNOT(wires=wires)
    ops.H(wires=wires[1])


@register_resources(partial(_pauli_ctrl_pauli_ppr_resources, pauli0=ops.Z, pauli1=ops.Z), name="to_ppr")
def cz_to_ppr(wires: AbstractWires):
    """Decompose ``CZ`` using the ``to_ppr`` strategy."""
    _pauli_ctrl_pauli_ppr(wires, ops.Z, ops.Z)


@register_resources(
    {
        ops.PauliMeasure('ZX', wires=Wire[2]): 1,
        ops.PauliMeasure('ZZ', wires=Wire[2]): 1,
        ops.PauliMeasure('X', wires=Wire[1]): 1,
        ops.Z: 3,
        ops.GlobalPhase: 1,
    },
    work_wires={'burnable': 1},
    name='lattice_surgery_ppm',
)
def cz_lattice_surgery_ppm(wires: WiresLike):
    """Decompose ``CZ`` using the ``lattice_surgery_ppm`` strategy."""
    _pauli_ctrl_pauli_ppm(wires, ops.Z, ops.Z)


add_decomps(CZ, cz_to_cps, cz_to_cnot, cz_to_ppr, cz_lattice_surgery_ppm)
add_decomps("Adjoint(CZ)", self_adjoint)
add_decomps("Pow(CZ)", pow_involutory)


@register_resources({CNOT: 2, Toffoli: 1}, name="to_cnot_toffoli")
def cswap_to_cnot_toffoli(wires: WiresLike, **__):
    """Decompose ``CSWAP`` using the ``to_cnot_toffoli`` strategy."""
    CNOT([wires[2], wires[1]])
    Toffoli(wires=[wires[0], wires[1], wires[2]])
    CNOT([wires[2], wires[1]])


@register_resources(
    {
        ops.PPR_8(-1, pauli_word='ZZZ', wires=Wire[3]): 1,
        ops.PPR_8(-1, pauli_word='ZYY', wires=Wire[3]): 1,
        ops.PPR_8(-1, pauli_word='ZXX', wires=Wire[3]): 1,
        ops.PPR_8(1, pauli_word='ZZ', wires=Wire[2]): 1,
        ops.PPR_8(1, pauli_word='YY', wires=Wire[2]): 1,
        ops.PPR_8(1, pauli_word='XX', wires=Wire[2]): 1,
        ops.PPR_8(1, pauli_word='Z', wires=Wire[1]): 1,
        ops.GlobalPhase: 1,
    },
    name='to_ppr',
)
def cswap_to_ppr(wires: WiresLike, **_):
    """Decompose ``CSWAP`` using the ``to_ppr`` strategy."""
    ops.PPR_8(-1, "ZZZ", wires=wires)
    ops.PPR_8(-1, "ZYY", wires=wires)
    ops.PPR_8(-1, "ZXX", wires=wires)
    ops.PPR_8(1, "ZZ", wires=wires[1:])
    ops.PPR_8(1, "YY", wires=wires[1:])
    ops.PPR_8(1, "XX", wires=wires[1:])
    ops.PPR_8(1, "Z", wires=wires[0])
    ops.GlobalPhase(-np.pi / 8)


add_decomps(CSWAP, cswap_to_cnot_toffoli, cswap_to_ppr)
add_decomps("Adjoint(CSWAP)", self_adjoint)
add_decomps("Pow(CSWAP)", pow_involutory)


@register_resources(
    {
        CNOT: 6,
        ops.adjoint(ops.T(Wire[1])): 3,
        ops.T: 4,
        ops.Hadamard: 2,
    },
    name='to_clifford_t',
)
def ccz_to_clifford_t(wires: WiresLike, **__):
    """Decompose ``CCZ`` using the ``to_clifford_t`` strategy."""
    CNOT(wires=[wires[1], wires[2]])
    ops.adjoint(ops.T(wires=wires[2]))
    CNOT(wires=[wires[0], wires[2]])
    ops.T(wires=wires[2])
    CNOT(wires=[wires[1], wires[2]])
    ops.adjoint(ops.T(wires=wires[2]))
    CNOT(wires=[wires[0], wires[2]])
    ops.T(wires=wires[2])
    ops.T(wires=wires[1])
    CNOT(wires=[wires[0], wires[1]])
    ops.Hadamard(wires=wires[2])
    ops.T(wires=wires[0])
    ops.adjoint(ops.T(wires=wires[1]))
    CNOT(wires=[wires[0], wires[1]])
    ops.Hadamard(wires=wires[2])


@register_resources({ops.Hadamard: 2, Toffoli: 1}, name="to_toffoli")
def ccz_to_toffoli(wires: WiresLike, **__):
    """Decompose ``CCZ`` using the ``to_toffoli`` strategy."""
    ops.Hadamard(wires[2])
    Toffoli(wires)
    ops.Hadamard(wires[2])


add_decomps(CCZ, ccz_to_clifford_t, ccz_to_toffoli)
add_decomps("Adjoint(CCZ)", self_adjoint)
add_decomps("Pow(CCZ)", pow_involutory)


@register_resources({ops.H: 2, CZ: 1}, name="to_cz_h")
def cnot_to_cz_h(wires: WiresLike):
    """Decompose ``CNOT`` using the ``to_cz_h`` strategy."""
    ops.H(wires[1])
    CZ(wires=wires)
    ops.H(wires[1])


@register_resources(partial(_pauli_ctrl_pauli_ppr_resources, pauli0=ops.Z, pauli1=ops.X), name="to_ppr")
def cnot_to_ppr(wires: AbstractWires):
    """Decompose ``CNOT`` using the ``to_ppr`` strategy."""
    _pauli_ctrl_pauli_ppr(wires, ops.Z, ops.X)


@register_resources(
    {
        ops.PauliMeasure('ZX', wires=Wire[2]): 2,
        ops.PauliMeasure('X', wires=Wire[1]): 1,
        ops.Z: 2,
        ops.X: 1,
        ops.GlobalPhase: 1,
    },
    work_wires={'burnable': 1},
    name='lattice_surgery_ppm',
)
def cnot_lattice_surgery_ppm(wires: WiresLike):
    """Decompose ``CNOT`` using the ``lattice_surgery_ppm`` strategy."""
    _pauli_ctrl_pauli_ppm(wires, ops.Z, ops.X)


add_decomps(CNOT, cnot_to_cz_h, cnot_to_ppr, cnot_lattice_surgery_ppm)
add_decomps("Adjoint(CNOT)", self_adjoint)
add_decomps("Pow(CNOT)", pow_involutory)


# pylint: disable-next=unused-argument
def _ctrl_cnot_to_mcx_resources(base, control_wires, control_values, work_wires, work_wire_type):
    ctrl_values = _resolve_ctrl_values(control_values, [True], len(control_wires))
    return {
        ops.MultiControlledX(
            Wire[len(control_wires) + 2],
            control_values=ctrl_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@register_resources(_ctrl_cnot_to_mcx_resources, name="to_mcx")
def ctrl_cnot_to_mcx(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose ``C(CNOT)`` using the ``to_mcx`` strategy."""
    ctrl_values = _resolve_ctrl_values(control_values, [True], len(control_wires))
    ops.MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        control_values=ctrl_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(CNOT)", ctrl_cnot_to_mcx)


@register_resources(
    {
        ops.Hadamard: 2,
        CNOT: 6,
        ops.T: 4,
        ops.adjoint(ops.T(Wire[1])): 3,
    },
    name='to_clifford_t',
)
def toffoli_to_clifford_t(wires: WiresLike):
    """Decompose ``Toffoli`` using the ``to_clifford_t`` strategy."""
    ops.Hadamard(wires=wires[2])
    CNOT(wires=[wires[1], wires[2]])
    ops.adjoint(ops.T(wires=wires[2]))
    CNOT(wires=[wires[0], wires[2]])
    ops.T(wires=wires[2])
    CNOT(wires=[wires[1], wires[2]])
    ops.adjoint(ops.T(wires=wires[2]))
    CNOT(wires=[wires[0], wires[2]])
    ops.T(wires=wires[2])
    ops.T(wires=wires[1])
    CNOT(wires=[wires[0], wires[1]])
    ops.Hadamard(wires=wires[2])
    ops.T(wires=wires[0])
    ops.adjoint(ops.T(wires=wires[1]))
    CNOT(wires=[wires[0], wires[1]])


@register_resources(
    {
        ops.PPR_8(-1, pauli_word='ZZ', wires=Wire[2]): 1,
        ops.PPR_8(-1, pauli_word='ZX', wires=Wire[2]): 2,
        ops.PPR_8(1, pauli_word='ZZX', wires=Wire[3]): 1,
        ops.PPR_8(1, pauli_word='X', wires=Wire[1]): 1,
        ops.PPR_8(1, pauli_word='Z', wires=Wire[1]): 2,
        ops.GlobalPhase: 1,
    },
    name='to_ppr',
)
def toffoli_to_ppr(wires: WiresLike):
    """Decompose ``Toffoli`` using the ``to_ppr`` strategy."""
    ops.PPR_8(-1, "ZZ", wires=wires[:2])
    ops.PPR_8(-1, "ZX", wires=[wires[0], wires[2]])
    ops.PPR_8(-1, "ZX", wires=wires[1:])
    ops.PPR_8(1, "ZZX", wires=wires)
    ops.PPR_8(1, "X", wires=wires[2])
    ops.PPR_8(1, "Z", wires=wires[1])
    ops.PPR_8(1, "Z", wires=wires[0])
    ops.GlobalPhase(-np.pi / 8)


@register_resources(
    {
        _change_op_basis_abstract(templates.Elbow, CNOT, ops.adjoint(templates.Elbow(Wire[3]))): 1,
    },
    work_wires={'zeroed': 1},
    name='elbow',
)
def toffoli_elbow(wires: WiresLike):
    """Decompose ``Toffoli`` using the ``elbow`` strategy."""
    with allocation.allocate(1, allocation.AllocateState.ZERO, restored=True) as work_wires:
        ops.change_op_basis(
            templates.Elbow([wires[0], wires[1], work_wires[0]]),
            CNOT([work_wires[0], wires[2]]),
        )


add_decomps(Toffoli, toffoli_to_clifford_t, toffoli_to_ppr, toffoli_elbow)
add_decomps("Adjoint(Toffoli)", self_adjoint)
add_decomps("Pow(Toffoli)", pow_involutory)


# pylint: disable-next=unused-argument
def _ctrl_toffoli_to_mcx_resources(base, control_wires, control_values, work_wires, work_wire_type):
    ctrl_values = _resolve_ctrl_values(control_values, [True, True], len(control_wires))
    return {
        ops.MultiControlledX(
            Wire[len(control_wires) + 3],
            control_values=ctrl_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@register_resources(_ctrl_toffoli_to_mcx_resources, name="to_mcx")
def ctrl_toffoli_to_mcx(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose ``C(Toffoli)`` using the ``to_mcx`` strategy."""
    ctrl_values = _resolve_ctrl_values(control_values, [True, True], len(control_wires))
    ops.MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        control_values=ctrl_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(Toffoli)", ctrl_toffoli_to_mcx)


@register_resources({ops.RZ: 2, ops.RY: 2, CNOT: 2}, name="to_rz_ry")
def crx_to_rz_ry(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRX`` using the ``to_rz_ry`` strategy."""
    ops.RZ(np.pi / 2, wires=wires[1])
    ops.RY(phi / 2, wires=wires[1])
    CNOT(wires=wires)
    ops.RY(-phi / 2, wires=wires[1])
    CNOT(wires=wires)
    ops.RZ(-np.pi / 2, wires=wires[1])


@register_resources({ops.RX: 2, CZ: 2}, name="to_rx_cz")
def crx_to_rx_cz(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRX`` using the ``to_rx_cz`` strategy."""
    ops.RX(phi / 2, wires=wires[1])
    CZ(wires=wires)
    ops.RX(-phi / 2, wires=wires[1])
    CZ(wires=wires)


@register_resources({ops.Hadamard: 2, CRZ: 1}, name="to_h_crz")
def crx_to_h_crz(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRX`` using the ``to_h_crz`` strategy."""
    ops.Hadamard(wires=wires[1])
    CRZ(phi, wires=wires)
    ops.Hadamard(wires=wires[1])


@register_resources(
    {
        ops.PauliRot(Float, pauli_word='ZX', wires=Wire[2]): 1,
        ops.PauliRot(Float, pauli_word='X', wires=Wire[1]): 1,
    },
    name='to_paulirot',
)
def crx_to_ppr(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRX`` using the ``to_paulirot`` strategy."""
    ops.PauliRot(phi / 2, "X", wires=wires[1])
    ops.PauliRot(-phi / 2, "ZX", wires=wires)


add_decomps(CRX, crx_to_rx_cz, crx_to_rz_ry, crx_to_h_crz, crx_to_ppr)
add_decomps("Adjoint(CRX)", _adjoint_negate_angle)
add_decomps("Pow(CRX)", _pow_scale_angle)


@register_resources({ops.RY: 2, CNOT: 2}, name="to_ry_cnot")
def cry_to_ry_cnot(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRY`` using the ``to_ry_cnot`` strategy."""
    ops.RY(phi / 2, wires=wires[1])
    CNOT(wires=wires)
    ops.RY(-phi / 2, wires=wires[1])
    CNOT(wires=wires)


@register_resources(
    {
        ops.PauliRot(Float, pauli_word='ZY', wires=Wire[2]): 1,
        ops.PauliRot(Float, pauli_word='Y', wires=Wire[1]): 1,
    },
    name='to_paulirot',
)
def cry_to_ppr(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRY`` using the ``to_paulirot`` strategy."""
    ops.PauliRot(phi / 2, "Y", wires=wires[1])
    ops.PauliRot(-phi / 2, "ZY", wires=wires)


add_decomps(CRY, cry_to_ry_cnot, cry_to_ppr)
add_decomps("Adjoint(CRY)", _adjoint_negate_angle)
add_decomps("Pow(CRY)", _pow_scale_angle)


@register_resources({ops.RZ: 2, CNOT: 2}, name="to_rz_cnot")
def crz_to_rz_cnot(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRZ`` using the ``to_rz_cnot`` strategy."""
    ops.RZ(phi / 2, wires=wires[1])
    CNOT(wires=wires)
    ops.RZ(-phi / 2, wires=wires[1])
    CNOT(wires=wires)


@register_resources(
    {
        ops.PauliRot(Float, pauli_word='ZZ', wires=Wire[2]): 1,
        ops.PauliRot(Float, pauli_word='Z', wires=Wire[1]): 1,
    },
    name='to_paulirot',
)
def crz_to_ppr(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRZ`` using the ``to_paulirot`` strategy."""
    ops.PauliRot(phi / 2, "Z", wires=wires[1])
    ops.PauliRot(-phi / 2, "ZZ", wires=wires)


add_decomps(CRZ, crz_to_rz_cnot, crz_to_ppr)
add_decomps("Adjoint(CRZ)", _adjoint_negate_angle)
add_decomps("Pow(CRZ)", _pow_scale_angle)


@register_resources({ops.RZ: 3, CNOT: 2, ops.RY: 2}, name="to_rot_cnot")
def crot_to_rot_cnot(phi: TensorLike, theta: TensorLike, omega: TensorLike, wires: WiresLike):
    """Decompose ``CRot`` using the ``to_rot_cnot`` strategy."""
    ops.RZ((phi - omega) / 2, wires=wires[1])
    CNOT(wires=wires)
    ops.RZ(-(phi + omega) / 2, wires=wires[1])
    ops.RY(-theta / 2, wires=wires[1])
    CNOT(wires=wires)
    ops.RY(theta / 2, wires=wires[1])
    ops.RZ(omega, wires=wires[1])


add_decomps(CRot, crot_to_rot_cnot)


@register_resources({CRot: 1}, name="reverse_angles")
def adjoint_crot_standard(base):
    """Decompose ``Adjoint(CRot)`` using the ``reverse_angles`` strategy."""
    CRot(-base.omega, -base.theta, -base.phi, wires=base.wires)


add_decomps("Adjoint(CRot)", adjoint_crot_standard)


@register_resources({ops.RZ: 3, CNOT: 2, ops.GlobalPhase: 1}, name="to_rz_cnot")
def cphase_to_rz_cnot(phi: TensorLike, wires: WiresLike, **__):
    """Decompose ``ControlledPhaseShift`` using the ``to_rz_cnot`` strategy."""
    ops.RZ(phi / 2, wires=wires[0])
    CNOT(wires=wires)
    ops.RZ(-phi / 2, wires=wires[1])
    CNOT(wires=wires)
    ops.RZ(phi / 2, wires=wires[1])
    ops.GlobalPhase(-phi / 4)


@register_resources(
    {
        ops.GlobalPhase: 1,
        ops.PauliRot(Float, pauli_word='Z', wires=Wire[1]): 2,
        ops.PauliRot(Float, pauli_word='ZZ', wires=Wire[2]): 1,
    },
    name='to_paulirot',
)
def cphase_to_ppr(phi: TensorLike, wires: WiresLike, **__):
    """Decompose ``ControlledPhaseShift`` using the ``to_paulirot`` strategy."""
    ops.PauliRot(-phi / 2, pauli_word="ZZ", wires=wires)
    ops.PauliRot(phi / 2, pauli_word="Z", wires=wires[1])
    ops.PauliRot(phi / 2, pauli_word="Z", wires=wires[0])
    ops.GlobalPhase(-phi / 4)


# pylint: disable-next=unused-argument
def _cphase_to_single_phase_resources(phi, wires):
    elbow = templates.TemporaryAND(wires=Wire[3])
    return {_change_op_basis_abstract(elbow, ops.PhaseShift, ops.adjoint(elbow)): 1}


@register_resources(
    _cphase_to_single_phase_resources, work_wires={"zeroed": 1}, name="to_single_phase"
)
def cphase_to_single_phase(phi: TensorLike, wires: WiresLike, **__):
    """Decompose ``ControlledPhaseShift`` using the ``to_single_phase`` strategy."""
    with allocation.allocate(1, state="zero", restored=True) as work_wire:
        elbow = templates.TemporaryAND([wires[0], wires[1], work_wire[0]])
        ops.change_op_basis(elbow, ops.PhaseShift(phi, work_wire))


add_decomps(ControlledPhaseShift, cphase_to_rz_cnot, cphase_to_ppr, cphase_to_single_phase)
add_decomps("Adjoint(ControlledPhaseShift)", _adjoint_negate_angle)
add_decomps("Pow(ControlledPhaseShift)", _pow_scale_angle)
