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

"""Built-in decomposition rules for qubit.parametric_ops_single_qubit."""

# pylint: disable=too-many-arguments

# pylint: disable=arguments-differ

import math as builtin_math
from itertools import combinations

import numpy as np

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition_rules.op_math.adjoint2 import adjoint_rotation as adjoint_rotation2
from pennylane.decomposition_rules.op_math.controlled2 import (
    flip_zero_control as flip_zero_control2,
)
from pennylane.decomposition_rules.op_math.pow2 import pow_rotation as pow_rotation2
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.op_math.prod import prod
from pennylane.ops.qubit.parametric_ops_single_qubit import RX, RY, RZ, U1, U2, U3, PhaseShift, Rot
from pennylane.typing import Float, Wire
from pennylane.wires import WiresLike, concatenate_wires


def _rx_to_rot_resources(*_, **__):
    return {qp.Rot: 1}


@register_resources(_rx_to_rot_resources)
def _rx_to_rot(phi, wires: WiresLike):
    qp.Rot(np.pi / 2, phi, 3.5 * np.pi, wires=wires)


def _rx_to_rz_ry_resources(*_, **__):
    return {qp.RZ: 2, qp.RY: 1}


@register_resources(_rx_to_rz_ry_resources)
def _rx_to_rz_ry(phi, wires: WiresLike):
    qp.RZ(np.pi / 2, wires=wires)
    qp.RY(phi, wires=wires)
    qp.RZ(-np.pi / 2, wires=wires)


def _rx_to_ry_cliff_resources(*_, **__):
    resources = {_change_op_basis_abstract(qp.S, qp.RY, qp.adjoint(qp.S(Wire[1]))): 1}
    return resources


@register_resources(_rx_to_ry_cliff_resources)
def _rx_to_ry_cliff(phi, wires: WiresLike):
    qp.change_op_basis(qp.S(wires), qp.RY(phi, wires))


def _rx_to_rz_cliff_resources(*_, **__):
    resources = {_change_op_basis_abstract(qp.Hadamard, qp.RZ, qp.Hadamard): 1}
    return resources


@register_resources(_rx_to_rz_cliff_resources)
def _rx_to_rz_cliff(phi, wires: WiresLike):
    qp.change_op_basis(qp.Hadamard(wires), qp.RZ(phi, wires), qp.Hadamard(wires))


def _rx_to_ppr_resources(*_, **__):
    return {qp.PauliRot(Float, pauli_word="X", wires=Wire[1]): 1}


@register_resources(_rx_to_ppr_resources)
def _rx_to_ppr(phi, wires):
    qp.PauliRot(phi, "X", wires=wires)


add_decomps(RX, _rx_to_rot, _rx_to_rz_ry, _rx_to_ppr, _rx_to_ry_cliff, _rx_to_rz_cliff)

add_decomps("Adjoint(RX)", adjoint_rotation2)

add_decomps("Pow(RX)", pow_rotation2)


# pylint: disable-next=unused-argument
def _controlled_rx_resource(base, control_wires, control_values, work_wires, work_wire_type):
    if len(control_wires) == 1:
        return {qp.CRX: 1}
    return {
        qp.H: 2,
        qp.RZ: 2,
        qp.MultiControlledX(
            Wire[len(control_wires) + 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 2,
    }


@register_resources(_controlled_rx_resource)
# pylint: disable-next=unused-argument
def _controlled_rx_decomp(base, control_wires, control_values, work_wires, work_wire_type):
    wires = concatenate_wires(control_wires, base.wires)
    if len(control_wires) == 1:
        qp.CRX(base.phi, wires=wires)
        return

    qp.H(wires=wires[-1])
    qp.RZ(base.phi / 2, wires=wires[-1])
    qp.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)
    qp.RZ(-base.phi / 2, wires=wires[-1])
    qp.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)
    qp.H(wires=wires[-1])


add_decomps("C(RX)", flip_zero_control2(_controlled_rx_decomp))


def _ry_to_rot_resources(*_, **__):
    return {qp.Rot: 1}


@register_resources(_ry_to_rot_resources)
def _ry_to_rot(phi, wires: WiresLike):
    qp.Rot(0, phi, 0, wires=wires)


def _ry_to_rz_rx_resources(*_, **__):
    return {qp.RZ: 2, qp.RX: 1}


@register_resources(_ry_to_rz_rx_resources)
def _ry_to_rz_rx(phi, wires: WiresLike):
    qp.RZ(-np.pi / 2, wires=wires)
    qp.RX(phi, wires=wires)
    qp.RZ(np.pi / 2, wires=wires)


def _ry_to_rx_cliff_resources(*_, **__):
    resources = {_change_op_basis_abstract(_adjoint_abstract(qp.S), qp.RX, qp.S): 1}
    return resources


@register_resources(_ry_to_rx_cliff_resources)
def _ry_to_rx_cliff(phi, wires: WiresLike):
    qp.change_op_basis(qp.adjoint(qp.S(wires)), qp.RX(phi, wires), qp.S(wires))


def _ry_to_rz_cliff_resources(*_, **__):
    resources = {
        _change_op_basis_abstract(
            prod(qp.H(Wire[1]), qp.adjoint(qp.S(Wire[1]))),
            qp.RZ,
            prod(qp.S(Wire[1]), qp.H(Wire[1])),
        ): 1
    }
    return resources


@register_resources(_ry_to_rz_cliff_resources)
def _ry_to_rz_cliff(phi, wires: WiresLike, **__):

    def _compute():
        qp.adjoint(qp.S(wires))
        qp.Hadamard(wires)

    def _uncompute():
        qp.Hadamard(wires)
        qp.S(wires)

    qp.change_op_basis(_compute, qp.RZ(phi, wires), _uncompute)


def _ry_to_ppr_resources(*_, **__):
    return {qp.PauliRot(Float, pauli_word="Y", wires=Wire[1]): 1}


@register_resources(_ry_to_ppr_resources)
def _ry_to_ppr(phi, wires):
    qp.PauliRot(phi, "Y", wires=wires)


add_decomps(RY, _ry_to_rot, _ry_to_rz_rx, _ry_to_ppr, _ry_to_rx_cliff, _ry_to_rz_cliff)

add_decomps("Adjoint(RY)", adjoint_rotation2)

add_decomps("Pow(RY)", pow_rotation2)


# pylint: disable-next=unused-argument
def _controlled_ry_resource(base, control_wires, control_values, work_wires, work_wire_type):
    if len(control_wires) == 1:
        return {qp.CRY: 1}
    return {
        qp.RY: 2,
        qp.MultiControlledX(
            Wire[len(control_wires) + 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 2,
    }


@register_resources(_controlled_ry_resource)
# pylint: disable-next=unused-argument
def _controlled_ry_decomp(base, control_wires, control_values, work_wires, work_wire_type):
    wires = concatenate_wires(control_wires, base.wires)
    if len(control_wires) == 1:
        qp.CRY(base.phi, wires=wires)
        return

    qp.RY(base.phi / 2, wires=wires[-1])
    qp.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)
    qp.RY(-base.phi / 2, wires=wires[-1])
    qp.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)


add_decomps("C(RY)", flip_zero_control2(_controlled_ry_decomp))


# pylint: disable=unused-argument
def _rz_to_ps_resources(phi, wires):
    return {qp.PhaseShift: 1, qp.GlobalPhase: 1}


@register_resources(_rz_to_ps_resources)
def _rz_to_ps(phi, wires: WiresLike):
    qp.PhaseShift(phi, wires)
    qp.GlobalPhase(phi / 2)


# pylint: disable=unused-argument
def _rz_to_rot_resources(phi, wires):
    return {qp.Rot: 1}


@register_resources(_rz_to_rot_resources)
def _rz_to_rot(phi, wires: WiresLike):
    qp.Rot(0, 0, phi, wires=wires)


# pylint: disable=unused-argument
def _rz_to_ry_rx_resources(phi, wires):
    return {qp.RY: 2, qp.RX: 1}


@register_resources(_rz_to_ry_rx_resources)
def _rz_to_ry_rx(phi, wires: WiresLike):
    qp.RY(np.pi / 2, wires=wires)
    qp.RX(phi, wires=wires)
    qp.RY(-np.pi / 2, wires=wires)


# pylint: disable=unused-argument
def _rz_to_rx_cliff_resources(phi, wires):
    resources = {_change_op_basis_abstract(qp.Hadamard, qp.RX, qp.Hadamard): 1}
    return resources


@register_resources(_rz_to_rx_cliff_resources)
def _rz_to_rx_cliff(phi, wires: WiresLike):
    qp.change_op_basis(qp.Hadamard(wires), qp.RX(phi, wires), qp.Hadamard(wires))


# pylint: disable=unused-argument
def _rz_to_ry_cliff_resources(phi, wires):
    resources = {
        _change_op_basis_abstract(
            prod(qp.S(Wire[1]), qp.H(Wire[1])),
            qp.RY,
            prod(qp.H(Wire[1]), qp.adjoint(qp.S(Wire[1]))),
        ): 1
    }
    return resources


@register_resources(_rz_to_ry_cliff_resources)
def _rz_to_ry_cliff(phi, wires: WiresLike):

    def _compute():
        qp.H(wires)
        qp.S(wires)

    def _uncompute():
        qp.adjoint(qp.S(wires))
        qp.H(wires)

    qp.change_op_basis(_compute, qp.RY(phi, wires), _uncompute)


def _rz_to_ppr_resources(phi, wires):
    return {qp.PauliRot(Float, pauli_word="Z", wires=Wire[1]): 1}


@register_resources(_rz_to_ppr_resources)
def _rz_to_ppr(phi, wires, **_):
    qp.PauliRot(phi, "Z", wires=wires)


add_decomps(RZ, _rz_to_ps, _rz_to_rot, _rz_to_ry_rx, _rz_to_ppr, _rz_to_rx_cliff, _rz_to_ry_cliff)

add_decomps("Adjoint(RZ)", adjoint_rotation2)

add_decomps("Pow(RZ)", pow_rotation2)


def _controlled_rz_resource(base, control_wires, control_values, work_wires, work_wire_type):
    if len(control_wires) == 1:
        return {qp.CRZ: 1}
    return {
        qp.RZ: 2,
        qp.MultiControlledX(
            Wire[len(control_wires) + 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 2,
    }


@register_resources(_controlled_rz_resource)
def _controlled_rz_decomp(base, control_wires, control_values, work_wires, work_wire_type):

    if len(control_wires) == 1:
        qp.CRZ(base.phi, wires=concatenate_wires(control_wires, base.wires))
        return

    qp.RZ(base.phi / 2, wires=base.wires)
    qp.MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    qp.RZ(-base.phi / 2, wires=base.wires)
    qp.MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(RZ)", flip_zero_control2(_controlled_rz_decomp))


def _phaseshift_to_rz_gp_resources(phi, wires):  # pylint: disable=unused-argument
    return {qp.RZ: 1, qp.GlobalPhase: 1}


@register_resources(_phaseshift_to_rz_gp_resources)
def _phaseshift_to_rz_gp(phi, wires: WiresLike, **__):
    RZ(phi, wires=wires)
    qp.GlobalPhase(-phi / 2)


# pylint: disable-next=unused-argument
def _controlled_phase_shift_resource(base, control_wires, *_, **__):
    # A single control wire specializes to ``ControlledPhaseShift``, matching the legacy
    # behaviour (and its cheaper ``RZ``/``CNOT`` decomposition) rather than the general
    # phase-polynomial (``PauliRot``) expansion, which is only used for multiple control wires.
    if len(control_wires) == 1:
        return {qp.ControlledPhaseShift: 1}
    num_control_wires = len(control_wires)
    resources = {
        qp.PauliRot(Float, pauli_word="Z" * i, wires=Wire[len("Z" * i)]): builtin_math.comb(
            num_control_wires + 1, i
        )
        for i in range(1, num_control_wires + 2)
    }
    resources[qp.GlobalPhase] = 1
    return resources


@register_resources(_controlled_phase_shift_resource)
def _controlled_phase_shift_decomp(base, control_wires, *_, **__):
    wires = concatenate_wires(control_wires, base.wires)
    if len(control_wires) == 1:
        qp.ControlledPhaseShift(base.phi, wires=wires)
        return

    n = len(wires)
    for l in range(1, n + 1):
        for sub_wires in combinations(wires, l):
            phi = -base.phi / 2 ** (n - 1) * (-1) ** l
            qp.PauliRot(phi, pauli_word="Z" * l, wires=sub_wires)
    qp.GlobalPhase(-base.phi / 2**n)


add_decomps(PhaseShift, _phaseshift_to_rz_gp)

add_decomps("Adjoint(PhaseShift)", adjoint_rotation2)

add_decomps("Pow(PhaseShift)", pow_rotation2)

add_decomps("C(PhaseShift)", flip_zero_control2(_controlled_phase_shift_decomp))


# pylint: disable-next=unused-argument
def _rot_to_rz_ry_rz_resources(phi, theta, omega, wires):
    return {qp.RZ: 2, qp.RY: 1}


@register_resources(_rot_to_rz_ry_rz_resources)
def _rot_to_rz_ry_rz(phi, theta, omega, wires: WiresLike):
    RZ(phi, wires=wires)
    RY(theta, wires=wires)
    RZ(omega, wires=wires)


add_decomps(Rot, _rot_to_rz_ry_rz)


@register_resources({Rot: 1})
def _adjoint_rot(base):
    Rot(-base.omega, -base.theta, -base.phi, wires=base.wires)


add_decomps("Adjoint(Rot)", _adjoint_rot)


def _controlled_rot_resource(base, control_wires, control_values, work_wires, work_wire_type):
    if len(control_wires) == 1:
        return {qp.CRot: 1}
    return {
        qp.RZ: 3,
        qp.RY: 2,
        qp.MultiControlledX(
            Wire[len(control_wires) + 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 2,
    }


@register_resources(_controlled_rot_resource)
def _controlled_rot_decomp(base, control_wires, control_values, work_wires, work_wire_type):

    phi, theta, omega = base.phi, base.theta, base.omega
    wires = concatenate_wires(control_wires, base.wires)

    if len(control_wires) == 1:
        qp.CRot(phi, theta, omega, wires=wires)
        return

    qp.RZ((phi - omega) / 2, wires=wires[-1])
    qp.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)
    qp.RZ(-(phi + omega) / 2, wires=wires[-1])
    qp.RY(-theta / 2, wires=wires[-1])
    qp.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)
    qp.RY(theta / 2, wires=wires[-1])
    qp.RZ(omega, wires=wires[-1])


add_decomps("C(Rot)", flip_zero_control2(_controlled_rot_decomp))


# pylint: disable=unused-argument
def _u1_phaseshift_resources(*_, **__):
    return {PhaseShift: 1}


@register_resources(_u1_phaseshift_resources)
def _u1_phaseshift(phi, wires):
    PhaseShift(phi, wires=wires)


add_decomps(U1, _u1_phaseshift)

add_decomps("Adjoint(U1)", adjoint_rotation2)

add_decomps("Pow(U1)", pow_rotation2)


# pylint: disable=unused-argument
def _u2_phaseshift_rot_resources(phi, delta, wires):
    return {PhaseShift: 2, Rot: 1}


@register_resources(_u2_phaseshift_rot_resources)
def _u2_phaseshift_rot(phi, delta, wires):
    pi_half = qp.math.ones_like(delta) * (np.pi / 2)
    Rot(delta, pi_half, -delta, wires=wires)
    PhaseShift(delta, wires=wires)
    PhaseShift(phi, wires=wires)


add_decomps(U2, _u2_phaseshift_rot)


@register_resources(lambda base: {abstractify(U2): 1})
def _adjoint_u2(base):
    new_delta = qp.math.mod((np.pi - base.phi), (2 * np.pi))
    new_phi = qp.math.mod((np.pi - base.delta), (2 * np.pi))
    U2(new_phi, new_delta, wires=base.wires)


add_decomps("Adjoint(U2)", _adjoint_u2)


# pylint: disable=unused-argument
def _u3_phaseshift_rot_resources(theta, phi, delta, wires):
    return {PhaseShift: 2, Rot: 1}


@register_resources(_u3_phaseshift_rot_resources)
def _u3_phaseshift_rot(theta, phi, delta, wires):
    Rot(delta, theta, -delta, wires=wires)
    PhaseShift(delta, wires=wires)
    PhaseShift(phi, wires=wires)


add_decomps(U3, _u3_phaseshift_rot)


@register_resources(lambda base: {abstractify(U3): 1})
def _adjoint_u3(base):
    new_delta = qp.math.mod((np.pi - base.phi), (2 * np.pi))
    new_phi = qp.math.mod((np.pi - base.delta), (2 * np.pi))
    U3(base.theta, new_phi, new_delta, wires=base.wires)


add_decomps("Adjoint(U3)", _adjoint_u3)
