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

import math as builtin_math
from collections import Counter
from functools import partial
from itertools import combinations

import numpy as np

import pennylane as qp
from pennylane import math
from pennylane.core.operator import Operator, abstractify
from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
)
from pennylane.decomposition_rules.op_math.adjoint2 import (
    _adjoint_negate_angle as adjoint_rotation2,
)
from pennylane.decomposition_rules.op_math.controlled2 import (
    flip_zero_control as flip_zero_control2,
)
from pennylane.decomposition_rules.op_math.pow2 import _pow_scale_angle as pow_rotation2
from pennylane.ops.op_math import change_op_basis
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import ChangeOpBasis2, _change_op_basis_abstract
from pennylane.ops.qubit.parametric_ops_single_qubit import RX, RY, RZ, U1, U2, U3, PhaseShift, Rot
from pennylane.ops.subcircuit import subcircuit
from pennylane.typing import Bool, Float, Wire
from pennylane.wires import WireError, Wires, WiresLike, concatenate_wires


def _rx_to_rot_resources(*_, **__):
    return {qp.Rot: 1}


@register_resources(_rx_to_rot_resources, name="to_rot")
def _rx_to_rot(phi, wires: WiresLike):
    qp.Rot(np.pi / 2, phi, 3.5 * np.pi, wires=wires)


def _rx_to_rz_ry_resources(*_, **__):
    return {qp.RZ: 2, qp.RY: 1}


@register_resources(_rx_to_rz_ry_resources, name="to_rz_ry")
def _rx_to_rz_ry(phi, wires: WiresLike):
    qp.RZ(np.pi / 2, wires=wires)
    qp.RY(phi, wires=wires)
    qp.RZ(-np.pi / 2, wires=wires)


def _rx_to_ry_cliff_resources(*_, **__):
    resources = {_change_op_basis_abstract(qp.S, qp.RY, qp.adjoint(qp.S(Wire[1]))): 1}
    return resources


@register_resources(_rx_to_ry_cliff_resources, name="to_ry_cliff")
def _rx_to_ry_cliff(phi, wires: WiresLike):
    qp.change_op_basis(qp.S(wires), qp.RY(phi, wires))


def _rx_to_rz_cliff_resources(*_, **__):
    resources = {_change_op_basis_abstract(qp.Hadamard, qp.RZ, qp.Hadamard): 1}
    return resources


@register_resources(_rx_to_rz_cliff_resources, name="to_rz_cliff")
def _rx_to_rz_cliff(phi, wires: WiresLike):
    qp.change_op_basis(qp.Hadamard(wires), qp.RZ(phi, wires), qp.Hadamard(wires))


def _rx_to_ppr_resources(*_, **__):
    return {qp.PauliRot(Float, pauli_word="X", wires=Wire[1]): 1}


@register_resources(_rx_to_ppr_resources, name="to_paulirot")
def _rx_to_ppr(phi, wires):
    qp.PauliRot(phi, "X", wires=wires)


add_decomps(RX, _rx_to_rot, _rx_to_rz_ry, _rx_to_ppr, _rx_to_ry_cliff, _rx_to_rz_cliff)

add_decomps("Adjoint(RX)", adjoint_rotation2)

add_decomps("Pow(RX)", pow_rotation2)


def _controlled_rx_standard_resources(
    base,  # pylint: disable=unused-argument
    control_wires,
    control_values,  # pylint: disable=unused-argument
    work_wires,
    work_wire_type,
):
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


@register_resources(_controlled_rx_standard_resources, name="to_rz_mcx")
# pylint: disable-next=unused-argument
def _controlled_rx_standard(base, control_wires, control_values, work_wires, work_wire_type):
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


add_decomps("C(RX)", flip_zero_control2(_controlled_rx_standard, name="to_rz_mcx"))


def _ry_to_rot_resources(*_, **__):
    return {qp.Rot: 1}


@register_resources(_ry_to_rot_resources, name="to_rot")
def _ry_to_rot(phi, wires: WiresLike):
    qp.Rot(0, phi, 0, wires=wires)


def _ry_to_rz_rx_resources(*_, **__):
    return {qp.RZ: 2, qp.RX: 1}


@register_resources(_ry_to_rz_rx_resources, name="to_rz_rx")
def _ry_to_rz_rx(phi, wires: WiresLike):
    qp.RZ(-np.pi / 2, wires=wires)
    qp.RX(phi, wires=wires)
    qp.RZ(np.pi / 2, wires=wires)


def _ry_to_rx_cliff_resources(*_, **__):
    resources = {_change_op_basis_abstract(_adjoint_abstract(qp.S), qp.RX, qp.S): 1}
    return resources


@register_resources(_ry_to_rx_cliff_resources, name="to_rx_cliff")
def _ry_to_rx_cliff(phi, wires: WiresLike):
    qp.change_op_basis(qp.adjoint(qp.S(wires)), qp.RX(phi, wires), qp.S(wires))


@subcircuit
@register_resources(lambda **_: {qp.Hadamard: 1, _adjoint_abstract(qp.S): 1})
def _YToZBasis(wires: WiresLike):
    qp.adjoint(qp.S(wires))
    qp.Hadamard(wires)


@subcircuit
@register_resources(lambda **_: {qp.Hadamard: 1, qp.S: 1})
def _ZToYBasis(wires: WiresLike):
    qp.Hadamard(wires)
    qp.S(wires)


def _ry_to_rz_cliff_resources(*_, **__):
    resources = {
        _change_op_basis_abstract(
            _YToZBasis(Wire[1]),
            qp.RZ,
            _ZToYBasis(Wire[1]),
        ): 1
    }
    return resources


@register_resources(_ry_to_rz_cliff_resources, name="to_rz_cliff")
def _ry_to_rz_cliff(phi, wires: WiresLike, **__):
    qp.change_op_basis(
        _YToZBasis(wires),
        qp.RZ(phi, wires),
        _ZToYBasis(wires),
    )


def _ry_to_ppr_resources(*_, **__):
    return {qp.PauliRot(Float, pauli_word="Y", wires=Wire[1]): 1}


@register_resources(_ry_to_ppr_resources, name="to_paulirot")
def _ry_to_ppr(phi, wires):
    qp.PauliRot(phi, "Y", wires=wires)


add_decomps(RY, _ry_to_rot, _ry_to_rz_rx, _ry_to_ppr, _ry_to_rx_cliff, _ry_to_rz_cliff)

add_decomps("Adjoint(RY)", adjoint_rotation2)

add_decomps("Pow(RY)", pow_rotation2)


def _controlled_ry_standard_resources(
    base,  # pylint: disable=unused-argument
    control_wires,
    control_values,  # pylint: disable=unused-argument
    work_wires,
    work_wire_type,
):
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


@register_resources(_controlled_ry_standard_resources, name="to_ry_mcx")
# pylint: disable-next=unused-argument
def _controlled_ry_standard(base, control_wires, control_values, work_wires, work_wire_type):
    wires = concatenate_wires(control_wires, base.wires)
    if len(control_wires) == 1:
        qp.CRY(base.phi, wires=wires)
        return

    qp.RY(base.phi / 2, wires=wires[-1])
    qp.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)
    qp.RY(-base.phi / 2, wires=wires[-1])
    qp.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)


add_decomps("C(RY)", flip_zero_control2(_controlled_ry_standard, name="to_ry_mcx"))


def _rz_to_ps_resources(phi, wires):  # pylint: disable=unused-argument
    return {qp.PhaseShift: 1, qp.GlobalPhase: 1}


@register_resources(_rz_to_ps_resources, name="to_ps")
def _rz_to_ps(phi, wires: WiresLike):
    qp.PhaseShift(phi, wires)
    qp.GlobalPhase(phi / 2)


def _rz_to_rot_resources(phi, wires):  # pylint: disable=unused-argument
    return {qp.Rot: 1}


@register_resources(_rz_to_rot_resources, name="to_rot")
def _rz_to_rot(phi, wires: WiresLike):
    qp.Rot(0, 0, phi, wires=wires)


def _rz_to_ry_rx_resources(phi, wires):  # pylint: disable=unused-argument
    return {qp.RY: 2, qp.RX: 1}


@register_resources(_rz_to_ry_rx_resources, name="to_ry_rx")
def _rz_to_ry_rx(phi, wires: WiresLike):
    qp.RY(np.pi / 2, wires=wires)
    qp.RX(phi, wires=wires)
    qp.RY(-np.pi / 2, wires=wires)


def _rz_to_rx_cliff_resources(phi, wires):  # pylint: disable=unused-argument
    resources = {_change_op_basis_abstract(qp.Hadamard, qp.RX, qp.Hadamard): 1}
    return resources


@register_resources(_rz_to_rx_cliff_resources, name="to_rx_cliff")
def _rz_to_rx_cliff(phi, wires: WiresLike):
    qp.change_op_basis(qp.Hadamard(wires), qp.RX(phi, wires), qp.Hadamard(wires))


def _rz_to_ry_cliff_resources(phi, wires):  # pylint: disable=unused-argument
    resources = {
        _change_op_basis_abstract(
            _ZToYBasis(Wire[1]),
            qp.RY,
            _YToZBasis(Wire[1]),
        ): 1
    }
    return resources


@register_resources(_rz_to_ry_cliff_resources, name="to_ry_cliff")
def _rz_to_ry_cliff(phi, wires: WiresLike):
    qp.change_op_basis(
        _ZToYBasis(wires),
        qp.RY(phi, wires),
        _YToZBasis(wires),
    )


def _rz_to_ppr_resources(phi, wires):  # pylint: disable=unused-argument
    return {qp.PauliRot(Float, pauli_word="Z", wires=Wire[1]): 1}


@register_resources(_rz_to_ppr_resources, name="to_paulirot")
def _rz_to_ppr(phi, wires, **_):
    qp.PauliRot(phi, "Z", wires=wires)


add_decomps(RZ, _rz_to_ps, _rz_to_rot, _rz_to_ry_rx, _rz_to_ppr, _rz_to_rx_cliff, _rz_to_ry_cliff)

add_decomps("Adjoint(RZ)", adjoint_rotation2)

add_decomps("Pow(RZ)", pow_rotation2)


def _controlled_rz_standard_resources(
    base,  # pylint: disable=unused-argument
    control_wires,
    control_values,  # pylint: disable=unused-argument
    work_wires,
    work_wire_type,
):
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


@register_resources(_controlled_rz_standard_resources, name="to_rz_mcx")
# pylint: disable-next=unused-argument
def _controlled_rz_standard(base, control_wires, control_values, work_wires, work_wire_type):

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


add_decomps("C(RZ)", flip_zero_control2(_controlled_rz_standard, name="to_rz_mcx"))


def _phaseshift_to_rz_gp_resources(phi, wires):  # pylint: disable=unused-argument
    return {qp.RZ: 1, qp.GlobalPhase: 1}


@register_resources(_phaseshift_to_rz_gp_resources, name="to_rz_gp")
def _phaseshift_to_rz_gp(phi, wires: WiresLike, **__):
    RZ(phi, wires=wires)
    qp.GlobalPhase(-phi / 2)


# pylint: disable-next=unused-argument
def _controlled_phase_shift_standard_resources(base, control_wires, *_, **__):
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


@register_resources(_controlled_phase_shift_standard_resources, name="to_pauli_rot")
def _controlled_phase_shift_standard(base, control_wires, *_, **__):
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


def _controlled_phase_shift_single_phase_resources(base, control_wires, *_, **__):
    #  pylint: disable=unused-argument
    num_control_wires = len(control_wires)
    return {
        qp.TemporaryAND: num_control_wires,
        PhaseShift: 1,
        qp.adjoint(qp.TemporaryAND(Wire[3])): num_control_wires,
    }


@register_condition(
    lambda control_wires, work_wires, work_wire_type, **_: work_wire_type == "zeroed"
    and len(control_wires) > 0
    and len(work_wires) >= len(control_wires)
)
@register_resources(_controlled_phase_shift_single_phase_resources, name="single_phase")
def _controlled_phase_shift_single_phase(base, control_wires, work_wires, **_):
    """Computes ``AND(*control_wires, base.wires[0])`` into ``len(control_wires)`` zeroed work
    wires with a ``TemporaryAND`` ladder, applies a single ``PhaseShift`` to the last work wire
    and uncomputes the ladder with measurements."""
    and_controls = [*control_wires, base.wires[0]]
    and_targets = list(work_wires[: len(control_wires)])
    and_wires = list(zip([and_controls[0], *and_targets[:-1]], and_controls[1:], and_targets))

    for wires in and_wires:
        qp.TemporaryAND(wires=wires)
    PhaseShift(base.phi, wires=and_targets[-1])
    for wires in reversed(and_wires):
        qp.adjoint(qp.TemporaryAND)(wires=wires)


add_decomps(PhaseShift, _phaseshift_to_rz_gp)

add_decomps("Adjoint(PhaseShift)", adjoint_rotation2)

add_decomps("Pow(PhaseShift)", pow_rotation2)

add_decomps(
    "C(PhaseShift)",
    flip_zero_control2(_controlled_phase_shift_standard, name="to_pauli_rot"),
    flip_zero_control2(_controlled_phase_shift_single_phase, name="single_phase"),
)


def _rot_to_rz_ry_rz_resources(phi, theta, omega, wires):  # pylint: disable=unused-argument
    return {qp.RZ: 2, qp.RY: 1}


@register_resources(_rot_to_rz_ry_rz_resources, name="to_rz_ry_rz")
def _rot_to_rz_ry_rz(phi, theta, omega, wires: WiresLike):
    RZ(phi, wires=wires)
    RY(theta, wires=wires)
    RZ(omega, wires=wires)


add_decomps(Rot, _rot_to_rz_ry_rz)


@register_resources({Rot: 1}, name="reverse_angles")
def _adjoint_rot_standard(base):
    Rot(-base.omega, -base.theta, -base.phi, wires=base.wires)


add_decomps("Adjoint(Rot)", _adjoint_rot_standard)


def _controlled_rot_standard_resources(
    base,  # pylint: disable=unused-argument
    control_wires,
    control_values,  # pylint: disable=unused-argument
    work_wires,
    work_wire_type,
):
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


@register_resources(_controlled_rot_standard_resources, name="to_rz_ry_mcx")
# pylint: disable-next=unused-argument
def _controlled_rot_standard(base, control_wires, control_values, work_wires, work_wire_type):

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


add_decomps("C(Rot)", flip_zero_control2(_controlled_rot_standard, name="to_rz_ry_mcx"))


def _u1_to_phaseshift_resources(*_, **__):
    return {PhaseShift: 1}


@register_resources(_u1_to_phaseshift_resources, name="to_phaseshift")
def _u1_to_phaseshift(phi, wires):
    PhaseShift(phi, wires=wires)


add_decomps(U1, _u1_to_phaseshift)

add_decomps("Adjoint(U1)", adjoint_rotation2)

add_decomps("Pow(U1)", pow_rotation2)


def _u2_to_phaseshift_rot_resources(phi, delta, wires):  # pylint: disable=unused-argument
    return {PhaseShift: 2, Rot: 1}


@register_resources(_u2_to_phaseshift_rot_resources, name="to_phaseshift_rot")
def _u2_to_phaseshift_rot(phi, delta, wires):
    pi_half = qp.math.ones_like(delta) * (np.pi / 2)
    Rot(delta, pi_half, -delta, wires=wires)
    PhaseShift(delta, wires=wires)
    PhaseShift(phi, wires=wires)


add_decomps(U2, _u2_to_phaseshift_rot)


@register_resources(lambda base: {abstractify(U2): 1}, name="invert_angles")
def _adjoint_u2_standard(base):
    new_delta = qp.math.mod((np.pi - base.phi), (2 * np.pi))
    new_phi = qp.math.mod((np.pi - base.delta), (2 * np.pi))
    U2(new_phi, new_delta, wires=base.wires)


add_decomps("Adjoint(U2)", _adjoint_u2_standard)


def _u3_to_phaseshift_rot_resources(theta, phi, delta, wires):  # pylint: disable=unused-argument
    return {PhaseShift: 2, Rot: 1}


@register_resources(_u3_to_phaseshift_rot_resources, name="to_phaseshift_rot")
def _u3_to_phaseshift_rot(theta, phi, delta, wires):
    Rot(delta, theta, -delta, wires=wires)
    PhaseShift(delta, wires=wires)
    PhaseShift(phi, wires=wires)


add_decomps(U3, _u3_to_phaseshift_rot)


@register_resources(lambda base: {abstractify(U3): 1}, name="invert_angles")
def _adjoint_u3_standard(base):
    new_delta = qp.math.mod((np.pi - base.phi), (2 * np.pi))
    new_phi = qp.math.mod((np.pi - base.delta), (2 * np.pi))
    U3(base.theta, new_phi, new_delta, wires=base.wires)


add_decomps("Adjoint(U3)", _adjoint_u3_standard)


def _rz_phase_gradient(
    phi: float, wire: Wires, angle_wires: Wires, phase_grad_wires: Wires, work_wires: Wires
) -> Operator:
    """Build the RZ phase-gradient circuit without its global phase.

    Precision is determined by the length of ``angle_wires``. Callers add the global phase
    ``phi / 2``.
    """
    precision = len(angle_wires)

    binary_int = math.binary_decimals(phi, precision, unit=2 * np.pi)

    control_wire = Wires(wire)[0]
    compute_op = qp.ctrl(qp.MultiX(binary_int, angle_wires), control=control_wire)
    target_op = qp.SemiAdder(angle_wires, phase_grad_wires, work_wires)

    return change_op_basis(compute_op, target_op, compute_op)


def validate_phase_gradient_wires(angle_wires, phase_grad_wires, work_wires):
    """Validate three wire registers to be valid for phase gradient decompositions.

    Args:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state.
        work_wires (Wires): work wires for :class:`~.SemiAdder` (and possibly :class:`~.QROM`)
            decomposition.

    """
    angle_wires = Wires(angle_wires)
    phase_grad_wires = Wires(phase_grad_wires)
    work_wires = Wires(work_wires)
    if len(angle_wires) != len(phase_grad_wires):
        raise WireError(
            "angle_wires and phase_grad wires must be of same size, but received "
            f"{len(angle_wires)} angle_wires and {len(phase_grad_wires)} phase_grad_wires."
        )
    if len(phase_grad_wires) - 1 > len(work_wires):
        raise WireError(
            f"work_wires need to be at least of size len(phase_grad_wires) - 1, but received "
            f"{len(work_wires)} work_wires and {len(phase_grad_wires)-1=}"
        )
    return angle_wires, phase_grad_wires, work_wires


# Units of the angles in the phase gradient decomposition of each operator type,
# see ``qp.math.binary_decimals``
_ANGLE_UNITS = {"RZ": 2 * np.pi, "CRZ": 4 * np.pi, "SelectPauliRot": 4 * np.pi}


def _angle_unit(op_type):
    """Unit of the angles in the phase gradient decomposition of ``op_type``, which may be an
    operator class or its name."""
    name = getattr(op_type, "__name__", op_type)
    if name not in _ANGLE_UNITS:
        raise ValueError(
            f"op_type must be one of {list(_ANGLE_UNITS)} or the corresponding class, but got "
            f"{op_type}."
        )
    return _ANGLE_UNITS[name]


def _required_precision(binary_int):
    """Position of the last set bit across all binary representations in ``binary_int``, with
    the precision along the last axis. This is the number of bits required to represent all of
    them exactly."""
    return qp.math.max(binary_int * qp.math.arange(1, qp.math.shape(binary_int)[-1] + 1))


def phase_gradient_precision_stats(angles, precision, op_type):
    r"""Compute the ``precision_stats`` argument of the phase gradient decomposition rule
    factories from the rotation angles of the operators that a rule will decompose.

    A phase gradient decomposition with ``precision`` angle wires represents an angle
    :math:`\phi` by the ``precision`` leading binary decimals of :math:`\phi/u` (rounded to
    nearest), where the unit is :math:`u=2\pi` for :class:`~.RZ` and :math:`u=4\pi` for
    :class:`~.CRZ` and :class:`~.SelectPauliRot`. If the trailing decimals are zero, the rotation
    can be realized with fewer bits, i.e., with a smaller :class:`~.SemiAdder` and less data
    loading.
    This function counts how many operators require each number of bits. Passing the result as
    ``precision_stats`` to :func:`~.make_rz_to_phase_gradient_decomp`,
    :func:`~.make_crz_to_phase_gradient_decomp` or
    :func:`~.make_selectpaulirot_to_phase_gradient_decomp` makes the produced rule apply each of
    these operators at exactly the number of bits it requires, and hints the conditional
    branches with the observed frequencies.

    Args:
        angles (Sequence[TensorLike]): The angle(s) of each operator. For operators with
            multiple angles, such as :class:`~.SelectPauliRot`, the bits required by the operator
            are the maximum over all of its angles. The angles must be concrete.
        precision (int): The full precision of the decomposition rule, i.e., the number of angle
            wires.
        op_type (type | str): The type of the operators, i.e., :class:`~.RZ`, :class:`~.CRZ`
            or :class:`~.SelectPauliRot`, or its name. It determines the unit of the angles in
            the respective decomposition rule.

    Returns:
        dict[int, int]: The number of operators requiring each number of bits, sorted by the
        number of bits.

    .. seealso:: :func:`~.make_rz_to_phase_gradient_decomp`,
        :func:`~.make_crz_to_phase_gradient_decomp`,
        :func:`~.make_selectpaulirot_to_phase_gradient_decomp`

    **Example**

    Consider four :class:`~.RZ` rotations that we want to decompose with four angle wires. The
    first angle, :math:`\tfrac{1}{2}\cdot 2\pi`, has the binary representation :math:`(0.1000)_2`
    and thus only requires one bit, whereas :math:`0.1 \cdot 2\pi` is rounded to
    :math:`(0.0010)_2`, requiring three bits:

    >>> from pennylane.decomposition_rules import (
    ...     phase_gradient_precision_stats,
    ... )
    >>> angles = np.array([1 / 2, 1 / 4, 3 / 8, 0.1]) * 2 * np.pi
    >>> phase_gradient_precision_stats(angles, precision=4, op_type=qp.RZ)
    {1: 1, 2: 1, 3: 2}

    See :func:`~.make_rz_to_phase_gradient_decomp` for an example that uses these statistics
    in a decomposition rule.

    For :class:`~.SelectPauliRot`, we pass the angles of each operator. The operator type may
    also be specified by its name:

    >>> ops = [
    ...     qp.SelectPauliRot(np.array([1 / 2, 1 / 4]) * 4 * np.pi, control_wires=[0], target_wire=1),
    ...     qp.SelectPauliRot(np.array([1 / 2, 0]) * 4 * np.pi, control_wires=[0], target_wire=1),
    ... ]
    >>> angles = [op.data[0] for op in ops]
    >>> phase_gradient_precision_stats(angles, precision=4, op_type="SelectPauliRot")
    {1: 1, 2: 1}

    .. note::

        A :class:`~.SelectPauliRot` without control wires is decomposed into a plain
        :class:`~.RX`, :class:`~.RY` or :class:`~.RZ` rotation by
        :func:`~.make_selectpaulirot_to_phase_gradient_decomp`. If the resulting :class:`~.RZ`
        rotations are decomposed by a rule from :func:`~.make_rz_to_phase_gradient_decomp`,
        include their angles in the statistics for ``op_type=qp.RZ``, which uses the unit
        :math:`2\pi` instead of :math:`4\pi`.
    """
    unit = _angle_unit(op_type)
    required = (
        int(_required_precision(qp.math.binary_decimals(a, precision, unit=unit))) for a in angles
    )
    return dict(sorted(Counter(required).items()))


def _precision_dispatcher(precision_stats, precision, op_type):
    """Validate ``precision_stats`` and create a function that applies a phase gradient circuit
    at the smallest precision in ``precision_stats`` that suffices for the given angle(s).

    Args:
        precision_stats (dict[int, int] | None): number of rotations for each required precision.
            Zero counts are ignored. Without non-zero counts, the full ``precision`` is always
            applied.
        precision (int): full precision, i.e., the number of angle wires.
        op_type (type | str): type of the decomposed operators, determining the angle unit.

    Returns:
        tuple[set[int], Callable]: The precisions that may be applied, including the full
        ``precision``, and a function ``dispatch(angles, apply_fn)``, where ``apply_fn(p)``
        applies the full-precision circuit to the leading ``p`` angle and phase gradient wires.
        No conditional is used if the full ``precision`` is the only precision. ``angles`` may
        be traced if program capture is enabled.
    """
    precision_stats = {p: c for p, c in (precision_stats or {}).items() if c} or {precision: 1}
    if any(not 0 <= p <= precision for p in precision_stats):
        raise ValueError(
            f"precision_stats keys must be between 0 and {precision}, but got "
            f"{sorted(precision_stats)}."
        )
    total = sum(precision_stats.values())
    probs = {p: precision_stats[p] / total for p in sorted(precision_stats)}
    precisions = {precision, *probs}
    unit = _angle_unit(op_type)

    def dispatch(angles, apply_fn):
        def _branch(p):
            if p:
                apply_fn(p)

        # If only a single precision or no precision_stats was provided, we just call the branch
        # function for that single precision, without cond branching.
        if precisions == {precision}:
            _branch(precision)
            return

        # The circuit at precision p reproduces the bits of the angles exactly if p >= required,
        # because the bits beyond p are zero, so that rounding at precision p yields the same
        # bits as rounding at the full precision.
        required = _required_precision(qp.math.binary_decimals(angles, precision, unit=unit))
        branches = [
            (required <= p, qp.hint(partial(_branch, p), {"branch-prob": prob}))
            for p, prob in probs.items()
        ]
        # Without full precision in ``precision_stats``, the fallback is unhinted (probability 0)
        otherwise = branches.pop()[1] if precision in probs else partial(_branch, precision)
        (pred, true_fn), *elifs = branches
        qp.cond(pred, true_fn, otherwise, elifs=elifs)()

    return precisions, dispatch


def make_rz_to_phase_gradient_decomp(
    angle_wires, phase_grad_wires, work_wires, precision_stats=None
):
    r"""
    Create a custom decomposition rule for :class:`~.RZ` gates.

    Here, the necessary auxiliary wires for the `phase gradient decomposition of RZ
    <https://pennylane.ai/compilation/phase-gradient/b-rotations>`__ are taken as inputs and
    bound into the created decomposition rule.
    This way, this function can be used in a workflow context that explicitly uses those wires
    to generate this decomposition rule, which can then be registered with
    :func:`~pennylane.decomposition.add_decomps`.

    Parameters:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state
        work_wires (Wires): additional work wires for :class:`~.SemiAdder` decomposition
        precision_stats (dict[int, int] | None): Optional statistics of the number of bits
            required by the rotation angles, mapping a number of bits to the number of rotations
            requiring that many bits. Defaults to ``None``.

    Returns:
        qp.decomposition.DecompositionRule: decomposition rule to be used within :func:`~.pennylane.decompose`.

    .. seealso:: :func:`~.make_selectpaulirot_to_phase_gradient_decomp`,
        :func:`~.phase_gradient_precision_stats`

    **Example**

    In this example we decompose a circuit containing only a single :class:`~.RZ` gate using the custom decomposition rule
    that we generate from within the context of the example, where all auxiliary wires exist.

    .. code-block:: python

        import pennylane as qp
        from pennylane.decomposition_rules.qubit.parametric_ops_single_qubit import (
            make_rz_to_phase_gradient_decomp,
        )
        import numpy as np

        prec = 3
        phi = (1/2 + 1/4 + 1/8) * 2 * np.pi # binary rep is (111)

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
        phase_grad_wires = qp.wires.Wires([f"phg_{i}" for i in range(prec)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])

        custom_decomp = make_rz_to_phase_gradient_decomp(
            angle_wires, phase_grad_wires, work_wires
        )

        gate_set = {"CNOT", "SemiAdder", "GlobalPhase", "PauliX"}

        @qp.transforms._tape_decompose(gate_set=gate_set, fixed_decomps={qp.RZ: custom_decomp})
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            qp.RZ(phi, 0)
            return qp.state()

    The resulting circuit corresponds to the `phase gradient decomposition <https://pennylane.ai/compilation/phase-gradient/b-rotations>`__ of RZ,
    containing two CNOT fanouts corresponding to the binary representation of the angle (111 in this case), the :class:`~SemiAdder`, and a :class:`~GlobalPhase`.

    >>> print(qp.draw(circuit)())
         0: ─╭GlobalPhase(2.75)─╭●─╭●─╭●────────────╭●─╭●─╭●─┤ ╭State
     aux_0: ─├GlobalPhase(2.75)─╰X─│──│──╭SemiAdder─╰X─│──│──┤ ├State
     aux_1: ─├GlobalPhase(2.75)────╰X─│──├SemiAdder────╰X─│──┤ ├State
     aux_2: ─├GlobalPhase(2.75)───────╰X─├SemiAdder───────╰X─┤ ├State
     phg_0: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
     phg_1: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
     phg_2: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
    work_0: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
    work_1: ─╰GlobalPhase(2.75)──────────╰SemiAdder──────────┤ ╰State

    .. details::
        :title: Usage details

        **Adaptive precision via ``precision_stats``**

        For angles that do not require the full precision determined by the angle encoding and
        phase gradient wires, we may use a smaller encoding and adder, reducing the cost of the
        phase gradient rotation. However, as rotation angles are not known statically in general,
        this adaptive precision is only supported if additional information about the distribution
        of rotation angles is provided via ``precision_stats``, a dictionary mapping a number
        of bits to the number of rotations requiring that many bits.

        The bits required by an angle :math:`\phi` are the
        position of the last set bit in ``qp.math.binary_decimals(phi, len(angle_wires),
        unit=2 * np.pi)``, i.e., the precision without trailing zeros. If provided,
        the data loading and the :class:`~.SemiAdder` are truncated to the smallest number of
        bits in ``precision_stats`` that suffices for a given angle, using :func:`~.cond`
        branches that are hinted with the corresponding probabilities via :func:`~.hint`.
        If no number of bits in ``precision_stats`` suffices, the full precision is used.
        Keys must lie between ``0`` and ``len(angle_wires)``. Zero counts are ignored, and
        a dictionary without non-zero counts is treated like ``None``. Use
        :func:`~.phase_gradient_precision_stats` to compute the statistics from the angles.

        **Example**

        .. code-block:: python

            from pennylane.decomposition_rules.qubit.parametric_ops_single_qubit import (
                phase_gradient_precision_stats,
            )

            angles = np.array([1/2, 1/4 + 1/8, 1/2 + 1/4 + 1/8]) * 2 * np.pi # (100), (011), (111)
            precision_stats = phase_gradient_precision_stats(angles, prec, qp.RZ)

            truncating_decomp = make_rz_to_phase_gradient_decomp(
                angle_wires, phase_grad_wires, work_wires, precision_stats=precision_stats
            )

            @qp.transforms._tape_decompose(
                gate_set=gate_set, fixed_decomps={qp.RZ: truncating_decomp}
            )
            @qp.qnode(qp.device("null.qubit"))
            def truncated_circuit(phi):
                qp.RZ(phi, 0)
                return qp.state()

        One rotation requires a single bit and two rotations require all three bits. The branches
        of the rule are hinted with these frequencies, and the rotation by the first angle only uses
        a single angle wire and phase gradient wire:

        >>> precision_stats
        {1: 1, 3: 2}
        >>> wire_order = [0] + angle_wires + phase_grad_wires + work_wires
        >>> print(qp.draw(truncated_circuit, wire_order=wire_order)(angles[0]))
             0: ─╭GlobalPhase(1.57)─╭●────────────╭●─┤ ╭State
         aux_0: ─├GlobalPhase(1.57)─╰X─╭SemiAdder─╰X─┤ ├State
         phg_0: ─├GlobalPhase(1.57)────├SemiAdder────┤ ├State
        work_0: ─├GlobalPhase(1.57)────├SemiAdder────┤ ├State
        work_1: ─╰GlobalPhase(1.57)────╰SemiAdder────┤ ╰State

        >>> print(qp.draw(truncated_circuit, wire_order=wire_order)(angles[1]))
             0: ─╭GlobalPhase(1.18)─╭●─╭●────────────╭●─╭●─┤ ╭State
         aux_0: ─├GlobalPhase(1.18)─│──│──╭SemiAdder─│──│──┤ ├State
         aux_1: ─├GlobalPhase(1.18)─╰X─│──├SemiAdder─╰X─│──┤ ├State
         aux_2: ─├GlobalPhase(1.18)────╰X─├SemiAdder────╰X─┤ ├State
         phg_0: ─├GlobalPhase(1.18)───────├SemiAdder───────┤ ├State
         phg_1: ─├GlobalPhase(1.18)───────├SemiAdder───────┤ ├State
         phg_2: ─├GlobalPhase(1.18)───────├SemiAdder───────┤ ├State
        work_0: ─├GlobalPhase(1.18)───────├SemiAdder───────┤ ├State
        work_1: ─╰GlobalPhase(1.18)───────╰SemiAdder───────┤ ╰State

        Finally, keep the following in mind when using ``precision_stats``:

        * The resource function of the rule adds up the cost of all supported precisions, so
          that the graph-based decomposition system accounts for the operators in all
          conditional branches. This over-reports the cost of the rule, so that the
          decomposition graph may prefer another rule over this one when it is registered with
          :func:`~pennylane.decomposition.add_decomps`. Use ``fixed_decomps`` to enforce this rule.
        * Angles that require more bits than the largest key in ``precision_stats`` fall back
          to the full precision. Unless ``len(angle_wires)`` is a key in ``precision_stats``,
          this fallback branch is hinted with probability ``0``, so that resource estimates
          based on the hints do not account for these angles.

    """
    angle_wires, phase_grad_wires, work_wires = validate_phase_gradient_wires(
        angle_wires, phase_grad_wires, work_wires
    )
    precisions, dispatch = _precision_dispatcher(precision_stats, len(angle_wires), "RZ")

    def _rz_phase_gradient_standard_resources(phi, wires):  # pylint: disable=unused-argument
        # rz decomposition costs, using information about angle_wires etc from the outer scope.
        # Each non-zero precision that may be applied is counted once.
        resources = {qp.GlobalPhase: 1}
        for p in precisions - {0}:
            target_op = qp.SemiAdder(Wire[p], Wire[p], Wire[len(work_wires)])
            fanout = qp.ctrl(qp.MultiX(Bool[p], Wire[p]), control=Wire[1])
            resources[ChangeOpBasis2(fanout, target_op, fanout)] = 1
        return resources

    # MultiX only emits a gate per set bit, so the gate count depends on the concrete angle.
    @qp.register_resources(
        _rz_phase_gradient_standard_resources, exact=False, name="phase_gradient"
    )
    def _rz_phase_gradient_standard(phi, wires):
        qp.GlobalPhase(phi / 2)
        dispatch(
            phi,
            lambda p: _rz_phase_gradient(
                phi, wires, angle_wires[:p], phase_grad_wires[:p], work_wires
            ),
        )

    return _rz_phase_gradient_standard
