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
from itertools import combinations

import numpy as np

from pennylane import math, ops, templates
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.ops.op_math.adjoint2 import _adjoint_negate_angle as adjoint_negate_angle
from pennylane.ops.op_math.controlled2 import flip_zero_control
from pennylane.ops.op_math.pow2 import _pow_scale_angle as pow_scale_angle
from pennylane.ops.op_math import change_op_basis
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.qubit.parametric_ops_single_qubit import RX, RY, RZ, U1, U2, U3, PhaseShift, Rot
from pennylane.ops.subcircuit import subcircuit
from pennylane.typing import Float, Wire
from pennylane.wires import WiresLike, concatenate_wires


@register_resources({Rot: 1}, name="to_rot")
def rx_to_rot(phi, wires: WiresLike):
    """Decompose ``RX`` using the ``to_rot`` strategy."""
    Rot(np.pi / 2, phi, 3.5 * np.pi, wires=wires)


@register_resources({RZ: 2, RY: 1}, name="to_rz_ry")
def rx_to_rz_ry(phi, wires: WiresLike):
    """Decompose ``RX`` using the ``to_rz_ry`` strategy."""
    RZ(np.pi / 2, wires=wires)
    RY(phi, wires=wires)
    RZ(-np.pi / 2, wires=wires)


@register_resources(
    {_change_op_basis_abstract(ops.S, RY, ops.adjoint(ops.S(Wire[1]))): 1},
    name="to_ry_cliff",
)
def rx_to_ry_cliff(phi, wires: WiresLike):
    """Decompose ``RX`` using the ``to_ry_cliff`` strategy."""
    change_op_basis(ops.S(wires), RY(phi, wires))


@register_resources(
    {_change_op_basis_abstract(ops.Hadamard, RZ, ops.Hadamard): 1},
    name="to_rz_cliff",
)
def rx_to_rz_cliff(phi, wires: WiresLike):
    """Decompose ``RX`` using the ``to_rz_cliff`` strategy."""
    change_op_basis(ops.Hadamard(wires), RZ(phi, wires), ops.Hadamard(wires))


@register_resources({ops.PauliRot(Float, pauli_word="X", wires=Wire[1]): 1}, name="to_paulirot")
def rx_to_ppr(phi, wires):
    """Decompose ``RX`` using the ``to_paulirot`` strategy."""
    ops.PauliRot(phi, "X", wires=wires)


add_decomps(RX, rx_to_rot, rx_to_rz_ry, rx_to_ppr, rx_to_ry_cliff, rx_to_rz_cliff)
add_decomps("Adjoint(RX)", adjoint_negate_angle)
add_decomps("Pow(RX)", pow_scale_angle)


def _controlled_rx_standard_resources(  # pylint: disable=unused-argument
    base,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
):
    if len(control_wires) == 1:
        return {ops.CRX: 1}
    return {
        ops.H: 2,
        RZ: 2,
        ops.MultiControlledX(
            Wire[len(control_wires) + 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 2,
    }


@register_resources(_controlled_rx_standard_resources, name="to_rz_mcx")
# pylint: disable-next=unused-argument
def controlled_rx_standard(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose ``C(RX)`` using the ``to_rz_mcx`` strategy."""
    wires = concatenate_wires(control_wires, base.wires)
    if len(control_wires) == 1:
        ops.CRX(base.phi, wires=wires)
        return

    ops.H(wires=wires[-1])
    RZ(base.phi / 2, wires=wires[-1])
    ops.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)
    RZ(-base.phi / 2, wires=wires[-1])
    ops.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)
    ops.H(wires=wires[-1])


add_decomps("C(RX)", flip_zero_control(controlled_rx_standard, name="to_rz_mcx"))


@register_resources({Rot: 1}, name="to_rot")
def ry_to_rot(phi, wires: WiresLike):
    """Decompose ``RY`` using the ``to_rot`` strategy."""
    Rot(0, phi, 0, wires=wires)


@register_resources({RZ: 2, RX: 1}, name="to_rz_rx")
def ry_to_rz_rx(phi, wires: WiresLike):
    """Decompose ``RY`` using the ``to_rz_rx`` strategy."""
    RZ(-np.pi / 2, wires=wires)
    RX(phi, wires=wires)
    RZ(np.pi / 2, wires=wires)


@register_resources(
    {_change_op_basis_abstract(_adjoint_abstract(ops.S), RX, ops.S): 1},
    name="to_rx_cliff",
)
def ry_to_rx_cliff(phi, wires: WiresLike):
    """Decompose ``RY`` using the ``to_rx_cliff`` strategy."""
    change_op_basis(ops.adjoint(ops.S(wires)), RX(phi, wires), ops.S(wires))


@subcircuit
@register_resources({ops.Hadamard: 1, _adjoint_abstract(ops.S): 1})
def _YToZBasis(wires: WiresLike):
    ops.adjoint(ops.S(wires))
    ops.Hadamard(wires)


@subcircuit
@register_resources({ops.Hadamard: 1, ops.S: 1})
def _ZToYBasis(wires: WiresLike):
    ops.Hadamard(wires)
    ops.S(wires)


@register_resources(
    {_change_op_basis_abstract(_YToZBasis(Wire[1]), RZ, _ZToYBasis(Wire[1])): 1},
    name="to_rz_cliff",
)
def ry_to_rz_cliff(phi, wires: WiresLike, **__):
    """Decompose ``RY`` using the ``to_rz_cliff`` strategy."""
    change_op_basis(_YToZBasis(wires), RZ(phi, wires), _ZToYBasis(wires))


@register_resources({ops.PauliRot(Float, pauli_word="Y", wires=Wire[1]): 1}, name="to_paulirot")
def ry_to_ppr(phi, wires):
    """Decompose ``RY`` using the ``to_paulirot`` strategy."""
    ops.PauliRot(phi, "Y", wires=wires)


add_decomps(RY, ry_to_rot, ry_to_rz_rx, ry_to_ppr, ry_to_rx_cliff, ry_to_rz_cliff)
add_decomps("Adjoint(RY)", adjoint_negate_angle)
add_decomps("Pow(RY)", pow_scale_angle)


def _controlled_ry_standard_resources(  # pylint: disable=unused-argument
    base,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
):
    if len(control_wires) == 1:
        return {ops.CRY: 1}
    return {
        RY: 2,
        ops.MultiControlledX(
            Wire[len(control_wires) + 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 2,
    }


@register_resources(_controlled_ry_standard_resources, name="to_ry_mcx")
# pylint: disable-next=unused-argument
def controlled_ry_standard(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose ``C(RY)`` using the ``to_ry_mcx`` strategy."""
    wires = concatenate_wires(control_wires, base.wires)
    if len(control_wires) == 1:
        ops.CRY(base.phi, wires=wires)
        return

    RY(base.phi / 2, wires=wires[-1])
    ops.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)
    RY(-base.phi / 2, wires=wires[-1])
    ops.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)


add_decomps("C(RY)", flip_zero_control(controlled_ry_standard, name="to_ry_mcx"))


@register_resources({PhaseShift: 1, ops.GlobalPhase: 1}, name="to_ps")
def rz_to_ps(phi, wires: WiresLike):
    """Decompose ``RZ`` using the ``to_ps`` strategy."""
    PhaseShift(phi, wires)
    ops.GlobalPhase(phi / 2)


@register_resources({Rot: 1}, name="to_rot")
def rz_to_rot(phi, wires: WiresLike):
    """Decompose ``RZ`` using the ``to_rot`` strategy."""
    Rot(0, 0, phi, wires=wires)


@register_resources({RY: 2, RX: 1}, name="to_ry_rx")
def rz_to_ry_rx(phi, wires: WiresLike):
    """Decompose ``RZ`` using the ``to_ry_rx`` strategy."""
    RY(np.pi / 2, wires=wires)
    RX(phi, wires=wires)
    RY(-np.pi / 2, wires=wires)


@register_resources(
    {_change_op_basis_abstract(ops.Hadamard, RX, ops.Hadamard): 1},
    name="to_rx_cliff",
)
def rz_to_rx_cliff(phi, wires: WiresLike):
    """Decompose ``RZ`` using the ``to_rx_cliff`` strategy."""
    change_op_basis(ops.Hadamard(wires), RX(phi, wires), ops.Hadamard(wires))


@register_resources(
    {_change_op_basis_abstract(_ZToYBasis(Wire[1]), RY, _YToZBasis(Wire[1])): 1},
    name="to_ry_cliff",
)
def rz_to_ry_cliff(phi, wires: WiresLike):
    """Decompose ``RZ`` using the ``to_ry_cliff`` strategy."""
    change_op_basis(_ZToYBasis(wires), RY(phi, wires), _YToZBasis(wires))


@register_resources({ops.PauliRot(Float, pauli_word="Z", wires=Wire[1]): 1}, name="to_paulirot")
def rz_to_ppr(phi, wires, **_):
    """Decompose ``RZ`` using the ``to_paulirot`` strategy."""
    ops.PauliRot(phi, "Z", wires=wires)


add_decomps(RZ, rz_to_ps, rz_to_rot, rz_to_ry_rx, rz_to_ppr, rz_to_rx_cliff, rz_to_ry_cliff)
add_decomps("Adjoint(RZ)", adjoint_negate_angle)
add_decomps("Pow(RZ)", pow_scale_angle)


def _controlled_rz_standard_resources(  # pylint: disable=unused-argument
    base,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
):
    if len(control_wires) == 1:
        return {ops.CRZ: 1}
    return {
        RZ: 2,
        ops.MultiControlledX(
            Wire[len(control_wires) + 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 2,
    }


@register_resources(_controlled_rz_standard_resources, name="to_rz_mcx")
# pylint: disable-next=unused-argument
def controlled_rz_standard(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose ``C(RZ)`` using the ``to_rz_mcx`` strategy."""
    if len(control_wires) == 1:
        ops.CRZ(base.phi, wires=concatenate_wires(control_wires, base.wires))
        return

    RZ(base.phi / 2, wires=base.wires)
    ops.MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    RZ(-base.phi / 2, wires=base.wires)
    ops.MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(RZ)", flip_zero_control(controlled_rz_standard, name="to_rz_mcx"))


@register_resources({RZ: 1, ops.GlobalPhase: 1}, name="to_rz_gp")
def phaseshift_to_rz_gp(phi, wires: WiresLike, **__):
    """Decompose ``PhaseShift`` using the ``to_rz_gp`` strategy."""
    RZ(phi, wires=wires)
    ops.GlobalPhase(-phi / 2)


# pylint: disable-next=unused-argument
def _controlled_phase_shift_standard_resources(base, control_wires, *_, **__):
    # A single control wire specializes to ``ControlledPhaseShift``, matching the legacy
    # behaviour (and its cheaper ``RZ``/``CNOT`` decomposition) rather than the general
    # phase-polynomial (``PauliRot``) expansion, which is only used for multiple control wires.
    if len(control_wires) == 1:
        return {ops.ControlledPhaseShift: 1}
    num_control_wires = len(control_wires)
    resources = {
        ops.PauliRot(Float, pauli_word="Z" * i, wires=Wire[len("Z" * i)]): builtin_math.comb(
            num_control_wires + 1, i
        )
        for i in range(1, num_control_wires + 2)
    }
    resources[ops.GlobalPhase] = 1
    return resources


@register_resources(_controlled_phase_shift_standard_resources, name="to_pauli_rot")
def controlled_phase_shift_standard(base, control_wires, *_, **__):
    """Decompose ``C(PhaseShift)`` using the ``to_pauli_rot`` strategy."""
    wires = concatenate_wires(control_wires, base.wires)
    if len(control_wires) == 1:
        ops.ControlledPhaseShift(base.phi, wires=wires)
        return

    n = len(wires)
    for l in range(1, n + 1):
        for sub_wires in combinations(wires, l):
            phi = -base.phi / 2 ** (n - 1) * (-1) ** l
            ops.PauliRot(phi, pauli_word="Z" * l, wires=sub_wires)
    ops.GlobalPhase(-base.phi / 2**n)


def _controlled_phase_shift_single_phase_resources(base, control_wires, *_, **__):
    #  pylint: disable=unused-argument
    num_control_wires = len(control_wires)
    return {
        templates.TemporaryAND: num_control_wires,
        PhaseShift: 1,
        ops.adjoint(templates.TemporaryAND(Wire[3])): num_control_wires,
    }


@register_condition(
    lambda control_wires, work_wires, work_wire_type, **_: work_wire_type == "zeroed"
    and len(control_wires) > 0
    and len(work_wires) >= len(control_wires)
)
@register_resources(_controlled_phase_shift_single_phase_resources, name="single_phase")
def controlled_phase_shift_single_phase(base, control_wires, work_wires, **_):
    """Computes ``AND(*control_wires, base.wires[0])`` into ``len(control_wires)`` zeroed work
    wires with a ``TemporaryAND`` ladder, applies a single ``PhaseShift`` to the last work wire
    and uncomputes the ladder with measurements."""
    and_controls = [*control_wires, base.wires[0]]
    and_targets = list(work_wires[: len(control_wires)])
    and_wires = list(zip([and_controls[0], *and_targets[:-1]], and_controls[1:], and_targets))

    for wires in and_wires:
        templates.TemporaryAND(wires=wires)
    PhaseShift(base.phi, wires=and_targets[-1])
    for wires in reversed(and_wires):
        ops.adjoint(templates.TemporaryAND)(wires=wires)


add_decomps(PhaseShift, phaseshift_to_rz_gp)
add_decomps("Adjoint(PhaseShift)", adjoint_negate_angle)
add_decomps("Pow(PhaseShift)", pow_scale_angle)
add_decomps(
    "C(PhaseShift)",
    flip_zero_control(controlled_phase_shift_standard, name="to_pauli_rot"),
    flip_zero_control(controlled_phase_shift_single_phase, name="single_phase"),
)


@register_resources({RZ: 2, RY: 1}, name="to_rz_ry_rz")
def rot_to_rz_ry_rz(phi, theta, omega, wires: WiresLike):
    """Decompose ``Rot`` using the ``to_rz_ry_rz`` strategy."""
    RZ(phi, wires=wires)
    RY(theta, wires=wires)
    RZ(omega, wires=wires)


add_decomps(Rot, rot_to_rz_ry_rz)


@register_resources({Rot: 1}, name="reverse_angles")
def adjoint_rot_standard(base):
    """Decompose ``Adjoint(Rot)`` using the ``reverse_angles`` strategy."""
    Rot(-base.omega, -base.theta, -base.phi, wires=base.wires)


add_decomps("Adjoint(Rot)", adjoint_rot_standard)


def _controlled_rot_standard_resources(  # pylint: disable=unused-argument
    base,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
):
    if len(control_wires) == 1:
        return {ops.CRot: 1}
    return {
        RZ: 3,
        RY: 2,
        ops.MultiControlledX(
            Wire[len(control_wires) + 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 2,
    }


@register_resources(_controlled_rot_standard_resources, name="to_rz_ry_mcx")
# pylint: disable-next=unused-argument
def controlled_rot_standard(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose ``C(Rot)`` using the ``to_rz_ry_mcx`` strategy."""
    phi, theta, omega = base.phi, base.theta, base.omega
    wires = concatenate_wires(control_wires, base.wires)

    if len(control_wires) == 1:
        ops.CRot(phi, theta, omega, wires=wires)
        return

    RZ((phi - omega) / 2, wires=wires[-1])
    ops.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)
    RZ(-(phi + omega) / 2, wires=wires[-1])
    RY(-theta / 2, wires=wires[-1])
    ops.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)
    RY(theta / 2, wires=wires[-1])
    RZ(omega, wires=wires[-1])


add_decomps("C(Rot)", flip_zero_control(controlled_rot_standard, name="to_rz_ry_mcx"))


@register_resources({PhaseShift: 1}, name="to_phaseshift")
def u1_to_phaseshift(phi, wires):
    """Decompose ``U1`` using the ``to_phaseshift`` strategy."""
    PhaseShift(phi, wires=wires)


add_decomps(U1, u1_to_phaseshift)
add_decomps("Adjoint(U1)", adjoint_negate_angle)
add_decomps("Pow(U1)", pow_scale_angle)


@register_resources({PhaseShift: 2, Rot: 1}, name="to_phaseshift_rot")
def u2_to_phaseshift_rot(phi, delta, wires):
    """Decompose ``U2`` using the ``to_phaseshift_rot`` strategy."""
    pi_half = math.ones_like(delta) * (np.pi / 2)
    Rot(delta, pi_half, -delta, wires=wires)
    PhaseShift(delta, wires=wires)
    PhaseShift(phi, wires=wires)


add_decomps(U2, u2_to_phaseshift_rot)


@register_resources({abstractify(U2): 1}, name="invert_angles")
def adjoint_u2_standard(base):
    """Decompose ``Adjoint(U2)`` using the ``invert_angles`` strategy."""
    new_delta = math.mod((np.pi - base.phi), (2 * np.pi))
    new_phi = math.mod((np.pi - base.delta), (2 * np.pi))
    U2(new_phi, new_delta, wires=base.wires)


add_decomps("Adjoint(U2)", adjoint_u2_standard)


@register_resources({PhaseShift: 2, Rot: 1}, name="to_phaseshift_rot")
def u3_to_phaseshift_rot(theta, phi, delta, wires):
    """Decompose ``U3`` using the ``to_phaseshift_rot`` strategy."""
    Rot(delta, theta, -delta, wires=wires)
    PhaseShift(delta, wires=wires)
    PhaseShift(phi, wires=wires)


add_decomps(U3, u3_to_phaseshift_rot)


@register_resources({abstractify(U3): 1}, name="invert_angles")
def adjoint_u3_standard(base):
    """Decompose ``Adjoint(U3)`` using the ``invert_angles`` strategy."""
    new_delta = math.mod((np.pi - base.phi), (2 * np.pi))
    new_phi = math.mod((np.pi - base.delta), (2 * np.pi))
    U3(base.theta, new_phi, new_delta, wires=base.wires)


add_decomps("Adjoint(U3)", adjoint_u3_standard)
