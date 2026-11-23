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


"""Built-in decomposition rules for qubit.qchem_ops."""

import numpy as np

from pennylane import ops
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition_rules.op_math.adjoint import _adjoint_negate_angle
from pennylane.decomposition_rules.op_math.adjoint2 import (
    _adjoint_negate_angle as adjoint_rotation2,
)
from pennylane.decomposition_rules.op_math.pow import _pow_scale_angle
from pennylane.decomposition_rules.op_math.pow2 import _pow_scale_angle as pow_rotation2
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.qubit.qchem_ops import (
    DoubleExcitation,
    FermionicSWAP,
    OrbitalRotation,
    SingleExcitation,
    SingleExcitationMinus,
    SingleExcitationPlus,
)
from pennylane.ops.subcircuit import subcircuit
from pennylane.typing import Float, TensorLike, Wire
from pennylane.wires import WiresLike


@subcircuit(static_argnames=("reverse",))
@register_resources(lambda *_, **__: {ops.Hadamard: 1, ops.CNOT: 1})
def _SingleExcitationBasis(wires: WiresLike, reverse: bool):
    if not reverse:
        ops.Hadamard(wires[0])
    ops.CNOT(wires)
    if reverse:
        ops.Hadamard(wires[0])


@subcircuit(dynamic_argnames=("phi",), arg_specs={"phi": Float, "wires": Wire[2]})
@register_resources(lambda *_, **__: {ops.RY: 2})
def _SingleExcitationRotations(phi: TensorLike, wires: WiresLike):
    ops.RY(-phi / 2, wires[0])
    ops.RY(-phi / 2, wires[1])


# needs to be wrapped in function due to circular dependencies
def _single_excitation_standard_resources(phi, wires):  # pylint: disable=unused-argument
    return {
        _change_op_basis_abstract(
            _SingleExcitationBasis(wires=Wire[2], reverse=False),
            _SingleExcitationRotations(Float, wires=Wire[2]),
            _SingleExcitationBasis(wires=Wire[2], reverse=True),
        ): 1
    }


@register_resources(_single_excitation_standard_resources, name="basis_change_ry")
def _single_excitation_standard(phi: TensorLike, wires: WiresLike):
    ops.change_op_basis(
        _SingleExcitationBasis(wires=wires, reverse=False),
        _SingleExcitationRotations(phi, wires=wires),
        _SingleExcitationBasis(wires=wires, reverse=True),
    )


@subcircuit(dynamic_argnames=("phi",), arg_specs={"phi": Float, "wires": Wire[2]})
@register_resources(lambda *_, **__: {ops.RZ: 2})
def _SingleExcitationRZRotations(phi: TensorLike, wires: WiresLike):
    ops.RZ(phi / 2, wires[0])
    ops.RZ(-phi / 2, wires[1])


def _single_excitation_to_ppr_resources(phi, wires):  # pylint: disable=unused-argument
    return {
        ops.PauliRot(Float, pauli_word="XY", wires=Wire[2]): 1,
        ops.PauliRot(Float, pauli_word="YX", wires=Wire[2]): 1,
    }


@register_resources(_single_excitation_to_ppr_resources, name="to_paulirot")
def _single_excitation_to_ppr(phi: TensorLike, wires: WiresLike):
    ops.PauliRot(phi / 2, "YX", wires=wires)
    ops.PauliRot(-phi / 2, "XY", wires=wires)


def _single_excitation_to_ppr_rz_resources(phi, wires):  # pylint: disable=unused-argument
    return {
        _change_op_basis_abstract(
            ops.PPR_4(1, "XX", wires=Wire[2]),
            _SingleExcitationRZRotations(Float, wires=Wire[2]),
            ops.PPR_4(-1, "XX", wires=Wire[2]),
        ): 1
    }


@register_resources(_single_excitation_to_ppr_rz_resources, name="to_ppr_rz")
def _single_excitation_to_ppr_rz(phi: TensorLike, wires: WiresLike):
    ops.change_op_basis(
        ops.PPR_4(1, "XX", wires=wires),
        _SingleExcitationRZRotations(phi, wires=wires),
        ops.PPR_4(-1, "XX", wires=wires),
    )


@subcircuit
@register_resources(lambda *_, **__: {ops.PPR_4(1, "XX", wires=Wire[2]): 1, ops.CNOT: 1})
def _SingleExcitationCRZBasis(wires: WiresLike):
    ops.PPR_4(1, "XX", wires=wires)
    ops.CNOT(wires)


def _single_excitation_to_ppr_crz_resources(phi, wires):  # pylint: disable=unused-argument
    compute = _SingleExcitationCRZBasis(wires=Wire[2])
    return {
        _change_op_basis_abstract(
            compute,
            ops.CRZ(Float, wires=Wire[2]),
            ops.adjoint(compute),
        ): 1
    }


@register_resources(_single_excitation_to_ppr_crz_resources, name="to_ppr_crz")
def _single_excitation_to_ppr_crz(phi: TensorLike, wires: WiresLike):
    r"""A single arbitrary-angle rotation per Givens rotation.

    In the basis of ``_single_excitation_to_ppr_rz``, the two rotations
    :math:`RZ(\phi/2)_i\, RZ(-\phi/2)_j = e^{-i\phi(Z_i - Z_j)/4}` are a rotation whose angle
    depends only on whether the two wires differ. In the parity frame ``CNOT(i, j)`` this reads
    :math:`e^{-i\phi Z_i n_j/2}`, i.e. a single :class:`~.CRZ` controlled on ``j`` and targeting
    ``i``. This halves the number of arbitrary-angle rotations whenever a controlled rotation is
    about as expensive as an uncontrolled one, as for phase-gradient synthesis.
    """

    ops.change_op_basis(
        _SingleExcitationCRZBasis(wires=wires), ops.CRZ(phi, wires=[wires[1], wires[0]])
    )


add_decomps(
    SingleExcitation,
    _single_excitation_standard,
    _single_excitation_to_ppr,
    _single_excitation_to_ppr_rz,
    _single_excitation_to_ppr_crz,
)

add_decomps("Adjoint(SingleExcitation)", adjoint_rotation2)

add_decomps("Pow(SingleExcitation)", pow_rotation2)


def _single_excitation_minus_standard_resources():
    return {
        ops.Hadamard: 2,
        ops.CY: 1,
        ops.CNOT: 2,
        ops.RY: 2,
        ops.S: 1,
        ops.RZ: 1,
        ops.GlobalPhase: 1,
    }


@register_resources(_single_excitation_minus_standard_resources, name="to_single_qubit_cnot_cy")
def _single_excitation_minus_standard(phi, wires: WiresLike, **__):
    ops.Hadamard(wires[1])
    ops.CNOT([wires[1], wires[0]])
    ops.RY(phi / 2, wires[0])
    ops.RY(phi / 2, wires[1])
    ops.CY([wires[1], wires[0]])
    ops.S(wires[1])
    ops.Hadamard(wires[1])
    ops.RZ(phi / 2, wires[1])
    ops.CNOT(wires)
    ops.GlobalPhase(phi / 4)


add_decomps(SingleExcitationMinus, _single_excitation_minus_standard)

add_decomps("Adjoint(SingleExcitationMinus)", _adjoint_negate_angle)

add_decomps("Pow(SingleExcitationMinus)", _pow_scale_angle)


def _single_excitation_plus_standard_resources():
    return {
        ops.Hadamard: 2,
        ops.CY: 1,
        ops.CNOT: 2,
        ops.RY: 2,
        ops.S: 1,
        ops.RZ: 1,
        ops.GlobalPhase: 1,
    }


@register_resources(_single_excitation_plus_standard_resources, name="to_single_qubit_cnot_cy")
def _single_excitation_plus_standard(phi, wires: WiresLike, **__):
    ops.Hadamard(wires[1])
    ops.CNOT([wires[1], wires[0]])
    ops.RY(phi / 2, wires[0])
    ops.RY(phi / 2, wires[1])
    ops.CY([wires[1], wires[0]])
    ops.S(wires[1])
    ops.Hadamard(wires[1])
    ops.RZ(-phi / 2, wires[1])
    ops.CNOT(wires)
    ops.GlobalPhase(-phi / 4)


add_decomps(SingleExcitationPlus, _single_excitation_plus_standard)

add_decomps("Adjoint(SingleExcitationPlus)", _adjoint_negate_angle)

add_decomps("Pow(SingleExcitationPlus)", _pow_scale_angle)


def _double_excitation_to_gates_resources(phi, wires):  # pylint: disable=unused-argument
    return {ops.CNOT: 14, ops.Hadamard: 6, ops.RY: 8}


@register_resources(_double_excitation_to_gates_resources, name="to_gates")
def _double_excitation_to_gates(phi: TensorLike, wires: WiresLike):
    ops.CNOT(wires=[wires[2], wires[3]])
    ops.CNOT(wires=[wires[0], wires[2]])
    ops.Hadamard(wires=wires[3])
    ops.Hadamard(wires=wires[0])
    ops.CNOT(wires=[wires[2], wires[3]])
    ops.CNOT(wires=[wires[0], wires[1]])
    ops.RY(phi / 8, wires=wires[1])
    ops.RY(-phi / 8, wires=wires[0])
    ops.CNOT(wires=[wires[0], wires[3]])
    ops.Hadamard(wires=wires[3])
    ops.CNOT(wires=[wires[3], wires[1]])
    ops.RY(phi / 8, wires=wires[1])
    ops.RY(-phi / 8, wires=wires[0])
    ops.CNOT(wires=[wires[2], wires[1]])
    ops.CNOT(wires=[wires[2], wires[0]])
    ops.RY(-phi / 8, wires=wires[1])
    ops.RY(phi / 8, wires=wires[0])
    ops.CNOT(wires=[wires[3], wires[1]])
    ops.Hadamard(wires=wires[3])
    ops.CNOT(wires=[wires[0], wires[3]])
    ops.RY(-phi / 8, wires=wires[1])
    ops.RY(phi / 8, wires=wires[0])
    ops.CNOT(wires=[wires[0], wires[1]])
    ops.CNOT(wires=[wires[2], wires[0]])
    ops.Hadamard(wires=wires[0])
    ops.Hadamard(wires=wires[3])
    ops.CNOT(wires=[wires[0], wires[2]])
    ops.CNOT(wires=[wires[2], wires[3]])


def _double_excitation_to_ppr_resources(phi, wires):  # pylint: disable=unused-argument
    return {
        ops.PauliRot(Float, pauli_word="YYYX", wires=Wire[4]): 1,
        ops.PauliRot(Float, pauli_word="YYXY", wires=Wire[4]): 1,
        ops.PauliRot(Float, pauli_word="YXYY", wires=Wire[4]): 1,
        ops.PauliRot(Float, pauli_word="YXXX", wires=Wire[4]): 1,
        ops.PauliRot(Float, pauli_word="XYYY", wires=Wire[4]): 1,
        ops.PauliRot(Float, pauli_word="XYXX", wires=Wire[4]): 1,
        ops.PauliRot(Float, pauli_word="XXYX", wires=Wire[4]): 1,
        ops.PauliRot(Float, pauli_word="XXXY", wires=Wire[4]): 1,
    }


@register_resources(_double_excitation_to_ppr_resources, name="to_paulirot")
def _double_excitation_to_ppr(phi: TensorLike, wires: WiresLike):
    ops.PauliRot(phi / 8, pauli_word="YYYX", wires=wires)
    ops.PauliRot(phi / 8, pauli_word="YYXY", wires=wires)
    ops.PauliRot(-phi / 8, pauli_word="YXYY", wires=wires)
    ops.PauliRot(phi / 8, pauli_word="YXXX", wires=wires)
    ops.PauliRot(-phi / 8, pauli_word="XYYY", wires=wires)
    ops.PauliRot(phi / 8, pauli_word="XYXX", wires=wires)
    ops.PauliRot(-phi / 8, pauli_word="XXYX", wires=wires)
    ops.PauliRot(-phi / 8, pauli_word="XXXY", wires=wires)


add_decomps(DoubleExcitation, _double_excitation_to_gates, _double_excitation_to_ppr)

add_decomps("Adjoint(DoubleExcitation)", adjoint_rotation2)

add_decomps("Pow(DoubleExcitation)", pow_rotation2)


add_decomps("Adjoint(DoubleExcitationPlus)", _adjoint_negate_angle)

add_decomps("Pow(DoubleExcitationPlus)", _pow_scale_angle)


add_decomps("Adjoint(DoubleExcitationMinus)", _adjoint_negate_angle)

add_decomps("Pow(DoubleExcitationMinus)", _pow_scale_angle)


# pylint: disable-next=unused-argument
def _orbital_rotation_standard_resources(phi: TensorLike, wires: WiresLike):
    return {FermionicSWAP: 2, SingleExcitation: 2}


@register_resources(
    _orbital_rotation_standard_resources, name="to_fermionic_swap_single_excitation"
)
def _orbital_rotation_standard(phi: TensorLike, wires: WiresLike):
    FermionicSWAP(np.pi, wires=[wires[1], wires[2]])
    SingleExcitation(phi, wires=[wires[0], wires[1]])
    SingleExcitation(phi, wires=[wires[2], wires[3]])
    FermionicSWAP(np.pi, wires=[wires[1], wires[2]])


add_decomps(OrbitalRotation, _orbital_rotation_standard)

add_decomps("Adjoint(OrbitalRotation)", adjoint_rotation2)

add_decomps("Pow(OrbitalRotation)", pow_rotation2)


# pylint: disable-next=unused-argument
def _fermionic_swap_standard_resources(phi: TensorLike = None, wires: WiresLike = None):
    return {
        ops.Hadamard: 4,
        ops.MultiRZ(Float, Wire[2]): 2,
        ops.RX: 4,
        ops.RZ: 2,
        ops.GlobalPhase: 1,
    }


@register_resources(_fermionic_swap_standard_resources, name="to_multirz_single_qubit")
def _fermionic_swap_standard(phi, wires: WiresLike, **__):
    ops.Hadamard(wires=wires[0])
    ops.Hadamard(wires=wires[1])
    ops.MultiRZ(phi / 2, wires=[wires[0], wires[1]])
    ops.Hadamard(wires=wires[0])
    ops.Hadamard(wires=wires[1])
    ops.RX(np.pi / 2, wires=wires[0])
    ops.RX(np.pi / 2, wires=wires[1])
    ops.MultiRZ(phi / 2, wires=[wires[0], wires[1]])
    ops.RX(-np.pi / 2, wires=wires[0])
    ops.RX(-np.pi / 2, wires=wires[1])
    ops.RZ(phi / 2, wires=wires[0])
    ops.RZ(phi / 2, wires=wires[1])
    # for correcting global phase
    ops.GlobalPhase(-0.5 * phi)


add_decomps(FermionicSWAP, _fermionic_swap_standard)

add_decomps("Adjoint(FermionicSWAP)", adjoint_rotation2)

add_decomps("Pow(FermionicSWAP)", pow_rotation2)
