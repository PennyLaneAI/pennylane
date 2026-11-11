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

# pylint: disable=arguments-differ

import numpy as np

import pennylane as qp
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition_rules.op_math.adjoint import adjoint_rotation
from pennylane.decomposition_rules.op_math.adjoint2 import adjoint_rotation as adjoint_rotation2
from pennylane.decomposition_rules.op_math.pow import pow_rotation
from pennylane.decomposition_rules.op_math.pow2 import pow_rotation as pow_rotation2
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.op_math.prod2 import Prod2
from pennylane.ops.qubit.qchem_ops import (
    DoubleExcitation,
    FermionicSWAP,
    OrbitalRotation,
    SingleExcitation,
    SingleExcitationMinus,
    SingleExcitationPlus,
)
from pennylane.typing import Float, TensorLike, Wire
from pennylane.wires import WiresLike


# needs to be wrapped in function due to circular dependencies
# pylint: disable-next=unused-argument
def _single_excitation_resources(phi, wires):
    # ``Prod2`` takes its operands in matrix order, i.e. reversed relative to the order in which
    # the decomposition below applies them.
    basis = Prod2((qp.CNOT(wires=Wire[2]), qp.Hadamard(wires=Wire[1])))
    unbasis = Prod2((qp.Hadamard(wires=Wire[1]), qp.CNOT(wires=Wire[2])))
    return {
        _change_op_basis_abstract(
            basis,
            Prod2((qp.RY(Float, wires=Wire[1]), qp.RY(Float, wires=Wire[1]))),
            unbasis,
        ): 1
    }


@register_resources(_single_excitation_resources)
def _single_excitation_decomp(phi: TensorLike, wires: WiresLike):
    def _to_basis():
        qp.Hadamard(wires[0])
        qp.CNOT(wires)

    def _rotations():
        qp.RY(-phi / 2, wires[0])
        qp.RY(-phi / 2, wires[1])

    def _from_basis():
        qp.CNOT(wires)
        qp.Hadamard(wires[0])

    qp.change_op_basis(_to_basis, _rotations, _from_basis)


# pylint: disable-next=unused-argument
def _single_excitation_ppr_resource(phi, wires):
    return {
        qp.PauliRot(Float, pauli_word="XY", wires=Wire[2]): 1,
        qp.PauliRot(Float, pauli_word="YX", wires=Wire[2]): 1,
    }


@register_resources(_single_excitation_ppr_resource)
def _single_excitation_ppr(phi: TensorLike, wires: WiresLike):
    qp.PauliRot(phi / 2, "YX", wires=wires)
    qp.PauliRot(-phi / 2, "XY", wires=wires)


# pylint: disable-next=unused-argument
def _single_excitation_ppr_rz_resource(phi, wires):
    return {
        _change_op_basis_abstract(
            qp.PPR_4(1, "XX", wires=Wire[2]),
            Prod2((qp.RZ(Float, wires=Wire[1]), qp.RZ(Float, wires=Wire[1]))),
            qp.PPR_4(-1, "XX", wires=Wire[2]),
        ): 1
    }


@register_resources(_single_excitation_ppr_rz_resource)
def _single_excitation_ppr_rz(phi: TensorLike, wires: WiresLike):
    def _rotations():
        qp.RZ(phi / 2, wires[0])
        qp.RZ(-phi / 2, wires[1])

    qp.change_op_basis(qp.PPR_4(1, "XX", wires=wires), _rotations, qp.PPR_4(-1, "XX", wires=wires))


# pylint: disable-next=unused-argument
def _single_excitation_ppr_crz_resource(phi, wires):
    compute = Prod2((qp.CNOT(Wire[2]), qp.PPR_4(1, "XX", wires=Wire[2])))
    return {
        _change_op_basis_abstract(
            compute,
            qp.CRZ(Float, wires=Wire[2]),
            qp.adjoint(compute),
        ): 1
    }


@register_resources(_single_excitation_ppr_crz_resource)
def _single_excitation_ppr_crz(phi: TensorLike, wires: WiresLike):
    r"""A single arbitrary-angle rotation per Givens rotation.

    In the basis of ``_single_excitation_ppr_rz``, the two rotations
    :math:`RZ(\phi/2)_i\, RZ(-\phi/2)_j = e^{-i\phi(Z_i - Z_j)/4}` are a rotation whose angle
    depends only on whether the two wires differ. In the parity frame ``CNOT(i, j)`` this reads
    :math:`e^{-i\phi Z_i n_j/2}`, i.e. a single :class:`~.CRZ` controlled on ``j`` and targeting
    ``i``. This halves the number of arbitrary-angle rotations whenever a controlled rotation is
    about as expensive as an uncontrolled one, as for phase-gradient synthesis.
    """

    def _compute():
        qp.PPR_4(1, "XX", wires=wires)
        qp.CNOT(wires)

    def _rotation():
        qp.CRZ(phi, wires=[wires[1], wires[0]])

    qp.change_op_basis(_compute, _rotation)


add_decomps(
    SingleExcitation,
    _single_excitation_decomp,
    _single_excitation_ppr,
    _single_excitation_ppr_rz,
    _single_excitation_ppr_crz,
)

add_decomps("Adjoint(SingleExcitation)", adjoint_rotation2)

add_decomps("Pow(SingleExcitation)", pow_rotation2)


def _single_excitation_minus_decomp_resources():
    return {
        qp.Hadamard: 2,
        qp.CY: 1,
        qp.CNOT: 2,
        qp.RY: 2,
        qp.S: 1,
        qp.RZ: 1,
        qp.GlobalPhase: 1,
    }


@register_resources(_single_excitation_minus_decomp_resources)
def _single_excitation_minus_decomp(phi, wires: WiresLike, **__):
    qp.Hadamard(wires[1])
    qp.CNOT([wires[1], wires[0]])
    qp.RY(phi / 2, wires[0])
    qp.RY(phi / 2, wires[1])
    qp.CY([wires[1], wires[0]])
    qp.S(wires[1])
    qp.Hadamard(wires[1])
    qp.RZ(phi / 2, wires[1])
    qp.CNOT(wires)
    qp.GlobalPhase(phi / 4)


add_decomps(SingleExcitationMinus, _single_excitation_minus_decomp)

add_decomps("Adjoint(SingleExcitationMinus)", adjoint_rotation)

add_decomps("Pow(SingleExcitationMinus)", pow_rotation)


def _single_excitation_plus_decomp_resources():
    return {
        qp.Hadamard: 2,
        qp.CY: 1,
        qp.CNOT: 2,
        qp.RY: 2,
        qp.S: 1,
        qp.RZ: 1,
        qp.GlobalPhase: 1,
    }


@register_resources(_single_excitation_plus_decomp_resources)
def _single_excitation_plus_decomp(phi, wires: WiresLike, **__):
    qp.Hadamard(wires[1])
    qp.CNOT([wires[1], wires[0]])
    qp.RY(phi / 2, wires[0])
    qp.RY(phi / 2, wires[1])
    qp.CY([wires[1], wires[0]])
    qp.S(wires[1])
    qp.Hadamard(wires[1])
    qp.RZ(-phi / 2, wires[1])
    qp.CNOT(wires)
    qp.GlobalPhase(-phi / 4)


add_decomps(SingleExcitationPlus, _single_excitation_plus_decomp)

add_decomps("Adjoint(SingleExcitationPlus)", adjoint_rotation)

add_decomps("Pow(SingleExcitationPlus)", pow_rotation)


def _doublexcit_resource(phi, wires):  # pylint: disable=unused-argument
    return {qp.CNOT: 14, qp.Hadamard: 6, qp.RY: 8}


@register_resources(_doublexcit_resource)
def _doublexcit(phi: TensorLike, wires: WiresLike):
    qp.CNOT(wires=[wires[2], wires[3]])
    qp.CNOT(wires=[wires[0], wires[2]])
    qp.Hadamard(wires=wires[3])
    qp.Hadamard(wires=wires[0])
    qp.CNOT(wires=[wires[2], wires[3]])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.RY(phi / 8, wires=wires[1])
    qp.RY(-phi / 8, wires=wires[0])
    qp.CNOT(wires=[wires[0], wires[3]])
    qp.Hadamard(wires=wires[3])
    qp.CNOT(wires=[wires[3], wires[1]])
    qp.RY(phi / 8, wires=wires[1])
    qp.RY(-phi / 8, wires=wires[0])
    qp.CNOT(wires=[wires[2], wires[1]])
    qp.CNOT(wires=[wires[2], wires[0]])
    qp.RY(-phi / 8, wires=wires[1])
    qp.RY(phi / 8, wires=wires[0])
    qp.CNOT(wires=[wires[3], wires[1]])
    qp.Hadamard(wires=wires[3])
    qp.CNOT(wires=[wires[0], wires[3]])
    qp.RY(-phi / 8, wires=wires[1])
    qp.RY(phi / 8, wires=wires[0])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.CNOT(wires=[wires[2], wires[0]])
    qp.Hadamard(wires=wires[0])
    qp.Hadamard(wires=wires[3])
    qp.CNOT(wires=[wires[0], wires[2]])
    qp.CNOT(wires=[wires[2], wires[3]])


def _doublexcit_ppr_resource(phi, wires):  # pylint: disable=unused-argument
    return {
        qp.PauliRot(Float, pauli_word="YYYX", wires=Wire[4]): 1,
        qp.PauliRot(Float, pauli_word="YYXY", wires=Wire[4]): 1,
        qp.PauliRot(Float, pauli_word="YXYY", wires=Wire[4]): 1,
        qp.PauliRot(Float, pauli_word="YXXX", wires=Wire[4]): 1,
        qp.PauliRot(Float, pauli_word="XYYY", wires=Wire[4]): 1,
        qp.PauliRot(Float, pauli_word="XYXX", wires=Wire[4]): 1,
        qp.PauliRot(Float, pauli_word="XXYX", wires=Wire[4]): 1,
        qp.PauliRot(Float, pauli_word="XXXY", wires=Wire[4]): 1,
    }


@register_resources(_doublexcit_ppr_resource)
def _doublexcit_ppr(phi: TensorLike, wires: WiresLike):
    qp.PauliRot(phi / 8, pauli_word="YYYX", wires=wires)
    qp.PauliRot(phi / 8, pauli_word="YYXY", wires=wires)
    qp.PauliRot(-phi / 8, pauli_word="YXYY", wires=wires)
    qp.PauliRot(phi / 8, pauli_word="YXXX", wires=wires)
    qp.PauliRot(-phi / 8, pauli_word="XYYY", wires=wires)
    qp.PauliRot(phi / 8, pauli_word="XYXX", wires=wires)
    qp.PauliRot(-phi / 8, pauli_word="XXYX", wires=wires)
    qp.PauliRot(-phi / 8, pauli_word="XXXY", wires=wires)


add_decomps(DoubleExcitation, _doublexcit, _doublexcit_ppr)

add_decomps("Adjoint(DoubleExcitation)", adjoint_rotation2)

add_decomps("Pow(DoubleExcitation)", pow_rotation2)

add_decomps("Adjoint(DoubleExcitationPlus)", adjoint_rotation)

add_decomps("Pow(DoubleExcitationPlus)", pow_rotation)

add_decomps("Adjoint(DoubleExcitationMinus)", adjoint_rotation)

add_decomps("Pow(DoubleExcitationMinus)", pow_rotation)


# pylint: disable-next=unused-argument
def _orbital_rotation_decomposition_resources(phi: TensorLike, wires: WiresLike):
    return {qp.FermionicSWAP: 2, qp.SingleExcitation: 2}


@register_resources(_orbital_rotation_decomposition_resources)
def _orbital_rotation_decomp(phi: TensorLike, wires: WiresLike):
    qp.FermionicSWAP(np.pi, wires=[wires[1], wires[2]])
    qp.SingleExcitation(phi, wires=[wires[0], wires[1]])
    qp.SingleExcitation(phi, wires=[wires[2], wires[3]])
    qp.FermionicSWAP(np.pi, wires=[wires[1], wires[2]])


add_decomps(OrbitalRotation, _orbital_rotation_decomp)

add_decomps("Adjoint(OrbitalRotation)", adjoint_rotation2)

add_decomps("Pow(OrbitalRotation)", pow_rotation2)


# pylint: disable-next=unused-argument
def _fermionic_swap_decomp_resources(phi: TensorLike = None, wires: WiresLike = None):
    return {
        qp.Hadamard: 4,
        qp.MultiRZ(Float, Wire[2]): 2,
        qp.RX: 4,
        qp.RZ: 2,
        qp.GlobalPhase: 1,
    }


@register_resources(_fermionic_swap_decomp_resources)
def _fermionic_swap_decomp(phi, wires: WiresLike, **__):
    qp.Hadamard(wires=wires[0])
    qp.Hadamard(wires=wires[1])
    qp.MultiRZ(phi / 2, wires=[wires[0], wires[1]])
    qp.Hadamard(wires=wires[0])
    qp.Hadamard(wires=wires[1])
    qp.RX(np.pi / 2, wires=wires[0])
    qp.RX(np.pi / 2, wires=wires[1])
    qp.MultiRZ(phi / 2, wires=[wires[0], wires[1]])
    qp.RX(-np.pi / 2, wires=wires[0])
    qp.RX(-np.pi / 2, wires=wires[1])
    qp.RZ(phi / 2, wires=wires[0])
    qp.RZ(phi / 2, wires=wires[1])
    # for correcting global phase
    qp.GlobalPhase(-0.5 * phi)


add_decomps(FermionicSWAP, _fermionic_swap_decomp)

add_decomps("Adjoint(FermionicSWAP)", adjoint_rotation2)

add_decomps("Pow(FermionicSWAP)", pow_rotation2)
