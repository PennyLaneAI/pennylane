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
from pennylane.decomposition.symbolic_decomposition import (
    _adjoint_negate_angle as _adjoint_negate_angle_legacy,
)
from pennylane.decomposition.symbolic_decomposition import (
    _pow_scale_angle as _pow_scale_angle_legacy,
)
from pennylane.ops.op_math.adjoint2 import _adjoint_negate_angle
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.op_math.pow2 import _pow_scale_angle
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
@register_resources({ops.Hadamard: 1, ops.CNOT: 1})
def _SingleExcitationBasis(wires: WiresLike, reverse: bool):
    if not reverse:
        ops.Hadamard(wires[0])
    ops.CNOT(wires)
    if reverse:
        ops.Hadamard(wires[0])


@subcircuit(dynamic_argnames=("phi",), arg_specs={"phi": Float, "wires": Wire[2]})
@register_resources({ops.RY: 2})
def _SingleExcitationRotations(phi: TensorLike, wires: WiresLike):
    ops.RY(-phi / 2, wires[0])
    ops.RY(-phi / 2, wires[1])


@register_resources(
    {
        _change_op_basis_abstract(
            _SingleExcitationBasis(wires=Wire[2], reverse=False),
            _SingleExcitationRotations(Float, wires=Wire[2]),
            _SingleExcitationBasis(wires=Wire[2], reverse=True),
        ): 1,
    },
    name="basis_change_ry",
)
def single_excitation_standard(phi: TensorLike, wires: WiresLike):
    """Decompose ``SingleExcitation`` using the ``basis_change_ry`` strategy."""
    ops.change_op_basis(
        _SingleExcitationBasis(wires=wires, reverse=False),
        _SingleExcitationRotations(phi, wires=wires),
        _SingleExcitationBasis(wires=wires, reverse=True),
    )


@subcircuit(dynamic_argnames=("phi",), arg_specs={"phi": Float, "wires": Wire[2]})
@register_resources({ops.RZ: 2})
def _SingleExcitationRZRotations(phi: TensorLike, wires: WiresLike):
    ops.RZ(phi / 2, wires[0])
    ops.RZ(-phi / 2, wires[1])


@register_resources(
    {
        ops.PauliRot(Float, pauli_word="XY", wires=Wire[2]): 1,
        ops.PauliRot(Float, pauli_word="YX", wires=Wire[2]): 1,
    },
    name="to_paulirot",
)
def single_excitation_to_ppr(phi: TensorLike, wires: WiresLike):
    """Decompose ``SingleExcitation`` using the ``to_paulirot`` strategy."""
    ops.PauliRot(phi / 2, "YX", wires=wires)
    ops.PauliRot(-phi / 2, "XY", wires=wires)


@register_resources(
    {
        _change_op_basis_abstract(
            ops.PPR_4(1, "XX", wires=Wire[2]),
            _SingleExcitationRZRotations(Float, wires=Wire[2]),
            ops.PPR_4(-1, "XX", wires=Wire[2]),
        ): 1,
    },
    name="to_ppr_rz",
)
def single_excitation_to_ppr_rz(phi: TensorLike, wires: WiresLike):
    """Decompose ``SingleExcitation`` using the ``to_ppr_rz`` strategy."""
    ops.change_op_basis(
        ops.PPR_4(1, "XX", wires=wires),
        _SingleExcitationRZRotations(phi, wires=wires),
        ops.PPR_4(-1, "XX", wires=wires),
    )


@subcircuit
@register_resources({ops.PPR_4(1, "XX", wires=Wire[2]): 1, ops.CNOT: 1})
def _SingleExcitationCRZBasis(wires: WiresLike):
    ops.PPR_4(1, "XX", wires=wires)
    ops.CNOT(wires)


@register_resources(
    {
        _change_op_basis_abstract(
            _SingleExcitationCRZBasis(wires=Wire[2]),
            ops.CRZ(Float, wires=Wire[2]),
            ops.adjoint(_SingleExcitationCRZBasis(wires=Wire[2])),
        ): 1,
    },
    name="to_ppr_crz",
)
def single_excitation_to_ppr_crz(phi: TensorLike, wires: WiresLike):
    r"""A single arbitrary-angle rotation per Givens rotation.

    In the basis of ``single_excitation_to_ppr_rz``, the two rotations
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
    single_excitation_standard,
    single_excitation_to_ppr,
    single_excitation_to_ppr_rz,
    single_excitation_to_ppr_crz,
)
add_decomps("Adjoint(SingleExcitation)", _adjoint_negate_angle)
add_decomps("Pow(SingleExcitation)", _pow_scale_angle)


@register_resources(
    {ops.Hadamard: 2, ops.CY: 1, ops.CNOT: 2, ops.RY: 2, ops.S: 1, ops.RZ: 1, ops.GlobalPhase: 1},
    name="to_single_qubit_cnot_cy",
)
def single_excitation_minus_standard(phi, wires: WiresLike, **__):
    """Decompose ``SingleExcitationMinus`` using the ``to_single_qubit_cnot_cy`` strategy."""
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


add_decomps(SingleExcitationMinus, single_excitation_minus_standard)
add_decomps("Adjoint(SingleExcitationMinus)", _adjoint_negate_angle_legacy)
add_decomps("Pow(SingleExcitationMinus)", _pow_scale_angle_legacy)


@register_resources(
    {ops.Hadamard: 2, ops.CY: 1, ops.CNOT: 2, ops.RY: 2, ops.S: 1, ops.RZ: 1, ops.GlobalPhase: 1},
    name="to_single_qubit_cnot_cy",
)
def single_excitation_plus_standard(phi, wires: WiresLike, **__):
    """Decompose ``SingleExcitationPlus`` using the ``to_single_qubit_cnot_cy`` strategy."""
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


add_decomps(SingleExcitationPlus, single_excitation_plus_standard)
add_decomps("Adjoint(SingleExcitationPlus)", _adjoint_negate_angle_legacy)
add_decomps("Pow(SingleExcitationPlus)", _pow_scale_angle_legacy)


@register_resources({ops.CNOT: 14, ops.Hadamard: 6, ops.RY: 8}, name="to_gates")
def double_excitation_to_gates(phi: TensorLike, wires: WiresLike):
    """Decompose ``DoubleExcitation`` using the ``to_gates`` strategy."""
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


@register_resources(
    {
        ops.PauliRot(Float, pauli_word="YYYX", wires=Wire[4]): 1,
        ops.PauliRot(Float, pauli_word="YYXY", wires=Wire[4]): 1,
        ops.PauliRot(Float, pauli_word="YXYY", wires=Wire[4]): 1,
        ops.PauliRot(Float, pauli_word="YXXX", wires=Wire[4]): 1,
        ops.PauliRot(Float, pauli_word="XYYY", wires=Wire[4]): 1,
        ops.PauliRot(Float, pauli_word="XYXX", wires=Wire[4]): 1,
        ops.PauliRot(Float, pauli_word="XXYX", wires=Wire[4]): 1,
        ops.PauliRot(Float, pauli_word="XXXY", wires=Wire[4]): 1,
    },
    name="to_paulirot",
)
def double_excitation_to_ppr(phi: TensorLike, wires: WiresLike):
    """Decompose ``DoubleExcitation`` using the ``to_paulirot`` strategy."""
    ops.PauliRot(phi / 8, pauli_word="YYYX", wires=wires)
    ops.PauliRot(phi / 8, pauli_word="YYXY", wires=wires)
    ops.PauliRot(-phi / 8, pauli_word="YXYY", wires=wires)
    ops.PauliRot(phi / 8, pauli_word="YXXX", wires=wires)
    ops.PauliRot(-phi / 8, pauli_word="XYYY", wires=wires)
    ops.PauliRot(phi / 8, pauli_word="XYXX", wires=wires)
    ops.PauliRot(-phi / 8, pauli_word="XXYX", wires=wires)
    ops.PauliRot(-phi / 8, pauli_word="XXXY", wires=wires)


add_decomps(DoubleExcitation, double_excitation_to_gates, double_excitation_to_ppr)
add_decomps("Adjoint(DoubleExcitation)", _adjoint_negate_angle)
add_decomps("Pow(DoubleExcitation)", _pow_scale_angle)
add_decomps("Adjoint(DoubleExcitationPlus)", _adjoint_negate_angle_legacy)
add_decomps("Pow(DoubleExcitationPlus)", _pow_scale_angle_legacy)
add_decomps("Adjoint(DoubleExcitationMinus)", _adjoint_negate_angle_legacy)
add_decomps("Pow(DoubleExcitationMinus)", _pow_scale_angle_legacy)


@register_resources(
    {FermionicSWAP: 2, SingleExcitation: 2},
    name="to_fermionic_swap_single_excitation",
)
def orbital_rotation_standard(phi: TensorLike, wires: WiresLike):
    """Decompose ``OrbitalRotation`` using the ``to_fermionic_swap_single_excitation`` strategy."""
    FermionicSWAP(np.pi, wires=[wires[1], wires[2]])
    SingleExcitation(phi, wires=[wires[0], wires[1]])
    SingleExcitation(phi, wires=[wires[2], wires[3]])
    FermionicSWAP(np.pi, wires=[wires[1], wires[2]])


add_decomps(OrbitalRotation, orbital_rotation_standard)
add_decomps("Adjoint(OrbitalRotation)", _adjoint_negate_angle)
add_decomps("Pow(OrbitalRotation)", _pow_scale_angle)


@register_resources(
    {ops.Hadamard: 4, ops.MultiRZ(Float, Wire[2]): 2, ops.RX: 4, ops.RZ: 2, ops.GlobalPhase: 1},
    name="to_multirz_single_qubit",
)
def fermionic_swap_standard(phi, wires: WiresLike, **__):
    """Decompose ``FermionicSWAP`` using the ``to_multirz_single_qubit`` strategy."""
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


add_decomps(FermionicSWAP, fermionic_swap_standard)
add_decomps("Adjoint(FermionicSWAP)", _adjoint_negate_angle)
add_decomps("Pow(FermionicSWAP)", _pow_scale_angle)
