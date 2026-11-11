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

"""Built-in decomposition rules for qubit.parametric_ops_multi_qubit."""

# pylint: disable=arguments-differ

import functools
from collections import Counter

import numpy as np

import pennylane as qp
from pennylane import math
from pennylane.capture.autograph import disable_autograph
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition_rules.op_math.adjoint import adjoint_rotation
from pennylane.decomposition_rules.op_math.adjoint2 import adjoint_rotation as adjoint_rotation2
from pennylane.decomposition_rules.op_math.controlled2 import flip_zero_control
from pennylane.decomposition_rules.op_math.pow import pow_rotation
from pennylane.decomposition_rules.op_math.pow2 import pow_rotation as pow_rotation2
from pennylane.math.decomposition import decomp_int_to_powers_of_two
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.op_math.prod2 import Prod2
from pennylane.ops.qubit.non_parametric_ops import Hadamard, PauliX
from pennylane.ops.qubit.parametric_ops_multi_qubit import (
    PSWAP,
    CPhaseShift00,
    CPhaseShift01,
    CPhaseShift10,
    IsingXX,
    IsingXY,
    IsingYY,
    IsingZZ,
    MultiRZ,
    PauliRot,
    PCPhase,
)
from pennylane.ops.qubit.parametric_ops_single_qubit import RX, RY, RZ, PhaseShift
from pennylane.typing import Bool, Float, TensorLike, Wire
from pennylane.wires import WiresLike


# pylint: disable=unused-argument
def _multi_rz_decomposition_resources(theta: TensorLike, wires: WiresLike):
    num_wires = len(wires)
    if num_wires == 1:
        return {qp.RZ: 1}
    cnots = tuple(qp.CNOT(wires=Wire[2]) for _ in range(num_wires - 1))
    # a two-wire ladder is a lone CNOT, which ``change_op_basis`` does not wrap in a product
    ladder = cnots[0] if num_wires == 2 else Prod2(cnots)
    # Reversing identical abstract CNOT reps would produce the same resource key.
    unladder = ladder
    return {_change_op_basis_abstract(ladder, RZ(Float, wires=Wire[1]), unladder): 1}


@register_resources(_multi_rz_decomposition_resources)
def _multi_rz_decomposition(theta: TensorLike, wires: WiresLike):
    """CNOT ladder around an ``RZ``, so a control applies only to the rotation."""
    if len(wires) == 1:
        qp.RZ(theta, wires=wires[0])
        return

    if qp.compiler.active() or qp.capture.enabled():
        wires = math.array(wires, like="jax")

    def _cnots(i):
        qp.CNOT(wires=(wires[i], wires[i - 1]))

    num_wires = len(wires)
    qp.change_op_basis(
        qp.for_loop(num_wires - 1, 0, -1)(_cnots),
        qp.RZ(theta, wires=wires[0]),
        qp.for_loop(1, num_wires)(_cnots),
    )


def _multi_rz_to_ppr_resources(theta: TensorLike, wires: WiresLike):
    num_wires = len(wires)
    return {qp.PauliRot(Float, pauli_word="Z" * num_wires, wires=Wire[num_wires]): 1}


@register_resources(_multi_rz_to_ppr_resources)
def _multi_rz_to_ppr(theta: TensorLike, wires: WiresLike):
    qp.PauliRot(theta, "Z" * len(wires), wires=wires)


add_decomps(MultiRZ, _multi_rz_decomposition, _multi_rz_to_ppr)

add_decomps("Adjoint(MultiRZ)", adjoint_rotation2)

add_decomps("Pow(MultiRZ)", pow_rotation2)


def _pauli_rot_resources(theta, pauli_word, wires):  # pylint: disable=unused-argument
    if set(pauli_word) == {"I"}:
        return {qp.GlobalPhase: 1}
    num_active_wires = len(pauli_word.replace("I", ""))
    # non-Z gates in matrix order, i.e. reversed relative to the order they are applied in
    basis_word = [gate for gate in reversed(pauli_word) if gate in "XY"]
    if not basis_word:
        # a pure-Z word needs no basis change, so there is nothing to conjugate
        return {qp.MultiRZ(Float, Wire[num_active_wires]): 1}

    # A Y is an X rotation by ``±π/2``, i.e. ``PPR_4(±1, "X")``; an X uses a Hadamard.
    # Hadamard is required here: a ``PPR_4`` on Y decomposes back into ``PauliRot``, which
    # would cycle with the Y basis change. The compute and uncompute bases differ only
    # in the sign of the Y rotation.
    def _basis(sign):
        gates = tuple(
            qp.Hadamard(wires=Wire[1]) if gate == "X" else qp.PPR_4(sign, "X", wires=Wire[1])
            for gate in basis_word
        )
        # a single-gate basis change is not wrapped in a product
        return gates[0] if len(gates) == 1 else Prod2(gates)

    return {
        _change_op_basis_abstract(
            _basis(1), qp.MultiRZ(Float, Wire[num_active_wires]), _basis(-1)
        ): 1
    }


@register_resources(_pauli_rot_resources)
@disable_autograph
def _pauli_rot_decomposition(theta: TensorLike, pauli_word: str, wires: WiresLike):
    if set(pauli_word) == {"I"}:
        qp.GlobalPhase(theta / 2)
        return
    active_wires, active_gates = zip(
        *[(wire, gate) for wire, gate in zip(wires, pauli_word, strict=True) if gate != "I"],
        strict=True,
    )

    if set(active_gates) == {"Z"}:
        # a pure-Z word needs no basis change, so there is nothing to conjugate
        qp.MultiRZ(theta, wires=list(active_wires))
        return

    def _apply_basis_change(sign):
        for wire, gate in zip(active_wires, active_gates, strict=True):
            if gate == "X":
                qp.Hadamard(wires=[wire])
            elif gate == "Y":
                qp.PPR_4(sign, "X", wires=[wire])

    qp.change_op_basis(
        functools.partial(_apply_basis_change, 1),
        qp.MultiRZ(theta, wires=list(active_wires)),
        functools.partial(_apply_basis_change, -1),
    )


add_decomps(PauliRot, _pauli_rot_decomposition)

add_decomps("Adjoint(PauliRot)", adjoint_rotation2)

add_decomps("Pow(PauliRot)", pow_rotation2)


def _ctrl_phase_shift_resource(subspace, n_control_wires, n_work_wires):
    if n_control_wires == 0:
        return {qp.PhaseShift: 1}
    return {
        qp.ctrl(
            qp.PhaseShift(Float, Wire[1]),
            control=Wire[n_control_wires],
            work_wires=Wire[n_work_wires],
            control_values=Bool[n_control_wires],
        ): 1,
        qp.X: 2 * (1 - subspace),
    }


def _ctrl_phase_shift(
    phi, target_wire, subspace, control_wires, control_values, work_wires
):  # pylint: disable=too-many-arguments
    r"""Implement a ((multi-)controlled) phase shift on the specified subspace of a
    target qubit/wire.

    Args:
        phi (float): Phase shift angle
        target_wire (~.Wires): the target wire to apply phase shift to.
        subspace (int): which subspace of the target wire the phase shift is applied to. 0 or 1.
        control_wires (WiresLike): the control wires
        control_values (Iterable[bool | int]): the control values.
        work_wires (WiresLike): the work wires

    Returns:
        float: any global phase produced in the process.

    The decomposition for subspace=1 always is a simple ``PhaseShift`` gate, or its controlled
    counterpart. The decomposition of a non-controlled phase shift for subspace=0 can be achieved
    in two ways: The first is to flip the angle of the phase shift and complementing it with a
    global phase, so that the (diagonal of the) gate is decomposed as

    .. math::

        (\exp(i\phi), 1) = (1, \exp(-i\phi)) (\exp(i\phi), \exp(i\phi)).

    The second is to conjugate a phase shift by Pauli-X operators on the same qubit, decomposing
    the gate matrix as

    / e^(i\phi)   0 \ -- / 0   1 \/ 1     0     \/ 0   1 \
    \     0       1 / -- \ 1   0 /\ 0 e^(i\phi) /\ 1   0 /.

    Without controls, the first approach is nicer, because global phases usually are free
    operations. With controls, however, this approach would lead to a controlled global
    phase, which is equivalent to a phase shift on top of the controlled phase shift. For the
    second approach, we may use
    `the ctrl(compute-uncompute) pattern <https://iopscience.iop.org/article/10.1088/2058-9565/aaa5cc>`__
    to avoid controlling the Pauli-X operations, yielding two non-controlled Pauli-X gates and the
    "main" controlled phase shift operation. We deem this decomposition to be better.

    """

    if subspace == 1:
        # If there are no control wires, we are dealing with the very first phase shift of
        # the decomposition, which should be adding projectors. So subspace should have been 0.
        assert len(control_wires) > 0
        qp.ctrl(
            qp.PhaseShift(phi, wires=target_wire),
            control=control_wires,
            control_values=control_values,
            work_wires=work_wires,
        )
        return 0.0

    if len(control_wires) == 0:
        # Flip angle for phase_shift(subspace=0) = phase_shift(subspace=1)*global_phase
        qp.PhaseShift(-phi, wires=target_wire)
        return -phi

    qp.X(target_wire)
    qp.ctrl(
        qp.PhaseShift(phi, wires=target_wire),
        control=control_wires,
        control_values=control_values,
        work_wires=work_wires,
    )
    qp.X(target_wire)
    return 0.0


# pylint: disable-next=unused-argument
def _decompose_pcphase_resource(phi: TensorLike, dim: int, wires: WiresLike):
    """Decompose the PCPhase operation into controlled phase shifts and Pauli-X gates."""

    num_wires = len(wires)
    gate_count = Counter()
    flipped, *powers_of_two = decomp_int_to_powers_of_two(dim, num_wires + 1)
    sigma = (-1) ** flipped
    powers_of_two = [sigma * val for val in powers_of_two]

    for i, c_i in enumerate(powers_of_two):
        if c_i != 0:
            subspace = int(c_i < 0)
            if flipped:
                subspace = 1 - subspace
            gate_count.update(
                _ctrl_phase_shift_resource(
                    subspace,
                    n_control_wires=i,
                    n_work_wires=num_wires - i - 1,
                )
            )

    gate_count[qp.GlobalPhase] += 1
    return dict(gate_count)


@register_resources(_decompose_pcphase_resource)
def _decompose_pcphase(phi: TensorLike, dim: int, wires: WiresLike):
    """Decompose the PCPhase operation into controlled phase shifts and Pauli-X gates."""

    # Use one more bit than there are wires, according to flipping all relevant bits for the
    # projector decomposition, or a global phase on the gate level. Afterwards, we have
    # dim <= 2**len(wires)=2**(n-1), which is a requirement of decomp_int_to_powers_of_two.
    flipped, *powers_of_two = decomp_int_to_powers_of_two(dim, len(wires) + 1)

    sigma = (-1) ** flipped

    # Overall global phase to implement I_N part of the generator
    global_phase = sigma * phi

    phi = 2 * sigma * phi

    # If in flipped (sigma=-1) mode, reverse the sign of all coefficients
    powers_of_two = [sigma * val for val in powers_of_two]

    assert len(powers_of_two) == len(wires)

    control_values = []
    for i, c_i in enumerate(powers_of_two):
        if c_i != 0:
            # Projector with rank 2**(n-1-i) needs to be added/subtracted
            subspace = int(c_i < 0)  # If c_i < 0, target |1> subspace, else target |0> subspace
            if flipped:  # Flip subspace if in flipped (sigma=-1) mode
                subspace = 1 - subspace
            global_phase += _ctrl_phase_shift(
                c_i * phi,
                target_wire=wires[i],
                subspace=subspace,
                control_wires=wires[:i],
                control_values=control_values,
                work_wires=wires[i + 1 :],  # Unused wires of PCPhase can be used as work wires
            )

        # The control value to be used on the current wire (it will be the same for all
        # subsequent operations) depends on whether we add or subtract next)
        d_i = next(iter(val for val in powers_of_two[i + 1 :] if val != 0), None)
        # If we add next, control into the |1> subspace, otherwise into |0> subspace
        # The control value is modified both if we are in the flipped global phase mode and if
        # the current loop iteration did not add a bit string/projector/gate (c_i==0)
        next_cval = d_i == 1
        if c_i == 0:
            next_cval = not next_cval
        if flipped:
            next_cval = not next_cval
        control_values.append(next_cval)

    qp.GlobalPhase(global_phase)


add_decomps(PCPhase, _decompose_pcphase)

add_decomps("Adjoint(PCPhase)", adjoint_rotation2)


# pylint: disable-next=unused-argument
def _isingxx_to_cnot_rx_cnot_resources(phi: TensorLike, wires: WiresLike | None = None):
    return {
        _change_op_basis_abstract(
            qp.CNOT(wires=Wire[2]),
            RX(Float, wires=Wire[1]),
            qp.CNOT(wires=Wire[2]),
        ): 1
    }


@register_resources(_isingxx_to_cnot_rx_cnot_resources)
def _isingxx_to_cnot_rx_cnot(phi: TensorLike, wires: WiresLike, **__):
    qp.change_op_basis(qp.CNOT(wires=wires), RX(phi, wires=[wires[0]]), qp.CNOT(wires=wires))


# pylint: disable-next=unused-argument
def _isingxx_to_ppr_resource(phi: TensorLike, wires: WiresLike | None = None):
    return {PauliRot(Float, pauli_word="XX", wires=Wire[2]): 1}


@register_resources(_isingxx_to_ppr_resource)
def _isingxx_to_ppr(phi: TensorLike, wires: WiresLike, **_):
    qp.PauliRot(phi, "XX", wires=wires)


add_decomps(IsingXX, _isingxx_to_cnot_rx_cnot, _isingxx_to_ppr)

add_decomps("Adjoint(IsingXX)", adjoint_rotation2)

add_decomps("Pow(IsingXX)", pow_rotation2)


# pylint: disable-next=unused-argument
def _isingyy_to_cy_ry_cy_resources(phi: TensorLike, wires: WiresLike | None = None):
    return {
        _change_op_basis_abstract(
            qp.CY(wires=Wire[2]),
            RY(Float, wires=Wire[1]),
            qp.CY(wires=Wire[2]),
        ): 1
    }


@register_resources(_isingyy_to_cy_ry_cy_resources)
def _isingyy_to_cy_ry_cy(phi: TensorLike, wires: WiresLike, **__):
    qp.change_op_basis(qp.CY(wires=wires), RY(phi, wires=[wires[0]]), qp.CY(wires=wires))


# pylint: disable-next=unused-argument
def _isingyy_to_ppr_resource(phi: TensorLike, wires: WiresLike | None = None):
    return {PauliRot(Float, pauli_word="YY", wires=Wire[2]): 1}


@register_resources(_isingyy_to_ppr_resource)
def _isingyy_to_ppr(phi: TensorLike, wires: WiresLike, **_):
    qp.PauliRot(phi, "YY", wires=wires)


add_decomps(IsingYY, _isingyy_to_cy_ry_cy, _isingyy_to_ppr)

add_decomps("Adjoint(IsingYY)", adjoint_rotation2)

add_decomps("Pow(IsingYY)", pow_rotation2)


# pylint: disable-next=unused-argument
def _isingzz_to_cnot_rz_cnot_resources(phi: TensorLike, wires: WiresLike | None = None):
    return {
        _change_op_basis_abstract(
            qp.CNOT(wires=Wire[2]),
            RZ(Float, wires=Wire[1]),
            qp.CNOT(wires=Wire[2]),
        ): 1
    }


@register_resources(_isingzz_to_cnot_rz_cnot_resources)
def _isingzz_to_cnot_rz_cnot(phi: TensorLike, wires: WiresLike, **__):
    r"""Expressing ``IsingZZ``  via :func:`~.change_op_basis` (instead of three bare gates) lets
    PennyLane's generic ``C(ChangeOpBasis)`` rule automatically control it in an efficient way."""
    qp.change_op_basis(qp.CNOT(wires=wires), RZ(phi, wires=[wires[1]]), qp.CNOT(wires=wires))


# pylint: disable-next=unused-argument
def _isingzz_to_ppr_resource(phi: TensorLike, wires: WiresLike | None = None):
    return {PauliRot(Float, pauli_word="ZZ", wires=Wire[2]): 1}


@register_resources(_isingzz_to_ppr_resource)
def _isingzz_to_ppr(phi: TensorLike, wires: WiresLike, **_):
    qp.PauliRot(phi, "ZZ", wires=wires)


def _controlled_ising_zz_resource(base, control_wires, control_values, work_wires, work_wire_type):
    return {
        qp.CNOT: 2,
        qp.ctrl(
            qp.RZ(Float, wires=Wire[1]),
            control=Wire[len(control_wires)],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 1,
    }


@qp.register_resources(_controlled_ising_zz_resource)
def _controlled_ising_zz(base, control_wires, control_values, work_wires, work_wire_type):
    r"""``ctrl_c[IsingZZ(phi)_{a,b}] = CNOT_{a,b} ctrl_c[RZ(phi)_b] CNOT_{a,b}``.

    ``IsingZZ(phi)`` phases the parity of its two wires, so copying that parity onto ``b`` with
    a CNOT turns it into a single-qubit ``RZ(phi)``. The CNOTs need no control: they cancel on
    the ``|0>_c`` branch. Verified exactly (to 0.0) against ``qp.ctrl(qp.IsingZZ(...))`` for one
    and two control wires.
    """
    a, b = base.wires
    qp.CNOT(wires=[a, b])
    qp.ctrl(
        qp.RZ(base.data[0], wires=b),
        control=control_wires,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    qp.CNOT(wires=[a, b])


controlled_ising_zz = flip_zero_control(_controlled_ising_zz)

add_decomps(IsingZZ, _isingzz_to_cnot_rz_cnot, _isingzz_to_ppr)

add_decomps("Adjoint(IsingZZ)", adjoint_rotation2)

add_decomps("Pow(IsingZZ)", pow_rotation2)

add_decomps("C(IsingZZ)", controlled_ising_zz)


# pylint: disable-next=unused-argument
def _isingxy_to_h_cy_resources(phi: TensorLike, wires: WiresLike | None = None):
    # ``Prod2`` takes its operands in matrix order, i.e. reversed relative to the order in which
    # the decomposition below applies them.
    basis = Prod2((qp.CY(wires=Wire[2]), Hadamard(wires=Wire[1])))
    unbasis = Prod2((Hadamard(wires=Wire[1]), qp.CY(wires=Wire[2])))
    return {
        _change_op_basis_abstract(
            basis,
            Prod2((RX(Float, wires=Wire[1]), RY(Float, wires=Wire[1]))),
            unbasis,
        ): 1
    }


@register_resources(_isingxy_to_h_cy_resources)
def _isingxy_to_h_cy(phi: TensorLike, wires: WiresLike, **__):
    def _to_basis():
        Hadamard(wires=[wires[0]])
        qp.CY(wires=wires)

    def _rotations():
        RY(phi / 2, wires=[wires[0]])
        RX(-phi / 2, wires=[wires[1]])

    def _from_basis():
        qp.CY(wires=wires)
        Hadamard(wires=[wires[0]])

    qp.change_op_basis(_to_basis, _rotations, _from_basis)


add_decomps(IsingXY, _isingxy_to_h_cy)

add_decomps("Adjoint(IsingXY)", adjoint_rotation2)

add_decomps("Pow(IsingXY)", pow_rotation2)


def _pswap_to_swap_cnot_phaseshift_cnot_resources():
    return {qp.SWAP: 1, qp.CNOT: 2, PhaseShift: 1}


@register_resources(_pswap_to_swap_cnot_phaseshift_cnot_resources)
def _pswap_to_swap_cnot_phaseshift_cnot(phi: TensorLike, wires: WiresLike, **__):
    qp.SWAP(wires=wires)
    qp.CNOT(wires=wires)
    PhaseShift(phi, wires=[wires[1]])
    qp.CNOT(wires=wires)


def _pswap_to_ppr_resources():
    return {
        qp.PPR_4(-1, pauli_word="XX", wires=Wire[2]): 1,
        qp.PPR_4(-1, pauli_word="YY", wires=Wire[2]): 1,
        PauliRot(Float, pauli_word="ZZ", wires=Wire[2]): 1,
        qp.GlobalPhase: 1,
    }


@register_resources(_pswap_to_ppr_resources)
def _pswap_to_ppr(phi: TensorLike, wires: WiresLike, **__):
    qp.PPR_4(-1, pauli_word="YY", wires=wires)
    qp.PPR_4(-1, pauli_word="XX", wires=wires)
    PauliRot(phi - np.pi / 2, pauli_word="ZZ", wires=wires)
    qp.GlobalPhase(np.pi / 4 - phi / 2)


add_decomps(PSWAP, _pswap_to_swap_cnot_phaseshift_cnot, _pswap_to_ppr)

add_decomps("Adjoint(PSWAP)", adjoint_rotation)


def _cphaseshift00_resources():
    return {PauliX: 4, PhaseShift: 3, qp.CNOT: 2}


@register_resources(_cphaseshift00_resources)
def _cphaseshift00(phi: TensorLike, wires: WiresLike, **__):
    PauliX(wires[0])
    PauliX(wires[1])
    PhaseShift(phi / 2, wires=[wires[0]])
    PhaseShift(phi / 2, wires=[wires[1]])
    qp.CNOT(wires=wires)
    PhaseShift(-phi / 2, wires=[wires[1]])
    qp.CNOT(wires=wires)
    PauliX(wires[1])
    PauliX(wires[0])


add_decomps(CPhaseShift00, _cphaseshift00)

add_decomps("Adjoint(CPhaseShift00)", adjoint_rotation)

add_decomps("Pow(CPhaseShift00)", pow_rotation)


def _cphaseshift01_resources():
    return {PauliX: 2, PhaseShift: 3, qp.CNOT: 2}


@register_resources(_cphaseshift01_resources)
def _cphaseshift01(phi: TensorLike, wires: WiresLike, **__):
    PauliX(wires[0])
    PhaseShift(phi / 2, wires=[wires[0]])
    PhaseShift(phi / 2, wires=[wires[1]])
    qp.CNOT(wires=wires)
    PhaseShift(-phi / 2, wires=[wires[1]])
    qp.CNOT(wires=wires)
    PauliX(wires[0])


add_decomps(CPhaseShift01, _cphaseshift01)

add_decomps("Adjoint(CPhaseShift01)", adjoint_rotation)

add_decomps("Pow(CPhaseShift01)", pow_rotation)


def _cphaseshift10_resources():
    return {PauliX: 2, PhaseShift: 3, qp.CNOT: 2}


@register_resources(_cphaseshift10_resources)
def _cphaseshift10(phi: TensorLike, wires: WiresLike, **__):
    PauliX(wires[1])
    PhaseShift(phi / 2, wires=[wires[0]])
    PhaseShift(phi / 2, wires=[wires[1]])
    qp.CNOT(wires=wires)
    PhaseShift(-phi / 2, wires=[wires[1]])
    qp.CNOT(wires=wires)
    PauliX(wires[1])


add_decomps(CPhaseShift10, _cphaseshift10)

add_decomps("Adjoint(CPhaseShift10)", adjoint_rotation)

add_decomps("Pow(CPhaseShift10)", pow_rotation)
