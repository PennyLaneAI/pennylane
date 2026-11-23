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

from collections import Counter

import numpy as np

from pennylane import capture, compiler, control_flow, math, ops
from pennylane.capture.autograph import disable_autograph
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition_rules.op_math.adjoint import _adjoint_negate_angle
from pennylane.decomposition_rules.op_math.adjoint2 import (
    _adjoint_negate_angle as adjoint_rotation2,
)
from pennylane.decomposition_rules.op_math.controlled2 import flip_zero_control
from pennylane.decomposition_rules.op_math.pow2 import _pow_scale_angle as pow_rotation2
from pennylane.math.decomposition import decomp_int_to_powers_of_two
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
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
from pennylane.ops.subcircuit import subcircuit
from pennylane.typing import Bool, Float, TensorLike, Wire
from pennylane.wires import WiresLike


@subcircuit(static_argnames=("reverse",))
@register_resources(lambda wires, **_: {ops.CNOT: len(wires) - 1})
def _MultiRZLadder(wires: WiresLike, reverse: bool):
    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
    num_wires = len(wires)
    start, stop, step = (1, num_wires, 1) if reverse else (num_wires - 1, 0, -1)

    def _cnots(index):
        ops.CNOT(wires=(wires[index], wires[index - 1]))

    control_flow.for_loop(start, stop, step)(_cnots)()


# pylint: disable-next=unused-argument
def _multi_rz_standard_resources(theta: TensorLike, wires: WiresLike):
    num_wires = len(wires)
    if num_wires == 1:
        return {RZ: 1}
    ladder = _MultiRZLadder(wires=Wire[num_wires], reverse=False)
    unladder = _MultiRZLadder(wires=Wire[num_wires], reverse=True)
    return {_change_op_basis_abstract(ladder, RZ(Float, wires=Wire[1]), unladder): 1}


@register_resources(_multi_rz_standard_resources, name="cnot_ladder")
def _multi_rz_standard(theta: TensorLike, wires: WiresLike):
    """CNOT ladder around an ``RZ``, so a control applies only to the rotation."""
    if len(wires) == 1:
        RZ(theta, wires=wires[0])
        return

    ops.change_op_basis(
        _MultiRZLadder(wires=wires, reverse=False),
        RZ(theta, wires=wires[0]),
        _MultiRZLadder(wires=wires, reverse=True),
    )


# pylint: disable-next=unused-argument
def _multi_rz_to_ppr_resources(theta: TensorLike, wires: WiresLike):
    num_wires = len(wires)
    return {PauliRot(Float, pauli_word="Z" * num_wires, wires=Wire[num_wires]): 1}


@register_resources(_multi_rz_to_ppr_resources, name="to_paulirot")
def _multi_rz_to_ppr(theta: TensorLike, wires: WiresLike):
    PauliRot(theta, "Z" * len(wires), wires=wires)


add_decomps(MultiRZ, _multi_rz_standard, _multi_rz_to_ppr)

add_decomps("Adjoint(MultiRZ)", adjoint_rotation2)

add_decomps("Pow(MultiRZ)", pow_rotation2)


def _pauli_rot_basis_resources(sign, pauli_word, **_):
    resources = {}
    if pauli_word.count("X"):
        resources[Hadamard] = pauli_word.count("X")
    if pauli_word.count("Y"):
        resources[ops.PPR_4(sign, "X", wires=Wire[1])] = pauli_word.count("Y")
    return resources


@subcircuit(static_argnames=("sign", "pauli_word"))
@register_resources(_pauli_rot_basis_resources)
@disable_autograph
def _PauliRotBasis(sign: int, pauli_word: str, wires: WiresLike):
    for wire, gate in zip(wires, pauli_word, strict=True):
        if gate == "X":
            # A Y-axis PPR_4 would decompose back into PauliRot and create a cycle.
            Hadamard(wires=[wire])
        elif gate == "Y":
            ops.PPR_4(sign, "X", wires=[wire])


def _pauli_rot_standard_resources(theta, pauli_word, wires):  # pylint: disable=unused-argument
    if set(pauli_word) == {"I"}:
        return {ops.GlobalPhase: 1}
    active_word = pauli_word.replace("I", "")
    num_active_wires = len(active_word)
    if set(active_word) == {"Z"}:
        # a pure-Z word needs no basis change, so there is nothing to conjugate
        return {MultiRZ(Float, Wire[num_active_wires]): 1}

    return {
        _change_op_basis_abstract(
            _PauliRotBasis(1, active_word, wires=Wire[num_active_wires]),
            MultiRZ(Float, Wire[num_active_wires]),
            _PauliRotBasis(-1, active_word, wires=Wire[num_active_wires]),
        ): 1
    }


@register_resources(_pauli_rot_standard_resources, name="basis_change_multirz")
@disable_autograph
def _pauli_rot_standard(theta: TensorLike, pauli_word: str, wires: WiresLike):
    if set(pauli_word) == {"I"}:
        ops.GlobalPhase(theta / 2)
        return
    active_wires, active_gates = zip(
        *[(wire, gate) for wire, gate in zip(wires, pauli_word, strict=True) if gate != "I"],
        strict=True,
    )

    if set(active_gates) == {"Z"}:
        # a pure-Z word needs no basis change, so there is nothing to conjugate
        MultiRZ(theta, wires=list(active_wires))
        return

    ops.change_op_basis(
        _PauliRotBasis(1, "".join(active_gates), wires=active_wires),
        MultiRZ(theta, wires=list(active_wires)),
        _PauliRotBasis(-1, "".join(active_gates), wires=active_wires),
    )


add_decomps(PauliRot, _pauli_rot_standard)

add_decomps("Adjoint(PauliRot)", adjoint_rotation2)

add_decomps("Pow(PauliRot)", pow_rotation2)


def _ctrl_phase_shift_resource(subspace, n_control_wires, n_work_wires):
    if n_control_wires == 0:
        return {PhaseShift: 1}
    return {
        ops.ctrl(
            PhaseShift(Float, Wire[1]),
            control=Wire[n_control_wires],
            work_wires=Wire[n_work_wires],
            control_values=Bool[n_control_wires],
        ): 1,
        PauliX: 2 * (1 - subspace),
    }


# pylint: disable-next=too-many-arguments
def _ctrl_phase_shift(phi, target_wire, subspace, control_wires, control_values, work_wires):
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
        ops.ctrl(
            PhaseShift(phi, wires=target_wire),
            control=control_wires,
            control_values=control_values,
            work_wires=work_wires,
        )
        return 0.0

    if len(control_wires) == 0:
        # Flip angle for phase_shift(subspace=0) = phase_shift(subspace=1)*global_phase
        PhaseShift(-phi, wires=target_wire)
        return -phi

    PauliX(target_wire)
    ops.ctrl(
        PhaseShift(phi, wires=target_wire),
        control=control_wires,
        control_values=control_values,
        work_wires=work_wires,
    )
    PauliX(target_wire)
    return 0.0


# pylint: disable-next=unused-argument
def _decompose_pcphase_standard_resources(phi: TensorLike, dim: int, wires: WiresLike):
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

    gate_count[ops.GlobalPhase] += 1
    return dict(gate_count)


@register_resources(_decompose_pcphase_standard_resources, name="controlled_phase_shifts")
def _decompose_pcphase_standard(phi: TensorLike, dim: int, wires: WiresLike):
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

    ops.GlobalPhase(global_phase)


add_decomps(PCPhase, _decompose_pcphase_standard)

add_decomps("Adjoint(PCPhase)", adjoint_rotation2)


# pylint: disable-next=unused-argument
def _isingxx_to_cnot_rx_cnot_resources(phi: TensorLike, wires: WiresLike | None = None):
    return {
        _change_op_basis_abstract(
            ops.CNOT(wires=Wire[2]),
            RX(Float, wires=Wire[1]),
            ops.CNOT(wires=Wire[2]),
        ): 1
    }


@register_resources(_isingxx_to_cnot_rx_cnot_resources, name="to_cnot_rx_cnot")
def _isingxx_to_cnot_rx_cnot(phi: TensorLike, wires: WiresLike, **__):
    ops.change_op_basis(ops.CNOT(wires=wires), RX(phi, wires=[wires[0]]), ops.CNOT(wires=wires))


# pylint: disable-next=unused-argument
def _isingxx_to_ppr_resources(phi: TensorLike, wires: WiresLike | None = None):
    return {PauliRot(Float, pauli_word="XX", wires=Wire[2]): 1}


@register_resources(_isingxx_to_ppr_resources, name="to_paulirot")
def _isingxx_to_ppr(phi: TensorLike, wires: WiresLike, **_):
    PauliRot(phi, "XX", wires=wires)


add_decomps(IsingXX, _isingxx_to_cnot_rx_cnot, _isingxx_to_ppr)

add_decomps("Adjoint(IsingXX)", adjoint_rotation2)

add_decomps("Pow(IsingXX)", pow_rotation2)


# pylint: disable-next=unused-argument
def _isingyy_to_cy_ry_cy_resources(phi: TensorLike, wires: WiresLike | None = None):
    return {
        _change_op_basis_abstract(
            ops.CY(wires=Wire[2]),
            RY(Float, wires=Wire[1]),
            ops.CY(wires=Wire[2]),
        ): 1
    }


@register_resources(_isingyy_to_cy_ry_cy_resources, name="to_cy_ry_cy")
def _isingyy_to_cy_ry_cy(phi: TensorLike, wires: WiresLike, **__):
    ops.change_op_basis(ops.CY(wires=wires), RY(phi, wires=[wires[0]]), ops.CY(wires=wires))


# pylint: disable-next=unused-argument
def _isingyy_to_ppr_resources(phi: TensorLike, wires: WiresLike | None = None):
    return {PauliRot(Float, pauli_word="YY", wires=Wire[2]): 1}


@register_resources(_isingyy_to_ppr_resources, name="to_paulirot")
def _isingyy_to_ppr(phi: TensorLike, wires: WiresLike, **_):
    PauliRot(phi, "YY", wires=wires)


add_decomps(IsingYY, _isingyy_to_cy_ry_cy, _isingyy_to_ppr)

add_decomps("Adjoint(IsingYY)", adjoint_rotation2)

add_decomps("Pow(IsingYY)", pow_rotation2)


# pylint: disable-next=unused-argument
def _isingzz_to_cnot_rz_cnot_resources(phi: TensorLike, wires: WiresLike | None = None):
    return {
        _change_op_basis_abstract(
            ops.CNOT(wires=Wire[2]),
            RZ(Float, wires=Wire[1]),
            ops.CNOT(wires=Wire[2]),
        ): 1
    }


@register_resources(_isingzz_to_cnot_rz_cnot_resources, name="to_cnot_rz_cnot")
def _isingzz_to_cnot_rz_cnot(phi: TensorLike, wires: WiresLike, **__):
    r"""Expressing ``IsingZZ``  via :func:`~.change_op_basis` (instead of three bare gates) lets
    PennyLane's generic ``C(ChangeOpBasis)`` rule automatically control it in an efficient way."""
    ops.change_op_basis(ops.CNOT(wires=wires), RZ(phi, wires=[wires[1]]), ops.CNOT(wires=wires))


# pylint: disable-next=unused-argument
def _isingzz_to_ppr_resources(phi: TensorLike, wires: WiresLike | None = None):
    return {PauliRot(Float, pauli_word="ZZ", wires=Wire[2]): 1}


@register_resources(_isingzz_to_ppr_resources, name="to_paulirot")
def _isingzz_to_ppr(phi: TensorLike, wires: WiresLike, **_):
    PauliRot(phi, "ZZ", wires=wires)


def _controlled_ising_zz_standard_resources(
    base,  # pylint: disable=unused-argument
    control_wires,
    control_values,  # pylint: disable=unused-argument
    work_wires,
    work_wire_type,
):
    return {
        ops.CNOT: 2,
        ops.ctrl(
            RZ(Float, wires=Wire[1]),
            control=Wire[len(control_wires)],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 1,
    }


@register_resources(_controlled_ising_zz_standard_resources, name="parity_controlled_rz")
# pylint: disable-next=unused-argument
def _controlled_ising_zz_standard(base, control_wires, control_values, work_wires, work_wire_type):
    r"""``ctrl_c[IsingZZ(phi)_{a,b}] = CNOT_{a,b} ctrl_c[RZ(phi)_b] CNOT_{a,b}``.

    ``IsingZZ(phi)`` phases the parity of its two wires, so copying that parity onto ``b`` with
    a CNOT turns it into a single-qubit ``RZ(phi)``. The CNOTs need no control: they cancel on
    the ``|0>_c`` branch. Verified exactly (to 0.0) against ``qp.ctrl(qp.IsingZZ(...))`` for one
    and two control wires.
    """
    a, b = base.wires
    ops.CNOT(wires=[a, b])
    ops.ctrl(
        RZ(base.data[0], wires=b),
        control=control_wires,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    ops.CNOT(wires=[a, b])


controlled_ising_zz = flip_zero_control(_controlled_ising_zz_standard, name="parity_controlled_rz")


add_decomps(IsingZZ, _isingzz_to_cnot_rz_cnot, _isingzz_to_ppr)

add_decomps("Adjoint(IsingZZ)", adjoint_rotation2)

add_decomps("Pow(IsingZZ)", pow_rotation2)

add_decomps("C(IsingZZ)", controlled_ising_zz)


@subcircuit(static_argnames=("reverse",))
@register_resources(lambda **_: {Hadamard: 1, ops.CY: 1})
def _IsingXYBasis(wires: WiresLike, reverse: bool):
    if not reverse:
        Hadamard(wires[0])
    ops.CY(wires)
    if reverse:
        Hadamard(wires[0])


@subcircuit(dynamic_argnames=("phi",), arg_specs={"phi": Float, "wires": Wire[2]})
@register_resources({RX: 1, RY: 1})
def _IsingXYRotations(phi: TensorLike, wires: WiresLike):
    RY(phi / 2, wires[0])
    RX(-phi / 2, wires[1])


# pylint: disable-next=unused-argument
def _isingxy_to_h_cy_resources(phi: TensorLike, wires: WiresLike | None = None):
    return {
        _change_op_basis_abstract(
            _IsingXYBasis(wires=Wire[2], reverse=False),
            _IsingXYRotations(Float, wires=Wire[2]),
            _IsingXYBasis(wires=Wire[2], reverse=True),
        ): 1
    }


@register_resources(_isingxy_to_h_cy_resources, name="to_h_cy")
def _isingxy_to_h_cy(phi: TensorLike, wires: WiresLike, **__):
    ops.change_op_basis(
        _IsingXYBasis(wires=wires, reverse=False),
        _IsingXYRotations(phi, wires=wires),
        _IsingXYBasis(wires=wires, reverse=True),
    )


add_decomps(IsingXY, _isingxy_to_h_cy)

add_decomps("Adjoint(IsingXY)", adjoint_rotation2)

add_decomps("Pow(IsingXY)", pow_rotation2)


def _pswap_to_swap_cnot_phaseshift_cnot_resources():
    return {ops.SWAP: 1, ops.CNOT: 2, PhaseShift: 1}


@register_resources(
    _pswap_to_swap_cnot_phaseshift_cnot_resources, name="to_swap_cnot_phaseshift_cnot"
)
def _pswap_to_swap_cnot_phaseshift_cnot(phi: TensorLike, wires: WiresLike, **__):
    ops.SWAP(wires=wires)
    ops.CNOT(wires=wires)
    PhaseShift(phi, wires=[wires[1]])
    ops.CNOT(wires=wires)


def _pswap_to_ppr_resources():
    return {
        ops.PPR_4(-1, pauli_word="XX", wires=Wire[2]): 1,
        ops.PPR_4(-1, pauli_word="YY", wires=Wire[2]): 1,
        PauliRot(Float, pauli_word="ZZ", wires=Wire[2]): 1,
        ops.GlobalPhase: 1,
    }


@register_resources(_pswap_to_ppr_resources, name="to_ppr")
def _pswap_to_ppr(phi: TensorLike, wires: WiresLike, **__):
    ops.PPR_4(-1, pauli_word="YY", wires=wires)
    ops.PPR_4(-1, pauli_word="XX", wires=wires)
    PauliRot(phi - np.pi / 2, pauli_word="ZZ", wires=wires)
    ops.GlobalPhase(np.pi / 4 - phi / 2)


add_decomps(PSWAP, _pswap_to_swap_cnot_phaseshift_cnot, _pswap_to_ppr)

add_decomps("Adjoint(PSWAP)", _adjoint_negate_angle)


def _cphaseshift00_to_x_phaseshift_cnot_resources(*_, **__):
    return {PauliX: 4, PhaseShift: 3, ops.CNOT: 2}


@register_resources(_cphaseshift00_to_x_phaseshift_cnot_resources, name="to_x_phaseshift_cnot")
def _cphaseshift00_to_x_phaseshift_cnot(phi: TensorLike, wires: WiresLike, **__):
    PauliX(wires[0])
    PauliX(wires[1])
    PhaseShift(phi / 2, wires=[wires[0]])
    PhaseShift(phi / 2, wires=[wires[1]])
    ops.CNOT(wires=wires)
    PhaseShift(-phi / 2, wires=[wires[1]])
    ops.CNOT(wires=wires)
    PauliX(wires[1])
    PauliX(wires[0])


add_decomps(CPhaseShift00, _cphaseshift00_to_x_phaseshift_cnot)

add_decomps("Adjoint(CPhaseShift00)", adjoint_rotation2)

add_decomps("Pow(CPhaseShift00)", pow_rotation2)


def _cphaseshift01_to_x_phaseshift_cnot_resources(*_, **__):
    return {PauliX: 2, PhaseShift: 3, ops.CNOT: 2}


@register_resources(_cphaseshift01_to_x_phaseshift_cnot_resources, name="to_x_phaseshift_cnot")
def _cphaseshift01_to_x_phaseshift_cnot(phi: TensorLike, wires: WiresLike, **__):
    PauliX(wires[0])
    PhaseShift(phi / 2, wires=[wires[0]])
    PhaseShift(phi / 2, wires=[wires[1]])
    ops.CNOT(wires=wires)
    PhaseShift(-phi / 2, wires=[wires[1]])
    ops.CNOT(wires=wires)
    PauliX(wires[0])


add_decomps(CPhaseShift01, _cphaseshift01_to_x_phaseshift_cnot)

add_decomps("Adjoint(CPhaseShift01)", adjoint_rotation2)

add_decomps("Pow(CPhaseShift01)", pow_rotation2)


def _cphaseshift10_to_x_phaseshift_cnot_resources(*_, **__):
    return {PauliX: 2, PhaseShift: 3, ops.CNOT: 2}


@register_resources(_cphaseshift10_to_x_phaseshift_cnot_resources, name="to_x_phaseshift_cnot")
def _cphaseshift10_to_x_phaseshift_cnot(phi: TensorLike, wires: WiresLike, **__):
    PauliX(wires[1])
    PhaseShift(phi / 2, wires=[wires[0]])
    PhaseShift(phi / 2, wires=[wires[1]])
    ops.CNOT(wires=wires)
    PhaseShift(-phi / 2, wires=[wires[1]])
    ops.CNOT(wires=wires)
    PauliX(wires[1])


add_decomps(CPhaseShift10, _cphaseshift10_to_x_phaseshift_cnot)

add_decomps("Adjoint(CPhaseShift10)", adjoint_rotation2)

add_decomps("Pow(CPhaseShift10)", pow_rotation2)
