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
"""Scheme-agnostic scaffolding shared by the fragmented-Hamiltonian Trotter templates
(:class:`~.TrotterCDF` and :class:`~.TrotterCGF`)."""

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.ops import CNOT, RZ, IsingZZ, cond

# pylint: disable=too-many-arguments


def _emit_one_body_rz(angle, target_wire, control_wires, double_phase):
    r"""Emit a one-body ``RZ`` rotation for the base, double-phase, or genuine controlled circuit.

    * No control wire: a plain ``RZ(angle)``.
    * Controlled, ``double_phase=True``: the CNOT-sandwich
        .. code-block::

                c: ─╭●────────╭●─┤
            wires: ─╰X──RZ(ϕ)─╰X─┤

      at the full angle (see Fig. 6 in https://arxiv.org/abs/2506.15784). This results in :math:`\text{diag}(U, U^\dagger)` overall, such that the relative phase between both branches is 2t (hence "double-phase").
    * Controlled, ``double_phase=False``: a genuine controlled-``RZ`` (the standard ``CRZ``
      decomposition), leading to a genuine controlled evolution :math:`\text{diag}(1, U)`.
    """
    if len(control_wires) == 0:
        RZ(angle, target_wire)
        return
    control_wire = control_wires[0]
    if double_phase:
        CNOT([control_wire, target_wire])
        RZ(angle, target_wire)
        CNOT([control_wire, target_wire])
        return
    RZ(angle / 2, target_wire)
    CNOT([control_wire, target_wire])
    RZ(-angle / 2, target_wire)
    CNOT([control_wire, target_wire])


def _emit_two_body_isingzz(angle, wire_a, wire_b, control_wires, double_phase):
    """Emit a single two-body ``IsingZZ`` for the base, double-phase, or genuine circuit.

    For the double-phase circuit the shared ``CNOT`` sandwich is applied once per diagonal
    block by the caller, so here we only emit the bare ``IsingZZ(angle)``. For the genuine
    controlled circuit every ``IsingZZ`` is individually controlled (a genuine
    controlled-``IsingZZ``) at the full ``angle``, so the control-0 branch is the identity.
    """
    if len(control_wires) == 0 or double_phase:
        IsingZZ(angle, [wire_a, wire_b])
        return
    control_wire = control_wires[0]
    CNOT([wire_a, wire_b])
    RZ(angle / 2, wire_b)
    CNOT([control_wire, wire_b])
    RZ(-angle / 2, wire_b)
    CNOT([control_wire, wire_b])
    CNOT([wire_a, wire_b])


def _run_trotter_steps(
    evolution_time,
    num_trotter_steps,
    hamiltonian,
    wires,
    control_wires,
    double_phase=False,
    *,
    apply_system_basis_rotation,
    apply_two_body_diagonal,
    apply_one_body_diagonal,
    merge_leaves,
    transpose_leaf,
):
    r"""Emit the second-order Trotter step sequence and the trailing basis rotation.

    This is the scheme-agnostic backbone shared by :class:`~.TrotterCDF` and
    :class:`~.TrotterCGF`. The scheme-specific behaviour (tensor ranks, loop
    nesting, angle prefactors) is injected through the keyword-only callables.

    The basis rotations are always uncontrolled and are time-independent; only the
    diagonal-rotation angles (linear in ``evolution_time``) and the way each diagonal
    rotation is controlled depend on ``control_wires``/``double_phase``:

    * Base (``control_wires`` empty): the plain :math:`e^{-iHt}` circuit.
    * Genuine controlled (``double_phase=False``): every diagonal rotation is
      individually controlled at the full angle, so the control-0 branch is the
      identity and the circuit is a genuine controlled-:math:`e^{-iHt}`.
    * Double-phase controlled (``double_phase=True``, Fig. 6 of `arXiv:2506.15784
      <https://arxiv.org/abs/2506.15784>`__): each diagonal block is CNOT-sandwiched by
      the control wire at the full angle, giving the full-time :math:`e^{\mp i H t}`
      Hadamard-test branches (this reproduces the original ``trotter_fragmented`` circuit).

    Args:
        evolution_time (float): total evolution time ``t``.
        num_trotter_steps (int): number of second-order Trotter steps (``> 0``).
        hamiltonian (dict): fragmented Hamiltonian data.
        wires (Wires): system wires.
        control_wires (Wires): control wires. Empty for the base (uncontrolled)
            circuit; a single wire for the controlled circuits.
        double_phase (bool): whether the controlled circuit is the double-phase
            (Fig. 6) construction (``True``) or a genuine controlled unitary (``False``).
        apply_system_basis_rotation (callable): ``(U, wires) -> None``.
        apply_two_body_diagonal (callable):
            ``(Z, wires, first_order_time_step, control_wires, double_phase) -> None``.
        apply_one_body_diagonal (callable):
            ``(Z, wires, first_order_time_step, control_wires, double_phase) -> None``.
        merge_leaves (callable): ``(U_prev, U_curr) -> U``.
        transpose_leaf (callable): ``(U) -> U``.
    """
    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        if len(control_wires) > 0:
            control_wires = math.array(control_wires, like="jax")

    second_order_time_step = evolution_time / num_trotter_steps
    first_order_time_step = second_order_time_step / 2

    num_two_body_fragments = hamiltonian["leaf_tensors"].shape[0] - 1

    def _trotter_step(step_idx, hamiltonian):
        # ``hamiltonian`` is carried through the for-loop (rather than closed over)
        # so the traced tensors remain valid loop-body inputs under jax capture.
        U_tensor = hamiltonian["leaf_tensors"]
        Z_tensor = hamiltonian["core_tensors"]

        def two_body_fragments(fragment_idx, prev_fragment_idx):
            # The first fragment of the circuit (prev_fragment_idx < 0) uses its own leaf; later
            # fragments merge with the previous fragment's leaf so consecutive basis rotations
            # telescope into a single one. This is classical array selection (no quantum ops in
            # either branch), branching on the loop-carried ``prev_fragment_idx``.
            U = cond(
                prev_fragment_idx < 0,
                lambda: U_tensor[fragment_idx],
                lambda: merge_leaves(U_tensor[prev_fragment_idx], U_tensor[fragment_idx]),
            )()
            Z = Z_tensor[fragment_idx]
            apply_system_basis_rotation(U, wires)
            apply_two_body_diagonal(Z, wires, first_order_time_step, control_wires, double_phase)
            return fragment_idx

        def one_body_fragment():
            U_one = U_tensor[0]
            U = merge_leaves(U_tensor[num_two_body_fragments], U_one)
            apply_system_basis_rotation(U, wires)
            apply_one_body_diagonal(
                Z_tensor[0], wires, first_order_time_step, control_wires, double_phase
            )

        prev_fragment_idx_forward = math.sign(2 * step_idx - 1)
        for_loop(1, num_two_body_fragments + 1)(two_body_fragments)(prev_fragment_idx_forward)

        one_body_fragment()

        prev_fragment_idx_backward = 0
        for_loop(num_two_body_fragments, 0, -1)(two_body_fragments)(prev_fragment_idx_backward)

        return hamiltonian

    for_loop(num_trotter_steps)(_trotter_step)(hamiltonian)

    very_last_U = transpose_leaf(hamiltonian["leaf_tensors"][1])
    apply_system_basis_rotation(very_last_U, wires)
