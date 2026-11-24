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
"""Shared functionality for fragmented-Hamiltonian Trotter templates
(:class:`~.TrotterCDF` and :class:`~.TrotterCGF`)."""

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.ops import CRZ, RZ, IsingZZ


def _emit_one_body_rz(angle, target_wire, control_wires, double_phase):
    r"""Emit a one-body ``RZ`` rotation for the base, double-phase, or genuine controlled circuit.

    * No control wire: a plain ``RZ(angle)``.
    * Controlled, ``double_phase=True``: ``IsingZZ(angle, [control, target])``, i.e. the same
      ``CNOT; RZ; CNOT`` sandwich as Fig. 6 in https://arxiv.org/abs/2506.15784.
      This results in :math:`\text{diag}(U, U^\dagger)` overall, such that the relative phase
      between both branches is 2t (hence "double-phase").
    * Controlled, ``double_phase=False``: ``CRZ(angle, [control, target])`` for a genuine
      controlled-:math:`e^{-iHt}` (``diag(1, U)`` on the control/target subspace).
    """
    if len(control_wires) == 0:
        RZ(angle, target_wire)
        return
    control_wire = control_wires[0]
    if double_phase:
        IsingZZ(angle, [control_wire, target_wire])
        return
    CRZ(angle, wires=[control_wire, target_wire])


def _run_trotter_steps(  # pylint: disable=too-many-arguments
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
    apply_fused_one_body=None,
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
            (Fig. 6 in `arXiv:2506.15784 <https://arxiv.org/abs/2506.15784>`__)
            construction (``True``) or a genuine controlled unitary (``False``).
        apply_system_basis_rotation (callable): Qfunc with inputs ``(U, wires)`` that applies
            a fragment's leaf tensor as :class:`~.BasisRotation`.
        apply_two_body_diagonal (callable):
            Qfunc with inputs ``(Z, wires, first_order_time_step, control_wires, double_phase)`` that applies the
            two-body :class:`~.IsingZZ` layer.
        apply_one_body_diagonal (callable):
            Qfunc with inputs ``(Z, wires, first_order_time_step, control_wires, double_phase)`` that applies the
            one-body :class:`~.RZ` layer, one per wire, from the diagonal of ``Z``.
        merge_leaves (callable): Function that merges the current unitary basis change
            with the previous: ``(U_prev, U_curr) -> U``. Combines consecutive leaves so their
            basis rotations telescope into one.
        transpose_leaf (callable): ``(U) -> U``. Inverse of a leaf, for the trailing basis
            rotation that closes the final fragment.
        apply_fused_one_body (callable | None): Optional qfunc ``(wires, control_wires,
            double_phase)`` that applies the full one-body evolution in the frame of the last
            two-body fragment. It replaces the basis rotations into and out of the one-body frame
            and the one-body diagonal layer, which are applied if it is ``None``.
    """
    U_tensor = hamiltonian.leaf_tensors
    Z_tensor = hamiltonian.core_tensors
    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        if len(control_wires) > 0:
            control_wires = math.array(control_wires, like="jax")
        # The diagonal layers index these tensors with traced loop indices, which a numpy
        # array cannot do.
        U_tensor = math.array(U_tensor, like="jax")
        Z_tensor = math.array(Z_tensor, like="jax")

    second_order_time_step = evolution_time / num_trotter_steps
    first_order_time_step = second_order_time_step / 2

    num_two_body_fragments = U_tensor.shape[0] - 1

    def rotate(prev_fragment_idx, fragment_idx):
        U = merge_leaves(U_tensor[prev_fragment_idx], U_tensor[fragment_idx])
        apply_system_basis_rotation(U, wires)

    def two_body_diagonal(fragment_idx, time_step):
        apply_two_body_diagonal(
            Z_tensor[fragment_idx], wires, time_step, control_wires, double_phase
        )

    def forward_fragment(fragment_idx):
        rotate(fragment_idx - 1, fragment_idx)
        two_body_diagonal(fragment_idx, first_order_time_step)

    def reverse_fragment(fragment_idx):
        two_body_diagonal(fragment_idx, first_order_time_step)
        rotate(fragment_idx, fragment_idx - 1)

    def one_body_block():
        # Starts and ends in the frame of the last two-body fragment.
        if apply_fused_one_body is not None:
            apply_fused_one_body(wires, control_wires, double_phase)
            return
        rotate(num_two_body_fragments, 0)
        apply_one_body_diagonal(
            Z_tensor[0], wires, first_order_time_step, control_wires, double_phase
        )
        rotate(0, num_two_body_fragments)

    def remainder_of_step(step_idx):
        for_loop(2, num_two_body_fragments + 1)(forward_fragment)()
        one_body_block()
        for_loop(num_two_body_fragments, 1, -1)(reverse_fragment)()
        # End internal steps with a full fragment-1 block and the final step with a half block.
        endpoint_time_step = math.where(
            step_idx < num_trotter_steps - 1,
            second_order_time_step,
            first_order_time_step,
        )
        two_body_diagonal(1, endpoint_time_step)

    apply_system_basis_rotation(U_tensor[1], wires)
    two_body_diagonal(1, first_order_time_step)
    for_loop(num_trotter_steps)(remainder_of_step)()

    very_last_U = transpose_leaf(U_tensor[1])
    apply_system_basis_rotation(very_last_U, wires)
