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
from pennylane.ops import RZ, IsingZZ
from pennylane.ops.op_math import ctrl

# pylint: disable=too-many-arguments


def _emit_one_body_rz(angle, target_wire, control_wires, double_phase):
    r"""Emit a one-body ``RZ`` rotation for the base, double-phase, or genuine controlled circuit.

    * No control wire: a plain ``RZ(angle)``.
    * Controlled, ``double_phase=True``: ``IsingZZ(angle, [control, target])``, i.e. the same
      ``CNOT; RZ; CNOT`` sandwich as Fig. 6 in https://arxiv.org/abs/2506.15784.
      This results in :math:`\text{diag}(U, U^\dagger)` overall, such that the relative phase
      between both branches is 2t (hence "double-phase").
    * Controlled, ``double_phase=False``: ``ctrl(RZ(angle, target))`` for a genuine
      controlled-:math:`e^{-iHt}` (``diag(1, U)`` on the control/target subspace).
    """
    if len(control_wires) == 0:
        RZ(angle, target_wire)
        return
    control_wire = control_wires[0]
    if double_phase:
        IsingZZ(angle, [control_wire, target_wire])
        return
    ctrl(RZ(angle, target_wire), control=[control_wire])


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
    """
    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        if len(control_wires) > 0:
            control_wires = math.array(control_wires, like="jax")

    second_order_time_step = evolution_time / num_trotter_steps
    first_order_time_step = second_order_time_step / 2

    num_two_body_fragments = hamiltonian.leaf_tensors.shape[0] - 1

    def _initial_fragment(_, hamiltonian):
        U_tensor = hamiltonian.leaf_tensors
        Z_tensor = hamiltonian.core_tensors
        apply_system_basis_rotation(U_tensor[1], wires)
        apply_two_body_diagonal(
            Z_tensor[1], wires, first_order_time_step, control_wires, double_phase
        )
        return hamiltonian

    hamiltonian = for_loop(1)(_initial_fragment)(hamiltonian)

    def _remainder_of_step(endpoint_time_step):
        """Create a loop body ending in fragment 1 at the supplied duration."""

        def remainder_of_step(_, hamiltonian):
            # ``hamiltonian`` is carried through the for-loop (rather than closed over)
            # so the traced tensors remain valid loop-body inputs under jax capture.
            U_tensor = hamiltonian.leaf_tensors
            Z_tensor = hamiltonian.core_tensors

            def two_body_fragment(fragment_idx, prev_fragment_idx):
                U = merge_leaves(U_tensor[prev_fragment_idx], U_tensor[fragment_idx])
                apply_system_basis_rotation(U, wires)
                apply_two_body_diagonal(
                    Z_tensor[fragment_idx],
                    wires,
                    first_order_time_step,
                    control_wires,
                    double_phase,
                )
                return fragment_idx

            for_loop(2, num_two_body_fragments + 1)(two_body_fragment)(1)

            U = merge_leaves(U_tensor[num_two_body_fragments], U_tensor[0])
            apply_system_basis_rotation(U, wires)
            apply_one_body_diagonal(
                Z_tensor[0], wires, first_order_time_step, control_wires, double_phase
            )

            # For one two-body fragment this loop has no iterations and returns the
            # initial frame index 0, matching the current one-body frame.
            prev_fragment_idx = for_loop(num_two_body_fragments, 1, -1)(two_body_fragment)(0)

            U = merge_leaves(U_tensor[prev_fragment_idx], U_tensor[1])
            apply_system_basis_rotation(U, wires)
            apply_two_body_diagonal(
                Z_tensor[1], wires, endpoint_time_step, control_wires, double_phase
            )
            return hamiltonian

        return remainder_of_step

    # Merge the two fragment-1 half blocks at each internal Trotter boundary into one
    # full block. Keeping this as a loop preserves constant-size captured control flow.
    hamiltonian = for_loop(num_trotter_steps - 1)(_remainder_of_step(2 * first_order_time_step))(
        hamiltonian
    )
    for_loop(1)(_remainder_of_step(first_order_time_step))(hamiltonian)

    very_last_U = transpose_leaf(hamiltonian.leaf_tensors[1])
    apply_system_basis_rotation(very_last_U, wires)
