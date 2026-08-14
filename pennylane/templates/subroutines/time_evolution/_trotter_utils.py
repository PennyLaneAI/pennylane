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

from pennylane import compiler, math
from pennylane.control_flow import for_loop

# pylint: disable=too-many-arguments

has_jax = True
try:
    import jax
except ImportError:  # pragma: no cover
    has_jax = False


def _run_trotter_steps(
    evolution_time,
    num_trotter_steps,
    hamiltonian,
    wires,
    control_wires,
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

    The diagonal-rotation angles are linear in ``evolution_time`` while the basis
    rotations are time-independent. The controlled (double-phase, Fig. 6 of
    `arXiv:2506.15784 <https://arxiv.org/abs/2506.15784>`__) decompositions
    therefore obtain their halved sandwiched angles simply by passing
    ``evolution_time / 2`` here; nothing else about the circuit changes.

    Args:
        evolution_time (float): total evolution time ``t`` (pass ``t / 2`` for the
            controlled double-phase decomposition).
        num_trotter_steps (int): number of second-order Trotter steps (``> 0``).
        hamiltonian (dict): fragmented Hamiltonian data.
        wires (Wires): system wires.
        control_wires (Wires): control wires. Empty for the base (uncontrolled)
            circuit; a single wire for the CNOT-sandwiched controlled circuit.
        apply_system_basis_rotation (callable): ``(U, wires) -> None``.
        apply_two_body_diagonal (callable):
            ``(Z, wires, first_order_time_step, control_wires) -> None``.
        apply_one_body_diagonal (callable):
            ``(Z, wires, first_order_time_step, control_wires) -> None``.
        merge_leaves (callable): ``(U_prev, U_curr) -> U``.
        transpose_leaf (callable): ``(U) -> U``.
    """
    if not has_jax:
        raise ImportError(
            "jax is required for TrotterCDF/TrotterCGF. Install it with: pip install jax jaxlib"
        )  # pragma: no cover

    if compiler.active():
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
            U = jax.lax.cond(
                prev_fragment_idx < 0,
                lambda U_tensor, fragment_idx, prev_fragment_idx: U_tensor[fragment_idx],
                lambda U_tensor, fragment_idx, prev_fragment_idx: merge_leaves(
                    U_tensor[prev_fragment_idx], U_tensor[fragment_idx]
                ),
                U_tensor,
                fragment_idx,
                prev_fragment_idx,
            )
            Z = Z_tensor[fragment_idx]
            apply_system_basis_rotation(U, wires)
            apply_two_body_diagonal(Z, wires, first_order_time_step, control_wires)
            return fragment_idx

        def one_body_fragment():
            U_one = U_tensor[0]
            U = merge_leaves(U_tensor[num_two_body_fragments], U_one)
            apply_system_basis_rotation(U, wires)
            apply_one_body_diagonal(Z_tensor[0], wires, first_order_time_step, control_wires)

        prev_fragment_idx_forward = math.sign(2 * step_idx - 1)
        for_loop(1, num_two_body_fragments + 1)(two_body_fragments)(prev_fragment_idx_forward)

        one_body_fragment()

        prev_fragment_idx_backward = 0
        for_loop(num_two_body_fragments, 0, -1)(two_body_fragments)(prev_fragment_idx_backward)

        return hamiltonian

    for_loop(num_trotter_steps)(_trotter_step)(hamiltonian)

    very_last_U = transpose_leaf(hamiltonian["leaf_tensors"][1])
    apply_system_basis_rotation(very_last_U, wires)
