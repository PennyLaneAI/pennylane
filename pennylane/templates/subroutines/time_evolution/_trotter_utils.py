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

from collections import defaultdict

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.ops import RZ, IsingZZ, cond
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


def _raw_key_data(key):
    """Convert a typed JAX PRNG key into its raw ``uint32`` key data. Other inputs, including
    ``None``, are returned unchanged."""
    if key is None:
        return None
    import jax  # pylint: disable=import-outside-toplevel

    if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        return jax.random.key_data(key)
    return key


def _block_counts(num_trotter_steps, num_two_body_fragments, randomized):
    """Numbers of basis rotations, two-body diagonal blocks, and one-body diagonal blocks
    emitted by :func:`~._run_trotter_steps`, as a list of candidate configurations.

    Without randomization, there is a single configuration. With randomization, the fragment at
    the center of each step, whose two half blocks are merged, is only known at runtime. All block
    counts are linear in the number of steps with the one-body fragment at their center, so each
    count is bounded by one of the two extreme configurations, in which the one-body fragment
    sits at the center of all steps or of none.
    """
    L, n = num_two_body_fragments, num_trotter_steps
    if not randomized:
        return [(2 * L * n + 2, (2 * L - 1) * n + 1, n)]
    num_basis_rotations = (2 * L + 1) * n + 1
    one_body_always_central = (num_basis_rotations, 2 * L * n, n)
    if L == 0:
        return [one_body_always_central]
    return [one_body_always_central, (num_basis_rotations, (2 * L - 1) * n, 2 * n)]


def _max_counts(counts_list):
    """Gate-wise maximum of an iterable of gate-count dictionaries."""
    max_counts = defaultdict(int)
    for counts in counts_list:
        for gate, count in counts.items():
            max_counts[gate] = max(max_counts[gate], count)
    return dict(max_counts)


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
    key=None,
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
        key (jax.Array | None): JAX PRNG key for randomized fragment orderings. If ``None``
            (default), every step uses the fixed ordering :math:`H_1, \dots, H_L, H_0` for its
            first-order half. Otherwise, each step draws an independent uniformly random ordering
            of all :math:`L+1` fragments, see :func:`~._run_randomized_trotter_steps`. Requires
            program capture.
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

    if key is not None:
        _run_randomized_trotter_steps(
            second_order_time_step,
            num_trotter_steps,
            U_tensor,
            Z_tensor,
            wires,
            control_wires,
            double_phase,
            key,
            apply_system_basis_rotation=apply_system_basis_rotation,
            apply_two_body_diagonal=apply_two_body_diagonal,
            apply_one_body_diagonal=apply_one_body_diagonal,
            merge_leaves=merge_leaves,
            transpose_leaf=transpose_leaf,
        )
        return

    num_two_body_fragments = U_tensor.shape[0] - 1

    apply_system_basis_rotation(U_tensor[1], wires)
    apply_two_body_diagonal(Z_tensor[1], wires, first_order_time_step, control_wires, double_phase)

    def main_loop(step_idx):
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

        # Carry the index of the fragment whose inverse was most recently applied.
        # For L=1 there is no reverse two-body loop, so the preceding frame is H0.
        prev_fragment_idx = for_loop(num_two_body_fragments, 1, -1)(two_body_fragment)(0)

        U = merge_leaves(U_tensor[prev_fragment_idx], U_tensor[1])
        apply_system_basis_rotation(U, wires)
        # End internal steps with a full fragment-1 block and the final step with a half block.
        endpoint_time_step = math.where(
            step_idx < num_trotter_steps - 1,
            second_order_time_step,
            first_order_time_step,
        )
        apply_two_body_diagonal(Z_tensor[1], wires, endpoint_time_step, control_wires, double_phase)

    for_loop(num_trotter_steps)(main_loop)()

    very_last_U = transpose_leaf(U_tensor[1])
    apply_system_basis_rotation(very_last_U, wires)


def _random_fragment_orderings(key, num_trotter_steps, num_fragments):
    """Draw one independent, uniformly random permutation of ``range(num_fragments)`` per
    Trotter step, as an integer array of shape ``(num_trotter_steps, num_fragments)``.

    The orderings are computed with ``jax.random`` inside the traced program, so a
    compiled circuit draws new orderings for each new (dynamic) ``key``.
    """
    import jax  # pylint: disable=import-outside-toplevel

    orderings = math.tile(math.arange(num_fragments, like="jax"), (num_trotter_steps, 1))
    return jax.random.permutation(key, orderings, axis=1, independent=True)


def _run_randomized_trotter_steps(
    second_order_time_step,
    num_trotter_steps,
    U_tensor,
    Z_tensor,
    wires,
    control_wires,
    double_phase,
    key,
    *,
    apply_system_basis_rotation,
    apply_two_body_diagonal,
    apply_one_body_diagonal,
    merge_leaves,
    transpose_leaf,
):
    r"""Emit second-order Trotter steps with a random fragment ordering per step.

    Each step :math:`k` draws a random permutation :math:`\pi_k` of all :math:`L+1` fragments
    (the one-body fragment :math:`H_0` included) and applies

    .. math::

        S_2^{(k)}(\Delta t) = \Big(\prod_{j=0}^{L-1} e^{-i H_{\pi_k(j)} \Delta t/2}\Big)\,
            e^{-i H_{\pi_k(L)} \Delta t}\,
            \Big(\prod_{j=L-1}^{0} e^{-i H_{\pi_k(j)} \Delta t/2}\Big) ,

    i.e. the first-order product formula in the order :math:`\pi_k` followed by its reverse.
    The two half blocks of the central fragment :math:`H_{\pi_k(L)}` are merged. Unlike in the
    deterministic scheme, the half blocks at step boundaries belong to different fragments in
    general and are not merged, and :math:`H_0` is no longer pinned to the center. Basis rotations
    of consecutive fragments are still merged throughout, including across step boundaries.

    Arguments are as in :func:`~._run_trotter_steps`, except that ``U_tensor`` and ``Z_tensor``
    must be JAX arrays, as they are indexed with traced fragment indices.
    """
    num_fragments = U_tensor.shape[0]
    first_order_time_step = second_order_time_step / 2
    orderings = _random_fragment_orderings(key, num_trotter_steps, num_fragments)

    def fragment(fragment_idx, U_prev, time_step):
        apply_system_basis_rotation(merge_leaves(U_prev, U_tensor[fragment_idx]), wires)

        @cond(fragment_idx == 0)
        def diagonal():
            # ``apply_one_body_diagonal`` evolves for twice the time step it is given.
            apply_one_body_diagonal(Z_tensor[0], wires, time_step / 2, control_wires, double_phase)

        @diagonal.otherwise
        def _():
            apply_two_body_diagonal(
                Z_tensor[fragment_idx], wires, time_step, control_wires, double_phase
            )

        diagonal()
        return U_tensor[fragment_idx]

    def step(step_idx, U_prev):
        ordering = orderings[step_idx]

        def half_block(position, U_prev):
            return fragment(ordering[position], U_prev, first_order_time_step)

        U_prev = for_loop(num_fragments - 1)(half_block)(U_prev)
        U_prev = fragment(ordering[num_fragments - 1], U_prev, second_order_time_step)
        return for_loop(num_fragments - 2, -1, -1)(half_block)(U_prev)

    # Merging with the identity leaf makes the very first basis rotation plain ``U_tensor[idx]``.
    identity_leaf = math.zeros_like(U_tensor[0]) + math.eye(U_tensor.shape[-1], like="jax")
    U_last = for_loop(num_trotter_steps)(step)(identity_leaf)
    apply_system_basis_rotation(transpose_leaf(U_last), wires)
