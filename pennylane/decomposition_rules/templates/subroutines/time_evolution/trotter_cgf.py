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

"""Built-in decomposition rules for templates.subroutines.time_evolution.trotter_cgf."""

# pylint: disable=too-many-arguments, no-value-for-parameter, unused-argument

from collections import defaultdict

import numpy as np

from pennylane import math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.decomposition_rules.op_math.controlled2 import (
    flip_zero_control as flip_zero_control2,
)
from pennylane.ops import CNOT, CRZ, RZ, GlobalPhase, IsingZZ, PhaseShift
from pennylane.ops.op_math import ctrl
from pennylane.templates.subroutines.qchem.basis_rotation import BasisRotation
from pennylane.templates.subroutines.time_evolution._trotter_utils import (
    _emit_one_body_rz,
    _run_trotter_steps,
)
from pennylane.templates.subroutines.time_evolution.trotter_cgf import TrotterCGF
from pennylane.typing import Float, Wire


def _apply_system_basis_rotation(U, wires):
    """Apply a fragment's per-mode basis rotation on the whole system."""
    num_modes, n_states, _ = U.shape
    # The fragment's leaf O stores the "rotate-from-bare-to-diagonal-basis" direction.
    # qml.BasisRotation(U) with U = O^T implements the single-particle map O^T (moving
    # from bare modal states to the diagonal basis), so we pass the transpose.
    for l in range(num_modes):
        U_l = math.swapaxes(U[l], -2, -1)
        mode_wires = wires[l * n_states : (l + 1) * n_states]
        if math.is_tracer(U_l) or not np.allclose(U_l, np.eye(n_states)):
            BasisRotation(unitary_matrix=U_l, wires=mode_wires)


def _merge_leaves(U_prev, U_curr):
    """Per-mode fragment-rotation merge rule ``U_curr @ U_prev^T`` (vectorized via einsum).

    ``_apply_system_basis_rotation`` emits ``BasisRotation(leaf^T)``, so the single
    :class:`~.BasisRotation` that replaces the ``leaf_prev`` un-rotation followed by the
    ``leaf_curr`` rotation must be ``BasisRotation((U_curr @ U_prev^T)^T)`` for the frames
    to telescope. (This differs from :class:`~.TrotterCDF`, whose ``apply`` does not
    transpose and whose merge is therefore ``U_prev^dagger @ U_curr``.)
    """
    return math.einsum("lij,lkj->lik", U_curr, U_prev)


def _transpose_leaf(U):
    """Batch-transpose a leaf over the leading mode axis."""
    return math.swapaxes(U, -2, -1)


def _apply_two_body_diagonal(Z, wires, first_order_time_step, control_wires, double_phase):
    """Apply the two-body ``IsingZZ`` layer (base / double-phase / genuine controlled).

    Genuine control and double-phase are mutually exclusive constructions for a controlled
    ``IsingZZ``, chosen once here via ``double_phase``: genuine control uses
    ``ctrl(IsingZZ(...))`` on each term; double-phase instead shares *one* ``CNOT`` sandwich
    across every term touching a given ``wire_lp``, which is cheaper since ``IsingZZ`` itself
    stays uncontrolled either way.
    """
    num_modes = Z.shape[0]
    n_states = Z.shape[2]

    for l in range(1, num_modes):
        for m in range(l):  # strict lower triangle: l > m
            Z_lm = Z[l, m]

            @for_loop(n_states)
            def _p_loop(p, Z_lm=Z_lm, l=l, m=m):
                wire_lp = wires[l * n_states + p]

                # Symmetrization is already taken into account here.
                def _angle(q):
                    return 0.5 * Z_lm[p, q] * first_order_time_step

                if double_phase:

                    @for_loop(n_states)
                    def _q_loop(q, wire_lp=wire_lp, m=m):
                        wire_mq = wires[m * n_states + q]
                        IsingZZ(_angle(q), [wire_lp, wire_mq])

                    CNOT([control_wires[0], wire_lp])
                    _q_loop()
                    CNOT([control_wires[0], wire_lp])
                else:

                    @for_loop(n_states)
                    def _q_loop(q, wire_lp=wire_lp, m=m):
                        wire_mq = wires[m * n_states + q]
                        angle = _angle(q)
                        zz_wires = [wire_lp, wire_mq]
                        if len(control_wires) == 0:
                            IsingZZ(angle, zz_wires)
                        else:
                            ctrl(IsingZZ(angle, zz_wires), control=control_wires)

                    _q_loop()

            _p_loop()


def _apply_one_body_diagonal(Z_one_body, wires, first_order_time_step, control_wires, double_phase):
    """Apply the one-body ``RZ`` layer (base / double-phase / genuine controlled)."""
    num_modes = Z_one_body.shape[0]
    n_states = Z_one_body.shape[2]

    @for_loop(num_modes)
    def mode_loop(l):

        @for_loop(n_states)
        def modal_loop(p):
            wire_lp = wires[l * n_states + p]
            # One-body prefactor derivation:
            #   n^l_p = (I - Z_{l,p}) / 2  ->  Z-piece has coefficient -eps/2
            #   target operator: exp(-i eps n t) has Z-piece exp(+i eps t / 2 Z)
            #     which equals RZ(-eps t).
            #   This fragment is visited ONCE per Trotter step, at duration
            #     first_order_time_step = dt_trotter / 2, so to accumulate a total
            #     RZ angle of -eps * dt_trotter we need angle-per-step
            #     = -eps * dt_trotter = -2 * eps * first_order_time_step. So alpha_oneB = -2.
            angle = -2.0 * Z_one_body[l, l, p, p] * first_order_time_step
            _emit_one_body_rz(angle, wire_lp, control_wires, double_phase)

        modal_loop()

    mode_loop()


def _energy_shift(hamiltonian):
    """Zero-of-energy shift applied as a ``GlobalPhase`` (or control ``RZ``)."""
    nuc_constant = hamiltonian.nuc_constant
    Z_tensor = hamiltonian.core_tensors
    # One-body diagonal: nested traces over axes (-2, -1) twice pick out the
    # (l == m, p == q) entries and sum eps^l_p.
    one_body_diag = math.trace(
        math.trace(Z_tensor[0], axis1=-2, axis2=-1),
        axis1=-2,
        axis2=-1,
    )  # scalar: sum_{l, p} eps^l_p
    return nuc_constant + one_body_diag / 2


_CGF_HELPERS = {
    "apply_system_basis_rotation": _apply_system_basis_rotation,
    "apply_two_body_diagonal": _apply_two_body_diagonal,
    "apply_one_body_diagonal": _apply_one_body_diagonal,
    "merge_leaves": _merge_leaves,
    "transpose_leaf": _transpose_leaf,
}


def _cgf_resource_counts(num_trotter_steps, hamiltonian, has_control, double_phase=False):
    """Shared (upper-bound) gate counts for the base and controlled CGF circuits.

    The exact gate count can be lower at runtime because fragments whose basis rotation
    happens to be the identity are skipped when the Hamiltonian data is concrete (not
    traced); this estimate assumes no such fragment is skipped.
    """
    if num_trotter_steps <= 0:
        return {}

    leaf_tensors = hamiltonian.leaf_tensors
    num_two_body_fragments = leaf_tensors.shape[0] - 1
    num_modes = leaf_tensors.shape[1]
    n_states = leaf_tensors.shape[2]

    resources = defaultdict(int)
    num_sysrot_calls = 2 * num_two_body_fragments * num_trotter_steps + 2
    num_twobody_blocks = (2 * num_two_body_fragments - 1) * num_trotter_steps + 1
    num_onebody_blocks = num_trotter_steps
    num_pairs = num_modes * (num_modes - 1) // 2
    num_twobody_rotations = num_twobody_blocks * num_pairs * n_states**2
    num_onebody_rotations = num_onebody_blocks * num_modes * n_states

    # NOTE: 'BasisRotation' decomposes differently depending on the dtype of the unitary matrix.
    dtype_leaf, _ = math.get_abstract_and_cast_dtype(leaf_tensors)
    sysrot_key = BasisRotation(dtype_leaf[n_states, n_states], wires=Wire[n_states])
    resources[sysrot_key] += num_modes * num_sysrot_calls

    if not has_control:
        resources[IsingZZ] += num_twobody_rotations
        resources[RZ] += num_onebody_rotations
        resources[GlobalPhase] += 1
    elif double_phase:
        # Double-phase (Fig. 6 https://arxiv.org/abs/2506.15784): ``IsingZZ`` on every diagonal
        # term (one- and two-body), plus one CNOT pair around each *two-body* block, plus an RZ on
        # the control wire for the global phase.
        resources[IsingZZ] += num_twobody_rotations + num_onebody_rotations
        resources[CNOT] += num_twobody_blocks * num_pairs * 2 * n_states
        resources[RZ] += 1
    else:
        # Genuine controlled: one-body ``CRZ(...)``, two-body ``ctrl(IsingZZ(...))``; the
        # global phase becomes a PhaseShift on the control wire. There are no bare IsingZZ/RZ gates.
        resources[CRZ] += num_onebody_rotations
        resources[ctrl(IsingZZ(Float, wires=Wire[2]), control=Wire[1])] += num_twobody_rotations
        resources[PhaseShift] += 1

    return dict(resources)


def _trotter_cgf_resources(evolution_time, num_trotter_steps, hamiltonian, wires, double_phase):
    return _cgf_resource_counts(num_trotter_steps, hamiltonian, has_control=False)


@register_resources(_trotter_cgf_resources, exact=False)
def _trotter_cgf_decomposition(evolution_time, num_trotter_steps, hamiltonian, wires, double_phase):
    # ``double_phase`` only affects the controlled decomposition; the base operator is
    # always the plain (uncontrolled) e^{-iHt} circuit.
    if num_trotter_steps > 0:
        _run_trotter_steps(
            evolution_time,
            num_trotter_steps,
            hamiltonian.normalize_leaf_determinant().align_one_body_leaf(),
            wires,
            (),
            **_CGF_HELPERS,
        )
        phi = (_energy_shift(hamiltonian) * evolution_time) % (4 * np.pi)
        GlobalPhase(phi)


add_decomps(TrotterCGF, _trotter_cgf_decomposition)


def _controlled_trotter_cgf_resource(
    base, control_wires, control_values, work_wires, work_wire_type
):
    return _cgf_resource_counts(
        base.arguments["num_trotter_steps"],
        base.arguments["hamiltonian"],
        has_control=True,
        double_phase=base.arguments["double_phase"],
    )


@register_condition(lambda base, control_wires, *_, **__: len(control_wires) == 1)
@register_resources(_controlled_trotter_cgf_resource, exact=False)
def _controlled_trotter_cgf_decomp(base, control_wires, control_values, work_wires, work_wire_type):
    evolution_time = base.arguments["evolution_time"]
    num_trotter_steps = base.arguments["num_trotter_steps"]
    hamiltonian = base.arguments["hamiltonian"]
    wires = base.arguments["wires"]
    double_phase = base.arguments["double_phase"]

    if num_trotter_steps == 0:
        return

    phi = (_energy_shift(hamiltonian) * evolution_time) % (4 * np.pi)
    hamiltonian = hamiltonian.normalize_leaf_determinant().align_one_body_leaf()

    if double_phase:
        # Double-phase (Fig. 6 https://arxiv.org/abs/2506.15784) circuit: each full-time diagonal block is CNOT-sandwiched by
        # the control wire, so the bare rotations give control-0 / control-1 branches
        # e^{-i(H - s)t} / e^{+i(H - s)t}, where s = _energy_shift is the identity part of H.
        # We apply the energy shift explicitly and symmetrically as RZ(2*phi) on the control
        # wire (= diag(e^{-i s t}, e^{+i s t})), making the branches exactly the full-time
        # e^{-iHt} / e^{+iHt}.
        _run_trotter_steps(
            evolution_time,
            num_trotter_steps,
            hamiltonian,
            wires,
            control_wires,
            True,
            **_CGF_HELPERS,
        )
        RZ(2 * phi, control_wires)
        return

    # Genuine controlled unitary: control each diagonal rotation at the full angle (basis
    # rotations stay uncontrolled and telescope to the identity in the control-0 branch),
    # and apply the global phase as controlled-GlobalPhase(phi) = PhaseShift(-phi).
    _run_trotter_steps(
        evolution_time,
        num_trotter_steps,
        hamiltonian,
        wires,
        control_wires,
        False,
        **_CGF_HELPERS,
    )
    PhaseShift(-phi, control_wires)


add_decomps("C(TrotterCGF)", flip_zero_control2(_controlled_trotter_cgf_decomp))
