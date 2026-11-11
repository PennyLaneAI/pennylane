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

"""Built-in decomposition rules for templates.subroutines.time_evolution.trotter_cdf."""

# pylint: disable=too-many-arguments, no-value-for-parameter, unused-argument

from collections import defaultdict
from functools import partial

import numpy as np

from pennylane import math
from pennylane.capture import hint
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.decomposition_rules.op_math import controlled2 as controlled2_rules
from pennylane.ops import CNOT, CRZ, RZ, GlobalPhase, IsingZZ, PhaseShift
from pennylane.ops.op_math import adjoint, ctrl
from pennylane.templates.subroutines.arithmetic.temporary_and import TemporaryAND
from pennylane.templates.subroutines.qchem.basis_rotation import BasisRotation
from pennylane.templates.subroutines.time_evolution._trotter_utils import (
    _emit_one_body_rz,
    _run_trotter_steps,
)
from pennylane.templates.subroutines.time_evolution.trotter_cdf import TrotterCDF
from pennylane.typing import Float, Wire
from pennylane.wires import validate_no_wire_overlaps


def _apply_system_basis_rotation(U, wires):
    """Apply a fragment's basis rotation on the alpha and beta spin channels."""
    if math.is_tracer(U) or not np.allclose(U, np.eye(len(U))):
        BasisRotation(unitary_matrix=U, wires=wires[::2])
        BasisRotation(unitary_matrix=U, wires=wires[1::2])


def _merge_leaves(U_prev, U_curr):
    """Fragment-rotation merge rule: conjugate-transpose of ``U_prev`` times ``U_curr``."""
    return _transpose_leaf(U_prev) @ U_curr


def _transpose_leaf(U):
    """Conjugate transpose (adjoint) of a leaf rotation."""
    return U.conj().T


def _apply_two_body_diagonal(Z, wires, first_order_time_step, control_wires, double_phase):
    r"""Apply the two-body ``IsingZZ`` layer (base / double-phase / genuine controlled).

    Genuine control and double-phase are mutually exclusive constructions for a controlled
    ``IsingZZ``, chosen once here via ``double_phase``: genuine control uses
    ``ctrl(IsingZZ(...))`` on each term; double-phase instead shares *one* ``CNOT`` sandwich
    across every term touching a given ``wire_idx0``, which is cheaper since ``IsingZZ`` itself
    stays uncontrolled either way.
    """
    num_cas = Z.shape[0]

    def _angle(wire_idx0, wire_idx1):
        # Two-body prefactor. In the fragment basis the generator is a sum over distinct wire
        # pairs (w, w') with coupling lambda = core_tensors[l][p, q] for their orbitals p, q.
        # With n = (I - Z)/2, the Z_w Z_w' piece of lambda n_w n_w' is (lambda/4) Z_w Z_w', and
        # exp(-i (lambda/4) Z_w Z_w' tau) = IsingZZ(lambda tau / 2) since
        # IsingZZ(theta) = exp(-i theta Z Z / 2). Each fragment is visited at
        # tau = first_order_time_step = dt_trotter / 2, so the angle per visit is
        #   IsingZZ(lambda tau / 2) = IsingZZ(0.5 * lambda * first_order_time_step),
        # i.e. beta_twoB = 0.5. The two visits per Trotter step accumulate
        # IsingZZ(lambda * dt_trotter / 2), i.e. IsingZZ(lambda t / 2) over the full evolution.
        return 0.5 * Z[wire_idx0 // 2, wire_idx1 // 2] * first_order_time_step

    def zz_rotations(wire_idx0):
        if double_phase:

            @for_loop(wire_idx0 + 1, 2 * num_cas)
            def _zz_rotations(wire_idx1):
                IsingZZ(_angle(wire_idx0, wire_idx1), [wires[wire_idx0], wires[wire_idx1]])

            CNOT([control_wires[0], wires[wire_idx0]])
            _zz_rotations()
            CNOT([control_wires[0], wires[wire_idx0]])
        else:

            @for_loop(wire_idx0 + 1, 2 * num_cas)
            def _zz_rotations(wire_idx1):
                angle = _angle(wire_idx0, wire_idx1)
                zz_wires = [wires[wire_idx0], wires[wire_idx1]]
                if len(control_wires) == 0:
                    IsingZZ(angle, zz_wires)
                else:
                    ctrl(IsingZZ(angle, zz_wires), control=control_wires)

            _zz_rotations()

    for wire_idx0 in range(2 * num_cas - 1):
        zz_rotations(wire_idx0)


def _apply_one_body_diagonal(Z_one_body, wires, first_order_time_step, control_wires, double_phase):
    """Apply the one-body ``RZ`` layer (base / double-phase / genuine controlled)."""
    num_cas = Z_one_body.shape[0]

    @for_loop(2 * num_cas)
    def z_rotations(wire_idx):
        # One-body prefactor derivation (identical to CGF).
        #   n_p = (I - Z_p) / 2  ->  the Z-piece of eps n_p has coefficient -eps/2.
        #   target operator exp(-i eps n_p t) has Z-piece exp(+i eps t / 2 Z) = RZ(-eps t).
        #   The one-body fragment is visited ONCE per Trotter step, at duration
        #   first_order_time_step = dt_trotter / 2, so to accumulate a total RZ angle of
        #   -eps * dt_trotter we need angle-per-step = -2 * eps * first_order_time_step,
        #   i.e. alpha_oneB = -2.
        angle = -2.0 * Z_one_body[wire_idx // 2, wire_idx // 2] * first_order_time_step
        _emit_one_body_rz(angle, wires[wire_idx], control_wires, double_phase)

    z_rotations()


def _apply_two_body_diagonal_with_work_wires(
    Z, wires, first_order_time_step, control_wires, double_phase, work_wires
):
    """Combine each orbital pair's four ZZ rotations using clean work wires.

    CNOTs store each orbital's alpha/beta parity on its beta wire. For p<q, an AND
    selects the two zero parities and a CRZ supplies the phase. Standard control
    shares the AND of the external control and the first orbital's zero parity
    across the inner loop. The work wires and original system basis are restored
    at the end of the layer.
    """
    assert len(control_wires) <= 1, "The work-wire decomposition supports at most one control wire."
    num_orbitals = Z.shape[0]
    standard_control = len(control_wires) > 0 and not double_phase

    @for_loop(num_orbitals)
    def orbital_parity_cnots(orbital):
        CNOT([wires[2 * orbital], wires[2 * orbital + 1]])

    @for_loop(num_orbitals)
    def same_orbital_rotations(orbital):
        _emit_one_body_rz(
            0.5 * Z[orbital, orbital] * first_order_time_step,
            wires[2 * orbital + 1],
            control_wires,
            double_phase,
        )

    @for_loop(num_orbitals - 1)
    def orbital_pairs(p):
        # The inner loop visits N-1, ..., 1 pairs, averaging N/2 iterations.
        @hint({"num-iters": num_orbitals / 2})
        @for_loop(p + 1, num_orbitals)
        def pair(q):
            if standard_control:
                pair_and_wires = [work_wires[0], wires[2 * q + 1], work_wires[1]]
                control_values = (1, 0)
                rotation_control = work_wires[1]
            else:
                pair_and_wires = [wires[2 * p + 1], wires[2 * q + 1], work_wires[0]]
                control_values = (0, 0)
                rotation_control = work_wires[0]
            TemporaryAND(pair_and_wires, control_values=control_values)
            rotation_target = wires[2 * q]
            CNOT([wires[2 * p], rotation_target])
            CRZ(2 * Z[p, q] * first_order_time_step, [rotation_control, rotation_target])
            CNOT([wires[2 * p], rotation_target])
            adjoint(TemporaryAND)(pair_and_wires, control_values=control_values)

        if standard_control:
            control_and_wires = [wires[2 * p + 1], control_wires[0], work_wires[0]]
            TemporaryAND(control_and_wires, control_values=(0, 1))
        # Flipping p-alpha reverses all (p, q) rotations, so one CNOT pair serves this loop.
        if double_phase:
            CNOT([control_wires[0], wires[2 * p]])
        pair()
        if double_phase:
            CNOT([control_wires[0], wires[2 * p]])
        if standard_control:
            adjoint(TemporaryAND)(control_and_wires, control_values=(0, 1))

    orbital_parity_cnots()
    same_orbital_rotations()
    orbital_pairs()
    orbital_parity_cnots()


def _energy_shift(hamiltonian):
    r"""Zero-of-energy shift applied as a ``GlobalPhase`` (or control ``RZ``).

    Collects ``nuc_constant`` (:math:`C`) and every identity contribution produced by the
    :math:`n = (I - Z)/2` substitutions into a single scalar :math:`s` (see "Implementation
    Details", step 5, in :class:`~.TrotterCDF`'s docstring for the full derivation):

    .. math::

        s = C + \sum_p \epsilon_p - \frac{1}{2}\sum_{l\ge 1}\sum_{p,q} \lambda^{(l)}_{pq}
              + \frac{1}{4}\sum_{l\ge 1}\sum_{p} \lambda^{(l)}_{pp} ,

    where :math:`\epsilon_p =` ``core_tensors[0][p, p]`` and :math:`\lambda^{(l)}_{pq} =`
    ``core_tensors[l][p, q]``. Below, ``phase_from_mod_one_body`` is :math:`\sum_p \epsilon_p`
    and ``phase_from_two_body`` is the two-body sum/trace pair.
    """
    nuc_constant = hamiltonian.nuc_constant
    Z_tensor = hamiltonian.core_tensors
    phase_from_mod_one_body = math.trace(Z_tensor[0])
    phase_from_two_body = (
        -math.sum(Z_tensor[1:]) / 2 + math.sum(math.trace(Z_tensor[1:], axis1=1, axis2=2)) / 4
    )
    return nuc_constant + phase_from_mod_one_body + phase_from_two_body


_CDF_HELPERS = {
    "apply_system_basis_rotation": _apply_system_basis_rotation,
    "apply_two_body_diagonal": _apply_two_body_diagonal,
    "apply_one_body_diagonal": _apply_one_body_diagonal,
    "merge_leaves": _merge_leaves,
    "transpose_leaf": _transpose_leaf,
}


def _use_work_wire_decomposition(num_work_wires, has_control, double_phase):
    """Whether enough clean work wires are supplied to combine the four ZZ rotations."""
    return num_work_wires >= 1 + int(has_control and not double_phase)


def _cdf_decomposition_helpers(work_wires, has_control=False, double_phase=False):
    """Select the two-body decomposition and bind its work wires when available."""
    if _use_work_wire_decomposition(len(work_wires), has_control, double_phase):
        return {
            **_CDF_HELPERS,
            "apply_two_body_diagonal": partial(
                _apply_two_body_diagonal_with_work_wires, work_wires=work_wires
            ),
        }
    return _CDF_HELPERS


def _cdf_resource_counts(
    num_trotter_steps, hamiltonian, has_control, double_phase=False, num_work_wires=0
):
    """Shared (upper-bound) gate counts for the base and controlled CDF circuits.

    The exact gate count can be lower at runtime because fragments whose basis rotation
    happens to be the identity are skipped when the Hamiltonian data is concrete (not
    traced); this estimate assumes no such fragment is skipped.
    """
    if num_trotter_steps <= 0:
        return {}

    leaf_tensors = hamiltonian.leaf_tensors
    num_two_body_fragments = leaf_tensors.shape[0] - 1
    num_cas = leaf_tensors.shape[-1]

    resources = defaultdict(int)
    num_sysrot_calls = 2 * num_two_body_fragments * num_trotter_steps + 2
    num_twobody_blocks = (2 * num_two_body_fragments - 1) * num_trotter_steps + 1
    num_onebody_blocks = num_trotter_steps
    num_twobody_rotations = num_twobody_blocks * num_cas * (2 * num_cas - 1)
    num_onebody_rotations = num_onebody_blocks * 2 * num_cas

    # NOTE: 'BasisRotation' decomposes differently depending on the dtype of the unitary matrix.
    dtype_leaf, _ = math.get_abstract_and_cast_dtype(leaf_tensors)
    sysrot_key = BasisRotation(dtype_leaf[num_cas, num_cas], wires=Wire[num_cas])
    resources[sysrot_key] += 2 * num_sysrot_calls

    if not has_control:
        one_body_rotation = RZ
        two_body_rotation = IsingZZ
        resources[GlobalPhase] += 1
    elif double_phase:
        one_body_rotation = two_body_rotation = IsingZZ
        resources[RZ] += 1
    else:
        one_body_rotation = CRZ
        two_body_rotation = ctrl(IsingZZ(Float, wires=Wire[2]), control=Wire[1])
        resources[PhaseShift] += 1
    resources[one_body_rotation] += num_onebody_rotations

    if _use_work_wire_decomposition(num_work_wires, has_control, double_phase):
        num_pairs = num_twobody_blocks * num_cas * (num_cas - 1) // 2
        num_same_orbital_rotations = num_twobody_blocks * num_cas
        standard_control = has_control and not double_phase
        if has_control and double_phase:
            resources[CNOT] += 2 * num_twobody_blocks * (num_cas - 1)
        resources[one_body_rotation] += num_same_orbital_rotations
        resources[CRZ] += num_pairs
        resources[CNOT] += 2 * num_twobody_blocks * num_cas + 2 * num_pairs
        num_ands = num_pairs
        if standard_control:
            num_ands += num_twobody_blocks * (num_cas - 1)
        resources[TemporaryAND] += num_ands
        resources[adjoint(TemporaryAND(Wire[3]))] += num_ands
    else:
        resources[two_body_rotation] += num_twobody_rotations
        if has_control and double_phase:
            resources[CNOT] += num_twobody_blocks * 2 * (2 * num_cas - 1)

    return {op: count for op, count in resources.items() if count}


def _trotter_cdf_resources(
    evolution_time, num_trotter_steps, hamiltonian, wires, double_phase, work_wires=()
):
    return _cdf_resource_counts(
        num_trotter_steps, hamiltonian, has_control=False, num_work_wires=len(work_wires)
    )


@register_resources(_trotter_cdf_resources, exact=False)
def _trotter_cdf_decomposition(
    evolution_time, num_trotter_steps, hamiltonian, wires, double_phase, work_wires=()
):
    # ``double_phase`` only affects the controlled decomposition; the base operator is
    # always the plain (uncontrolled) e^{-iHt} circuit.
    if num_trotter_steps > 0:
        _run_trotter_steps(
            evolution_time,
            num_trotter_steps,
            hamiltonian.normalize_leaf_determinant(),
            wires,
            (),
            **_cdf_decomposition_helpers(work_wires),
        )
        phi = (_energy_shift(hamiltonian) * evolution_time) % (4 * np.pi)
        GlobalPhase(phi)


add_decomps(TrotterCDF, _trotter_cdf_decomposition)


def _controlled_trotter_cdf_resource(
    base, control_wires, control_values, work_wires, work_wire_type
):
    return _cdf_resource_counts(
        base.arguments["num_trotter_steps"],
        base.arguments["hamiltonian"],
        has_control=True,
        double_phase=base.arguments["double_phase"],
        num_work_wires=len(base.arguments["work_wires"]),
    )


@register_condition(lambda base, control_wires, *_, **__: len(control_wires) == 1)
@register_resources(_controlled_trotter_cdf_resource, exact=False)
def _controlled_trotter_cdf_decomp(base, control_wires, control_values, work_wires, work_wire_type):
    evolution_time = base.arguments["evolution_time"]
    num_trotter_steps = base.arguments["num_trotter_steps"]
    hamiltonian = base.arguments["hamiltonian"]
    wires = base.arguments["wires"]
    double_phase = base.arguments["double_phase"]
    cdf_work_wires = base.arguments["work_wires"]
    validate_no_wire_overlaps({"control_wires": control_wires, "work_wires": cdf_work_wires})

    if num_trotter_steps == 0:
        return

    helpers = _cdf_decomposition_helpers(
        cdf_work_wires, has_control=True, double_phase=double_phase
    )
    phi = (_energy_shift(hamiltonian) * evolution_time) % (4 * np.pi)
    hamiltonian = hamiltonian.normalize_leaf_determinant()

    if double_phase:
        # Double-phase (Fig. 6 in https://arxiv.org/abs/2506.15784) circuit: each full-time diagonal block is CNOT-sandwiched by
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
            **helpers,
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
        **helpers,
    )
    PhaseShift(-phi, control_wires)


@register_condition(controlled2_rules.ctrl_many_zeroed_work_wires.is_applicable)
@register_condition(lambda base, **_: not base.arguments["double_phase"])
@register_resources(
    lambda *args, **kwargs: controlled2_rules.ctrl_many_zeroed_work_wires.compute_resources(
        *args, **kwargs
    ).gate_counts,
    exact=False,
)
def _multi_controlled_trotter_cdf_decomp(
    base, control_wires, control_values, work_wires, work_wire_type
):
    """Reuse the single-control CDF decomposition after combining multiple controls."""
    validate_no_wire_overlaps(
        {
            "control_wires": control_wires,
            "control_work_wires": work_wires,
            "work_wires": base.arguments["work_wires"],
        }
    )
    controlled2_rules.ctrl_many_zeroed_work_wires(
        base, control_wires, control_values, work_wires, work_wire_type
    )


add_decomps(
    "C(TrotterCDF)",
    controlled2_rules.flip_zero_control(_controlled_trotter_cdf_decomp),
    _multi_controlled_trotter_cdf_decomp,
)
