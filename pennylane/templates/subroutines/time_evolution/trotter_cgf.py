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
"""Contains the ``TrotterCGF`` template for Christiansen Greedy Fragmentation Hamiltonians."""

from collections import defaultdict

import numpy as np

from pennylane import math
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.ops import CNOT, RZ, GlobalPhase, IsingZZ
from pennylane.ops.op_math.controlled2 import flip_zero_control as flip_zero_control2
from pennylane.templates.subroutines.qchem.basis_rotation import BasisRotation
from pennylane.typing import Complex, Wire

from ._trotter_utils import _run_trotter_steps

# pylint: disable=too-many-arguments, no-value-for-parameter, unused-argument


class TrotterCGF(Operator2):
    r"""Second-order Trotter time evolution for a Christiansen Greedy Fragmentation (CGF) Hamiltonian.

    Implements :math:`e^{-iHt}` for a vibrational Hamiltonian in the Christiansen Greedy
    Fragmentation form, see `arXiv:2508.11865, Sec. III C <https://arxiv.org/abs/2508.11865>`__.

    Controlling this operator with :func:`~pennylane.ctrl` (a single control wire)
    produces the double-phase Hadamard-test circuit of `Fig. 6 of arXiv:2506.15784
    <https://arxiv.org/abs/2506.15784>`__: each diagonal rotation is sandwiched by a
    pair of ``CNOT`` gates from the control wire (an ancilla-system ``ZZ`` coupling)
    and its angle is halved.

    .. code-block::

            c: ─╭●───────╭●─┤
        wires: ─╰X──U(ϕ)─╰X─┤

    Args:
        evolution_time (float): Total evolution time ``t``.
        num_trotter_steps (int): Number of second-order Trotter steps.
        hamiltonian (dict): A CGF Hamiltonian as a dictionary with keys ``nuc_constant``,
            ``core_tensors``, and ``leaf_tensors``. The expected shapes are
            ``core_tensors: (L+1, M, M, N, N)`` and ``leaf_tensors: (L+1, M, N, N)``,
            where ``M`` is the number of modes, ``N`` is the number of modals per mode,
            and ``L`` is the number of two-body fragments.
        wires (Wires): The system wires. CGF expects ``M*N`` wires arranged mode-major:
            wire ``l*N + p`` corresponds to modal ``p`` of mode ``l`` (unary/SBE layout).

    **Example**

    Let us create mock CGF Hamiltonian data with the correct tensor shapes.

    .. code-block:: python

        rng = np.random.default_rng(42)
        L = 2; M = 2; N = 3
        hamiltonian = {
            "core_tensors": rng.random((L, M, M, N, N)),
            "leaf_tensors": rng.random((L, M, N, N)),
            "nuc_constant": 0.5,
        }

    With this, we can set up a Hadamard test circuit, consisting of a :class:`~.Hadamard`
    gate and a controlled Trotter evolution obtained via :func:`~pennylane.ctrl`.

    .. code-block:: python

        registers = qp.registers({"hadamard": 1, "system": M * N})

        gate_set = {"Hadamard", "BasisRotation", "RZ", "IsingZZ", "CNOT", "ForLoop"}

        @qp.qjit
        @qp.transforms.decompose(gate_set=gate_set)
        @qp.qnode(qp.device("lightning.qubit"))
        def trotter_circuit():
            qp.H(registers["hadamard"])

            qp.ctrl(
                qp.TrotterCGF(
                    evolution_time=1.0, num_trotter_steps=10, hamiltonian=hamiltonian,
                    wires=registers["system"],
                ),
                control=registers["hadamard"],
            )

            return qp.expval(qp.X(registers["hadamard"]))

    We can now run this circuit consisting of just ``10`` Trotter steps.

    >>> trotter_circuit()
    Array(-0.04733941, dtype=float64)

    Or check the quantum resources required for this task. Note that the order of the keys in
    the ``quantum_operations`` dictionary is not guaranteed, so we sort it before printing:

    >>> specs = qp.specs(trotter_circuit)()["resources"].quantum_operations
    >>> dict(sorted(specs.items()))
    {'CNOT': 240, 'Hadamard': 1, 'IsingZZ': 180, 'PhaseShift': 62, 'RZ': 61, 'SingleExcitation': 186}

    The :class:`~.SingleExcitation` gates are due to :class:`~.BasisRotation` decomposing into
    :class:`~.PhaseShift` and :class:`~.SingleExcitation` on ``lightning.qubit``.
    """

    dynamic_argnames = ("evolution_time",)
    hybrid_argnames = ("hamiltonian",)
    # `hybrid_argnames` and `compilable_argnames` cannot both be non-empty on the same
    # operator, so `num_trotter_steps` (a plain Python int that drives Python-level
    # control flow) is treated as `static_argnames` instead.
    static_argnames = ("num_trotter_steps",)

    def __init__(self, evolution_time, num_trotter_steps, hamiltonian, wires):
        Z = hamiltonian["core_tensors"]
        U = hamiltonian["leaf_tensors"]
        if not (Z.ndim == 5 and U.ndim == 4):
            raise ValueError(
                "TrotterCGF expects a CGF Hamiltonian with core_tensors.ndim == 5 and "
                f"leaf_tensors.ndim == 4. Got core_tensors.ndim={Z.ndim}, "
                f"leaf_tensors.ndim={U.ndim}. For electronic (CDF) Hamiltonians, use TrotterCDF."
            )
        super().__init__(evolution_time, num_trotter_steps, hamiltonian, wires)


def _apply_system_basis_rotation(U, wires):
    """Apply a fragment's per-mode basis rotation on the whole system."""
    num_modes, n_states, _ = U.shape
    # The fragment's leaf O stores the "rotate-from-bare-to-diagonal-basis" direction.
    # qml.BasisRotation(U) with U = O^T implements the single-particle map O^T (moving
    # from bare modal states to the diagonal basis), so we pass the transpose.
    for l in range(num_modes):
        U_l = math.swapaxes(U[l], -2, -1)
        mode_wires = wires[l * n_states : (l + 1) * n_states]
        if math.is_abstract(U_l) or not np.allclose(U_l, np.eye(n_states)):
            BasisRotation(unitary_matrix=U_l, wires=mode_wires)


def _merge_leaves(U_prev, U_curr):
    """Per-mode fragment-rotation merge rule ``U_prev^T @ U_curr`` (vectorized via einsum)."""
    return math.einsum("lji,ljk->lik", U_prev, U_curr)


def _transpose_leaf(U):
    """Batch-transpose a leaf over the leading mode axis."""
    return math.swapaxes(U, -2, -1)


def _apply_two_body_diagonal(Z, wires, first_order_time_step, control_wires):
    """Apply the two-body ``IsingZZ`` layer, optionally CNOT-sandwiched by the control."""
    num_modes = Z.shape[0]
    n_states = Z.shape[2]

    for l in range(1, num_modes):
        for m in range(l):  # strict lower triangle: l > m
            Z_lm = Z[l, m]

            @for_loop(n_states)
            def _p_loop(p, Z_lm=Z_lm, l=l, m=m):
                wire_lp = wires[l * n_states + p]

                @for_loop(n_states)
                def _q_loop(q, Z_lm=Z_lm, p=p, wire_lp=wire_lp, l=l, m=m):
                    # Symmetrization is already taken into account here.
                    wire_mq = wires[m * n_states + q]
                    lam = Z_lm[p, q]
                    angle = 0.5 * lam * first_order_time_step
                    IsingZZ(angle, [wire_lp, wire_mq])

                # A single control wire wraps each diagonal block in a CNOT pair; the
                # halved angle is supplied upstream by feeding evolution_time / 2.
                if len(control_wires) > 0:
                    CNOT([control_wires[0], wire_lp])
                _q_loop()
                if len(control_wires) > 0:
                    CNOT([control_wires[0], wire_lp])

            _p_loop()


def _apply_one_body_diagonal(Z_one_body, wires, first_order_time_step, control_wires):
    """Apply the one-body ``RZ`` layer, optionally CNOT-sandwiched by the control."""
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

            if len(control_wires) > 0:
                CNOT([control_wires[0], wire_lp])
            RZ(angle, wire_lp)
            if len(control_wires) > 0:
                CNOT([control_wires[0], wire_lp])

        modal_loop()

    mode_loop()


def _energy_shift(hamiltonian):
    """Zero-of-energy shift applied as a ``GlobalPhase`` (or control ``RZ``)."""
    nuc_constant = hamiltonian.get("nuc_constant", 0.0)
    Z_tensor = hamiltonian["core_tensors"]
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


def _cgf_resource_counts(num_trotter_steps, hamiltonian, has_control):
    """Shared (upper-bound) gate counts for the base and controlled CGF circuits.

    The exact gate count can be lower at runtime because fragments whose basis rotation
    happens to be the identity are skipped when the Hamiltonian data is concrete (not
    traced); this estimate assumes no such fragment is skipped.
    """
    if num_trotter_steps <= 0:
        return {}

    leaf_tensors = hamiltonian["leaf_tensors"]
    num_two_body_fragments = leaf_tensors.shape[0] - 1
    num_modes = leaf_tensors.shape[1]
    n_states = leaf_tensors.shape[2]

    resources = defaultdict(int)
    num_sysrot_calls = num_trotter_steps * (2 * num_two_body_fragments + 1) + 1
    num_twobody_blocks = num_trotter_steps * 2 * num_two_body_fragments
    num_onebody_blocks = num_trotter_steps

    sysrot_key = BasisRotation(Complex[n_states, n_states], wires=Wire[n_states])
    resources[sysrot_key] += num_modes * num_sysrot_calls

    num_pairs = num_modes * (num_modes - 1) // 2
    resources[IsingZZ] += num_twobody_blocks * num_pairs * n_states**2
    resources[RZ] += num_onebody_blocks * num_modes * n_states

    if has_control:
        resources[CNOT] += num_twobody_blocks * num_pairs * 2 * n_states
        resources[CNOT] += num_onebody_blocks * 2 * num_modes * n_states
        # Controlled global phase -> RZ on the control wire.
        resources[RZ] += 1
    else:
        resources[GlobalPhase] += 1

    return dict(resources)


def _trotter_cgf_resources(evolution_time, num_trotter_steps, hamiltonian, wires):
    return _cgf_resource_counts(num_trotter_steps, hamiltonian, has_control=False)


@register_resources(_trotter_cgf_resources, exact=False)
def _trotter_cgf_decomposition(evolution_time, num_trotter_steps, hamiltonian, wires):
    if num_trotter_steps > 0:
        _run_trotter_steps(
            evolution_time, num_trotter_steps, hamiltonian, wires, (), **_CGF_HELPERS
        )
        phi = (_energy_shift(hamiltonian) * evolution_time) % (4 * np.pi)
        GlobalPhase(phi)


add_decomps(TrotterCGF, _trotter_cgf_decomposition)


def _controlled_trotter_cgf_resource(
    base, control_wires, control_values, work_wires, work_wire_type
):
    return _cgf_resource_counts(
        base.arguments["num_trotter_steps"], base.arguments["hamiltonian"], has_control=True
    )


@register_condition(lambda control_wires, **__: len(control_wires) == 1)
@register_resources(_controlled_trotter_cgf_resource, exact=False)
def _controlled_trotter_cgf_decomp(base, control_wires, control_values, work_wires, work_wire_type):
    evolution_time = base.arguments["evolution_time"]
    num_trotter_steps = base.arguments["num_trotter_steps"]
    hamiltonian = base.arguments["hamiltonian"]
    wires = base.arguments["wires"]

    if num_trotter_steps > 0:
        # The double-phase (Fig. 6) controlled circuit is identical to the base circuit
        # with each diagonal block CNOT-sandwiched by the control wire; the sandwiched
        # angles are halved, which is achieved by evolving for evolution_time / 2.
        _run_trotter_steps(
            evolution_time / 2, num_trotter_steps, hamiltonian, wires, control_wires, **_CGF_HELPERS
        )
        # A genuine controlled global phase at the full time: controlled-GlobalPhase(phi)
        # equals RZ(phi) on the control wire (up to an unobservable global phase).
        phi = (_energy_shift(hamiltonian) * evolution_time) % (4 * np.pi)
        RZ(phi, control_wires)


add_decomps("C(TrotterCGF)", flip_zero_control2(_controlled_trotter_cgf_decomp))
