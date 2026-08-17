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
"""Contains the ``TrotterCDF`` template for Compressed Double Factorization Hamiltonians."""

from collections import defaultdict

import numpy as np

from pennylane import math
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.ops import CNOT, RZ, GlobalPhase, IsingZZ, PhaseShift
from pennylane.ops.op_math.controlled2 import flip_zero_control as flip_zero_control2
from pennylane.templates.subroutines.qchem.basis_rotation import BasisRotation
from pennylane.typing import Complex, Wire

from ._trotter_utils import _emit_one_body_rz, _emit_two_body_isingzz, _run_trotter_steps

# pylint: disable=too-many-arguments, no-value-for-parameter, unused-argument


class TrotterCDF(Operator2):
    r"""Second-order Trotter time evolution for a Compressed Double Factorization (CDF) Hamiltonian.

    Implements :math:`e^{-iHt}` for an electronic-structure Hamiltonian in the
    Compressed Double Factorization form, see `arXiv:2506.15784, Sec. III A
    <https://arxiv.org/abs/2506.15784>`__.

    Args:
        evolution_time (float): Total evolution time ``t``.
        num_trotter_steps (int): Number of second-order Trotter steps.
        hamiltonian (dict): A CDF Hamiltonian as a dictionary with keys ``nuc_constant``,
            ``core_tensors``, and ``leaf_tensors``. The expected shapes are
            ``core_tensors: (L+1, N, N)`` (diagonal per fragment) and
            ``leaf_tensors: (L+1, N, N)``, where ``N`` is the number of orbitals and
            ``L`` is the number of two-body fragments.
        wires (Wires): The system wires. CDF expects ``2N`` wires (alpha / beta interleaved).
        double_phase (bool): Only affects the controlled decomposition. If ``False`` (default),
            :func:`~pennylane.ctrl` produces a genuine controlled unitary. If ``True``, it
            produces the double-phase (Fig. 6) Hadamard-test circuit. Has no effect on the
            uncontrolled operator. See usage details below.

    **Example**

    Let us create mock CDF Hamiltonian data with the correct tensor shapes and evolve
    with a few Trotter steps.

    .. code-block:: python

        rng = np.random.default_rng(42)
        N = 2  # orbitals
        L = 1  # two-body fragments
        hamiltonian = {
            "core_tensors": rng.random((L + 1, N, N)),
            "leaf_tensors": rng.random((L + 1, N, N)),
            "nuc_constant": 0.5,
        }

        gate_set = {"BasisRotation", "RZ", "IsingZZ", "GlobalPhase", "ForLoop"}

        @qp.qjit
        @qp.transforms.decompose(gate_set=gate_set)
        @qp.qnode(qp.device("lightning.qubit", wires=2 * N))
        def trotter_circuit():
            qp.TrotterCDF(
                evolution_time=1.0, num_trotter_steps=10, hamiltonian=hamiltonian,
                wires=range(2 * N),
            )
            return qp.state()

    We can inspect the quantum resources required. Note that the order of the keys in
    the ``quantum_operations`` dictionary is not guaranteed, so we sort it before printing:

    >>> specs = qp.specs(trotter_circuit)()["resources"].quantum_operations
    >>> dict(sorted(specs.items()))
    {'GlobalPhase': 1, 'IsingZZ': 120, 'PhaseShift': 62, 'RZ': 40, 'SingleExcitation': 62}

    The :class:`~.PhaseShift` and :class:`~.SingleExcitation` gates are due to
    :class:`~.BasisRotation` decomposing further on ``lightning.qubit``.

    .. details ::
        :title: Usage Details

        Controlling this operator with :func:`~pennylane.ctrl` (a single control wire)
        produces, by default (``double_phase=False``), a genuine controlled evolution.
        In particular,each diagonal rotation is individually controlled at its full angle (the basis
        rotations remain uncontrolled), so the control-0 branch is the identity and the
        circuit implements :math:`|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes e^{-iHt}`.

        With ``double_phase=True`` it instead produces the double-phase Hadamard-test circuit
        of `Fig. 6 of arXiv:2506.15784 <https://arxiv.org/abs/2506.15784>`__: each diagonal
        rotation is sandwiched by a pair of ``CNOT`` gates from the control wire (an
        ancilla-system ``ZZ`` coupling) and its angle is halved, so the control-0 and
        control-1 branches evolve by :math:`e^{-iHt/2}` and :math:`e^{+iHt/2}` respectively.

        .. code-block::

            double_phase=True:  c: ─╭●───────╭●─┤
                            wires: ─╰X──U(ϕ)─╰X─┤
    """

    dynamic_argnames = ("evolution_time",)
    hybrid_argnames = ("hamiltonian",)
    # `hybrid_argnames` and `compilable_argnames` cannot both be non-empty on the same
    # operator, so `num_trotter_steps` (a plain Python int that drives Python-level
    # control flow) is treated as `static_argnames` instead.
    static_argnames = ("num_trotter_steps", "double_phase")

    def __init__(self, evolution_time, num_trotter_steps, hamiltonian, wires, double_phase=False):
        Z = hamiltonian["core_tensors"]
        U = hamiltonian["leaf_tensors"]
        if not (Z.ndim == 3 and U.ndim == 3):
            raise ValueError(
                "TrotterCDF expects a CDF Hamiltonian with core_tensors.ndim == 3 and "
                f"leaf_tensors.ndim == 3. Got core_tensors.ndim={Z.ndim}, "
                f"leaf_tensors.ndim={U.ndim}. For vibrational (CGF) Hamiltonians, use TrotterCGF."
            )
        super().__init__(evolution_time, num_trotter_steps, hamiltonian, wires, double_phase)


def _apply_system_basis_rotation(U, wires):
    """Apply a fragment's basis rotation on the alpha and beta spin channels."""
    if math.is_abstract(U) or not np.allclose(U, np.eye(len(U))):
        BasisRotation(unitary_matrix=U, wires=wires[::2])
        BasisRotation(unitary_matrix=U, wires=wires[1::2])


def _merge_leaves(U_prev, U_curr):
    """Fragment-rotation merge rule ``U_prev^T @ U_curr`` (single ``(N, N)`` matmul)."""
    return U_prev.T @ U_curr


def _transpose_leaf(U):
    """Transpose (adjoint for real orthogonal rotations) of a leaf."""
    return U.T


def _apply_two_body_diagonal(Z, wires, first_order_time_step, control_wires, double_phase):
    """Apply the two-body ``IsingZZ`` layer (base / double-phase / genuine controlled)."""
    num_cas = Z.shape[0]
    # The shared CNOT sandwich is only used by the double-phase construction; the genuine
    # controlled circuit controls each IsingZZ individually inside ``_emit_two_body_isingzz``.
    sandwich = len(control_wires) > 0 and double_phase

    def zz_rotations(wire_idx0):

        @for_loop(wire_idx0 + 1, 2 * num_cas)
        def _zz_rotations(wire_idx1):
            # Prefactor breakdown:
            #   1/8  from (A29)
            #   2    from exp(-i H t) -> IsingZZ(phi) = exp(-i phi Z Z / 2)
            #   2    from symmetrization (k<->l for sigma=tau,
            #                             (k,sigma)<->(l,tau) for sigma!=tau)
            #   => 1/2, times the second-order Trotter split -> 1/4 with a sign.
            angle = -0.25 * Z[wire_idx0 // 2, wire_idx1 // 2] * first_order_time_step
            _emit_two_body_isingzz(
                angle, wires[wire_idx0], wires[wire_idx1], control_wires, double_phase
            )

        if sandwich:
            CNOT([control_wires[0], wires[wire_idx0]])
        _zz_rotations()
        if sandwich:
            CNOT([control_wires[0], wires[wire_idx0]])

    for wire_idx0 in range(2 * num_cas - 1):
        zz_rotations(wire_idx0)


def _apply_one_body_diagonal(Z_one_body, wires, first_order_time_step, control_wires, double_phase):
    """Apply the one-body ``RZ`` layer (base / double-phase / genuine controlled)."""
    num_cas = Z_one_body.shape[0]

    @for_loop(2 * num_cas)
    def z_rotations(wire_idx):
        # Prefactor breakdown:
        #   -1/2 from (A29)
        #   2    from exp(-i H t) -> RZ(phi) = exp(-i phi Z / 2)
        #   2    from merging forward+backward occurrences of the 2nd-order
        #        Trotter formula
        #   => 1
        angle = Z_one_body[wire_idx // 2, wire_idx // 2] * first_order_time_step
        _emit_one_body_rz(angle, wires[wire_idx], control_wires, double_phase)

    z_rotations()


def _energy_shift(hamiltonian):
    """Zero-of-energy shift applied as a ``GlobalPhase`` (or control ``RZ``)."""
    nuc_constant = hamiltonian.get("nuc_constant", 0.0)
    Z_tensor = hamiltonian["core_tensors"]
    # Eq. (A29) first line: nuc + sum_k Z^(0)_{k,k}
    #   - (sum_{l,k,l'} Z^(l)_{k,l'}) / 2 + (sum_{l,k} Z^(l)_{k,k}) / 4
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


def _cdf_resource_counts(num_trotter_steps, hamiltonian, has_control, double_phase=False):
    """Shared (upper-bound) gate counts for the base and controlled CDF circuits.

    The exact gate count can be lower at runtime because fragments whose basis rotation
    happens to be the identity are skipped when the Hamiltonian data is concrete (not
    traced); this estimate assumes no such fragment is skipped.
    """
    if num_trotter_steps <= 0:
        return {}

    leaf_tensors = hamiltonian["leaf_tensors"]
    num_two_body_fragments = leaf_tensors.shape[0] - 1
    num_cas = leaf_tensors.shape[-1]

    resources = defaultdict(int)
    num_sysrot_calls = num_trotter_steps * (2 * num_two_body_fragments + 1) + 1
    num_twobody_blocks = num_trotter_steps * 2 * num_two_body_fragments
    num_onebody_blocks = num_trotter_steps
    num_twobody_rotations = num_twobody_blocks * num_cas * (2 * num_cas - 1)
    num_onebody_rotations = num_onebody_blocks * 2 * num_cas

    sysrot_key = BasisRotation(Complex[num_cas, num_cas], wires=Wire[num_cas])
    resources[sysrot_key] += 2 * num_sysrot_calls

    if not has_control:
        resources[IsingZZ] += num_twobody_rotations
        resources[RZ] += num_onebody_rotations
        resources[GlobalPhase] += 1
    elif double_phase:
        # Double-phase (Fig. 6): bare IsingZZ / RZ rotations, plus one CNOT pair around
        # each diagonal block, plus an RZ on the control wire for the global phase.
        resources[IsingZZ] += num_twobody_rotations
        resources[RZ] += num_onebody_rotations
        resources[CNOT] += num_twobody_blocks * 2 * (2 * num_cas - 1)
        resources[CNOT] += num_onebody_blocks * 4 * num_cas
        resources[RZ] += 1
    else:
        # Genuine controlled: each IsingZZ -> controlled-IsingZZ (4 CNOT + 2 RZ) and each
        # RZ -> controlled-RZ (2 CNOT + 2 RZ); the global phase becomes a PhaseShift on
        # the control wire. There are no bare IsingZZ gates.
        resources[RZ] += 2 * num_twobody_rotations + 2 * num_onebody_rotations
        resources[CNOT] += 4 * num_twobody_rotations + 2 * num_onebody_rotations
        resources[PhaseShift] += 1

    return dict(resources)


def _trotter_cdf_resources(evolution_time, num_trotter_steps, hamiltonian, wires, double_phase):
    return _cdf_resource_counts(num_trotter_steps, hamiltonian, has_control=False)


@register_resources(_trotter_cdf_resources, exact=False)
def _trotter_cdf_decomposition(evolution_time, num_trotter_steps, hamiltonian, wires, double_phase):
    # ``double_phase`` only affects the controlled decomposition; the base operator is
    # always the plain (uncontrolled) e^{-iHt} circuit.
    if num_trotter_steps > 0:
        _run_trotter_steps(
            evolution_time, num_trotter_steps, hamiltonian, wires, (), **_CDF_HELPERS
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
    )


@register_condition(lambda control_wires, **__: len(control_wires) == 1)
@register_resources(_controlled_trotter_cdf_resource, exact=False)
def _controlled_trotter_cdf_decomp(base, control_wires, control_values, work_wires, work_wire_type):
    evolution_time = base.arguments["evolution_time"]
    num_trotter_steps = base.arguments["num_trotter_steps"]
    hamiltonian = base.arguments["hamiltonian"]
    wires = base.arguments["wires"]
    double_phase = base.arguments["double_phase"]

    if num_trotter_steps == 0:
        return

    phi = (_energy_shift(hamiltonian) * evolution_time) % (4 * np.pi)

    if double_phase:
        # Double-phase (Fig. 6) circuit: each diagonal block is CNOT-sandwiched by the
        # control wire with its angle halved (achieved by evolving for evolution_time / 2),
        # and the global phase becomes RZ(phi) on the control wire.
        _run_trotter_steps(
            evolution_time / 2,
            num_trotter_steps,
            hamiltonian,
            wires,
            control_wires,
            True,
            **_CDF_HELPERS,
        )
        RZ(phi, control_wires)
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
        **_CDF_HELPERS,
    )
    PhaseShift(-phi, control_wires)


add_decomps("C(TrotterCDF)", flip_zero_control2(_controlled_trotter_cdf_decomp))
