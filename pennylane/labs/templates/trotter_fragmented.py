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
"""Contains the Trotter templates for fragmented Hamiltonians."""

import pennylane as qp
import numpy as np


def trotter_factorized(evolution_time, num_trotter_steps, hamiltonian, wires,
                       control_wires=None):
    r"""Second-order Trotter time evolution for a factorized Hamiltonian.

    Works for both electronic CDF and vibrational CGF Hamiltonians.

    Args:
        evolution_time (float): Total evolution time ``t``.
        num_trotter_steps (int): Number of second-order Trotter steps.
        hamiltonian (dict): Dictionary with keys ``"core_tensors"`` and ``"leaf_tensors"``.
            ``"nuc_constant"`` is optional (defaults to 0.0).
            * CDF shapes: ``core_tensors: (L+1, N, N)`` (diagonal per fragment),
              ``leaf_tensors: (L+1, N, N)``.
            * CGF shapes: ``core_tensors: (L+1, M, M, N, N)``,
              ``leaf_tensors:  (L+1, M, N, N)``.
        wires (Wires): System wires. CDF expects ``2N`` wires (alpha / beta interleaved).
            CGF expects ``M*N`` wires arranged mode-major: wire ``l*N + p``
            corresponds to modal ``p`` of mode ``l`` (unary/SBE layout).
        control_wires (Wires | None): Optional control wires for a Hadamard test. When present, each
            diagonal Pauli rotation is sandwiched by CNOTs controlled on
            ``control_wires[0]`` and a global phase correction is applied.
    """

    Z = hamiltonian["core_tensors"]
    U = hamiltonian["leaf_tensors"]

    if Z.ndim == 3 and U.ndim == 3:
        frag_scheme = "cdf"
    elif Z.ndim == 5 and U.ndim == 4:
        frag_scheme = "cgf"
    else:
        raise ValueError(
        "Could not auto-detect Hamiltonian type. "
        f"Got core_tensors.ndim={Z.ndim}, leaf_tensors.ndim={U.ndim}. ")

    if num_trotter_steps > 0:
        second_order_time_step = evolution_time / num_trotter_steps
    else:
        second_order_time_step = 0.0

    @qp.for_loop(num_trotter_steps)
    def trotter_steps(step_idx, hamiltonian):
        _trotter_step(step_idx, second_order_time_step, hamiltonian, wires,
                      control_wires, frag_scheme)
        return hamiltonian

    trotter_steps(hamiltonian)

    if num_trotter_steps > 0:
        U_tensor = hamiltonian["leaf_tensors"]
        very_last_U = _transpose_leaf(U_tensor[1], frag_scheme)
        _apply_system_basis_rotation(very_last_U, wires, frag_scheme)

    # Global phase from the Hamiltonian zero-energy shift is only relevant
    # when the evolution is controlled (Hadamard test).
    if control_wires is not None:
        energy_shift = _energy_shift(hamiltonian, frag_scheme)
        phi = (energy_shift * evolution_time) % (4 * np.pi)
        # exp(-i phi) would be PhaseShift(-phi) controlled. The double-phase
        # trick turns it into RZ(-phi) (differ only by an unobservable global
        # phase, but we keep it exact for bookkeeping).
        qp.RZ(-phi, control_wires)

def _trotter_step(step_idx, second_order_time_step, hamiltonian, wires,
                  control_wires, frag_scheme):
    """Single second-order Trotter step for either CDF or CGF Hamiltonian.
    """
    if qp.compiler.active():
        wires = qp.math.array(wires, like="jax")
        if control_wires is not None:
            control_wires = qp.math.array(control_wires, like="jax")

    U_tensor = hamiltonian["leaf_tensors"]
    Z_tensor = hamiltonian["core_tensors"]

    # U_tensor contains rotations for one-body term and L two-body terms
    # we denote L as ``num_two_body_fragments``.
    num_two_body_fragments = U_tensor.shape[0] - 1

    first_order_time_step = second_order_time_step / 2


    def two_body_fragments(fragment_idx, prev_fragment_idx):
        U = jax.lax.cond(
            prev_fragment_idx < 0,
            lambda U_tensor, fragment_idx, prev_fragment_idx: U_tensor[fragment_idx],
            lambda U_tensor, fragment_idx, prev_fragment_idx:
                _merge_leaves(U_tensor[prev_fragment_idx],
                              U_tensor[fragment_idx], frag_scheme),
            U_tensor, fragment_idx, prev_fragment_idx,
        )
        Z = Z_tensor[fragment_idx]
        _apply_system_basis_rotation(U, wires, frag_scheme)
        _apply_two_body_diagonal(Z, wires, first_order_time_step,
                                 control_wires, frag_scheme)
        return fragment_idx

    def one_body_fragment():
        U_one = U_tensor[0] if frag_scheme == "cdf" else U_tensor[0]
        U = _merge_leaves(U_tensor[num_two_body_fragments], U_one, frag_scheme)
        _apply_system_basis_rotation(U, wires, frag_scheme)
        _apply_one_body_diagonal(Z_tensor[0], wires, first_order_time_step,
                                 control_wires, frag_scheme)

    prev_fragment_idx_forward = qp.math.sign(2 * step_idx - 1)
    qp.for_loop(1, num_two_body_fragments + 1)(two_body_fragments)(
        prev_fragment_idx_forward
    )

    one_body_fragment()

    prev_fragment_idx_backward = 0
    qp.for_loop(num_two_body_fragments, 0, -1)(two_body_fragments)(
        prev_fragment_idx_backward
    )


def _apply_system_basis_rotation(U, wires, frag_scheme):
    """Apply a fragment's basis rotation on the whole system."""
    if frag_scheme == "cdf":
        if qp.math.is_abstract(U):
            qp.BasisRotation(unitary_matrix=U, wires=wires[::2])
            qp.BasisRotation(unitary_matrix=U, wires=wires[1::2])
        elif not np.allclose(U, np.eye(len(U))):
            qp.BasisRotation(unitary_matrix=U, wires=wires[::2])
            qp.BasisRotation(unitary_matrix=U, wires=wires[1::2])
    else:
        num_modes, n_states, _ = U.shape
        # The fragment's leaf O stores the "rotate-from-bare-to-diagonal-basis"
        # direction.  The circuit needs to rotate the qubits so that the
        # diagonal layer acts in the correct basis: qml.BasisRotation(U) with
        # U = O^T implements the single-particle map O^T (moving from bare
        # modal states to the diagonal basis).  Hence we pass the transpose.
        for l in range(num_modes):
            U_l = qp.math.swapaxes(U[l], -2, -1)
            mode_wires = wires[l * n_states:(l + 1) * n_states]
            if qp.math.is_abstract(U_l):
                qp.BasisRotation(unitary_matrix=U_l, wires=mode_wires)
            elif not np.allclose(U_l, np.eye(n_states)):
                qp.BasisRotation(unitary_matrix=U_l, wires=mode_wires)


def _merge_leaves(U_prev, U_curr, frag_scheme):
    """Fragment-rotation merge rule: ``U_prev^dagger @ U_curr``.

    CDF: a single ``(N, N)`` matmul.
    CGF: per-mode matmul, vectorized via ``einsum``.
    """
    if frag_scheme == "cdf":
        return U_prev.T @ U_curr
    return qp.math.einsum("lji,ljk->lik", U_prev, U_curr)


def _transpose_leaf(U, frag_scheme):
    """Transpose (i.e. adjoint for real orthogonal rotations) of a leaf."""
    if frag_scheme == "cdf":
        return U.T
    # CGF: batch-transpose over the leading mode axis.
    return qp.math.swapaxes(U, -2, -1)


def _apply_two_body_diagonal(Z, wires, first_order_time_step, control_wires,
                             mode):
    if mode == "cdf":
        num_cas = Z.shape[0]

        @qp.for_loop(2 * num_cas - 1)
        def zz_rotations(wire_idx0):

            @qp.for_loop(wire_idx0 + 1, 2 * num_cas)
            def _zz_rotations(wire_idx1):
                # Prefactor breakdown (see original trotter_cdf.py):
                #   1/8  from (A29)
                #   2    from exp(-i H t) -> IsingZZ(phi) = exp(-i phi Z Z / 2)
                #   -1/2 from the double-phase trick
                #   2    from symmetrization (k<->l for sigma=tau,
                #                             (k,sigma)<->(l,tau) for sigma!=tau)
                #   => -1/4
                angle = -0.25 * Z[wire_idx0 // 2, wire_idx1 // 2] * first_order_time_step
                qp.IsingZZ(angle, [wires[wire_idx0], wires[wire_idx1]])

            # TODO: support multiple control wires (not needed for Hadamard tests)
            if control_wires is not None:
                qp.CNOT([control_wires[0], wires[wire_idx0]])
            _zz_rotations()
            if control_wires is not None:
                qp.CNOT([control_wires[0], wires[wire_idx0]])

        zz_rotations()
    else:
        num_modes = Z.shape[0]
        n_states = Z.shape[2]

        ising_coeff = 0.5

        for l in range(1, num_modes):
            for m in range(l):  # strict lower triangle: l > m
                Z_lm = Z[l, m]

                @qp.for_loop(n_states)
                def _p_loop(p, Z_lm=Z_lm, l=l, m=m):
                    wire_lp = wires[l * n_states + p]

                    @qp.for_loop(n_states)
                    def _q_loop(q, Z_lm=Z_lm, p=p, wire_lp=wire_lp, l=l, m=m):
                        wire_mq = wires[m * n_states + q]
                        lam = Z_lm[p, q]
                        angle = ising_coeff * lam * first_order_time_step
                        if control_wires is not None:
                            qp.CNOT([control_wires[0], wire_lp])
                        qp.IsingZZ(angle, [wire_lp, wire_mq])
                        if control_wires is not None:
                            qp.CNOT([control_wires[0], wire_lp])

                    _q_loop()

                _p_loop()

def _apply_one_body_diagonal(Z_one_body, wires, first_order_time_step,
                             control_wires, mode):
    if mode == "cdf":
        num_cas = Z_one_body.shape[0]

        @qp.for_loop(2 * num_cas)
        def z_rotations(wire_idx):
            # Prefactor breakdown:
            #   -1/2 from (A29)
            #   2    from exp(-i H t) -> RZ(phi) = exp(-i phi Z / 2)
            #   -1/2 from the double-phase trick
            #   2    from merging forward+backward occurrences of the 2nd-order
            #        Trotter formula
            #   => 1
            angle = Z_one_body[wire_idx // 2, wire_idx // 2] * first_order_time_step

            if control_wires is not None:
                qp.CNOT([control_wires[0], wires[wire_idx]])
            qp.RZ(angle, wires[wire_idx])
            if control_wires is not None:
                qp.CNOT([control_wires[0], wires[wire_idx]])

        z_rotations()
    else:
        num_modes = Z_one_body.shape[0]
        n_states = Z_one_body.shape[2]

        @qp.for_loop(num_modes)
        def mode_loop(l):

            @qp.for_loop(n_states)
            def modal_loop(p):
                wire_lp = wires[l * n_states + p]
                # One-body prefactor derivation:
                #   n^l_p = (I - Z_{l,p}) / 2  ->  Z-piece has coefficient -eps/2
                #   target operator: exp(-i eps n t) has Z-piece exp(+i eps t / 2 Z)
                #     which equals RZ(-eps t).
                #   This fragment is visited ONCE per Trotter step, at duration
                #     first_order_time_step = dt_trotter / 2, so to accumulate a
                #     total RZ angle of -eps * dt_trotter we need angle-per-step
                #     = -eps * dt_trotter = -2 * eps * first_order_time_step.
                # So alpha_oneB = -2.
                angle = -2.0 * Z_one_body[l, l, p, p] * first_order_time_step

                if control_wires is not None:
                    qp.CNOT([control_wires[0], wire_lp])
                qp.RZ(angle, wire_lp)
                if control_wires is not None:
                    qp.CNOT([control_wires[0], wire_lp])

            modal_loop()

        mode_loop()



# ---- CGF diagonal gates (SBE / unary encoding) -----------------------------
#
# Encoding assumption: each of the M modes is encoded on N qubits with
# exactly one qubit excited (unary / single-boson encoding, SBE). With this
# choice, the number operator on modal p of mode l maps to
#     n^l_p   ===   (I - Z_{l*N + p}) / 2
# on the logical subspace. Therefore:
#   * exp(-i t eps n^l_p) = exp(-i t eps / 2) * RZ(-t eps, wire)
#   * exp(-i t lam n^l_p n^m_q)
#       = exp(-i t lam / 4) * RZ(-t lam / 2, p) * RZ(-t lam / 2, q)
#         * IsingZZ(t lam / 2, [p, q])
# The leading "identity phases" are collected into ``_energy_shift`` below.
#
# Trotter bookkeeping (matching CDF conventions):
#   * Each two-body fragment is swept twice per step (forward + backward) at
#     ``first_order_time_step = t / (2 * num_trotter_steps)``. No extra
#     factor-of-2 is needed because we already feed it to both passes.
#   * The one-body fragment appears once per step with a factor-of-2 from
#     merging the forward+backward halves (matches CDF one-body comment).
#   * Controlled (double-phase) evolution introduces an extra -1/2 factor on
#     the angle. To stay consistent with ``trotter_cdf.py`` we bake that
#     -1/2 permanently into the prefactor regardless of ``control_wires``.


# ---------------------------------------------------------------------------
# Zero-energy shift (global phase on controlled evolution)
# ---------------------------------------------------------------------------

def _energy_shift(hamiltonian, mode):
    """Compute the zero-of-energy shift that must be applied as a controlled
    ``RZ`` during a Hadamard test."""
    nuc_constant = hamiltonian.get("nuc_constant", 0.0)
    Z_tensor = hamiltonian["core_tensors"]

    if mode == "cdf":
        # Eq. (A29) first line: nuc + sum_k Z^(0)_{k,k}
        #   - (sum_{l,k,l'} Z^(l)_{k,l'}) / 2
        #   + (sum_{l,k} Z^(l)_{k,k}) / 4
        phase_from_mod_one_body = qp.math.trace(Z_tensor[0])
        phase_from_two_body = (
            -qp.math.sum(Z_tensor[1:]) / 2
            + qp.math.sum(qp.math.trace(Z_tensor[1:], axis1=1, axis2=2)) / 4
        )
        return nuc_constant + phase_from_mod_one_body + phase_from_two_body


    # One-body diagonal: nested traces over axes (-2, -1) twice pick out the
    # (l == m, p == q) entries and sum ε^l_p.
    one_body_diag = qp.math.trace(
        qp.math.trace(Z_tensor[0], axis1=-2, axis2=-1),
        axis1=-2,
        axis2=-1,
    )  # scalar: Σ_{l, p} ε^l_p
    phase_from_one_body = one_body_diag / 2

    return nuc_constant + phase_from_one_body
