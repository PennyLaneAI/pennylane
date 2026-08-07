# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantum functions that synthesize a matrix product state into a circuit.

Includes MPS construction, canonical-form checks, unitary completion, the
circuit builder :func:`mps_synthesis`, and the queuing template
:func:`mps_preparation`.
"""

import numpy as np
from scipy.linalg import null_space

import pennylane as qp

from .flag import (
    PartiallyMultiplexedFlag,
    _map_nested_ops,
    add_control,
    flatten_ops,
    merge_partially_multiplexed_flags_in_circuit,
    recursive_generalized_flag_decomposition,
)
from .linalg import (
    csd,
    embed_unitary,
    get_fractal_embedding_states,
    ints_to_control_bits,
    split_diagonal_into_partially_multiplexed_rz,
)

# ============================================================================
# MPS construction, canonical form, and unitary completion
# ============================================================================


def is_right_canonical(tensors, atol=1e-10):
    """
    True iff every tensor satisfies sum_s A^s A^s^H = I_{chi_L}.
    """
    return all(
        np.allclose(
            A.reshape(A.shape[0], -1) @ A.reshape(A.shape[0], -1).conj().T,
            np.eye(A.shape[0]),
            atol=atol,
        )
        for A in tensors
    )


def split_mps(mps):
    """
    Split an MPS into (left, bulk, right).
    Segments are identified by their bond  profile:
        left  : chi_L < chi_R   (1 -> 2 -> ... -> chi)
        bulk  : chi_L == chi_R  (chi -> chi, maximal bond)
        right : chi_L > chi_R   (chi -> ... -> 2 -> 1)
    """
    left, bulk, right = [], [], []
    for A in mps:
        chi_L, _, chi_R = A.shape
        if chi_R > chi_L:
            left.append(A)
        elif chi_R < chi_L:
            right.append(A)
        else:
            bulk.append(A)
    return left, bulk, right


def unitary_completion_tensor(A):
    """
    Unitary completion of a local tensor (chi_L, d, chi_R).

    Three cases:
      - Right interface (only chi_L non-2^k): embed input columns on the
        fractal-active set -> returns (N, N), N = d*chi_R.
      - Left interface (only chi_R non-2^k): embed output rows on the
        fractal-active set -> returns (d*N, d*N), N = 2**ceil(log2 chi_R).
      - Plain completion otherwise: both bonds 2^k (ordinary bulk/boundary) or
        both bonds non-2^k (non-2^k bulk). No embedding here; for the non-2^k
        bulk the fractal embedding is deferred to the flag-decomposition step.
    """
    chi_L, d, chi_R = A.shape

    def is_pow2(k):
        return (k & (k - 1)) == 0

    V = A.transpose(1, 2, 0).reshape(d * chi_R, chi_L)

    # Right interface
    if not is_pow2(chi_L) and is_pow2(chi_R):
        N = d * chi_R
        active, inactive = get_fractal_embedding_states(chi_L, N)
        M = np.zeros((N, N), dtype=complex)
        M[:, active] = V
        if inactive:
            M[:, inactive] = null_space(V.conj().T)
        return M

    # Left interface
    if is_pow2(chi_L) and not is_pow2(chi_R):
        N = 1 << int(np.ceil(np.log2(chi_R)))
        active, _ = get_fractal_embedding_states(d * chi_R, d * N)
        M = np.zeros((d * N, chi_L), dtype=complex)
        M[active, :] = V
        return np.hstack([M, null_space(M.conj().T)])

    # Plain completion
    return np.hstack([V, null_space(V.conj().T)])


def unitary_completion_mps(tensors):
    """
    Unitary completion of a list of right-canonical local tensors.
    Each tensor of shape (chi_L, d, chi_R) = (in_bond, physical, out_bond).
    Physical index as the MSB of the output register.
    """
    completion = [unitary_completion_tensor(A) for A in tensors]
    return completion


def asymmetric_csd(Gk, chi):
    """
    Eq. (E.8) asymmetric CSD.
    Gk is a 4*chi x 4*chi unitary completion of a boundary isometry that maps
    Returns K00, K01 in U(2*chi), theta (chi angles),
    K10 in U(chi).  K11 (U(3*chi)) is never needed (Eqs. E.9-E.11).
    """
    W = Gk[:, :chi]  # The |00>-input isometry, 4*chi x chi
    W0, W1 = W[: 2 * chi], W[2 * chi :]  # Top and bottom part of the isometry

    # Eigenvalues and eigenvectors of the top part (cos^2)
    evals, V = np.linalg.eigh(W0.conj().T @ W0)
    # Sort the eigenvalues in descending order
    idx = np.argsort(-evals)
    # Clip the eigenvalues to be between 0 and 1, and sort the eigenvectors accordingly
    evals, V = np.clip(evals[idx], 0, 1), V[:, idx]
    c, s = np.sqrt(evals), np.sqrt(1 - evals)  # cos, sin
    theta = 2 * np.arccos(np.clip(c, -1, 1))  # RY angles

    def build_U(Wk, diag):
        """Build the left unitaries K00, K01 matrices."""
        cols = [(Wk @ V[:, i]) / d for i, d in enumerate(diag) if d > 1e-9]
        B = np.array(cols).T if cols else np.zeros((2 * chi, 0), complex)
        N = null_space(B.conj().T) if B.shape[1] < 2 * chi else np.zeros((2 * chi, 0), complex)
        return np.hstack([B, N])[:, : 2 * chi]

    K00, K01 = build_U(W0, c), build_U(W1, s)
    return K00, K01, theta, V.conj().T


# ============================================================================
# MPS synthesis and preparation template
# ============================================================================


def merge_staircase_unitaries(unitaries):
    """
    Merge a "staircase" list of unitaries into a single n-qubit unitary.
    This needs to be the unitary completion if not power-of-two.
            ┌────┐
    q0 ─────┤    ├─────────────────
            │    │  ┌────┐
    q1 ─────┤W[0]├──┤    ├─────────
            │    │  │W[1]│  ┌────┐
    q2 ─────┤    ├──┤    ├──┤W[2]├─
            └────┘  └────┘  └────┘
    """
    n = len(unitaries)
    dim = 2**n
    W = np.eye(dim, dtype=complex)
    for k, U in enumerate(unitaries):
        pad = 2**k  # Identity on the k most-significant qubits
        U_full = np.kron(np.eye(pad, dtype=complex), U) if pad > 1 else U
        W = U_full @ W
    return W


def parallel_synthesis_steps(left_boundary, bulk, right_boundary):
    """
    Parallel synthesis steps for the left boundary, bulk, and right boundary.
    The inputs are the unitary completions.
    """

    def _absorb_to_left(cells, trailing):
        """
        This function merges unitaries from right to left and takes into account
        the embedding of the unitaries.
        """
        vs = [c["V"] for c in cells[1:]] + [trailing]
        for cell, V in zip(cells, vs):
            Vshape, Ushape = V.shape[0], cell["K00"].shape[0]
            if Vshape == Ushape:
                cell["K00"] = V @ cell["K00"]
                cell["K01"] = V @ cell["K01"]
            else:
                chi = Ushape if Vshape > Ushape else Vshape
                N = 1 << int(np.ceil(np.log2(chi)))
                active, _ = get_fractal_embedding_states(chi, N)
                if Vshape > Ushape:
                    cell["K00"] = embed_unitary(cell["K00"], N, active)
                    cell["K01"] = embed_unitary(cell["K01"], N, active)
                else:
                    V = embed_unitary(V, N, active)
                cell["K00"] = V @ cell["K00"]
                cell["K01"] = V @ cell["K01"]
        for cell in cells[1:]:
            cell["V"] = None
        return cells

    bulk_cells, boundary_cells = [], []

    # Apply asymmetric CSD to left boundary, Eq. (E.8)-(E.11)
    for Gk in left_boundary:
        chi = Gk.shape[0] // 4
        K00, K01, theta, V = asymmetric_csd(Gk, chi)
        boundary_cells.append({"K00": K00, "K01": K01, "theta": theta, "V": V})

    # Apply standard CSD to bulk, Eq. (E.1)
    for Gk in bulk:
        K00, K01, theta, V, _, _, _, _ = csd(Gk, shift=True)
        bulk_cells.append({"K00": K00, "K01": K01, "theta": theta, "V": V})

    # Merge unitaries from right to left cells
    W = merge_staircase_unitaries(right_boundary)
    bulk_cells = _absorb_to_left(bulk_cells, W)
    boundary_cells = _absorb_to_left(boundary_cells, bulk_cells[0]["V"])

    return boundary_cells, bulk_cells


def flag_decompose_multiplexor(K, phys2, bond):
    """
    Decompose the unitaries K appearing in Eq. (E.12).
    |0⟩ ───────■───────  phys1
             ┌─┴─┐
    |0⟩ ─────┤   ├─────  phys2
             │ K │
    k-1 ──/──┤   ├─────  bond
             └───┘
    K acts on the wires (phys2, bond) and phys2 is the MSQ.
    """

    def _flag_decompose(U, w):
        # Base case: a 1x1 unitary on an empty bond register is just a phase.
        if len(w) == 0:
            return [], np.asarray(U, dtype=complex).reshape(-1)
        return recursive_generalized_flag_decomposition(U, w, _top=False)

    # Symmetric CSD, split on phys2
    R0, R1, theta_csd, L0, _, _, _, _ = csd(K, shift=True)

    # Flag-decompose on bond register (base-case safe)
    FL, DL = _flag_decompose(L0, bond)

    # Merge DL into R_i
    R0, R1 = R0 * DL, R1 * DL

    # Flag-decompose R_i
    FR0, DR0 = _flag_decompose(R0, bond)
    FR1, DR1 = _flag_decompose(R1, bond)
    FR0 = _map_nested_ops(FR0, lambda op: add_control(op, phys2, 0))
    FR1 = _map_nested_ops(FR1, lambda op: add_control(op, phys2, 1))
    DR = np.concatenate((DR0, DR1))

    return FL, FR0, FR1, DR, theta_csd


def boundary_sequential_step(cells, aux_wires, phys_wires):
    """
    Sequential (left-to-right) synthesis of the growing boundary tensors.
    This corresponds to Eqs. (E.13) - (E.15).

    Boundary bond dimensions are always powers of two (besides for the last one).
    The flags are therefore fully multiplexed (except for the last one).
    """
    circuit, delta = [], None  # Initial residual diagonal

    for j, cell in enumerate(cells):
        n_sub = cell["K00"].shape[0].bit_length() - 1
        phys1, sub = phys_wires[j], aux_wires[-n_sub:]
        phys2, bond = sub[0], sub[1:]
        chi = cell["K00"].shape[0] // 2  # Left virtual bond dimension

        # Get control values for fully multiplexed flags
        cvs_bond = ints_to_control_bits(range(chi), len(bond)) if bond else []
        cvs_sub = ints_to_control_bits(range(2 * chi), n_sub)

        # Apply flag decomposition to each control value
        branch_ops, branch_diags = [], []
        for cv, K in [[0, cell["K00"]], [1, cell["K01"]]]:
            if delta is not None:  # Merge diagonal from previous tensor
                K = K * np.kron(np.ones(2), delta)

            FL, FR0, FR1, DR, theta_csd = flag_decompose_multiplexor(K, phys2, bond)
            RY = PartiallyMultiplexedFlag(
                np.zeros_like(theta_csd), theta_csd, bond + [phys2], cvs_bond
            )

            # Attach the MSQ as control qubit
            ops = _map_nested_ops(
                flatten_ops([FL, RY, FR0, FR1]),
                lambda op, w=phys1, c=cv: add_control(op, w, c),
            )
            branch_ops.append(ops)
            branch_diags.append(DR)

        Gamma = merge_partially_multiplexed_flags_in_circuit(flatten_ops(branch_ops))
        Delta = np.concatenate(branch_diags)  # Full diagonal

        # Split into RZ on phys1
        phi1, rem1, _ = split_diagonal_into_partially_multiplexed_rz(
            Delta, sub + [phys1], range(2 * chi)
        )
        delta = rem1[: 2 * chi]  # To be merged into next tensor

        circuit.append(
            PartiallyMultiplexedFlag(
                np.zeros_like(cell["theta"]), cell["theta"], bond + [phys1], cvs_bond
            )
        )
        circuit += list(flatten_ops(Gamma))
        circuit.append(
            PartiallyMultiplexedFlag(
                np.asarray(phi1, float), np.zeros(len(phi1)), sub + [phys1], cvs_sub
            )
        )

    return merge_partially_multiplexed_flags_in_circuit(circuit), delta


def cascade_diagonal_into_rz_msq(full_diagonal, wires):
    """
    Cascade a full diagonal into a staircase of multiplexed Rz gates.

          ┌─────┐
     q0 ──┤ Rzθ ├─────────────────────────────────
          └──┬──┘   ┌─────┐
     q1 ─────■──────┤ Rzθ ├───────────────────────
             │      └──┬──┘   ┌─────┐
     q2 ─────■─────────■──────┤ Rzθ ├─────────────
             │         │      └──┬──┘   ┌─────┐
     q3 ─────■─────────■─────────■──────┤ Rzφ ├───
                                        └───-─┘
    """
    diag = np.asarray(full_diagonal, dtype=complex)
    assert diag.shape == (2 ** len(wires),), "len(diagonal) must be 2**len(wires)"
    flags = []
    while wires:
        msq, controls = wires[0], wires[1:]  # MSQ as target
        num_ctrl = len(wires) - 1
        control_states = list(range(2**num_ctrl))  # Fully multiplexed
        angles, remaining, _ = split_diagonal_into_partially_multiplexed_rz(
            diag, controls + [msq], control_states
        )
        cvs = ints_to_control_bits(control_states, num_ctrl) if controls else []
        flags.append(
            PartiallyMultiplexedFlag(
                np.asarray(angles, float), np.zeros(len(angles)), controls + [msq], cvs
            )
        )
        diag = remaining[: 2**num_ctrl]
        wires = controls
    global_phase = complex(diag[0])
    return flags, global_phase


def bulk_sequential_step(cells, delta, aux_wires, phys_wires):
    """
    Sequential (left-to-right) synthesis of the bulk tensors.
    Bulk bond dimensions can be non-power-of-two. In that case,
    special care is taken to handle the embedding of the unitaries
    at the interaces.
    """
    n = len(aux_wires)
    N_bond = 2**n
    chi = len(np.atleast_1d(cells[0]["theta"]))
    active_bond, inactive_bond = get_fractal_embedding_states(chi, N_bond)
    cvs = ints_to_control_bits(active_bond, n)
    full = list(range(N_bond))
    cvs_full = ints_to_control_bits(full, n)

    ops = []
    for j, cell in enumerate(cells):
        phys = phys_wires[j]
        U0, U1 = cell["K00"], cell["K01"]
        is_last = U0.shape[0] == N_bond  # Last cell has W merged

        if is_last:
            # Last cell absorbs the full residual
            U0, U1 = U0 * delta, U1 * delta
            carry = np.ones(N_bond, complex)
        else:
            # Interior cells keep the active/inactive split and carry inactive phases
            U0, U1 = U0 * delta[active_bond], U1 * delta[active_bond]
            carry = np.ones(N_bond, complex)
            carry[inactive_bond] = delta[inactive_bond]

        # Flag-decompose the unitaries
        F0, D0 = recursive_generalized_flag_decomposition(U0, aux_wires, _top=False)
        F1, D1 = recursive_generalized_flag_decomposition(U1, aux_wires, _top=False)
        F0 = _map_nested_ops(F0, lambda op, w=phys: add_control(op, w, 0))
        F1 = _map_nested_ops(F1, lambda op, w=phys: add_control(op, w, 1))
        Gamma = merge_partially_multiplexed_flags_in_circuit(flatten_ops([F0, F1]))
        D0, D1 = D0 * carry, D1 * carry
        Delta = np.concatenate((D0, D1))

        # Fully multiplex over N_bond for the last cell
        if is_last:
            ctrl, cv_use = full, cvs_full
        else:
            ctrl, cv_use = active_bond, cvs

        # Split diagonal into Rz and residual
        phi, residual_full, _ = split_diagonal_into_partially_multiplexed_rz(
            Delta, aux_wires + [phys], ctrl
        )
        phi = np.asarray(phi, float)
        delta = residual_full[:N_bond]

        ops.append(PartiallyMultiplexedFlag(np.zeros(chi), cell["theta"], aux_wires + [phys], cvs))
        ops += list(flatten_ops(Gamma))
        ops.append(PartiallyMultiplexedFlag(phi, np.zeros_like(phi), aux_wires + [phys], cv_use))

    return ops, delta


def mps_synthesis(mps_tensors, aux_wires, phys_wires):
    r"""Synthesize a right-canonical matrix product state into a list of operations.

    This is the builder underlying :func:`mps_preparation`. It returns the flag
    circuit together with the residual global phase, without queuing anything.

    Args:
        mps_tensors (Sequence[np.ndarray]): Right-canonical MPS tensors, each of
            shape ``(chi_L, d, chi_R)`` (input bond, physical, output bond).
        aux_wires (Sequence): Auxiliary (bond) wires; there must be
            ``ceil(log2(chi))`` of them, where ``chi`` is the maximal bond dimension.
        phys_wires (Sequence): Physical wires, one per MPS tensor, in circuit order.

    Returns:
        tuple[list, complex]: ``(circuit, global_phase)`` where ``circuit`` is a
        list of operations (mostly :class:`PartiallyMultiplexedFlag`) and
        ``global_phase`` is the residual scalar phase.
    """
    assert is_right_canonical(mps_tensors), "MPS tensors must be right-canonical"

    # Obtain the unitary completions
    left, bulk, right = split_mps(mps_tensors)
    left_completion = unitary_completion_mps(left)
    right_completion = unitary_completion_mps(right)
    bulk_completion = unitary_completion_mps(bulk)

    # Split the physical wires
    n_left = len(left)
    phys_wires_left = phys_wires[:n_left]
    phys_wires_bulk = phys_wires[n_left:]

    # Perform the initial merges
    boundary_cells, bulk_cells = parallel_synthesis_steps(
        left_completion, bulk_completion, right_completion
    )

    # Synthesize the left boundary
    left_ops, delta = boundary_sequential_step(boundary_cells, aux_wires, phys_wires_left)
    # Synthesize the bulk
    bulk_ops, delta = bulk_sequential_step(bulk_cells, delta, aux_wires, phys_wires_bulk)
    # Synthesize the right boundary
    right_ops, global_phase = cascade_diagonal_into_rz_msq(delta, aux_wires)

    circuit = left_ops + bulk_ops + right_ops

    return circuit, global_phase


def mps_preparation(mps_tensors, aux_wires, phys_wires):
    r"""Prepare a right-canonical matrix product state on a quantum register.

    Queues the flag circuit that maps the all-zero state to the state whose
    amplitudes are given by ``mps_tensors``. The physical wires carry the state;
    the auxiliary (bond) wires are returned to :math:`\lvert 0 \rangle`.

    The synthesis follows the generalized-flag / MPS-preparation construction:
    the boundary and bulk tensors are unitary-completed, decomposed via
    Cosine-Sine and recursive flag decompositions into
    :class:`PartiallyMultiplexedFlag` operations, and the residual diagonal is
    cascaded into multiplexed :math:`R_z` gates plus a global phase.

    Args:
        mps_tensors (Sequence[np.ndarray]): Right-canonical MPS tensors, each of
            shape ``(chi_L, d, chi_R)`` (input bond, physical, output bond).
        aux_wires (Sequence): Auxiliary (bond) wires; there must be
            ``ceil(log2(chi))`` of them, where ``chi`` is the maximal bond dimension.
        phys_wires (Sequence): Physical wires, one per MPS tensor, in circuit order.

    Returns:
        complex: The global phase applied by :class:`~pennylane.GlobalPhase`.

    **Example**

    Given a list of right-canonical MPS tensors ``mps`` (each of shape
    ``(chi_L, d, chi_R)``), with physical wires ``phys`` (one per tensor) and
    ``ceil(log2(chi))`` auxiliary bond wires ``aux``:

    .. code-block:: python

        dev = qp.device("default.qubit", wires=sorted(set(phys) | set(aux)))

        @qp.qnode(dev)
        def circuit():
            mps_preparation(mps, aux, phys)
            return qp.state()
    """
    # Build the circuit with queuing suspended: mps_synthesis instantiates many
    # intermediate operators that would otherwise auto-queue onto the tape.
    with qp.QueuingManager.stop_recording():
        circuit, global_phase = mps_synthesis(mps_tensors, aux_wires, phys_wires)
    for op in circuit:
        qp.apply(op)
    qp.GlobalPhase(-np.angle(global_phase))
    return global_phase
