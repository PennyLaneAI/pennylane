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
    embed_unitary,
    get_fractal_embedding_states,
    ints_to_control_bits,
    split_diagonal_into_partially_multiplexed_rz,
    synthesis_csd,
)

# ============================================================================
# MPS construction, canonical form, and unitary completion
# ============================================================================


def is_right_canonical(tensors, atol=1e-10):
    """
    True iff every tensor satisfies sum_s A^s A^s^H = I_{chi_L}.
    """
    return all(
        qp.math.allclose(
            A.reshape(A.shape[0], -1) @ A.reshape(A.shape[0], -1).conj().T,
            qp.math.eye(A.shape[0]),
            atol=atol,
        )
        for A in tensors
    )


def split_mps(mps):
    """
    Split an MPS into contiguous ``(left, bulk, right)`` segments, preserving site order.

    Segments are identified by their bond profile:
        left  : chi_L < chi_R   (1 -> 2 -> ... -> chi)
        bulk  : chi_L == chi_R  (chi -> chi, maximal bond)
        right : chi_L > chi_R   (chi -> ... -> 2 -> 1)

    Only a single unimodal profile is supported: all ``left`` sites, then all ``bulk``
    sites, then all ``right`` sites (bond dimension non-decreasing then non-increasing).
    A non-unimodal profile such as ``1 -> 2 -> 1 -> 2 -> 1`` is rejected with a
    ``ValueError`` rather than being silently reordered.
    """
    # Phase index per site: left -> 0, bulk -> 1, right -> 2. A supported profile has a
    # non-decreasing phase sequence, i.e. lefts, then bulks, then rights, contiguously.
    phases = []
    for A in mps:
        chi_L, _, chi_R = A.shape
        phases.append(0 if chi_R > chi_L else (1 if chi_R == chi_L else 2))

    if any(later < earlier for earlier, later in zip(phases, phases[1:])):
        bond_profile = [A.shape[0] for A in mps] + [mps[-1].shape[2]]
        raise ValueError(
            "mps_synthesis only supports a single left/bulk/right bond profile "
            "(bond dimension non-decreasing then non-increasing). Got the non-unimodal "
            f"bond profile {bond_profile}."
        )

    left = [A for A, phase in zip(mps, phases) if phase == 0]
    bulk = [A for A, phase in zip(mps, phases) if phase == 1]
    right = [A for A, phase in zip(mps, phases) if phase == 2]
    return left, bulk, right


def unitary_completion_tensor(A):
    r"""Unitary completion of a local MPS tensor of shape ``(chi_L, d, chi_R)``.

    Three cases by bond parity:
      - right interface (only ``chi_L`` non-``2^k``): embed input columns on the
        fractal-active set, returning an ``(N, N)`` matrix with ``N = d * chi_R``;
      - left interface (only ``chi_R`` non-``2^k``): embed output rows, ``N = 2**ceil(log2 chi_R)``;
      - plain completion otherwise (fractal embedding of a non-``2^k`` bulk is deferred
        to the flag-decomposition step).
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
    r"""Asymmetric Cosine-Sine Decomposition of a boundary isometry (Eq. (E.8)).

    ``Gk`` is a ``4*chi x 4*chi`` unitary completion of the boundary isometry.
    Returns:
        tuple: ``(K00, K01, theta, K10)`` — ``K00, K01`` in ``U(2*chi)``, the ``chi``
        :math:`R_y` angles ``theta``, and ``K10`` in ``U(chi)``. ``K11`` is never needed
        (Eqs. (E.9)-(E.11) of Kottmann et al. <https://arxiv.org/abs/2603.20376>`__).
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
    r"""Decompose each MPS segment's unitary completion independently (parallelizable).

    Follows the MPS-preparation scheme of
    `Kottmann et al. <https://arxiv.org/abs/2603.20376>`__. Left-boundary isometries are
    decomposed via the asymmetric CSD (Eqs. (E.8)-(E.11)) into :math:`K_{10}`, a multiplexed
    :math:`R_y`, and :math:`K_{0j}`:

    .. code-block::

               ┌─────┐                          ┌────┐
        q0|0⟩ ─┤     ├─      q0|0⟩ ─────────────┤ Ry ├──────■──────
               │     │                          └─┬──┘   ┌──┴──┐
        q1|0⟩ ─┤ G_k ├─  =   q1|0⟩ ───────────────■──────┤     ├───
               │     │               ┌──────┐     │      │ K_0j│
        k-1 ─/─┤     ├─      k-1 ─/──┤ K_10 ├─────■──────┤     ├───
               └─────┘               └──────┘            └─────┘

    Bulk unitaries use the standard CSD (Eq. (E.1)) into :math:`V`, a multiplexed :math:`R_y`,
    and :math:`K_{0j}`:

    .. code-block::

               ┌─────┐                        ┌────┐
        q0|0⟩ ─┤     ├─     q0|0⟩ ────────────┤ Ry ├──────■──────
               │ G_k │  =           ┌───┐     └─┬──┘   ┌──┴───┐
        n ──/──┤     ├─     n ──/───┤ V ├───────■──────┤ K_0j ├───
               └─────┘              └───┘              └──────┘

    The right-boundary tensors are merged into one trailing unitary and absorbed right-to-left,
    removing the uncontrolled :math:`V` factors. The resulting left-boundary cell is:

    .. code-block::

               ┌────┐             │
        q0|0⟩ ─┤ Ry ├─────■───────┘        (physical bond)
               └─┬──┘  ┌──┴───┐
        q1|0⟩ ───┼─────┤      ├──────      (additional virtual bond)
                 │     │ K_0j │
        k-1 ─/───■─────┤      ├──────      (previous bond register)
                       └──────┘

    and the resulting bulk cell is:

    .. code-block::

               ┌────┐             │
        q0|0⟩ ─┤ Ry ├─────■───────┘        (physical bond)
               └─┬──┘  ┌──┴───┐
                 │     │ K_0j │
        k-1 ─/───■─────┤      ├──────      (full bond register)
                       └──────┘

    Args:
        left_boundary (Sequence[np.ndarray]): left-boundary unitary completions
            (``4*chi x 4*chi``).
        bulk (Sequence[np.ndarray]): bulk unitary completions (``2*chi x 2*chi``).
        right_boundary (Sequence[np.ndarray]): right-boundary completions, merged right-to-left.

    Returns:
        tuple[list[dict], list[dict]]: the boundary and bulk cells, each a dict with keys
        ``"K00"``, ``"K01"``, ``"theta"``, and ``"V"`` (``None`` once absorbed).

    .. seealso:: :func:`~.asymmetric_csd`, :func:`~.boundary_sequential_step`,
        :func:`~.bulk_sequential_step`
    """

    def _absorb_to_left(cells, trailing):
        """Fold each cell's right unitary into its ``K00``/``K01`` (in place, fractal-embedding at non-2^k interfaces)."""
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
        K00, K01, theta, V, _, _, _, _ = synthesis_csd(Gk, shift=True)
        bulk_cells.append({"K00": K00, "K01": K01, "theta": theta, "V": V})

    # Merge unitaries from right to left cells
    W = merge_staircase_unitaries(right_boundary)
    bulk_cells = _absorb_to_left(bulk_cells, W)
    # With no bulk, the boundary abuts the right boundary directly and absorbs W;
    # otherwise it absorbs the leftmost bulk cell's right unitary.
    trailing = bulk_cells[0]["V"] if bulk_cells else W
    boundary_cells = _absorb_to_left(boundary_cells, trailing)

    return boundary_cells, bulk_cells


def flag_decompose_multiplexor(K, bond_additional, bond_previous):
    r"""Flag-decompose the multiplexed unitary :math:`K` of Eq. (E.12).

    :math:`K` (from `Kottmann et al. <https://arxiv.org/abs/2603.20376>`__) acts on the
    ``bond_additional`` (second) wire and the ``bond_previous`` register, with
    ``bond_additional`` as the most-significant qubit, controlled by the physical-bond wire:

    .. code-block::

            ───────■───────  (physical bond)
                 ┌─┴─┐
        |0⟩ ─────┤   ├─────  (additional virtual bond)
                 │ K │
        k-1 ──/──┤   ├─────  (previous bond register)
                 └───┘

    It factorizes into a flag ⚑ on the register, a multiplexed :math:`R_y` on the second wire,
    a second flag ⚑, and a trailing diagonal :math:`D`:

    .. code-block::

                                                ┌───┐
            ──────■─────────■───────────■───────┤   ├──  (physical bond)
                  │       ┌─┴──┐        │       │   │
        |0⟩ ──────┼───────┤ Ry ├────────■───────┤ D ├──  (additional virtual bond)
                ┌─┴─┐     └─┬──┘      ┌─┴─┐     │   │
        k-1 ─/──┤ ⚑ ├───────■─────────┤ ⚑ ├─────┤   ├──  (previous bond register)
                └───┘                 └───┘     └───┘

    Args:
        K (np.ndarray): the multiplexed unitary (``bond_additional`` as MSB).
        bond_additional (int): the additional virtual-bond (second) wire.
        bond_previous (Sequence[int]): the previous bond register (``k-1`` wires).

    Returns:
        tuple: ``(FL, FR0, FR1, DR, theta_csd)`` — the leading flag, the flags controlled on
        ``bond_additional`` = 0/1, the trailing diagonal, and the multiplexed :math:`R_y` angles.
    """

    def _flag_decompose(U, w):
        # Base case: a 1x1 unitary on an empty bond register is just a phase.
        if len(w) == 0:
            return [], qp.math.flatten(qp.math.asarray(U, dtype=complex))
        return recursive_generalized_flag_decomposition(U, w, _top=False)

    # Symmetric CSD, split on bond_additional
    R0, R1, theta_csd, L0, _, _, _, _ = synthesis_csd(K, shift=True)

    # Flag-decompose on bond register (base-case safe)
    FL, DL = _flag_decompose(L0, bond_previous)

    # Merge DL into R_i
    R0, R1 = R0 * DL, R1 * DL

    # Flag-decompose R_i
    FR0, DR0 = _flag_decompose(R0, bond_previous)
    FR1, DR1 = _flag_decompose(R1, bond_previous)
    FR0 = _map_nested_ops(FR0, lambda op: add_control(op, bond_additional, 0))
    FR1 = _map_nested_ops(FR1, lambda op: add_control(op, bond_additional, 1))
    DR = qp.math.concatenate((DR0, DR1))

    return FL, FR0, FR1, DR, theta_csd


def boundary_sequential_step(cells, aux_wires, phys_wires):
    r"""Sequentially synthesize the growing boundary tensors (Eqs. (E.13)-(E.15)).

    Boundary bond dimensions are powers of two (except the last), so the flags are fully
    multiplexed. Returns the merged circuit and the residual diagonal ``delta`` to carry
    into the bulk.
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
                K = K * qp.math.concatenate((delta, delta))

            FL, FR0, FR1, DR, theta_csd = flag_decompose_multiplexor(K, phys2, bond)
            RY = PartiallyMultiplexedFlag(
                qp.math.zeros_like(theta_csd), theta_csd, bond + [phys2], cvs_bond
            )

            # Attach the MSQ as control qubit
            ops = _map_nested_ops(
                flatten_ops([FL, RY, FR0, FR1]),
                lambda op, w=phys1, c=cv: add_control(op, w, c),
            )
            branch_ops.append(ops)
            branch_diags.append(DR)

        Gamma = merge_partially_multiplexed_flags_in_circuit(flatten_ops(branch_ops))
        Delta = qp.math.concatenate(branch_diags)  # Full diagonal

        # Split into RZ on phys1
        phi1, rem1, _ = split_diagonal_into_partially_multiplexed_rz(
            Delta, sub + [phys1], range(2 * chi)
        )
        delta = rem1[: 2 * chi]  # To be merged into next tensor

        circuit.append(
            PartiallyMultiplexedFlag(
                qp.math.zeros_like(cell["theta"]), cell["theta"], bond + [phys1], cvs_bond
            )
        )
        circuit += list(flatten_ops(Gamma))
        circuit.append(
            PartiallyMultiplexedFlag(
                qp.math.asarray(phi1, dtype=float), qp.math.zeros(len(phi1)), sub + [phys1], cvs_sub
            )
        )

    return merge_partially_multiplexed_flags_in_circuit(circuit), delta


def cascade_diagonal_into_rz_msq(full_diagonal, wires):
    r"""Cascade a full diagonal into a staircase of fully multiplexed :math:`R_z` gates.

    .. code-block::

          ┌─────┐
     q0 ──┤ Rzθ ├─────────────────────────────────
          └──┬──┘   ┌─────┐
     q1 ─────■──────┤ Rzθ ├───────────────────────
             │      └──┬──┘   ┌─────┐
     q2 ─────■─────────■──────┤ Rzθ ├─────────────
             │         │      └──┬──┘   ┌─────┐
     q3 ─────■─────────■─────────■──────┤ Rzφ ├───
                                        └───-─┘

    Returns the list of :math:`R_z` flags and the leftover ``global_phase``.
    """
    diag = qp.math.asarray(full_diagonal, dtype=complex)
    assert qp.math.shape(diag) == (2 ** len(wires),), "len(diagonal) must be 2**len(wires)"
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
                qp.math.asarray(angles, dtype=float),
                qp.math.zeros(len(angles)),
                controls + [msq],
                cvs,
            )
        )
        diag = remaining[: 2**num_ctrl]
        wires = controls
    global_phase = complex(diag[0])
    return flags, global_phase


def bulk_sequential_step(cells, delta, aux_wires, phys_wires):
    r"""Sequentially synthesize the bulk tensors, carrying the residual diagonal ``delta``.

    Bulk bond dimensions may be non-power-of-two, so the unitaries are fractally embedded at
    the interfaces. Returns the circuit ops and the residual diagonal for the right boundary.
    """
    if not cells:
        return [], delta

    n = len(aux_wires)
    N_bond = 2**n
    chi = len(qp.math.atleast_1d(cells[0]["theta"]))
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
            carry = qp.math.ones_like(delta)
        else:
            # Interior cells keep the active/inactive split and carry inactive phases
            delta_active = qp.math.gather(delta, active_bond)
            U0, U1 = U0 * delta_active, U1 * delta_active
            # Functional build of ``carry``: ``delta`` on the inactive bond, 1 elsewhere.
            inactive_mask = np.zeros(N_bond, dtype=bool)
            inactive_mask[inactive_bond] = True
            carry = qp.math.where(inactive_mask, delta, qp.math.ones_like(delta))

        # Flag-decompose the unitaries
        F0, D0 = recursive_generalized_flag_decomposition(U0, aux_wires, _top=False)
        F1, D1 = recursive_generalized_flag_decomposition(U1, aux_wires, _top=False)
        F0 = _map_nested_ops(F0, lambda op, w=phys: add_control(op, w, 0))
        F1 = _map_nested_ops(F1, lambda op, w=phys: add_control(op, w, 1))
        Gamma = merge_partially_multiplexed_flags_in_circuit(flatten_ops([F0, F1]))
        D0, D1 = D0 * carry, D1 * carry
        Delta = qp.math.concatenate((D0, D1))

        # Fully multiplex over N_bond for the last cell
        if is_last:
            ctrl, cv_use = full, cvs_full
        else:
            ctrl, cv_use = active_bond, cvs

        # Split diagonal into Rz and residual
        phi, residual_full, _ = split_diagonal_into_partially_multiplexed_rz(
            Delta, aux_wires + [phys], ctrl
        )
        phi = qp.math.asarray(phi, dtype=float)
        delta = residual_full[:N_bond]

        ops.append(
            PartiallyMultiplexedFlag(qp.math.zeros(chi), cell["theta"], aux_wires + [phys], cvs)
        )
        ops += list(flatten_ops(Gamma))
        ops.append(
            PartiallyMultiplexedFlag(phi, qp.math.zeros_like(phi), aux_wires + [phys], cv_use)
        )

    return ops, delta


@qp.QueuingManager.stop_recording()
def mps_synthesis(mps_tensors, wires):
    r"""Synthesize a right-canonical matrix product state into a list of operations.

    The builder underlying :func:`mps_preparation`: it returns the flag circuit and the
    residual global phase without queuing anything.

    Args:
        mps_tensors (Sequence[np.ndarray]): right-canonical MPS tensors, each of shape
            ``(chi_L, d, chi_R)`` (input bond, physical, output bond).
        wires (Sequence): the full register — ``ceil(log2(chi))`` auxiliary (bond) wires
            followed by the physical wires.

    Returns:
        tuple[list, complex]: ``(circuit, global_phase)`` — the operations (mostly
        :class:`PartiallyMultiplexedFlag`) and the residual scalar phase.

    Raises:
        ValueError: if ``mps_tensors`` is not right-canonical, or has an unsupported
            (non-unimodal) bond profile.
    """
    if not is_right_canonical(mps_tensors):
        raise ValueError("MPS tensors must be right-canonical")

    chi = max(A.shape[0] for A in mps_tensors)  # maximum bond dimension
    n_aux = int(qp.math.ceil_log2(chi))

    aux_wires = list(wires[:n_aux])
    phys_wires = list(wires[n_aux:])

    phys_canon = list(range(len(phys_wires)))
    aux_canon = list(range(len(phys_wires), len(phys_wires) + len(aux_wires)))
    wire_map = dict(zip(phys_canon + aux_canon, phys_wires + aux_wires))

    # Obtain the unitary completions
    left, bulk, right = split_mps(mps_tensors)
    left_completion = unitary_completion_mps(left)
    right_completion = unitary_completion_mps(right)
    bulk_completion = unitary_completion_mps(bulk)

    # Split the (canonical) physical wires
    n_left = len(left)
    phys_wires_left = phys_canon[:n_left]
    phys_wires_bulk = phys_canon[n_left:]

    # Perform the initial merges
    boundary_cells, bulk_cells = parallel_synthesis_steps(
        left_completion, bulk_completion, right_completion
    )

    # Synthesize the left boundary
    left_ops, delta = boundary_sequential_step(boundary_cells, aux_canon, phys_wires_left)
    # Synthesize the bulk
    bulk_ops, delta = bulk_sequential_step(bulk_cells, delta, aux_canon, phys_wires_bulk)
    # Synthesize the right boundary
    right_ops, global_phase = cascade_diagonal_into_rz_msq(delta, aux_canon)

    circuit = left_ops + bulk_ops + right_ops

    # Relabel from canonical wires back to the user-provided wires.
    circuit = [op.map_wires(wire_map) for op in circuit]

    return circuit, global_phase


def mps_preparation(mps_tensors, wires):
    r"""Prepare a right-canonical matrix product state on a quantum register.

    Queues the flag circuit mapping :math:`\lvert 0 \rangle` to the state with amplitudes given
    by ``mps_tensors``. The physical wires carry the state; the auxiliary (bond) wires are
    returned to :math:`\lvert 0 \rangle`. See :func:`mps_synthesis` for the underlying builder.

    Args:
        mps_tensors (Sequence[np.ndarray]): right-canonical MPS tensors, each of shape
            ``(chi_L, d, chi_R)`` (input bond, physical, output bond).
        wires (Sequence): the full register — ``ceil(log2(chi))`` auxiliary (bond) wires
            followed by the physical wires.

    Returns:
        complex: the global phase applied via :class:`~pennylane.GlobalPhase`.

    Raises:
        ValueError: if ``mps_tensors`` is not right-canonical or has an unsupported bond profile.

    **Example**

    Given right-canonical MPS tensors ``mps`` (each of shape ``(chi_L, d, chi_R)``) and a
    combined ``wires`` register holding the ``ceil(log2(chi))`` auxiliary (bond) wires
    followed by the physical wires:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(0)

        def rc(chi_l, chi_r, d=2):
            g = rng.standard_normal((d * chi_r, chi_l)) + 1j * rng.standard_normal((d * chi_r, chi_l))
            q, _ = np.linalg.qr(g)
            return q.conj().T.reshape(chi_l, d, chi_r)

        # Right-canonical MPS with bond dimension chi = 2 (three tensors).
        mps = [rc(1, 2), rc(2, 2), rc(2, 1)]

        # ceil(log2(2)) = 1 bond (auxiliary) wire (label 2), listed first, then the two
        # physical wires (labels 0, 1); the register length equals the number of tensors.
        wires = [2, 0, 1]

        dev = qp.device("default.qubit", wires=sorted(wires))

        @qp.qnode(dev)
        def circuit():
            mps_preparation(mps, wires)
            return qp.state()

    .. details::
        :title: Usage Details

        **Wire register.** ``wires`` is a single register whose first
        :math:`\lceil \log_2 \chi \rceil` entries are the bond (auxiliary) wires and whose
        remaining entries are the physical wires, where :math:`\chi` is the maximal bond
        dimension (inferred from ``mps_tensors``). Its total length equals the number of MPS
        tensors. The synthesis is invariant to the choice of wire labels and their order:
        operators are built on canonical labels internally and remapped onto ``wires`` only at
        the end, so any labelling prepares the same state up to the corresponding relabelling.

    """
    # Build the circuit with queuing suspended: mps_synthesis instantiates many
    # intermediate operators that would otherwise auto-queue onto the tape.
    with qp.QueuingManager.stop_recording():
        circuit, global_phase = mps_synthesis(mps_tensors, wires)
    for op in circuit:
        qp.apply(op)
    qp.GlobalPhase(-qp.math.angle(global_phase))
    return global_phase
