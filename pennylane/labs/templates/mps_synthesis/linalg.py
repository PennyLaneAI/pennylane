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
"""Linear-algebra and circuit-transformation helpers for MPS preparation.

Bundles small array/control-bit utilities, fractal embeddings for
non-power-of-two dimensions, the Cosine-Sine Decomposition, and
diagonal-splitting circuit transformations.
"""

import numpy as np
from scipy.linalg import cossin

import pennylane as qp

# ============================================================================
# Array and control-bit utilities
# ============================================================================


def get_nwires(d):
    """Return the number of wires used to embed a ``d``-dimensional space."""
    nwires = int(np.ceil(np.log2(d)))
    if d == 2:
        nwires += 1
    return nwires


def ints_to_control_bits(control_states, num_control_bits):
    r"""Convert each integer control state into a ``num_control_bits``-long
    list of bits (MSB first, zero-padded)."""
    return [[int(bit) for bit in format(c, f"0{num_control_bits}b")] for c in control_states]


def control_bits_to_ints(control_bits):
    """Convert a list of bits (MSB first) to an integer."""
    value = 0
    for b in control_bits:
        value = (value << 1) | b
    return value


# ============================================================================
# Fractal embedding
# ============================================================================


def split_d(d):
    """
    Splits the dimension d into two parts p=floor(d/2) and q=ceil(d/2) such that p + q = d.
    """
    p = d // 2
    q = d - p
    return p, q


def get_fractal_sequence(N):
    r"""Return the fractal ordering of a ``\log_2(N)``-qubit register.
    The register is recursively halved and the two child orderings are interleaved
    at each level, yielding a permutation of ``range(N)``. The active ``d`` states are
    the length-``d`` suffix ``fractal_sequence[N - d:]``.
    Args:
        N (int): size of the register; must be a power of 2.
    Returns:
        list[int]: a permutation of ``range(N)`` in fractal order.
    Raises:
        ValueError: if ``N`` is not a positive power of 2.
    """
    if N < 1 or (N & (N - 1)) != 0:
        raise ValueError("N must be a power of 2.")

    def build(n_bits):
        if n_bits <= 1:
            return list(range(2**n_bits))
        half = 2 ** (n_bits - 1)
        left = build(n_bits - 1)
        right = [index + half for index in build(n_bits - 1)]
        if n_bits == 2:
            return left + right
        out = []
        for left_i, right_i in zip(left, right):
            out.extend((left_i, right_i))
        return out

    return build(int(np.log2(N)))


def get_fractal_embedding_states(d, N):
    r"""Return the active and inactive basis indices for a fractal embedding.
    The fractal ordering of the ``N`` basis states is split so that the last ``d``
    states are active and the remaining ``N - d`` are inactive.
    Args:
        d (int): number of active states.
        N (int): Hilbert-space dimension; must be a power of 2.
    Returns:
        tuple[list[int], list[int]]: the sorted active and inactive basis indices.
    """
    master_sequence = get_fractal_sequence(N)
    inactive_states = master_sequence[: N - d]
    active_states = master_sequence[N - d :]
    return sorted(active_states), sorted(inactive_states)


def embed_unitary(U, N: int, active_indices: list):
    r"""Embed the submatrix ``U`` into an ``N x N`` identity on the active subspace.
    ``U`` acts on the basis states given by ``active_indices``; all remaining
    (inactive) basis states are left unchanged (identity).
    Args:
        U (tensor_like): the ``len(active_indices) x len(active_indices)`` matrix to embed.
        N (int): dimension of the full register.
        active_indices (list[int]): basis indices that ``U`` acts on.
    Returns:
        tensor_like: the ``N x N`` matrix acting as ``U`` on the active subspace
        and as the identity elsewhere.
    """
    U = qp.math.asarray(U)
    active_indices = list(active_indices)
    # 0/1 selection matrix (structural) mapping the active subspace into the register.
    selection = np.eye(N, dtype=complex)[:, active_indices]
    inactive_identity = np.eye(N, dtype=complex) - selection @ selection.T
    return selection @ U @ selection.T + inactive_identity


def find_mismatched_block_start(block_dims):
    r"""Find the starting state index of the first mismatched block in the second half.
    The blocks are split into two equal halves and compared pairwise (block ``i`` in
    the first half against block ``i + num_blocks // 2`` in the second). The returned
    index is the cumulative state offset, within the second half, of the first pair
    whose dimensions differ.
    Args:
        block_dims (Sequence[int]): dimension (number of states) of each block.
    Returns:
        int or None: the second-half state index where the first size mismatch occurs,
        or ``None`` if the two halves are perfectly symmetric.
    """
    num_blocks = len(block_dims)
    half_blocks = num_blocks // 2

    # The second half starts exactly after the total sum of all states in the first half
    states_in_first_half = sum(block_dims[:half_blocks])
    current_state_idx_second = states_in_first_half

    for i in range(half_blocks):
        size_first_half = block_dims[i]
        size_second_half = block_dims[i + half_blocks]

        # If the block dimensions don't match, we return the current second-half state index
        if size_first_half != size_second_half:
            return current_state_idx_second

        # Add the size of the current block in the second half to shift the index forward
        current_state_idx_second += size_second_half

    return None  # Returns None if all blocks are perfectly symmetric


def get_active_block_sizes(d, N):
    r"""Compute the sizes of the active blocks in a fractal embedding.
    The ``N``-dimensional register is recursively halved, splitting the ``d`` active
    states with a floor-first partition (``p = floor``, ``q = ceil``) until each chunk
    has dimension ``<= 4``. Fully inactive chunks (``curr_d == 0``) are omitted.
    Args:
        d (int): number of active states; must satisfy ``0 <= d <= N``.
        N (int): dimension of the register; must be a power of 2.
    Returns:
        list[int]: the active-state count of each non-empty block, left to right.
    Raises:
        ValueError: if ``N`` is not a power of 2, or ``d`` is outside ``[0, N]``.
    """
    if N < 1 or (N & (N - 1)) != 0:
        raise ValueError("N must be a power of 2.")
    if d < 0 or d > N:
        raise ValueError("d must be between 0 and N.")

    def _get_sizes(curr_d, curr_N):
        # Base case: We've reached the embedding block
        if curr_N <= 4:
            # curr_d is the exact number of active states in this chunk.
            # If curr_d is 0, it's an entirely inactive block, so we omit it
            return [curr_d] if curr_d > 0 else []

        # Recursive split (p = floor, q = ceil)
        p, q = split_d(curr_d)
        half_N = curr_N // 2

        # Traverse left and right branches
        left_sizes = _get_sizes(p, half_N)
        right_sizes = _get_sizes(q, half_N)

        return left_sizes + right_sizes

    return _get_sizes(d, N)


# ============================================================================
# Cosine-Sine Decomposition
# ============================================================================


def shift_csd_one(U, CS, V_H, target_index):
    r"""Shift the uncoupled ``1`` of a Cosine-Sine Decomposition to a target index.
    Given a CSD ``U @ CS @ V_H`` with total dimension ``d = 2p + 1`` (so the lone
    uncoupled ``1`` sits at position ``p`` in the ``CS`` matrix), this permutes the
    columns/rows within the second block ``[p, d)`` so the ``1`` lands at
    ``target_index``, returning the updated factors.
    Args:
        U (tensor_like): left factor of the CSD.
        CS (tensor_like): the ``d x d`` cosine-sine (diagonal-block) factor.
        V_H (tensor_like): right factor of the CSD.
        target_index (int): destination index for the uncoupled ``1``; must lie in
            ``[p, d)``.
    Returns:
        tuple[tensor_like, tensor_like, tensor_like]: the permuted ``(U, CS, V_H)``.
    Raises:
        ValueError: if ``target_index`` is outside its block partition ``[p, d - 1]``.
    """
    d = CS.shape[0]
    p = d // 2  # Integer division to find p

    # Validate the splitting of blocks
    assert np.isclose(CS[p, p], 1.0), "Splitting of blocks is incorrect"
    block_start = p
    block_end = d

    # Validate the target index
    if not block_start <= target_index < block_end:
        raise ValueError(
            f"Target index {target_index} is invalid. "
            f"It must remain within its block partition: [{block_start}, {block_end - 1}]."
        )

    # Calculate the relative position within the sub-block
    rel_target = target_index - block_start
    block_size = block_end - block_start

    # Create the custom column order for the sub-block
    custom_order = list(range(1, block_size))
    custom_order.insert(rel_target, 0)

    # Build the permutation matrices
    P_sub = np.eye(block_size)[:, custom_order]
    P = np.eye(d)
    P[block_start:block_end, block_start:block_end] = P_sub

    # Apply the permutations
    U_custom = U @ P
    CS_custom = P.T @ CS @ P
    V_H_custom = P.T @ V_H

    return U_custom, CS_custom, V_H_custom


def synthesis_csd(V, shift=False, return_all=False):
    r"""Cosine-Sine Decomposition of a unitary, specialized for fractal-embedded synthesis.
    Splits a :math:`d \times d` unitary (via :func:`scipy.linalg.cossin`) into blocks
    ``K00, K01`` / ``K10, K11`` and a multiplexed :math:`R_Y` angle array ``theta``.
    When ``shift=True`` and ``d`` is odd, the uncoupled ``1`` of the cosine-sine block is
    relocated to match the fractal-embedding block structure.
    Args:
        V (np.ndarray): a :math:`d \times d` unitary matrix.
        shift (bool): align the odd-``d`` uncoupled ``1`` with the embedding blocks.
        return_all (bool): also return the raw CSD factors ``U, CS, V_H``.
    Returns:
        tuple: ``(K00, K01, theta, K10, K11, U, CS, V_H)``; the last three are ``None``
        unless ``return_all=True``.
    """
    d = V.shape[0]
    p, q = split_d(d)
    N = 2 ** get_nwires(d)  # Hilbert-space dimension (power of 2)
    U, CS, V_H = None, None, None

    shift_idx = None
    if p != q and shift:
        shift_idx = find_mismatched_block_start(get_active_block_sizes(d, N))
    if shift_idx is not None:
        U_init, CS_init, V_H_init = cossin(V, p=p, q=p, separate=False)
        _, theta, _ = cossin(V, p, p, separate=True)
        U, CS, V_H = shift_csd_one(U_init, CS_init, V_H_init, shift_idx)
        K00 = U[:p, :p]
        K01 = U[p:, p:]
        K10 = V_H[:p, :p]
        K11 = V_H[p:, p:]
    else:
        (K00, K01), theta, (K10, K11) = cossin(V, p=p, q=p, separate=True)
    if not return_all:
        U, CS, V_H = None, None, None
    elif shift_idx is None:
        U, CS, V_H = cossin(V, p=p, q=p, separate=False)
    # RY(alpha) is defined with half-angles while SciPy returns full angles
    theta *= 2.0
    return K00, K01, theta, K10, K11, U, CS, V_H


# ============================================================================
# Diagonal and controlled-unitary transformations
# ============================================================================


def split_diagonal_into_partially_multiplexed_rz(full_diagonal, wires, control_states):
    r"""Factor a diagonal into a partially multiplexed :math:`R_z` and a remainder.
    ``full_diagonal`` is MSB-first on ``sorted(wires)``; ``wires[-1]`` is the :math:`R_z`
    target and ``wires[:-1]`` the controls, restricted to ``control_states``. Satisfies
    ``full_diagonal = rz * remaining`` element-wise.
    Args:
        full_diagonal (tensor_like): the diagonal to factor.
        wires (Sequence): control wires followed by the :math:`R_z` target wire.
        control_states (Sequence[int]): control basis indices to act on.
    Returns:
        tuple: ``(angles, remaining, rz)`` — the :math:`R_z` angles, the residual
        diagonal, and the extracted :math:`R_z` diagonal.
    """
    wires = list(wires)
    target = wires[-1]

    full_diagonal = qp.math.asarray(full_diagonal, dtype=complex)
    wire_order = sorted(wires)
    n = len(wire_order)

    # Find pair of full-register diagonal indices that the target qubit's Rz acts on.
    bit_pos = n - 1 - wire_order.index(target)
    c = np.asarray(control_states)
    idx_0 = ((c >> bit_pos) << (bit_pos + 1)) | (c & ((1 << bit_pos) - 1))
    idx_1 = idx_0 | (1 << bit_pos)

    # Rz angle = phase difference between the |0>/|1> pair;
    # residual = shared phase left behind.
    val_0, val_1 = full_diagonal[idx_0], full_diagonal[idx_1]
    angles = qp.math.angle(val_1 / val_0)
    half = qp.math.exp(1j * angles / 2)
    residual = val_0 * half

    # Strip the Rz phase from both paired entries, leaving the shared residual.
    # Functional scatter: "set to residual" == add the delta from the current value.
    remaining = qp.math.scatter_element_add(full_diagonal, [idx_0], residual - val_0)
    remaining = qp.math.scatter_element_add(remaining, [idx_1], residual - val_1)

    # Reconstruct the extracted Rz as its own diagonal: e^{-i angle/2} on |0>
    # and e^{+i angle/2} on |1>, identity everywhere else.
    rz = qp.math.ones_like(full_diagonal)
    rz = qp.math.scatter_element_add(rz, [idx_0], qp.math.conj(half) - 1)
    rz = qp.math.scatter_element_add(rz, [idx_1], half - 1)

    return angles, remaining, rz


def split_diagonal_into_control_branches(diag, wires):
    r"""Split a diagonal into its :math:`\vert 0\rangle`- and :math:`\vert 1\rangle`-controlled branches.
    ``wires[0]`` is the control qubit (indexed MSB-first on ``sorted(wires)``); the rest
    are targets.
    Args:
        diag (tensor_like): a length-``2^n`` diagonal.
        wires (Sequence): control wire followed by target wires.
    Returns:
        tuple: ``(d0, d1, target_d0, target_d1)`` — the full diagonals controlled on
        :math:`\vert 0\rangle` / :math:`\vert 1\rangle` (identity off-branch) and the
        corresponding target-only diagonals.
    """
    diag = qp.math.asarray(diag)
    wires = list(wires)
    n_states = qp.math.shape(diag)[0]
    n_qubits = int(round(float(np.log2(n_states))))

    all_wires = sorted(wires)
    control_wire = wires[0]
    ctrl_bit_pos = n_qubits - 1 - all_wires.index(control_wire)

    # Divide basis states into |0>- and |1>-controlled groups (structural index masks).
    indices = np.arange(n_states)
    is_zero = ((indices >> ctrl_bit_pos) & 1) == 0

    # Build the |0>- and |1>-controlled diagonals functionally (identity off-branch).
    ones = qp.math.ones_like(diag)
    d0 = qp.math.where(is_zero, diag, ones)
    d1 = qp.math.where(~is_zero, diag, ones)

    return d0, d1, diag[indices[is_zero]], diag[indices[~is_zero]]


def get_controlled_unitary_msq(U, wires, control_value, active_indices=None):
    r"""Lift a target operator ``U`` to a full-register operator controlled on ``wires[0]``.
    Indexing is MSB-first on ``sorted(wires)``. If ``U`` is smaller than the target
    dimension, ``active_indices`` says which target states it acts on (identity elsewhere).
    A purely diagonal 2D ``U`` is handled via the cheaper diagonal path.
    Args:
        U (tensor_like): target operator on ``wires[1:]`` — a length-``2**(len(wires)-1)``
            diagonal (or active subset) or a 2D unitary.
        wires (Sequence): ``wires[0]`` is the control; ``wires[1:]`` the targets.
        control_value (int): control bit (0 or 1) that triggers ``U``.
        active_indices (Sequence[int], optional): target indices ``U`` acts on when it is
            smaller than the full target dimension.
    Returns:
        tensor_like: a length-``2**len(wires)`` diagonal if ``U`` is diagonal, else a
        ``2**len(wires)`` square matrix.
    Raises:
        ValueError: if ``U`` is smaller than the target dimension and ``active_indices`` is ``None``.
    """
    wires = list(wires)
    U = qp.math.asarray(U)
    # Treat a 2D U with no off-diagonal entries as a diagonal (cheaper).
    is_diagonal = len(qp.math.shape(U)) == 1
    if len(qp.math.shape(U)) == 2:
        off_diag = U - qp.math.diag(qp.math.diag(U))
        if qp.math.allclose(off_diag, 0):
            U = qp.math.diag(U)
            is_diagonal = True

    control_wire = wires[0]
    wire_order = sorted(wires)
    n_qubits = len(wire_order)
    full_dim = 2**n_qubits
    sorted_target = [w for w in wire_order if w != control_wire]
    n_target = len(sorted_target)
    target_dim = 2**n_target
    op_size = qp.math.shape(U)[0]

    # Expand U to the full target subspace: use it as-is if already full size,
    # otherwise pad with identity on the inactive target indices.
    if op_size == target_dim:
        U_target = qp.math.cast(U, complex)
    elif active_indices is None:
        raise ValueError(
            "active_indices must be provided when U is smaller than the target dimension"
        )
    elif is_diagonal:
        active_indices = np.asarray(active_indices)
        # Identity diagonal with U scattered onto the active indices.
        base = qp.math.convert_like(np.ones(target_dim, dtype=complex), U)
        U_target = qp.math.scatter_element_add(base, [active_indices], U - 1)
    else:
        active_indices = np.asarray(active_indices)
        U_target = embed_unitary(U, target_dim, active_indices)

    # Select the full-register rows where the control qubit equals control_value.
    idx = np.arange(full_dim)
    ctrl_bit = n_qubits - 1 - wire_order.index(control_wire)
    mask = ((idx >> ctrl_bit) & 1) == control_value

    # Map each full-register index to its target-subspace index by gathering the
    # target wires' bits (MSB-first over sorted_target).
    target_sub = np.zeros(full_dim, dtype=int)
    for j, w in enumerate(sorted_target):
        bit = n_qubits - 1 - wire_order.index(w)
        target_sub |= ((idx >> bit) & 1) << (n_target - 1 - j)

    # Diagonal case: gather U's diagonal onto the controlled rows, 1 elsewhere.
    if is_diagonal:
        gathered = qp.math.gather(U_target, target_sub)
        return qp.math.where(mask, gathered, qp.math.ones_like(gathered))

    # Dense case: place U into the controlled block, identity on the rest, using a 0/1
    # selection matrix ``S`` so ``S @ U_target @ S.T`` scatters the block without mutation.
    slice_idx = np.where(mask)[0]
    sub = target_sub[slice_idx]
    selection = np.zeros((full_dim, target_dim), dtype=complex)
    selection[slice_idx, sub] = 1.0
    identity_rest = np.eye(full_dim, dtype=complex) - selection @ selection.T
    return selection @ U_target @ selection.T + identity_rest


def propagate_diagonal_through_unitary(full_diag, U, wires, control_val, active_indices):
    r"""Absorb a diagonal into a unitary ``U`` on the controlled subspace.

    On the rows where ``wires[0]`` equals ``control_val``, the target diagonal carried by
    ``full_diag`` is multiplied onto ``U``'s active columns and stripped from the diagonal.
    All indexing is MSB-first on ``sorted(wires)``.

    Args:
        full_diag (tensor_like): length-``2**len(wires)`` diagonal.
        U (tensor_like): unitary on the active target subspace.
        wires (Sequence): all wires; ``wires[0]`` is the control.
        control_val (int): control value ``U`` acts on.
        active_indices (Sequence[int]): target indices ``U`` occupies.

    Returns:
        tuple: ``(new_U, new_full_diag, controlled_new_U)`` — ``U`` with the diagonal folded
        in, the diagonal with the absorbed part removed, and ``new_U`` lifted to the full register.
    """
    active_indices = np.asarray(active_indices)
    wires = list(wires)
    n_qubits = len(wires)
    n_control = 1
    n_target = n_qubits - n_control
    full_dim = 2**n_qubits
    target_dim = 2**n_target

    U = qp.math.asarray(U)
    full_diag = qp.math.asarray(full_diag, dtype=complex)

    # Control mask + full-register -> target-subspace index map (MSB-first).
    idx = np.arange(full_dim)
    mask = np.ones(full_dim, dtype=bool)
    target_sub = np.zeros(full_dim, dtype=int)
    for i in range(n_control):
        bit = n_qubits - 1 - i
        mask &= ((idx >> bit) & 1) == ((control_val >> (n_control - 1 - i)) & 1)
    for j in range(n_target):
        bit = n_qubits - 1 - (n_control + j)
        target_sub |= ((idx >> bit) & 1) << (n_target - 1 - j)

    # Collapse the controlled rows into a target-only diagonal (bijective, so just gather).
    controlled_rows = np.where(mask)[0]
    order = controlled_rows[np.argsort(target_sub[controlled_rows])]
    target_diag = qp.math.gather(full_diag, order)

    # Fold the active diagonal into U; inactive entries stay as the remaining diagonal.
    new_U = U * qp.math.gather(target_diag, active_indices)
    active_mask = np.zeros(target_dim, dtype=bool)
    active_mask[active_indices] = True
    remaining_target = qp.math.where(active_mask, qp.math.ones_like(target_diag), target_diag)

    # Scatter the leftover diagonal back onto the controlled rows.
    gathered = qp.math.gather(remaining_target, target_sub)
    new_full_diag = qp.math.where(mask, gathered, full_diag)

    # Lift new_U to the full register (U on the controlled block, identity elsewhere).
    U_target = embed_unitary(new_U, target_dim, active_indices)
    sub = target_sub[controlled_rows]
    selection = np.zeros((full_dim, target_dim), dtype=complex)
    selection[controlled_rows, sub] = 1.0
    identity_rest = np.eye(full_dim, dtype=complex) - selection @ selection.T
    controlled_new_U = selection @ U_target @ selection.T + identity_rest

    return new_U, new_full_diag, controlled_new_U
