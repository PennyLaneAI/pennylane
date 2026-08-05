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
"""Linear-algebra and circuit-transformation helpers for MPS preparation.

Bundles small array/control-bit utilities, fractal embeddings for
non-power-of-two dimensions, the Cosine-Sine Decomposition, and
diagonal-splitting circuit transformations.
"""

import numpy as np
from scipy.linalg import cossin
from scipy.stats import unitary_group

# ============================================================================
# Array and control-bit utilities
# ============================================================================


def get_nwires(d):
    """Return the number of wires used to embed a ``d``-dimensional space."""
    nwires = int(np.ceil(np.log2(d)))
    if d == 2:
        nwires += 1
    return nwires


def create_unitary(d):
    """Return a random ``d x d`` unitary and the wires used to embed it."""
    nwires = get_nwires(d)
    wires = list(range(nwires))
    U_d = unitary_group.rvs(d)
    return U_d, wires


def create_unitary_diagonal(d):
    """Return a length-``d`` diagonal of random complex phases."""
    random_phases = np.random.uniform(0, 2 * np.pi, d)
    full_diag = np.exp(1j * random_phases)
    return full_diag


def ints_to_control_bits(control_states, num_control_bits):
    r"""
    Converts a list of integer control states into a list of lists
    containing individual bit values (MSB first).
    """
    return [[int(bit) for bit in format(c, f"0{num_control_bits}b")] for c in control_states]


def control_bits_to_ints(control_bits):
    """Convert a list of bits (MSB first) to an integer."""
    value = 0
    for b in control_bits:
        value = (value << 1) | b
    return value


def count_nontrivial_diagonal(diagonal, atol=1e-8):
    """
    Count the number of entries in a diagonal that are not equal to 1.

    Useful for measuring how much of a diagonal remainder is non-trivial
    (entries equal to 1 act as identity and carry no phase).
    """
    diagonal = np.asarray(diagonal)
    return int(np.count_nonzero(~np.isclose(diagonal, 1.0, atol=atol)))


def extract_active_submatrix(matrix, active_indices):
    """
    Extracts the active submatrix from a larger embedded matrix by removing
    inactive basis states (states that only apply the identity operation).
    """
    matrix = np.asarray(matrix)
    return matrix[np.ix_(active_indices, active_indices)]


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


def get_master_sequence(N):
    r"""
    Return the fractal master sequence for a ``\log_2(N)``-qubit register by
    recursively halving the register and interleaving the two child orderings at each level.

    The sequence is a permutation of ``range(N)``. Active ``d`` states are the
    length-``d`` suffix ``master_sequence[N - d:]``.
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
    r"""
    Active and inactive basis indices for fractal embedding.

    Args:
        d: Number of active states.
        N: Hilbert-space dimension (power of 2).
    """
    master_sequence = get_master_sequence(N)
    inactive_states = master_sequence[: N - d]
    active_states = master_sequence[N - d :]
    return sorted(active_states), sorted(inactive_states)


def embed_unitary(U: np.ndarray, N: int, active_indices: list):
    """Embed the active submatrix ``U`` into an ``N x N`` identity."""
    embedded = np.eye(N, dtype=complex)
    ix = np.ix_(active_indices, active_indices)
    embedded[ix] = U
    return embedded


def print_embedding_matrix(d, N):
    """
    Prints an NxN diagonal-like matrix where inactive states are '1'
    and active states form dense blocks of '*' (2x2 or 3x3) along the diagonal.
    """
    active_states, inactive_states = get_fractal_embedding_states(d, N)

    # Group the active states into contiguous blocks
    active_states.sort()
    active_blocks = []

    if active_states:
        current_block = [active_states[0]]
        for i in range(1, len(active_states)):
            if active_states[i] == current_block[-1] + 1:
                current_block.append(active_states[i])
            else:
                active_blocks.append(current_block)
                current_block = [active_states[i]]
        active_blocks.append(current_block)

    # Initialize the N x N grid with '.' for empty zeroes
    grid = [["." for _ in range(N)] for _ in range(N)]

    # Populate the inactive states
    for i in inactive_states:
        grid[i][i] = "1"

    # Populate the active states
    for block in active_blocks:
        for r in block:
            for c in block:
                grid[r][c] = "*"

    # Print the matrix
    print(f"Fractal embedding matrix (N={N}, d={d})")
    print("-" * (N * 2 - 1))
    for row in grid:
        print(" ".join(row))


def find_mismatched_block_start(block_dims):
    """
    Takes a list of block dimensions and returns the first active state index
    of the mismatched block in the second half.
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

        # Add the size of the current block in the SECOND half to shift the index forward
        current_state_idx_second += size_second_half

    return None  # Returns None if all blocks are perfectly symmetric


def get_active_block_sizes(d, N):
    """
    Recursively calculates the dimensions of the active blocks
    using a floor_first partition (p=floor, q=ceil).
    """
    if N < 1 or (N & (N - 1)) != 0:
        raise ValueError("N must be a power of 2.")
    if d < 0 or d > N:
        raise ValueError("d must be between 0 and N.")

    def _get_sizes(curr_d, curr_N):
        # Base case: We've reached the embedding block
        if curr_N <= 4:
            # curr_d is the exact number of active states in this chunk!
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
    """
    Shifts the uncoupled '1' in a Cosine-Sine Decomposition to a target index.
    Assumes a total dimension d = 2p + 1.
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


def csd(V, shift=False, return_all=False):
    r"""
    Perform the Cosine-Sine Decomposition (CSD) of a unitary matrix :math:`V`.

    Decomposes a :math:`d \times d` unitary into two :math:`p \times p` and two :math:`q \times q`
    (with :math:`p + q = d` and :math:`q-p\in\{0,1\}`) unitary blocks and a partially multiplexed
    :math:`R_Y` rotation as used in the Quantum Shannon Decomposition.
    Uses the standard implementation in SciPy.

    Args:
        V (np.ndarray): A :math:`d \times d` complex unitary matrix.
        shift (bool): Whether to shift the uncoupled '1' in the CSD.
        return_all (bool): Whether to return U, CS, V_H.
    Returns:
        tuple[np.ndarray]: (K00, K01, theta, K10, K11, U, CS, V_H) where:
            - K00, K01 (np.ndarray): :math:`p \times p` and :math:`q \times q` blocks of the left factor.
            - theta (np.ndarray): Array of :math:`p` angles for a partially multiplexed :math:`R_Y`.
            - K10, K11 (np.ndarray): :math:`p \times p` and :math:`q \times q` blocks of the right factor.
            - U (np.ndarray): The unitary matrix :math:`U` of the CSD.
            - CS (np.ndarray): The cosine-sine matrix :math:`CS` of the CSD.
            - V_H (np.ndarray): The hermitian conjugate of the unitary matrix :math:`V_H` of the CSD.
    """
    d = V.shape[0]
    p, q = split_d(d)
    N = 2 ** get_nwires(d)  # Hilbert-space dimension (power of 2)

    U, CS, V_H = None, None, None

    if p == q:
        (K00, K01), theta, (K10, K11) = cossin(V, p=p, q=p, separate=True)
    elif shift:
        U_init, CS_init, V_H_init = cossin(V, p=p, q=p, separate=False)
        _, theta, _ = cossin(V, p, p, separate=True)
        shift_idx = find_mismatched_block_start(get_active_block_sizes(d, N))
        U, CS, V_H = shift_csd_one(U_init, CS_init, V_H_init, shift_idx)
        K00 = U[:p, :p]
        K01 = U[p:, p:]
        K10 = V_H[:p, :p]
        K11 = V_H[p:, p:]
    else:
        (K00, K01), theta, (K10, K11) = cossin(V, p=p, q=p, separate=True)

    if not return_all:
        U, CS, V_H = None, None, None
    if return_all and not shift:
        U, CS, V_H = cossin(V, p=p, q=p, separate=False)

    # RY(alpha) is defined with half-angles while SciPy returns full angles
    theta *= 2.0

    return K00, K01, theta, K10, K11, U, CS, V_H


# ============================================================================
# Diagonal and controlled-unitary transformations
# ============================================================================


def split_diagonal_into_partially_multiplexed_rz(full_diagonal, wires, control_states):
    """
    Split a diagonal into partially multiplexed Rz angles and a remainder.

    ``full_diagonal`` is indexed in MSB-first order on
    ``sorted(wires)``. ``wires[-1]`` is the Rz target; ``wires[:-1]`` are
    controls (any order). ``control_states`` are basis indices in that ordering.

    ``full_diagonal = rz_diagonal * remaining_diagonal`` (element-wise).
    """
    wires = list(wires)
    target = wires[-1]

    full_diagonal = np.asarray(full_diagonal, dtype=complex)
    wire_order = sorted(wires)
    n = len(wire_order)
    expected = 2**n

    # Find pair of full-register diagonal indices that the target qubit's Rz acts on.
    bit_pos = n - 1 - wire_order.index(target)
    c = np.asarray(control_states)
    idx_0 = ((c >> bit_pos) << (bit_pos + 1)) | (c & ((1 << bit_pos) - 1))
    idx_1 = idx_0 | (1 << bit_pos)

    # Rz angle = phase difference between the |0>/|1> pair;
    # residual = shared phase left behind.
    val_0, val_1 = full_diagonal[idx_0], full_diagonal[idx_1]
    angles = np.angle(val_1 / val_0)
    residual = val_0 * np.exp(1j * angles / 2)

    # Strip the Rz phase from both paired entries, leaving the shared residual.
    remaining = full_diagonal.copy()
    remaining[idx_0] = remaining[idx_1] = residual

    # Reconstruct the extracted Rz as its own diagonal: e^{-i angle/2} on |0>
    # and e^{+i angle/2} on |1>, identity everywhere else.
    half = np.exp(1j * angles / 2)
    rz = np.ones(expected, dtype=complex)
    rz[idx_0] = np.conj(half)
    rz[idx_1] = half

    return angles.tolist(), remaining, rz


def split_diagonal_into_control_branches(diag, wires):
    """
    Splits a diagonal into |0>- and |1>-controlled diagonals.
    Returns the |0>-controlled diagonal, the |1>-controlled diagonal,
    and the corresponding target-only diagonals.

    ``wires[0]`` is the control qubit; remaining entries are target wires.

    Args:
        diag (np.ndarray): A 1D array of length 2^n representing the diagonal.
        wires (list or array-like): The wires the operation acts on.

    Returns:
        tuple:
            - d0 / d1 (np.ndarray): Full-sized diagonal operation controlled by |0> / |1>.
            - target_d0 / target_d1 (np.ndarray): Target diagonal when control is |0> / |1>.
    """
    diag = np.asarray(diag)
    wires = list(wires)
    n_states = len(diag)
    n_qubits = int(np.round(np.log2(n_states)))

    all_wires = sorted(wires)
    control_wire = wires[0]
    ctrl_bit_pos = n_qubits - 1 - all_wires.index(control_wire)

    # Divide basis states into |0>- and |1>-controlled groups.
    indices = np.arange(n_states)
    ctrl_bit = (indices >> ctrl_bit_pos) & 1
    idx0 = indices[ctrl_bit == 0]
    idx1 = indices[ctrl_bit == 1]

    # Get the |0>- and |1>-controlled diagonals.
    d0 = np.ones(n_states, dtype=diag.dtype)
    d1 = np.ones(n_states, dtype=diag.dtype)
    d0[idx0] = diag[idx0]
    d1[idx1] = diag[idx1]

    return d0, d1, diag[idx0], diag[idx1]


def get_controlled_unitary_msq(U, wires, control_value, active_indices=None):
    """
    Turns a target operator ``U`` into a full-register operator controlled on
    ``wires[0]`` by the control value ``control_value``.

    All indexing is MSB-first on
    ``sorted(wires)``. A 2D ``U`` that is purely diagonal is treated as diagonal.

    Args:
        U (array-like): Target operator on ``wires[1:]``. A 1D diagonal of length
            ``2**(len(wires)-1)`` (or its active subset), or a 2D unitary.
        wires (list): ``wires[0]`` is the control; ``wires[1:]`` are targets.
        control_value (int): Control bit (0 or 1) the operation triggers on.
        active_indices (array-like, optional): Target-subspace indices ``U`` acts
            on when ``U`` is smaller than the full target dimension; the rest is
            padded with identity.

    Returns:
        np.ndarray: Length ``2**len(wires)`` diagonal vector if ``U`` is diagonal,
        otherwise a ``2**len(wires)`` square matrix.
    """
    wires = list(wires)
    U = np.asarray(U)
    # Treat a 2D U with no off-diagonal entries as a diagonal (cheaper path).
    is_diagonal = U.ndim == 1
    if U.ndim == 2:
        off_diag = U - np.diag(np.diag(U))
        if np.allclose(off_diag, 0):
            U = np.diag(U)
            is_diagonal = True

    control_wire = wires[0]
    wire_order = sorted(wires)
    n_qubits = len(wire_order)
    full_dim = 2**n_qubits
    sorted_target = [w for w in wire_order if w != control_wire]
    n_target = len(sorted_target)
    target_dim = 2**n_target
    op_size = U.size if is_diagonal else U.shape[0]

    # Expand U to the full target subspace: use it as-is if already full size,
    # otherwise pad with identity on the inactive target indices.
    if op_size == target_dim:
        U_target = np.asarray(U, dtype=complex)
    elif active_indices is not None:
        active_indices = np.asarray(active_indices)
        if is_diagonal:
            U_target = np.ones(target_dim, dtype=complex)
            U_target[active_indices] = U
        else:
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

    # Diagonal case: scatter U's diagonal onto the controlled rows, 1 elsewhere.
    if is_diagonal:
        controlled = np.ones(full_dim, dtype=complex)
        controlled[mask] = U_target[target_sub[mask]]
        return controlled

    # Dense case: place U into the controlled block, identity on the rest.
    controlled = np.eye(full_dim, dtype=complex)
    slice_idx = np.where(mask)[0]
    sub = target_sub[slice_idx]
    controlled[np.ix_(slice_idx, slice_idx)] = U_target[np.ix_(sub, sub)]
    return controlled


def propagate_diagonal_through_unitary(full_diag, U, wires, control_val, active_indices):
    """
    Propagates a diagonal through a ``d x d`` unitary ``U``.

    On the subspace where the leading control wire equals ``control_val``, the
    target diagonal carried by ``full_diag`` is merged into ``U`` (multiplied
    onto its columns at ``active_indices``) and stripped from the diagonal.

    Args:
        full_diag (array-like): Length-``2**len(wires)`` diagonal, MSB-first on
            ``sorted(wires)``.
        U (np.ndarray): ``d x d`` unitary on the active target subspace.
        wires (list): All wires; the first wire is the control.
        control_val (int): Control value that ``U`` acts on.
        active_indices (array-like): Target-subspace indices ``U`` occupies.

    Returns:
        tuple:
            - new_U (np.ndarray): ``U`` with the target diagonal folded in.
            - new_full_diag (np.ndarray): Full diagonal with the absorbed part removed.
            - controlled_new_U (np.ndarray): ``new_U`` lifted to the full register.
    """
    active_indices = np.asarray(active_indices)
    wires = list(wires)
    n_qubits = len(wires)
    n_control = 1
    n_target = n_qubits - n_control
    full_dim = 2**n_qubits
    target_dim = 2**n_target

    full_diag = np.asarray(full_diag, dtype=complex)

    # Build the control mask (rows matching control_val) and a map from each
    # full-register index to its target-subspace index. Controls are the leading
    # wires, targets the trailing ones (both MSB-first).
    idx = np.arange(full_dim)
    mask = np.ones(full_dim, dtype=bool)
    target_sub = np.zeros(full_dim, dtype=int)
    for i in range(n_control):
        bit = n_qubits - 1 - i
        mask &= ((idx >> bit) & 1) == ((control_val >> (n_control - 1 - i)) & 1)
    for j in range(n_target):
        bit = n_qubits - 1 - (n_control + j)
        target_sub |= ((idx >> bit) & 1) << (n_target - 1 - j)

    # Collapse the controlled rows of full_diag down to a target-only diagonal.
    target_diag = np.zeros(target_dim, dtype=complex)
    target_diag[target_sub[mask]] = full_diag[mask]

    # Merge the active part of that diagonal into U; the inactive entries become
    # the remaining diagonal (set to 1 where absorbed).
    new_U = U * target_diag[active_indices]
    remaining_target = target_diag.copy()
    remaining_target[active_indices] = 1.0

    # Write the leftover diagonal back onto the controlled rows only.
    new_full_diag = full_diag.copy()
    new_full_diag[mask] = remaining_target[target_sub[mask]]

    # Lift new_U back to the full register: U on the controlled block, identity elsewhere.
    U_target = embed_unitary(new_U, target_dim, active_indices)
    slice_idx = np.where(mask)[0]
    sub = target_sub[slice_idx]
    controlled_new_U = np.eye(full_dim, dtype=complex)
    controlled_new_U[np.ix_(slice_idx, slice_idx)] = U_target[np.ix_(sub, sub)]

    return new_U, new_full_diag, controlled_new_U
