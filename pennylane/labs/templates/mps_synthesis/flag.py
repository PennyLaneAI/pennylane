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
"""The :class:`PartiallyMultiplexedFlag` operation and flag decompositions.

Contains the multiplexed-flag operation, helpers to merge flags, and the
recursive generalized flag decomposition of unitaries into flag circuits.
"""

import itertools

import numpy as np

import pennylane as qp
from pennylane.math.decomposition import zyz_rotation_angles

from .linalg import (
    control_bits_to_ints,
    count_nontrivial_diagonal,
    csd,
    extract_active_submatrix,
    get_controlled_unitary_msq,
    get_fractal_embedding_states,
    ints_to_control_bits,
    propagate_diagonal_through_unitary,
    split_d,
    split_diagonal_into_control_branches,
    split_diagonal_into_partially_multiplexed_rz,
)

# ============================================================================
# PartiallyMultiplexedFlag operation and flag merging
# ============================================================================


class PartiallyMultiplexedFlag(qp.operation.Operation):
    """
    A partially multiplexed sequence of single qubit flags.
    Each entry applies ``R_z(phi) R_y(theta)`` to the target wire, multi-controlled
    by specific computational basis states.

    The control wires are ``wires[:-1]`` and carry explicit ``control_values``;
    the target wire is ``wires[-1]``. Angles are stored as a flat array with one
    ``(rz, ry)`` pair per control pattern (so its length equals
    ``len(control_values)``). Uncontrolled gates (one wire) use ``control_values=[]``.
    """

    num_params = 2

    @staticmethod
    def _angle_arrays(rz_angles, ry_angles):
        phi = qp.math.flatten(qp.math.atleast_1d(rz_angles))
        theta = qp.math.flatten(qp.math.atleast_1d(ry_angles))
        if len(phi) != len(theta):
            raise ValueError("rz_angles and ry_angles must have the same length")
        return phi, theta

    def __init__(self, rz_angles, ry_angles, wires, control_values=None):
        wires = qp.wires.Wires(wires)
        control_wires = wires[:-1]

        phi, theta = self._angle_arrays(rz_angles, ry_angles)
        num_patterns = len(phi)

        if control_values is None:
            if not control_wires:
                control_values = []
            else:
                max_state = (2 ** len(control_wires)) - 1
                # Default control values are counted backwards
                control_values = ints_to_control_bits(
                    range(max_state, max_state - num_patterns, -1),
                    len(control_wires),
                )
        else:
            if not control_wires and control_values:
                raise ValueError("control_values must be [] for a flag with no control wires")
            if control_wires and len(control_values) != num_patterns:
                raise ValueError("len(control_values) must equal the number of patterns ")

        self._hyperparameters = {
            "control_values": control_values,
        }
        super().__init__(phi, theta, wires=wires)

    def __repr__(self):
        """Custom string representation to display control structure."""
        phi_vals = self.parameters[0]
        theta_vals = self.parameters[1]
        wire_list = self.wires.tolist()
        c_vals = self.hyperparameters["control_values"]
        return (
            f"PartiallyMultiplexedFlag({phi_vals}, {theta_vals}, wires={wire_list}, "
            f"control_values={c_vals})"
        )

    def add_control(self, control_wire, control_value):
        """
        Creates a new PartiallyMultiplexedFlag with an additional control qubit
        prepended.

        Args:
            control_wire (int or str): The new wire to act as a control.
            control_value (int): The binary state (0 or 1) the new wire controls on.

        Returns:
            PartiallyMultiplexedFlag: A new operation instance with updated wires
            and control values.
        """
        # Target wire stays at the very end (wires[-1]); new control goes first.
        new_wires = qp.wires.Wires(control_wire) + self.wires

        num_patterns = len(self.parameters[0])

        old_control_values = self.hyperparameters["control_values"]
        if not old_control_values:
            new_control_values = [[control_value] for _ in range(num_patterns)]
        else:
            new_control_values = [[control_value] + list(state) for state in old_control_values]

        return PartiallyMultiplexedFlag(
            rz_angles=self.parameters[0],
            ry_angles=self.parameters[1],
            wires=new_wires,
            control_values=new_control_values,
        )

    @staticmethod
    def compute_decomposition(
        rz_angles, ry_angles, wires, control_values=None, **_
    ):  # pylint: disable=arguments-differ
        wires = qp.wires.Wires(wires)
        target_wire = wires[-1]
        control_wires = list(wires[:-1])

        phi, theta = PartiallyMultiplexedFlag._angle_arrays(rz_angles, ry_angles)

        ops = []
        for p, (phi_p, theta_p) in enumerate(zip(phi, theta)):
            for angle, gate in ((phi_p, qp.RZ), (theta_p, qp.RY)):
                op = gate(angle, wires=target_wire)
                if control_wires:
                    op = qp.ctrl(op, control=control_wires, control_values=list(control_values[p]))
                ops.append(op)

        return ops


def merge_partially_multiplexed_flags(flags):
    """
    Merge several ``PartiallyMultiplexedFlag`` operations that share the same
    wires into a single ``PartiallyMultiplexedFlag``.

    Args:
        flags (Sequence[PartiallyMultiplexedFlag]): Flags to merge. They must
            all act on the same wires (in the same order).

    Returns:
        PartiallyMultiplexedFlag: The merged operation.
    """
    flags = list(flags)
    for flag in flags:
        if not isinstance(flag, PartiallyMultiplexedFlag):
            raise TypeError(
                "merge_partially_multiplexed_flags only accepts "
                f"PartiallyMultiplexedFlag instances; got {type(flag).__name__}"
            )

    wires = flags[0].wires
    for flag in flags[1:]:
        if flag.wires != wires:
            raise ValueError(
                "all flags must share the same wires to be merged; "
                f"got {wires.tolist()} and {flag.wires.tolist()}"
            )

    rz_angles = qp.math.concatenate([qp.math.atleast_1d(flag.parameters[0]) for flag in flags])
    ry_angles = qp.math.concatenate([qp.math.atleast_1d(flag.parameters[1]) for flag in flags])

    control_values = []
    for flag in flags:
        control_values.extend(flag.hyperparameters["control_values"])

    # Sort the control values by their integer representation
    if control_values:
        order = sorted(
            range(len(control_values)),
            key=lambda i: control_bits_to_ints([int(bit) for bit in control_values[i]]),
        )
        control_values = [control_values[i] for i in order]
        rz_angles = qp.math.stack([rz_angles[i] for i in order])
        ry_angles = qp.math.stack([ry_angles[i] for i in order])

    return PartiallyMultiplexedFlag(
        rz_angles=rz_angles,
        ry_angles=ry_angles,
        wires=wires,
        control_values=control_values,
    )


def _is_diagonal_op(op):
    """
    Conservative check whether ``op`` is diagonal in the computational basis.

    Returns ``True`` only for cases we can guarantee:
      * a ``PartiallyMultiplexedFlag`` whose ``RY`` angles are all zero (only
        ``RZ`` phases remain),
      * ``CZ`` / ``CCZ`` and controlled-``CZ`` ops.
    Anything else returns ``False`` (treated as possibly non-diagonal).
    """
    if isinstance(op, PartiallyMultiplexedFlag):
        ry = qp.math.atleast_1d(op.parameters[1])
        return bool(qp.math.allclose(ry, 0.0))
    name = getattr(op, "name", "")
    if isinstance(op, qp.CZ) or name in ("CZ", "CCZ"):
        return True
    if isinstance(op, qp.ops.op_math.ControlledOp):
        return _is_diagonal_op(op.base)
    return False


def _ops_commute(a, b):
    """
    Commutation test used to decide whether a flag may be
    slid past an intervening operator. Only returns ``True`` for disjoint wires
    or if both operators are diagonal in the computational basis. May return
    ``False`` for ops that actually commute (never the reverse).
    """
    if set(a.wires).isdisjoint(set(b.wires)):
        return True
    return _is_diagonal_op(a) and _is_diagonal_op(b)


def merge_partially_multiplexed_flags_in_circuit(ops):
    """
    Merge all ``PartiallyMultiplexedFlag`` operations in ``ops`` that can be
    combined, including flags separated by other operators when it is provably
    safe to bring them together.

    Args:
        ops (Sequence): A list of operators (e.g. a flag decomposition). Non-flag
            operators are passed through unchanged.

    Returns:
        list: A new list of operators with mergeable flags combined.
    """
    result = []
    for op in ops:
        if isinstance(op, PartiallyMultiplexedFlag):
            target_idx = None
            for j in range(len(result) - 1, -1, -1):
                prev = result[j]
                if isinstance(prev, PartiallyMultiplexedFlag) and prev.wires == op.wires:
                    target_idx = j
                    break
                if not _ops_commute(op, prev):
                    break
            if target_idx is not None:
                result[target_idx] = merge_partially_multiplexed_flags([result[target_idx], op])
                continue
        result.append(op)
    return result


def add_control(op, control_wire, control_value):
    """
    Add one control qubit to a decomposition op.

    Args:
        op: A ``PartiallyMultiplexedFlag``, a ``qp.CZ``, or an already-controlled
            op (e.g. a ``ControlledOp`` produced by a previous ``add_control``
            during a deeper level of the recursion).
        control_wire: Wire index/label for the new control.
        control_value: 0 or 1 — state on which the op is active.

    Returns:
        A new operator with the control applied.
    """
    if isinstance(op, PartiallyMultiplexedFlag):
        return op.add_control(control_wire, control_value)
    if isinstance(op, (qp.CZ, qp.ops.op_math.ControlledOp)) or getattr(op, "name", "") in (
        "CZ",
        "CCZ",
    ):
        return qp.ctrl(op, control=control_wire, control_values=control_value)
    raise TypeError(
        "add_control supports PartiallyMultiplexedFlag, CZ, and controlled ops; "
        f"got {type(op).__name__}"
    )


# ============================================================================
# Recursive generalized flag decomposition
# ============================================================================


def _zyz_flag(matrix: np.ndarray) -> tuple:
    """ZYZ Euler decomposition of a ``2x2`` matrix into ``(phi, theta)`` flag
    angles and the trailing two-element diagonal ``delta``."""
    phi, theta, omega, alpha = zyz_rotation_angles(matrix)
    delta = np.exp(1j * np.array([-omega / 2 + alpha, omega / 2 + alpha]))
    return phi, theta, delta


def one_qubit_flag_decomp(matrix: np.ndarray, wires: list) -> tuple[list, np.ndarray]:
    """
    Implements the one-qubit flag decomposition returning the two-gate flag circuit
    and the trailing two-element diagonal. This is based on a standard Euler decomposition.

    Args:
        matrix (np.ndarray): Matrix of shape ``(2, 2)`` to be decomposed.
        wires (list): Wires on which the operations should act. Should have length 1.

    Returns:
        tuple[list, np.ndarray]: List of a single operation (a ``MultiplexedFlag``) and a
        one-dimensional array of length ``2`` representing the diagonal.
    """
    phi, theta, Delta = _zyz_flag(matrix)
    F = [PartiallyMultiplexedFlag(phi, theta, wires)]
    return F, Delta


def d2_generalized_flag_decomp(
    matrix: np.ndarray, wires: list, control_value=1
) -> tuple[list, np.ndarray]:
    """``d=2`` flag decomposition on a two-qubit register (N=4)."""
    if len(wires) != 2:
        raise ValueError(f"d=2 base case requires two wires (N=4), got {len(wires)}")

    wires = list(wires)
    phi, theta, delta = _zyz_flag(matrix)
    ops = [PartiallyMultiplexedFlag(phi, theta, wires, control_values=[[control_value]])]
    controlled = get_controlled_unitary_msq(delta, wires, control_value=control_value)
    return ops, controlled


def d3_generalized_flag_decomp(matrix: np.ndarray, wires: list) -> tuple[list, np.ndarray]:
    """``d=3`` flag decomposition on a two-qubit register (N=4)."""
    if len(wires) != 2:
        raise ValueError(f"d=3 base case requires two wires (N=4), got {len(wires)}")

    wires = list(wires)
    ops = []

    K00, K01, theta_Y, K10, K11, _, _, _ = csd(matrix, shift=False)
    theta_Y = theta_Y.item()

    F0, diag0 = d2_generalized_flag_decomp(K11, wires)
    ops += F0

    diag1 = get_controlled_unitary_msq(np.array([1.0, K10.item()]), wires, control_value=0)
    diag2 = diag1 * diag0

    diag2_0, _, _, diag2_1_target = split_diagonal_into_control_branches(diag2, wires[::-1])
    Ry = qp.matrix(qp.RY(theta_Y, wires=wires[0]))
    V = Ry * diag2_1_target

    F1, diag3 = d2_generalized_flag_decomp(V, wires[::-1])
    ops += F1

    diag4 = diag2_0 * diag3
    diag4_0, _, _, diag4_1_target = split_diagonal_into_control_branches(diag4, wires)

    W = K01 * diag4_1_target
    F2, diag5 = d2_generalized_flag_decomp(W, wires)
    ops += F2

    diag6 = get_controlled_unitary_msq(np.array([1.0, K00.item()]), wires, control_value=0)
    diag = diag4_0 * diag5 * diag6
    return ops, diag


def flatten_ops(structure):
    """Flatten a (possibly nested) list of flags/ops into a flat list of operators
    in circuit order. Handles arbitrary nesting (flat list, single op, or nested
    lists); a flat list is returned unchanged."""
    if isinstance(structure, list):
        flat = []
        for s in structure:
            flat.extend(flatten_ops(s))
        return flat
    return [structure]


def _map_nested_ops(structure, fn):
    """Apply ``fn`` to every operator in a (possibly nested) op structure,
    preserving the nesting. Works on a flat list, a single op, or a nested list."""
    if isinstance(structure, list):
        return [_map_nested_ops(s, fn) for s in structure]
    return fn(structure)


def _interleave_branches(B_p, B_q):
    """Interleave the ``wires[0]=0`` branch (``B_p``) and the ``wires[0]=1`` branch
    (``B_q``) by walking both branch trees in parallel.

    Both branches are produced by the same recursive procedure on the same wires,
    so their nested ``[FL, FCS, FR]`` structure is congruent except at the base
    cases (a ``d=2`` leaf is a single flag, a ``d=3`` leaf is three). Recursing
    position by position merges every corresponding pair of flags into one fully
    multiplexed flag via the Multiplexer Extension Property (this reproduces
    flagsynth's fully multiplexed single-qubit flags for the qubit case), while a
    shape mismatch only leaves a short, unpaired tail at the leaf where it occurs.
    """
    # Base case: two leaf flags on the same wires -> Multiplexer Extension Property.
    if isinstance(B_p, PartiallyMultiplexedFlag) and isinstance(B_q, PartiallyMultiplexedFlag):
        if B_p.wires == B_q.wires:
            return merge_partially_multiplexed_flags([B_p, B_q])
        return [B_p, B_q]
    # Otherwise recurse positionally, treating a lone flag as a singleton branch.
    P = B_p if isinstance(B_p, list) else [B_p]
    Q = B_q if isinstance(B_q, list) else [B_q]
    paired = [_interleave_branches(a, b) for a, b in zip(P, Q)]
    return paired + P[len(paired) :] + Q[len(paired) :]


def _decompose_branch(diag_in, R, wires, control_val, size):
    """Propagate ``diag_in`` through branch unitary ``R`` (selected by
    ``control_val`` on ``wires[0]``), recursively flag-decompose the result, and
    re-apply the control. Returns ``(flags, propagated_diagonal, controlled_diagonal)``."""
    N = 2 ** len(wires)
    active_states, _ = get_fractal_embedding_states(size, N // 2)
    R_dash, D_dash, _ = propagate_diagonal_through_unitary(
        diag_in, R, wires, control_val=control_val, active_indices=active_states
    )
    FR, DR = recursive_generalized_flag_decomposition(R_dash, wires[1:], _top=False)
    FR = _map_nested_ops(FR, lambda op: add_control(op, wires[0], control_val))
    DR = get_controlled_unitary_msq(DR, wires, control_value=control_val)
    return FR, D_dash, DR


def recursive_generalized_flag_decomposition(U_d, wires, _top=True):
    """Recursively flag-decompose a unitary on ``wires``.

    Returns ``(ops, diagonal)`` where ``ops`` is a nested list of
    :class:`~.PartiallyMultiplexedFlag` operations and ``diagonal`` is the
    trailing diagonal factor. When ``_top`` is ``True``, the input may be a
    non-power-of-two active submatrix that is fractally embedded first.
    """
    d = U_d.shape[0]
    wires = list(wires)

    # Base cases
    if len(wires) == 1:
        return one_qubit_flag_decomp(U_d, wires)
    if len(wires) == 2:
        if d == 2:
            return d2_generalized_flag_decomp(U_d, wires)
        if d == 3:
            return d3_generalized_flag_decomp(U_d, wires)

    nwires = len(wires)
    N = 2**nwires

    p, q = split_d(d)
    R_p, R_q, theta, L_p, L_q, _, _, _ = csd(U_d, shift=True)

    # Flag decomposition of L_p and L_q
    FL_p, DL_p = recursive_generalized_flag_decomposition(L_p, wires[1:], _top=False)
    FL_q, DL_q = recursive_generalized_flag_decomposition(L_q, wires[1:], _top=False)

    # Add control to every flag, keeping whatever nesting the sub-result has
    FL_p = _map_nested_ops(FL_p, lambda op: add_control(op, wires[0], 0))
    FL_q = _map_nested_ops(FL_q, lambda op: add_control(op, wires[0], 1))
    DL = np.concatenate((DL_p, DL_q))

    # Split diagonal into partially multiplexed Rz and remaining diagonal
    control_states, _ = get_fractal_embedding_states(min(p, q), N // 2)
    phi, new_full_diagonal, _ = split_diagonal_into_partially_multiplexed_rz(
        DL, wires[1:] + wires[:1], control_states
    )

    control_values = ints_to_control_bits(control_states, nwires - 1)
    FCS = PartiallyMultiplexedFlag(phi, theta, wires[1:] + wires[:1], control_values)

    # Propagate, flag-decompose and re-control each branch (R_p then R_q).
    FR_p, DR_p_dash, DR_p = _decompose_branch(new_full_diagonal, R_p, wires, 0, p)
    DR_total = DR_p_dash * DR_p
    FR_q, DR_q_dash, DR_q = _decompose_branch(DR_total, R_q, wires, 1, q)
    diag = DR_q * DR_q_dash

    # Interleave the branches recursively over the nested structure
    FL = _interleave_branches(FL_p, FL_q)
    FR = _interleave_branches(FR_p, FR_q)

    ops = [FL, FCS, FR]
    if _top:
        ops = merge_partially_multiplexed_flags_in_circuit(flatten_ops(ops))
    return ops, diag


def reconstruct_unitary(ops, diagonal, wires, d=None):
    """Reconstruct the active ``d x d`` block from a flag decomposition.
    If d=None, no extraction of the active submatrix is performed.
    Accepts either a flat or a nested op structure."""
    ops = flatten_ops(ops)
    N = 2 ** len(wires)
    U_d_reconstructed = np.diag(diagonal) @ qp.matrix(ops, wire_order=wires)
    if d is not None:
        active_states, _ = get_fractal_embedding_states(d, N)
        U_d_reconstructed = extract_active_submatrix(
            U_d_reconstructed, active_indices=active_states
        )
    return U_d_reconstructed


def check_reconstruction(U_d, ops, diag, wires, d):
    """Print whether ``ops`` and ``diag`` reconstruct the active submatrix of ``U_d``."""
    U_d_reconstructed_active = reconstruct_unitary(ops, diag, wires, d)
    print(f"Reconstruction correct (d={d}): ", np.allclose(U_d, U_d_reconstructed_active))


def get_parameter_count(ops, diag):
    """Return ``(num_angles, num_diag_params)`` for a flag decomposition."""
    ops = flatten_ops(ops)
    tape = qp.tape.QuantumScript(ops)
    params = tape.get_parameters()  # list of the stored phi/theta arrays
    num_angles = sum(qp.math.size(p) for p in params)
    num_diag_params = count_nontrivial_diagonal(diag)
    return num_angles, num_diag_params


def check_parameter_count(ops, diag, d):
    """Print whether the flag decomposition uses exactly ``d**2`` parameters."""
    num_angles, num_diag_params = get_parameter_count(ops, diag)
    print(f"Parameter count correct (d^2={d**2}): ", num_angles + num_diag_params == d**2)


def _state_value(bitstring, order):
    """Integer value of ``bitstring`` reading bit positions ``order`` MSB->LSB."""
    return int("".join(str(bitstring[w]) for w in order), 2) if order else 0


def _count_blocks(bitstrings, order):
    """Number of maximal runs of consecutive integer values under ``order``."""
    vals = sorted(_state_value(bs, order) for bs in bitstrings)
    return 1 + sum(1 for a, b in zip(vals, vals[1:]) if b != a + 1)


def optimize_unary_sequence(bitstrings, wires):
    """
    Choose the wire ordering that minimizes the number of contiguous blocks the
    control states split into.

    wires: list of wire positions, MSB -> LSB (including wires[-1] target)

    Returns:
        tuple: (wire_ordering, subsequences_rewired, index_blocks)
            wire_ordering: bit positions, MSB -> LSB.
            subsequences_rewired: list of blocks of rewired bitstrings.
            index_blocks: same shape, holding the original index of each state
                          so associated angles can be reordered to match.
    """
    if not bitstrings:
        return [], [], []
    if len(bitstrings) == 1:
        return wires, [], []
    if len(wires) != len(bitstrings[0]) + 1:
        raise ValueError("wires must be one longer than the bitstrings")

    num_bits = len(bitstrings[0])
    num_states = len(bitstrings)

    # Stability of each bit: how many states share its majority value.
    ones = [sum(bs[i] for bs in bitstrings) for i in range(num_bits)]
    stability = [max(o, num_states - o) for o in ones]

    # Exhaustive search: minimize the block count, then prefer the ordering whose
    # most stable bits sit in the most significant positions.
    order = list(
        min(
            itertools.permutations(range(num_bits)),
            key=lambda o: (_count_blocks(bitstrings, o), tuple(-stability[w] for w in o)),
        )
    )

    def value(bs):
        return _state_value(bs, order)

    # Distinct control states commute -> free to reorder. Sort to maximize runs.
    perm = sorted(range(num_states), key=lambda k: value(bitstrings[k]))

    subsequences, index_blocks, prev = [], [], None
    for k in perm:
        v = value(bitstrings[k])
        rewired = [bitstrings[k][w] for w in order]
        if subsequences and v == prev + 1:
            subsequences[-1].append(rewired)
            index_blocks[-1].append(k)
        else:
            subsequences.append([rewired])
            index_blocks.append([k])
        prev = v

    # Place the target wire back at its original spot
    target = int(wires[-1])
    reordered_control = [wires[w] for w in order]
    reordered_wires = reordered_control[:target] + [target] + reordered_control[target:]

    return reordered_wires, subsequences, index_blocks


def get_rewired_control_sequences(ops, wires, separate_target=False):
    """Rewrite control wire orderings for each flag to minimize unary blocks.

    Args:
        ops (list): Flag operations whose control values will be rewired.
        wires (list): Default wire ordering used for single-control flags.
        separate_target (bool): If ``True``, return ``(control_wires, target_wires)``
            instead of the combined rewired wire lists.

    Returns:
        list or tuple: Rewired wire sequences, or separated control/target lists.
    """
    routed_wires = []
    for F in ops:
        bitstrings = [list(pattern) for pattern in F.hyperparameters["control_values"]]
        reordered_wires, _, _ = optimize_unary_sequence(bitstrings, list(F.wires))
        if len(bitstrings) == 1:
            routed_wires.append(list(wires))
        else:
            routed_wires.append(reordered_wires)

    if separate_target:
        target_wires = [op.wires[-1] for op in ops]
        control_wires = [
            [w for w in group if w != target] for group, target in zip(routed_wires, target_wires)
        ]
        return control_wires, target_wires
    return routed_wires
