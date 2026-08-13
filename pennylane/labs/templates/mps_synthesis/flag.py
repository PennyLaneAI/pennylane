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
    get_controlled_unitary_msq,
    get_fractal_embedding_states,
    ints_to_control_bits,
    propagate_diagonal_through_unitary,
    split_d,
    split_diagonal_into_control_branches,
    split_diagonal_into_partially_multiplexed_rz,
    synthesis_csd,
)

# ============================================================================
# PartiallyMultiplexedFlag operation and flag merging
# ============================================================================


class PartiallyMultiplexedFlag(qp.operation.Operation):
    r"""A partially multiplexed sequence of single-qubit flags.

    Applies ``R_z(phi) R_y(theta)`` to the target wire (``wires[-1]``), multi-controlled by
    the control wires (``wires[:-1]``) on the given ``control_values``. Angles are flat arrays
    with one ``(rz, ry)`` pair per control pattern (length ``len(control_values)``); an
    uncontrolled flag uses ``control_values=[]``.
    """

    num_params = 2

    @staticmethod
    def _angle_arrays(rz_angles, ry_angles):
        r"""Flatten the ``R_z``/``R_y`` angles to 1D arrays, requiring equal length."""
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

        # Store as a tuple of tuples so the hyperparameter is hashable.
        control_values = tuple(tuple(int(b) for b in pattern) for pattern in control_values)

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
        """Return a copy of this flag with ``control_wire`` prepended as a new control.
        Args:
            control_wire (int or str): the new control wire.
            control_value (int): the bit value (0 or 1) it controls on.
        Returns:
            PartiallyMultiplexedFlag: the flag with the added control.
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
        """Decompose into per-pattern multi-controlled ``R_z(phi)`` and ``R_y(theta)`` gates."""
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
    """Merge same-wire ``PartiallyMultiplexedFlag`` operations into one, sorted by control value.

    Args:
        flags (Sequence[PartiallyMultiplexedFlag]): flags acting on identical wires and
            covering disjoint control patterns.
    Returns:
        PartiallyMultiplexedFlag: the merged flag.
    Raises:
        TypeError: if any element is not a ``PartiallyMultiplexedFlag``.
        ValueError: if the flags' wires differ or their control patterns overlap.
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

    if control_values:
        # Check for unallowed duplicate control values
        seen = set()
        for cv in control_values:
            key = tuple(int(b) for b in cv)
            if key in seen:
                raise ValueError(
                    f"duplicate control value {key} when merging flags; "
                    "flags to be merged must cover disjoint control patterns"
                )
            seen.add(key)

        # Sort the control values by their integer representation
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
    """Conservatively check whether ``op`` is diagonal in the computational basis.

    Returns ``True`` only for guaranteed cases — a ``PartiallyMultiplexedFlag`` with
    all-zero ``R_y`` angles, or ``CZ``/``CCZ`` (and controlled variants); ``False`` otherwise.
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
    """Conservatively test whether ``a`` and ``b`` commute.

    Returns ``True`` only for disjoint wires or two diagonal ops; may return ``False``
    for ops that actually commute, but never the reverse.
    """
    if set(a.wires).isdisjoint(set(b.wires)):
        return True
    return _is_diagonal_op(a) and _is_diagonal_op(b)


def merge_partially_multiplexed_flags_in_circuit(ops):
    """Merge same-wire ``PartiallyMultiplexedFlag`` ops in ``ops``, sliding past provably commuting ops.

    Args:
        ops (Sequence): operators to scan; non-flag ops pass through unchanged.
    Returns:
        list: the operators with mergeable flags combined.
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
    """Add one control qubit to a decomposition op.

    Args:
        op: a ``PartiallyMultiplexedFlag``, ``CZ``/``CCZ``, or an already-controlled op.
        control_wire: wire for the new control.
        control_value (int): state (0 or 1) the op is active on.
    Returns:
        the controlled operator.
    Raises:
        TypeError: if ``op`` is not a supported type.
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
    delta = qp.math.exp(1j * qp.math.stack([-omega / 2 + alpha, omega / 2 + alpha]))
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

    K00, K01, theta_Y, K10, K11, _, _, _ = synthesis_csd(matrix, shift=False)
    theta_Y = theta_Y.item()

    F0, diag0 = d2_generalized_flag_decomp(K11, wires)
    ops += F0

    diag1 = get_controlled_unitary_msq(qp.math.stack([1.0, K10[0, 0]]), wires, control_value=0)
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

    diag6 = get_controlled_unitary_msq(qp.math.stack([1.0, K00[0, 0]]), wires, control_value=0)
    diag = diag4_0 * diag5 * diag6
    return ops, diag


def flatten_ops(structure):
    """Flatten a (possibly nested) list of ops into a flat list in circuit order."""
    if isinstance(structure, list):
        flat = []
        for s in structure:
            flat.extend(flatten_ops(s))
        return flat
    return [structure]


def _map_nested_ops(structure, fn):
    """Apply ``fn`` to every op in a (possibly nested) op structure, preserving nesting."""
    if isinstance(structure, list):
        return [_map_nested_ops(s, fn) for s in structure]
    return fn(structure)


def _interleave_branches(branch0, branch1):
    """Interleave the ``wires[0]=0`` and ``wires[0]=1`` branch trees, merging paired flags.

    The two branches share a congruent ``[left_flags, cs_flag, right_flags]`` structure, so
    recursing position by position merges each corresponding flag pair into one fully
    multiplexed flag (Multiplexer Extension Property); shape mismatches at leaves leave a
    short unpaired tail.
    """
    # Base case: two leaf flags on the same wires -> Multiplexer Extension Property.
    if isinstance(branch0, PartiallyMultiplexedFlag) and isinstance(
        branch1, PartiallyMultiplexedFlag
    ):
        if branch0.wires == branch1.wires:
            return merge_partially_multiplexed_flags([branch0, branch1])
        return [branch0, branch1]
    # Otherwise recurse positionally, treating a lone flag as a singleton branch.
    list0 = branch0 if isinstance(branch0, list) else [branch0]
    list1 = branch1 if isinstance(branch1, list) else [branch1]
    paired = [_interleave_branches(a, b) for a, b in zip(list0, list1)]
    return paired + list0[len(paired) :] + list1[len(paired) :]


def _decompose_branch(diag_in, right_block, wires, control_val, size):
    """Propagate ``diag_in`` through a branch unitary, flag-decompose it, and re-apply the control.

    The branch ``right_block`` is selected by ``control_val`` on ``wires[0]``. Returns
    ``(branch_flags, residual_diag, controlled_diag)``.
    """
    N = 2 ** len(wires)
    active_states, _ = get_fractal_embedding_states(size, N // 2)
    right_block_prop, residual_diag, _ = propagate_diagonal_through_unitary(
        diag_in, right_block, wires, control_val=control_val, active_indices=active_states
    )
    branch_flags, controlled_diag = recursive_generalized_flag_decomposition(
        right_block_prop, wires[1:], _top=False
    )
    branch_flags = _map_nested_ops(branch_flags, lambda op: add_control(op, wires[0], control_val))
    controlled_diag = get_controlled_unitary_msq(controlled_diag, wires, control_value=control_val)
    return branch_flags, residual_diag, controlled_diag


def recursive_generalized_flag_decomposition(U_d, wires, _top=True):
    r"""Recursively flag-decompose a unitary on ``wires`` via the asymmetric CSD.

    Returns ``(ops, diagonal)``, where ``ops`` is a nested list of
    :class:`~.PartiallyMultiplexedFlag` operations and ``diagonal`` is the trailing
    diagonal factor. When ``_top=True``, a non-power-of-two active submatrix is fractally
    embedded first and the flags are merged into a flat circuit.
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
    right_block0, right_block1, cs_angles, left_block0, left_block1, _, _, _ = synthesis_csd(
        U_d, shift=True
    )

    # Flag decomposition of the left CSD blocks
    left_flags0, left_diag0 = recursive_generalized_flag_decomposition(
        left_block0, wires[1:], _top=False
    )
    left_flags1, left_diag1 = recursive_generalized_flag_decomposition(
        left_block1, wires[1:], _top=False
    )

    # Add control to every flag, keeping whatever nesting the sub-result has
    left_flags0 = _map_nested_ops(left_flags0, lambda op: add_control(op, wires[0], 0))
    left_flags1 = _map_nested_ops(left_flags1, lambda op: add_control(op, wires[0], 1))
    left_diag = qp.math.concatenate((left_diag0, left_diag1))

    # Split diagonal into partially multiplexed Rz and remaining diagonal
    control_states, _ = get_fractal_embedding_states(min(p, q), N // 2)
    rz_angles, new_full_diagonal, _ = split_diagonal_into_partially_multiplexed_rz(
        left_diag, wires[1:] + wires[:1], control_states
    )

    control_values = ints_to_control_bits(control_states, nwires - 1)
    cs_flag = PartiallyMultiplexedFlag(rz_angles, cs_angles, wires[1:] + wires[:1], control_values)

    # Propagate, flag-decompose and re-control each branch (block 0 then block 1).
    right_flags0, right_diag0_prop, right_diag0 = _decompose_branch(
        new_full_diagonal, right_block0, wires, 0, p
    )
    right_diag0_total = right_diag0_prop * right_diag0
    right_flags1, right_diag1_prop, right_diag1 = _decompose_branch(
        right_diag0_total, right_block1, wires, 1, q
    )
    trailing_diag = right_diag1 * right_diag1_prop

    # Interleave the branches recursively over the nested structure
    left_flags = _interleave_branches(left_flags0, left_flags1)
    right_flags = _interleave_branches(right_flags0, right_flags1)

    ops = [left_flags, cs_flag, right_flags]
    if _top:
        ops = merge_partially_multiplexed_flags_in_circuit(flatten_ops(ops))
    return ops, trailing_diag


def _state_value(bitstring, order):
    """Integer value of ``bitstring`` reading bit positions ``order`` MSB->LSB."""
    return int("".join(str(bitstring[w]) for w in order), 2) if order else 0


def _count_blocks(bitstrings, order):
    """Number of maximal runs of consecutive integer values under ``order``."""
    vals = sorted(_state_value(bs, order) for bs in bitstrings)
    return 1 + sum(1 for a, b in zip(vals, vals[1:]) if b != a + 1)


def optimize_unary_sequence(bitstrings, wires):
    """Pick the wire ordering that splits the control states into the fewest contiguous blocks.

    Args:
        bitstrings (list): control states, MSB-first; ``wires`` is one longer (target last).
        wires (list): wire positions MSB -> LSB, including the target ``wires[-1]``.
    Returns:
        tuple: ``(reordered_wires, subsequences, index_blocks)`` — the chosen wire order, the
        rewired bitstrings grouped into blocks, and each state's original index (to reorder
        the matching angles).
    Raises:
        ValueError: if ``wires`` is not exactly one longer than the bitstrings.
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
    """Rewire each flag's control wire ordering to minimize unary blocks.

    Args:
        ops (list): flag operations to rewire.
        wires (list): default ordering for single-control flags.
        separate_target (bool): if ``True``, return ``(control_wires, target_wires)``.
    Returns:
        list or tuple: the rewired wire sequences, or separated control/target lists.
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
