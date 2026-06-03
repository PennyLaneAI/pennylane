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
r"""Contains the PartialUnaryStatePreparation template."""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

import pennylane as qp
from pennylane import allocate, math
from pennylane.exceptions import DecompositionUndefinedError
from pennylane.operation import Operation
from pennylane.wires import Wires, WiresLike

# =============================================================================
# Classical Algorithm for Finding the Isometry (Algorithm 1 from the paper)
# =============================================================================


def _int_to_binary(vals, n_bits):
    """Convert integer(s) to binary array representation (MSB first).

    Args:
        vals: int or array-like of ints
        n_bits: number of bits

    Returns:
        2D numpy array of shape (len(vals), n_bits) or 1D if scalar input
    """
    vals = np.asarray(vals)
    scalar = vals.ndim == 0
    vals = np.atleast_1d(vals)
    # Bit positions from MSB to LSB
    shifts = np.arange(n_bits - 1, -1, -1)
    bits = ((vals[:, None] >> shifts) & 1).astype(np.int8)
    return bits[0] if scalar else bits


@dataclass
class Batch:
    states: list = field(default_factory=list)
    qubit_positions: list = field(default_factory=list)
    states_set: set = field(default_factory=set)

    def append(self, state, qubit_pos):
        self.states.append(state)
        self.states_set.add(state)
        self.qubit_positions.append(qubit_pos)

    def __len__(self):
        return len(self.states)

    def __bool__(self):
        return bool(self.states)


class IsometryFinder:
    """
    Classical algorithm that finds the isometry circuit and bijection.
    """

    def __init__(self, basis_states: list, n_qubits: int):
        self.s = len(basis_states)
        self.n = n_qubits
        self.l = max(math.ceil_log2(self.s), 1)
        self.r_size = self.n - self.l
        self.m = 1 << int(math.floor(math.log2(max(self.r_size, 1))))

        # Initialize tableau as int8 for memory efficiency
        self.tableau = _int_to_binary(basis_states, self.n)
        self.circuit_ops = []
        self.bijection = {}

        # Pre-compute subspace membership (track incrementally)
        self._in_subspace = np.all(self.tableau[:, self.l :] == 0, axis=1)

    def _update_subspace_status(self):
        """Recompute which rows are in subspace (all non-subspace bits zero)."""
        self._in_subspace = np.all(self.tableau[:, self.l :] == 0, axis=1)

    def cx(self, control: int, target: int, wires: WiresLike):
        """Add a CNOT operation to the circuit ops and apply CNOT to the tableau."""
        self.circuit_ops.append(qp.CNOT([wires[control], wires[target]]))
        self.tableau[:, target] ^= self.tableau[:, control]

    def apply_multi_controlled_x(self, controls, control_values, target: int):
        """Apply multi-controlled X to the tableau."""
        if isinstance(controls, slice):
            ctrl_cols = self.tableau[:, controls]
            cv = np.asarray(control_values)
        else:
            ctrl_cols = self.tableau[:, controls]
            cv = np.asarray(control_values)
        # A row is flipped iff all control bits match control_values
        match = np.all(ctrl_cols == cv, axis=1)
        self.tableau[:, target] ^= match.astype(np.int8)

    def pui(self, k, b, batch_qubit_positions, wires, work_wires):
        """Add a Partial unary iterator circuit (in form of `Select`) to the circuit ops and
        apply the corresponding multicontrolled bit flips to the tableau."""

        subspace_wires = wires[: self.l]
        nonsubspace_wires = wires[self.l :]

        k_start = k - b
        target_wires = [nonsubspace_wires[batch_qubit_positions[idx]] for idx in range(b)]
        self.circuit_ops.append(qp.BasisEmbedding(k_start, subspace_wires))
        sub_ops = [qp.X(w) for w in target_wires]
        self.circuit_ops.append(
            qp.Select(sub_ops, control=subspace_wires, work_wires=work_wires, partial=False)
        )
        self.circuit_ops.append(qp.BasisEmbedding(k_start, subspace_wires))

        # Update tableau for PUI effect
        target_bits = _int_to_binary(k_start + np.arange(b), self.l)
        for j, _bits in enumerate(target_bits):
            ns_qubit = self.l + batch_qubit_positions[j]
            self.apply_multi_controlled_x(slice(0, self.l), _bits, ns_qubit)

        # Update subspace status after PUI
        self._update_subspace_status()

    def swap(self, w0, w1, wires):
        """Add a SWAP operation to the circuit ops and apply SWAP to the tableau."""
        self.circuit_ops.append(qp.SWAP([wires[w0], wires[w1]]))
        self.tableau[:, [w0, w1]] = self.tableau[:, [w1, w0]]

    def _next_state_with_target_set(self, target_qubit):
        actual_qubit = self.l + target_qubit

        # which rows have bit at actual_qubit set?
        col = self.tableau[:, actual_qubit]
        search_idx = np.where(col)[0]
        return int(search_idx[0]) if len(search_idx) else None

    def _try_swap(self, remaining, target_qubit, wires):
        remaining_arr = np.array(remaining)
        ns_cols = self.tableau[remaining_arr, self.l + target_qubit + 1 : self.l + self.r_size]
        if ns_cols.size > 0:
            # Find first (row, col) with a 1
            row_idx, col_idx = np.where(ns_cols)
            if len(row_idx):
                swap_from = self.l + target_qubit + 1 + int(col_idx[0])
                self.swap(self.l + target_qubit, swap_from, wires)
                return int(remaining_arr[row_idx[0]])

        return None

    def _get_diff_qubit(self, idx, j, actual_qubit, batch):
        mismatches = np.where(self.tableau[idx] != self.tableau[batch.states[j]])[0]
        if len(mismatches) and mismatches[0] != actual_qubit:
            return int(mismatches[0])
        if len(mismatches) > 1:
            return int(mismatches[1] if mismatches[0] == actual_qubit else mismatches[0])
        return None

    def _toffoli_trick(self, remaining, target_qubit, batch, wires, work_wires):
        actual_qubit = self.l + target_qubit

        for idx in remaining:
            ns = self.tableau[idx, self.l :]
            for j in range(target_qubit):
                if ns[j] == 1:
                    diff_qubit = self._get_diff_qubit(idx, j, actual_qubit, batch)

                    if diff_qubit is None:
                        continue

                    ctrl_val = int(self.tableau[idx, diff_qubit])
                    mcx_wires = [wires[self.l + j], wires[diff_qubit], wires[actual_qubit]]
                    self.circuit_ops.append(
                        qp.MultiControlledX(
                            mcx_wires,
                            control_values=[1, ctrl_val],
                            work_wires=work_wires[0],
                            work_wire_type="zeroed",
                        )
                    )
                    self.apply_multi_controlled_x(
                        [self.l + j, diff_qubit],
                        np.array([1, ctrl_val]),
                        actual_qubit,
                    )
                    return idx

        return None

    def _zero_non_subspace(self, found_state, target_qubit, wires):
        actual_qubit = self.l + target_qubit
        ns = self.tableau[found_state, self.l :]
        for j in np.where(ns)[0]:
            j = int(j)
            if j != target_qubit:
                self.cx(actual_qubit, self.l + j, wires)

    def _set_subspace_to_k(self, found_state, k, target_qubit, wires):
        actual_qubit = self.l + target_qubit
        k_bits = _int_to_binary(k, self.l)
        current_sub = self.tableau[found_state, : self.l]
        for j in np.where(current_sub != k_bits)[0]:
            self.cx(actual_qubit, int(j), wires)

    def flush(self, k, b, qubit_positions, wires, work_wires):
        self.pui(k, b, batch.qubit_positions, wires, work_wires)
        return Batch()

    @qp.QueuingManager.stop_recording()
    def find_isometry(self, wires, work_wires):
        """
        Find isometry using the batched PUI approach (Algorithm 1).
        Optimized: tracks subspace membership incrementally, minimizes
        redundant numpy operations.
        """
        k = 0
        # Pre-allocate set for batch membership checks
        batch = Batch()

        while True:
            b = len(batch)

            # Get remaining indices not in subspace and not in batch
            remaining = np.where(~self._in_subspace)[0]
            remaining = [int(i) for i in remaining if i not in batch.states_set]

            need_flush = (b == self.m) or (not remaining and batch)

            if b == self.m and batch:
                batch = self.flush(k, b, batch.qubit_positions, wires, work_wires)
                continue

            if not remaining and batch:
                if batch:
                    batch = self.flush(k, b, batch.qubit_positions, wires, work_wires)
                continue

            if not remaining:
                # Nothing remaining, exit.
                break

            # Find a state with the b-th non-subspace qubit set to 1
            target_qubit = b
            found_state = self._next_state_with_target_set(target_qubit)

            if found_state is None:
                # Try swapping with a qubit further right to get a state with b-th
                # non-subspace qubit set to 1
                found_state = self._try_swap(remaining, target_qubit, wires)

            if found_state is None and batch:
                # Use Toffoli trick to find a state
                found_state = self._toffoli_trick(remaining, target_qubit, batch, wires, work_wires)

            if found_state is None:
                # If really no state could be found, flush the PUI and start a new batch.
                if batch:
                    self.pui(k, b, batch.qubit_positions, wires, work_wires)
                    batch = Batch()
                continue

            # Transform found_state: zero out other non-subspace bits using CX
            self._zero_non_subspace(found_state, target_qubit, wires)

            # Transform subspace to |k>
            self._set_subspace_to_k(found_state, k, target_qubit, wires)

            # Append the found state to the batch and register it in the state bijection
            batch.append(found_state, target_qubit)
            self.bijection[found_state] = k
            k += 1

        # Assign bijection for states already in subspace
        vals = self.tableau[:, : self.l] @ (2 ** np.arange(self.l - 1, -1, -1))
        for i, val in enumerate(vals):
            self.bijection.setdefault(i, int(val))

        return self.circuit_ops, self.bijection


class PartialUnaryStatePreparation(Operation):
    r"""Prepare an arbitrary quantum state with the partial unary iteration technique.

    This operation prepares an arbitrary state

    .. math:: |\psi\rangle = \sum_{\ell \in L } c_\ell |\ell\rangle,

    where :math:`L` denotes the set of ``indices`` and :math:`c_\ell` is the ``coefficient``
    corresponding to the index :math:`\ell\in L`.
    The state :math:`|\ell\rangle` is a computational basis state, interpreted via the
    binary representation of :math:`\ell`.

    This state preparation technique was introduced in
    `Rupprecht and Wölk, arXiv:2601.09388 <https://arxiv.org/abs/2601.09388>`__.

    Args:
        coefficients (np.ndarray): Coefficients of the sparse state to prepare. The ordering should
            match that in ``indices``.
        wires (qp.wires.WiresLike): Wires on which to prepare the state. All work wires will be
            allocated dynamically with :func:`~.allocate`.
        indices (tuple[int]): Indices of the sparse state to prepare. The ordering should match
            that in ``coefficients``.

    .. warning::

        Note that we require ``coefficients`` to be treated as numerical data in the form of an
        array, whereas the ``indices`` need to be hashable, and thus will be treated as static
        information. This is because ``indices`` significantly impacts the structure and size of
        the circuit that realizes the state preparation.

    **Example**

    #TODO

    """

    resource_keys = {"num_entries", "num_wires", "num_work_wires"}

    @property
    def resource_params(self):
        return {
            "num_entries": len(self.hyperparameters["indices"]),
            "num_wires": len(self.wires),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
        }

    def __init__(self, coefficients, wires, indices, work_wires):
        work_wires = Wires([] if work_wires is None else work_wires)
        super().__init__(coefficients, wires=wires)
        self.hyperparameters["indices"] = indices
        self.hyperparameters["work_wires"] = Wires(work_wires)

    @property
    def has_decomposition(self):
        """We are using ``qp.allocate`` in the decomposition, so the validation for
        decomposition in the old system breaks. Hence we manually deactivate the fallback
        of ``compute_decomposition`` to the new decomp system that is implemented in
        ``Operator.compute_decomposition``. Accordingly we set ``has_decomposition=False`` here."""
        return False

    @staticmethod
    def compute_decomposition(*_, **__):  # pylint: disable=arguments-differ
        """We are using ``qp.allocate`` in the decomposition, so the validation for
        decomposition in the old system breaks. Hence we manually deactivate the fallback
        of ``compute_decomposition`` to the new decomp system that is implemented in
        ``Operator.compute_decomposition``."""
        raise DecompositionUndefinedError


def _pui_state_prep_resources(num_entries, num_wires, num_work_wires):
    """Compute the resources for _pui_state_prep."""
    ell = max(math.ceil_log2(num_entries), 1)
    if num_entries == 1:
        return {qp.resource_rep(qp.BasisState, num_wires=num_wires): 1}

    resources = defaultdict(int)
    if num_work_wires < ell - 1:
        resources[qp.resource_rep(qp.allocation.Allocate)] += 1
        resources[qp.resource_rep(qp.allocation.Deallocate)] += 1
    num_work_wires = max(num_work_wires, ell - 1, 1)
    resources[qp.resource_rep(qp.MultiplexerStatePreparation, num_wires=ell)] += 1
    R = num_wires - ell
    main_pui_batch_size = 1 << int(math.floor(math.log2(max(R, 1))))
    select_rep = qp.resource_rep(
        qp.Select,
        op_reps=(qp.resource_rep(qp.X),) * main_pui_batch_size,
        num_control_wires=ell,
        num_work_wires=num_work_wires,
        partial=False,
    )
    resources[select_rep] += max(num_entries // main_pui_batch_size, 1)

    rest_pui_batch_size = num_entries % main_pui_batch_size or main_pui_batch_size
    select_rep = qp.resource_rep(
        qp.Select,
        op_reps=(qp.resource_rep(qp.X),) * rest_pui_batch_size,
        num_control_wires=ell,
        num_work_wires=num_work_wires,
        partial=False,
    )
    resources[select_rep] += 1

    resources[qp.resource_rep(qp.CNOT)] += int(num_entries**1.15 * 4)
    embed_rep = qp.resource_rep(qp.BasisState, num_wires=ell)
    resources[embed_rep] += max(num_entries // 4, 1)
    swap_rep = qp.resource_rep(qp.SWAP)
    resources[swap_rep] += num_wires

    guess = int(num_wires / 10) + 1

    for num, num_zeroed in zip(
        [max(guess // 2, 1), max(guess - guess // 2, 1)], [0, 1], strict=True
    ):
        mcx_rep = qp.resource_rep(
            qp.MultiControlledX,
            num_control_wires=2,
            num_zero_control_values=num_zeroed,
            num_work_wires=1,
            work_wire_type="zeroed",
        )
        resources[mcx_rep] += num
    return resources


def _pui_state_prep_core(coefficients, wires, indices, work_wires):
    """Compute the decomposition of the partial unary iteration state preparation technique."""
    s = len(indices)
    if s == 1:
        qp.BasisState(indices[0], wires)
        return

    ell = max(math.ceil_log2(s), 1)
    iso_finder = IsometryFinder(indices, len(wires))
    ops, bijection = iso_finder.find_isometry(wires, work_wires)

    subspace_wires = wires[:ell]

    # Step 1: Dense state preparation
    dense_state = np.zeros(2**ell, dtype=complex)
    for i, amp in enumerate(coefficients):
        dense_state[bijection[i]] = amp

    qp.MultiplexerStatePreparation(dense_state, subspace_wires)

    # Step 2: Apply the inverse of the isometry circuit
    for op in reversed(ops):
        if qp.QueuingManager.recording():
            qp.apply(op)


def _pui_state_prep_provided_work_wires_condition(num_entries, num_wires, num_work_wires):
    # pylint: disable=unused-argument
    ell = max(math.ceil_log2(num_entries), 1)
    needed_work_wires = max(ell - 1, 1)
    return num_work_wires >= needed_work_wires


@qp.register_condition(_pui_state_prep_provided_work_wires_condition)
@qp.register_resources(_pui_state_prep_resources, exact=False)
def _pui_state_prep_provided_work_wires(coefficients, wires, indices, work_wires, **__):
    """Compute the decomposition of the partial unary iteration state preparation technique."""
    _pui_state_prep_core(coefficients, wires, indices, work_wires)


def _pui_state_prep_work_wires(num_entries, num_wires, num_work_wires):
    # pylint: disable=unused-argument
    ell = max(math.ceil_log2(num_entries), 1)
    needed_work_wires = max(ell - 1, 1)
    return {"zeroed": max(needed_work_wires - num_work_wires, 0)}


def _pui_state_prep_dyn_work_wires_condition(num_entries, num_wires, num_work_wires):
    # pylint: disable=unused-argument
    ell = max(math.ceil_log2(num_entries), 1)
    needed_work_wires = max(ell - 1, 1)
    return num_work_wires < needed_work_wires


@qp.register_condition(_pui_state_prep_dyn_work_wires_condition)
@qp.register_resources(
    _pui_state_prep_resources, work_wires=_pui_state_prep_work_wires, exact=False
)
def _pui_state_prep_dyn_work_wires(coefficients, wires, indices, **__):
    """Compute the decomposition of the partial unary iteration state preparation technique."""
    need_to_allocate = max(math.ceil_log2(len(indices)) - 1, 1)
    with allocate(need_to_allocate, state="zero", restored=True) as work_wires:
        _pui_state_prep_core(coefficients, wires, indices, work_wires)


qp.add_decomps(
    PartialUnaryStatePreparation,
    _pui_state_prep_dyn_work_wires,
    _pui_state_prep_provided_work_wires,
)
