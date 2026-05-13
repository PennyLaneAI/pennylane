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

import numpy as np

import pennylane as qp
from pennylane import allocate, math
from pennylane.decomposition import (
    add_decomps,
    register_resources,
    resource_rep,
)
from pennylane.exceptions import DecompositionUndefinedError
from pennylane.operation import Operation
from pennylane.wires import Wires

# =============================================================================
# Classical Algorithm for Finding the Isometry (Algorithm 1 from the paper)
# =============================================================================


class IsometryFinder:
    """
    Classical algorithm that finds the isometry circuit and bijection.

    Optimized: uses incremental tracking of subspace membership, avoids
    redundant recomputation, and uses vectorized numpy operations.
    """

    def __init__(self, basis_states: list, n_qubits: int):
        self.s = len(basis_states)
        self.n = n_qubits
        self.l = max(math.ceil_log2(self.s), 1)
        self.r_size = self.n - self.l
        self.m = 1 << int(math.floor(math.log2(max(self.r_size, 1))))

        # Initialize tableau as int8 for memory efficiency
        self.tableau = math.int_to_binary(basis_states, self.n)
        self.circuit_ops = []
        self.bijection = {}

        # Pre-compute subspace membership (track incrementally)
        self._in_subspace = np.all(self.tableau[:, self.l :] == 0, axis=1)

    def _update_subspace_status(self):
        """Recompute which rows are in subspace (all non-subspace bits zero)."""
        self._in_subspace = np.all(self.tableau[:, self.l :] == 0, axis=1)

    def apply_cx_to_tableau(self, control_qubit: int, target_qubit: int):
        """Apply CX gate to the tableau (vectorized)."""
        self.tableau[:, target_qubit] ^= self.tableau[:, control_qubit]

    def apply_x_to_tableau(self, qubit: int):
        """Apply X gate to the tableau."""
        self.tableau[:, qubit] ^= 1

    def apply_multi_controlled_x(self, controls, control_values, target: int):
        """Apply multi-controlled X to the tableau (vectorized)."""
        if isinstance(controls, slice):
            ctrl_cols = self.tableau[:, controls]
            cv = np.asarray(control_values)
        else:
            ctrl_cols = self.tableau[:, controls]
            cv = np.asarray(control_values)
        # A row is flipped iff all control bits match control_values
        match = np.all(ctrl_cols == cv, axis=1)
        self.tableau[:, target] ^= match.astype(np.int8)

    def pui(self, k, b, subspace_wires, nonsubspace_wires, batch_qubit_positions, pui_work_wires):
        """Add a Partial unary iterator circuit (in form of `Select`) to the circuit ops and
        apply the corresponding multicontrolled bit flips to the tableau."""
        k_start = k - b
        target_wires = [nonsubspace_wires[batch_qubit_positions[idx]] for idx in range(b)]
        self.circuit_ops.append(qp.BasisEmbedding(k_start, subspace_wires))
        sub_ops = [qp.X(w) for w in target_wires]
        self.circuit_ops.append(
            qp.Select(sub_ops, control=subspace_wires, work_wires=pui_work_wires, partial=False)
        )
        self.circuit_ops.append(qp.BasisEmbedding(k_start, subspace_wires))

        # Update tableau for PUI effect
        target_bits = math.int_to_binary(k_start + np.arange(b), self.l)
        for j, _bits in enumerate(target_bits):
            ns_qubit = self.l + batch_qubit_positions[j]
            self.apply_multi_controlled_x(slice(0, self.l), _bits, ns_qubit)

        # Update subspace status after PUI
        self._update_subspace_status()

    def swap(self, w0, w1, wires):
        """Add a SWAP operation to the circuit ops and apply SWAP to the tableau."""
        self.circuit_ops.append(qp.SWAP([wires[w0], wires[w1]]))
        self.tableau[:, [w0, w1]] = self.tableau[:, [w1, w0]]

    @qp.QueuingManager.stop_recording()
    def find_isometry(self, wires, pui_work_wires):
        """
        Find isometry using the batched PUI approach (Algorithm 1).
        Optimized: tracks subspace membership incrementally, minimizes
        redundant numpy operations.
        """
        k = 0
        batch = []
        batch_qubit_positions = []
        subspace_wires = wires[: self.l]
        nonsubspace_wires = wires[self.l :]

        # Only allocate work wires if l > 1 (otherwise Select needs none)
        need_to_allocate = len(pui_work_wires) - (self.l - 1)
        if need_to_allocate:
            with qp.queuing.AnnotatedQueue() as q:
                pui_work_wires = allocate(need_to_allocate, state="zero", restored=True)
            alloc = q.queue[0]
            dealloc = qp.allocation.Deallocate(alloc.wires)
            self.circuit_ops.append(dealloc)  # Reverse ordering
        else:
            alloc = None

        # Pre-allocate set for batch membership checks
        batch_set = set()

        while True:
            b = len(batch)

            # Get remaining indices not in subspace and not in batch
            remaining = np.where(~self._in_subspace)[0]
            remaining = [int(i) for i in remaining if i not in batch_set]

            need_flush = (b == self.m) or (not remaining and batch)

            if need_flush and batch:
                self.pui(
                    k, b, subspace_wires, nonsubspace_wires, batch_qubit_positions, pui_work_wires
                )
                batch = []
                batch_qubit_positions = []
                batch_set = set()
                continue

            if not remaining:
                break

            # Find a state with the b-th non-subspace qubit set to 1
            target_qubit_pos = b
            actual_qubit = self.l + target_qubit_pos

            # Vectorized search: which rows have bit at actual_qubit set?
            col = self.tableau[:, actual_qubit]
            search_idx = np.where(col)[0]
            found_state = int(search_idx[0]) if len(search_idx) else None

            if found_state is None:
                # Try swapping with a qubit further right
                remaining_arr = np.array(remaining)
                ns_cols = self.tableau[
                    remaining_arr, self.l + target_qubit_pos + 1 : self.l + self.r_size
                ]
                if ns_cols.size > 0:
                    # Find first (row, col) with a 1
                    row_idx, col_idx = np.where(ns_cols)
                    if len(row_idx):
                        swap_from = self.l + target_qubit_pos + 1 + int(col_idx[0])
                        self.swap(self.l + target_qubit_pos, swap_from, wires)
                        found_state = int(remaining_arr[row_idx[0]])

                if found_state is None and batch:
                    # Toffoli trick
                    for idx in remaining:
                        ns = self.tableau[idx, self.l :]
                        for j in range(target_qubit_pos):
                            if ns[j] == 1:
                                batch_elem = batch[j]
                                mismatches = np.where(
                                    self.tableau[idx] != self.tableau[batch_elem]
                                )[0]
                                if len(mismatches) and mismatches[0] != actual_qubit:
                                    diff_qubit = int(mismatches[0])
                                elif len(mismatches) > 1:
                                    diff_qubit = int(
                                        mismatches[1]
                                        if mismatches[0] == actual_qubit
                                        else mismatches[0]
                                    )
                                else:
                                    diff_qubit = None

                                if diff_qubit is not None:
                                    ctrl_val = int(self.tableau[idx, diff_qubit])
                                    self.circuit_ops.append(
                                        qp.MultiControlledX(
                                            [
                                                wires[self.l + j],
                                                wires[diff_qubit],
                                                wires[actual_qubit],
                                            ],
                                            control_values=[1, ctrl_val],
                                            work_wires=pui_work_wires[0],
                                            work_wire_type="zeroed",
                                        )
                                    )
                                    self.apply_multi_controlled_x(
                                        [self.l + j, diff_qubit],
                                        np.array([1, ctrl_val]),
                                        actual_qubit,
                                    )
                                    found_state = idx
                                    break
                        if found_state is not None:
                            break

                if found_state is None:
                    if batch:
                        self.pui(
                            k,
                            b,
                            subspace_wires,
                            nonsubspace_wires,
                            batch_qubit_positions,
                            pui_work_wires,
                        )
                        batch = []
                        batch_qubit_positions = []
                        batch_set = set()
                    continue

            # Transform found_state: zero out other non-subspace bits using CX
            ns = self.tableau[found_state, self.l :]
            nonzero_positions = np.where(ns)[0]
            for j in nonzero_positions:
                j = int(j)
                if j != target_qubit_pos:
                    _wires = [actual_qubit, self.l + j]
                    self.circuit_ops.append(qp.CNOT([wires[_wires[0]], wires[_wires[1]]]))
                    self.apply_cx_to_tableau(*_wires)

            # Transform subspace to |k>
            k_bits = math.int_to_binary(k, self.l)
            current_sub = self.tableau[found_state, : self.l]
            diff_bits = np.where(current_sub != k_bits)[0]
            for j in diff_bits:
                j = int(j)
                _wires = [actual_qubit, j]
                self.circuit_ops.append(qp.CNOT([wires[_wires[0]], wires[_wires[1]]]))
                self.apply_cx_to_tableau(*_wires)

            batch.append(found_state)
            batch_qubit_positions.append(target_qubit_pos)
            batch_set.add(found_state)
            self.bijection[found_state] = k
            k += 1

        if alloc is not None:
            self.circuit_ops.append(alloc)

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

    def __init__(self, coefficients, wires, indices, work_wires=None):
        work_wires = [] if work_wires is None else work_wires
        all_wires = Wires.all_wires([wires, work_wires])
        super().__init__(coefficients, wires=all_wires)
        self.hyperparameters["indices"] = indices
        self.hyperparameters["work_wires"] = work_wires

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


def _pui_state_prep_resources(num_entries, num_wires, **_):
    """Compute the resources for _pui_state_prep."""
    ell = max(math.ceil_log2(num_entries), 1)
    if num_entries == 1:
        return {resource_rep(qp.BasisState, num_wires=num_wires): 1}

    resources = defaultdict(int)
    resources[resource_rep(qp.MultiplexerStatePreparation, num_wires=ell)] += 1
    R = num_wires - ell
    main_pui_batch_size = 1 << int(math.floor(math.log2(max(R, 1))))
    select_rep = resource_rep(
        qp.Select,
        op_reps=[resource_rep(qp.X)] * main_pui_batch_size,
        num_control_wires=ell,
        num_work_wires=ell - 1,
        partial=False,
    )
    resources[select_rep] += num_entries // 8

    rest_pui_batch_size = num_entries % main_pui_batch_size
    select_rep = resource_rep(
        qp.Select,
        op_reps=[resource_rep(qp.X)] * rest_pui_batch_size,
        num_control_wires=ell,
        num_work_wires=ell - 1,
        partial=False,
    )
    resources[select_rep] += 1

    resources[resource_rep(qp.CNOT)] += int(num_entries**1.15 * 4)
    embed_rep = resource_rep(qp.BasisEmbedding, num_wires=ell)
    resources[embed_rep] += num_entries // 4
    swap_rep = resource_rep(qp.SWAP)
    resources[swap_rep] += num_wires

    mcx_rep = resource_rep(
        qp.MultiControlledX,
        num_control_wires=2,
        num_zero_control_values=1,
        num_work_wires=1,
        work_wire_type="clean",
    )
    resources[mcx_rep] += int(num_wires / 10) + 1

    return resources


def _pui_state_prep_work_wires(num_entries, num_wires, num_work_wires):
    # pylint: disable=unused-argument
    ell = max(math.ceil_log2(num_entries), 1)
    needed_work_wires = ell - 1
    return max(needed_work_wires - num_work_wires, 0)


@register_resources(_pui_state_prep_resources, work_wires=_pui_state_prep_work_wires)
def _pui_state_prep(coefficients, wires, indices, work_wires, **__):
    """Compute the decomposition of the partial unary iteration state preparation technique."""
    s = len(indices)
    if s == 1:
        qp.BasisState(indices[0], wires)
        return

    iso_finder = IsometryFinder(indices, len(wires))
    ops, bijection = iso_finder.find_isometry(wires, work_wires)
    ell = max(math.ceil_log2(s), 1)
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


add_decomps(PartialUnaryStatePreparation, _pui_state_prep)
