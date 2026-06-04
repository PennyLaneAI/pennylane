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
from pennylane.decomposition import controlled_resource_rep
from pennylane.exceptions import DecompositionUndefinedError
from pennylane.operation import Operation
from pennylane.wires import Wires


class IsometryFinder:
    """Classical algorithm that finds the isometry circuit and bijection for
    PartialUnaryStatePreparation."""

    def __init__(self, basis_states: list, n_qubits: int):
        self.n = n_qubits
        self.l = max(math.ceil_log2(len(basis_states)), 1)
        r = self.n - self.l
        self.m = 1 << int(math.floor(math.log2(max(r, 1))))

        self.tableau = qp.math.int_to_binary(basis_states, self.n)
        # self.circuit_ops = []
        self.circuit = []

        # Pre-compute subspace membership (will track incrementally)
        self._in_subspace = np.all(self.tableau[:, self.l :] == 0, axis=1)

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

    def pui(self, k, batch_size):
        """Add a Partial unary iterator circuit (in form of `Select`) to the circuit ops and
        apply the corresponding multicontrolled bit flips to the tableau."""
        k_start = k - batch_size
        self.circuit.append(("PUI", k_start, k))

        # Update tableau for PUI effect
        target_bits = qp.math.int_to_binary(np.arange(k_start, k), self.l)
        for j, _bits in enumerate(target_bits):
            self.apply_multi_controlled_x(slice(0, self.l), _bits, self.l + j)

        # Update subspace status after PUI
        self._in_subspace = np.all(self.tableau[:, self.l :] == 0, axis=1)

    def fanout(self, control: int, bits: np.ndarray):
        """Add a CNOT operation to the circuit ops and apply CNOT to the tableau."""
        self.circuit.append(("Fanout", control, bits))
        ctrl_bits = self.tableau[:, control]
        for i, bit in enumerate(bits[:control]):
            if bit:
                self.tableau[:, i] ^= ctrl_bits
        for i, bit in enumerate(bits[control:], start=control + 1):
            if bit:
                self.tableau[:, i] ^= ctrl_bits

    def mcx(self, controls, control_values, target):
        """Add a MultiControlledX operation to the circuit ops and apply it to the tableau."""
        self.circuit.append(("Toffoli", controls + [target], control_values))
        self.apply_multi_controlled_x(controls, control_values, target)

    def swap(self, w0, w1):
        """Add a SWAP operation to the circuit ops and apply SWAP to the tableau."""
        self.circuit.append(("SWAP", [w0, w1]))
        self.tableau[:, [w0, w1]] = self.tableau[:, [w1, w0]]

    def _next_state_with_target_set(self, target_qubit):
        actual_qubit = self.l + target_qubit

        # which rows have bit at actual_qubit set?
        col = self.tableau[:, actual_qubit]
        search_idx = np.where(col)[0]
        return int(search_idx[0]) if len(search_idx) else None

    def _try_swap(self, remaining, target_qubit):
        ns_cols = self.tableau[np.array(remaining), self.l + target_qubit + 1 : self.n]
        if ns_cols.size > 0:
            # Find first (row, col) with a 1
            row_idx, col_idx = np.where(ns_cols)
            if len(row_idx):
                swap_from = self.l + target_qubit + 1 + int(col_idx[0])
                self.swap(self.l + target_qubit, swap_from)
                return int(remaining[row_idx[0]])

        return None

    def _get_diff_qubit(self, idx, j, actual_qubit, batch):
        mismatches = np.where(self.tableau[idx] != self.tableau[batch[j]])[0]
        if len(mismatches) and mismatches[0] != actual_qubit:
            return int(mismatches[0])
        if len(mismatches) > 1:
            return int(mismatches[1] if mismatches[0] == actual_qubit else mismatches[0])
        return None

    def _toffoli_trick(self, remaining, target_qubit, batch):
        actual_qubit = self.l + target_qubit

        for idx in remaining:
            ns = self.tableau[idx, self.l :]
            for j in range(target_qubit):
                if ns[j] == 1:
                    diff_qubit = self._get_diff_qubit(idx, j, actual_qubit, batch)

                    if diff_qubit is None:
                        continue

                    controls = [self.l + j, diff_qubit]
                    control_values = [1, int(self.tableau[idx, diff_qubit])]
                    self.mcx(controls, control_values, actual_qubit)
                    return idx

        return None

    def _map_full_state(self, found_state, k, target_qubit):
        actual_qubit = self.l + target_qubit
        k_bits = qp.math.int_to_binary(k, self.l)
        target_state = qp.math.concatenate([k_bits, np.zeros(self.n - self.l, dtype=int)])
        current_state = self.tableau[found_state]
        diff = np.delete(np.bitwise_xor(target_state, current_state), actual_qubit)
        self.fanout(actual_qubit, diff)

    def find_isometry(self):
        """
        Find isometry using the batched PUI approach (Algorithm 1).
        """
        bijection = {}
        batch = []

        k = 0
        while True:
            b = len(batch)

            # Get remaining indices not in subspace and not in batch
            remaining = np.where(~self._in_subspace)[0]
            remaining = [int(i) for i in remaining if i not in batch]

            if b == self.m or (not remaining and b > 0):
                self.pui(k, len(batch))
                batch = []
                continue

            if not remaining:
                # We know that b == 0 because (not remaining and b>0) is caught above
                # Hence, the batch is empty and there are not states remaining to be handled.
                # We thus are done and exit the loop.
                break

            # Find a state with the b-th non-subspace qubit set to 1
            target_qubit = b
            found_state = self._next_state_with_target_set(target_qubit)

            if found_state is None:
                # Try swapping with a qubit further right to get a state with b-th
                # non-subspace qubit set to 1
                found_state = self._try_swap(remaining, target_qubit)

            if found_state is None and b > 0:
                # Use Toffoli trick to find a state
                found_state = self._toffoli_trick(remaining, target_qubit, batch)

            if found_state is None:
                # If really no state could be found, flush the PUI and start a new batch.
                if b > 0:
                    self.pui(k, len(batch))
                    batch = []
                continue

            # Transform found_state: zero out other non-subspace bits using CX
            # Then transform subspace to |k>
            self._map_full_state(found_state, k, target_qubit)

            # Append the found state to the batch and register it in the state bijection
            batch.append(found_state)
            bijection[found_state] = k
            k += 1

        # Assign bijection for states already in subspace
        vals = self.tableau[:, : self.l] @ (2 ** np.arange(self.l - 1, -1, -1))
        for i, val in enumerate(vals):
            bijection.setdefault(i, int(val))

        return self.circuit, bijection


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

    Consider a sparse state, specified in terms of its ``coefficients``, or amplitudes, and the
    corresponding computational basis state ``indices``:

    .. code-block:: python

        import pennylane as qp
        import numpy as np
        qp.decomposition.enable_graph()

        coefficients = np.array([0.25, 0.25j, -0.25, 0.5, 0.5, 0.25, -0.25j, 0.25, -0.25, 0.25])
        indices = (0, 1, 4, 13, 14, 17, 19, 22, 23, 25)

    Let's prepare this state on a six-qubit register, using three auxiliary qubits:


    .. code-block:: python

        wires = list(range(6))
        work_wires = list(range(6, 9))

        dev = qp.device("lightning.qubit", wires=9)

        @qp.qnode(dev)
        def circuit():
            qp.PartialUnaryStatePreparation(coefficients, wires, indices, work_wires)
            return qp.state()

        prepared_state = circuit()[::8] # Slice out three work wires

    We can check that the correct basis states are populated with the correct amplitudes:

    >>> where = np.where(prepared_state)[0]
    >>> print(tuple(where)==indices)
    True
    >>> print(np.allclose(prepared_state[where], coefficients))
    True

    The preparation circuit looks like this:

    >>> print(qp.draw(qp.decompose(circuit, max_expansion=1), max_length=200, show_matrices=False)())
    0: ─╭MultiplexerStatePreparation(M0)─╭|Ψ⟩─╭QROM(M1)─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭QROM(M1)─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭QROM(M1)─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭QROM(M1)─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭QROM(M1)─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─┤ ╭State
    1: ─├MultiplexerStatePreparation(M0)─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─┤ ├State
    2: ─├MultiplexerStatePreparation(M0)─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─┤ ├State
    3: ─╰MultiplexerStatePreparation(M0)─╰|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─╰|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─╰|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─╰|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─╰|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─╰|Ψ⟩─┤ ├State
    4: ───────────────────────────────────────├QROM(M1)─├|Ψ⟩─├●────────├QROM(M1)─├|Ψ⟩─├●────────├QROM(M1)─├|Ψ⟩─├●────────├QROM(M1)─├|Ψ⟩─├●────────├QROM(M1)─├|Ψ⟩─├●────────┤ ├State
    5: ───────────────────────────────────────├QROM(M1)─╰●───╰|Ψ⟩──────├QROM(M1)─╰●───╰|Ψ⟩──────├QROM(M1)─╰●───╰|Ψ⟩──────├QROM(M1)─╰●───╰|Ψ⟩──────├QROM(M1)─╰●───╰|Ψ⟩──────┤ ├State
    6: ───────────────────────────────────────├QROM(M1)────────────────├QROM(M1)────────────────├QROM(M1)────────────────├QROM(M1)────────────────├QROM(M1)────────────────┤ ├State
    7: ───────────────────────────────────────├QROM(M1)────────────────├QROM(M1)────────────────├QROM(M1)────────────────├QROM(M1)────────────────├QROM(M1)────────────────┤ ├State
    8: ───────────────────────────────────────╰QROM(M1)────────────────╰QROM(M1)────────────────╰QROM(M1)────────────────╰QROM(M1)────────────────╰QROM(M1)────────────────┤ ╰State

    We can make out the dense state preparation on the subspace wires ``0`` through ``3``, followed
    by the isometry circuit consisting of partial unary iteration circuits and ``CNOT`` blocks.
    The basis state preparations denoted as ``|Ψ⟩`` are layers of simple ``PauliX`` bit flips
    that allow us to shift the range of the partial unary iterations to start from ``0``, and
    thus to reuse generic :class:`~.Select` subroutines, independent of the actual iteration range.

    For more complex examples, the circuit may also contain :class:`~.SWAP` and/or Toffoli
    gates (in the form of :class:`~.MultiControlledX`):

    .. code-block:: python

        np.random.seed(31)
        K = 2553
        coefficients = np.random.random(K)
        coefficients /= np.linalg.norm(coefficients)
        indices = np.random.choice(2**15, K, replace=False)
        wires = list(range(15))
        num_work_wires = qp.math.ceil_log2(K) - 1
        work_wires = list(range(15, 15 + num_work_wires))

    >>> print(qp.specs(qp.decompose(circuit, max_expansion=1), compute_depth=False)()["resources"])
    Wire allocations: 26
    Total gates: 4834
    Gate counts:
    - MultiplexerStatePreparation: 1
    - BasisState: 1208
    - QROM: 1207
    - C(BasisState): 2414
    - MultiControlledX: 3
    - SWAP: 1
    Measurements:
    - state(all wires): 1
    Depth: Not computed

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
    """Compute the resources for _pui_state_prep.
    These resource counts are numerically obtained heuristics, extended to guarantee all
    resource reps that may appear are included at least once."""
    if num_entries == 1:
        return {qp.resource_rep(qp.BasisState, num_wires=num_wires): 1}

    ell = max(math.ceil_log2(num_entries), 1)
    resources = defaultdict(int)
    if num_work_wires < max(ell - 1, 1):
        resources[qp.resource_rep(qp.allocation.Allocate)] += 1
        resources[qp.resource_rep(qp.allocation.Deallocate)] += 1

    num_work_wires = max(num_work_wires, ell - 1, 1)
    resources[qp.resource_rep(qp.MultiplexerStatePreparation, num_wires=ell)] += 1

    R = num_wires - ell
    main_pui_batch_size = 1 << int(math.floor(math.log2(max(R, 1))))
    rest_pui_batch_size = num_entries % main_pui_batch_size or main_pui_batch_size

    qrom_rep = qp.resource_rep(
        qp.QROM,
        num_bitstrings=main_pui_batch_size,
        num_control_wires=ell,
        num_target_wires=main_pui_batch_size,
        num_work_wires=num_work_wires,
        clean=True,
    )
    resources[qrom_rep] += max(num_entries // main_pui_batch_size, 1)

    last_qrom_rep = qp.resource_rep(
        qp.QROM,
        num_bitstrings=rest_pui_batch_size,
        num_control_wires=ell,
        num_target_wires=rest_pui_batch_size,
        num_work_wires=num_work_wires,
        clean=True,
    )
    resources[last_qrom_rep] += 1

    resources[
        controlled_resource_rep(
            qp.BasisState, base_params={"num_wires": num_wires - 1}, num_control_wires=1
        )
    ] += num_entries

    embed_rep = qp.resource_rep(qp.BasisState, num_wires=ell)
    resources[embed_rep] += num_entries // main_pui_batch_size + 2

    swap_rep = qp.resource_rep(qp.SWAP)
    resources[swap_rep] += num_wires

    num_toffolis = int(num_wires / 10) + 1
    toffoli_params = {"num_control_wires": 2, "num_work_wires": 1, "work_wire_type": "zeroed"}
    mcx_rep_0 = qp.resource_rep(qp.MultiControlledX, num_zero_control_values=0, **toffoli_params)
    resources[mcx_rep_0] += max(num_toffolis // 2, 1)
    mcx_rep_1 = qp.resource_rep(qp.MultiControlledX, num_zero_control_values=1, **toffoli_params)
    resources[mcx_rep_1] += max(num_toffolis - num_toffolis // 2, 1)

    return resources


def _pui_state_prep_core(coefficients, wires, indices, work_wires):
    """Compute the decomposition of the partial unary iteration state preparation technique.
    This core method is used by the two rules below, which only differ by the work
    wire management."""
    s = len(indices)
    if s == 1:
        qp.BasisState(indices[0], wires)
        return

    ell = max(math.ceil_log2(s), 1)
    iso_finder = IsometryFinder(np.array(indices), len(wires))
    circuit, bijection = iso_finder.find_isometry()

    subspace_wires = Wires(wires[:ell])
    nonsubspace_wires = Wires(wires[ell:])

    # Step 1: Dense state preparation
    dense_state = np.zeros(2**ell, dtype=complex)
    for i, amp in enumerate(coefficients):
        dense_state[bijection[i]] = amp

    qp.MultiplexerStatePreparation(dense_state, subspace_wires)

    wires = Wires(wires)

    last_pui_k = 0
    # Step 2: Apply the inverse of the isometry circuit
    for _type, *data in reversed(circuit):
        if _type == "PUI":
            k_start, k = data
            qp.BasisState(np.bitwise_xor(k_start, last_pui_k), subspace_wires)
            last_pui_k = k_start
            b = k - k_start
            qp.QROM(np.eye(b), subspace_wires, nonsubspace_wires[:b], work_wires)
            continue
        if _type == "Fanout":
            control, bits = data
            _wires = wires[:control] + wires[control + 1 :]
            qp.ctrl(qp.BasisState(bits, _wires), control)
            continue

        ids = data[0]
        _wires = [wires[idx] for idx in ids]
        if _type == "SWAP":
            qp.SWAP(_wires)
        elif _type == "Toffoli":
            qp.MultiControlledX(_wires, data[1], work_wires=work_wires[0], work_wire_type="zeroed")
        else:
            raise NotImplementedError

    qp.BasisState(last_pui_k, subspace_wires)


# Decomposition rule with statically given work_wires to PartialUnaryStatePreparation


def _pui_state_prep_provided_work_wires_condition(num_entries, num_wires, num_work_wires):
    # pylint: disable=unused-argument
    ell = max(math.ceil_log2(num_entries), 1)
    needed_work_wires = max(ell - 1, 1)
    return num_work_wires >= needed_work_wires


@qp.register_condition(_pui_state_prep_provided_work_wires_condition)
@qp.register_resources(_pui_state_prep_resources, exact=False)
def _pui_state_prep_provided_work_wires(coefficients, wires, indices, work_wires, **__):
    """Compute the decomposition of the partial unary iteration state preparation technique.
    Uses the work_wires given to PartialUnaryStatePreparation as an argument."""
    _pui_state_prep_core(coefficients, wires, indices, work_wires)


# Decomposition rule with dynamic work wire allocation


def _pui_state_prep_work_wires(num_entries, num_wires, num_work_wires):
    # pylint: disable=unused-argument
    return {"zeroed": max(math.ceil_log2(num_entries) - 1, 1)}


def _pui_state_prep_dyn_work_wires_condition(num_entries, num_wires, num_work_wires):
    # pylint: disable=unused-argument
    return num_work_wires < max(math.ceil_log2(num_entries) - 1, 1)


@qp.register_condition(_pui_state_prep_dyn_work_wires_condition)
@qp.register_resources(
    _pui_state_prep_resources, work_wires=_pui_state_prep_work_wires, exact=False
)
def _pui_state_prep_dyn_work_wires(coefficients, wires, indices, **__):
    """Compute the decomposition of the partial unary iteration state preparation technique.
    This decomposition dynamically allocates work wires. If PartialUnaryStatePreparation
    has work_wires but too few of them, they will **not** be used here."""
    need_to_allocate = max(math.ceil_log2(len(indices)) - 1, 1)
    with allocate(need_to_allocate, state="zero", restored=True) as work_wires:
        _pui_state_prep_core(coefficients, wires, indices, work_wires)


qp.add_decomps(
    PartialUnaryStatePreparation,
    _pui_state_prep_dyn_work_wires,
    _pui_state_prep_provided_work_wires,
)
