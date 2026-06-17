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
from pennylane.core.operator import Operation
from pennylane.decomposition import controlled_resource_rep
from pennylane.exceptions import DecompositionUndefinedError
from pennylane.wires import Wires


class PUIIsometryFinder:
    r"""Classical algorithm that finds the isometry circuit and bijection for
    :class:`~.PartialUnaryStatePreparation`. The goal is to compute an isometry that maps
    given computational basis states :math:`\{|\ell\rangle\}_{\ell in L}` to the first consecutive
    computational basis states :math:`\{|j\rangle\}_{0\leq j < |L|}`. The state preparation
    circuit will then prepare the amplitude :math:`c_\ell` on the computational basis state that
    :math:`|\ell\rangle` was mapped to, and then runs the isometry backwards to distribute the
    amplitudes to the states :math:`\{|\ell\rangle\}`.

    Args:
        basis_states (list[int]): Computational basis state indices :math:`L` that we want to map
            to the first :math:`|L|` consecutive basis states.
        n_qubits (int): Number of qubits on which the state needs to be prepared.

    .. details::
        :title: Theoretical background
        :href: theory

        The core idea for this isometry mapping stems from
        `Malvetti et al. (2021) <https://quantum-journal.org/papers/q-2021-03-15-412/>`__.
        To prepare the mapping, we split the overall wires (excluding dedicated work wires) into
        a subspace register of size :math:`n_{\text{subspace}}=\lceil \log_2(L)\rceil` and the
        remainder register of size :math:`n_r =n-n_{\text{subspace}}. The goal then is
        to map all states :math:`|\ell\rangle = |\ell_s\rangle \otimes |\ell_r\rangle` to some
        unique subspace state :math:`|f(\ell)\rangle = |f(\ell)\rangle \otimes |0\rangle`,
        where :math:`f` is a bijection.
        Throughout, we will switch between integers and their bit string representation as needed,
        for example in the split of :math:`\ell` into its first :math:`n_{\text{subspace}}` bits
        :math:`\ell_s` and the remaining substring :math:`\ell_r`.
        Finally, we also define a *batch size* :math:`m = 2^{\lfloor \log_2(n_r)\rfloor}`.

        The algorithm now proceeds iteratively in batches. We will denote some actions in bold,
        which are both carried out on the bit strings of all the states to be encoded (stored
        in a bit tableau), and recorded in the circuit representation for the isometry.

        0. Initialize an empty batch :math:`\mathcal{B}=\{\}` of integer pairs, a global
           bijection :math:`f` of integers, and a global counter :math:`k=0`.

        1. If :math:`j\coloneqq|\mathcal{B}| < m-1`, go to step 2, otherwise to step 4.

        2. Search for the next :math:`\ell'\in L\setminus\mathcal{B}` that has the :math:`j`\ th
           remainder bit set to one. There are three stages to this, where the latter two are
           only used if the previous ones did not succeed.

           - Search for the desired :math:`\ell'` in :math:`L\setminus \mathcal{B}`.
             If none is found move to the next stage, else continue with step 3.

           - **Swap** remainder qubit :math:`j` with a higher-index remainder qubit that is
             set to one in at least one bitstring. If none is found and :math:`j>0`, move to
             the next stage. If none is found and :math:`j=0`, move to step 4. If a possible swap
             move was found, continue with step 3, knowing that we now found a desired :math:`\ell'`.

           - Find a bitstring that has a one in one of the first :math:`j` (note we asserted
             :math:`j>0` above) remainder qubits
             (at least one such bitstring must exist) and identify a bit in which this bit string
             differs from the :math:`j`\ th bitstring in :math:`\mathcal{B}` (there is at least
             one such qubit because the bitstrings are unique).
             **Controlled** on the found remainder qubit and the differing qubit, **flip** the
             :math:`j`\ th remainder qubit. We then have
             our desired :math:`\ell'` with a one on that qubit, and continue with step 3.

           If the above stages did not find an :math:`\ell'` and
           :math:`\mathcal{B}=\{\}`, we know that all remaining bit strings lie in the subspace
           register exclusively, and go to step 5.

        3. **Controlled** on qubit :math:`j`, **flip** all other remainder qubits of the
           found :math:`\ell` and set the subspace qubits to the bit string of :math:`k`.
           Append :math:`(\ell, k)` to :math:`\mathcal{B}`, set :math:`f(\ell)=k`,
           and increment :math:`k`. Go to step 1.

        4. Else, **flip** the :math:`i`\ th remainder qubit **controlled** on the bitstring
           :math:`k_i` of the :math:`i`\ th integer pair in :math:`\mathcal{B}`.
           Note that this can be done in a batched
           manner, using `unary iteration <https://pennylane.ai/compilation/unary-iteration>`__.
           Reset the batch to :math:`\mathcal{B}=\{\}` (but not the counter :math:`k`) and
           go to step 1.

        5. For all bitstrings :math:`\ell_0` that do not have any bit in the remainder register
           set to one, set :math:`f(\ell_0)=\ell_0`, i.e. register :math:`f` to be the identity
           mapping on those states. Note that this step happens (most likely) after the bit strings
           have been manipulated repeatedly by the previous steps.

        We now have the complete bijection :math:`f` and the recording of the isometry operations
        required to map :math:`L` to the consecutive integers :math:`0\leq j<|L|`. The former
        is used to prepare the dense state :math:`|\phi_0\rangle=\sum_{\ell\in L} c_{\ell}|f(\ell)\rangle`
        on the subspace register. The latter can be executed in reverse to map the states to the
        desired :math:`|\psi\rangle = \sum_{\ell \in L } c_\ell |\ell\rangle`.

        **Why does this work?**

        We will not go into too much detail about the correctness but want to leave some comments.
        The elegant core idea of the isometry finding is that any operations we perform controlled
        on at least one remainder qubit cannot modify the bitstrings that we already mapped to the
        subspace. This allows us to successively treat the bitstrings, bringing them into the
        subspace without undoing previous work. The bit strings that have not been mapped, however,
        are being modified, and for this, it is crucial to keep track of all modifications we make.
        This class does this with its ``tableau`` attribute, which is updated for each operation
        that we perform during the mapping.
        (side note: the same logic applies for the swapping step; for already mapped bit strings
        we just swap two zeroed remainder qubits).

        The second neat part of this algorithm is the batched zeroing of remainder qubits via
        unary iteration. Due to the consecutive integers to which the states of the batch are
        mapped, unary iteration can be used out of the box, lowering the non-Clifford cost of
        the isometry circuit.
        As we already keep track of bit strings, we can reduce the cost further by using
        `partial Select circuits <https://pennylane.ai/compilation/partial-select>`__
        for this step.        Those modify the remainder qubits not
        only for the specific control states in the subspace register, but also for addition
        control states. Tracking this change in the tableau, we can take these additional
        modifications into account as a simple side effect of the unary iteration steps.

        Note that unfortunately, in our documentation and learning resources the word
        "partial" is used for this flavour of Select/unary iteration with side effects, whereas
        `Rupprecht and Wölk <https://arxiv.org/abs/2601.09388>`__ use the word to point to the
        sliced range the used unary iteration circuits.

    """

    def __init__(self, basis_states: list, n_qubits: int):
        s = len(basis_states)
        self.n = n_qubits
        self.n_subspace = max(math.ceil_log2(s), 1)
        n_r = self.n - self.n_subspace
        # Largest power of 2 less than or equal to n_r, the remainder register size
        self.m = 1 << int(math.floor(math.log2(max(n_r, 1))))

        self.tableau = qp.math.int_to_binary(basis_states, self.n).astype(np.int8)
        self.circuit = []

        # Pre-compute subspace membership (will track incrementally)
        self._in_subspace = np.all(self.tableau[:, self.n_subspace :] == 0, axis=1)

    def apply_multi_controlled_x(self, controls, control_values, target: int):
        """Apply multi-controlled X to the tableau."""
        ctrl_cols = self.tableau[:, controls]
        # A row is flipped iff all control bits match control_values
        match = np.all(ctrl_cols == control_values, axis=1)
        self.tableau[:, target] ^= match.astype(np.int8)

    def pui(self, k, batch_size):
        """Add a Partial unary iterator circuit (in form of `Select`) to the circuit ops and
        apply the corresponding multicontrolled bit flips to the tableau."""
        k_start = k - batch_size
        self.circuit.append(("PUI", k_start, k))

        # Update tableau for PUI effect
        target_bits = qp.math.int_to_binary(np.arange(k_start, k), self.n_subspace)

        # Broadcasted version of `apply_multi_controlled_x`.
        ctrl_cols = self.tableau[None, :, : self.n_subspace]
        # A row is flipped iff all control bits match control_values
        match = np.all(ctrl_cols == target_bits[:, None, :], axis=2)
        self.tableau[
            :, np.arange(self.n_subspace, len(target_bits) + self.n_subspace)
        ] ^= match.astype(np.int8).T

        # Update subspace status after PUI
        self._in_subspace = np.all(self.tableau[:, self.n_subspace :] == 0, axis=1)

    def fanout(self, control: int, bits: np.ndarray):
        """Add a Fanout operation to the circuit ops and apply corresponding CNOTs to the tableau."""
        self.circuit.append(("Fanout", control, np.delete(bits, control)))
        ctrl_bits = self.tableau[:, control].copy()
        target_bits = np.where(bits)[0]
        self.tableau[:, target_bits] ^= ctrl_bits[:, None]
        self.tableau[:, control] = ctrl_bits

    def toffoli(self, controls, control_values, target):
        """Add a MultiControlledX operation to the circuit ops and apply it to the tableau."""
        self.circuit.append(("Toffoli", controls + [target], control_values))
        self.apply_multi_controlled_x(controls, np.array(control_values), target)

    def swap(self, w0, w1):
        """Add a SWAP operation to the circuit ops and apply SWAP to the tableau."""
        self.circuit.append(("SWAP", [w0, w1]))
        self.tableau[:, [w0, w1]] = self.tableau[:, [w1, w0]]

    def _next_state_with_target_set(self, target_qubit):
        """Stage 1 of step 2.
        Find the next state in the tableau that has the remainder qubit with
        index ``target_qubit`` set to one."""
        # Translate from remainder index to total index
        actual_qubit = self.n_subspace + target_qubit

        # which rows have bit at actual_qubit set?
        search_idx = np.where(self.tableau[:, actual_qubit])[0]
        return int(search_idx[0]) if len(search_idx) else None

    def _try_swap(self, remaining, target_qubit):
        """Stage 2 of step 2
        Find a remainder qubit that is set on at least one bit string, and swap it with
        ``target_qubit``."""
        # Translate from remainder index to total index
        actual_qubit = self.n_subspace + target_qubit
        # We know that no bit string has the actual_qubit set to 1 because
        # _next_state_with_target_set failed to find it. We look for a one on a higher-index
        # qubit in the remainder register
        ns_cols = self.tableau[np.array(remaining), actual_qubit + 1 :]
        if ns_cols.size > 0:
            # Find first (row, col) with a 1
            row_idx, col_idx = np.where(ns_cols)
            if len(row_idx):
                # Find absolute qubit index from relative index within ns_cols and apply swap.
                swap_from = actual_qubit + 1 + int(col_idx[0])
                self.swap(actual_qubit, swap_from)
                # Return the index of the bitstring that had the qubit set to 1, and now has
                # actual_qubit set to one due to the swap.
                return int(remaining[row_idx[0]])

        return None

    def _try_toffoli(self, remaining, target_qubit, batch):
        """Stage 3 of step 2.
        Find a bitstring that has one of the remainder qubits set to one that are already
        used in ``batch`` but differs on a different qubit. Then apply the Toffoli trick
        to flip the ``target_qubit`` of the found bitstring. We know that the latter must be 0,
        otherwise _next_state_with_target_set would have identified the bit string as next
        candidate in stage 1 of this step already."""
        actual_qubit = self.n_subspace + target_qubit

        # Take the next-best bitstring that we still need to map
        idx = remaining[0]
        # Find a remainder qubit lower than actual_qubit that is set to one. We know that
        # idx must have such a qubit because
        # - it must have at least one remainder qubit set to one because it is in `remaining`
        # - it can't have any qubits at or above the index `actual_qubit` set, because if it did,
        #   _next_state_with_target_set or _try_swap would have been successful.
        active_remainder_bit = np.where(self.tableau[idx, self.n_subspace : actual_qubit])[0][0]
        # Find a qubit at which the bitstrings A at position batch[active_remainder_bit] and
        # B at position `idx` differ. Note that we know that there is a differing qubit because
        # the bitstrings are unique. Also, note that actual_qubit can't be a differing qubit,
        # because A only has ones in the subspace and at position
        # active_remainder_bit < actual_qubit, and B does not have actual_qubit set because
        # in that case, _next_state_with_target_set would have found it already.
        A, B = self.tableau[idx], self.tableau[batch[active_remainder_bit]]
        diff_qubit = int(np.where(A != B)[0][0])

        controls = [self.n_subspace + active_remainder_bit, diff_qubit]
        control_values = [1, int(self.tableau[idx, diff_qubit])]
        self.toffoli(controls, control_values, actual_qubit)
        return idx

    def _map_full_state(self, found_state, k, target_qubit):
        """Execute step 3 of the algorithm, zeroing all remainder qubits controlled on
        ``target_qubit`` (except for ``target_qubit`` itself) and setting the subspace qubits
        to the integer ``k``."""
        actual_qubit = self.n_subspace + target_qubit
        k_bits = qp.math.int_to_binary(k, self.n_subspace)
        target_state = qp.math.concatenate([k_bits, np.zeros(self.n - self.n_subspace, dtype=int)])
        current_state = self.tableau[found_state]
        diff = np.bitwise_xor(target_state, current_state)
        self.fanout(actual_qubit, diff)

    def find_isometry(self):
        """Main method to find the isometry. See main docstring for a detailed description."""
        bijection = {}  # Bijection f between desired states and consecutive basis states
        batch = []

        k = 0
        while True:
            b = len(batch)

            # Get remaining indices not in subspace and not in batch
            remaining = np.where(~self._in_subspace)[0]
            remaining = [int(i) for i in remaining if i not in batch]

            if b == self.m or (not remaining and b > 0):
                # Step 4
                # Need to flush because the batch is full, or because there are no remaining
                # bit strings to map but the batch still has some entries.
                self.pui(k, len(batch))
                batch = []
                continue

            if not remaining:
                # We know that b == 0 because (not remaining and b>0) is caught above
                # Hence, the batch is empty and there are not states remaining to be handled.
                # We thus are done and exit the loop.
                break

            # Step 2, three stages:
            # Stage 1: Find a state with the b-th non-subspace qubit set to 1
            target_qubit = b
            found_state = self._next_state_with_target_set(target_qubit)

            if found_state is None:
                # Stage 2: Try swapping with a qubit further right to get a state with b-th
                # non-subspace qubit set to 1
                found_state = self._try_swap(remaining, target_qubit)

            if found_state is None:
                # If we still have to map at least one bit string, the batch is currently empty,
                # and neither _next_state_with_target_set nor _try_swap were successful, we know
                # that _try_toffoli can't work (because it needs a reference batch state to
                # discriminate against). Thus, we would be stuck in an infinite loop of trying the
                # three stages of step 2.
                assert (
                    b > 0
                ), "This scenario should never happen because it would lead to infinite recursion."
                # Stage 3: Use Toffoli trick to fabricate a suitable state instead
                found_state = self._try_toffoli(remaining, target_qubit, batch)

            # Step 3
            #  - Transform found_state: zero out other non-subspace bits using CX
            #    Then transform subspace to |k>
            self._map_full_state(found_state, k, target_qubit)

            #  - Append the found state to the batch and register it in the state bijection
            batch.append(found_state)
            bijection[found_state] = k
            k += 1

        # Step 5: Assign bijection for states already in subspace
        vals = self.tableau[:, : self.n_subspace] @ (2 ** np.arange(self.n_subspace - 1, -1, -1))
        for i, val in enumerate(vals):
            # setdefault means that no states are overwritten. We only did isometric steps,
            # so we exactly set all other states that we did not take care of yet with this.
            bijection.setdefault(i, int(val))

        return self.circuit, bijection


class PartialUnaryStatePreparation(Operation):
    r"""Prepare a sparse quantum state with the partial unary iteration technique.

    This operation prepares an arbitrary state

    .. math:: |\psi\rangle = \sum_{\ell \in L } c_\ell |\ell\rangle,

    where :math:`L` denotes the set of ``indices`` and :math:`c_\ell` is the ``coefficient``
    corresponding to the index :math:`\ell\in L`.
    The state :math:`|\ell\rangle` is a computational basis state, interpreted via the
    binary representation of :math:`\ell`.

    This state preparation technique was introduced in
    `Rupprecht and Wölk, arXiv:2601.09388 <https://arxiv.org/abs/2601.09388>`__. It consists
    of a dense state preparation of the amplitudes on a subspace register, and an isometric mapping
    that permutes the prepared amplitudes into the correct positions for the target state.
    As such, this method is tailored to sparse states. However, it approaches
    :class:`~.MultiplexerStatePreparation` for less sparse states, effectively providing an
    interpolation between sparse and dense state preparation.

    .. seealso::

        :class:`~.PUIIsometryFinder` for the classical algorithm that finds the isometry for
        the circuit, as well as :class:`~.SumOfSlatersPrep` for another sparse state
        preparation technique.

    Args:
        coefficients (np.ndarray): Coefficients of the sparse state to prepare. The ordering should
            match that in ``indices``.
        wires (qp.wires.WiresLike): Wires on which to prepare the state. All work wires will be
            allocated dynamically with :func:`~.allocate`.
        indices (tuple[int]): Indices of the sparse state to prepare. The ordering should match
            that in ``coefficients``.
        work_wires (qp.wires.WiresLike): Work wires used for the state preparation. For
            :math:`L` entries in the state, :math:`\max(\lceil \log_2(L)\rceil-1, 1)` work wires
            are needed.

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

    >>> where = np.where(np.round(prepared_state, 3))[0]
    >>> print(tuple(where)==indices)
    True
    >>> print(np.allclose(prepared_state[where], coefficients))
    True

    The preparation circuit looks like this:

    >>> print(qp.draw(qp.decompose(circuit, max_expansion=1), max_length=200, show_matrices=False)())
    0: ─╭MultiplexerStatePreparation(M0)─╭|Ψ⟩─╭QROM(M1)─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭QROM(M1)─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭QROM(M1)─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭QROM(M1)─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭QROM(M1)─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─┤ ╭State
    1: ─├MultiplexerStatePreparation(M0)─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─┤ ├State
    2: ─├MultiplexerStatePreparation(M0)─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├QROM(M1)─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─┤ ├State
    3: ─╰MultiplexerStatePreparation(M0)─╰|Ψ⟩─├QROM(M1)─╰|Ψ⟩─├|Ψ⟩─├|Ψ⟩─╰|Ψ⟩─├QROM(M1)─╰|Ψ⟩─├|Ψ⟩─├|Ψ⟩─╰|Ψ⟩─├QROM(M1)─╰|Ψ⟩─├|Ψ⟩─├|Ψ⟩─╰|Ψ⟩─├QROM(M1)─╰|Ψ⟩─├|Ψ⟩─├|Ψ⟩─╰|Ψ⟩─├QROM(M1)─╰|Ψ⟩─├|Ψ⟩─├|Ψ⟩─┤ ├State
    4: ───────────────────────────────────────├QROM(M1)──────├|Ψ⟩─├●────────├QROM(M1)──────├|Ψ⟩─├●────────├QROM(M1)──────├|Ψ⟩─├●────────├QROM(M1)──────├|Ψ⟩─├●────────├QROM(M1)──────├|Ψ⟩─├●───┤ ├State
    5: ───────────────────────────────────────├QROM(M1)──────╰●───╰|Ψ⟩──────├QROM(M1)──────╰●───╰|Ψ⟩──────├QROM(M1)──────╰●───╰|Ψ⟩──────├QROM(M1)──────╰●───╰|Ψ⟩──────├QROM(M1)──────╰●───╰|Ψ⟩─┤ ├State
    6: ───────────────────────────────────────├QROM(M1)─────────────────────├QROM(M1)─────────────────────├QROM(M1)─────────────────────├QROM(M1)─────────────────────├QROM(M1)────────────────┤ ├State
    7: ───────────────────────────────────────├QROM(M1)─────────────────────├QROM(M1)─────────────────────├QROM(M1)─────────────────────├QROM(M1)─────────────────────├QROM(M1)────────────────┤ ├State
    8: ───────────────────────────────────────╰QROM(M1)─────────────────────╰QROM(M1)─────────────────────╰QROM(M1)─────────────────────╰QROM(M1)─────────────────────╰QROM(M1)────────────────┤ ╰State

    We can make out the dense state preparation on the subspace wires ``0`` through ``3``, followed
    by the isometry circuit consisting of partial unary iteration circuits and ``CNOT`` blocks.
    The basis state preparations denoted as ``|Ψ⟩`` are layers of simple ``PauliX`` bit flips
    that allow us to shift the range of the partial unary iterations to start from ``0``, and
    thus to reuse generic :class:`~.Select` subroutines, independent of the actual iteration range.

    For more complex examples, the circuit may also contain :class:`~.SWAP` and/or Toffoli
    gates (in the form of :class:`~.MultiControlledX`):

    .. code-block:: python

        np.random.seed(31)
        L = 2553
        coefficients = np.random.random(L)
        coefficients /= np.linalg.norm(coefficients)
        indices = np.random.choice(2**15, L, replace=False)
        wires = list(range(15))
        num_work_wires = qp.math.ceil_log2(L) - 1
        work_wires = list(range(15, 15 + num_work_wires))

    >>> print(qp.specs(qp.decompose(circuit, max_expansion=1), compute_depth=False)()["resources"])
    Wire allocations: 26
    Total gates: 6,040
    Gate counts:
    - MultiplexerStatePreparation: 1
    - BasisState: 2,414
    - QROM: 1,207
    - C(BasisState): 2,414
    - MultiControlledX: 3
    - SWAP: 1
    Measurements:
    - state(all wires): 1
    Depth: Not computed

    Note that passing more work wires than the needed :math:`\max(\lceil \log_2(L)\rceil-1, 1)`
    makes the isometry circuit of the state preparation cheaper:

    >>> new_num_work_wires = 3*num_work_wires
    >>> work_wires = list(range(15, 15 + new_num_work_wires))
    >>> print(qp.specs(qp.decompose(circuit, max_expansion=1), compute_depth=False)()["resources"])
    Wire allocations: 48
    Total gates: 3,056
    Gate counts:
    - MultiplexerStatePreparation: 1
    - BasisState: 320
    - QROM: 160
    - C(BasisState): 2,553
    - MultiControlledX: 6
    - SWAP: 16
    Measurements:
    - state(all wires): 1
    Depth: Not computed

    We used just ``160`` ``QROM``\ s instead of ``1207``, and as their size is dictated only by the
    number of indices :math:`L`, it is the same between the two decompositions.

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
        s = len(indices)
        if len(set(indices)) != s:
            raise ValueError("The state indices must be unique.")
        if len(coefficients) != s:
            raise ValueError(
                "The number of coefficients and the number of state indices must match."
            )
        if max(indices) > 2 ** len(wires) - 1:
            raise ValueError(
                f"The state indices must be smaller than {2**len(wires)=}. Largest index is {max(indices)}"
            )

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
    """Compute the resources for _pui_state_prep, the partial unary iteration state prep.
    These resource counts are numerically obtained heuristics, extended to guarantee all
    resource reps that may appear are included at least once."""
    if num_entries == 1:
        return {qp.resource_rep(qp.BasisState, num_wires=num_wires): 1}

    n_subspace = max(math.ceil_log2(num_entries), 1)
    resources = defaultdict(int)
    if num_work_wires < max(n_subspace - 1, 1):
        resources[qp.resource_rep(qp.allocation.Allocate)] += 1
        resources[qp.resource_rep(qp.allocation.Deallocate)] += 1

    num_work_wires = max(num_work_wires, n_subspace - 1, 1)
    resources[qp.resource_rep(qp.MultiplexerStatePreparation, num_wires=n_subspace)] += 1

    R = num_wires - n_subspace
    main_pui_batch_size = 1 << int(math.floor(math.log2(max(R, 1))))

    qrom_reps = {
        p: qp.resource_rep(
            qp.QROM,
            num_bitstrings=p,
            num_control_wires=n_subspace,
            num_target_wires=p,
            num_work_wires=n_subspace - 1,
            clean=True,
        )
        for p in range(1, main_pui_batch_size + 1)
    }

    resources[qrom_reps[main_pui_batch_size]] += max(num_entries // main_pui_batch_size, 1)
    for p in range(1, main_pui_batch_size):
        resources[qrom_reps[p]] += 1

    resources[
        controlled_resource_rep(qp.BasisState, {"num_wires": num_wires - 1}, num_control_wires=1)
    ] += num_entries

    embed_rep = qp.resource_rep(qp.BasisState, num_wires=n_subspace)
    resources[embed_rep] += 2 * (num_entries // main_pui_batch_size + 1)

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

    n_subspace = max(math.ceil_log2(s), 1)
    needed_work_wires = max(n_subspace - 1, 1)  # Need n_subspace-1 for QROM, 1 for Toffoli
    wires = Wires(wires)
    if len(work_wires) > needed_work_wires:
        # There is an easy way to use more work wires: Pretend they are part of the system wires,
        # which gives the isometry circuit more space to move states around. Adding the excess
        # work wires to the beginning of the `wires` register allows us to keep `indices`
        # unchanged. It is not obvious whether this approach or a different wire ordering will
        # be cheaper in terms of quantum resources used in the isometry circuit.
        wires = Wires(work_wires[needed_work_wires:]) + wires

    iso_finder = PUIIsometryFinder(np.array(indices), len(wires))
    circuit, bijection = iso_finder.find_isometry()

    subspace_wires = Wires(wires[:n_subspace])
    nonsubspace_wires = Wires(wires[n_subspace:])

    # Step 1: Dense state preparation
    dense_state = np.zeros(2**n_subspace, dtype=complex)
    ids = np.array([bijection[i] for i in range(len(coefficients))])
    dense_state[ids] = coefficients
    qp.MultiplexerStatePreparation(dense_state, subspace_wires)

    # Step 2: Apply the inverse of the isometry circuit
    for _type, *data in reversed(circuit):
        if _type == "PUI":
            k_start, k = data
            qp.BasisState(k_start, subspace_wires)
            b = k - k_start
            qp.QROM(np.eye(b), subspace_wires, nonsubspace_wires[:b], work_wires[: n_subspace - 1])
            qp.BasisState(k_start, subspace_wires)
            continue
        if _type == "Fanout":
            control, bits = data
            qp.ctrl(qp.BasisState(bits, wires[:control] + wires[control + 1 :]), wires[control])
            continue

        ids = data[0]
        _wires = [wires[idx] for idx in ids]
        if _type == "SWAP":
            qp.SWAP(_wires)
        elif _type == "Toffoli":
            qp.MultiControlledX(_wires, data[1], work_wires=work_wires[0], work_wire_type="zeroed")
        else:
            raise NotImplementedError  # pragma: no cover


# Decomposition rule with statically given work_wires to PartialUnaryStatePreparation


def _pui_state_prep_provided_work_wires_condition(num_entries, num_wires, num_work_wires):
    # pylint: disable=unused-argument
    if num_entries == 1:
        return True
    return num_work_wires >= max(math.ceil_log2(num_entries) - 1, 1)


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
    if num_entries == 1:
        return False  # Just use _pui_state_prep_provided_work_wires, we don't need work wires
    return num_work_wires < max(math.ceil_log2(num_entries) - 1, 1)


@qp.register_condition(_pui_state_prep_dyn_work_wires_condition)
@qp.register_resources(
    _pui_state_prep_resources, work_wires=_pui_state_prep_work_wires, exact=False
)
def _pui_state_prep_dyn_work_wires(coefficients, wires, indices, **__):
    """Compute the decomposition of the partial unary iteration state preparation technique.
    This decomposition dynamically allocates work wires. If PartialUnaryStatePreparation
    has work_wires but too few of them, they will **not** be used here."""
    # The case num_entries=1 is excluded via _pui_state_prep_dyn_work_wires_condition, so
    # we know that we want to allocate at least one work wire.
    need_to_allocate = max(math.ceil_log2(len(indices)) - 1, 1)
    with allocate(need_to_allocate, state="zero", restored=True) as work_wires:
        _pui_state_prep_core(coefficients, wires, indices, work_wires)


qp.add_decomps(
    PartialUnaryStatePreparation,
    _pui_state_prep_dyn_work_wires,
    _pui_state_prep_provided_work_wires,
)
