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
from pennylane.typing import Wire
from pennylane.wires import Wires

_U64 = np.uint64


# pylint: disable-next=too-many-instance-attributes
class PUIsometryFinder:
    r"""Classical algorithm that finds the isometry circuit and bijection for
    :class:`~.PartialUnaryStatePreparation`. The goal is to compute an isometry that maps
    given computational basis states :math:`\{|\ell\rangle\}_{\ell \in L}` to the first consecutive
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
        a *subspace register* of size :math:`n_{\text{subspace}}=\lceil \log_2(|L|)\rceil` and the
        *remainder register* of size :math:`n_r =n-n_{\text{subspace}}`. The goal then is
        to map all states :math:`|\ell\rangle = |\ell_s\rangle \otimes |\ell_r\rangle` to some
        unique subspace state :math:`|f(\ell)\rangle = |f(\ell)\rangle \otimes |0\rangle`,
        where :math:`f` is a bijection.
        The algorithm works because it successively maps the states to the desired subspace, and
        does so exclusively with operations that leave the subspace itself untouched, so that all
        states that have been handled remain static when handling the next ones.
        Throughout, we will switch between integers and their bit string representation as needed,
        for example in the split of :math:`\ell` into its first :math:`n_{\text{subspace}}` bits
        :math:`\ell_s` and the remaining substring :math:`\ell_r`.
        Finally, we also define a *batch size* :math:`m = 2^{\lfloor \log_2(n_r)\rfloor}`.

        **Algorithm description**

        The algorithm now proceeds iteratively in batches. We will denote some actions in bold,
        which are both carried out on the bit strings of all the states to be encoded, and
        recorded in the circuit representation for the isometry.
        The former is stored in a packed bit tableau, representing each row in a single
        ``np.uint64`` (for :math:`n\leq 63` qubits) or a large Python integer (for :math:`n>63`).

        0. Initialize an empty batch :math:`\mathcal{B}=\{\}` of integer pairs, a global
           bijection :math:`f` of integers, and a global counter :math:`k=0`.

        1. If :math:`j:=|\mathcal{B}| < m-1`, go to step 2, otherwise to step 4.

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

        **Space-time tradeoff**

        The isometry finder can naturally make use of a space-time tradeoff; to understand this,
        note that the batches passed to the partial unary iterators are limited by the maximal size
        :math:`m` inferred from the number of remainder qubits. As the subspace register size is
        exclusively determined by :math:`|L|`, adding work wires (or more system wires) directly
        adds them to the remainder register, thus increasing :math:`m`.

        Overall, the isometry circuit needs to iterate over the entire range :math:`[0,|L|)`,
        but the smaller the (maximal) batch size, the more iterators are needed, each of which
        comes with some overhead for the control structure that selects the subspace on which
        it acts. Increasing the batch size therefore allows for fewer iterators, for which
        we thus pay less overhead.

    """

    def __init__(self, basis_states: list, n_qubits: int):
        num_entries = len(basis_states)
        if n_qubits < 1 or not isinstance(n_qubits, int):
            raise ValueError(f"n_qubits must be a positive integer, got {n_qubits}.")
        self.n = n_qubits
        if (num_dist := len(set(basis_states))) != num_entries:
            raise ValueError(
                f"Computational basis states must be unique, got {num_entries} basis states but "
                f"just {num_dist} distinct basis states."
            )
        if num_entries < 2:  # No need for this algorithm
            raise ValueError(f"At least two basis states are required. Got {num_entries}.")
        # Choose the packing representation. ``uint64`` is the fast native path for
        # n <= 63 qubits; for wider registers a single 64-bit word cannot hold a row, so we fall
        # back to Python big integers stored in a ``dtype=object`` array. ``_word`` converts a
        # Python int to the packing scalar type.
        if self.n > 63:
            self._packed_dtype = object
            self._word = int
        else:
            self._packed_dtype = _U64
            self._word = _U64

        self.n_subspace = max(math.ceil_log2(num_entries), 1)
        self.n_r = self.n - self.n_subspace

        # Packed tableau: one word per row. Column 0 is the MSB (weight 2**(n-1)), matching
        # the MSB-first convention of ``int_to_binary``, so a row's integer value == its index.
        # ``int(x)`` normalizes both numpy-int and Python-int inputs (the latter occur when the
        # indices exceed int64 for n >= 64 and are already stored in an object array).
        self.tableau = np.array([int(x) for x in basis_states], dtype=self._packed_dtype)
        self.circuit = []  # Forward circuit realizing the isometry _to_ densified basis sates

        # Largest power of 2 less than or equal to n_r, the remainder register size
        if self.n_r == 0:
            self.m = 0
            return

        self.m = 1 << int(math.floor(math.log2(self.n_r)))

        # Frequently used word constants, precomputed in the packing type to avoid any casts
        # inside the hot loop.
        self._zero = self._word(0)
        self._one = self._word(1)
        self._nr_shift = self._word(self.n_r)

        # Mask selecting the remainder register (the low n_r bits).
        self.rem_mask = self._word((1 << self.n_r) - 1)

        # Pre-compute subspace membership: a row is in the subspace iff all remainder bits are
        # zero. This is refreshed only inside ``pui`` (never after ``fanout``).
        self._in_subspace = (self.tableau & self.rem_mask) == self._zero
        # Number of rows not in the subspace. Maintained incrementally, it only changes when
        # ``pui`` is called (the only op that refreshes ``_in_subspace``). Rows entering the
        # current batch stay marked as "not in subspace" until the next ``pui``, so that the
        # number of _remaining_ rows to be mapped (not in subspace and not in the batch)
        # will equal ``_n_not_subspace - len(batch)``.
        self._n_not_subspace = int(np.count_nonzero(~self._in_subspace))

        # Precompute per-column single-bit masks (big endian) and shift amounts. These are
        # looked up many times in the main loop, so caching avoids repeated casting to _word type.
        self._shifts = [self._word(self.n - 1 - c) for c in range(self.n)]
        self._col_masks = [self._one << s for s in self._shifts]

    def _col_bit(self, col: int):
        """Return the word mask with a single set bit at tableau column ``col``."""
        return self._col_masks[col]

    def _diff_bits(self, diff_val: int) -> np.ndarray:
        """Return the length-``n`` MSB-first binary representation of ``diff_val`` (a Python
        int) as an ``int8`` array. Uses ``math.int_to_binary`` on the single-word path; that
        function overflows for values wider than 63 bits, so the multi-word path uses an explicit
        big-int extraction instead."""
        if self.n <= 63:
            # Single-word scenario
            return math.int_to_binary(diff_val, self.n).astype(np.int8)

        # Multi-word scenario
        _shifts = ((self.n - 1 - c) for c in range(self.n))
        return np.fromiter(((diff_val >> s) & 1 for s in _shifts), dtype=np.int8, count=self.n)

    def pui(self, k: int, batch_size: int):
        """Add a Partial unary iterator circuit (in form of ``Select``) to the circuit ops and
        apply the corresponding multicontrolled bit flips to the tableau."""
        k_start = k - batch_size
        self.circuit.append(("PUI", k_start, k))

        # Update tableau for PUI effect
        # For batch element j, rows whose subspace value equals k_start + j get remainder
        # qubit j flipped. Remainder qubit j is tableau column n_subspace + j, whose cached
        # single-bit mask has value 2**(n_r - 1 - j).
        subspace_val = self.tableau >> self._nr_shift
        for j in range(batch_size):
            mask = subspace_val == (k_start + j)
            self.tableau[mask] ^= self._col_masks[self.n_subspace + j]

        # Update subspace status after PUI.
        self._in_subspace = (self.tableau & self.rem_mask) == self._zero
        self._n_not_subspace = int(np.count_nonzero(~self._in_subspace))

    def fanout(self, control: int, bits: np.ndarray):
        """Add a Fanout operation to the circuit ops and apply corresponding CNOTs to the tableau.

        ``bits`` is a length-``n`` binary array (the diff vector marking the bits to be flipped).
        All columns set in ``bits`` except ``control`` itself are XORed with the ``control``
        column (per row), while the ``control`` column is left unchanged.
        """
        self.circuit.append(("Fanout", control, np.delete(bits, control)))

        # Packed flip mask: all set bits of ``bits`` except the control bit.
        flip_mask = self._zero
        for c in np.nonzero(bits)[0]:
            if int(c) != control:
                flip_mask |= self._col_bit(int(c))

        # Vectorized XOR: rows whose control bit is 1 get ``flip_mask`` applied.
        # Broadcasting ``ctrl_bit`` (0/1) to the full flip word avoids a boolean fancy-index
        # write, which causes overhead for large registers.
        ctrl_bit = (self.tableau >> self._shifts[control]) & self._one
        self.tableau ^= ctrl_bit * flip_mask

    def toffoli(self, controls: list, second_ctrl_val: int, target: int):
        """Add a MultiControlledX operation to the circuit ops and apply it to the tableau."""
        self.circuit.append(("Toffoli", controls + [target], second_ctrl_val))

        # Apply multi-controlled X to the tableau.
        # Create control mask and the control pattern we want to match, from ``controls`` and
        # ``control_values``. This is fast enough because we only ever use this function to realize
        # bit flips from Toffoli gates, so len(control)=len(control_values)=2
        ctrl_mask = self._zero
        ctrl_pattern = self._zero
        for c, v in zip(controls, (1, second_ctrl_val), strict=True):
            bit = self._col_bit(c)
            ctrl_mask |= bit
            if v:
                ctrl_pattern |= bit
        match = (self.tableau & ctrl_mask) == ctrl_pattern
        # Where the control pattern matches, we flip the target bit by XORing with the bitstring
        # having only the target bit set to one.
        self.tableau[match] ^= self._col_bit(target)

    def swap(self, w0: int, w1: int):
        """Add a SWAP operation to the circuit ops and apply SWAP to the tableau."""
        self.circuit.append(("SWAP", [w0, w1]))
        # positions for the two qubits
        p0 = self._shifts[w0]
        p1 = self._shifts[w1]
        # masks rows which have the bits w0 / w1 set, respectively
        b0 = (self.tableau >> p0) & self._one
        b1 = (self.tableau >> p1) & self._one
        diff = b0 ^ b1  # masks rows where the two bits differ
        self.tableau ^= (diff << p0) | (diff << p1)  # Flip differing bits on correct positions

    def _next_state_with_target_set(self, target_qubit: int) -> int | None:
        """Stage 1 of step 2.
        Find the next state in the tableau that has the remainder qubit with
        index ``target_qubit`` set to one."""
        # Translate from remainder index to total index
        actual_qubit = self.n_subspace + target_qubit
        bit = self._col_bit(actual_qubit)
        # masks rows where the actual target qubit is set
        hits = self.tableau & bit
        # We only need the *first* row with the bit set. ``argmax`` finds it in a single scan
        # with no output allocation (unlike ``nonzero``).
        # ``argmax`` returns 0 both when row 0 is the first hit and when there is no hit, so we
        # disambiguate by checking explicitly that the bit is set in the returned row.
        first = int(np.argmax(hits != self._zero))
        if hits[first]:
            # The bit is indeed set
            return first
        # We got first=0 but because there was no hit, not because row 0 had the target qubit set.
        return None

    def _try_swap(self, remaining: np.ndarray, target_qubit: int) -> int | None:
        """Stage 2 of step 2:
        find a remainder qubit (of higher index than ``target_qubit``)
        that is set on at least one remaining bit string, and swap it with ``target_qubit``."""
        actual_qubit = self.n_subspace + target_qubit
        # We know that no bit string has the actual_qubit set to 1 because
        # _next_state_with_target_set failed to find it. We look for a one on a higher-index
        # qubit in the remainder register, i.e. columns actual_qubit+1 .. n-1.
        # Create a mask for higher-index remainder qubits.
        lower_mask = self._col_bit(actual_qubit) - self._one
        if lower_mask == self._zero:
            return None

        rem = np.asarray(remaining)
        masked = self.tableau[rem] & lower_mask
        rows_with_qubit_set = np.nonzero(masked)[0]
        if len(rows_with_qubit_set) == 0:
            return None

        # First remaining row (smallest position) that has such a bit set.
        r_pos = int(rows_with_qubit_set[0])
        row_val = int(masked[r_pos])
        # First matching column == highest set bit within the region.
        # column c has weight 2**(n-1-c); highest set bit -> smallest column index.
        swap_from = self.n - row_val.bit_length()
        self.swap(actual_qubit, swap_from)
        return int(rem[r_pos])

    def _try_toffoli(self, remaining: np.ndarray, target_qubit: int, batch: list) -> int:
        """Stage 3 of step 2.
        Find a bitstring that has one of the remainder qubits set to one that are already
        used in ``batch`` but differs on a different qubit. Then apply the Toffoli trick
        to flip the ``target_qubit`` of the found bitstring. We know that the latter must be 0,
        otherwise _next_state_with_target_set would have identified the bit string as next
        candidate in stage 1 of this step already."""
        actual_qubit = self.n_subspace + target_qubit
        idx = remaining[0]

        # Find a remainder qubit lower than actual_qubit that is set to one. We know that
        # idx must have such a qubit because
        # - it must have at least one remainder qubit set to one because it is in `remaining`
        # - it can't have any qubits at or above the index `actual_qubit` set, because if it did,
        #   _next_state_with_target_set or _try_swap would have been successful.
        # Create a mask for this by XORing away the higher-index bits from the
        # ``rem_mask`` constant.
        region_mask = self.rem_mask ^ (self._col_bit(actual_qubit) - self._one)
        region_val = int(self.tableau[idx] & region_mask)
        # Highest set bit (smallest column index) within the region.
        active_col = self.n - region_val.bit_length()
        # Translate to index in remainder space
        active_remainder_bit = active_col - self.n_subspace

        # Find a qubit at which the bitstrings A at position batch[active_remainder_bit] and
        # B at position `idx` differ. Note that we know that there is a differing qubit because
        # the bitstrings are unique. Also, note that actual_qubit can't be a differing qubit,
        # because A only has ones in the subspace and at position
        # active_remainder_bit < actual_qubit, and B does not have actual_qubit set because
        # in that case, _next_state_with_target_set would have found it already.
        xor_val = int(self.tableau[idx] ^ self.tableau[batch[active_remainder_bit]])
        diff_qubit = self.n - xor_val.bit_length()

        controls = [self.n_subspace + active_remainder_bit, diff_qubit]
        second_ctrl_val = int((int(self.tableau[idx]) >> (self.n - 1 - diff_qubit)) & 1)
        self.toffoli(controls, second_ctrl_val, actual_qubit)
        return idx

    def _map_full_state(self, found_state: int, k: int, target_qubit: int):
        """Execute step 3 of the algorithm, zeroing all remainder qubits controlled on
        ``target_qubit`` (except for ``target_qubit`` itself) and setting the subspace qubits
        to the integer ``k``."""
        # Target packed value: subspace bits = k, remainder bits = 0:  k << n_r.
        found_val = int(self.tableau[found_state])
        target_val = k << self.n_r
        diff_val = target_val ^ found_val

        # Reconstruct the length-n diff bit array for faithful circuit record.
        diff_bits = self._diff_bits(diff_val)
        actual_qubit = self.n_subspace + target_qubit
        self.fanout(actual_qubit, diff_bits)

    def find_isometry(self):
        """Main method to find the isometry. See main docstring for a detailed description."""
        if self.m == 0:
            return self.circuit, {i: int(val) for i, val in enumerate(self.tableau)}

        bijection = {}  # Bijection f between desired states and densified basis states
        batch = []

        k = 0
        while True:
            b = len(batch)

            # Number of remaining rows (not in subspace and not in batch). ``_n_not_subspace``
            # counts rows not in the subspace; batch members are a subset of those (they stay
            # marked until the next ``pui``), so subtracting ``b`` gives the remaining count.
            # The actual ``remaining`` index array is only materialized on demand for the
            # rare fallback stages below, so we save those cost in most iterations
            n_remaining = self._n_not_subspace - b

            if b == self.m or (n_remaining == 0 and b > 0):
                # Step 4
                # Need to flush because the batch is full, or because there are no remaining
                # bit strings to map but the batch still has some entries.
                self.pui(k, len(batch))
                batch = []
                continue

            if n_remaining == 0:
                # We know that b == 0 because (n_remaining==0 and b>0) is caught above
                # Hence, the batch is empty and there are not states remaining to be handled.
                # We thus are done and exit the loop.
                break

            # Step 2, three stages:
            # Stage 1: Find a state with the b-th non-subspace qubit set to 1
            target_qubit = b
            found_state = self._next_state_with_target_set(target_qubit)

            if found_state is None:
                # Stage 2: Try swapping with a qubit further right to get a state with b-th
                # non-subspace qubit set to 1. Now we need the explicit ``remaining`` index array.
                mask = ~self._in_subspace
                if batch:
                    # Make sure not to overwrite _in_subspace
                    mask = mask.copy()
                    mask[batch] = False
                remaining = np.nonzero(mask)[0]

                found_state = self._try_swap(remaining, target_qubit)

                if found_state is None:
                    # If we still have to map at least one bit string, the batch is currently empty,
                    # and neither _next_state_with_target_set nor _try_swap were successful, we know
                    # that _try_toffoli can't work (because it needs a reference batch state to
                    # discriminate against). Thus, we would be stuck in an infinite loop of trying the
                    # three stages of step 2.
                    assert b > 0, (
                        "This scenario should never happen because it would lead to "
                        "infinite recursion."
                    )
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
        tableau = (self.tableau >> self._nr_shift).astype(object)
        for i, val in enumerate(tableau):
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
    `Rupprecht and Wölk, arXiv:2601.09388 <https://arxiv.org/abs/2601.09388>`__. It consists of a
    dense state preparation of the amplitudes on a *subspace register*, and an isometric mapping
    that permutes the prepared amplitudes into the correct positions for the target state.
    As such, this method is tailored to sparse states. However, it approaches
    :class:`~.MultiplexerStatePreparation` for less sparse states, effectively providing an
    interpolation between sparse and dense state preparation.

    .. seealso::

        :class:`~.PUIsometryFinder` for the classical algorithm that finds the isometry for
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
            :math:`|L|` entries in the state, :math:`\max(\lceil \log_2(|L|)\rceil-1, 1)` work wires
            are needed. Adding more work wires reduces the depth and gate count of the circuit.

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
    0: ─╭MultiplexerStatePreparation(M0)─╭|Ψ⟩─╭◑────────╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭◑────────╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭◑────────╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭◑────────╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭◑────────╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─┤ ╭State
    1: ─├MultiplexerStatePreparation(M0)─├|Ψ⟩─├◑────────├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├◑────────├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├◑────────├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├◑────────├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├◑────────├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─┤ ├State
    2: ─├MultiplexerStatePreparation(M0)─├|Ψ⟩─├◑────────├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├◑────────├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├◑────────├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├◑────────├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─├◑────────├|Ψ⟩─├|Ψ⟩─├|Ψ⟩─┤ ├State
    3: ─╰MultiplexerStatePreparation(M0)─╰|Ψ⟩─├◑────────╰|Ψ⟩─├|Ψ⟩─├|Ψ⟩─╰|Ψ⟩─├◑────────╰|Ψ⟩─├|Ψ⟩─├|Ψ⟩─╰|Ψ⟩─├◑────────╰|Ψ⟩─├|Ψ⟩─├|Ψ⟩─╰|Ψ⟩─├◑────────╰|Ψ⟩─├|Ψ⟩─├|Ψ⟩─╰|Ψ⟩─├◑────────╰|Ψ⟩─├|Ψ⟩─├|Ψ⟩─┤ ├State
    4: ───────────────────────────────────────├QROM(M1)──────├|Ψ⟩─├●────────├QROM(M1)──────├|Ψ⟩─├●────────├QROM(M1)──────├|Ψ⟩─├●────────├QROM(M1)──────├|Ψ⟩─├●────────├QROM(M1)──────├|Ψ⟩─├●───┤ ├State
    5: ───────────────────────────────────────├QROM(M1)──────╰●───╰|Ψ⟩──────├QROM(M1)──────╰●───╰|Ψ⟩──────├QROM(M1)──────╰●───╰|Ψ⟩──────├QROM(M1)──────╰●───╰|Ψ⟩──────├QROM(M1)──────╰●───╰|Ψ⟩─┤ ├State
    6: ───────────────────────────────────────├work─────────────────────────├work─────────────────────────├work─────────────────────────├work─────────────────────────├work────────────────────┤ ├State
    7: ───────────────────────────────────────├work─────────────────────────├work─────────────────────────├work─────────────────────────├work─────────────────────────├work────────────────────┤ ├State
    8: ───────────────────────────────────────╰work─────────────────────────╰work─────────────────────────╰work─────────────────────────╰work─────────────────────────╰work────────────────────┤ ╰State

    We can make out the dense state preparation on the *subspace register* ``0`` through ``3``,
    followed by the isometry circuit consisting of partial unary iteration circuits and ``CNOT``
    blocks. The basis state preparations denoted as ``|Ψ⟩`` are layers of simple ``PauliX`` bit
    flips that allow us to shift the range of the partial unary iterations to start from ``0``, and
    thus to reuse generic :class:`~.Select` subroutines, independent of the actual iteration range.

    For more complex examples, the circuit may also contain :class:`~.SWAP` and/or Toffoli
    gates (in the form of :class:`~.MultiControlledX`):

    .. code-block:: python

        np.random.seed(31)
        num_entries = 2553
        coefficients = np.random.random(num_entries)
        coefficients /= np.linalg.norm(coefficients)
        indices = np.random.choice(2**15, num_entries, replace=False)
        wires = list(range(15))
        num_work_wires = qp.math.ceil_log2(num_entries) - 1
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

    Note that passing more work wires than the needed :math:`\max(\lceil \log_2(|L|)\rceil-1, 1)`
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
    number of indices :math:`|L|`, it is the same between the two decompositions.

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
        num_entries = len(indices)
        if len(set(indices)) != num_entries:
            raise ValueError("The state indices must be unique.")
        if len(coefficients) != num_entries:
            raise ValueError(
                "The number of coefficients and the number of state indices must match."
            )
        if max(indices) > 2 ** len(wires) - 1:
            raise ValueError(
                f"The state indices must be smaller than {2**len(wires)=}. Largest index is {max(indices)}"
            )
        if min(indices) < 0:
            raise ValueError(
                f"The state indices must be positive. Smallest index is {min(indices)}"
            )

        work_wires = Wires([] if work_wires is None else work_wires)
        super().__init__(coefficients, wires=wires)
        self.hyperparameters["indices"] = indices
        self.hyperparameters["work_wires"] = Wires(work_wires)


def _pui_state_prep_resources(num_entries, num_wires, num_work_wires):
    """Compute the resources for _pui_state_prep, the partial unary iteration state prep.
    These resource counts are numerically obtained heuristics, extended to guarantee all
    resource reps that may appear are included at least once."""
    if num_entries == 1:
        return {qp.resource_rep(qp.BasisState, num_wires=num_wires): 1}

    n_subspace = max(math.ceil_log2(num_entries), 1)
    resources = defaultdict(int)
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

    resources[qp.SWAP] += num_wires

    num_toffolis = int(num_wires / 10) + 1
    mcx_rep = qp.MultiControlledX(Wire[3], work_wires=Wire[1], work_wire_type="zeroed")
    resources[mcx_rep] += num_toffolis

    return resources


def _pui_state_prep_core(coefficients, wires, indices, work_wires):
    """Compute the decomposition of the partial unary iteration state preparation technique.
    This core method is used by the two rules below, which only differ by the work
    wire management."""
    num_entries = len(indices)
    if num_entries == 1:
        qp.BasisState(indices[0], wires)
        return

    n_subspace = max(math.ceil_log2(num_entries), 1)
    needed_work_wires = max(n_subspace - 1, 1)  # Need n_subspace-1 for QROM, 1 for Toffoli
    wires = Wires(wires)
    if len(work_wires) > needed_work_wires:
        # There is an easy way to use more work wires: Pretend they are part of the system wires,
        # which gives the isometry circuit more space to move states around. Adding the excess
        # work wires to the beginning of the `wires` register allows us to keep `indices`
        # unchanged. It is not obvious whether this approach or a different wire ordering will
        # be cheaper in terms of quantum resources used in the isometry circuit.
        wires = Wires(work_wires[needed_work_wires:]) + wires

    iso_finder = PUIsometryFinder(np.array(indices), len(wires))
    circuit, bijection = iso_finder.find_isometry()

    subspace_wires = Wires(wires[:n_subspace])
    nonsubspace_wires = Wires(wires[n_subspace:])

    # Step 1: Dense state preparation
    dense_state = np.zeros(2**n_subspace, dtype=complex)
    ids = np.array([bijection[i] for i in range(num_entries)])
    dense_state[ids] = coefficients
    qp.MultiplexerStatePreparation(dense_state, subspace_wires)

    if not circuit:
        return

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
            target_wires = wires[:control] + wires[control + 1 :]
            qp.ctrl(qp.BasisState(bits, target_wires), wires[control])
            continue

        ids = data[0]
        _wires = [wires[idx] for idx in ids]
        if _type == "SWAP":
            qp.SWAP(_wires)
        elif _type == "Toffoli":
            qp.MultiControlledX(_wires, [1, data[1]], work_wires[0], work_wire_type="zeroed")
        else:
            raise NotImplementedError  # pragma: no cover


# Decomposition rule with statically given work_wires to PartialUnaryStatePreparation


# pylint: disable=unused-argument
def _pui_state_prep_provided_work_wires_condition(num_entries, num_wires, num_work_wires):
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
