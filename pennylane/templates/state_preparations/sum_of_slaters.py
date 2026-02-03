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
r"""Contains the SumOfSlatersPrep template."""

from collections import defaultdict
from itertools import combinations, product

import numpy as np

import pennylane as qml
from pennylane import allocate, for_loop, math
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import Operation
from pennylane.queuing import AnnotatedQueue, QueuingManager


def _columns_differ(bits: np.ndarray) -> bool:
    r"""Check whether all columns of a bit array differ pairwise.

    Args:
        bits (np.ndarray): Array of bits (zeros and ones) to check for differing column count.

    Returns:
        bool: Whether all columns of ``bits`` differ.

    Note that this is not a rank check over :math:`\mathbb{Z}_2`, because
    we test the columns to differ, not to be linearly independent (linear
    independence implies being different, but not conversely).

    **Example**

    Consider three differing columns of length 2:

    >>> from pennylane.templates.state_preparations.sum_of_slaters import _columns_differ
    >>> differing_bits = np.array([[1, 0, 1], [0, 1, 1]])
    >>> print(differing_bits)
    [[1 0 1]
     [0 1 1]]
    >>> _columns_differ(differing_bits)
    True

    This is opposed to columns of length 2 were the first and third column are the same:

    >>> redundant_bits = np.array([[1, 0, 1], [0, 1, 0]])
    >>> print(redundant_bits)
    [[1 0 1]
     [0 1 0]]
    >>> _columns_differ(redundant_bits)
    False

    """
    # In order to be representable faithfully with 64-bit integers, the strings can't be too large
    # Note that we do not care about the range of the integers, just about them being unique.
    # This limitation can easily be fixed in a number of ways (e.g., compare the raw columns,
    # or compare tuples of int64 integers, where each integer collects 64 rows)
    if bits.shape[0] > 64:
        raise ValueError(
            "Column comparison uses 64-bit integers internally. Can't compare bitstrings longer "
            "than 64 bits."
        )

    # Powers of two to compute integer representations of columns
    pot = 2 ** np.arange(bits.shape[0])
    # Compute the integer representation of each column
    ints = np.dot(pot, bits)
    # Return whether the integers all differ
    return len(set(ints)) == len(ints)


def select_sos_rows(bits: np.ndarray) -> tuple[list[int], np.ndarray]:
    r"""Select rows of a bit array of differing columns such that the stacked array of the
    selected rows still contains differing columns. Also memorizes the row indices of the input
    array that were selected.

    Args:
        bits (np.ndarray): Bit array with differing columns

    Returns:
        tuple[list[int], np.ndarray]: Selected row indices to obtain the reduced bit array,
        and the reduced bit array itself. If ``bits`` had shape ``(n_rows, n_cols)`` and the list
        of row indices has length ``r``, the reduced bit array has shape ``(r, n_cols)``.

    .. note::

        This function does not come with an optimality guarantee about the number of selected rows.
        That is, there may be selections of fewer rows that maintain the columns to differ.

    Under the hood, this method is not selecting rows, but instead greedily removes rows from the
    input. We first attempt to remove rows with a mean weight far away from 0.5, as they are
    generically less likely to differentiate many columns from each other.

    **Example**

    Let's generate a random bit array of ``D=8`` differing columns of length ``n=6``, by first
    sampling unique integers from the range ``(0, 2**n)`` and converting them to bitstrings.

    >>> np.random.seed(31)
    >>> D = 8
    >>> n = 6
    >>> ids = np.random.choice(2**n, size=D, replace=False)
    >>> bitstrings = ((ids[:, None] >> np.arange(n-1, -1, -1)[None, :]) % 2).T
    >>> print(bitstrings)
    [[0 0 0 1 0 0 1 0]
     [0 0 0 1 0 1 1 1]
     [0 0 0 0 0 1 0 0]
     [0 1 1 1 1 1 1 1]
     [1 0 0 1 1 1 0 0]
     [0 0 1 1 1 0 1 1]]

    Then let's select rows that maintain the uniqueness of the rows:

    >>> from pennylane.templates.state_preparations.sum_of_slaters import select_sos_rows
    >>> selectors, new_bits = select_sos_rows(bitstrings)
    >>> selectors
    [0, 1, 4, 5]

    Indeed, selecting the indicated rows of ``bitstrings``, we still find
    unique columns:

    >>> print(new_bits)
    [[0 0 0 1 0 0 1 0]
     [0 0 0 1 0 1 1 1]
     [1 0 0 1 1 1 0 0]
     [0 0 1 1 1 0 1 1]]

    In general, the number of rows :math:`r` selected by the method will satisfy
    :math:`\log_2(n_{\text{col}})\leq r\leq \min(n_{\text{row}}, n_{\text{col}})` if
    :math:`(n_{\text{row}}, n_{\text{col}})` is the shape of the input array.

    """
    selectors = list(range(len(bits)))

    while True:
        # compute weight of each row. We'll try to first remove rows with a
        # mean weight far away from 0.5
        weights = math.mean(bits, axis=1)
        ordering = math.argsort(math.abs(0.5 - weights))
        for i in reversed(ordering):
            # Check whether the array with row ``i`` removed still has unique columns
            new_bits = math.concatenate([bits[:i], bits[i + 1 :]])
            if _columns_differ(new_bits):
                # If the columns remain unique, remove the row and the row index from selectors
                del selectors[i]
                bits = new_bits
                break
        else:
            # If no row could be removed, stop the process
            break

    return selectors, bits


def _find_ell(set_M: np.ndarray, set_N: np.ndarray, bits_basis: np.ndarray) -> np.ndarray:
    r"""Find replacement vector :math:`\ell` to construct lower-rank set of bit strings during
    recursive computation of kernel space :math:`\mathcal{W}` in ``_get_w_vectors``."""

    # Number of bits
    r = len(bits_basis)
    # Fector to be replaced
    v_r = bits_basis[:, -1]
    # Map the set N to N', containing the components that are spanned by
    # all basis vectors except for v_r
    set_N_prime = set_N + v_r[:, None]
    # Compute all differences (=sums, over ℤ_2) of vectors in M and N'
    if set_M.shape[1] == 0 or set_N.shape[1] == 0:
        combs = np.zeros((r, 0), dtype=int)
    else:
        combs = np.array([m + m_prime for m, m_prime in product(set_M.T, set_N_prime.T)]).T
    zero = np.zeros((r, 1), dtype=int)
    # Collect all bitstrings that ell may not be equal to. Note that we have
    # ``set_M.shape[1] + set_N.shape[1] + 1 = D``, where D is the number of total bitstrings in
    # the algorithm. Thus, the size of ``all_bitstrings_to_avoid`` is bounded from above by
    # D + max_{0≤n≤D-1} (n * (D-1-n)) = (D+1)^2/4.
    # Further, we only call ``_find_ell`` in ``_find_w`` for ``t>1``, i.e.
    # we know that r≥m+2=2d+1. Thus, 2^(r-2)≥D^2/2>(D+1)^2/4 for D>2.
    all_bitstrings_to_avoid = np.concatenate([set_M, set_N_prime, combs, zero], axis=1) % 2

    for i in range(2 ** (r - 2)):
        ell_bits_in_basis = (i >> np.arange(bits_basis.shape[1] - 2, -1, -1)) % 2
        ell = (bits_basis[:, :-1] @ ell_bits_in_basis) % 2
        if not np.any(np.all(ell[:, None] == all_bitstrings_to_avoid, axis=0)):
            break
    else:
        raise ValueError("ell should always be guaranteed to exist.")  # pragma: no cover
    return ell


def _find_single_w(bits):
    r"""Brute force search a single vector :math:`\{w_1\}` that differs from all input bits
    and all pairwise differences (=sums, over Z_2) of bits.

    Note that this function uses `int64` intepretation of bit strings for fast comparisons,
    limiting it to ``r<=64``.
    """
    r, D = bits.shape
    if r > 64:
        raise ValueError(
            "Bitstring search _find_single_w uses 64-bit integers internally. Can't process "
            f"bitstrings longer than 64 bits. Got {bits.shape[0]} bits per column."
        )
    # Power of two
    pot = 2 ** np.arange(r - 1, -1, -1)
    if D > 1:
        diffs = np.array([(v_i - v_j) for v_i, v_j in combinations(bits.T, r=2)]).T % 2
    else:
        diffs = np.zeros((r, 0), dtype=int)
    # Compute the integer representation of each column and each difference of columns
    ints = set(np.dot(pot, np.concatenate([bits, diffs], axis=1)))
    unoccupied = next(i for i in range(1, len(ints) + 1) if i not in ints)
    w = (unoccupied >> np.arange(r - 1, -1, -1)) % 2
    return w[:, None]


def _find_w(bits_basis, other_bits, t):
    """Compute the kernel space W from the original bit strings, including a basis for their
    column space. See the documentation of ``compute_sos_encoding`` for details.
    """
    r = bits_basis.shape[0]
    if t == 1:
        # Step 3: brute-force search of a single vector w_1
        all_bits = np.concatenate([bits_basis, other_bits], axis=1)
        # Compute set(V) ∪ (set(V)+set(V)). We will skip 0 by starting our search at 1 below
        diffs = np.array([(v_i - v_j) for v_i, v_j in combinations(all_bits.T, r=2)]).T
        all_bitstrings_to_avoid = np.concatenate([all_bits, diffs], axis=1) % 2

        # Note that the set ``all_bitstrings_to_avoid`` has size at most D+(D^2-D)/2=(D^2+D)/2
        # For r<=m+1, we actually never call ``_find_w``, so that we know r≥m+2=2d+1, and thus
        # 2^(r-1) ≥ D^2 > (D^2+D)/2 (for D>1), so that 2^(r-1) candidates are enough to find a new
        # vector.
        for i in range(1, 2 ** (r - 1)):
            w_bits_in_basis = (i >> np.arange(bits_basis.shape[1] - 2, -1, -1)) % 2
            w = (bits_basis[:, :-1] @ w_bits_in_basis) % 2
            if not np.any(np.all(w[:, None] == all_bitstrings_to_avoid, axis=0)):
                break

        W = w[:, None]
        return W

    v_r = bits_basis[:, -1]
    bits_without_v_r = np.concatenate([bits_basis[:, :-1], other_bits], axis=1)

    # Step 4: Split the set of bitstrings into three sets:
    #  - set_M <> \mathcal{M}
    #  - v_r <> \{v_l\}
    #  - set_N <> \mathcal{N}
    indep_of_reduced_basis = np.array(
        [math.binary_is_independent(vec, bits_basis[:, :-1]) for vec in bits_without_v_r.T]
    )

    # Note that the first t-1 columns of set_M are guaranteed to match bits_basis[:, :-1]
    set_M = bits_without_v_r[:, np.where(~indep_of_reduced_basis)[0]]
    set_N = bits_without_v_r[:, np.where(indep_of_reduced_basis)[0]]

    # Step 6: Brute-force search bitstring ell to replace v_l. Step 5 is included in _find_ell
    ell = _find_ell(set_M, set_N, bits_basis)

    # Step 7: Compute new set sub_bits <> \mathcal{V}' of bitstrings in span of bits_basis[:, :-1]
    #         and call _find_w recursively on the sub_bits to compute sub_W <> \mathcal{W}'
    w_t = (v_r + ell) % 2
    _set_N = (set_N + w_t[:, None]) % 2
    sub_bits = np.concatenate([set_M, ell[:, None], _set_N], axis=1)
    sub_W = _find_w(bits_basis[:, :-1], sub_bits[:, bits_basis.shape[1] - 1 :], t - 1)

    # Step 8: Append the replacement vector w_t to sub_W
    W = np.concatenate([sub_W, w_t[:, None]], axis=1)
    return W


def _find_U_from_W(W):
    """Compute a linearly independent set of vectors ``{u_i}_i`` that satisfy the equations
    ``u_i @ W = 0`` for a "tall" rectangular matrix ``W`` with maximal rank.

    That is, we compute a basis for the kernel of ``W``, which has the shape ``(r, t)`` with
    ``r>=t`` and has rank ``t``. For this, we construct the augmented matrix ``A = (W | 1_r)``,
    where ``1_r`` is the identity matrix in ``r`` dimensions. Then, we compute the (reduced)
    row echelon form of ``A`` and read out the last ``r-t`` rows of the appended part in ``A``,
    i.e. the last ``r-t`` rows of the last ``r`` columns.
    """
    r, t = W.shape
    # Create augmented matrix for Gauss-Jordan elimination.
    A = np.concatenate([W, np.eye(r, dtype=int)], axis=1)
    A = math.binary_finite_reduced_row_echelon(A, inplace=True)
    U = A[t:, t:]
    return U


def compute_sos_encoding(bits):
    r"""Compute the bitstrings :math:`U` and :math:`b` from Lemma 1 in
    the Sum of Slaters paper
    (`Fomichev et al., PRX Quantum 5, 040339 <https://doi.org/10.1103/PRXQuantum.5.040339>`__).
    This is the major classical coprocessing required for the state preparation.

    Args:
        bits (np.ndarray): Reduced bitstrings that are input into Lemma 1. The i-th bitstring
            v_i is stored in the i-th column, so that the input shape is ``(r, D)``.

    Returns:
        tuple[np.ndarray]: Two bit arrays. The first is :math:`U`, which maps the input ``bits``
        to ``D`` distinct bitstrings :math:`\{b_i\}` of length :math:`\min(r, m)`, where
        :math:`m=2\lceil \log_2(D)\rceil-1`. The second array are the bitstrings
        :math:`\{b_i\}` themselves, stored as columns.

    **Example**

    Consider an array of bits with distinct columns:

    >>> bits = np.array([
    ...     [0, 1, 1, 0, 1, 0],
    ...     [0, 0, 1, 1, 1, 0],
    ...     [1, 1, 1, 1, 1, 1],
    ...     [0, 0, 1, 0, 0, 0],
    ...     [1, 1, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 1],
    ... ])
    >>> from pennylane.templates.state_preparations.sum_of_slaters import (
    ...     compute_sos_encoding, _columns_differ
    ... )
    >>> print(_columns_differ(bits))
    True
    >>> print(bits.shape)
    (6, 6)

    Our goal is to encode these bitstrings as new, distinct bitstrings of length ``m=5``:

    >>> D = bits.shape[1]
    >>> m = 2 * qml.math.ceil_log2(D) - 1
    >>> print(m)
    5

    We can achieve this with ``compute_sos_encoding``, which computes the encoding matrix ``U``
    and the obtained encoded bitstrings ``b``:

    >>> U, b = compute_sos_encoding(bits)
    >>> print(U)
    [[1 0 0 0 0 0]
     [0 1 0 0 0 0]
     [0 0 1 0 0 0]
     [0 0 0 1 0 0]
     [0 0 0 0 1 0]]
    >>> print(b)
    [[0 1 1 0 1 0]
     [0 0 1 1 1 0]
     [1 1 1 1 1 1]
     [0 0 1 0 0 0]
     [1 1 0 0 0 0]]
    >>> print(_columns_differ(b))
    True

    Note that in this particular example, we could have achieved the reduction simply by selecting
    ``4<m`` rows of the input bits:

    >>> from pennylane.templates.state_preparations.sum_of_slaters import select_sos_rows
    >>> select_ids, sub_bits = select_sos_rows(bits)
    >>> print(sub_bits)
    [[0 1 1 0 1 0]
     [0 0 1 1 1 0]
     [0 0 1 0 0 0]
     [1 1 0 0 0 0]]

    In practice, this sub-selection of bits is combined with ``compute_sos_encoding`` to achieve
    lowest cost.

    .. details::
        :title: Implementation notes

        In the following, we slightly rewrite the first part of the proof.
        Then, we clarify the algorithmic structure for constructing the space :math:`\mathcal{W}`.

        Our goal is to find a linear map :math:`U:\mathbb{Z}_2^{r}\to \mathbb{Z}_2^{m}` from
        :math:`D` distinct bitstrings :math:`\{v_i\}` with length :math:`r` to :math:`D` bitstrings
        with length :math:`m\leq 2d-1`, where :math:`d:=\lceil\log_2(D)\rceil`, such that

        .. math::

            U(v_i-v_j)\neq 0 \forall i, j, \text{and} U(v_i)\neq 0 \forall i \text{unless} v_i=0.

        It will be instructive to rewrite this as

        .. math::

            v_i-v_j\not\in \ker U \forall i, j, \text{and} v_i\not\in \ker U \forall i
            \text{unless} v_i=0.   (1)

        We will speak of :math:`U` and its matrix representation of zeros and ones interchangeably.
        Since :math:`k` bits can represent at most :math:`2^k` different bitstrings, we know that
        :math:`D` different bitstrings :math:`\{v_i\}` require at least :math:`d` bits to
        be represented, i.e. we know that :math:`r\geq d`. We will proceed in two cases from
        here on, differentiated by :math:`r`.

        **Case 1: :math:`d\leq r\leq 2d-1`**

        In this case, we do not really need to do anything; the bitstrings :math:`\{v_i\}` already
        have length :math:`m:=r\leq 2d-1`, so we simply set :math:`U` to be the identity map.

        **Case 2: :math:`2d-1 < r`**

        Fix :math:`m=2d-1` and define :math:`t:=r-m` so that :math:`r=m+t`. According to the rank
        theorem, any candidate linear map :math:`U` satisfies :math:`\dim \Im U + \dim \ker U=r`.
        If we guarantee linear independence of the rows of :math:`U`, we know that
        :math:`\dim \Im U` matches the dimensions of the target space, :math:`m`, and thus
        :math:`\dim \ker U=r-m=t`.

        Our strategy now will be to find :math:`t` linearly independent vectors :math:`\{w_k\}`
        such that the space :math:`\mathcal{W}:=\span\{w_1, \dots w_t\}` spanned by them satisfies

        .. math::

            v_i-v_j\not\in \mathcal{W} \forall i, j, \text{and} v_i\not\in \mathcal{W} \forall i
            \text{unless} v_i=0.   (2)

        and to construct a map :math:`U` with linearly independent rows such that
        :math:`U w_k=0 \forall k`, i.e. :math:`\mathcal{W}\subset\ker U`. Given that we know the
        kernel dimension to be :math:`t` and :math:`\dim\mathcal{W}=t`, this implies
        :math:`\mathcal{W}=\ker U`. To see that this actually ensures :math:`U` to have the
        properties we are after, assume that :math:`U v_i=0` for some :math:`i` with
        :math:`v_i\neq 0` (or :math:`U(v_i-v_j)` for some :math:`(i,j)`). Due to
        :math:`\ker U=\mathcal{W}`, this would imply :math:`v_i\in\mathcal{W}` (or
        :math:`v_i-v_j\in\mathcal{W}`), which is false by construction of the vectors
        :math:`\{w_k\}`, in particular due to Eq. (2)

        The main work thus is to show that we can actually construct these vectors :math:`\{w_k\}`
        with the required properties. The construction of the map :math:`U` itself will then be a
        simple linear algebraic task. We perform the construction of the vectors iteratively,
        corresponding to a proof by induction on :math:`t`.

        From here on, we focus on the algorithmic structure and implementation, and do not
        reproduce the proof of correctness from the paper. Given :math:`D` vectors :math:`\{v_i\}`
        of length :math:`r` with rank :math:`r` (there are at least :math:`r` linearly independent
        vectors), our task is to build :math:`t=r-(2\lceil \log_2 (D)\rceil -1)` linearly
        independent vectors :math:`\{w_k\}` from the space :math:`\span\{v_i\}` such that the
        resulting vector space :math:`\mathcal{W}=\span\{w_k\}` does not contain the
        :math:`\{v_i\}` or their pairwise differences, see Eq.(2). If :math:`t=1`, there is a
        particularly simple method: we can brute-force a search of :math:`w_1` over
        :math:`\mathbb{Z}_2^r\setminus (\{v_i\}\cup \{v_i-v_j\})`. This is implemented
        in ``_find_single_w``.

        If :math:`t>1`, we recursively construct the :math:`\{w_k\}`.
        We proceed in the following steps, thinking of ordered sets whenever we speak of sets.

        1. First, we select :math:`r` of the input vectors :math:`\mathcal{V}=\{v_i\}` that are
           linearly independent (and thus form a--likely not orthonormal--basis of
           :math:`\mathbb{Z}_2^r`). We relabel the vectors so that this selection of vectors has
           the indices :math:`\{1,\dots, r\}`, i.e. the basis is :math:`\mathcal{B}=\{v_1,\dots,v_r\}`.
           This is implemented in ``qml.math.binary_select_basis``, which returns the basis and
           the remaining columns separately.
        2. If :math:`t=1` (which can happen despite :math:`t>1` initially, because we will use
           recursion), go to step 3. Else go to step 4.
        3. Brute-force search a linear combination :math:`w_1` of the basis vectors
           :math:`\mathcal{B}` that is not contained in the set
           :math:`\mathcal{V}\cup(\mathcal{V}-\mathcal{V})\cup \{0\}`, where the set difference
           is meant pairwise between elements. Return :math:`\{w_1\}`.
        4. We split :math:`\mathcal{V}` into three sets: First, :math:`\mathcal{M}` contains all
           vectors that lie in the span :math:`\mathcal{K}` of all but the last basis vector,
           which we denote as :math:`v_l`. The second set is simply :math:`\{v_l\}`. Third,
           :math:`\mathcal{N}` contains all vectors that require both :math:`v_l` and a linear
           combination of the other basis vectors. We maintain the relative ordering within each
           set, so that the first vectors in :math:`\mathcal{M}` correspond to the basis vectors
           (except for :math:`v_l` which is not in :math:`\mathcal{M}` by definition).
        5. Map each vector in :math:`\mathcal{N}` to the space :math:`\mathcal{K}` by
           adding :math:`v_l`, and call the resulting set :math:`\mathcal{N}'`.
        6. Brute-force search a bitstring :math:`\ell` that is not contained in the set
           :math:`\mathcal{L}=\mathcal{M}\cup\mathcal{N}'\cup(\mathcal{M}+\mathcal{N})\cup\{0\}`,
           where addition of sets denotes pairwise addition of elements.
           :math:`\ell` is guaranteed to exist.
        7. Form a new set of vectors
           :math:`\mathcal{V}'=\mathcal{M}\cup(\mathcal{N}'+\ell)\cup\{\ell\}`, while preserving
           ordering. Split off the first vectors :math:`\mathcal{B}'` that correspond to the
           original basis vectors (except for :math:`v_l`) off this set. Set
           :math:`\mathcal{B}\gets\mathcal{B}'`, :math:`\mathcal{V}\gets\mathcal{V}'` and
           :math:`t\gets t'=t-1`, and go to step 2 to compute a set of :math:`t'` vectors
           :math:`\mathcal{W}'` that satisfy the desired properties.
        8. Append :math:`w_t=\ell+v_l` to :math:`\mathcal{W}'` to obtain :math:`\mathcal{W}`
           and return :math:`\mathcal{W}`.

        This procedure will produce the desired linearly independent vectors :math:`\{w_k\}`.
        Computing the matrix :math:`U` such that the vectors are in its kernel is rather simple
        then. This is because the problem is self-dual, i.e., we actually need to find vectors
        in the kernel of :math:`W^T`, and will construct :math:`U` from the kernel vectors as rows.
    """
    r, D = bits.shape
    d = math.ceil_log2(D)
    m = 2 * d - 1
    if r <= m:
        # Case 1: We can use the identity mapping
        U = np.eye(r, dtype=int)
        return U, bits

    # Case 2 splits into two scenarios: t=1 (simple scenario) or t>1 (more involved, see _find_w)
    if r == m + 1:
        # Particularly simple brute force solution for a single vector w
        W = _find_single_w(bits)
    else:
        # Step 1: Construct a basis for the input bits
        bits_basis, other_bits = math.binary_select_basis(bits)
        # Construct the kernel space W, in terms of linearly independent vectors
        W = _find_w(bits_basis, other_bits, t=r - m)

    # Construct the matrix U from the designed kernel W
    U = _find_U_from_W(W)

    # Compute the encoding bit strings b.
    b = (U @ bits) % 2
    return U, b


class SumOfSlatersPrep(Operation):
    """Prepare a sum-of-Slaters state.
    This operation implements the state preparation as introduced by
    `Fomichev et al., PRX Quantum 5, 040339 <https://doi.org/10.1103/PRXQuantum.5.040339>`__, which
    is tailored to sparse states.

    .. seealso:: :func:`~.compute_sos_encoding` for the required classical coprocessing.

    Args:

    """

    resource_keys = {"D", "num_bits", "num_wires"}

    @property
    def resource_params(self):
        indices = self.hyperparameters["indices"]
        wires = self.hyperparameters["target_wires"]
        D = len(indices)
        n = len(wires)
        v_bits = _int_to_binary(np.array(indices), n)  # v_bits.shape = (n, D)
        selector_ids, _ = select_sos_rows(v_bits)
        r = len(selector_ids)
        return {"D": D, "num_bits": r, "num_wires": n}

    ndim_params = (1,)

    def __init__(
        self,
        coefficients,
        target_wires,
        enumeration_wires,
        identification_wires,
        qrom_work_wires,
        mcx_work_wires,
        indices,
    ):
        wire_inputs = [
            target_wires,
            enumeration_wires,
            identification_wires,
            qrom_work_wires,
            mcx_work_wires,
        ]
        all_wires = sum((list(w) for w in wire_inputs), start=[])
        super().__init__(coefficients, wires=all_wires)
        self.hyperparameters["target_wires"] = target_wires
        self.hyperparameters["enumeration_wires"] = enumeration_wires
        self.hyperparameters["identification_wires"] = identification_wires
        self.hyperparameters["qrom_work_wires"] = qrom_work_wires
        self.hyperparameters["mcx_work_wires"] = mcx_work_wires
        self.hyperparameters["indices"] = indices

    @staticmethod
    def compute_decomposition(
        coefficients,
        target_wires,
        enumeration_wires,
        identification_wires,
        qrom_work_wires,
        mcx_work_wires,
        indices,
    ):  # pylint: disable=arguments-differ
        with AnnotatedQueue() as q:
            _sos_state_prep(
                coefficients,
                target_wires,
                enumeration_wires,
                identification_wires,
                qrom_work_wires,
                mcx_work_wires,
                indices,
            )

        op_list = []
        if QueuingManager.recording():
            for op in q:
                op_list.append(qml.apply(op))
        return op_list


def _sos_state_prep_resources(D, num_bits, num_wires):
    if D == 1:
        return {resource_rep(qml.BasisState, num_wires=num_wires): 1}
    d = math.ceil_log2(D)
    m = min(num_bits, 2 * d - 1)
    resources = defaultdict(int)

    # Step 1
    resources[resource_rep(qml.StatePrep, num_wires=d)] += 1

    # Step 2
    qrom_params = {
        "num_bitstrings": D,
        "num_control_wires": d,
        "num_target_wires": num_wires,
        "num_work_wires": d - 1,
        "clean": True,
    }
    resources[resource_rep(qml.QROM, **qrom_params)] += 1

    ## Step 3 & 4:
    resources[resource_rep(qml.CNOT)] += m * num_wires  # size {u_k} * bits in u_k

    ## Step 5:
    # TODO [dwierichs]: Revisit the following "hack" once [sc-110068] is completed.
    # We have to hack the MultiControlledX resources a little bit, even with exact=False.
    # This is because MultiControlledX resources with differing `num_zero_control_values`
    # are considered different by the framework. Overall we know that there are 2D MultiControlledX
    # instances in the decomposition, so we create 2D-m+1 with maximal zeroed control value count,
    # and one with each other zeroed control value count.
    mcx_params = {
        "num_control_wires": m,
        "num_zero_control_values": m,
        "num_work_wires": m - 1,
        "work_wire_type": "zeroed",
    }
    # We use two MultiControlledX operators per bitstring
    resources[resource_rep(qml.MultiControlledX, **mcx_params)] += 2 * D - m + 1
    for i in range(m):
        mcx_params["num_zero_control_values"] = i
        resources[resource_rep(qml.MultiControlledX, **mcx_params)] += 1
    # We use up to d CNOTs for any given bitstring, leading to d*D CNOTs naively.
    # However, as we actually count up from 0 to D, we know that overall we will have to flip each
    # bit at most half of the time, so that we have an upper bound of d*D/2
    resources[resource_rep(qml.CNOT)] += d * D // 2

    ## Step 6:
    resources[resource_rep(qml.CNOT)] += m * num_wires  # size {u_k} * bits in u_k

    return resources


def _int_to_binary(x: np.ndarray, length: int) -> np.ndarray:
    return (x[None] >> np.arange(length - 1, -1, -1)[:, None]) % 2


def _sos_state_prep_work_wires(D, num_bits, **_):
    d = math.ceil_log2(D)
    m = min(num_bits, 2 * d - 1)
    return {"zeroed": d + m + (d - 1) + m}


@register_resources(_sos_state_prep_resources, exact=False, work_wires=_sos_state_prep_work_wires)
def _sos_state_prep(
    coefficients,
    target_wires,
    enumeration_wires,
    identification_wires,
    qrom_work_wires,
    mcx_work_wires,
    indices,
):
    """Compute the decomposition of the sum-of-Slaters state preparation technique."""
    # pylint: disable=no-value-for-parameter
    n = len(target_wires)
    D = len(coefficients)
    v_bits = _int_to_binary(np.array(indices), n)  # Shape (n, D)
    if D == 1:
        qml.BasisState(v_bits[:, 0], wires=target_wires)
        return
    assert v_bits.shape == (n, D)

    # Preprocessing: Compute u_bits and b_bits via the algorithm described in Lemma 1,
    # if not passed to instantiation already
    # if precomputed_bits is None:
    # if selector_ids has length r, vtilde_bits has shape (r, D)
    selector_ids, vtilde_bits = select_sos_rows(v_bits)
    # u_bits has shape (2d-1, r), b_bits has shape (2d-1, D)
    u_bits, b_bits = compute_sos_encoding(vtilde_bits)  # u_bits has shape
    # else:
    # selector_ids, u_bits, b_bits = precomputed_bits

    r = len(selector_ids)
    d = math.ceil_log2(D)
    m = min(r, 2 * d - 1)
    print(f"{d=}")
    print(f"{m=}")
    assert u_bits.shape == (m, r), f"{u_bits.shape=}, {(m, r)=}"
    assert b_bits.shape == (m, D)
    # kw = {"state": "zero", "restored": True}
    enumeration_wires = enumeration_wires[:d]
    identification_wires = identification_wires[:m]
    qrom_work_wires = qrom_work_wires[: d - 1]
    mcx_work_wires = mcx_work_wires[:m]
    print(enumeration_wires)
    print(identification_wires)
    print(qrom_work_wires)
    print(mcx_work_wires)

    # Step 1: Dense state preparation in enumeration register
    # Need to add work wires and correct decomposition
    # qml.MottonenStatePreparation(coefficients, wires=enumeration_wires)
    qml.StatePrep(coefficients, wires=enumeration_wires, pad_with=0.0)

    # Step 2: QROM to load v_bits into system register
    qml.QROM(
        v_bits.T,
        control_wires=enumeration_wires,
        target_wires=target_wires,
        work_wires=qrom_work_wires,
    )

    # Step 3-4): Encode the b_bits from Lemma 1 in the identification register
    @for_loop(m)
    def encoding(i):
        u = u_bits[i]

        @for_loop(r)
        def inner_loop(j):
            qml.cond(u[j], qml.CNOT([target_wires[selector_ids[j]], identification_wires[i]]))

        inner_loop()

    encoding()

    # Step 5): Use the identification register to uncompute the enumeration register
    @for_loop(D)
    def uncompute_enumeration(k, b_bits):
        bits = list(map(int, b_bits[:, k]))

        qml.MultiControlledX(
            wires=qml.math.concatenate([identification_wires, mcx_work_wires[:1]]),
            control_values=bits,
            work_wires=mcx_work_wires[1:],
            work_wire_type="zeroed",
        )

        @for_loop(d)
        def inner_loop(j):
            bit_is_set = (k >> d - j) & 1
            qml.cond(bit_is_set, qml.CNOT([mcx_work_wires[0], enumeration_wires[j]]))

        inner_loop()

        qml.MultiControlledX(
            wires=qml.math.concatenate([identification_wires, mcx_work_wires[:1]]),
            control_values=bits,
            work_wires=mcx_work_wires[1:],
            work_wire_type="zeroed",
        )
        return b_bits

    uncompute_enumeration(b_bits)

    # Step 6): Uncompute the b_i in the identification register (self-adjoint)
    encoding()


add_decomps(SumOfSlatersPrep, _sos_state_prep)
