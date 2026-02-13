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
from pennylane.exceptions import DecompositionUndefinedError
from pennylane.operation import Operation


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

    This is opposed to columns of length 2 where the first and third column are the same:

    >>> redundant_bits = np.array([[1, 0, 1], [0, 1, 0]])
    >>> print(redundant_bits)
    [[1 0 1]
     [0 1 0]]
    >>> _columns_differ(redundant_bits)
    False

    """
    if bits.size == 0 or bits.shape[1] <= 1:
        return True
    sorted_col_indices = np.lexsort(bits)
    sorted_bits = bits[:, sorted_col_indices]
    differences = np.any(sorted_bits[:, 1:] != sorted_bits[:, :-1], axis=0)
    # if all adjacent pairs are different, then all columns are unique.
    return all(differences)


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
        That is, there may be selections of fewer rows that maintain differing columns.

    Under the hood, this method is not selecting rows, but instead greedily removes rows from the
    input. We first attempt to remove rows with a mean weight far away from 0.5, as they are
    generically less likely to differentiate many columns from each other.

    **Example**

    Let's generate a random bit array of ``D = 8`` differing columns of length ``n = 6``, by first
    sampling unique integers from the range ``(0, 2**n)`` and converting them to bitstrings.

    >>> np.random.seed(31)
    >>> D = 8
    >>> n = 6
    >>> ids = np.random.choice(2**n, size=D, replace=False)
    >>> bitstrings = qml.math.int_to_binary(ids, width=n).T
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
    if bits.shape[1] == 1:
        # If there is a single column, we can make our life a bit easier
        return [0], bits[:1]

    selectors = list(range(len(bits)))

    while len(selectors) > 1:  # We will want to keep at least one row
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


def _find_ell(bits_basis: np.ndarray, set_M: np.ndarray, set_N: np.ndarray) -> np.ndarray:
    r"""Find replacement vector :math:`\ell` to construct lower-rank set of bit strings during
    recursive computation of kernel space :math:`\mathcal{W}` in ``_get_w_vectors``.

    In more general terms, we get as input a basis ``bits_basis``,
    a set ``set_M`` of bitstrings that are spanned by all but the last bitstring in that basis
    (i.e. all these bitstrings have coefficient ``0`` for the last basis bitstring),
    as well as a set ``set_N`` of bitstrings that require the last bitstring of the basis
    (i.e. all these bitstrings have coefficient ``1`` for the last basis bitstring).
    For this second set, define ``set_N_prime`` to be the bitstrings in ``set_N`` where we
    subtracted away the contribution of the last basis bitstring
    (i.e. all these bitstrings have been toggled manually to have coefficient ``0`` for the last
    bitstring).

    For this input, the task is then to find a vector/bitstring ``ell`` that is spanned by
    all but the last bitstrings in ``bits_basis`` such that it is not part of ``set_M``,
    ``set_N_prime`` or the set of differences (=sums, over ℤ_2) between bitstrings in ``set_M`` and
    ``set_N_prime``. Finally, we also avoid the zero bitstring.
    Note that this search for ``ell`` is brute-forced: We iterate over all
    integers in a suitable range, translate them to bitstrings, multiply those bitstrings with the
    basis vectors, and check whether the result is in the set of bitstrings we want to avoid.

    Args:
        bits_basis (np.ndarray): Basis of ``B`` bitstrings of length ``r``, should have
            shape ``(r, A)``.
        set_M (np.ndarray): Set of ``n`` bitstrings of length ``r`` that are representable by
            all but the last basis bitstring in ``bits_basis``. Should have shape ``(r, n)``
        set_N (np.ndarray): Set of ``D-1-n`` bitstrings (where ``n`` is given by the shape of
            ``set_M`` and ``D`` is the number of all Slaters in the SOS algorithm) that
            require the last basis bitstring in their representation in ``bits_basis``. Should have
            shape ``(r, D-1-n)``.

    Returns:
        np.ndarray: New bitstring that avoids the described set of bitstrings, and is spanned by
        all but the last basis vector in ``bits_basis``. Will have shape ``(r,)``.

    """
    # Number of bits
    r = len(bits_basis)
    # Vector to be replaced
    v_r = bits_basis[:, -1]
    # Step 5: Map the set N to N', containing the components that are spanned by
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
        # Translate number to bitstring, interpret it as component vector for the basis (by
        # multiplying the basis into the bitstring), and check whether it avoids all bitstrings
        # that we need to avoid (stored in ``all_bitstrings_to_avoid``)
        ell_bits_in_basis = math.int_to_binary(i, width=bits_basis.shape[1] - 1)
        ell = (bits_basis[:, :-1] @ ell_bits_in_basis) % 2
        if not np.any(np.all(ell[:, None] == all_bitstrings_to_avoid, axis=0)):
            break
    else:
        raise ValueError(
            "Unexpected exception: ell should always be guaranteed to exist."
        )  # pragma: no cover
    return ell


def _find_single_w(bits):
    r"""Brute force search a bitstring that differs from all input bitstrings (read as columns
    of the input array) and from all pairwise differences (=sums, over Z_2) of the input
    bitstrings.

    Args:
        bits (np.ndarray): Bitstrings to find a differing bitstring from (taking differences into
            account as well). We denote its shape as ``(r, D)``.

    Returns:
        np.ndarray: Bitstring that differs from all columns in ``bits`` and from all pairwise
        differences of columns in ``bits``. Has shape ``(r, 1)``.

    This function uses comparison of 64-bit integers to test for uniqueness. The resulting
    limitation to ``r<=64`` is avoided by working with the first 64 bits only if ``r>64``.
    Memory limitations of ``bits.size<1.199e21`` guarantee that there will still be a new
    bitstring that we can find. We then just append ``r-64`` zeroes to pad the found bitstring
    to the required length ``r``.
    """
    r, D = bits.shape
    if r > 64:
        # Even if there are more than 64 bits, we can assume that the first 64 bits have
        # enough room for a new unique bitstring, because 2**64 bitstrings of length 65
        # don't fit into any reasonable memory (150Exabytes raw information content)
        # Once we found a differing bitstring on the first 64 bits, we can just append any bits
        # to bring ``w`` to length ``r``.
        bits = bits[:64]
        zeroes_to_append = r - 64
        r = 64
    else:
        zeroes_to_append = 0

    if D > 1:
        diffs = np.array([(v_i - v_j) for v_i, v_j in combinations(bits.T, r=2)]).T % 2
    else:
        diffs = np.zeros((r, 0), dtype=int)

    # Powers of two
    powers = 2 ** np.arange(r - 1, -1, -1)
    # Compute the integer representation of each column and each difference of columns
    ints = set(np.dot(powers, np.concatenate([bits, diffs], axis=1)))
    unoccupied = next(i for i in range(1, len(ints) + 1) if i not in ints)
    w = math.int_to_binary(unoccupied, width=r)
    if zeroes_to_append:
        w = np.concatenate([w, np.zeros(zeroes_to_append, dtype=int)])
    return w[:, None]


def _step_3_in_find_w(bits_basis, other_bits):
    """Step 3 of _find_w, which is triggered when t=1 in the recursion but initially we had t>1."""
    r = bits_basis.shape[0]
    all_bits = np.concatenate([bits_basis, other_bits], axis=1)
    # Compute set(V) ∪ (set(V) - set(V)). We will skip 0 by starting our search at 1 below
    diffs = np.array([(v_i - v_j) for v_i, v_j in combinations(all_bits.T, r=2)]).T % 2
    all_bitstrings_to_avoid = np.concatenate([all_bits, diffs], axis=1)

    # Note that the set ``all_bitstrings_to_avoid`` has size at most D+(D^2-D)/2=(D^2+D)/2
    # We want to find a new bitstring that avoids all these bit strings, so we need to iterate
    # over a range that is larger than (D^2+D)/2 by at least one.
    # For r<=m+1, we actually never call ``_find_w``, so that we know r≥m+2=2d+1. This implies:
    # 2^(r-1) ≥ D^2 > (D^2+D)/2 (for D>1),
    # so that 2^(r-1) candidates are enough to find a new vector.
    for i in range(1, 2 ** (r - 1)):
        w_bits_in_basis = math.int_to_binary(i, width=bits_basis.shape[1] - 1)
        w = (bits_basis[:, :-1] @ w_bits_in_basis) % 2
        if not np.any(np.all(w[:, None] == all_bitstrings_to_avoid, axis=0)):
            return w[:, None]

    raise ValueError("Unexpected exception: There should always be a w found.")  # pragma: no cover


def _find_w(bits_basis, other_bits, t):
    """Compute the kernel space W from the original bit strings, including a basis for their
    column space. See the documentation of ``compute_sos_encoding`` for details.
    """
    if t == 1:
        # Step 3: brute-force search of a single vector w_1
        return _step_3_in_find_w(bits_basis, other_bits)

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
    ell = _find_ell(bits_basis, set_M, set_N)

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
    r"""Map :math:`D` different bitstrings of length :math:`r` to :math:`D` different
    bitstrings :math:`b` of length :math:`m = \min(r, 2d-1)` where
    :math:`d=\lceil\log_2(D)\rceil`. The function computes both the mapping :math:`U` and the
    output bitstrings :math:`b`.
    This algorithm forms the constructive proof of Lemma 1 in
    `Fomichev et al., PRX Quantum 5, 040339 <https://doi.org/10.1103/PRXQuantum.5.040339>`__.
    It is the main classical coprocessing step required for the sparse state preparation
    (sum-of-Slaters preparation) presented in this paper, enabling its resource efficiency.

    Args:
        bits (np.ndarray): Bitstrings of length :math:`r` that are input into Lemma 1. The i-th
            bitstring :math:`v_i` is stored in the :math:`i`-th column, so that for :math:`D`
            bitstrings, the input shape is :math:`(r, D)`.

    Returns:
        tuple[np.ndarray]: Two bit arrays. The first is :math:`U`, which maps the input ``bits``
        to :math:`D` distinct bitstrings :math:`\{b_i\}` of length :math:`\min(r, m)`, where
        :math:`m=2\lceil \log_2(D)\rceil-1`. The second array are the bitstrings
        :math:`\{b_i\}` themselves, stored as columns.

    .. warning::

        It is recommended to first subselect bits via :func:`~.select_sos_rows` in order to
        work with a reduced input here.
        Furthermore, this function assumes that the bitstrings are not overly redundant, so
        that it might error out if ``select_sos_rows`` is not used.

    .. seealso:: :func:`~.select_sos_rows`

    **Example**

    Consider an array of bits with distinct columns:

    >>> bits = np.array([
    ...     [0, 1, 1, 0, 1, 0, 0],
    ...     [0, 0, 1, 1, 1, 0, 0],
    ...     [1, 1, 1, 1, 1, 1, 1],
    ...     [0, 0, 1, 0, 0, 0, 1],
    ...     [1, 1, 0, 0, 0, 0, 1],
    ...     [0, 0, 0, 0, 0, 1, 1],
    ... ])
    >>> from pennylane.templates.state_preparations.sum_of_slaters import (
    ...     compute_sos_encoding, _columns_differ
    ... )
    >>> print(_columns_differ(bits))
    True
    >>> print(bits.shape)
    (6, 7)

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
     [[0 1 1 0 1 0 0]
     [0 0 1 1 1 0 0]
     [1 1 1 1 1 1 1]
     [0 0 1 0 0 0 1]
     [1 1 0 0 0 0 1]]
    >>> print(_columns_differ(b))
    True

    The encoded bitstrings ``b`` are provided for convenience. They can equivalently be computed
    from ``U`` and the input ``bits`` via ``(U @ bits) % 2``:

    >>> np.array_equal((U @ bits) % 2, b)
    True

    Note that in this particular example, we could have achieved the reduction simply by selecting
    :math:`4<m` rows of the input bits, still obtaining different bitstrings. There is a function
    that does just that:

    >>> from pennylane.templates.state_preparations.sum_of_slaters import select_sos_rows
    >>> select_ids, sub_bits = select_sos_rows(bits)
    >>> print(sub_bits)
    [[0 1 1 0 1 0 0]
     [0 0 1 1 1 0 0]
     [0 0 1 0 0 0 1]
     [1 1 0 0 0 0 1]]

    In practice, this sub-selection of bits via ``select_sos_rows`` is combined with
    ``compute_sos_encoding`` to achieve lowest cost. Note that there may be edge cases where
    ``compute_sos_encoding`` errors out if ``select_sos_rows`` is not used before, because
    the input bitstrings are too redundant in this case.

    .. details::
        :title: Implementation notes

        We are given :math:`D` distinct bitstrings :math:`\{v_i\}` with length :math:`r`.
        We assume :math:`D\geq r` and :math:`\operatorname{rank}(V)\geq r`, which can always
        be achieved by first calling ``select_sos_rows`` on the bitstrings.

        Our goal is to find a linear map :math:`U:\mathbb{Z}_2^{r}\to \mathbb{Z}_2^{m}` from
        the input bitstrings to :math:`D` new distinct bitstrings with length
        :math:`m\leq 2d-1`, where :math:`d:=\lceil\log_2(D)\rceil`, such that

        .. math::

            U(v_i-v_j)\neq 0 \ \ \forall i, j, \quad\text{and}\quad
            U(v_i)\neq 0 \ \ \forall i \quad\text{unless}\quad v_i=0.

        It will be instructive to rewrite this as

        .. math::

            v_i-v_j\not\in \ker U \ \ \forall i, j, \quad\text{and}\quad
            v_i\not\in \ker U \ \ \forall i \quad\text{unless}\quad v_i=0.\qquad(1)

        We will speak of :math:`U` and its matrix representation of zeros and ones interchangeably.
        Since :math:`k` bits can represent at most :math:`2^k` different bitstrings, we know that
        :math:`D` different bitstrings :math:`\{v_i\}` require at least :math:`d` bits to
        be represented, i.e. we know that :math:`r\geq d`. We will proceed in two cases from
        here on, differentiated by :math:`r`.

        **Case 1:** :math:`d\leq r\leq 2d-1`

        In this case, we do not need to do anything; the bitstrings :math:`\{v_i\}` already
        have length :math:`m:=r\leq 2d-1`, so we simply set :math:`U` to be the identity map.
        This scenario may actually occur in practice, and it leads to simplifications of the
        quantum circuit for the state preparation. This depends on the specific bitstrings, though.
        This case is handled directly in the main function of ``compute_sos_encoding``.

        **Case 2:** :math:`2d-1 < r`

        Fix :math:`m=2d-1` and define :math:`t:=r-m` so that :math:`r=m+t`. According to the rank
        theorem, any candidate linear map :math:`U` satisfies :math:`\dim (\mathrm{Im} U) + \dim (\ker U)=r`.
        If we guarantee linear independence of the rows of :math:`U`, we know that
        :math:`\dim (\mathrm{Im} U)` matches the dimensions of the target space, :math:`m`, and thus
        :math:`\dim (\ker U)=r-m=t`.

        Our strategy now will be to find :math:`t` linearly independent vectors :math:`\{w_k\}`
        such that the space :math:`\mathcal{W}:=\operatorname{span}\{w_1, \dots w_t\}` spanned by them satisfies

        .. math::

            v_i-v_j\not\in \mathcal{W} \ \ \forall i, j, \quad\text{and}\quad
            v_i\not\in \mathcal{W} \ \ \forall i \quad\text{unless}\quad v_i=0.\qquad (2)

        and to construct a map :math:`U` with linearly independent rows such that
        :math:`U w_k=0 \ \ \forall k`, i.e. :math:`\mathcal{W}\subset\ker U`. Given that we know the
        kernel dimension to be :math:`t` and :math:`\dim(\mathcal{W})=t`, this will imply
        :math:`\mathcal{W}=\ker U`.

        .. admonition:: Math comment

            To see that this strategy actually ensures :math:`U` to have the
            properties we are after, assume that :math:`U v_i=0` for some :math:`i` with
            :math:`v_i\neq 0` (or :math:`U(v_i-v_j)` for some :math:`(i,j)`). Due to
            :math:`\ker U=\mathcal{W}`, this would imply :math:`v_i\in\mathcal{W}` (or
            :math:`v_i-v_j\in\mathcal{W}`), which is false by construction of the vectors
            :math:`\{w_k\}`, in particular due to Eq. (2).

        The main work thus is to actually construct these vectors :math:`\{w_k\}`
        with the required properties. The construction of the map :math:`U` itself will then be a
        simple linear algebraic task. We perform the construction of the vectors iteratively,
        corresponding to a proof by induction on :math:`t`.

        Given :math:`D` vectors :math:`\{v_i\}` of length :math:`r` with rank :math:`r` (i.e.,
        there are at least :math:`r` linearly independent vectors), our task is to build
        :math:`t=r-(2\lceil \log_2 (D)\rceil -1)` linearly independent vectors :math:`\{w_k\}`
        from the space :math:`\operatorname{span}\{v_i\}` such that the
        resulting vector space :math:`\mathcal{W}=\operatorname{span}\{w_k\}` does not contain the
        :math:`\{v_i\}` or their pairwise differences, see Eq.(2).
        We can again separate two scenarios.

        **Case 2a:** :math:`t=1`

        For this case, there is a particularly simple method: we can brute-force a search of
        :math:`w_1` over :math:`\mathbb{Z}_2^r\setminus (\{v_i\}\cup \{v_i-v_j\})`. This is
        implemented in ``_find_single_w``. See "Computing ``U``" below for the remaining
        thing we need to do in this case.

        **Case 2b:** :math:`t>1`

        If :math:`t>1`, we recursively construct the :math:`\{w_k\}`, which is implemented in
        ``_find_w``.
        We proceed in the following steps, thinking of ordered sets whenever we speak of sets.

        1. First, we select :math:`r` of the input vectors :math:`\mathcal{V}=\{v_i\}` that are
           linearly independent (and thus form a--likely not orthonormal--basis of
           :math:`\mathbb{Z}_2^r`). We relabel the vectors so that this selection of vectors has
           the indices :math:`\{1,\dots, r\}`, i.e. the basis is :math:`\mathcal{B}=\{v_1,\dots,v_r\}`.
           This is implemented in ``qml.math.binary_select_basis``, which returns the basis and
           the remaining columns separately. This step will only ever be executed once.
        2. If :math:`t=1` (which can happen despite :math:`t>1` initially, because we will use
           recursion), go to step 3. Else go to step 4.
        3. Brute-force search a linear combination :math:`w_1` of the basis vectors
           :math:`\mathcal{B}` that is not contained in the set
           :math:`\mathcal{V}\cup(\mathcal{V}-\mathcal{V})\cup \{0\}`, where the set difference
           is meant pairwise between elements. Return :math:`\{w_1\}`.
           This is implemented in ``_step_3_in_find_w``.
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

        **Computing** ``U``

        Computing the matrix :math:`U` such that the vectors :math:`\{w_k\}` are in its kernel is
        rather simple. This is because the problem is self-dual, i.e., we actually need to find
        vectors in the kernel of :math:`W^T`, and will construct :math:`U` from the kernel vectors
        as rows. The same is true for case 2b), where we only had to compute a single :math:`w_1`.
    """
    r, D = bits.shape
    d = math.ceil_log2(D)
    m = 2 * d - 1
    if r <= m:
        # Case 1: We can use the identity mapping
        U = np.eye(r, dtype=int)
        return U, bits

    # Case 2 splits into two: case 2a): t=1 (simple) and case 2b): t>1 (more involved)
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
        coefficients (np.ndarray): Coefficients of the sparse state to prepare. The ordering should
            match that in ``indices``.
        wires (qml.wires.WiresLike): Wires on which to prepare the state. All work wires will be
            allocated dynamically with :func:`~.allocate`.
        indices (tuple[int]): Indices of the sparse state to prepare. The ordering should match
            that in ``coefficients``.

    .. warning::

        Note that we require ``coefficients`` to be treated as numerical data in the form of an
        array, whereas the ``indices`` need to be hashable, and thus will be treated as static
        information. This is because ``indices`` significantly impacts the structure and size of
        the circuit that realizes the state preparation.

    **Example**

    Consider a sparse state specified by normalized coefficients and statevector
    indices pointing to the populated computational basis states:

    >>> coefficients = np.array([0.25, 0.25j, -0.25, 0.5, 0.5, 0.25, -0.25j, 0.25, -0.25, 0.25])
    >>> indices = (0, 1, 4, 13, 14, 17, 19, 22, 23, 25)

    Given that we have the index ``25`` in ``indices``, we need at least five qubits (:math:`2^5 = 32`) to represent
    this state (in practical use cases, the number of qubits is typical given from context). This is all the information we require to create the state preparation. Let's take a look at how the preparation is implemented:

    >>> qml.decomposition.enable_graph()
    >>> decomp_call = qml.decompose(qml.SumOfSlatersPrep, max_expansion=1, num_work_wires=10)
    >>> decomp_call = qml.transforms.resolve_dynamic_wires(decomp_call, zeroed=range(5, 15))
    >>> print(qml.draw(decomp_call, show_matrices=False)(coefficients, wires, indices))
     0: ──────╭QROM(M0)─╭○─╭○─╭○─╭○─╭○─╭●─╭●─╭●─╭●─╭●──────────╭●─╭●─╭●─╭●─┤
     1: ──────├QROM(M0)─├○─├○─├●─├●─├●─├○─├○─├○─├○─├○──────────├○─├○─├●─├●─┤
     2: ──────├QROM(M0)─├○─├●─├●─├●─├●─├○─├○─├○─├○─├●──────────├●─├●─├○─├○─┤
     3: ──────├QROM(M0)─├○─├○─├○─├○─├●─├○─├○─├●─├●─├●──────────├●─├●─├○─├○─┤
     4: ──────├QROM(M0)─├●─├○─├●─├●─├○─├●─├●─├●─├●─├○──────────├○─├●─├●─├●─┤
     7: ──────│─────────│──│──│──│──│──│──│──│──│──╰X─╭●─╭●─╭●─╰X─│──│──│──┤
     8: ──────├QROM(M0)─│──│──│──│──│──│──│──│──│─────│──│──│─────│──│──│──┤
     9: ──────├QROM(M0)─│──│──│──│──│──│──│──│──│─────│──│──│─────│──│──│──┤
    10: ──────├QROM(M0)─│──│──│──│──│──│──│──│──│─────│──│──│─────│──│──│──┤
    11: ─╭|Ψ⟩─├QROM(M0)─╰X─│──│──╰X─│──│──╰X─│──│─────│──│──╰X────│──│──╰X─┤
    12: ─├|Ψ⟩─├QROM(M0)────╰X─╰X────│──│─────│──╰X────│──╰X───────│──│─────┤
    13: ─├|Ψ⟩─├QROM(M0)─────────────╰X─╰X────╰X───────╰X──────────│──│─────┤
    14: ─╰|Ψ⟩─╰QROM(M0)───────────────────────────────────────────╰X─╰X────┤
    .. code-block:: python3
        
        qml.decomposition.enable_graph()
        wires = qml.wires.Wires(range(5))

        @qml.transforms.resolve_dynamic_wires(min_int=max(wires)+1)
        @qml.decompose(gate_set={"QROM", "MultiControlledX", "StatePrep", "CNOT"}, num_work_wires=10)
        @qml.qnode(qml.device("lightning.qubit", wires=wires))
        def circuit():
            qml.SumOfSlatersPrep(coefficients, wires, indices)
            return qml.state()
            
    >>> print(qml.draw(circuit, show_matrices=False)())
    0: ──────╭QROM(M0)─╭○─╭○─╭○─╭○─╭○─╭●─╭●─╭●─╭●─╭●──────────╭●─╭●─╭●─╭●─┤  State
    1: ──────├QROM(M0)─├○─├○─├●─├●─├●─├○─├○─├○─├○─├○──────────├○─├○─├●─├●─┤  State
    2: ──────├QROM(M0)─├○─├●─├●─├●─├●─├○─├○─├○─├○─├●──────────├●─├●─├○─├○─┤  State
    3: ──────├QROM(M0)─├○─├○─├○─├○─├●─├○─├○─├●─├●─├●──────────├●─├●─├○─├○─┤  State
    4: ──────├QROM(M0)─├●─├○─├●─├●─├○─├●─├●─├●─├●─├○──────────├○─├●─├●─├●─┤  State
    5: ─╭|Ψ⟩─├QROM(M0)─│──│──│──│──│──│──│──│──│──│───────────│──╰X─╰X─│──┤  State
    6: ─├|Ψ⟩─├QROM(M0)─│──│──│──│──╰X─╰X─│──╰X─│──│──╭X───────│────────│──┤  State
    7: ─├|Ψ⟩─├QROM(M0)─│──╰X─╰X─│────────│─────╰X─│──│──╭X────│────────│──┤  State
    8: ─╰|Ψ⟩─├QROM(M0)─╰X───────╰X───────╰X───────│──│──│──╭X─│────────╰X─┤  State
    9: ──────├QROM(M0)────────────────────────────│──│──│──│──│───────────┤  State
   10: ──────├QROM(M0)────────────────────────────│──│──│──│──│───────────┤  State
   11: ──────╰QROM(M0)────────────────────────────│──│──│──│──│───────────┤  State
   12: ───────────────────────────────────────────╰X─╰●─╰●─╰●─╰X──────────┤  State
    Note that wires with labels ``5`` to ``12`` were dynamically allocated. We can see an initial dense state preparation via :class:`~.StatePrep` on fewer qubits (depicted as ``|Ψ⟩`` on the first four dynamic wires in the above diagram),
    a :class:`~.QROM` and a sequence of :class:`~.MultiControlledX` gates, some of which are
    mediated with a caching qubit (qubit index ``12``) and :class:`~.CNOT` gates.

    Note that we guessed the required number of work wires (``num_work_wires``) in  :func:`~.decompose` and employed :func:`~.transforms.resolve_dynamic_wires` to assign integer wire labels to those dynamically allocated wires. If we want to know
    the required wire register sizes ahead of time, they can be computed with
    ``SumOfSlatersPrep.required_register_sizes``:

    >>> prep_op = qml.SumOfSlatersPrep(coefficients, wires, indices)
    >>> prep_op.required_register_sizes(**prep_op.resource_params)
    {'wires': 5,
     'enumeration_wires': 4,
     'identification_wires': 0,
     'qrom_work_wires': 3,
     'mcx_cache_wires': 1}

    .. admonition:: Gotchas of reported work register sizes

        Note that these register sizes might be upper bounds in some scenarios, and that further
        decomposing the circuit efficiently may require additional work wires, for example for
        the ``MultiControlledX`` gates. In contrast, the QROM work wires are explicitly accounted for,
        which is due to some internal technical limitation.

    """

    resource_keys = {"D", "num_bits", "num_wires"}

    @property
    def resource_params(self):
        indices = self.hyperparameters["indices"]
        n = len(self.wires)
        v_bits = math.int_to_binary(np.array(indices), n).T
        selector_ids, _ = select_sos_rows(v_bits)
        return {"D": len(indices), "num_bits": len(selector_ids), "num_wires": n}

    def __init__(self, coefficients, wires, indices):
        super().__init__(coefficients, wires)
        self.hyperparameters["indices"] = indices

    @property
    def has_decomposition(self):
        """We are using ``qml.allocate`` in the decomposition, so the validation for
        decomposition in the old system breaks. Hence we manually deactivate the fallback
        of ``compute_decomposition`` to the new decomp system that is implemented in
        ``Operator.compute_decomposition``. Accordingly we set ``has_decomposition=False`` here."""
        return False

    @staticmethod
    def compute_decomposition(coefficients, wires, indices):  # pylint: disable=arguments-differ
        """We are using ``qml.allocate`` in the decomposition, so the validation for
        decomposition in the old system breaks. Hence we manually deactivate the fallback
        of ``compute_decomposition`` to the new decomp system that is implemented in
        ``Operator.compute_decomposition``."""
        raise DecompositionUndefinedError

    @staticmethod
    def required_register_sizes(D, num_bits, num_wires):
        """Compute the register sizes required for ``SumOfSlatersPrep``, for given
        numbers of bitstrings ``D``, of bits per bitstring (``num_bits``, already reduced via
        ``select_sos_rows``) and target wires (``num_wires``).

        Args:
            D (int): Number of bitstrings encoded by ``SumOfSlatersPrep``.
            num_bits (int): Number of bits per bitstring.
            num_wires (int): Number of target wires on which ``SumOfSlatersPrep`` will prepare
                the state.

        Returns:
            dict[str, int]: Required register size per register name

        """
        if D == 1:
            # Simple computational basis state preparation, does not require auxiliary qubits
            return {
                "wires": num_wires,
                "enumeration_wires": 0,
                "identification_wires": 0,
                "qrom_work_wires": 0,
                "mcx_cache_wires": 0,
            }

        d = math.ceil_log2(D)
        m = 2 * d - 1
        if num_bits <= m:
            # Identity encoding. We do not need the identification register but can use the
            # (subselection of) system wires directly
            num_identification = 0
        else:
            # Non-identity encoding, we need 2d-1 auxiliary qubits for the identification register
            num_identification = m

        # If D<=7, we only have encoded bits with bit count at most 2, so that we will not use
        # a cache qubit for the MultiControlledX ops.
        num_mcx_cache = int(D > 7)

        return {
            "wires": num_wires,
            "enumeration_wires": d,
            "identification_wires": num_identification,
            "qrom_work_wires": d - 1,
            "mcx_cache_wires": num_mcx_cache,
        }


def _sos_state_prep_resources(D, num_bits, num_wires):
    """Compute the resources for _sos_state_prep. These are upper-bounded resources due to
    the way MultiControlledX gates are accounted for at the moment.
    We can remedy this once [sc-110068] is completed."""
    if D == 1:
        return {resource_rep(qml.BasisState, num_wires=num_wires): 1}
    d = math.ceil_log2(D)
    m = min(num_bits, 2 * d - 1)

    identity_encoding = num_bits == m

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

    if not identity_encoding:
        ## Step 3 & 4:
        resources[resource_rep(qml.CNOT)] += m * num_wires  # size {u_k} * bits in u_k

    mcx_params = {
        "num_work_wires": 0,  # Work wires will be allocated by MCX itself
        "work_wire_type": "borrowed",
        "num_control_wires": m,
        "num_zero_control_values": m,
    }
    mcx_rep = resource_rep(qml.MultiControlledX, **mcx_params)

    # Calculate the bit counts of all integers that need to be uncomputed. Depending on the bit
    # count, we need to apply one or two MCX gates or two MCX and multiple CNOT gates, see below
    bit_counts = np.bitwise_count(np.arange(1, D)).astype(int)
    counts = dict(zip(*np.unique(bit_counts, return_counts=True)))

    # If k is a power of two, we can directly use an MCX gate
    resources[mcx_rep] += counts.pop(1, 0)
    # If k is a sum of two powers of two, we can directly use two MCX gates
    resources[mcx_rep] += 2 * counts.pop(2, 0)
    # If k has more than 2 bits set, it is cheaper to first flip an aux bit and use that as
    # control to flip the targets via ``bit_count`` many CNOTs
    resources[mcx_rep] += 2 * sum(counts.values())
    for bit_count, count in counts.items():
        resources[resource_rep(qml.CNOT)] += count * bit_count

    ## Step 5:
    # TODO [dwierichs]: Revisit the following "hack" once [sc-110068] is completed.
    # We have to hack the MultiControlledX resources a little bit, even with exact=False.
    # This is because MultiControlledX resources with differing `num_zero_control_values`
    # are considered different by the framework. Overall we accounted for all MultiControlledX
    # instances as having the maximal number of zeroed control values above.
    # Here we replace m of them with all possible instances of zeroed control values. If we would
    # be reaching negative counts, we simply add more MCX gates, leading to an upper bound instead
    resources[mcx_rep] = max(resources[mcx_rep] - m, 1)
    for num_zeroed in range(m):
        if num_zeroed == 0 and m == 2:
            # For this special case PL casts to Toffoli instead of MCX
            resources[resource_rep(qml.Toffoli)] += 1
        else:
            mcx_params["num_zero_control_values"] = num_zeroed
            resources[resource_rep(qml.MultiControlledX, **mcx_params)] += 1

    ## Step 6:
    if not identity_encoding:
        resources[resource_rep(qml.CNOT)] += m * num_wires  # size {u_k} * bits in u_k

    return resources


def _sos_state_prep_work_wires(D, num_bits, num_wires):
    """See SumOfSlatersPrep.required_register_sizes for details."""
    sizes = SumOfSlatersPrep.required_register_sizes(D, num_bits, num_wires)
    return {"zeroed": sum(sizes.values()) - num_wires}


def _preprocess(v_bits, wires):
    """Preprocess the bits for SumOfSlatersPrep and compute some characterizing integers."""
    D = v_bits.shape[1]
    # if selector_ids has length r, vtilde_bits has shape (r, D)
    selector_ids, vtilde_bits = select_sos_rows(v_bits)
    selected_target_wires = [wires[idx] for idx in selector_ids]
    # u_bits has shape (2d-1, r), b_bits has shape (2d-1, D)
    u_bits, b_bits = compute_sos_encoding(vtilde_bits)

    r = len(vtilde_bits)
    d = math.ceil_log2(D)
    m = min(r, 2 * d - 1)
    assert u_bits.shape == (m, r), f"{u_bits.shape=}, {(m, r)=}"
    assert b_bits.shape == (m, D)

    return selected_target_wires, u_bits, b_bits, d, m, r


@register_resources(_sos_state_prep_resources, exact=False, work_wires=_sos_state_prep_work_wires)
def _sos_state_prep(coefficients, wires, indices, **__):
    """Compute the decomposition of the sum-of-Slaters state preparation technique."""
    # pylint: disable=no-value-for-parameter
    n = len(wires)
    D = len(indices)
    v_bits = math.int_to_binary(np.array(indices), n).T  # Shape (n, D)
    if D == 1:
        qml.BasisState(v_bits[:, 0], wires=wires)
        return
    assert v_bits.shape == (n, D)

    selected_target_wires, u_bits, b_bits, d, m, r = _preprocess(v_bits, wires)
    identity_encoding = r == m

    sizes = SumOfSlatersPrep.required_register_sizes(D, r, n)
    all_allocate_wires = sum(sizes.values()) - n
    with allocate(all_allocate_wires, state="zero", restored=True) as allocated:
        start = 0
        # There is no implementation of QROM with allocate yet, so we allocate its work wires here
        enumeration_wires = allocated[start : (start := start + sizes["enumeration_wires"])]
        identification_wires = allocated[start : (start := start + sizes["identification_wires"])]
        qrom_work_wires = allocated[start : (start := start + sizes["qrom_work_wires"])]
        mcx_cache_wires = allocated[start : (start := start + sizes["mcx_cache_wires"])]
        # Step 1: Dense state preparation in enumeration register
        qml.StatePrep(coefficients, wires=enumeration_wires, pad_with=0.0)

        # Step 2: QROM to load v_bits into system register
        qml.QROM(
            v_bits.T,
            control_wires=enumeration_wires,
            target_wires=wires,
            work_wires=qrom_work_wires,
        )

        if not identity_encoding:
            # Step 3-4): Encode the b_bits from Lemma 1 in the identification register
            # Note that we skip this step if identity_encoding=True, because the encoding is
            # trivial in this case. This is an additional optimization compared to the paper.
            @for_loop(m)
            def encoding(i):
                u = u_bits[i]

                @for_loop(r)
                def inner_loop(j):
                    qml.cond(u[j], qml.CNOT)([selected_target_wires[j], identification_wires[i]])

                inner_loop()

            encoding()

        # Step 5): Use the identification register to uncompute the enumeration register
        mcx_ctrl_wires = selected_target_wires if identity_encoding else identification_wires

        @for_loop(1, D)
        def uncompute_enumeration(k):
            bits = list(map(int, b_bits[:, k]))

            bit_count = np.bitwise_count(k)
            if bit_count == 1:
                # If k is a power of two, we can directly use an MCX gate
                # This is an additional optimization compared to the paper, saving one MCX gate
                target = math.ceil_log2(k)
                qml.MultiControlledX(
                    wires=qml.wires.Wires.all_wires([mcx_ctrl_wires, enumeration_wires[~target]]),
                    control_values=bits,
                )
            elif bit_count == 2:
                # If k is a sum of two powers of two, we can directly use two MCX gates
                # This is an additional optimization compared to the paper, saving aux wire usage
                target0 = math.ceil_log2(k) - 1
                target1 = math.ceil_log2(k - 2**target0)
                for target in [target0, target1]:
                    qml.MultiControlledX(
                        wires=qml.wires.Wires.all_wires(
                            [mcx_ctrl_wires, enumeration_wires[~target]]
                        ),
                        control_values=bits,
                    )
            else:
                # If k has more than 2 bits set, it is cheaper to first flip an aux bit and use that as
                # control to flip the targets
                mcx_kwargs = {
                    "wires": qml.wires.Wires.all_wires([mcx_ctrl_wires, mcx_cache_wires]),
                    "control_values": bits,
                }

                qml.MultiControlledX(**mcx_kwargs)

                @for_loop(d)
                def inner_loop(j):
                    bit_is_set = (k >> (d - 1 - j)) & 1
                    qml.cond(bit_is_set, qml.CNOT)([mcx_cache_wires[0], enumeration_wires[j]])

                inner_loop()

                qml.MultiControlledX(**mcx_kwargs)

        uncompute_enumeration()

        # Step 6): Uncompute the b_i in the identification register (self-adjoint)
        if not identity_encoding:
            encoding()


add_decomps(SumOfSlatersPrep, _sos_state_prep)
