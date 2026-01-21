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
r"""Contains the SumOfSlatersStatePreparation template."""

from itertools import combinations, product

import numpy as np

from pennylane.math import ceil_log2
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
    # Powers of two to compute integer representations of columns
    pot = 2 ** np.arange(bits.shape[0])
    # Compute the integer representation of each column
    ints = np.dot(pot, bits)
    # Return whether the integers all differ
    return len(set(ints)) == len(ints)


def _select_rows(bits: np.ndarray) -> tuple[list[int], np.ndarray]:
    r"""Select rows of a bit array of differing columns such that the stacked array of the
    selected rows still contains differing columns. Also memorizes the row indices of the input
    array that were selected.

    Args:
        bits (np.ndarray): Bit array with differing columns

    Returns:
        tuple[list[int], np.ndarray]: Selected row indices to obtain the reduced bit array,
        and the reduced bit array itself. If ``bits`` had shape ``(n_rows, n_cols)`` and the list
        of row indices has length ``r``, the reduced bit array has shape ``(r, n_cols)``.

    Under the hood, this method is not selecting rows, but instead greedily removes rows from the
    input. We first attempt to remove rows with a mean weight far away from 0.5, as they are
    generically less likely to differentiate many columns from each other.

    **Example**

    Let's generate a random bit array of ``D=8`` differing columns of length ``n``, by first
    sampling unique integers from the range ``(0, 2**n)`` and converting them to bitstrings.

    >>> np.random.seed(355)
    >>> ids = np.random.choice(2**n, size=D, replace=False)
    >>> bitstrings = ((ids[:, None] >> np.arange(n-1, -1, -1)[None, :]) % 2).T
    >>> bitstrings
    [[0 1 1 0 1 0 0 0]
     [0 0 0 0 0 0 1 1]
     [1 0 1 0 0 1 1 1]
     [1 1 0 0 0 0 1 0]
     [0 0 0 0 0 1 0 0]
     [0 0 0 1 0 1 0 1]]

    Then let's select rows that maintain the uniqueness of the rows:

    >>> selectors, new_bits = _select_rows(bitstrings)
    >>> seletors
    [1, 2, 3, 5]

    Indeed, selecting the indicated rows of ``bitstrings``, we still find
    unique columns:

    >>> new_bits
    [[0 0 0 0 0 0 1 1]
     [1 0 1 0 0 1 1 1]
     [1 1 0 0 0 0 1 0]
     [0 0 0 1 0 1 0 1]]

    In general, the number of rows :math:`r` selected by the method will satisfy
    :math:`\log_2(n_{\text{col}})\leq r\leq \min(n_{\text{row}}, n_{\text{col}})` if
    :math:`(n_{\text{row}}, n_{\text{col}})` is the shape of the input array.

    """
    selectors = list(range(len(bits)))

    while True:
        # compute weight of each row. We'll try to first remove rows with a
        # mean weight far away from 0.5
        weights = np.mean(bits, axis=1)
        ordering = np.argsort(np.abs(0.5 - weights))
        x = np.abs(0.5 - weights)
        for i in reversed(ordering):
            # Check whether the array with row ``i`` removed still has unique columns
            _bits = np.concatenate([bits[:i], bits[i + 1 :]])
            if _columns_differ(_bits):
                # If the columns remain unique, remove the row and the row index from selectors
                del selectors[i]
                bits = _bits
                break
        else:
            # If no row could be removed, stop the process
            break

    return selectors, bits


def _rank_over_z2(bits):
    """
    # Source - https://stackoverflow.com/a
    # Posted by Mark Dickinson, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-01-15, License - CC BY-SA 4.0

    Find rank of a matrix over Z_2.

    The rows of the matrix are given as nonnegative integers, thought
    of as bit-strings.

    This function modifies the input list. Use _rank_over_z2(rows.copy())
    instead of _rank_over_z2(rows) to avoid modifying rows.
    """
    assert isinstance(bits, np.ndarray) and bits.ndim == 2
    num_rows, num_bits = bits.shape
    if num_bits > 64:
        if num_rows <= 64:
            return _rank_over_z2(bits.T)
        raise NotImplementedError(
            "Rank computation requires one of the axes of the input bits to be smaller "
            f"than or equal to 64. Got input shape {bits.shape}"
        )

    ints = list(np.array(bits) @ 2 ** np.arange(len(bits[0]) - 1, -1, -1))
    rank = 0
    while ints:
        pivot_row = ints.pop()
        if pivot_row:
            rank += 1
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(ints):
                if row & lsb:
                    ints[index] = row ^ pivot_row
    return rank


def _lin_indep(col, new_cols, new_cols_rank=None):
    assert isinstance(col, np.ndarray) and col.ndim == 1, f"{col.ndim=}"
    r = col.shape[0]
    assert isinstance(new_cols, np.ndarray) and new_cols.ndim == 2, f"{new_cols.shape=}"
    assert new_cols.shape[1] == r, f"{new_cols.shape=}, {r=}"
    if new_cols_rank is None:
        new_cols_rank = _rank_over_z2(new_cols)
    rk = _rank_over_z2(np.concatenate([new_cols, [col]]))
    return rk > new_cols_rank


def _get_bits_basis(bits: np.ndarray):
    """Select bit strings from a set of bitstrings that form a basis
    for the column space.

    Args:
        bits (np.ndarray): Input bitstrings.
    """
    r, D = bits.shape
    basis = np.zeros((0, r), dtype=int)
    other_cols = []
    basis_rank = 0
    for i, col in enumerate(bits.T):
        if basis_rank < r and _lin_indep(col, basis, basis_rank):
            basis = np.concatenate([basis, [col]])
            basis_rank += 1
        else:
            other_cols.append(col)

    if other_cols == []:
        other_cols = np.zeros((0, r), dtype=int)
    else:
        other_cols = np.array(other_cols)
    return basis.T, np.concatenate([basis[:-1], other_cols]).T


def _find_ell(set_M: np.ndarray, set_N: np.ndarray, bits_basis: np.ndarray) -> np.ndarray:
    r"""Find replacement vector :math:`\ell` to construct lower-rank set of bit strings during
    recursive computation of kernel space :math:`\mathcal{W}` in ``_get_w_vectors``."""

    r = len(bits_basis)
    v_r = bits_basis[:, -1]
    if set_M.shape[1] == 0 or set_N.shape[1] == 0:
        combinations = np.zeros((r, 0), dtype=int)
    else:
        combinations = np.array(
            [m + m_prime_p_v_r + v_r for m, m_prime_p_v_r in product(set_M.T, set_N.T)]
        ).T
    zero = np.zeros((r, 1), dtype=int)
    all_bitstrings_to_avoid = (
        np.concatenate([set_M, set_N + v_r[:, None], combinations, zero], axis=1) % 2
    )

    for i in range(2 ** (r - 2)):
        ell_bits_in_basis = (i >> np.arange(bits_basis.shape[1] - 2, -1, -1)) % 2
        ell = (bits_basis[:, :-1] @ ell_bits_in_basis) % 2
        if not np.any(np.all(ell[:, None] == all_bitstrings_to_avoid, axis=0)):
            break
    else:
        raise ValueError()
    return ell
    pot = 2 ** np.arange(r - 2, -1, -1)
    ints = set(np.dot(pot, (bits_basis[:, :-1].T @ all_bitstrings_to_avoid) % 2))
    unoccupied = next(i for i in range(1, len(ints) + 1) if i not in ints)
    ell_bits = (unoccupied >> np.arange(r - 2, -1, -1)) % 2
    ell = (bits_basis[:, :-1] @ ell_bits) % 2
    return ell


def _bits_not_in_space(bits, W, incl_diffs=True):
    assert all(_lin_indep(bitstring, W.T) for bitstring in bits.T)
    if incl_diffs:
        assert all(
            _lin_indep((bits0 + bits1) % 2, W.T) for bits0, bits1 in combinations(bits.T, r=2)
        )


def _bits_in_space(bits, basis):
    assert _rank_over_z2(np.concatenate([bits, basis], axis=1)) == _rank_over_z2(basis)


def _get_w_vectors(bits, r, t, bits_basis=None):
    if t == 1:
        if bits_basis is None:
            # Power of two
            pot = 2 ** np.arange(r - 1, -1, -1)
            diffs = np.array([(v_i - v_j) for v_i, v_j in combinations(bits.T, r=2)]).T
            diffs = diffs % 2
            # Compute the integer representation of each column and each difference of columns
            ints = set(np.dot(pot, np.concatenate([bits, diffs], axis=1)))
            unoccupied = next(i for i in range(1, len(ints) + 1) if i not in ints)
            w = (unoccupied >> np.arange(r - 1, -1, -1)) % 2

        else:
            v_r = bits_basis[:, -1]
            diffs = np.array([(v_i - v_j) for v_i, v_j in combinations(bits.T, r=2)]).T
            zero = np.zeros((r, 1), dtype=int)
            all_bitstrings_to_avoid = np.concatenate([bits, diffs, zero], axis=1) % 2

            for i in range(1, 2 ** (r)):
                w_bits_in_basis = (i >> np.arange(bits_basis.shape[1] - 2, -1, -1)) % 2
                w = (bits_basis[:, :-1] @ w_bits_in_basis) % 2
                if not np.any(np.all(w[:, None] == all_bitstrings_to_avoid, axis=0)):
                    break
            else:
                raise ValueError()

        W = np.array([w]).T
        _bits_not_in_space(bits, W)
        if bits_basis is not None:
            _bits_in_space(W, bits_basis)
        return W

    if bits_basis is not None:
        # Basis known from previous iteration, and we stick to that basis choice
        bits_without_v_r = np.array(
            [col for col in bits.T if not np.allclose(col, bits_basis[:, -1])]
        ).T
    else:
        bits_basis, bits_without_v_r = _get_bits_basis(bits)

    v_r = bits_basis[:, -1]
    spanned_by_reduced_basis = np.array(
        [not _lin_indep(vec, bits_basis[:, :-1].T) for vec in bits_without_v_r.T]
    )

    set_M = bits_without_v_r[:, np.where(spanned_by_reduced_basis)[0]]
    _bits_in_space(set_M, bits_basis[:, :-1])

    set_N = bits_without_v_r[:, np.where(~spanned_by_reduced_basis)[0]]
    _bits_not_in_space(set_N, bits_basis[:, :-1], incl_diffs=False)
    ell = _find_ell(set_M, set_N, bits_basis)

    w_t = (v_r + ell) % 2
    _bits_not_in_space(np.array([w_t]).T, bits_basis[:, :-1], incl_diffs=False)

    _set_N = (set_N + w_t[:, None]) % 2
    _bits_in_space(_set_N, bits_basis[:, :-1])

    prev_bits = np.concatenate([set_M, ell[:, None], _set_N], axis=1)
    _bits_in_space(prev_bits, bits_basis[:, :-1])

    W_prev = _get_w_vectors(prev_bits, r, t - 1, bits_basis[:, :-1])
    _bits_in_space(W_prev, bits_basis[:, :-1])
    # _bits_in_space(W_prev[:, :-1], bits_basis[:, :-2]) # TO be re-added

    W = np.concatenate([W_prev, w_t[:, None]], axis=1)
    _bits_in_space(W, bits_basis)
    _bits_not_in_space(bits, W)
    """
    for i, v_i in enumerate(bits.T):
        for j, v_j in enumerate(bits.T[i+1:]):
            if not _lin_indep((v_i+v_j)%2, list(W.T)):
                print(i, j)
                print(f"{_lin_indep(v_i, bits_basis[:,:-1].T)=} (exp: False?)")
                print(f"{_lin_indep(v_j, bits_basis[:,:-1].T)=} (exp: False?)")
                print(f"{_lin_indep(w_t.flat, bits_basis[:,:-1].T)=} (exp: True)")
                print(f"{_lin_indep(v_r, bits_basis[:,:-1].T)=} (exp: True)")
                print(f"{_rank_over_z2(bits_basis[:, :-1])==_rank_over_z2(np.concatenate([W_prev, bits_basis[:, :-1]], axis=1))=} (exp: True)")
    print(f"{W.shape=}")
    print(f"{_rank_over_z2(W)=}")
    assert _rank_over_z2(bits_basis)==_rank_over_z2(np.concatenate([W, bits_basis], axis=1)), f"{_rank_over_z2(bits_basis)=},{_rank_over_z2(np.concatenate([W, bits_basis], axis=1))=}"
    """
    return W


def _find_U_from_W(W, r, m):

    # Create augmented matrix for Gauss-Jordan elimination.
    M = np.concatenate([W.T, np.eye(r, dtype=int)], axis=0)

    t = W.shape[1]
    for k in range(t):
        # Find next column with non-zero entry in ``k``th row
        i_max = next(iter(i for i in range(k, r) if M[k, i] == 1), None)
        # This only happens for underdetermined systems, whereas we assume M to be regular
        if i_max is None:
            raise ValueError(
                "Did not find next pivot in Gauss-Jordan elimination, indicating a singular "
                "matrix. Only regular matrices are supported by _solve_regular_linear_system_z2 "
                f"in RowCol. Extended matrix at time of error:\n{M}"
            )

        # Swap columns
        M[:, [k, i_max]] = M[:, [i_max, k]]

        # Iterate through columns and add current column with index ``k`` to them if they
        # have a 1 in the current row with index ``k``
        for i in range(r):
            if i == k:  # Exclude the ``k``th column itself
                continue
            if M[k, i] == 0:  # No need to do anything if the target entry already is zero
                continue
            # We use addition of rows modulo 2, which is implementable with bitwise xor, or ``^``.
            M[k:, i] ^= M[k:, k]

    # Solution is written into the last column of the (augmented) matrix
    U = M[t:, t:].T
    return U


def compute_sos_encoding(bits):
    """

    Args:
        bits (np.ndarray): Reduced bitstrings that are input into Lemma 1.
            The i-th bitstring v_i is stored in the i-th column.

    """
    r, D = bits.shape
    d = ceil_log2(D)
    m = 2 * d - 1
    print(f"{r=}, {D=}, {d=}, {m=}")
    if r <= m:
        print("Case 1")
        U = np.eye(r, dtype=int)
        b = bits
        return U, b

    t = r - m
    W = _get_w_vectors(bits, r, t)

    U = _find_U_from_W(W, r, m)
    assert all(_lin_indep(col, W.T) for col in bits.T)
    assert all(
        _lin_indep((col0 + col1) % 2, W.T) for col0, col1 in combinations(bits.T, r=2)
    ), f"Unchanged"
    assert np.allclose((U @ W) % 2, 0)
    b = (U @ bits) % 2
    return U, b


# class SumOfSlatersStatePreparation(Operation):


r"""Rewritten proof of Lemma 1.

_Proof._
Our goal is to find a linear map $U:\mathbb{Z}_2^{r}\to \mathbb{Z}_2^{m}$
from $D$ distinct bitstrings $\{v_i\}$ with length $r$ to $D$ bitstrings with length $m\leq 2d-1$, where $d:=\lceil\log_2(D)\rceil$, such that

$
U(v_i-v_j)\neq 0 \forall i, j, \text{and} U(v_i)\neq 0 \forall i \text{unless} v_i=0.
$

It will be instructive to rewrite this as

$
v_i-v_j\not\in \ker U \forall i, j, \text{and} v_i\not\in \ker U \forall i \text{unless} v_i=0.   (1)
$

We will speak of $U$ and its matrix representation of zeros and ones interchangeably.
Since $k$ bits can represent at most $2^k$ different bitstrings, we know that $D$ different bitstrings $\{v_i\}$ require at least $d$ bits to
be represented, i.e. we know that $r\geq d$.
We will proceed in two cases from here on, differentiated by $r$.

Case 1: $d\leq r\leq 2d-1$
==========================

In this case, we do not really need to do anything; the bitstrings $\{v_i\}$ already have length $m:=r\leq 2d-1$, so we simply set $U$ to be
the identity map.

Case 2: $2d-1 < r$
==================

Fix $m=2d-1$ and define $t:=r-m$ so that $r=m+t$.
According to the rank theorem, any candidate linear map $U$ satisfies $\dim \Im U + \dim \ker U=r$.
If we guarantee linear independence of the rows of $U$, we know that
$\dim \Im U$ matches the dimensions of the target space, $m$, and thus
$\dim \ker U=r-m=t$.

Our strategy now will be to find $t$ linearly independent vectors
$\{w_k\}$ such that the space $\mathcal{W}:=\span\{w_1, \dots w_t\}$ spanned by them satisfies

$
v_i-v_j\not\in \mathcal{W} \forall i, j, \text{and} v_i\not\in \mathcal{W} \forall i \text{unless} v_i=0.   (2)
$

and to construct a map $U$ with linearly independent rows such that $U w_k=0 \forall k$,
i.e. $\mathcal{W}\subset\ker U$. Given that we know the kernel dimension to be $t$ and
$\dim\mathcal{W}=t$, this implies $\mathcal{W}=\ker U$.
To see that this actually ensures $U$ to have the properties we are after, assume that $U v_i=0$
for some $i$ with $v_i\neq 0$ (or $U(v_i-v_j)$ for some $(i,j)$). Due to $\ker U=\mathcal{W}$,
this would imply $v_i\in\mathcal{W}$ (or $v_i-v_j\in\mathcal{W}$), which is false by
construction of the vectors $\{w_k\}$, in particular due to Eq.(2)

The main work thus is to show that we can actually construct these vectors
$\{w_k\}$ with the required properties. The construction of the map $U$ itself will then be a simple linear algebraic task.
We perform the construction of the vectors iteratively, corresponding to a proof by induction on $t$.
For the base case, we deviate from the proof in the paper and start with $t=0$.
The claim is trivially true for $t=0$, because we can find the empty set $\{\}$ of $0$ vectors $\{w_k\}_{k=1}^{t}$ such that Eq. (2) is satisfied.
For the induction step, let $t>0$ and assume that we are given $t-1$ vectors $\{w_1,\dots w_{t-1}\}$
"""
