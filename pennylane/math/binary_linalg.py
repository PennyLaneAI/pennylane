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
"""This module contains linear algebraic functions over the binary numbers,
also referred to as finite field F_2, Galois field GF_2, or ℤ_2."""
import numpy as np


def binary_finite_reduced_row_echelon(binary_matrix, inplace=False):
    r"""Computes the `reduced row echelon form (RREF)
    <https://en.wikipedia.org/wiki/Row_echelon_form>`__ of a matrix with
    entries from the binary numbers, a.k.a. the finite field :math:`\mathbb{Z}_2`.

    Args:
        binary_matrix (array[int]): binary matrix

    Returns:
        array[int]: reduced row-echelon form of the given ``binary_matrix``. The output has the
        same shape as the input.

    **Example**

    >>> binary_matrix = np.array([[1, 0, 0, 0, 0, 1, 0, 0],
    ...                           [1, 0, 1, 0, 0, 0, 1, 0],
    ...                           [0, 0, 0, 1, 1, 0, 0, 1]])
    >>> print(qml.math.binary_finite_reduced_row_echelon(binary_matrix))
    [[1, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 1, 0, 0, 1, 1, 0],
     [0, 0, 0, 1, 1, 0, 0, 1]]
    """
    if inplace:
        rref_mat = binary_matrix
    else:
        rref_mat = binary_matrix.copy()
    shape = rref_mat.shape
    icol = 0

    for irow in range(shape[0]):
        while icol < shape[1] and not rref_mat[irow][icol]:
            # get the nonzero indices in the remainder of column icol
            non_zero_idx = rref_mat[irow:, icol].nonzero()[0]

            if len(non_zero_idx) == 0:  # if remainder of column icol is all zero
                icol += 1
            else:
                # find value and index of largest element in remainder of column icol
                krow = irow + non_zero_idx[0]

                # swap rows krow and irow
                rref_mat[irow, icol:], rref_mat[krow, icol:] = (
                    rref_mat[krow, icol:].copy(),
                    rref_mat[irow, icol:].copy(),
                )
        if icol < shape[1] and rref_mat[irow][icol]:
            # store remainder right hand side columns of the pivot row irow
            rpvt_cols = rref_mat[irow, icol:].copy()

            # get the column icol and set its irow element to 0 to avoid XORing pivot row with itself
            currcol = rref_mat[:, icol].copy()
            currcol[irow] = 0

            # XOR the right hand side of the pivot row irow with all of the other rows
            rref_mat[:, icol:] ^= np.outer(currcol, rpvt_cols)
            icol += 1

    return rref_mat


def binary_rank(binary_matrix: np.ndarray) -> int:
    r"""
    Find rank of a matrix over Z_2.

    Args:
        binary_matrix (np.ndarray): Matrix of binary entries.

    Returns:
        int: Rank of ``binary_matrix`` over the finite field :math:`\mathbb{Z}_2`.

    Note that the ranks of a matrix over :math:`\mathbb{Z}_2` and over :math:`\mathbb{Z}`
    may differ.
    This function does not modify the input.

    **Example**

    """
    binary_matrix = binary_matrix.copy()
    rank = 0
    while len(binary_matrix):
        # Take last row as potential pivot and reduce remaining matrix
        binary_matrix, pivot = binary_matrix[:-1], binary_matrix[-1]
        if np.any(pivot):
            # If it the potential pivot is not all zeros, it provides a new degree of freedom
            rank += 1
            # Find least significant bit that is set to 1 in the pivot
            least_sig_bit = np.where(pivot)[0][-1]
            # XOR all rows with the pivot that have the bit in the position least_sig_bit set to 1
            binary_matrix[np.where(binary_matrix[:, least_sig_bit])] ^= pivot
    return rank


def solve_binary_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""Solve the linear system of equations A.x=b over the Booleans/:math:`\mathbb{Z}_2`,
    where A is assumed to be regular, i.e., non-singular.
    The implementation is based on Gaussian elimination to bring the augmented matrix :math:`(A|b)`
    into (reduced) row echelon form via :func:`~.math.binary_finite_reduced_row_echelon`.

    with simplifications based on the regularity of A (we always find a next pivot and have

    Args:
        A (np.ndarray): Square matrix with coefficients (0 or 1).
        b (np.ndarray): Coefficient vector with same length as ``A`` and entries 0 or 1.

    Returns:
        np.ndarray: Solution vector with same length as ``b`` and entries 0 or 1.

    **Example**

    Consider a simple regular Boolean matrix ``A`` and a coefficient vector ``b``:

    >>> A = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1]])
    >>> b = np.array([1, 1, 1])

    Then we can solve the system ``A@x=b`` for ``x`` over :math:`\mathbb{Z}_2` with the
    Gauss-Jordan elimination of the extended matrix ``A | b``.

    >>> x = qml.math.solve_binary_linear_system(A, b)
    >>> print(x)
    [1 1 0]

    Indeed, we can verify that ``A@x=b`` (over :math:`\mathbb{Z}_2`):

    >>> print(np.allclose((A @ x)%2, b))
    True

    Note that the solution is unique if ``A`` is regular.

    The regularity
    is used in the algorithm to simplify its logic.
    """
    # Create augmented matrix for Gauss-Jordan elimination. This also creates a copy so that the
    # input ``A`` is not modified in place, and we can skip copying within the echelon form call
    rref = np.hstack([A, b[:, None]])
    rref = binary_finite_reduced_row_echelon(rref, inplace=True)

    if np.any(np.sum(rref[:, :-1], axis=1) == 0):
        # Matrix A is singular
        raise np.linalg.LinAlgError("Singular binary matrix.")

    # Potential solution is written in the last column of the (augmented) matrix after obtained RREF
    return rref[:, -1]
