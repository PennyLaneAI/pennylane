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


def binary_finite_reduced_row_echelon(binary_matrix):
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
    >>> qml.math.binary_finite_reduced_row_echelon(binary_matrix)
    array([[1, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0, 1, 1, 0],
           [0, 0, 0, 1, 1, 0, 0, 1]])
    """
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

    return rref_mat.astype(int)


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
