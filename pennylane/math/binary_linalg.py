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


def int_to_binary(integer: int | np.ndarray, width: int) -> np.ndarray:
    """Convert an integer or an array of integers to an array of bitstrings of
    given length, representing the integers as binaries.

    Args:
        integer (int | np.ndarray): Integer(s) to convert. Either a single integers or an
            array of integers.
        width (int): Length of the bitstrings to which the integer(s) are converted. Note
            that the ``width`` **least** significant bits corresponding to
            ``integers % 2**width`` are returned, discarding the most
            significant contributions if ``integer > 2**(width-1)-1``.

    Returns:
        np.ndarray: Array of bitstrings representing the ``integer`` input. If ``integer`` is an
        ``int``, the returned array has shape ``(width,)``. If ``integer`` is an array of
        shape ``S``, the returned array has shape ``(*S, width)``.


    **Example**

    We may compute the binary representation of the integer ``13`` on five bits, for example:

    >>> width = 5
    >>> print(qml.math.int_to_binary(13, width=width))
    [0 1 1 0 1]

    This matches the output of ``np.binary_repr`` but returns a numerical array instead
    of a string:

    >>> print(np.binary_repr(13, width=width))
    01101

    For an array-typed input, we obtain a new array with an additinoal axis in the last position,
    of size ``width``:

    >>> x = np.array([[7, 3], [17, 9], [2, 8]])
    >>> print(x.shape)
    (3, 2)
    >>> bits = qml.math.int_to_binary(x, width=width)
    >>> print(bits.shape)
    (3, 2, 5)

    The input ``integer`` can be reconstructed from the bit strings via

    >>> powers_of_two = 2**np.arange(width-1, -1, -1)
    >>> reconstruction = bits @ powers_of_two
    >>> print(reconstruction)
    [[ 7  3]
     [17  9]
     [ 2  8]]

    """
    shifts = np.arange(width - 1, -1, -1)  # [width-1, width-2, ... 1, 0]
    if isinstance(integer, int):
        return (integer >> shifts) % 2
    return (integer[..., None] >> shifts) % 2  # Broadcasting (new) shape (*S, 1) with (width,)


def binary_finite_reduced_row_echelon(binary_matrix, inplace=False):
    r"""Computes the `reduced row echelon form (RREF)
    <https://en.wikipedia.org/wiki/Row_echelon_form>`__ of a matrix with
    entries from the binary numbers, a.k.a. the finite field :math:`\mathbb{Z}_2`.

    Args:
        binary_matrix (array[int]): binary matrix
        inplace (bool): Whether to perform the modification of ``binary_matrix`` in place. Defaults
            to ``False``, making a copy before running the calculation.

    Returns:
        array[int]: reduced row-echelon form of the given ``binary_matrix``. The output has the
        same shape as the input. If ``inplace=True``, the returned array is the same object as
        the input ``binary_matrix``, which then has been modified in place.

    .. note::

        This function is currently not compatible with JAX.

    **Example**

    >>> binary_matrix = np.array([[1, 0, 0, 0, 0, 1, 0, 0],
    ...                           [1, 0, 1, 0, 0, 0, 1, 0],
    ...                           [0, 0, 0, 1, 1, 0, 0, 1]])
    >>> print(qml.math.binary_finite_reduced_row_echelon(binary_matrix))
    [[1, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 1, 0, 0, 1, 1, 0],
     [0, 0, 0, 1, 1, 0, 0, 1]]
    """

    rref_mat = binary_matrix if inplace else binary_matrix.copy()
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


def binary_matrix_rank(binary_matrix: np.ndarray) -> int:
    r"""
    Find rank of a matrix over :math:`\mathbb{Z}_2`.

    Args:
        binary_matrix (np.ndarray): Matrix of binary entries.

    Returns:
        int: Rank of ``binary_matrix`` over the finite field :math:`\mathbb{Z}_2`.

    Note that the ranks of a matrix over :math:`\mathbb{Z}_2` and over :math:`\mathbb{Z}` (or
    :math:`\mathbb{R}`) may differ.

    This function does not modify the input.

    .. note::

        This function is currently not compatible with JAX.

    **Example**

    Consider the following binary matrix of shape ``(4, 4)``:

    >>> binary_matrix = np.array([[0, 1, 1, 0], [0, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0]])
    >>> print(binary_matrix.shape)
    (4, 4)

    We may compute its rank over :math:`\mathbb{Z}_2` and find that it does not have full rank:

    >>> print(qml.math.binary_matrix_rank(binary_matrix))
    3

    Note that it would have full rank over the real numbers :math:`\mathbb{R}`:

    >>> print(qml.math.linalg.matrix_rank(binary_matrix))
    4

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


def binary_solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""Solve the linear system of equations :math:`Ax=b` over the Booleans/:math:`\mathbb{Z}_2`,
    where :math:`A` is assumed to be regular, i.e., non-singular.
    The implementation is based on Gaussian elimination to bring the augmented matrix :math:`(A|b)`
    into (reduced) row echelon form via :func:`~.math.binary_finite_reduced_row_echelon`.

    Args:
        A (np.ndarray): Square matrix of the linear system of equations with binary entries.
        b (np.ndarray): Binary coefficient vector with same length as ``A``.

    Returns:
        np.ndarray: Binary solution vector with same length as ``A`` and ``b``.

    .. note::

        This function is currently not compatible with JAX.

    **Example**

    Consider a simple regular Boolean matrix ``A`` and a coefficient vector ``b``:

    >>> A = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1]])
    >>> b = np.array([1, 1, 1])

    Then we can solve the system ``A@x=b`` for ``x`` over :math:`\mathbb{Z}_2`:

    >>> x = qml.math.binary_solve_linear_system(A, b)
    >>> print(x)
    [1 1 0]

    Internally, this is done with the Gauss-Jordan elimination of the extended matrix ``(A | b)``.
    Indeed, we can verify that ``A@x=b`` (over :math:`\mathbb{Z}_2`):

    >>> print(np.allclose((A @ x)%2, b))
    True

    """
    # Create augmented matrix for Gauss-Jordan elimination. This also creates a copy so that the
    # input ``A`` is not modified in place, and we can skip copying within the echelon form call
    rref = np.hstack([A, b[:, None]])
    rref = binary_finite_reduced_row_echelon(rref, inplace=True)

    if np.any(np.sum(rref[:, :-1], axis=1) == 0):
        # Matrix A is singular
        raise np.linalg.LinAlgError("Singular binary matrix.")

    # Potential solution is written in the last column of the augmented matrix after obtained RREF
    return rref[:, -1]


def binary_is_independent(vector: np.ndarray, basis: np.ndarray) -> bool:
    r"""Check whether a binary vector, i.e., a bitstring, is
    linearly independent (over :math:`\mathbb{Z}_2`) of a basis of binary vectors, given as column
    vectors of a matrix.

    Args:
        vector (np.ndarray): Binary vector to check.
        basis (np.ndarray): Basis of binary vectors against which ``vector`` is checked. If
            ``vector`` has shape ``(r,)``, ``basis`` should have shape ``(r, m)`` and rank
            ``min(r, m)``, i.e., the columns of ``basis`` all need to be linearly independent.

    Returns:
        bool: Whether ``vector`` is linearly independent of ``basis`` over :math:`\mathbb{Z}_2`.

    .. note::

        This function is currently not compatible with JAX.

    """
    # We assume ``basis`` to have full rank.
    r = vector.shape[0]
    if basis.shape[0] != r:
        raise ValueError(
            "The columns of `basis` should have the same length as `vector`. "
            f"Got {vector.shape=} and {basis.shape=}"
        )
    basis_rank = min(basis.shape)
    rk = binary_matrix_rank(np.concatenate([basis, vector[:, None]], axis=1))
    return rk > basis_rank


def binary_select_basis(bitstrings: np.ndarray):
    r"""Select bitstrings from an array of bitstrings that form a basis
    for the column space of the array. Also returns the bitstrings that were not selected.

    Args:
        bitstrings (np.ndarray): Input bitstrings. The columns of the array span the space for
            which a basis is selected.

    Returns:
        tuple[np.ndarray]: Two binary array. The first contains a selection of columns from
        ``bitstrings`` that form a basis for the column space of ``bitstrings`` over
        :math:`\mathbb{Z}_2`. The second contains all other columns.

    .. note::

        This function is currently not compatible with JAX.
    """
    r, _ = bitstrings.shape
    basis = np.zeros((r, 0), dtype=int)
    other_cols = []
    for col in bitstrings.T:
        if basis.shape[1] < r and binary_is_independent(col, basis):
            basis = np.concatenate([basis, col[:, None]], axis=1)
        else:
            other_cols.append(col)

    if not other_cols:
        other_cols = np.zeros((r, 0), dtype=int)
    else:
        other_cols = np.array(other_cols).T
    return basis, other_cols
