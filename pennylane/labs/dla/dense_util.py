# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility tools for dense Lie algebra representations"""

from typing import Optional

import numpy as np

import pennylane as qml
from pennylane.ops.qubit.matrix_ops import _walsh_hadamard_transform
from pennylane.pauli import PauliSentence, PauliWord
from pennylane.typing import TensorLike


def _make_phase_mat(n: int) -> np.ndarray:
    r"""Create an array with shape ``(2**n, 2**n)`` containing powers of :math:`i`.
    For the entry at position ``(i, k)``, the entry is ``1j**p(i,k)`` with the
    power being the Hamming weight of the product of the binary bitstrings of
    ``i`` and ``k``.
    """
    _slice = (slice(None), None, None)
    # Compute the bitwise and of the row and column index for all matrix entries
    ids_bit_and = np.bitwise_and(np.arange(2**n)[:, None], np.arange(2**n)[None])
    # Compute the Hamming weight of the bitwise and. Can be replaced by bitwise_count with np>=2.0
    hamming_weight = np.sum(
        np.bitwise_and(ids_bit_and[None] >> np.arange(n + 1)[_slice], 1), axis=0
    )
    return 1j**hamming_weight


def _make_permutation_indices(dim: int) -> list[np.ndarray]:
    r"""Make a list of ``dim`` arrays of length ``dim`` containing the indices
    ``0`` through ``dim-1`` in a specific permutation order to match the Walsh-Hadamard
    transform to the Pauli decomposition task."""
    indices = [qml.math.arange(dim)]
    for idx in range(dim - 1):
        indices.append(qml.math.bitwise_xor(indices[-1], (idx + 1) ^ (idx)))
    return indices


def _make_extraction_indices(n: int) -> tuple[tuple]:
    r"""Create a tuple of two tuples of indices to extract Pauli basis coefficients.
    The first tuple of indices as bit strings encodes the presence of a Pauli Z or Pauli Y
    on the ``k``\ th wire in the ``k``\ th bit. The second tuple encodes the presence of
    Pauli X or Pauli Y. That is, for a given position, the four different Pauli operators
    are encoded as ``(0, 0) = "I"``, ``(0, 1) = "X"``, ``(1, 0) = "Z"`` and ``(1, 1) = "Y"``.
    """
    if n == 1:
        return ((0, 0, 1, 1), (0, 1, 1, 0))

    ids0, ids1 = np.array(_make_extraction_indices(n - 1))
    return (
        tuple(np.concatenate([ids0, ids0, ids0 + 2 ** (n - 1), ids0 + 2 ** (n - 1)])),
        tuple(np.concatenate([ids1, ids1 + 2 ** (n - 1), ids1 + 2 ** (n - 1), ids1])),
    )


def pauli_coefficients(H: TensorLike) -> np.ndarray:
    r"""Computes the coefficients of a Hermitian matrix in the Pauli basis.

    Args:
        H (tensor_like[complex]): a Hermitian matrix of dimension ``(2**n, 2**n)`` or a collection
        of Hermitian matrices of dimension ``(batch, 2**n, 2**n)``.

    Returns:
        np.ndarray: The coefficients of ``H`` in the Pauli basis with shape ``(4**n,)`` for a single
        matrix input and ``(batch, 4**n)`` for a collection of matrices. The output is real-valued.

    **Examples**

    Consider the Hamiltonian :math:`H=\frac{1}{4} X_0 + \frac{2}{5} Z_0 X_1` with matrix

    >>> H = 1 / 4 * qml.X(0) + 2 / 5 * qml.Z(0) @ qml.X(1)
    >>> mat = H.matrix()
    array([[ 0.  +0.j,  0.4 +0.j,  0.25+0.j,  0.  +0.j],
           [ 0.4 +0.j,  0.  +0.j,  0.  +0.j,  0.25+0.j],
           [ 0.25+0.j,  0.  +0.j,  0.  +0.j, -0.4 +0.j],
           [ 0.  +0.j,  0.25+0.j, -0.4 +0.j,  0.  +0.j]])

    Then we can obtain the coefficients of :math:`H` in the Pauli basis via

    >>> from pennylane.labs.dla import pauli_coefficients
    >>> pauli_coefficients(mat)
    array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.  ,
            0.  , -0.  ,  0.  ,  0.  ,  0.4 ,  0.  ,  0.  ])

    The function can be used on a batch of matrices:

    >>> ops = [1 / 4 * qml.X(0), 1 / 2 * qml.Z(0), 3 / 5 * qml.Y(0)]
    >>> batch = np.stack([op.matrix() for op in ops])
    >>> pauli_coefficients(batch)
    array([[0.  , 0.25, 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.5 ],
           [0.  , 0.  , 0.6 , 0.  ]])

    """
    # Preparations
    shape = H.shape
    batch = shape[0] if H.ndim == 3 else None
    dim = shape[-1]
    n = int(np.round(np.log2(dim)))
    assert dim == 2**n

    # Permutation
    indices = _make_permutation_indices(dim)
    # Apply the permutation by slicing and stacking again
    sliced_H = [
        qml.math.take(H[..., idx, :], _indices, axis=-1) for idx, _indices in enumerate(indices)
    ]
    sliced_H = qml.math.cast(qml.math.stack(sliced_H), complex)
    # Move leading axis (different permutation slices) to last position and combine broadcasting axis
    # and slicing axis into one leading axis (because `_walsh_hadamard_transform` only takes one batch axis)
    term_mat = qml.math.reshape(qml.math.moveaxis(sliced_H, 0, -1), (-1, dim))
    # Apply Walsh-Hadamard transform
    hadamard_transform_mat = _walsh_hadamard_transform(term_mat)
    # Reshape again to separate actual broadcasting axis and previous slicing axis
    hadamard_transform_mat = qml.math.reshape(hadamard_transform_mat, shape)
    # _make phase matrix that allows us to figure out phase contributions from Pauli Y terms.
    phase_mat = qml.math.convert_like(_make_phase_mat(n), H)
    # Multiply phase matrix to Hadamard transformed matrix and transpose the two Hilbert-space-dim axes
    coefficients = qml.math.moveaxis(
        qml.math.real(qml.math.multiply(hadamard_transform_mat, phase_mat)), -2, -1
    )
    # Extract the coefficients by reordering them according to the encoding in `qml.pauli.pauli_decompose`
    indices = _make_extraction_indices(n)
    new_shape = (dim**2,) if batch is None else (batch, dim**2)
    return qml.math.reshape(coefficients[..., indices[0], indices[1]], new_shape)


_pauli_strings = (None, "X", "Y", "Z")


def _idx_to_pw(idx, n):
    pw = {}
    wire = n - 1
    while idx > 0:
        p = _pauli_strings[idx % 4]
        if p:
            pw[wire] = p
        idx //= 4
        wire -= 1
    return PauliWord(pw)


def pauli_decompose(H: TensorLike, tol: Optional[float] = None, pauli: bool = False):
    r"""Decomposes a Hermitian matrix into a linear combination of Pauli operators.

    Args:
        H (tensor_like[complex]): a Hermitian matrix of dimension ``(2**n, 2**n)`` or a collection
            of Hermitian matrices of dimension ``(batch, 2**n, 2**n)``.
        tol (float): Tolerance below which Pauli coefficients are discarded.
        pauli (bool): Whether to format the output as :class:`~.PauliSentence`.

    Returns:
        Union[~.Hamiltonian, ~.PauliSentence]: the matrix (matrices) decomposed as a
        linear combination of Pauli operators, returned either as a :class:`~.Hamiltonian`
        or :class:`~.PauliSentence` instance.

    .. seealso:: :func:`~.pauli_coefficients`

    **Examples**

    Consider the Hamiltonian :math:`H=\frac{1}{4} X_0 + \frac{2}{5} Z_0 X_1`. We can compute its
    matrix and get back the Pauli representation via ``pauli_decompose``.

    >>> from pennylane.labs.dla import pauli_decompose
    >>> H = 1 / 4 * qml.X(0) + 2 / 5 * qml.Z(0) @ qml.X(1)
    >>> mat = H.matrix()
    >>> op = pauli_decompose(mat)
    >>> op
    0.25 * X(1) + 0.4 * Z(1)
    >>> type(op)
    pennylane.ops.op_math.sum.Sum

    We can choose to receive a :class:`~.PauliSentence` instead as output instead, by setting
    ``pauli=True``:

    >>> op = pauli_decompose(mat, pauli=True)
    >>> type(op)
    pennylane.pauli.pauli_arithmetic.PauliSentence

    This function supports batching and will return a list of operations for a batched input:

    >>> ops = [1 / 4 * qml.X(0), 1 / 2 * qml.Z(0) + 1e-7 * qml.Y(0)]
    >>> batch = np.stack([op.matrix() for op in ops])
    >>> pauli_decompose(batch)
    [0.25 * X(0), 1e-07 * Y(0) + 0.5 * Z(0)]

    Small contributions can be removed by specifying the ``tol`` parameter, which defaults
    to ``1e-10``, accordingly:

    >>> pauli_decompose(batch, tol=1e-6)
    [0.25 * X(0), 0.5 * Z(0)]

    """
    if tol is None:
        tol = 1e-10
    coeffs = pauli_coefficients(H)
    if single_H := qml.math.ndim(coeffs) == 1:
        coeffs = [coeffs]

    n = int(np.round(np.log2(qml.math.shape(coeffs)[1]))) // 2

    H_ops = []
    for _coeffs in coeffs:
        ids = qml.math.where(qml.math.abs(_coeffs) > tol)[0]
        sentence = PauliSentence({_idx_to_pw(idx, n): c for c, idx in zip(_coeffs[ids], ids)})
        if pauli:
            H_ops.append(sentence)
        else:
            H_ops.append(sentence.operation())

    if single_H:
        return H_ops[0]
    return H_ops