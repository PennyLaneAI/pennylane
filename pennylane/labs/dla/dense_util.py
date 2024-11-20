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
# pylint: disable=possibly-used-before-assignment
from itertools import combinations_with_replacement
from typing import Iterable, Optional, Union

import numpy as np
from scipy.linalg import sqrtm

import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops.qubit.matrix_ops import _walsh_hadamard_transform
from pennylane.pauli import PauliSentence, PauliVSpace, PauliWord
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

    The coefficients are ordered lexicographically in the Pauli group.
    I.e. for ``n=2`` qubits we have the following ordering.

    .. code-block::

        [I(0),
         Z(1),
         Z(0),
         Z(0) @ Z(1),
         X(1),
         Y(1),
         Z(0) @ X(1),
         Z(0) @ Y(1),
         X(0),
         X(0) @ Z(1),
         Y(0),
         Y(0) @ Z(1),
         X(0) @ X(1),
         X(0) @ Y(1),
         Y(0) @ X(1),
         Y(0) @ Y(1)
        ]

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


def trace_inner_product(
    A: Union[PauliSentence, Operator, np.ndarray], B: Union[PauliSentence, Operator, np.ndarray]
):
    r"""Trace inner product

    Implementation of the trace inner product :math:`\langle A, B \rangle = \text{tr}\left(A^\dagger B\right)/\text{dim}(A)`

    If the inputs are ``np.ndarray``, leading broadcasting axes are supported for either or both
    inputs.

    Args:
        A (Union[PauliSentence, Operator, np.ndarray]): First operator
        B (Union[PauliSentence, Operator, np.ndarray]): Second operator

    Returns:
        Union[float, np.ndarray]: Result is either a single float or a batch of floats.

    **Example**

    >>> from pennylane.labs.dla import trace_inner_product
    >>> trace_inner_product(qml.X(0) + qml.Y(0), qml.Y(0) + qml.Z(0))
    1.0

    """
    if getattr(A, "pauli_rep", None) is not None and getattr(B, "pauli_rep", None) is not None:
        return (A.pauli_rep @ B.pauli_rep).trace()

    if not isinstance(A, type(B)):
        raise TypeError("Both input operators need to be of the same type")

    if isinstance(A, np.ndarray):
        assert A.shape[-2:] == B.shape[-2:]
        # The axes of the first input are switched, compared to tr[A@B], because we need to
        # transpose A.
        return np.tensordot(A.conj(), B, axes=[[-1, -2], [-1, -2]]) / A.shape[-1]

    if isinstance(A, (PauliSentence, PauliWord)):
        return (A @ B).trace()

    raise NotImplementedError


def adjvec_to_op(adj_vecs, basis, is_orthogonal=True):
    """Transform vectors representing operators in an operator basis back into operator format.

    Args:
        adj_vecs (np.ndarray): collection of vectors with shape ``(batch, len(basis))``
        basis (List[Union[PauliSentence, Operator, np.ndarray]]): collection of basis operators
        is_orthogonal (bool): Whether the ``basis`` consists of orthogonal elements.

    Returns:
        list: collection of operators corresponding to the input vectors read in the input basis.
        The operators are in the format specified by the elements in ``basis``.

    """
    assert qml.math.shape(adj_vecs)[1] == len(basis)

    if all(isinstance(op, PauliSentence) for op in basis):
        if not is_orthogonal:
            gram = _gram_ps(basis)
            adj_vecs = np.tensordot(adj_vecs, sqrtm(np.linalg.pinv(gram)), axes=[[1], [0]])
        res = []
        for vec in adj_vecs:
            op_j = sum(c * op for c, op in zip(vec, basis))
            op_j.simplify()
            res.append(op_j)
        return res

    if all(isinstance(op, Operator) for op in basis):
        if not is_orthogonal:
            basis_ps = [op.pauli_rep for op in basis]
            gram = _gram_ps(basis_ps)
            adj_vecs = np.tensordot(adj_vecs, sqrtm(np.linalg.pinv(gram)), axes=[[1], [0]])
        res = []
        for vec in adj_vecs:
            op_j = sum(c * op for c, op in zip(vec, basis))
            op_j = qml.simplify(op_j)
            res.append(op_j)
        return res

    if isinstance(basis, np.ndarray) or all(isinstance(op, np.ndarray) for op in basis):
        if not is_orthogonal:
            gram = np.tensordot(basis, basis, axes=[[1, 2], [2, 1]]).real / basis[0].shape[0]
            adj_vecs = np.tensordot(adj_vecs, sqrtm(np.linalg.pinv(gram)), axes=[[1], [0]])
        return np.tensordot(adj_vecs, basis, axes=1)

    raise NotImplementedError(
        "At least one operator in the specified basis is of unsupported type, "
        "or not all operators are of the same type."
    )


def _gram_ps(basis: Iterable[PauliSentence]):
    gram = np.zeros((len(basis), len(basis)))
    for (i, b_i), (j, b_j) in combinations_with_replacement(enumerate(basis), r=2):
        gram[i, j] = gram[j, i] = (b_i @ b_j).trace()
    return gram


def _op_to_adjvec_ps(ops: PauliSentence, basis: PauliSentence, is_orthogonal: bool = True):
    """Pauli sentence branch of ``op_to_adjvec``."""
    if not all(isinstance(op, PauliSentence) for op in ops):
        ops = [op.pauli_rep for op in ops]

    res = []
    if is_orthogonal:
        norms_squared = [(basis_i @ basis_i).trace() for basis_i in basis]
    else:
        # Fake the norm correction if we anyways will apply the inverse Gram matrix later
        norms_squared = np.ones(len(basis))
        gram = _gram_ps(basis)
        inv_gram = sqrtm(np.linalg.pinv(gram))

    for op in ops:
        rep = np.zeros((len(basis),))
        for i, basis_i in enumerate(basis):
            # v = ∑ (v · e_j / ||e_j||^2) * e_j
            rep[i] = (basis_i @ op).trace() / norms_squared[i]

        res.append(rep)
    res = np.array(res)
    if not is_orthogonal:
        res = np.einsum("ij,kj->ki", inv_gram, res)

    return res


def op_to_adjvec(
    ops: Union[PauliSentence, Operator, np.ndarray],
    basis: Union[PauliSentence, Operator, np.ndarray],
    is_orthogonal: bool = True,
):
    """Decompose a batch of operators onto a given operator basis.

    Args:
        ops (Union[PauliSentence, Operator, np.ndarray]): Operators to decompose
        basis (Iterable[Union[PauliSentence, Operator, np.ndarray]]): Operator basis
        is_orthogonal (bool): Whether the basis is orthogonal with respect to the trace inner
            product. Defaults to ``True``, which allows to skip some computations.

    Returns:
        np.ndarray: The batch of coefficient vectors of the operators ``ops`` expressed in
        ``basis``. The shape is ``(len(ops), len(basis)``.

    The format of the resulting operators is determined by the ``type`` in ``basis``.
    If ``is_orthogonal=True`` (the default), only normalization is taken into account
    in the projection. For ``is_orthogonal=False``, orthogonalization also is considered.
    """
    if isinstance(basis, PauliVSpace):
        basis = basis.basis

    if all(isinstance(op, Operator) for op in basis):
        ops = [op.pauli_rep for op in ops]
        basis = [op.pauli_rep for op in basis]

    # PauliSentence branch
    if all(isinstance(op, PauliSentence) for op in basis):
        return _op_to_adjvec_ps(ops, basis, is_orthogonal)

    # dense branch
    if all(isinstance(op, TensorLike) for op in basis):
        if not all(isinstance(op, TensorLike) for op in ops):
            _n = int(np.round(np.log2(basis[0].shape[-1])))
            ops = np.array([qml.matrix(op, wire_order=range(_n)) for op in ops])

        basis = np.array(basis)
        res = trace_inner_product(np.array(ops), basis).real
        if is_orthogonal:
            norm = np.einsum("bij,bji->b", basis, basis) / basis[0].shape[0]
            return res / norm
        gram = trace_inner_product(basis, basis).real
        return np.einsum("ij,kj->ki", sqrtm(np.linalg.pinv(gram)), res)

    raise NotImplementedError(
        "At least one operator in the specified basis is of unsupported type, "
        "or not all operators are of the same type."
    )
