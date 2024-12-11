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
# pylint: disable=too-many-return-statements, missing-function-docstring, possibly-used-before-assignment
from functools import reduce
from itertools import combinations, combinations_with_replacement
from typing import Iterable, List, Optional, Union

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
    r"""Computes the coefficients of one or multiple Hermitian matrices in the Pauli basis.

    The coefficients are ordered lexicographically in the Pauli group, ``["III", "IIX", "IIY", "IIZ", "IXI", ...]``.

    Args:
        H (tensor_like[complex]): a Hermitian matrix of dimension ``(2**n, 2**n)`` or a collection
            of Hermitian matrices of dimension ``(batch, 2**n, 2**n)``.

    Returns:
        np.ndarray: The coefficients of ``H`` in the Pauli basis with shape ``(4**n,)`` for a single
        matrix input and ``(batch, 4**n)`` for a collection of matrices. The output is real-valued.

    See :func:`~.pennylane.pauli.pauli_decompose` for theoretical background information.

    **Examples**

    Consider the Hamiltonian :math:`H=\frac{1}{4} X_0 + \frac{2}{5} Z_0 X_1` with matrix

    >>> H = 1 / 4 * qml.X(0) + 2 / 5 * qml.Z(0) @ qml.X(1)
    >>> mat = H.matrix()
    >>> mat
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
    # make phase matrix that allows us to figure out phase contributions from Pauli Y terms.
    phase_mat = qml.math.convert_like(_make_phase_mat(n), H)
    # Multiply phase matrix to Hadamard transformed matrix and transpose the two Hilbert-space-dim axes
    coefficients = qml.math.moveaxis(
        qml.math.real(qml.math.multiply(hadamard_transform_mat, phase_mat)), -2, -1
    )
    # Extract the coefficients by reordering them according to the encoding in `qml.pauli.batched_pauli_decompose`
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


def batched_pauli_decompose(H: TensorLike, tol: Optional[float] = None, pauli: bool = False):
    r"""Decomposes a Hermitian matrix or a batch of matrices into a linear combination
    of Pauli operators.

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
    matrix and get back the Pauli representation via ``batched_pauli_decompose``.

    >>> from pennylane.labs.dla import batched_pauli_decompose
    >>> H = 1 / 4 * qml.X(0) + 2 / 5 * qml.Z(0) @ qml.X(1)
    >>> mat = H.matrix()
    >>> op = batched_pauli_decompose(mat)
    >>> op
    0.25 * X(1) + 0.4 * Z(1)
    >>> type(op)
    pennylane.ops.op_math.sum.Sum

    We can choose to receive a :class:`~.PauliSentence` instead as output instead, by setting
    ``pauli=True``:

    >>> op = batched_pauli_decompose(mat, pauli=True)
    >>> type(op)
    pennylane.pauli.pauli_arithmetic.PauliSentence

    This function supports batching and will return a list of operations for a batched input:

    >>> ops = [1 / 4 * qml.X(0), 1 / 2 * qml.Z(0) + 1e-7 * qml.Y(0)]
    >>> batch = np.stack([op.matrix() for op in ops])
    >>> batched_pauli_decompose(batch)
    [0.25 * X(0), 1e-07 * Y(0) + 0.5 * Z(0)]

    Small contributions can be removed by specifying the ``tol`` parameter, which defaults
    to ``1e-10``, accordingly:

    >>> batched_pauli_decompose(batch, tol=1e-6)
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


def check_commutation(ops1, ops2, vspace):
    r"""Helper function to check :math:`[\text{ops1}, \text{ops2}] \subseteq \text{vspace}`.

    .. warning:: This function is expensive to compute

    Args:
        ops1 (Iterable[PauliSentence]): First set of operators
        ops2 (Iterable[PauliSentence]): Second set of operators
        vspace (:class:`~PauliVSpace`): The vector space in form of a :class:`~PauliVSpace` that the operators should map to

    Returns:
        bool: Whether or not :math:`[\text{ops1}, \text{ops2}] \subseteq \text{vspace}`

    **Example**

    >>> from pennylane.labs.dla import check_commutation
    >>> ops1 = [qml.X(0).pauli_rep]
    >>> ops2 = [qml.Y(0).pauli_rep]
    >>> vspace1 = qml.pauli.PauliVSpace([qml.X(0).pauli_rep, qml.Y(0).pauli_rep], dtype=complex)

    Because :math:`[X_0, Y_0] = 2i Z_0`, the commutators do not map to the selected vector space.

    >>> check_commutation(ops1, ops2, vspace1)
    False

    Instead, we need the full :math:`\mathfrak{su}(2)` space.

    >>> vspace2 = qml.pauli.PauliVSpace([qml.X(0).pauli_rep, qml.Y(0).pauli_rep, qml.Z(0).pauli_rep], dtype=complex)
    >>> check_commutation(ops1, ops2, vspace2)
    True
    """
    for o1 in ops1:
        for o2 in ops2:
            com = o1.commutator(o2)
            com.simplify()
            if len(com) != 0:
                if vspace.is_independent(com):
                    return False

    return True


def check_all_commuting(ops: List[Union[PauliSentence, np.ndarray, Operator]]):
    r"""Helper function to check if all operators in ``ops`` commute.

    .. warning:: This function is expensive to compute

    Args:
        ops (List[Union[PauliSentence, np.ndarray, Operator]]): List of operators to check for mutual commutation

    Returns:
        bool: Whether or not all operators commute with each other

    **Example**

    >>> from pennylane.labs.dla import check_all_commuting
    >>> from pennylane import X
    >>> ops = [X(i) for i in range(10)]
    >>> check_all_commuting(ops)
    True

    Operators on different wires (trivially) commute with each other.
    """
    if all(isinstance(op, PauliSentence) for op in ops):
        for oi, oj in combinations(ops, 2):
            com = oj.commutator(oi)
            com.simplify()
            if len(com) != 0:
                return False

        return True

    if all(isinstance(op, Operator) for op in ops):
        for oi, oj in combinations(ops, 2):
            com = qml.simplify(qml.commutator(oj, oi))
            if not qml.equal(com, 0 * qml.Identity()):
                return False

        return True

    if all(isinstance(op, np.ndarray) for op in ops):
        for oi, oj in combinations(ops, 2):
            com = oj @ oi - oi @ oj
            if not np.allclose(com, np.zeros_like(com)):
                return False

        return True

    return NotImplemented


def check_cartan_decomp(k: List[PauliSentence], m: List[PauliSentence], verbose=True):
    r"""Helper function to check the validity of a Cartan decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}.`

    Check whether of not the following properties are fulfilled.

    .. math::

            [\mathfrak{k}, \mathfrak{k}] \subseteq \mathfrak{k} & \text{ (subalgebra)}\\
            [\mathfrak{k}, \mathfrak{m}] \subseteq \mathfrak{m} & \text{ (reductive property)}\\
            [\mathfrak{m}, \mathfrak{m}] \subseteq \mathfrak{k} & \text{ (symmetric property)}

    .. warning:: This function is expensive to compute

    Args:
        k (List[PauliSentence]): List of operators of the vertical subspace
        m (List[PauliSentence]): List of operators of the horizontal subspace
        verbose: Whether failures to meet one of the criteria should be printed

    Returns:
        bool: Whether or not all properties are fulfilled

    .. seealso:: :func:`~cartan_decomp`

    **Example**

    We first construct a Lie algebra.

    >>> from pennylane import X, Z
    >>> from pennylane.labs.dla import concurrence_involution, even_odd_involution, cartan_decomp
    >>> generators = [X(0) @ X(1), Z(0), Z(1)]
    >>> g = qml.lie_closure(generators)
    >>> g
    [X(0) @ X(1),
     Z(0),
     Z(1),
     -1.0 * (Y(0) @ X(1)),
     -1.0 * (X(0) @ Y(1)),
     -1.0 * (Y(0) @ Y(1))]

    We compute the Cartan decomposition with respect to the :func:`~concurrence_involution`.

    >>> k, m = cartan_decomp(g, concurrence_involution)
    >>> k, m
    ([-1.0 * (Y(0) @ X(1)), -1.0 * (X(0) @ Y(1))],
     [X(0) @ X(1), Z(0), Z(1), -1.0 * (Y(0) @ Y(1))])

    We can check the validity of the decomposition using ``check_cartan_decomp``.

    >>> from pennylane.labs.dla import check_cartan_decomp
    >>> check_cartan_decomp(k, m)
    True

    """
    if any(isinstance(op, np.ndarray) for op in k):
        k = [qml.pauli_decompose(op).pauli_rep for op in k]
    if any(isinstance(op, np.ndarray) for op in m):
        m = [qml.pauli_decompose(op).pauli_rep for op in m]

    if any(isinstance(op, Operator) for op in k):
        k = [op.pauli_rep for op in k]
    if any(isinstance(op, Operator) for op in m):
        m = [op.pauli_rep for op in m]

    k_space = qml.pauli.PauliVSpace(k, dtype=complex)
    m_space = qml.pauli.PauliVSpace(m, dtype=complex)

    # Commutation relations for Cartan pair
    if not (check_kk := check_commutation(k, k, k_space)):
        _ = print("[k, k] sub k not fulfilled") if verbose else None
    if not (check_km := check_commutation(k, m, m_space)):
        _ = print("[k, m] sub m not fulfilled") if verbose else None
    if not (check_mm := check_commutation(m, m, k_space)):
        _ = print("[m, m] sub k not fulfilled") if verbose else None

    return all([check_kk, check_km, check_mm])


def orthonormalize(basis: Iterable[Union[PauliSentence, Operator, np.ndarray]]) -> np.ndarray:
    r"""Orthonormalize a list of basis vectors.

    Args:
        basis (Iterable[Union[PauliSentence, Operator, np.ndarray]]): List of basis vectors.

    Returns:
        np.ndarray: Orthonormalized basis vectors.

    .. seealso:: :func:`~trace_inner_product`, :func:`~orthonormalize`

    **Example**

    >>> from pennylane.labs.dla import orthonormalize, check_orthonormal, trace_inner_product
    >>> ops = [qml.X(0), qml.X(0) + qml.Y(0), qml.Y(0) + qml.Z(0)]
    >>> check_orthonormal(ops, trace_inner_product)
    False
    >>> ops_orth = orthonormalize(ops)
    >>> check_orthonormal(ops_orth, trace_inner_product)
    True

    This works also for lists of dense matrices as inputs
    >>> ops_m = [qml.matrix(op) for op in ops]
    >>> ops_m_orth = orthonormalize(ops_m)
    >>> ops_m_orth.shape
    (3, 2, 2)
    """

    if isinstance(basis, PauliVSpace) or all(
        isinstance(op, (PauliSentence, Operator)) for op in basis
    ):
        return _orthonormalize_ps(basis)

    if all(isinstance(op, np.ndarray) for op in basis):
        return _orthonormalize_np(basis)

    raise NotImplementedError(
        f"orthonormalize not implemented for basis of type {type(basis[0])}:\n{basis}"
    )


def _orthonormalize_np(basis: Iterable[np.ndarray]):
    basis = np.array(basis)
    gram_inv = np.linalg.pinv(sqrtm(trace_inner_product(basis, basis).real))
    return np.tensordot(gram_inv, basis, axes=1)


def _orthonormalize_ps(basis: Union[PauliVSpace, Iterable[Union[PauliSentence, Operator]]]):
    # We are generating a sparse pauli representation of the basis, where each entry of a basis vector corresponds to one of the Pauli words
    if isinstance(basis, PauliVSpace):
        basis = basis.basis

    if not all(isinstance(op, PauliSentence) for op in basis):
        basis = [op.pauli_rep for op in basis]

    if len(basis) == 0:
        return basis

    # Set up all unique pauli words in the basis
    all_pws = reduce(set.__or__, [set(ps.keys()) for ps in basis])
    num_pw = len(all_pws)

    # map pauli words to indices and back
    _pw_to_idx = {pw: i for i, pw in enumerate(all_pws)}
    _idx_to_pw = dict(enumerate(all_pws))

    # dense matrix representation of the basis in the sparse pauli representation
    _M = np.zeros((num_pw, len(basis)), dtype=float)

    for i, gen in enumerate(basis):
        for pw, value in gen.items():
            _M[_pw_to_idx[pw], i] = value

    # orthonormalize dense matrix using QR decomposition
    def gram_schmidt(X):
        Q, _ = np.linalg.qr(X)
        return Q

    OM = gram_schmidt(_M)

    # make sure the resulting matrix is orthonormal
    assert np.allclose(np.tensordot(OM.T, OM, axes=1), np.eye(OM.shape[1]))

    # reconstruct orthonormalized operators
    generators_orthogonal = []
    for i in range(len(basis)):
        u1 = PauliSentence({})
        for j in range(num_pw):
            u1 += _idx_to_pw[j] * OM[j, i]
        u1.simplify()
        generators_orthogonal.append(u1)

    return generators_orthogonal


def check_orthonormal(g: Iterable[Union[PauliSentence, Operator]], inner_product: callable) -> bool:
    r"""
    Utility function to check if operators in ``g`` are orthonormal with respect to the provided ``inner_product``.

    Args:
        g (Iterable[Union[PauliSentence, Operator]]): List of operators
        inner_product (callable): Inner product function to check orthonormality

    Returns:
        bool: ``True`` if the operators are orthonormal, ``False`` otherwise.

    .. seealso:: :func:`~trace_inner_product`, :func:`~orthonormalize`

    **Example**

    >>> from pennylane.labs.dla import orthonormalize, check_orthonormal, trace_inner_product
    >>> ops = [qml.X(0), qml.X(0) + qml.Y(0), qml.Y(0) + qml.Z(0)]
    >>> check_orthonormal(ops, trace_inner_product)
    False
    >>> ops_orth = orthonormalize(ops)
    >>> check_orthonormal(ops_orth, trace_inner_product)
    True
    """
    for op in g:
        if not np.isclose(inner_product(op, op), 1.0):
            return False
    for opi, opj in combinations(g, r=2):
        if not np.isclose(inner_product(opi, opj), 0.0):
            return False
    return True


def trace_inner_product(
    A: Union[PauliSentence, Operator, np.ndarray], B: Union[PauliSentence, Operator, np.ndarray]
):
    r"""Implementation of the trace inner product :math:`\langle A, B \rangle = \text{tr}\left(A B\right)/\text{dim}(A)` between two Hermitian operators :math:`A` and :math:`B`.

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

    If both operators are dense arrays, a leading batch dimension is broadcasted.

    >>> batch = 10
    >>> ops1 = np.random.rand(batch, 16, 16)
    >>> op2 = np.random.rand(16, 16)
    >>> trace_inner_product(ops1, op2).shape
    (10,)
    >>> trace_inner_product(op2, ops1).shape
    (10,)

    We can also have both arguments broadcasted.

    >>> trace_inner_product(ops1, ops1).shape
    (10, 10)

    """
    if getattr(A, "pauli_rep", None) is not None and getattr(B, "pauli_rep", None) is not None:
        return (A.pauli_rep @ B.pauli_rep).trace()

    if all(isinstance(op, np.ndarray) for op in A) and all(isinstance(op, np.ndarray) for op in B):
        A = np.array(A)
        B = np.array(B)

    if not isinstance(A, type(B)):
        raise TypeError("Both input operators need to be of the same type")

    if isinstance(A, np.ndarray):
        assert A.shape[-2:] == B.shape[-2:]
        # The axes of the first input are switched, compared to tr[A@B], because we need to
        # transpose A.
        return np.tensordot(A, B, axes=[[-1, -2], [-2, -1]]) / A.shape[-1]

    raise NotImplementedError


def change_basis_ad_rep(adj: np.ndarray, basis_change: np.ndarray):
    r"""Apply a ``basis_change`` between bases of operators to the adjoint representation ``adj``.

    Assume the adjoint repesentation is given in terms of a basis :math:`\{b_j\}`,
    :math:`\text{ad_\mu}_{\alpha \beta} \propto \text{tr}\left(b_\mu \cdot [b_\alpha, b_\beta] \right)`.
    We can represent the adjoint representation in terms of a new basis :math:`c_i = \sum_j T_{ij} b_j`
    with the basis transformation matrix :math:`T` using ``change_basis_ad_rep``.

    Args:
        adj (numpy.ndarray): Adjoint representation in old basis.
        basis_change (numpy.ndarray): Basis change matrix from old to new basis.

    Returns:
        numpy.ndarray: Adjoint representation in new basis.

    **Example**

    We choose a basis of a Lie algebra, compute its adjoint representation.

    >>> from pennylane.labs.dla import change_basis_ad_rep
    >>> basis = [qml.X(0), qml.Y(0), qml.Z(0)]
    >>> adj = qml.structure_constants(basis)

    Now we change the basis and re-compute the adjoint representation in that new basis.

    >>> basis_change = np.array([[1., 1., 0.], [0., 1., 1.], [0., 1., 1.]])
    >>> new_ops = [qml.sum(*[basis_change[i,j] * basis[j] for j in range(3)]) for i in range(3)]
    >>> new_adj = qml.structure_constants(new_ops)

    We confirm that instead of re-computing the adjoint representation (typically expensive), we can
    transform the old adjoint representation with the change of basis matrix.

    >>> new_adj_re = change_basis_ad_rep(adj, basis_change)
    np.allclose(new_adj, new_adj_re)
    """
    # Perform the einsum contraction "mnp, hm, in, jp -> hij" via three einsum steps
    new_adj = np.einsum("mnp,im->inp", adj, np.linalg.pinv(basis_change.T))
    new_adj = np.einsum("mnp,in->mip", new_adj, basis_change)
    return np.einsum("mnp,ip->mni", new_adj, basis_change)


def adjvec_to_op(adj_vecs, basis, is_orthogonal=True):
    r"""Transform adjoint vector representations back into operator format.

    This function simply reconstructs :math:`\hat{O} = \sum_j c_j \hat{b}_j` given the adjoint vector
    representation :math:`c_j` and basis :math:`\hat{b}_j`.

    .. seealso:: :func:`~op_to_adjvec`

    Args:
        adj_vecs (np.ndarray): collection of vectors with shape ``(batch, len(basis))``
        basis (List[Union[PauliSentence, Operator, np.ndarray]]): collection of basis operators
        is_orthogonal (bool): Whether the ``basis`` consists of orthogonal elements.

    Returns:
        list: collection of operators corresponding to the input vectors read in the input basis.
        The operators are in the format specified by the elements in ``basis``.

    **Example**

    >>> from pennylane.labs.dla import adjvec_to_op
    >>> c = np.array([[0.5, 0.3, 0.7]])
    >>> basis = [qml.X(0), qml.Y(0), qml.Z(0)]
    >>> adjvec_to_op(c, basis)
    [0.5 * X(0) + 0.3 * Y(0) + 0.7 * Z(0)]

    """
    assert qml.math.shape(adj_vecs)[1] == len(basis)

    if all(isinstance(op, PauliSentence) for op in basis):
        if not is_orthogonal:
            gram = _gram_ps(basis)
            adj_vecs = np.tensordot(adj_vecs, np.linalg.pinv(sqrtm(gram)), axes=[[1], [0]])
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
            adj_vecs = np.tensordot(adj_vecs, np.linalg.pinv(sqrtm(gram)), axes=[[1], [0]])
        res = []
        for vec in adj_vecs:
            op_j = sum(c * op for c, op in zip(vec, basis))
            op_j = qml.simplify(op_j)
            res.append(op_j)
        return res

    if isinstance(basis, np.ndarray) or all(isinstance(op, np.ndarray) for op in basis):
        if not is_orthogonal:
            gram = trace_inner_product(basis, basis).real
            adj_vecs = np.tensordot(adj_vecs, np.linalg.pinv(sqrtm(gram)), axes=[[1], [0]])
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
        inv_gram = np.linalg.pinv(sqrtm(gram))

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
    ops: Iterable[Union[PauliSentence, Operator, np.ndarray]],
    basis: Union[PauliSentence, Operator, np.ndarray],
    is_orthogonal: bool = True,
):
    r"""Decompose a batch of operators onto a given operator basis.

    The adjoint vector representation is provided by the coefficients :math:`c_j` in a given operator
    basis of the operator :math:`\hat{b}_j` such that the input operator can be written as
    :math:`\hat{O} = \sum_j c_j \hat{b}_j`.

    .. seealso:: :func:`~adjvec_to_op`

    Args:
        ops (Iterable[Union[PauliSentence, Operator, np.ndarray]]): List of operators to decompose
        basis (Iterable[Union[PauliSentence, Operator, np.ndarray]]): Operator basis
        is_orthogonal (bool): Whether the basis is orthogonal with respect to the trace inner
            product. Defaults to ``True``, which allows to skip some computations.

    Returns:
        np.ndarray: The batch of coefficient vectors of the operators' ``ops`` expressed in
        ``basis``. The shape is ``(len(ops), len(basis)``.

    The format of the resulting operators is determined by the ``type`` in ``basis``.
    If ``is_orthogonal=True`` (the default), only normalization is taken into account
    in the projection. For ``is_orthogonal=False``, orthogonalization also is considered.

    **Example**

    The basis can be numerical or operators.

    >>> from pennylane.labs.dla import op_to_adjvec
    >>> op = qml.X(0) + 0.5 * qml.Y(0)
    >>> basis = [qml.X(0), qml.Y(0), qml.Z(0)]
    >>> op_to_adjvec([op], basis)
    array([[1. , 0.5, 0. ]])
    >>> op_to_adjvec([op], [op.matrix() for op in basis])
    array([[1. , 0.5, 0. ]])

    Note how the function always expects an ``Iterable`` of operators as input.

    The ``ops`` can also be numerical, but then ``basis`` has to be numerical as well.

    >>> op = op.matrix()
    >>> op_to_adjvec([op], [op.matrix() for op in basis])
    array([[1. , 0.5, 0. ]])
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
            norm = np.einsum("bij,bji->b", basis, basis).real / basis[0].shape[0]
            return res / norm
        gram = trace_inner_product(basis, basis).real
        sqrtm_gram = sqrtm(gram)
        # Imaginary component is an artefact
        assert np.allclose(sqrtm_gram.imag, 0.0, atol=1e-16)
        return np.einsum("ij,kj->ki", np.linalg.pinv(sqrtm_gram.real), res)

    raise NotImplementedError(
        "At least one operator in the specified basis is of unsupported type, "
        "or not all operators are of the same type."
    )
