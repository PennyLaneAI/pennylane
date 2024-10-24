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
# pylint: disable=too-many-return-statements
from itertools import combinations, product
from typing import List, Union

import numpy as np

import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops.qubit.matrix_ops import _walsh_hadamard_transform
from pennylane.pauli import PauliSentence, PauliWord


def _make_phase_mat(n):
    phase_mat = qml.math.ones((2,) * (2 * n), dtype=complex)
    for idx in range(n):
        index = [slice(None)] * (2 * n)
        index[idx] = index[idx + n] = 1
        phase_mat[tuple(index)] *= 1j
    phase_mat = qml.math.reshape(phase_mat, (2**n, 2**n))
    return phase_mat


def _make_permutation_indices(dim):
    indices = [qml.math.arange(dim)]
    for idx in range(dim - 1):
        indices.append(qml.math.bitwise_xor(indices[-1], (idx + 1) ^ (idx)))
    return indices


def _make_extraction_indices(n):
    indices = []
    for pauli_rep in product("IXYZ", repeat=n):
        bit_array = qml.math.array(
            [[(rep in "YZ"), (rep in "XY")] for rep in pauli_rep], dtype=int
        ).T
        indices.append(tuple(int("".join(map(str, x)), 2) for x in bit_array))
    return tuple(zip(*indices))


def pauli_coefficients(H):
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


def pauli_decompose(H, tol=None, pauli: bool = False):
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


def check_commutation(ops1, ops2, vspace):
    """Helper function to check things like [k, m] subspace m; expensive"""
    assert_vals = []
    for o1 in ops1:
        for o2 in ops2:
            com = o1.commutator(o2)
            com.simplify()
            if len(com) != 0:
                assert_vals.append(not vspace.is_independent(com))
            else:
                assert_vals.append(True)

    return all(assert_vals)


def check_all_commuting(ops: List[Union[PauliSentence, np.ndarray, Operator]]):
    """Helper function to check if all operators in a set of operators commute"""
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
    """Helper function to check the validity of a Cartan decomposition by checking its commutation relations"""
    if any(isinstance(op, np.ndarray) for op in k):
        k = [qml.pauli_decompose(op).pauli_rep for op in k]
    if any(isinstance(op, np.ndarray) for op in m):
        m = [qml.pauli_decompose(op).pauli_rep for op in m]

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


def apply_basis_change(change_op, targets):
    """Helper function for recursive Cartan decompositions"""
    if single_target := np.ndim(targets) == 2:
        targets = [targets]
    if isinstance(targets, list):
        targets = np.array(targets)
    # Compute x V^\dagger for all x in ``targets``. ``moveaxis`` brings the batch axis to the front
    out = np.moveaxis(np.tensordot(change_op, targets, axes=[[1], [1]]), 1, 0)
    out = np.tensordot(out, change_op.conj().T, axes=[[2], [0]])
    if single_target:
        return out[0]
    return out
