# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility functions to convert between ``~.PauliSentence`` and other PennyLane operators.
"""
from collections import defaultdict
from functools import reduce, singledispatch
from itertools import product
from operator import matmul

import numpy as np
import scipy.sparse as sps

import pennylane as qp
from pennylane import math
from pennylane.math.utils import is_abstract
from pennylane.ops import Identity, LinearCombination, PauliX, PauliY, PauliZ, Prod, SProd, Sum
from pennylane.ops.qubit.matrix_ops import _walsh_hadamard_transform

from .pauli_arithmetic import I, PauliSentence, PauliWord, X, Y, Z, op_map
from .utils import is_pauli_word


def _validate_and_normalize_decomposition_inputs(shape, wire_order=None, is_sparse=False):
    """Validate matrix shape and wire order for Pauli decomposition.

    Args:
        shape: Matrix shape tuple (rows, cols)
        wire_order: Optional list of wires. If None, will be set to range(num_qubits)
        is_sparse: Whether the matrix is sparse (for additional empty matrix check)

    Returns:
        tuple: (num_qubits, wire_order) where wire_order is normalized

    Raises:
        ValueError: If shape is invalid or wire_order is incompatible
    """
    if shape[0] != shape[1]:
        raise ValueError(
            f"The matrix should be square, got {shape}. Use 'padding=True' for rectangular matrices."
        )

    if is_sparse and shape[0] == 0:
        raise ValueError("Cannot decompose an empty matrix.")

    if (
        shape[0] & (shape[0] - 1) != 0
    ):  # 2 powers are of 100... binary, minus 1 to get 011..., sharing no common bit; any other integers share at least one bit with their minus one
        raise ValueError(
            f"Dimension of the matrix should be a power of 2, got {shape}. Use 'padding=True' for these matrices."
        )

    num_qubits = int(math.log2(shape[0]))

    if wire_order is not None and len(wire_order) != num_qubits:
        raise ValueError(
            f"number of wires {len(wire_order)} is not compatible with the number of qubits {num_qubits}"
        )

    if wire_order is None:
        wire_order = range(num_qubits)

    return num_qubits, wire_order


def _generalized_pauli_decompose(  # pylint: disable=too-many-branches
    matrix, hide_identity=False, wire_order=None, pauli=False, padding=False
) -> tuple[qp.typing.TensorLike, list]:
    r"""Decomposes any matrix into a linear combination of Pauli operators.

    This method converts any matrix to a weighted sum of Pauli words acting on :math:`n` qubits
    in time :math:`O(n 4^n)`. The input matrix is first padded with zeros if its dimensions are not
    :math:`2^n\times 2^n` and written as a quantum state in the computational basis following the
    `channel-state duality <https://en.wikipedia.org/wiki/Channel-state_duality>`_.
    A Bell basis transformation is then performed using the
    `Walsh-Hadamard transform <https://en.wikipedia.org/wiki/Hadamard_transform>`_, after which
    coefficients for each of the :math:`4^n` Pauli words are computed while accounting for the
    phase from each ``PauliY`` term occurring in the word.

    Args:
        matrix (tensor_like[complex]): any matrix M, the keyword argument ``padding=True``
            should be provided if the dimension of M is not :math:`2^n\times 2^n`.
        hide_identity (bool): does not include the Identity observable within
            the tensor products of the decomposition if ``True``.
        wire_order (list[Union[int, str]]): the ordered list of wires with respect
            to which the operator is represented as a matrix.
        pauli (bool): return a PauliSentence instance if ``True``.
        padding (bool): makes the function compatible with rectangular matrices and square matrices
            that are not of shape :math:`2^n\times 2^n` by padding them with zeros if ``True``.

    Returns:
        Tuple[qp.math.array[complex], list]: the matrix decomposed as a linear combination of Pauli operators
        as a tuple consisting of an array of complex coefficients and a list of corresponding Pauli terms.

    **Example:**

    We can use this function to compute the Pauli operator decomposition of an arbitrary matrix:

    >>> A = np.array(
    ... [[-2, -2+1j, -2, -2], [-2-1j,  0,  0, -1], [-2,  0, -2, -1], [-2, -1, -1,  1j]])
    >>> coeffs, obs = qp.pauli.conversion._generalized_pauli_decompose(A)
    >>> coeffs
    array([-1. +0.25j, -1.5+0.j  , -0.5+0.j  , -1. -0.25j, -1.5+0.j  ,
       -1. +0.j  , -0.5+0.j  ,  1. -0.j  ,  0. -0.25j, -0.5+0.j  ,
       -0.5+0.j  ,  0. +0.25j])
    >>> obs
    [I(0) @ I(1),
    I(0) @ X(1),
    I(0) @ Y(1),
    I(0) @ Z(1),
    X(0) @ I(1),
    X(0) @ X(1),
    X(0) @ Z(1),
    Y(0) @ Y(1),
    Z(0) @ I(1),
    Z(0) @ X(1),
    Z(0) @ Y(1),
    Z(0) @ Z(1)]

    We can also set custom wires using the ``wire_order`` argument:

    >>> coeffs, obs = qp.pauli.conversion._generalized_pauli_decompose(A, wire_order=['a', 'b'])
    >>> obs
    [I('a') @ I('b'),
    I('a') @ X('b'),
    I('a') @ Y('b'),
    I('a') @ Z('b'),
    X('a') @ I('b'),
    X('a') @ X('b'),
    X('a') @ Z('b'),
    Y('a') @ Y('b'),
    Z('a') @ I('b'),
    Z('a') @ X('b'),
    Z('a') @ Y('b'),
    Z('a') @ Z('b')]

    .. details::
        :title: Advanced Usage Details
        :href: usage-decompose-operation

        For non-square matrices, we need to provide the ``padding=True`` keyword argument:

        >>> A = np.array([[-2, -2 + 1j]])
        >>> coeffs, obs = qp.pauli.conversion._generalized_pauli_decompose(A, padding=True)
        >>> coeffs
        array([-1. +0.j , -1. +0.5j, -0.5-1.j , -1. +0.j ])
        >>> obs
        [I(0), X(0), Y(0), Z(0)]

        We can also use the method within a differentiable workflow and obtain gradients:

        >>> A = qp.numpy.array([[-2, -2 + 1j]], requires_grad=True)
        >>> dev = qp.device("default.qubit", wires=1)
        >>> @qp.qnode(dev)
        ... def circuit(A):
        ...    coeffs, _ = qp.pauli.conversion._generalized_pauli_decompose(A, padding=True)
        ...    qp.RX(qp.math.real(coeffs[2]), 0)
        ...    return qp.expval(qp.Z(0))
        >>> qp.grad(circuit)(A)
        array([[0.+0.j        , 0.+0.2397...j]])

    """
    # Ensuring original matrix is not manipulated and we support builtin types.
    matrix = qp.math.convert_like(matrix, next(iter([*matrix[0]]), []))

    # Pad with zeros to make the matrix shape equal and a power of two.
    if padding:
        shape = qp.math.shape(matrix)
        num_qubits = int(qp.math.ceil(qp.math.log2(qp.math.max(shape))))
        if shape[0] != shape[1] or shape[0] != 2**num_qubits:
            padd_diffs = qp.math.abs(qp.math.array(shape) - 2**num_qubits)
            padding = (
                ((0, padd_diffs[0]), (0, padd_diffs[1]))
                if qp.math.get_interface(matrix) != "torch"
                else ((padd_diffs[0], 0), (padd_diffs[1], 0))
            )
            matrix = qp.math.pad(matrix, padding, mode="constant", constant_values=0)

    shape = qp.math.shape(matrix)
    num_qubits, wire_order = _validate_and_normalize_decomposition_inputs(
        shape, wire_order, is_sparse=False
    )

    # Permute by XORing
    indices = [qp.math.array(range(shape[0]))]
    for idx in range(shape[0] - 1):
        indices.append(qp.math.bitwise_xor(indices[-1], (idx + 1) ^ (idx)))
    term_mat = qp.math.cast(
        qp.math.stack(
            [qp.math.gather(matrix[idx], indice) for idx, indice in enumerate(indices)]
        ),
        complex,
    )

    # Perform Hadamard transformation on coloumns
    hadamard_transform_mat = _walsh_hadamard_transform(qp.math.transpose(term_mat))

    # Account for the phases from Y
    phase_mat = qp.math.ones(shape, dtype=complex).reshape((2,) * (2 * num_qubits))
    for idx in range(num_qubits):
        index = [slice(None)] * (2 * num_qubits)
        index[idx] = index[idx + num_qubits] = 1
        phase_mat[tuple(index)] *= 1j
    phase_mat = qp.math.convert_like(qp.math.reshape(phase_mat, shape), matrix)

    # c_00 + c_11 -> I; c_00 - c_11 -> Z; c_01 + c_10 -> X; 1j*(c_10 - c_01) -> Y
    # https://quantumcomputing.stackexchange.com/a/31790
    term_mat = qp.math.transpose(qp.math.multiply(hadamard_transform_mat, phase_mat))

    # Obtain the coefficients for each Pauli word
    coeffs, obs = [], []
    for pauli_rep in product("IXYZ", repeat=num_qubits):
        bit_array = qp.math.array(
            [[(rep in "YZ"), (rep in "XY")] for rep in pauli_rep], dtype=int
        ).T
        coefficient = term_mat[tuple(int("".join(map(str, x)), 2) for x in bit_array)]

        if not is_abstract(matrix) and qp.math.allclose(coefficient, 0):
            continue

        observables = (
            [(o, w) for w, o in zip(wire_order, pauli_rep) if o != I]
            if hide_identity and not all(t == I for t in pauli_rep)
            else [(o, w) for w, o in zip(wire_order, pauli_rep)]
        )
        if observables:
            coeffs.append(coefficient)
            obs.append(observables)

    coeffs = qp.math.stack(coeffs)

    if not pauli:
        with qp.QueuingManager.stop_recording():
            obs = [reduce(matmul, [op_map[o](w) for o, w in obs_term]) for obs_term in obs]

    return (coeffs, obs)


def _generalized_pauli_decompose_sparse(  # pylint: disable=too-many-statements,too-many-branches
    matrix, hide_identity=False, wire_order=None, pauli=False, padding=False
) -> tuple[qp.typing.TensorLike, list]:
    r"""Sparse SciPy implementation of the generalized Pauli decomposition.

    This function computes a weighted sum of Pauli words that is equivalent to the input
    matrix, using a sparsity-aware routine that iterates over the nonzero entries without
    converting the matrix to a dense array. It supports padding for non-power-of-two or
    rectangular inputs and returns either operator tensors or Pauli-word data depending on
    the ``pauli`` flag.

    Args:
        matrix (scipy.sparse matrix): Any sparse matrix. If its dimension is not
            :math:`2^n \times 2^n`, use ``padding=True`` to pad with zeros to the next power of two.
        hide_identity (bool): If ``True``, Identity factors are omitted within tensor products
            of the decomposition terms.
        wire_order (list[Union[int, str]] | None): The ordered list of wires corresponding to
            the matrix qubit order. If ``None``, uses ``range(n)``.
        pauli (bool): If ``True``, returns a list of Pauli-word specifications as ``(char, wire)``
            pairs per term. If ``False``, returns PennyLane operator tensors for each term.
        padding (bool): If ``True``, enables zero-padding to make the matrix square with
            side length a power of two.

    Ordering convention:
        Pauli words are constructed MSB-first; the leftmost character corresponds to
        ``wire_order[0]`` and the rightmost to ``wire_order[-1]``.

    Returns:
        Tuple[qp.typing.TensorLike, list]:
            A tuple ``(coeffs, terms)`` where ``coeffs`` is a complex-valued array of coefficients.
            ``terms`` is either a list of operator tensors (if ``pauli=False``) or a list of
            lists of ``(pauli_char, wire)`` pairs (if ``pauli=True``).

    Raises:
        ValueError: If the input has the wrong shape (not square or not a power of two when
            ``padding=False``), or if the matrix is empty.

    Example:
        >>> import pennylane as qp
        >>> import scipy.sparse as sps
        >>> # Decompose a 2-qubit sparse matrix: Z(0) @ Z(1) + 0.5 * X(0)
        >>> # Matrix: [[1, 0, 0.5, 0], [0, -1, 0, 0.5], [0.5, 0, -1, 0], [0, 0.5, 0, 1]]
        >>> sparse_matrix = sps.csr_matrix(
        ...     [[1, 0, 0.5, 0], [0, -1, 0, 0.5], [0.5, 0, -1, 0], [0, 0.5, 0, 1]]
        ... )
        >>> coeffs, terms = qp.pauli.conversion._generalized_pauli_decompose_sparse(
        ...     sparse_matrix, wire_order=[0, 1]
        ... )
        >>> coeffs
        array([1. +0.j, 0.5+0.j])
        >>> terms
        [Z(0) @ Z(1), X(0) @ I(1)]
    """
    sparse_matrix = sps.coo_matrix(matrix)
    # Sum duplicate (row, col) entries as COO format allows multiple entries
    # for the same position, which must be combined before processing.
    sparse_matrix.sum_duplicates()
    sparse_matrix.eliminate_zeros()
    shape = sparse_matrix.shape

    if padding:
        max_dim = max(shape)
        if max_dim == 0:
            target_dim = 1
        else:
            target_dim = int(2 ** math.ceil(math.log2(max_dim)))
        if shape != (target_dim, target_dim):
            sparse_matrix = sps.coo_matrix(
                (sparse_matrix.data, (sparse_matrix.row, sparse_matrix.col)),
                shape=(target_dim, target_dim),
            )
            shape = sparse_matrix.shape

    num_qubits, wire_order = _validate_and_normalize_decomposition_inputs(
        shape, wire_order, is_sparse=True
    )

    coeffs_map: dict[str, complex] = defaultdict(complex)
    rows, cols, data = sparse_matrix.row, sparse_matrix.col, sparse_matrix.data

    # Decompose each nonzero matrix entry into Pauli word contributions
    for row, col, value in zip(rows, cols, data):
        contributions = [("", complex(value))]

        # Process each qubit position (MSB first)
        for wire in range(num_qubits):
            bit_index = num_qubits - 1 - wire
            row_bit = (row >> bit_index) & 1
            col_bit = (col >> bit_index) & 1

            # Determine Pauli operators diagonal (I/Z) or off-diagonal (X/Y)
            if row_bit == col_bit:
                z_coeff = 0.5 if row_bit == 0 else -0.5
                options = (("I", 0.5), ("Z", z_coeff))
            else:
                if row_bit == 0:
                    options = (("X", 0.5), ("Y", 0.5j))
                else:
                    options = (("X", 0.5), ("Y", -0.5j))

            # Expand contributions each prefix branches into I/Z or X/Y options
            new_contributions = []
            for prefix, coeff in contributions:
                for pauli_char, factor in options:
                    new_contributions.append((prefix + pauli_char, coeff * factor))
            contributions = new_contributions

        for word, coeff in contributions:
            coeffs_map[word] += coeff

    # Filter out coefficients close to zero
    coeffs = []
    obs_terms = []
    for word, coeff in coeffs_map.items():
        if qp.math.allclose(coeff, 0):
            continue
        if hide_identity and not all(char == I for char in word):
            observables = [(char, wire) for wire, char in zip(wire_order, word) if char != I]
        else:
            observables = [(char, wire) for wire, char in zip(wire_order, word)]
        coeffs.append(coeff)
        obs_terms.append(observables)

    if not coeffs:
        coeffs = qp.math.cast(qp.math.array([], dtype=complex), complex)
    else:
        coeffs = qp.math.cast(qp.math.stack(coeffs), complex)

    if not pauli:
        with qp.QueuingManager.stop_recording():
            obs_terms = [reduce(matmul, [op_map[o](w) for o, w in term]) for term in obs_terms]

    return (coeffs, obs_terms)


def _validate_sparse_matrix_shape(shape):
    """Validate that a sparse matrix has the correct shape for decomposition.

    Args:
        shape: Matrix shape tuple (rows, cols)

    Raises:
        ValueError: If shape is invalid for decomposition
    """
    if shape[0] == 0:
        raise ValueError("Cannot decompose an empty matrix.")
    if shape[0] != shape[1]:
        raise ValueError(
            f"The matrix should be square, got {shape}. Use 'padding=True' for rectangular matrices."
        )
    if (
        shape[0] & (shape[0] - 1) != 0
    ):  # 2 powers are of 100... binary, minus 1 to get 011..., sharing no common bit; any other integers share at least one bit with their minus one
        raise ValueError(
            f"Dimension of the matrix should be a power of 2, got {shape}. Use 'padding=True' for these matrices."
        )


def _check_hermitian_sparse(H):
    """Check if a sparse matrix is Hermitian.

    Args:
        H: Sparse matrix to check

    Raises:
        ValueError: If the matrix is not Hermitian
    """
    adjoint = H.getH() if hasattr(H, "getH") else H.transpose().conjugate()
    diff = H - adjoint
    diff.eliminate_zeros()
    nnz = getattr(diff, "nnz", None)
    if nnz is None:
        nnz = diff.count_nonzero()
    if nnz:
        max_diff = np.abs(diff.data).max()
        if max_diff > 1e-8:
            raise ValueError(f"The matrix is not Hermitian. (max diff: {max_diff})")


def pauli_decompose(
    H, hide_identity=False, wire_order=None, pauli=False, check_hermitian=True
) -> LinearCombination | PauliSentence:
    r"""Decomposes a Hermitian matrix into a linear combination of Pauli operators.

    Args:
        H (tensor_like[complex] or scipy.sparse matrix): a Hermitian matrix of dimension :math:`2^n\times 2^n`.
            Scipy sparse matrices are also supported and are processed natively without converting to dense format,
            enabling efficient decomposition of large sparse matrices.
        hide_identity (bool): does not include the Identity observable within
            the tensor products of the decomposition if ``True``.
        wire_order (list[Union[int, str]]): the ordered list of wires with respect
            to which the operator is represented as a matrix.
        pauli (bool): return a :class:`~.PauliSentence` instance if ``True``.
        check_hermitian (bool): check if the provided matrix is Hermitian if ``True``.

    Returns:
        Union[~.LinearCombination, ~.PauliSentence]: the matrix decomposed as a linear combination
        of Pauli operators, returned either as a :class:`~.ops.LinearCombination` or :class:`~.PauliSentence`
        instance.

    **Example:**

    We can use this function to compute the Pauli operator decomposition of an arbitrary Hermitian
    matrix:

    >>> import pennylane as qp
    >>> import numpy as np
    >>> A = np.array(
    ... [[-2, -2+1j, -2, -2], [-2-1j,  0,  0, -1], [-2,  0, -2, -1], [-2, -1, -1,  0]])
    >>> H = qp.pauli_decompose(A)
    >>> import pprint
    >>> pprint.pprint(H)
    (
        -1.0 * (I(0) @ I(1))
      + -1.5 * (I(0) @ X(1))
      + -0.5 * (I(0) @ Y(1))
      + -1.0 * (I(0) @ Z(1))
      + -1.5 * (X(0) @ I(1))
      + -1.0 * (X(0) @ X(1))
      + -0.5 * (X(0) @ Z(1))
      + 1.0 * (Y(0) @ Y(1))
      + -0.5 * (Z(0) @ X(1))
      + -0.5 * (Z(0) @ Y(1))
    )

    We can return a :class:`~.PauliSentence` instance by using the keyword argument ``pauli=True``:

    >>> ps = qp.pauli_decompose(A, pauli=True)
    >>> print(ps)
    -1.0 * I
    + -1.5 * X(1)
    + -0.5 * Y(1)
    + -1.0 * Z(1)
    + -1.5 * X(0)
    + -1.0 * X(0) @ X(1)
    + -0.5 * X(0) @ Z(1)
    + 1.0 * Y(0) @ Y(1)
    + -0.5 * Z(0) @ X(1)
    + -0.5 * Z(0) @ Y(1)

    By default the wires are numbered [0, 1, ..., n], but we can also set custom wires using the
    ``wire_order`` argument:

    >>> ps = qp.pauli_decompose(A, pauli=True, wire_order=['a', 'b'])
    >>> print(ps)
    -1.0 * I
    + -1.5 * X(b)
    + -0.5 * Y(b)
    + -1.0 * Z(b)
    + -1.5 * X(a)
    + -1.0 * X(a) @ X(b)
    + -0.5 * X(a) @ Z(b)
    + 1.0 * Y(a) @ Y(b)
    + -0.5 * Z(a) @ X(b)
    + -0.5 * Z(a) @ Y(b)

    .. details::
        :title: Theory
        :href: theory

        This method internally uses a generalized decomposition routine to convert the matrix to a
        weighted sum of Pauli words acting on :math:`n` qubits in time :math:`O(n 4^n)`. The input
        matrix is written as a quantum state in the computational basis following the
        `channel-state duality <https://en.wikipedia.org/wiki/Channel-state_duality>`_.
        A Bell basis transformation is then performed using the
        `Walsh-Hadamard transform <https://en.wikipedia.org/wiki/Hadamard_transform>`_, after which
        coefficients for each of the :math:`4^n` Pauli words are computed while accounting for the
        phase from each ``PauliY`` term occurring in the word.

        Scipy sparse matrices are also supported and processed natively without converting to
        dense format, enabling efficient decomposition of large sparse matrices. For example:

        >>> import scipy.sparse as sps
        >>> sparse_H = sps.csr_matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        >>> qp.pauli_decompose(sparse_H)
        1.0 * (Z(0) @ Z(1))

    """
    is_sparse = sps.issparse(H)
    shape = H.shape if is_sparse else qp.math.shape(H)

    if is_sparse:
        _validate_sparse_matrix_shape(shape)

    n = int(math.log2(shape[0]))
    N = 2**n

    if check_hermitian:
        if shape != (N, N):
            raise ValueError("The matrix should have shape (2**n, 2**n), for any qubit number n>=1")

        if not is_abstract(H):
            if is_sparse:
                _check_hermitian_sparse(H)
            else:
                if not qp.math.allclose(H, qp.math.conj(qp.math.transpose(H))):
                    raise ValueError("The matrix is not Hermitian")

    _pauli_decompose = (
        _generalized_pauli_decompose_sparse if is_sparse else _generalized_pauli_decompose
    )
    coeffs, obs = _pauli_decompose(
        H, hide_identity=hide_identity, wire_order=wire_order, pauli=pauli, padding=True
    )

    if check_hermitian:
        coeffs = qp.math.real(coeffs)

    if pauli:
        return PauliSentence(
            {
                PauliWord({w: o for o, w in obs_n_wires}): coeff
                for coeff, obs_n_wires in zip(coeffs, obs)
            }
        )

    return qp.Hamiltonian(coeffs, obs)


def pauli_sentence(op):
    """Return the PauliSentence representation of an arithmetic operator or Hamiltonian.

    Args:
        op (~.Operator): The operator or Hamiltonian that needs to be converted.

    Raises:
        ValueError: Op must be a linear combination of Pauli operators

    Returns:
        .PauliSentence: the PauliSentence representation of an arithmetic operator or Hamiltonian
    """

    if isinstance(op, PauliWord):
        return PauliSentence({op: 1.0})

    if isinstance(op, PauliSentence):
        return op

    if (ps := op.pauli_rep) is not None:
        return ps

    return _pauli_sentence(op)


def is_pauli_sentence(op):
    """Returns True of the operator is a PauliSentence and False otherwise."""
    if op.pauli_rep is not None:
        return True
    return False


@singledispatch
def _pauli_sentence(op):
    """Private function to dispatch"""
    raise ValueError(f"Op must be a linear combination of Pauli operators only, got: {op}")


@_pauli_sentence.register
def _(op: PauliX):
    return PauliSentence({PauliWord({op.wires[0]: X}): 1.0})


@_pauli_sentence.register
def _(op: PauliY):
    return PauliSentence({PauliWord({op.wires[0]: Y}): 1.0})


@_pauli_sentence.register
def _(op: PauliZ):
    return PauliSentence({PauliWord({op.wires[0]: Z}): 1.0})


@_pauli_sentence.register
def _(op: Identity):
    return PauliSentence({PauliWord({}): 1.0})


@_pauli_sentence.register
def _(op: Prod):
    factors = (_pauli_sentence(factor) for factor in op)
    return reduce(lambda a, b: a @ b, factors)


@_pauli_sentence.register
def _(op: SProd):
    ps = _pauli_sentence(op.base)
    for pw, coeff in ps.items():
        ps[pw] = coeff * op.scalar
    return ps


@_pauli_sentence.register(LinearCombination)
def _(op: LinearCombination):
    if not all(is_pauli_word(o) for o in op.ops):
        raise ValueError(f"Op must be a linear combination of Pauli operators only, got: {op}")

    return op._build_pauli_rep()  # pylint: disable=protected-access


@_pauli_sentence.register
def _(op: Sum):
    ps = PauliSentence()
    for term in op:
        ps += _pauli_sentence(term)
    return ps
