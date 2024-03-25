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
from functools import reduce, singledispatch
from itertools import product
from operator import matmul
from typing import Union, Tuple

import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import (
    Hamiltonian,
    LinearCombination,
    Identity,
    PauliX,
    PauliY,
    PauliZ,
    Prod,
    SProd,
    Sum,
)
from pennylane.ops.qubit.matrix_ops import _walsh_hadamard_transform

from .pauli_arithmetic import I, PauliSentence, PauliWord, X, Y, Z, op_map
from .utils import is_pauli_word


# pylint: disable=too-many-branches
def _generalized_pauli_decompose(
    matrix, hide_identity=False, wire_order=None, pauli=False, padding=False
) -> Tuple[qml.typing.TensorLike, list]:
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
        Tuple[qml.math.array[complex], list]: the matrix decomposed as a linear combination of Pauli operators
        as a tuple consisting of an array of complex coefficients and a list of corresponding Pauli terms.

    **Example:**

    We can use this function to compute the Pauli operator decomposition of an arbitrary matrix:

    >>> A = np.array(
    ... [[-2, -2+1j, -2, -2], [-2-1j,  0,  0, -1], [-2,  0, -2, -1], [-2, -1, -1,  1j]])
    >>> coeffs, obs = qml.pauli.conversion._generalized_pauli_decompose(A)
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

    >>> coeffs, obs = qml.pauli.conversion._generalized_pauli_decompose(A, wire_order=['a', 'b'])
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
        >>> coeffs, obs = qml.pauli.conversion._generalized_pauli_decompose(A, padding=True)
        >>> coeffs
        ([-1. +0.j , -1. +0.5j, -0.5-1.j , -1. +0.j ])
        >>> obs
        [I(0), X(0), Y(0), Z(0)]

        We can also use the method within a differentiable workflow and obtain gradients:

        >>> A = qml.numpy.array([[-2, -2 + 1j]], requires_grad=True)
        >>> dev = qml.device("default.qubit", wires=1)
        >>> @qml.qnode(dev)
        ... def circuit(A):
        ...    coeffs, _ = qml.pauli.conversion._generalized_pauli_decompose(A, padding=True)
        ...    qml.RX(qml.math.real(coeffs[2]), 0)
        ...    return qml.expval(qml.Z(0))
        >>> qml.grad(circuit)(A)
        array([[0.+0.j        , 0.+0.23971277j]])

    """
    # Ensuring original matrix is not manipulated and we support builtin types.
    matrix = qml.math.convert_like(matrix, next(iter([*matrix[0]]), []))

    # Pad with zeros to make the matrix shape equal and a power of two.
    if padding:
        shape = qml.math.shape(matrix)
        num_qubits = int(qml.math.ceil(qml.math.log2(qml.math.max(shape))))
        if shape[0] != shape[1] or shape[0] != 2**num_qubits:
            padd_diffs = qml.math.abs(qml.math.array(shape) - 2**num_qubits)
            padding = (
                ((0, padd_diffs[0]), (0, padd_diffs[1]))
                if qml.math.get_interface(matrix) != "torch"
                else ((padd_diffs[0], 0), (padd_diffs[1], 0))
            )
            matrix = qml.math.pad(matrix, padding, mode="constant", constant_values=0)

    shape = qml.math.shape(matrix)
    if shape[0] != shape[1]:
        raise ValueError(
            f"The matrix should be square, got {shape}. Use 'padding=True' for rectangular matrices."
        )

    num_qubits = int(qml.math.log2(shape[0]))
    if shape[0] != 2**num_qubits:
        raise ValueError(
            f"Dimension of the matrix should be a power of 2, got {shape}. Use 'padding=True' for these matrices."
        )

    if wire_order is not None and len(wire_order) != num_qubits:
        raise ValueError(
            f"number of wires {len(wire_order)} is not compatible with the number of qubits {num_qubits}"
        )

    if wire_order is None:
        wire_order = range(num_qubits)

    # Permute by XORing
    indices = [qml.math.array(range(shape[0]))]
    for idx in range(shape[0] - 1):
        indices.append(qml.math.bitwise_xor(indices[-1], (idx + 1) ^ (idx)))
    term_mat = qml.math.cast(
        qml.math.stack(
            [qml.math.gather(matrix[idx], indice) for idx, indice in enumerate(indices)]
        ),
        complex,
    )

    # Perform Hadamard transformation on coloumns
    hadamard_transform_mat = _walsh_hadamard_transform(qml.math.transpose(term_mat))

    # Account for the phases from Y
    phase_mat = qml.math.ones(shape, dtype=complex).reshape((2,) * (2 * num_qubits))
    for idx in range(num_qubits):
        index = [slice(None)] * (2 * num_qubits)
        index[idx] = index[idx + num_qubits] = 1
        phase_mat[tuple(index)] *= 1j
    phase_mat = qml.math.convert_like(qml.math.reshape(phase_mat, shape), matrix)

    # c_00 + c_11 -> I; c_00 - c_11 -> Z; c_01 + c_10 -> X; 1j*(c_10 - c_01) -> Y
    # https://quantumcomputing.stackexchange.com/a/31790
    term_mat = qml.math.transpose(qml.math.multiply(hadamard_transform_mat, phase_mat))

    # Obtain the coefficients for each Pauli word
    coeffs, obs = [], []
    for pauli_rep in product("IXYZ", repeat=num_qubits):
        bit_array = qml.math.array(
            [[(rep in "YZ"), (rep in "XY")] for rep in pauli_rep], dtype=int
        ).T
        coefficient = term_mat[tuple(int("".join(map(str, x)), 2) for x in bit_array)]

        if not qml.math.allclose(coefficient, 0):
            observables = (
                [(o, w) for w, o in zip(wire_order, pauli_rep) if o != I]
                if hide_identity and not all(t == I for t in pauli_rep)
                else [(o, w) for w, o in zip(wire_order, pauli_rep)]
            )
            if observables:
                coeffs.append(coefficient)
                obs.append(observables)

    coeffs = qml.math.stack(coeffs)

    if not pauli:
        with qml.QueuingManager.stop_recording():
            obs = [reduce(matmul, [op_map[o](w) for o, w in obs_term]) for obs_term in obs]

    return (coeffs, obs)


def pauli_decompose(
    H, hide_identity=False, wire_order=None, pauli=False, check_hermitian=True
) -> Union[Hamiltonian, PauliSentence]:
    r"""Decomposes a Hermitian matrix into a linear combination of Pauli operators.

    Args:
        H (tensor_like[complex]): a Hermitian matrix of dimension :math:`2^n\times 2^n`.
        hide_identity (bool): does not include the Identity observable within
            the tensor products of the decomposition if ``True``.
        wire_order (list[Union[int, str]]): the ordered list of wires with respect
            to which the operator is represented as a matrix.
        pauli (bool): return a :class:`~.PauliSentence` instance if ``True``.
        check_hermitian (bool): check if the provided matrix is Hermitian if ``True``.

    Returns:
        Union[~.Hamiltonian, ~.PauliSentence]: the matrix decomposed as a linear combination
        of Pauli operators, returned either as a :class:`~.Hamiltonian` or :class:`~.PauliSentence`
        instance.

    **Example:**

    We can use this function to compute the Pauli operator decomposition of an arbitrary Hermitian
    matrix:

    >>> A = np.array(
    ... [[-2, -2+1j, -2, -2], [-2-1j,  0,  0, -1], [-2,  0, -2, -1], [-2, -1, -1,  0]])
    >>> H = qml.pauli_decompose(A)
    >>> print(H)
    (-1.5) [I0 X1]
    + (-1.5) [X0 I1]
    + (-1.0) [I0 I1]
    + (-1.0) [I0 Z1]
    + (-1.0) [X0 X1]
    + (-0.5) [I0 Y1]
    + (-0.5) [X0 Z1]
    + (-0.5) [Z0 X1]
    + (-0.5) [Z0 Y1]
    + (1.0) [Y0 Y1]

    We can return a :class:`~.PauliSentence` instance by using the keyword argument ``pauli=True``:

    >>> ps = qml.pauli_decompose(A, pauli=True)
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

    >>> ps = qml.pauli_decompose(A, pauli=True, wire_order=['a', 'b'])
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

    """
    shape = qml.math.shape(H)
    n = int(qml.math.log2(shape[0]))
    N = 2**n

    if check_hermitian:
        if shape != (N, N):
            raise ValueError("The matrix should have shape (2**n, 2**n), for any qubit number n>=1")

        if not qml.math.allclose(H, qml.math.conj(qml.math.transpose(H))):
            raise ValueError("The matrix is not Hermitian")

    coeffs, obs = _generalized_pauli_decompose(
        H, hide_identity=hide_identity, wire_order=wire_order, pauli=pauli, padding=True
    )

    if check_hermitian:
        coeffs = qml.math.real(coeffs)

    if pauli:
        return PauliSentence(
            {
                PauliWord({w: o for o, w in obs_n_wires}): coeff
                for coeff, obs_n_wires in zip(coeffs, obs)
            }
        )

    return qml.Hamiltonian(coeffs, obs)


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
def _(op: Identity):  # pylint:disable=unused-argument
    return PauliSentence({PauliWord({}): 1.0})


@_pauli_sentence.register
def _(op: Tensor):
    if not is_pauli_word(op):
        raise ValueError(f"Op must be a linear combination of Pauli operators only, got: {op}")

    factors = (_pauli_sentence(factor) for factor in op.obs)
    return reduce(lambda a, b: a @ b, factors)


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


@_pauli_sentence.register(qml.ops.Hamiltonian)
def _(op: qml.ops.Hamiltonian):
    if not all(is_pauli_word(o) for o in op.ops):
        raise ValueError(f"Op must be a linear combination of Pauli operators only, got: {op}")

    def term_2_pauli_word(term):
        if isinstance(term, Tensor):
            pw = {obs.wires[0]: obs.name[-1] for obs in term.non_identity_obs}
        elif isinstance(term, Identity):
            pw = {}
        else:
            pw = dict([(term.wires[0], term.name[-1])])
        return PauliWord(pw)

    ps = PauliSentence()
    for coeff, term in zip(*op.terms()):
        sub_ps = PauliSentence({term_2_pauli_word(term): coeff})
        ps += sub_ps

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
