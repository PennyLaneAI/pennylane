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
from typing import Union

import numpy as np

import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd, Sum

from .pauli_arithmetic import I, PauliSentence, PauliWord, X, Y, Z, mat_map, op_map
from .utils import is_pauli_word


def pauli_decompose(
    H, hide_identity=False, wire_order=None, pauli=False
) -> Union[Hamiltonian, PauliSentence]:
    r"""Decomposes a Hermitian matrix into a linear combination of Pauli operators.

    Args:
        H (array[complex]): a Hermitian matrix of dimension :math:`2^n\times 2^n`.
        hide_identity (bool): does not include the Identity observable within
            the tensor products of the decomposition if ``True``.
        wire_order (list[Union[int, str]]): the ordered list of wires with respect
            to which the operator is represented as a matrix.
        pauli (bool): return a PauliSentence instance if ``True``.

    Returns:
        Union[~.Hamiltonian, ~.PauliSentence]: the matrix decomposed as a linear combination
        of Pauli operators, either as a :class:`~.Hamiltonian` or :class:`~.PauliSentence` instance.

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

    We can also set custom wires using the ``wire_order`` argument:

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
    """
    n = int(np.log2(len(H)))
    N = 2**n

    if wire_order is not None and len(wire_order) != n:
        raise ValueError(
            f"number of wires {len(wire_order)} is not compatible with number of qubits {n}"
        )

    if wire_order is None:
        wire_order = range(n)

    if H.shape != (N, N):
        raise ValueError("The matrix should have shape (2**n, 2**n), for any qubit number n>=1")

    if not np.allclose(H, H.conj().T):
        raise ValueError("The matrix is not Hermitian")

    obs_lst = []
    coeffs = []

    for term in product([I, X, Y, Z], repeat=n):
        matrices = [mat_map[i] for i in term]
        coeff = np.trace(reduce(np.kron, matrices) @ H) / N
        coeff = np.real_if_close(coeff).item()

        if not np.allclose(coeff, 0):
            obs_term = (
                [(o, w) for w, o in zip(wire_order, term) if o != I]
                if hide_identity and not all(t == I for t in term)
                else [(o, w) for w, o in zip(wire_order, term)]
            )

            if obs_term:
                coeffs.append(coeff)
                obs_lst.append(obs_term)

    if pauli:
        return PauliSentence(
            {
                PauliWord({w: o for o, w in obs_n_wires}): coeff
                for coeff, obs_n_wires in zip(coeffs, obs_lst)
            }
        )

    obs = [reduce(matmul, [op_map[o](w) for o, w in obs_term]) for obs_term in obs_lst]
    return Hamiltonian(coeffs, obs)


def pauli_decompose_fast(matrix, pauli=False, wire_order=None):
    r"""Decomposes a square matrix into a linear combination of Pauli operators.

    Args:
        H (array[complex]): a Hermitian matrix of dimension :math:`2^n\times 2^n`.
        hide_identity (bool): does not include the Identity observable within
            the tensor products of the decomposition if ``True``.
        wire_order (list[Union[int, str]]): the ordered list of wires with respect
            to which the operator is represented as a matrix.
        pauli (bool): return a PauliSentence instance if ``True``.

    Returns:
        Union[~.Hamiltonian, ~.PauliSentence]: the matrix decomposed as a linear combination
        of Pauli operators, either as a :class:`~.Hamiltonian` or :class:`~.PauliSentence` instance.

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

    We can also set custom wires using the ``wire_order`` argument:

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
    """

    shape = matrix.shape
    if shape[0] != shape[1]:
        raise ValueError(f"Matrix should be square, got {shape}")

    num_qubits = int(np.log2(shape[0]))
    if shape[0] != 2**num_qubits:
        raise ValueError(f"Dimension of the matrix should be a power of 2, got {shape}")

    if wire_order is None or len(wire_order) != num_qubits:
        wire_map = {wire: wire for wire in range(num_qubits)}
    else:
        wire_map = {wire: idx for idx, wire in enumerate(wire_order)}

    # Permute by XORing
    term_mat = np.zeros(shape, dtype=complex)
    indices = np.array(range(shape[0]))
    for idx in range(shape[0]):
        term_mat[idx, :] = matrix[idx, indices]
        indices ^= (idx + 1) ^ (idx)

    # Hadamard transform
    # c_00 + c_11 -> I; c_00 - c_11 -> Z; c_01 + c_10 -> X; 1j*(c_10 - c_01) -> Y
    term_mat.shape = (2,) * (2 * num_qubits)
    for idx in range(num_qubits):
        index = [slice(None)] * (2 * num_qubits)
        slices, indice, ind = [0, 0, num_qubits], [0, 1, 1], []
        for sc, ic in zip(slices, indice):
            index[idx + sc] = ic
            ind.append(tuple(index))
        a, b, c = ind
        term_mat[a], term_mat[b] = (term_mat[a] + term_mat[b], term_mat[a] - term_mat[b])
        term_mat[c] *= 1j
    term_mat /= shape[0]
    term_mat.shape = shape

    # Convert to Hamiltonian and PauliSentence
    terms = ([], [])
    for pauli_rep in product("IXYZ", repeat=num_qubits):
        bit_array = np.array([[(rep in "YZ"), (rep in "XY")] for rep in pauli_rep], dtype=int).T
        coefficient = term_mat[tuple(int("".join(map(str, x)), 2) for x in bit_array)]
        if coefficient:
            terms[0].append(coefficient)
            terms[1].append(qml.pauli.string_to_pauli_word("".join(pauli_rep), wire_map=wire_map))

    if pauli:
        return qml.pauli.pauli_sentence(qml.Hamiltonian(*terms))

    return qml.Hamiltonian(*terms)

def pauli_decompose_fast_non_square(matrix, pauli=False, wire_order=None):
    r"""Decomposes any matrix into a linear combination of Pauli operators."""

    # Pad with zeros to make the matrix shape equal and a power of two.
    shape = matrix.shape
    num_qubits = int(np.ceil(np.log2(max(shape))))
    if shape[0] != shape[1]:
        padd_diffs = np.abs(np.array(shape) - 2**num_qubits)
        padding = ((0, padd_diffs[0]), (0, padd_diffs[1]))
        matrix = np.pad(matrix, padding, mode='constant', constant_values=0)

    shape = matrix.shape
    if shape[0] != shape[1]:
        raise ValueError(f"Matrix should be square, got {shape}")

    num_qubits = int(np.log2(shape[0]))
    if shape[0] != 2**num_qubits:
        raise ValueError(f"Dimension of the matrix should be a power of 2, got {shape}")

    if wire_order is None or len(wire_order) != num_qubits:
        wire_map = {wire: wire for wire in range(num_qubits)}
    else:
        wire_map = {wire: idx for idx, wire in enumerate(wire_order)}

    # Permute by XORing
    term_mat = np.zeros(shape, dtype=complex)
    indices = np.array(range(shape[0]))
    for idx in range(shape[0]):
        term_mat[idx, :] = matrix[idx, indices]
        indices ^= (idx + 1) ^ (idx)

    # Hadamard transform
    # c_00 + c_11 -> I; c_00 - c_11 -> Z; c_01 + c_10 -> X; 1j*(c_10 - c_01) -> Y
    t1 = time.time()
    term_mat.shape = (2,) * (2 * num_qubits)
    for idx in range(num_qubits):
        index = [slice(None)] * (2 * num_qubits)
        slices, indice, ind = [0, 0, num_qubits], [0, 1, 1], []
        for sc, ic in zip(slices, indice):
            index[idx + sc] = ic
            ind.append(tuple(index))
        a, b, c = ind
        term_mat[a], term_mat[b] = (term_mat[a] + term_mat[b], term_mat[a] - term_mat[b])
        term_mat[c] *= 1j
    term_mat /= shape[0]
    term_mat.shape = shape
    t2 = time.time()

    # Convert to Hamiltonian and PauliSentence
    terms = ([], [])
    for pauli_rep in product("IXYZ", repeat=num_qubits):
        bit_array = np.array([[(rep in "YZ"), (rep in "XY")] for rep in pauli_rep], dtype=int).T
        coefficient = term_mat[tuple(int("".join(map(str, x)), 2) for x in bit_array)]
        if coefficient:
            terms[0].append(coefficient)
            terms[1].append(qml.pauli.string_to_pauli_word("".join(pauli_rep), wire_map=wire_map))

    if pauli:
        return qml.pauli.pauli_sentence(qml.Hamiltonian(*terms))

    return qml.Hamiltonian(*terms)

@singledispatch
def pauli_sentence(op):
    """Return the PauliSentence representation of an arithmetic operator or Hamiltonian.

    Args:
        op (~.Operator): The operator or Hamiltonian that needs to be converted.

    Raises:
        ValueError: Op must be a linear combination of Pauli operators

    Returns:
        .PauliSentence: the PauliSentence representation of an arithmetic operator or Hamiltonian
    """
    raise ValueError(f"Op must be a linear combination of Pauli operators only, got: {op}")


@pauli_sentence.register
def _(op: PauliX):
    return PauliSentence({PauliWord({op.wires[0]: X}): 1.0})


@pauli_sentence.register
def _(op: PauliY):
    return PauliSentence({PauliWord({op.wires[0]: Y}): 1.0})


@pauli_sentence.register
def _(op: PauliZ):
    return PauliSentence({PauliWord({op.wires[0]: Z}): 1.0})


@pauli_sentence.register
def _(op: Identity):  # pylint:disable=unused-argument
    return PauliSentence({PauliWord({}): 1.0})


@pauli_sentence.register
def _(op: Tensor):
    if not is_pauli_word(op):
        raise ValueError(f"Op must be a linear combination of Pauli operators only, got: {op}")

    factors = (pauli_sentence(factor) for factor in op.obs)
    return reduce(lambda a, b: a * b, factors)


@pauli_sentence.register
def _(op: Prod):
    factors = (pauli_sentence(factor) for factor in op)
    return reduce(lambda a, b: a * b, factors)


@pauli_sentence.register
def _(op: SProd):
    ps = pauli_sentence(op.base)
    for pw, coeff in ps.items():
        ps[pw] = coeff * op.scalar
    return ps


@pauli_sentence.register
def _(op: Hamiltonian):
    if not all(is_pauli_word(o) for o in op.ops):
        raise ValueError(f"Op must be a linear combination of Pauli operators only, got: {op}")

    summands = []
    for coeff, term in zip(*op.terms()):
        ps = pauli_sentence(term)
        for pw, sub_coeff in ps.items():
            ps[pw] = coeff * sub_coeff
        summands.append(ps)

    return reduce(lambda a, b: a + b, summands)


@pauli_sentence.register
def _(op: Sum):
    summands = (pauli_sentence(summand) for summand in op)
    return reduce(lambda a, b: a + b, summands)
