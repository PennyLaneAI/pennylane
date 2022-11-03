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
from operator import matmul
from itertools import product
from typing import Union
from functools import reduce, singledispatch

import numpy as np

from .pauli_arithmetic import PauliWord, PauliSentence, I, X, Y, Z, mat_map, op_map, op_to_str_map
from .utils import is_pauli_word

from pennylane.ops import Sum, Prod, SProd, PauliX, PauliY, PauliZ, Identity
from pennylane.operation import Tensor
from pennylane.ops.qubit.hamiltonian import Hamiltonian


@singledispatch
def pauli_sentence(op):
    """Return the PauliSentence representation of an arithmetic operator or Hamiltonian.

    Args:
        op (~.Operator): The operator or hamiltonian that needs to converted.

    Raises:
        ValueError: Op must be a linear combination of Pauli operators

    Returns ~.PauliSentence
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
def _(op: Identity):
    return PauliSentence({PauliWord({}): 1.0})


@pauli_sentence.register
def _(op: Tensor):
    if not is_pauli_word(op):
        raise ValueError(f"Op must be a linear combination of Pauli operators only, got: {op}")

    pw = {}
    for factor in op.obs:
        wire, pauli = (factor.wires[0], op_to_str_map[factor.__class__])
        pw[wire] = pauli

    return PauliSentence({PauliWord(pw): 1.0})


@pauli_sentence.register
def _(op: Prod):
    factors = [pauli_sentence(factor) for factor in op.operands]
    return reduce(lambda a, b: a * b, factors)


@pauli_sentence.register
def _(op: SProd):
    ps = pauli_sentence(op.base)
    for pw, coeff in ps.items():
        ps[pw] = coeff * op.scalar
    return ps


@pauli_sentence.register
def _(op: Hamiltonian):
    if not is_pauli_word(op):
        raise ValueError(f"Op must be a linear combination of Pauli operators only, got: {op}")

    summands = []
    for coeff, op in zip(*op.terms()):
        ps = pauli_sentence(op)
        for pw, sub_coeff in ps.items():
            ps[pw] = coeff * sub_coeff
        summands.append(ps)

    return reduce(lambda a, b: a + b, summands)


@pauli_sentence.register
def _(op: Sum):
    summands = [pauli_sentence(summand) for summand in op.operands]
    return reduce(lambda a, b: a + b, summands)


def decompose(
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
            of Pauli operators, either as a ``~.Hamiltonian`` or ``~.PauliSentence`` instance.

    **Example:**

    We can use this function to compute the Pauli operator decomposition of an arbitrary Hermitian
    matrix:

    >>> A = np.array(
    ... [[-2, -2+1j, -2, -2], [-2-1j,  0,  0, -1], [-2,  0, -2, -1], [-2, -1, -1,  0]])
    >>> H = decompose(A)
    >>> print(H)
    (-1.0) [I0 I1]
    + (-1.5) [X1]
    + (-0.5) [Y1]
    + (-1.0) [Z1]
    + (-1.5) [X0]
    + (-1.0) [X0 X1]
    + (-0.5) [X0 Z1]
    + (1.0) [Y0 Y1]
    + (-0.5) [Z0 X1]
    + (-0.5) [Z0 Y1]

    We can return a ``~.PauliSentence`` instance by using the keyword argument ``pauli=True``:

    >>> ps = decompose(A, pauli=True)
    >>> print(ps)
    """
    n = int(np.log2(len(H)))
    N = 2**n

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
                if hide_identity
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
