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
"""This module contains functions to map bosonic operators to qubit operators."""

from functools import singledispatch
from typing import Union

import numpy as np
import pennylane as qml
from pennylane.pauli import PauliSentence, PauliWord

from .bosonic import BoseSentence, BoseWord


def _get_pauli_op(i, j, qub_id):
    r"""Returns expression to convert qubit-local term ::math::``\ket{x_j}\bra{x^{'}}``
    to qubit operators as given in Eq. (6-9) <https://www.nature.com/articles/s41534-020-0278-0>"""

    c1, c2 = 0.5, -0.5 if i == 1 else 0.5

    if i != j:
        return PauliSentence({PauliWord({qub_id: "X"}): c1, PauliWord({qub_id: "Y"}): c2 * 1j})

    return PauliSentence({PauliWord({}): c1, PauliWord({qub_id: "Z"}): c2})


def binary_mapping(
    bose_operator: Union[BoseWord, BoseSentence],
    nstates_boson: int = 2,
    ps: bool = True,
    wire_map: dict = None,
    tol: float = None,
):
    r"""Convert a bosonic operator to a qubit operator using the standard-binary mapping.
    
    The mapping procedure is described in `arXiv:1507.03271 <https://arxiv.org/pdf/1507.03271>`_.

    Args:
      bose_operator(BoseWord, BoseSentence): the bosonic operator
      nstates_boson(int): Number of states in the boson.
      ps (bool): whether to return the result as a PauliSentence instead of an operator. Defaults to False.
      wire_map (dict): a dictionary defining how to map the orbitals of
      the Bose operator to qubit wires. If None, the integers used to
      order the orbitals will be used as wire labels. Defaults to None.
      tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
      a linear combination of qubit operators

    **Example**

    >>> w = qml.labs.vibrational.BoseWord({(0, 0): "+"})
    >>> binary_mapping(w, nstates_boson=4)
    0.6830127018922193 * X(0)
    + -0.1830127018922193 * X(0) @ Z(1)
    + -0.6830127018922193j * Y(0)
    + 0.1830127018922193j * Y(0) @ Z(1)
    + 0.3535533905932738 * X(0) @ X(1)
    + -0.3535533905932738j * X(0) @ Y(1)
    + 0.3535533905932738j * Y(0) @ X(1)
    + (0.3535533905932738+0j) * Y(0) @ Y(1)
    """

    qubit_operator = _binary_mapping_dispatch(bose_operator, nstates_boson, tol=tol)

    wires = list(bose_operator.wires) or [0]
    identity_wire = wires[0]
    if not ps:
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator


@singledispatch
def _binary_mapping_dispatch(bose_operator, nstates_boson):
    """Dispatches to appropriate function if bose_operator is a BoseWord or BoseSentence."""
    raise ValueError(f"bose_operator must be a BoseWord or BoseSentence, got: {bose_operator}")


@_binary_mapping_dispatch.register
def _(bose_operator: BoseWord, nstates_boson, tol=None):

    if nstates_boson < 2:
        raise ValueError(
            f"Number of bosonic states cannot be less than 2, provided {nstates_boson}."
        )
    nqub_per_boson = int(np.ceil(np.log2(nstates_boson)))

    cr_op = np.zeros((nstates_boson, nstates_boson))
    for s in range(nstates_boson - 1):
        cr_op[s + 1, s] = np.sqrt(s + 1.0)

    op_mat = {"+": cr_op, "-": cr_op.T}

    qubit_operator = PauliSentence({PauliWord({}): 1.0})

    for item in bose_operator.items():
        (_, b_idx), sign = item
        op = PauliSentence()
        sparse_opmat = np.nonzero(op_mat[sign])
        for i, j in zip(*sparse_opmat):
            coeff = op_mat[sign][i][j]

            binary_row = list(map(int, bin(i)[2:]))[::-1]
            if nqub_per_boson > len(binary_row):
                binary_row += [0] * (nqub_per_boson - len(binary_row))

            binary_col = list(map(int, bin(j)[2:]))[::-1]
            if nqub_per_boson > len(binary_col):
                binary_col += [0] * (nqub_per_boson - len(binary_col))

            pauliOp = PauliSentence({PauliWord({}): 1.0})
            for n in range(nqub_per_boson):
                pauliOp @= _get_pauli_op(binary_row[n], binary_col[n], n + b_idx * nqub_per_boson)

            op += coeff * pauliOp
        qubit_operator @= op

    for pw in qubit_operator:
        if tol is not None and abs(qml.math.imag(qubit_operator[pw])) <= tol:
            qubit_operator[pw] = qml.math.real(qubit_operator[pw])

    qubit_operator.simplify(tol=1e-16)

    return qubit_operator


@_binary_mapping_dispatch.register
def _(bose_operator: BoseSentence, nstates_boson, tol=None):

    qubit_operator = PauliSentence()

    for bw, coeff in bose_operator.items():
        bose_word_as_ps = binary_mapping(bw, nstates_boson=nstates_boson, ps=True)

        for pw in bose_word_as_ps:
            qubit_operator[pw] = qubit_operator[pw] + bose_word_as_ps[pw] * coeff

            if tol is not None and abs(qml.math.imag(qubit_operator[pw])) <= tol:
                qubit_operator[pw] = qml.math.real(qubit_operator[pw])

    qubit_operator.simplify(tol=1e-16)

    return qubit_operator