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

    c1, c2 = 0.5, 0.5
    if i == 1:
        c2 = -c2

    if i != j:
        return PauliSentence({PauliWord({qub_id: "X"}): c1, PauliWord({qub_id: "Y"}):c2*1j})

    return PauliSentence({PauliWord({}): c1, PauliWord({qub_id: "Z"}):c2})

def binary_mapping(
    bose_operator: Union[BoseWord, BoseSentence],
    d: int = 2,
    ps: bool = True,
    wire_map: dict = None,
    tol: float = None,
):
    r"""Convert a bosonic operator to a qubit operator using the standard-binary mapping
    as described in <https://www.nature.com/articles/s41534-020-0278-0>
    Args:
      bose_operator(BoseWord, BoseSentence): the bosonic operator
      d(int): Number of states in the boson.
      ps (bool): whether to return the result as a PauliSentence instead of an operator. Defaults to False.
      wire_map (dict): a dictionary defining how to map the orbitals of
      the Bose operator to qubit wires. If None, the integers used to
      order the orbitals will be used as wire labels. Defaults to None.
      tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
      a linear combination of qubit operators

    **Example**

    >>> w = qml.vibrational_ham.BoseWord()
    >>> binary_mapping(w, d=2)

    """

    return _binary_mapping_dispatch(bose_operator, d, ps, wire_map, tol)


@singledispatch
def _binary_mapping_dispatch(bose_operator, d, ps=False, wires_map=None, tol=None):
    """Dispatches to appropriate function if bose_operator is a BoseWord or BoseSentence."""
    raise ValueError(f"bose_operator must be a BoseWord or BoseSentence, got: {bose_operator}")


@_binary_mapping_dispatch.register
def _(bose_operator: BoseWord, d, ps=False, wire_map=None, tol=None):
    nqub_per_boson = int(np.ceil(np.log2(d)))

    cr = np.zeros((d, d))
    for s in range(d - 1):
        cr[s + 1, s] = np.sqrt(s + 1.0)

    d_mat = {"+": cr, "-": cr.T}

    qubit_operator = PauliSentence({PauliWord({}): 1.0})

    for item in bose_operator.items():
        (_, boson), sign = item
        oper = PauliSentence()
        non_zero_d = np.nonzero(d_mat[sign])
        for i, j in zip(*non_zero_d):
            coeff = d_mat[sign][i][j]

            binary_row = list(map(int, bin(i)[2:]))[::-1]
            if nqub_per_boson > len(binary_row):
                binary_row += [0] * (nqub_per_boson - len(binary_row))

            binary_col = list(map(int, bin(j)[2:]))[::-1]
            if nqub_per_boson > len(binary_col):
                binary_col += [0] * (nqub_per_boson - len(binary_col))

            pauliOp = PauliSentence({PauliWord({}): 1.0})
            for n in range(nqub_per_boson):
                pauliOp @= _get_pauli_op(
                    binary_row[n], binary_col[n], n + boson * nqub_per_boson
                )

            oper += coeff * pauliOp
        qubit_operator @= oper
    qubit_operator.simplify()

    for pw in qubit_operator:
        if tol is not None and abs(qml.math.imag(qubit_operator[pw])) <= tol:
            qubit_operator[pw] = qml.math.real(qubit_operator[pw])

    wires = list(bose_operator.wires) or [0]
    identity_wire = wires[0]
    if not ps:
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator


@_binary_mapping_dispatch.register
def _(bose_operator: BoseSentence, d, ps=False, wire_map=None, tol=None):

    qubit_operator = PauliSentence()

    for bw, coeff in bose_operator.items():
        bose_word_as_ps = binary_mapping(bw, d=d, ps=True)

        for pw in bose_word_as_ps:
            qubit_operator[pw] = qubit_operator[pw] + bose_word_as_ps[pw] * coeff

            if tol is not None and abs(qml.math.imag(qubit_operator[pw])) <= tol:
                qubit_operator[pw] = qml.math.real(qubit_operator[pw])

    qubit_operator.simplify(tol=1e-16)

    wires = list(bose_operator.wires) or [0]
    identity_wire = wires[0]
    if not ps:
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator
