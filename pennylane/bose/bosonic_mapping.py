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

from collections import defaultdict
from functools import singledispatch

import numpy as np

from pennylane import math
from pennylane.pauli import PauliSentence, PauliWord

from .bosonic import BoseSentence, BoseWord


def _get_pauli_op(i, j, qub_id):
    r"""Returns expression to convert qubit-local term ::math::``\ket{x_i}\bra{x_j}``
    to qubit operators as given in :math:`Eq. (6-9)` in `arXiv.1909.12847 <https://arxiv.org/abs/1909.12847>_`.
    """

    c1, c2 = 0.5, -0.5 if i == 1 else 0.5

    if i != j:
        return PauliSentence({PauliWord({qub_id: "X"}): c1, PauliWord({qub_id: "Y"}): c2 * 1j})

    return PauliSentence({PauliWord({}): c1, PauliWord({qub_id: "Z"}): c2})


def binary_mapping(
    bose_operator: BoseWord | BoseSentence,
    n_states: int = 2,
    ps: bool = False,
    wire_map: dict = None,
    tol: float = None,
):
    r"""Convert a bosonic operator to a qubit operator using the standard-binary mapping.

    The mapping procedure is described in equations :math:`27-29` in `arXiv:1507.03271 <https://arxiv.org/pdf/1507.03271>`_.

    Args:
        bose_operator (BoseWord, BoseSentence): the bosonic operator
        n_states (int): Maximum number of allowed bosonic states. Defaults to ``2``.
        ps (bool): Whether to return the result as a ``PauliSentence`` instead of an
            operator. Defaults to ``False``.
        wire_map (dict): A dictionary defining how to map the states of
            the Bose operator to qubit wires. If ``None``, integers used to
            label the bosonic states will be used as wire labels. Defaults to ``None``.
        tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
        Union[PauliSentence, Operator]: a linear combination of qubit operators

    **Example**

    >>> w = qml.BoseWord({(0, 0): "+"})
    >>> qml.binary_mapping(w, n_states=4)
    (
        0.6830127018922193 * X(0)
      + -0.1830127018922193 * (X(0) @ Z(1))
      + -0.6830127018922193j * Y(0)
      + 0.1830127018922193j * (Y(0) @ Z(1))
      + 0.3535533905932738 * (X(0) @ X(1))
      + -0.3535533905932738j * (X(0) @ Y(1))
      + 0.3535533905932738j * (Y(0) @ X(1))
      + (0.3535533905932738+0j) * (Y(0) @ Y(1))
    )
    """

    qubit_operator = _binary_mapping_dispatch(bose_operator, n_states, tol=tol)

    wires = list(bose_operator.wires) or [0]
    identity_wire = wires[0]
    if not ps:
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator


@singledispatch
def _binary_mapping_dispatch(bose_operator, n_states, tol):
    """Dispatches to appropriate function if bose_operator is a BoseWord or BoseSentence."""
    raise TypeError(f"bose_operator must be a BoseWord or BoseSentence, got: {bose_operator}")


@_binary_mapping_dispatch.register
def _(bose_operator: BoseWord, n_states, tol=None):

    if n_states < 2:
        raise ValueError(
            f"Number of allowed bosonic states cannot be less than 2, provided {n_states}."
        )
    nqub_per_boson = int(np.ceil(np.log2(n_states)))

    creation = np.zeros((n_states, n_states))
    for s in range(n_states - 1):
        creation[s + 1, s] = np.sqrt(s + 1.0)

    coeff_mat = {"+": creation, "-": creation.T}

    qubit_operator = PauliSentence({PauliWord({}): 1.0})

    for (_, b_idx), sign in bose_operator.items():
        op = PauliSentence()
        sparse_coeffmat = np.nonzero(coeff_mat[sign])
        for i, j in zip(*sparse_coeffmat):
            coeff = coeff_mat[sign][i][j]

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
        if tol is not None and abs(math.imag(qubit_operator[pw])) <= tol:
            qubit_operator[pw] = math.real(qubit_operator[pw])

    qubit_operator.simplify(tol=1e-16)

    return qubit_operator


@_binary_mapping_dispatch.register
def _(bose_operator: BoseSentence, n_states, tol=None):

    qubit_operator = PauliSentence()

    for bw, coeff in bose_operator.items():
        bose_word_as_ps = binary_mapping(bw, n_states=n_states, ps=True)

        for pw in bose_word_as_ps:
            qubit_operator[pw] = qubit_operator[pw] + bose_word_as_ps[pw] * coeff

            if tol is not None and abs(math.imag(qubit_operator[pw])) <= tol:
                qubit_operator[pw] = math.real(qubit_operator[pw])

    qubit_operator.simplify(tol=1e-16)

    return qubit_operator


def unary_mapping(
    bose_operator: BoseWord | BoseSentence,
    n_states: int = 2,
    ps: bool = False,
    wire_map: dict = None,
    tol: float = None,
):
    r"""Convert a bosonic operator to a qubit operator using the unary mapping.

    The mapping procedure is described in `arXiv.1909.12847 <https://arxiv.org/abs/1909.12847>`_.

    Args:
        bose_operator(BoseWord, BoseSentence): the bosonic operator
        n_states(int): Maximum number of allowed bosonic states. Defaults to ``2``.
        ps (bool): Whether to return the result as a ``PauliSentence`` instead of an
            operator. Defaults to ``False``.
        wire_map (dict): A dictionary defining how to map the states of
            the Bose operator to qubit wires. If ``None``, integers used to
            label the bosonic states will be used as wire labels. Defaults to ``None``.
        tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
        Union[PauliSentence, Operator]: a linear combination of qubit operators

    **Example**

    >>> w = qml.BoseWord({(0, 0): "+"})
    >>> qml.unary_mapping(w, n_states=4)
    (
        0.25 * (X(0) @ X(1))
      + -0.25j * (X(0) @ Y(1))
      + 0.25j * (Y(0) @ X(1))
      + (0.25+0j) * (Y(0) @ Y(1))
      + 0.3535533905932738 * (X(1) @ X(2))
      + -0.3535533905932738j * (X(1) @ Y(2))
      + 0.3535533905932738j * (Y(1) @ X(2))
      + (0.3535533905932738+0j) * (Y(1) @ Y(2))
      + 0.4330127018922193 * (X(2) @ X(3))
      + -0.4330127018922193j * (X(2) @ Y(3))
      + 0.4330127018922193j * (Y(2) @ X(3))
      + (0.4330127018922193+0j) * (Y(2) @ Y(3))
    )
    """

    qubit_operator = _unary_mapping_dispatch(bose_operator, n_states, tol=tol)

    wires = list(bose_operator.wires) or [0]
    identity_wire = wires[0]
    if not ps:
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator


@singledispatch
def _unary_mapping_dispatch(bose_operator, n_states, ps=False, wires_map=None, tol=None):
    """Dispatches to appropriate function if bose_operator is a BoseWord or BoseSentence."""
    raise TypeError(f"bose_operator must be a BoseWord or BoseSentence, got: {bose_operator}")


@_unary_mapping_dispatch.register
def _(bose_operator: BoseWord, n_states, tol=None):

    if n_states < 2:
        raise ValueError(
            f"Number of allowed bosonic states cannot be less than 2, provided {n_states}."
        )

    creation = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        creation[i + 1, i] = np.sqrt(i + 1.0)

    coeff_mat = {"+": creation, "-": creation.T}

    qubit_operator = PauliSentence({PauliWord({}): 1.0})

    ops_per_idx = defaultdict(list)

    # Avoiding superfluous terms by taking the product of
    # coefficient matrices.
    for (_, b_idx), sign in bose_operator.items():
        ops_per_idx[b_idx].append(sign)

    for b_idx, signs in ops_per_idx.items():
        coeff_mat_prod = np.eye(n_states)
        for sign in signs:
            coeff_mat_prod = np.dot(coeff_mat_prod, coeff_mat[sign])

        op = PauliSentence()
        sparse_coeffmat = np.nonzero(coeff_mat_prod)
        for i, j in zip(*sparse_coeffmat):
            coeff = coeff_mat_prod[i][j]

            row = np.zeros(n_states)
            row[i] = 1

            col = np.zeros(n_states)
            col[j] = 1

            pauliOp = PauliSentence({PauliWord({}): 1.0})
            for n in range(n_states):
                if row[n] == 1 or col[n] == 1:
                    pauliOp @= _get_pauli_op(row[n], col[n], n + b_idx * n_states)
            op += coeff * pauliOp
        qubit_operator @= op

    for pw in qubit_operator:
        if tol is not None and abs(math.imag(qubit_operator[pw])) <= tol:
            qubit_operator[pw] = math.real(qubit_operator[pw])
    qubit_operator.simplify(tol=1e-16)

    return qubit_operator


@_unary_mapping_dispatch.register
def _(bose_operator: BoseSentence, n_states, tol=None):

    qubit_operator = PauliSentence()

    for bw, coeff in bose_operator.items():
        bose_word_as_ps = unary_mapping(bw, n_states=n_states, ps=True)

        for pw in bose_word_as_ps:
            qubit_operator[pw] = qubit_operator[pw] + bose_word_as_ps[pw] * coeff

            if tol is not None and abs(math.imag(qubit_operator[pw])) <= tol:
                qubit_operator[pw] = math.real(qubit_operator[pw])

    qubit_operator.simplify(tol=1e-16)

    return qubit_operator


def christiansen_mapping(
    bose_operator: BoseWord | BoseSentence,
    ps: bool = False,
    wire_map: dict = None,
    tol: float = None,
):
    r"""Convert a bosonic operator to a qubit operator using the Christiansen mapping.

    This mapping assumes that the maximum number of allowed bosonic states is 2 and works only for
    Christiansen bosons defined in `J. Chem. Phys. 120, 2140 (2004)
    <https://pubs.aip.org/aip/jcp/article-abstract/120/5/2140/534128/A-second-quantization-formulation-of-multimode?redirectedFrom=fulltext>`_.
    The bosonic creation and annihilation operators are mapped to the Pauli operators as

    .. math::

        b^{\dagger}_0 =  \left (\frac{X_0 - iY_0}{2}  \right ), \:\: \text{...,} \:\:
        b^{\dagger}_n = \frac{X_n - iY_n}{2},

    and

    .. math::

        b_0 =  \left (\frac{X_0 + iY_0}{2}  \right ), \:\: \text{...,} \:\:
        b_n = \frac{X_n + iY_n}{2},

    where :math:`X`, :math:`Y`, and :math:`Z` are the Pauli operators.

    Args:
        bose_operator(BoseWord, BoseSentence): the bosonic operator
        ps (bool): Whether to return the result as a ``PauliSentence`` instead of an
            operator. Defaults to ``False``.
        wire_map (dict): A dictionary defining how to map the states of
            the Bose operator to qubit wires. If ``None``, integers used to
            label the bosonic states will be used as wire labels. Defaults to ``None``.
        tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
        Union[PauliSentence, Operator]: A linear combination of qubit operators.

    **Example**

    >>> w = qml.bose.BoseWord({(0,0):"+", (1,1): "-"})
    >>> qml.christiansen_mapping(w)
    (
        0.25 * (X(0) @ X(1))
      + 0.25j * (X(0) @ Y(1))
      + -0.25j * (Y(0) @ X(1))
      + (0.25+0j) * (Y(0) @ Y(1))
    )
    """

    qubit_operator = _christiansen_mapping_dispatch(bose_operator, tol)

    wires = list(bose_operator.wires) or [0]
    identity_wire = wires[0]
    if not ps:
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator


@singledispatch
def _christiansen_mapping_dispatch(bose_operator, tol):
    """Dispatches to appropriate function if bose_operator is a BoseWord or BoseSentence."""
    raise TypeError(f"bose_operator must be a BoseWord or BoseSentence, got: {bose_operator}")


@_christiansen_mapping_dispatch.register
def _(bose_operator: BoseWord, tol=None):

    qubit_operator = PauliSentence({PauliWord({}): 1.0})

    coeffs = {"+": -0.5j, "-": 0.5j}

    for (_, b_idx), sign in bose_operator.items():

        qubit_operator @= PauliSentence(
            {
                PauliWord({**{b_idx: "X"}}): 0.5,
                PauliWord({**{b_idx: "Y"}}): coeffs[sign],
            }
        )

    for pw in qubit_operator:
        if tol is not None and abs(math.imag(qubit_operator[pw])) <= tol:
            qubit_operator[pw] = math.real(qubit_operator[pw])

    qubit_operator.simplify(tol=1e-16)

    return qubit_operator


@_christiansen_mapping_dispatch.register
def _(bose_operator: BoseSentence, tol=None):

    qubit_operator = PauliSentence()

    for bw, coeff in bose_operator.items():
        bose_word_as_ps = christiansen_mapping(bw, ps=True)

        for pw in bose_word_as_ps:
            qubit_operator[pw] = qubit_operator[pw] + bose_word_as_ps[pw] * coeff

            if tol is not None and abs(math.imag(qubit_operator[pw])) <= tol:
                qubit_operator[pw] = math.real(qubit_operator[pw])

    qubit_operator.simplify(tol=1e-16)

    return qubit_operator
