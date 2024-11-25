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

import pennylane as qml
from pennylane.pauli import PauliSentence, PauliWord

from .bosonic import BoseSentence, BoseWord


def _test_double_occupancy(bose_operator):
    r"""Tests and raises an error if the BoseSentence or BoseWord contains terms with double occupancy."""
    ordered_op = bose_operator.normal_order()
    for bw in ordered_op:
        bw_terms = list(bw.keys())
        for i in range(len(bw) - 1):
            if bw_terms[i][1] == bw_terms[i + 1][1] and bw[bw_terms[i]] == bw[bw_terms[i + 1]]:
                raise ValueError(
                    "The provided bose_operator contains terms that require more than 2 states to "
                    "represent a bosonic mode, consider using binary_mapping or unary_mapping for this operator."
                )


def christiansen_mapping(
    bose_operator: Union[BoseWord, BoseSentence],
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
        ps (bool): Whether to return the result as a PauliSentence instead of an
            operator. Defaults to False.
        wire_map (dict): A dictionary defining how to map the states of
            the Bose operator to qubit wires. If None, integers used to
            label the bosonic states will be used as wire labels. Defaults to None.
        tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
        Union[PauliSentence, Operator]: A linear combination of qubit operators.
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
    raise ValueError(f"bose_operator must be a BoseWord or BoseSentence, got: {bose_operator}")


@_christiansen_mapping_dispatch.register
def _(bose_operator: BoseWord, tol=None):

    _test_double_occupancy(bose_operator)

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
        if tol is not None and abs(qml.math.imag(qubit_operator[pw])) <= tol:
            qubit_operator[pw] = qml.math.real(qubit_operator[pw])

    qubit_operator.simplify(tol=1e-16)

    return qubit_operator


@_christiansen_mapping_dispatch.register
def _(bose_operator: BoseSentence, tol=None):

    _test_double_occupancy(bose_operator)

    qubit_operator = PauliSentence()

    for bw, coeff in bose_operator.items():
        bose_word_as_ps = christiansen_mapping(bw, ps=True)

        for pw in bose_word_as_ps:
            qubit_operator[pw] = qubit_operator[pw] + bose_word_as_ps[pw] * coeff

            if tol is not None and abs(qml.math.imag(qubit_operator[pw])) <= tol:
                qubit_operator[pw] = qml.math.real(qubit_operator[pw])

    qubit_operator.simplify(tol=1e-16)

    return qubit_operator
