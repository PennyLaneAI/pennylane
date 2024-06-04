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
This module contains the functions needed to convert OpenFermion ``QubitOperator`` objects to PennyLane ``Sum`` and ``LinearCombination`` and vice versa.
"""

from functools import singledispatch
from typing import Union

import pennylane as qml
from pennylane import numpy as np
from pennylane.qchem.convert import (
    _openfermion_to_pennylane,
    _pennylane_to_openfermion,
    _process_wires,
)

from pennylane.fermi.fermionic import FermiWord, FermiSentence
from pennylane.ops import Sum, LinearCombination
from pennylane.wires import Wires

import openfermion


def from_openfermion(of_qubit_operator, wires=None, tol=None, return_sum=False):
    r"""Convert OpenFermion ``QubitOperator`` to a :class:`~.LinearCombination` object in PennyLane representing a linear combination of qubit operators.

    Args:
        qubit_operator (QubitOperator): fermionic-to-qubit transformed operator in terms of
            Pauli matrices
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert the qubit operator
            to an observable terms measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).
        tol (float): tolerance value to decide whether the imaginary part of the coefficients is retained
        return_sum (bool): flag indicating whether a ``Sum`` object is returned

    Returns:
        pennylane.ops.LinearCombination: a linear combination of Pauli matrices

    **Example**

    >>> q_op = QubitOperator('X0', 1.2) + QubitOperator('Z1', 2.4)
    >>> q_op
    1.2 [X0] +
    2.4 [Z1]
    >>> from_openfermion(q_op)
    1.2 * X(0) + 2.4 * Z(1)
    """

    coeffs, ops = _openfermion_to_pennylane(of_qubit_operator, wires=wires, tol=tol)
    pl_term = qml.ops.LinearCombination(coeffs, ops)

    if return_sum:
        pl_term.__class__ = qml.ops.Sum

    return pl_term


def to_openfermion(
    pl_operator: Union[Sum, LinearCombination, FermiWord, FermiSentence], wires=None, tol=None
) -> Union[openfermion.QubitOperator, openfermion.ops.FermionOperator]:
    r"""Convert ``LinearCombination`` object in PennyLane representing a linear combination of qubit operators to a OpenFermion ``QubitOperator``.

    Args:
        pl_linear_combination (pennylane.ops.LinearCombination): linear combination of operators
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert the qubit operator
            to an observable terms measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).

    Returns:
        QubitOperator: a qubit operator in terms of Pauli matrices

    **Example**

    >>> pl_term = 1.2 * qml.X(0) + 2.4 * qml.Z(1)
    >>> pl_term
    1.2 * X(0) + 2.4 * Z(1)
    >>> q_op = to_openfermion(q_op)
    >>> q_op
    1.2 [X0] +
    2.4 [Z1]
    """

    return _to_openfermion_dispatch(pl_operator, wires=wires, tol=tol)


@singledispatch
def _to_openfermion_dispatch(pl_operator, wires=None, tol=None):
    """Dispatches to appropriate function if pl_operator is a ``Sum``, ``LinearCombination, ``FermiWord`` or ``FermiSentence``."""
    raise ValueError(
        f"pl_operator must be a Sum, LinearCombination, FermiWord or FermiSentence, got: {pl_operator}."
    )


@_to_openfermion_dispatch.register
def _(pl_operator: Sum, wires=None, tol=None):
    coeffs, ops = pl_operator.terms()
    return _pennylane_to_openfermion(np.array(coeffs), ops, wires=wires, tol=tol)


@_to_openfermion_dispatch.register
def _(pl_operator: FermiWord, wires=None, tol=None):
    if wires:
        # Check wheter wires cover all wires defined in `pl_operator`.
        all_wires = Wires.all_wires(pl_operator.wires, sort=True)
        mapped_wires = _process_wires(wires)
        if not set(all_wires).issubset(set(mapped_wires)):
            raise ValueError("Supplied `wires` does not cover all wires defined in `pl_operator`.")

        # Map the FermiWord based on the ordering provided in `wires`.
        pl_operator_mapped = {}
        for (loc, orbital) in pl_operator.keys():
            pl_operator_mapped[(loc, mapped_wires.index(orbital))] = pl_operator[(loc, orbital)]
        # Replace the FermiWord by the mapped FermiWord.
        pl_operator = FermiWord(pl_operator_mapped)

    return openfermion.ops.FermionOperator(pl_operator.to_openfermion_string())


@_to_openfermion_dispatch.register
def _(pl_operator: FermiSentence, wires=None, tol=None):
    fermion_op = openfermion.ops.FermionOperator()
    # Convert each FermiWord to a FermionOperator in OpenFermion.
    # The coverage of the wire mapping is checked in the conversion of each FermiWord.
    for fermi_word in pl_operator:
        if tol:
            if np.abs(pl_operator[fermi_word]) > tol:
                fermion_operator += to_openfermion(fermi_word, wires=wires)
        else:
            fermion_operator += to_openfermion(fermi_word, wires=wires)

    return fermion_op
