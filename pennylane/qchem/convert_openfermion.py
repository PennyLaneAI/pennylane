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
import numpy as np

import pennylane as qml
from pennylane.qchem.convert import _openfermion_to_pennylane, _pennylane_to_openfermion


def from_openfermion(of_qubit_operator, wires=None, tol=None):
    r"""Convert OpenFermion ``QubitOperator`` to a ``LinearCombination`` object in PennyLane representing a linear combination of qubit operators.

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
    return qml.ops.LinearCombination(coeffs, ops)


def to_openfermion(pl_linear_combination, wires=None):
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
    1.2 [X0] +
    2.4 [Z1]
    """
    
    coeffs, ops = pl_linear_combination.terms()
    return _pennylane_to_openfermion(coeffs, ops, wires=wires)

