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
This module contains the functions needed for creating fermionic and qubit observables.
"""

import numpy as np

import pennylane as qml
from pennylane.fermi import FermiSentence, FermiWord
from pennylane.pauli import PauliSentence


def fermionic_observable(constant, one=None, two=None, cutoff=1.0e-12):
    r"""Create a fermionic observable from molecular orbital integrals.

    Args:
        constant (array[float]): the contribution of the core orbitals and nuclei
        one (array[float]): the one-particle molecular orbital integrals
        two (array[float]): the two-particle molecular orbital integrals
        cutoff (float): cutoff value for discarding the negligible integrals

    Returns:
        ~.FermiSentence: fermionic observable

    **Example**

    >>> constant = np.array([1.0])
    >>> integral = np.array([[0.5, -0.8270995], [-0.8270995, 0.5]])
    >>> fermionic_observable(constant, integral)
    1.0 * I
    + 0.5 * a⁺(0) a(0)
    + -0.8270995 * a⁺(0) a(2)
    + 0.5 * a⁺(1) a(1)
    + -0.8270995 * a⁺(1) a(3)
    + -0.8270995 * a⁺(2) a(0)
    + 0.5 * a⁺(2) a(2)
    + -0.8270995 * a⁺(3) a(1)
    + 0.5 * a⁺(3) a(3)
    """
    coeffs = qml.math.array([])

    if not qml.math.allclose(constant, 0.0):
        coeffs = qml.math.concatenate((coeffs, constant))
        operators = [[]]
    else:
        operators = []

    if one is not None:
        indices_one = qml.math.argwhere(abs(one) >= cutoff)
        # up-up + down-down terms
        operators_one = (indices_one * 2).tolist() + (indices_one * 2 + 1).tolist()
        coeffs_one = qml.math.tile(one[abs(one) >= cutoff], 2)
        coeffs = qml.math.convert_like(coeffs, one)
        coeffs = qml.math.concatenate((coeffs, coeffs_one))
        operators = operators + operators_one

    if two is not None:
        indices_two = np.array(qml.math.argwhere(abs(two) >= cutoff))
        n = len(indices_two)
        operators_two = (
            [(indices_two[i] * 2).tolist() for i in range(n)]  # up-up-up-up
            + [(indices_two[i] * 2 + [0, 1, 1, 0]).tolist() for i in range(n)]  # up-down-down-up
            + [(indices_two[i] * 2 + [1, 0, 0, 1]).tolist() for i in range(n)]  # down-up-up-down
            + [(indices_two[i] * 2 + 1).tolist() for i in range(n)]  # down-down-down-down
        )
        coeffs_two = qml.math.tile(two[abs(two) >= cutoff], 4) / 2

        coeffs = qml.math.concatenate((coeffs, coeffs_two))
        operators = operators + operators_two

    sentence = FermiSentence({FermiWord({}): constant[0]})
    for c, o in sorted(zip(coeffs, operators), key=lambda item: item[1]):

        if len(o) == 2:
            sentence.update({FermiWord({(0, o[0]): "+", (1, o[1]): "-"}): c})
        if len(o) == 4:
            sentence.update(
                {FermiWord({(0, o[0]): "+", (1, o[1]): "+", (2, o[2]): "-", (3, o[3]): "-"}): c}
            )
    sentence.simplify()

    return sentence


def qubit_observable(o_ferm, cutoff=1.0e-12, mapping="jordan_wigner"):
    r"""Convert a fermionic observable to a PennyLane qubit observable.

    Args:
        o_ferm (Union[~.FermiWord, ~.FermiSentence]): fermionic operator
        cutoff (float): cutoff value for discarding the negligible terms
        mapping (str): Specifies the fermion-to-qubit mapping. Input values can
            be ``'jordan_wigner'``, ``'parity'`` or ``'bravyi_kitaev'``.
    Returns:
        Operator: Simplified PennyLane Hamiltonian

    **Example**

    >>> w1 = qml.FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w2 = qml.FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> s = qml.FermiSentence({w1 : 1.2, w2: 3.1})
    >>> print(qubit_observable(s))
    -0.775j * (Y(0) @ X(1)) + 0.775 * (Y(0) @ Y(1)) + 0.775 * (X(0) @ X(1)) + 0.775j * (X(0) @ Y(1))
    """
    mapping = mapping.strip().lower()
    if mapping == "jordan_wigner":
        h = qml.jordan_wigner(o_ferm, ps=True, tol=cutoff)
    elif mapping == "parity":
        qubits = len(o_ferm.wires)
        h = qml.parity_transform(o_ferm, qubits, ps=True, tol=cutoff)
    elif mapping == "bravyi_kitaev":
        qubits = len(o_ferm.wires)
        h = qml.bravyi_kitaev(o_ferm, qubits, ps=True, tol=cutoff)
    else:
        raise ValueError(
            f"The '{mapping}' transformation is not available."
            f"Please set mapping to 'jordan_wigner', 'parity', or 'bravyi_kitaev'"
        )

    if list(h.wires) != sorted(list(h.wires)):
        h = PauliSentence(
            sorted(h.items(), key=lambda item: max(item[0].wires.tolist(), default=0))
        )

    h.simplify(tol=cutoff)

    if not h.wires:
        return h.operation(wire_order=[0])
    return h.operation()
