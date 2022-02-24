# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains the functions needed for operator and observable arithmetic.
"""
# pylint: disable= too-many-branches, too-many-arguments, too-many-locals, too-many-nested-blocks
import pennylane as qml
from pennylane import numpy as np


def simplify(h, cutoff=1.0e-12):
    r"""Add together identical terms in the Hamiltonian.

    The Hamiltonian terms with identical Pauli words are added together and eliminated if the
    overall coefficient is smaller than a cutoff value.

    Args:
        h (Hamiltonian): PennyLane Hamiltonian
        cutoff (float): cutoff value for discarding the negligible terms

    Returns:
        Hamiltonian: Simplified PennyLane Hamiltonian

    **Example**

    >>> c = np.array([0.5, 0.5])
    >>> h = qml.Hamiltonian(c, [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliY(1)])
    >>> print(simplify(h))
    (1.0) [X0 Y1]
    """
    wiremap = dict(zip(h.wires, range(len(h.wires) + 1)))

    c = []
    o = []
    for i, op in enumerate(h.terms()[1]):
        op = qml.operation.Tensor(op).prune()
        op = qml.grouping.pauli_word_to_string(op, wire_map=wiremap)
        if op not in o:
            c.append(h.terms()[0][i])
            o.append(op)
        else:
            c[o.index(op)] += h.terms()[0][i]

    coeffs = []
    ops = []
    nonzero_ind = np.argwhere(abs(np.array(c)) > cutoff).flatten()
    for i in nonzero_ind:
        coeffs.append(c[i])
        ops.append(qml.grouping.string_to_pauli_word(o[i], wire_map=wiremap))

    try:
        coeffs = qml.math.stack(coeffs)
    except ValueError:
        pass

    return qml.Hamiltonian(coeffs, ops)


def _pauli_mult(p1, p2):
    r"""Return the result of multiplication between two tensor products of Pauli operators.

    The Pauli operator :math:`(P_0)` is denoted by [(0, 'P')], where :math:`P` represents
    :math:`X`, :math:`Y` or :math:`Z`.

    Args:
        p1 (list[tuple[int, str]]): the first tensor product of Pauli operators
        p2 (list[tuple[int, str]]): the second tensor product of Pauli operators

    Returns
        tuple(list[tuple[int, str]], complex): list of the Pauli operators and the coefficient

    **Example**

    >>> p1 = [(0, "X"), (1, "Y")],  # X_0 @ Y_1
    >>> p2 = [(0, "X"), (2, "Y")],  # X_0 @ Y_2
    >>> _pauli_mult(p1, p2)
    ([(2, "Y"), (1, "Y")], 1.0) # p1 @ p2 = X_0 @ Y_1 @ X_0 @ Y_2
    """
    c = 1.0

    t1 = [t[0] for t in p1]
    t2 = [t[0] for t in p2]

    k = []

    for i in p1:
        if i[0] in t1 and i[0] not in t2:
            k.append((i[0], pauli_mult[i[1]]))
        for j in p2:
            if j[0] in t2 and j[0] not in t1:
                k.append((j[0], pauli_mult[j[1]]))

            if i[0] == j[0]:
                if i[1] + j[1] in pauli_coeff:
                    k.append((i[0], pauli_mult[i[1] + j[1]]))
                    c = c * pauli_coeff[i[1] + j[1]]
                else:
                    k.append((i[0], pauli_mult[i[1] + j[1]]))

    k = [i for i in k if "I" not in i[1]]

    for item in k:
        k_ = [i for i, x in enumerate(k) if x == item]
        if len(k_) >= 2:
            for j in k_[::-1][:-1]:
                del k[j]

    return k, c


def _return_pauli(p):
    r"""Return the PennyLane Pauli operator.

    Args:
        args (str): symbol representing the Pauli operator

    Returns:
        pennylane.ops: the PennyLane Pauli operator

    **Example**

    >>> _return_pauli('X')
    qml.PauliX
    """
    if p == "X":
        return qml.PauliX
    if p == "Y":
        return qml.PauliY

    return qml.PauliZ


pauli_mult = {
    "XX": "I",
    "YY": "I",
    "ZZ": "I",
    "ZX": "Y",
    "XZ": "Y",
    "ZY": "X",
    "YZ": "X",
    "XY": "Z",
    "YX": "Z",
    "IX": "X",
    "IY": "Y",
    "IZ": "Z",
    "XI": "X",
    "YI": "Y",
    "ZI": "Z",
    "I": "I",
    "II": "I",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
}

pauli_coeff = {
    "ZX": 1.0j,
    "XZ": -1.0j,
    "ZY": -1.0j,
    "YZ": 1.0j,
    "XY": 1.0j,
    "YX": -1.0j,
}
