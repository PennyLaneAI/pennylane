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
# pylint: disable= too-many-branches,
from functools import reduce

import autograd.numpy as anp

import pennylane as qml
from pennylane import numpy as np


def fermionic_observable(constant, one=None, two=None, cutoff=1.0e-12):
    r"""Create a fermionic observable from molecular orbital integrals.

    Args:
        constant (array[float]): the contribution of the core orbitals and nuclei
        one (array[float]): the one-particle molecular orbital integrals
        two (array[float]): the two-particle molecular orbital integrals
        cutoff (float): cutoff value for discarding the negligible integrals

    Returns:
        tuple(array[float], list[int]): fermionic coefficients and operators

    **Example**

    >>> constant = np.array([1.0])
    >>> integral = np.array([[0.5, -0.8270995], [-0.8270995, 0.5]])
    >>> coeffs, ops = fermionic_observable(constant, integral)
    >>> ops
    [[], [0, 0], [0, 2], [1, 1], [1, 3], [2, 0], [2, 2], [3, 1], [3, 3]]
    """
    coeffs = anp.array([])

    if constant != anp.array([0.0]):
        coeffs = anp.concatenate((coeffs, constant))
        operators = [[]]
    else:
        operators = []

    if one is not None:
        indices_one = anp.argwhere(abs(one) >= cutoff)
        # up-up + down-down terms
        operators_one = (indices_one * 2).tolist() + (indices_one * 2 + 1).tolist()
        coeffs_one = anp.tile(one[abs(one) >= cutoff], 2)
        coeffs = anp.concatenate((coeffs, coeffs_one))
        operators = operators + operators_one

    if two is not None:
        indices_two = anp.argwhere(abs(two) >= cutoff)
        n = len(indices_two)
        operators_two = (
            [(indices_two[i] * 2).tolist() for i in range(n)]  # up-up-up-up
            + [(indices_two[i] * 2 + [0, 1, 1, 0]).tolist() for i in range(n)]  # up-down-down-up
            + [(indices_two[i] * 2 + [1, 0, 0, 1]).tolist() for i in range(n)]  # down-up-up-down
            + [(indices_two[i] * 2 + 1).tolist() for i in range(n)]  # down-down-down-down
        )
        coeffs_two = anp.tile(two[abs(two) >= cutoff], 4) / 2

        coeffs = anp.concatenate((coeffs, coeffs_two))
        operators = operators + operators_two

    indices_sort = [operators.index(i) for i in sorted(operators)]

    return coeffs[indices_sort], sorted(operators)


def qubit_observable(o_ferm, cutoff=1.0e-12):
    r"""Convert a fermionic observable to a PennyLane qubit observable.

    The fermionic operator is a tuple containing the fermionic coefficients and operators. For
    instance, the one-body fermionic operator :math:`a_2^\dagger a_0` is specified as [2, 0] and the
    two-body operator :math:`a_4^\dagger a_3^\dagger a_2 a_1` is specified as [4, 3, 2, 1].

    Args:
        o_ferm tuple(array[float], list[int]): fermionic operator
        cutoff (float): cutoff value for discarding the negligible terms

    Returns:
        Hamiltonian: Simplified PennyLane Hamiltonian

    **Example**

    >>> coeffs = np.array([1.0, 1.0])
    >>> ops = [[0, 0], [0, 0]]
    >>> f = (coeffs, ops)
    >>> print(qubit_observable(f))
    ((-1+0j)) [Z0]
    + ((1+0j)) [I0]
    """
    ops = []
    coeffs = anp.array([])

    for n, t in enumerate(o_ferm[1]):
        if len(t) == 0:
            ops = ops + [qml.Identity(0)]
            coeffs = anp.array([0.0])
            coeffs = coeffs + o_ferm[0][n]
        else:
            op = jordan_wigner(t)
            if op != 0:
                ops = ops + op[1]
                coeffs = anp.concatenate([coeffs, anp.array(op[0]) * o_ferm[0][n]])

    o_qubit = simplify(qml.Hamiltonian(coeffs, ops), cutoff=cutoff)

    return o_qubit


def jordan_wigner(op):
    r"""Convert a fermionic operator to a qubit operator using the Jordan-Wigner mapping.

    For instance, the one-body fermionic operator :math:`a_2^\dagger a_0` should be constructed as
    [2, 0] and the two-body operator :math:`a_4^\dagger a_3^\dagger a_2 a_1` should be constructed
    as [4, 3, 2, 1].

    Args:
        op (list[int]): the fermionic operator

    Returns
        tuple(list[complex], list[list[int, str]]): list of coefficients and the qubit-operator terms

    **Example**

    >>> f  = [0, 0]
    >>> q = jordan_wigner(f)
    >>> q
    ([(0.5+0j), (-0.5+0j)], [Identity(wires=[0]), PauliZ(wires=[0])]) # corresponds to :math:`\frac{1}{2}(I_0 - Z_0)`
    """
    if len(op) == 1:
        op = [((op[0], 1),)]

    if len(op) == 2:
        op = [((op[0], 1), (op[1], 0))]

    if len(op) == 4:
        op = [((op[0], 1), (op[1], 1), (op[2], 0), (op[3], 0))]

        if op[0][0][0] == op[0][1][0] or op[0][2][0] == op[0][3][0]:
            return 0

    for t in op:
        for l in t:
            z = [(index, "Z") for index in range(l[0])]
            x = z + [(l[0], "X"), 0.5]

            if l[1]:
                y = z + [(l[0], "Y"), -0.5j]

            else:
                y = z + [(l[0], "Y"), 0.5j]

            if t.index(l) == 0:
                q = [x, y]
            else:
                m = []
                for t1 in q:
                    for t2 in [x, y]:
                        q1, c1 = _pauli_mult(t1[:-1], t2[:-1])
                        m.append(q1 + [c1 * t1[-1] * t2[-1]])
                q = m

    c = [p[-1] for p in q]
    o = [p[:-1] for p in q]

    for item in o:
        k = [i for i, x in enumerate(o) if x == item]
        if len(k) >= 2:
            for j in k[::-1][:-1]:
                del o[j]
                c[k[0]] = c[k[0]] + c[j]
                del c[j]

    pauli_map = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    for i, term in enumerate(o):
        if len(term) == 0:
            o[i] = qml.Identity(0)
        if len(term) == 1:
            o[i] = pauli_map[term[0][1]](term[0][0])
        if len(term) > 1:
            k = [pauli_map[t[1]](t[0]) for t in term]
            o[i] = reduce(lambda x, y: x @ y, k)

    return c, o


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

    c, o = [], []
    for i, op in enumerate(h.ops):
        op = qml.operation.Tensor(op).prune()
        op = qml.grouping.pauli_word_to_string(op, wire_map=wiremap)
        if op not in o:
            c.append(h.coeffs[i])
            o.append(op)
        else:
            c[o.index(op)] += h.coeffs[i]

    coeffs, ops = [], []
    nonzero_ind = np.argwhere(abs(np.array(c)) > cutoff).flatten()
    for i in nonzero_ind:
        coeffs.append(c[i])
        ops.append(qml.grouping.string_to_pauli_word(o[i], wire_map=wiremap))

    try:
        coeffs = qml.math.stack(coeffs)
    except ValueError:
        pass

    return qml.Hamiltonian(np.array(coeffs), ops)


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

    >>> p1 = [(0, "X"), (1, "Y")]  # X_0 @ Y_1
    >>> p2 = [(0, "X"), (2, "Y")]  # X_0 @ Y_2
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
