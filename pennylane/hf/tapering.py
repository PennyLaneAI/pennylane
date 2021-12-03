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
This module contains the functions needed for tapering qubits.
"""
# pylint: disable=unnecessary-lambda
import functools
import itertools

import autograd.numpy as anp
import pennylane as qml
from pennylane import numpy as np
from pennylane.grouping import pauli_mult_with_phase, pauli_word_to_string, string_to_pauli_word


def observable_mult(obs_a, obs_b):
    r"""Multiply two PennyLane observables together.

    Each observable is a linear combination of Pauli words, e.g., :math:`\sum_{k=0}^{N} c_k O_k`,
    and is represented by a PennyLane Hamiltonian.

    Args:
        obs_a (Hamiltonian): first observable
        obs_b (Hamiltonian): second observable

    Returns:
        .Hamiltonian: Observable expressed as a PennyLane Hamiltonian

    **Example**

    >>> c = np.array([0.5, 0.5])
    >>> obs_a = qml.Hamiltonian(c, [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliZ(1)])
    >>> obs_b = qml.Hamiltonian(c, [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0) @ qml.PauliZ(1)])
    >>> print(observable_mult(obs_a, obs_b))
      (-0.25j) [Z1]
    + (-0.25j) [Y0]
    + ( 0.25j) [Y1]
    + ((0.25+0j)) [Y0 X1]
    """
    o = []
    c = []
    for i in range(len(obs_a.terms[0])):
        for j in range(len(obs_b.terms[0])):
            op, phase = pauli_mult_with_phase(obs_a.terms[1][i], obs_b.terms[1][j])
            o.append(op)
            c.append(phase * obs_a.terms[0][i] * obs_b.terms[0][j])

    return simplify(qml.Hamiltonian(qml.math.stack(c), o))


def simplify(h):
    r"""Add together identical terms in the Hamiltonian.

    The Hamiltonian terms with identical Pauli words are added together and eliminated if the
    overall coefficient is zero.

    Args:
        h (Hamiltonian): PennyLane Hamiltonian

    Returns:
        .Hamiltonian: Simplified PennyLane Hamiltonian

    **Example**

    >>> c = np.array([0.5, 0.5])
    >>> h = qml.Hamiltonian(c, [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliY(1)])
    >>> print(simplify(h))
    (1.0) [X0 Y1]
    """
    s = []
    wiremap = dict(zip(h.wires, h.wires))
    for term in h.terms[1]:
        term = qml.operation.Tensor(term).prune()
        s.append(pauli_word_to_string(term, wire_map=wiremap))

    o = list(set(s))
    c = [0.0] * len(o)
    for i, item in enumerate(s):
        c[o.index(item)] += h.terms[0][i]

    nonzero_ind = np.nonzero(c)[0]
    c = [c[i] for i in nonzero_ind]
    o = [o[i] for i in nonzero_ind]
    o = [string_to_pauli_word(i) for i in o]

    return qml.Hamiltonian(qml.math.stack(c), o)


def clifford(generator, paulix_wires):
    r"""Compute a Clifford operator from a set of generators and Pauli operators.

    This function computes :math:`U = U_0U_1...U_k` for a set of :math:`k` generators and
    :math:`k` PauliX operators.

    Args:
        generator (list[Hamiltonian]): generators expressed as PennyLane Hamiltonians
        paulix_wires (list[int]): indices of the wires the PauliX operator acts on

    Returns:
        .Hamiltonian: Clifford operator expressed as a PennyLane Hamiltonian

    **Example**

    >>> t1 = qml.Hamiltonian([1.0], [string_to_pauli_word('ZZII')])
    >>> t2 = qml.Hamiltonian([1.0], [string_to_pauli_word('ZIZI')])
    >>> t3 = qml.Hamiltonian([1.0], [string_to_pauli_word('ZIIZ')])
    >>> generator = [t1, t2, t3]
    >>> paulix_wires = [1, 2, 3]
    >>> u = clifford(generator, paulix_wires)
    >>> print(u)
      (0.3535533905932737) [Z1 Z2 X3]
    + (0.3535533905932737) [X1 X2 X3]
    + (0.3535533905932737) [Z1 X2 Z3]
    + (0.3535533905932737) [X1 Z2 Z3]
    + (0.3535533905932737) [Z0 X1 X2 Z3]
    + (0.3535533905932737) [Z0 Z1 Z2 Z3]
    + (0.3535533905932737) [Z0 X1 Z2 X3]
    + (0.3535533905932737) [Z0 Z1 X2 X3]
    """
    cliff = []
    for i, t in enumerate(generator):
        cliff.append(1 / 2 ** 0.5 * (qml.PauliX(paulix_wires[i]) + t))

    u = functools.reduce(lambda i, j: observable_mult(i, j), cliff)

    return u


def transform_hamiltonian(h, generator, paulix_wires, paulix_sector=None):
    r"""Transform a Hamiltonian with a Clifford operator and taper qubits.

    The Hamiltonian is transformed as :math:`H' = U^{\dagger} H U` where :math:`U` is a Clifford
    operator. The transformed Hamiltonian acts trivially on some qubits which are then replaced
    with the eigenvalues of their corresponding Pauli/Identity operator.

    Args:
        h (Hamiltonian): PennyLane Hamiltonian
        generator (list[Hamiltonian]): generators expressed as PennyLane Hamiltonians
        paulix_wires (list[int]): indices of the wires the PauliX operator acts on
        paulix_sector list([list[int]]): list of eigenvalues of the PauliX operators

    Returns:
        (list[tuple[list[int], qml.Hamiltonian]]): paulix sector and its corresponding Hamiltonian

    **Example**

    >>> symbols = ["H", "H"]
    >>> geometry = np.array([[0.0, 0.0, -0.69440367], [0.0, 0.0, 0.69440367]], requires_grad=False)
    >>> mol = qml.hf.Molecule(symbols, geometry)
    >>> H = qml.hf.generate_hamiltonian(mol)()
    >>> t1 = qml.Hamiltonian([1.0], [string_to_pauli_word('ZZII')])
    >>> t2 = qml.Hamiltonian([1.0], [string_to_pauli_word('ZIZI')])
    >>> t3 = qml.Hamiltonian([1.0], [string_to_pauli_word('ZIIZ')])
    >>> generator = [t1, t2, t3]
    >>> paulix_wires = [1, 2, 3]
    >>> paulix_sector = [[1, -1, -1]]
    >>> transform_hamiltonian(H, generator, paulix_wires, [paulix_sector])
    [([1, -1, -1], <Hamiltonian: terms=4, wires=[0]>)]
    """
    u = clifford(generator, paulix_wires)
    h = observable_mult(observable_mult(u, h), u)

    if paulix_sector is None:
        paulix_sector = itertools.product([1, -1], repeat=len(paulix_wires))

    h_tapered = []
    wiremap = dict(zip(h.wires, h.wires))

    for sector in paulix_sector:
        val = np.ones(len(h.terms[0])) * complex(1.0)

        for idx, w in enumerate(paulix_wires):
            for i in range(len(h.terms[0])):
                s = pauli_word_to_string(h.terms[1][i], wire_map=wiremap)
                if s[w] == "X":
                    val[i] *= sector[idx]

        o = []
        for i in range(len(h.terms[0])):
            s = pauli_word_to_string(h.terms[1][i], wire_map=wiremap)
            wires = [x for x in h.wires if x not in paulix_wires]
            o.append(string_to_pauli_word("".join([s[i] for i in wires])))

        c = anp.multiply(val, h.terms[0])
        c = qml.math.stack(c)
        h_tapered.append((sector, simplify(qml.Hamiltonian(c, o))))

    return h_tapered
