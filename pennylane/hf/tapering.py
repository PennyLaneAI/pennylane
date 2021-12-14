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

import autograd.numpy as anp
import pennylane as qml
from pennylane import numpy as np


def _observable_mult(obs_a, obs_b):
    r"""Multiply two PennyLane observables together.

    Each observable should be a linear combination of Pauli words, e.g.,
    :math:`\sum_{k=0}^{N} c_k P_k`, and represented as a PennyLane Hamiltonian.

    Args:
        obs_a (Hamiltonian): first observable
        obs_b (Hamiltonian): second observable

    Returns:
        qml.Hamiltonian: Observable expressed as a PennyLane Hamiltonian

    **Example**

    >>> c = np.array([0.5, 0.5])
    >>> obs_a = qml.Hamiltonian(c, [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliZ(1)])
    >>> obs_b = qml.Hamiltonian(c, [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0) @ qml.PauliZ(1)])
    >>> print(_observable_mult(obs_a, obs_b))
      (-0.25j) [Z1]
    + (-0.25j) [Y0]
    + ( 0.25j) [Y1]
    + ((0.25+0j)) [Y0 X1]
    """
    o = []
    c = []
    for i in range(len(obs_a.terms[0])):
        for j in range(len(obs_b.terms[0])):
            op, phase = qml.grouping.pauli_mult_with_phase(obs_a.terms[1][i], obs_b.terms[1][j])
            o.append(op)
            c.append(phase * obs_a.terms[0][i] * obs_b.terms[0][j])

    return _simplify(qml.Hamiltonian(qml.math.stack(c), o))


def _simplify(h, cutoff=1.0e-12):
    r"""Add together identical terms in the Hamiltonian.

    The Hamiltonian terms with identical Pauli words are added together and eliminated if the
    overall coefficient is zero.

    Args:
        h (Hamiltonian): PennyLane Hamiltonian
        cutoff (float): cutoff value for discarding the negligible terms

    Returns:
        qml.Hamiltonian: Simplified PennyLane Hamiltonian

    **Example**

    >>> c = np.array([0.5, 0.5])
    >>> h = qml.Hamiltonian(c, [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliY(1)])
    >>> print(_simplify(h))
    (1.0) [X0 Y1]
    """
    s = []
    w = list(range(max(h.wires.tolist()) + 1))
    wiremap = dict(zip(w, w))
    for term in h.terms[1]:
        term = qml.operation.Tensor(term).prune()
        s.append(qml.grouping.pauli_word_to_string(term, wire_map=wiremap))

    o = list(set(s))
    c = [0.0] * len(o)
    for i, item in enumerate(s):
        c[o.index(item)] += h.terms[0][i]
    c = qml.math.stack(c)

    coeffs = []
    ops = []
    nonzero_ind = np.argwhere(abs(c) > cutoff).flatten()
    for i in nonzero_ind:
        coeffs.append(c[i])
        ops.append(qml.grouping.string_to_pauli_word(o[i]))
    try:
        coeffs = qml.math.stack(coeffs)
    except ValueError:
        pass

    return qml.Hamiltonian(coeffs, ops)


def clifford(generator, paulix_wires):
    r"""Compute a Clifford operator from a set of generators and Pauli operators.

    This function computes :math:`U = U_0U_1...U_k` for a set of :math:`k` generators and
    :math:`k` Pauli-X operators.

    Args:
        generator (list[Hamiltonian]): generators expressed as PennyLane Hamiltonians
        paulix_wires (list[int]): indices of the wires the Pauli-X operator acts on

    Returns:
        qml.Hamiltonian: Clifford operator expressed as a PennyLane Hamiltonian

    **Example**

    >>> t1 = qml.Hamiltonian([1.0], [qml.grouping.string_to_pauli_word('ZZII')])
    >>> t2 = qml.Hamiltonian([1.0], [qml.grouping.string_to_pauli_word('ZIZI')])
    >>> t3 = qml.Hamiltonian([1.0], [qml.grouping.string_to_pauli_word('ZIIZ')])
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

    u = functools.reduce(lambda i, j: _observable_mult(i, j), cliff)

    return u


def transform_hamiltonian(h, generator, paulix_wires, paulix_sector):
    r"""Transform a Hamiltonian with a Clifford operator and taper qubits.

    The Hamiltonian is transformed as :math:`H' = U^{\dagger} H U` where :math:`U` is a Clifford
    operator. The transformed Hamiltonian acts trivially on some qubits which are then replaced
    with the eigenvalues of their corresponding Pauli/Identity operator. The list of these
    eigenvalues used for each qubit is defined as the Pauli sector.

    Args:
        h (Hamiltonian): PennyLane Hamiltonian
        generator (list[Hamiltonian]): generators expressed as PennyLane Hamiltonians
        paulix_wires (list[int]): indices of the wires the Pauli-X operator acts on
        paulix_sector llist[int]: eigenvalues of the Pauli-X operators

    Returns:
        qml.Hamiltonian:: the tapered Hamiltonian

    **Example**

    >>> symbols = ["H", "H"]
    >>> geometry = np.array([[0.0, 0.0, -0.69440367], [0.0, 0.0, 0.69440367]])
    >>> mol = qml.hf.Molecule(symbols, geometry)
    >>> H = qml.hf.generate_hamiltonian(mol)(geometry)
    >>> t1 = qml.Hamiltonian([1.0], [qml.grouping.string_to_pauli_word('ZZII')])
    >>> t2 = qml.Hamiltonian([1.0], [qml.grouping.string_to_pauli_word('ZIZI')])
    >>> t3 = qml.Hamiltonian([1.0], [qml.grouping.string_to_pauli_word('ZIIZ')])
    >>> generator = [t1, t2, t3]
    >>> paulix_wires = [1, 2, 3]
    >>> paulix_sector = [1, -1, -1]
    >>> transform_hamiltonian(H, generator, paulix_wires, paulix_sector)
    <Hamiltonian: terms=4, wires=[0]>
    """
    u = clifford(generator, paulix_wires)
    h = _observable_mult(_observable_mult(u, h), u)

    w = list(range(max(h.wires.tolist()) + 1))
    wiremap = dict(zip(w, w))

    val = np.ones(len(h.terms[0])) * complex(1.0)

    for idx, w in enumerate(paulix_wires):
        for i in range(len(h.terms[0])):
            s = qml.grouping.pauli_word_to_string(h.terms[1][i], wire_map=wiremap)
            if s[w] == "X":
                val[i] *= paulix_sector[idx]

    o = []
    for i in range(len(h.terms[0])):
        s = qml.grouping.pauli_word_to_string(h.terms[1][i], wire_map=wiremap)
        wires = [x for x in h.wires if x not in paulix_wires]
        o.append(qml.grouping.string_to_pauli_word("".join([s[i] for i in wires])))

    c = anp.multiply(val, h.terms[0])
    c = qml.math.stack(c)

    return _simplify(qml.Hamiltonian(c, o))
