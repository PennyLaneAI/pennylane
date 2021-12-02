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

import pennylane as qml
import autograd.numpy as anp
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
    r"""..."""

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


def transform_hamiltonian(h, symmetry, paulix_wires, paulix_sector):
    r"""..."""

    cliff = []
    for i, t in enumerate(symmetry):
        cliff.append(1 / 2 ** 0.5 * (qml.PauliX(paulix_wires[i]) + t))

    u = functools.reduce(lambda i, j: observable_mult(i, j), cliff)

    uhu = observable_mult(observable_mult(u, h), u)

    h = simplify(qml.Hamiltonian(uhu.coeffs, uhu.ops))

    h_red = qml.Hamiltonian(h.coeffs.copy(), h.ops.copy())

    c = anp.ones(len(h_red.terms[0])) * complex(1.0)

    for idx, w_i in enumerate(paulix_wires):
        for i in range(len(h_red.terms[0])):
            s = pauli_word_to_string(h_red.terms[1][i], wire_map=dict(zip(h.wires, h.wires)))
            if s[w_i] == "X":
                c[i] *= paulix_sector[idx]

    s = []
    for i in range(len(h_red.terms[0])):
        str = pauli_word_to_string(h_red.terms[1][i], wire_map=dict(zip(h.wires, h.wires)))
        wires = [x for x in h.wires if x not in paulix_wires]
        s.append(string_to_pauli_word("".join([str[i] for i in wires])))
        c[i] = c[i] * h_red.terms[0][i]

    return simplify(qml.Hamiltonian(c, s))
