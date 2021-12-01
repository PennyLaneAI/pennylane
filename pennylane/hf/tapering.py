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
from pennylane.grouping import pauli_mult_with_phase, pauli_word_to_string, string_to_pauli_word


def hamiltonian_mult(h1, h2):
    r"""..."""
    o = []
    c = []
    for i in range(len(h1.terms[0])):
        for j in range(len(h2.terms[0])):
            op, phase = pauli_mult_with_phase(h1.terms[1][i], h2.terms[1][j])
            o.append(op)
            c.append(phase * h1.terms[0][i] * h2.terms[0][j])
    return qml.Hamiltonian(c, o)


def simplify(h):
    r"""..."""

    c = []
    o = []
    s = []

    for term in h.terms[1]:
        term = qml.operation.Tensor(term).prune()
        s.append(qml.grouping.pauli_word_to_string(term, wire_map=dict(zip(h.wires, h.wires))))

    o_set = list(set(s))
    c_set = anp.zeros(len(o_set))

    for i, item in enumerate(s):
        o_ = anp.zeros(len(o_set))
        o_[o_set.index(item)] = 1.0
        c_set = c_set + o_ * h.terms[0][i]

    for i, coeff in enumerate(c_set):
        if not anp.allclose(coeff, 0.0):
            c.append(coeff)
            o.append(o_set[i])

    o = [qml.grouping.string_to_pauli_word(i) for i in o]

    return qml.Hamiltonian(c, o)


def transform_hamiltonian(h, symmetry, paulix_wires, paulix_sector):
    r"""..."""

    cliff = []
    for i, t in enumerate(symmetry):
        cliff.append(1 / 2 ** 0.5 * (qml.PauliX(paulix_wires[i]) + t))

    u = functools.reduce(lambda i, j: hamiltonian_mult(i, j), cliff)

    uhu = hamiltonian_mult(hamiltonian_mult(u, h), u)

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
