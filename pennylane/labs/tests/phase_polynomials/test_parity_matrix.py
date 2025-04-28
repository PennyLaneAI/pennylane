# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pennylane/labs/phase_polynomials/parity_matrix.py"""
from itertools import permutations

import numpy as np
import pytest

import pennylane as qml
from pennylane.labs.phase_polynomials import parity_matrix

circ1 = qml.tape.QuantumScript(
    [
        qml.CNOT((3, 2)),
        qml.CNOT((0, 2)),
        qml.CNOT((2, 1)),
        qml.CNOT((3, 2)),
        qml.CNOT((3, 0)),
        qml.CNOT((0, 2)),
    ],
)
P1 = np.array(
    [[1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]]
)

circ2 = qml.tape.QuantumScript(
    [qml.SWAP((0, 1)), qml.SWAP((1, 2)), qml.SWAP((2, 3))], []
).expand()  # expand into CNOTs
P2 = np.array(
    [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
)


@pytest.mark.parametrize("circ, P_true", ((circ1, P1), (circ2, P2)))
def test_parity_matrix(circ, P_true):
    """Test parity matrix computation"""
    P = parity_matrix(circ, wire_order=range(len(circ.wires)))

    assert np.allclose(P, P_true)


wire_order_abcd = ["a", "b", "c", "d"]
(circ1_letters,), _ = qml.map_wires(circ1, dict(enumerate(wire_order_abcd)))


def test_parity_matrix_with_string_wires():
    """Test parity matrix computation with string valued wires"""
    wire_order1 = ["a", "b", "c", "d"]
    P_re = parity_matrix(circ1_letters, wire_order=wire_order1)

    assert np.allclose(P_re, P1)


@pytest.mark.parametrize("wire_order", list(permutations(range(4), 4)))
def test_parity_matrix_wire_order(wire_order):
    """Test wire_order works as expected"""
    P1_re = parity_matrix(circ1, wire_order=wire_order)
    assert np.allclose(P1_re[np.argsort(wire_order)][:, np.argsort(wire_order)], P1)
