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
# pylint: disable = no-self-use
from itertools import permutations

import numpy as np
import pytest

import pennylane as qml
from pennylane.labs.intermediate_reps import parity_matrix

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
wire_order_abcd = ["a", "b", "c", "d"]
(circ1_letters,), _ = qml.map_wires(circ1, dict(enumerate(wire_order_abcd)))

P1 = np.array([[1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])

circ2 = qml.tape.QuantumScript(
    [qml.SWAP((0, 1)), qml.SWAP((1, 2)), qml.SWAP((2, 3))], []
).expand()  # expand into CNOTs
P2 = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])


class TestParityMatrix:
    """Tests for parity_matrix function"""

    @pytest.mark.parametrize("circ, P_true", ((circ1, P1), (circ2, P2)))
    def test_parity_matrix(self, circ, P_true):
        """Test parity matrix computation"""
        P = parity_matrix(circ, wire_order=range(len(circ.wires)))

        assert np.allclose(P, P_true)

    def test_parity_matrix_with_string_wires(self):
        """Test parity matrix computation with string valued wires"""
        wire_order1 = ["a", "b", "c", "d"]
        P_re = parity_matrix(circ1_letters, wire_order=wire_order1)

        assert np.allclose(P_re, P1)

    @pytest.mark.parametrize("wire_order", list(permutations(range(4), 4)))
    def test_parity_matrix_wire_order(self, wire_order):
        """Test wire_order works as expected"""
        P1_re = parity_matrix(circ1, wire_order=wire_order)
        assert np.allclose(P1_re[np.argsort(wire_order)][:, np.argsort(wire_order)], P1)

    def test_WireError(self):
        """Test that WireError is raised when wires in the provided wire_order dont match the circuit wires"""
        with pytest.raises(qml.wires.WireError, match="The provided wire_order"):
            _ = parity_matrix(circ1, wire_order=[1, 2, 3, 4])

    def test_input_validation(self):
        """Test that input circuits are correctly validated"""
        circ = qml.tape.QuantumScript([qml.CNOT((0, 2)), qml.RZ(0.5, 0)])

        with pytest.raises(TypeError, match="parity_matrix requires all input circuits"):
            _ = parity_matrix(circ)
