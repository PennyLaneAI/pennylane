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
"""Tests for pennylane/labs/phase_polynomials/phase_polynomial.py"""
# pylint: disable = no-self-use
import numpy as np
import pytest

import pennylane as qml
from pennylane.labs.intermediate_reps import phase_polynomial

# trivial example
circ0 = qml.tape.QuantumScript([])
pmat0 = np.eye(4)
ptab0 = np.array([])
angles0 = np.array([])

# example from fig. 1 in https://arxiv.org/abs/2104.00934
circ1 = qml.tape.QuantumScript(
    [
        qml.CNOT((1, 0)),
        qml.RZ(1, 0),
        qml.CNOT((2, 0)),
        qml.RZ(2, 0),
        qml.CNOT((0, 1)),
        qml.CNOT((3, 1)),
        qml.RZ(3, 1),
    ],
    [],
)

pmat1 = np.array([[1, 1, 1, 0], [1, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
ptab1 = np.array([[1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1]])
angles1 = np.array([1, 2, 3])


class TestPhasePolynomial:
    """Tests for qml.labs.dla.phase_polynomials.phase_polynomial"""

    @pytest.mark.parametrize(
        "circ, res",
        (
            (circ0, (pmat0, ptab0, angles0)),
            (circ1, (pmat1, ptab1, angles1)),
        ),
    )
    def test_computation(self, circ, res):
        """Test phase_polynomial computes the correct parity matrix, parity table and angles"""

        pmat, ptab, angles = phase_polynomial(circ, wire_order=range(4))
        pmat_true, ptab_true, angles_true = res

        assert np.allclose(pmat, pmat_true)
        assert np.allclose(ptab, ptab_true)
        assert np.allclose(angles, angles_true)

    def test_verbose(self, capsys):
        """Test the verbose output of phase_polynomial"""
        _ = phase_polynomial(circ1, verbose=True)
        captured = capsys.readouterr()
        assert "Operator CNOT - #0" in captured.out
        assert "Operator RZ - #1" in captured.out
        assert "Operator CNOT - #2" in captured.out

    def test_wire_order_string_wires(self):
        """Test wire_order with string wires"""
        wire_order_abcd = ["a", "b", "c", "d"]

        (circ1_abcd,), _ = qml.map_wires(circ1, dict(enumerate(wire_order_abcd)))

        pmat, ptab, angles = phase_polynomial(circ1_abcd, wire_order=["a", "b", "c", "d"])

        assert np.allclose(pmat, pmat1)
        assert np.allclose(ptab, ptab1)
        assert np.allclose(angles, angles1)
