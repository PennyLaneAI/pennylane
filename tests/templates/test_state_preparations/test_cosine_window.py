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
Unit tests for the CosineWindow template.
"""
# pylint: disable=too-few-public-methods
import pytest
import numpy as np
import pennylane as qml
from pennylane.wires import WireError


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    def test_correct_gates_single_wire(self):
        """Test that the correct gates are applied."""

        op = qml.CosineWindow(wires=[0])
        queue = op.expand().operations

        assert queue[0].name == "Hadamard"
        assert queue[1].name == "RZ"
        assert queue[2].name == "Adjoint(QFT)"
        assert queue[3].name == "PhaseShift"

        assert np.isclose(queue[3].data[0], np.pi / 2)

    def test_correct_gates_many_wires(self):
        """Test that the correct gates are applied on two wires."""

        op = qml.CosineWindow(wires=[0, 1, 2, 3, 4])
        queue = op.decomposition()

        assert queue[0].name == "Hadamard"
        assert queue[1].name == "RZ"
        assert queue[2].name == "Adjoint(QFT)"

        for ind, q in enumerate(queue[3:]):
            assert q.name == "PhaseShift"
            assert np.isclose(q.data[0], np.pi / 2 ** (ind + 1))

    def test_custom_wire_labels(self):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.CosineWindow(wires=range(3))
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.CosineWindow(wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2)
        assert np.allclose(state1, state2)


class TestInputs:
    """Test inputs and pre-processing."""

    def test_id(self):
        """Tests that the id attribute can be set."""
        wires = [0, 1, 2]
        template = qml.CosineWindow(wires=wires, id="a")
        assert template.id == "a"
        assert template.wires == qml.Wires(wires)


class TestStateVector:
    """Test the state_vector() method of various CosineWindow operations."""

    def test_CosineWindow_state_vector(self):
        """Tests that the state vector is correct for a single wire."""
        op = qml.CosineWindow(wires=[0])
        res = op.state_vector()
        expected = np.array([0.0, 1.0])
        assert np.allclose(res, expected)

        op = qml.CosineWindow(wires=[0, 1])
        res = np.reshape(op.state_vector() ** 2, (-1,))
        expected = np.array([0.0, 0.25, 0.5, 0.25])
        assert np.allclose(res, expected)

    def test_CosineWindow_state_vector_bad_wire_order(self):
        """Tests that the provided wire_order must contain the wires in the operation."""
        qsv_op = qml.CosineWindow(wires=[0, 1])
        with pytest.raises(WireError, match="wire_order must contain all CosineWindow wires"):
            qsv_op.state_vector(wire_order=[1, 2])

    def test_CosineWindow_state_vector_wire_order(self):
        """Tests that the state vector works with a different order of wires."""
        op = qml.CosineWindow(wires=[0, 1])
        res = np.reshape(op.state_vector(wire_order=[1, 0]) ** 2, (-1,))
        expected = np.array([0.0, 0.5, 0.25, 0.25])
        assert np.allclose(res, expected)
