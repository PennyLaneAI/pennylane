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
Unit tests for the the arithmetic qubit operations
"""
import pytest
import numpy as np

import pennylane as qml


label_data = [
    (qml.QubitCarry(wires=(0, 1, 2, 3)), "QubitCarry"),
    (qml.QubitSum(wires=(0, 1, 2)), "Σ"),
]


@pytest.mark.parametrize("op, label", label_data)
def test_label(op, label):
    assert op.label() == label
    assert op.label(decimals=2) == label
    op.inv()
    assert op.label() == label + "⁻¹"


class TestQubitCarry:
    """Tests the QubitCarry operator."""

    @pytest.mark.parametrize(
        "wires,input_string,output_string,expand",
        [
            ([0, 1, 2, 3], "0000", "0000", True),
            ([0, 1, 2, 3], "0001", "0001", True),
            ([0, 1, 2, 3], "0010", "0010", True),
            ([0, 1, 2, 3], "0011", "0011", True),
            ([0, 1, 2, 3], "0100", "0110", True),
            ([0, 1, 2, 3], "0101", "0111", True),
            ([0, 1, 2, 3], "0110", "0101", True),
            ([0, 1, 2, 3], "0111", "0100", True),
            ([0, 1, 2, 3], "1000", "1000", True),
            ([0, 1, 2, 3], "1001", "1001", True),
            ([0, 1, 2, 3], "1010", "1011", True),
            ([0, 1, 2, 3], "1011", "1010", True),
            ([0, 1, 2, 3], "1100", "1111", True),
            ([0, 1, 2, 3], "1101", "1110", True),
            ([0, 1, 2, 3], "1110", "1101", True),
            ([0, 1, 2, 3], "1111", "1100", True),
            ([3, 1, 2, 0], "0110", "1100", True),
            ([3, 2, 0, 1], "1010", "0110", True),
            ([0, 1, 2, 3], "0000", "0000", False),
            ([0, 1, 2, 3], "0001", "0001", False),
            ([0, 1, 2, 3], "0010", "0010", False),
            ([0, 1, 2, 3], "0011", "0011", False),
            ([0, 1, 2, 3], "0100", "0110", False),
            ([0, 1, 2, 3], "0101", "0111", False),
            ([0, 1, 2, 3], "0110", "0101", False),
            ([0, 1, 2, 3], "0111", "0100", False),
            ([0, 1, 2, 3], "1000", "1000", False),
            ([0, 1, 2, 3], "1001", "1001", False),
            ([0, 1, 2, 3], "1010", "1011", False),
            ([0, 1, 2, 3], "1011", "1010", False),
            ([0, 1, 2, 3], "1100", "1111", False),
            ([0, 1, 2, 3], "1101", "1110", False),
            ([0, 1, 2, 3], "1110", "1101", False),
            ([0, 1, 2, 3], "1111", "1100", False),
            ([3, 1, 2, 0], "0110", "1100", False),
            ([3, 2, 0, 1], "1010", "0110", False),
        ],
    )
    def test_output(self, wires, input_string, output_string, expand, mocker):
        """Test if ``QubitCarry`` produces the right output and is expandable."""
        dev = qml.device("default.qubit", wires=4)
        spy = mocker.spy(qml.QubitCarry, "decomposition")

        with qml.tape.QuantumTape() as tape:
            for i in range(len(input_string)):
                if input_string[i] == "1":
                    qml.PauliX(i)
            qml.QubitCarry(wires=wires)
            qml.probs(wires=[0, 1, 2, 3])

        if expand:
            tape = tape.expand()
        result = dev.execute(tape)
        result = np.argmax(result)
        result = format(result, "04b")
        assert result == output_string

        # checks that decomposition is only used when intended
        assert expand is (len(spy.call_args_list) != 0)

    def test_superposition(self):
        """Test if ``QubitCarry`` works for superposition input states."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=1)
            qml.Hadamard(wires=2)
            qml.QubitCarry(wires=[0, 1, 2, 3])
            return qml.probs(wires=3)

        result = circuit()
        assert np.allclose(result, 0.5)

    def test_matrix_representation(self, tol):
        """Test that the matrix representation is defined correctly"""

        res_static = qml.QubitCarry.compute_matrix()
        res_dynamic = qml.QubitCarry(wires=[0, 1, 2, 3]).matrix()
        expected = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)


class TestQubitSum:
    """Tests for the QubitSum operator"""

    # fmt: off
    @pytest.mark.parametrize(
        "wires,input_state,output_state,expand",
        [
            ([0, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], True),
            ([0, 1, 2], [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], True),
            ([0, 1, 2], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], True),
            ([0, 1, 2], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], True),
            ([0, 1, 2], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], True),
            ([0, 1, 2], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], True),
            ([0, 1, 2], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0], True),
            ([0, 1, 2], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], True),
            ([2, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], True),
            ([1, 2, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], True),
            ([0, 1, 2], [0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5, 0, 0.5, 0.5, 0], True),
            ([0, 1, 2], [np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8),
                         np.sqrt(1 / 8), np.sqrt(1 / 8)],
             [np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8),
              np.sqrt(1 / 8), np.sqrt(1 / 8)], True),
            ([0, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], False),
            ([0, 1, 2], [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], False),
            ([0, 1, 2], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], False),
            ([0, 1, 2], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], False),
            ([0, 1, 2], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], False),
            ([0, 1, 2], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], False),
            ([0, 1, 2], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0], False),
            ([0, 1, 2], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], False),
            ([2, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], False),
            ([1, 2, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], False),
            ([0, 1, 2], [0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5, 0, 0.5, 0.5, 0], False),
            ([0, 1, 2], [np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8),
                         np.sqrt(1 / 8), np.sqrt(1 / 8)],
             [np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8),
              np.sqrt(1 / 8), np.sqrt(1 / 8)], False),
        ],
    )
    # fmt: on
    def test_output(self, wires, input_state, output_state, expand, mocker):
        """Test if ``QubitSum`` produces the correct output"""
        dev = qml.device("default.qubit", wires=3)
        spy = mocker.spy(qml.QubitSum, "decomposition")

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(input_state, wires=[0, 1, 2])

            if expand:
                qml.QubitSum(wires=wires).expand()
            else:
                qml.QubitSum(wires=wires)

            qml.state()

        result = dev.execute(tape)
        assert np.allclose(result, output_state)

        # checks that decomposition is only used when intended
        assert expand is (len(spy.call_args_list) != 0)

    def test_adjoint(self):
        """Test the adjoint method of QubitSum by reconstructing the unitary matrix and checking
        if it is equal to qml.QubitSum's matrix representation (recall that the operation is self-adjoint)"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def f(state):
            qml.QubitStateVector(state, wires=range(3))
            qml.adjoint(qml.QubitSum)(wires=range(3))
            return qml.probs(wires=range(3))

        u = np.array([f(state) for state in np.eye(2**3)]).T
        assert np.allclose(u, qml.QubitSum.compute_matrix())

    def test_matrix_representation(self, tol):
        """Test that the matrix representation is defined correctly"""

        res_static = qml.QubitSum.compute_matrix()
        res_dynamic = qml.QubitSum(wires=[0, 1, 2]).matrix()
        expected = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)
