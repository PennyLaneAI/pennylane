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
from copy import copy

import pytest
import numpy as np

import pennylane as qml
from pennylane.wires import Wires

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


class TestIntegerComparator:
    """Tests for the IntegerComparator"""

    @pytest.mark.parametrize(
        "value,geq,wires,expected_error_message",
        [
            (4.20, False, [0, 1, 2], "The comparable value must be an integer."),
            (2, True, None, "Must specify the target wire where the operation acts on."),
            (
                2,
                True,
                [1],
                r"IntegerComparator: wrong number of wires. 1 wire\(s\) given. Need at least 2.",
            ),
        ],
    )
    def test_invalid_mixed_polarity_controls(self, value, geq, wires, expected_error_message):
        """Test if IntegerComparator properly handles invalid mixed-polarity
        control values."""

        with pytest.raises(ValueError, match=expected_error_message):
            qml.IntegerComparator(value, geq=geq, wires=wires).matrix()

    def test_compute_matrix_geq_True(self):
        """Test compute_matrix for geq=True"""
        mat1 = qml.IntegerComparator.compute_matrix(value=2, control_wires=[0, 1], geq=True)
        mat2 = np.zeros((8, 8))

        mat2[0, 0] = 1
        mat2[1, 1] = 1
        mat2[2, 2] = 1
        mat2[3, 3] = 1
        mat2[4, 5] = 1
        mat2[5, 4] = 1
        mat2[6, 7] = 1
        mat2[7, 6] = 1

        assert np.allclose(mat1, mat2)

    def test_compute_matrix_geq_False(self):
        """Test compute_matrix for geq=False"""
        mat1 = qml.IntegerComparator.compute_matrix(value=2, control_wires=[0, 1], geq=False)
        mat2 = np.zeros((8, 8))

        mat2[0, 1] = 1
        mat2[1, 0] = 1
        mat2[2, 3] = 1
        mat2[3, 2] = 1
        mat2[4, 4] = 1
        mat2[5, 5] = 1
        mat2[6, 6] = 1
        mat2[7, 7] = 1

        assert np.allclose(mat1, mat2)

    @pytest.mark.parametrize(
        "value,control_wires,geq,expected_error_message",
        [
            (None, [0, 1], True, "The value to compare to must be specified."),
            (4.20, [0, 1], False, "The compared value must be an int. Got <class 'float'>."),
        ],
    )
    def test_invalid_args_compute_matrix(self, value, control_wires, geq, expected_error_message):
        """Test if compute_matrix properly handles invalid arguments."""
        with pytest.raises(ValueError, match=expected_error_message):
            qml.IntegerComparator.compute_matrix(value=value, control_wires=control_wires, geq=geq)

    def test_compute_matrix_large_value(self):
        """Test if compute_matrix properly handles values exceeding the Hilbert space of the control
        wires."""

        mat1 = qml.IntegerComparator.compute_matrix(value=10, control_wires=[0, 1], geq=True)
        mat2 = np.eye(8)

        assert np.allclose(mat1, mat2)

    def test_compute_matrix_value_zero(self):
        """Test if compute_matrix properly handles value=0 when geq=False."""

        mat1 = qml.IntegerComparator.compute_matrix(value=0, control_wires=[0, 1], geq=False)
        mat2 = np.eye(8)

        assert np.allclose(mat1, mat2)

    def test_adjoint_method(self):
        """Test ``adjoint()`` method."""
        op = (qml.IntegerComparator(2, wires=(0, 1, 2, 3)),)
        adj_op = copy.copy(op)
        for _ in range(4):
            adj_op = adj_op.adjoint()

            assert adj_op.name == op.name

    def test_control_wires(self):
        """Test ``control_wires`` attribute for non-parametrized operations."""
        op, control_wires = qml.IntegerComparator(2, wires=(0, 1, 2, 3)), Wires([0, 1, 2])
        assert op.control_wires == control_wires

    def test_label_method(self):
        """Test label method"""

        op, label1, label2 = qml.IntegerComparator(2, wires=(0, 1, 2, 3)), ">=2", ">=2"

        assert op.label() == label1
        assert op.label(decimals=2) == label1

        op.inv()
        assert op.label() == label2
