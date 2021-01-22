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
"""Unit tests for the RewindTape tape"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.tape.tapes.rewind import dot_product, operation_derivative


class TestOperationDerivative:
    """Tests for operation_derivative function"""

    def test_no_generator_raise(self):
        """Tests if the function raises a ValueError if the input operation has no generator"""
        op = qml.Rot(0.1, 0.2, 0.3, wires=0)

        with pytest.raises(ValueError, match="Operation Rot does not have a generator"):
            operation_derivative(op)

    def test_multiparam_raise(self):
        """Test if the function raises a ValueError if the input operation is composed of multiple
        parameters"""
        class RotWithGen(qml.Rot):
            generator = [np.zeros((2, 2)), 1]

        op = RotWithGen(0.1, 0.2, 0.3, wires=0)

        with pytest.raises(ValueError, match="Operation RotWithGen is not written in terms of"):
            operation_derivative(op)

    def test_rx(self):
        """Test if the function correctly returns the derivative of RX"""
        p = 0.3
        op = qml.RX(p, wires=0)

        derivative = operation_derivative(op)

        expected_derivative = 0.5 * np.array([[-np.sin(p / 2), -1j * np.cos(p / 2)],[-1j * np.cos(p / 2), - np.sin(p / 2)]])

        assert np.allclose(derivative, expected_derivative)

        op.inv()
        derivative_inv = operation_derivative(op)
        expected_derivative_inv = 0.5 * np.array([[-np.sin(p / 2), 1j * np.cos(p / 2)],[1j * np.cos(p / 2), -np.sin(p / 2)]])

        assert not np.allclose(derivative, derivative_inv)
        assert np.allclose(derivative_inv, expected_derivative_inv)

    def test_phase(self):
        """Test if the function correctly returns the derivative of PhaseShift"""
        p = 0.3
        op = qml.PhaseShift(p, wires=0)

        derivative = operation_derivative(op)
        expected_derivative = np.array([[0, 0], [0, 1j * np.exp(1j * p)]])
        assert np.allclose(derivative, expected_derivative)

    def test_cry(self):
        """Test if the function correctly returns the derivative of CRY"""
        p = 0.3
        op = qml.CRY(p, wires=[0, 1])

        derivative = operation_derivative(op)
        expected_derivative = 0.5 * np.array([[0, 0, 0, 0], [0, 0, 0, 0],
                                        [0, 0, -np.sin(p / 2), - np.cos(p / 2)],
                                        [0, 0, np.cos(p / 2), - np.sin(p / 2)],
                                        ])
        assert np.allclose(derivative, expected_derivative)
