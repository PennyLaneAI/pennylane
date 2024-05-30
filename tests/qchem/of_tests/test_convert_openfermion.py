# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for functions needed for converting ``QubitOperator``from OpenFermion to 
PennyLane ``LinearCombination`` and vice versa.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np

openfermion = pytest.importorskip("openfermion")


def test_from_openfermion():
    """Test the from_openfermion function."""
    q_op = openfermion.QubitOperator("X0", 1.2) + openfermion.QubitOperator("Z1", 2.4)
    pl_linear_combination = qml.from_openfermion(q_op)

    assert str(pl_linear_combination) == "1.2 * X(0) + 2.4 * Z(1)"


def test_from_openfermion_tol():
    """Test the from_openfermion function with complex coefficients."""
    q_op = openfermion.QubitOperator("X0", complex(1.0, 1e-8)) + openfermion.QubitOperator("Z1", complex(1.3, 1e-8))
    # The method should discard the imaginary part of the coefficients.
    pl_linear_combination = qml.from_openfermion(q_op, tol=1e-6)
    # Check whether coefficients do not contain imaginary part.
    assert ~np.any( pl_linear_combination.coeffs.imag )
    pl_linear_combination = qml.from_openfermion(q_op, tol=1e-10)
    # Check whether coefficients do contain imaginary part since imaginary part exceeds treshold.
    assert np.any( pl_linear_combination.coeffs.imag )


def test_from_openfermion_custom_wires():
    """Test the from_openfermion function with custom (swapped) wires."""
    q_op = openfermion.QubitOperator("X0", 1.2) + openfermion.QubitOperator("Z1", 2.4) + openfermion.QubitOperator("Y2", 0.1)
    pl_linear_combination = qml.from_openfermion(q_op, wires={0: 2, 1: 1, 2: 0})

    assert str(pl_linear_combination) == "1.2 * X(2) + 2.4 * Z(1) + 0.1 * Y(0)"


def test_to_openfermion():
    """Test the to_openfermion function with default wires."""

    pl_linear_combination = 1.2 * qml.X(0) + 2.4 * qml.Z(1)
    q_op = qml.to_openfermion(pl_linear_combination)

    q_op_str = str(q_op)

    # Remove new line characters
    q_op_str = q_op_str.replace("\n", " ")
    expected = "(1.2+0j) [X0] + (2.4+0j) [Z1]"
    assert q_op_str == expected


def test_to_openfermion_custom_wires():
    """Test the to_openfermion function with custom (swapped) wires."""

    pl_linear_combination = 1.2 * qml.X(0) + 2.4 * qml.Z(1) + 0.2 * qml.Y(2)
    # Custom mapping where the first and third qubits are swapped.
    q_op = qml.to_openfermion(pl_linear_combination, wires={0: 2, 1: 1, 2: 0})

    q_op_str = str(q_op)

    # Remove new line characters
    q_op_str = q_op_str.replace("\n", " ")
    # The qubit operator should now reflect the swapped order of the first and third qubits.
    expected = "(0.2+0j) [Y0] + (2.4+0j) [Z1] + (1.2+0j) [X2]"
    assert q_op_str == expected
