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
Unit tests for the get_unitary_matrix transform
"""
from functools import reduce
import pytest
import numpy as np
from gate_data import I, X, H, S, CNOT
import pennylane as qml

from pennylane.transforms.get_unitary_matrix import get_unitary_matrix

# test non-parametric single qubit gates
nonparam_1qubit_op_list = [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.S, qml.T, qml.SX]


@pytest.mark.parametrize("op", nonparam_1qubit_op_list)
@pytest.mark.parametrize("wire", [0, 1, 2])
def test_get_unitary_matrix_nonparam_1qubit_ops(op, wire):
    """Check the matrices for different nonparametrized single-qubit gates, which are acting on different qubits in a space of three qubits."""
    wires = [0, 1, 2]

    def testcircuit(wire):
        op(wires=wire)

    get_matrix = get_unitary_matrix(testcircuit, wires)
    matrix = get_matrix(wire)

    if wire == 0:
        expected_matrix = np.kron(op(wires=wire).matrix, np.eye(4))
    if wire == 1:
        expected_matrix = np.kron(np.eye(2), np.kron(op(wires=wire).matrix, np.eye(2)))
    if wire == 2:
        expected_matrix = np.kron(np.eye(4), op(wires=wire).matrix)

    assert np.allclose(matrix, expected_matrix)


# Test a circuit containing multiple gates
def test_get_unitary_matrix_multiple_ops():
    """Check the total matrix for a circuit containing multiple gates. Also checks that non-integer wires work"""
    wires = ["a", "b", "c"]

    def testcircuit():
        qml.PauliX(wires="a")
        qml.S(wires="b")
        qml.Hadamard(wires="c")
        qml.CNOT(wires=["b", "c"])

    get_matrix = get_unitary_matrix(testcircuit, wires)
    matrix = get_matrix()

    expected_matrix = np.kron(I, CNOT) @ np.kron(X, np.kron(S, H))

    assert np.allclose(matrix, expected_matrix)


def test_get_unitary_matrix_more_ops():
    "Test some more operators. Same circuit as test_single_qubit_fusion_multiple_qubits() in test_single_qubit_fusion.py"
    wires = ["a", "b"]

    def testcircuit():
        qml.RZ(0.3, wires="a")
        qml.RY(0.5, wires="a")
        qml.Rot(0.1, 0.2, 0.3, wires="b")
        qml.RX(0.1, wires="a")
        qml.CNOT(wires=["b", "a"])
        qml.SX(wires="b")
        qml.S(wires="b")
        qml.PhaseShift(0.3, wires="b")

    op1 = np.kron(qml.RZ(0.3, wires="a").matrix, I)
    op2 = np.kron(qml.RY(0.5, wires="a").matrix, I)
    op3 = np.kron(I, qml.Rot(0.1, 0.2, 0.3, wires="b").matrix)
    op4 = np.kron(qml.RX(0.1, wires="a").matrix, I)
    op5 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    op6 = np.kron(I, qml.SX(wires="b").matrix)
    op7 = np.kron(I, S)
    op8 = np.kron(I, qml.PhaseShift(0.3, wires="b").matrix)

    ops = [op8, op7, op6, op5, op4, op3, op2, op1]

    expected_matrix = np.linalg.multi_dot(ops)

    get_matrix = get_unitary_matrix(testcircuit, wires)
    matrix = get_matrix()

    assert np.allclose(matrix, expected_matrix)


@pytest.mark.parametrize("target_wire", [0, 2, 3, 4])
def test_get_unitary_matrix_CNOT(target_wire):
    """Test CNOT: 2-qubit gate with different target wires, some non-adjacent."""
    wires = [0, 1, 2, 3, 4]

    def testcircuit():
        qml.CNOT(wires=[1, target_wire])

    get_matrix = get_unitary_matrix(testcircuit, wires)
    matrix = get_matrix()

    # test the matrix operation on a state
    state0 = [1, 0]
    state1 = [0, 1]
    teststate = reduce(np.kron, [state1, state1, state1, state1, state1])

    if target_wire == 0:
        expected_state = reduce(np.kron, [state0, state1, state1, state1, state1])
    elif target_wire == 2:
        expected_state = reduce(np.kron, [state1, state1, state0, state1, state1])
    elif target_wire == 3:
        expected_state = reduce(np.kron, [state1, state1, state1, state0, state1])
    elif target_wire == 4:
        expected_state = reduce(np.kron, [state1, state1, state1, state1, state0])

    obtained_state = matrix @ teststate

    assert np.allclose(obtained_state, expected_state)


def test_get_unitary_matrix_CRX():
    """Test controlled rotation with non-adjacent control- and target wires"""
    testangle = np.pi / 4

    wires = [0, 1, 2]

    def testcircuit():
        qml.CRX(testangle, wires=[2, 0])

    # test applying to state
    state0 = [1, 0]
    state1 = [0, 1]

    # perform controlled rotation
    teststate1 = reduce(np.kron, [state1, state1, state1])
    # do not perform controlled rotation
    teststate0 = reduce(np.kron, [state1, state1, state0])

    expected_state1 = reduce(np.kron, [qml.RX(testangle, wires=1).matrix @ state1, state1, state1])
    expected_state0 = teststate0

    get_matrix = get_unitary_matrix(testcircuit, wires)
    matrix = get_matrix()

    obtained_state1 = matrix @ teststate1
    obtained_state0 = matrix @ teststate0

    assert np.allclose(obtained_state1, expected_state1)
    assert np.allclose(obtained_state0, expected_state0)


def test_get_unitary_matrix_Toffoli():
    """Check the Toffoli matrix by its action on states"""
    wires = [0, "a", 2, "c", 4]

    def testcircuit():
        qml.Toffoli(wires=[0, 4, "a"])

    # test applying to state
    state0 = [1, 0]
    state1 = [0, 1]

    teststate1 = reduce(np.kron, [state1, state1, state1, state1, state1])
    teststate2 = reduce(np.kron, [state0, state0, state1, state1, state0])

    expected_state1 = reduce(np.kron, [state1, state0, state1, state1, state1])
    expected_state2 = teststate2

    get_matrix = get_unitary_matrix(testcircuit, wires)
    matrix = get_matrix()

    obtained_state1 = matrix @ teststate1
    obtained_state2 = matrix @ teststate2

    assert np.allclose(obtained_state1, expected_state1)
    assert np.allclose(obtained_state2, expected_state2)


# Test MultiControlledX
def test_get_unitary_matrix_MultiControlledX():
    """Test with many control wires"""
    wires = [0, 1, 2, 3, 4, 5]

    def testcircuit():
        qml.MultiControlledX(control_wires=[0, 2, 4, 5], wires=3)

    state0 = [1, 0]
    state1 = [0, 1]

    teststate1 = reduce(np.kron, [state1, state1, state1, state1, state1, state1])
    teststate2 = reduce(np.kron, [state0, state1, state0, state0, state1, state0])

    expected_state1 = reduce(np.kron, [state1, state1, state1, state0, state1, state1])
    expected_state2 = teststate2

    get_matrix = get_unitary_matrix(testcircuit, wires)
    matrix = get_matrix()

    obtained_state1 = matrix @ teststate1
    obtained_state2 = matrix @ teststate2

    assert np.allclose(obtained_state1, expected_state1)
    assert np.allclose(obtained_state2, expected_state2)
