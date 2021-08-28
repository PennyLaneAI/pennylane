import pennylane as qml
import numpy as np
import pytest
from gate_data import I, X, Y, Z, H, S, T
from functools import reduce


np.set_printoptions(suppress=True, linewidth=np.nan, precision=3)

from pennylane.transforms.get_unitary_matrix import get_unitary_matrix

# test non-parametric single qubit gates
nonparam_1qubit_op_list = [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.S, qml.T, qml.SX]


@pytest.mark.parametrize("op", nonparam_1qubit_op_list)
@pytest.mark.parametrize("wire", [0, 1, 2])
def test_get_unitary_matrix_nonparam_1qubit_ops(op, wire):

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

    wires = ["a", "b", "c"]

    def testcircuit():
        qml.PauliX(wires="a")
        qml.S(wires="b")
        qml.Hadamard(wires="c")
        qml.CNOT(wires=["b", "c"])

    get_matrix = get_unitary_matrix(testcircuit, wires)
    matrix = get_matrix()

    expected_matrix = np.kron(I, qml.CNOT.matrix) @ np.kron(X, np.kron(S, H))

    assert np.allclose(matrix, expected_matrix)


# Test CNOT: 2-qubit gate with different target wires, some non-adjacent
@pytest.mark.parametrize("target_wire", [0, 2, 3, 4])
def test_get_unitary_matrix_CNOT(target_wire):
    wires = [0, 1, 2, 3, 4]

    def testcircuit():
        qml.CNOT(wires=[1, target_wire])

    get_matrix = get_unitary_matrix(testcircuit, wires)
    matrix = get_matrix()

    # test applying to state
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


# TEST CRX with non-adjacent wire
def test_get_unitary_matrix_CRX():

    testangle = np.pi / 4

    wires = [0, 1, 2]

    def testcircuit():
        qml.CRX(testangle, wires=[0, 2])

    # test applying to state
    state0 = [1, 0]
    state1 = [0, 1]

    # perform controlled rotation
    teststate1 = reduce(np.kron, [state1, state1, state1])
    # do not perform controlled rotation
    teststate0 = reduce(np.kron, [state0, state1, state1])

    expected_state1 = reduce(np.kron, [state1, state1, qml.RX(testangle, wires=1).matrix @ state1])
    expected_state0 = teststate0

    get_matrix = get_unitary_matrix(testcircuit, wires)
    matrix = get_matrix()

    obtained_state1 = matrix @ teststate1
    obtained_state0 = matrix @ teststate0

    assert np.allclose(obtained_state1, expected_state1)
    assert np.allclose(obtained_state0, expected_state0)


# Test Toffoli
def test_get_unitary_matrix_Toffoli():

    wires = [0, 1, 2, 3, 4]

    def testcircuit():
        qml.Toffoli(wires=[0, 4, 1])

    # test applying to state
    state0 = [1, 0]
    state1 = [0, 1]

    teststate1 = reduce(np.kron, [state1, state1, state1, state1, state1])
    teststate2 = reduce(np.kron, [state0, state1, state1, state1, state1])

    expected_state1 = reduce(np.kron, [state1, state0, state1, state1, state1])
    expected_state2 = teststate2

    get_matrix = get_unitary_matrix(testcircuit, wires)
    matrix = get_matrix()

    obtained_state1 = matrix @ teststate1
    obtained_state2 = matrix @ teststate2

    assert np.allclose(obtained_state1, expected_state1)
    assert np.allclose(obtained_state2, expected_state2)
