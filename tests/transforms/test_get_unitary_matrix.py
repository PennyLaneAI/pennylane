import pennylane as qml
import numpy as np
import pytest
from gate_data import I, X, Y, Z, H, S, T

np.set_printoptions(suppress=True, linewidth=np.nan, precision=3)

from pennylane.transforms.get_unitary_matrix import get_unitary_matrix

# test non-parametric single qubit gates
nonparam_1qubit_op_list = [
    qml.PauliX,
    qml.PauliY,
    qml.PauliZ,
    qml.Hadamard,
    qml.S,
    #    qml.T,
    #    qml.SX
]


@pytest.mark.parametrize("op", nonparam_1qubit_op_list)
@pytest.mark.parametrize("wire", [0, 1, 2])
def test_compute_matrix_nonparam_1qubit_ops(op, wire):

    wires = [0, 1, 2]

    def testcircuit(wire):
        op(wires=wire)

    get_matrix = get_unitary_matrix(testcircuit, wires)
    matrix = get_matrix(wire)

    if wire == 0:
        correct_matrix = np.kron(op(wires=wire).matrix, np.eye(4))
    if wire == 1:
        correct_matrix = np.kron(np.eye(2), np.kron(op(wires=wire).matrix, np.eye(2)))
    if wire == 2:
        correct_matrix = np.kron(np.eye(4), op(wires=wire).matrix)

    assert np.allclose(matrix, correct_matrix)


def test_compute_matrix_many_nonparam_1qubit_ops():

    wires = ["a", "b", "c"]

    def testcircuit():
        qml.S(wires="a")
        qml.S(wires="b")
        qml.S(wires="c")

    get_matrix = get_unitary_matrix(testcircuit, wires)
    matrix = get_matrix()

    correct_matrix = np.kron(X, np.kron(S, H))
