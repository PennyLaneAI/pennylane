import numpy as np
import pytest

import pennylane as qml

def test_standard_checks():
    """Run standard validity tests."""
    lcu = qml.dot([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)])
    control = [0]

    op = qml.PrepSelectPrep(lcu, control)
    qml.ops.functions.assert_valid(op)

def test_repr():
    """Test the repr method."""
    lcu = qml.dot([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)])
    control = [0]

    op = qml.PrepSelectPrep(lcu, control)
    assert repr(op) == "PrepSelectPrep(coeffs=(0.25, 0.75), ops=(Z(2), X(1) @ X(2)), control=<Wires = [0]>)"

def manual_circuit(lcu, control):
    coeffs, unitaries = lcu.terms()
    normalized_coeffs = np.sqrt(coeffs) / np.linalg.norm(np.sqrt(coeffs))

    qml.StatePrep(normalized_coeffs, wires=control)
    qml.Select(unitaries, control=control)
    qml.adjoint(qml.StatePrep(normalized_coeffs, wires=control))

    return qml.state()

def prepselectprep_circuit(lcu, control):
    qml.PrepSelectPrep(lcu, control)
    return qml.state()

def test_decomposition():
    dev = qml.device("default.qubit")
    manual = qml.QNode(manual_circuit, dev)
    prepselectprep = qml.QNode(prepselectprep_circuit, dev)

    lcu = qml.dot([0.25, 0+0.75j], [qml.Z(2), qml.X(1) @ qml.X(2)])
    assert np.array_equal(
        qml.matrix(manual, wire_order=[0, 1, 2])(lcu, control=0), 
        qml.matrix(prepselectprep, wire_order=[0, 1, 2])(lcu, control=0))

    coeffs = np.array([1/2, 1/2])
    unitaries = [qml.Identity(0), qml.PauliZ(0)]
    lcu = qml.dot(coeffs, unitaries)
    print(
        qml.matrix(manual, wire_order=[0, 'ancilla'])(lcu, control='ancilla'), 
        qml.matrix(prepselectprep, wire_order=[0, 'ancilla'])(lcu, control='ancilla'))
    assert np.array_equal(
        qml.matrix(manual, wire_order=[0, 'ancilla'])(lcu, control='ancilla'), 
        qml.matrix(prepselectprep, wire_order=[0, 'ancilla'])(lcu, control='ancilla'))

    a = 0.25
    b = 0.75
    A = np.array(
        [[a, 0, 0, b],
         [0, -a, b, 0],
         [0, b, a, 0],
         [b, 0, 0, -a]])
    lcu = qml.pauli_decompose(A)
    coeffs, unitaries = lcu.terms()
    unitaries = [qml.map_wires(op, {0: 1, 1: 2}) for op in unitaries]
    lcu = qml.dot(coeffs, unitaries)
    assert np.array_equal(
        qml.matrix(manual, wire_order=[0, 1, 2])(lcu, control=0), 
        qml.matrix(prepselectprep, wire_order=[0, 1, 2])(lcu, control=0))
