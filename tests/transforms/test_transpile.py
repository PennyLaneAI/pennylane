"""
Unit tests for transpiler function.
"""
import pytest
from math import isclose

from pennylane import numpy as np
import pennylane as qml
from pennylane.transforms.transpile import transpile


def build_qfunc(wires):
    def qfunc(x, y, z):
        qml.Hadamard(wires=wires[0])
        qml.RZ(z, wires=wires[2])
        qml.CNOT(wires=[wires[2], wires[0]])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RX(x, wires=wires[0])
        qml.CNOT(wires=[wires[0], wires[2]])
        qml.RZ(-z, wires=wires[2])
        qml.RX(y, wires=wires[0])
        qml.PauliY(wires=wires[2])
        qml.CY(wires=[wires[1], wires[2]])
        return qml.expval(qml.PauliZ(wires=wires[0]))

    return qfunc


class TestTranspile:
    """ Unit tests for transpile function """

    def test_transpile_invalid_coupling(self):
        """ test that error is raised when coupling_map is invalid"""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        # build circuit
        original_qfunc = build_qfunc([0, 1, 2])
        transpiled_qfunc = transpile(coupling_map=[(0, 1)])(original_qfunc)
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        with pytest.raises(ValueError):
            transpiled_qnode(0.1, 0.2, 0.3)

    def test_transpile_qfunc_transpiled_equivalent(self):
        """ test that transpile does not alter output """
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        # build circuit without transpile
        original_qfunc = build_qfunc([0, 1, 2])
        original_qnode = qml.QNode(original_qfunc, dev)
        original_expectation = original_qnode(0.1, 0.2, 0.3)

        # build circuit with transpile
        transpiled_qfunc = transpile(coupling_map=[(0, 1), (1, 2)])(original_qfunc)
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        transpiled_expectation = transpiled_qnode(0.1, 0.2, 0.3)

        assert isclose(original_expectation, transpiled_expectation, abs_tol=np.finfo(float).eps)
