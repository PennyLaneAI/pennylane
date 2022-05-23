"""
Unit tests for transpiler function.
"""
from math import isclose
import pytest

from pennylane import numpy as np
import pennylane as qml
from pennylane.transforms.transpile import transpile


def build_qfunc_probs(wires):
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
        return qml.probs(wires=[0, 1])

    return qfunc


def build_qfunc_pauli_z(wires):
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
    """Unit tests for transpile function"""

    def test_transpile_invalid_coupling(self):
        """test that error is raised when coupling_map is invalid"""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        # build circuit
        original_qfunc = build_qfunc_pauli_z([0, 1, 2])
        transpiled_qfunc = transpile(coupling_map=[(0, 1)])(original_qfunc)
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        err_msg = (
            "Not all wires present in coupling map! wires: \[0, 2, 1\], coupling map: \[0, 1\]"
        )
        with pytest.raises(ValueError, match=err_msg):
            transpiled_qnode(0.1, 0.2, 0.3)

    def test_transpile_raise_not_implemented_hamiltonain_mmt(self):
        """test that error is raised when measurement is expectation of a Hamiltonian"""
        dev = qml.device("default.qubit", wires=[0, 1, 2, 3])
        coeffs = [1]
        obs = [qml.PauliZ(0) @ qml.PauliZ(1)]
        H = qml.Hamiltonian(coeffs, obs)

        def circuit():
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 3])
            return qml.expval(H)

        # build circuit
        transpiled_qfunc = transpile(coupling_map=[(0, 1), (1, 2), (2, 3)])(circuit)
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        err_msg = (
            "Measuring expectation values of tensor products or Hamiltonians is not yet supported"
        )
        with pytest.raises(NotImplementedError, match=err_msg):
            transpiled_qnode()

    def test_transpile_raise_not_implemented_tensorproduct_mmt(self):
        """test that error is raised when measurement is expectation of a Tensor product"""
        dev = qml.device("default.qubit", wires=[0, 1, 2, 3])

        def circuit():
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 3])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        # build circuit
        transpiled_qfunc = transpile(coupling_map=[(0, 1), (1, 2), (2, 3)])(circuit)
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        err_msg = (
            r"Measuring expectation values of tensor products or Hamiltonians is not yet supported"
        )
        with pytest.raises(NotImplementedError, match=err_msg):
            transpiled_qnode()

    def test_transpile_qfunc_transpiled_mmt_obs(self):
        """test that transpile does not alter output for expectation value of an observable"""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        # build circuit without transpile
        original_qfunc = build_qfunc_pauli_z([0, 1, 2])
        original_qnode = qml.QNode(original_qfunc, dev)
        original_expectation = original_qnode(0.1, 0.2, 0.3)

        # build circuit with transpile
        transpiled_qfunc = transpile(coupling_map=[(0, 1), (1, 2)])(original_qfunc)
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        transpiled_expectation = transpiled_qnode(0.1, 0.2, 0.3)

        assert isclose(original_expectation, transpiled_expectation, abs_tol=np.finfo(float).eps)

    def test_transpile_qfunc_transpiled_mmt_probs(self):
        """test that transpile does not alter output for probs measurement"""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        # build circuit without transpile
        original_qfunc = build_qfunc_probs([0, 1, 2])
        original_qnode = qml.QNode(original_qfunc, dev)
        original_probs = original_qnode(0.1, 0.2, 0.3)

        # build circuit with transpile
        transpiled_qfunc = transpile(coupling_map=[(0, 1), (1, 2)])(original_qfunc)
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        transpiled_probs = transpiled_qnode(0.1, 0.2, 0.3)

        assert all(
            isclose(po, pt, abs_tol=np.finfo(float).eps)
            for po, pt in zip(original_probs, transpiled_probs)
        )

    @pytest.mark.autograd
    def test_transpile_differentiable(self):
        """test that circuit remains differentiable after transpilation"""
        dev = qml.device("default.qubit", wires=3)

        def circuit(parameters):
            qml.RX(parameters[0], wires=0)
            qml.RY(parameters[1], wires=1)
            qml.CNOT(wires=[0, 2])
            qml.PhaseShift(parameters[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        transpiled_circ = transpile(coupling_map=[(0, 1), (1, 2)])(circuit)
        transpiled_qnode = qml.QNode(transpiled_circ, dev, interface="autograd")
        params = np.array([0.5, 0.1, 0.2], requires_grad=True)
        qml.gradients.param_shift(transpiled_qnode)(params)
