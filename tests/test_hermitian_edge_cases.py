import pytest
import numpy as np
from scipy.linalg import block_diag

import pennylane as qml
from pennylane.qnodes.qubit import QubitQNode
from pennylane.qnodes.base import QuantumFunctionError
from pennylane.plugins.default_qubit import Y, Z


THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


@pytest.mark.parametrize("analytic", [True, False])
@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestEdgeHermitian:
    def test_hermitian_two_wires_identity_expectation_only_hermitian(self, analytic, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = qml.device("default.qubit", wires=3, analytic=analytic, shots=1000000)

        A = np.array([[1.02789352, 1.61296440 - 0.3498192j],
                      [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

        Identity = np.array([[1, 0],[0, 1]])
        obs = np.kron(np.kron(Identity, Identity), A)

        @qml.qnode(dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(obs, wires=[2, 1, 0]))

        res = circuit()

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]

        expected = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        assert np.allclose(res, expected, atol=0.01, rtol=0)

    def test_hermitian_two_wires_identity_expectation_with_tensor(self, analytic, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = qml.device("default.qubit", wires=3, analytic=analytic, shots=1000000)

        A = np.array([[1.02789352, 1.61296440 - 0.3498192j],
                      [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

        Identity = np.array([[1, 0],[0, 1]])
        obs = np.kron(Identity, A)

        @qml.qnode(dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(obs, wires=[2,0]) @ qml.Identity(1))

        res = circuit()

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]

        expected = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        assert np.allclose(res, expected, atol=0.01, rtol=0)
