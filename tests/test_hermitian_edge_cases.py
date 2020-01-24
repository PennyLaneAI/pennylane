import itertools
import pytest
import numpy as np
from scipy.linalg import block_diag

import pennylane as qml
from pennylane.qnodes.qubit import QubitQNode
from pennylane.qnodes.base import QuantumFunctionError
from gate_data import Y, Z


THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


@pytest.mark.parametrize("analytic", [True, False])
@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestEdgeHermitian:
    def test_hermitian_two_wires_identity_expectation_only_hermitian(
        self, analytic, theta, phi, varphi
    ):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = qml.device("default.qubit", wires=3, analytic=analytic, shots=1000000)

        A = np.array(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
        )

        Identity = np.array([[1, 0], [0, 1]])
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

    def test_hermitian_two_wires_identity_expectation_with_tensor(
        self, analytic, theta, phi, varphi
    ):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = qml.device("default.qubit", wires=3, analytic=analytic, shots=1000000)

        A = np.array(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
        )

        Identity = np.array([[1, 0], [0, 1]])
        obs = np.kron(Identity, A)

        @qml.qnode(dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(obs, wires=[2, 0]) @ qml.Identity(1))

        res = circuit()

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]

        expected = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        assert np.allclose(res, expected, atol=0.01, rtol=0)

    @pytest.mark.parametrize("w1, w2", list(itertools.permutations(range(4), 2)))
    def test_hermitian_two_wires_permuted(self, w1, w2, analytic, theta, phi, varphi):
        """Test that an hermitian expectation with various wires permuted works"""
        dev = qml.device("default.qubit", wires=4, shots=1000000, analytic=analytic)
        theta = 0.543

        A = np.array(
            [
                [1, 2j, 1 - 2j, 0.5j],
                [-2j, 0, 3 + 4j, 1],
                [1 + 2j, 3 - 4j, 0.75, 1.5 - 2j],
                [-0.5j, 1, 1.5 + 2j, -1],
            ]
        )

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[w1])
            qml.RY(2 * theta, wires=[w2])
            qml.CNOT(wires=[w1, w2])
            return qml.expval(qml.Hermitian(A, wires=[w1, w2]))

        res = circuit()

        # make sure the mean matches the analytic mean
        expected = (
            88 * np.sin(theta)
            + 24 * np.sin(2 * theta)
            - 40 * np.sin(3 * theta)
            + 5 * np.cos(theta)
            - 6 * np.cos(2 * theta)
            + 27 * np.cos(3 * theta)
            + 6
        ) / 32

        assert np.allclose(res, expected, atol=0.01, rtol=0)
