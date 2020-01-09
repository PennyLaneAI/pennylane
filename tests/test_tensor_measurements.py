# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Integration tests to ensure that tensor observables return the correct result.
"""
import pytest

import numpy as np
import pennylane as qml
from pennylane import expval, var, sample


Z = np.array([[1, 0], [0, -1]])
THETA = np.linspace(0.11, 3, 5)
PHI = np.linspace(0.32, 3, 5)
VARPHI = np.linspace(0.02, 3, 5)


def ansatz(a, b, c):
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(c, wires=2)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])


@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorExpval:
    """Test tensor expectation values"""

    def test_tensor_product(self, theta, phi, varphi, tol):
        """Test that a tensor product ZxZ gives the same result as simply
        using an Hermitian matrix"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit1(a, b, c):
            ansatz(a, b, c)
            return expval(qml.PauliZ(0) @ qml.PauliZ(2))

        @qml.qnode(dev)
        def circuit2(a, b, c):
            ansatz(a, b, c)
            return expval(qml.Hermitian(np.kron(Z, Z), wires=[0, 2]))

        res1 = circuit1(theta, phi, varphi)
        res2 = circuit2(theta, phi, varphi)

        assert np.allclose(res1, res2, atol=tol, rtol=0)

    def test_combine_tensor_with_non_tensor(self, theta, phi, varphi, tol):
        """Test that a tensor product along with a non-tensor product
        continues to function correctly"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit1(a, b, c):
            ansatz(a, b, c)
            return expval(qml.PauliZ(0) @ qml.PauliZ(2)), expval(qml.PauliZ(1))

        @qml.qnode(dev)
        def circuit2(a, b, c):
            ansatz(a, b, c)
            return expval(qml.Hermitian(np.kron(Z, Z), wires=[0, 2]))

        @qml.qnode(dev)
        def circuit3(a, b, c):
            ansatz(a, b, c)
            return expval(qml.PauliZ(1))

        res1 = circuit1(theta, phi, varphi)
        res2 = circuit2(theta, phi, varphi), circuit3(theta, phi, varphi)

        assert np.allclose(res1, res2, atol=tol, rtol=0)

    def test_paulix_tensor_pauliy(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return expval(qml.PauliX(0) @ qml.PauliY(2))

        res = circuit(theta, phi, varphi)
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_paulix_tensor_pauliy_gradient(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return expval(qml.PauliX(0) @ qml.PauliY(2))

        dcircuit = qml.grad(circuit, 0)
        res = dcircuit(theta, phi, varphi)
        expected = np.cos(theta) * np.sin(phi) * np.sin(varphi)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_tensor_identity(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and Identity works correctly"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return expval(qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2))

        res = circuit(theta, phi, varphi)
        expected = np.cos(varphi) * np.cos(phi)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_tensor_hadamard(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and hadamard works correctly"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return expval(qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2))

        res = circuit(theta, phi, varphi)
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix works correctly"""
        dev = qml.device("default.qubit", wires=3)

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return expval(qml.PauliZ(0) @ qml.Hermitian(A, [1, 2]))

        res = circuit(theta, phi, varphi)
        expected = 0.5 * (
            -6 * np.cos(theta) * (np.cos(varphi) + 1)
            - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
            + 3 * np.cos(varphi) * np.sin(phi)
            + np.sin(phi)
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_tensor_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving two Hermitian matrices works correctly"""
        dev = qml.device("default.qubit", wires=3)

        A1 = np.array([[1, 2], [2, 4]])

        A2 = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return expval(qml.Hermitian(A1, 0) @ qml.Hermitian(A2, [1, 2]))

        res = circuit(theta, phi, varphi)
        expected = 0.25 * (
            -30
            + 4 * np.cos(phi) * np.sin(theta)
            + 3 * np.cos(varphi) * (-10 + 4 * np.cos(phi) * np.sin(theta) - 3 * np.sin(phi))
            - 3 * np.sin(phi)
            - 2
            * (5 + np.cos(phi) * (6 + 4 * np.sin(theta)) + (-3 + 8 * np.sin(theta)) * np.sin(phi))
            * np.sin(varphi)
            + np.cos(theta)
            * (
                18
                + 5 * np.sin(phi)
                + 3 * np.cos(varphi) * (6 + 5 * np.sin(phi))
                + 2 * (3 + 10 * np.cos(phi) - 5 * np.sin(phi)) * np.sin(varphi)
            )
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_tensor_identity_expectation(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix and the identity works correctly"""
        dev = qml.device("default.qubit", wires=2)

        A = np.array(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
        )

        @qml.qnode(dev)
        def circuit(a, b, c):
            qml.RY(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return expval(qml.Hermitian(A, 0) @ qml.Identity(1))

        res = circuit(theta, phi, varphi)

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        expected = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2

        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorVar:
    """Tests for variance of tensor observables"""

    def test_paulix_tensor_pauliy(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return var(qml.PauliX(0) @ qml.PauliY(2))

        res = circuit(theta, phi, varphi)
        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_tensor_hadamard(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and hadamard works correctly"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return var(qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2))

        res = circuit(theta, phi, varphi)
        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_tensor_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.qubit", wires=3)

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return var(qml.PauliZ(0) @ qml.Hermitian(A, [1, 2]))

        res = circuit(theta, phi, varphi)
        expected = (
            1057
            - np.cos(2 * phi)
            + 12 * (27 + np.cos(2 * phi)) * np.cos(varphi)
            - 2 * np.cos(2 * varphi) * np.sin(phi) * (16 * np.cos(phi) + 21 * np.sin(phi))
            + 16 * np.sin(2 * phi)
            - 8 * (-17 + np.cos(2 * phi) + 2 * np.sin(2 * phi)) * np.sin(varphi)
            - 8 * np.cos(2 * theta) * (3 + 3 * np.cos(varphi) + np.sin(varphi)) ** 2
            - 24 * np.cos(phi) * (np.cos(phi) + 2 * np.sin(phi)) * np.sin(2 * varphi)
            - 8
            * np.cos(theta)
            * (
                4 * np.cos(phi)
                * (4 + 8 * np.cos(varphi) + np.cos(2 * varphi) - (1 + 6 * np.cos(varphi)) * np.sin(varphi))
                + np.sin(phi) * (15 + 8 * np.cos(varphi) - 11 * np.cos(2 * varphi) + 42 * np.sin(varphi) + 3 * np.sin(2 * varphi))
            )
        ) / 16

        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorSample:
    """Tests for samples of tensor observables"""

    def test_paulix_tensor_pauliy(self, theta, phi, varphi, monkeypatch, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return sample(qml.PauliX(0) @ qml.PauliY(2))

        with monkeypatch.context() as m:
            m.setattr("numpy.random.choice", lambda x, y, p: (x, p))
            s1, p = circuit(theta, phi, varphi)

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, atol=tol, rtol=0)

        mean = s1 @ p
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        assert np.allclose(mean, expected, atol=tol, rtol=0)

        var = (s1 ** 2) @ p - (s1 @ p).real ** 2
        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16
        assert np.allclose(var, expected, atol=tol, rtol=0)

    def test_pauliz_tensor_hadamard(self, theta, phi, varphi, monkeypatch, tol):
        """Test that a tensor product involving PauliZ and hadamard works correctly"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return sample(qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2))

        with monkeypatch.context() as m:
            m.setattr("numpy.random.choice", lambda x, y, p: (x, p))
            s1, p = circuit(theta, phi, varphi)

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, atol=tol, rtol=0)

        mean = s1 @ p
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)
        assert np.allclose(mean, expected, atol=tol, rtol=0)

        var = (s1 ** 2) @ p - (s1 @ p).real ** 2
        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4
        assert np.allclose(var, expected, atol=tol, rtol=0)

    def test_tensor_hermitian(self, theta, phi, varphi, monkeypatch, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.qubit", wires=3)

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return sample(qml.PauliZ(0) @ qml.Hermitian(A, [1, 2]))

        with monkeypatch.context() as m:
            m.setattr("numpy.random.choice", lambda x, y, p: (x, p))
            s1, p = circuit(theta, phi, varphi)

        # s1 should only contain the eigenvalues of
        # the hermitian matrix tensor product Z
        Z = np.diag([1, -1])
        eigvals = np.linalg.eigvalsh(np.kron(Z, A))
        assert set(np.round(s1, 8)).issubset(set(np.round(eigvals, 8)))

        mean = s1 @ p
        expected = 0.5 * (
            -6 * np.cos(theta) * (np.cos(varphi) + 1)
            - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
            + 3 * np.cos(varphi) * np.sin(phi)
            + np.sin(phi)
        )
        assert np.allclose(mean, expected, atol=tol, rtol=0)

        var = (s1 ** 2) @ p - (s1 @ p).real ** 2
        expected = (
            1057
            - np.cos(2 * phi)
            + 12 * (27 + np.cos(2 * phi)) * np.cos(varphi)
            - 2 * np.cos(2 * varphi) * np.sin(phi) * (16 * np.cos(phi) + 21 * np.sin(phi))
            + 16 * np.sin(2 * phi)
            - 8 * (-17 + np.cos(2 * phi) + 2 * np.sin(2 * phi)) * np.sin(varphi)
            - 8 * np.cos(2 * theta) * (3 + 3 * np.cos(varphi) + np.sin(varphi)) ** 2
            - 24 * np.cos(phi) * (np.cos(phi) + 2 * np.sin(phi)) * np.sin(2 * varphi)
            - 8
            * np.cos(theta)
            * (
                4 * np.cos(phi)
                * (4 + 8 * np.cos(varphi) + np.cos(2 * varphi) - (1 + 6 * np.cos(varphi)) * np.sin(varphi))
                + np.sin(phi) * (15 + 8 * np.cos(varphi) - 11 * np.cos(2 * varphi) + 42 * np.sin(varphi) + 3 * np.sin(2 * varphi))
            )
        ) / 16

        assert np.allclose(var, expected, atol=tol, rtol=0)
