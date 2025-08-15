# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for edge cases of the Hermitian class.
"""
import itertools

import numpy as np
import pytest

import pennylane as qml

# pylint:disable=too-many-arguments

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)


@pytest.mark.parametrize("shots", [None, 1000000])
class TestEdgeHermitian:
    """Test Hermitian edge cases."""

    @pytest.mark.parametrize("theta,phi", list(zip(THETA, PHI)))
    def test_hermitian_two_wires_identity_expectation_only_hermitian(self, shots, theta, phi):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = qml.device("default.qubit", wires=3)

        A = np.array(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
        )

        Identity = np.array([[1, 0], [0, 1]])
        obs = np.kron(np.kron(Identity, Identity), A)

        @qml.set_shots(shots)
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

    @pytest.mark.parametrize("theta,phi", list(zip(THETA, PHI)))
    def test_hermitian_two_wires_identity_expectation_with_tensor(self, shots, theta, phi):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = qml.device("default.qubit", wires=3)

        A = np.array(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
        )

        Identity = np.array([[1, 0], [0, 1]])
        obs = np.kron(Identity, A)

        @qml.set_shots(shots)
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

    @pytest.mark.parametrize("theta", THETA)
    @pytest.mark.parametrize("w1, w2", list(itertools.permutations(range(4), 2)))
    def test_hermitian_two_wires_permuted(self, w1, w2, shots, theta, seed):
        """Test that an hermitian expectation with various wires permuted works"""
        dev = qml.device("default.qubit", wires=4, seed=seed)
        theta = 0.543

        A = np.array(
            [
                [1, 2j, 1 - 2j, 0.5j],
                [-2j, 0, 3 + 4j, 1],
                [1 + 2j, 3 - 4j, 0.75, 1.5 - 2j],
                [-0.5j, 1, 1.5 + 2j, -1],
            ]
        )

        @qml.set_shots(shots)
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
