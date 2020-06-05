# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests and integration tests for the ``default.qubit.tf`` device.
"""
from itertools import product

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", minversion="2.0")

import pennylane as qml
from pennylane.beta.plugins.default_qubit_tf import DefaultQubitTF
from gate_data import (
    I,
    X,
    Y,
    Z,
    H,
    S,
    T,
    CNOT,
    CZ,
    SWAP,
    CNOT,
    Toffoli,
    CSWAP,
    Rphi,
    Rotx,
    Roty,
    Rotz,
    Rot3,
    CRotx,
    CRoty,
    CRotz,
    CRot3,
)

np.random.seed(42)


#####################################################
# Test matrices
#####################################################

U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)

U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])


#####################################################
# Define standard qubit operations
#####################################################

single_qubit = [(qml.S, S), (qml.T, T), (qml.PauliX, X), (qml.PauliY, Y), (qml.PauliZ, Z), (qml.Hadamard, H)]
single_qubit_param = [(qml.PhaseShift, Rphi), (qml.RX, Rotx), (qml.RY, Roty), (qml.RZ, Rotz)]
two_qubit = [(qml.CZ, CZ), (qml.CNOT, CNOT), (qml.SWAP, SWAP)]
two_qubit_param = [(qml.CRX, CRotx), (qml.CRY, CRoty), (qml.CRZ, CRotz)]
three_qubit = [(qml.Toffoli, Toffoli), (qml.CSWAP, CSWAP)]


#####################################################
# Fixtures
#####################################################


@pytest.fixture
def init_state(scope="session"):
    """Generates a random initial state"""

    def _init_state(n):
        """random initial state"""
        state = np.random.random([2 ** n]) + np.random.random([2 ** n]) * 1j
        state /= np.linalg.norm(state)
        return state

    return _init_state


#####################################################
# Device-level integration tests
#####################################################


class TestApply:
    """Test application of PennyLane operations."""

    def test_basis_state(self, tol):
        """Test basis state initialization"""
        dev = DefaultQubitTF(wires=4)
        state = np.array([0, 0, 1, 0])

        dev.apply([qml.BasisState(state, wires=[0, 1, 2, 3])])

        res = dev.state
        expected = np.zeros([2 ** 4])
        expected[np.ravel_multi_index(state, [2] * 4)] = 1

        assert isinstance(res, tf.Tensor)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_invalid_basis_state_length(self, tol):
        """Test that an exception is raised if the basis state is the wrong size"""
        dev = DefaultQubitTF(wires=4)
        state = np.array([0, 0, 1, 0])

        with pytest.raises(
            ValueError, match=r"BasisState parameter and wires must be of equal length"
        ):
            dev.apply([qml.BasisState(state, wires=[0, 1, 2])])

    def test_invalid_basis_state(self, tol):
        """Test that an exception is raised if the basis state is invalid"""
        dev = DefaultQubitTF(wires=4)
        state = np.array([0, 0, 1, 2])

        with pytest.raises(
            ValueError, match=r"BasisState parameter must consist of 0 or 1 integers"
        ):
            dev.apply([qml.BasisState(state, wires=[0, 1, 2, 3])])

    def test_qubit_state_vector(self, init_state, tol):
        """Test qubit state vector application"""
        dev = DefaultQubitTF(wires=1)
        state = init_state(1)

        dev.apply([qml.QubitStateVector(state, wires=[0])])

        res = dev.state
        expected = state
        assert isinstance(res, tf.Tensor)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_invalid_qubit_state_vector_size(self):
        """Test that an exception is raised if the state
        vector is the wrong size"""
        dev = DefaultQubitTF(wires=2)
        state = np.array([0, 1])

        with pytest.raises(ValueError, match=r"State vector must be of length 2\*\*wires"):
            dev.apply([qml.QubitStateVector(state, wires=[0, 1])])

    def test_invalid_qubit_state_vector_norm(self):
        """Test that an exception is raised if the state
        vector is not normalized"""
        dev = DefaultQubitTF(wires=2)
        state = np.array([0, 12])

        with pytest.raises(ValueError, match=r"Sum of amplitudes-squared does not equal one"):
            dev.apply([qml.QubitStateVector(state, wires=[0])])

    def test_invalid_state_prep(self):
        """Test that an exception is raised if a state preparation is not the
        first operation in the circuit."""
        dev = DefaultQubitTF(wires=2)
        state = np.array([0, 12])

        with pytest.raises(
            qml.DeviceError,
            match=r"cannot be used after other Operations have already been applied",
        ):
            dev.apply([qml.PauliZ(0), qml.QubitStateVector(state, wires=[0])])

    @pytest.mark.parametrize("op,mat", single_qubit)
    def test_single_qubit_no_parameters(self, init_state, op, mat, tol):
        """Test non-parametrized single qubit operations"""
        dev = DefaultQubitTF(wires=1)
        state = init_state(1)

        queue = [qml.QubitStateVector(state, wires=[0])]
        queue += [op(wires=0)]
        dev.apply(queue)

        res = dev.state
        expected = mat @ state
        assert isinstance(res, tf.Tensor)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters(self, init_state, op, func, theta, tol):
        """Test parametrized single qubit operations"""
        dev = DefaultQubitTF(wires=1)
        state = init_state(1)

        queue = [qml.QubitStateVector(state, wires=[0])]
        queue += [op(theta, wires=0)]
        dev.apply(queue)

        res = dev.state
        expected = func(theta) @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_rotation(self, init_state, tol):
        """Test three axis rotation gate"""
        dev = DefaultQubitTF(wires=1)
        state = init_state(1)

        a = 0.542
        b = 1.3432
        c = -0.654

        queue = [qml.QubitStateVector(state, wires=[0])]
        queue += [qml.Rot(a, b, c, wires=0)]
        dev.apply(queue)

        res = dev.state
        expected = Rot3(a, b, c) @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_controlled_rotation(self, init_state, tol):
        """Test three axis controlled-rotation gate"""
        dev = DefaultQubitTF(wires=2)
        state = init_state(2)

        a = 0.542
        b = 1.3432
        c = -0.654

        queue = [qml.QubitStateVector(state, wires=[0, 1])]
        queue += [qml.CRot(a, b, c, wires=[0, 1])]
        dev.apply(queue)

        res = dev.state
        expected = CRot3(a, b, c) @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_inverse_operation(self, init_state, tol):
        """Test that the inverse of an operation is correctly applied"""
        """Test three axis rotation gate"""
        dev = DefaultQubitTF(wires=1)
        state = init_state(1)

        a = 0.542
        b = 1.3432
        c = -0.654

        queue = [qml.QubitStateVector(state, wires=[0])]
        queue += [qml.Rot(a, b, c, wires=0).inv()]
        dev.apply(queue)

        res = dev.state
        expected = np.linalg.inv(Rot3(a, b, c)) @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op,mat", two_qubit)
    def test_two_qubit_no_parameters(self, init_state, op, mat, tol):
        """Test non-parametrized two qubit operations"""
        dev = DefaultQubitTF(wires=2)
        state = init_state(2)

        queue = [qml.QubitStateVector(state, wires=[0, 1])]
        queue += [op(wires=[0, 1])]
        dev.apply(queue)

        res = dev.state
        expected = mat @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, init_state, mat, tol):
        """Test application of arbitrary qubit unitaries"""
        N = int(np.log2(len(mat)))
        dev = DefaultQubitTF(wires=N)
        state = init_state(N)

        queue = [qml.QubitStateVector(state, wires=range(N))]
        queue += [qml.QubitUnitary(mat, wires=range(N))]
        dev.apply(queue)

        res = dev.state
        expected = mat @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op, mat", three_qubit)
    def test_three_qubit_no_parameters(self, init_state, op, mat, tol):
        """Test non-parametrized three qubit operations"""
        dev = DefaultQubitTF(wires=3)
        state = init_state(3)

        queue = [qml.QubitStateVector(state, wires=[0, 1, 2])]
        queue += [op(wires=[0, 1, 2])]
        dev.apply(queue)

        res = dev.state
        expected = mat @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", two_qubit_param)
    def test_two_qubit_parameters(self, init_state, op, func, theta, tol):
        """Test two qubit parametrized operations"""
        dev = DefaultQubitTF(wires=2)
        state = init_state(2)

        queue = [qml.QubitStateVector(state, wires=[0, 1])]
        queue += [op(theta, wires=[0, 1])]
        dev.apply(queue)

        res = dev.state
        expected = func(theta) @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)


THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestExpval:
    """Test expectation values"""

    # test data; each tuple is of the form (GATE, OBSERVABLE, EXPECTED)
    single_wire_expval_test_data = [
        (qml.RX, qml.Identity, lambda t, p: np.array([1, 1])),
        (qml.RX, qml.PauliZ, lambda t, p: np.array([np.cos(t), np.cos(t) * np.cos(p)])),
        (qml.RY, qml.PauliX, lambda t, p: np.array([np.sin(t) * np.sin(p), np.sin(p)])),
        (qml.RX, qml.PauliY, lambda t, p: np.array([0, -np.cos(t) * np.sin(p)])),
        (
            qml.RY,
            qml.Hadamard,
            lambda t, p: np.array(
                [np.sin(t) * np.sin(p) + np.cos(t), np.cos(t) * np.cos(p) + np.sin(p)]
            )
            / np.sqrt(2),
        ),
    ]

    @pytest.mark.parametrize("gate,obs,expected", single_wire_expval_test_data)
    def test_single_wire_expectation(self, gate, obs, expected, theta, phi, varphi, tol):
        """Test that identity expectation value (i.e. the trace) is 1"""
        dev = DefaultQubitTF(wires=2)
        queue = [gate(theta, wires=0), gate(phi, wires=1), qml.CNOT(wires=[0, 1])]
        observables = [obs(wires=[i]) for i in range(2)]

        for i in range(len(observables)):
            observables[i].return_type = qml.operation.Expectation

        res = dev.execute(qml.CircuitGraph(queue + observables, {}))
        assert np.allclose(res, expected(theta, phi), atol=tol, rtol=0)

    def test_hermitian_expectation(self, theta, phi, varphi, tol):
        """Test that arbitrary Hermitian expectation values are correct"""
        dev = DefaultQubitTF(wires=2)
        queue = [qml.RY(theta, wires=0), qml.RY(phi, wires=1), qml.CNOT(wires=[0, 1])]
        observables = [qml.Hermitian(A, wires=[i]) for i in range(2)]

        for i in range(len(observables)):
            observables[i].return_type = qml.operation.Expectation

        res = dev.execute(qml.CircuitGraph(queue + observables, {}))

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        ev1 = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        ev2 = ((a - d) * np.cos(theta) * np.cos(phi) + 2 * re_b * np.sin(phi) + a + d) / 2
        expected = np.array([ev1, ev2])

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multi_mode_hermitian_expectation(self, theta, phi, varphi, tol):
        """Test that arbitrary multi-mode Hermitian expectation values are correct"""
        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        dev = DefaultQubitTF(wires=2)
        queue = [qml.RY(theta, wires=0), qml.RY(phi, wires=1), qml.CNOT(wires=[0, 1])]
        observables = [qml.Hermitian(A, wires=[0, 1])]

        for i in range(len(observables)):
            observables[i].return_type = qml.operation.Expectation

        res = dev.execute(qml.CircuitGraph(queue + observables, {}))

        # below is the analytic expectation value for this circuit with arbitrary
        # Hermitian observable A
        expected = 0.5 * (
            6 * np.cos(theta) * np.sin(phi)
            - np.sin(theta) * (8 * np.sin(phi) + 7 * np.cos(phi) + 3)
            - 2 * np.sin(phi)
            - 6 * np.cos(phi)
            - 6
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_paulix_pauliy(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)
        dev.reset()

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
        )

        res = dev.expval(obs)

        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_identity(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and Identity works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)
        dev.reset()

        obs = qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
        )

        res = dev.expval(obs)

        expected = np.cos(varphi)*np.cos(phi)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.reset()
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
        )

        res = dev.expval(obs)

        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)
        dev.reset()

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.PauliZ(0) @ qml.Hermitian(A, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
        )

        res = dev.expval(obs)

        expected = 0.5 * (
            -6 * np.cos(theta) * (np.cos(varphi) + 1)
            - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
            + 3 * np.cos(varphi) * np.sin(phi)
            + np.sin(phi)
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving two Hermitian matrices works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)

        A1 = np.array([[1, 2],
                       [2, 4]])

        A2 = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.Hermitian(A1, wires=[0]) @ qml.Hermitian(A2, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
        )

        res = dev.expval(obs)

        expected = 0.25 * (
            -30
            + 4 * np.cos(phi) * np.sin(theta)
            + 3 * np.cos(varphi) * (-10 + 4 * np.cos(phi) * np.sin(theta) - 3 * np.sin(phi))
            - 3 * np.sin(phi)
            - 2 * (5 + np.cos(phi) * (6 + 4 * np.sin(theta)) + (-3 + 8 * np.sin(theta)) * np.sin(phi))
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

    def test_hermitian_identity_expectation(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix and the identity works correctly"""
        dev = qml.device("default.qubit.tf", wires=2)

        A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

        obs = qml.Hermitian(A, wires=[0]) @ qml.Identity(wires=[1])

        dev.apply(
            [
                qml.RY(theta, wires=[0]),
                qml.RY(phi, wires=[1]),
                qml.CNOT(wires=[0, 1])
            ],
            obs.diagonalizing_gates()
        )

        res = dev.expval(obs)

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        expected = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_two_wires_identity_expectation(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = qml.device("default.qubit.tf", wires=3, analytic=True)

        A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])
        Identity = np.array([[1, 0],[0, 1]])
        H = np.kron(np.kron(Identity,Identity), A)
        obs = qml.Hermitian(H, wires=[2, 1, 0])

        dev.apply(
            [
                qml.RY(theta, wires=[0]),
                qml.RY(phi, wires=[1]),
                qml.CNOT(wires=[0, 1])
            ],
            obs.diagonalizing_gates()
        )
        res = dev.expval(obs)

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]

        expected = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestVar:
    """Tests for the variance"""

    def test_var(self, theta, phi, varphi, tol):
        """Tests for variance calculation"""
        dev = DefaultQubitTF(wires=1)
        # test correct variance for <Z> of a rotated state

        queue = [qml.RX(phi, wires=0), qml.RY(theta, wires=0)]
        observables = [qml.PauliZ(wires=[0])]

        for i in range(len(observables)):
            observables[i].return_type = qml.operation.Variance

        res = dev.execute(qml.CircuitGraph(queue + observables, {}))
        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_var_hermitian(self, theta, phi, varphi, tol):
        """Tests for variance calculation using an arbitrary Hermitian observable"""
        dev = DefaultQubitTF(wires=2)

        # test correct variance for <H> of a rotated state
        H = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        queue = [qml.RX(phi, wires=0), qml.RY(theta, wires=0)]
        observables = [qml.Hermitian(H, wires=[0])]

        for i in range(len(observables)):
            observables[i].return_type = qml.operation.Variance

        res = dev.execute(qml.CircuitGraph(queue + observables, {}))
        expected = 0.5 * (
            2 * np.sin(2 * theta) * np.cos(phi) ** 2
            + 24 * np.sin(phi) * np.cos(phi) * (np.sin(theta) - np.cos(theta))
            + 35 * np.cos(2 * phi)
            + 39
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_paulix_pauliy(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
        )

        res = dev.var(obs)

        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.reset()
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
        )

        res = dev.var(obs)

        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.PauliZ(0) @ qml.Hermitian(A, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
        )

        res = dev.var(obs)

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
                4
                * np.cos(phi)
                * (
                    4
                    + 8 * np.cos(varphi)
                    + np.cos(2 * varphi)
                    - (1 + 6 * np.cos(varphi)) * np.sin(varphi)
                )
                + np.sin(phi)
                * (
                    15
                    + 8 * np.cos(varphi)
                    - 11 * np.cos(2 * varphi)
                    + 42 * np.sin(varphi)
                    + 3 * np.sin(2 * varphi)
                )
            )
        ) / 16

        assert np.allclose(res, expected, atol=tol, rtol=0)


#####################################################
# QNode-level integration tests
#####################################################


class TestQNodeIntegration:
    """Integration tests for default.qubit.tf. This test ensures it integrates
    properly with the PennyLane UI, in particular the new QNode."""

    def test_load_tensornet_tf_device(self):
        """Test that the tensor network plugin loads correctly"""
        dev = qml.device("default.qubit.tf", wires=2)
        assert dev.num_wires == 2
        assert dev.shots == 1000
        assert dev.analytic
        assert dev.short_name == "default.qubit.tf"
        assert dev.capabilities()["passthru_interface"] == "tf"

    def test_qubit_circuit(self, tol):
        """Test that the tensor network plugin provides correct
        result for a simple circuit using the old QNode."""
        p = tf.Variable(0.543)

        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, interface="tf")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -tf.math.sin(p)

        assert isinstance(circuit, qml.qnodes.PassthruQNode)
        assert np.isclose(circuit(p), expected, atol=tol, rtol=0)

    def test_correct_state(self, tol):
        """Test that the device state is correct after applying a
        quantum function on the device"""

        dev = qml.device("default.qubit.tf", wires=2)

        state = dev.state
        expected = np.array([1, 0, 0, 0])
        assert np.allclose(state, expected, atol=tol, rtol=0)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit()
        state = dev.state

        expected = tf.constant([1.0, 0, 1.0, 0]) / np.sqrt(2)
        assert np.allclose(state, expected, atol=tol, rtol=0)


class TestPassthruIntegration:
    """Tests for integration with the PassthruQNode"""

    def test_jacobian_variable_multiply(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.tf device
        gives the correct result in the case of parameters multiplied by scalars"""
        x = tf.Variable(0.43316321)
        y = tf.Variable(0.2162158)
        z = tf.Variable(0.75110998)

        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(p):
            qml.RX(3 * p[0], wires=0)
            qml.RY(p[1], wires=0)
            qml.RX(p[2] / 2, wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit([x, y, z])

        expected = tf.math.cos(3 * x) * tf.math.cos(y) * tf.math.cos(z / 2) - tf.math.sin(
            3 * x
        ) * tf.math.sin(z / 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tf.concat(tape.jacobian(res, [x, y, z]), axis=0)

        expected = np.array(
            [
                -3
                * (
                    tf.math.sin(3 * x) * tf.math.cos(y) * tf.math.cos(z / 2)
                    + tf.math.cos(3 * x) * tf.math.sin(z / 2)
                ),
                -tf.math.cos(3 * x) * tf.math.sin(y) * tf.math.cos(z / 2),
                -0.5
                * (
                    tf.math.sin(3 * x) * tf.math.cos(z / 2)
                    + tf.math.cos(3 * x) * tf.math.cos(y) * tf.math.sin(z / 2)
                ),
            ]
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_repeated(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.tf device
        gives the correct result in the case of repeated parameters"""
        x = 0.43316321
        y = 0.2162158
        z = 0.75110998
        p = tf.Variable([x, y, z])
        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(p)

        expected = np.cos(y) ** 2 - np.sin(x) * np.sin(y) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, p)

        expected = np.array(
            [-np.cos(x) * np.sin(y) ** 2, -2 * (np.sin(x) + 1) * np.sin(y) * np.cos(y), 0]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_agrees_backprop_parameter_shift(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.tf device
        gives the correct result with respect to the parameter-shift method"""
        p = np.array([0.43316321, 0.2162158, 0.75110998, 0.94714242])

        def circuit(x):
            for i in range(0, len(p), 2):
                qml.RX(x[i], wires=0)
                qml.RY(x[i + 1], wires=1)
            for i in range(2):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))

        dev1 = qml.device("default.qubit.tf", wires=3)
        dev2 = qml.device("default.qubit.tf", wires=3)

        circuit1 = qml.QNode(circuit, dev1, diff_method="backprop", interface="tf")
        circuit2 = qml.QNode(circuit, dev2, diff_method="parameter-shift")

        p_tf = tf.Variable(p)
        with tf.GradientTape() as tape:
            res = circuit1(p_tf)

        assert np.allclose(res, circuit2(p), atol=tol, rtol=0)

        res = tape.jacobian(res, p_tf)
        assert np.allclose(res, circuit2.jacobian([p]), atol=tol, rtol=0)

    def test_state_differentiability(self, tol):
        """Test that the device state can be differentiated"""
        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit(a):
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable(0.54)

        with tf.GradientTape() as tape:
            circuit(a)
            res = tf.abs(dev.state) ** 2
            res = res[1] - res[0]

        grad = tape.gradient(res, a)
        expected = tf.sin(a)
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_prob_differentiability(self, tol):
        """Test that the device probability can be differentiated"""
        dev = qml.device("default.qubit.tf", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = tf.Variable(0.54)
        b = tf.Variable(0.12)

        with tf.GradientTape() as tape:
            # get the probability of wire 1
            prob_wire_1 = circuit(a, b)[0]
            # compute Prob(|1>_1) - Prob(|0>_1)
            res = prob_wire_1[1] - prob_wire_1[0]

        expected = -tf.cos(a) * tf.cos(b)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad = tape.gradient(res, [a, b])
        expected = [tf.sin(a) * tf.cos(b), tf.cos(a) * tf.sin(b)]
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_backprop_gradient(self, tol):
        """Tests that the gradient of the qnode is correct"""
        dev = qml.device("default.qubit.tf", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = -0.234
        b = 0.654

        a_tf = tf.Variable(a, dtype=tf.float64)
        b_tf = tf.Variable(b, dtype=tf.float64)

        with tf.GradientTape() as tape:
            tape.watch([a_tf, b_tf])
            res = circuit(a_tf, b_tf)

        # the analytic result of evaluating circuit(a, b)
        expected_cost = 0.5 * (np.cos(a) * np.cos(b) + np.cos(a) - np.cos(b) + 1)

        # the analytic result of evaluating grad(circuit(a, b))
        expected_grad = np.array(
            [-0.5 * np.sin(a) * (np.cos(b) + 1), 0.5 * np.sin(b) * (1 - np.cos(a))]
        )

        assert np.allclose(res.numpy(), expected_cost, atol=tol, rtol=0)

        res = tape.gradient(res, [a_tf, b_tf])
        assert np.allclose(res, expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("operation", [qml.U3, qml.U3.decomposition])
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "finite-diff"])
    def test_tf_interface_gradient(self, operation, diff_method, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the TensorFlow interface, using a variety of differentiation methods."""
        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(x, weights, w=None):
            """In this example, a mixture of scalar
            arguments, array arguments, and keyword arguments are used."""
            qml.QubitStateVector(1j * np.array([1, -1]) / np.sqrt(2), wires=w)
            operation(x, weights[0], weights[1], wires=w)
            return qml.expval(qml.PauliX(w))

        # Check that the correct QNode type is being used.
        if diff_method == "backprop":
            assert isinstance(circuit, qml.qnodes.PassthruQNode)
            assert not hasattr(circuit, "jacobian")
        else:
            assert not isinstance(circuit, qml.qnodes.PassthruQNode)
            assert hasattr(circuit, "jacobian")

        def cost(params):
            """Perform some classical processing"""
            return circuit(params[0], params[1:], w=0) ** 2

        theta = 0.543
        phi = -0.234
        lam = 0.654

        params = tf.Variable([theta, phi, lam], dtype=tf.float64)

        with tf.GradientTape() as tape:
            tape.watch(params)
            res = cost(params)

        # check that the result is correct
        expected_cost = (np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi)) ** 2
        assert np.allclose(res.numpy(), expected_cost, atol=tol, rtol=0)

        res = tape.gradient(res, params)

        # check that the gradient is correct
        expected_grad = (
            np.array(
                [
                    np.sin(theta) * np.cos(lam) * np.cos(phi),
                    np.cos(theta) * np.cos(lam) * np.sin(phi) + np.sin(lam) * np.cos(phi),
                    np.cos(theta) * np.sin(lam) * np.cos(phi) + np.cos(lam) * np.sin(phi),
                ]
            )
            * 2
            * (np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi))
        )
        assert np.allclose(res.numpy(), expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["autograd", "torch"])
    def test_error_backprop_wrong_interface(self, interface, tol):
        """Tests that an error is raised if diff_method='backprop' but not using
        the TF interface"""
        dev = qml.device("default.qubit.tf", wires=1)

        def circuit(x, w=None):
            qml.RZ(x, wires=w)
            return qml.expval(qml.PauliX(w))

        with pytest.raises(
            ValueError,
            match="default.qubit.tf only supports diff_method='backprop' when using the tf interface",
        ):
            qml.qnode(dev, diff_method="backprop", interface=interface)(circuit)


class TestSamplesNonAnalytic:
    """Tests for sampling and non-analytic mode"""

    def test_sample_observables(self):
        """Test that the device allows for sampling from observables."""
        shots = 100
        dev = qml.device("default.qubit.tf", wires=2, shots=shots)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(0))

        a = tf.Variable(0.54)
        res = circuit(a)

        assert isinstance(res, tf.Tensor)
        assert res.shape == (1, shots)
        assert set(res[0].numpy()) == {-1, 1}

    def test_sample_observables_non_differentiable(self):
        """Test that sampled observables cannot be differentiated."""
        shots = 100
        dev = qml.device("default.qubit.tf", wires=2, shots=shots)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(0))

        a = tf.Variable(0.54)

        with tf.GradientTape() as tape:
            res = circuit(a)

        assert tape.gradient(res, a) is None

    def test_estimating_marginal_probability(self, tol):
        """Test that the probability of a subset of wires is accurately estimated."""
        dev = qml.device("default.qubit.tf", wires=2, analytic=False, shots=1000)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit():
            qml.PauliX(0)
            return qml.probs(wires=[0])

        res = circuit()

        assert isinstance(res, tf.Tensor)

        expected = np.array([0, 1])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_estimating_full_probability(self, tol):
        """Test that the probability of a subset of wires is accurately estimated."""
        dev = qml.device("default.qubit.tf", wires=2, analytic=False, shots=1000)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit():
            qml.PauliX(0)
            qml.PauliX(1)
            return qml.probs(wires=[0, 1])

        res = circuit()

        assert isinstance(res, tf.Tensor)

        expected = np.array([0, 0, 0, 1])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_estimating_expectation_values(self, tol):
        """Test that estimating expectation values using a finite number
        of shots produces a numeric tensor"""
        dev = qml.device("default.qubit.tf", wires=3, analytic=False, shots=1000)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit(a, b):
            qml.RX(a, wires=[0])
            qml.RX(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        a = tf.Variable(0.543)
        b = tf.Variable(0.43)

        res = circuit(a, b)
        assert isinstance(res, tf.Tensor)

        # We don't check the expected value due to stochasticity, but
        # leave it here for completeness.
        # expected = [tf.cos(a), tf.cos(a) * tf.cos(b)]
        # assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_estimating_expectation_values_not_differentiable(self, tol):
        """Test that analytic=False results in non-differentiable QNodes"""
        dev = qml.device("default.qubit.tf", wires=3, analytic=False)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit(a, b):
            qml.RX(a, wires=[0])
            qml.RX(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        a = tf.Variable(0.543)
        b = tf.Variable(0.43)

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        assert isinstance(res, tf.Tensor)
        grad = tape.gradient(res, [a, b])
        assert grad == [None, None]


class TestHighLevelIntegration:
    """Tests for integration with higher level components of PennyLane."""

    def test_qnode_collection_integration(self):
        """Test that a PassthruQNode default.qubit.tf works with QNodeCollections."""
        dev = qml.device("default.qubit.tf", wires=2)

        obs_list = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1)]
        qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev, interface="tf")

        assert qnodes.interface == "tf"

        weights = tf.Variable(qml.init.strong_ent_layers_normal(n_wires=2, n_layers=2))

        @tf.function
        def cost(weights):
            return tf.reduce_sum(qnodes(weights))

        with tf.GradientTape() as tape:
            res = qnodes(weights)

        grad = tape.gradient(res, weights)

        assert isinstance(grad, tf.Tensor)
        assert grad.shape == weights.shape
