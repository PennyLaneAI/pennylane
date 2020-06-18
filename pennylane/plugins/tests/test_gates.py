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
"""Tests that application of operations works correctly in the plugin devices"""
import pytest

import numpy as np
import pennylane as qml
from scipy.linalg import block_diag

from conftest import U, U2, A

np.random.seed(42)


# ==========================================================
# Some useful global variables

# non-parametrized qubit gates
I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CZ = np.diag([1, 1, 1, -1])
toffoli = np.diag([1 for i in range(8)])
toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])
CSWAP = block_diag(I, I, SWAP)

# parametrized qubit gates
phase_shift = lambda phi: np.array([[1, 0], [0, np.exp(1j * phi)]])
rx = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * X
ry = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Y
rz = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Z
rot = lambda a, b, c: rz(c) @ (ry(b) @ rz(a))
crz = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.exp(-1j * theta / 2), 0],
        [0, 0, 0, np.exp(1j * theta / 2)],
    ]
)

# list of all non-parametrized single-qubit gates,
# along with the PennyLane operation name
single_qubit = [
    (qml.PauliX, X),
    (qml.PauliY, Y),
    (qml.PauliZ, Z),
    (qml.Hadamard, H),
    (qml.S, S),
    (qml.S.inv(), S.conj().T),
    (qml.T, T),
    (qml.T.inv(), T.conj().T),
]

# list of all parametrized single-qubit gates
single_qubit_param = [
  # (qml.PhaseShift(0, wires=0), phase_shift),
    (qml.RX, rx),
    (qml.RY, ry),
    (qml.RZ, rz)]
# list of all non-parametrized two-qubit gates
two_qubit = [
    (qml.CNOT, CNOT),
    (qml.SWAP, SWAP),
    (qml.CZ, CZ)
]
# list of all parametrized two-qubit gates
two_qubit_param = [
    (qml.CRZ, crz)
]
# list of all three-qubit gates
three_qubit = [
    (qml.Toffoli, toffoli),
    (qml.CSWAP, CSWAP)
    ]

# single qubit unitary matrix
U = np.array([[0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
              [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j]])

# two qubit unitary matrix
U2 = np.array([[0, 1, 1, 1],
               [1, 0, 1, -1],
               [1, -1, 0, 1],
               [1, 1, -1, 0]]) / np.sqrt(3)

# single qubit Hermitian observable
A = np.array([[1.02789352, 1.61296440 - 0.3498192j],
              [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

# ===============================================================

class TestGatesAnalytic:
    """Test the device's probability vector after application of gates.
    These tests will only run for devices where the state can be accessed,
    which are devices with the attribute ``analytic=True``."""

    @pytest.mark.parametrize("basis_state", [np.array([0, 0, 1, 0]),
                                       np.array([0, 0, 1, 0]),
                                       np.array([1, 0, 1, 0]),
                                       np.array([1, 1, 1, 1])]
                             )
    def test_basis_state(self, device_name, basis_state, tol, skip_if_not_analytic):
        """Test basis state initialization."""
        dev = qml.device(device_name, wires=4)
        skip_if_not_analytic(dev.analytic)

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(basis_state, wires=range(4))
            return qml.expval(qml.Identity(wires=0))

        circuit()
        state_vec = dev.state
        res = np.abs(state_vec) ** 2

        expected = np.zeros([2 ** 4])
        expected[np.ravel_multi_index(basis_state, [2] * 4)] = 1
        assert np.allclose(res, expected, tol)

    def test_qubit_state_vector(self, device_name, init_state, tol, skip_if_not_analytic):
        """Test QubitStateVector initialisation."""
        dev = qml.device(device_name, wires=1)
        skip_if_not_analytic(dev.analytic)

        rnd_state = init_state(1)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=[0])
            return qml.expval(qml.Identity(wires=0))

        circuit()
        state_vec = dev.state
        res = np.abs(state_vec) ** 2

        expected = np.abs(rnd_state) ** 2
        assert np.allclose(res, expected, tol)

    def test_invalid_qubit_state_vector(self, device_name, skip_if_not_analytic):
        """Test that an exception is raised if the state
        vector is the wrong size."""
        dev = qml.device(device_name, wires=2)
        skip_if_not_analytic(dev.analytic)

        invalid_state = np.array([0, 123.432])

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(invalid_state, wires=[0])
            return qml.expval(qml.Identity(wires=0))

        with pytest.raises(ValueError, match=r"State vector must be of length 2\*\*wires"):
            circuit()

    @pytest.mark.parametrize("op,mat", single_qubit)
    def test_single_qubit_no_parameters(self, device_name, init_state, op, mat, tol, skip_if_not_analytic):
        """Test PauliX application."""
        dev = qml.device(device_name, wires=1)
        skip_if_not_analytic(dev.analytic)

        rnd_state = init_state(1)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=[0])
            op(wires=[0])
            return qml.expval(qml.Identity(wires=0))

        circuit()
        state_vec = dev.state
        res = np.abs(state_vec) ** 2

        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters(self, device_name, init_state, op, func, theta, tol, skip_if_not_analytic):
        """Test single qubit gates."""
        dev = qml.device(device_name, wires=1)
        skip_if_not_analytic(dev.analytic)

        rnd_state = init_state(1)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=[0])
            op(theta, wires=[0])
            return qml.expval(qml.Identity(wires=0))

        circuit()
        state_vec = dev.state
        res = np.abs(state_vec) ** 2

        expected = np.abs(func(theta) @ rnd_state) ** 2
        assert np.allclose(res, expected, tol)

    def test_rotation(self, device_name, init_state, tol, skip_if_not_analytic):
        """Test three axis rotation gate."""
        dev = qml.device(device_name, wires=1)
        skip_if_not_analytic(dev.analytic)

        rnd_state = init_state(1)
        a = 0.542
        b = 1.3432
        c = -0.654

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=[0])
            qml.Rot(a, b, c, wires=0)
            return qml.expval(qml.Identity(wires=0))

        circuit()
        state_vec = dev.state
        res = np.abs(state_vec) ** 2

        expected = np.abs(rot(a, b, c) @ rnd_state) ** 2
        assert np.allclose(res, expected, tol)

    @pytest.mark.parametrize("op,mat", two_qubit)
    def test_two_qubit_no_parameters(self, device_name, init_state, op, mat, tol, skip_if_not_analytic):
        """Test two qubit gates."""
        dev = qml.device(device_name, wires=2)
        skip_if_not_analytic(dev.analytic)

        rnd_state = init_state(2)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=[0])
            op(wires=[0, 1])
            return qml.expval(qml.Identity(wires=0))

        circuit()
        state_vec = dev.state
        res = np.abs(state_vec) ** 2

        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", two_qubit_param)
    def test_two_qubit_parameters(self, device_name, init_state, op, func, theta, tol, skip_if_not_analytic):
        """Test parametrized two qubit gates."""
        dev = qml.device(device_name, wires=2)
        skip_if_not_analytic(dev.analytic)

        rnd_state = init_state(2)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=[0])
            op(theta, wires=[0, 1])
            return qml.expval(qml.Identity(wires=0))

        circuit()
        state_vec = dev.state
        res = np.abs(state_vec) ** 2

        expected = np.abs(func(theta) @ rnd_state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, device_name, init_state, mat, tol, skip_if_not_analytic):
        """Test QubitUnitary gate."""
        N = int(np.log2(len(mat)))
        dev = qml.device(device_name, wires=N)
        skip_if_not_analytic(dev.analytic)

        rnd_state = init_state(N)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=[0])
            qml.QubitUnitary(mat, wires=list(range(N)))
            return qml.expval(qml.Identity(wires=0))

        circuit()
        state_vec = dev.state
        res = np.abs(state_vec) ** 2

        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, tol)

    def test_invalid_qubit_unitary(self, device_name, init_state, skip_if_not_analytic):
        """Test that an exception is raised if the
        unitary matrix is the wrong size."""
        dev = qml.device(device_name, wires=2)
        skip_if_not_analytic(dev.analytic)

        rnd_state = init_state(2)
        invalid_unitary = np.array([[1, 0], [0, 1]])

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=[0])
            qml.QubitUnitary(invalid_unitary, wires=[0, 1])
            return qml.expval(qml.Identity(wires=0))

        with pytest.raises(ValueError, match=r"Unitary matrix must be of shape"):
            circuit()

    @pytest.mark.parametrize("op, mat", three_qubit)
    def test_three_qubit_no_parameters(self, device_name, init_state, op, mat, tol, skip_if_not_analytic):
        """Test three qubit gates without parameters."""
        dev = qml.device(device_name, wires=3)
        skip_if_not_analytic(dev.analytic)

        rnd_state = init_state(3)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=[0])
            op(wires=[0, 1, 2])
            return qml.expval(qml.Identity(wires=0))

        circuit()
        state_vec = dev.state
        res = np.abs(state_vec) ** 2

        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, tol)
