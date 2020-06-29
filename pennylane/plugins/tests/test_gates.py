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
Tests that application of gates and state preparations
works correctly an a device.
"""
import pytest

import numpy as np
import pennylane as qml
from scipy.linalg import block_diag
from flaky import flaky

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
cry = lambda theta: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, math.cos(theta / 2), -math.sin(theta / 2)], [0, 0, math.sin(theta / 2), math.cos(theta / 2)]])
crx = lambda theta: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, math.cos(theta / 2), 1j * math.sin(-theta / 2)], [0, 0, 1j * math.sin(-theta / 2), math.cos(theta / 2)]])
crot = lambda phi, theta, omega: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, cmath.exp(-0.5j * (phi + omega)) * math.cos(theta / 2), -cmath.exp(0.5j * (phi - omega)) * math.sin(theta / 2)],
        [0, 0, cmath.exp(-0.5j * (phi - omega)) * math.sin(theta / 2), cmath.exp(0.5j * (phi + omega)) * math.cos(theta / 2)],
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
    (qml.T, T),
]

# list of all parametrized single-qubit gates
single_qubit_param = [
    # (qml.PhaseShift(0, wires=0), phase_shift),
    (qml.RX, rx),
    (qml.RY, ry),
    (qml.RZ, rz),
]
# list of all non-parametrized two-qubit gates
two_qubit = [(qml.CNOT, CNOT), (qml.SWAP, SWAP), (qml.CZ, CZ)]
# list of all parametrized two-qubit gates
two_qubit_param = [(qml.CRX, crx), (qml.CRY, cry), (qml.CRZ, crz)]
two_qubit_multi_param = [(qml.CRot, crot)]
# list of all three-qubit gates
three_qubit = [(qml.Toffoli, toffoli), (qml.CSWAP, CSWAP)]

# single qubit unitary matrix
U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)

# two qubit unitary matrix
U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)

# single qubit Hermitian observable
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])


# ===============================================================


@flaky(max_runs=10)
class TestGatesQubit:
    """Test qubit-based devices' probability vector after application of gates.
    """

    @pytest.mark.parametrize(
        "basis_state",
        [
            np.array([0, 0, 1, 0]),
            np.array([0, 0, 1, 0]),
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 1, 1]),
        ],
    )
    def test_basis_state(self, device, basis_state, tol, skip_if):
        """Test basis state initialization."""
        n_wires = 4
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(basis_state, wires=range(n_wires))
            return qml.probs(wires=range(n_wires))

        res = circuit()

        expected = np.zeros([2 ** n_wires])
        expected[np.ravel_multi_index(basis_state, [2] * n_wires)] = 1
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    def test_qubit_state_vector(self, device, init_state, tol, skip_if):
        """Test QubitStateVector initialisation."""
        n_wires = 1
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            return qml.probs(range(n_wires))

        res = circuit()
        expected = np.abs(rnd_state) ** 2

        assert np.allclose(res, expected, atol=tol(dev.analytic))

    @pytest.mark.parametrize("op,mat", single_qubit)
    def test_single_qubit_no_parameters(self, device, init_state, op, mat, tol, skip_if):
        """Test PauliX application."""
        n_wires = 1
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            op(wires=range(n_wires))
            return qml.probs(wires=range(n_wires))

        res = circuit()

        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters(self, device, init_state, op, func, theta, tol, skip_if):
        """Test single qubit gates taking a single scalar argument."""
        n_wires = 1
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            op(theta, wires=range(n_wires))
            return qml.probs(wires=range(n_wires))

        res = circuit()

        expected = np.abs(func(theta) @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    def test_rotation(self, device, init_state, tol, skip_if):
        """Test three axis rotation gate."""
        n_wires = 1
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        rnd_state = init_state(n_wires)

        rnd_state = init_state(n_wires)
        a = 0.542
        b = 1.3432
        c = -0.654

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            qml.Rot(a, b, c, wires=range(n_wires))
            return qml.probs(wires=range(n_wires))

        res = circuit()

        expected = np.abs(rot(a, b, c) @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    @pytest.mark.parametrize("op,mat", two_qubit)
    def test_two_qubit_no_parameters(self, device, init_state, op, mat, tol, skip_if):
        """Test two qubit gates."""
        n_wires = 2
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        rnd_state = init_state(n_wires)

        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            op(wires=range(n_wires))
            return qml.probs(wires=range(n_wires))

        res = circuit()

        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", two_qubit_param)
    def test_two_qubit_parameters(self, device, init_state, op, func, theta, tol, skip_if):
        """Test parametrized two qubit gates."""
        n_wires = 2
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        rnd_state = init_state(n_wires)

        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            op(theta, wires=range(n_wires))
            return qml.probs(wires=range(n_wires))

        res = circuit()

        expected = np.abs(func(theta) @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, device, init_state, mat, tol, skip_if):
        """Test QubitUnitary gate."""
        n_wires = int(np.log2(len(mat)))
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            qml.QubitUnitary(mat, wires=list(range(n_wires)))
            return qml.probs(wires=range(n_wires))

        res = circuit()

        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    @pytest.mark.parametrize("op, mat", three_qubit)
    def test_three_qubit_no_parameters(self, device, init_state, op, mat, tol, skip_if):
        """Test three qubit gates without parameters."""
        n_wires = 3
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            op(wires=[0, 1, 2])
            return qml.probs(wires=range(n_wires))

        res = circuit()

        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))


@flaky(max_runs=10)
class TestInverseGatesQubit:
    """Test the device's probability vector after application of inverse of gates."""

    @pytest.mark.parametrize("op,mat", single_qubit)
    def test_single_qubit_no_parameters(self, device, init_state, op, mat, tol, skip_if):
        """Test inverse single qubit gate application."""
        n_wires = 1
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")
        skip_if("inverse_operations" not in capabilities)
        skip_if("inverse_operations" in capabilities and not capabilities["inverse_operations"])

        rnd_state = init_state(1)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            op(wires=range(n_wires)).inv()
            return qml.probs(wires=range(n_wires))

        res = circuit()

        mat = mat.conj().T
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters(self, device, init_state, op, func, theta, tol, skip_if):
        """Test inverse single qubit gates taking one scalar parameter."""
        n_wires = 1
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("inverse_operations" not in capabilities)
        skip_if("inverse_operations" in capabilities and not capabilities["inverse_operations"])
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            op(theta, wires=range(n_wires)).inv()
            return qml.probs(wires=range(n_wires))

        res = circuit()

        mat = func(theta)
        mat = mat.conj().T
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    def test_rotation(self, device, init_state, tol, skip_if):
        """Test inverse three axis rotation gate."""
        n_wires = 1
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("inverse_operations" not in capabilities)
        skip_if("inverse_operations" in capabilities and not capabilities["inverse_operations"])
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        rnd_state = init_state(1)
        a = 0.542
        b = 1.3432
        c = -0.654

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            qml.Rot(a, b, c, wires=range(n_wires)).inv()
            return qml.probs(wires=range(n_wires))

        res = circuit()

        mat = rot(a, b, c)
        mat = mat.conj().T
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    @pytest.mark.parametrize("op,mat", two_qubit)
    def test_two_qubit_no_parameters(self, device, init_state, op, mat, tol, skip_if):
        """Test inverse two qubit gates."""
        n_wires = 2
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("inverse_operations" not in capabilities)
        skip_if("inverse_operations" in capabilities and not capabilities["inverse_operations"])
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            op(wires=range(n_wires)).inv()
            return qml.probs(wires=range(n_wires))

        res = circuit()

        mat = mat.conj().T
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", two_qubit_param)
    def test_two_qubit_parameters(self, device, init_state, op, func, theta, tol, skip_if):
        """Test inverse parametrized two qubit gates."""
        n_wires = 2
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("inverse_operations" not in capabilities)
        skip_if("inverse_operations" in capabilities and not capabilities["inverse_operations"])
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        rnd_state = init_state(2)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            op(theta, wires=range(n_wires)).inv()
            return qml.probs(wires=range(n_wires))

        res = circuit()

        mat = func(theta)
        mat = mat.conj().T
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, device, init_state, mat, tol, skip_if):
        """Test inverse QubitUnitary gate."""
        n_wires = int(np.log2(len(mat)))
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("inverse_operations" not in capabilities)
        skip_if("inverse_operations" in capabilities and not capabilities["inverse_operations"])
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            qml.QubitUnitary(mat, wires=list(range(n_wires))).inv()
            return qml.probs(wires=range(n_wires))

        res = circuit()

        mat = mat.conj().T
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    @pytest.mark.parametrize("op, mat", three_qubit)
    def test_three_qubit_no_parameters(self, device, init_state, op, mat, tol, skip_if):
        """Test inverse three qubit gates without parameters."""
        n_wires = 3
        dev = device(n_wires)
        capabilities = dev.__class__.capabilities()
        skip_if("inverse_operations" not in capabilities)
        skip_if("inverse_operations" in capabilities and not capabilities["inverse_operations"])
        skip_if("model" not in capabilities or not capabilities["model"] == "qubit")

        rnd_state = init_state(3)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            op(wires=range(n_wires)).inv()
            return qml.probs(wires=range(n_wires))

        res = circuit()

        mat = mat.conj().T
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))
