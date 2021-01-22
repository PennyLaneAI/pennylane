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
# pylint: disable=no-self-use
# pylint: disable=too-many-arguments
# pylint: disable=pointless-statement
from cmath import exp
from math import cos, sin, sqrt

import pytest
import numpy as np
import pennylane as qml

from scipy.linalg import block_diag
from flaky import flaky

pytestmark = [pytest.mark.skip_unsupported, pytest.mark.usefixtures("tape_mode")]

np.random.seed(42)

# ==========================================================
# Some useful global variables

# gates for which device support is tested
ops = {
    "BasisState": qml.BasisState(np.array([0]), wires=[0]),
    "CNOT": qml.CNOT(wires=[0, 1]),
    "CRX": qml.CRX(0, wires=[0, 1]),
    "CRY": qml.CRY(0, wires=[0, 1]),
    "CRZ": qml.CRZ(0, wires=[0, 1]),
    "CRot": qml.CRot(0, 0, 0, wires=[0, 1]),
    "CSWAP": qml.CSWAP(wires=[0, 1, 2]),
    "CZ": qml.CZ(wires=[0, 1]),
    "CY": qml.CY(wires=[0, 1]),
    "DiagonalQubitUnitary": qml.DiagonalQubitUnitary(np.array([1, 1]), wires=[0]),
    "Hadamard": qml.Hadamard(wires=[0]),
    "MultiRZ": qml.MultiRZ(0, wires=[0]),
    "PauliX": qml.PauliX(wires=[0]),
    "PauliY": qml.PauliY(wires=[0]),
    "PauliZ": qml.PauliZ(wires=[0]),
    "PhaseShift": qml.PhaseShift(0, wires=[0]),
    "QubitStateVector": qml.QubitStateVector(np.array([1.0, 0.0]), wires=[0]),
    "QubitUnitary": qml.QubitUnitary(np.eye(2), wires=[0]),
    "RX": qml.RX(0, wires=[0]),
    "RY": qml.RY(0, wires=[0]),
    "RZ": qml.RZ(0, wires=[0]),
    "Rot": qml.Rot(0, 0, 0, wires=[0]),
    "S": qml.S(wires=[0]),
    "SWAP": qml.SWAP(wires=[0, 1]),
    "T": qml.T(wires=[0]),
    "SX": qml.SX(wires=[0]),
    "Toffoli": qml.Toffoli(wires=[0, 1, 2]),
}

all_ops = ops.keys()

# non-parametrized qubit gates
I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CZ = np.diag([1, 1, 1, -1])
CY = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])
toffoli = np.diag([1 for i in range(8)])
toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])
CSWAP = block_diag(I, I, SWAP)

# parametrized qubit gates
phase_shift = lambda phi: np.array([[1, 0], [0, np.exp(1j * phi)]])
rx = lambda theta: cos(theta / 2) * I + 1j * sin(-theta / 2) * X
ry = lambda theta: cos(theta / 2) * I + 1j * sin(-theta / 2) * Y
rz = lambda theta: cos(theta / 2) * I + 1j * sin(-theta / 2) * Z
rot = lambda a, b, c: rz(c) @ (ry(b) @ rz(a))
crz = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.exp(-1j * theta / 2), 0],
        [0, 0, 0, np.exp(1j * theta / 2)],
    ]
)
cry = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, cos(theta / 2), -sin(theta / 2)],
        [0, 0, sin(theta / 2), cos(theta / 2)],
    ]
)
crx = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, cos(theta / 2), 1j * sin(-theta / 2)],
        [0, 0, 1j * sin(-theta / 2), cos(theta / 2)],
    ]
)
crot = lambda phi, theta, omega: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [
            0,
            0,
            exp(-0.5j * (phi + omega)) * cos(theta / 2),
            -exp(0.5j * (phi - omega)) * sin(theta / 2),
        ],
        [
            0,
            0,
            exp(-0.5j * (phi - omega)) * sin(theta / 2),
            exp(0.5j * (phi + omega)) * cos(theta / 2),
        ],
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
    (qml.SX, SX),
]

# list of all parametrized single-qubit gates
# taking a single parameter
single_qubit_param = [
    (qml.PhaseShift, phase_shift),
    (qml.RX, rx),
    (qml.RY, ry),
    (qml.RZ, rz),
]
# list of all non-parametrized two-qubit gates
two_qubit = [(qml.CNOT, CNOT), (qml.SWAP, SWAP), (qml.CZ, CZ), (qml.CY, CY)]
# list of all parametrized two-qubit gates
two_qubit_param = [(qml.CRX, crx), (qml.CRY, cry), (qml.CRZ, crz)]
two_qubit_multi_param = [(qml.CRot, crot)]
# list of all three-qubit gates
three_qubit = [(qml.Toffoli, toffoli), (qml.CSWAP, CSWAP)]

# single qubit unitary matrix
theta = 0.8364
phi = -0.1234
U = np.array(
    [
        [
            np.cos(theta / 2) * np.exp(np.complex(0, -phi / 2)),
            -np.sin(theta / 2) * np.exp(np.complex(0, phi / 2)),
        ],
        [
            np.sin(theta / 2) * np.exp(np.complex(0, -phi / 2)),
            np.cos(theta / 2) * np.exp(np.complex(0, phi / 2)),
        ],
    ]
)

# two qubit unitary matrix
U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / sqrt(3)

# single qubit Hermitian observable
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])


# ===============================================================


class TestSupportedGates:
    """Test that the device can implement all gates that it claims to support."""

    @pytest.mark.parametrize("operation", all_ops)
    def test_supported_gates_can_be_implemented(self, device_kwargs, operation):
        """Test that the device can implement all its supported gates."""
        device_kwargs["wires"] = 3  # maximum size of current gates
        dev = qml.device(**device_kwargs)

        assert hasattr(dev, "operations")
        if operation in dev.operations:

            @qml.qnode(dev)
            def circuit():
                ops[operation]
                return qml.expval(qml.Identity(wires=0))

            assert isinstance(circuit(), (float, np.ndarray))

    @pytest.mark.parametrize("operation", all_ops)
    def test_inverse_gates_can_be_implemented(self, device_kwargs, operation):
        """Test that the device can implement the inverse of all its supported gates.
        This test is skipped for devices that do not support inverse operations."""
        device_kwargs["wires"] = 3
        dev = qml.device(**device_kwargs)
        supports_inv = (
            "supports_inverse_operations" in dev.capabilities()
            and dev.capabilities()["supports_inverse_operations"]
        )
        if not supports_inv:
            pytest.skip("Device does not support inverse operations.")

        assert hasattr(dev, "operations")
        if operation in dev.operations:

            @qml.qnode(dev)
            def circuit():
                ops[operation].queue().inv()
                return qml.expval(qml.Identity(wires=0))

            assert isinstance(circuit(), (float, np.ndarray))


@flaky(max_runs=10)
class TestGatesQubit:
    """Test qubit-based devices' probability vector after application of gates."""

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
        skip_if(dev, {"returns_probs": False})

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
        skip_if(dev, {"returns_probs": False})

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
        skip_if(dev, {"returns_probs": False})

        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            op(wires=range(n_wires))
            return qml.probs(wires=range(n_wires))

        res = circuit()

        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    @pytest.mark.parametrize("gamma", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters(self, device, init_state, op, func, gamma, tol, skip_if):
        """Test single qubit gates taking a single scalar argument."""
        n_wires = 1
        dev = device(n_wires)
        skip_if(dev, {"returns_probs": False})

        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            op(gamma, wires=range(n_wires))
            return qml.probs(wires=range(n_wires))

        res = circuit()

        expected = np.abs(func(gamma) @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    def test_rotation(self, device, init_state, tol, skip_if):
        """Test three axis rotation gate."""
        n_wires = 1
        dev = device(n_wires)
        skip_if(dev, {"returns_probs": False})

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
        skip_if(dev, {"returns_probs": False})

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
        """Test parametrized two qubit gates taking a single scalar argument."""
        n_wires = 2
        dev = device(n_wires)
        skip_if(dev, {"returns_probs": False})

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

        if "QubitUnitary" not in dev.operations:
            pytest.skip("Skipped because device does not support QubitUnitary.")

        skip_if(dev, {"returns_probs": False})

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

        skip_if(dev, {"returns_probs": False})

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
        skip_if(dev, {"supports_inverse_operations": False})
        skip_if(dev, {"returns_probs": False})

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

    @pytest.mark.parametrize("gamma", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters(self, device, init_state, op, func, gamma, tol, skip_if):
        """Test inverse single qubit gates taking one scalar parameter."""
        n_wires = 1
        dev = device(n_wires)
        skip_if(dev, {"supports_inverse_operations": False})
        skip_if(dev, {"returns_probs": False})

        rnd_state = init_state(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            op(gamma, wires=range(n_wires)).inv()
            return qml.probs(wires=range(n_wires))

        res = circuit()

        mat = func(gamma)
        mat = mat.conj().T
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    def test_rotation(self, device, init_state, tol, skip_if):
        """Test inverse three axis rotation gate."""
        n_wires = 1
        dev = device(n_wires)
        skip_if(dev, {"supports_inverse_operations": False})
        skip_if(dev, {"returns_probs": False})

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
        skip_if(dev, {"supports_inverse_operations": False})
        skip_if(dev, {"returns_probs": False})

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

    @pytest.mark.parametrize("gamma", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", two_qubit_param)
    def test_two_qubit_parameters(self, device, init_state, op, func, gamma, tol, skip_if):
        """Test inverse of two qubit gates taking one parameter."""
        n_wires = 2
        dev = device(n_wires)
        skip_if(dev, {"supports_inverse_operations": False})
        skip_if(dev, {"returns_probs": False})

        rnd_state = init_state(2)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(rnd_state, wires=range(n_wires))
            op(gamma, wires=range(n_wires)).inv()
            return qml.probs(wires=range(n_wires))

        res = circuit()

        mat = func(gamma)
        mat = mat.conj().T
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.analytic))

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, device, init_state, mat, tol, skip_if):
        """Test inverse QubitUnitary gate."""
        n_wires = int(np.log2(len(mat)))
        dev = device(n_wires)
        skip_if(dev, {"supports_inverse_operations": False})
        skip_if(dev, {"returns_probs": False})

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
        skip_if(dev, {"supports_inverse_operations": False})
        skip_if(dev, {"returns_probs": False})

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
