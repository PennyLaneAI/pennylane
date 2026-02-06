# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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

# pylint: disable=unnecessary-lambda-assignment

from cmath import exp
from math import cos, sin, sqrt

import numpy as np
import pytest
from flaky import flaky
from scipy.linalg import block_diag

import pennylane as qp
from pennylane.exceptions import DeviceError

pytestmark = pytest.mark.skip_unsupported


# ==========================================================
# Some useful global variables

# gates for which device support is tested
ops = {
    "Identity": qp.Identity(wires=[0]),
    "Snapshot": qp.Snapshot("label"),
    "BasisState": qp.BasisState(np.array([0]), wires=[0]),
    "BlockEncode": qp.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
    "CNOT": qp.CNOT(wires=[0, 1]),
    "CRX": qp.CRX(0, wires=[0, 1]),
    "CRY": qp.CRY(0, wires=[0, 1]),
    "CRZ": qp.CRZ(0, wires=[0, 1]),
    "CRot": qp.CRot(0, 0, 0, wires=[0, 1]),
    "CSWAP": qp.CSWAP(wires=[0, 1, 2]),
    "CZ": qp.CZ(wires=[0, 1]),
    "CCZ": qp.CCZ(wires=[0, 1, 2]),
    "CY": qp.CY(wires=[0, 1]),
    "CH": qp.CH(wires=[0, 1]),
    "DiagonalQubitUnitary": qp.DiagonalQubitUnitary(np.array([1, 1]), wires=[0]),
    "Hadamard": qp.Hadamard(wires=[0]),
    "H": qp.H(wires=[0]),
    "MultiRZ": qp.MultiRZ(0, wires=[0]),
    "PauliX": qp.X(0),
    "PauliY": qp.Y(0),
    "PauliZ": qp.Z(0),
    "X": qp.X([0]),
    "Y": qp.Y([0]),
    "Z": qp.Z([0]),
    "PhaseShift": qp.PhaseShift(0, wires=[0]),
    "PCPhase": qp.PCPhase(0, 1, wires=[0, 1]),
    "ControlledPhaseShift": qp.ControlledPhaseShift(0, wires=[0, 1]),
    "CPhaseShift00": qp.CPhaseShift00(0, wires=[0, 1]),
    "CPhaseShift01": qp.CPhaseShift01(0, wires=[0, 1]),
    "CPhaseShift10": qp.CPhaseShift10(0, wires=[0, 1]),
    "StatePrep": qp.StatePrep(np.array([1.0, 0.0]), wires=[0]),
    "QubitDensityMatrix": qp.QubitDensityMatrix(np.array([[0.5, 0.0], [0, 0.5]]), wires=[0]),
    "QubitUnitary": qp.QubitUnitary(np.eye(2), wires=[0]),
    "SpecialUnitary": qp.SpecialUnitary(np.array([0.2, -0.1, 2.3]), wires=1),
    "ControlledQubitUnitary": qp.ControlledQubitUnitary(np.eye(2), wires=[1, 0]),
    "MultiControlledX": qp.MultiControlledX(wires=[1, 2, 0]),
    "IntegerComparator": qp.IntegerComparator(1, geq=True, wires=[0, 1, 2]),
    "RX": qp.RX(0, wires=[0]),
    "RY": qp.RY(0, wires=[0]),
    "RZ": qp.RZ(0, wires=[0]),
    "Rot": qp.Rot(0, 0, 0, wires=[0]),
    "S": qp.S(wires=[0]),
    "Adjoint(S)": qp.adjoint(qp.S(wires=[0])),
    "SWAP": qp.SWAP(wires=[0, 1]),
    "ISWAP": qp.ISWAP(wires=[0, 1]),
    "PSWAP": qp.PSWAP(0, wires=[0, 1]),
    "ECR": qp.ECR(wires=[0, 1]),
    "Adjoint(ISWAP)": qp.adjoint(qp.ISWAP(wires=[0, 1])),
    "T": qp.T(wires=[0]),
    "Adjoint(T)": qp.adjoint(qp.T(wires=[0])),
    "SX": qp.SX(wires=[0]),
    "Adjoint(SX)": qp.adjoint(qp.SX(wires=[0])),
    "Barrier": qp.Barrier(wires=[0, 1, 2]),
    "WireCut": qp.WireCut(wires=[0]),
    "Toffoli": qp.Toffoli(wires=[0, 1, 2]),
    "QFT": qp.templates.QFT(wires=[0, 1, 2]),
    "IsingXX": qp.IsingXX(0, wires=[0, 1]),
    "IsingYY": qp.IsingYY(0, wires=[0, 1]),
    "IsingZZ": qp.IsingZZ(0, wires=[0, 1]),
    "IsingXY": qp.IsingXY(0, wires=[0, 1]),
    "SingleExcitation": qp.SingleExcitation(0, wires=[0, 1]),
    "SingleExcitationPlus": qp.SingleExcitationPlus(0, wires=[0, 1]),
    "SingleExcitationMinus": qp.SingleExcitationMinus(0, wires=[0, 1]),
    "DoubleExcitation": qp.DoubleExcitation(0, wires=[0, 1, 2, 3]),
    "DoubleExcitationPlus": qp.DoubleExcitationPlus(0, wires=[0, 1, 2, 3]),
    "DoubleExcitationMinus": qp.DoubleExcitationMinus(0, wires=[0, 1, 2, 3]),
    "QubitCarry": qp.QubitCarry(wires=[0, 1, 2, 3]),
    "QubitSum": qp.QubitSum(wires=[0, 1, 2]),
    "PauliRot": qp.PauliRot(0, "XXYY", wires=[0, 1, 2, 3]),
    "U1": qp.U1(0, wires=0),
    "U2": qp.U2(0, 0, wires=0),
    "U3": qp.U3(0, 0, 0, wires=0),
    "SISWAP": qp.SISWAP(wires=[0, 1]),
    "Adjoint(SISWAP)": qp.adjoint(qp.SISWAP(wires=[0, 1])),
    "OrbitalRotation": qp.OrbitalRotation(0, wires=[0, 1, 2, 3]),
    "FermionicSWAP": qp.FermionicSWAP(0, wires=[0, 1]),
    "GlobalPhase": qp.GlobalPhase(0.123),
}

all_ops = ops.keys()

# All qubit operations should be available to test in the device test suite
# Linting check disabled as static analysis can misidentify qp.ops as the set instance qp.ops.qubit.ops
all_available_ops = qp.ops._qubit__ops__.copy()  # pylint: disable=protected-access
all_available_ops.remove("CPhase")  # CPhase is an alias of ControlledPhaseShift
all_available_ops.remove("SQISW")  # SQISW is an alias of SISWAP
all_available_ops.add("QFT")  # QFT was recently moved to being a template, but let's keep it here

symbolic_ops = {"Adjoint(S)", "Adjoint(T)", "Adjoint(SX)", "Adjoint(ISWAP)", "Adjoint(SISWAP)"}

if not set(all_ops) == all_available_ops.union(symbolic_ops):
    raise ValueError(
        "A qubit operation has been added that is not being tested in the "
        "device test suite. Please add to the ops dictionary in "
        "pennylane/devices/tests/test_gates.py"
    )

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
ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
ECR = np.array(
    [
        [0, 0, 1 / sqrt(2), 1j * 1 / sqrt(2)],
        [0, 0, 1j * 1 / sqrt(2), 1 / sqrt(2)],
        [1 / sqrt(2), -1j * 1 / sqrt(2), 0, 0],
        [-1j * 1 / sqrt(2), 1 / sqrt(2), 0, 0],
    ]
)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CZ = np.diag([1, 1, 1, -1])
CCZ = np.diag([1, 1, 1, 1, 1, 1, 1, -1])
CY = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])
CH = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1 / sqrt(2), 1 / sqrt(2)],
        [0, 0, 1 / sqrt(2), -1 / sqrt(2)],
    ]
)
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

IsingXX = lambda phi: np.array(
    [
        [cos(phi / 2), 0, 0, -1j * sin(phi / 2)],
        [0, cos(phi / 2), -1j * sin(phi / 2), 0],
        [0, -1j * sin(phi / 2), cos(phi / 2), 0],
        [-1j * sin(phi / 2), 0, 0, cos(phi / 2)],
    ]
)

IsingXY = lambda phi: np.array(
    [
        [1, 0, 0, 0],
        [0, cos(phi / 2), 1j * sin(phi / 2), 0],
        [0, 1j * sin(phi / 2), cos(phi / 2), 0],
        [0, 0, 0, 1],
    ]
)

IsingYY = lambda phi: np.array(
    [
        [cos(phi / 2), 0, 0, 1j * sin(phi / 2)],
        [0, cos(phi / 2), -1j * sin(phi / 2), 0],
        [0, -1j * sin(phi / 2), cos(phi / 2), 0],
        [1j * sin(phi / 2), 0, 0, cos(phi / 2)],
    ]
)

IsingZZ = lambda phi: np.array(
    [
        [exp(-1.0j * phi / 2), 0, 0, 0],
        [0, exp(1.0j * phi / 2), 0, 0],
        [0, 0, exp(1.0j * phi / 2), 0],
        [0, 0, 0, exp(-1.0j * phi / 2)],
    ]
)

PSWAP = lambda phi: np.array(
    [
        [1, 0, 0, 0],
        [0, 0, exp(1.0j * phi), 0],
        [0, exp(1.0j * phi), 0, 0],
        [0, 0, 0, 1],
    ]
)


def adjoint_tuple(op, orig_mat):
    """Returns op constructor and matrix for provided base ops."""
    mat = qp.math.conj(qp.math.transpose(orig_mat))
    return (qp.adjoint(op), mat)


# list of all non-parametrized single-qubit gates,
# along with the PennyLane operation name
single_qubit = [
    (qp.PauliX, X),
    (qp.PauliY, Y),
    (qp.PauliZ, Z),
    (qp.X, X),
    (qp.Y, Y),
    (qp.Z, Z),
    (qp.Hadamard, H),
    (qp.H, H),
    (qp.S, S),
    (qp.T, T),
    (qp.SX, SX),
    adjoint_tuple(qp.S, S),
    adjoint_tuple(qp.T, T),
    adjoint_tuple(qp.SX, SX),
]

# list of all parametrized single-qubit gates
# taking a single parameter
single_qubit_param = [
    (qp.PhaseShift, phase_shift),
    (qp.RX, rx),
    (qp.RY, ry),
    (qp.RZ, rz),
]
# list of all non-parametrized two-qubit gates
two_qubit = [
    (qp.CNOT, CNOT),
    (qp.SWAP, SWAP),
    (qp.ISWAP, ISWAP),
    (qp.ECR, ECR),
    (qp.CZ, CZ),
    (qp.CY, CY),
    (qp.CH, CH),
    adjoint_tuple(qp.ISWAP, ISWAP),
]
# list of all parametrized two-qubit gates
two_qubit_param = [
    (qp.CRX, crx),
    (qp.CRY, cry),
    (qp.CRZ, crz),
    (qp.IsingXX, IsingXX),
    (qp.IsingXY, IsingXY),
    (qp.IsingYY, IsingYY),
    (qp.IsingZZ, IsingZZ),
    (qp.PSWAP, PSWAP),
]
two_qubit_multi_param = [(qp.CRot, crot)]
# list of all three-qubit gates
three_qubit = [(qp.Toffoli, toffoli), (qp.CSWAP, CSWAP), (qp.CCZ, CCZ)]

# single qubit unitary matrix
theta = 0.8364
phi = -0.1234
U = np.array(
    [
        [
            np.cos(theta / 2) * np.exp(np.complex128(-phi / 2j)),
            -np.sin(theta / 2) * np.exp(np.complex128(phi / 2j)),
        ],
        [
            np.sin(theta / 2) * np.exp(np.complex128(-phi / 2j)),
            np.cos(theta / 2) * np.exp(np.complex128(phi / 2j)),
        ],
    ]
)

# two qubit unitary matrix
U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / sqrt(3)

# single qubit Hermitian observable
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

# ===============================================================


# pylint: disable=too-few-public-methods
class TestSupportedGates:
    """Test that the device can implement all gates that it claims to support."""

    @pytest.mark.parametrize("operation", all_ops)
    def test_supported_gates_can_be_implemented(self, device_kwargs, operation):
        """Test that the device can implement all its supported gates."""
        device_kwargs["wires"] = 4  # maximum size of current gates
        dev = qp.device(**device_kwargs)

        if isinstance(dev, qp.devices.LegacyDevice):
            if operation not in dev.operations:
                pytest.skip("operation not supported.")
        else:
            if ops[operation].name == "QubitDensityMatrix":
                prog = dev.preprocess_transforms()
                tape = qp.tape.QuantumScript([ops[operation]])
                try:
                    prog((tape,))
                except DeviceError:
                    pytest.skip("operation not supported on the device")

        @qp.qnode(dev)
        def circuit():
            qp.apply(ops[operation])
            return qp.expval(qp.Identity(wires=0))

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
        if isinstance(dev, qp.devices.LegacyDevice):
            skip_if(dev, {"returns_probs": False})

        @qp.qnode(dev)
        def circuit():
            qp.BasisState(basis_state, wires=range(n_wires))
            return qp.probs(wires=range(n_wires))

        res = circuit()

        expected = np.zeros([2**n_wires])
        expected[np.ravel_multi_index(basis_state, [2] * n_wires)] = 1
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_state_prep(self, device, init_state, tol, skip_if):
        """Test StatePrep initialisation."""
        n_wires = 1
        dev = device(n_wires)
        if isinstance(dev, qp.devices.LegacyDevice):
            skip_if(dev, {"returns_probs": False})

        rnd_state = init_state(n_wires)

        @qp.qnode(dev)
        def circuit():
            qp.StatePrep(rnd_state, wires=range(n_wires))
            return qp.probs(range(n_wires))

        res = circuit()
        expected = np.abs(rnd_state) ** 2

        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize("op,mat", single_qubit)
    def test_single_qubit_no_parameters(self, device, init_state, op, mat, tol, skip_if, benchmark):
        """Test PauliX application."""
        n_wires = 1
        dev = device(n_wires)
        if isinstance(dev, qp.devices.LegacyDevice):
            skip_if(dev, {"returns_probs": False})

        rnd_state = init_state(n_wires)

        @qp.qnode(dev)
        def circuit():
            qp.StatePrep(rnd_state, wires=range(n_wires))
            op(wires=range(n_wires))
            return qp.probs(wires=range(n_wires))

        res = benchmark(circuit)

        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize("gamma", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters(
        self, device, init_state, op, func, gamma, tol, skip_if, benchmark
    ):
        """Test single qubit gates taking a single scalar argument."""
        n_wires = 1
        dev = device(n_wires)
        if isinstance(dev, qp.devices.LegacyDevice):
            skip_if(dev, {"returns_probs": False})

        rnd_state = init_state(n_wires)

        @qp.qnode(dev)
        def circuit():
            qp.StatePrep(rnd_state, wires=range(n_wires))
            op(gamma, wires=range(n_wires))
            return qp.probs(wires=range(n_wires))

        res = benchmark(circuit)

        expected = np.abs(func(gamma) @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_rotation(self, device, init_state, tol, skip_if, benchmark):
        """Test three axis rotation gate."""
        n_wires = 1
        dev = device(n_wires)
        if isinstance(dev, qp.devices.LegacyDevice):
            skip_if(dev, {"returns_probs": False})

        rnd_state = init_state(n_wires)
        a = 0.542
        b = 1.3432
        c = -0.654

        @qp.qnode(dev)
        def circuit():
            qp.StatePrep(rnd_state, wires=range(n_wires))
            qp.Rot(a, b, c, wires=range(n_wires))
            return qp.probs(wires=range(n_wires))

        res = benchmark(circuit)

        expected = np.abs(rot(a, b, c) @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize("op,mat", two_qubit)
    def test_two_qubit_no_parameters(self, device, init_state, op, mat, tol, skip_if, benchmark):
        """Test two qubit gates."""
        n_wires = 2
        dev = device(n_wires)
        if isinstance(dev, qp.devices.LegacyDevice):
            skip_if(dev, {"returns_probs": False})
            if not dev.supports_operation(op(wires=range(n_wires)).name):
                pytest.skip("op not supported")

        rnd_state = init_state(n_wires)

        @qp.qnode(dev)
        def circuit():
            qp.StatePrep(rnd_state, wires=range(n_wires))
            op(wires=range(n_wires))
            return qp.probs(wires=range(n_wires))

        res = benchmark(circuit)

        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize("param", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", two_qubit_param)
    def test_two_qubit_parameters(
        self, device, init_state, op, func, param, tol, skip_if, benchmark
    ):
        """Test parametrized two qubit gates taking a single scalar argument."""
        n_wires = 2
        dev = device(n_wires)
        if isinstance(dev, qp.devices.LegacyDevice):
            skip_if(dev, {"returns_probs": False})

        rnd_state = init_state(n_wires)

        @qp.qnode(dev)
        def circuit():
            qp.StatePrep(rnd_state, wires=range(n_wires))
            op(param, wires=range(n_wires))
            return qp.probs(wires=range(n_wires))

        res = benchmark(circuit)

        expected = np.abs(func(param) @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, device, init_state, mat, tol, skip_if, benchmark):
        """Test QubitUnitary gate."""
        n_wires = int(np.log2(len(mat)))
        dev = device(n_wires)

        if isinstance(dev, qp.devices.LegacyDevice):
            if "QubitUnitary" not in dev.operations:
                pytest.skip("Skipped because device does not support QubitUnitary.")

            skip_if(dev, {"returns_probs": False})

        rnd_state = init_state(n_wires)

        @qp.qnode(dev)
        def circuit():
            qp.StatePrep(rnd_state, wires=range(n_wires))
            qp.QubitUnitary(mat, wires=list(range(n_wires)))
            return qp.probs(wires=range(n_wires))

        res = benchmark(circuit)

        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize("theta_", [np.array([0.4, -0.1, 0.2]), np.ones(15) / 3])
    def test_special_unitary(self, device, init_state, theta_, tol, skip_if, benchmark):
        """Test SpecialUnitary gate."""
        n_wires = int(np.log(len(theta_) + 1) / np.log(4))
        dev = device(n_wires)

        if isinstance(dev, qp.devices.LegacyDevice):
            if "SpecialUnitary" not in dev.operations:
                pytest.skip("Skipped because device does not support SpecialUnitary.")

            skip_if(dev, {"returns_probs": False})

        rnd_state = init_state(n_wires)

        @qp.qnode(dev)
        def circuit():
            qp.StatePrep(rnd_state, wires=range(n_wires))
            qp.SpecialUnitary(theta_, wires=list(range(n_wires)))
            return qp.probs(wires=range(n_wires))

        res = benchmark(circuit)

        # Disabling Pylint test because qp.ops can be misunderstood as qp.ops.qubit.ops
        basis_fn = qp.ops.qubit.special_unitary.pauli_basis_matrices
        basis = basis_fn(n_wires)
        mat = qp.math.expm(1j * np.tensordot(theta_, basis, axes=[[0], [0]]))
        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.parametrize("op, mat", three_qubit)
    def test_three_qubit_no_parameters(self, device, init_state, op, mat, tol, skip_if, benchmark):
        """Test three qubit gates without parameters."""
        n_wires = 3
        dev = device(n_wires)

        if isinstance(dev, qp.devices.LegacyDevice):
            skip_if(dev, {"returns_probs": False})

        rnd_state = init_state(n_wires)

        @qp.qnode(dev)
        def circuit():
            qp.StatePrep(rnd_state, wires=range(n_wires))
            op(wires=[0, 1, 2])
            return qp.probs(wires=range(n_wires))

        res = benchmark(circuit)

        expected = np.abs(mat @ rnd_state) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))
