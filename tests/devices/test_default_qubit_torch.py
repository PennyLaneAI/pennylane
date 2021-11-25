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
Unit tests and integration tests for the ``default.qubit.torch`` device.
"""
from itertools import product

import numpy as np
import pytest
import cmath
import math

torch = pytest.importorskip("torch", minversion="1.8.1")

import pennylane as qml
from pennylane import numpy as pnp
from pennylane import DeviceError
from pennylane.wires import Wires
from pennylane.devices.default_qubit_torch import DefaultQubitTorch
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
    IsingXX,
    IsingYY,
    IsingZZ,
    MultiRZ1,
    MultiRZ2,
    ControlledPhaseShift,
    SingleExcitation,
    SingleExcitationPlus,
    SingleExcitationMinus,
    DoubleExcitation,
    DoubleExcitationPlus,
    DoubleExcitationMinus,
    OrbitalRotation,
)

np.random.seed(42)


#####################################################
# Test matrices
#####################################################

U = torch.tensor(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ],
    dtype=torch.complex128,
)

U2 = torch.tensor(
    [[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]], dtype=torch.complex128
) / np.sqrt(3)


#####################################################
# Define standard qubit operations
#####################################################

single_qubit = [
    (qml.S, torch.tensor(S)),
    (qml.T, torch.tensor(T)),
    (qml.PauliX, torch.tensor(X, dtype=torch.complex128)),
    (qml.PauliY, torch.tensor(Y)),
    (qml.PauliZ, torch.tensor(Z, dtype=torch.complex128)),
    (qml.Hadamard, torch.tensor(H, dtype=torch.complex128)),
]
single_qubit_param = [
    (qml.PhaseShift, Rphi),
    (qml.RX, Rotx),
    (qml.RY, Roty),
    (qml.RZ, Rotz),
    (qml.MultiRZ, MultiRZ1),
]
two_qubit = [(qml.CZ, CZ), (qml.CNOT, CNOT), (qml.SWAP, SWAP)]
two_qubit_param = [
    (qml.CRX, CRotx),
    (qml.CRY, CRoty),
    (qml.CRZ, CRotz),
    (qml.IsingXX, IsingXX),
    (qml.IsingYY, IsingYY),
    (qml.IsingZZ, IsingZZ),
    (qml.MultiRZ, MultiRZ2),
    (qml.ControlledPhaseShift, ControlledPhaseShift),
    (qml.SingleExcitation, SingleExcitation),
    (qml.SingleExcitationPlus, SingleExcitationPlus),
    (qml.SingleExcitationMinus, SingleExcitationMinus),
]
three_qubit = [(qml.Toffoli, Toffoli), (qml.CSWAP, CSWAP)]
four_qubit_param = [
    (qml.DoubleExcitation, DoubleExcitation),
    (qml.DoubleExcitationPlus, DoubleExcitationPlus),
    (qml.DoubleExcitationMinus, DoubleExcitationMinus),
    (qml.OrbitalRotation, OrbitalRotation),
]


#####################################################
# Fixtures
#####################################################


@pytest.fixture
def init_state(scope="session"):
    """Generates a random initial state"""

    def _init_state(n):
        """random initial state"""
        torch.manual_seed(42)
        state = torch.rand([2 ** n], dtype=torch.complex128) + torch.rand([2 ** n]) * 1j
        state /= torch.linalg.norm(state)
        return state

    return _init_state


#####################################################
# Initialization test
#####################################################


def test_analytic_deprecation():
    """Tests if the kwarg `analytic` is used and displays error message."""
    msg = "The analytic argument has been replaced by shots=None. "
    msg += "Please use shots=None instead of analytic=True."

    with pytest.raises(DeviceError, match=msg):
        qml.device("default.qubit.torch", wires=1, shots=1, analytic=True)


#####################################################
# Device-level integration tests
#####################################################


class TestApply:
    """Test application of PennyLane operations."""

    def test_basis_state(self, tol):
        """Test basis state initialization"""
        dev = DefaultQubitTorch(wires=4)
        state = torch.tensor([0, 0, 1, 0])

        dev.apply([qml.BasisState(state, wires=[0, 1, 2, 3])])

        res = dev.state
        expected = torch.zeros([2 ** 4], dtype=torch.complex128)
        expected[2] = 1

        assert isinstance(res, torch.Tensor)
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_invalid_basis_state_length(self, tol):
        """Test that an exception is raised if the basis state is the wrong size"""
        dev = DefaultQubitTorch(wires=4)
        state = torch.tensor([0, 0, 1, 0])

        with pytest.raises(
            ValueError, match=r"BasisState parameter and wires must be of equal length"
        ):
            dev.apply([qml.BasisState(state, wires=[0, 1, 2])])

    def test_invalid_basis_state(self, tol):
        """Test that an exception is raised if the basis state is invalid"""
        dev = DefaultQubitTorch(wires=4)
        state = torch.tensor([0, 0, 1, 2])

        with pytest.raises(
            ValueError, match=r"BasisState parameter must consist of 0 or 1 integers"
        ):
            dev.apply([qml.BasisState(state, wires=[0, 1, 2, 3])])

    def test_qubit_state_vector(self, init_state, tol):
        """Test qubit state vector application"""
        dev = DefaultQubitTorch(wires=1)
        state = init_state(1)

        dev.apply([qml.QubitStateVector(state, wires=[0])])

        res = dev.state
        expected = state
        assert isinstance(res, torch.Tensor)
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_full_subsystem_statevector(self, mocker):
        """Test applying a state vector to the full subsystem"""
        dev = DefaultQubitTorch(wires=["a", "b", "c"])
        state = torch.tensor([1, 0, 0, 0, 1, 0, 1, 1], dtype=torch.complex128) / 2.0
        state_wires = qml.wires.Wires(["a", "b", "c"])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        assert torch.allclose(torch.reshape(dev._state, (-1,)), state)
        spy.assert_not_called()

    def test_partial_subsystem_statevector(self, mocker):
        """Test applying a state vector to a subset of wires of the full subsystem"""
        dev = DefaultQubitTorch(wires=["a", "b", "c"])
        state = torch.tensor([1, 0, 1, 0], dtype=torch.complex128) / math.sqrt(2.0)
        state_wires = qml.wires.Wires(["a", "c"])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)
        res = torch.reshape(torch.sum(dev._state, axis=(1,)), [-1])

        assert torch.allclose(res, state)
        spy.assert_called()

    def test_invalid_qubit_state_vector_size(self):
        """Test that an exception is raised if the state
        vector is the wrong size"""
        dev = DefaultQubitTorch(wires=2)
        state = torch.tensor([0, 1])

        with pytest.raises(ValueError, match=r"State vector must be of length 2\*\*wires"):
            dev.apply([qml.QubitStateVector(state, wires=[0, 1])])

    @pytest.mark.parametrize(
        "state", [torch.tensor([0, 12]), torch.tensor([1.0, -1.0], requires_grad=True)]
    )
    def test_invalid_qubit_state_vector_norm(self, state):
        """Test that an exception is raised if the state
        vector is not normalized"""
        dev = DefaultQubitTorch(wires=2)

        with pytest.raises(ValueError, match=r"Sum of amplitudes-squared does not equal one"):
            dev.apply([qml.QubitStateVector(state, wires=[0])])

    def test_invalid_state_prep(self):
        """Test that an exception is raised if a state preparation is not the
        first operation in the circuit."""
        dev = DefaultQubitTorch(wires=2)
        state = torch.tensor([0, 12])

        with pytest.raises(
            qml.DeviceError,
            match=r"cannot be used after other Operations have already been applied",
        ):
            dev.apply([qml.PauliZ(0), qml.QubitStateVector(state, wires=[0])])

    @pytest.mark.parametrize("op,mat", single_qubit)
    def test_single_qubit_no_parameters(self, init_state, op, mat, tol):
        """Test non-parametrized single qubit operations"""
        dev = DefaultQubitTorch(wires=1)
        state = init_state(1)

        queue = [qml.QubitStateVector(state, wires=[0])]
        queue += [op(wires=0)]
        dev.apply(queue)

        res = dev.state
        # assert mat.dtype == state.dtype
        expected = torch.matmul(mat, state)
        assert isinstance(res, torch.Tensor)
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters(self, init_state, op, func, theta, tol):
        """Test parametrized single qubit operations"""
        dev = DefaultQubitTorch(wires=1)
        state = init_state(1)

        queue = [qml.QubitStateVector(state, wires=[0])]
        queue += [op(theta, wires=0)]
        dev.apply(queue)

        res = dev.state
        op_mat = torch.tensor(func(theta), dtype=torch.complex128)
        expected = torch.matmul(op_mat, state)
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters_inverse(self, init_state, op, func, theta, tol):
        """Test parametrized single qubit operations"""
        dev = DefaultQubitTorch(wires=1)
        state = init_state(1)

        queue = [qml.QubitStateVector(state, wires=[0])]
        queue += [op(theta, wires=0).inv()]
        dev.apply(queue)

        res = dev.state
        op_mat = torch.tensor(func(theta), dtype=torch.complex128)
        op_mat = torch.transpose(torch.conj(op_mat), 0, 1)
        expected = torch.matmul(op_mat, state)
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_rotation(self, init_state, tol):
        """Test three axis rotation gate"""
        dev = DefaultQubitTorch(wires=1)
        state = init_state(1)

        a = 0.542
        b = 1.3432
        c = -0.654

        queue = [qml.QubitStateVector(state, wires=[0])]
        queue += [qml.Rot(a, b, c, wires=0)]
        dev.apply(queue)

        res = dev.state
        op_mat = torch.tensor(Rot3(a, b, c), dtype=torch.complex128)
        expected = op_mat @ state
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_controlled_rotation(self, init_state, tol):
        """Test three axis controlled-rotation gate"""
        dev = DefaultQubitTorch(wires=2)
        state = init_state(2)

        a = 0.542
        b = 1.3432
        c = -0.654

        queue = [qml.QubitStateVector(state, wires=[0, 1])]
        queue += [qml.CRot(a, b, c, wires=[0, 1])]
        dev.apply(queue)

        res = dev.state
        op_mat = torch.tensor(CRot3(a, b, c), dtype=torch.complex128)
        expected = op_mat @ state
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_inverse_operation(self, init_state, tol):
        """Test that the inverse of an operation is correctly applied"""
        """Test three axis rotation gate"""
        dev = DefaultQubitTorch(wires=1)
        state = init_state(1)

        a = 0.542
        b = 1.3432
        c = -0.654

        queue = [qml.QubitStateVector(state, wires=[0])]
        queue += [qml.Rot(a, b, c, wires=0).inv()]
        dev.apply(queue)

        res = dev.state
        op_mat = torch.tensor(Rot3(a, b, c), dtype=torch.complex128)
        expected = torch.linalg.inv(op_mat) @ state
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op,mat", two_qubit)
    def test_two_qubit_no_parameters(self, init_state, op, mat, tol):
        """Test non-parametrized two qubit operations"""
        dev = DefaultQubitTorch(wires=2)
        state = init_state(2)

        queue = [qml.QubitStateVector(state, wires=[0, 1])]
        queue += [op(wires=[0, 1])]
        dev.apply(queue)

        res = dev.state
        expected = torch.tensor(mat, dtype=torch.complex128) @ state
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, init_state, mat, tol):
        """Test application of arbitrary qubit unitaries"""
        N = int(math.log(len(mat), 2))
        dev = DefaultQubitTorch(wires=N)
        state = init_state(N)

        queue = [qml.QubitStateVector(state, wires=range(N))]
        queue += [qml.QubitUnitary(mat, wires=range(N))]
        dev.apply(queue)

        res = dev.state
        expected = mat @ state
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_diagonal_qubit_unitary(self, init_state, tol):
        """Tests application of a diagonal qubit unitary"""
        dev = DefaultQubitTorch(wires=1)
        state = init_state(1)

        diag = torch.tensor(
            [-1.0 + 1j, 1.0 + 1j], requires_grad=True, dtype=torch.complex128
        ) / math.sqrt(2)

        queue = [qml.QubitStateVector(state, wires=0), qml.DiagonalQubitUnitary(diag, wires=0)]
        dev.apply(queue)

        res = dev.state
        expected = torch.diag(diag) @ state
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_diagonal_qubit_unitary_inverse(self, init_state, tol):
        """Tests application of a diagonal qubit unitary"""
        dev = DefaultQubitTorch(wires=1)
        state = init_state(1)

        diag = torch.tensor(
            [-1.0 + 1j, 1.0 + 1j], requires_grad=True, dtype=torch.complex128
        ) / math.sqrt(2)

        queue = [
            qml.QubitStateVector(state, wires=0),
            qml.DiagonalQubitUnitary(diag, wires=0).inv(),
        ]
        dev.apply(queue)

        res = dev.state
        expected = torch.diag(diag).conj() @ state
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op, mat", three_qubit)
    def test_three_qubit_no_parameters(self, init_state, op, mat, tol):
        """Test non-parametrized three qubit operations"""
        dev = DefaultQubitTorch(wires=3)
        state = init_state(3)

        queue = [qml.QubitStateVector(state, wires=[0, 1, 2])]
        queue += [op(wires=[0, 1, 2])]
        dev.apply(queue)

        res = dev.state
        expected = torch.tensor(mat, dtype=torch.complex128) @ state
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", two_qubit_param)
    def test_two_qubit_parameters(self, init_state, op, func, theta, tol):
        """Test two qubit parametrized operations"""
        dev = DefaultQubitTorch(wires=2)
        state = init_state(2)

        queue = [qml.QubitStateVector(state, wires=[0, 1])]
        queue += [op(theta, wires=[0, 1])]
        dev.apply(queue)

        res = dev.state
        op_mat = torch.tensor(func(theta), dtype=torch.complex128)
        expected = op_mat @ state
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", four_qubit_param)
    def test_four_qubit_parameters(self, init_state, op, func, theta, tol):
        """Test two qubit parametrized operations"""
        dev = DefaultQubitTorch(wires=4)
        state = init_state(4)

        queue = [qml.QubitStateVector(state, wires=[0, 1, 2, 3])]
        queue += [op(theta, wires=[0, 1, 2, 3])]
        dev.apply(queue)

        res = dev.state
        op_mat = torch.tensor(func(theta), dtype=torch.complex128)
        expected = op_mat @ state
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_apply_ops_above_8_wires_using_special(self):
        """Test that special apply methods that involve slicing function correctly when using 9
        wires"""
        dev = DefaultQubitTorch(wires=9)
        dev._apply_ops = {"CNOT": dev._apply_cnot}

        queue = [qml.CNOT(wires=[1, 2])]
        dev.apply(queue)


THETA = torch.linspace(0.11, 1, 3, dtype=torch.float64)
PHI = torch.linspace(0.32, 1, 3, dtype=torch.float64)
VARPHI = torch.linspace(0.02, 1, 3, dtype=torch.float64)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestExpval:
    """Test expectation values"""

    # test data; each tuple is of the form (GATE, OBSERVABLE, EXPECTED)
    single_wire_expval_test_data = [
        (qml.RX, qml.Identity, lambda t, p: torch.tensor([1.0, 1.0], dtype=torch.float64)),
        (
            qml.RX,
            qml.PauliZ,
            lambda t, p: torch.tensor(
                [torch.cos(t), torch.cos(t) * torch.cos(p)], dtype=torch.float64
            ),
        ),
        (
            qml.RY,
            qml.PauliX,
            lambda t, p: torch.tensor(
                [torch.sin(t) * torch.sin(p), torch.sin(p)], dtype=torch.float64
            ),
        ),
        (
            qml.RX,
            qml.PauliY,
            lambda t, p: torch.tensor([0, -torch.cos(t) * torch.sin(p)], dtype=torch.float64),
        ),
        (
            qml.RY,
            qml.Hadamard,
            lambda t, p: torch.tensor(
                [
                    torch.sin(t) * torch.sin(p) + torch.cos(t),
                    torch.cos(t) * torch.cos(p) + torch.sin(p),
                ],
                dtype=torch.float64,
            )
            / math.sqrt(2),
        ),
    ]

    @pytest.mark.parametrize("gate,obs,expected", single_wire_expval_test_data)
    def test_single_wire_expectation(self, gate, obs, expected, theta, phi, varphi, tol):
        """Test that single qubit gates with single qubit expectation values"""
        dev = DefaultQubitTorch(wires=2)

        with qml.tape.QuantumTape() as tape:
            queue = [gate(theta, wires=0), gate(phi, wires=1), qml.CNOT(wires=[0, 1])]
            observables = [qml.expval(obs(wires=[i])) for i in range(2)]

        res = dev.execute(tape)

        expected_res = expected(theta, phi)
        assert torch.allclose(res, expected_res, atol=tol, rtol=0)

    def test_hermitian_expectation(self, theta, phi, varphi, tol):
        """Test that arbitrary Hermitian expectation values are correct"""
        dev = DefaultQubitTorch(wires=2)

        Hermitian_mat = torch.tensor(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]],
            dtype=torch.complex128,
        )

        with qml.tape.QuantumTape() as tape:
            queue = [qml.RY(theta, wires=0), qml.RY(phi, wires=1), qml.CNOT(wires=[0, 1])]
            observables = [qml.expval(qml.Hermitian(Hermitian_mat, wires=[i])) for i in range(2)]

        res = dev.execute(tape)

        a = Hermitian_mat[0, 0]
        re_b = Hermitian_mat[0, 1].real
        d = Hermitian_mat[1, 1]
        ev1 = (
            (a - d) * torch.cos(theta) + 2 * re_b * torch.sin(theta) * torch.sin(phi) + a + d
        ) / 2
        ev2 = ((a - d) * torch.cos(theta) * torch.cos(phi) + 2 * re_b * torch.sin(phi) + a + d) / 2
        expected = torch.tensor([ev1, ev2], dtype=torch.float64)

        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_multi_mode_hermitian_expectation(self, theta, phi, varphi, tol):
        """Test that arbitrary multi-mode Hermitian expectation values are correct"""
        Hermit_mat2 = torch.tensor(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ],
            dtype=torch.complex128,
        )

        dev = DefaultQubitTorch(wires=2)

        with qml.tape.QuantumTape() as tape:
            queue = [qml.RY(theta, wires=0), qml.RY(phi, wires=1), qml.CNOT(wires=[0, 1])]
            observables = [qml.expval(qml.Hermitian(Hermit_mat2, wires=[0, 1]))]

        res = dev.execute(tape)

        # below is the analytic expectation value for this circuit with arbitrary
        # Hermitian observable Hermit_mat2
        expected = 0.5 * (
            6 * torch.cos(theta) * torch.sin(phi)
            - torch.sin(theta) * (8 * torch.sin(phi) + 7 * torch.cos(phi) + 3)
            - 2 * torch.sin(phi)
            - 6 * torch.cos(phi)
            - 6
        )

        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_paulix_pauliy(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit.torch", wires=3)
        dev.reset()

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = torch.sin(theta) * torch.sin(phi) * torch.sin(varphi)

        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_identity(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and Identity works correctly"""
        dev = qml.device("default.qubit.torch", wires=3)
        dev.reset()

        obs = qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = torch.cos(varphi) * torch.cos(phi)

        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("default.qubit.torch", wires=3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.reset()
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = -(
            torch.cos(varphi) * torch.sin(phi) + torch.sin(varphi) * torch.cos(theta)
        ) / math.sqrt(2)

        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.qubit.torch", wires=3)
        dev.reset()

        Hermit_mat3 = torch.tensor(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ],
            dtype=torch.complex128,
        )

        obs = qml.PauliZ(0) @ qml.Hermitian(Hermit_mat3, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = 0.5 * (
            -6 * torch.cos(theta) * (torch.cos(varphi) + 1)
            - 2 * torch.sin(varphi) * (torch.cos(theta) + torch.sin(phi) - 2 * torch.cos(phi))
            + 3 * torch.cos(varphi) * torch.sin(phi)
            + torch.sin(phi)
        )

        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving two Hermitian matrices works correctly"""
        dev = qml.device("default.qubit.torch", wires=3)

        A1 = torch.tensor([[1, 2], [2, 4]], dtype=torch.complex128)

        A2 = torch.tensor(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ],
            dtype=torch.complex128,
        )

        obs = qml.Hermitian(A1, wires=[0]) @ qml.Hermitian(A2, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = 0.25 * (
            -30
            + 4 * torch.cos(phi) * torch.sin(theta)
            + 3
            * torch.cos(varphi)
            * (-10 + 4 * torch.cos(phi) * torch.sin(theta) - 3 * torch.sin(phi))
            - 3 * torch.sin(phi)
            - 2
            * (
                5
                + torch.cos(phi) * (6 + 4 * torch.sin(theta))
                + (-3 + 8 * torch.sin(theta)) * torch.sin(phi)
            )
            * torch.sin(varphi)
            + torch.cos(theta)
            * (
                18
                + 5 * torch.sin(phi)
                + 3 * torch.cos(varphi) * (6 + 5 * torch.sin(phi))
                + 2 * (3 + 10 * torch.cos(phi) - 5 * torch.sin(phi)) * torch.sin(varphi)
            )
        )

        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_identity_expectation(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix and the identity works correctly"""
        dev = qml.device("default.qubit.torch", wires=2)

        A = torch.tensor(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]],
            dtype=torch.complex128,
        )

        obs = qml.Hermitian(A, wires=[0]) @ qml.Identity(wires=[1])

        dev.apply(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        expected = (
            (a - d) * torch.cos(theta) + 2 * re_b * torch.sin(theta) * torch.sin(phi) + a + d
        ) / 2

        assert torch.allclose(res, torch.real(expected), atol=tol, rtol=0)

    def test_hermitian_two_wires_identity_expectation(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = qml.device("default.qubit.torch", wires=3, shots=None)

        A = torch.tensor(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]],
            dtype=torch.complex128,
        )
        Identity = torch.tensor([[1, 0], [0, 1]])
        H = torch.kron(torch.kron(Identity, Identity), A)
        obs = qml.Hermitian(H, wires=[2, 1, 0])

        dev.apply(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            obs.diagonalizing_gates(),
        )
        res = dev.expval(obs)

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]

        expected = (
            (a - d) * torch.cos(theta) + 2 * re_b * torch.sin(theta) * torch.sin(phi) + a + d
        ) / 2
        assert torch.allclose(res, torch.real(expected), atol=tol, rtol=0)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestVar:
    """Tests for the variance"""

    def test_var(self, theta, phi, varphi, tol):
        """Tests for variance calculation"""
        dev = DefaultQubitTorch(wires=1)
        # test correct variance for <Z> of a rotated state

        with qml.tape.QuantumTape() as tape:
            queue = [qml.RX(phi, wires=0), qml.RY(theta, wires=0)]
            observables = [qml.var(qml.PauliZ(wires=[0]))]

        res = dev.execute(tape)
        expected = 0.25 * (
            3 - torch.cos(2 * theta) - 2 * torch.cos(theta) ** 2 * torch.cos(2 * phi)
        )
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_var_hermitian(self, theta, phi, varphi, tol):
        """Tests for variance calculation using an arbitrary Hermitian observable"""
        dev = DefaultQubitTorch(wires=2)

        # test correct variance for <H> of a rotated state
        H = torch.tensor([[4, -1 + 6j], [-1 - 6j, 2]], dtype=torch.complex128)

        with qml.tape.QuantumTape() as tape:
            queue = [qml.RX(phi, wires=0), qml.RY(theta, wires=0)]
            observables = [qml.var(qml.Hermitian(H, wires=[0]))]

        res = dev.execute(tape)
        expected = 0.5 * (
            2 * torch.sin(2 * theta) * torch.cos(phi) ** 2
            + 24 * torch.sin(phi) * torch.cos(phi) * (torch.sin(theta) - torch.cos(theta))
            + 35 * torch.cos(2 * phi)
            + 39
        )

        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_paulix_pauliy(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit.torch", wires=3)

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.var(obs)

        expected = (
            8 * torch.sin(theta) ** 2 * torch.cos(2 * varphi) * torch.sin(phi) ** 2
            - torch.cos(2 * (theta - phi))
            - torch.cos(2 * (theta + phi))
            + 2 * torch.cos(2 * theta)
            + 2 * torch.cos(2 * phi)
            + 14
        ) / 16

        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("default.qubit.torch", wires=3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.reset()
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.var(obs)

        expected = (
            3
            + torch.cos(2 * phi) * torch.cos(varphi) ** 2
            - torch.cos(2 * theta) * torch.sin(varphi) ** 2
            - 2 * torch.cos(theta) * torch.sin(phi) * torch.sin(2 * varphi)
        ) / 4

        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.qubit.torch", wires=3)

        A = torch.tensor(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ],
            dtype=torch.complex128,
        )

        obs = qml.PauliZ(0) @ qml.Hermitian(A, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.var(obs)

        expected = (
            1057
            - torch.cos(2 * phi)
            + 12 * (27 + torch.cos(2 * phi)) * torch.cos(varphi)
            - 2
            * torch.cos(2 * varphi)
            * torch.sin(phi)
            * (16 * torch.cos(phi) + 21 * torch.sin(phi))
            + 16 * torch.sin(2 * phi)
            - 8 * (-17 + torch.cos(2 * phi) + 2 * torch.sin(2 * phi)) * torch.sin(varphi)
            - 8 * torch.cos(2 * theta) * (3 + 3 * torch.cos(varphi) + torch.sin(varphi)) ** 2
            - 24 * torch.cos(phi) * (torch.cos(phi) + 2 * torch.sin(phi)) * torch.sin(2 * varphi)
            - 8
            * torch.cos(theta)
            * (
                4
                * torch.cos(phi)
                * (
                    4
                    + 8 * torch.cos(varphi)
                    + torch.cos(2 * varphi)
                    - (1 + 6 * torch.cos(varphi)) * torch.sin(varphi)
                )
                + torch.sin(phi)
                * (
                    15
                    + 8 * torch.cos(varphi)
                    - 11 * torch.cos(2 * varphi)
                    + 42 * torch.sin(varphi)
                    + 3 * torch.sin(2 * varphi)
                )
            )
        ) / 16

        assert torch.allclose(res, expected, atol=tol, rtol=0)


#####################################################
# QNode-level integration tests
#####################################################


class TestQNodeIntegration:
    """Integration tests for default.qubit.torch. This test ensures it integrates
    properly with the PennyLane UI, in particular the new QNode."""

    def test_defines_correct_capabilities(self):
        """Test that the device defines the right capabilities"""

        dev = qml.device("default.qubit.torch", wires=1)
        cap = dev.capabilities()
        capabilities = {
            "model": "qubit",
            "supports_finite_shots": True,
            "supports_tensor_observables": True,
            "returns_probs": True,
            "returns_state": True,
            "supports_reversible_diff": False,
            "supports_inverse_operations": True,
            "supports_analytic_computation": True,
            "passthru_interface": "torch",
            "passthru_devices": {
                "torch": "default.qubit.torch",
                "tf": "default.qubit.tf",
                "autograd": "default.qubit.autograd",
                "jax": "default.qubit.jax",
            },
        }
        assert cap == capabilities

    def test_load_torch_device(self):
        """Test that the torch device plugin loads correctly"""
        dev = qml.device("default.qubit.torch", wires=2)
        assert dev.num_wires == 2
        assert dev.shots is None
        assert dev.short_name == "default.qubit.torch"
        assert dev.capabilities()["passthru_interface"] == "torch"

    def test_qubit_circuit(self, tol):
        """Test that the torch device provides correct
        result for a simple circuit using the old QNode."""
        p = torch.tensor(0.543, dtype=torch.float64)

        dev = qml.device("default.qubit.torch", wires=1)

        @qml.qnode(dev, interface="torch")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -torch.sin(p)

        assert circuit.gradient_fn == "backprop"
        assert torch.allclose(circuit(p), expected, atol=tol, rtol=0)

    def test_correct_state(self, tol):
        """Test that the device state is correct after applying a
        quantum function on the device"""

        dev = qml.device("default.qubit.torch", wires=2)

        state = dev.state
        expected = torch.tensor([1, 0, 0, 0], dtype=torch.complex128)
        assert torch.allclose(state, expected, atol=tol, rtol=0)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit():
            qml.Hadamard(wires=0)
            qml.RZ(math.pi / 4, wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit()
        state = dev.state

        amplitude = cmath.exp(-1j * cmath.pi / 8) / cmath.sqrt(2)

        expected = torch.tensor([amplitude, 0, amplitude.conjugate(), 0], dtype=torch.complex128)
        assert torch.allclose(state, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_one_qubit_param_gates(self, theta, op, func, init_state, tol):
        """Test the integration of the one-qubit single parameter rotations by passing
        a Torch data structure as a parameter"""
        dev = qml.device("default.qubit.torch", wires=1)
        state = init_state(1)

        @qml.qnode(dev, interface="torch")
        def circuit(params):
            qml.QubitStateVector(state, wires=[0])
            op(params[0], wires=[0])
            return qml.expval(qml.PauliZ(0))

        params = torch.tensor([theta])
        circuit(params)
        res = dev.state
        expected = torch.tensor(func(theta), dtype=torch.complex128) @ state
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, 4.213])
    @pytest.mark.parametrize("op,func", two_qubit_param)
    def test_two_qubit_param_gates(self, theta, op, func, init_state, tol):
        """Test the integration of the two-qubit single parameter rotations by passing
        a Torch data structure as a parameter"""
        dev = qml.device("default.qubit.torch", wires=2)
        state = init_state(2)

        @qml.qnode(dev, interface="torch")
        def circuit(params):
            qml.QubitStateVector(state, wires=[0, 1])
            op(params[0], wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        # Pass a Torch Variable to the qfunc
        params = torch.tensor([theta])
        circuit(params)
        res = dev.state
        expected = torch.tensor(func(theta), dtype=torch.complex128) @ state
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, 4.213])
    @pytest.mark.parametrize("op,func", four_qubit_param)
    def test_four_qubit_param_gates(self, theta, op, func, init_state, tol):
        """Test the integration of the four-qubit single parameter rotations by passing
        a Torch data structure as a parameter"""
        dev = qml.device("default.qubit.torch", wires=4)
        state = init_state(4)

        @qml.qnode(dev, interface="torch")
        def circuit(params):
            qml.QubitStateVector(state, wires=[0, 1, 2, 3])
            op(params[0], wires=[0, 1, 2, 3])
            return qml.expval(qml.PauliZ(0))

        # Pass a Torch Variable to the qfunc
        params = torch.tensor([theta])
        circuit(params)
        res = dev.state
        expected = torch.tensor(func(theta), dtype=torch.complex128) @ state
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_controlled_rotation_integration(self, init_state, tol):
        """Test the integration of the two-qubit controlled rotation by passing
        a Torch data structure as a parameter"""
        dev = qml.device("default.qubit.torch", wires=2)
        a = 1.7
        b = 1.3432
        c = -0.654
        state = init_state(2)

        @qml.qnode(dev, interface="torch")
        def circuit(params):
            qml.QubitStateVector(state, wires=[0, 1])
            qml.CRot(params[0], params[1], params[2], wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        # Pass a Torch Variable to the qfunc
        params = torch.tensor([a, b, c])
        circuit(params)
        res = dev.state
        expected = torch.tensor(CRot3(a, b, c), dtype=torch.complex128) @ state
        assert torch.allclose(res, expected, atol=tol, rtol=0)


class TestPassthruIntegration:
    """Tests for integration with the PassthruQNode"""

    def test_jacobian_variable_multiply(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.torch device
        gives the correct result in the case of parameters multiplied by scalars"""
        x = torch.tensor(0.43316321, dtype=torch.float64, requires_grad=True)
        y = torch.tensor(0.43316321, dtype=torch.float64, requires_grad=True)
        z = torch.tensor(0.43316321, dtype=torch.float64, requires_grad=True)

        dev = qml.device("default.qubit.torch", wires=1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(p):
            qml.RX(3 * p[0], wires=0)
            qml.RY(p[1], wires=0)
            qml.RX(p[2] / 2, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit([x, y, z])
        res.backward()

        expected = torch.cos(3 * x) * torch.cos(y) * torch.cos(z / 2) - torch.sin(
            3 * x
        ) * torch.sin(z / 2)
        assert torch.allclose(res, expected, atol=tol, rtol=0)

        x_grad = -3 * (
            torch.sin(3 * x) * torch.cos(y) * torch.cos(z / 2) + torch.cos(3 * x) * torch.sin(z / 2)
        )
        y_grad = -torch.cos(3 * x) * torch.sin(y) * torch.cos(z / 2)
        z_grad = -0.5 * (
            torch.sin(3 * x) * torch.cos(z / 2) + torch.cos(3 * x) * torch.cos(y) * torch.sin(z / 2)
        )

        assert torch.allclose(x.grad, x_grad)
        assert torch.allclose(y.grad, y_grad)
        assert torch.allclose(z.grad, z_grad)

    def test_jacobian_repeated(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.torch device
        gives the correct result in the case of repeated parameters"""
        x = torch.tensor(0.43316321, dtype=torch.float64, requires_grad=True)
        y = torch.tensor(0.2162158, dtype=torch.float64, requires_grad=True)
        z = torch.tensor(0.75110998, dtype=torch.float64, requires_grad=True)
        p = torch.tensor([x, y, z], requires_grad=True)
        dev = qml.device("default.qubit.torch", wires=1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(p)
        res.backward()

        expected = torch.cos(y) ** 2 - torch.sin(x) * torch.sin(y) ** 2

        assert torch.allclose(res, expected, atol=tol, rtol=0)

        expected_grad = torch.tensor(
            [
                -torch.cos(x) * torch.sin(y) ** 2,
                -2 * (torch.sin(x) + 1) * torch.sin(y) * torch.cos(y),
                0,
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(p.grad, expected_grad, atol=tol, rtol=0)

    def test_jacobian_agrees_backprop_parameter_shift(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.torch device
        gives the correct result with respect to the parameter-shift method"""
        p = pnp.array([0.43316321, 0.2162158, 0.75110998, 0.94714242], requires_grad=True)

        def circuit(x):
            for i in range(0, len(p), 2):
                qml.RX(x[i], wires=0)
                qml.RY(x[i + 1], wires=1)
            for i in range(2):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))  # , qml.var(qml.PauliZ(1))

        dev1 = qml.device("default.qubit.torch", wires=3)
        dev2 = qml.device("default.qubit", wires=3)

        circuit1 = qml.QNode(circuit, dev1, diff_method="backprop", interface="torch")
        circuit2 = qml.QNode(circuit, dev2, diff_method="parameter-shift")

        p_torch = torch.tensor(p, requires_grad=True)
        res = circuit1(p_torch)
        res.backward()

        assert np.allclose(res.detach().numpy(), circuit2(p).numpy(), atol=tol, rtol=0)

        p_grad = p_torch.grad
        assert np.allclose(p_grad.detach().numpy(), qml.jacobian(circuit2)(p), atol=tol, rtol=0)

    def test_state_differentiability(self, tol):
        """Test that the device state can be differentiated"""
        dev = qml.device("default.qubit.torch", wires=1)

        @qml.qnode(dev, diff_method="backprop", interface="torch")
        def circuit(a):
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = torch.tensor(0.54, requires_grad=True)

        circuit(a)
        res = torch.abs(dev.state) ** 2
        res = res[1] - res[0]
        res.backward()

        grad = a.grad
        expected = torch.sin(a)
        assert torch.allclose(grad, expected, atol=tol, rtol=0)

    def test_prob_differentiability(self, tol):
        """Test that the device probability can be differentiated"""
        dev = qml.device("default.qubit.torch", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="torch")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = torch.tensor(0.54, requires_grad=True, dtype=torch.float64)
        b = torch.tensor(0.12, requires_grad=True, dtype=torch.float64)

        # get the probability of wire 1
        prob_wire_1 = circuit(a, b)
        # compute Prob(|1>_1) - Prob(|0>_1)
        res = prob_wire_1[1] - prob_wire_1[0]
        res.backward()

        expected = -torch.cos(a) * torch.cos(b)
        assert torch.allclose(res, expected, atol=tol, rtol=0)

        assert torch.allclose(a.grad, torch.sin(a) * torch.cos(b), atol=tol, rtol=0)
        assert torch.allclose(b.grad, torch.cos(a) * torch.sin(b), atol=tol, rtol=0)

    def test_backprop_gradient(self, tol):
        """Tests that the gradient of the qnode is correct"""
        dev = qml.device("default.qubit.torch", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="torch")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = torch.tensor(-0.234, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(0.654, dtype=torch.float64, requires_grad=True)

        res = circuit(a, b)
        res.backward()

        # the analytic result of evaluating circuit(a, b)
        expected_cost = 0.5 * (torch.cos(a) * torch.cos(b) + torch.cos(a) - torch.cos(b) + 1)

        assert torch.allclose(res, expected_cost, atol=tol, rtol=0)

        assert torch.allclose(a.grad, -0.5 * torch.sin(a) * (torch.cos(b) + 1), atol=tol, rtol=0)
        assert torch.allclose(b.grad, 0.5 * torch.sin(b) * (1 - torch.cos(a)))

    @pytest.mark.parametrize("operation", [qml.U3, qml.U3.decomposition])
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "finite-diff"])
    def test_torch_interface_gradient(self, operation, diff_method, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the PyTorch interface, using a variety of differentiation methods."""
        dev = qml.device("default.qubit.torch", wires=1)

        @qml.qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(x, weights, w):
            """In this example, a mixture of scalar
            arguments, array arguments, and keyword arguments are used."""
            qml.QubitStateVector(1j * torch.tensor([1, -1]) / math.sqrt(2), wires=w)
            operation(x, weights[0], weights[1], wires=w)
            return qml.expval(qml.PauliX(w))

        # Check that the correct QNode type is being used.
        if diff_method == "backprop":
            assert circuit.gradient_fn == "backprop"
        elif diff_method == "finite-diff":
            assert circuit.gradient_fn is qml.gradients.finite_diff

        def cost(params):
            """Perform some classical processing"""
            return circuit(params[0], params[1:], w=0) ** 2

        theta = torch.tensor(0.543, dtype=torch.float64)
        phi = torch.tensor(-0.234, dtype=torch.float64)
        lam = torch.tensor(0.654, dtype=torch.float64)

        params = torch.tensor([theta, phi, lam], dtype=torch.float64, requires_grad=True)

        res = cost(params)
        res.backward()

        # check that the result is correct
        expected_cost = (
            torch.sin(lam) * torch.sin(phi) - torch.cos(theta) * torch.cos(lam) * torch.cos(phi)
        ) ** 2
        assert torch.allclose(res, expected_cost, atol=tol, rtol=0)

        # check that the gradient is correct
        expected_grad = (
            torch.tensor(
                [
                    torch.sin(theta) * torch.cos(lam) * torch.cos(phi),
                    torch.cos(theta) * torch.cos(lam) * torch.sin(phi)
                    + torch.sin(lam) * torch.cos(phi),
                    torch.cos(theta) * torch.sin(lam) * torch.cos(phi)
                    + torch.cos(lam) * torch.sin(phi),
                ]
            )
            * 2
            * (torch.sin(lam) * torch.sin(phi) - torch.cos(theta) * torch.cos(lam) * torch.cos(phi))
        )
        assert torch.allclose(params.grad, expected_grad, atol=tol, rtol=0)

    def test_inverse_operation_jacobian_backprop(self, tol):
        """Test that inverse operations work in backprop
        mode"""
        dev = qml.device("default.qubit.torch", wires=1)

        @qml.qnode(dev, diff_method="backprop", interface="torch")
        def circuit(param):
            qml.RY(param, wires=0).inv()
            return qml.expval(qml.PauliX(0))

        x = torch.tensor(0.3, requires_grad=True, dtype=torch.float64)

        res = circuit(x)
        res.backward()

        assert torch.allclose(res, -torch.sin(x), atol=tol, rtol=0)

        grad = x.grad
        assert torch.allclose(grad, -torch.cos(x), atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["autograd", "torch"])
    def test_error_backprop_wrong_interface(self, interface, tol):
        """Tests that an error is raised if diff_method='backprop' but not using
        the torch interface"""
        dev = qml.device("default.qubit.torch", wires=1)

        def circuit(x, w=None):
            qml.RZ(x, wires=w)
            return qml.expval(qml.PauliX(w))

        with pytest.raises(Exception) as e:
            assert qml.qnode(dev, diff_method="autograd", interface=interface)(circuit)
        assert (
            str(e.value)
            == "Differentiation method autograd not recognized. Allowed options are ('best', 'parameter-shift', 'backprop', 'finite-diff', 'device', 'reversible', 'adjoint')."
        )


class TestSamples:
    """Tests for sampling outputs"""

    def test_sample_observables(self):
        """Test that the device allows for sampling from observables."""
        shots = 100
        dev = qml.device("default.qubit.torch", wires=2, shots=shots)

        @qml.qnode(dev, diff_method=None, interface="torch")
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(0))

        a = torch.tensor(0.54, dtype=torch.float64)
        res = circuit(a)

        assert torch.is_tensor(res)
        assert res.shape == (shots,)
        assert torch.allclose(torch.unique(res), torch.tensor([-1, 1], dtype=torch.int64))

    def test_estimating_marginal_probability(self, tol):
        """Test that the probability of a subset of wires is accurately estimated."""
        dev = qml.device("default.qubit.torch", wires=2, shots=1000)

        @qml.qnode(dev, diff_method=None, interface="torch")
        def circuit():
            qml.PauliX(0)
            return qml.probs(wires=[0])

        res = circuit()

        assert torch.is_tensor(res)

        expected = torch.tensor([0, 1], dtype=torch.float64)
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_estimating_full_probability(self, tol):
        """Test that the probability of a subset of wires is accurately estimated."""
        dev = qml.device("default.qubit.torch", wires=2, shots=1000)

        @qml.qnode(dev, diff_method=None, interface="torch")
        def circuit():
            qml.PauliX(0)
            qml.PauliX(1)
            return qml.probs(wires=[0, 1])

        res = circuit()

        assert torch.is_tensor(res)

        expected = torch.tensor([0, 0, 0, 1], dtype=torch.float64)
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_estimating_expectation_values(self, tol):
        """Test that estimating expectation values using a finite number
        of shots produces a numeric tensor"""
        dev = qml.device("default.qubit.torch", wires=3, shots=1000)

        @qml.qnode(dev, diff_method=None, interface="torch")
        def circuit(a, b):
            qml.RX(a, wires=[0])
            qml.RX(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        a = torch.tensor(0.543, dtype=torch.float64)
        b = torch.tensor(0.43, dtype=torch.float64)

        res = circuit(a, b)
        assert torch.is_tensor(res)

        # We don't check the expected value due to stochasticity, but
        # leave it here for completeness.
        # expected = [torch.cos(a), torch.cos(a) * torch.cos(b)]
        # assert np.allclose(res, expected, atol=tol, rtol=0)


class TestHighLevelIntegration:
    """Tests for integration with higher level components of PennyLane."""

    def test_qnode_collection_integration(self):
        """Test that a PassthruQNode default.qubit.torch works with QNodeCollections."""
        dev = qml.device("default.qubit.torch", wires=2)

        obs_list = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1)]
        qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev, interface="torch")

        assert qnodes.interface == "torch"

        torch.manual_seed(42)
        weights = torch.rand(
            qml.templates.StronglyEntanglingLayers.shape(n_wires=2, n_layers=2), requires_grad=True
        )

        def cost(weights):
            return torch.sum(qnodes(weights))

        res = cost(weights)
        res.backward()

        grad = weights.grad

        assert torch.is_tensor(res)
        assert grad.shape == weights.shape

    def test_sampling_analytic_mode(self):
        """Test that when sampling with shots=None, dev uses 1000 shots and
        raises an error.
        """
        dev = qml.device("default.qubit.torch", wires=1, shots=None)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit():
            return qml.sample(qml.PauliZ(wires=0))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The number of shots has to be explicitly set on the device",
        ):
            res = circuit()
