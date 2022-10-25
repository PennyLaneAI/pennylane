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
Unit tests for the :mod:`pennylane.devices.DefaultMixed` device.
"""

import numpy as np
import pytest

import pennylane as qml
from pennylane import BasisState, DeviceError, QubitStateVector
from pennylane.devices import DefaultMixed
from pennylane.ops import (
    CNOT,
    CZ,
    AmplitudeDamping,
    DepolarizingChannel,
    Hadamard,
    Identity,
    PauliError,
    PauliX,
    PauliZ,
    ResetError,
)
from pennylane.wires import Wires

INV_SQRT2 = 1 / np.sqrt(2)


# functions for creating different states used in testing
def basis_state(index, nr_wires):
    rho = np.zeros((2**nr_wires, 2**nr_wires), dtype=np.complex128)
    rho[index, index] = 1
    return rho


def hadamard_state(nr_wires):
    """Equal superposition state (Hadamard on all qubits)"""
    return np.ones((2**nr_wires, 2**nr_wires), dtype=np.complex128) / (2**nr_wires)


def max_mixed_state(nr_wires):
    return np.eye(2**nr_wires, dtype=np.complex128) / (2**nr_wires)


def root_state(nr_wires):
    """Pure state with equal amplitudes but phases equal to roots of unity"""
    dim = 2**nr_wires
    ket = [np.exp(1j * 2 * np.pi * n / dim) / np.sqrt(dim) for n in range(dim)]
    return np.outer(ket, np.conj(ket))


@pytest.mark.parametrize("nr_wires", [1, 2, 3])
class TestCreateBasisState:
    """Unit tests for the method `_create_basis_state()`"""

    def test_shape(self, nr_wires):
        """Tests that the basis state has the correct shape"""
        dev = qml.device("default.mixed", wires=nr_wires)

        assert [2] * (2 * nr_wires) == list(np.shape(dev._create_basis_state(0)))

    @pytest.mark.parametrize("index", [0, 1])
    def test_expected_state(self, nr_wires, index, tol):
        """Tests output basis state against the expected one"""
        rho = np.zeros((2**nr_wires, 2**nr_wires))
        rho[index, index] = 1
        rho = np.reshape(rho, [2] * (2 * nr_wires))
        dev = qml.device("default.mixed", wires=nr_wires)

        assert np.allclose(rho, dev._create_basis_state(index), atol=tol, rtol=0)


@pytest.mark.parametrize("nr_wires", [2, 3])
class TestState:
    """Tests for the method `state()`, which retrieves the state of the system"""

    def test_shape(self, nr_wires):
        """Tests that the state has the correct shape"""
        dev = qml.device("default.mixed", wires=nr_wires)

        assert (2**nr_wires, 2**nr_wires) == np.shape(dev.state)

    def test_init_state(self, nr_wires, tol):
        """Tests that the state is |0...0><0...0| after initialization of the device"""
        rho = np.zeros((2**nr_wires, 2**nr_wires))
        rho[0, 0] = 1
        dev = qml.device("default.mixed", wires=nr_wires)

        assert np.allclose(rho, dev.state, atol=tol, rtol=0)

    @pytest.mark.parametrize("op", [CNOT, CZ])
    def test_state_after_twoqubit(self, nr_wires, op, tol):
        """Tests that state is correctly retrieved after applying two-qubit operations on the
        first wires"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([op(wires=[0, 1])])
        current_state = np.reshape(dev._state, (2**nr_wires, 2**nr_wires))

        assert np.allclose(dev.state, current_state, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "op",
        [
            AmplitudeDamping(0.5, wires=0),
            DepolarizingChannel(0.5, wires=0),
            ResetError(0.1, 0.5, wires=0),
            PauliError("X", 0.5, wires=0),
            PauliError("ZY", 0.3, wires=[1, 0]),
        ],
    )
    def test_state_after_channel(self, nr_wires, op, tol):
        """Tests that state is correctly retrieved after applying a channel on the first wires"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([op])
        current_state = np.reshape(dev._state, (2**nr_wires, 2**nr_wires))

        assert np.allclose(dev.state, current_state, atol=tol, rtol=0)

    @pytest.mark.parametrize("op", [PauliX, PauliZ, Hadamard])
    def test_state_after_gate(self, nr_wires, op, tol):
        """Tests that state is correctly retrieved after applying operations on the first wires"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([op(wires=0)])
        current_state = np.reshape(dev._state, (2**nr_wires, 2**nr_wires))

        assert np.allclose(dev.state, current_state, atol=tol, rtol=0)


@pytest.mark.parametrize("nr_wires", [2, 3])
class TestReset:
    """Unit tests for the method `reset()`"""

    def test_reset_basis(self, nr_wires, tol):
        dev = qml.device("default.mixed", wires=nr_wires)
        dev._state = dev._create_basis_state(1)
        dev.reset()

        assert np.allclose(dev._state, dev._create_basis_state(0), atol=tol, rtol=0)

    @pytest.mark.parametrize("op", [CNOT, CZ])
    def test_reset_after_twoqubit(self, nr_wires, op, tol):
        """Tests that state is correctly reset after applying two-qubit operations on the first
        wires"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([op(wires=[0, 1])])
        dev.reset()

        assert np.allclose(dev._state, dev._create_basis_state(0), atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "op",
        [
            AmplitudeDamping(0.5, wires=[0]),
            DepolarizingChannel(0.5, wires=[0]),
            ResetError(0.1, 0.5, wires=[0]),
            PauliError("X", 0.5, wires=0),
            PauliError("ZY", 0.3, wires=[1, 0]),
        ],
    )
    def test_reset_after_channel(self, nr_wires, op, tol):
        """Tests that state is correctly reset after applying a channel on the first
        wires"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([op])
        dev.reset()

        assert np.allclose(dev._state, dev._create_basis_state(0), atol=tol, rtol=0)

    @pytest.mark.parametrize("op", [PauliX, PauliZ, Hadamard])
    def test_reset_after_channel(self, nr_wires, op, tol):
        """Tests that state is correctly reset after applying gates on the first
        wire"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([op(wires=0)])
        dev.reset()

        assert np.allclose(dev._state, dev._create_basis_state(0), atol=tol, rtol=0)


@pytest.mark.parametrize("nr_wires", [1, 2, 3])
class TestAnalyticProb:
    """Unit tests for the method `analytic_probability()`"""

    def test_prob_init_state(self, nr_wires, tol):
        """Tests that we obtain the correct probabilities for the state |0...0><0...0|"""
        dev = qml.device("default.mixed", wires=nr_wires)
        probs = np.zeros(2**nr_wires)
        probs[0] = 1

        assert np.allclose(probs, dev.analytic_probability(), atol=tol, rtol=0)

    def test_prob_basis_state(self, nr_wires, tol):
        """Tests that we obtain correct probabilities for the basis state |1...1><1...1|"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev._state = dev._create_basis_state(2**nr_wires - 1)
        probs = np.zeros(2**nr_wires)
        probs[-1] = 1

        assert np.allclose(probs, dev.analytic_probability(), atol=tol, rtol=0)

    def test_prob_hadamard(self, nr_wires, tol):
        """Tests that we obtain correct probabilities for the equal superposition state"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev._state = hadamard_state(nr_wires)
        probs = np.ones(2**nr_wires) / (2**nr_wires)

        assert np.allclose(probs, dev.analytic_probability(), atol=tol, rtol=0)

    def test_prob_mixed(self, nr_wires, tol):
        """Tests that we obtain correct probabilities for the maximally mixed state"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev._state = max_mixed_state(nr_wires)
        probs = np.ones(2**nr_wires) / (2**nr_wires)

        assert np.allclose(probs, dev.analytic_probability(), atol=tol, rtol=0)

    def test_prob_root(self, nr_wires, tol):
        """Tests that we obtain correct probabilities for the root state"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev._state = root_state(nr_wires)
        probs = np.ones(2**nr_wires) / (2**nr_wires)

        assert np.allclose(probs, dev.analytic_probability(), atol=tol, rtol=0)

    def test_none_state(self, nr_wires):
        """Tests that return is `None` when the state is `None`"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev._state = None

        assert dev.analytic_probability() is None

    def test_probability_not_negative(self, nr_wires):
        """Test that probabilities are always real"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev._state = np.zeros([2**nr_wires, 2**nr_wires])
        dev._state[0, 0] = 1
        dev._state[1, 1] = -5e-17

        assert np.all(dev.analytic_probability() >= 0)


class TestKrausOps:
    """Unit tests for the method `_get_kraus_ops()`"""

    unitary_ops = [
        (PauliX(wires=0), np.array([[0, 1], [1, 0]])),
        (Hadamard(wires=0), np.array([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]])),
        (CNOT(wires=[0, 1]), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])),
        (
            PauliError("X", 0.5, wires=0),
            [np.sqrt(0.5) * np.eye(2), np.sqrt(0.5) * np.array([[0, 1], [1, 0]])],
        ),
        (
            PauliError("Y", 0.3, wires=0),
            [np.sqrt(0.7) * np.eye(2), np.sqrt(0.3) * np.array([[0, -1j], [1j, 0]])],
        ),
    ]

    @pytest.mark.parametrize("ops", unitary_ops)
    def test_unitary_kraus(self, ops, tol):
        """Tests that matrices of non-diagonal unitary operations are retrieved correctly"""
        dev = qml.device("default.mixed", wires=2)

        assert np.allclose(dev._get_kraus(ops[0]), [ops[1]], atol=tol, rtol=0)

    diagonal_ops = [
        (PauliZ(wires=0), np.array([1, -1])),
        (CZ(wires=[0, 1]), np.array([1, 1, 1, -1])),
    ]

    @pytest.mark.parametrize("ops", diagonal_ops)
    def test_diagonal_kraus(self, ops, tol):
        """Tests that matrices of diagonal unitary operations are retrieved correctly"""
        dev = qml.device("default.mixed", wires=2)

        assert np.allclose(dev._get_kraus(ops[0]), ops[1], atol=tol, rtol=0)

    p = 0.5
    p_0, p_1 = 0.1, 0.5

    channel_ops = [
        (
            AmplitudeDamping(p, wires=0),
            [np.diag([1, np.sqrt(1 - p)]), np.sqrt(p) * np.array([[0, 1], [0, 0]])],
        ),
        (
            DepolarizingChannel(p, wires=0),
            [
                np.sqrt(1 - p) * np.eye(2),
                np.sqrt(p / 3) * np.array([[0, 1], [1, 0]]),
                np.sqrt(p / 3) * np.array([[0, -1j], [1j, 0]]),
                np.sqrt(p / 3) * np.array([[1, 0], [0, -1]]),
            ],
        ),
        (
            ResetError(p_0, p_1, wires=0),
            [
                np.sqrt(1 - p_0 - p_1) * np.eye(2),
                np.sqrt(p_0) * np.array([[1, 0], [0, 0]]),
                np.sqrt(p_0) * np.array([[0, 1], [0, 0]]),
                np.sqrt(p_1) * np.array([[0, 0], [1, 0]]),
                np.sqrt(p_1) * np.array([[0, 0], [0, 1]]),
            ],
        ),
        (
            PauliError("X", p_0, wires=0),
            [np.sqrt(1 - p_0) * np.eye(2), np.sqrt(p_0) * np.array([[0, 1], [1, 0]])],
        ),
    ]

    @pytest.mark.parametrize("ops", channel_ops)
    def test_channel_kraus(self, ops, tol):
        """Tests that kraus matrices of non-unitary channels are retrieved correctly"""
        dev = qml.device("default.mixed", wires=1)

        assert np.allclose(dev._get_kraus(ops[0]), ops[1], atol=tol, rtol=0)


class TestApplyChannel:
    """Unit tests for the method `_apply_channel()`"""

    x_apply_channel_init = [
        [1, PauliX(wires=0), basis_state(1, 1)],
        [1, Hadamard(wires=0), np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]])],
        [2, CNOT(wires=[0, 1]), basis_state(0, 2)],
        [1, AmplitudeDamping(0.5, wires=0), basis_state(0, 1)],
        [
            1,
            DepolarizingChannel(0.5, wires=0),
            np.array([[2 / 3 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1 / 3 + 0.0j]]),
        ],
        [
            1,
            ResetError(0.1, 0.5, wires=0),
            np.array([[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.5 + 0.0j]]),
        ],
        [1, PauliError("Z", 0.3, wires=0), basis_state(0, 1)],
        [2, PauliError("XY", 0.5, wires=[0, 1]), 0.5 * basis_state(0, 2) + 0.5 * basis_state(3, 2)],
    ]

    @pytest.mark.parametrize("x", x_apply_channel_init)
    def test_channel_init(self, x, tol):
        """Tests that channels are correctly applied to the default initial state"""
        nr_wires = x[0]
        op = x[1]
        target_state = np.reshape(x[2], [2] * 2 * nr_wires)
        dev = qml.device("default.mixed", wires=nr_wires)
        kraus = dev._get_kraus(op)
        if op.name == "CNOT" or (isinstance(op, PauliError) and nr_wires == 2):
            dev._apply_channel(kraus, wires=Wires([0, 1]))
        else:
            dev._apply_channel(kraus, wires=Wires(0))

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)

    x_apply_channel_mixed = [
        [1, PauliX(wires=0), max_mixed_state(1)],
        [2, Hadamard(wires=0), max_mixed_state(2)],
        [2, CNOT(wires=[0, 1]), max_mixed_state(2)],
        [
            1,
            AmplitudeDamping(0.5, wires=0),
            np.array([[0.75 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.25 + 0.0j]]),
        ],
        [
            1,
            DepolarizingChannel(0.5, wires=0),
            np.array([[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.5 + 0.0j]]),
        ],
        [
            1,
            ResetError(0.1, 0.5, wires=0),
            np.array([[0.3 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.7 + 0.0j]]),
        ],
        [1, PauliError("Z", 0.3, wires=0), max_mixed_state(1)],
        [2, PauliError("XY", 0.5, wires=[0, 1]), max_mixed_state(2)],
    ]

    @pytest.mark.parametrize("x", x_apply_channel_mixed)
    def test_channel_mixed(self, x, tol):
        """Tests that channels are correctly applied to the maximally mixed state"""
        nr_wires = x[0]
        op = x[1]
        target_state = np.reshape(x[2], [2] * 2 * nr_wires)
        dev = qml.device("default.mixed", wires=nr_wires)
        max_mixed = np.reshape(max_mixed_state(nr_wires), [2] * 2 * nr_wires)
        dev._state = max_mixed
        kraus = dev._get_kraus(op)
        if op.name == "CNOT" or (isinstance(op, PauliError) and nr_wires == 2):
            dev._apply_channel(kraus, wires=Wires([0, 1]))
        else:
            dev._apply_channel(kraus, wires=Wires(0))

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)

    x_apply_channel_root = [
        [1, PauliX(wires=0), np.array([[0.5 + 0.0j, -0.5 + 0.0j], [-0.5 - 0.0j, 0.5 + 0.0j]])],
        [1, Hadamard(wires=0), np.array([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]])],
        [
            2,
            CNOT(wires=[0, 1]),
            np.array(
                [
                    [0.25 + 0.0j, 0.0 - 0.25j, 0.0 + 0.25j, -0.25],
                    [0.0 + 0.25j, 0.25 + 0.0j, -0.25 + 0.0j, 0.0 - 0.25j],
                    [0.0 - 0.25j, -0.25 + 0.0j, 0.25 + 0.0j, 0.0 + 0.25j],
                    [-0.25 + 0.0j, 0.0 + 0.25j, 0.0 - 0.25j, 0.25 + 0.0j],
                ]
            ),
        ],
        [
            1,
            AmplitudeDamping(0.5, wires=0),
            np.array([[0.75 + 0.0j, -0.35355339 - 0.0j], [-0.35355339 + 0.0j, 0.25 + 0.0j]]),
        ],
        [
            1,
            DepolarizingChannel(0.5, wires=0),
            np.array([[0.5 + 0.0j, -1 / 6 + 0.0j], [-1 / 6 + 0.0j, 0.5 + 0.0j]]),
        ],
        [
            1,
            ResetError(0.1, 0.5, wires=0),
            np.array([[0.3 + 0.0j, -0.2 + 0.0j], [-0.2 + 0.0j, 0.7 + 0.0j]]),
        ],
        [
            1,
            PauliError("Z", 0.3, wires=0),
            np.array([[0.5 + 0.0j, -0.2 + 0.0j], [-0.2 + 0.0j, 0.5 + 0.0j]]),
        ],
        [
            2,
            PauliError("XY", 0.5, wires=[0, 1]),
            np.array(
                [
                    [0.25 + 0.0j, 0.0 - 0.25j, -0.25 + 0.0j, 0.0 + 0.25j],
                    [0.0 + 0.25j, 0.25 + 0.0j, 0.0 - 0.25j, -0.25 + 0.0j],
                    [-0.25 + 0.0j, 0.0 + 0.25j, 0.25 + 0.0j, 0.0 - 0.25j],
                    [0.0 - 0.25j, -0.25 + 0.0j, 0.0 + 0.25j, 0.25 + 0.0j],
                ]
            ),
        ],
    ]

    @pytest.mark.parametrize("x", x_apply_channel_root)
    def test_channel_root(self, x, tol):
        """Tests that channels are correctly applied to root state"""
        nr_wires = x[0]
        op = x[1]
        target_state = np.reshape(x[2], [2] * 2 * nr_wires)
        dev = qml.device("default.mixed", wires=nr_wires)
        root = np.reshape(root_state(nr_wires), [2] * 2 * nr_wires)
        dev._state = root
        kraus = dev._get_kraus(op)
        if op.name == "CNOT" or (isinstance(op, PauliError) and nr_wires == 2):
            dev._apply_channel(kraus, wires=Wires([0, 1]))
        else:
            dev._apply_channel(kraus, wires=Wires(0))

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)


class TestApplyDiagonal:
    """Unit tests for the method `_apply_diagonal_unitary()`"""

    x_apply_diag_init = [
        [1, PauliZ(0), basis_state(0, 1)],
        [2, CZ(wires=[0, 1]), basis_state(0, 2)],
    ]

    @pytest.mark.parametrize("x", x_apply_diag_init)
    def test_diag_init(self, x, tol):
        """Tests that diagonal gates are correctly applied to the default initial state"""
        nr_wires = x[0]
        op = x[1]
        target_state = np.reshape(x[2], [2] * 2 * nr_wires)
        dev = qml.device("default.mixed", wires=nr_wires)
        kraus = dev._get_kraus(op)
        if op.name == "CZ":
            dev._apply_diagonal_unitary(kraus, wires=Wires([0, 1]))
        else:
            dev._apply_diagonal_unitary(kraus, wires=Wires(0))

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)

    x_apply_diag_mixed = [
        [1, PauliZ(0), max_mixed_state(1)],
        [2, CZ(wires=[0, 1]), max_mixed_state(2)],
    ]

    @pytest.mark.parametrize("x", x_apply_diag_mixed)
    def test_diag_mixed(self, x, tol):
        """Tests that diagonal gates are correctly applied to the maximally mixed state"""
        nr_wires = x[0]
        op = x[1]
        target_state = np.reshape(x[2], [2] * 2 * nr_wires)
        dev = qml.device("default.mixed", wires=nr_wires)
        max_mixed = np.reshape(max_mixed_state(nr_wires), [2] * 2 * nr_wires)
        dev._state = max_mixed
        kraus = dev._get_kraus(op)
        if op.name == "CZ":
            dev._apply_diagonal_unitary(kraus, wires=Wires([0, 1]))
        else:
            dev._apply_diagonal_unitary(kraus, wires=Wires(0))

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)

    x_apply_diag_root = [
        [1, PauliZ(0), np.array([[0.5, 0.5], [0.5, 0.5]])],
        [
            2,
            CZ(wires=[0, 1]),
            np.array(
                [
                    [0.25, -0.25j, -0.25, -0.25j],
                    [0.25j, 0.25, -0.25j, 0.25],
                    [-0.25, 0.25j, 0.25, 0.25j],
                    [0.25j, 0.25, -0.25j, 0.25],
                ]
            ),
        ],
    ]

    @pytest.mark.parametrize("x", x_apply_diag_root)
    def test_diag_root(self, x, tol):
        """Tests that diagonal gates are correctly applied to root state"""
        nr_wires = x[0]
        op = x[1]
        target_state = np.reshape(x[2], [2] * 2 * nr_wires)
        dev = qml.device("default.mixed", wires=nr_wires)
        root = np.reshape(root_state(nr_wires), [2] * 2 * nr_wires)
        dev._state = root
        kraus = dev._get_kraus(op)
        if op.name == "CZ":
            dev._apply_diagonal_unitary(kraus, wires=Wires([0, 1]))
        else:
            dev._apply_diagonal_unitary(kraus, wires=Wires(0))

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)


class TestApplyBasisState:
    """Unit tests for the method `_apply_basis_state"""

    @pytest.mark.parametrize("nr_wires", [1, 2, 3])
    def test_all_ones(self, nr_wires, tol):
        """Tests that the state |11...1> is applied correctly"""
        dev = qml.device("default.mixed", wires=nr_wires)
        state = np.ones(nr_wires)
        dev._apply_basis_state(state, wires=Wires(range(nr_wires)))
        b_state = basis_state(2**nr_wires - 1, nr_wires)
        target_state = np.reshape(b_state, [2] * 2 * nr_wires)

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)

    fixed_states = [[3, np.array([0, 1, 1])], [5, np.array([1, 0, 1])], [6, np.array([1, 1, 0])]]

    @pytest.mark.parametrize("state", fixed_states)
    def test_fixed_states(self, state, tol):
        """Tests that different basis states are applied correctly"""
        nr_wires = 3
        dev = qml.device("default.mixed", wires=nr_wires)
        dev._apply_basis_state(state[1], wires=Wires(range(nr_wires)))
        b_state = basis_state(state[0], nr_wires)
        target_state = np.reshape(b_state, [2] * 2 * nr_wires)

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)

    wire_subset = [(6, [0, 1]), (5, [0, 2]), (3, [1, 2])]

    @pytest.mark.parametrize("wires", wire_subset)
    def test_subset_wires(self, wires, tol):
        """Tests that different basis states are applied correctly when applied to a subset of
        wires"""
        nr_wires = 3
        dev = qml.device("default.mixed", wires=nr_wires)
        state = np.ones(2)
        dev._apply_basis_state(state, wires=Wires(wires[1]))
        b_state = basis_state(wires[0], nr_wires)
        target_state = np.reshape(b_state, [2] * 2 * nr_wires)

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)

    def test_wrong_dim(self):
        """Checks that an error is raised if state has the wrong dimension"""
        dev = qml.device("default.mixed", wires=3)
        state = np.ones(2)
        with pytest.raises(ValueError, match="BasisState parameter and wires"):
            dev._apply_basis_state(state, wires=Wires(range(3)))

    def test_not_01(self):
        """Checks that an error is raised if state doesn't have entries in {0,1}"""
        dev = qml.device("default.mixed", wires=2)
        state = np.array([INV_SQRT2, INV_SQRT2])
        with pytest.raises(ValueError, match="BasisState parameter must"):
            dev._apply_basis_state(state, wires=Wires(range(2)))


class TestApplyStateVector:
    """Unit tests for the method `_apply_state_vector()`"""

    @pytest.mark.parametrize("nr_wires", [1, 2, 3])
    def test_apply_equal(self, nr_wires, tol):
        """Checks that an equal superposition state is correctly applied"""
        dev = qml.device("default.mixed", wires=nr_wires)
        state = np.ones(2**nr_wires) / np.sqrt(2**nr_wires)
        dev._apply_state_vector(state, Wires(range(nr_wires)))
        eq_state = hadamard_state(nr_wires)
        target_state = np.reshape(eq_state, [2] * 2 * nr_wires)

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)

    @pytest.mark.parametrize("nr_wires", [1, 2, 3])
    def test_apply_root(self, nr_wires, tol):
        """Checks that a root state is correctly applied"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dim = 2**nr_wires
        state = np.array([np.exp(1j * 2 * np.pi * n / dim) / np.sqrt(dim) for n in range(dim)])
        dev._apply_state_vector(state, Wires(range(nr_wires)))
        r_state = root_state(nr_wires)
        target_state = np.reshape(r_state, [2] * 2 * nr_wires)

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)

    subset_wires = [(4, 0), (2, 1), (1, 2)]

    @pytest.mark.parametrize("wires", subset_wires)
    def test_subset_wires(self, wires, tol):
        """Tests that applying state |1> on each individual single wire prepares the correct basis
        state"""
        nr_wires = 3
        dev = qml.device("default.mixed", wires=nr_wires)
        state = np.array([0, 1])
        dev._apply_state_vector(state, Wires(wires[1]))
        b_state = basis_state(wires[0], nr_wires)
        target_state = np.reshape(b_state, [2] * 2 * nr_wires)

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)

    def test_wrong_dim(self):
        """Checks that an error is raised if state has the wrong dimension"""
        dev = qml.device("default.mixed", wires=3)
        state = np.ones(7) / np.sqrt(7)
        with pytest.raises(ValueError, match="State vector must be"):
            dev._apply_state_vector(state, Wires(range(3)))

    def test_not_normalized(self):
        """Checks that an error is raised if state is not normalized"""
        dev = qml.device("default.mixed", wires=3)
        state = np.ones(8) / np.sqrt(7)
        with pytest.raises(ValueError, match="Sum of amplitudes"):
            dev._apply_state_vector(state, Wires(range(3)))

    def test_wires_as_list(self, tol):
        """Checks that state is correctly prepared when device wires are given as a list,
        not a number. This test helps with coverage"""
        nr_wires = 2
        dev = qml.device("default.mixed", wires=[0, 1])
        state = np.ones(2**nr_wires) / np.sqrt(2**nr_wires)
        dev._apply_state_vector(state, Wires(range(nr_wires)))
        eq_state = hadamard_state(nr_wires)
        target_state = np.reshape(eq_state, [2] * 2 * nr_wires)

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)


class TestApplyDensityMatrix:
    """Unit tests for the method `_apply_density_matrix()`"""

    def test_instantiate_density_mat(self, tol):
        """Checks that the specific density matrix is initialized"""
        dev = qml.device("default.mixed", wires=2)
        initialize_state = basis_state(1, 2)

        @qml.qnode(dev)
        def circuit():
            qml.QubitDensityMatrix(initialize_state, wires=[0, 1])
            return qml.state()

        final_state = circuit()
        assert np.allclose(final_state, initialize_state, atol=tol, rtol=0)

    @pytest.mark.parametrize("nr_wires", [1, 2, 3])
    def test_apply_equal(self, nr_wires, tol):
        """Checks that an equal superposition state is correctly applied"""
        dev = qml.device("default.mixed", wires=nr_wires)
        state = np.ones(2**nr_wires) / np.sqrt(2**nr_wires)
        rho = np.outer(state, state.conj())
        dev._apply_density_matrix(rho, Wires(range(nr_wires)))
        eq_state = hadamard_state(nr_wires)
        target_state = np.reshape(eq_state, [2] * 2 * nr_wires)

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)

    @pytest.mark.parametrize("nr_wires", [1, 2, 3])
    def test_apply_root(self, nr_wires, tol):
        """Checks that a root state is correctly applied"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dim = 2**nr_wires
        state = np.array([np.exp(1j * 2 * np.pi * n / dim) / np.sqrt(dim) for n in range(dim)])
        rho = np.outer(state, state.conj())
        dev._apply_density_matrix(rho, Wires(range(nr_wires)))
        r_state = root_state(nr_wires)
        target_state = np.reshape(r_state, [2] * 2 * nr_wires)

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)

    subset_wires = [(4, 0), (2, 1), (1, 2)]

    @pytest.mark.parametrize("wires", subset_wires)
    def test_subset_wires_with_filling_remaining(self, wires, tol):
        """Tests that applying state |1><1| on a subset of wires prepares the correct state
        |1><1| ⊗ |0><0|"""
        nr_wires = 3
        dev = qml.device("default.mixed", wires=nr_wires)
        state = np.array([0, 1])
        rho = np.outer(state, state.conj())
        dev._apply_density_matrix(rho, Wires(wires[1]))
        b_state = basis_state(wires[0], nr_wires)
        target_state = np.reshape(b_state, [2] * 2 * nr_wires)
        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)

    subset_wires = [(7, (0, 1, 2), ()), (5, (0, 2), (1,)), (6, (0, 1), (2,))]

    @pytest.mark.parametrize("wires", subset_wires)
    def test_subset_wires_without_filling_remaining(self, wires, tol):
        """Tests that does nothing |1><1| on a subset of wires prepares the correct state
        |1><1| ⊗ ρ if `fill_remaining=False`"""
        nr_wires = 3
        dev = qml.device("default.mixed", wires=nr_wires)
        state0 = np.array([1, 0])
        rho0 = np.outer(state0, state0.conj())
        state1 = np.array([0, 1])
        rho1 = np.outer(state1, state1.conj())
        for wire in wires[1]:
            dev._apply_density_matrix(rho1, Wires(wire))
        for wire in wires[2]:
            dev._apply_density_matrix(rho0, Wires(wire))
        b_state = basis_state(wires[0], nr_wires)
        target_state = np.reshape(b_state, [2] * 2 * nr_wires)
        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)

    def test_wrong_dim(self):
        """Checks that an error is raised if state has the wrong dimension"""
        dev = qml.device("default.mixed", wires=3)
        state = np.ones(7) / np.sqrt(7)
        rho = np.outer(state, state.conj())
        with pytest.raises(ValueError, match="Density matrix must be"):
            dev._apply_density_matrix(rho, Wires(range(3)))

    def test_not_normalized(self):
        """Checks that an error is raised if state is not normalized"""
        dev = qml.device("default.mixed", wires=3)
        state = np.ones(8) / np.sqrt(7)
        rho = np.outer(state, state.conj())
        with pytest.raises(ValueError, match="Trace of density matrix"):
            dev._apply_density_matrix(rho, Wires(range(3)))

    def test_wires_as_list(self, tol):
        """Checks that state is correctly prepared when device wires are given as a list,
        not a number. This test helps with coverage"""
        nr_wires = 2
        dev = qml.device("default.mixed", wires=[0, 1])
        state = np.ones(2**nr_wires) / np.sqrt(2**nr_wires)
        rho = np.outer(state, state.conj())
        dev._apply_density_matrix(rho, Wires(range(nr_wires)))
        eq_state = hadamard_state(nr_wires)
        target_state = np.reshape(eq_state, [2] * 2 * nr_wires)

        assert np.allclose(dev._state, target_state, atol=tol, rtol=0)


class TestApplyOperation:
    """Unit tests for the method `_apply_operation()`. Since this just calls `_apply_channel()`
    and `_apply_diagonal_unitary()`, we just check that the correct method is called"""

    def test_diag_apply_op(self, mocker):
        """Tests that when applying a diagonal gate, only `_apply_diagonal_unitary` is called,
        exactly once"""
        spy_channel = mocker.spy(DefaultMixed, "_apply_channel")
        spy_diag = mocker.spy(DefaultMixed, "_apply_diagonal_unitary")
        dev = qml.device("default.mixed", wires=1)
        dev._apply_operation(PauliZ(0))

        spy_channel.assert_not_called()
        spy_diag.assert_called_once()

    def test_channel_apply_op(self, mocker):
        """Tests that when applying a non-diagonal gate, only `_apply_channel` is called,
        exactly once"""
        spy_channel = mocker.spy(DefaultMixed, "_apply_channel")
        spy_diag = mocker.spy(DefaultMixed, "_apply_diagonal_unitary")
        dev = qml.device("default.mixed", wires=1)
        dev._apply_operation(PauliX(0))

        spy_diag.assert_not_called()
        spy_channel.assert_called_once()

    def test_identity_skipped(self, mocker):
        """Test that applying the identity does not perform any additional computations."""

        op = qml.Identity(0)
        dev = qml.device("default.mixed", wires=1)

        spy_diagonal_unitary = mocker.spy(dev, "_apply_diagonal_unitary")
        spy_apply_channel = mocker.spy(dev, "_apply_channel")

        initialstate = dev.state.__copy__()

        dev._apply_operation(op)

        assert qml.math.allclose(dev.state, initialstate)

        spy_diagonal_unitary.assert_not_called()
        spy_apply_channel.assert_not_called()


class TestApply:
    """Unit tests for the main method `apply()`. We check that lists of operations are applied
    correctly, rather than single operations"""

    ops_and_true_state = [(None, basis_state(0, 2)), (Hadamard, hadamard_state(2))]

    @pytest.mark.parametrize("op, true_state", ops_and_true_state)
    def test_identity(self, op, true_state, tol):
        """Tests that applying the identity operator doesn't change the state"""
        num_wires = 2
        dev = qml.device("default.mixed", wires=num_wires)  # prepare basis state

        if op is not None:
            ops = [op(i) for i in range(num_wires)]
            dev.apply(ops)

        # Apply Identity:
        dev.apply([Identity(i) for i in range(num_wires)])

        assert np.allclose(dev.state, true_state, atol=tol, rtol=0)

    def test_bell_state(self, tol):
        """Tests that we correctly prepare a Bell state by applying a Hadamard then a CNOT"""
        dev = qml.device("default.mixed", wires=2)
        ops = [Hadamard(0), CNOT(wires=[0, 1])]
        dev.apply(ops)
        bell = np.zeros((4, 4))
        bell[0, 0] = bell[0, 3] = bell[3, 0] = bell[3, 3] = 1 / 2

        assert np.allclose(bell, dev.state, atol=tol, rtol=0)

    @pytest.mark.parametrize("nr_wires", [1, 2, 3])
    def test_hadamard_state(self, nr_wires, tol):
        """Tests that applying Hadamard gates on all qubits produces an equal superposition over
        all basis states"""
        dev = qml.device("default.mixed", wires=nr_wires)
        ops = [Hadamard(i) for i in range(nr_wires)]
        dev.apply(ops)

        assert np.allclose(dev.state, hadamard_state(nr_wires), atol=tol, rtol=0)

    @pytest.mark.parametrize("nr_wires", [1, 2, 3])
    def test_max_mixed_state(self, nr_wires, tol):
        """Tests that applying damping channel on all qubits to the state |11...1> produces a
        maximally mixed state"""
        dev = qml.device("default.mixed", wires=nr_wires)
        flips = [PauliX(i) for i in range(nr_wires)]
        damps = [AmplitudeDamping(0.5, wires=i) for i in range(nr_wires)]
        ops = flips + damps
        dev.apply(ops)

        assert np.allclose(dev.state, max_mixed_state(nr_wires), atol=tol, rtol=0)

    @pytest.mark.parametrize("nr_wires", [1, 2, 3])
    def test_undo_rotations(self, nr_wires, tol):
        """Tests that rotations are correctly applied by adding their inverse as initial
        operations"""
        dev = qml.device("default.mixed", wires=nr_wires)
        ops = [Hadamard(i) for i in range(nr_wires)]
        rots = ops
        dev.apply(ops, rots)
        basis = np.reshape(basis_state(0, nr_wires), [2] * (2 * nr_wires))
        # dev.state = pre-rotated state, dev._state = state after rotations
        assert np.allclose(dev.state, hadamard_state(nr_wires), atol=tol, rtol=0)
        assert np.allclose(dev._state, basis, atol=tol, rtol=0)

    @pytest.mark.parametrize("nr_wires", [1, 2, 3])
    def test_apply_basis_state(self, nr_wires, tol):
        """Tests that we correctly apply a `BasisState` operation for the |11...1> state"""
        dev = qml.device("default.mixed", wires=nr_wires)
        state = np.ones(nr_wires)
        dev.apply([BasisState(state, wires=range(nr_wires))])

        assert np.allclose(dev.state, basis_state(2**nr_wires - 1, nr_wires), atol=tol, rtol=0)

    @pytest.mark.parametrize("nr_wires", [1, 2, 3])
    def test_apply_state_vector(self, nr_wires, tol):
        """Tests that we correctly apply a `QubitStateVector` operation for the root state"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dim = 2**nr_wires
        state = np.array([np.exp(1j * 2 * np.pi * n / dim) / np.sqrt(dim) for n in range(dim)])
        dev.apply([QubitStateVector(state, wires=range(nr_wires))])

        assert np.allclose(dev.state, root_state(nr_wires), atol=tol, rtol=0)

    def test_apply_state_vector_wires(self, tol):
        """Tests that we correctly apply a `QubitStateVector` operation for the root state when
        wires are passed as an ordered list"""
        nr_wires = 3
        dev = qml.device("default.mixed", wires=[0, 1, 2])
        dim = 2**nr_wires
        state = np.array([np.exp(1j * 2 * np.pi * n / dim) / np.sqrt(dim) for n in range(dim)])
        dev.apply([QubitStateVector(state, wires=[0, 1, 2])])

        assert np.allclose(dev.state, root_state(nr_wires), atol=tol, rtol=0)

    def test_apply_state_vector_subsystem(self, tol):
        """Tests that we correctly apply a `QubitStateVector` operation when the
        wires passed are a strict subset of the device wires"""
        nr_wires = 2
        dev = qml.device("default.mixed", wires=[0, 1, 2])
        dim = 2**nr_wires
        state = np.array([np.exp(1j * 2 * np.pi * n / dim) / np.sqrt(dim) for n in range(dim)])
        dev.apply([QubitStateVector(state, wires=[0, 1])])

        expected = np.array([1, 0, 1j, 0, -1, 0, -1j, 0]) / 2
        expected = np.outer(expected, np.conj(expected))

        assert np.allclose(dev.state, expected, atol=tol, rtol=0)

    def test_raise_order_error_basis_state(self):
        """Tests that an error is raised if a state is prepared after BasisState has been
        applied"""
        dev = qml.device("default.mixed", wires=1)
        state = np.array([0])
        ops = [PauliX(0), BasisState(state, wires=0)]

        with pytest.raises(DeviceError, match="Operation"):
            dev.apply(ops)

    def test_raise_order_error_qubit_state(self):
        """Tests that an error is raised if a state is prepared after QubitStateVector has been
        applied"""
        dev = qml.device("default.mixed", wires=1)
        state = np.array([1, 0])
        ops = [PauliX(0), QubitStateVector(state, wires=0)]

        with pytest.raises(DeviceError, match="Operation"):
            dev.apply(ops)

    def test_apply_toffoli(self, tol):
        """Tests that Toffoli gate is correctly applied on state |111> to give state |110>"""
        nr_wires = 3
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([PauliX(0), PauliX(1), PauliX(2), qml.Toffoli(wires=[0, 1, 2])])

        assert np.allclose(dev.state, basis_state(6, 3), atol=tol, rtol=0)

    def test_apply_qubitunitary(self, tol):
        """Tests that custom qubit unitary is correctly applied"""
        nr_wires = 1
        theta = 0.42
        U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([qml.QubitUnitary(U, wires=[0])])
        ket = np.array([np.cos(theta) + 0j, np.sin(theta) + 0j])
        target_rho = np.outer(ket, np.conj(ket))

        assert np.allclose(dev.state, target_rho, atol=tol, rtol=0)

    def test_apply_pauli_error(self, tol):
        """Tests that PauliError gate is correctly applied"""
        nr_wires = 3
        p = 0.3
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([PauliError("XYZ", p, wires=[0, 1, 2])])
        target = 0.7 * basis_state(0, 3) + 0.3 * basis_state(6, 3)

        assert np.allclose(dev.state, target, atol=tol, rtol=0)


class TestReadoutError:
    """Tests for measurement readout error"""

    prob_and_expected_expval = [
        (0, np.array([1, 1])),
        (0.5, np.array([0, 0])),
        (1, np.array([-1, -1])),
    ]

    @pytest.mark.parametrize("nr_wires", [2, 3])
    @pytest.mark.parametrize("prob, expected", prob_and_expected_expval)
    def test_readout_expval_pauliz(self, nr_wires, prob, expected):
        """Tests the measurement results for expval of PauliZ"""
        dev = qml.device("default.mixed", wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        res = circuit()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("nr_wires", [2, 3])
    @pytest.mark.parametrize("prob, expected", prob_and_expected_expval)
    def test_readout_expval_paulix(self, nr_wires, prob, expected):
        """Tests the measurement results for expval of PauliX"""
        dev = qml.device("default.mixed", wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        res = circuit()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("prob", [0, 0.5, 1])
    @pytest.mark.parametrize(
        "nr_wires, expected", [(1, np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]))]
    )
    def test_readout_state(self, nr_wires, prob, expected):
        """Tests the state output is not affected by readout error"""
        dev = qml.device("default.mixed", wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            return qml.state()

        res = circuit()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("nr_wires", [2, 3])
    @pytest.mark.parametrize("prob", [0, 0.5, 1])
    def test_readout_density_matrix(self, nr_wires, prob):
        """Tests the density matrix output is not affected by readout error"""
        dev = qml.device("default.mixed", wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            return qml.density_matrix(wires=1)

        res = circuit()
        expected = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]])
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("prob", [0, 0.5, 1])
    @pytest.mark.parametrize("nr_wires", [2, 3])
    def test_readout_vnentropy_and_mutualinfo(self, nr_wires, prob):
        """Tests the output of qml.vn_entropy and qml.mutual_info is not affected by readout error"""
        dev = qml.device("default.mixed", wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            return qml.vn_entropy(wires=0, log_base=2), qml.mutual_info(
                wires0=[0], wires1=[1], log_base=2
            )

        res = circuit()
        expected = np.array([0, 0])
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("nr_wires", [2, 3])
    @pytest.mark.parametrize(
        "prob, expected", [(0, [np.zeros(2), np.zeros(2)]), (1, [np.ones(2), np.ones(2)])]
    )
    def test_readout_sample(self, nr_wires, prob, expected):
        """Tests the sample output with readout error"""
        dev = qml.device("default.mixed", shots=2, wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            return qml.sample(wires=[0, 1])

        res = circuit()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("nr_wires", [2, 3])
    @pytest.mark.parametrize("prob, expected", [(0, {"00": 100}), (1, {"11": 100})])
    def test_readout_counts(self, nr_wires, prob, expected):
        """Tests the counts output with readout error"""
        dev = qml.device("default.mixed", shots=100, wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            return qml.counts(wires=[0, 1])

        res = circuit()
        assert res == expected

    prob_and_expected_probs = [
        (0, np.array([1, 0])),
        (0.5, np.array([0.5, 0.5])),
        (1, np.array([0, 1])),
    ]

    @pytest.mark.parametrize("nr_wires", [2, 3])
    @pytest.mark.parametrize("prob, expected", prob_and_expected_probs)
    def test_readout_probs(self, nr_wires, prob, expected):
        """Tests the measurement results for probs"""
        dev = qml.device("default.mixed", wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            return qml.probs(wires=0)

        res = circuit()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("nr_wires", [2, 3])
    def test_prob_out_of_range(self, nr_wires):
        """Tests that an error is raised when readout error probability is outside [0,1]"""
        with pytest.raises(ValueError, match="should be in the range"):
            qml.device("default.mixed", wires=nr_wires, readout_prob=2)

    @pytest.mark.parametrize("nr_wires", [2, 3])
    def test_prob_type(self, nr_wires):
        """Tests that an error is raised for wrong data type of readout error probability"""
        with pytest.raises(TypeError, match="should be an integer or a floating-point number"):
            qml.device("default.mixed", wires=nr_wires, readout_prob="RandomNum")


class TestInit:
    """Tests related to device initializtion"""

    def test_nr_wires(self):
        """Tests that an error is raised if the device is initialized with more than 23 wires"""
        with pytest.raises(ValueError, match="This device does not currently"):
            qml.device("default.mixed", wires=24)
