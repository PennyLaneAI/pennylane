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

import pytest
import pennylane as qml
from pennylane.devices import DefaultMixed
from pennylane.ops import PauliZ, CZ, PauliX, Hadamard, CNOT, AmplitudeDamping, DepolarizingChannel
from pennylane.wires import Wires

import numpy as np

INV_SQRT2 = 1 / np.sqrt(2)

# Operations used for testing
diags = [PauliZ, CZ]
channels = [AmplitudeDamping, DepolarizingChannel]
gates = [PauliX, Hadamard, CNOT]
non_diags = [PauliX, Hadamard, CNOT, AmplitudeDamping, DepolarizingChannel]
all_ops = [PauliZ, CZ, PauliX, Hadamard, CNOT, AmplitudeDamping, DepolarizingChannel]


# functions for creating different states used in testing
def basis_state(index, nr_wires):
    rho = np.zeros((2 ** nr_wires, 2 ** nr_wires), dtype=np.complex128)
    rho[index, index] = 1
    return rho


def hadamard_state(nr_wires):
    """Equal superposition state (Hadamard on all qubits)"""
    return np.ones((2 ** nr_wires, 2 ** nr_wires), dtype=np.complex128) / (2 ** nr_wires)


def max_mixed_state(nr_wires):
    return np.eye(2 ** nr_wires, dtype=np.complex128) / (2 ** nr_wires)


def root_state(nr_wires):
    """Pure state with equal amplitudes but phases equal to roots of unity"""
    dim = 2 ** nr_wires
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
    def test_expected_state(self, nr_wires, index):
        """Tests output basis state against the expected one"""
        rho = np.zeros((2 ** nr_wires, 2 ** nr_wires))
        rho[index, index] = 1
        rho = np.reshape(rho, [2] * (2 * nr_wires))
        dev = qml.device("default.mixed", wires=nr_wires)

        assert np.allclose(rho, dev._create_basis_state(index))


@pytest.mark.parametrize("nr_wires", [2, 3])
class TestState:
    """Tests for the method `state()`, which retrieves the state of the system"""

    def test_shape(self, nr_wires):
        """Tests that the state has the correct shape"""
        dev = qml.device("default.mixed", wires=nr_wires)

        assert (2 ** nr_wires, 2 ** nr_wires) == np.shape(dev.state)

    def test_init_state(self, nr_wires):
        """Tests that the state is |0...0><0...0| after initialization of the device"""
        rho = np.zeros((2 ** nr_wires, 2 ** nr_wires))
        rho[0, 0] = 1
        dev = qml.device("default.mixed", wires=nr_wires)

        assert np.allclose(rho, dev.state)

    @pytest.mark.parametrize("op", [CNOT, CZ])
    def test_state_after_twoqubit(self, nr_wires, op):
        """Tests that state is correctly retrieved after applying two-qubit operations on the
        first wires"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([op(wires=[0, 1])])
        current_state = np.reshape(dev._state, (2 ** nr_wires, 2 ** nr_wires))

        assert np.allclose(dev.state, current_state)

    @pytest.mark.parametrize("op", channels)
    def test_state_after_channel(self, nr_wires, op):
        """Tests that state is correctly retrieved after applying a channel on the first wires"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([op(0.5, wires=0)])
        current_state = np.reshape(dev._state, (2 ** nr_wires, 2 ** nr_wires))

        assert np.allclose(dev.state, current_state)

    @pytest.mark.parametrize("op", [PauliX, PauliZ, Hadamard])
    def test_state_after_gate(self, nr_wires, op):
        """Tests that state is correctly retrieved after applying operations on the first wires"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([op(wires=0)])
        current_state = np.reshape(dev._state, (2 ** nr_wires, 2 ** nr_wires))

        assert np.allclose(dev.state, current_state)


@pytest.mark.parametrize("nr_wires", [2, 3])
class TestReset:
    """Unit tests for the method `reset()`"""

    def test_reset_basis(self, nr_wires):
        dev = qml.device("default.mixed", wires=nr_wires)
        dev._state = dev._create_basis_state(1)
        dev.reset()

        assert np.allclose(dev._state, dev._create_basis_state(0))

    @pytest.mark.parametrize("op", [CNOT, CZ])
    def test_reset_after_twoqubit(self, nr_wires, op):
        """Tests that state is correctly reset after applying two-qubit operations on the first
        wires"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([op(wires=[0, 1])])
        dev.reset()

        assert np.allclose(dev._state, dev._create_basis_state(0))

    @pytest.mark.parametrize("op", channels)
    def test_reset_after_channel(self, nr_wires, op):
        """Tests that state is correctly reset after applying a channel on the first
        wires"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([op(0.5, wires=[0])])
        dev.reset()

        assert np.allclose(dev._state, dev._create_basis_state(0))

    @pytest.mark.parametrize("op", [PauliX, PauliZ, Hadamard])
    def test_reset_after_channel(self, nr_wires, op):
        """Tests that state is correctly reset after applying gates on the first
        wire"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([op(wires=0)])
        dev.reset()

        assert np.allclose(dev._state, dev._create_basis_state(0))


@pytest.mark.parametrize("nr_wires", [1, 2, 3])
class TestAnalyticProb:
    """Unit tests for the method `analytic_probability()` """

    def test_prob_init_state(self, nr_wires):
        """Tests that we obtain the correct probabilities for the state |0...0><0...0|"""
        dev = qml.device("default.mixed", wires=nr_wires)
        probs = np.zeros(2 ** nr_wires)
        probs[0] = 1

        assert np.allclose(probs, dev.analytic_probability())

    def test_prob_basis_state(self, nr_wires):
        """Tests that we obtain correct probabilities for the basis state |1...1><1...1|"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev._state = dev._create_basis_state(2 ** nr_wires - 1)
        probs = np.zeros(2 ** nr_wires)
        probs[-1] = 1

        assert np.allclose(probs, dev.analytic_probability())

    def test_prob_hadamard(self, nr_wires):
        """Tests that we obtain correct probabilities for the equal superposition state"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev._state = hadamard_state(nr_wires)
        probs = np.ones(2 ** nr_wires) / (2 ** nr_wires)
        assert np.allclose(probs, dev.analytic_probability())

    def test_prob_mixed(self, nr_wires):
        """Tests that we obtain correct probabilities for the maximally mixed state"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev._state = max_mixed_state(nr_wires)
        probs = np.ones(2 ** nr_wires) / (2 ** nr_wires)
        assert np.allclose(probs, dev.analytic_probability())

    def test_prob_root(self, nr_wires):
        """Tests that we obtain correct probabilities for the root state"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev._state = root_state(nr_wires)
        probs = np.ones(2 ** nr_wires) / (2 ** nr_wires)

        assert np.allclose(probs, dev.analytic_probability())

    def test_none_state(self, nr_wires):
        """Tests that return is `None` when the state is `None`"""
        dev = qml.device("default.mixed", wires=nr_wires)
        dev._state = None

        assert dev.analytic_probability() == None


class TestKrausOps:
    """Unit tests for the method `_get_kraus_ops()`"""

    unitary_ops = [
        (PauliX, np.array([[0, 1], [1, 0]])),
        (Hadamard, np.array([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]])),
        (CNOT, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])),
    ]

    @pytest.mark.parametrize("ops", unitary_ops)
    def test_unitary_kraus(self, ops):
        """Tests that matrices of non-diagonal unitary operations are retrieved correctly"""
        dev = qml.device("default.mixed", wires=2)

        assert np.allclose(dev._get_kraus(ops[0]), [ops[1]])

    diagonal_ops = [
        (PauliZ(wires=0), np.array([1, -1])),
        (CZ(wires=[0, 1]), np.array([1, 1, 1, -1])),
    ]

    @pytest.mark.parametrize("ops", diagonal_ops)
    def test_diagonal_kraus(self, ops):
        """Tests that matrices of non-diagonal unitary operations are retrieved correctly"""
        dev = qml.device("default.mixed", wires=2)

        assert np.allclose(dev._get_kraus(ops[0]), ops[1])

    p = 0.5
    K0 = np.sqrt(1 - p) * np.eye(2)
    K1 = np.sqrt(p / 3) * np.array([[0, 1], [1, 0]])
    K2 = np.sqrt(p / 3) * np.array([[0, -1j], [1j, 0]])
    K3 = np.sqrt(p / 3) * np.array([[1, 0], [0, -1]])

    channel_ops = [
        (
            AmplitudeDamping(p, wires=0),
            [np.diag([1, np.sqrt(1 - p)]), np.sqrt(p) * np.array([[0, 1], [0, 0]])],
        ),
        (DepolarizingChannel(p, wires=0), [K0, K1, K2, K3]),
    ]

    @pytest.mark.parametrize("ops", channel_ops)
    def test_channel_kraus(self, ops):
        """Tests that kraus matrices of non-unitary channels are retrieved correctly"""
        dev = qml.device("default.mixed", wires=1)

        assert np.allclose(dev._get_kraus(ops[0]), ops[1])


class TestApplyChannel:
    """Unit tests for the method `_apply_channel()`"""

    x_apply_channel_init = [
        [1, PauliX, basis_state(1, 1)],
        [1, Hadamard, np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]])],
        [2, CNOT, basis_state(0, 2)],
        [1, AmplitudeDamping(0.5, wires=0), basis_state(0, 1)],
        [
            1,
            DepolarizingChannel(0.5, wires=0),
            np.array([[2 / 3 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1 / 3 + 0.0j]]),
        ],
    ]

    @pytest.mark.parametrize("x", x_apply_channel_init)
    def test_channel_init(self, x):
        """Tests that channels are correctly applied to the default initial state"""
        nr_wires = x[0]
        op = x[1]
        target_state = np.reshape(x[2], [2] * 2 * nr_wires)
        dev = qml.device("default.mixed", wires=nr_wires)
        kraus = dev._get_kraus(op)
        if op == CNOT:
            dev._apply_channel(kraus, wires=Wires([0, 1]))
        else:
            kraus = dev._get_kraus(op)
            dev._apply_channel(kraus, wires=Wires(0))

        assert np.allclose(dev._state, target_state)

    x_apply_channel_mixed = [
        [1, PauliX, max_mixed_state(1)],
        [2, Hadamard, max_mixed_state(2)],
        [2, CNOT, max_mixed_state(2)],
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
    ]

    @pytest.mark.parametrize("x", x_apply_channel_mixed)
    def test_channel_mixed(self, x):
        """Tests that channels are correctly applied to the maximally mixed state"""
        nr_wires = x[0]
        op = x[1]
        target_state = np.reshape(x[2], [2] * 2 * nr_wires)
        dev = qml.device("default.mixed", wires=nr_wires)
        max_mixed = np.reshape(max_mixed_state(nr_wires), [2] * 2 * nr_wires)
        dev._state = max_mixed
        kraus = dev._get_kraus(op)
        if op == CNOT:
            dev._apply_channel(kraus, wires=Wires([0, 1]))
        else:
            kraus = dev._get_kraus(op)
            dev._apply_channel(kraus, wires=Wires(0))

        assert np.allclose(dev._state, target_state)

    x_apply_channel_root = [
        [1, PauliX, np.array([[0.5 + 0.0j, -0.5 + 0.0j], [-0.5 - 0.0j, 0.5 + 0.0j]])],
        [1, Hadamard, np.array([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]])],
        [
            2,
            CNOT,
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
    ]

    @pytest.mark.parametrize("x", x_apply_channel_root)
    def test_channel_root(self, x):
        """Tests that channels are correctly applied to root state"""
        nr_wires = x[0]
        op = x[1]
        target_state = np.reshape(x[2], [2] * 2 * nr_wires)
        dev = qml.device("default.mixed", wires=nr_wires)
        root = np.reshape(root_state(nr_wires), [2] * 2 * nr_wires)
        dev._state = root
        kraus = dev._get_kraus(op)
        if op == CNOT:
            dev._apply_channel(kraus, wires=Wires([0, 1]))
        else:
            kraus = dev._get_kraus(op)
            dev._apply_channel(kraus, wires=Wires(0))

        assert np.allclose(dev._state, target_state)


class TestApplyDiagonal:
    """Unit tests for the method `_apply_diagonal_unitary()`"""

    x_apply_diag_init = [[1, PauliZ, basis_state(0, 1)], [2, CZ, basis_state(0, 2)]]

    @pytest.mark.parametrize("x", x_apply_diag_init)
    def test_diag_init(self, x):
        """Tests that diagonal gates are correctly applied to the default initial state"""
        nr_wires = x[0]
        op = x[1]
        target_state = np.reshape(x[2], [2] * 2 * nr_wires)
        dev = qml.device("default.mixed", wires=nr_wires)
        kraus = dev._get_kraus(op)
        if op == CZ:
            dev._apply_channel(kraus, wires=Wires([0, 1]))
        else:
            kraus = dev._get_kraus(op)
            dev._apply_channel(kraus, wires=Wires(0))

        assert np.allclose(dev._state, target_state)

    x_apply_diag_mixed = [[1, PauliZ, max_mixed_state(1)], [2, CZ, max_mixed_state(2)]]

    @pytest.mark.parametrize("x", x_apply_diag_mixed)
    def test_diag_mixed(self, x):
        """Tests that diagonal gates are correctly applied to the maximally mixed state"""
        nr_wires = x[0]
        op = x[1]
        target_state = np.reshape(x[2], [2] * 2 * nr_wires)
        dev = qml.device("default.mixed", wires=nr_wires)
        max_mixed = np.reshape(max_mixed_state(nr_wires), [2] * 2 * nr_wires)
        dev._state = max_mixed
        kraus = dev._get_kraus(op)
        if op == CZ:
            dev._apply_channel(kraus, wires=Wires([0, 1]))
        else:
            kraus = dev._get_kraus(op)
            dev._apply_channel(kraus, wires=Wires(0))

        assert np.allclose(dev._state, target_state)

    x_apply_diag_root = [
        [1, PauliZ, np.array([[0.5, 0.5], [0.5, 0.5]])],
        [
            2,
            CZ,
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
    def test_diag_root(self, x):
        """Tests that diagonal gates are correctly applied to root state"""
        nr_wires = x[0]
        op = x[1]
        target_state = np.reshape(x[2], [2] * 2 * nr_wires)
        dev = qml.device("default.mixed", wires=nr_wires)
        root = np.reshape(root_state(nr_wires), [2] * 2 * nr_wires)
        dev._state = root
        kraus = dev._get_kraus(op)
        if op == CZ:
            dev._apply_channel(kraus, wires=Wires([0, 1]))
        else:
            kraus = dev._get_kraus(op)
            dev._apply_channel(kraus, wires=Wires(0))

        assert np.allclose(dev._state, target_state)


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


class TestApply:
    """Unit tests for the main method `apply()`. We check that lists of operations are applied
    correctly, rather than single operations"""

    def test_bell_state(self):
        """Tests that we correctly prepare a Bell state by applying a Hadamard then a CNOT"""
        dev = qml.device("default.mixed", wires=2)
        ops = [Hadamard(0), CNOT(wires=[0, 1])]
        dev.apply(ops)
        bell = np.zeros((4, 4))
        bell[0, 0] = bell[0, 3] = bell[3, 0] = bell[3, 3] = 1 / 2

        assert np.allclose(bell, dev.state)

    @pytest.mark.parametrize("nr_wires", [1, 2, 3])
    def test_hadamard_state(self, nr_wires):
        """Tests that applying Hadamard gates on all qubits produces an equal superposition over
        all basis states"""
        dev = qml.device("default.mixed", wires=nr_wires)
        ops = [Hadamard(i) for i in range(nr_wires)]
        dev.apply(ops)

        assert np.allclose(dev.state, hadamard_state(nr_wires))

    @pytest.mark.parametrize("nr_wires", [1, 2, 3])
    def test_max_mixed_state(self, nr_wires):
        """Tests that applying damping channel on all qubits to the state |11...1> produces a
        maximally mixed state"""
        dev = qml.device("default.mixed", wires=nr_wires)
        flips = [PauliX(i) for i in range(nr_wires)]
        damps = [AmplitudeDamping(0.5, wires=i) for i in range(nr_wires)]
        ops = flips + damps
        dev.apply(ops)

        assert np.allclose(dev.state, max_mixed_state(nr_wires))

    @pytest.mark.parametrize("nr_wires", [1, 2, 3])
    def test_undo_rotations(self, nr_wires):
        """Tests that rotations are correctly applied by adding their inverse as initial
        operations"""
        dev = qml.device("default.mixed", wires=nr_wires)
        ops = [Hadamard(i) for i in range(nr_wires)]
        rots = ops
        dev.apply(ops, rots)
        basis = np.reshape(basis_state(0, nr_wires), [2] * (2 * nr_wires))
        # dev.state = pre-rotated state, dev._state = state after rotations
        assert np.allclose(dev.state, hadamard_state(nr_wires))
        assert np.allclose(dev._state, basis)

    def test_apply_toffoli(self):
        """Tests that Toffoli gate is correctly applied on state |111> to give state |110>"""
        nr_wires = 3
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([PauliX(0), PauliX(1), PauliX(2), qml.Toffoli(wires=[0, 1, 2])])

        assert np.allclose(dev.state, basis_state(6, 3))

    def test_apply_qubitunitary(self):
        """Tests that custom qubit unitary is correctly applied"""
        nr_wires = 1
        theta = 0.42
        U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        dev = qml.device("default.mixed", wires=nr_wires)
        dev.apply([qml.QubitUnitary(U, wires=[0])])
        ket = np.array([np.cos(theta) + 0j, np.sin(theta) + 0j])
        target_rho = np.outer(ket, np.conj(ket))

        assert np.allclose(dev.state, target_rho)


class TestInit:
    """Tests related to device initializtion"""

    def test_nr_wires(self):
        """Tests that an error is raised if the device is initialized with more than 23 wires"""
        with pytest.raises(ValueError, match="This device does not currently"):
            qml.device("default.mixed", wires=24)
