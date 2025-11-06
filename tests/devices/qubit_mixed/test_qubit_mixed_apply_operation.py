# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for apply_operation in devices/qubit_mixed/apply_operation."""

from functools import reduce

import numpy as np
import pytest
from dummy_debugger import Debugger
from scipy.stats import unitary_group

import pennylane as qml
from pennylane import (
    CNOT,
    ISWAP,
    AmplitudeDamping,
    DepolarizingChannel,
    Hadamard,
    PauliError,
    PauliX,
    ResetError,
    math,
)
from pennylane.devices.qubit_mixed import apply_operation, measure
from pennylane.devices.qubit_mixed.apply_operation import (
    apply_operation_einsum,
    apply_operation_tensordot,
)
from pennylane.operation import _UNSET_BATCH_SIZE

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]


def basis_state(index, nr_wires):
    """Generate the density matrix of the computational basis state
    indicated by ``index``."""
    rho = np.zeros((2**nr_wires, 2**nr_wires), dtype=np.complex128)
    rho[index, index] = 1
    return rho


def base0(nr_wires):
    """Generate the density matrix of the computational basis state
    indicated by ``00...0``."""
    return basis_state(0, nr_wires)


def base1(nr_wires):
    """Generate the density matrix of the computational basis state
    indicated by ``11...1``."""
    return basis_state(2**nr_wires - 1, nr_wires)


def cat_state(nr_wires):
    """Generate the density matrix of the cat state. |0...0> + |1...1>"""
    rho = np.zeros((2**nr_wires, 2**nr_wires), dtype=np.complex128)
    first = 0
    last = 2**nr_wires - 1
    # Make the four corners of the matrix 0.5
    rho[first, first] = 0.5
    rho[first, last] = 0.5
    rho[last, first] = 0.5
    rho[last, last] = 0.5
    return rho


def hadamard_state(nr_wires):
    """Generate the equal superposition state (Hadamard on all qubits)"""
    return np.ones((2**nr_wires, 2**nr_wires), dtype=np.complex128) / (2**nr_wires)


def max_mixed_state(nr_wires):
    """Generate the maximally mixed state."""
    return np.eye(2**nr_wires, dtype=np.complex128) / (2**nr_wires)


def root_state(nr_wires):
    """Pure state with equal amplitudes but phases equal to roots of unity"""
    dim = 2**nr_wires
    ket = [np.exp(1j * 2 * np.pi * n / dim) / np.sqrt(dim) for n in range(dim)]
    return np.outer(ket, np.conj(ket))


special_state_generator = [base0, base1, cat_state, hadamard_state, max_mixed_state, root_state]


def get_expected_state(expanded_operator, state, num_q):
    """Finds expected state after applying operator"""
    # Convert the state into numpy
    state = np.asarray(state)
    shape = (2**num_q,) * 2
    flattened_state = state.reshape(shape)
    adjoint_matrix = np.conj(expanded_operator).T

    new_state = expanded_operator @ flattened_state @ adjoint_matrix
    return new_state.reshape([2] * (num_q * 2))


def expand_matrices(op, num_q, batch_size=0):
    """Find expanded operator matrices, independent of qml implementation"""
    pre_wires_identity = np.eye(2 ** op.wires[0])
    post_wires_identity = np.eye(2 ** ((num_q - 1) - op.wires[-1]))
    mat = op.matrix()

    def expand_matrix(matrix):
        return reduce(np.kron, (pre_wires_identity, matrix, post_wires_identity))

    if batch_size:
        return [expand_matrix(mat[i]) for i in range(batch_size)]
    return expand_matrix(mat)


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestOperation:  # pylint: disable=too-few-public-methods
    """Tests that broadcasted operations (not channels) are applied correctly."""

    unbroadcasted_ops = [
        qml.Hadamard(wires=0),
        qml.RX(np.pi / 3, wires=0),
        qml.RY(2 * np.pi / 3, wires=1),
        qml.RZ(np.pi / 6, wires=2),
        qml.X(wires=0),
        qml.Z(wires=1),
        qml.S(wires=2),
        qml.T(wires=0),
        qml.PhaseShift(np.pi / 7, wires=1),
        qml.CNOT(wires=[0, 1]),
        qml.MultiControlledX(wires=(0, 1, 2), control_values=[1, 0]),
        qml.SWAP(wires=[0, 1]),
        qml.CSWAP(wires=[0, 1, 2]),
        qml.Toffoli(wires=[0, 1, 2]),
        qml.CZ(wires=[0, 1]),
        qml.CY(wires=[0, 1]),
        qml.CH(wires=[0, 1]),
        qml.GroverOperator(wires=[0, 1, 2]),
        qml.GroverOperator(wires=[1, 2]),
    ]
    diagonal_ops = [
        qml.PauliZ(wires=0),  # Most naive one
        qml.RZ(np.pi / 6, wires=2),  # single-site op, diagonal but complex eigvals
        qml.IsingZZ(0.5, wires=[0, 1]),  # two site
        qml.CCZ(wires=[0, 1, 2]),  # three site
    ]
    num_qubits = [3, 9]
    num_batched = 4

    @classmethod
    def circuit_matrices(cls, op, num_q, batch_size=0):
        """defines the circuit matrices, an alternative to expand_matrices"""

        def circuit():
            op(wires=op.wires)

        matrix_fn = qml.matrix(circuit, wire_order=range(num_q))
        if batch_size:
            return [matrix_fn() for _ in range(batch_size)]
        return matrix_fn()

    @pytest.mark.parametrize("op", unbroadcasted_ops)
    @pytest.mark.parametrize("num_q", num_qubits)
    def test_no_broadcasting(self, op, num_q, ml_framework, random_mixed_state):
        """
        Tests that unbatched operations are applied correctly to an unbatched state.

        Args:
            op (Operation): Quantum operation to apply.
            ml_framework (str): The machine learning framework in use (numpy, autograd, etc.).
        """
        state = math.asarray(random_mixed_state(num_q), like=ml_framework)
        res = apply_operation(op, state)
        res_tensordot = apply_operation_tensordot(op, state)
        res_einsum = apply_operation_einsum(op, state)

        expanded_operator = expand_matrices(op, num_q)
        expected = get_expected_state(np.array(expanded_operator), np.array(state), num_q)

        # assert math.get_interface(res) == ml_framework
        assert math.allclose(res, expected), f"Operation {op} failed. {res} \n != {expected}"
        assert math.allclose(
            res_tensordot, expected
        ), f"Tensordot and einsum results do not match. {res_tensordot} != {res_einsum}"

    @pytest.mark.parametrize("op", diagonal_ops)
    @pytest.mark.parametrize("num_q", num_qubits)
    def test_diagonal(self, op, num_q, ml_framework, random_mixed_state):
        """
        Tests that diagonal operations are applied correctly to an unbatched state.

        Args:
            op (Operation): Quantum operation to apply.
            ml_framework (str): The machine learning framework in use (numpy, autograd, etc.).
        """
        state_np = random_mixed_state(num_q)
        state = math.asarray(state_np, like=ml_framework)
        res = apply_operation(op, state)

        expanded_operator = expand_matrices(op, num_q)
        expected = get_expected_state(expanded_operator, state_np, num_q)

        assert math.allclose(res, expected), f"Operation {op} failed. {res} != {expected}"

    @pytest.mark.parametrize("num_q", num_qubits)
    def test_identity(self, num_q, ml_framework, random_mixed_state):
        """Tests that the identity operation is applied correctly to an unbatched state."""
        state_np = random_mixed_state(num_q)
        state = math.asarray(state_np, like=ml_framework)
        op = qml.Identity(wires=0)
        res = apply_operation(op, state)

        assert math.allclose(res, state), f"Operation {op} failed. {res} != {state}"

    @pytest.mark.parametrize("num_q", num_qubits)
    def test_globalphase(self, num_q, ml_framework, random_mixed_state):
        """Tests that the identity operation is applied correctly to an unbatched state."""
        state_np = random_mixed_state(num_q)
        state = math.asarray(state_np, like=ml_framework)
        op = qml.GlobalPhase(np.pi / 7, wires=0)
        res = apply_operation(op, state)

        assert math.allclose(res, state), f"Operation {op} failed. {res} != {state}"

    @pytest.mark.parametrize("op", unbroadcasted_ops)
    @pytest.mark.parametrize("num_q", num_qubits)
    def test_unbroadcasted_ops_batched(self, op, num_q, ml_framework, random_mixed_state):
        """Test that unbroadcasted operations are applied correctly to batched states."""
        batch_size = self.num_batched
        state = np.array([random_mixed_state(num_q) for _ in range(batch_size)])
        state = math.asarray(state, like=ml_framework)
        res = apply_operation(op, state, is_state_batched=True)

        expanded_operator = expand_matrices(op, num_q)
        expanded_operator = math.expand_matrix(op.matrix(), op.wires, wire_order=range(num_q))
        expected = np.array(
            [
                get_expected_state(
                    expanded_operator,
                    s,
                    num_q,
                )
                for s in state
            ]
        )

        # Make both res and expected the same shape
        res = np.array(res).reshape(expected.shape)

        assert math.allclose(res, expected), f"Operation {op} failed. {res} != {expected}"

    @pytest.mark.parametrize("op", diagonal_ops)
    @pytest.mark.parametrize("num_q", num_qubits)
    def test_diagonal_ops_batched(self, op, num_q, ml_framework, random_mixed_state):
        """Test that diagonal operations are applied correctly to batched states."""
        batch_size = self.num_batched
        state = np.array([random_mixed_state(num_q) for _ in range(batch_size)])
        state = math.asarray(state, like=ml_framework)
        res = apply_operation(op, state, is_state_batched=True)

        expanded_operator = expand_matrices(op, num_q)
        expected = np.array([get_expected_state(expanded_operator, s, num_q) for s in state])

        assert math.allclose(res, expected), f"Operation {op} failed. {res} != {expected}"

    @pytest.mark.parametrize("state_gen", special_state_generator)
    @pytest.mark.parametrize("op", unbroadcasted_ops)
    @pytest.mark.parametrize("num_q", num_qubits)
    def test_special_states(self, ml_framework, state_gen, op, num_q):
        """Test that special states are handled correctly."""
        state = math.asarray(state_gen(num_q), like=ml_framework)
        state = math.reshape(state, [2] * (2 * num_q))
        res = apply_operation(op, state)

        expanded_operator = expand_matrices(op, num_q)
        expected = get_expected_state(expanded_operator, state, num_q)

        assert math.allclose(res, expected), f"Operation {op} failed. {res} != {expected}"


class TestApplyGroverOperator:
    """Test that GroverOperator is applied correctly to mixed states."""

    @pytest.mark.parametrize(
        "num_wires, expected_method",
        [
            (2, "einsum"),
            (3, "tensordot"),
            (8, "tensordot"),
            (9, "custom"),
        ],
    )
    def test_dispatch_method(self, num_wires, expected_method, mocker, random_mixed_state):
        """Test that the correct dispatch method is used based on the number of wires."""
        state = random_mixed_state(num_wires)

        op = qml.GroverOperator(wires=range(num_wires))

        spy_einsum = mocker.spy(math, "einsum")
        spy_tensordot = mocker.spy(math, "tensordot")

        apply_operation(op, state)

        if expected_method == "einsum":
            assert spy_einsum.called
            assert not spy_tensordot.called
        elif expected_method == "tensordot":
            assert not spy_einsum.called
            assert spy_tensordot.called
        else:  # custom method
            assert not spy_einsum.called
            # Not assert not spy tensordot since in the method implemented in qubit.apply_operation it is indeed in use
            # assert not spy_tensordot.called

    @pytest.mark.parametrize("num_wires", [2, 3, 7, 8, 9])
    def test_correctness(self, num_wires, random_mixed_state):
        """Test that the GroverOperator is applied correctly for various wire numbers."""
        state = random_mixed_state(num_wires)

        op = qml.GroverOperator(wires=range(num_wires))
        op_mat = op.matrix()
        flat_shape = op_mat.shape

        result = apply_operation(op, state)

        state_flat = state.reshape(flat_shape)
        expected = op_mat @ state_flat @ op_mat.conj().T

        assert np.allclose(result.reshape(flat_shape), expected)

    @pytest.mark.parametrize("num_wires", [2, 3, 7, 8, 9])
    def test_batched_state(self, num_wires, random_mixed_state):
        """Test that the GroverOperator works correctly with batched states."""
        batch_size = 3
        state = np.array([random_mixed_state(num_wires) for _ in range(batch_size)])

        op = qml.GroverOperator(wires=range(num_wires))
        op_mat = op.matrix()
        # Make new shape, considering the batch dimension as the first
        flat_shape = (batch_size,) + op_mat.shape

        result = apply_operation(op, state, is_state_batched=True)

        state_flat = state.reshape(flat_shape)
        expected = np.array([op.matrix() @ s @ op.matrix().conj().T for s in state_flat])

        assert np.allclose(result.reshape(flat_shape), expected)

    @pytest.mark.parametrize("interface", ml_frameworks_list)
    def test_interface_compatibility(self, interface, random_mixed_state):
        """Test that the GroverOperator works with different interfaces."""
        num_wires = 5
        state = random_mixed_state(num_wires)
        state = math.asarray(state, like=interface)

        op = qml.GroverOperator(wires=range(num_wires))

        # Test with bruteforce
        op_mat = op.matrix()
        op_mat = expand_matrices(op, num_wires)
        result_bf = get_expected_state(op_mat, state, num_wires)

        result = apply_operation(op, state)

        assert np.allclose(result_bf, result)


class TestApplyMultiControlledX:
    """Test that MultiControlledX is applied correctly to mixed states."""

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    @pytest.mark.parametrize(
        "num_wires, interface, expected_method",
        [
            (3, "numpy", "tensordot"),
            (8, "numpy", "tensordot"),
            (9, "numpy", "custom"),
            (3, "autograd", "tensordot"),
            (8, "autograd", "tensordot"),
            (9, "autograd", "custom"),
        ],
    )
    def test_dispatch_method(
        self, num_wires, expected_method, interface, mocker, random_mixed_state
    ):
        """Test that the correct dispatch method is used based on the number of wires
        for numpy and autograd."""
        state = random_mixed_state(num_wires)
        # Convert to interface
        state = math.asarray(state, like=interface)

        op = qml.MultiControlledX(wires=range(num_wires))

        spy_einsum = mocker.spy(math, "einsum")
        spy_tensordot = mocker.spy(math, "tensordot")

        apply_operation(op, state)

        if expected_method == "einsum":
            assert spy_einsum.called
            assert not spy_tensordot.called
        elif expected_method == "tensordot":
            assert not spy_einsum.called
            assert spy_tensordot.called
        else:  # custom method
            assert not spy_einsum.called
            assert not spy_tensordot.called

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    @pytest.mark.parametrize("interface", ml_frameworks_list[2:])
    @pytest.mark.parametrize(
        "num_wires, expected_method",
        [
            (3, "einsum"),
            (7, "einsum"),
            (8, "tensordot"),
            (9, "custom"),
        ],
    )
    def test_dispatch_method_interfaces(
        self, num_wires, expected_method, interface, mocker, random_mixed_state
    ):
        """Test that the correct dispatch method is used based on the number of wires
        for torch, tensorflow, and jax."""
        state = random_mixed_state(num_wires)
        # Convert to interface
        state = math.asarray(state, like=interface)

        op = qml.MultiControlledX(wires=range(num_wires))

        spy_einsum = mocker.spy(math, "einsum")
        spy_tensordot = mocker.spy(math, "tensordot")

        apply_operation(op, state)

        if expected_method == "einsum":
            assert spy_einsum.called
            assert not spy_tensordot.called
        elif expected_method == "tensordot":
            assert not spy_einsum.called
            assert spy_tensordot.called
        else:  # custom method
            assert not spy_einsum.called
            assert not spy_tensordot.called

    @pytest.mark.parametrize("num_wires", [2, 3, 7, 8, 9])
    def test_correctness(self, num_wires, random_mixed_state):
        """Test that the MultiControlledX is applied correctly for various wire numbers."""
        state = random_mixed_state(num_wires)

        op = qml.MultiControlledX(wires=range(num_wires))
        op_mat = op.matrix()
        flat_shape = op_mat.shape

        result = apply_operation(op, state)

        state_flat = state.reshape(flat_shape)
        expected = op_mat @ state_flat @ op_mat.conj().T

        assert np.allclose(result.reshape(flat_shape), expected)

    @pytest.mark.parametrize("num_wires", [2, 3, 7, 8, 9])
    def test_batched_state(self, num_wires, random_mixed_state):
        """Test that the MultiControlledX works correctly with batched states."""
        batch_size = 3
        state = np.array([random_mixed_state(num_wires) for _ in range(batch_size)])

        op = qml.MultiControlledX(wires=range(num_wires))
        op_mat = op.matrix()
        # Make new shape, considering the batch dimension as the first
        flat_shape = (batch_size,) + op_mat.shape

        result = apply_operation(op, state, is_state_batched=True)

        state_flat = state.reshape(flat_shape)
        expected = np.array([op.matrix() @ s @ op.matrix().conj().T for s in state_flat])

        assert np.allclose(result.reshape(flat_shape), expected)

    @pytest.mark.parametrize("interface", ml_frameworks_list)
    def test_interface_compatibility(self, interface, random_mixed_state):
        """Test that the MultiControlledX works with different interfaces."""
        num_wires = 5
        state = random_mixed_state(num_wires)
        state = math.asarray(state, like=interface)

        op = qml.MultiControlledX(wires=range(num_wires))

        result = apply_operation(op, state)

        # Test with bruteforce
        op_mat = op.matrix()
        op_mat = expand_matrices(op, num_wires)
        result_bf = get_expected_state(op_mat, state, num_wires)

        assert np.allclose(result_bf, result)


@pytest.mark.parametrize("apply_method", [apply_operation_einsum, apply_operation_tensordot])
class TestApplyChannel:
    """Unit tests for apply operation of channels"""

    x_apply_channel_init = [
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
    def test_channel_init(self, x, tol, apply_method):
        """Tests that channels are correctly applied to the default initial state"""
        nr_wires = x[0]
        op = x[1]
        shape_state = [2] * 2 * nr_wires
        init_state = basis_state(0, nr_wires)
        init_state = np.reshape(init_state, shape_state)
        target_state = np.reshape(x[2], shape_state)
        res = apply_method(op, init_state)

        assert np.allclose(res, target_state, atol=tol, rtol=0)

    x_apply_channel_mixed = [
        [1, PauliX(wires=0), max_mixed_state(1)],
        [2, Hadamard(wires=0), max_mixed_state(2)],
        [2, CNOT(wires=[0, 1]), max_mixed_state(2)],
        [2, ISWAP(wires=[0, 1]), max_mixed_state(2)],
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
    def test_channel_mixed(self, x, tol, apply_method):
        """Tests that channels are correctly applied to the maximally mixed state"""
        nr_wires = x[0]
        op = x[1]
        shape_state = [2] * 2 * nr_wires
        init_state = np.reshape(max_mixed_state(nr_wires), shape_state)
        target_state = np.reshape(x[2], shape_state)
        res = apply_method(op, init_state)

        assert np.allclose(res, target_state, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ml_frameworks_list)
    @pytest.mark.parametrize("x", x_apply_channel_init)
    def test_channel_init_interface(self, x, tol, apply_method, interface):
        """Tests that channels are correctly applied to the default initial state with different interfaces"""
        nr_wires = x[0]
        op = x[1]
        shape_state = [2] * 2 * nr_wires
        init_state = basis_state(0, nr_wires)
        init_state = np.reshape(init_state, shape_state)
        init_state = math.asarray(init_state, like=interface)
        target_state = np.reshape(x[2], shape_state)
        target_state = math.asarray(target_state, like=interface)
        res = apply_method(op, init_state)

        assert math.allclose(res, target_state, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ml_frameworks_list)
    @pytest.mark.parametrize("x", x_apply_channel_mixed)
    def test_channel_mixed_interface(self, x, tol, apply_method, interface):
        """Tests that channels are correctly applied to the maximally mixed state with different interfaces"""
        nr_wires = x[0]
        op = x[1]
        shape_state = [2] * 2 * nr_wires
        init_state = np.reshape(max_mixed_state(nr_wires), shape_state)
        init_state = math.asarray(init_state, like=interface)
        target_state = np.reshape(x[2], shape_state)
        target_state = math.asarray(target_state, like=interface)
        res = apply_method(op, init_state)

        assert math.allclose(res, target_state, atol=tol, rtol=0)


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestBroadcasting:  # pylint: disable=too-few-public-methods
    """Tests that broadcasted operations are applied correctly."""

    broadcasted_ops = [
        qml.RX(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=2),
        qml.PhaseShift(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=2),
        qml.IsingXX(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=[1, 2]),
        qml.QubitUnitary(
            np.array([unitary_group.rvs(8), unitary_group.rvs(8), unitary_group.rvs(8)]),
            wires=[0, 1, 2],
        ),
    ]

    unbroadcasted_ops = [
        qml.PauliX(2),
        qml.PauliZ(2),
        qml.CNOT([1, 2]),
        qml.RX(np.pi, wires=2),
        qml.PhaseShift(np.pi / 2, wires=2),
        qml.IsingXX(np.pi / 2, wires=[1, 2]),
        qml.QubitUnitary(unitary_group.rvs(8), wires=[0, 1, 2]),
    ]

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op(self, op, ml_framework, random_mixed_state):
        """Tests that batched operations are applied correctly to an unbatched state."""
        num_q = 3
        state = math.asarray(random_mixed_state(num_q), like=ml_framework)

        res = apply_operation(op, state)

        expanded_mat = expand_matrices(op, 3, batch_size=3)
        expected = [(get_expected_state(expanded_mat[i], state, num_q)) for i in range(3)]

        assert math.get_interface(res) == ml_framework
        assert math.allclose(res, expected)

    @pytest.mark.parametrize("op", unbroadcasted_ops)
    def test_broadcasted_state(self, op, ml_framework, random_mixed_state):
        """Tests that batched operations are applied correctly to an unbatched state."""
        num_q = 3
        state = [math.asarray(random_mixed_state(num_q), like=ml_framework) for _ in range(3)]
        state = math.stack(state)

        res = apply_operation(op, state, is_state_batched=True)

        expanded_mat = expand_matrices(op, 3)
        expected = [(get_expected_state(expanded_mat, state[i], num_q)) for i in range(3)]

        assert math.get_interface(res) == ml_framework
        assert math.allclose(res, expected)

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op_broadcasted_state(self, op, ml_framework, random_mixed_state):
        """Tests that batched operations are applied correctly to batched state."""
        num_q = 3
        state = [math.asarray(random_mixed_state(num_q), like=ml_framework) for _ in range(3)]
        state = math.stack(state)

        res = apply_operation(op, state, is_state_batched=True)

        expanded_mat = expand_matrices(op, 3, batch_size=3)
        expected = [(get_expected_state(expanded_mat[i], state[i], num_q)) for i in range(3)]

        assert math.get_interface(res) == ml_framework
        assert math.allclose(res, expected)

    def test_batch_size_set_if_missing(self, ml_framework):
        """Tests that the batch_size is set on an operator if it was missing before.
        Mostly useful for TF-autograph since it may have batch size set to None."""
        param = qml.math.asarray([0.1, 0.2, 0.3], like=ml_framework)
        state = np.ones((2, 2)) / 2
        op = qml.RX(param, 0)
        assert op._batch_size is _UNSET_BATCH_SIZE  # pylint:disable=protected-access
        state = apply_operation_einsum(op, state)
        assert state.shape == (3, 2, 2)
        assert op.batch_size == 3


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize(
    "state,shape", [("two_qubit_state", (4, 4)), ("two_qubit_batched_state", (2, 4, 4))]
)
class TestSnapshot:
    """Test that apply_operation works for Snapshot ops"""

    @pytest.mark.usefixtures("two_qubit_state")
    def test_no_debugger(self, ml_framework, state, shape, request):
        """Test that nothing happens when there is no debugger"""
        state = request.getfixturevalue(state)
        initial_state = math.asarray(state, like=ml_framework)

        new_state = apply_operation(qml.Snapshot(), initial_state, is_state_batched=len(shape) != 2)
        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

    def test_empty_tag(self, ml_framework, state, shape, request):
        """Test a snapshot is recorded properly when there is no tag"""
        state = request.getfixturevalue(state)
        initial_state = math.asarray(state, like=ml_framework)

        debugger = Debugger()
        new_state = apply_operation(
            qml.Snapshot(), initial_state, debugger=debugger, is_state_batched=len(shape) != 2
        )

        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [0]
        assert debugger.snapshots[0].shape == shape
        assert math.allclose(debugger.snapshots[0], math.reshape(initial_state, shape))

    def test_provided_tag(self, ml_framework, state, shape, request):
        """Test a snapshot is recorded properly when provided a tag"""
        state = request.getfixturevalue(state)
        initial_state = math.asarray(state, like=ml_framework)

        debugger = Debugger()
        tag = "dense"
        new_state = apply_operation(
            qml.Snapshot(tag), initial_state, debugger=debugger, is_state_batched=len(shape) != 2
        )

        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [tag]
        assert debugger.snapshots[tag].shape == shape
        assert math.allclose(debugger.snapshots[tag], math.reshape(initial_state, shape))

    def test_snapshot_with_measurement(self, ml_framework, state, shape, request):
        """Test a snapshot with measurement"""
        state = request.getfixturevalue(state)
        initial_state = math.asarray(state, like=ml_framework)
        tag = "expected_value"

        debugger = Debugger()

        new_state = apply_operation(
            qml.Snapshot(tag, measurement=qml.expval(qml.PauliZ(0))),
            initial_state,
            debugger=debugger,
            is_state_batched=len(shape) != 2,
        )

        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [tag]

        if len(shape) == 2:
            assert debugger.snapshots[tag].shape == ()
            # Expected value for PauliZ measurement would depend on the initial state
            # This value should be calculated based on your test state
            expected_value = measure(qml.expval(qml.PauliZ(0)), initial_state)
            assert math.allclose(debugger.snapshots[tag], expected_value)
        else:
            assert debugger.snapshots[tag].shape == (2,)
            expected_values = measure(
                qml.expval(qml.PauliZ(0)), initial_state, is_state_batched=True
            )
            assert math.allclose(debugger.snapshots[tag], expected_values)

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    @pytest.mark.parametrize(
        "measurement",
        [
            qml.sample(wires=[0, 1]),
            qml.counts(wires=[0, 1]),
        ],
    )
    def test_snapshot_with_shots_and_measurement(
        self, measurement, ml_framework, state, shape, request
    ):
        """Test snapshots with shots for various measurement types."""
        state = request.getfixturevalue(state)
        initial_state = math.asarray(state, like=ml_framework)
        tag = "measurement_snapshot"
        is_state_batched = len(shape) != 2

        shots = qml.measurements.Shots(1000)
        debugger = Debugger()

        new_state = apply_operation(
            qml.Snapshot(tag, measurement=measurement),
            initial_state,
            debugger=debugger,
            is_state_batched=is_state_batched,
            tape_shots=shots,
        )

        # Check state is unchanged
        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

        # Check snapshot was stored
        assert list(debugger.snapshots.keys()) == [tag]

        snapshot_result = debugger.snapshots[tag]

        # Verify snapshot result based on measurement type
        if isinstance(measurement, qml.measurements.SampleMP):
            len_measured_wires = len(measurement.wires)
            assert (
                snapshot_result.shape == (1000, len_measured_wires)
                if not is_state_batched
                else (2, 1000, len_measured_wires)
            )
            assert set(np.unique(snapshot_result)) <= {0, 1}
        elif isinstance(measurement, qml.measurements.CountsMP):
            if is_state_batched:
                snapshot_result = snapshot_result[0]
            assert isinstance(snapshot_result, dict)
            assert all(isinstance(k, str) for k in snapshot_result.keys())
            assert sum(snapshot_result.values()) == 1000


def get_valid_density_matrix(num_wires):
    """Helper function to create a valid density matrix"""
    # Create a pure state first
    state = np.zeros(2**num_wires, dtype=np.complex128)
    state[0] = 1 / np.sqrt(2)
    state[-1] = 1 / np.sqrt(2)
    # Convert to density matrix
    return np.outer(state, state.conjugate())


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestDensityMatrix:
    """Test that apply_operation works for QubitDensityMatrix"""

    num_qubits = [1, 2, 3]

    @pytest.mark.parametrize("num_q", num_qubits)
    def test_valid_density_matrix(self, num_q, ml_framework):
        """Test applying a valid density matrix to the state"""
        density_matrix = get_valid_density_matrix(num_q)
        # Convert density matrix to the given ML framework and ensure complex dtype
        density_matrix = math.asarray(density_matrix, like=ml_framework)
        density_matrix = math.cast(density_matrix, dtype=complex)  # ensure complex

        op = qml.QubitDensityMatrix(density_matrix, wires=range(num_q))

        # Create the initial state as zeros in the same framework and ensure complex dtype
        shape = (2,) * (2 * num_q)
        state = np.zeros(shape, dtype=np.complex128)
        state = math.asarray(state, like=ml_framework)
        state = math.cast(state, dtype=complex)

        # Apply operation
        result = qml.devices.qubit_mixed.apply_operation(op, state)

        # Reshape and cast expected result
        expected = math.reshape(density_matrix, shape)
        expected = math.cast(expected, dtype=complex)

        assert math.allclose(result, expected)

    @pytest.mark.parametrize("num_q", num_qubits)
    def test_batched_state(self, num_q, ml_framework):
        """Test applying density matrix to batched states"""
        batch_size = 3
        density_matrix = get_valid_density_matrix(num_q)
        density_matrix = math.asarray(density_matrix, like=ml_framework)
        density_matrix = math.cast(density_matrix, dtype=complex)

        op = qml.QubitDensityMatrix(density_matrix, wires=range(num_q))

        shape = (batch_size,) + (2,) * (2 * num_q)
        state = math.zeros(shape, like=ml_framework)
        state = math.cast(state, dtype=complex)

        result = qml.devices.qubit_mixed.apply_operation(op, state, is_state_batched=True)

        expected_single = math.reshape(density_matrix, (2,) * (2 * num_q))
        expected_single = math.cast(expected_single, dtype=complex)
        # Tile along batch dimension
        expected = math.stack([expected_single] * batch_size, axis=0)

        assert math.allclose(result, expected)

    @pytest.mark.parametrize("num_q", num_qubits)
    def test_batched_eigvals(self, num_q, ml_framework):
        """Test applying density matrix with batched eigenvalues"""

        density_matrix = get_valid_density_matrix(num_q)
        density_matrix = math.asarray(density_matrix, like=ml_framework)
        density_matrix = math.cast(density_matrix, dtype=complex)

        batched_params = math.asarray([0, 1, 2], like=ml_framework)
        op = qml.RX(batched_params, wires=0)

        shape = (2,) * (2 * num_q)
        state = math.zeros(shape, like=ml_framework)
        state = math.cast(state, dtype=complex)

        result = apply_operation(op, state)
        assert result.shape == (len(batched_params),) + shape

    def test_partial_trace_single_qubit_update(self, ml_framework):
        """Minimal test for partial tracing when applying QubitDensityMatrix to a subset of wires."""

        # Initial 2-qubit state as a (4,4) density matrix representing |00><00|
        # |00> in vector form = [1,0,0,0]
        # |00><00| as a 4x4 matrix = diag([1,0,0,0])
        initial_state = np.zeros((4, 4), dtype=complex)
        initial_state[0, 0] = 1.0
        initial_state = math.asarray(initial_state, like=ml_framework)

        # Define the single-qubit density matrix |+><+| = 0.5 * [[1,1],[1,1]]
        plus_state = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        plus_state = math.asarray(plus_state, like=ml_framework)

        # Apply QubitDensityMatrix on the first wire (wire=0)
        op = qml.QubitDensityMatrix(plus_state, wires=[0])

        # The expected final state should be |+><+| ⊗ |0><0|
        # |0><0| = [[1,0],[0,0]]
        zero_dm = np.array([[1, 0], [0, 0]], dtype=complex)
        expected = np.kron(plus_state, zero_dm)  # shape (4,4)
        expected = math.reshape(expected, [2] * 4)
        # Apply the operation
        result = qml.devices.qubit_mixed.apply_operation(op, initial_state)

        assert math.allclose(result, expected, atol=1e-8)

    def test_partial_trace_batched_update(self, ml_framework):
        """Minimal test for partial tracing when applying QubitDensityMatrix to a subset of wires, batched."""

        batch_size = 3

        # Initial 2-qubit state as a (4,4) density matrix representing |00><00| batched
        initial_state = np.zeros((batch_size, 4, 4), dtype=complex)
        for b in range(batch_size):
            initial_state[b, 0, 0] = 1.0
        initial_state = math.asarray(initial_state, like=ml_framework)

        # Define the single-qubit density matrix |+><+| = 0.5 * [[1,1],[1,1]]
        plus_state = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        plus_state = math.asarray(plus_state, like=ml_framework)

        # Apply QubitDensityMatrix on the first wire (wire=0)
        op = qml.QubitDensityMatrix(plus_state, wires=[0])

        # The expected final state should be |+><+| ⊗ |0><0| for each batch
        zero_dm = np.array([[1, 0], [0, 0]], dtype=complex)
        expected_single = np.kron(plus_state, zero_dm)  # shape (4,4)
        expected = np.stack([expected_single] * batch_size, axis=0)
        expected = math.reshape(expected, [batch_size] + [2] * 4)

        # Apply the operation
        result = qml.devices.qubit_mixed.apply_operation(op, initial_state, is_state_batched=True)

        assert math.allclose(result, expected, atol=1e-8)
