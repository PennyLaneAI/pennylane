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
from scipy.stats import unitary_group

import pennylane as qml
import pennylane.math as math
from pennylane.devices.qubit_mixed import apply_operation
from pennylane.devices.qubit_mixed.apply_operation import (
    GLOBALPHASE_WARNING,
    apply_operation_einsum,
    apply_operation_tensordot,
)
from pennylane.devices.qubit_mixed.constants import QUDIT_DIM

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]


def get_random_mixed_state(num_qubits):
    """
    Generates a random mixed state for testing purposes.

    Args:
        num_qubits (int): The number of qubits in the mixed state.

    Returns:
        np.ndarray: A tensor representing the random mixed state.
    """
    dim = QUDIT_DIM**num_qubits

    rng = np.random.default_rng(seed=4774)
    basis = unitary_group(dim=dim, seed=584545).rvs()
    schmidt_weights = rng.dirichlet(np.ones(dim), size=1).astype(complex)[0]
    mixed_state = np.zeros((dim, dim)).astype(complex)
    for i in range(dim):
        mixed_state += schmidt_weights[i] * np.outer(np.conj(basis[i]), basis[i])

    return mixed_state.reshape([QUDIT_DIM] * (2 * num_qubits))


@pytest.fixture
def three_qubit_state_fixture():
    """Fixture for a random three-qubit mixed state."""
    return get_random_mixed_state(3)


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
    def get_expected_state(cls, expanded_operator, state, num_q):
        """Finds expected state after applying operator"""
        # Convert the state into numpy
        state = np.asarray(state)
        shape = (QUDIT_DIM**num_q,) * 2
        flattened_state = state.reshape(shape)
        adjoint_matrix = np.conj(expanded_operator).T

        new_state = expanded_operator @ flattened_state @ adjoint_matrix
        return new_state.reshape([QUDIT_DIM] * (num_q * 2))

    @classmethod
    def expand_matrices(cls, op, num_q, batch_size=0):
        """Find expanded operator matrices, independent of qml implementation"""
        pre_wires_identity = np.eye(QUDIT_DIM ** op.wires[0])
        post_wires_identity = np.eye(QUDIT_DIM ** ((num_q - 1) - op.wires[-1]))
        mat = op.matrix()

        def expand_matrix(matrix):
            return reduce(np.kron, (pre_wires_identity, matrix, post_wires_identity))

        if batch_size:
            return [expand_matrix(mat[i]) for i in range(batch_size)]
        return expand_matrix(mat)

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
    def test_no_broadcasting(self, op, num_q, ml_framework):
        """
        Tests that unbatched operations are applied correctly to an unbatched state.

        Args:
            op (Operation): Quantum operation to apply.
            ml_framework (str): The machine learning framework in use (numpy, autograd, etc.).
        """
        state = qml.math.asarray(get_random_mixed_state(num_q), like=ml_framework)
        res = apply_operation(op, state)
        res_tensordot = apply_operation_tensordot(op, state)
        res_einsum = apply_operation_einsum(op, state)

        expanded_operator = self.expand_matrices(op, num_q)
        expected = self.get_expected_state(np.array(expanded_operator), np.array(state), num_q)

        # assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected), f"Operation {op} failed. {res} \n != {expected}"
        assert qml.math.allclose(
            res_tensordot, expected
        ), f"Tensordot and einsum results do not match. {res_tensordot} != {res_einsum}"

    @pytest.mark.parametrize("op", diagonal_ops)
    @pytest.mark.parametrize("num_q", num_qubits)
    def test_diagonal(self, op, num_q, ml_framework):
        """
        Tests that diagonal operations are applied correctly to an unbatched state.

        Args:
            op (Operation): Quantum operation to apply.
            ml_framework (str): The machine learning framework in use (numpy, autograd, etc.).
        """
        state_np = get_random_mixed_state(num_q)
        state = qml.math.asarray(state_np, like=ml_framework)
        res = apply_operation(op, state)

        expanded_operator = self.expand_matrices(op, num_q)
        expected = self.get_expected_state(expanded_operator, state_np, num_q)

        assert qml.math.allclose(res, expected), f"Operation {op} failed. {res} != {expected}"

    @pytest.mark.parametrize("num_q", num_qubits)
    def test_identity(self, num_q, ml_framework):
        """Tests that the identity operation is applied correctly to an unbatched state."""
        state_np = get_random_mixed_state(num_q)
        state = qml.math.asarray(state_np, like=ml_framework)
        op = qml.Identity(wires=0)
        res = apply_operation(op, state)

        assert qml.math.allclose(res, state), f"Operation {op} failed. {res} != {state}"

    @pytest.mark.parametrize("num_q", num_qubits)
    def test_globalphase(self, num_q, ml_framework):
        """Tests that the identity operation is applied correctly to an unbatched state."""
        state_np = get_random_mixed_state(num_q)
        state = qml.math.asarray(state_np, like=ml_framework)
        op = qml.GlobalPhase(np.pi / 7, wires=0)
        with pytest.warns(UserWarning, match=GLOBALPHASE_WARNING):
            res = apply_operation(op, state)

        assert qml.math.allclose(res, state), f"Operation {op} failed. {res} != {state}"

    @pytest.mark.parametrize("op", unbroadcasted_ops)
    @pytest.mark.parametrize("num_q", num_qubits)
    def test_unbroadcasted_ops_batched(self, op, num_q, ml_framework):
        """Test that unbroadcasted operations are applied correctly to batched states."""
        batch_size = self.num_batched
        state = np.array([get_random_mixed_state(num_q) for _ in range(batch_size)])
        state = qml.math.asarray(state, like=ml_framework)
        res = apply_operation(op, state, is_state_batched=True)

        expanded_operator = self.expand_matrices(op, num_q)
        expanded_operator = math.expand_matrix(op.matrix(), op.wires, wire_order=range(num_q))
        expected = np.array(
            [
                self.get_expected_state(
                    expanded_operator,
                    s,
                    num_q,
                )
                for s in state
            ]
        )

        # Make both res and expected the same shape
        res = np.array(res).reshape(expected.shape)

        assert qml.math.allclose(res, expected), f"Operation {op} failed. {res} != {expected}"

    @pytest.mark.parametrize("op", diagonal_ops)
    @pytest.mark.parametrize("num_q", num_qubits)
    def test_diagonal_ops_batched(self, op, num_q, ml_framework):
        """Test that diagonal operations are applied correctly to batched states."""
        batch_size = self.num_batched
        state = np.array([get_random_mixed_state(num_q) for _ in range(batch_size)])
        state = qml.math.asarray(state, like=ml_framework)
        res = apply_operation(op, state, is_state_batched=True)

        expanded_operator = self.expand_matrices(op, num_q)
        expected = np.array([self.get_expected_state(expanded_operator, s, num_q) for s in state])

        assert qml.math.allclose(res, expected), f"Operation {op} failed. {res} != {expected}"


class TestApplyGroverOperator:
    """Test that GroverOperator is applied correctly to mixed states."""

    @pytest.mark.parametrize(
        "num_wires, expected_method",
        [
            (2, "einsum"),
            (3, "tensordot"),
            (7, "tensordot"),
            (8, "tensordot"),
            (9, "custom"),
            # (13, "custom"),
        ],
    )
    def test_dispatch_method(self, num_wires, expected_method, mocker):
        """Test that the correct dispatch method is used based on the number of wires."""
        state = get_random_mixed_state(num_wires)

        op = qml.GroverOperator(wires=range(num_wires))

        spy_einsum = mocker.spy(qml.math, "einsum")
        spy_tensordot = mocker.spy(qml.math, "moveaxis")

        apply_operation(op, state)

        if expected_method == "einsum":
            assert spy_einsum.called
        elif expected_method == "tensordot":
            assert not spy_einsum.called
            assert spy_tensordot.called
        else:  # custom method
            assert not spy_einsum.called

    @pytest.mark.parametrize("num_wires", [2, 3, 7, 8, 9])
    def test_correctness(self, num_wires):
        """Test that the GroverOperator is applied correctly for various wire numbers."""
        state = get_random_mixed_state(num_wires)

        op = qml.GroverOperator(wires=range(num_wires))
        op_mat = op.matrix()
        flat_shape = op_mat.shape

        result = apply_operation(op, state)

        state_flat = state.reshape(flat_shape)
        expected = op_mat @ state_flat @ op_mat.conj().T

        assert np.allclose(result.reshape(flat_shape), expected)

    @pytest.mark.parametrize("num_wires", [2, 3, 7, 8, 9])
    def test_batched_state(self, num_wires):
        """Test that the GroverOperator works correctly with batched states."""
        batch_size = 3
        state = np.array([get_random_mixed_state(num_wires) for _ in range(batch_size)])

        op = qml.GroverOperator(wires=range(num_wires))
        op_mat = op.matrix()
        # Make new shape, considering the batch dimension as the first
        flat_shape = (batch_size,) + op_mat.shape

        result = apply_operation(op, state, is_state_batched=True)

        state_flat = state.reshape(flat_shape)
        expected = np.array([op.matrix() @ s @ op.matrix().conj().T for s in state_flat])

        assert np.allclose(result.reshape(flat_shape), expected)

    def test_interface_compatibility(self):
        """Test that the GroverOperator works with different interfaces."""
        num_wires = 5
        state = get_random_mixed_state(num_wires)

        op = qml.GroverOperator(wires=range(num_wires))

        # Test with numpy interface
        result_numpy = apply_operation(op, state)

        # Test with autograd interface
        import autograd.numpy as anp

        state_autograd = anp.array(state)
        result_autograd = apply_operation(op, state_autograd)

        assert np.allclose(result_numpy, result_autograd)


class TestApplyMultiControlledX:
    """Test that MultiControlledX is applied correctly to mixed states."""

    @pytest.mark.parametrize(
        "num_wires, expected_method",
        [
            (3, "tensordot"),
            (7, "tensordot"),
            (8, "tensordot"),
            (9, "custom"),
            # (13, "custom"),
        ],
    )
    def test_dispatch_method(self, num_wires, expected_method, mocker):
        """Test that the correct dispatch method is used based on the number of wires."""
        state = get_random_mixed_state(num_wires)

        op = qml.MultiControlledX(wires=range(num_wires))

        spy_einsum = mocker.spy(qml.math, "einsum")
        spy_tensordot = mocker.spy(qml.math, "moveaxis")

        apply_operation(op, state)

        if expected_method == "einsum":
            assert spy_einsum.called
        elif expected_method == "tensordot":
            assert not spy_einsum.called
            assert spy_tensordot.called
        else:  # custom method
            assert not spy_einsum.called

    @pytest.mark.parametrize("num_wires", [2, 3, 7, 8, 9])
    def test_correctness(self, num_wires):
        """Test that the MultiControlledX is applied correctly for various wire numbers."""
        state = get_random_mixed_state(num_wires)

        op = qml.MultiControlledX(wires=range(num_wires))
        op_mat = op.matrix()
        flat_shape = op_mat.shape

        result = apply_operation(op, state)

        state_flat = state.reshape(flat_shape)
        expected = op_mat @ state_flat @ op_mat.conj().T

        assert np.allclose(result.reshape(flat_shape), expected)

    @pytest.mark.parametrize("num_wires", [2, 3, 7, 8, 9])
    def test_batched_state(self, num_wires):
        """Test that the MultiControlledX works correctly with batched states."""
        batch_size = 3
        state = np.array([get_random_mixed_state(num_wires) for _ in range(batch_size)])

        op = qml.MultiControlledX(wires=range(num_wires))
        op_mat = op.matrix()
        # Make new shape, considering the batch dimension as the first
        flat_shape = (batch_size,) + op_mat.shape

        result = apply_operation(op, state, is_state_batched=True)

        state_flat = state.reshape(flat_shape)
        expected = np.array([op.matrix() @ s @ op.matrix().conj().T for s in state_flat])

        assert np.allclose(result.reshape(flat_shape), expected)

    def test_interface_compatibility(self):
        """Test that the MultiControlledX works with different interfaces."""
        num_wires = 5
        state = get_random_mixed_state(num_wires)

        op = qml.MultiControlledX(wires=range(num_wires))

        # Test with numpy interface
        result_numpy = apply_operation(op, state)

        # Test with autograd interface
        import autograd.numpy as anp

        state_autograd = anp.array(state)
        result_autograd = apply_operation(op, state_autograd)

        assert np.allclose(result_numpy, result_autograd)
