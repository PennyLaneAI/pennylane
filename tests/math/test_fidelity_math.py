# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for differentiable quantum entropies.
"""

import numpy as onp
import pytest

import pennylane as qml
from pennylane import numpy as np

pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class TestFidelityMath:
    """Tests for Fidelity function between two states (state vectors or density matrix)."""

    state_vectors = [
        ([1, 0], [0, 1], 0),
        ([0, 1], [0, 1], 1.0),
        ([1, 0], [1, 1] / np.sqrt(2), 0.5),
    ]

    density_mats = [
        ([[1, 0], [0, 0]], [[0.5, 0], [0, 0.5]], 0.5),
        ([[0, 0], [0, 1]], [[0.5, 0], [0, 0.5]], 0.5),
        ([[1, 0], [0, 0]], [[0.5, 0.5], [0.5, 0.5]], 0.5),
        ([[0, 0], [0, 1]], [[0.5, 0.5], [0.5, 0.5]], 0.5),
    ]

    array_funcs = [
        lambda x: x,
        onp.array,
        np.array,
        jnp.array,
        torch.tensor,
        tf.Variable,
        tf.constant,
    ]

    check_state = [True, False]

    @pytest.mark.parametrize("states_fid", state_vectors)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("func", array_funcs)
    def test_state_vector_fidelity(self, states_fid, check_state, func):
        """Test fidelity between different quantum states."""
        state0, state1, fid = states_fid
        state0 = func(state0)
        state1 = func(state1)

        fidelity = qml.math.fidelity_statevector(state0, state1, check_state)
        assert qml.math.allclose(fid, fidelity)

    @pytest.mark.parametrize("states_fid", density_mats)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("func", array_funcs)
    def test_density_mat_fidelity(self, states_fid, check_state, func):
        """Test fidelity between different quantum states."""
        state0, state1, fid = states_fid
        state0 = func(state0)
        state1 = func(state1)

        fidelity = qml.math.fidelity(state0, state1, check_state)
        assert qml.math.allclose(fid, fidelity)

    state_wrong_amp = [([0.5, 0], [0, 1]), ([0, 1], [0.5, 0])]

    @pytest.mark.parametrize("state0,state1", state_wrong_amp)
    def test_state_vector_wrong_amplitudes(self, state0, state1):
        """Test that a message is raised when a state does not have right norm"""
        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            qml.math.fidelity_statevector(state0, state1, check_state=True)

    state_wrong_shape = [([0, 1, 1], [0, 1]), ([0, 1], [0, 1, 1])]

    @pytest.mark.parametrize("state0,state1", state_wrong_shape)
    def test_state_vector_wrong_shape(self, state0, state1):
        """Test that a message is raised when the state does not have the right shape."""
        with pytest.raises(ValueError, match="State vector must be of shape"):
            qml.math.fidelity_statevector(state0, state1, check_state=True)

    d_mat_wrong_shape = [
        ([[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 0], [0, 0]]),
        ([[1, 0], [0, 0]], [[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
    ]

    @pytest.mark.parametrize("state0,state1", d_mat_wrong_shape)
    def test_density_matrix_wrong_shape(self, state0, state1):
        """Test that a message is raised when the density matrix does not have the right shape."""
        with pytest.raises(ValueError, match="Density matrix must be of shape"):
            qml.math.fidelity(state0, state1, check_state=True)

    d_mat_wrong_trace = [
        ([[1, 0], [0, -1]], [[1, 0], [0, 0]]),
        ([[1, 0], [0, 0]], [[1, 0], [0, -1]]),
    ]

    @pytest.mark.parametrize("state0,state1", d_mat_wrong_trace)
    def test_density_matrix_wrong_trace(self, state0, state1):
        """Test that a message is raised when the density matrix does not have the right trace."""
        with pytest.raises(ValueError, match="The trace of the density matrix should be one"):
            qml.math.fidelity(state0, state1, check_state=True)

    d_mat_not_hermitian = [
        ([[1, 1], [0, 0]], [[1, 0], [0, 0]]),
        ([[1, 0], [0, 0]], [[1, 1], [0, 0]]),
    ]

    @pytest.mark.parametrize("state0,state1", d_mat_not_hermitian)
    def test_density_matrix_not_hermitian(self, state0, state1):
        """Test that a message is raised when the density matrix is not Hermitian."""
        with pytest.raises(ValueError, match="The matrix is not Hermitian"):
            qml.math.fidelity(state0, state1, check_state=True)

    d_mat_not_positive = [
        ([[2, 0], [0, -1]], [[1, 0], [0, 0]]),
        ([[1, 0], [0, 0]], [[2, 0], [0, -1]]),
    ]

    @pytest.mark.parametrize("state0,state1", d_mat_not_positive)
    def test_density_matrix_not_positive_semi_def(self, state0, state1):
        """Test that a message is raised when the density matrix is not positive semi def."""
        with pytest.raises(ValueError, match="The matrix is not positive semi"):
            qml.math.fidelity(state0, state1, check_state=True)

    def test_same_number_wires(self):
        """Test that the two states must act on the same number of wires"""
        state0 = [0, 1, 0, 0]
        state1 = [1, 0]
        with pytest.raises(
            qml.QuantumFunctionError, match="The two states must have the same number of wires"
        ):
            qml.math.fidelity_statevector(state0, state1, check_state=True)

    def test_same_number_wires_dm(self):
        """Test that the two states must act on the same number of wires"""
        state0 = np.diag([0, 1, 0, 0])
        state1 = [[1, 0], [0, 0]]
        with pytest.raises(
            qml.QuantumFunctionError, match="The two states must have the same number of wires"
        ):
            qml.math.fidelity(state0, state1, check_state=True)

    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("func", array_funcs)
    def test_broadcast_sv_sv(self, check_state, func):
        """Test broadcasting works for fidelity and state vectors"""
        state0 = func([[1, 0], [0, 1], [1, 0]])
        state1 = func([[0, 1], [0, 1], [1, 1] / np.sqrt(2)])
        expected = [0, 1, 0.5]

        fidelity = qml.math.fidelity_statevector(state0, state1, check_state)
        assert qml.math.allclose(fidelity, expected)

    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("func", array_funcs)
    def test_broadcast_sv_sv_unbatched(self, check_state, func):
        """Test broadcasting works for fidelity and state vectors when one input is unbatched"""
        state0 = func([1, 0])
        state1 = func([[0, 1], [1, 0], [1, 1] / np.sqrt(2)])
        expected = [0, 1, 0.5]

        fidelity = qml.math.fidelity_statevector(state0, state1, check_state)
        assert qml.math.allclose(fidelity, expected)

    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("func", array_funcs)
    def test_broadcast_dm_dm(self, check_state, func):
        """Test broadcasting works for fidelity and density matrices"""
        state0 = func([[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[1, 0], [0, 0]], [[0, 0], [0, 1]]])
        state1 = func(
            [
                [[0.5, 0], [0, 0.5]],
                [[0.5, 0], [0, 0.5]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, 0.5], [0.5, 0.5]],
            ]
        )
        expected = [0.5, 0.5, 0.5, 0.5]

        fidelity = qml.math.fidelity(state0, state1, check_state)
        assert qml.math.allclose(fidelity, expected)

    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("func", array_funcs)
    def test_broadcast_dm_dm_unbatched(self, check_state, func):
        """Test broadcasting works for fidelity and density matrices when one input is unbatched"""
        state0 = func(
            [
                [[0.5, -0.5], [-0.5, 0.5]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
            ]
        )
        state1 = func([[0.5, 0.5], [0.5, 0.5]])

        expected = [0, 1, 0.5, 0.5]

        fidelity = qml.math.fidelity(state0, state1, check_state)
        assert qml.math.allclose(fidelity, expected)


class TestGradient:
    """Test the gradient of qml.math.fidelity"""

    @staticmethod
    def cost_fn_single(x):
        state1 = qml.math.diag([qml.math.cos(x / 2) ** 2, qml.math.sin(x / 2) ** 2])
        state2 = qml.math.convert_like(qml.math.diag([1, 0]), state1)
        return qml.math.fidelity(state1, state2) + qml.math.fidelity(state2, state1)

    @staticmethod
    def cost_fn_multi(x):
        state1 = qml.math.diag([qml.math.cos(x / 2) ** 2, 0, 0, qml.math.sin(x / 2) ** 2])
        state2 = qml.math.convert_like(qml.math.diag([1, 0, 0, 0]), state1)
        return qml.math.fidelity(state1, state2) + qml.math.fidelity(state2, state1)

    @staticmethod
    def expected_res(x):
        return 2 * qml.math.cos(x / 2) ** 2

    @staticmethod
    def expected_grad(x):
        return -qml.math.sin(x)

    @pytest.mark.autograd
    @pytest.mark.parametrize("x", [0.0, 0.456, np.pi / 2])
    def test_single_wire_autograd(self, x, tol):
        """Test gradients are correct for a single wire for autograd"""
        x = np.array(x)
        res = self.cost_fn_single(x)
        grad = qml.grad(self.cost_fn_single)(x)

        assert qml.math.allclose(res, self.expected_res(x), tol)
        assert qml.math.allclose(grad, self.expected_grad(x), tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("x", [0.0, 0.456, np.pi / 2])
    def test_single_wire_jax(self, x, tol):
        """Test gradients are correct for a single wire for jax"""
        x = jnp.array(x)
        res = self.cost_fn_single(x)
        grad = jax.grad(self.cost_fn_single)(x)

        assert qml.math.allclose(res, self.expected_res(x), tol)
        assert qml.math.allclose(grad, self.expected_grad(x), tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("x", [0.0, 0.456, np.pi / 2])
    def test_single_wire_jax_jit(self, x, tol):
        """Test gradients are correct for a single wire for jax-jit"""
        x = jnp.array(x)

        jitted_cost = jax.jit(self.cost_fn_single)
        res = jitted_cost(x)
        grad = jax.grad(jitted_cost)(x)

        assert qml.math.allclose(res, self.expected_res(x), tol)
        assert qml.math.allclose(grad, self.expected_grad(x), tol)

    @pytest.mark.torch
    @pytest.mark.parametrize("x", [0.0, 0.456, np.pi / 2])
    def test_single_wire_torch(self, x, tol):
        """Test gradients are correct for a single wire for torch"""
        x = torch.from_numpy(np.array(x)).requires_grad_()
        res = self.cost_fn_single(x)
        res.backward()
        grad = x.grad

        assert qml.math.allclose(res, self.expected_res(x), tol)
        assert qml.math.allclose(grad, self.expected_grad(x), tol)

    @pytest.mark.tf
    @pytest.mark.parametrize("x", [0.0, 0.456, np.pi / 2])
    def test_single_wire_tf(self, x, tol):
        """Test gradients are correct for a single wire for tf"""
        x = tf.Variable(x, trainable=True)

        with tf.GradientTape() as tape:
            res = self.cost_fn_single(x)

        grad = tape.gradient(res, x)

        assert qml.math.allclose(res, self.expected_res(x), tol)
        assert qml.math.allclose(grad, self.expected_grad(x), tol)

    @pytest.mark.autograd
    @pytest.mark.parametrize("x", [0.0, 0.456, np.pi / 2])
    def test_multi_wire_autograd(self, x, tol):
        """Test gradients are correct for multiple wires for autograd"""
        x = np.array(x)
        res = self.cost_fn_multi(x)
        grad = qml.grad(self.cost_fn_multi)(x)

        assert qml.math.allclose(res, self.expected_res(x), tol)
        assert qml.math.allclose(grad, self.expected_grad(x), tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("x", [0.0, 0.456, np.pi / 2])
    def test_multi_wire_jax(self, x, tol):
        """Test gradients are correct for multiple wires for jax"""
        x = jnp.array(x)
        res = self.cost_fn_multi(x)
        grad = jax.grad(self.cost_fn_multi)(x)

        assert qml.math.allclose(res, self.expected_res(x), tol)
        assert qml.math.allclose(grad, self.expected_grad(x), tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("x", [0.0, 0.456, np.pi / 2])
    def test_multi_wire_jax_jit(self, x, tol):
        """Test gradients are correct for multiple wires for jax-jit"""
        x = jnp.array(x)

        jitted_cost = jax.jit(self.cost_fn_multi)
        res = jitted_cost(x)
        grad = jax.grad(jitted_cost)(x)

        assert qml.math.allclose(res, self.expected_res(x), tol)
        assert qml.math.allclose(grad, self.expected_grad(x), tol)

    @pytest.mark.torch
    @pytest.mark.parametrize("x", [0.0, 0.456, np.pi / 2])
    def test_multi_wire_torch(self, x, tol):
        """Test gradients are correct for multiple wires for torch"""
        x = torch.from_numpy(np.array(x)).requires_grad_()
        res = self.cost_fn_multi(x)
        res.backward()
        grad = x.grad

        assert qml.math.allclose(res, self.expected_res(x), tol)
        assert qml.math.allclose(grad, self.expected_grad(x), tol)

    @pytest.mark.tf
    @pytest.mark.parametrize("x", [0.0, 0.456, np.pi / 2])
    def test_multi_wire_tf(self, x, tol):
        """Test gradients are correct for a multiple wires for tf"""
        x = tf.Variable(x, trainable=True)

        with tf.GradientTape() as tape:
            res = self.cost_fn_multi(x)

        grad = tape.gradient(res, x)

        assert qml.math.allclose(res, self.expected_res(x), tol)
        assert qml.math.allclose(grad, self.expected_grad(x), tol)
