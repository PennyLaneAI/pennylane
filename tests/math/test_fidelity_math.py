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
"""Unit tests for differentiable quantum entropies."""

import numpy as onp
import pytest

import pennylane as qml
from pennylane import numpy as np

pytestmark = pytest.mark.all_interfaces

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
        with pytest.raises(ValueError, match="The two states must have the same number of wires"):
            qml.math.fidelity_statevector(state0, state1, check_state=True)

    def test_same_number_wires_dm(self):
        """Test that the two states must act on the same number of wires"""
        state0 = np.diag([0, 1, 0, 0])
        state1 = [[1, 0], [0, 0]]
        with pytest.raises(ValueError, match="The two states must have the same number of wires"):
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


def cost_fn_single(x):
    first_term = qml.math.convert_like(qml.math.diag([1.0, 0]), x)
    second_term = qml.math.convert_like(qml.math.diag([0, 1.0]), x)

    x = qml.math.cast_like(x, first_term)
    if len(qml.math.shape(x)) == 0:
        state1 = qml.math.cos(x / 2) ** 2 * first_term + qml.math.sin(x / 2) ** 2 * second_term
    else:
        # broadcasting
        x = x[:, None, None]
        state1 = qml.math.cos(x / 2) ** 2 * first_term + qml.math.sin(x / 2) ** 2 * second_term

    state2 = qml.math.convert_like(qml.math.diag([1, 0]), state1)

    return qml.math.fidelity(state1, state2) + qml.math.fidelity(state2, state1)


def cost_fn_multi1(x):
    first_term = qml.math.convert_like(qml.math.diag([1.0, 0, 0, 0]), x)
    second_term = qml.math.convert_like(qml.math.diag([0, 0, 0, 1.0]), x)

    x = qml.math.cast_like(x, first_term)

    if len(qml.math.shape(x)) == 0:
        state1 = qml.math.cos(x / 2) ** 2 * first_term + qml.math.sin(x / 2) ** 2 * second_term
    else:
        # broadcasting
        x = x[:, None, None]
        state1 = qml.math.cos(x / 2) ** 2 * first_term + qml.math.sin(x / 2) ** 2 * second_term

    state2 = qml.math.convert_like(qml.math.diag([1, 0, 0, 0]), state1)

    return qml.math.fidelity(state1, state2) + qml.math.fidelity(state2, state1)


def cost_fn_multi2(x):
    first_term = qml.math.convert_like(np.ones((4, 4)) / 4, x)
    second_term = np.zeros((4, 4))
    second_term[1:3, 1:3] = np.array([[1, -1], [-1, 1]]) / 2
    second_term = qml.math.convert_like(second_term, x)

    x = qml.math.cast_like(x, first_term)

    if len(qml.math.shape(x)) == 0:
        state1 = qml.math.cos(x / 2) ** 2 * first_term + qml.math.sin(x / 2) ** 2 * second_term
    else:
        # broadcasting
        x = x[:, None, None]
        state1 = qml.math.cos(x / 2) ** 2 * first_term + qml.math.sin(x / 2) ** 2 * second_term

    state2 = qml.math.convert_like(qml.math.diag([1, 0, 0, 0]), state1)

    return qml.math.fidelity(state1, state2) + qml.math.fidelity(state2, state1)


def expected_res_single(x):
    return 2 * qml.math.cos(x / 2) ** 2


def expected_res_multi1(x):
    return 2 * qml.math.cos(x / 2) ** 2


def expected_res_multi2(x):
    return qml.math.cos(x / 2) ** 2 / 2


def expected_grad_single(x):
    return -qml.math.sin(x)


def expected_grad_multi1(x):
    return -qml.math.sin(x)


def expected_grad_multi2(x):
    return -qml.math.sin(x) / 4


class TestGradient:
    """Test the gradient of qml.math.fidelity"""

    # pylint: disable=too-many-arguments

    cost_fns = [
        (cost_fn_single, expected_res_single, expected_grad_single),
        (cost_fn_multi1, expected_res_multi1, expected_grad_multi1),
        (cost_fn_multi2, expected_res_multi2, expected_grad_multi2),
    ]

    @pytest.mark.autograd
    @pytest.mark.parametrize("x", [0.0, 1e-7, 0.456, np.pi / 2 - 1e-7, np.pi / 2])
    @pytest.mark.parametrize("cost_fn, expected_res, expected_grad", cost_fns)
    def test_grad_autograd(self, x, cost_fn, expected_res, expected_grad, tol):
        """Test gradients are correct for autograd"""
        x = np.array(x)
        res = cost_fn(x)
        grad = qml.grad(cost_fn)(x)

        assert qml.math.allclose(res, expected_res(x), tol)
        assert qml.math.allclose(grad, expected_grad(x), tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("x", [0.0, 1e-7, 0.456, np.pi / 2 - 1e-7, np.pi / 2])
    @pytest.mark.parametrize("cost_fn, expected_res, expected_grad", cost_fns)
    def test_grad_jax(self, x, cost_fn, expected_res, expected_grad, tol):
        """Test gradients are correct for jax"""
        x = jnp.array(x)
        res = cost_fn(x)
        grad = jax.grad(cost_fn)(x)

        assert qml.math.allclose(res, expected_res(x), tol)
        assert qml.math.allclose(grad, expected_grad(x), tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("x", [0.0, 1e-7, 0.456, np.pi / 2 - 1e-7, np.pi / 2])
    @pytest.mark.parametrize("cost_fn, expected_res, expected_grad", cost_fns)
    def test_grad_jax_jit(self, x, cost_fn, expected_res, expected_grad, tol):
        """Test gradients are correct for jax-jit"""
        x = jnp.array(x)

        jitted_cost = jax.jit(cost_fn)
        res = jitted_cost(x)
        grad = jax.grad(jitted_cost)(x)

        assert qml.math.allclose(res, expected_res(x), tol)
        assert qml.math.allclose(grad, expected_grad(x), tol)

    @pytest.mark.torch
    @pytest.mark.parametrize("x", [0.0, 1e-7, 0.456, np.pi / 2 - 1e-7, np.pi / 2])
    @pytest.mark.parametrize("cost_fn, expected_res, expected_grad", cost_fns)
    def test_grad_torch(self, x, cost_fn, expected_res, expected_grad, tol):
        """Test gradients are correct for torch"""
        x = torch.from_numpy(np.array(x)).requires_grad_()
        res = cost_fn(x)
        res.backward()
        grad = x.grad

        assert qml.math.allclose(res, expected_res(x), tol)
        assert qml.math.allclose(grad, expected_grad(x), tol)

    @pytest.mark.autograd
    @pytest.mark.parametrize("cost_fn, expected_res, expected_grad", cost_fns)
    def test_broadcast_autograd(self, cost_fn, expected_res, expected_grad, tol):
        """Test gradients are correct for a broadcasted input for autograd"""
        x = np.array([0.0, 1e-7, 0.456, np.pi / 2 - 1e-7, np.pi / 2])
        res = cost_fn(x)
        grad = qml.math.diag(qml.jacobian(cost_fn)(x))

        assert qml.math.allclose(res, expected_res(x), tol)
        assert qml.math.allclose(grad, expected_grad(x), tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("cost_fn, expected_res, expected_grad", cost_fns)
    def test_broadcast_jax(self, cost_fn, expected_res, expected_grad, tol):
        """Test gradients are correct for a broadcasted input for jax"""
        x = jnp.array([0.0, 1e-7, 0.456, np.pi / 2 - 1e-7, np.pi / 2])
        res = cost_fn(x)
        grad = qml.math.diag(jax.jacobian(cost_fn)(x))

        assert qml.math.allclose(res, expected_res(x), tol)
        assert qml.math.allclose(grad, expected_grad(x), tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("cost_fn, expected_res, expected_grad", cost_fns)
    def test_broadcast_jax_jit(self, cost_fn, expected_res, expected_grad, tol):
        """Test gradients are correct for a broadcasted input for jax-jit"""
        x = jnp.array([0.0, 1e-7, 0.456, np.pi / 2 - 1e-7, np.pi / 2])

        jitted_cost = jax.jit(cost_fn)
        res = jitted_cost(x)
        grad = qml.math.diag(jax.jacobian(jitted_cost)(x))

        assert qml.math.allclose(res, expected_res(x), tol)
        assert qml.math.allclose(grad, expected_grad(x), tol)

    @pytest.mark.torch
    @pytest.mark.parametrize("cost_fn, expected_res, expected_grad", cost_fns)
    def test_broadcast_torch(self, cost_fn, expected_res, expected_grad, tol):
        """Test gradients are correct for a broadcasted input for torch"""
        x = torch.from_numpy(
            np.array([0.0, 1e-7, 0.456, np.pi / 2 - 1e-7, np.pi / 2])
        ).requires_grad_()

        res = cost_fn(x)
        grad = qml.math.diag(torch.autograd.functional.jacobian(cost_fn, x))

        assert qml.math.allclose(res, expected_res(x), tol)
        assert qml.math.allclose(grad, expected_grad(x), tol)
