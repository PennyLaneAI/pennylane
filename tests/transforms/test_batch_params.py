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
Unit tests for the batch params transform.
"""
# pylint:disable=comparison-with-callable
import functools

import pytest

import pennylane as qml
from pennylane import numpy as np


def test_simple_circuit(mocker):
    """Test that batching works for a simple circuit"""
    dev = qml.device("default.qubit", wires=3)

    @qml.batch_params
    @qml.qnode(dev, interface="autograd")
    def circuit(data, x, weights):
        qml.templates.AmplitudeEmbedding(data, wires=[0, 1, 2], normalize=True)
        qml.RX(x, wires=0)
        qml.RY(0.2, wires=1)
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 2])

    batch_size = 5
    data = np.random.random((batch_size, 8))
    x = np.linspace(0.1, 0.5, batch_size, requires_grad=True)
    weights = np.ones((batch_size, 10, 3, 3), requires_grad=True)

    spy = mocker.spy(circuit.device, "execute")
    res = circuit(data, x, weights)
    assert res.shape == (batch_size, 4)
    assert len(spy.call_args[0][0]) == batch_size


def test_simple_circuit_one_batch(mocker):
    """Test that batching works for a simple circuit when the batch size is 1"""
    dev = qml.device("default.qubit", wires=3)

    @qml.batch_params
    @qml.qnode(dev, interface="autograd")
    def circuit(data, x, weights):
        qml.templates.AmplitudeEmbedding(data, wires=[0, 1, 2], normalize=True)
        qml.RX(x, wires=0)
        qml.RY(0.2, wires=1)
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 2])

    batch_size = 1
    data = np.random.random((batch_size, 8))
    x = np.linspace(0.1, 0.5, batch_size, requires_grad=True)
    weights = np.ones((batch_size, 10, 3, 3), requires_grad=True)

    spy = mocker.spy(circuit.device, "execute")
    res = circuit(data, x, weights)
    assert res.shape == (batch_size, 4)
    assert len(spy.call_args[0][0]) == batch_size


def test_simple_circuit_with_prep(mocker):
    """Test that batching works for a simple circuit with a state preparation"""
    dev = qml.device("default.qubit", wires=3)

    init_state = np.array([0, 0, 0, 0, 1, 0, 0, 0], requires_grad=False)

    @qml.batch_params
    @qml.qnode(dev, interface="autograd")
    def circuit(data, x, weights):
        qml.StatePrep(init_state, wires=[0, 1, 2])
        qml.RX(x, wires=0)
        qml.RY(0.2, wires=1)
        qml.RZ(data, wires=1)
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 2])

    batch_size = 5
    data = np.random.random((batch_size,))
    x = np.linspace(0.1, 0.5, batch_size, requires_grad=True)
    weights = np.ones((batch_size, 10, 3, 3), requires_grad=True)

    spy = mocker.spy(circuit.device, "execute")
    res = circuit(data, x, weights)
    assert res.shape == (batch_size, 4)
    assert len(spy.call_args[0][0]) == batch_size


def test_basic_entangler_layers(mocker):
    """Test that batching works for BasicEngtanglerLayers"""
    dev = qml.device("default.qubit", wires=2)

    @qml.batch_params
    @qml.qnode(dev, interface="autograd")
    def circuit(weights):
        qml.templates.BasicEntanglerLayers(weights, wires=[0, 1])
        qml.RY(0.2, wires=1)
        return qml.probs(wires=[0, 1])

    batch_size = 5
    weights = np.random.random((batch_size, 2, 2))

    spy = mocker.spy(circuit.device, "execute")
    res = circuit(weights)
    assert res.shape == (batch_size, 4)
    assert len(spy.call_args[0][0]) == batch_size


def test_angle_embedding(mocker):
    """Test that batching works for AngleEmbedding"""
    dev = qml.device("default.qubit", wires=3)

    @qml.batch_params
    @qml.qnode(dev, interface="autograd")
    def circuit(data):
        qml.templates.AngleEmbedding(data, wires=[0, 1, 2])
        qml.RY(0.2, wires=1)
        return qml.probs(wires=[0, 2])

    batch_size = 5
    data = np.random.random((batch_size, 3))

    spy = mocker.spy(circuit.device, "execute")
    res = circuit(data)
    assert res.shape == (batch_size, 4)
    assert len(spy.call_args[0][0]) == batch_size


def test_mottonenstate_preparation(mocker):
    """Test that batching works for MottonenStatePreparation"""
    dev = qml.device("default.qubit", wires=3)

    @qml.batch_params
    @qml.qnode(dev, interface="autograd")
    def circuit(data, weights):
        qml.templates.MottonenStatePreparation(data, wires=[0, 1, 2])
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 1, 2])

    batch_size = 3

    # create a batched input statevector
    data = np.random.random((batch_size, 2**3))
    data /= np.linalg.norm(data, axis=1).reshape(-1, 1)  # normalize
    weights = np.random.random((batch_size, 10, 3, 3))

    spy = mocker.spy(circuit.device, "execute")
    res = circuit(data, weights)
    assert res.shape == (batch_size, 2**3)
    assert len(spy.call_args[0][0]) == batch_size

    # check the results against individually executed circuits (no batching)
    @qml.qnode(dev)
    def circuit2(data, weights):
        qml.templates.MottonenStatePreparation(data, wires=[0, 1, 2])
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 1, 2])

    indiv_res = []
    for state, weight in zip(data, weights):
        indiv_res.append(circuit2(state, weight))
    assert np.allclose(res, indiv_res)


def test_qubit_state_prep(mocker):
    """Test that batching works for StatePrep"""
    dev = qml.device("default.qubit", wires=3)

    @qml.batch_params
    @qml.qnode(dev, interface="autograd")
    def circuit(data, weights):
        qml.StatePrep(data, wires=[0, 1, 2])
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 1, 2])

    batch_size = 3

    # create a batched input statevector
    data = np.random.random((batch_size, 2**3))
    data /= np.linalg.norm(data, axis=1).reshape(-1, 1)  # normalize
    weights = np.random.random((batch_size, 10, 3, 3))

    spy = mocker.spy(circuit.device, "execute")
    res = circuit(data, weights)
    assert res.shape == (batch_size, 2**3)
    assert len(spy.call_args[0][0]) == batch_size

    # check the results against individually executed circuits (no batching)
    @qml.qnode(dev)
    def circuit2(data, weights):
        qml.StatePrep(data, wires=[0, 1, 2])
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 1, 2])

    indiv_res = []
    for state, weight in zip(data, weights):
        indiv_res.append(circuit2(state, weight))
    assert np.allclose(res, indiv_res)


def test_multi_returns():
    """Test that batching works for a simple circuit with multiple returns"""
    dev = qml.device("default.qubit", wires=3)

    @qml.batch_params
    @qml.qnode(dev)
    def circuit(data, x, weights):
        qml.templates.AmplitudeEmbedding(data, wires=[0, 1, 2], normalize=True)
        qml.RX(x, wires=0)
        qml.RY(0.2, wires=1)
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 2])

    batch_size = 6
    data = np.random.random((batch_size, 8))
    x = np.linspace(0.1, 0.5, batch_size, requires_grad=True)
    weights = np.ones((batch_size, 10, 3, 3), requires_grad=True)

    res = circuit(data, x, weights)

    assert isinstance(res, tuple)
    assert len(res) == 2

    assert res[0].shape == (batch_size,)
    assert res[1].shape == (batch_size, 4)


def test_shot_vector():
    """Test that batching works for a simple circuit with a shot vector"""
    # pylint:disable=not-an-iterable
    dev = qml.device("default.qubit", wires=3)

    @qml.batch_params
    @qml.set_shots((100, (200, 3), 300))
    @qml.qnode(dev)
    def circuit(data, x, weights):
        qml.templates.AngleEmbedding(data, wires=[0, 1, 2])
        qml.RX(x, wires=0)
        qml.RY(0.2, wires=1)
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 2])

    batch_size = 6
    data = np.random.random((batch_size, 3))
    x = np.linspace(0.1, 0.5, batch_size, requires_grad=True)
    weights = np.ones((batch_size, 10, 3, 3), requires_grad=True)

    res = circuit(data, x, weights)

    assert isinstance(res, tuple)
    assert len(res) == 5
    # pylint:disable=not-an-iterable
    assert all(shot_res.shape == (batch_size, 4) for shot_res in res)


def test_multi_returns_shot_vector():
    """Test that batching works for a simple circuit with multiple returns
    and with a shot vector"""
    dev = qml.device("default.qubit", wires=3)

    @qml.batch_params
    @qml.set_shots((100, (200, 3), 300))
    @qml.qnode(dev)
    def circuit(data, x, weights):
        qml.templates.AngleEmbedding(data, wires=[0, 1, 2])
        qml.RX(x, wires=0)
        qml.RY(0.2, wires=1)
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 2])

    batch_size = 6
    data = np.random.random((batch_size, 3))
    x = np.linspace(0.1, 0.5, batch_size, requires_grad=True)
    weights = np.ones((batch_size, 10, 3, 3), requires_grad=True)

    res = circuit(data, x, weights)

    assert isinstance(res, tuple)
    assert len(res) == 5
    assert all(isinstance(shot_res, tuple) for shot_res in res)
    assert all(len(shot_res) == 2 for shot_res in res)
    assert all(shot_res[0].shape == (batch_size,) for shot_res in res)
    assert all(shot_res[1].shape == (batch_size, 4) for shot_res in res)


class TestDiffSingle:
    """Test gradients for a single measurement"""

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift"])
    def test_autograd(self, diff_method, tol):
        """Test derivatives when using autograd"""
        dev = qml.device("default.qubit", wires=2)

        @qml.batch_params
        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.1, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        def cost(x):
            return np.sum(circuit(x))

        batch_size = 3
        x = np.linspace(0.1, 0.5, batch_size, requires_grad=True)

        res = qml.grad(cost)(x)
        expected = -np.sin(0.1) * np.sin(x)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto", "jax"])
    def test_jax(self, diff_method, tol, interface):
        """Test derivatives when using JAX."""
        import jax

        jnp = jax.numpy
        dev = qml.device("default.qubit", wires=2)

        @qml.batch_params
        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.1, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        def cost(x):
            return jnp.sum(circuit(x))

        batch_size = 3
        x = jnp.linspace(0.1, 0.5, batch_size)

        res = jax.grad(cost)(x)
        expected = -np.sin(0.1) * np.sin(x)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ["adjoint", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto", "jax", "jax-jit"])
    def test_jax_jit(self, diff_method, interface, tol):
        """Test derivatives when using JAX and JIT."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit", wires=2)

        @qml.batch_params
        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.1, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        @jax.jit
        def cost(x):
            return jnp.sum(circuit(x))

        batch_size = 3
        x = jnp.linspace(0.1, 0.5, batch_size)

        res = jax.grad(cost)(x)
        expected = -np.sin(0.1) * np.sin(x)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.torch
    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto", "torch"])
    def test_torch(self, diff_method, tol, interface):
        """Test derivatives when using Torch"""
        import torch

        dev = qml.device("default.qubit", wires=2)

        @qml.batch_params
        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.1, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        def cost(x):
            return torch.sum(circuit(x))

        batch_size = 3
        x = torch.linspace(0.1, 0.5, batch_size, requires_grad=True)

        loss = cost(x)
        loss.backward()

        res = x.grad
        expected = -np.sin(0.1) * torch.sin(x)
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto"])
    def test_tf(self, diff_method, tol, interface):
        """Test derivatives when using TF"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        @qml.batch_params
        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.1, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        def cost(x):
            return tf.reduce_sum(circuit(x))

        batch_size = 3
        x = tf.Variable(tf.linspace(0.1, 0.5, batch_size))

        with tf.GradientTape() as tape:
            loss = cost(x)

        res = tape.gradient(loss, x)
        expected = -np.sin(0.1) * tf.sin(x)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("interface", ["auto", "tf-autograph"])
    def test_tf_autograph(self, tol, interface):
        """Test derivatives when using TF and autograph"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        @qml.batch_params
        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.1, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        @tf.function
        def cost(x):
            return tf.reduce_sum(circuit(x))

        batch_size = 3
        x = tf.Variable(tf.linspace(0.1, 0.5, batch_size))

        with tf.GradientTape() as tape:
            loss = cost(x)

        res = tape.gradient(loss, x)
        expected = -np.sin(0.1) * tf.sin(x)
        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestDiffMulti:
    """Test gradients for multiple measurements"""

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_autograd(self, diff_method, tol):
        """Test derivatives when using autograd"""
        dev = qml.device("default.qubit", wires=2)

        @qml.batch_params
        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        def cost(x):
            res = circuit(x)
            return qml.math.concatenate([qml.math.expand_dims(res[0], 1), res[1]], axis=1)

        batch_size = 3
        x = np.linspace(0.1, 0.5, batch_size, requires_grad=True)

        res = cost(x)
        expected = qml.math.transpose(
            qml.math.stack(
                [
                    np.cos(x),
                    np.cos(x / 2) ** 2,
                    np.zeros_like(x),
                    np.zeros_like(x),
                    np.sin(x / 2) ** 2,
                ]
            )
        )
        assert qml.math.allclose(res, expected, atol=tol)

        grad = qml.jacobian(cost)(x)
        expected = qml.math.stack(
            [-np.sin(x), -np.sin(x) / 2, np.zeros_like(x), np.zeros_like(x), np.sin(x) / 2]
        ) * qml.math.expand_dims(np.eye(batch_size), 1)

        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto", "jax", "jax-jit"])
    def test_jax(self, diff_method, tol, interface):
        """Test derivatives when using JAX"""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit", wires=2)

        @functools.partial(qml.batch_params, all_operations=True)
        @qml.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        batch_size = 3
        x = jnp.linspace(0.1, 0.5, batch_size)

        res = circuit(x)
        expected = (
            jnp.cos(x),
            qml.math.transpose(
                qml.math.stack(
                    [jnp.cos(x / 2) ** 2, jnp.zeros_like(x), jnp.zeros_like(x), jnp.sin(x / 2) ** 2]
                )
            ),
        )

        assert isinstance(res, tuple)
        assert len(res) == 2
        for r, exp in zip(res, expected):
            assert qml.math.allclose(r, exp, atol=tol)

        grad = jax.jacobian(circuit)(x)
        expected = (
            -np.sin(x) * np.eye(batch_size),
            qml.math.stack([-np.sin(x) / 2, np.zeros_like(x), np.zeros_like(x), np.sin(x) / 2])
            * qml.math.expand_dims(np.eye(batch_size), 1),
        )

        assert isinstance(grad, tuple)
        assert len(grad) == 2
        for g, exp in zip(grad, expected):
            assert qml.math.allclose(g, exp, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ["parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto", "jax", "jax-jit"])
    def test_jax_jit(self, diff_method, tol, interface):
        """Test derivatives when using JAX"""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit", wires=2)

        @jax.jit
        @functools.partial(qml.batch_params, all_operations=True)
        @qml.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        batch_size = 3
        x = jnp.linspace(0.1, 0.5, batch_size)

        res = circuit(x)
        expected = (
            jnp.cos(x),
            qml.math.transpose(
                qml.math.stack(
                    [jnp.cos(x / 2) ** 2, jnp.zeros_like(x), jnp.zeros_like(x), jnp.sin(x / 2) ** 2]
                )
            ),
        )

        assert isinstance(res, tuple)
        assert len(res) == 2
        for r, exp in zip(res, expected):
            assert qml.math.allclose(r, exp, atol=tol)

        grad = jax.jacobian(circuit)(x)
        expected = (
            -np.sin(x) * np.eye(batch_size),
            qml.math.stack([-np.sin(x) / 2, np.zeros_like(x), np.zeros_like(x), np.sin(x) / 2])
            * qml.math.expand_dims(np.eye(batch_size), 1),
        )

        assert isinstance(grad, tuple)
        assert len(grad) == 2
        for g, exp in zip(grad, expected):
            assert qml.math.allclose(g, exp, atol=tol, rtol=0)

    @pytest.mark.torch
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto", "torch"])
    def test_torch(self, diff_method, tol, interface):
        """Test derivatives when using torch"""
        import torch

        dev = qml.device("default.qubit", wires=2)

        @qml.batch_params
        @qml.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        batch_size = 3
        x = torch.tensor(np.linspace(0.1, 0.5, batch_size), requires_grad=True)

        res = circuit(x)
        expected = (
            torch.cos(x),
            qml.math.transpose(
                qml.math.stack(
                    [
                        torch.cos(x / 2) ** 2,
                        torch.zeros_like(x),
                        torch.zeros_like(x),
                        torch.sin(x / 2) ** 2,
                    ]
                )
            ),
        )

        assert isinstance(res, tuple)
        assert len(res) == 2
        for r, exp in zip(res, expected):
            assert qml.math.allclose(r, exp, atol=tol)

        grad = torch.autograd.functional.jacobian(circuit, x)
        expected = (
            -torch.sin(x) * torch.eye(batch_size),
            qml.math.stack(
                [-torch.sin(x) / 2, torch.zeros_like(x), torch.zeros_like(x), torch.sin(x) / 2]
            )
            * qml.math.expand_dims(torch.eye(batch_size), 1),
        )

        assert isinstance(grad, tuple)
        assert len(grad) == 2
        for g, exp in zip(grad, expected):
            assert qml.math.allclose(g, exp, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto"])
    def test_tf(self, diff_method, tol, interface):
        """Test derivatives when using TF"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        @qml.batch_params
        @qml.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        batch_size = 3
        x = tf.Variable(np.linspace(0.1, 0.5, batch_size))

        with tf.GradientTape() as tape:
            res = circuit(x)
            res = qml.math.concatenate([qml.math.expand_dims(res[0], 1), res[1]], axis=1)

        expected = qml.math.transpose(
            qml.math.stack(
                [
                    np.cos(x),
                    np.cos(x / 2) ** 2,
                    np.zeros_like(x),
                    np.zeros_like(x),
                    np.sin(x / 2) ** 2,
                ]
            )
        )
        assert qml.math.allclose(res, expected, atol=tol)

        grad = tape.jacobian(res, x)
        expected = qml.math.stack(
            [-np.sin(x), -np.sin(x) / 2, np.zeros_like(x), np.zeros_like(x), np.sin(x) / 2]
        ) * qml.math.expand_dims(np.eye(batch_size), 1)

        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto"])
    def test_tf_autograph(self, diff_method, tol, interface):
        """Test derivatives when using TF"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        @tf.function
        @qml.batch_params
        @qml.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        batch_size = 3
        x = tf.Variable(np.linspace(0.1, 0.5, batch_size))

        with tf.GradientTape() as tape:
            res = circuit(x)
            res = qml.math.concatenate([qml.math.expand_dims(res[0], 1), res[1]], axis=1)

        expected = qml.math.transpose(
            qml.math.stack(
                [
                    np.cos(x),
                    np.cos(x / 2) ** 2,
                    np.zeros_like(x),
                    np.zeros_like(x),
                    np.sin(x / 2) ** 2,
                ]
            )
        )
        assert qml.math.allclose(res, expected, atol=tol)

        grad = tape.jacobian(res, x)
        expected = qml.math.stack(
            [-np.sin(x), -np.sin(x) / 2, np.zeros_like(x), np.zeros_like(x), np.sin(x) / 2]
        ) * qml.math.expand_dims(np.eye(batch_size), 1)

        assert np.allclose(grad, expected, atol=tol, rtol=0)


def test_all_operations(mocker):
    """Test that a batch dimension can be added to all operations"""
    dev = qml.device("default.qubit", wires=3)

    @functools.partial(qml.batch_params, all_operations=True)
    @qml.qnode(dev, interface="autograd")
    def circuit(x, weights):
        qml.RX(x, wires=0)
        qml.RY([0.2, 0.3, 0.3], wires=1)
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 2])

    batch_size = 3
    x = np.linspace(0.1, 0.5, batch_size, requires_grad=True)
    weights = np.ones((batch_size, 10, 3, 3), requires_grad=False)

    spy = mocker.spy(circuit.device, "execute")
    res = circuit(x, weights)
    assert res.shape == (batch_size, 4)
    assert len(spy.call_args[0][0]) == batch_size


def test_unbatched_parameter():
    """Test that an exception is raised if a parameter
    is not batched"""

    dev = qml.device("default.qubit", wires=1)

    @qml.batch_params
    @qml.qnode(dev)
    def circuit(x, y):
        qml.RY(x, wires=[0])
        qml.RX(y, wires=[0])
        return qml.expval(qml.PauliZ(0))

    x = np.array([0.3, 0.4, 0.5])
    y = np.array(0.2)

    with pytest.raises(ValueError, match="0.2 has incorrect batch dimension"):
        circuit(x, y)


def test_initial_unbatched_parameter():
    """Test that an exception is raised if an initial parameter
    is not batched"""

    dev = qml.device("default.qubit", wires=1)

    @qml.batch_params
    @qml.qnode(dev)
    def circuit(x, y):
        qml.RY(x, wires=[0])
        qml.RX(y, wires=[0])
        return qml.expval(qml.PauliZ(0))

    x = np.array(0.2)
    y = np.array([0.3, 0.4, 0.5])

    with pytest.raises(ValueError, match="Parameter 0.2 does not contain a batch"):
        circuit(x, y)


def test_no_batch_param_error():
    """Test that the right error is thrown when there is nothing to batch"""
    dev = qml.device("default.qubit", wires=1)

    @qml.batch_params
    @qml.qnode(dev)
    def circuit(x):
        qml.RY(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    x = [0.2, 0.6, 3]
    with pytest.raises(ValueError, match="There are no operations to transform"):
        circuit(x)


def test_unbatched_not_copied():
    """Test that operators containing unbatched parameters are not copied"""

    batch_size = 5
    data = np.random.random((batch_size, 8))
    weights = np.ones((batch_size, 10, 3, 3), requires_grad=True)
    x = np.array(0.4, requires_grad=False)

    ops = [
        qml.templates.AmplitudeEmbedding(data, wires=[0, 1, 2], normalize=True),
        qml.RX(x, wires=0),
        qml.RY(0.2, wires=1),
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2]),
    ]
    meas = [qml.probs(wires=[0, 2])]

    tape = qml.tape.QuantumScript(ops, meas)
    tape.trainable_params = [0, 3]

    new_tapes = qml.batch_params(tape)[0]
    assert len(new_tapes) == batch_size

    for new_tape in new_tapes:
        # same instance of RX and RY operators
        assert new_tape.operations[1] is tape.operations[1]
        assert new_tape.operations[2] is tape.operations[2]

        # different instance of AmplitudeEmbedding and StronglyEntanglingLayers
        assert new_tape.operations[0] is not tape.operations[0]
        assert new_tape.operations[3] is not tape.operations[3]
