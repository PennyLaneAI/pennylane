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
Unit tests for the ``batch_inputs`` transform.
"""
# pylint: disable=too-few-public-methods,no-value-for-parameter,comparison-with-callable

import pytest

import pennylane as qp
from pennylane import numpy as np


def test_simple_circuit():
    """Test that batching works for a simple circuit"""
    dev = qp.device("default.qubit", wires=2)

    @qp.batch_input(argnum=1)
    @qp.qnode(dev, diff_method="parameter-shift")
    def circuit(inputs, weights):
        qp.RY(weights[0], wires=0)
        qp.AngleEmbedding(inputs, wires=range(2), rotation="Y")
        qp.RY(weights[1], wires=1)
        return qp.expval(qp.PauliZ(1))

    batch_size = 5
    inputs = np.random.uniform(0, np.pi, (batch_size, 2), requires_grad=False)
    weights = np.random.uniform(-np.pi, np.pi, (2,))

    res = circuit(inputs, weights)
    assert res.shape == (batch_size,)


def test_simple_circuit_one_batch():
    """Test that batching works for a simple circuit when the batch size is 1"""
    dev = qp.device("default.qubit", wires=2)

    @qp.batch_input(argnum=1)
    @qp.qnode(dev, diff_method="parameter-shift")
    def circuit(inputs, weights):
        qp.RY(weights[0], wires=0)
        qp.AngleEmbedding(inputs, wires=range(2), rotation="Y")
        qp.RY(weights[1], wires=1)
        return qp.expval(qp.PauliZ(1))

    batch_size = 1
    inputs = np.random.uniform(0, np.pi, (batch_size, 2), requires_grad=False)
    weights = np.random.uniform(-np.pi, np.pi, (2,))

    res = circuit(inputs, weights)
    assert res.shape == (batch_size,)


def test_simple_circuit_with_prep():
    """Test that batching works for a simple circuit with a state preparation"""
    dev = qp.device("default.qubit", wires=2)

    @qp.batch_input(argnum=1)
    @qp.qnode(dev)
    def circuit(inputs, weights):
        qp.StatePrep(np.array([0, 0, 1, 0]), wires=[0, 1])
        qp.RX(inputs, wires=0)
        qp.RY(weights[0], wires=0)
        qp.RY(weights[1], wires=1)
        return qp.expval(qp.PauliZ(1))

    batch_size = 5
    inputs = np.random.uniform(0, np.pi, (batch_size,), requires_grad=False)
    weights = np.random.uniform(-np.pi, np.pi, (2,))

    res = circuit(inputs, weights)
    assert res.shape == (batch_size,)


def test_circuit_non_param_operator_before_batched_operator():
    """Test a circuit where a non-parametric operation is located before a batched operator."""
    dev = qp.device("default.qubit", wires=2)

    @qp.batch_input(argnum=0)
    @qp.qnode(dev)
    def circuit(input):
        qp.CNOT(wires=[0, 1])
        qp.RY(input, wires=1)
        qp.RX(0.1, wires=0)
        return qp.expval(qp.PauliZ(0) @ qp.PauliX(1))

    batch_size = 3

    input = np.linspace(0.1, 0.5, batch_size, requires_grad=False)

    res = circuit(input)

    assert res.shape == (batch_size,)


def test_value_error():
    """Test if the batch_input raises relevant errors correctly"""

    dev = qp.device("default.qubit", wires=2)

    class Embedding(qp.AngleEmbedding):
        """Variant of qp.AngleEmbedding that does not provide fixed
        ``ndim_params`` in order to allow for the detection of inconsistent
        batching in ``batch_input``."""

        @property
        def ndim_params(self):
            return self._ndim_params

    @qp.batch_input(argnum=[0, 2])
    @qp.qnode(dev, diff_method="parameter-shift")
    def circuit(input1, input2, weights):
        Embedding(input1, wires=range(2), rotation="Y")
        qp.RY(weights[0], wires=0)
        qp.RY(input2[0], wires=0)
        qp.RY(weights[1], wires=1)
        return qp.expval(qp.PauliZ(1))

    batch_size = 5
    input1 = np.random.uniform(0, np.pi, (batch_size, 2), requires_grad=False)
    input2 = np.random.uniform(0, np.pi, (4, 1), requires_grad=False)
    weights = np.random.uniform(-np.pi, np.pi, (2,))

    with pytest.raises(ValueError, match="Batch dimension for all gate arguments"):
        circuit(input1, input2, weights)


def test_batch_input_with_trainable_parameters_raises_error():
    """Test that using the batch_input method with trainable parameters raises a ValueError."""
    dev = qp.device("default.qubit", wires=2)

    @qp.batch_input(argnum=0)
    @qp.qnode(dev)
    def circuit(input):
        qp.RY(input, wires=1)
        qp.CNOT(wires=[0, 1])
        qp.RX(0.1, wires=0)
        return qp.expval(qp.PauliZ(0) @ qp.PauliX(1))

    batch_size = 3

    input = np.linspace(0.1, 0.5, batch_size, requires_grad=True)

    with pytest.raises(
        ValueError,
        match="Batched inputs must be non-trainable."
        + " Please make sure that the parameters indexed by "
        + "'argnum' are not marked as trainable.",
    ):
        circuit(input)


def test_mottonenstate_preparation(mocker):
    """Test that batching works for MottonenStatePreparation"""
    dev = qp.device("default.qubit", wires=3)

    @qp.batch_input(argnum=0)
    @qp.qnode(dev, interface="autograd")
    def circuit(data, weights):
        qp.templates.MottonenStatePreparation(data, wires=[0, 1, 2])
        qp.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qp.probs(wires=[0, 1, 2])

    batch_size = 3

    # create a batched input statevector
    data = np.random.random((batch_size, 2**3), requires_grad=False)
    data /= np.linalg.norm(data, axis=1).reshape(-1, 1)  # normalize

    # weights is not batched
    weights = np.random.random((10, 3, 3), requires_grad=True)

    spy = mocker.spy(circuit.device, "execute")
    res = circuit(data, weights)
    assert res.shape == (batch_size, 2**3)
    assert len(spy.call_args[0][0]) == batch_size

    # check the results against individually executed circuits (no batching)
    @qp.qnode(dev)
    def circuit2(data, weights):
        qp.templates.MottonenStatePreparation(data, wires=[0, 1, 2])
        qp.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qp.probs(wires=[0, 1, 2])

    indiv_res = []
    for state in data:
        indiv_res.append(circuit2(state, weights))
    assert np.allclose(res, indiv_res)


def test_qubit_state_prep(mocker):
    """Test that batching works for StatePrep"""

    dev = qp.device("default.qubit", wires=3)

    @qp.batch_input(argnum=0)
    @qp.qnode(dev, interface="autograd")
    def circuit(data, weights):
        qp.StatePrep(data, wires=[0, 1, 2])
        qp.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qp.probs(wires=[0, 1, 2])

    batch_size = 3

    # create a batched input statevector
    data = np.random.random((batch_size, 2**3), requires_grad=False)
    data /= np.linalg.norm(data, axis=1).reshape(-1, 1)  # normalize

    # weights is not batched
    weights = np.random.random((10, 3, 3), requires_grad=True)

    spy = mocker.spy(circuit.device, "execute")
    res = circuit(data, weights)
    assert res.shape == (batch_size, 2**3)
    assert len(spy.call_args[0][0]) == batch_size

    # check the results against individually executed circuits (no batching)
    @qp.qnode(dev)
    def circuit2(data, weights):
        qp.StatePrep(data, wires=[0, 1, 2])
        qp.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qp.probs(wires=[0, 1, 2])

    indiv_res = []
    for state in data:
        indiv_res.append(circuit2(state, weights))

    assert np.allclose(res, indiv_res)


def test_multi_returns():
    """Test that batching works for a simple circuit with multiple returns"""
    dev = qp.device("default.qubit", wires=2)

    @qp.batch_input(argnum=1)
    @qp.qnode(dev, diff_method="parameter-shift")
    def circuit(inputs, weights):
        qp.RY(weights[0], wires=0)
        qp.AngleEmbedding(inputs, wires=range(2), rotation="Y")
        qp.RY(weights[1], wires=1)
        return qp.expval(qp.PauliZ(1)), qp.probs(wires=[0, 1])

    batch_size = 6
    inputs = np.random.uniform(0, np.pi, (batch_size, 2), requires_grad=False)
    weights = np.random.uniform(-np.pi, np.pi, (2,))

    res = circuit(inputs, weights)
    assert isinstance(res, tuple)
    assert len(res) == 2

    assert res[0].shape == (batch_size,)
    assert res[1].shape == (batch_size, 4)


def test_shot_vector():
    """Test that batching works for a simple circuit with a shot vector"""
    dev = qp.device("default.qubit", wires=2)

    @qp.batch_input(argnum=1)
    @qp.set_shots((100, (200, 3), 300))
    @qp.qnode(dev, diff_method="parameter-shift")
    def circuit(inputs, weights):
        qp.RY(weights[0], wires=0)
        qp.AngleEmbedding(inputs, wires=range(2), rotation="Y")
        qp.RY(weights[1], wires=1)
        return qp.probs(wires=[0, 1])

    batch_size = 6
    inputs = np.random.uniform(0, np.pi, (batch_size, 2), requires_grad=False)
    weights = np.random.uniform(-np.pi, np.pi, (2,))

    res = circuit(inputs, weights)

    assert isinstance(res, tuple)
    assert len(res) == 5
    # pylint:disable=not-an-iterable
    assert all(shot_res.shape == (batch_size, 4) for shot_res in res)


def test_multi_returns_shot_vector():
    """Test that batching works for a simple circuit with multiple returns
    and with a shot vector"""
    dev = qp.device("default.qubit", wires=2)

    @qp.batch_input(argnum=1)
    @qp.set_shots((100, (200, 3), 300))
    @qp.qnode(dev, diff_method="parameter-shift")
    def circuit(inputs, weights):
        qp.RY(weights[0], wires=0)
        qp.AngleEmbedding(inputs, wires=range(2), rotation="Y")
        qp.RY(weights[1], wires=1)
        return qp.expval(qp.PauliZ(1)), qp.probs(wires=[0, 1])

    batch_size = 6
    inputs = np.random.uniform(0, np.pi, (batch_size, 2), requires_grad=False)
    weights = np.random.uniform(-np.pi, np.pi, (2,))

    res = circuit(inputs, weights)

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
        dev = qp.device("default.qubit", wires=2)

        @qp.batch_input(argnum=0)
        @qp.qnode(dev, diff_method=diff_method)
        def circuit(input, x):
            qp.RY(input, wires=1)
            qp.CNOT(wires=[0, 1])
            qp.RX(x, wires=0)
            return qp.expval(qp.PauliZ(0) @ qp.PauliX(1))

        batch_size = 3

        def cost(input, x):
            return np.sum(circuit(input, x))

        input = np.linspace(0.1, 0.5, batch_size, requires_grad=False)
        x = np.array(0.1, requires_grad=True)

        res = qp.grad(cost)(input, x)
        expected = -np.sin(0.1) * sum(np.sin(input))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto", "jax"])
    def test_jax(self, diff_method, tol, interface):
        """Test derivatives when using JAX"""
        import jax
        import jax.numpy as jnp

        dev = qp.device("default.qubit", wires=2)

        @qp.batch_input(argnum=0)
        @qp.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(input, x):
            qp.RY(input, wires=1)
            qp.CNOT(wires=[0, 1])
            qp.RX(x, wires=0)
            return qp.expval(qp.PauliZ(0) @ qp.PauliX(1))

        batch_size = 3

        def cost(input, x):
            return jnp.sum(circuit(input, x))

        input = jnp.linspace(0.1, 0.5, batch_size)
        x = jnp.array(0.1)

        res = jax.grad(cost, argnums=1)(input, x)
        expected = -np.sin(0.1) * sum(np.sin(input))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ["adjoint", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto", "jax", "jax-jit"])
    def test_jax_jit(self, diff_method, tol, interface):
        """Test derivatives when using JAX"""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        dev = qp.device("default.qubit", wires=2)

        @jax.jit
        @qp.batch_input(argnum=0)
        @qp.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(input, x):
            qp.RY(input, wires=1)
            qp.CNOT(wires=[0, 1])
            qp.RX(x, wires=0)
            return qp.expval(qp.PauliZ(0) @ qp.PauliX(1))

        batch_size = 3

        def cost(input, x):
            return jnp.sum(circuit(input, x))

        input = jnp.linspace(0.1, 0.5, batch_size)
        x = jnp.array(0.1)

        res = jax.grad(cost, argnums=1)(input, x)
        expected = -np.sin(0.1) * sum(np.sin(input))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.torch
    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto", "torch"])
    def test_torch(self, diff_method, tol, interface):
        """Test derivatives when using torch"""
        import torch

        dev = qp.device("default.qubit", wires=2)

        @qp.batch_input(argnum=0)
        @qp.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(input, x):
            qp.RY(input, wires=1)
            qp.CNOT(wires=[0, 1])
            qp.RX(x, wires=0)
            return qp.expval(qp.PauliZ(0) @ qp.PauliX(1))

        batch_size = 3

        def cost(input, x):
            return torch.sum(circuit(input, x))

        input = torch.linspace(0.1, 0.5, batch_size, requires_grad=False)
        x = torch.tensor(0.1, requires_grad=True)

        loss = cost(input, x)
        loss.backward()

        res = x.grad
        expected = -np.sin(0.1) * torch.sum(torch.sin(input))
        assert qp.math.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto"])
    def test_tf(self, diff_method, tol, interface):
        """Test derivatives when using TF"""
        import tensorflow as tf

        dev = qp.device("default.qubit", wires=2)

        @qp.batch_input(argnum=0)
        @qp.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(input, x):
            qp.RY(input, wires=1)
            qp.CNOT(wires=[0, 1])
            qp.RX(x, wires=0)
            return qp.expval(qp.PauliZ(0) @ qp.PauliX(1))

        batch_size = 3
        input = tf.Variable(np.linspace(0.1, 0.5, batch_size), trainable=False)
        x = tf.Variable(0.1, trainable=True)

        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(circuit(input, x))

        res = tape.gradient(loss, x)
        expected = -np.sin(0.1) * tf.reduce_sum(tf.sin(input))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto", "tf-autograph"])
    def test_tf_autograph(self, diff_method, tol, interface):
        """Test derivatives when using TF and autograph"""
        import tensorflow as tf

        dev = qp.device("default.qubit", wires=2)

        @tf.function
        @qp.batch_input(argnum=0)
        @qp.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(input, x):
            qp.RY(input, wires=1)
            qp.CNOT(wires=[0, 1])
            qp.RX(x, wires=0)
            return qp.expval(qp.PauliZ(0) @ qp.PauliX(1))

        batch_size = 3
        input = tf.Variable(np.linspace(0.1, 0.5, batch_size), trainable=False)
        x = tf.Variable(0.1, trainable=True, dtype=tf.float64)

        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(circuit(input, x))

        res = tape.gradient(loss, x)
        expected = -np.sin(0.1) * tf.reduce_sum(tf.sin(input))
        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestDiffMulti:
    """Test gradients for multiple measurements"""

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_autograd(self, diff_method, tol):
        """Test derivatives when using autograd"""
        dev = qp.device("default.qubit", wires=2)

        @qp.batch_input(argnum=0)
        @qp.qnode(dev, diff_method=diff_method)
        def circuit(input, x):
            qp.RY(input, wires=0)
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0)), qp.probs(wires=[0, 1])

        def cost(input, x):
            res = circuit(input, x)
            return qp.math.concatenate([qp.math.expand_dims(res[0], 1), res[1]], axis=1)

        batch_size = 3
        input = np.linspace(0.1, 0.5, batch_size, requires_grad=False)
        x = np.array(0.1, requires_grad=True)

        res = cost(input, x)
        expected = qp.math.transpose(
            qp.math.stack(
                [
                    np.cos(input + x),
                    np.cos((input + x) / 2) ** 2,
                    np.zeros_like(input),
                    np.zeros_like(input),
                    np.sin((input + x) / 2) ** 2,
                ]
            )
        )
        assert qp.math.allclose(res, expected, atol=tol)

        grad = qp.jacobian(lambda x: cost(input, x))(x)
        expected = qp.math.transpose(
            qp.math.stack(
                [
                    -np.sin(input + x),
                    -np.sin(input + x) / 2,
                    np.zeros_like(input),
                    np.zeros_like(input),
                    np.sin(input + x) / 2,
                ]
            )
        )
        assert qp.math.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto", "jax"])
    def test_jax(self, diff_method, tol, interface):
        """Test derivatives when using JAX"""
        import jax
        import jax.numpy as jnp

        dev = qp.device("default.qubit", wires=2)

        @qp.batch_input(argnum=0)
        @qp.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(input, x):
            qp.RY(input, wires=0)
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0)), qp.probs(wires=[0, 1])

        batch_size = 3
        input = jnp.linspace(0.1, 0.5, batch_size)
        x = jnp.array(0.1)

        res = circuit(input, x)
        expected = (
            jnp.cos(input + x),
            qp.math.transpose(
                qp.math.stack(
                    [
                        np.cos((input + x) / 2) ** 2,
                        np.zeros_like(input),
                        np.zeros_like(input),
                        np.sin((input + x) / 2) ** 2,
                    ]
                )
            ),
        )

        assert isinstance(res, tuple)
        assert len(res) == 2
        for r, exp in zip(res, expected):
            assert qp.math.allclose(r, exp, atol=tol)

        grad = jax.jacobian(circuit, argnums=1)(input, x)
        expected = (
            -jnp.sin(input + x),
            qp.math.transpose(
                qp.math.stack(
                    [
                        -jnp.sin(input + x) / 2,
                        jnp.zeros_like(input),
                        jnp.zeros_like(input),
                        jnp.sin(input + x) / 2,
                    ]
                )
            ),
        )

        assert isinstance(grad, tuple)
        assert len(grad) == 2
        for g, exp in zip(grad, expected):
            assert qp.math.allclose(g, exp, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ["parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto", "jax", "jax-jit"])
    def test_jax_jit(self, diff_method, tol, interface):
        """Test derivatives when using JAX and jitting"""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        dev = qp.device("default.qubit", wires=2)

        @jax.jit
        @qp.batch_input(argnum=0)
        @qp.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(input, x):
            qp.RY(input, wires=0)
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0)), qp.probs(wires=[0, 1])

        batch_size = 3
        input = jnp.linspace(0.1, 0.5, batch_size)
        x = jnp.array(0.1)

        res = circuit(input, x)
        expected = (
            jnp.cos(input + x),
            qp.math.transpose(
                qp.math.stack(
                    [
                        np.cos((input + x) / 2) ** 2,
                        np.zeros_like(input),
                        np.zeros_like(input),
                        np.sin((input + x) / 2) ** 2,
                    ]
                )
            ),
        )

        assert isinstance(res, tuple)
        assert len(res) == 2
        for r, exp in zip(res, expected):
            assert qp.math.allclose(r, exp, atol=tol)

        grad = jax.jacobian(circuit, argnums=1)(input, x)
        expected = (
            -jnp.sin(input + x),
            qp.math.transpose(
                qp.math.stack(
                    [
                        -jnp.sin(input + x) / 2,
                        jnp.zeros_like(input),
                        jnp.zeros_like(input),
                        jnp.sin(input + x) / 2,
                    ]
                )
            ),
        )

        assert isinstance(grad, tuple)
        assert len(grad) == 2
        for g, exp in zip(grad, expected):
            assert qp.math.allclose(g, exp, atol=tol, rtol=0)

    @pytest.mark.torch
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto", "torch"])
    def test_torch(self, diff_method, tol, interface):
        """Test derivatives when using torch"""
        import torch

        dev = qp.device("default.qubit", wires=2)

        @qp.batch_input(argnum=0)
        @qp.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(input, x):
            qp.RY(input, wires=0)
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0)), qp.probs(wires=[0, 1])

        batch_size = 3
        input = torch.tensor(np.linspace(0.1, 0.5, batch_size), requires_grad=False)
        x = torch.tensor(0.1, requires_grad=True)

        res = circuit(input, x)
        expected = (
            torch.cos(input + x),
            qp.math.transpose(
                qp.math.stack(
                    [
                        torch.cos((input + x) / 2) ** 2,
                        torch.zeros_like(input),
                        torch.zeros_like(input),
                        torch.sin((input + x) / 2) ** 2,
                    ]
                )
            ),
        )

        assert isinstance(res, tuple)
        assert len(res) == 2
        for r, exp in zip(res, expected):
            assert qp.math.allclose(r, exp, atol=tol)

        grad = torch.autograd.functional.jacobian(lambda x: circuit(input, x), x)
        expected = (
            -torch.sin(input + x),
            qp.math.transpose(
                qp.math.stack(
                    [
                        -torch.sin(input + x) / 2,
                        torch.zeros_like(input),
                        torch.zeros_like(input),
                        torch.sin(input + x) / 2,
                    ]
                )
            ),
        )

        assert isinstance(grad, tuple)
        assert len(grad) == 2
        for g, exp in zip(grad, expected):
            assert qp.math.allclose(g, exp, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto"])
    def test_tf(self, diff_method, tol, interface):
        """Test derivatives when using TF"""
        import tensorflow as tf

        dev = qp.device("default.qubit", wires=2)

        @qp.batch_input(argnum=0)
        @qp.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(input, x):
            qp.RY(input, wires=0)
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0)), qp.probs(wires=[0, 1])

        batch_size = 3
        input = tf.Variable(np.linspace(0.1, 0.5, batch_size), trainable=False)
        x = tf.Variable(0.1, trainable=True, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(input, x)
            res = qp.math.concatenate([qp.math.expand_dims(res[0], 1), res[1]], axis=1)

        expected = qp.math.transpose(
            qp.math.stack(
                [
                    np.cos(input + x),
                    np.cos((input + x) / 2) ** 2,
                    np.zeros_like(input),
                    np.zeros_like(input),
                    np.sin((input + x) / 2) ** 2,
                ]
            )
        )
        assert qp.math.allclose(res, expected, atol=tol)

        grad = tape.jacobian(res, x)
        expected = qp.math.transpose(
            qp.math.stack(
                [
                    -np.sin(input + x),
                    -np.sin(input + x) / 2,
                    np.zeros_like(input),
                    np.zeros_like(input),
                    np.sin(input + x) / 2,
                ]
            )
        )
        assert qp.math.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    @pytest.mark.parametrize("interface", ["auto", "tf-autograph"])
    def test_tf_autograph(self, diff_method, tol, interface):
        """Test derivatives when using TF and autograph"""
        import tensorflow as tf

        dev = qp.device("default.qubit", wires=2)

        @tf.function
        @qp.batch_input(argnum=0)
        @qp.qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(input, x):
            qp.RY(input, wires=0)
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0)), qp.probs(wires=[0, 1])

        batch_size = 3
        input = tf.Variable(np.linspace(0.1, 0.5, batch_size), trainable=False)
        x = tf.Variable(0.1, trainable=True, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(input, x)
            res = qp.math.concatenate([qp.math.expand_dims(res[0], 1), res[1]], axis=1)

        expected = qp.math.transpose(
            qp.math.stack(
                [
                    np.cos(input + x),
                    np.cos((input + x) / 2) ** 2,
                    np.zeros_like(input),
                    np.zeros_like(input),
                    np.sin((input + x) / 2) ** 2,
                ]
            )
        )
        assert qp.math.allclose(res, expected, atol=tol)

        grad = tape.jacobian(res, x)
        expected = qp.math.transpose(
            qp.math.stack(
                [
                    -np.sin(input + x),
                    -np.sin(input + x) / 2,
                    np.zeros_like(input),
                    np.zeros_like(input),
                    np.sin(input + x) / 2,
                ]
            )
        )
        assert qp.math.allclose(grad, expected, atol=tol, rtol=0)


def test_unbatched_not_copied():
    """Test that operators containing unbatched parameters are not copied"""
    batch_size = 5
    inputs = np.random.uniform(0, np.pi, (batch_size, 2), requires_grad=False)
    weights = np.random.uniform(-np.pi, np.pi, (2,))

    ops = [
        qp.RY(weights[0], wires=0),
        qp.AngleEmbedding(inputs, wires=range(2), rotation="Y"),
        qp.RY(weights[1], wires=1),
    ]
    meas = [qp.expval(qp.PauliZ(1))]

    tape = qp.tape.QuantumScript(ops, meas)
    tape.trainable_params = [0, 2]

    new_tapes = qp.batch_input(tape, argnum=1)[0]
    assert len(new_tapes) == batch_size

    for new_tape in new_tapes:
        # same instance of RY operators
        assert new_tape.operations[0] is tape.operations[0]
        assert new_tape.operations[2] is tape.operations[2]

        # different instance of AngleEmbedding
        assert new_tape.operations[1] is not tape.operations[1]
