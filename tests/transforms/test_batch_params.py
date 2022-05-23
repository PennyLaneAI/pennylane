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
import functools
import pytest

import pennylane as qml
from pennylane import numpy as np


def test_simple_circuit(mocker):
    """Test that batching works for a simple circuit"""
    dev = qml.device("default.qubit", wires=3)

    @qml.batch_params
    @qml.qnode(dev)
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

    spy = mocker.spy(circuit.device, "batch_execute")
    res = circuit(data, x, weights)
    assert res.shape == (batch_size, 4)
    assert len(spy.call_args[0][0]) == batch_size


def test_basic_entangler_layers(mocker):
    """Test that batching works for BasicEngtanglerLayers"""
    dev = qml.device("default.qubit", wires=2)

    @qml.batch_params
    @qml.qnode(dev)
    def circuit(weights):
        qml.templates.BasicEntanglerLayers(weights, wires=[0, 1])
        qml.RY(0.2, wires=1)
        return qml.probs(wires=[0, 1])

    batch_size = 5
    weights = np.random.random((batch_size, 2, 2))

    spy = mocker.spy(circuit.device, "batch_execute")
    res = circuit(weights)
    assert res.shape == (batch_size, 4)
    assert len(spy.call_args[0][0]) == batch_size


def test_angle_embedding(mocker):
    """Test that batching works for AngleEmbedding"""
    dev = qml.device("default.qubit", wires=3)

    @qml.batch_params
    @qml.qnode(dev)
    def circuit(data):
        qml.templates.AngleEmbedding(data, wires=[0, 1, 2])
        qml.RY(0.2, wires=1)
        return qml.probs(wires=[0, 2])

    batch_size = 5
    data = np.random.random((batch_size, 3))

    spy = mocker.spy(circuit.device, "batch_execute")
    res = circuit(data)
    assert res.shape == (batch_size, 4)
    assert len(spy.call_args[0][0]) == batch_size


def test_mottonenstate_preparation(mocker):
    """Test that batching works for MottonenStatePreparation"""
    dev = qml.device("default.qubit", wires=3)

    @qml.batch_params
    @qml.qnode(dev)
    def circuit(data, weights):
        qml.templates.MottonenStatePreparation(data, wires=[0, 1, 2])
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 1, 2])

    batch_size = 3

    # create a batched input statevector
    data = np.random.random((batch_size, 2**3))
    data /= np.linalg.norm(data, axis=1).reshape(-1, 1)  # normalize
    weights = np.random.random((batch_size, 10, 3, 3))

    spy = mocker.spy(circuit.device, "batch_execute")
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


def test_basis_state_preparation(mocker):
    """Test that batching works for BasisStatePreparation"""
    dev = qml.device("default.qubit", wires=4)

    @qml.batch_params
    @qml.qnode(dev)
    def circuit(data, weights):
        qml.templates.BasisStatePreparation(data, wires=[0, 1, 2, 3])
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2, 3])
        return qml.probs(wires=[0, 1, 2, 3])

    batch_size = 5

    # create random batched basis states
    data = np.random.randint(2, size=(batch_size, 4))
    weights = np.random.random((batch_size, 10, 4, 3))

    spy = mocker.spy(circuit.device, "batch_execute")
    res = circuit(data, weights)
    assert res.shape == (batch_size, 2**4)
    assert len(spy.call_args[0][0]) == batch_size

    # check the results against individually executed circuits (no batching)
    @qml.qnode(dev)
    def circuit2(data, weights):
        qml.templates.BasisStatePreparation(data, wires=[0, 1, 2, 3])
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2, 3])
        return qml.probs(wires=[0, 1, 2, 3])

    indiv_res = []
    for state, weight in zip(data, weights):
        indiv_res.append(circuit2(state, weight))
    assert np.allclose(res, indiv_res)


@pytest.mark.autograd
@pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift"])
def test_autograd(diff_method, tol):
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
def test_jax(diff_method, tol):
    """Test derivatives when using JAX."""
    import jax

    jnp = jax.numpy
    dev = qml.device("default.qubit", wires=2)

    @qml.batch_params
    @qml.qnode(dev, interface="jax", diff_method=diff_method)
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
@pytest.mark.parametrize("interface", ["jax", "jax-jit"])
def test_jax_jit(diff_method, interface, tol):
    """Test derivatives when using JAX and JIT."""
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
def test_torch(diff_method, tol):
    """Test derivatives when using Torch"""
    import torch

    dev = qml.device("default.qubit", wires=2)

    @qml.batch_params
    @qml.qnode(dev, interface="torch", diff_method=diff_method)
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
def test_tf(diff_method, tol):
    """Test derivatives when using TF"""
    import tensorflow as tf

    dev = qml.device("default.qubit", wires=2)

    @qml.batch_params
    @qml.qnode(dev, interface="tf", diff_method=diff_method)
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
def test_tf_autograph(tol):
    """Test derivatives when using TF and autograph"""
    import tensorflow as tf

    dev = qml.device("default.qubit", wires=2)

    @qml.batch_params
    @qml.qnode(dev, interface="tf", diff_method="backprop")
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


def test_all_operations(mocker):
    """Test that a batch dimension can be added to all operations"""
    dev = qml.device("default.qubit", wires=3)

    @functools.partial(qml.batch_params, all_operations=True)
    @qml.qnode(dev)
    def circuit(x, weights):
        qml.RX(x, wires=0)
        qml.RY([0.2, 0.3, 0.3], wires=1)
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 2])

    batch_size = 3
    x = np.linspace(0.1, 0.5, batch_size, requires_grad=True)
    weights = np.ones((batch_size, 10, 3, 3), requires_grad=False)

    spy = mocker.spy(circuit.device, "batch_execute")
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
