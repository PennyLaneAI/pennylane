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
"""
Unit tests for the batch partial transform.
"""
# pylint: disable=too-few-public-methods
import re

import pytest

import pennylane as qml
from pennylane import numpy as np


def test_partial_evaluation():
    """Test partial evaluation matches individual full evaluations"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the partial argument to construct a new circuit with
    y = np.random.uniform(size=2)

    # the batched argument to the new partial circuit
    x = np.random.uniform(size=batch_size)

    batched_partial_circuit = qml.batch_partial(circuit, y=y)
    res = batched_partial_circuit(x)

    # check the results against individually executed circuits
    indiv_res = []
    for x_indiv in x:
        indiv_res.append(circuit(x_indiv, y))

    assert np.allclose(res, indiv_res)


def test_partial_evaluation_kwargs():
    """Test partial evaluation matches individual full evaluations
    when the keyword syntax is used to call the partial object"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the partial argument to construct a new circuit with
    y = np.random.uniform(size=2)

    # the batched argument to the new partial circuit
    x = np.random.uniform(size=batch_size)

    batched_partial_circuit = qml.batch_partial(circuit, y=y)
    res = batched_partial_circuit(x=x)

    # check the results against individually executed circuits
    indiv_res = []
    for x_indiv in x:
        indiv_res.append(circuit(x_indiv, y))

    assert np.allclose(res, indiv_res)


def test_partial_evaluation_multi_args():
    """Test partial evaluation matches individual full evaluations
    for multiple pre-supplied arguments"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x, y, z):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        qml.RX(z, wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the partial arguments to construct a new circuit with
    y = np.random.uniform(size=2)
    z = np.random.uniform(size=())

    # the batched argument to the new partial circuit
    x = np.random.uniform(size=batch_size)

    batched_partial_circuit = qml.batch_partial(circuit, y=y, z=z)
    res = batched_partial_circuit(x)

    # check the results against individually executed circuits
    indiv_res = []
    for x_indiv in x:
        indiv_res.append(circuit(x_indiv, y, z))

    assert np.allclose(res, indiv_res)


def test_partial_evaluation_nonnumeric1():
    """Test partial evaluation matches individual full evaluations
    for non-numeric pre-supplied arguments"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x, y, measurement):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.apply(measurement)

    batch_size = 4

    # the partial arguments to construct a new circuit with
    y = np.random.uniform(size=2)
    measurement = qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    # the batched argument to the new partial circuit
    x = np.random.uniform(size=batch_size)

    batched_partial_circuit = qml.batch_partial(circuit, y=y, measurement=measurement)
    res = batched_partial_circuit(x)

    # check the results against individually executed circuits
    indiv_res = []
    for x_indiv in x:
        indiv_res.append(circuit(x_indiv, y, measurement))

    assert np.allclose(res, indiv_res)


def test_partial_evaluation_nonnumeric2():
    """Test partial evaluation matches individual full evaluations
    for non-numeric pre-supplied arguments"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x, y, func):
        qml.RX(func(x), wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the partial arguments to construct a new circuit with
    y = np.random.uniform(size=2)
    func = np.cos

    # the batched argument to the new partial circuit
    x = np.random.uniform(size=batch_size)

    batched_partial_circuit = qml.batch_partial(circuit, y=y, func=func)
    res = batched_partial_circuit(x)

    # check the results against individually executed circuits
    indiv_res = []
    for x_indiv in x:
        indiv_res.append(circuit(x_indiv, y, func))

    assert np.allclose(res, indiv_res)


def test_partial_evaluation_nonnumeric3():
    """Test partial evaluation matches individual full evaluations
    for non-numeric pre-supplied arguments"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x, y, op):
        qml.apply(op(x, wires=0))
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the partial arguments to construct a new circuit with
    y = np.random.uniform(size=2)
    op = qml.RX

    # the batched argument to the new partial circuit
    x = np.random.uniform(size=batch_size)

    batched_partial_circuit = qml.batch_partial(circuit, y=y, op=op)
    res = batched_partial_circuit(x)

    # check the results against individually executed circuits
    indiv_res = []
    for x_indiv in x:
        indiv_res.append(circuit(x_indiv, y, op))

    assert np.allclose(res, indiv_res)


@pytest.mark.autograd
@pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift", "finite-diff"])
def test_partial_evaluation_autograd(diff_method):
    """Test gradient of partial evaluation matches gradients of
    individual full evaluations using autograd"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the partial argument to construct a new circuit with
    y = np.random.uniform(size=2)

    # the batched argument to the new partial circuit
    x = np.random.uniform(size=batch_size, requires_grad=True)

    batched_partial_circuit = qml.batch_partial(circuit, y=y)

    # we could also sum over the batch dimension and use the regular
    # gradient instead of the jacobian, but either works
    grad = np.diagonal(qml.jacobian(batched_partial_circuit)(x))

    # check the results against individually executed circuits
    indiv_grad = []
    for x_indiv in x:
        indiv_grad.append(qml.grad(circuit, argnum=0)(x_indiv, y))

    assert np.allclose(grad, indiv_grad)


@pytest.mark.jax
@pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift", "finite-diff"])
def test_partial_evaluation_jax(diff_method):
    """Test gradient of partial evaluation matches gradients of
    individual full evaluations using jax"""
    import jax
    import jax.numpy as jnp

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the partial argument to construct a new circuit with
    y = jnp.asarray(np.random.uniform(size=2))

    # the batched argument to the new partial circuit
    x = jnp.asarray(np.random.uniform(size=batch_size))

    batched_partial_circuit = qml.batch_partial(circuit, all_operations=True, y=y)

    # we could also sum over the batch dimension and use the regular
    # gradient instead of the jacobian, but either works
    grad = jnp.diagonal(jax.jacrev(batched_partial_circuit)(x))

    # check the results against individually executed circuits
    indiv_grad = []
    for x_indiv in x:
        indiv_grad.append(jax.grad(circuit, argnums=0)(x_indiv, y))

    assert np.allclose(grad, indiv_grad)


@pytest.mark.tf
@pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift", "finite-diff"])
def test_partial_evaluation_tf(diff_method):
    """Test gradient of partial evaluation matches gradients of
    individual full evaluations using TF"""
    import tensorflow as tf

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the partial argument to construct a new circuit with
    y = tf.Variable(np.random.uniform(size=2), trainable=True)

    # the batched argument to the new partial circuit
    x = tf.Variable(np.random.uniform(size=batch_size), trainable=True)

    batched_partial_circuit = qml.batch_partial(circuit, y=y)

    with tf.GradientTape() as tape:
        out = batched_partial_circuit(x)

    # we could also sum over the batch dimension and use the regular
    # gradient instead of the jacobian, but either works
    grad = tf.linalg.tensor_diag_part(tape.jacobian(out, x))

    # check the results against individually executed circuits
    indiv_grad = []
    for x_indiv in tf.unstack(x):
        # create a new variable since tensors created by
        # indexing a trainable variable aren't themselves trainable
        x_indiv = tf.Variable(x_indiv, trainable=True)

        with tf.GradientTape() as tape:
            out_indiv = circuit(x_indiv, y)
        indiv_grad.append(tape.gradient(out_indiv, x_indiv))

    assert np.allclose(grad, indiv_grad)


@pytest.mark.torch
@pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift", "finite-diff"])
def test_partial_evaluation_torch(diff_method):
    """Test gradient of partial evaluation matches gradients of
    individual full evaluations using PyTorch"""
    import torch
    import torch.autograd.functional as F

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the partial argument to construct a new circuit with
    y = torch.tensor(np.random.uniform(size=2), requires_grad=True)

    # the batched argument to the new partial circuit
    x = torch.tensor(np.random.uniform(size=batch_size), requires_grad=True)

    batched_partial_circuit = qml.batch_partial(circuit, y=y)

    # we could also sum over the batch dimension and use the regular
    # gradient instead of the jacobian, but either works
    grad = torch.diagonal(F.jacobian(batched_partial_circuit, x))

    # check the results against individually executed circuits
    indiv_grad = []
    for x_indiv in x:
        # create a new variable since tensors created by
        # indexing a trainable variable aren't themselves trainable
        x_indiv = x_indiv.clone().detach().requires_grad_(True)

        out_indiv = circuit(x_indiv, y)
        out_indiv.backward()

        indiv_grad.append(x_indiv.grad)

    assert np.allclose(grad, indiv_grad)


def test_lambda_evaluation():
    """Test lambda argument replacement matches individual full evaluations"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the first partial argument
    x = np.random.uniform(size=())

    # the base value of the second partial argument
    y = np.random.uniform(size=2)

    # the second partial argument as a function of the inputs
    fn = lambda y0: y + y0 * np.ones(2)

    # values for the second argument
    y0 = np.random.uniform(size=batch_size)

    batched_partial_circuit = qml.batch_partial(circuit, x=x, preprocess={"y": fn})
    res = batched_partial_circuit(y0)

    # check the results against individually executed circuits
    indiv_res = []
    for y0_indiv in y0:
        indiv_res.append(circuit(x, y + y0_indiv * np.ones(2)))

    assert np.allclose(res, indiv_res)


@pytest.mark.autograd
@pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift", "finite-diff"])
def test_lambda_evaluation_autograd(diff_method):
    """Test gradient of lambda argument replacement matches
    gradients of individual full evaluations using autograd"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the first partial argument
    x = np.random.uniform(size=())

    # the base value of the second partial argument
    y = np.random.uniform(size=2)

    # the second partial argument as a function of the inputs
    fn = lambda y0: y + y0 * np.ones(2)

    # values for the second argument
    y0 = np.random.uniform(size=batch_size, requires_grad=True)

    batched_partial_circuit = qml.batch_partial(circuit, x=x, preprocess={"y": fn})

    # we could also sum over the batch dimension and use the regular
    # gradient instead of the jacobian, but either works
    grad = qml.math.diagonal(qml.jacobian(batched_partial_circuit)(y0))

    # check the results against individually executed circuits
    indiv_grad = []
    for y0_indiv in y0:
        grad_wrt_second_arg = qml.grad(circuit, argnum=1)(x, y + y0_indiv * np.ones(2))
        grad_wrt_y0 = qml.math.sum(grad_wrt_second_arg)
        indiv_grad.append(grad_wrt_y0)

    assert np.allclose(grad, indiv_grad)


@pytest.mark.jax
@pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift", "finite-diff"])
def test_lambda_evaluation_jax(diff_method):
    """Test gradient of lambda argument replacement matches
    gradients of individual full evaluations using JAX"""
    import jax
    import jax.numpy as jnp

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the first partial argument
    x = jnp.asarray(np.random.uniform(size=()))

    # the base value of the second partial argument
    y = jnp.asarray(np.random.uniform(size=2))

    # the second partial argument as a function of the inputs
    fn = lambda y0: y + y0 * jnp.ones(2)

    # values for the second argument
    y0 = jnp.asarray(np.random.uniform(size=batch_size))

    batched_partial_circuit = qml.batch_partial(
        circuit, all_operations=True, x=x, preprocess={"y": fn}
    )

    # we could also sum over the batch dimension and use the regular
    # gradient instead of the jacobian, but either works
    grad = jnp.diagonal(jax.jacrev(batched_partial_circuit)(y0))

    # check the results against individually executed circuits
    indiv_grad = []
    for y0_indiv in y0:
        grad_wrt_second_arg = jax.grad(circuit, argnums=1)(x, y + y0_indiv * np.ones(2))
        grad_wrt_y0 = jnp.sum(grad_wrt_second_arg)
        indiv_grad.append(grad_wrt_y0)

    assert np.allclose(grad, indiv_grad)


@pytest.mark.tf
@pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift", "finite-diff"])
def test_lambda_evaluation_tf(diff_method):
    """Test gradient of lambda argument replacement matches
    gradients of individual full evaluations using TF"""
    import tensorflow as tf

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the first partial argument
    x = tf.Variable(np.random.uniform(size=()))

    # the base value of the second partial argument
    y = tf.Variable(np.random.uniform(size=2))

    # the second partial argument as a function of the inputs
    fn = lambda y0: y + y0 * tf.ones(2, dtype=tf.float64)

    # values for the second argument
    y0 = tf.Variable(np.random.uniform(size=batch_size), trainable=True)

    batched_partial_circuit = qml.batch_partial(circuit, x=x, preprocess={"y": fn})

    with tf.GradientTape() as tape:
        out = batched_partial_circuit(y0)

    # we could also sum over the batch dimension and use the regular
    # gradient instead of the jacobian, but either works
    grad = tf.linalg.tensor_diag_part(tape.jacobian(out, y0))

    # check the results against individually executed circuits
    indiv_grad = []
    for y0_indiv in tf.unstack(y0):
        # create a new variable since tensors created by
        # indexing a trainable variable aren't themselves trainable
        y0_indiv = tf.Variable(y0_indiv, trainable=True)

        with tf.GradientTape() as tape:
            out_indiv = circuit(x, y + y0_indiv * tf.ones(2, dtype=tf.float64))

        indiv_grad.append(tape.gradient(out_indiv, y0_indiv))

    assert np.allclose(grad, indiv_grad)


@pytest.mark.torch
@pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift", "finite-diff"])
def test_lambda_evaluation_torch(diff_method):
    """Test gradient of lambda argument replacement matches
    gradients of individual full evaluations using PyTorch"""
    import torch
    import torch.autograd.functional as F

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the first partial argument
    x = torch.tensor(np.random.uniform(size=()), requires_grad=True)

    # the base value of the second partial argument
    y = torch.tensor(np.random.uniform(size=2), requires_grad=True)

    # the second partial argument as a function of the inputs
    fn = lambda y0: y + y0 * torch.ones(2)

    # values for the second argument
    y0 = torch.tensor(np.random.uniform(size=batch_size), requires_grad=True)

    batched_partial_circuit = qml.batch_partial(circuit, x=x, preprocess={"y": fn})

    # we could also sum over the batch dimension and use the regular
    # gradient instead of the jacobian, but either works
    grad = torch.diagonal(F.jacobian(batched_partial_circuit, y0))

    # check the results against individually executed circuits
    indiv_grad = []
    for y0_indiv in y0:
        # create a new variable since tensors created by
        # indexing a trainable variable aren't themselves trainable
        y0_indiv = y0_indiv.clone().detach().requires_grad_(True)

        out_indiv = circuit(x, y + y0_indiv * torch.ones(2))
        out_indiv.backward()

        indiv_grad.append(y0_indiv.grad)

    assert np.allclose(grad, indiv_grad)


def test_full_evaluation_error():
    """Test that an error is raised when all arguments to QNode
    are provided to a partial evaluation."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    # the partial arguments
    x = np.random.uniform(size=())
    y = np.random.uniform(size=2)

    with pytest.raises(
        ValueError, match="Partial evaluation must leave at least one unevaluated parameter"
    ):
        qml.batch_partial(circuit, x=x, y=y)


def test_incomplete_evaluation_error():
    """Test that an error is raised when not all arguments to QNode
    are provided to a callable wrapper"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    # the second partial argument as a function of the inputs
    y = np.random.uniform(size=2)
    fn = lambda y0: y + y0 * np.ones(2)

    with pytest.raises(
        ValueError, match="Callable argument requires all other arguments to QNode be provided"
    ):
        qml.batch_partial(circuit, preprocess={"y": fn})


def test_kwargs_callable_error():
    """Test that an error is raised when keyword arguments
    are provided to a callable wrapper"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size = 4

    # the partial arguments
    x = np.random.uniform(size=())

    y = np.random.uniform(size=2)
    fn = lambda y0: y + y0 * np.ones(2)
    y0 = np.random.uniform(size=batch_size)

    batched_partial_circuit = qml.batch_partial(circuit, x=x, preprocess={"y": fn})

    with pytest.raises(
        ValueError,
        match="Arguments must not be passed as keyword arguments to callable within partial function",
    ):
        batched_partial_circuit(y=y0)

    with pytest.raises(
        ValueError,
        match="Arguments must not be passed as keyword arguments to callable within partial function",
    ):
        batched_partial_circuit(y0=y0)


def test_no_batchdim_error():
    """Test that an error is raised when no batch
    dimension is given to the decorated QNode"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y[..., 0], wires=0)
        qml.RY(y[..., 1], wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    # the second partial argument
    y = np.random.uniform(size=2)

    # the incorrectly batched argument to the new partial circuit
    x = np.random.uniform(size=())

    batched_partial_circuit = qml.batch_partial(circuit, y=y)

    with pytest.raises(ValueError, match="Parameter with batch dimension must be provided"):
        batched_partial_circuit(x=x)


def test_different_batchdim_error():
    """Test that an error is raised when different batch
    dimensions are given to the decorated QNode"""
    dev = qml.device("default.qubit", wires=2)

    # To test this error message, we need to use operations that do
    # not report problematic broadcasting dimensions (in place of problematic
    # batch dimensions) at tape creation. For this, we "delete" `ndim_params`.

    class RX_no_ndim(qml.RX):
        ndim_params = property(lambda self: self._ndim_params)

    class RY_no_ndim(qml.RY):
        ndim_params = property(lambda self: self._ndim_params)

    @qml.qnode(dev)
    def circuit(x, y, z):
        RX_no_ndim(x, wires=0)
        RY_no_ndim(y[..., 0], wires=0)
        RY_no_ndim(y[..., 1], wires=1)
        RX_no_ndim(z, wires=1)
        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    batch_size1, batch_size2 = 5, 4

    # the second partial argument
    y = np.random.uniform(size=2)

    # the batched arguments to the new partial circuit
    x = np.random.uniform(size=batch_size1)
    z = np.random.uniform(size=batch_size2)

    batched_partial_circuit = qml.batch_partial(circuit, y=y)

    msg = "has incorrect batch dimension. Expecting first dimension of length 5."
    msg = re.escape(msg)
    with pytest.raises(ValueError, match=msg):
        batched_partial_circuit(x=x, z=z)
