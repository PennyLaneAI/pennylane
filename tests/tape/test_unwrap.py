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
"""Unit tests for tape unwrapping"""

import numpy as np
import pytest

import pennylane as qml


def test_tapeunwrap_class_deprecation_warning():
    """Test that a deprecation warning is raised when initializing UnwrapTape"""
    ops = [qml.RX(0.1, 0)]
    tape = qml.tape.QuantumTape(ops)

    with pytest.warns(UserWarning, match="The UnwrapTape class is deprecated and will be removed"):
        _ = qml.tape.unwrap.UnwrapTape(tape)


def test_unwrap_class_deprecation_warning():
    """Test that a deprecation warning is raised when initializing UnwrapTape"""
    ops = [qml.RX(0.1, 0)]
    tape = qml.tape.QuantumTape(ops)

    with pytest.warns(UserWarning, match="The Unwrap class is deprecated and will be removed"):
        _ = qml.tape.unwrap.Unwrap(tape)


def test_tapeunwrap_method_deprecation_warning():
    """Test that a deprecation warning is raised when using the tape.unwrap() method"""
    ops = [qml.RX(0.1, 0)]
    tape = qml.tape.QuantumTape(ops)

    with pytest.warns(
        UserWarning, match="The method tape.unwrap is deprecated and will be removed"
    ):
        _ = tape.unwrap()


@pytest.mark.tf
def test_unwrap_tensorflow():
    """Test that unwrapping a tape with TensorFlow parameters
    works as expected"""
    import tensorflow as tf

    p = [tf.Variable(0.1), tf.constant(0.2), np.array(0.5), tf.Variable(0.3)]

    with tf.GradientTape():
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(p[0], wires=0)
            qml.RY(p[1], wires=0)
            qml.PhaseShift(p[2], wires=0)
            qml.RZ(p[3], wires=0)

        tape = qml.tape.QuantumScript.from_queue(q)
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

        with tape.unwrap():
            # inside the context manager, all parameters
            # will be unwrapped to NumPy arrays
            params = tape.get_parameters(trainable_only=False)
            assert all(isinstance(i, (float, np.float32)) for i in params)
            assert np.allclose(params, [0.1, 0.2, 0.5, 0.3])
            assert tape.trainable_params == [0, 3]

    # outside the context, the original parameters have been restored.
    assert tape.get_parameters(trainable_only=False) == p


@pytest.mark.torch
def test_unwrap_torch():
    """Test that unwrapping a tape with Torch parameters
    works as expected"""
    import torch

    p = [
        torch.tensor(0.1, requires_grad=True),
        torch.tensor(0.2),
        np.array(0.5),
        torch.tensor(0.3, requires_grad=True),
    ]

    with qml.queuing.AnnotatedQueue() as q:
        qml.RX(p[0], wires=0)
        qml.RY(p[1], wires=0)
        qml.PhaseShift(p[2], wires=0)
        qml.RZ(p[3], wires=0)

    tape = qml.tape.QuantumScript.from_queue(q)
    params = tape.get_parameters(trainable_only=False)
    tape.trainable_params = qml.math.get_trainable_indices(params)

    with tape.unwrap():
        # inside the context manager, all parameters
        # will be unwrapped to NumPy arrays
        params = tape.get_parameters(trainable_only=False)
        assert all(isinstance(i, float) for i in params)
        assert np.allclose(params, [0.1, 0.2, 0.5, 0.3])
        assert tape.trainable_params == [0, 3]

    # outside the context, the original parameters have been restored.
    assert tape.get_parameters(trainable_only=False) == p


@pytest.mark.autograd
def test_unwrap_autograd():
    """Test that unwrapping a tape with Autograd parameters
    works as expected"""
    from pennylane import numpy as anp

    p = [
        anp.tensor(0.1, requires_grad=True),
        anp.tensor(0.2, requires_grad=False),
        0.5,
        anp.tensor(0.3, requires_grad=True),
    ]

    with qml.queuing.AnnotatedQueue() as q:
        qml.RX(p[0], wires=0)
        qml.RY(p[1], wires=0)
        qml.PhaseShift(p[2], wires=0)
        qml.RZ(p[3], wires=0)

    tape = qml.tape.QuantumScript.from_queue(q)
    with tape.unwrap():
        # inside the context manager, all parameters
        # will be unwrapped to NumPy arrays
        params = tape.get_parameters(trainable_only=False)
        assert all(isinstance(i, float) for i in params)
        assert np.allclose(params, [0.1, 0.2, 0.5, 0.3])

    # outside the context, the original parameters have been restored.
    assert tape.get_parameters(trainable_only=False) == p


def test_unwrap_autograd_backward():
    """Test that unwrapping a tape with Autograd parameters
    works as expected during a backwards pass"""
    from pennylane import numpy as anp
    from autograd.numpy.numpy_boxes import ArrayBox

    p = [
        anp.tensor([0.1, 0.5, 0.3], requires_grad=True),
        anp.tensor(0.2, requires_grad=False),
    ]

    def cost(*p):
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(p[0][0], wires=0)
            qml.RY(p[1], wires=0)
            qml.PhaseShift(p[0][1], wires=0)
            qml.RZ(p[0][2], wires=0)

        tape = qml.tape.QuantumScript.from_queue(q)
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

        with tape.unwrap():
            # inside the context manager, all parameters
            # will be unwrapped to NumPy arrays
            params = tape.get_parameters(trainable_only=False)
            assert all(isinstance(i, float) for i in params)
            assert np.allclose(params, [0.1, 0.2, 0.5, 0.3])
            assert tape.trainable_params == [0, 2, 3]

        # outside the context, the original parameters have been restored.
        params = tape.get_parameters(trainable_only=False)
        assert any(isinstance(i, ArrayBox) for i in params)

        return p[0][0] * p[1] ** 2 * anp.sin(p[0][1]) * anp.exp(-0.5 * p[0][2])

    qml.grad(cost)(*p)
    qml.jacobian(qml.grad(cost))(*p)


@pytest.mark.jax
def test_unwrap_jax():
    """Test that unwrapping a tape with JAX parameters
    works as expected"""
    from jax import numpy as jnp

    p = [
        jnp.array(0.1),
        jnp.array(0.2),
        0.5,
        jnp.array(0.3),
    ]

    with qml.queuing.AnnotatedQueue() as q:
        qml.RX(p[0], wires=0)
        qml.RY(p[1], wires=0)
        qml.PhaseShift(p[2], wires=0)
        qml.RZ(p[3], wires=0)

    tape = qml.tape.QuantumScript.from_queue(q)
    params = tape.get_parameters(trainable_only=False)
    tape.trainable_params = qml.math.get_trainable_indices(params)

    with tape.unwrap():
        # inside the context manager, all parameters
        # will be unwrapped to NumPy arrays
        params = tape.get_parameters(trainable_only=False)
        assert all(isinstance(i, float) for i in params)
        assert np.allclose(params, [0.1, 0.2, 0.5, 0.3])

        assert tape.trainable_params == []

    # outside the context, the original parameters have been restored.
    assert tape.get_parameters(trainable_only=False) == p


@pytest.mark.jax
def test_unwrap_jax_backward():
    """Test that unwrapping a tape with JAX parameters
    works as expected during a backwards pass"""
    import jax
    from jax import numpy as jnp
    from jax.interpreters.ad import JVPTracer

    p = [
        jnp.array([0.1, 0.5, 0.3]),
        jnp.array(0.2),
    ]

    def cost(*p):
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(p[0][0], wires=0)
            qml.RY(p[1], wires=0)
            qml.PhaseShift(p[0][1], wires=0)
            qml.RZ(p[0][2], wires=0)

        tape = qml.tape.QuantumScript.from_queue(q)
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

        with tape.unwrap():
            # inside the context manager, all parameters
            # will be unwrapped to NumPy arrays
            params = tape.get_parameters(trainable_only=False)
            assert all(isinstance(i, float) for i in params)
            assert np.allclose(params, [0.1, 0.2, 0.5, 0.3])
            assert tape.trainable_params == [0, 2, 3]

        # outside the context, the original parameters have been restored.
        params = tape.get_parameters(trainable_only=False)
        assert all(isinstance(i, (jax.Array, JVPTracer)) for i in params)

        return p[0][0] * p[1] ** 2 * jnp.sin(p[0][1]) * jnp.exp(-0.5 * p[0][2])

    jax.grad(cost, argnums=0)(*p)
    jax.jacobian(jax.grad(cost, argnums=0), argnums=0)(*p)


@pytest.mark.torch
def test_multiple_unwrap():
    """Test that unwrapping multiple tapes at once works correctly"""
    import torch

    p = [
        torch.tensor(0.1, requires_grad=True),
        torch.tensor(0.2),
        np.array(0.5),
        torch.tensor(0.3, requires_grad=True),
    ]

    with qml.queuing.AnnotatedQueue() as q1:
        qml.RX(p[0], wires=0)
        qml.RY(p[1], wires=0)
        qml.PhaseShift(p[2], wires=0)
        qml.RZ(p[3], wires=0)

    tape1 = qml.tape.QuantumScript.from_queue(q1)
    with qml.queuing.AnnotatedQueue() as q2:
        qml.RX(p[1], wires=0)
        qml.RY(p[3], wires=0)
        qml.PhaseShift(p[0], wires=0)
        qml.RZ(p[2], wires=0)

    tape2 = qml.tape.QuantumScript.from_queue(q2)
    for t in [tape1, tape2]:
        params = t.get_parameters(trainable_only=False)
        t.trainable_params = qml.math.get_trainable_indices(params)

    with qml.tape.Unwrap(tape1, tape2):
        # inside the context manager, all parameters
        # will be unwrapped to NumPy arrays
        params = tape1.get_parameters(trainable_only=False)
        assert all(isinstance(i, float) for i in params)
        assert tape1.trainable_params == [0, 3]

        params = tape2.get_parameters(trainable_only=False)
        assert all(isinstance(i, float) for i in params)
        assert tape2.trainable_params == [1, 2]

    # outside the context, the original parameters have been restored.
    assert tape1.get_parameters(trainable_only=False) == p
    assert tape2.get_parameters(trainable_only=False) == [p[1], p[3], p[0], p[2]]
