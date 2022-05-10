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
Unit tests for the :func:`pennylane.math.is_abstract` function.
"""
import functools

import numpy as np
import pytest

import pennylane as qml


@pytest.mark.tf
class TestTensorFlow:
    """Test that tensorflow integrates with is_abstract"""

    def test_eager(self):
        """Test that no tensors are abstract when in eager mode"""
        import tensorflow as tf

        def cost(x, w):
            y = x**2
            z = tf.ones([2, 2])

            assert tf.executing_eagerly()

            assert not qml.math.is_abstract(w)
            assert not qml.math.is_abstract(x)
            assert not qml.math.is_abstract(y)
            assert not qml.math.is_abstract(z)

            return tf.reduce_sum(y)

        x = tf.Variable([0.5, 0.1])
        w = tf.constant(0.1)

        assert not qml.math.is_abstract(w)
        assert not qml.math.is_abstract(x)

        with tf.GradientTape() as tape:
            res = cost(x, w)

        assert res == 0.26

        grad = tape.gradient(res, x)
        assert np.allclose(grad, 2 * x)

    @pytest.mark.parametrize("jit_compile", [True, False])
    def test_jit(self, jit_compile):
        """Test that all tensors are abstract when in autograd mode"""
        import tensorflow as tf

        @tf.function(jit_compile=jit_compile)
        def cost(x, w):
            y = x**2
            z = tf.ones([2, 2])

            assert not tf.executing_eagerly()

            assert qml.math.is_abstract(w)
            assert qml.math.is_abstract(x)
            assert qml.math.is_abstract(y)
            assert qml.math.is_abstract(z)

            return tf.reduce_sum(y)

        x = tf.Variable([0.5, 0.1])
        w = tf.constant(0.1)

        assert not qml.math.is_abstract(w)
        assert not qml.math.is_abstract(x)

        with tf.GradientTape() as tape:
            res = cost(x, w)

        assert res == 0.26

        grad = tape.gradient(res, x)
        assert np.allclose(grad, 2 * x)


@pytest.mark.jax
class TestJAX:
    """Test that JAX integrates with is_abstract"""

    def test_eager(self):
        """Test that no tensors are abstract when in eager mode"""
        import jax
        import jax.numpy as jnp

        def cost(x, w):
            y = x**2
            z = jnp.ones([2, 2])

            assert not qml.math.is_abstract(w)
            assert not qml.math.is_abstract(x)
            assert not qml.math.is_abstract(y)
            assert not qml.math.is_abstract(z)

            return jnp.sum(y)

        x = jnp.array([0.5, 0.1])
        w = jnp.array(0.1)

        assert not qml.math.is_abstract(w)
        assert not qml.math.is_abstract(x)

        res = cost(x, w)
        assert res == 0.26

        grad = jax.grad(cost, argnums=0)(x, w)
        assert np.allclose(grad, 2 * x)

    def test_jit(self):
        """Test that all tensors are abstract when in JIT mode.
        Note that `jax.grad` has slightly different behaviour, and will
        avoid making abstract tensors for non-differentiable arguments."""
        import jax
        import jax.numpy as jnp

        @functools.partial(jax.jit, static_argnums=[2])
        def cost(x, w, w_is_abstract=True):
            y = x**2
            z = jnp.ones([2, 2])

            assert qml.math.is_abstract(w) == w_is_abstract

            assert qml.math.is_abstract(x)
            assert qml.math.is_abstract(y)
            assert qml.math.is_abstract(z)

            return jnp.sum(y)

        x = jnp.array([0.5, 0.1])
        w = jnp.array(0.1)

        assert not qml.math.is_abstract(w)
        assert not qml.math.is_abstract(x)

        res = cost(x, w)
        assert res == 0.26

        # NOTE: As of JAX 0.3.1, JAX will trace non-differentiated arguments.
        # As a result, it is no longer possible to assume that non-differentiated
        # arguments will not be abstract.

        # # since we only specify argnums=0, w will not be abstract
        # grad = jax.grad(cost, argnums=0)(x, w, w_is_abstract=False)
        # assert np.allclose(grad, 2 * x)

        # Otherwise, w will be abstract
        grad = jax.grad(cost, argnums=[0, 1])(x, w, w_is_abstract=True)
        assert np.allclose(grad[0], 2 * x)
