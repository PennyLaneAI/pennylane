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
"""Tests for the gradients.jvp module."""
from functools import partial

import pytest

import pennylane as qml
from pennylane import numpy as np


class TestComputeVJP:
    """Tests for the numeric computation of VJPs"""

    def test_computation(self):
        """Test that the correct JVP is returned"""
        dy = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = np.array([[[1.0, 0.1, 0.2], [0.2, 0.6, 0.1]], [[0.4, -0.7, 1.2], [-0.5, -0.6, 0.7]]])

        jvp = qml.gradients.compute_jvp(dy, jac)
        print(jvp)
        # assert jvp.shape == (3,)
        # assert np.all(jvp == np.tensordot(jac, dy, axes=[[0, 1], [0, 1]]))

    def test_jacobian_is_none(self):
        """A None Jacobian returns a None JVP"""

        dy = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = None

        vjp = qml.gradients.compute_vjp(dy, jac)
        assert vjp is None

    def test_zero_dy(self):
        """A zero dy vector will return a zero matrix"""
        dy = np.zeros([2, 2])
        jac = np.array([[[1.0, 0.1, 0.2], [0.2, 0.6, 0.1]], [[0.4, -0.7, 1.2], [-0.5, -0.6, 0.7]]])

        vjp = qml.gradients.compute_jvp(dy, jac)
        assert np.all(vjp == np.zeros([3]))

    @pytest.mark.torch
    @pytest.mark.parametrize("dtype1,dtype2", [("float32", "float64"), ("float64", "float32")])
    def test_dtype_torch(self, dtype1, dtype2):
        """Test that using the Torch interface the dtype of the result is
        determined by the dtype of the dy."""
        import torch

        dtype1 = getattr(torch, dtype1)
        dtype2 = getattr(torch, dtype2)

        dy = torch.ones(4, dtype=dtype1)
        jac = torch.ones((4, 4), dtype=dtype2)

        assert qml.gradients.compute_vjp(dy, jac).dtype == dtype1

    @pytest.mark.tf
    @pytest.mark.parametrize("dtype1,dtype2", [("float32", "float64"), ("float64", "float32")])
    def test_dtype_tf(self, dtype1, dtype2):
        """Test that using the TensorFlow interface the dtype of the result is
        determined by the dtype of the dy."""
        import tensorflow as tf

        dtype1 = getattr(tf, dtype1)
        dtype2 = getattr(tf, dtype2)

        dy = tf.ones(4, dtype=dtype1)
        jac = tf.ones((4, 4), dtype=dtype2)

        assert qml.gradients.compute_vjp(dy, jac).dtype == dtype1

    @pytest.mark.jax
    @pytest.mark.parametrize("dtype1,dtype2", [("float32", "float64"), ("float64", "float32")])
    def test_dtype_jax(self, dtype1, dtype2):
        """Test that using the JAX interface the dtype of the result is
        determined by the dtype of the dy."""
        import jax
        from jax.config import config

        config.update("jax_enable_x64", True)

        dtype1 = getattr(jax.numpy, dtype1)
        dtype2 = getattr(jax.numpy, dtype2)

        dy = jax.numpy.array([0, 1], dtype=dtype1)
        jac = jax.numpy.array([[0, 1], [2, 3]], dtype=dtype2)

        assert qml.gradients.compute_vjp(dy, jac).dtype == dtype1
