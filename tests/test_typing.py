# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the ``typing.py`` file."""
import numpy as np
import pytest

import pennylane.numpy as pnp
from pennylane.typing import TensorLike


class TestTensorLike:
    def test_isinstance_unknown_type(self):
        """Test that an unknown type returns False."""

        # pylint: disable=too-few-public-methods
        class UnknownType:
            """Unknown type."""

        assert not isinstance(UnknownType(), TensorLike)

    def test_isinstance_numpy_array(self):
        """Tests that a numpy array is a Tensor"""
        assert isinstance(np.array(1), TensorLike)

    def test_isinstance_pennylane_tensor(self):
        """Tests that a PennyLane numpy tensor is a Tensor"""
        assert isinstance(pnp.array(1), TensorLike)

    @pytest.mark.jax
    def test_isinstance_jax_array_is_tensor_like(self):
        """Tests that a jax DeviceArray is a Tensor"""
        import jax

        tensor = jax.numpy.array(1)
        assert isinstance(tensor, TensorLike)

    @pytest.mark.torch
    def test_isinstance_torch_tensor_is_tensor_like(self):
        """Tests that a torch Tensor is a Tensor"""
        import torch

        tensor = torch.Tensor(1)
        assert isinstance(tensor, TensorLike)

    @pytest.mark.tf
    def test_isinstance_tf_tensor_is_tensor_like(self):
        """Tests that a tensorflow Tensor is a Tensor"""
        import tensorflow as tf

        tensor = tf.constant([1, 2, 3])
        assert isinstance(tensor, tf.Tensor)
        assert isinstance(tensor, TensorLike)
        var = tf.Variable(9)
        assert isinstance(var, TensorLike)

    def test_subclass_unknown_type(self):
        """Test that an unknown type returns False."""

        # pylint: disable=too-few-public-methods
        class UnknownType:
            """Unknown type."""

        assert not issubclass(UnknownType, TensorLike)

    def test_subclass_numpy_array(self):
        """Tests that a numpy array is a Tensor"""
        assert issubclass(np.ndarray, TensorLike)

    def test_subclass_pennylane_tensor(self):
        """Tests that a PennyLane numpy tensor is a Tensor"""
        assert issubclass(pnp.ndarray, TensorLike)

    @pytest.mark.jax
    def test_subclass_jax_array_is_tensor_like(self):
        """Tests that a jax DeviceArray is a Tensor"""
        import jax

        assert issubclass(jax.numpy.ndarray, TensorLike)

    @pytest.mark.torch
    def test_subclass_torch_tensor_is_tensor_like(self):
        """Tests that a torch Tensor is a Tensor"""
        import torch

        assert issubclass(torch.Tensor, TensorLike)

    @pytest.mark.tf
    def test_subclass_tf_tensor_is_tensor_like(self):
        """Tests that a tensorflow Tensor is a Tensor"""
        import tensorflow as tf

        assert issubclass(tf.Tensor, TensorLike)
        assert issubclass(tf.Variable, TensorLike)
