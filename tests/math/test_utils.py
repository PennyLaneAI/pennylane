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
"""Unit tests for functions in qml.math.utils
"""

import pytest
from pennylane.math.utils import tensor_like
from pennylane import numpy as np


class TestTypeAlias:
    def test_pennylane_tensor_is_tensor_like(self):
        """Tests that a PennyLane numpy tensor is tensor_like"""
        assert isinstance(np.tensor(1), tensor_like)

    @pytest.mark.jax
    def test_jax_array_is_tensor_like(self):
        """Tests that a jax DeviceArray is tensor_like"""
        import jax

        tensor = jax.numpy.array(1)
        assert isinstance(tensor, jax.Array)
        assert isinstance(tensor, tensor_like)

    @pytest.mark.torch
    def test_torch_tensor_is_tensor_like(self):
        """Tests that a torch Tensor is tensor_like"""
        import torch

        tensor = torch.Tensor(1)
        assert isinstance(tensor, tensor_like)

    @pytest.mark.tensorflow
    def test_jax_array_is_tensor_like(self):
        """Tests that a tensorflow Tensor is tensor_like"""
        import tensorflow as tf

        tensor = tf.constant([1, 2, 3])
        assert isinstance(tensor, tf.Tensor)
        assert isinstance(tensor, tensor_like)
