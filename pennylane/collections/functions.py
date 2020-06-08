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
"""
ML interface functions to be used with collections.
"""
import sys
from unittest.mock import MagicMock

import numpy as np
from pennylane import numpy as anp
from scipy.linalg import block_diag


try:
    import torch
except ImportError as e:
    torch = MagicMock()


try:
    import tensorflow as tf
except ImportError as e:
    tf = MagicMock()


#############################################################################
# NumPy functions
#############################################################################


class NumPyFunctions:
    float64 = anp.float64
    complex128 = anp.complex128

    abs = anp.abs
    asarray = anp.asarray
    block_diag = block_diag
    cast = anp.asarray
    diag = anp.diag
    dot = anp.dot
    einsum = anp.einsum
    flatten = lambda array: array.flatten()
    gather = lambda array, indices: array[indices]
    reshape = anp.reshape
    sum = anp.sum
    stack = anp.stack
    tensordot = anp.tensordot
    transpose = lambda array: array.T
    zeros = anp.zeros

    @staticmethod
    def scatter_element_add(array, index, value):
        array[tuple(index)] = value
        return array

    @staticmethod
    def reduce_sum(array, axes):
        for a in reversed(axes):
            array = anp.sum(array, axis=a)
        return array


#############################################################################
# TensorFlow functions
#############################################################################


class TensorFlowFunctions:
    float64 = tf.float64
    complex128 = tf.complex128

    abs = tf.abs
    asarray = tf.convert_to_tensor
    cast = tf.cast
    diag = tf.linalg.diag
    dot = lambda x, y: tf.tensordot(x, y, axes=1)
    einsum = tf.einsum
    flatten = lambda tensor: tf.reshape(tensor, [-1])
    gather = tf.gather
    reduce_sum = tf.reduce_sum
    reshape = tf.reshape
    scatter = tf.scatter_nd
    stack = tf.stack
    sum = tf.reduce_sum
    tensordot = tf.tensordot
    transpose = tf.transpose
    zeros = tf.zeros

    @staticmethod
    def scatter_element_add(tensor, index, value):
        return tensor + TensorFlowFunctions.scatter_element(index, value, tensor.shape)

    @staticmethod
    def scatter_element(index, value, new_dimensions):
        indices = tf.expand_dims(index, 0)
        tensor = tf.expand_dims(value, 0)
        return TensorFlowFunctions.scatter(indices, tensor, new_dimensions)

    @staticmethod
    def block_diag(*tensors):
        linop_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in tensors]
        linop_block_diagonal = tf.linalg.LinearOperatorBlockDiag(linop_blocks)
        return linop_block_diagonal.to_dense()


#############################################################################
# Torch functions
#############################################################################


class TorchFunctions:
    float64 = torch.float64
    complex128 = torch.complex128

    abs = torch.abs
    asarray = torch.as_tensor
    cast = lambda tensor, dtype: tensor.to(dtype=dtype)
    diag = lambda tensor: torch.diag(torch.stack(tensor))
    dot = torch.dot
    einsum = torch.einsum
    flatten = torch.flatten
    gather = lambda array, indices: array[indices]
    reduce_sum = torch.sum
    reshape = torch.reshape
    stack = torch.stack
    sum = torch.sum
    tensordot = torch.tensordot
    transpose = lambda tensor: tensor.T
    zeros = torch.zeros

    @staticmethod
    def scatter_element_add(tensor, index, value):
        tensor[tuple(index)] = value
        return tensor

    @staticmethod
    def scatter(indices, tensor, new_dimensions):
        new_array = torch.zeros(new_dimensions, dtype=tensor.dtype)
        new_array[indices] = tensor
        return new_array

    @staticmethod
    def block_diag(*tensors):
        sizes = [t.shape[0] for t in tensors]
        res = torch.zeros([sum(sizes), sum(sizes)], dtype=tensors[0].dtype)

        p = np.cumsum(sizes)
        indices = np.vstack([p - sizes, p]).T

        for t, p in zip(tensors, indices):
            row = np.arange(*p).reshape(-1, 1)
            col = np.arange(*p).reshape(1, -1)
            res[row, col] = t

        return res


MLFunctionWrapper = {
    "np": NumPyFunctions,
    "numpy": NumPyFunctions,
    "autograd": NumPyFunctions,
    "tf": TensorFlowFunctions,
    "torch": TorchFunctions,
}

sys.modules[__name__] = MLFunctionWrapper
