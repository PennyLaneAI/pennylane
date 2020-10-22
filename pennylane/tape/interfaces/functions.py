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
Wrappers for common functions that manipulate or create
NumPy, TensorFlow, and Torch data structures.
"""
from unittest.mock import MagicMock

import numpy as np
import scipy as sp
from pennylane import numpy as anp
from scipy.linalg import block_diag
from scipy import sparse


try:
    import torch
except ImportError as e:
    torch = MagicMock()


try:
    import tensorflow as tf
except ImportError as e:
    tf = MagicMock()


class WrapperFunctions:

    @staticmethod
    def wrapper_class(obj):
        """Return the correct wrapper class for the type of input.
        """
        if isinstance(obj, (np.ndarray, sp.sparse.spmatrix)):
            return NumpyArrayFunctions
        elif isinstance(obj, (tf.Tensor, tf.Variable)):
            return TfTensorFunctions
        elif isinstance(obj, torch.Tensor):
            return TorchTensorFunctions
        elif isinstance(obj, (list, tuple)):
            return NumpyArrayFunctions
        else:
            raise ValueError("No wrapper defined for input of type {}".format(type(obj)))

    @staticmethod
    def abs(array):
        """Entry-wise absolute value of array.
        Args:
            array (array-like): array with real or complex-valued entries
        Returns:
            array-like
        """
        fn = WrapperFunctions.wrapper_class(array)
        return fn.abs(array)

    @staticmethod
    def angle(array):
        """Entry-wise complex angle of array.
        Args:
            array (array-like): array with real or complex-valued entries
        Returns:
            array-like
        """
        fn = WrapperFunctions.wrapper_class(array)
        return fn.angle(array)

    @staticmethod
    def arcsin(array):
        """Entry-wise arcsine of array.
        Args:
            array (array-like): array with real or complex-valued entries
        Returns:
            array-like
        """
        fn = WrapperFunctions.wrapper_class(array)
        return fn.arcsin(array)

    @staticmethod
    def block_diag(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.block_diag(array)

    @staticmethod
    def diag(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.diag(array)

    @staticmethod
    def dot(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.dot(array)

    @staticmethod
    def einsum(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.einsum(array)

    @staticmethod
    def expand_dims(array, axis):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.expand_dims(array, axis=axis)

    @staticmethod
    def flatten(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.flatten(array)

    @staticmethod
    def gather(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.gather(array)

    @staticmethod
    def isclose(array, val, atol=None):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.isclose(array, val, atol)

    @staticmethod
    def len(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.len(array)

    @staticmethod
    def reduce_sum(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.reduce_sum(array)

    @staticmethod
    def reshape(array, new_shape):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.reshape(array, new_shape)

    @staticmethod
    def scatter(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.scatter(array)

    @staticmethod
    def scatter_element(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.scatter_element(array)

    @staticmethod
    def scatter_element_add(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.scatter_element_add(array)

    @staticmethod
    def shape(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.shape(array)

    @staticmethod
    def sparse_matrix(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.sparse_matrix(array)

    @staticmethod
    def sqrt(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.sqrt(array)

    @staticmethod
    def stack(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.stack(array)

    @staticmethod
    def sum(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.sum(array)

    @staticmethod
    def sum(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.sum(array)

    @staticmethod
    def tensordot(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.tensordot(array)

    @staticmethod
    def transpose(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.transpose(array)

    @staticmethod
    def zeros(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.zeros(array)

# Problem: asarray, cast?


class NumpyArrayFunctions:
    """Wrapper functions taking ndarrays."""

    float64 = anp.float64
    complex128 = anp.complex128

    abs = anp.abs
    isclose = lambda array, other, atol: anp.isclose(array, other, atol=atol)
    angle = anp.angle
    arcsin = anp.arcsin
    asarray = anp.asarray
    block_diag = block_diag
    cast = anp.asarray
    diag = anp.diag
    dot = anp.dot
    einsum = anp.einsum
    expand_dims = lambda array, axis: anp.expand_dims(array, axis=axis)
    flatten = lambda array: array.flatten()
    gather = lambda array, indices: array[indices]
    len = len
    reshape = anp.reshape
    shape = lambda obj: obj.shape
    sparse_matrix = lambda shape: sparse.dok_matrix(shape, dtype=np.float64)
    sqrt = np.sqrt
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


class TfTensorFunctions:
    """Wrapper functions taking TensorFlow tensors."""

    float64 = tf.float64
    complex128 = tf.complex128

    abs = tf.math.abs
    angle = tf.math.angle
    arcsin = tf.math.asin
    asarray = tf.convert_to_tensor
    cast = tf.cast
    diag = tf.linalg.diag
    dot = lambda x, y: tf.tensordot(x, y, axes=1)
    einsum = tf.einsum
    expand_dims = lambda array, axis: tf.expand_dims(array, axis=axis)
    flatten = lambda tensor: tf.reshape(tensor, [-1])
    gather = tf.gather
    len = lambda obj: tf.shape(obj)[0]
    reduce_sum = tf.reduce_sum
    reshape = tf.reshape
    scatter = tf.scatter_nd
    shape = lambda obj: obj.shape
    sparse_matrix = lambda shape: tf.SparseTensor(dense_shape=shape)
    sqrt = tf.math.sqrt
    stack = tf.stack
    sum = tf.reduce_sum
    tensordot = tf.tensordot
    transpose = tf.transpose
    zeros = tf.zeros

    @staticmethod
    def isclose(array, val, atol):
        return tf.abs(array - val) <= atol

    @staticmethod
    def scatter_element(index, value, new_dimensions):
        indices = tf.expand_dims(index, 0)
        tensor = tf.expand_dims(value, 0)
        return TfTensorFunctions.scatter(indices, tensor, new_dimensions)

    @staticmethod
    def block_diag(*tensors):
        linop_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in tensors]
        linop_block_diagonal = tf.linalg.LinearOperatorBlockDiag(linop_blocks)
        return linop_block_diagonal.to_dense()

    @staticmethod
    def scatter_element_add(tensor, index, value):
        return tensor + TfTensorFunctions.scatter_element(index, value, tensor.shape)


class TorchTensorFunctions:
    """Wrapper functions taking PyTorch tensors."""

    float64 = torch.float64
    complex128 = torch.complex128

    abs = torch.abs
    angle = torch.angle
    arcsin = torch.asin
    asarray = torch.as_tensor
    cast = lambda tensor, dtype: tensor.to(dtype=dtype)
    diag = lambda tensor: torch.diag(torch.stack(tensor))
    dot = torch.dot
    einsum = torch.einsum
    expand_dims = lambda array, axis: torch.unsqueeze(array, dim=axis)
    flatten = torch.flatten
    gather = lambda array, indices: array[indices]
    len = lambda obj: obj.size()[0]
    reduce_sum = torch.sum
    reshape = torch.reshape
    shape = lambda obj: obj.size()
    sparse_matrix = lambda shape: torch.sparse.FloatTensor(*shape)
    sqrt = torch.sqrt
    stack = torch.stack
    sum = torch.sum
    tensordot = torch.tensordot
    transpose = lambda tensor: tensor.T
    zeros = torch.zeros

    @staticmethod
    def isclose(array, val, atol):
        val = torch.tensor(val, dtype=array.dtype)
        return torch.isclose(array, val, atol=atol)

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
