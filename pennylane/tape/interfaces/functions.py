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
import pennylane as qml
from pennylane import numpy as anp


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
        if isinstance(obj, (np.ndarray, qml.numpy.tensor)):
            return NumpyArrayFunctions
        elif isinstance(obj, (tf.Tensor, tf.Variable, tf.python.ops.resource_variable_ops.ResourceVariable)):
            return TfTensorFunctions
        elif isinstance(obj, torch.Tensor):
            return TorchTensorFunctions
        elif isinstance(obj, (list, tuple)):
            return NumpyArrayFunctions
        else:
            raise ValueError("No wrapper defined for input of type {}".format(type(obj)))

    @staticmethod
    def expand_dims(array, axis):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.expand_dims(array, axis=axis)

    @staticmethod
    def ones_like(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.ones_like(array)


    @staticmethod
    def to_ndarray(array):
        fn = WrapperFunctions.wrapper_class(array)
        return fn.to_ndarray(array)


class NumpyArrayFunctions:
    """Wrapper functions taking ndarrays."""

    expand_dims = lambda array, axis: anp.expand_dims(array, axis=axis)

    @staticmethod
    def to_ndarray(array):
        return anp.array(array)

    @staticmethod
    def ones_like(array):
        return anp.ones_like(array)


class TfTensorFunctions:
    """Wrapper functions taking TensorFlow tensors."""

    expand_dims = lambda array, axis: tf.expand_dims(array, axis=axis)

    @staticmethod
    def to_ndarray(array):
        return array.numpy()

    @staticmethod
    def ones_like(array):
        return tf.ones_like(array)


class TorchTensorFunctions:
    """Wrapper functions taking PyTorch tensors."""

    expand_dims = lambda array, axis: torch.unsqueeze(array, dim=axis)

    @staticmethod
    def to_ndarray(array):
        return array.detach.numpy()

    @staticmethod
    def ones_like(array):
        return torch.ones_like(array)
