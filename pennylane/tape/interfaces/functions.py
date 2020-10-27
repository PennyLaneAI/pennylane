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
        elif isinstance(obj, (tf.Tensor, tf.Variable)):
            return TfTensorFunctions
        elif isinstance(obj, torch.Tensor):
            return TorchTensorFunctions
        elif isinstance(obj, (list, tuple)):
            return NumpyArrayFunctions
        else:
            raise ValueError("No wrapper defined for input of type {}".format(type(obj)))

    # @staticmethod
    # def abs(array):
    #     """Entry-wise absolute value of array.
    #     Args:
    #         array (array-like): array with real or complex-valued entries
    #     Returns:
    #         array-like
    #     """
    #     fn = WrapperFunctions.wrapper_class(array)
    #     return fn.abs(array)


class NumpyArrayFunctions:
    """Wrapper functions taking ndarrays."""


class TfTensorFunctions:
    """Wrapper functions taking TensorFlow tensors."""


class TorchTensorFunctions:
    """Wrapper functions taking PyTorch tensors."""

