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
"""This file contains different PennyLane types."""
import contextlib

# pylint: disable=import-outside-toplevel, too-few-public-methods
import sys
from typing import Union

import numpy as np
from autograd.numpy.numpy_boxes import ArrayBox


def is_jax_instance(instance):
    """Check if instance is a jax tensor."""
    if "jax" in sys.modules:
        with contextlib.suppress(ImportError):
            from jax.numpy import ndarray
            from jax import Array

            return isinstance(instance, (ndarray, Array))
    return False


def is_tensorflow_instance(instance):
    """Check if instance is a tensorflow tensor."""
    if "tensorflow" in sys.modules or "tensorflow-macos" in sys.modules:
        with contextlib.suppress(ImportError):
            from tensorflow import Tensor as tfTensor
            from tensorflow import Variable

            return isinstance(instance, (tfTensor, Variable))
    return False


def is_torch_instance(instance):
    """Check if instance is a torch tensor."""
    if "torch" in sys.modules:
        with contextlib.suppress(ImportError):
            from torch import Tensor as torchTensor

            return isinstance(instance, torchTensor)
    return False


tensor_like_base = Union[int, float, bool, complex, bytes, str, list, tuple, np.ndarray, ArrayBox]


class TensorLikeMETA(type):
    """TensorLike metaclass."""

    def __instancecheck__(cls, instance):
        return (
            isinstance(instance, tensor_like_base)
            or is_jax_instance(instance)
            or is_torch_instance(instance)
            or is_tensorflow_instance(instance)
        )


class TensorLike(metaclass=TensorLikeMETA):
    """Returns a ``Union`` of all tensor-like types, which includes any scalar or sequence
    that can be interpreted as a pennylane tensor, including lists and tuples. Any argument
    accepted by ``pnp.array`` is tensor-like.

    **Examples**

    >>> from pennylane import typing
    >>> isinstance(4, typing.TensorLike)
    True
    >>> isinstance([2, 6, 8], typing.TensorLike)
    True
    >>> isinstance(torch.tensor([1, 2, 3]), typing.TensorLike)
    True
    """
