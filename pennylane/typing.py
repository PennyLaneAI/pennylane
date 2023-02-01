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

_TensorLike = Union[int, float, bool, complex, bytes, str, list, tuple, np.ndarray, ArrayBox]


class TensorLikeMETA(type):
    """TensorLike metaclass."""

    def __instancecheck__(cls, other):
        return (
            isinstance(other, _TensorLike)
            or _is_jax_instance(other)
            or _is_torch_instance(other)
            or _is_tensorflow_instance(other)
        )

    def __subclasscheck__(cls, other):
        return (
            issubclass(other, _TensorLike)
            or _is_jax_subclass(other)
            or _is_torch_subclass(other)
            or _is_tensorflow_subclass(other)
        )


class TensorLike(metaclass=TensorLikeMETA):
    """Returns a ``Union`` of all tensor-like types, which includes any scalar or sequence
    that can be interpreted as a pennylane tensor, including lists and tuples. Any argument
    accepted by ``pnp.array`` is tensor-like.

    **Examples**

    >>> from pennylane.typing import TensorLike
    >>> isinstance(4, TensorLike)
    True
    >>> isinstance([2, 6, 8], TensorLike)
    True
    >>> isinstance(torch.tensor([1, 2, 3]), TensorLike)
    True
    >>> issubclass(list, TensorLike)
    True
    >>> issubclass(jax.Array, TensorLike)
    True
    """


def _is_jax_instance(instance):
    """Check if instance is a jax tensor."""
    if "jax" in sys.modules:
        with contextlib.suppress(ImportError):
            from jax.numpy import ndarray
            from jax import Array

            return isinstance(instance, (ndarray, Array))
    return False


def _is_tensorflow_instance(instance):
    """Check if instance is a tensorflow tensor."""
    if "tensorflow" in sys.modules or "tensorflow-macos" in sys.modules:
        with contextlib.suppress(ImportError):
            from tensorflow import Tensor as tfTensor
            from tensorflow import Variable

            return isinstance(instance, (tfTensor, Variable))
    return False


def _is_torch_instance(instance):
    """Check if instance is a torch tensor."""
    if "torch" in sys.modules:
        with contextlib.suppress(ImportError):
            from torch import Tensor as torchTensor

            return isinstance(instance, torchTensor)
    return False


def _is_jax_subclass(c):
    """Check if class is a subclass of a jax tensor."""
    if "jax" in sys.modules:
        with contextlib.suppress(ImportError):
            from jax.numpy import ndarray
            from jax import Array

            return issubclass(c, (ndarray, Array))
    return False


def _is_tensorflow_subclass(c):
    """Check if class is a subclass of a tensorflow tensor."""
    if "tensorflow" in sys.modules or "tensorflow-macos" in sys.modules:
        with contextlib.suppress(ImportError):
            from tensorflow import Tensor as tfTensor
            from tensorflow import Variable

            return issubclass(c, (tfTensor, Variable))
    return False


def _is_torch_subclass(c):
    """Check if class is a subclass of a torch tensor."""
    if "torch" in sys.modules:
        with contextlib.suppress(ImportError):
            from torch import Tensor as torchTensor

            return issubclass(c, torchTensor)
    return False
