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
from typing import Union, TypeVar, Tuple

import numpy as np
from autograd.numpy.numpy_boxes import ArrayBox

_TensorLike = Union[int, float, bool, complex, bytes, list, tuple, np.ndarray, ArrayBox, np.generic]


class TensorLikeMETA(type):
    """TensorLike metaclass that defines dunder methods for the ``isinstance`` and ``issubclass``
    checks.

    .. note:: These special dunder methods can only be defined inside a metaclass.
    """

    def __instancecheck__(cls, other):
        """Dunder method used to check if an object is a `TensorLike` instance."""
        return (
            isinstance(other, _TensorLike.__args__)  # TODO: Remove __args__ when python>=3.10
            or _is_jax(other)
            or _is_torch(other)
            or _is_tensorflow(other)
        )

    def __subclasscheck__(cls, other):
        """Dunder method that checks if a class is a subclass of ``TensorLike``."""
        return (
            issubclass(other, _TensorLike.__args__)  # TODO: Remove __args__ when python>=3.10
            or _is_jax(other, subclass=True)
            or _is_torch(other, subclass=True)
            or _is_tensorflow(other, subclass=True)
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


def _is_jax(other, subclass=False):
    """Check if other is an instance or a subclass of a jax tensor."""
    # pylint: disable=c-extension-no-member
    if "jax" in sys.modules:
        with contextlib.suppress(ImportError):
            import jax
            import jaxlib
            from jax.numpy import ndarray

            JaxTensor = Union[
                ndarray,
                (
                    jax.Array  # TODO: keep this after jax>=0.4 is required
                    if hasattr(jax, "Array")
                    else Union[jaxlib.xla_extension.DeviceArray, jax.core.Tracer]
                ),  # pylint: disable=c-extension-no-member
            ]
            check = issubclass if subclass else isinstance

            return check(other, JaxTensor)
    return False


def _is_tensorflow(other, subclass=False):
    """Check if other is an instance or a subclass of a tensorflow tensor."""
    if "tensorflow" in sys.modules or "tensorflow-macos" in sys.modules:
        with contextlib.suppress(ImportError):
            from tensorflow import Tensor as tfTensor
            from tensorflow import Variable

            check = issubclass if subclass else isinstance

            return check(other, (tfTensor, Variable))
    return False


def _is_torch(other, subclass=False):
    """Check if other is an instance or a subclass of a torch tensor."""
    if "torch" in sys.modules:
        with contextlib.suppress(ImportError):
            from torch import Tensor as torchTensor

            check = issubclass if subclass else isinstance

            return check(other, torchTensor)
    return False


Result = TypeVar("Result", Tuple, TensorLike)

ResultBatch = Tuple[Result]
