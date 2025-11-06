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

# pylint: disable=import-outside-toplevel,too-few-public-methods
import sys
from collections.abc import Callable, Sequence
from typing import Optional, TypeVar, Union

import numpy as np
from autograd.numpy.numpy_boxes import ArrayBox


class InterfaceTensorMeta(type):
    """defines dunder methods for the ``isinstance`` and ``issubclass`` checks.

    .. note:: These special dunder methods can only be defined inside a metaclass.

    """

    def __instancecheck__(cls, other):
        """Dunder method used to check if an object is a `InterfaceTensor` instance."""
        return _is_jax(other) or _is_torch(other) or _is_tensorflow(other)

    def __subclasscheck__(cls, other):
        """Dunder method that checks if a class is a subclass of ``InterfaceTensor``."""
        return (
            _is_jax(other, subclass=True)
            or _is_torch(other, subclass=True)
            or _is_tensorflow(other, subclass=True)
        )


class InterfaceTensor(metaclass=InterfaceTensorMeta):
    """Adds support for runtime instance checking of interface-specific tensor-like data"""


TensorLike = Union[
    int, float, bool, complex, bytes, list, tuple, np.ndarray, np.generic, ArrayBox, InterfaceTensor
]
"""A type for all tensor-like data.

TensorLike includes any scalar or sequence that can be interpreted as a pennylane tensor,
including lists and tuples. Any argument accepted by ``qml.numpy.array`` is tensor-like.

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
    if "jax" in sys.modules:
        with contextlib.suppress(ImportError):
            from jax import Array
            from jax.core import Tracer
            from jax.numpy import ndarray

            JaxTensor = ndarray | Array | Tracer
            check = issubclass if subclass else isinstance

            return check(other, JaxTensor)
    return False


def _is_tensorflow(
    other, subclass=False
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
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


Result = TypeVar("Result", dict, tuple, TensorLike)

ResultBatch = Sequence[Result]

PostprocessingFn = Callable[[ResultBatch], Result]
BatchPostprocessingFn = Callable[[ResultBatch], ResultBatch]

JSON = Optional[int | str | bool | list["JSON"] | dict[str, "JSON"]]
