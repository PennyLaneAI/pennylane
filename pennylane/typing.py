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
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod
from numbers import Number
from typing import Any, Optional, TypeVar, Union

import numpy as np
from autograd.numpy.numpy_boxes import ArrayBox


class InterfaceTensorMeta(type):
    """defines dunder methods for the ``isinstance`` and ``issubclass`` checks.

    .. note:: These special dunder methods can only be defined inside a metaclass.

    """

    def __instancecheck__(cls, other):
        """Dunder method used to check if an object is a `InterfaceTensor` instance."""
        return _is_jax(other) or _is_torch(other) or _is_tensorflow(other)  # pragma: no cover

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
including lists and tuples. Any argument accepted by ``qp.numpy.array`` is tensor-like.

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


@dataclass(frozen=True)
class AbstractArray:
    """An abstract representation of an array that contains the shape and dtype
    attributes necessary for resource calculations.

    Args:
        shape (tuple(int)): the dimensions of the array. ``()`` corresponds to a scalar
        dtype (type): the data type of the array
    """

    shape: tuple[int | types.EllipsisType, ...]
    dtype: np.dtype | type[Number]

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    @property
    def size(self) -> int:
        """Total number of elements."""
        try:
            return prod(self.shape)
        except TypeError as e:
            if any(s == ... for s in self.shape):
                raise TypeError(
                    f"size is undefined for {self} with unknown shape dimension specified by Ellipsis."
                ) from e
            raise e

    @property
    def T(self) -> "AbstractArray":
        """Transpose view of the array."""
        return AbstractArray(self.shape[::-1], self.dtype)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    def __getitem__(self, item):
        if self.shape:
            raise IndexError(
                "Only scalar AbstractArray's can be indexed into to create new versions."
            )

        if isinstance(item, int) or item == ...:
            item = (item,)
        if not isinstance(item, tuple) or not all(isinstance(n, int) or n == ... for n in item):
            raise TypeError(
                "AbstractArray scalars can only be subscripted with integers and Ellipsis."
            )
        return AbstractArray(item, self.dtype)

    def __setitem__(self, *_, **__):
        raise IndexError("Cannot index into an abstract array.")

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("len() of unsized object.")
        return self.shape[0]

    def __eq__(self, other) -> bool:
        # This should probably just raise an error
        if isinstance(other, AbstractArray):
            return self.shape == other.shape and self.dtype == other.dtype

        raise TypeError("Tried to check equality against an abstract array.")

    def __hash__(self) -> int:
        return hash(("AbstractArray", self.shape, self.dtype))


@dataclass(frozen=True)
class AbstractWires:
    """An abstract representation of a sequence of wires that contains the number
    of wires, useful for resource calculations.

    Args:
        num_wires (int): The number of wires
    """

    num_wires: int | types.EllipsisType

    def __eq__(self, other) -> bool:
        if isinstance(other, AbstractWires):
            return self.num_wires == other.num_wires

        raise TypeError("Tried to check equality against an abstract wire register.")

    def __hash__(self):
        return hash(("AbstractWires", self.num_wires))

    def __len__(self) -> int:
        return self.num_wires

    @classmethod
    def from_wires(cls, wires: Sequence[Any]) -> "AbstractWires":
        """Create an AbstractWires instance from a concrete sequence of wires."""
        if not isinstance(wires, Sequence):
            raise TypeError(f"Cannot create AbstractWires from {wires}.")

        return cls(len(wires))

    def __class_getitem__(cls, item) -> "AbstractWires":
        if not isinstance(item, int) and item != ...:
            raise TypeError(
                f"AbstractWires can only be subscripted with integers and Ellipsis. Got {item}."
            )
        return AbstractWires(item)


Int = AbstractArray((), int)
"""TODO"""


Float = AbstractArray((), float)
"""TODO"""

Bool = AbstractArray((), bool)
"""TODO"""


Complex = AbstractArray((), complex)
"""TODO"""
