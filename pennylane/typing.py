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
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from math import prod
from numbers import Number
from typing import Any, Optional, TypeVar, Union

import numpy as np
from autograd.numpy.numpy_boxes import ArrayBox

FlatPytree = tuple[Sequence[Any], Hashable]


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
        shape (tuple[int, ...] | Ellipsis): the dimensions of the array.
            ``()`` corresponds to a scalar. ``Ellipsis`` (``...``) means the number of axes
            is unknown. Within a shape tuple, ``-1`` marks an axis whose size is unknown.
        dtype (type): the data type of the array. Can either be a ``builtin`` like
            ``float`` or ``int``, or a numpy dtype like ``np.complex64``.

    >>> from pennylane.typing import AbstractArray
    >>> AbstractArray((4, 2), float)
    AbstractArray((4, 2), 'float64', weak_type=True)

    Use ``-1`` for an axis with unknown size:

    >>> aa = AbstractArray((-1, 2), np.int32)
    >>> isinstance(np.ones((4, 2), np.int32), aa)
    True

    Use ``Ellipsis`` when the rank itself is unknown:

    >>> isinstance(np.ones((4, 2)), AbstractArray(..., float))
    True

    """

    shape: tuple[int, ...] | types.EllipsisType
    dtype: np.dtype
    _weak_type: bool = field(init=False, hash=True)

    def __post_init__(self):
        weak_type = (
            isinstance(self.dtype, type)
            and issubclass(self.dtype, Number)
            # Need to check this because builtin numpy dtypes (like np.float32) are
            # subclasses of Number
            and self.dtype.__module__ == "builtins"
        )
        if self.shape is not Ellipsis:
            shape = tuple(self.shape)
            if Ellipsis in shape:
                raise ValueError(
                    f"Ellipsis cannot appear inside a shape tuple, but got {shape}. "
                    "Use -1 for axes with unknown sizes, or shape=Ellipsis when the "
                    "number of axes is unknown."
                )
            if not all(isinstance(s, int) for s in shape):
                raise ValueError(
                    f"Shapes can only be initialized with integer values, but got {shape}. "
                    "For axes with unknown sizes, use -1. For an unknown number of axes, "
                    "use shape=Ellipsis."
                )
            object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", np.dtype(self._resolve_dtype(self.dtype)))
        object.__setattr__(self, "_weak_type", weak_type)

    def __instancecheck__(self, instance):
        dtype = getattr(instance, "dtype", None)
        if dtype is None or self._resolve_dtype(dtype) != self.dtype:
            return False

        if self.shape is Ellipsis:
            return True

        shape = getattr(instance, "shape", None)
        return shape is not None and self._shape_matches(shape)

    def is_compatible_with(self, val) -> bool:
        """Check whether an input value is compatible with an ``AbstractArray``."""
        # No need to create a new array if value is already an array
        val = np.array(val) if isinstance(val, (Number, list, tuple)) else val
        shape = val.shape
        dtype = self._resolve_dtype(val.dtype)

        # If self._weak_type, then ``instance``'s dtype must be promotable to self.dtype. For
        # example, int64 can be promoted to float64, and float32 can be promoted to float64.
        # However, the inverse is not true in either case.
        casting = "safe" if self._weak_type else "equiv"
        # In the general case with weak types, this check will fail if there would be subtype
        # promotion but precision demotion. For example, comparing ``dtype == int64`` and
        # ``self.dtype == float32`` will fail, which is inconsistent with weak typing. However,
        # this is fine because our dtype is weak only if the input dtype was a Python builtin
        # number type. In that case, ``self.dtype`` will always have the default precision
        # NumPy uses. For example, ``np.dtype(int) == np.int64``, so ``self.dtype`` will always
        # have a precision for which this check will behave as expected.
        if not np.can_cast(dtype, self.dtype, casting=casting):
            return False

        if self.shape is Ellipsis:
            return True

        return self._shape_matches(shape)

    def _shape_matches(self, shape) -> bool:
        if len(shape) != len(self.shape):
            return False
        return all(s2 in {s1, -1} for s1, s2 in zip(shape, self.shape, strict=True))

    @property
    def size(self) -> int:
        """Total number of elements."""
        if self.shape is Ellipsis or any(s == -1 for s in self.shape):
            raise TypeError(f"size is undefined for {self} with incomplete shape.")
        return prod(self.shape)

    @property
    def T(self) -> "AbstractArray":
        """Transpose view of the array."""
        new_shape = self.shape if self.shape is Ellipsis else self.shape[::-1]
        return AbstractArray(new_shape, self.dtype)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        if self.shape is Ellipsis:
            raise TypeError(f"ndim is undefined for {self} with incomplete shape.")
        return len(self.shape)

    def _resolve_dtype(self, dtype):
        """Convert an arbitrary dtype into a numpy dtype."""
        if dtype.__class__.__module__.split(".")[0] == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            dummy = torch.tensor((), dtype=dtype)
            dtype = dummy.numpy().dtype

        return np.dtype(dtype)

    def __repr__(self):
        shape_repr = "?" if self.shape is Ellipsis else self.shape
        args = f"{shape_repr}, '{self.dtype.name}'"
        if self._weak_type:
            args += ", weak_type=True"
        return f"AbstractArray({args})"

    def __getitem__(self, item):
        raise IndexError("Cannot index into an AbstractArray.")

    def __setitem__(self, *_, **__):
        raise IndexError("Cannot index into an AbstractArray.")

    def __len__(self) -> int:
        if not self.shape or self.shape is Ellipsis:
            raise TypeError("len() of unsized object.")
        if self.shape[0] < 0:
            raise TypeError(f"len() is undefined for {self} with unknown size for the first axis.")
        return self.shape[0]

    def __eq__(self, other) -> bool:
        # This should probably just raise an error
        if isinstance(other, AbstractArray):
            return self.shape == other.shape and self.dtype == other.dtype

        raise TypeError(f"Cannot check equality between AbstractArray and {type(other)}.")

    def __hash__(self) -> int:
        return hash(("AbstractArray", self.shape, self.dtype))
