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
from dataclasses import dataclass
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
        shape (tuple(int | types.EllipsisType)): the dimensions of the array.
            ``()`` corresponds to a scalar, and ``...`` corresponds to an unknown dimension.
        dtype (type): the data type of the array. Can either be a ``builtin`` like
            ``float`` or ``int``, or a numpy dtype like ``np.complex64``.

    >>> from pennylane.typing import AbstractArray
    >>> AbstractArray((4, 2), float)
    AbstractArray(shape=(4, 2), dtype=dtype('float64'))

    Ellipsis (``...``) can be used as a placeholder for an unknown, arbitrary sized dimension.

    >>> aa = AbstractArray((..., 2), np.int32)
    >>> aa
    AbstractArray(shape=(Ellipsis, 2), dtype=dtype('int32'))

    ``AbstractArray``'s can be used together with ``isinstance`` checks:

    >>> isinstance(np.ones((4,2), np.int32), aa)
    True

    """

    shape: tuple[int | types.EllipsisType, ...]
    dtype: np.dtype | type[Number]

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        if self.dtype.__class__.__module__.split(".")[0] == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            dummy = torch.tensor((), dtype=self.dtype)
            object.__setattr__(self, "dtype", dummy.numpy().dtype)
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    def __instancecheck__(self, instance):
        if not getattr(instance, "dtype", None) == self.dtype:
            return False
        shape = getattr(instance, "shape", None)
        if shape is None or len(shape) != len(self.shape):
            return False
        return all(s2 in {s1, ...} for s1, s2 in zip(shape, self.shape, strict=True))

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
            raise e  # pragma: no cover

    @property
    def T(self) -> "AbstractArray":
        """Transpose view of the array."""
        return AbstractArray(self.shape[::-1], self.dtype)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    def __getitem__(self, item):
        raise IndexError("Cannot index into an AbstractArray.")

    def __setitem__(self, *_, **__):
        raise IndexError("Cannot index into an AbstractArray.")

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("len() of unsized object.")
        return self.shape[0]

    def __eq__(self, other) -> bool:
        # This should probably just raise an error
        if isinstance(other, AbstractArray):
            return self.shape == other.shape and self.dtype == other.dtype

        raise TypeError(f"Cannot check equality between AbstractArray and {type(other)}.")

    def __hash__(self) -> int:
        return hash(("AbstractArray", self.shape, self.dtype))


class AbstractTypeFactory(AbstractArray):
    """
    An abstraction that enables the generation of AbstractArrays via a highly user-friendly type notation,
    using an override of the __getitem__ method.
    """

    def __init__(self, dtype):
        super().__init__((), dtype)

    def __getitem__(self, shape):
        """
        Overrides the indexing mechanism in python to achieve a user-friendly
        sized type notation.

        Args:
            shape: Gives the shape of the desired abstract type. This is not an index,
                but should be thought of as type notation for sized types.

        Returns:
            An instance of AbstractArray with the desired shape.
        """

        if isinstance(shape, int) or shape == ...:
            shape = (shape,)
        if not isinstance(shape, tuple) or not all(isinstance(n, int) or n == ... for n in shape):
            raise TypeError(
                "AbstractTypeFactory's can only be subscripted with integers and Ellipsis."
            )
        return AbstractArray(shape, self.dtype)


Int = AbstractTypeFactory(int)
"""An :class:`~.AbstractArray` of ``dtype=np.int64``. On it's own, it corresponds to a single scalar, but
can be indexed into to create the :class:`~.AbstractArray` for arbitrary dimensions.

>>> isinstance(np.array(2), qp.typing.Int)
True
>>> qp.typing.Int[4,2]
AbstractArray(shape=(4, 2), dtype=dtype('int64'))

"""


Float = AbstractTypeFactory(float)
"""An :class:`~.AbstractArray` of ``dtype=np.float64``. On it's own, it corresponds to a single scalar, but
can be indexed into to create the :class:`~.AbstractArray` for arbitrary dimensions.

>>> isinstance(np.array(2.0), qp.typing.Float)
True
>>> qp.typing.Int[4,2]
AbstractArray(shape=(4, 2), dtype=dtype('float64'))

"""

Bool = AbstractTypeFactory(bool)
"""An :class:`~.AbstractArray` of ``dtype=np.bool``. On it's own, it corresponds to a single scalar, but
can be indexed into to create the :class:`~.AbstractArray` for arbitrary dimensions.

>>> isinstance(np.array(False), qp.typing.Bool)
True
>>> qp.typing.Bool[4]
AbstractArray(shape=(4,), dtype=dtype('bool'))

"""


Complex = AbstractTypeFactory(complex)
"""An :class:`~.AbstractArray` of ``dtype=np.complex128``. On it's own, it corresponds to a single scalar, but
can be indexed into to create the :class:`~.AbstractArray` for arbitrary dimensions.

>>> isinstance(np.array(0+1.2j), qp.typing.Complex)
True
>>> qp.typing.Complex[..., 2]
AbstractArray(shape=(Ellipsis, 2), dtype=dtype('complex128'))

"""


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

    @property
    def shape(self) -> tuple[int]:
        """The number of wires expressed as shape ``(num_wires, )``."""
        return (self.num_wires,)

    @property
    def dtype(self):
        """np.int64.  The dtype of wires when used with Catalyst."""
        return np.int64

    def __hash__(self):
        return hash(("AbstractWires", self.num_wires))

    def __len__(self) -> int:
        return self.num_wires

    def __instancecheck__(self, instance):
        if not instance.__class__.__name__ == "Wires":
            return False
        return len(instance) == self.num_wires


class AbstractWireTypeFactory(AbstractWires):
    """
    An abstraction that enables the generation of AbstractWires via a highly user-friendly type notation,
    using an override of the __getitem__ method.
    """

    def __init__(self):
        super().__init__(1)

    def __getitem__(self, shape):
        """
        Overrides the indexing mechanism in python to achieve a user-friendly
        sized type notation.

        Args:
            shape: Gives the shape of the desired abstract wires type. This is not an index,
                but should be thought of as type notation for sized types.

        Returns:
            An instance of AbstractWires with the desired shape.
        """

        if not (isinstance(shape, int) or shape == ...):
            raise TypeError(
                "AbstractWireTypeFactory's can only be subscripted with integers and Ellipsis."
            )
        return AbstractWires(shape)


WireType = AbstractWireTypeFactory()
"""An :class:`~.AbstractWires`` subclass. On it's own, it corresponds to a single scalar, but
can be indexed into to create the :class:`~.AbstractWires` for arbitrary dimensions.

>>> isinstance(Wires([0, 1]), qp.typing.WireType[2])
True
>>> qp.typing.WireType[2]
AbstractWires(num_wires=2)

"""
