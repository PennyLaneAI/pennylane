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
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from math import prod
from numbers import Number
from types import EllipsisType
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
        shape (tuple[int, ...] | types.EllipsisType): the dimensions of the array.
            ``()`` corresponds to a scalar. ``Ellipsis`` (``...``) means the number of axes
            is unknown. Within a shape tuple, ``-1`` marks an axis whose size is unknown.
        dtype (type): the data type of the array. Can either be a ``builtin`` like
            ``float`` or ``int``, or a numpy dtype like ``np.complex64``.

    >>> from pennylane.typing import AbstractArray
    >>> AbstractArray((4, 2), float)
    AbstractArray((4, 2), float64, weak_type=True)

    Use ``-1`` for an axis with unknown size:

    >>> aa = AbstractArray((-1, 2), np.int32)
    >>> isinstance(np.ones((4, 2), np.int32), aa)
    True

    Use ``Ellipsis`` when the rank itself is unknown:

    >>> isinstance(np.ones((4, 2)), AbstractArray(..., float))
    True

    """

    shape: tuple[int, ...] | EllipsisType
    dtype: np.dtype
    shape_fixed: bool = field(init=False)
    _weak_type: bool = field(init=False)

    def __post_init__(self):
        weak_type = (
            isinstance(self.dtype, type)
            and issubclass(self.dtype, Number)
            # Need to check this because builtin numpy dtypes (like np.float32) are
            # subclasses of Number
            and self.dtype.__module__ == "builtins"
        )

        if self.shape is Ellipsis:
            object.__setattr__(self, "shape_fixed", False)
        else:
            shape = tuple(self.shape)
            shape_fixed = True

            for s in shape:
                if not isinstance(s, int) or s < -1:
                    raise ValueError(
                        f"Shapes can only be initialized with integer values, but got {shape}. "
                        "For axes with unknown sizes, use -1. For an unknown number of axes, "
                        "use shape=Ellipsis."
                    )
                if s == -1:
                    shape_fixed = False

            object.__setattr__(self, "shape", shape)
            object.__setattr__(self, "shape_fixed", shape_fixed)

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
        """Check whether an input value is compatible with an ``AbstractArray``. A value
        is considered to be compatible if it has a shape and dtype that can be safely cast
        to the shape and dtype of the ``AbstractArray``.

        Args:
            val (Any): input value to check for compatibility

        Returns:
            bool: ``True`` if ``val`` is compatible, ``False`` otherwise

        For shapes, the following conditions must be met to be considered compatible:

        * If the ``AbstractArray`` allows any rank, i.e., ``shape = ...``, then the
          input value can have any shape.
        * If the ``AbstractArray`` has size ``-1`` for any of the axes, the input
          value can have any size **for those axes**, but must have the same size
          as the ``AbstractArray`` for the rest of the axes.
        * If the ``AbstractArray`` has a fixed shape, then the shape of the input
          value must match exactly.

        For dtypes, the following conditions must be met to be considered compatible:

        * If the ``AbstractArray``'s dtype is weak, i.e., it was initialized with
          a Python builtin number type (``int``, ``float``, etc.), then any dtypes
          that can be safely cast to the ``AbstractArray``'s dtype are compatible.
          For example, ``np.bool`` and ``np.int32`` can be safely cast to ``np.int64``,
          but ``np.float32`` cannot be.
        * If the ``AbstractArray``'s dtype is not weak, i.e., it was initialized with
          a dtype with a specific precision, then the dtype of the input value must
          match the dtype of the ``AbstractArray`` exactly.

        If all the above conditions are met, an input value will be considered compatible.

        **Example**

        >>> aa = AbstractArray((-1, 2), int)
        >>> aa.is_compatible_with(np.ones((5, 2), dtype=np.int32))
        True
        >>> aa.is_compatible_with(np.ones((5, 2), dtype=np.bool))
        True
        >>> aa.is_compatible_with(np.ones((5, 3), dtype=np.int32))
        False
        >>> aa.is_compatible_with(np.ones((5, 2), dtype=np.float64))
        False
        <BLANKLINE>
        >>> aa = AbstractArray((3, 2), np.int32)
        >>> aa.is_compatible_with(np.ones((3, 2), dtype=np.int32))
        True
        >>> aa.is_compatible_with(np.ones((5, 2), dtype=np.int32))
        False
        >>> aa.is_compatible_with(np.ones((3, 2), dtype=np.int16))
        False
        """
        # No need to create a new array if value is already an array
        val = np.array(val) if isinstance(val, (Number, list, tuple)) else val
        shape = getattr(val, "shape", None)
        dtype = getattr(val, "dtype", None)
        if shape is None or dtype is None:
            return False

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
        if shape is Ellipsis:
            return False
        if len(shape) != len(self.shape):
            return False
        return all(s2 in (s1, -1) for s1, s2 in zip(shape, self.shape, strict=True))

    @property
    def size(self) -> int:
        """Total number of elements."""
        if not self.shape_fixed:
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

            dummy = torch.empty((), dtype=dtype)
            dtype = dummy.numpy().dtype

        return np.dtype(dtype)

    def __repr__(self):
        shape_repr = "..." if self.shape is Ellipsis else self.shape
        args = f"{shape_repr}, {self.dtype.name}"
        if self._weak_type:
            args += ", weak_type=True"
        return f"AbstractArray({args})"

    __str__ = __repr__

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


class _AbstractTypeFactory(AbstractArray):
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
        if isinstance(shape, int):
            shape = (shape,)

        if shape is not Ellipsis and (
            not isinstance(shape, tuple) or not all(isinstance(n, int) and n >= -1 for n in shape)
        ):
            raise TypeError(
                "AbstractTypeFactories can only be subscripted with integers and ellipsis."
            )

        res = AbstractArray(shape, self.dtype)
        object.__setattr__(res, "_weak_type", self._weak_type)
        return res


Int = _AbstractTypeFactory(int)
"""An :class:`~.AbstractArray` of ``dtype=int``. On its own, it corresponds to a single scalar, but
can be indexed into to create the :class:`~.AbstractArray` with arbitrary dimensions.

>>> isinstance(np.array(2), qp.typing.Int)
True
>>> qp.typing.Int[4, 2]
AbstractArray((4, 2), int64, weak_type=True)
>>> qp.typing.Int[-1, 10]
AbstractArray((-1, 10), int64, weak_type=True)

"""


Float = _AbstractTypeFactory(float)
"""An :class:`~.AbstractArray` of ``dtype=float``. On its own, it corresponds to a single scalar, but
can be indexed into to create the :class:`~.AbstractArray` with arbitrary dimensions.

>>> isinstance(np.array(2.0), qp.typing.Float)
True
>>> qp.typing.Float[4, 2]
AbstractArray((4, 2), float64, weak_type=True)
>>> qp.typing.Float[-1, 10]
AbstractArray((-1, 10), float64, weak_type=True)

"""

Bool = _AbstractTypeFactory(bool)
"""An :class:`~.AbstractArray` of ``dtype=bool``. On its own, it corresponds to a single scalar, but
can be indexed into to create the :class:`~.AbstractArray` with arbitrary dimensions.

>>> isinstance(np.array(False), qp.typing.Bool)
True
>>> qp.typing.Bool[4, 2]
AbstractArray((4, 2), bool, weak_type=True)
>>> qp.typing.Bool[-1, 10]
AbstractArray((-1, 10), bool, weak_type=True)

"""


Complex = _AbstractTypeFactory(complex)
"""An :class:`~.AbstractArray` of ``dtype=complex``. On its own, it corresponds to a single scalar, but
can be indexed into to create the :class:`~.AbstractArray` with arbitrary dimensions.

>>> isinstance(np.array(0 + 1.2j), qp.typing.Complex)
True
>>> qp.typing.Complex[4, 2]
AbstractArray((4, 2), complex128, weak_type=True)
>>> qp.typing.Complex[-1, 10]
AbstractArray((-1, 10), complex128, weak_type=True)

"""


@dataclass(frozen=True)
class AbstractWires:
    """An abstract representation of a sequence of wires that contains the number
    of wires, useful for resource calculations.

    Args:
        num_wires (int): The number of wires. Use ``-1`` when the wire count is unknown.
    """

    num_wires: int
    shape_fixed: bool = field(init=False)

    def __post_init__(self):
        if not isinstance(self.num_wires, int):
            raise TypeError(
                f"'num_wires' must be an integer, but got {type(self.num_wires).__name__}."
            )
        if self.num_wires < -1:
            raise ValueError(
                f"'num_wires' must be a non-negative integer or -1, but got {self.num_wires}. "
                "For a dynamic number of wires, use -1."
            )
        object.__setattr__(self, "shape_fixed", self.num_wires != -1)

    def __eq__(self, other) -> bool:
        if isinstance(other, AbstractWires):
            return self.num_wires == other.num_wires

        raise TypeError(
            "Cannot check equality between AbstractWires and an object of type "
            f"'{type(other).__name__}'. AbstractWires equality is only supported "
            "against other AbstractWires instances."
        )

    @property
    def shape(self) -> tuple[int]:
        """The number of wires expressed as shape ``(num_wires,)``."""
        return (self.num_wires,)

    @property
    def dtype(self):
        """np.int64. The dtype of wires when used with Catalyst."""
        return np.int64

    def __hash__(self):
        return hash(("AbstractWires", self.num_wires))

    def __len__(self) -> int:
        if self.num_wires < 0:
            raise TypeError(f"len() is undefined for {self} with unknown number of wires.")
        return self.num_wires

    def __repr__(self):
        return f"AbstractWires({self.num_wires})"

    __str__ = __repr__

    def __instancecheck__(self, instance):
        if instance.__class__.__name__ != "Wires":
            return False
        if self.num_wires == -1:
            return True
        return len(instance) == self.num_wires


class _AbstractWireTypeFactory(AbstractWires):
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

        if not isinstance(shape, int):
            raise TypeError("_AbstractWireTypeFactory's can only be subscripted with integers.")
        return AbstractWires(shape)


Wire = _AbstractWireTypeFactory()
"""An :class:`~.AbstractWires` subclass. On it's own, it corresponds to a single wire, but
can be indexed to create :class:`~.AbstractWires` with a fixed or dynamic wire count.

>>> isinstance(Wires([0, 1]), qp.typing.Wire[2])
True
>>> qp.typing.Wire[2]
AbstractWires(2)
>>> qp.typing.Wire[-1]
AbstractWires(-1)

"""
