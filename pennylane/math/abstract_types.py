# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains data structures to represent abstract arrays."""

from dataclasses import dataclass
from math import prod
from numbers import Number

import numpy as np


@dataclass(frozen=True)
class AbstractArray:
    """Abstract array type."""

    shape: tuple[int, ...]
    dtype: np.dtype | type[Number] = np.int64

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    @property
    def size(self) -> int:
        """Total number of elements."""
        return prod(self.shape)

    @property
    def T(self) -> "AbstractArray":
        """Transpose view of the array."""
        return AbstractArray(self.shape[::-1], self.dtype)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    def __getitem__(self, *_, **__):
        raise IndexError("Cannot index into an abstract array.")

    def __setitem__(self, *_, **__):
        raise IndexError("Cannot index into an abstract array.")

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("len() of unsized object.")
        return self.shape[0]

    def __eq__(self, other: "AbstractArray") -> bool:
        # This should probably just raise an error
        if isinstance(other, AbstractArray):
            return self.shape == other.shape and self.dtype == other.dtype

        raise TypeError("Tried to check equality against an abstract array.")

    def __hash__(self) -> int:
        return hash((self.shape, self.dtype))


AbstractBool = AbstractArray((), bool)
AbstractInt = AbstractArray((), int)
AbstractFloat = AbstractArray((), float)
AbstractComplex = AbstractArray((), complex)


@dataclass(frozen=True)
class AbstractWires(AbstractArray):
    """Abstract wires."""

    num_wires: int

    def __post_init__(self):
        object.__setattr__(self, "shape", (self.num_wires,))
        object.__setattr__(self, "dtype", int)

    def __eq__(self, other: "AbstractWires"):
        if isinstance(other, AbstractWires):
            return self.num_wires == other.num_wires

        raise TypeError("Tried to check equality against an abstract wire register.")

    def __hash__(self):
        return hash(("AbstractWires", self.num_wires))
