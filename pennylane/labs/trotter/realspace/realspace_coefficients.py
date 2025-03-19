# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tree representation of coefficients of a realspace operator"""

from __future__ import annotations

from enum import Enum
from itertools import product
from typing import Any, Iterator, Tuple, Union

import numpy as np
from numpy import allclose, isclose, ndarray, zeros

# pylint: disable=too-many-arguments,too-many-positional-arguments,protected-access,too-many-return-statements


class _NodeType(Enum):
    """Enum containing the types of nodes"""

    SUM = 1
    OUTER = 2
    SCALAR = 3
    TENSOR = 4
    FLOAT = 5


class RealspaceCoeffs:  # pylint: disable=too-many-instance-attributes
    """RealspaceCoeffss of a tree used to compute the coefficients of a RealspaceOperator"""

    def __init__(
        self,
        node_type: _NodeType,
        l_child: RealspaceCoeffs = None,
        r_child: RealspaceCoeffs = None,
        l_shape: Tuple[int] = (),
        r_shape: Tuple[int] = (),
        tensor: ndarray = None,
        scalar: float = None,
        value: float = None,
        is_zero: bool = None,
        label: Tuple[str, Any] = None,
    ) -> RealspaceCoeffs:

        self.node_type = node_type
        self.l_child = l_child
        self.r_child = r_child
        self.l_shape = l_shape
        self.r_shape = r_shape
        self.tensor = tensor
        self.scalar = scalar
        self.value = value
        self.is_zero = is_zero
        self.label = label

        match node_type:
            case _NodeType.SUM:
                self.shape = l_child.shape
            case _NodeType.OUTER:
                self.shape = l_child.shape + r_child.shape
            case _NodeType.SCALAR:
                self.shape = l_child.shape
            case _NodeType.TENSOR:
                self.shape = tensor.shape
            case _NodeType.FLOAT:
                self.shape = ()
            case _:
                raise ValueError(f"{node_type} is not a valid _NodeType.")

    @classmethod
    def coeffs(cls, tensor: Union[np.ndarray, float], label: str):
        """User facing method to construct a coefficient tensor"""
        return cls.tensor_node(tensor, label)

    @classmethod
    def sum_node(cls, l_child: RealspaceCoeffs, r_child: RealspaceCoeffs) -> RealspaceCoeffs:
        """Construct a SUM node"""
        if l_child.shape != r_child.shape:
            raise ValueError(
                f"Cannot add RealspaceCoeffs of shape {l_child.shape} with RealspaceCoeffs of shape {r_child.shape}."
            )

        return cls(
            node_type=_NodeType.SUM,
            l_child=l_child,
            r_child=r_child,
            l_shape=l_child.shape,
            r_shape=r_child.shape,
            is_zero=l_child.is_zero and r_child.is_zero,
        )

    @classmethod
    def outer_node(
        cls,
        l_child: RealspaceCoeffs,
        r_child: RealspaceCoeffs,
    ) -> RealspaceCoeffs:
        """Construct a OUTER node"""

        return cls(
            node_type=_NodeType.OUTER,
            l_child=l_child,
            r_child=r_child,
            l_shape=l_child.shape,
            r_shape=r_child.shape,
            is_zero=l_child.is_zero or r_child.is_zero,
        )

    @classmethod
    def tensor_node(cls, tensor: ndarray, label: str = None) -> RealspaceCoeffs:
        """Construct a TENSOR node"""

        if len(tensor.shape):
            return cls(
                node_type=_NodeType.TENSOR,
                tensor=tensor,
                label=label,
                is_zero=allclose(tensor, zeros(tensor.shape)),
            )

        return cls(
            node_type=_NodeType.FLOAT,
            value=tensor,
            is_zero=(tensor == 0),
        )

    @classmethod
    def scalar_node(cls, scalar: float, child: RealspaceCoeffs) -> RealspaceCoeffs:
        """Construct a SCALAR node"""

        return cls(
            node_type=_NodeType.SCALAR,
            l_child=child,
            l_shape=child.shape,
            scalar=scalar,
            is_zero=child.is_zero or isclose(scalar, 0),
        )

    def __add__(self, other: RealspaceCoeffs) -> RealspaceCoeffs:
        return self.__class__.sum_node(self, other)

    def __mul__(self, scalar: float) -> RealspaceCoeffs:
        return self.__class__.scalar_node(scalar, self)

    __rmul__ = __mul__

    def __matmul__(self, other: RealspaceCoeffs) -> RealspaceCoeffs:
        return self.__class__.outer_node(self, other)

    def __eq__(self, other: RealspaceCoeffs) -> bool:
        if self.node_type != other.node_type:
            return False

        if self.shape != other.shape:
            return False

        if self.node_type == _NodeType.OUTER:
            if self.l_shape != other.l_shape:
                return False

            if self.r_shape != other.r_shape:
                return False

            return (self.l_child == other.l_child) and (self.r_child == other.r_child)

        if self.node_type == _NodeType.SCALAR:
            if self.scalar != other.scalar:
                return False

            return self.l_child == other.l_child

        if self.node_type == _NodeType.SUM:
            return (self.l_child == other.l_child) and (self.r_child == other.r_child)

        if self.node_type == _NodeType.TENSOR:
            return allclose(self.tensor, other.tensor)

        if self.node_type == _NodeType.FLOAT:
            return self.value == other.value

        raise ValueError(
            f"RealspaceCoeffs was constructed with invalid _NodeType {self.node_type}."
        )

    def __str__(self) -> str:
        indices = [f"idx{i}" for i in range(len(self.shape))]

        return str(self._str(indices))

    def _str(self, indices) -> str:

        match self.node_type:
            case _NodeType.TENSOR:
                return f"{self.label}[{','.join(indices)}]"
            case _NodeType.FLOAT:
                return f"{self.value}"
            case _NodeType.SCALAR:
                return f"{self.scalar} * ({self.l_child._str(indices)})"
            case _NodeType.OUTER:
                l_indices = indices[: len(self.l_shape)]
                r_indices = indices[len(self.l_shape) :]

                return f"({self.l_child._str(l_indices)}) * ({self.r_child._str(r_indices)})"
            case _NodeType.SUM:
                return f"({self.l_child._str(indices)}) + ({self.r_child._str(indices)})"
            case _:
                raise ValueError(
                    f"RealspaceCoeffs was constructed with invalid _NodeType {self.node_type}."
                )

    def compute(self, index: Tuple[int]) -> float:
        """Compute the coefficient at the given index"""

        if not self._validate_index(index):
            raise ValueError(f"Given index {index} is not compatible with shape {self.shape}")

        match self.node_type:
            case _NodeType.TENSOR:
                return self.tensor[index]
            case _NodeType.FLOAT:
                return self.value
            case _NodeType.SCALAR:
                return self.scalar * self.l_child.compute(index)
            case _NodeType.SUM:
                return self.l_child.compute(index) + self.r_child.compute(index)
            case _NodeType.OUTER:
                l_index = index[: len(self.l_shape)]
                r_index = index[len(self.l_shape) :]
                return self.l_child.compute(l_index) * self.r_child.compute(r_index)
            case _:
                raise ValueError(
                    f"RealspaceCoeffs was constructed with invalid _NodeType {self.node_type}."
                )

    def __getitem__(self, index):
        return self.compute(index)

    def nonzero(self) -> Iterator[Tuple[int]]:
        """Compute the nonzero indices"""

        match self.node_type:
            case _NodeType.TENSOR:
                return zip(*self.tensor.nonzero())
            case _NodeType.FLOAT:
                return iter(((),)) if self.value else iter(())
            case _NodeType.SCALAR:
                return self.l_child.nonzero() if self.scalar else iter(())
            case _NodeType.SUM:
                return _uniq_chain(self.l_child.nonzero(), self.r_child.nonzero())
            case _NodeType.OUTER:
                return _flatten_product(self.l_child.nonzero(), self.r_child.nonzero())
            case _:
                raise ValueError(
                    f"RealspaceCoeffs was constructed with invalid _NodeType {self.node_type}."
                )

    def _validate_index(self, index: Tuple[int]) -> bool:
        if len(index) != len(self.shape):
            return False

        for x, y in zip(index, self.shape):
            if x < 0:
                return False

            if x >= y:
                return False

        return True

    def get_coefficients(self, threshold: float = 0.0):
        """Return the coefficients in a dictionary."""

        match self.node_type:
            case _NodeType.TENSOR:
                return _numpy_to_dict(self.tensor, threshold)
            case _NodeType.FLOAT:
                return {(): self.value}
            case _NodeType.SCALAR:
                return _scale_dict(self.scalar, self.l_child.get_coefficients(threshold), threshold)
            case _NodeType.SUM:
                return _add_dicts(
                    self.l_child.get_coefficients(threshold),
                    self.r_child.get_coefficients(threshold),
                    threshold,
                )
            case _NodeType.OUTER:
                return _mul_dicts(
                    self.l_child.get_coefficients(threshold),
                    self.r_child.get_coefficients(threshold),
                    threshold,
                )
            case _:
                raise ValueError(
                    f"RealspaceCoeffs was constructed with invalid _NodeType {self.node_type}."
                )


def _add_dicts(d1, d2, threshold):
    add_dict = {}

    d1_keys = {d1.keys()}
    d2_keys = {d2.keys()}

    for key in d1_keys.intersection(d2_keys):
        if abs(d1[key] + d2[key]) > threshold:
            add_dict[key] = d1[key] + d2[key]

    for key in d1_keys.difference(d2_keys):
        add_dict[key] = d1[key]

    for key in d2_keys.difference(d1_keys):
        add_dict[key] = d2[key]

    return add_dict


def _mul_dicts(d1, d2, threshold):
    mul_dict = {}

    for key1, key2 in product(d1.keys(), d2.keys()):
        if abs(d1[key1] * d2[key2]) > threshold:
            mul_dict[key1 + key2] = d1[key1] * d2[key2]

    return mul_dict


def _scale_dict(scalar, d, threshold):
    scaled = {}
    for key in d.keys():
        if abs(scalar * d[key]) > threshold:
            scaled[key] = scalar * d[key]

    return scaled


def _numpy_to_dict(arr, threshold):
    nz = arr.nonzero()
    d = {}

    for index in zip(*nz):
        if abs(arr[index]) > threshold:
            d[index] = arr[index]

    return d


def _flatten_product(iter1: Iterator[Tuple], iter2: Iterator[Tuple]) -> Iterator[Tuple]:
    for a, b in product(iter1, iter2):
        yield (*a, *b)


def _uniq_chain(iter1: Iterator, iter2: Iterator) -> Iterator:
    seen = set()

    for a in iter1:
        seen.add(a)
        yield a

    for b in iter2:
        if b not in seen:
            yield b
