"""Tree representation of coefficients of a realspace operator"""

from __future__ import annotations

import re
from enum import Enum
from itertools import product
from typing import Any, Iterator, Tuple, Union

import numpy as np
from numpy import allclose, isclose, ndarray, zeros

# pylint: disable=too-many-arguments,too-many-positional-arguments,protected-access,too-many-return-statements


class NodeType(Enum):
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
        node_type: NodeType,
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
            case NodeType.SUM:
                self.shape = l_child.shape
            case NodeType.OUTER:
                self.shape = l_child.shape + r_child.shape
            case NodeType.SCALAR:
                self.shape = l_child.shape
            case NodeType.TENSOR:
                self.shape = tensor.shape
            case NodeType.FLOAT:
                self.shape = ()
            case _:
                raise ValueError(f"{node_type} is not a valid NodeType.")

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
            node_type=NodeType.SUM,
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
            node_type=NodeType.OUTER,
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
                node_type=NodeType.TENSOR,
                tensor=tensor,
                label=label,
                is_zero=allclose(tensor, zeros(tensor.shape)),
            )

        return cls(
            node_type=NodeType.FLOAT,
            value=tensor,
            is_zero=(tensor == 0),
        )

    @classmethod
    def scalar_node(cls, scalar: float, child: RealspaceCoeffs) -> RealspaceCoeffs:
        """Construct a SCALAR node"""

        return cls(
            node_type=NodeType.SCALAR,
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

        if self.node_type == NodeType.OUTER:
            if self.l_shape != other.l_shape:
                return False

            if self.r_shape != other.r_shape:
                return False

            return (self.l_child == other.l_child) and (self.r_child == other.r_child)

        if self.node_type == NodeType.SCALAR:
            if self.scalar != other.scalar:
                return False

            return self.l_child == other.l_child

        if self.node_type == NodeType.SUM:
            return (self.l_child == other.l_child) and (self.r_child == other.r_child)

        if self.node_type == NodeType.TENSOR:
            return allclose(self.tensor, other.tensor)

        if self.node_type == NodeType.FLOAT:
            return self.value == other.value

        raise ValueError(f"RealspaceCoeffs was constructed with invalid NodeType {self.node_type}.")

    def __str__(self) -> str:
        indices = [f"idx{i}" for i in range(len(self.shape))]

        return str(self._str(indices))

    def _str(self, indices) -> str:

        match self.node_type:
            case NodeType.TENSOR:
                return f"{self.label}[{','.join(indices)}]"
            case NodeType.FLOAT:
                return f"{self.value}"
            case NodeType.SCALAR:
                return f"{self.scalar} * ({self.l_child._str(indices)})"
            case NodeType.OUTER:
                l_indices = indices[: len(self.l_shape)]
                r_indices = indices[len(self.l_shape) :]

                return f"({self.l_child._str(l_indices)}) * ({self.r_child._str(r_indices)})"
            case NodeType.SUM:
                return f"({self.l_child._str(indices)}) + ({self.r_child._str(indices)})"
            case _:
                raise ValueError(f"RealspaceCoeffs was constructed with invalid NodeType {self.node_type}.")

    def compile(self, to_numpy: bool = False):
        """Compile to a simple arithmetic expression"""

        indices = [f"idx{i}" for i in range(len(self.shape))]
        local_vars = {}
        expr, local_vars = self._compile(indices, local_vars)

        if to_numpy:
            local_vars["np_abs"] = np.abs
            local_vars["np_sum"] = np.sum

            indices = np.array(list(self.nonzero())).T
            for i, row in enumerate(indices):
                local_vars[f"idx{i}"] = row

            str_rep = f"np_sum(np_abs({expr}))"
        else:
            str_rep = f"{expr}"

        return compile(str_rep, "", "eval"), local_vars

    def _compile(self, indices: Tuple[int], local_vars: dict) -> (str, dict):

        if self.node_type == NodeType.TENSOR:
            index_str = ",".join(indices)

            if self.label:
                var_expr, reference = self.label
                var_name = re.sub(r"\[.*\]", "", var_expr)
                local_vars[var_name] = reference
            else:
                var_expr = (
                    "nparray("
                    + "".join(np.array2string(self.tensor, separator=",").splitlines())
                    + ")"
                )
                local_vars["nparray"] = np.array

            return f"{var_expr}[{index_str}]", local_vars

        if self.node_type == NodeType.FLOAT:
            return f"{self.value}", local_vars

        if self.node_type == NodeType.SCALAR:
            compiled, local_vars = self.l_child._compile(indices, local_vars)

            return f"{self.scalar} * ({compiled})", local_vars

        if self.node_type == NodeType.SUM:
            l_compiled, l_vars = self.l_child._compile(indices, local_vars)
            r_compiled, r_vars = self.r_child._compile(indices, local_vars)
            local_vars = l_vars | r_vars

            return f"({l_compiled}) + ({r_compiled})", local_vars

        if self.node_type == NodeType.OUTER:
            l_indices = indices[: len(self.l_shape)]
            r_indices = indices[len(self.l_shape) :]

            l_compiled, l_vars = self.l_child._compile(l_indices, local_vars)
            r_compiled, r_vars = self.r_child._compile(r_indices, local_vars)
            local_vars = l_vars | r_vars

            return f"({l_compiled}) * ({r_compiled})", local_vars

        raise ValueError(f"RealspaceCoeffs was constructed with invalid NodeType {self.node_type}.")

    def compute(self, index: Tuple[int]) -> float:
        """Compute the coefficient at the given index"""

        if not self._validate_index(index):
            raise ValueError(f"Given index {index} is not compatible with shape {self.shape}")

        match self.node_type:
            case NodeType.TENSOR:
                return self.tensor[index]
            case NodeType.FLOAT:
                return self.value
            case NodeType.SCALAR:
                return self.scalar * self.l_child.compute(index)
            case NodeType.SUM:
                return self.l_child.compute(index) + self.r_child.compute(index)
            case NodeType.OUTER:
                l_index = index[: len(self.l_shape)]
                r_index = index[len(self.l_shape) :]
                return self.l_child.compute(l_index) * self.r_child.compute(r_index)
            case _:
                raise ValueError(f"RealspaceCoeffs was constructed with invalid NodeType {self.node_type}.")

    def __getitem__(self, index):
        return self.compute(index)

    def nonzero(self) -> Iterator[Tuple[int]]:
        """Compute the nonzero indices"""

        match self.node_type:
            case NodeType.TENSOR:
                return zip(*self.tensor.nonzero())
            case NodeType.FLOAT:
                return iter(((),)) if self.value else iter(())
            case NodeType.SCALAR:
                return self.l_child.nonzero() if self.scalar else iter(())
            case NodeType.SUM:
                return _uniq_chain(self.l_child.nonzero(), self.r_child.nonzero())
            case NodeType.OUTER:
                return _flatten_product(self.l_child.nonzero(), self.r_child.nonzero())
            case _:
                raise ValueError(f"RealspaceCoeffs was constructed with invalid NodeType {self.node_type}.")

    def _validate_index(self, index: Tuple[int]) -> bool:
        if len(index) != len(self.shape):
            return False

        for x, y in zip(index, self.shape):
            if x < 0:
                return False

            if x >= y:
                return False

        return True


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
