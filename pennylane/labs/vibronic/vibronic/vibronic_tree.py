"""The VibronicTree class"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

from numpy import allclose, average, ndarray, zeros


class NodeType(Enum):
    """Enum containing the types of nodes in a VibronicTree"""

    SUM = 1
    OUTER = 2
    HADAMARD = 3
    SCALAR = 4
    TENSOR = 5


class Node:  # pylint: disable=too-many-instance-attributes
    """Nodes of a tree used to compute the coefficients of a VibronicTerm"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        node_type: NodeType,
        l_child: Node = None,
        r_child: Node = None,
        l_shape: Tuple[int] = tuple(),
        r_shape: Tuple[int] = tuple(),
        tensor: ndarray = None,
        average: float = None,
        scalar: float = None,
    ) -> Node:

        self.node_type = node_type
        self.l_child = l_child
        self.r_child = r_child
        self.l_shape = l_shape
        self.r_shape = r_shape
        self.tensor = tensor
        self.average = average
        self.scalar = scalar

        if node_type == NodeType.SUM:
            self.shape = l_child.shape
        if node_type == NodeType.OUTER:
            self.shape = l_child.shape + r_child.shape
        if node_type == NodeType.HADAMARD:
            self.shape = l_child.shape
        if node_type == NodeType.SCALAR:
            self.shape = l_child.shape
        if node_type == NodeType.TENSOR:
            self.shape = tensor.shape

    @classmethod
    def sum_node(cls, l_child: Node, r_child: Node) -> Node:
        """Construct a SUM node"""
        if l_child.shape != r_child.shape:
            raise ValueError(
                f"Cannot add Node of shape {l_child.shape} with Node of shape {r_child.shape}."
            )

        return cls(
            node_type=NodeType.SUM,
            l_child=l_child,
            r_child=r_child,
            l_shape=l_child.shape,
            r_shape=r_child.shape,
        )

    @classmethod
    def outer_node(
        cls,
        l_child: Node,
        r_child: Node,
    ) -> Node:
        """Construct a OUTER node"""
        return cls(
            node_type=NodeType.OUTER,
            l_child=l_child,
            r_child=r_child,
            l_shape=l_child.shape,
            r_shape=r_child.shape,
        )

    @classmethod
    def hadamard_node(
        cls,
        l_child: Node,
        r_child: Node,
    ) -> Node:
        """Construct a HADAMARD node"""
        if l_child.shape != r_child.shape:
            raise ValueError(
                f"Cannot take Hadamard product of Node of shape {l_child.shape} with Node of shape {r_child.shape}."
            )

        return cls(
            node_type=NodeType.HADAMARD,
            l_child=l_child,
            r_child=r_child,
            l_shape=l_child.shape,
            r_shape=r_child.shape,
        )

    @classmethod
    def tensor_node(cls, tensor: ndarray) -> Node:
        """Construct a TENSOR node"""
        return cls(node_type=NodeType.TENSOR, tensor=tensor, average=average(tensor))

    @classmethod
    def scalar_node(cls, scalar: float, child: Node) -> Node:
        """Construct a SCALAR node"""
        return cls(node_type=NodeType.SCALAR, l_child=child, l_shape=child.shape, scalar=scalar)

    def __add__(self, other: Node) -> Node:
        return self.__class__.sum_node(self, other)

    def __mul__(self, scalar: float) -> Node:
        return self.__class__.scalar_node(scalar, self)

    __rmul__ = __mul__

    def __matmul__(self, other: Node) -> Node:
        return self.__class__.outer_node(self, other)

    def __eq__(self, other: Node) -> bool:  # pylint: disable=too-many-return-statements
        if self.node_type != other.node_type:
            return False

        if self.shape != other.shape:
            return False

        if self.node_type in (NodeType.SUM, NodeType.HADAMARD):
            return (self.l_child == other.l_child) and (self.r_child == other.r_child)

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

        if self.node_type == NodeType.TENSOR:
            return allclose(self.tensor, other.tensor)

        raise ValueError(f"Node was constructed with invalid NodeType {self.node_type}.")

    def __repr__(self) -> str:
        if self.node_type == NodeType.TENSOR:
            return f"Node(TENSOR, {self.tensor})"

        if self.node_type == NodeType.SCALAR:
            return f"Node(SCALAR, {self.scalar}, {self.l_child})"

        if self.node_type == NodeType.OUTER:
            return f"Node(OUTER, {self.l_shape}, {self.r_shape}, {self.l_child}, {self.r_child})"

        if self.node_type == NodeType.SUM:
            return f"Node(SUM, {self.l_child}, {self.r_child})"

        if self.node_type == NodeType.HADAMARD:
            return f"Node(HADAMARD, {self.l_child}, {self.r_child})"

        raise ValueError(f"Node was constructed with invalid NodeType {self.node_type}.")

    def __str__(self) -> str:
        return self._str(0)

    def _str(self, level: int = 0) -> str:
        # pylint: disable=protected-access

        ret = "\t" * level
        if self.node_type == NodeType.TENSOR:
            ret += f"(TENSOR, {self.tensor})"

        if self.node_type == NodeType.SCALAR:
            ret += f"(SCALAR, {self.scalar}, \n"
            ret += self.l_child._str(level + 1)
            ret += ")"

        if self.node_type == NodeType.OUTER:
            ret += f"(OUTER, {self.l_shape}, {self.r_shape}, \n"
            ret += self.l_child._str(level + 1) + ",\n"
            ret += self.r_child._str(level + 1)
            ret += ")"

        if self.node_type == NodeType.SUM:
            ret += "(SUM, \n"
            ret += self.l_child._str(level + 1) + ",\n"
            ret += self.r_child._str(level + 1)
            ret += ")"

        if self.node_type == NodeType.HADAMARD:
            ret += "(HADAMARD, \n"
            ret += self.l_child._str(level + 1) + ",\n"
            ret += self.r_child._str(level + 1)
            ret += ")"

        return ret

    def compute(self, index: Tuple[int]) -> float:
        """Compute the coefficient at the given index"""

        if not self._validate_index(index):
            raise ValueError(f"Given index {index} is not compatible with shape {self.shape}")

        if self.node_type == NodeType.TENSOR:
            return self.tensor[index]

        if self.node_type == NodeType.SCALAR:
            return self.scalar * self.l_child.compute(index)

        if self.node_type == NodeType.SUM:
            return self.l_child.compute(index) + self.r_child.compute(index)

        if self.node_type == NodeType.OUTER:
            l_index = index[: len(self.l_shape)]
            r_index = index[len(self.l_shape) :]
            return self.l_child.compute(l_index) * self.r_child.compute(r_index)

        if self.node_type == NodeType.HADAMARD:
            return self.l_child.compute(index) * self.r_child.compute(index)

        raise ValueError(f"Node was constructed with invalid NodeType {self.node_type}.")

    def compute_average(self) -> float:
        """Compute the average of the term's coefficients"""

        if self.node_type == NodeType.TENSOR:
            return self.average

        if self.node_type == NodeType.SCALAR:
            return self.scalar * self.l_child.compute_average()

        if self.node_type == NodeType.SUM:
            return (self.l_child.compute_average() + self.r_child.compute_average()) / 2

        if self.node_type == NodeType.OUTER:
            return self.l_child.compute_average() * self.r_child.compute_average()

        if self.node_type == NodeType.HADAMARD:
            raise NotImplementedError("HADAMARD nodes not supported")

        raise ValueError(f"Node was constructed with invalid NodeType {self.node_type}.")

    def is_zero(self) -> bool:
        """Returning true means the tree computes zero on every index, however there are false negatives"""

        if self.node_type == NodeType.TENSOR:
            return allclose(self.tensor, zeros(self.tensor.shape))

        if self.node_type == NodeType.SCALAR:
            return self.scalar == 0 or self.l_child.is_zero()

        if self.node_type == NodeType.SUM:
            return self.l_child.is_zero() and self.r_child.is_zero()

        if self.node_type == NodeType.OUTER:
            return self.l_child.is_zero() or self.r_child.is_zero()

        if self.node_type == NodeType.HADAMARD:
            return self.l_child.is_zero() or self.r_child.is_zero()

        raise ValueError(f"Node was constructed with invalid NodeType {self.node_type}.")

    def _validate_index(self, index: Tuple[int]) -> bool:
        if len(index) != len(self.shape):
            return False

        for x, y in zip(index, self.shape):
            if not isinstance(x, type(y)):
                return False

            if x < 0:
                return False

            if x >= y:
                return False

        return True
