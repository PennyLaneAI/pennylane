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
"""The BCHTree class"""

from __future__ import annotations

import copy

from collections import defaultdict
from enum import Enum
from typing import Dict

from numpy import isclose


class BCHNode(Enum):
    """Enum for nodes in the BCH tree"""

    ADD = "ADD"
    COMMUTATOR = "COMMUTATOR"
    FRAGMENT = "FRAGMENT"
    MATMUL = "MATMUL"
    SCALAR = "SCALAR"
    SUB = "SUB"
    ZERO = "ZERO"


class BCHTree:
    """An AST for BCH expressions"""

    def __init__(
        self,
        node_type: BCHNode,
        l_child: BCHTree = None,
        r_child: BCHTree = None,
        fragment: int = None,
        scalar: float = None,
    ):

        self.node_type = node_type

        if node_type in (BCHNode.ADD, BCHNode.COMMUTATOR, BCHNode.MATMUL, BCHNode.SUB):
            if not isinstance(l_child, BCHTree):
                raise TypeError(f"`l_child` must be of type BCHTree, got type {type(l_child)}.")

            if not isinstance(r_child, BCHTree):
                raise TypeError(f"`r_child` must be of type BCHTree, got type {type(r_child)}.")

            self.l_child = l_child
            self.r_child = r_child

        elif node_type == BCHNode.SCALAR:
            if not isinstance(l_child, BCHTree):
                raise TypeError(f"`l_child` must be of type BCHTree, got type {type(l_child)}.")

            if not isinstance(scalar, float):
                raise TypeError(f"`scalar` must be of type float, got type {type(scalar)}.")

            self.l_child = l_child
            self.scalar = scalar

        elif node_type == BCHNode.FRAGMENT:
            if not isinstance(fragment, int):
                raise TypeError(f"`fragment` must be of type int, got type {type(fragment)}.")

            self.fragment = fragment

        else:
            if node_type != BCHNode.ZERO:
                raise ValueError(f"Invalid node type: {node_type}.")

    @classmethod
    def add_node(cls, l_child: BCHTree, r_child: BCHTree) -> BCHTree:
        """Instantiate an ADD node"""

        if l_child.node_type == BCHNode.ZERO:
            return r_child
        if r_child.node_type == BCHNode.ZERO:
            return l_child

        return cls(BCHNode.ADD, l_child=l_child, r_child=r_child)

    @classmethod
    def commutator_node(cls, l_child: BCHTree, r_child: BCHTree) -> BCHTree:
        """Instantiate a COMMUTATOR node"""

        if l_child.node_type == BCHNode.ZERO:
            return BCHTree.zero_node()
        if r_child.node_type == BCHNode.ZERO:
            return BCHTree.zero_node()
        if l_child == r_child:
            return BCHTree.zero_node()

        return cls(BCHNode.COMMUTATOR, l_child=l_child, r_child=r_child)

    @classmethod
    def matmul_node(cls, l_child: BCHTree, r_child: BCHTree) -> BCHTree:
        """Instantiate a MATMUL node"""

        if l_child.node_type == BCHNode.ZERO:
            return BCHTree.zero_node()
        if r_child.node_type == BCHNode.ZERO:
            return BCHTree.zero_node()

        return cls(BCHNode.MATMUL, l_child=l_child, r_child=r_child)

    @classmethod
    def fragment_node(cls, fragment: int) -> BCHTree:
        """Instantiate a FRAGMENT node"""
        return cls(BCHNode.FRAGMENT, fragment=fragment)

    @classmethod
    def scalar_node(cls, child: BCHTree, scalar: float) -> BCHTree:
        """Instantiate a SCALAR node"""

        if child.node_type == BCHNode.ZERO or isclose(scalar, 0):
            return BCHTree.zero_node()

        return cls(BCHNode.SCALAR, l_child=child, scalar=scalar)

    @classmethod
    def sub_node(cls, l_child: BCHTree, r_child: BCHTree):
        """Instantiate a SUB node"""

        if l_child.node_type == BCHNode.ZERO:
            return (-1)*r_child
        if r_child.node_type == BCHNode.ZERO:
            return l_child

        return cls(BCHNode.SUB, l_child=l_child, r_child=r_child)

    @classmethod
    def zero_node(cls):
        """Instantiate a ZERO node"""
        return cls(BCHNode.ZERO)

    def __add__(self, other: BCHTree) -> BCHTree:
        return BCHTree.add_node(self, other)

    def __sub__(self, other: BCHTree) -> BCHTree:
        return BCHTree.sub_node(self, other)

    def __matmul__(self, other: BCHTree) -> BCHTree:
        return BCHTree.matmul_node(self, other)

    def __mul__(self, scalar: float) -> BCHTree:
        return BCHTree.scalar_node(self, scalar)

    __rmul__ = __mul__

    def __repr__(self):

        def wrap(tree: BCHTree) -> str:
            if tree.node_type in (BCHNode.COMMUTATOR, BCHNode.FRAGMENT):
                return str(tree)

            return f"({tree})"

        if self.node_type == BCHNode.ADD:
            return f"{self.l_child} + {self.r_child}"

        if self.node_type == BCHNode.COMMUTATOR:
            return f"[{self.l_child}, {self.r_child}]"

        if self.node_type == BCHNode.MATMUL:
            return f"{wrap(self.l_child)}{wrap(self.r_child)}"

        if self.node_type == BCHNode.FRAGMENT:
            return f"H{self.fragment}"

        if self.node_type == BCHNode.SCALAR:
            return f"{self.scalar}*{wrap(self.l_child)}"

        if self.node_type == BCHNode.SUB:
            return f"{wrap(self.l_child)} - {wrap(self.r_child)}"

        if self.node_type == BCHNode.ZERO:
            return "0"

        raise ValueError(f"Found invalid node type: {self.node_type}.")

    def __eq__(self, other: BCHTree) -> bool:
        if self.node_type != other.node_type:
            return False

        if self.node_type in (BCHNode.ADD, BCHNode.COMMUTATOR, BCHNode.MATMUL, BCHNode.SUB):
            return self.l_child == other.l_child and self.r_child == other.r_child

        if self.node_type == BCHNode.FRAGMENT:
            return self.fragment == other.fragment

        if self.node_type == BCHNode.SCALAR:
            return self.l_child == other.l_child and isclose(self.scalar, other.scalar)

        if self.node_type == BCHNode.ZERO:
            return True

        raise ValueError(f"Found invalid node type {self.node_type}.")

    def simplify(self) -> BCHTree:
        """Simplify the AST according to commutator identities"""
        return _simplify(self)


def bch_approx(x: BCHTree, y: BCHTree) -> BCHTree:
    """Return first term of the BCH expansion"""
    xy = BCHTree.commutator_node(x, y)
    xxy = BCHTree.commutator_node(x, BCHTree.commutator_node(x, y))
    yxy = BCHTree.commutator_node(y, BCHTree.commutator_node(x, y))

    return (1/12)*xxy - (1/12)*yxy

def _simplify(x: BCHTree) -> BCHTree:
    old_ast = copy.copy(x)
    x = _distribute_scalar(x)

    return x if old_ast == x else _simplify(x)


def _commutator_linearity_left(x: BCHTree) -> BCHTree:
    if x.node_type != BCHNode.COMMUTATOR:
        return x

    return x


def _commmutator_linearity_right(x: BCHTree) -> BCHTree:
    if x.node_type != BCHNode.COMMUTATOR:
        return x

    return x

def _group_terms(x: BCHTree) -> Dict[BCHTree, float]:
    term_dict = defaultdict(int)
    term_dict[x] = 1
    return _group_terms_recursion(term_dict)

def _group_terms_recursion(d: Dict[BCHTree, float]) -> Dict[BCHTree, float]:

    new_dict = defaultdict(int)
    for key, val in d.items():

        if key.node_type == BCHNode.ADD:
            pass

        if key.node_type == BCHNode.COMMUTATOR:
            pass

        if key.node_type == BCHNode.SCALAR:
            pass

        if key.node_type == BCHNode.ZERO:
            pass

def _distribute_scalar(x: BCHTree) -> BCHTree:
    if x.node_type != BCHNode.SCALAR:
        return x

    if x.l_child.node_type == BCHNode.ADD:
        l_child = x.scalar * x.l_child.l_child
        r_child = x.scalar * x.l_child.r_child

        return l_child + r_child

    if x.l_child.node_type == BCHNode.SUB:
        l_child = x.scalar * x.l_child.l_child
        r_child = x.scalar * x.l_child.r_child

        return l_child - r_child

    if x.l_child.node_type == BCHNode.SCALAR:
        return (x.scalar * x.l_child.scalar) * x.l_child.l_child

    if x.l_child.node_type == BCHNode.ZERO:
        return BCHTree.zero_node()

    return x
