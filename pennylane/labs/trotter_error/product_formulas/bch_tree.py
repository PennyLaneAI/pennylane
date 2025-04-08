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
from typing import Sequence, Set, Tuple, Union

from numpy import isclose

from pennylane.labs.trotter_error import Fragment
from pennylane.labs.trotter_error.abstract import commutator as frag_commutator


class BCHNode(Enum):
    """Enum for nodes in the BCH tree"""

    ADD = "ADD"
    COMMUTATOR = "COMMUTATOR"
    FRAGMENT = "FRAGMENT"
    MATMUL = "MATMUL"
    SCALAR = "SCALAR"
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
        order: int = None,
        fragment_set: Set[int] = None,
    ):

        self.node_type = node_type

        if not isinstance(order, int):
            raise TypeError(f"`order` must be of type int, got type {type(order)}.")

        self.order = order
        self.fragment_set = fragment_set

        if node_type in (BCHNode.ADD, BCHNode.COMMUTATOR, BCHNode.MATMUL):
            if not isinstance(l_child, BCHTree):
                raise TypeError(f"`l_child` must be of type BCHTree, got type {type(l_child)}.")

            if not isinstance(r_child, BCHTree):
                raise TypeError(f"`r_child` must be of type BCHTree, got type {type(r_child)}.")

            self.l_child = l_child
            self.r_child = r_child

        elif node_type == BCHNode.SCALAR:
            if not isinstance(l_child, BCHTree):
                raise TypeError(f"`l_child` must be of type BCHTree, got type {type(l_child)}.")

            if not isinstance(scalar, (float, int)):
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

        return cls(
            BCHNode.ADD,
            l_child=l_child,
            r_child=r_child,
            order=max(l_child.order, r_child.order),
            fragment_set=l_child.fragment_set.union(r_child.fragment_set),
        )

    @classmethod
    def commutator_node(cls, l_child: BCHTree, r_child: BCHTree) -> BCHTree:
        """Instantiate a COMMUTATOR node"""

        if l_child.node_type == BCHNode.ZERO:
            return BCHTree.zero_node()
        if r_child.node_type == BCHNode.ZERO:
            return BCHTree.zero_node()
        if l_child == r_child:
            return BCHTree.zero_node()

        return cls(
            BCHNode.COMMUTATOR,
            l_child=l_child,
            r_child=r_child,
            order=l_child.order + r_child.order,
            fragment_set=l_child.fragment_set.union(r_child.fragment_set),
        )

    @classmethod
    def fragment_node(cls, fragment: int) -> BCHTree:
        """Instantiate a FRAGMENT node"""
        return cls(BCHNode.FRAGMENT, fragment=fragment, order=1, fragment_set={fragment})

    @classmethod
    def matmul_node(cls, l_child: BCHTree, r_child: BCHTree) -> BCHTree:
        """Instantiate a MATMUL node"""

        if l_child.node_type == BCHNode.ZERO:
            return BCHTree.zero_node()
        if r_child.node_type == BCHNode.ZERO:
            return BCHTree.zero_node()

        return cls(
            BCHNode.MATMUL,
            l_child=l_child,
            r_child=r_child,
            order=l_child.order + r_child.order,
            fragment_set=l_child.fragment_set.union(r_child.fragment_set),
        )

    @classmethod
    def scalar_node(cls, child: BCHTree, scalar: float) -> BCHTree:
        """Instantiate a SCALAR node"""

        if child.node_type == BCHNode.ZERO or isclose(scalar, 0):
            return BCHTree.zero_node()

        return cls(
            BCHNode.SCALAR,
            l_child=child,
            scalar=scalar,
            order=child.order,
            fragment_set=child.fragment_set,
        )

    @classmethod
    def zero_node(cls):
        """Instantiate a ZERO node"""
        return cls(BCHNode.ZERO, order=0, fragment_set={})

    def __add__(self, other: BCHTree) -> BCHTree:
        return BCHTree.add_node(self, other)

    def __sub__(self, other: BCHTree) -> BCHTree:
        return self + (-1) * other

    def __matmul__(self, other: BCHTree) -> BCHTree:
        return BCHTree.matmul_node(self, other)

    def __mul__(self, scalar: float) -> BCHTree:

        if isclose(scalar, 1):
            return self

        return BCHTree.scalar_node(self, scalar)

    __rmul__ = __mul__

    def __neg__(self) -> BCHTree:
        return (-1) * self

    def __repr__(self):

        def wrap(tree: BCHTree) -> str:
            if tree.node_type in (BCHNode.COMMUTATOR, BCHNode.FRAGMENT):
                return str(tree)

            return f"({tree})"

        if self.node_type == BCHNode.ADD:
            return f"{self.l_child} + {self.r_child}"

        if self.node_type == BCHNode.COMMUTATOR:
            return f"[{self.l_child}, {self.r_child}]"

        if self.node_type == BCHNode.FRAGMENT:
            return f"H{self.fragment}"

        if self.node_type == BCHNode.MATMUL:
            return f"{wrap(self.l_child)}{wrap(self.r_child)}"

        if self.node_type == BCHNode.SCALAR:
            return f"{self.scalar}*{wrap(self.l_child)}"

        if self.node_type == BCHNode.ZERO:
            return "0"

        raise ValueError(f"Found invalid node type: {self.node_type}.")

    def __eq__(self, other: BCHTree) -> bool:
        if not isinstance(other, BCHTree):
            raise TypeError(f"Expected type BCHTree, got type {type(other)}.")

        if self.node_type != other.node_type:
            return False

        if self.node_type in (BCHNode.ADD, BCHNode.COMMUTATOR, BCHNode.MATMUL):
            return self.l_child == other.l_child and self.r_child == other.r_child

        if self.node_type == BCHNode.FRAGMENT:
            return self.fragment == other.fragment

        if self.node_type == BCHNode.SCALAR:
            return self.l_child == other.l_child and isclose(self.scalar, other.scalar)

        if self.node_type == BCHNode.ZERO:
            return True

        raise ValueError(f"Found invalid node type {self.node_type}.")

    def simplify(self, max_order: int = 4) -> BCHTree:
        """Simplify the AST according to commutator identities"""

        return _simplify(self, max_order)

    def evaluate(self, fragments: Sequence[Fragment]):
        """Evaluate the AST"""

        if len(self.fragment_set) != len(fragments):
            raise ValueError(
                f"Product formula requires {len(self.fragment_set)} fragments, but only {len(fragments)} were given"
            )

        if self.node_type == BCHNode.ADD:
            return self.l_child.evaluate(fragments) + self.r_child.evaluate(fragments)

        if self.node_type == BCHNode.COMMUTATOR:
            return frag_commutator(
                self.l_child.evaluate(fragments), self.r_child.evaluate(fragments)
            )

        if self.node_type == BCHNode.FRAGMENT:
            return fragments[self.fragment]

        if self.node_type == BCHNode.MATMUL:
            return self.l_child.evaluate(fragments) @ self.r_child.evaluate(fragments)

        if self.node_type == BCHNode.SCALAR:
            return self.scalar * self.l_child.evaluate(fragments)

        if self.node_type == BCHNode.ZERO:
            return fragments[0].__class__().zero()

        raise ValueError(f"Found invalid node type: {self.node_type}.")


def bch_approx(x: BCHTree, y: BCHTree) -> BCHTree:
    """Return first term of the BCH expansion"""

    xy = BCHTree.commutator_node(x, y)

    return x + y + (1 / 2) * xy


def _simplify(x: BCHTree, max_order: int) -> BCHTree:
    old_ast = copy.copy(x)

    if x.node_type in (BCHNode.ADD, BCHNode.COMMUTATOR):
        x.l_child = _simplify(x.l_child, max_order)
        x.r_child = _simplify(x.r_child, max_order)

    if x.node_type == BCHNode.SCALAR:
        x.l_child = _simplify(x.l_child, max_order)

    x = _distribute_scalar(x)
    x = _factor_scalar_from_commutator(x)
    x = _commutator_linearity(x)
    x = _commutator_is_zero(x)
    x = _order_commutator(x)
    x = _drop_high_order(x, max_order)

    return _group_terms(x) if x == old_ast else _simplify(x, max_order)


def _group_terms(x: BCHTree) -> BCHTree:

    fragments = [
        coeff * BCHTree.fragment_node(fragment) for fragment, coeff in _fragment_dict(x).items()
    ]

    commutator_dict = _commutator_dict(x)

    commutators = [
        coeff * _tuple_to_commutator(commutator)
        for commutator, coeff in _commutator_dict(x).items()
    ]

    return sum(fragments + commutators, BCHTree.zero_node())


def _commutator_dict(x: BCHTree) -> defaultdict:
    d = defaultdict(int)

    if x.node_type == BCHNode.ADD:
        d_left = _commutator_dict(x.l_child)
        d_right = _commutator_dict(x.r_child)

        for commutator, coeff in d_left.items():
            d[commutator] += coeff

        for commutator, coeff in d_right.items():
            d[commutator] += coeff

    if x.node_type == BCHNode.COMMUTATOR:
        d[_commutator_to_tuple(x)] += 1

    if x.node_type == BCHNode.SCALAR:
        d_child = _commutator_dict(x.l_child)
        for fragment, coeff in d_child.items():
            d[fragment] = x.scalar * coeff

    return d


def _commutator_to_tuple(x: BCHTree) -> Tuple[Union[int, Tuple], Union[int, Tuple]]:
    if x.node_type != BCHNode.COMMUTATOR:
        raise ValueError(f"Expected node type COMMUTATOR, got {x.node_type}.")

    l_type = x.l_child.node_type
    r_type = x.r_child.node_type
    l_ret, r_ret = None, None

    if l_type == BCHNode.FRAGMENT:
        l_ret = x.l_child.fragment

    if l_type == BCHNode.COMMUTATOR:
        l_ret = _commutator_to_tuple(x.l_child)

    if r_type == BCHNode.FRAGMENT:
        r_ret = x.r_child.fragment

    if r_type == BCHNode.COMMUTATOR:
        r_ret = _commutator_to_tuple(x.r_child)

    if l_ret is None or r_ret is None:
        raise ValueError(
            f"Children must be of type BCHNode.FRAGMENT, BCHNode.COMMUTATOR, got {x.l_child.node_type} and {x.r_child.node_type}."
        )

    return (l_ret, r_ret)


def _tuple_to_commutator(x: Tuple[Union[int, Tuple], Union[int, Tuple]]) -> BCHTree:
    left, right = x

    if isinstance(left, int) and isinstance(right, int):
        return BCHTree.commutator_node(BCHTree.fragment_node(left), BCHTree.fragment_node(right))

    if isinstance(left, int) and isinstance(right, tuple):
        return BCHTree.commutator_node(BCHTree.fragment_node(left), _tuple_to_commutator(right))

    if isinstance(left, tuple) and isinstance(right, int):
        return BCHTree.commutator_node(_tuple_to_commutator(left), BCHTree.fragment_node(right))

    if isinstance(left, tuple) and isinstance(right, tuple):
        return BCHTree.commutator_node(_tuple_to_commutator(left), _tuple_to_commutator(right))

    raise ValueError(
        f"Expected tuple compents to be of types int or tuple, got {type(left)} and {type(right)}."
    )


def _fragment_dict(x: BCHTree) -> defaultdict:
    d = defaultdict(int)

    if x.node_type == BCHNode.ADD:
        d_left = _fragment_dict(x.l_child)
        d_right = _fragment_dict(x.r_child)

        for fragment, coeff in d_left.items():
            d[fragment] += coeff

        for fragment, coeff in d_right.items():
            d[fragment] += coeff

    if x.node_type == BCHNode.FRAGMENT:
        d[x.fragment] += 1

    if x.node_type == BCHNode.SCALAR:
        d_child = _fragment_dict(x.l_child)
        for fragment, coeff in d_child.items():
            d[fragment] = x.scalar * coeff

    return d


def _commutator_linearity(x: BCHTree) -> BCHTree:
    if x.node_type != BCHNode.COMMUTATOR:
        return x

    if x.l_child.node_type == BCHNode.ADD:
        l_comm = BCHTree.commutator_node(x.l_child.l_child, x.r_child)
        r_comm = BCHTree.commutator_node(x.l_child.r_child, x.r_child)

        return l_comm + r_comm

    if x.r_child.node_type == BCHNode.ADD:
        l_comm = BCHTree.commutator_node(x.l_child, x.r_child.l_child)
        r_comm = BCHTree.commutator_node(x.l_child, x.r_child.r_child)

        return l_comm + r_comm

    return x


def _factor_scalar_from_commutator(x: BCHTree) -> BCHTree:
    if x.node_type != BCHNode.COMMUTATOR:
        return x

    if x.l_child.node_type == BCHNode.SCALAR and x.r_child.node_type == BCHNode.SCALAR:
        return (
            x.l_child.scalar
            * x.r_child.scalar
            * BCHTree.commutator_node(x.l_child.l_child, x.r_child.l_child)
        )

    if x.l_child.node_type == BCHNode.SCALAR:
        return x.l_child.scalar * BCHTree.commutator_node(x.l_child.l_child, x.r_child)

    if x.r_child.node_type == BCHNode.SCALAR:
        return x.r_child.scalar * BCHTree.commutator_node(x.l_child, x.r_child.l_child)

    return x


def _distribute_scalar(x: BCHTree) -> BCHTree:
    if x.node_type != BCHNode.SCALAR:
        return x

    if x.l_child.node_type == BCHNode.ADD:
        l_child = x.scalar * x.l_child.l_child
        r_child = x.scalar * x.l_child.r_child

        return l_child + r_child

    if x.l_child.node_type == BCHNode.SCALAR:
        return (x.scalar * x.l_child.scalar) * x.l_child.l_child

    if x.l_child.node_type == BCHNode.ZERO:
        return BCHTree.zero_node()

    return x


def _commutator_is_zero(x: BCHTree) -> BCHTree:
    if x.node_type != BCHNode.COMMUTATOR:
        return x

    l_type = x.l_child.node_type
    r_type = x.r_child.node_type

    if l_type == BCHNode.FRAGMENT and r_type == BCHNode.FRAGMENT:
        return BCHTree.zero_node() if x.l_child.fragment == x.r_child.fragment else x

    if l_type == BCHNode.COMMUTATOR and r_type == BCHNode.COMMUTATOR:
        l_tuple = _commutator_to_tuple(x.l_child)
        r_tuple = _commutator_to_tuple(x.r_child)

        return BCHTree.zero_node() if l_tuple == (r_tuple[1], r_tuple[0]) else x

    if l_type == BCHNode.SCALAR and r_type == BCHNode.SCALAR:
        return (
            x.l_child.scalar
            * x.r_child.scalar
            * _commutator_is_zero(BCHTree.commutator_node(x.l_child.l_child, x.r_child.l_child))
        )

    if l_type == BCHNode.SCALAR:
        return x.l_child.scalar * _commutator_is_zero(
            BCHTree.commutator_node(x.l_child.l_child, x.r_child)
        )

    if r_type == BCHNode.SCALAR:
        return x.r_child.scalar * _commutator_is_zero(
            BCHTree.commutator_node(x.l_child, x.r_child.l_child)
        )

    return x


def _order_commutator(x: BCHTree) -> BCHTree:
    if x.node_type != BCHNode.COMMUTATOR:
        return x

    l_type = x.l_child.node_type
    r_type = x.r_child.node_type

    if l_type == BCHNode.FRAGMENT and r_type == BCHNode.FRAGMENT:
        if x.l_child.fragment <= x.r_child.fragment:
            return x
        return -BCHTree.commutator_node(x.r_child, x.l_child)

    if l_type == BCHNode.COMMUTATOR and r_type == BCHNode.FRAGMENT:
        return -BCHTree.commutator_node(x.r_child, x.l_child)

    return x


def _drop_high_order(x: BCHTree, max_order: int):
    if x.node_type not in (BCHNode.COMMUTATOR, BCHNode.FRAGMENT):
        return x

    if x.order > max_order:
        return BCHTree.zero_node()

    return x


def _distribute_into_commutator(x: BCHTree):
    if x.node_type != BCHNode.SCALAR:
        return x

    if x.l_child.node_type != BCHNode.COMMUTATOR:
        return x

    return BCHTree.commutator_node(x.scalar * x.l_child.l_child, x.l_child.r_child)


def _combine_with_left_linearity(commutators: defaultdict) -> BCHTree:

    combined = {}
    for commutator, coeff in commutators.items():
        left, right = commutator

        if right in combined:
            combined[right].append((left, coeff))
        else:
            combined[right] = [(left, coeff)]

    ret = BCHTree.zero_node()
    for right, left in combined.items():
        term_sum = BCHTree.zero_node()
        for term in left:
            if isinstance(term[1], int):
                term_sum += term[1] * BCHTree.fragment_node(term[0])

            if isinstance(term[1], tuple):
                term_sum += term[1] * _tuple_to_commutator(term[0])

        right_node = (
            BCHTree.fragment_node(right) if isinstance(right, int) else _tuple_to_commutator(right)
        )
        ret += BCHTree.commutator_node(term_sum, right_node)

    return ret
