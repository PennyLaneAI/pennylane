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

from collections import defaultdict
from enum import Enum
from typing import Dict, Hashable, Sequence, Set, Tuple

from numpy import isclose

from pennylane.labs.trotter_error import Fragment
from pennylane.labs.trotter_error.abstract import commutator as frag_commutator


class BCHNode(Enum):
    """Enum for nodes in the BCH tree"""

    ADD = "ADD"
    COMMUTATOR = "COMMUTATOR"
    FRAGMENT = "FRAGMENT"
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
        scalar: complex = None,
        fragment_set: Set[Hashable] = None,
    ):

        self.node_type = node_type
        self.fragment_set = fragment_set

        if node_type in (BCHNode.ADD, BCHNode.COMMUTATOR):
            if not isinstance(l_child, BCHTree):
                raise TypeError(f"`l_child` must be of type BCHTree, got type {type(l_child)}.")

            if not isinstance(r_child, BCHTree):
                raise TypeError(f"`r_child` must be of type BCHTree, got type {type(r_child)}.")

            self.l_child = l_child
            self.r_child = r_child

        elif node_type == BCHNode.SCALAR:
            if not isinstance(l_child, BCHTree):
                raise TypeError(f"`l_child` must be of type BCHTree, got type {type(l_child)}.")

            if not isinstance(scalar, (float, int, complex)):
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
            fragment_set=l_child.fragment_set.union(r_child.fragment_set),
        )

    @classmethod
    def fragment_node(cls, fragment: Hashable) -> BCHTree:
        """Instantiate a FRAGMENT node"""
        return cls(BCHNode.FRAGMENT, fragment=fragment, fragment_set={fragment})

    @classmethod
    def scalar_node(cls, child: BCHTree, scalar: complex) -> BCHTree:
        """Instantiate a SCALAR node"""

        if child.node_type == BCHNode.ZERO or isclose(scalar, 0):
            return BCHTree.zero_node()

        return cls(
            BCHNode.SCALAR,
            l_child=child,
            scalar=scalar,
            fragment_set=child.fragment_set,
        )

    @classmethod
    def zero_node(cls) -> BCHTree:
        """Instantiate a ZERO node"""
        return cls(BCHNode.ZERO, fragment_set={})

    @classmethod
    def nested_commutator(cls, fragments: Sequence[BCHTree]) -> BCHTree:
        """Instantiate a nested commutator"""

        if len(fragments) == 2:
            return cls.commutator_node(fragments[0], fragments[1])

        head, *tail = fragments
        return cls.commutator_node(head, cls.nested_commutator(tail))

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

        if self.node_type in (BCHNode.ADD, BCHNode.COMMUTATOR):
            return self.l_child == other.l_child and self.r_child == other.r_child

        if self.node_type == BCHNode.FRAGMENT:
            return self.fragment == other.fragment

        if self.node_type == BCHNode.SCALAR:
            return self.l_child == other.l_child and isclose(self.scalar, other.scalar)

        if self.node_type == BCHNode.ZERO:
            return True

        raise ValueError(f"Found invalid node type {self.node_type}.")

    def __hash__(self) -> int:
        if self.node_type == BCHNode.ADD:
            return hash((self.node_type, frozenset((self.l_child, self.r_child))))

        if self.node_type == BCHNode.COMMUTATOR:
            return hash((self.node_type, self.l_child, self.r_child))

        if self.node_type == BCHNode.FRAGMENT:
            return hash((self.node_type, self.fragment))

        if self.node_type == BCHNode.SCALAR:
            return hash((self.node_type, self.scalar, self.l_child))

        if self.node_type == BCHNode.ZERO:
            return 0

        raise ValueError(f"Found invalid node type {self.node_type}.")

    def simplify(self, max_order: int = 3) -> BCHTree:
        """Simplify the AST according to commutator identities"""

        return _group_terms(_simplify(self, max_order))

    def evaluate(self, fragments: Dict[Hashable, Fragment]):
        """Evaluate the AST"""

        if self.fragment_set != set(fragments.keys()):
            raise ValueError("Fragment labels do not match leaf fragments of AST.")

        if self.node_type == BCHNode.ADD:
            return self.l_child.evaluate(fragments) + self.r_child.evaluate(fragments)

        if self.node_type == BCHNode.COMMUTATOR:
            return frag_commutator(
                self.l_child.evaluate(fragments), self.r_child.evaluate(fragments)
            )

        if self.node_type == BCHNode.FRAGMENT:
            return fragments[self.fragment]

        if self.node_type == BCHNode.SCALAR:
            return self.scalar * self.l_child.evaluate(fragments)

        if self.node_type == BCHNode.ZERO:
            return fragments[0].__class__().zero()

        raise ValueError(f"Found invalid node type: {self.node_type}.")


def bch_approx(x: BCHTree, y: BCHTree) -> BCHTree:
    """Return first term of the BCH expansion"""

    return (
        x
        + y
        + (1 / 2) * BCHTree.commutator_node(x, y)
        + (1 / 12) * BCHTree.nested_commutator([x, x, y])
        + (1 / 12) * BCHTree.nested_commutator([y, y, x])
        + (1 / 24) * BCHTree.nested_commutator([y, x, x, y])
        - (1 / 720) * BCHTree.nested_commutator([y, y, y, y, x])
        - (1 / 720) * BCHTree.nested_commutator([x, x, x, x, y])
        + (1 / 360) * BCHTree.nested_commutator([x, y, y, y, x])
        + (1 / 360) * BCHTree.nested_commutator([y, x, x, x, y])
        + (1 / 120) * BCHTree.nested_commutator([y, x, y, x, y])
        + (1 / 120) * BCHTree.nested_commutator([x, y, x, y, x])
        # + (1 / 240) * BCHTree.nested_commutator([x, y, x, y, x, y])
        # + (1 / 720) * BCHTree.nested_commutator([x, y, x, x, x, y])
        # - (1 / 720) * BCHTree.nested_commutator([x, x, y, y, x, y])
        # + (1 / 1440) * BCHTree.nested_commutator([x, y, y, y, x, y])
        # - (1 / 1440) * BCHTree.nested_commutator([x, x, y, x, x, y])
    )


def _simplify(x: BCHTree, max_order: int) -> BCHTree:

    changed = False

    if x.node_type in (BCHNode.ADD, BCHNode.COMMUTATOR):
        x.l_child = _simplify(x.l_child, max_order)
        x.r_child = _simplify(x.r_child, max_order)

    if x.node_type == BCHNode.SCALAR:
        x.l_child = _simplify(x.l_child, max_order)

    x, y = _distribute_scalar(x)
    changed = changed or y

    x, y = _factor_scalar_from_commutator(x)
    changed = changed or y

    x, y = _commutator_linearity(x)
    changed = changed or y

    x, y = _order_commutator_terms(x)
    changed = changed or y

    x, y = _drop_high_order(x, max_order)
    changed = changed or y

    return _simplify(x, max_order) if changed else x


def _group_terms(x: BCHTree) -> BCHTree:

    fragments = []
    commutators = []

    for term, coeff in _term_dict(x).items():
        if isclose(coeff, 0):
            continue

        if term.node_type == BCHNode.COMMUTATOR:
            commutators.append(coeff * term)

        if term.node_type == BCHNode.FRAGMENT:
            fragments.append(coeff * term)

    commutators = _combine_left_linearity(commutators)

    return sum(fragments + commutators, BCHTree.zero_node())


def _combine_left_linearity(commutators: Sequence[BCHTree]) -> Sequence[BCHTree]:
    right_sides = {}

    for commutator in commutators:
        if commutator.node_type == BCHNode.COMMUTATOR:
            left = commutator.l_child
            right = commutator.r_child

        elif commutator.node_type == BCHNode.SCALAR:
            if commutator.l_child.node_type != BCHNode.COMMUTATOR:
                raise ValueError(
                    f"Expected child node type BCHNode.COMMUTATOR, got {commutator.l_child.node_type} instead."
                )

            left = commutator.scalar * commutator.l_child.l_child
            right = commutator.l_child.r_child

        else:
            raise ValueError(
                f"Expected node type BCHNode.COMMUTATOR or BCHNode.SCALAR, got {commutator.node_type} instead."
            )

        if right in right_sides:
            right_sides[right] += left
        else:
            right_sides[right] = left

    return [BCHTree.commutator_node(left, right) for right, left in right_sides.items()]


def _term_dict(x: BCHTree) -> Dict[BCHTree, float]:
    d = defaultdict(int)

    if x.node_type in (BCHNode.COMMUTATOR, BCHNode.FRAGMENT):
        d[x] = 1

    if x.node_type == BCHNode.ADD:
        d_left = _term_dict(x.l_child)
        d_right = _term_dict(x.r_child)

        for term, coeff in d_left.items():
            d[term] += coeff

        for term, coeff in d_right.items():
            d[term] += coeff

    if x.node_type == BCHNode.SCALAR:
        d_child = _term_dict(x.l_child)
        for term, coeff in d_child.items():
            d[term] = x.scalar * coeff

    return d


def _commutator_linearity(x: BCHTree) -> Tuple[BCHTree, bool]:
    if x.node_type != BCHNode.COMMUTATOR:
        return x, False

    l_type = x.l_child.node_type
    r_type = x.r_child.node_type

    if l_type == BCHNode.ADD:
        l_comm = BCHTree.commutator_node(x.l_child.l_child, x.r_child)
        r_comm = BCHTree.commutator_node(x.l_child.r_child, x.r_child)

        return l_comm + r_comm, True

    if r_type == BCHNode.ADD:
        l_comm = BCHTree.commutator_node(x.l_child, x.r_child.l_child)
        r_comm = BCHTree.commutator_node(x.l_child, x.r_child.r_child)

        return l_comm + r_comm, True

    return x, False


def _factor_scalar_from_commutator(x: BCHTree) -> Tuple[BCHTree, bool]:
    if x.node_type != BCHNode.COMMUTATOR:
        return x, False

    if x.l_child.node_type == BCHNode.SCALAR and x.r_child.node_type == BCHNode.SCALAR:
        return (
            x.l_child.scalar
            * x.r_child.scalar
            * BCHTree.commutator_node(x.l_child.l_child, x.r_child.l_child)
        ), True

    if x.l_child.node_type == BCHNode.SCALAR:
        return x.l_child.scalar * BCHTree.commutator_node(x.l_child.l_child, x.r_child), True

    if x.r_child.node_type == BCHNode.SCALAR:
        return x.r_child.scalar * BCHTree.commutator_node(x.l_child, x.r_child.l_child), True

    return x, False


def _distribute_scalar(x: BCHTree) -> Tuple[BCHTree, bool]:
    if x.node_type != BCHNode.SCALAR:
        return x, False

    if x.l_child.node_type == BCHNode.ADD:
        l_child = x.scalar * x.l_child.l_child
        r_child = x.scalar * x.l_child.r_child

        return l_child + r_child, True

    if x.l_child.node_type == BCHNode.SCALAR:
        return (x.scalar * x.l_child.scalar) * x.l_child.l_child, True

    if x.l_child.node_type == BCHNode.ZERO:
        return BCHTree.zero_node(), True

    return x, False


def _order_commutator_terms(x: BCHTree) -> Tuple[BCHTree, bool]:
    if x.node_type != BCHNode.COMMUTATOR:
        return x, False

    l_type = x.l_child.node_type
    r_type = x.r_child.node_type

    if l_type == BCHNode.FRAGMENT and r_type == BCHNode.FRAGMENT:
        if x.l_child.fragment <= x.r_child.fragment:
            return x, False

        return -BCHTree.commutator_node(x.r_child, x.l_child), True

    if l_type == BCHNode.COMMUTATOR and r_type == BCHNode.FRAGMENT:
        return -BCHTree.commutator_node(x.r_child, x.l_child), True

    return x, False


def _compute_commutator_order(x: BCHTree) -> int:
    l_type = x.l_child.node_type
    r_type = x.r_child.node_type

    if l_type == BCHNode.SCALAR:
        l_term = x.l_child.l_child
    else:
        l_term = x.l_child

    if r_type == BCHNode.SCALAR:
        r_term = x.r_child.l_child
    else:
        r_term = x.r_child

    if l_type == BCHNode.FRAGMENT and r_type == BCHNode.FRAGMENT:
        return 2

    if l_type == BCHNode.FRAGMENT and r_type == BCHNode.COMMUTATOR:
        return _compute_commutator_order(r_term) + 1

    if l_type == BCHNode.COMMUTATOR and r_type == BCHNode.FRAGMENT:
        return _compute_commutator_order(l_term) + 1

    if l_type == BCHNode.COMMUTATOR and r_type == BCHNode.COMMUTATOR:
        return _compute_commutator_order(l_term) + _compute_commutator_order(r_term)

    raise ValueError(
        f"Expected child nodes to be of type BCHNode.FRAGMENT, BCHNode.COMMUTATOR, or BCHNode.SCALAR. Got {l_type} and {r_type} instead."
    )


def _drop_high_order(x: BCHTree, max_order: int) -> Tuple[BCHTree, bool]:
    if x.node_type != BCHNode.COMMUTATOR:
        return x, False

    if x.l_child.node_type not in (BCHNode.COMMUTATOR, BCHNode.FRAGMENT):
        return x, False

    if x.r_child.node_type not in (BCHNode.COMMUTATOR, BCHNode.FRAGMENT):
        return x, False

    if _compute_commutator_order(x) > max_order:
        return BCHTree.zero_node(), True

    return x, False
