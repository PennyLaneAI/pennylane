import copy
import typing
from typing import Hashable, List

import pytest


class Node:
    """Abstract base class for all nodes in the commutator tree."""

    pass


class LeafNode(Node):
    """
    Represents a leaf node in the commutator tree, holding a base value.

    The value can be any hashable object.
    The concrete value is of no concern to the commutator tree structure class.
    """

    def __init__(self, value: Hashable):
        self.value = copy.deepcopy(value)
        self.order = 1

    def __eq__(self, other):
        if not isinstance(other, LeafNode):
            return False
        return self.value == other.value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def __getitem__(self, i):
        assert i == 0
        return self.value

    def __setitem__(self, i, v):
        assert i == 0
        self.value = copy.deepcopy(v)

    def expand(self) -> dict:
        """
        Expands the leaf node into a dictionary representing a linear combination.
        For a leaf 'A', the expansion is {('A',): 1}.
        """
        return {(str(self.value),): 1}


class CommutatorNode(Node):
    """
    Represents an internal commutator node [left, right] in the tree.

    This node contains the method to search for and replace child nodes.
    """

    def __init__(self, left: Node, right: Node):
        if not isinstance(left, Node) or not isinstance(right, Node):
            raise TypeError("Both left and right children must be Node instances.")
        self.left = copy.deepcopy(left)
        self.right = copy.deepcopy(right)
        self.order = self.left.order + self.right.order

    def __str__(self):
        return f"[{self.left}, {self.right}]"

    def __repr__(self):
        return f"[{self.left}, {self.right}]"

    def __eq__(self, other):
        if not isinstance(other, CommutatorNode):
            return False
        return self.left == other.left and self.right == other.right

    def __getitem__(self, i):
        if i >= self.order or i < 0:
            raise RuntimeError("Index out of range.")
        if i < self.left.order:
            return self.left[i]
        else:
            return self.right[i - self.left.order]

    def __setitem__(self, i, v):
        if i >= self.order or i < 0:
            raise RuntimeError("Index out of range.")
        if i < self.left.order:
            self.left[i] = v
        else:
            self.right[i - self.left.order] = v

    def _replace_node_impl(self, target_node: Node, new_nodes: List[Node]):
        """
        Recursively finds a target child node and replaces it.
        This method traverses the tree starting from the current node's children.

        Args:
            target_node: The specific node object to be replaced.
            new_node: The new node object to insert in its place.
        """

        if not new_nodes:
            raise RuntimeError(
                "Got fewer replacement nodes than the number of targte nodes in the commutator tree."
            )

        if self.left == target_node:
            new_node = new_nodes.pop(0)
            if not isinstance(new_node, Node):
                raise TypeError("The replacement node must be a Node instance.")
            self.left = new_node

        if self.right == target_node:
            new_node = new_nodes.pop(0)
            if not isinstance(new_node, Node):
                raise TypeError("The replacement node must be a Node instance.")
            self.right = new_node

        # If not a direct child, recurse into children that are CommutatorNodes
        if isinstance(self.left, CommutatorNode):
            self.left._replace_node_impl(target_node, new_nodes)

        if isinstance(self.right, CommutatorNode):
            self.right._replace_node_impl(target_node, new_nodes)

    def replace_node(self, target_node: Node, new_nodes: List[Node]):
        """
        Replace target_node in the tree with the new_nodes.
        Each next occurence of target_node will be replaced with the next node in new_nodes.

        Args:
            target_node: The specific node object to be replaced.
            new_node: The new node object to insert in its place.
        """
        self._replace_node_impl(target_node, new_nodes)
        if len(new_nodes) > 0:
            raise RuntimeError(
                "Got more replacement nodes than the number of targte nodes in the commutator tree."
            )

    def expand(self) -> dict:
        """
        Recursively expands the commutator [L, R] using the identity LR - RL.

        Returns:
            A dictionary where keys are tuples of symbols (representing a product)
            and values are their integer coefficients.
        """
        left_expansion = self.left.expand()
        right_expansion = self.right.expand()

        result = {}

        # Calculate the +LR term
        for l_prod, l_coeff in left_expansion.items():
            for r_prod, r_coeff in right_expansion.items():
                prod = l_prod + r_prod
                coeff = l_coeff * r_coeff
                result[prod] = result.get(prod, 0) + coeff

        # Calculate the -RL term
        for r_prod, r_coeff in right_expansion.items():
            for l_prod, l_coeff in left_expansion.items():
                prod = r_prod + l_prod
                coeff = l_coeff * r_coeff
                result[prod] = result.get(prod, 0) - coeff

        # Remove terms with a zero coefficient for a cleaner output
        return {prod: coeff for prod, coeff in result.items() if coeff != 0}


def is_mergeable(node1: Node, node2: Node, k: int):
    """
    Check if two nodes are mergeable at index k.

    The index of a symbol in a (possibly nested) commutator is the index of the symbol in the
    commutator's nested representation.
    For example in [[A, B], [C, [D, E]]], C is the component at k=2.

    Two commutators are mergable at index k if they have the same nesting structure, and their
    symbols only differ at index k.
    """
    if not isinstance(node1, Node) or not isinstance(node2, Node):
        raise TypeError("Only Node instances can be checked for mergeability.")

    if type(node1) is not type(node2):
        # different tree structure, not mergeable
        return False

    if isinstance(node1, LeafNode):
        return k == 0

    if node1.left.order != node2.left.order or node1.right.order != node2.right.order:
        return False

    if k < node1.left.order:
        # k on the left
        if node1.right != node2.right:
            return False
        return is_mergeable(node1.left, node2.left, k)
    else:
        # k on the right
        if node1.left != node2.left:
            return False
        return is_mergeable(node1.right, node2.right, k - node1.left.order)


def merge(node1: Node, node2: Node, k: int):
    """
    Merge two nodes at index k.
    """

    if not is_mergeable(node1, node2, k):
        raise RuntimeError(f"The two commutators are not mergeable at index {k}.")

    sum_comm = copy.deepcopy(node1)
    if not all(isinstance(v, set) for v in [node1[k], node2[k]]):
        raise RuntimeError("Merging commutators can only take sets at the merged position.")
    sum_comm[k] = node1[k] | node2[k]
    return sum_comm
