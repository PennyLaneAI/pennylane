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


#
# tests
#


def test_merge():
    A = LeafNode({"A"})
    B = LeafNode({"B"})
    C = LeafNode({"C"})
    D = LeafNode({"D"})
    comm1 = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, B), D)
    )

    comm2 = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), D)
    )
    sum_comm = merge(comm1, comm2, 4)

    merged_node = LeafNode({"B", "D"})
    expected = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, merged_node), D)
    )
    assert sum_comm == expected


@pytest.mark.parametrize("value", ["A", 12, [1, 2, 3], {"X", "Y", 1.2}])
def test_mergeability(value):
    A = LeafNode(value)
    B = LeafNode("B")
    C = LeafNode("C")
    D = LeafNode("D")
    comm1 = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, B), D)
    )

    comm2 = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), D)
    )

    comm3 = CommutatorNode(
        CommutatorNode(CommutatorNode(A, B), C), CommutatorNode(CommutatorNode(A, D), D)
    )

    assert is_mergeable(comm1, comm2, 0) == False
    assert is_mergeable(comm1, comm2, 4) == True
    assert is_mergeable(comm1, comm3, 0) == False


def test_replacement():
    A = LeafNode([1, 2])
    B = LeafNode("B")
    C = LeafNode("C")
    D = LeafNode("D")
    E = LeafNode("E")
    X = LeafNode("X")
    Y = LeafNode("Y")
    big_comm = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), E)
    )

    replacements = [CommutatorNode(X, Y), E]
    big_comm.replace_node(A, replacements)
    assert str(big_comm) == "[[[X, Y], [B, C]], [[E, D], E]]"


def test_replacement_too_many():
    A = LeafNode("A")
    B = LeafNode("B")
    C = LeafNode("C")
    D = LeafNode("D")
    E = LeafNode("E")
    X = LeafNode("X")
    Y = LeafNode("Y")
    big_comm = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), E)
    )

    replacements = [CommutatorNode(X, Y), E, X]
    with pytest.raises(RuntimeError, match="Got more replacement nodes"):
        big_comm.replace_node(A, replacements)


def test_replacement_too_few():
    A = LeafNode("A")
    B = LeafNode("B")
    C = LeafNode("C")
    D = LeafNode("D")
    E = LeafNode("E")
    X = LeafNode("X")
    Y = LeafNode("Y")
    big_comm = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), E)
    )

    replacements = [CommutatorNode(X, Y)]
    with pytest.raises(RuntimeError, match="Got fewer replacement nodes"):
        big_comm.replace_node(A, replacements)
