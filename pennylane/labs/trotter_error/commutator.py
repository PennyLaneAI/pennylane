import typing
from typing import List

import pytest


class Node:
    """Abstract base class for all nodes in the commutator tree."""

    pass


class LeafNode(Node):
    """
    Represents a leaf node in the commutator tree, holding a base value.

    Think of this as the 'A', 'B', or 'C' in an expression like [[A, B], C].
    In other words, this is a symbol.
    """

    def __init__(self, value: any):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class CommutatorNode(Node):
    """
    Represents an internal commutator node [left, right] in the tree.

    This node contains the method to search for and replace child nodes.
    """

    def __init__(self, left: Node, right: Node):
        if not isinstance(left, Node) or not isinstance(right, Node):
            raise TypeError("Both left and right children must be Node instances.")
        self.left = left
        self.right = right

    def __str__(self):
        return f"[{self.left}, {self.right}]"

    def __repr__(self):
        return f"[{self.left}, {self.right}]"

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

        if self.left is target_node:
            new_node = new_nodes.pop(0)
            if not isinstance(new_node, Node):
                raise TypeError("The replacement node must be a Node instance.")
            self.left = new_node

        if self.right is target_node:
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


def is_symbol(char):
    non_symbol = ["[", "]", " ", ","]
    return char not in non_symbol


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

    string1 = str(node1)  # .replace("[", "").replace("]", "").replace(",", "").replace(" ", "")
    string2 = str(node2)  # .replace("[", "").replace("]", "").replace(",", "").replace(" ", "")

    if len(string1) != len(string2):
        return False

    symbol_count = 0
    for i, (char1, char2) in enumerate(zip(string1, string2)):
        if not is_symbol(char1) and char1 != char2:
            return False
        else:
            if is_symbol(char1):
                if not is_symbol(char2):
                    return False
                symbol_count += 1
            if symbol_count - 1 != k:
                if char1 != char2:
                    return False
    return True


def test_mergeability():
    A = LeafNode("A")
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
