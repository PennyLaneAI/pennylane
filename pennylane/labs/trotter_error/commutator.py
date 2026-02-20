import copy
import math
import typing
from collections import defaultdict
from itertools import product
from typing import Dict, Generator, Hashable, List, Set


class Node:
    """Abstract base class for all nodes in the commutator tree."""

    def __hash__(self):
        return hash(str(self))


class LeafNode(Node):
    """
    Represents a leaf node in the commutator tree, holding a base value.

    The value can be any hashable object.
    The concrete value is of no concern to the commutator tree structure class.
    """

    def __init__(self, value: Hashable):
        # Convert set to frozenset for hashability
        if isinstance(value, set):
            value = frozenset(value)

        if not isinstance(value, Hashable):
            raise TypeError(f"LeafNode value must be hashable, but got {type(value)}")
        self.value = copy.deepcopy(value)

    def __eq__(self, other):
        if not isinstance(other, LeafNode):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        if isinstance(self.value, frozenset):
            # To ensure a canonical string representation for tests,
            # we sort the elements before printing.
            try:
                # This works if all elements are sortable (e.g., all ints)
                sorted_elements = sorted(list(self.value))
                return f"{{{', '.join(map(str, sorted_elements))}}}"
            except TypeError:
                # Fallback if elements are not sortable (e.g., mixed types)
                return str(set(self.value))
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def __getitem__(self, i):
        assert i == 0
        return self.value

    def __setitem__(self, i, v):
        assert i == 0
        self.value = copy.deepcopy(v)

    @property
    def order(self) -> int:
        return 1

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

    def __str__(self):
        return f"[{self.left}, {self.right}]"

    def __repr__(self):
        return f"[{self.left}, {self.right}]"

    def __eq__(self, other):
        if not isinstance(other, CommutatorNode):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash((self.left, self.right))

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

    @property
    def order(self) -> int:
        return self.left.order + self.right.order

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


def _replace_node_impl(source_node: Node, target_node: Node, new_nodes: List[Node]):
    """
    Recursively finds a target child node and replaces it.
    This method traverses the tree starting from the current node's children.
    """
    if isinstance(source_node, LeafNode):
        return

    # Check and replace left child
    if source_node.left == target_node:
        if not new_nodes:
            raise RuntimeError(
                "Got fewer replacement nodes than the number of target nodes in the commutator tree."
            )
        source_node.left = new_nodes.pop(0)

    elif isinstance(source_node.left, CommutatorNode):
        _replace_node_impl(source_node.left, target_node, new_nodes)

    # Check and replace right child
    if source_node.right == target_node:
        if not new_nodes:
            raise RuntimeError(
                "Got fewer replacement nodes than the number of target nodes in the commutator tree."
            )
        source_node.right = new_nodes.pop(0)

    elif isinstance(source_node.right, CommutatorNode):
        _replace_node_impl(source_node.right, target_node, new_nodes)


def replace_node(source_node: Node, target_node: Node, new_nodes: List[Node]):
    """
    Replace target_node in the tree with the new_nodes.
    Each next occurence of target_node will be replaced with the next node in new_nodes.

    Args:
        source_node: The original source node. The method leaves the original source node unchanged.
        target_node: The specific node object to be replaced.
        new_node: The new node object to insert in its place.

    Returns:
        A Node object with replacements.
    """
    result = copy.deepcopy(source_node)
    # Use a copy of the list so the original is not modified
    replacements = list(new_nodes)
    _replace_node_impl(result, target_node, replacements)
    if len(replacements) > 0:
        raise RuntimeError(
            "Got more replacement nodes than the number of target nodes in the commutator tree."
        )
    return result


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
    if not all(isinstance(v, (set, frozenset)) for v in [node1[k], node2[k]]):
        raise RuntimeError(
            "Merging commutators can only take sets or frozensets at the merged position."
        )
    sum_comm[k] = node1[k] | node2[k]
    return sum_comm


### BCH recursive merge


def find_leaves(node: Node) -> List[LeafNode]:
    """Recursively finds all LeafNode instances in a tree."""
    if isinstance(node, LeafNode):
        return [node]
    if isinstance(node, CommutatorNode):
        return find_leaves(node.left) + find_leaves(node.right)
    return []


def _partitions_positive(m: int, n: int) -> Generator[tuple[int], None, None]:
    """Yields tuples of m positive integers that sum to n."""
    if m <= 0 or n <= 0:
        return
    if m == 1:
        if n > 0:
            yield (n,)
        return
    if n < m:
        return
    for i in range(1, n - m + 2):
        for partition in _partitions_positive(m - 1, n - i):
            yield (i,) + partition


def bilinear_expansion(
    commutator_node: Node,
    terms: Dict[LeafNode, List[Dict[Node, complex]]],
    max_order: int,
    bch_coeff: complex,
) -> List[Dict[Node, complex]]:
    r"""
    Performs bilinear expansion of a commutator template.

    This version assumes all leaf nodes in the template are placeholders
    that have a corresponding expansion in the `terms` dictionary.
    """
    merged_bch = [defaultdict(complex) for _ in range(max_order)]

    target_leaves = find_leaves(commutator_node)
    num_leaves = len(target_leaves)

    for final_order in range(1, max_order + 1):
        for partition in _partitions_positive(num_leaves, final_order):
            try:
                kept_terms_for_product = [
                    terms[leaf][partition[j] - 1].items() for j, leaf in enumerate(target_leaves)
                ]
            except IndexError:
                # This partition is not possible with the given expansions, so we skip it.
                continue

            for combo in product(*kept_terms_for_product):
                term_coeff = math.prod(item[1] for item in combo) * bch_coeff

                # Group replacements by the target leaf they correspond to.
                # This correctly handles templates with repeated placeholders like [X, X].
                target_map = defaultdict(list)
                for target, (replacement_node, _) in zip(target_leaves, combo):
                    target_map[target].append(replacement_node)

                # Iteratively apply the replacements for each unique target leaf.
                new_node = copy.deepcopy(commutator_node)
                for target_node, replacement_nodes in target_map.items():
                    new_node = replace_node(new_node, target_node, replacement_nodes)

                if new_node.order <= max_order:
                    merged_bch[new_node.order - 1][new_node] += term_coeff

    return merged_bch


def is_tree_isomorphic(node1: Node, node2: Node) -> bool:
    """
    Recursively checks if two nodes have the same bracketing structure (tree isomorphism).
    """
    if type(node1) is not type(node2):
        return False

    if isinstance(node1, LeafNode):
        return True

    if isinstance(node1, CommutatorNode) and isinstance(node2, CommutatorNode):
        return is_tree_isomorphic(node1.left, node2.left) and is_tree_isomorphic(
            node1.right, node2.right
        )

    return False


class CommutatorsPartitioner:
    """
    Maintains a partition of commutators based on their bracketing structure.

    This class groups commutators into sets, where each set represents an
    isomorphism class. All commutators within a single set have the same
    tree structure.
    """

    def __init__(self):
        self.partitions: List[Set[Node]] = []

    def __str__(self):
        if not self.partitions:
            return "CommutatorsPartitioner(empty)"

        partition_strs = []
        for i, iso_class in enumerate(self.partitions):
            class_str = f"  Class {i}: " + ", ".join(str(node) for node in iso_class)
            partition_strs.append(class_str)
        return "CommutatorsPartitioner:\n" + "\n".join(partition_strs)

    def __repr__(self):
        return self.__str__()

    def add_to_partition(self, commutator: Node):
        """
        Adds a commutator to the appropriate isomorphism class.

        It finds the set of commutators with the same bracketing structure and adds
        the new commutator to it. If no such set exists, a new one is created.
        """
        for iso_class in self.partitions:
            if iso_class:
                representative = next(iter(iso_class))
                if is_tree_isomorphic(commutator, representative):
                    iso_class.add(commutator)
                    return

        # No matching class was found, create a new isomorphism class containing this commutator.
        self.partitions.append({commutator})
