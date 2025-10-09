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
"""AST for commutator expressions"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Hashable
from itertools import chain, product
from typing import Dict, Generator, List, Sequence, Tuple

import numpy as np


class Node:
    """Abstract base class for all nodes in the commutator tree."""


class SymbolNode(Node):
    """
    Represents a leaf node in the commutator tree, holding a base value.

    The value can be any hashable object.
    The concrete value is of no concern to the commutator tree structure class.
    """

    def __init__(
        self, symbols: Hashable | Sequence[Hashable], coeffs: complex | Sequence[complex] = 1.0
    ):
        if not isinstance(symbols, Sequence):
            symbols = (symbols,)
        if not isinstance(coeffs, Sequence):
            coeffs = (coeffs,)

        if len(symbols) != len(coeffs):
            raise ValueError("Number of symbols must match number of coefficients")

        tmp_symbol_set = set()
        for symbol, coeff in zip(symbols, coeffs):
            tmp_symbol_set.add((symbol, coeff))

        self.symbols = frozenset(tmp_symbol_set)

    def __eq__(self, other):
        if not isinstance(other, SymbolNode):
            return False
        return self.symbols == other.symbols

    def __hash__(self):
        return hash(self.symbols)

    def __str__(self):
        strings = [
            str(symbol) if np.isclose(coeff, 1.0) else f"{coeff}*{symbol}"
            for (symbol, coeff) in self.symbols
        ]

        return " + ".join(strings)

    def __repr__(self):
        return str(self)

    def __getitem__(self, i):
        assert i == 0
        return self.symbols

    def __setitem__(self, i, v):
        assert i == 0
        self.symbols = copy.deepcopy(v)

    @property
    def order(self) -> int:
        return 1

    def expand(self) -> Dict[Tuple[SymbolNode], int]:
        """
        Expands the leaf node into a dictionary representing a linear combination.
        For a leaf 'A', the expansion is {('A',): 1}.
        """
        return {(self,): 1}


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

    def expand(self) -> Dict[Tuple[SymbolNode], int]:
        """
        Recursively expands the commutator [L, R] using the identity LR - RL.

        Returns:
            A dictionary where keys are tuples of symbols (representing a product)
            and values are their integer coefficients.
        """
        left_expansion = self.left.expand()
        right_expansion = self.right.expand()

        result = defaultdict(int)

        for l_prod, l_coeff in left_expansion.items():
            for r_prod, r_coeff in right_expansion.items():
                prod = l_prod + r_prod
                coeff = l_coeff * r_coeff
                result[prod] += coeff

        for r_prod, r_coeff in right_expansion.items():
            for l_prod, l_coeff in left_expansion.items():
                prod = r_prod + l_prod
                coeff = l_coeff * r_coeff
                result[prod] -= coeff

        return {prod: coeff for prod, coeff in result.items() if coeff != 0}

    def is_zero(self) -> bool:
        if isinstance(self.left, SymbolNode) and isinstance(self.right, SymbolNode):
            return self.left


def find_leaves(node: Node) -> List[SymbolNode]:
    """Recursively finds all SymbolNode instances in a tree."""
    if isinstance(node, SymbolNode):
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
    terms: Dict[SymbolNode, Dict[Node, complex]],
    max_order: int,
    bch_coeff: complex,
) -> Dict[Node, complex]:
    r"""
    Substitutes BCH expansions into a commutator tree structure.

    This version assumes all leaf nodes in the template are placeholders
    that have a corresponding expansion in the `terms` dictionary.
    """

    expanded = defaultdict(complex)
    target_leaves = find_leaves(commutator_node)
    num_leaves = len(target_leaves)

    terms_by_order = {}
    for key, value in terms.items():
        terms_by_order[key] = _separate_by_order(value, max_order)

    for final_order in range(1, max_order + 1):
        for partition in _partitions_positive(num_leaves, final_order):
            try:
                kept_terms_for_product = [
                    terms_by_order[leaf][partition[j] - 1].items()
                    for j, leaf in enumerate(target_leaves)
                ]
            except IndexError:
                continue

            for combo in product(*kept_terms_for_product):
                term_coeff = math.prod(item[1] for item in combo) * bch_coeff

                target_map = defaultdict(list)
                for target, (replacement_node, _) in zip(target_leaves, combo):
                    target_map[target].append(replacement_node)

                new_node = copy.deepcopy(commutator_node)
                for target_node, replacement_nodes in target_map.items():
                    new_node = replace_node(new_node, target_node, replacement_nodes)

                if new_node.order <= max_order:
                    expanded[new_node] += term_coeff

    return expanded


def _separate_by_order(d: Dict[Node, complex], max_order: int) -> List[Dict[Node, complex]]:
    ret = [{} for _ in range(max_order)]
    for key, value in d.items():
        ret[key.order - 1][key] = value

    return ret


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
        return False

    if isinstance(node1, SymbolNode):
        return k == 0

    if node1.left.order != node2.left.order or node1.right.order != node2.right.order:
        return False

    if k < node1.left.order:
        if node1.right != node2.right:
            return False
        return is_mergeable(node1.left, node2.left, k)

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

    left_node = node1[k]
    right_node = node2[k]

    tmp_dict = defaultdict(complex)
    for symbol, coeff in chain(left_node, right_node):
        tmp_dict[symbol] += coeff

    sum_comm[k] = frozenset(tmp_dict.items())

    return sum_comm


def is_tree_isomorphic(node1: Node, node2: Node) -> bool:
    """
    Recursively checks if two nodes have the same bracketing structure (tree isomorphism).
    """
    if type(node1) is not type(node2):
        return False

    if isinstance(node1, SymbolNode):
        return True

    if isinstance(node1, CommutatorNode) and isinstance(node2, CommutatorNode):
        return is_tree_isomorphic(node1.left, node2.left) and is_tree_isomorphic(
            node1.right, node2.right
        )

    return False


def _replace_node_impl(source_node: Node, target_node: Node, new_nodes: List[Node]):
    """
    Recursively finds a target child node and replaces it.
    This method traverses the tree starting from the current node's children.
    """
    if isinstance(source_node, SymbolNode):
        return

    # Check and replace left child
    if source_node.left == target_node:
        if not new_nodes:
            raise RuntimeError(
                "Got fewer replacement nodes than the number of target nodes in the commutator tree."
            )
        source_node.left = copy.deepcopy(new_nodes.pop(0))

    elif isinstance(source_node.left, CommutatorNode):
        _replace_node_impl(source_node.left, target_node, new_nodes)

    # Check and replace right child
    if source_node.right == target_node:
        if not new_nodes:
            raise RuntimeError(
                "Got fewer replacement nodes than the number of target nodes in the commutator tree."
            )
        source_node.right = copy.deepcopy(new_nodes.pop(0))

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

    if isinstance(source_node, SymbolNode):
        if len(new_nodes) != 1:
            raise RuntimeError("`new_nodes` must be length 1 when `source_node` is a SymbolNode.")

        return copy.deepcopy(new_nodes[0])

    result = copy.deepcopy(source_node)
    replacements = copy.deepcopy(new_nodes)
    _replace_node_impl(result, target_node, replacements)
    if len(replacements) > 0:
        raise RuntimeError(
            "Got more replacement nodes than the number of target nodes in the commutator tree."
        )
    return result
