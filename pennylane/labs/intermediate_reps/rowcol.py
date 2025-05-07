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
"""CNOT routing algorithm ROWCOL as described in https://arxiv.org/abs/1910.14478."""

from typing import Iterable

import galois
import networkx as nx
import numpy as np
from networkx.algorithms.approximation import steiner_tree


def postorder_traverse(tree: nx.Graph, source: int, source_parent: int = None):
    """Post-order traverse a tree graph, starting from (but excluding) the node ``source``.

    Args:
        tree (nx.Graph): Tree graph to traverse. Must contain ``source``. Must contain
            ``source_parent`` if it is not None. Typing assumes integer-labeled nodes.
        source (int): Node to start the traversal from
        source_parent (Optional[int]): Parent node of ``source`` in ``tree``.

    Returns:
        list[tuple[int]]: Pairs of nodes that constitute post-order traversal of ``tree``
        starting at ``source``. Strictly speaking, the traversal is encoded in the first
        entry of each pair, and the second entry simply is the parent node for each first entry.

    A useful illustration of depth-first tree traversals can be found on
    `Wikipedia <https://en.wikipedia.org/wiki/Tree_traversal#Depth-first%20search>`__.
    """
    out = []
    # First, traverse the subtrees attached to the current node ``source``
    for child in tree.neighbors(source):
        # The graph does not know of the tree structure, so we need to make sure we skip
        # the parent of ``source`` among its neighbors.
        if child != source_parent:
            # Recurse
            out.extend(postorder_traverse(tree, child, source))
    # Second, attach the current node itself, together with its parent
    if source_parent is not None:
        out.append((source, source_parent))

    return out


def preorder_traverse(tree: nx.Graph, source: int, source_parent: int = None):
    """Pre-order traverse a tree graph, starting from (but excluding) the node ``source``.

    Args:
        tree (nx.Graph): Tree graph to traverse. Must contain ``source``. Must contain
            ``source_parent`` if it is not None. Typing assumes integer-labeled nodes.
        source (int): Node to start the traversal from
        source_parent (Optional[int]): Parent node of ``source`` in ``tree``.

    Returns:
        list[tuple[int]]: Pairs of nodes that constitute pre-order traversal of ``tree``
        starting at ``source``. Strictly speaking, the traversal is encoded in the first
        entry of each pair, and the second entry simply is the parent node for each first entry.

    A useful illustration of depth-first tree traversals can be found on
    `Wikipedia <https://en.wikipedia.org/wiki/Tree_traversal#Depth-first%20search>`__.
    """
    out = []

    # First, attach the current node itself, together with its parent
    if source_parent is not None:
        out.append((source, source_parent))
    # Second, traverse the subtrees attached to the current node ``source``
    for child in tree.neighbors(source):
        # The graph does not know of the tree structure, so we need to make sure we skip
        # the parent of ``source`` among its neighbors.
        if child != source_parent:
            # Recurse
            out.extend(preorder_traverse(tree, child, source))

    return out


def _update(P: np.ndarray, cnots: list[tuple[int]], control: int, target: int):
    """In-place apply update corresponding to a CNOT on wires ``(control, target)``
    to parity matrix P and list of CNOT gates ``cnots``."""
    P[target] += P[control]
    cnots.append((control, target))


F_2 = galois.GF(2)


def _get_S(P: np.ndarray, idx: int, node_set: Iterable[int], mode: str):
    # Find S (S') either by simply extracting a column or by solving a linear system for the row
    if mode == "column":
        b = P[:, idx]
    elif mode == "row":
        P = F_2(P)
        e_i = F_2.Zeros(len(P))
        e_i[idx] = 1
        b = np.linalg.solve(P.T, e_i)
    S = set(np.where(b)[0])
    # Add the node ``idx`` itself
    S.add(idx)
    # Manually remove nodes from S that are no longer part of ``connectivity``.
    S = S.intersection(node_set)
    return S


def _eliminate(P: np.ndarray, connectivity: nx.Graph, idx: int, mode: str, verbose: bool):
    """Eliminate the column or row with index ``idx`` of the parity matrix P,
    respecting the connectivity constraints given by ``connectivity``.

    Args:
        P (np.ndarray): Parity matrix
        connectivity (nx.Graph): Connectivity graph
        idx (int): Column or row index to eliminate
        mode (str): Whether to eliminate the column (``column``) or row (``row``) of ``P``.
        verbose (bool): Whether to print elimination results

    Returns:
        tuple[np.ndarray, list[tuple[int]]]: Updated parity matrix and list of CNOTs that
        accomplish the update, in terms of ``(control, target)`` qubit pairs.
    """

    # i.1.1/i.2.1 Construct S (mode="column") or S' (mode="row"), respectively.
    S = _get_S(P, idx, connectivity.nodes(), mode)
    if len(S) == 1:
        # idx is in S for sure, so this column/row already has the right content. No need to update
        if verbose:
            print(f"CNOTs from {mode} elimination ({idx}): {[]}")
        return P, []

    cnots = []
    # i.1.1/i.2.1 Find Steiner tree within S (S').
    T = steiner_tree(connectivity, list(S))

    # Need post-order nodes in any case
    visit_nodes = postorder_traverse(T, source=idx)

    if mode == "column":
        # For column mode use post-order and parities constraint in first pass (i.1.2)...
        _ = [
            _update(P, cnots, child, parent)
            for child, parent in visit_nodes
            if P[child, idx] == 1 and P[parent, idx] == 0
        ]
        # ... and no constraints for second pass (i.1.3)
        _ = [_update(P, cnots, parent, child) for child, parent in visit_nodes]

    elif mode == "row":
        # For row mode use pre-order and constraints from S' in first pass (i.2.2)...
        previsit_nodes = preorder_traverse(T, source=idx)
        _ = [_update(P, cnots, child, parent) for child, parent in previsit_nodes if child not in S]
        # ... and no constraints for second pass (i.2.3)
        _ = [_update(P, cnots, child, parent) for child, parent in visit_nodes]

    if verbose:
        print(f"CNOTs from {mode} elimination ({idx}): {cnots}")
    return P % 2, cnots


def rowcol(P: np.ndarray, connectivity: nx.Graph = None, verbose: bool = False) -> list[tuple[int]]:
    """CNOT routing algorithm ROWCOL.

    This algorithm was introduced by `Wu et al. <https://arxiv.org/abs/1910.14478>`__ and is
    detailed in the `compilation hub <https://pennylane.ai/compilation/rowcol-algorithm>`__,
    where examples can be found as well.

    Args:
        P (np.ndarray): Parity matrix to implement. Will not be altered
        connectivity (nx.Graph): Connectivity graph to route into. If not given,
            full connectivity is assumed. May be altered by this function
        verbose (bool): Whether or not to print progress of obtained CNOT gates.

    Returns:
        list[tuple[int]]: Wire pairs for CNOTs that implement the parity matrix (control first,
        target second).

    """
    P = P.copy()
    n = len(P)
    # If no connectivity is given, assume full connectivity
    if connectivity is None:
        connectivity = nx.complete_graph(n)
        cut_vertices = set()
    else:
        cut_vertices = set(nx.articulation_points(connectivity))

    cnots = []
    while connectivity.number_of_nodes() > 1:
        # Pick a vertex that is not a cut vertex of the (remaining) connectivity graph
        i = next(v for v in connectivity.nodes() if v not in cut_vertices)
        # Eliminate column and row of parity matrix with index v (Steps i.1 and i.2 in compilation hub)
        for mode in ["column", "row"]:
            P, new_cnots = _eliminate(P, connectivity, i, mode, verbose)
            # Memorize CNOTs required to eliminate column/row
            cnots.extend(new_cnots)
        # Remove vertex i
        connectivity.remove_nodes_from([i])

        # Recompute cut vertices
        cut_vertices = set(nx.articulation_points(connectivity))

    # Assert that the parity matrix was transformed into the identity matrix
    assert np.allclose(np.eye(n), P)
    # Return CNOTs in reverse order
    return P, cnots[::-1]
