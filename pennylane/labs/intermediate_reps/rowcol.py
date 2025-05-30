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
        source_parent (Optional[int]): Parent node of ``source`` in ``tree``. Should not provided
            manually but is used in recursion.

    Returns:
        list[tuple[int]]: Pairs of nodes that constitute post-order traversal of ``tree``
        starting at ``source``. Strictly speaking, the traversal is encoded in the first
        entry of each pair, and the second entry simply is the parent node for each first entry.

    A useful illustration of depth-first tree traversals can be found on
    `Wikipedia <https://en.wikipedia.org/wiki/Tree_traversal#Depth-first%20search>`__.

    **Example**

    Consider the tree

    .. code-block:: python

                          (4)
                           |
        (6) - (2) - (0) - (1) - (3) - (8)
               |           |
              (7)         (5)

    and consider ``(0)`` to be the source, or root, of the tree.
    We may construct this tree as a ``nx.Graph`` by providing the edge data:

    >>> import networkx as nx
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 8)])

    As for every tree traversal, post-order traversal results in a reordering of the nodes of
    the tree, with each node appearing exactly once. Post-order traversing the graph means that
    for every node we reach, we first visit its child nodes (in standard sorting of the node
    labels/indices) and then append the node itself to the ordering.

    The post-order traversal reads ``[8, 3, 4, 5, 1, 6, 7, 2, 0]``.
    For the output convention of this function, each node is accompanied by its
    parent node, because this is useful information needed down the line, and it is not easy to
    retrieve from the ``nx.Graph`` itself. In addition, the last entry, which always is the root
    of the tree provided via the ``source`` argument, is *not* included in the output.

    >>> from pennylane.labs.intermediate_reps import postorder_traverse
    >>> traversal = postorder_traverse(G, 0)
    >>> print(traversal)
    [(8, 3), (3, 1), (4, 1), (5, 1), (1, 0), (6, 2), (7, 2), (2, 0)]
    >>> expected = [8, 3, 4, 5, 1, 6, 7, 2] # Skipping trailing root
    >>> all(child == exp for (child, parent), exp in zip(traversal, expected, strict=True))
    True

      To see how this comes about, start at the root ``(0)`` and perform the following steps:

      #. Visit ``(1)``, ``(3)`` and then ``(8)``, without appending any of them.
      #. Append ``(8)`` because there are no child nodes to visit (it is a leaf node).
      #. Append ``(3)`` because all children of it have been visited.
      #. Visit the next child of ``(1)``, which is ``(4)``, and append it because it is a leaf node.
      #. Visit the last child of ``(1)``, which is ``(5)``, and append it because it is a leaf node.
      #. Append ``(1)`` because all its children have been visited.
      #. Visit ``(2)`` and then ``(6)``, the latter of which is appended (leaf node).
      #. Visit the second and last child of ``(2))``, which is ``(7)``, and append it (leaf node).
      #. Append ``(2)`` becaues all its children have been visited.
      #. Append ``(0)`` becaues all its children have been visited.

    Note that the ``source_parent`` argument should not be provided when calling the function
    but is used for the internal recursive structure.
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
        source_parent (Optional[int]): Parent node of ``source`` in ``tree``. Should not provided
            manually but is used in recursion.

    Returns:
        list[tuple[int]]: Pairs of nodes that constitute pre-order traversal of ``tree``
        starting at ``source``. Strictly speaking, the traversal is encoded in the first
        entry of each pair, and the second entry simply is the parent node for each first entry.

    A useful illustration of depth-first tree traversals can be found on
    `Wikipedia <https://en.wikipedia.org/wiki/Tree_traversal#Depth-first%20search>`__.

    **Example**

    Consider the tree

    .. code-block:: python

                          (4)
                           |
        (6) - (2) - (0) - (1) - (3) - (8)
               |           |
              (7)         (5)

    and consider ``(0)`` to be the source, or root, of the tree.
    We may construct this tree as a ``nx.Graph`` by providing the edge data:

    >>> import networkx as nx
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 8)])

    As for every tree traversal, pre-order traversal results in a reordering of the nodes of
    the tree, with each node appearing exactly once. Pre-order traversing the graph means that
    for every node we reach, we first append the node itself to the ordering and then visit its
    child nodes (in standard sorting of the node labels/indices).

    The pre-order traversal reads ``[0, 1, 3, 8, 4, 5, 2, 6, 7]``.
    For the output convention of this function, each node is accompanied by its
    parent node, because this is useful information needed down the line, and it is not easy to
    retrieve from the ``nx.Graph`` itself. In addition, the first entry, which always is the root
    of the tree provided via the ``source`` argument, is *not* included in the output.

    >>> from pennylane.labs.intermediate_reps import preorder_traverse
    >>> traversal = preorder_traverse(G, 0)
    >>> print(traversal)
    [(1, 0), (3, 1), (8, 3), (4, 1), (5, 1), (2, 0), (6, 2), (7, 2)]
    >>> expected = [1, 3, 8, 4, 5, 2, 6, 7] # Skipping leading root
    >>> all(child == exp for (child, parent), exp in zip(traversal, expected, strict=True))
    True

      To see how this comes about, start at the root ``(0)`` and perform the following steps:

      #. Append ``(0)``, the root.
      #. Visit the first child of ``(0)``, which is ``(1)``, and append it.
      #. Visit the first child of ``(1)``, which is ``(3)``, and append it.
      #. Visit the first and only child of ``(3)``, which is ``(8)``, and append it.
      #. Visit the next child of ``(1)``, which is ``(4)``, and append it.
      #. Visit the last child of ``(1)``, which is ``(5)``, and append it.
      #. Visit the second and last child of ``(0)``, which is ``(2)``, and append it.
      #. Visit the first child of ``(2)``, which is ``(6)``, and append it.
      #. Visit the second and last child of ``(2)``, which is ``(7)``, and append it.

    Note that the ``source_parent`` argument should not be provided when calling the function
    but is used for the internal recursive structure.
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
    return P, cnots


# Create the Galois number field F_2, in which we can create arrays in _get_S.
F_2 = galois.GF(2)


def _get_S(P: np.ndarray, idx: int, node_set: Iterable[int], mode: str):
    # Find S (S') either by simply extracting a column or by solving a linear system for the row
    if mode == "column":
        b = P[:, idx]
    else:
        P = F_2(P)
        e_i = F_2.Zeros(len(P))
        e_i[idx] = 1
        b = np.linalg.solve(P.T, e_i)  # This solve step is over F_2!
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

    # For some reason, Pylint does not understand this usage of the Walrus operator.
    state = (P, cnots)  # pylint: disable=unused-variable
    if mode == "column":
        # For column mode use post-order and parities constraint in first pass (i.1.2)...
        _ = [
            state := _update(*state, child, parent)
            for child, parent in visit_nodes
            if P[child, idx] == 1 and P[parent, idx] == 0
        ]
        # ... and no constraints for second pass (i.1.3)
        _ = [state := _update(*state, parent, child) for child, parent in visit_nodes]

    else:
        # For row mode use pre-order and constraints from S' in first pass (i.2.2)...
        previsit_nodes = preorder_traverse(T, source=idx)
        _ = [
            state := _update(*state, child, parent)
            for child, parent in previsit_nodes
            if child not in S
        ]
        # ... and no constraints for second pass (i.2.3)
        _ = [state := _update(*state, child, parent) for child, parent in visit_nodes]

    if verbose:
        print(f"CNOTs from {mode} elimination ({idx}): {cnots}")
    return P % 2, cnots


def rowcol(P: np.ndarray, connectivity: nx.Graph = None, verbose: bool = False) -> list[tuple[int]]:
    r"""CNOT routing algorithm ROWCOL.

    This algorithm was introduced by `Wu et al. <https://arxiv.org/abs/1910.14478>`__ and is
    detailed in the `compilation hub <https://pennylane.ai/compilation/rowcol-algorithm>`__,
    where examples can be found as well.

    Args:
        P (np.ndarray): Parity matrix to implement. Will not be altered
        connectivity (nx.Graph): Connectivity graph to route into. If ``None`` (the default),
            full connectivity is assumed.
        verbose (bool): Whether or not to print progress of obtained CNOT gates. Default is ``False``.

    Returns:
        list[tuple[int]]: Wire pairs for CNOTs that implement the parity matrix (control first,
        target second).

    **Example**

    Here we compute the example 1 from `Wu et al. <https://arxiv.org/abs/1910.14478>`__ in code.
    We also compute it by hand in the manual example section below.

    To start, we have the connectivity graph

    .. code-block:: python

        (0) - (3) - (4)
               |
              (2)
               |
              (1)

    and define it in code as a ``networkx.Graph``
    (`networkx documentation <https://networkx.org/documentation/stable/index.html>`__).

    >>> import networkx as nx
    >>> G = nx.Graph([(0, 3), (1, 2), (2, 3), (3, 4)])

    We would like to find a CNOT circuit implementing the following parity matrix ``P``:

    >>> P = np.array([
    ...     [1, 1, 0, 1, 1],
    ...     [0, 0, 1, 1, 0],
    ...     [1, 0, 1, 0 ,1],
    ...     [1, 1, 0, 1, 0],
    ...     [1, 1, 1, 1, 0],
    ... ])

    We import ``rowcol`` and the function ``parity_matrix`` that will allow us to
    check that the computed circuit produces the input parity matrix.
    Then we run the algorithm:

    >>> from pennylane.labs.intermediate_reps import rowcol, parity_matrix
    >>> cnots = rowcol(P, G)

    The constructed circuit is the one found in the paper as well:

    >>> circ = qml.tape.QuantumScript([qml.CNOT(pair) for pair in cnots])
    >>> print(qml.drawer.tape_text(circ, wire_order=range(5)))
    0: ──────────────────────────────────╭X───────╭X─╭●───────┤
    1: ─────────────╭X───────╭X─╭●────╭X─│────────│──│────────┤
    2: ────╭X────╭●─╰●────╭X─╰●─╰X─╭●─╰●─│─────╭●─│──│─────╭X─┤
    3: ─╭X─╰●─╭X─╰X─╭●─╭X─╰●─╭X────╰X────╰●─╭X─╰X─╰●─╰X─╭●─╰●─┤
    4: ─╰●────╰●────╰X─╰●────╰●─────────────╰●──────────╰X────┤

    We can confirm that this circuit indeed implements the original parity matrix:

    >>> recon_P = parity_matrix(circ, wire_order=range(5))
    >>> np.allclose(P, recon_P)
    True

    .. details::
        :title: Algorithm overview
        :href: algorithm-overview

        Given a parity matrix :math:`P` and a connectivity graph :math:`G(V, E)`, as well as an
        empty list ``L`` to which we will append CNOT gates,
        the algorithm consists of the following steps:

        #. If :math:`|V|=1` (there is a single node), go to step 5. Else, select a vertex
           :math:`i\in V` that is not a cut vertex, i.e. a vertex that we can remove without
           disconnecting the graph.
        #. Eliminate column ``P[:, i]``
            a. Construct :math:`S = \{j | P_{ji} \neq 0\} \cup \{i\}` and find a Steiner tree
               :math:`T \supseteq S`.
            b. `Post-order traverse <https://en.wikipedia.org/wiki/Tree_traversal#Post-order,_LRN>`__
               the Steiner tree :math:`T` starting from :math:`i` as root. If ``P[c, i]=1`` for node
               :math:`c` in the traversal and ``P[p, i]=0`` for parent node :math:`p` of node :math:`c`,
               add row ``P[c]`` to row ``P[p]`` and append ``CNOT([c, p])`` to ``L``.
            c. Post-order traverse :math:`T` again starting from :math:`i`. For each node :math:`p`
               in the traversal, add row ``P[p]`` to row ``P[c]``, where :math:`c` is the child node
               of :math:`p`, and append ``CNOT([p, c])`` to ``L``.
        #. Eliminate row ``P[i]``
            a. Construct :math:`S'` such that the corresponding rows of :math:`P` add up to the
               :math:`i`\ th basis vector, :math:`\sum_{j\in S'} P[j] = e_i`. Find a Steiner tree
               :math:`T' \supseteq S'`.
            b. `Pre-order traverse <https://en.wikipedia.org/wiki/Tree_traversal#Pre-order,_NLR>`__
               the Steiner tree :math:`T'` starting from :math:`i` and for each node :math:`c` in
               the traversal for which :math:`c \notin S'`, add row :math:`P[c]` to row :math:`P[p]`,
               where :math:`p` is the parent node of :math:`c`, and append ``CNOT([c, p])`` to ``L``.
            c. Post-order traverse :math:`T` starting from :math:`i`. For each node :math:`c`
               in the traversal, add row ``P[c]`` to row ``P[p]``, where :math:`p` is the parent node
               of :math:`c`, and append ``CNOT([c, p])`` to ``L``.
        #. Delete vertex :math:`i` from :math:`G` and ignore row and column with index :math:`i` in
           the following. Go to step 1.
        #. Revert the list ``L`` of CNOT gates in order to go from the circuit that transforms
           :math:`P` to the identity to the circuit that implements :math:`P` starting from the
           identity.

    .. details::
        :title: Manual example
        :href: manual-example

        We walk through example 1 in `Wu et al. <https://arxiv.org/abs/1910.14478>`__, which
        was demonstrated in code above, in detail. The steps are numbered according to
        the algorithm overview above.
        We restrict ourselves to the following connectivity graph.

        .. code-block:: python

            (0) - (3) - (4)
                   |
                  (2)
                   |
                  (1)

        We look at a CNOT circuit with the following parity matrix. The first column is
        highlighted as we are going to start by eliminating it (``(0)`` is not a cut vertex).

        .. math::

            P = \begin{pmatrix}
                \color{blue}1 & 1 & 0 & 1 & 1 \\
                \color{blue}0 & 0 & 1 & 1 & 0 \\
                \color{blue}1 & 0 & 1 & 0 & 1 \\
                \color{blue}1 & 1 & 0 & 1 & 0 \\
                \color{blue}1 & 1 & 1 & 1 & 0
            \end{pmatrix}

        *Step 1:* Pick vertex ``i=0``

        *Step 2:* Eliminate ``P[:, 0]``

        *Step 2.a:* Find :math:`S` and :math:`T`

        All non-zero vertices are ``S = [0, 2, 3, 4]``. A Steiner tree ``T`` connecting those
        vertices is given by

        .. code-block:: python

            (0) - (3) - (4)
                   |
                  (2)

        *Step 2.b:* First post-order traversal

        We post-order traverse :math:`T` from :math:`0`, giving the ordering :math:`[2, 4, 3, 0]`,
        and check for nodes :math:`c` with parent :math:`p` where :math:`P_{ci}=0` and
        :math:`P_{pi}=1`. This can only happen when we have elements in :math:`T` that are not in
        :math:`S` or when the vertex :math:`i` itself has a value :math:`0`. Neither is the case here;
        therefore, no action is required.

        *Step 2.c:* Second post-order traversal

        We again post-order traverse the tree and add *every* vertex to its children, i.e.
        we get the CNOT operations :math:`\text{CNOT}_{3 2} \text{CNOT}_{3 4} \text{CNOT}_{0 3}`.
        Simultaneously, we update the parity matrix with the corresponding row additions (modulo 2).

        .. math::

            P = \begin{pmatrix}
                1 & 1 & 0 & 1 & 1 \\
                0 & 0 & 1 & 1 & 0 \\
                0 & 1 & 1 & 1 & 1 \\
                0 & 0 & 0 & 0 & 1 \\
                0 & 0 & 1 & 0 & 0
            \end{pmatrix}.

        *Step 3:* Eliminate ``P[0, :]``

        Now we eliminate the first row, marked blue below.

        .. math::

            P = \begin{pmatrix}
                \color{blue}1 & \color{blue}1 & \color{blue}0 & \color{blue}1 & \color{blue}1 \\
                0 & 0 & 1 & 1 & 0 \\
                0 & 1 & 1 & 1 & 1 \\
                0 & 0 & 0 & 0 & 1 \\
                0 & 0 & 1 & 0 & 0
            \end{pmatrix}

        *Step 3.a:* Find :math:`S'` and :math:`T'`

        We see that we can add rows :math:`S' = \{0, 2, 4\}` together to obtain the
        first unit vector :math:`e_0`.
        The Steiner tree :math:`T'` encapsulating :math:`S'` is then

        .. code-block:: python

            (0) - (3) - (4)
                   |
                  (2)

        where :math:`3 \notin S'`.

        *Step 3.b:* Pre-order traverse

        We pre-order traverse :math:`T'` (yielding :math:`[0, 3, 2, 4]`) and for all
        :math:`c \notin S'`, we add them to their parent. In particular, we get an
        additional :math:`\text{CNOT}_{3 0}`.
        This is necessary because we are going to traverse through this node in the next step and we
        need to undo this step again to eliminate the targeted row.

        *Step 3.c:* Post-order traverse

        Similarly to step 2.c, we post-order traverse :math:`T'`, but this time adding every node's
        row to that of its parent. This yields the same circuit as in step 2.c (because the Steiner
        trees are the same), but with the roles of target and control qubits reversed.

        *Step 4:* Delete and iterate

        We delete vertex :math:`i=0` from :math:`G` and ignore the first row and column of :math:`P`
        in the following. Then we return to step 1.

        *Step 1:* Pick vertex ``i=1``

        *Step 2:* Eliminate ``P[:, 1]``

        We repeat the procedure for the next vertex :math:`i=1` from the resulting parity matrix

        .. math::

            P = \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 \\
                0 & \color{blue}0 & 1 & 1 & 0 \\
                0 & \color{blue}1 & 1 & 1 & 1 \\
                0 & \color{blue}1 & 0 & 1 & 0 \\
                0 & \color{blue}0 & 1 & 0 & 0
            \end{pmatrix}.

        *Step 2.a:* Find :math:`S` and :math:`T`

        The set :math:`S` of non-zero entries in column ``P[:,1]`` together with :math:`i` itself
        is :math:`S = [1, 2, 3]` and the encapsulating Steiner tree :math:`T` is simply

        .. code-block:: python

            (3)
             |
            (2)
             |
            (1)

        *Step 2.b:* First post-order traversal

        We post-order traverse :math:`T` from :math:`1` (yielding :math:`[3, 2, 1]`) to check
        if any parent has :math:`P_{pi}=0`, which is the case for :math:`p=1`, so we add
        :math:`\text{CNOT}_{2 1}`.

        *Step 2.c:* Second post-order traversal

        Once again, add every parent node row to the row of its children while post-order traversing.
        We additionally get :math:`\text{CNOT}_{2 3}\text{CNOT}_{1 2}`.

        *Step 3:* Eliminate ``P[1]``

        Now we eliminate the second row, marked blue below.

        .. math::

            P = \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 \\
                0 & \color{blue}1 & \color{blue}0 & \color{blue}0 & \color{blue}1 \\
                0 & 0 & 1 & 1 & 0 \\
                0 & 0 & 1 & 0 & 1 \\
                0 & 0 & 1 & 0 & 0
            \end{pmatrix}

        *Step 3.a:* Find :math:`S'` and :math:`T'`

        We first construct :math:`S' = [1, 3, 4]`, which labels rows whose sum yields :math:`e_1`.
        The encapsulating Steiner tree :math:`T'` is

        .. code-block:: python

            (3) - (4)
             |
            (2)
             |
            (1)

        where :math:`2 \notin S'`.

        *Step 3.b:* Pre-order traverse

        Add :math:`2 \notin S'` to its parent during pre-order traversal (:math:`[1, 2, 3, 4]`),
        i.e. add :math:`\text{CNOT}_{2 1}`.

        *Step 3.c:* Post-order traverse

        Post-order traverse the tree (:math:`[4, 3, 2, 1]`) and add every row to its parent:
        :math:`\text{CNOT}_{4 3}\text{CNOT}_{3 2}\text{CNOT}_{2 1}`.

        *Step 4:* Delete and iterate

        We delete vertex :math:`i=1` from :math:`G` and ignore the second row and column of :math:`P`
        in the following. Then we return to step 1.

        **Remaining operations**

        At this point, we have covered all the cases. We defer to the compilation hub
        for a more extensive explanation of this example and here simply list the remaining gates.

        *Step 1:* Select vertex :math:`i=2`

        *Step 2.b:* :math:`\text{CNOT}_{4 3}`

        *Step 2.c:* :math:`\text{CNOT}_{3 4}\text{CNOT}_{2 3}`

        *Step 3.a:* no gate

        *Step 3.c:* :math:`\text{CNOT}_{4 3}\text{CNOT}_{3 2}`

        *Step 4:* Delete :math:`i=2`

        *Step 1:* Select vertex :math:`i=3`

        *Step 2.b:* no gate

        *Step 2.c:* no gate

        *Step 3.a:* no gate

        *Step 3.c:* :math:`\text{CNOT}_{4 3}`

        *Step 4:* Delete :math:`i=3`

        *Step 1:* :math:`|V|=|\{4\}|=1`, terminate.

        The final circuit is the reverse of the synthesized subcircuits, matching the
        result from the code example above:

        >>> print(qml.drawer.tape_text(circ, wire_order=range(5)))
        0: ──────────────────────────────────╭X───────╭X─╭●───────┤
        1: ─────────────╭X───────╭X─╭●────╭X─│────────│──│────────┤
        2: ────╭X────╭●─╰●────╭X─╰●─╰X─╭●─╰●─│─────╭●─│──│─────╭X─┤
        3: ─╭X─╰●─╭X─╰X─╭●─╭X─╰●─╭X────╰X────╰●─╭X─╰X─╰●─╰X─╭●─╰●─┤
        4: ─╰●────╰●────╰X─╰●────╰●─────────────╰●──────────╰X────┤

    """
    P = P.copy()
    connectivity = connectivity.copy()
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
        for mode in ("column", "row"):
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
    return cnots[::-1]
