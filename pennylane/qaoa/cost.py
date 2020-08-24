# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Methods for generating QAOA cost Hamiltonians corresponding to
different optimization problems.
"""

import networkx as nx
import pennylane as qml
from pennylane import qaoa


########################
# Hamiltonian components


def bit_driver(wires, n):
    r"""Returns the bit-driver cost Hamiltonian component.

    This Hamiltonian is defined as:

    .. math:: H \ = \ (-1)^{n + 1} \displaystyle\sum_{i} Z_i

    where :math:`Z_i` is the Pauli-Z operator acting on the
    :math:`i`-th wire and :math:`n \ \in \ \{0, \ 1\}`. This Hamiltonian is often used as a term when
    constructing larger QAOA cost Hamiltonians.

    Args:
        wires (Iterable or Wires): The wires on which the returned Hamiltonian acts
        n (int): Either :math:`0` or :math:`1`. Determines whether the Hamiltonian assigns
                 lower energies to bitstrings with more :math:`0`s or :math:`1`s, respectively.

    Returns:
        .Hamiltonian

    **Example**

    >>> wires = range(3)
    >>> hamiltonian = qaoa.pauli_driver(wires, 1)
    >>> print(hamiltonian)
    (1.0) [Z0] + (1.0) [Z1] + (1.0) [Z2]
    """
    if n == 0:
        coeffs = [-1 for _ in wires]
    elif n == 1:
        coeffs = [1 for _ in wires]
    else:
        raise ValueError("'state' argument must be either 0 or 1, got {}".format(n))

    ops = [qml.PauliZ(w) for w in wires]
    return qml.Hamiltonian(coeffs, ops)


def edge_driver(graph, reward):
    r"""Returns the edge-driver cost Hamiltonian component.

    Given some graph, :math:`G`, this method will return a Hamiltonian that "rewards"
    bitstrings encoding graph colourings of :math:`G` (assigns them a lower
    energy) with edges that end in nodes with colourings supplied in ``reward``.

    See usage details for more information.

    Args:
         graph (nx.Graph): The graph on which the Hamiltonian is defined
         reward (list[str]): The list of two-bit bitstrings that are "rewarded" by the Hamiltonian

    Returns:
        .Hamiltonian

    **Example**

    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> hamiltonian = qaoa.edge_driver(graph, ["11", "10", "01"])
    >>> print(hamiltonian)
    (1.0) [Z0 Z1] + (1.0) [Z0] + (2.0) [Z1] + (1.0) [Z1 Z2] + (1.0) [Z2]
    """

    allowed = ["00", "01", "10", "11"]

    if not all([e in allowed for e in reward]):
        raise ValueError("Encountered invalid entry in 'reward', expected 2-bit bitstrings.")

    if "01" in reward and "10" not in reward or "10" in reward and "01" not in reward:
        raise ValueError(
            "'reward' cannot contain either '10' or '01', must contain neither, or both."
        )

    if not isinstance(graph, nx.Graph):
        raise ValueError("Input graph must be a nx.Graph, got {}".format(type(graph).__name__))

    coeffs = []
    ops = []

    if len(reward) == 0 or len(reward) == 4:
        coeffs = [1 for _ in graph.nodes]
        ops = [qml.Identity(v) for v in graph.nodes]

    else:

        reward = list(set(reward) - {"01"})
        sign = -1

        if len(reward) == 2:
            reward = list({"00", "10", "11"} - set(reward))
            sign = 1

        reward = reward[0]

        if reward == "00":
            for e in graph.edges:
                coeffs.extend([0.5 * sign, 0.5 * sign, 0.5 * sign])
                ops.extend(
                    [qml.PauliZ(e[0]) @ qml.PauliZ(e[1]), qml.PauliZ(e[0]), qml.PauliZ(e[1])]
                )

        if reward == "10":
            for e in graph.edges:
                coeffs.append(-1 * sign)
                ops.append(qml.PauliZ(e[0]) @ qml.PauliZ(e[1]))

        if reward == "11":
            for e in graph.edges:
                coeffs.extend([0.5 * sign, -0.5 * sign, -0.5 * sign])
                ops.extend(
                    [qml.PauliZ(e[0]) @ qml.PauliZ(e[1]), qml.PauliZ(e[0]), qml.PauliZ(e[1])]
                )

    return qml.Hamiltonian(coeffs, ops)


#######################
# Optimization problems


def maxcut(graph):
    r"""Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the
    MaxCut problem, for a given graph.

    The goal of the MaxCut problem for a particular graph is to find a partition of nodes into two sets,
    such that the number of edges in the graph with endpoints in different sets is maximized. Formally,
    we wish to find the `cut of the graph <https://en.wikipedia.org/wiki/Cut_(graph_theory)>`__ such
    that the number of edges crossing the cut is maximized.

    The MaxCut cost Hamiltonian is defined as:

    .. math:: H_C \ = \ \frac{1}{2} \displaystyle\sum_{(i, j) \in E(G)} \big( Z_i Z_j \ - \ \mathbb{I} \big),

    where :math:`G` is a graph, :math:`\mathbb{I}` is the identity, and :math:`Z_i` and :math:`Z_j` are
    the Pauli-Z operators on the :math:`i`-th and :math:`j`-th wire respectively.

    The mixer Hamiltonian returned from :func:`~qaoa.maxcut` is :func:`~qaoa.x_mixer` applied to all wires.

    .. note::

        **Recommended initialization circuit:**
            Even superposition over all basis states

    Args:
        graph (nx.Graph): a graph defining the pairs of wires on which each term of the Hamiltonian acts

    Returns:
        (.Hamiltonian, .Hamiltonian):

    **Example**

    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> cost_h, mixer_h = qml.qaoa.maxcut(graph)
    >>> print(cost_h)
    >>> print(mixer_h)
    (0.5) [Z0 Z1] + (0.5) [Z1 Z2] + (-1.0) [I0]
    (1.0) [X0] + (1.0) [X1] + (1.0) [X2]
    """

    if not isinstance(graph, nx.Graph):
        raise ValueError("Input graph must be a nx.Graph, got {}".format(type(graph).__name__))

    identity_h = qml.Hamiltonian(
        [-0.5 for e in graph.edges], [qml.Identity(e[0]) @ qml.Identity(e[1]) for e in graph.edges]
    )
    H = 0.5 * edge_driver(graph, ["10", "01"]) + identity_h
    return (H, qaoa.x_mixer(graph.nodes))


def max_independent_set(graph, constrained=True):
    r"""Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the MaxIndependentSet problem,
    for a given graph.

    The goal of MaxIndependentSet is to find the largest possible independent set of a graph. Given some graph :math:`G`,
    an independent set of :math:`G` is a set of vertices such that no two of the vertices in the set share a common edge.

    Args:
        graph (nx.Graph): a graph defining the pairs of wires on which each term of the Hamiltonian acts
        constrained (bool): specifies the variant of QAOA that is performed (constrained or unconstrained)

    Returns:
        (.Hamiltonian, .Hamiltonian):

    .. UsageDetails::

        There are two variations of QAOA for this problem, constrained and unconstrained:

        **Constrained**

        .. note::

            This method of constrained QAOA was introduced by Hadfield, Wang, Gorman, Rieffel, Venturelli, and Biswas
            in `[arXiv:1709.03489] <https://arxiv.org/abs/1709.03489>`__.

        The constrained MaxIndependentSet cost Hamiltonian is defined as:

        .. math:: H_C \ = \ \displaystyle\sum_{v \in V(G)} Z_{v}

        where :math:`V(G)` is the set of vertices of the input graph, and :math:`Z_i` is the Pauli-Z operator applied to the :math:`i`-th
        vertex.

        The returned mixer Hamiltonian is `~qaoa.bit_flip_mixer` applied to :math:`G`.

        .. note::

            **Recommended initialization circuit:**
                Each wire in the :math:`|0\rangle` state

        **Unconstrained**

        The unconstrained MaxIndependentSet cost Hamiltonian is defined as:

        .. math:: H_C \ = \ \frac{(i, j) \in E(G)} (Z_i Z_j \ - \ Z_i \ - \ Z_j) \ + \ \displaystyle\sum_{i \in V(G)} Z_i

        where :math:`E(G)` is the edges of :math:`G`, :math:`V(G)` is the set of vertices, and :math:`Z_i` is the Pauli-Z operator
        acting on the :math:`i`-th vertex.

        The returned mixer Hamiltonian is `~qaoa.x_mixer` applied to all wires.

        .. note::

            **Recommended initialization circuit:**
                Even superposition over all basis states
    """

    if not isinstance(graph, nx.Graph):
        raise ValueError("Input graph must be a nx.Graph, got {}".format(type(graph).__name__))

    if constrained:
        return (bit_driver(graph.nodes, 1), qaoa.bit_flip_mixer(graph, 0))

    cost_h = edge_driver(graph, ['10', '01', '00']) + bit_driver(graph.nodes, 1)
    mixer_h = qaoa.x_mixer(graph.nodes)

    return (cost_h, mixer_h)


def min_vertex_cover(graph, constrained=True):
    r"""Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the Minimum Vertex Cover problem,
    for a given graph.

    The goal of the Minimum Vertex Cover problem is to find the smallest
    `vertex cover <https://en.wikipedia.org/wiki/Vertex_cover>`__ of a graph (a collection of nodes such that
    every edge in the graph has one of the nodes as an endpoint).

    Args:
        graph (nx.Graph): a graph defining the pairs of wires on which each term of the Hamiltonian acts
        constrained (bool): specifies the variant of QAOA that is performed (constrained or unconstrained)

    Returns:
        (.Hamiltonian, .Hamiltonian):

    .. UsageDetails::

        There are two variations of QAOA for this problem, constrained and unconstrained:

        **Constrained**

        .. note::

            This method of constrained QAOA was introduced by Hadfield, Wang, Gorman, Rieffel, Venturelli, and Biswas
            in `[arXiv:1709.03489] <https://arxiv.org/abs/1709.03489>`__.

        The constrained MinVertexCover cost Hamiltonian is defined as:

        .. math:: H_C \ = \ - \displaystyle\sum_{v \in V(G)} Z_{v}

        where :math:`V(G)` is the set of vertices of the input graph, and :math:`Z_i` is the Pauli-Z operator applied to the :math:`i`-th
        vertex.

        The returned mixer Hamiltonian is `~qaoa.bit_flip_mixer` applied to :math:`G`.

        .. note::

            **Recommended initialization circuit:**
                Each wire in the :math:`|1\rangle` state

        **Unconstrained**

        The Minimum Vertex Cover cost Hamiltonian is defined as:

        .. math:: H_C \ = \ \frac{(i, j) \in E(G)} (Z_i Z_j \ + \ Z_i \ + \ Z_j) \ - \ \displaystyle\sum_{i \in V(G)} Z_i

        where :math:`E(G)` is the edges of :math:`G`, :math:`V(G)` is the set of vertices, and :math:`Z_i` is the Pauli-Z operator
        acting on the :math:`i`-th vertex.

        The returned mixer Hamiltonian is `~qaoa.x_mixer` applied to all wires.

        .. note::

            **Recommended initialization circuit:**
                Even superposition over all basis states
    """

    if not isinstance(graph, nx.Graph):
        raise ValueError("Input graph must be a nx.Graph, got {}".format(type(graph).__name__))

    if constrained:
        return (bit_driver(graph.nodes, 0), qaoa.bit_flip_mixer(graph, 1))

    cost_h = edge_driver(graph, ['11', '10', '01']) + bit_driver(graph.nodes, 0)
    mixer_h = qaoa.x_mixer(graph.nodes)

    return (cost_h, mixer_h)


def maxclique(graph, constrained=True):
    r"""Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the MaxClique problem,
    for a given graph.

    The goal of MaxClique is to find the largest `clique <https://en.wikipedia.org/wiki/Clique_(graph_theory)>`__ of a
    graph (the largest subgraph with all nodes sharing an edge).

    Args:
        graph (nx.Graph): a graph defining the pairs of wires on which each term of the Hamiltonian acts
        constrained (bool): specifies the variant of QAOA that is performed (constrained or unconstrained)

    Returns:
        (.Hamiltonian, .Hamiltonian):

    .. UsageDetails::

        There are two variations of QAOA for this problem, constrained and unconstrained:

        **Constrained**

        .. note::

            This method of constrained QAOA was introduced by Hadfield, Wang, Gorman, Rieffel, Venturelli, and Biswas
            in `[arXiv:1709.03489] <https://arxiv.org/abs/1709.03489>`__.

        The constrained MaxClique cost Hamiltonian is defined as:

        .. math:: H_C \ = \ \displaystyle\sum_{v \in V(G)} Z_{v}

        where :math:`V(G)` is the set of vertices of the input graph, and :math:`Z_i` is the Pauli-Z operator applied to the :math:`i`-th
        vertex.

        The returned mixer Hamiltonian is `~qaoa.bit_flip_mixer` applied to :math:`\bar{G}` (the complement of the graph).

        .. note::

            **Recommended initialization circuit:**
                Each wire in the :math:`|0\rangle` state

        **Unconstrained**

        The unconstrained MaxClique cost Hamiltonian is defined as:

        .. math:: H_C \ = \ \frac{(i, j) \in E(\bar{G})} (Z_i Z_j \ - \ Z_i \ - \ Z_j) \ + \ \displaystyle\sum_{i \in V(G)} Z_i

        where :math:`V(G)` is the vertices of the input graph :math:`G`, :math:`E(\bar{G})` is the edges of the
        complement of :math:`G` and :math:`Z_i` is the Pauli-Z operator applied to the :math:`i`-th vertex.

        The returned mixer Hamiltonian is `~qaoa.x_mixer` applied to all wires.

        .. note::

            **Recommended initialization circuit:**
                Even superposition over all basis states
    """

    if not isinstance(graph, nx.Graph):
        raise ValueError("Input graph must be a nx.Graph, got {}".format(type(graph).__name__))

    if constrained:
        return (bit_driver(graph.nodes, 1), qaoa.bit_flip_mixer(nx.complement(graph), 0))

    cost_h = edge_driver(nx.complement(graph), ['10', '01', '00']) + bit_driver(graph.nodes, 1)
    mixer_h = qaoa.x_mixer(graph.nodes)

    return (cost_h, mixer_h)
