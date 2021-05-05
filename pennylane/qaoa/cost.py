# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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


def bit_driver(wires, b):
    r"""Returns the bit-driver cost Hamiltonian.

    This Hamiltonian is defined as:

    .. math:: H \ = \ (-1)^{b + 1} \displaystyle\sum_{i} Z_i

    where :math:`Z_i` is the Pauli-Z operator acting on the
    :math:`i`-th wire and :math:`b \ \in \ \{0, \ 1\}`. This Hamiltonian is often used when
    constructing larger QAOA cost Hamiltonians.

    Args:
        wires (Iterable or Wires): The wires on which the Hamiltonian acts
        b (int): Either :math:`0` or :math:`1`. Determines whether the Hamiltonian assigns
                 lower energies to bitstrings with a majority of bits being :math:`0` or
                 a majority of bits being :math:`1`, respectively.

    Returns:
        .Hamiltonian:

    **Example**

    >>> wires = range(3)
    >>> hamiltonian = qaoa.bit_driver(wires, 1)
    >>> print(hamiltonian)
      (1) [Z0]
    + (1) [Z1]
    + (1) [Z2]
    """
    if b == 0:
        coeffs = [-1 for _ in wires]
    elif b == 1:
        coeffs = [1 for _ in wires]
    else:
        raise ValueError("'b' must be either 0 or 1, got {}".format(b))

    ops = [qml.PauliZ(w) for w in wires]
    return qml.Hamiltonian(coeffs, ops)


def edge_driver(graph, reward):
    r"""Returns the edge-driver cost Hamiltonian.

    Given some graph, :math:`G` with each node representing a wire, and a binary
    colouring where each node/wire is assigned either :math:`|0\rangle` or :math:`|1\rangle`, the edge driver
    cost Hamiltonian will assign a lower energy to edges represented by qubit states with endpoint colourings
    supplied in ``reward``.

    For instance, if ``reward`` is ``["11"]``, then edges
    with both endpoints coloured as ``1`` (the state :math:`|11\rangle`) will be assigned a lower energy, while
    the other colourings  (``"00"``, ``"10"``, and ``"01"`` corresponding to states
    :math:`|00\rangle`, :math:`|10\rangle`, and :math:`|10\rangle`, respectively) will be assigned a higher energy.

    See usage details for more information.

    Args:
         graph (nx.Graph): The graph on which the Hamiltonian is defined
         reward (list[str]): The list of two-bit bitstrings that are assigned a lower energy by the Hamiltonian

    Returns:
        .Hamiltonian:

    **Example**

    >>> import networkx as nx
    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> hamiltonian = qaoa.edge_driver(graph, ["11", "10", "01"])
    >>> print(hamiltonian)
      (0.25) [Z0]
    + (0.25) [Z1]
    + (0.25) [Z1]
    + (0.25) [Z2]
    + (0.25) [Z0 Z1]
    + (0.25) [Z1 Z2]

    In the above example, ``"11"``, ``"10"``, and ``"01"`` are assigned a lower
    energy than ``"00"``. For example, a quick calculation of expectation values gives us:

    .. math:: \langle 000 | H | 000 \rangle \ = \ 1.5
    .. math:: \langle 100 | H | 100 \rangle \ = \ 0.5
    .. math:: \langle 110 | H | 110\rangle \ = \ -0.5

    In the first example, both vertex pairs are not in ``reward``. In the second example, one pair is in ``reward`` and
    the other is not. Finally, in the third example, both pairs are in ``reward``.

    .. UsageDetails::

        The goal of many combinatorial problems that can be solved with QAOA is to
        find a `Graph colouring <https://en.wikipedia.org/wiki/Graph_coloring>`__ of some supplied
        graph :math:`G`, that minimizes some cost function. With QAOA, it is natural to consider the class
        of graph colouring problems that only admit two colours, as we can easily encode these two colours
        using the :math:`|1\rangle` and :math:`|0\rangle` states of qubits. Therefore, given
        some graph :math:`G`, each edge of the graph can be described by a pair of qubits, :math:`|00\rangle`,
        :math:`|01\rangle`, :math:`|10\rangle`, or :math:`|11\rangle`, corresponding to the colourings of its endpoints.

        When constructing QAOA cost functions, one must "penalize" certain states of the graph, and "reward"
        others, by assigning higher and lower energies to these respective configurations. Given a set of vertex-colour
        pairs (which each describe a possible  state of a graph edge), the ``edge_driver()``
        function outputs a Hamiltonian that rewards the pairs in the set, and penalizes the others.

        For example, given the reward set: :math:`\{|00\rangle, \ |01\rangle, \ |10\rangle\}` and the graph :math:`G`,
        the ``edge_driver()`` function will output the following Hamiltonian:

        .. math:: H \ = \ \frac{1}{4} \displaystyle\sum_{(i, j) \in E(G)} \big( Z_{i} Z_{j} \ - \ Z_{i} \ - \ Z_{j} \big)

        where :math:`E(G)` is the set of edges of :math:`G`, and :math:`Z_i` is the Pauli-Z operator acting on the
        :math:`i`-th wire. As can be checked, this Hamiltonian assigns an energy of :math:`-1/4` to the states
        :math:`|00\rangle`, :math:`|01\rangle` and :math:`|10\rangle`, and an energy of :math:`3/4` to the state
        :math:`|11\rangle`.

        .. Note::

            ``reward`` must always contain both :math:`|01\rangle` and :math:`|10\rangle`, or neither of the two.
            Within an undirected graph, there is no notion of "order"
            of edge endpoints, so these two states are effectively the same. Therefore, there is no well-defined way to
            penalize one and reward the other.

        .. Note::

            The absolute difference in energy between colourings in ``reward`` and colourings in its
            complement is always :math:`1`.

    """

    allowed = ["00", "01", "10", "11"]

    if not all(e in allowed for e in reward):
        raise ValueError("Encountered invalid entry in 'reward', expected 2-bit bitstrings.")

    if "01" in reward and "10" not in reward or "10" in reward and "01" not in reward:
        raise ValueError(
            "'reward' cannot contain either '10' or '01', must contain neither or both."
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
                coeffs.extend([0.25 * sign, 0.25 * sign, 0.25 * sign])
                ops.extend(
                    [qml.PauliZ(e[0]) @ qml.PauliZ(e[1]), qml.PauliZ(e[0]), qml.PauliZ(e[1])]
                )

        if reward == "10":
            for e in graph.edges:
                coeffs.append(-0.5 * sign)
                ops.append(qml.PauliZ(e[0]) @ qml.PauliZ(e[1]))

        if reward == "11":
            for e in graph.edges:
                coeffs.extend([0.25 * sign, -0.25 * sign, -0.25 * sign])
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
        (.Hamiltonian, .Hamiltonian): The cost and mixer Hamiltonians

    **Example**

    >>> import networkx as nx
    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> cost_h, mixer_h = qml.qaoa.maxcut(graph)
    >>> print(cost_h)
      (-1.0) [I0]
    + (0.5) [Z0 Z1]
    + (0.5) [Z1 Z2]
    >>> print(mixer_h)
      (1) [X0]
    + (1) [X1]
    + (1) [X2]
    """

    if not isinstance(graph, nx.Graph):
        raise ValueError("Input graph must be a nx.Graph, got {}".format(type(graph).__name__))

    identity_h = qml.Hamiltonian(
        [-0.5 for e in graph.edges], [qml.Identity(e[0]) @ qml.Identity(e[1]) for e in graph.edges]
    )
    H = edge_driver(graph, ["10", "01"]) + identity_h
    return (H, qaoa.x_mixer(graph.nodes))


def max_independent_set(graph, constrained=True):
    r"""For a given graph, returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the Maximum Independent Set problem.

    Given some graph :math:`G`, an independent set is a set of vertices such that no pair of vertices in the set
    share a common edge. The Maximum Independent Set problem, is the problem of finding the largest such set.

    Args:
        graph (nx.Graph): a graph whose edges define the pairs of vertices on which each term of the Hamiltonian acts
        constrained (bool): specifies the variant of QAOA that is performed (constrained or unconstrained)

    Returns:
        (.Hamiltonian, .Hamiltonian): The cost and mixer Hamiltonians

    .. UsageDetails::

        There are two variations of QAOA for this problem, constrained and unconstrained:

        **Constrained**

        .. note::

            This method of constrained QAOA was introduced by Hadfield, Wang, Gorman, Rieffel, Venturelli, and Biswas
            in arXiv:1709.03489.

        The Maximum Independent Set cost Hamiltonian for constrained QAOA is defined as:

        .. math:: H_C \ = \ \displaystyle\sum_{v \in V(G)} Z_{v},

        where :math:`V(G)` is the set of vertices of the input graph, and :math:`Z_i` is the Pauli-Z
        operator applied to the :math:`i`-th vertex.

        The returned mixer Hamiltonian is :func:`~qaoa.bit_flip_mixer` applied to :math:`G`.

        .. note::

            **Recommended initialization circuit:**
                Each wire in the :math:`|0\rangle` state.

        **Unconstrained**

        The Maximum Independent Set cost Hamiltonian for unconstrained QAOA is defined as:

        .. math:: H_C \ = \ 3 \sum_{(i, j) \in E(G)} (Z_i Z_j \ - \ Z_i \ - \ Z_j) \ + \
                  \displaystyle\sum_{i \in V(G)} Z_i

        where :math:`E(G)` is the set of edges of :math:`G`, :math:`V(G)` is the set of vertices,
        and :math:`Z_i` is the Pauli-Z operator acting on the :math:`i`-th vertex.

        The returned mixer Hamiltonian is :func:`~qaoa.x_mixer` applied to all wires.

        .. note::

            **Recommended initialization circuit:**
                Even superposition over all basis states.

    """

    if not isinstance(graph, nx.Graph):
        raise ValueError("Input graph must be a nx.Graph, got {}".format(type(graph).__name__))

    if constrained:
        return (bit_driver(graph.nodes, 1), qaoa.bit_flip_mixer(graph, 0))

    cost_h = 3 * edge_driver(graph, ["10", "01", "00"]) + bit_driver(graph.nodes, 1)
    mixer_h = qaoa.x_mixer(graph.nodes)

    return (cost_h, mixer_h)


def min_vertex_cover(graph, constrained=True):
    r"""Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the Minimum Vertex Cover problem,
    for a given graph.

    To solve the Minimum Vertex Cover problem, we attempt to find the smallest
    `vertex cover <https://en.wikipedia.org/wiki/Vertex_cover>`__ of a graph --- a collection of vertices such that
    every edge in the graph has one of the vertices as an endpoint.

    Args:
        graph (nx.Graph): a graph whose edges define the pairs of vertices on which each term of the Hamiltonian acts
        constrained (bool): specifies the variant of QAOA that is performed (constrained or unconstrained)

    Returns:
        (.Hamiltonian, .Hamiltonian): The cost and mixer Hamiltonians

    .. UsageDetails::

        There are two variations of QAOA for this problem, constrained and unconstrained:

        **Constrained**

        .. note::

            This method of constrained QAOA was introduced by Hadfield, Wang, Gorman, Rieffel, Venturelli, and Biswas
            in arXiv:1709.03489.

        The Minimum Vertex Cover cost Hamiltonian for constrained QAOA is defined as:

        .. math:: H_C \ = \ - \displaystyle\sum_{v \in V(G)} Z_{v},

        where :math:`V(G)` is the set of vertices of the input graph, and :math:`Z_i` is the Pauli-Z operator
        applied to the :math:`i`-th vertex.

        The returned mixer Hamiltonian is :func:`~qaoa.bit_flip_mixer` applied to :math:`G`.

        .. note::

            **Recommended initialization circuit:**
                Each wire in the :math:`|1\rangle` state.

        **Unconstrained**

        The Minimum Vertex Cover cost Hamiltonian for unconstrained QAOA is defined as:

        .. math:: H_C \ = \ 3 \sum_{(i, j) \in E(G)} (Z_i Z_j \ + \ Z_i \ + \ Z_j) \ - \
                  \displaystyle\sum_{i \in V(G)} Z_i

        where :math:`E(G)` is the set of edges of :math:`G`, :math:`V(G)` is the set of vertices,
        and :math:`Z_i` is the Pauli-Z operator acting on the :math:`i`-th vertex.

        The returned mixer Hamiltonian is :func:`~qaoa.x_mixer` applied to all wires.

        .. note::

            **Recommended initialization circuit:**
                Even superposition over all basis states.

    """

    if not isinstance(graph, nx.Graph):
        raise ValueError("Input graph must be a nx.Graph, got {}".format(type(graph).__name__))

    if constrained:
        return (bit_driver(graph.nodes, 0), qaoa.bit_flip_mixer(graph, 1))

    cost_h = 3 * edge_driver(graph, ["11", "10", "01"]) + bit_driver(graph.nodes, 0)
    mixer_h = qaoa.x_mixer(graph.nodes)

    return (cost_h, mixer_h)


def max_clique(graph, constrained=True):
    r"""Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the Maximum Clique problem,
    for a given graph.

    The goal of Maximum Clique is to find the largest `clique <https://en.wikipedia.org/wiki/Clique_(graph_theory)>`__ of a
    graph --- the largest subgraph such that all vertices are connected by an edge.

    Args:
        graph (nx.Graph): a graph whose edges define the pairs of vertices on which each term of the Hamiltonian acts
        constrained (bool): specifies the variant of QAOA that is performed (constrained or unconstrained)

    Returns:
        (.Hamiltonian, .Hamiltonian): The cost and mixer Hamiltonians

    .. UsageDetails::

        There are two variations of QAOA for this problem, constrained and unconstrained:

        **Constrained**

        .. note::

            This method of constrained QAOA was introduced by Hadfield, Wang, Gorman, Rieffel, Venturelli, and Biswas
            in arXiv:1709.03489.

        The Maximum Clique cost Hamiltonian for constrained QAOA is defined as:

        .. math:: H_C \ = \ \displaystyle\sum_{v \in V(G)} Z_{v},

        where :math:`V(G)` is the set of vertices of the input graph, and :math:`Z_i` is the Pauli-Z operator
        applied to the :math:`i`-th
        vertex.

        The returned mixer Hamiltonian is :func:`~qaoa.bit_flip_mixer` applied to :math:`\bar{G}`,
        the complement of the graph.

        .. note::

            **Recommended initialization circuit:**
                Each wire in the :math:`|0\rangle` state.

        **Unconstrained**

        The Maximum Clique cost Hamiltonian for unconstrained QAOA is defined as:

        .. math:: H_C \ = \ 3 \sum_{(i, j) \in E(\bar{G})}
                  (Z_i Z_j \ - \ Z_i \ - \ Z_j) \ + \ \displaystyle\sum_{i \in V(G)} Z_i

        where :math:`V(G)` is the set of vertices of the input graph :math:`G`, :math:`E(\bar{G})` is the set of
        edges of the complement of :math:`G`, and :math:`Z_i` is the Pauli-Z operator applied to the
        :math:`i`-th vertex.

        The returned mixer Hamiltonian is :func:`~qaoa.x_mixer` applied to all wires.

        .. note::

            **Recommended initialization circuit:**
                Even superposition over all basis states.

    """

    if not isinstance(graph, nx.Graph):
        raise ValueError("Input graph must be a nx.Graph, got {}".format(type(graph).__name__))

    if constrained:
        return (bit_driver(graph.nodes, 1), qaoa.bit_flip_mixer(nx.complement(graph), 0))

    cost_h = 3 * edge_driver(nx.complement(graph), ["10", "01", "00"]) + bit_driver(graph.nodes, 1)
    mixer_h = qaoa.x_mixer(graph.nodes)

    return (cost_h, mixer_h)
