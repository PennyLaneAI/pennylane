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
# pylint: disable=unnecessary-lambda-assignment
from typing import Iterable, Union
import networkx as nx
import rustworkx as rx

import pennylane as qml
from pennylane import qaoa


########################
# Hamiltonian components


def bit_driver(wires: Union[Iterable, qaoa.Wires], b: int):
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
    1 * Z(0) + 1 * Z(1) + 1 * Z(2)
    """
    if b == 0:
        coeffs = [-1 for _ in wires]
    elif b == 1:
        coeffs = [1 for _ in wires]
    else:
        raise ValueError(f"'b' must be either 0 or 1, got {b}")

    ops = [qml.Z(w) for w in wires]
    return qml.Hamiltonian(coeffs, ops)


def edge_driver(graph: Union[nx.Graph, rx.PyGraph], reward: list):
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
         graph (nx.Graph or rx.PyGraph): The graph on which the Hamiltonian is defined
         reward (list[str]): The list of two-bit bitstrings that are assigned a lower energy by the Hamiltonian

    Returns:
        .Hamiltonian:

    **Example**

    >>> import networkx as nx
    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> hamiltonian = qaoa.edge_driver(graph, ["11", "10", "01"])
    >>> print(hamiltonian)
    0.25 * (Z(0) @ Z(1)) + 0.25 * Z(0) + 0.25 * Z(1) + 0.25 * (Z(1) @ Z(2)) + 0.25 * Z(1) + 0.25 * Z(2)

    >>> import rustworkx as rx
    >>> graph = rx.PyGraph()
    >>> graph.add_nodes_from([0, 1, 2])
    >>> graph.add_edges_from([(0, 1,""), (1,2,"")])
    >>> hamiltonian = qaoa.edge_driver(graph, ["11", "10", "01"])
    >>> print(hamiltonian)
    0.25 * (Z(0) @ Z(1)) + 0.25 * Z(0) + 0.25 * Z(1) + 0.25 * (Z(1) @ Z(2)) + 0.25 * Z(1) + 0.25 * Z(2)

    In the above example, ``"11"``, ``"10"``, and ``"01"`` are assigned a lower
    energy than ``"00"``. For example, a quick calculation of expectation values gives us:

    .. math:: \langle 000 | H | 000 \rangle \ = \ 1.5
    .. math:: \langle 100 | H | 100 \rangle \ = \ 0.5
    .. math:: \langle 110 | H | 110\rangle \ = \ -0.5

    In the first example, both vertex pairs are not in ``reward``. In the second example, one pair is in ``reward`` and
    the other is not. Finally, in the third example, both pairs are in ``reward``.

    .. details::
        :title: Usage Details

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

    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph, got {type(graph).__name__}"
        )

    coeffs = []
    ops = []

    is_rx = isinstance(graph, rx.PyGraph)
    graph_nodes = graph.nodes()
    graph_edges = sorted(graph.edge_list()) if is_rx else graph.edges

    # In RX each node is assigned to an integer index starting from 0;
    # thus, we use the following lambda function to get node-values.
    get_nvalue = lambda i: graph_nodes[i] if is_rx else i

    if len(reward) == 0 or len(reward) == 4:
        coeffs = [1 for _ in graph_nodes]
        ops = [qml.Identity(v) for v in graph_nodes]

    else:
        reward = list(set(reward) - {"01"})
        sign = -1

        if len(reward) == 2:
            reward = list({"00", "10", "11"} - set(reward))
            sign = 1

        reward = reward[0]

        if reward == "00":
            for e in graph_edges:
                coeffs.extend([0.25 * sign, 0.25 * sign, 0.25 * sign])
                ops.extend(
                    [
                        qml.Z(get_nvalue(e[0])) @ qml.Z(get_nvalue(e[1])),
                        qml.Z(get_nvalue(e[0])),
                        qml.Z(get_nvalue(e[1])),
                    ]
                )

        if reward == "10":
            for e in graph_edges:
                coeffs.append(-0.5 * sign)
                ops.append(qml.Z(get_nvalue(e[0])) @ qml.Z(get_nvalue(e[1])))

        if reward == "11":
            for e in graph_edges:
                coeffs.extend([0.25 * sign, -0.25 * sign, -0.25 * sign])
                ops.extend(
                    [
                        qml.Z(get_nvalue(e[0])) @ qml.Z(get_nvalue(e[1])),
                        qml.Z(get_nvalue(e[0])),
                        qml.Z(get_nvalue(e[1])),
                    ]
                )

    return qml.Hamiltonian(coeffs, ops)


#######################
# Optimization problems


def maxcut(graph: Union[nx.Graph, rx.PyGraph]):
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
        graph (nx.Graph or rx.PyGraph): a graph defining the pairs of wires on which each term of the Hamiltonian acts

    Returns:
        (.Hamiltonian, .Hamiltonian): The cost and mixer Hamiltonians

    **Example**

    >>> import networkx as nx
    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> cost_h, mixer_h = qml.qaoa.maxcut(graph)
    >>> print(cost_h)
    0.5 * (Z(0) @ Z(1)) + 0.5 * (Z(1) @ Z(2)) + -0.5 * (I(0) @ I(1)) + -0.5 * (I(1) @ I(2))
    >>> print(mixer_h)
    1 * X(0) + 1 * X(1) + 1 * X(2)

    >>> import rustworkx as rx
    >>> graph = rx.PyGraph()
    >>> graph.add_nodes_from([0, 1, 2])
    >>> graph.add_edges_from([(0, 1,""), (1,2,"")])
    >>> cost_h, mixer_h = qml.qaoa.maxcut(graph)
    >>> print(cost_h)
    0.5 * (Z(0) @ Z(1)) + 0.5 * (Z(1) @ Z(2)) + -0.5 * (I(0) @ I(1)) + -0.5 * (I(1) @ I(2))
    >>> print(mixer_h)
    1 * X(0) + 1 * X(1) + 1 * X(2)
    """

    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph, got {type(graph).__name__}"
        )

    is_rx = isinstance(graph, rx.PyGraph)
    graph_nodes = graph.nodes()
    graph_edges = sorted(graph.edge_list()) if is_rx else graph.edges

    # In RX each node is assigned to an integer index starting from 0;
    # thus, we use the following lambda function to get node-values.
    get_nvalue = lambda i: graph_nodes[i] if is_rx else i

    identity_h = qml.Hamiltonian(
        [-0.5 for e in graph_edges],
        [qml.Identity(get_nvalue(e[0])) @ qml.Identity(get_nvalue(e[1])) for e in graph_edges],
    )
    H = edge_driver(graph, ["10", "01"]) + identity_h
    # store the valuable information that all observables are in one commuting group
    H.grouping_indices = [list(range(len(H.ops)))]
    return (H, qaoa.x_mixer(graph_nodes))


def max_independent_set(graph: Union[nx.Graph, rx.PyGraph], constrained: bool = True):
    r"""For a given graph, returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the Maximum Independent Set problem.

    Given some graph :math:`G`, an independent set is a set of vertices such that no pair of vertices in the set
    share a common edge. The Maximum Independent Set problem, is the problem of finding the largest such set.

    Args:
        graph (nx.Graph or rx.PyGraph): a graph whose edges define the pairs of vertices on which each term of the Hamiltonian acts
        constrained (bool): specifies the variant of QAOA that is performed (constrained or unconstrained)

    Returns:
        (.Hamiltonian, .Hamiltonian): The cost and mixer Hamiltonians

    .. details::
        :title: Usage Details

        There are two variations of QAOA for this problem, constrained and unconstrained:

        **Constrained**

        .. note::

            This method of constrained QAOA was introduced by
            `Hadfield, Wang, Gorman, Rieffel, Venturelli, and Biswas (2019) <https://doi.org/10.3390/a12020034>`__.

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

    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph, got {type(graph).__name__}"
        )

    graph_nodes = graph.nodes()

    if constrained:
        cost_h = bit_driver(graph_nodes, 1)
        cost_h.grouping_indices = [list(range(len(cost_h.ops)))]
        return (cost_h, qaoa.bit_flip_mixer(graph, 0))

    cost_h = 3 * edge_driver(graph, ["10", "01", "00"]) + bit_driver(graph_nodes, 1)
    mixer_h = qaoa.x_mixer(graph_nodes)

    # store the valuable information that all observables are in one commuting group
    cost_h.grouping_indices = [list(range(len(cost_h.ops)))]

    return (cost_h, mixer_h)


def min_vertex_cover(graph: Union[nx.Graph, rx.PyGraph], constrained: bool = True):
    r"""Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the Minimum Vertex Cover problem,
    for a given graph.

    To solve the Minimum Vertex Cover problem, we attempt to find the smallest
    `vertex cover <https://en.wikipedia.org/wiki/Vertex_cover>`__ of a graph --- a collection of vertices such that
    every edge in the graph has one of the vertices as an endpoint.

    Args:
        graph (nx.Graph or rx.PyGraph): a graph whose edges define the pairs of vertices on which each term of the Hamiltonian acts
        constrained (bool): specifies the variant of QAOA that is performed (constrained or unconstrained)

    Returns:
        (.Hamiltonian, .Hamiltonian): The cost and mixer Hamiltonians

    .. details::
        :title: Usage Details

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

    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph, got {type(graph).__name__}"
        )

    graph_nodes = graph.nodes()

    if constrained:
        cost_h = bit_driver(graph_nodes, 0)
        cost_h.grouping_indices = [list(range(len(cost_h.ops)))]
        return (cost_h, qaoa.bit_flip_mixer(graph, 1))

    cost_h = 3 * edge_driver(graph, ["11", "10", "01"]) + bit_driver(graph_nodes, 0)
    mixer_h = qaoa.x_mixer(graph_nodes)

    # store the valuable information that all observables are in one commuting group
    cost_h.grouping_indices = [list(range(len(cost_h.ops)))]

    return (cost_h, mixer_h)


def max_clique(graph: Union[nx.Graph, rx.PyGraph], constrained: bool = True):
    r"""Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the Maximum Clique problem,
    for a given graph.

    The goal of Maximum Clique is to find the largest `clique <https://en.wikipedia.org/wiki/Clique_(graph_theory)>`__ of a
    graph --- the largest subgraph such that all vertices are connected by an edge.

    Args:
        graph (nx.Graph or rx.PyGraph): a graph whose edges define the pairs of vertices on which each term of the Hamiltonian acts
        constrained (bool): specifies the variant of QAOA that is performed (constrained or unconstrained)

    Returns:
        (.Hamiltonian, .Hamiltonian): The cost and mixer Hamiltonians

    .. details::
        :title: Usage Details

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

    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph, got {type(graph).__name__}"
        )

    graph_nodes = graph.nodes()
    graph_complement = (
        rx.complement(graph) if isinstance(graph, rx.PyGraph) else nx.complement(graph)
    )

    if constrained:
        cost_h = bit_driver(graph_nodes, 1)
        cost_h.grouping_indices = [list(range(len(cost_h.ops)))]
        return (cost_h, qaoa.bit_flip_mixer(graph_complement, 0))

    cost_h = 3 * edge_driver(graph_complement, ["10", "01", "00"]) + bit_driver(graph_nodes, 1)
    mixer_h = qaoa.x_mixer(graph_nodes)

    # store the valuable information that all observables are in one commuting group
    cost_h.grouping_indices = [list(range(len(cost_h.ops)))]

    return (cost_h, mixer_h)


def max_weight_cycle(graph: Union[nx.Graph, rx.PyGraph, rx.PyDiGraph], constrained: bool = True):
    r"""Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the
    maximum-weighted cycle problem, for a given graph.

    The maximum-weighted cycle problem is defined in the following way (see
    `here <https://1qbit.com/whitepaper/arbitrage/>`__ for more details).
    The product of weights of a subset of edges in a graph is given by

    .. math:: P = \prod_{(i, j) \in E} [(c_{ij} - 1)x_{ij} + 1]

    where :math:`E` are the edges of the graph, :math:`x_{ij}` is a binary number that selects
    whether to include the edge :math:`(i, j)` and :math:`c_{ij}` is the corresponding edge weight.
    Our objective is to maximimize :math:`P`, subject to selecting the :math:`x_{ij}` so that
    our subset of edges composes a `cycle <https://en.wikipedia.org/wiki/Cycle_(graph_theory)>`__.

    Args:
        graph (nx.Graph or rx.PyGraph or rx.PyDiGraph): the directed graph on which the Hamiltonians are defined
        constrained (bool): specifies the variant of QAOA that is performed (constrained or unconstrained)

    Returns:
        (.Hamiltonian, .Hamiltonian, dict): The cost and mixer Hamiltonians, as well as a dictionary
        mapping from wires to the graph's edges

    .. details::
        :title: Usage Details

        There are two variations of QAOA for this problem, constrained and unconstrained:

        **Constrained**

        .. note::

            This method of constrained QAOA was introduced by Hadfield, Wang, Gorman, Rieffel,
            Venturelli, and Biswas in `arXiv:1709.03489 <https://arxiv.org/abs/1709.03489>`__.

        The maximum weighted cycle cost Hamiltonian for unconstrained QAOA is

        .. math:: H_C = H_{\rm loss}.

        Here, :math:`H_{\rm loss}` is a loss Hamiltonian:

        .. math:: H_{\rm loss} = \sum_{(i, j) \in E} Z_{ij}\log c_{ij}

        where :math:`E` are the edges of the graph and :math:`Z_{ij}` is a qubit Pauli-Z matrix
        acting upon the wire specified by the edge :math:`(i, j)` (see :func:`~.loss_hamiltonian`
        for more details).

        The returned mixer Hamiltonian is :func:`~.cycle_mixer` given by

        .. math:: H_M = \frac{1}{4}\sum_{(i, j)\in E}
                \left(\sum_{k \in V, k\neq i, k\neq j, (i, k) \in E, (k, j) \in E}
                \left[X_{ij}X_{ik}X_{kj} +Y_{ij}Y_{ik}X_{kj} + Y_{ij}X_{ik}Y_{kj} - X_{ij}Y_{ik}Y_{kj}\right]
                \right).

        This mixer provides transitions between collections of cycles, i.e., any subset of edges
        in :math:`E` such that all the graph's nodes :math:`V` have zero net flow
        (see the :func:`~.net_flow_constraint` function).

        .. note::

            **Recommended initialization circuit:**
                Your circuit must prepare a state that corresponds to a cycle (or a superposition
                of cycles). Follow the example code below to see how this is done.

        **Unconstrained**

        The maximum weighted cycle cost Hamiltonian for constrained QAOA is defined as:

        .. math:: H_C \ = H_{\rm loss} + 3 H_{\rm netflow} + 3 H_{\rm outflow}.

        The netflow constraint Hamiltonian :func:`~.net_flow_constraint` is given by

        .. math:: H_{\rm netflow} = \sum_{i \in V} \left((d_{i}^{\rm out} - d_{i}^{\rm in})\mathbb{I} -
                \sum_{j, (i, j) \in E} Z_{ij} + \sum_{j, (j, i) \in E} Z_{ji} \right)^{2},

        where :math:`d_{i}^{\rm out}` and :math:`d_{i}^{\rm in}` are
        the outdegree and indegree, respectively, of node :math:`i`. It is minimized whenever a
        subset of edges in :math:`E` results in zero net flow from each node in :math:`V`.

        The outflow constraint Hamiltonian :func:`~.out_flow_constraint` is given by

        .. math:: H_{\rm outflow} = \sum_{i\in V}\left(d_{i}^{out}(d_{i}^{out} - 2)\mathbb{I}
                - 2(d_{i}^{out}-1)\sum_{j,(i,j)\in E}\hat{Z}_{ij} +
                \left( \sum_{j,(i,j)\in E}\hat{Z}_{ij} \right)^{2}\right).

        It is minimized whenever a subset of edges in :math:`E` results in an outflow of at most one
        from each node in :math:`V`.

        The returned mixer Hamiltonian is :func:`~.x_mixer` applied to all wires.

        .. note::

            **Recommended initialization circuit:**
                Even superposition over all basis states.

        **Example**

        First set up a simple graph:

        .. code-block:: python

            import pennylane as qml
            import numpy as np
            import networkx as nx

            a = np.random.random((4, 4))
            np.fill_diagonal(a, 0)
            g = nx.DiGraph(a)

        The cost and mixer Hamiltonian as well as the mapping from wires to edges can be loaded
        using:

        >>> cost, mixer, mapping = qml.qaoa.max_weight_cycle(g, constrained=True)

        Since we are using ``constrained=True``, we must ensure that the input state to the QAOA
        algorithm corresponds to a cycle. Consider the mapping:

        >>> mapping
        {0: (0, 1),
         1: (0, 2),
         2: (0, 3),
         3: (1, 0),
         4: (1, 2),
         5: (1, 3),
         6: (2, 0),
         7: (2, 1),
         8: (2, 3),
         9: (3, 0),
         10: (3, 1),
         11: (3, 2)}

        A simple cycle is given by the edges ``(0, 1)`` and ``(1, 0)`` and corresponding wires
        ``0`` and ``3``. Hence, the state :math:`|100100000000\rangle` corresponds to a cycle and
        can be prepared using :class:`~.BasisState` or simple :class:`~.PauliX` rotations on the
        ``0`` and ``3`` wires.
    """
    if not isinstance(graph, (nx.Graph, rx.PyGraph, rx.PyDiGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph or rx.PyDiGraph, got {type(graph).__name__}"
        )

    mapping = qaoa.cycle.wires_to_edges(graph)

    if constrained:
        cost_h = qaoa.cycle.loss_hamiltonian(graph)
        cost_h.grouping_indices = [list(range(len(cost_h.ops)))]
        return (cost_h, qaoa.cycle.cycle_mixer(graph), mapping)

    cost_h = qaoa.cycle.loss_hamiltonian(graph) + 3 * (
        qaoa.cycle.net_flow_constraint(graph) + qaoa.cycle.out_flow_constraint(graph)
    )
    mixer_h = qaoa.x_mixer(mapping.keys())

    return (cost_h, mixer_h, mapping)
