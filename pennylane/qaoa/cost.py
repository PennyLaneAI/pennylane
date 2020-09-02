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

    Given some graph, :math:`G`, this method will return a Hamiltonian that assigns
    lower energies to two-bit bitstrings supplied in ``reward``. Each bitstring corresponds
    to the state of some edge in :math:`G`, determined by the states of its endpoints.

    See usage details for more information.

    Args:
         graph (nx.Graph): The graph on which the Hamiltonian is defined
         reward (list[str]): The list of two-bit bitstrings that are assigned lower energy by the Hamiltonian

    Returns:
        .Hamiltonian

    **Example**

    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> hamiltonian = qaoa.edge_driver(graph, ["11", "10", "01"])
    >>> print(hamiltonian)
    (0.25) [Z0 Z1] + (0.25) [Z0] + (0.25) [Z1] + (0.25) [Z1 Z2] + (0.25) [Z2]

    ..UsageDetails::

        The goal of many combinatorial problems that can be solved with QAOA is to
        find a `Graph colouring <https://en.wikipedia.org/wiki/Graph_coloring>`__ of some supplied
        graph :math:`G` that minimizes some cost function.

        It is oftentimes natural to consider the class
        of graph colouring problems that only admit two colours, as we can easily encode these two colours
        using the :math:`|1\rangle` and :math:`|0\rangle` states of qubits. Therefore, given
        some graph :math:`G`, each edge of the graph can be described by a pair of qubits, :math:`|00\rangle`,
        :math:`01\rangle`, :math:`|10\rangle`, or :math:`|11\rangle`, corresponding to the colourings of its endpoints.

        When constructing QAOA cost functions, one must "penalize" certain states of the graph, and "reward"
        others, by assigning higher and lower energies to these respective configurations. Given a set of vertex-colour
        pairs (which describe an edge), the `edge_driver` method will output a Hamiltonian that rewards the edge-stats
        in the set, and penalizes the others. For example, given the set:
        :math:`\{|00\rangle, \ |01\rangle, \ |10\rangle}` and the graph :math:`G`,
        the `edge_driver` method will output the following Hamiltonian:

        ..math:: H \ = \ \frac{1}{4} \displaystyle\sum_{(i, j) \in E(G)} \big( Z_{i} Z_{j} \ - \ Z_{i} \ - \ Z_{j} \big)

        where :math:`E(G)` is the set of edges of :math:`G` and :math:`Z_i` is the Pauli-Z operator acting on the
        :math:`i`-th wire. As can be checked, this Hamiltonian assigns an energy of :math:`-1/4` to the states
        :math:`|00\rangle`, :math:`|01\rangle` and :math:`|10\rangle`, and an energy of :math:`3/4` to the state
        :math:`|11\rangle`.

        .. Note::

            If either of the states :math:`\01\rangle` or :math:`|10\rangle` is contained in ``reward``, then so too
            must :math:`|10\rangle` or :math:`|01\rangle`, respectively. Within a graph, there is no notion of "order"
            of edge endpoints, so these two states are effectively the same.
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
    H = edge_driver(graph, ["10", "01"]) + identity_h
    return (H, qaoa.x_mixer(graph.nodes))
