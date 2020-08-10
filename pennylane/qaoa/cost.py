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
This file contains functions that generate QAOA cost Hamiltonians corresponding to
different optimization problems.
"""

import networkx as nx
import pennylane as qml
from pennylane import qaoa
from pennylane.qaoa.mixers import permutation_mixer

def maxcut(graph):
    r"""Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the
    MaxCut problem, for a given graph.

    The goal of the MaxCut problem for a particular graph is to find a partition of nodes into two sets,
    such that the number of edges in the graph with endpoints in different sets is maximized. Formally,
    we wish to find the `cut of the graph <https://en.wikipedia.org/wiki/Cut_(graph_theory)>`__ such
    that the number of edges crossing the cut is maximized.

    The MaxCut Hamiltonian is defined as:

    .. math:: H_C \ = \ \frac{1}{2} \displaystyle\sum_{(i, j) \in E(G)} \big( Z_i Z_j \ - \ \mathbb{I} \big),

    where :math:`G` is some graph, :math:`\mathbb{I}` is the identity, and :math:`Z_i` and :math:`Z_j` are
    the Pauli-Z operators on the :math:`i`-th and :math:`j`-th wire respectively.

    The mixer Hamiltonian returned from :func:`~qaoa.maxcut` is :func:`~qaoa.x_mixer` applied to all wires.

    .. note::

        **Recommended initialization circuit:**
            Even superposition over all basis states

    Args:
        graph (nx.Graph): The graph on which MaxCut is performed. This defines the pairs of wires
                          on which each term of the Hamiltonian acts.

    Returns:
        (.Hamiltonian, .Hamiltonian):

    **Example**

    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> cost_h, mixer_h = qml.qaoa.maxcut(graph)
    >>> print(cost_h)
    >>> print(mixer_h)
    (-0.5) [I0 I1] + (0.5) [Z0 Z1] + (-0.5) [I1 I2] + (0.5) [Z1 Z2]
    (1.0) [X0] + (1.0) [X1] + (1.0) [X2]
    """

    if not isinstance(graph, nx.Graph):
        raise ValueError(
            "Input graph must be a nx.Graph object, got {}".format(type(graph).__name__)
        )

    edges = graph.edges

    coeffs = []
    obs = []

    for node1, node2 in edges:

        obs.append(qml.Identity(node1) @ qml.Identity(node2))
        coeffs.append(-0.5)

        obs.append(qml.PauliZ(node1) @ qml.PauliZ(node2))
        coeffs.append(0.5)

    return (qml.Hamiltonian(coeffs, obs), qaoa.x_mixer(graph.nodes))


def travelling_salesman(graph, node_ordering, wire_matrix):
    r"""Returns the QAOA cost Hamiltonian and the recommended mixer corresponding the the Travelling
    Salesman problem.

    The cost Hamiltonian for TSP is defined as:

    .. math:: H_C \ = \ \displaystyle\sum_{\{u, \v} \in E} d_{u, v} \displaystyle\sum_{j = 1}^{n}
              ( Z_{u, j} Z_{v, j+1} \ + \ Z_{v, j} Z_{u, j+1})

    where :math:`d_{u, v}` is the distance between vertices :math:`u` and :math:`v` and :math:`Z_{v, t}` is
    the Pauli-Z operator acting on the wire representing vertex :math:`v` at time-step :math:`t`.
    See [`arXiv:1805.03265 <https://arxiv.org/abs/1805.03265>`__] for more information.

    The mixer Hamiltonian returned from :func:`~qaoa.travelling_salesman` is `~qaoa.plus_minus_mixer`,
    applied between nodes of the graph, separated by one temporal step.

    .. note::

        **Recommended initialization circuit:**
            Even superposition over all basis states

    Args:
        graph (nx.Graph): The graph on which TSP is being performed. Note that this graph must be weighted and
                          complete (as a distance between any two pairs of "cities" must be defined). In addition, the
                          nodes of the graph's nodes must be labelled as 0, 1, 2, ..., k.
        wire_matrix (array): A two-dimensional array that encodes the wire labels used for TSP (see Usage Details for
                             more information).

    Returns:
        (.Hamiltonian, .Hamiltonian):

    **Example**

    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> cost_h, mixer_h = qml.qaoa.maxcut(graph)
    >>> print(cost_h)
    >>> print(mixer_h)
    (-0.5) [I0 I1] + (0.5) [Z0 Z1] + (-0.5) [I1 I2] + (0.5) [Z1 Z2]
    (1.0) [X0] + (1.0) [X1] + (1.0) [X2]

    .. UsageDetails::

        **Qubit Embedding Scheme**

        In order to

        **The TSP Graph and Cost Hamiltonian**

        **The TSP Mixer**
    """

    # Constructs the cost Hamiltonian

    coeffs = []
    obs = []

    for u, v, d in graph.edges(data=True):

        u_val = node_ordering[u]
        v_val = node_ordering[v]

        for j in range(0, len(graph.nodes)-1):

            term1 = qml.PauliZ(wires=wire_matrix[j][u_val]) @ qml.PauliZ(wires=wire_matrix[j+1][v_val])
            term2 = qml.PauliZ(wires=wire_matrix[j][v_val]) @ qml.PauliZ(wires=wire_matrix[j+1][u_val])

            coeffs.extend([d['weight'], d['weight']])
            obs.extend([term1, term2])

    cost_h = qml.Hamiltonian(coeffs, obs)

    # Constructs the mixer Hamiltonian

    mixer_h = None

    for u, v, d in graph.edges(data=True):

        u_val = node_ordering[u]
        v_val = node_ordering[v]

        for j in range(0, len(graph.nodes) - 1):

            mixer_term = permutation_mixer(
                ["-++-", "+--+"],
                [1, 1],
                    wires=[wire_matrix[j][u_val], wire_matrix[j+1][u_val], wire_matrix[j][v_val], wire_matrix[j+1][v_val]]
                )

            if mixer_h is None:
                mixer_h = mixer_term
            else:
                mixer_h += mixer_term

    # Returns the cost and mixer

    return cost_h, mixer_h



