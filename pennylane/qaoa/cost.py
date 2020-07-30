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


def MaxCut(graph):
    r"""A method that builds a QAOA cost Hamiltonian corresponding to the MaxCut problem, for a given graph.

    The goal of the MaxCut problem for a particular graph is to find a partition of nodes into two sets, such
    that the number of edges in the graph with endpoints in different sets is maximized. Formally, we wish to
    find the cut of the graph such that the number of edges crossing the cut is maximized
    (see `Cut (graph theory) <https://en.wikipedia.org/wiki/Cut_(graph_theory)>`__).

    The MaxCut Hamiltonian is defined as:

    .. math:: H_C \ = \ \frac{1}{2} \displaystyle\sum_{(i, j) \in E(G)} Z_i Z_j \ - \ \mathbb{I}

    where :math:`G` is some graph and :math:`Z_i` and :math:`Z_j` are the Pauli-Z operators on the :math:`i`-th and
    :math:`j`-th wire respectively.

    As one can check, the states :math:`|01\rangle` and
    :math:`|10\rangle` (representing a cut) both have eigenvalues with respect to :math:`H_C` of :math:`-1`. One can
    also see that :math:`|00\rangle` and :math:`|11\rangle`` (no cut) have eigenvalues of :math:`0`.
    Thus, for a given basis state, with each entry of the state vector representing a node of the graph, and :math:`0` and
    :math:`1` being the labels of the two partitioned sets, the MaxCut cost Hamiltonian effectively counts the number
    of edges crossing the cut and multiplies it by :math:`-1`. Upon minimization, we are left with the basis
    state that yields the maximum cut.

    Recommended mixer Hamiltonian: ~.qaoa.x_mixer

    Recommended initialization circuit: Even superposition over all basis states

    Args:
         graph (nx.Graph) A graph defining the pairs of wires on which each term of the Hamiltonian acts.

    Returns:
        ~.Hamiltonian:

    .. UsageDetails::

        The MaxCut cost Hamiltonian can be called as follows:

        .. code-block:: python

            from pennylane import qaoa
            from networkx import Graph

            graph = Graph([(0, 1), (1, 2)])
            cost_h = qaoa.MaxCut(graph)

        >>> print(cost_h)
        (-0.5) [I0 I1] + (0.5) [Z0 Z1] + (-0.5) [I1 I2] + (0.5) [Z1 Z2]
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

    return qml.Hamiltonian(coeffs, obs)
