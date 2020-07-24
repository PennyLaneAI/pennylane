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

    Recommended mixer Hamiltonian: pennylane.qaoa.x_mixer

    Recommended initialization circuit: pennylane.templates.even_superposition

    Args:
         graph (nx.Graph) A graph defining the pairs of indices of wires on which each term of the Hamiltonian acts.

    Returns:
        ~.Hamiltonian:
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
        coeffs.append(0.5)

        obs.append(qml.PauliZ(node1) @ qml.PauliZ(node2))
        coeffs.append(-0.5)

    return qml.Hamiltonian(coeffs, obs)
