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
This file contains functions that generate cost Hamiltonians corresponding to
different optimization problems, for use in QAOA workflows.
"""
import pennylane as qml
import networkx
from .utils import check_iterable_graph
from collections.abc import Iterable

def MaxCut(graph):
    r"""A method that build the MaxCut Hamiltonian for a given graph.

    The MaxCut problem can be formulated as follows: given some graph :math:`G`, what is the colouring
    :math:`C : V(G) \ \rightarrow \ \{0, \ 1\}` of :math:`G` that yields the maximum number of node pairs
    :math:`(i, \ j) \ \in \ E` such that :math:`C(i) \ \neq \ C(j)`?

    More simply put, if we are tasked with "painting" each node in a graph either black or white, what is
    the configuration of colourings that yields the maximum number of edges with different coloured endpoints?

    This problem is of some practical relevance in physics and circuit design, but within quantum computing,
    MaxCut is an easy problem to implement and solve with QAOA, making it useful for benchmarking NISQ quantum devices.

    Args:
         graph (networkx.Graph) A graph defining the pairs of qubits on which each term of the Hamiltonian acts.

    Returns:
         A qml.Hamiltonian encoding the MaxCut Hamiltonian.
    """

    ##############
    # Input checks

    if isinstance(graph, networkx.Graph):
        graph = graph.edges

    elif isinstance(graph, Iterable):
        check_iterable_graph(graph)

    else:
        raise ValueError(
            "Inputted graph must be a networkx.Graph object or Iterable, got {}".format(type(graph).__name__))

    ##############

    coeffs = []
    obs = []

    for e in graph:

        obs.append(qml.Identity(e[0]) @ qml.Identity(e[1]))
        coeffs.append(0.5)

        obs.append(qml.PauliZ(e[0]) @ qml.PauliZ(e[1]))
        coeffs.append(-0.5)

    return qml.Hamiltonian(coeffs, obs)
