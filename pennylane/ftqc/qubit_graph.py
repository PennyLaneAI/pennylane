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

"""This module contains the data structures to represent the qubit memory model in the FTQC/MBQC
framework, as well as the API to access and manipulate these data structures.
"""

import networkx as nx


class QubitGraph:
    """A class to represent nested graphs of qubits in the FTQC/MBQC framework.

    A QubitGraph is a qubit that contains a graph of underlying qubits, where each underlying qubit
    is itself a QubitGraph. This representation allows for nesting of lower-level qubits with
    arbitrary depth to allow easy insertion of arbitrarily many levels of abstractions between
    logical qubits and physical qubits.

    This is a WIP! Still to do:

        * How to represent operations on qubits?
        * Should be able to broadcast operations to underlying qubits
        * Tensor-like indexing and slicing.
    """

    def __init__(self):
        self._graph_qubits = None  # The qubits underlying (nested within) the current qubit

    @property
    def nodes(self):
        """Gets the set of nodes in the underlying qubit graph.

        Returns:
            networkx.NodeView: The set of nodes, with native support for operations such as
                `len(g.nodes)`, `n in g.nodes`, `g.nodes & h.nodes`, etc. See the networkx
                documentation for more information.
        """
        return self._graph_qubits.nodes

    @property
    def edges(self):
        """Gets the set of edges in the underlying qubit graph.

        Returns:
            networkx.EdgeView: The set of edges, with native support for operations such as
                `len(g.edges)`, `e in g.edges`, `g.edges & h.edges`, etc. See the networkx
                documentation for more information.
        """
        return self._graph_qubits.edges

    def init_graph_2d_grid(self, m: int, n: int):
        """Initialize the QubitGraph's underlying qubits as a 2-dimensional m-by-n grid of other
        QubitGraphs.

        Args:
            m, n (int): The number of rows, m, and columns, n, in the grid. The nodes are indexed as
                (0, 0), (0, 1), ..., (0, n-1), (1, 0), ..., (m-1, n-1).
        """
        self._graph_qubits = nx.grid_2d_graph(m, n)
        for node in self._graph_qubits.nodes:
            # TODO: Having the QubitGraph stored in a dictionary under the key 'qubit' feels clunky
            self._graph_qubits.nodes[node]["qubits"] = QubitGraph()
