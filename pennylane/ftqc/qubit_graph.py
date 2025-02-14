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

import warnings
from typing import Union

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
    def graph(self):
        """Gets the underlying qubit graph."""
        return self._graph_qubits

    @property
    def nodes(self):
        """Gets the set of nodes in the underlying qubit graph.

        If the underlying qubit graph has not been initialized, emit a UserWarning and return None.

        Returns:
            networkx.NodeView: The set of nodes, with native support for operations such as
                `len(g.nodes)`, `n in g.nodes`, `g.nodes & h.nodes`, etc. See the networkx
                documentation for more information.
        """
        if self._graph_qubits is None:
            self._warn_uninitialized()
            return None

        return self._graph_qubits.nodes

    @property
    def edges(self):
        """Gets the set of edges in the underlying qubit graph.

        If the underlying qubit graph has not been initialized, emit a UserWarning and return None.

        Returns:
            networkx.EdgeView: The set of edges, with native support for operations such as
                `len(g.edges)`, `e in g.edges`, `g.edges & h.edges`, etc. See the networkx
                documentation for more information.
        """
        if self._graph_qubits is None:
            self._warn_uninitialized()
            return None

        return self._graph_qubits.edges

    @property
    def is_initialized(self):
        """Checks if the underlying qubits have been initialized.

        Returns:
            bool: Returns True if the underlying qubits have been initialized, False otherwise.
        """
        return self._graph_qubits is not None

    def clear(self):
        """Clears the graph of underlying qubits."""
        self._graph_qubits = None

    def connected_qubits(self, node):
        """Returns an iterator over all of the qubits connected to the qubit with label ``node``.

        Args:
            node (node like): The label of a node in the qubit graph.

        Returns:
            iterator: An iterator over all QubitGraph objects connected to the qubit with label
                ``node``.
        """
        if not self.is_initialized:
            self._warn_uninitialized()
            return

        for neighbor in self._graph_qubits.neighbors(node):
            yield self._graph_qubits.nodes[neighbor]["qubits"]

    def _warn_uninitialized(self):
        """Emit a UserWarning when attempting to access an uninitialized graph."""
        warnings.warn("Attempting to access an uninitialized QubitGraph.", UserWarning)

    def _warn_reinitialization(self):
        """Emit a UserWarning when attempting to initialize an already-initialized graph."""
        warnings.warn(
            "Attempting to re-initialize a QubitGraph. If you wish to initialize the underlying "
            "qubits with a new graph structure, you must first call QubitGraph.clear() and then "
            "call the initialization method.",
            UserWarning,
        )

    def init_graph(self, g: nx.Graph):
        """Initialize the QubitGraph's underlying qubits with the given graph.

        Args:
            g (nx.Graph): The graph to use as the QubitGraph's underlying qubits.
        """
        if self.is_initialized:
            self._warn_reinitialization()
            return

        self._graph_qubits = g
        for node in self._graph_qubits:
            # TODO: Having the QubitGraph stored in a dictionary under the key 'qubit' feels clunky
            self._graph_qubits.nodes[node]["qubits"] = QubitGraph()

    def init_graph_2d_grid(self, m: int, n: int):
        """Initialize the QubitGraph's underlying qubits as a 2-dimensional Cartesian grid of other
        QubitGraphs.

        Args:
            m, n (int): The number of rows, m, and columns, n, in the grid.

        Example:

            >>> q = QubitGraph()
            >>> q.init_graph_2d_grid(2, 3)

            This example initializes the underlying qubits as a 2x3 2-dimensional Cartesian grid
            with graph structure and qubit indexing below:

                (0,0) --- (0,1) --- (0,2)
                  |         |         |
                (1,0) --- (1,1) --- (1,2)
        """
        if self.is_initialized:
            self._warn_reinitialization()
            return

        self._graph_qubits = nx.grid_2d_graph(m, n)
        for node in self._graph_qubits.nodes:
            # TODO: Having the QubitGraph stored in a dictionary under the key 'qubit' feels clunky
            self._graph_qubits.nodes[node]["qubits"] = QubitGraph()

    def init_graph_nd_grid(self, dim: Union[list[int], tuple[int]]):
        """Initialize the QubitGraph's underlying qubits as an n-dimensional Cartesian grid of other
        QubitGraphs.

        Args:
            dim (list or tuple of ints): The size of each dimension.

        Example:

            >>> q = QubitGraph()
            >>> q.init_graph_2d_grid(2, 2, 3)

            This example initializes the underlying qubits as a 2x2x3 3-dimensional Cartesian grid
            with graph structure and qubit indexing below:

                      (2,0,0) ------------- (2,0,1)
                     /|                    /|
                   (1,0,0) ------------- (1,0,1)
                  /|  |                 /|  |
                (0,0,0) ------------- (0,0,1)
                |  |  |               |  |  |
                |  |  (2,1,0) --------|--|- (2,1,1)
                |  | /                |  | /
                |  (1,1,0) -----------|- (1,1,1)
                | /                   | /
                (0,1,0) ------------- (0,1,1)

        """
        if self.is_initialized:
            self._warn_reinitialization()
            return

        self._graph_qubits = nx.grid_graph(dim)
        for node in self._graph_qubits.nodes:
            # TODO: Having the QubitGraph stored in a dictionary under the key 'qubit' feels clunky
            self._graph_qubits.nodes[node]["qubits"] = QubitGraph()

    def init_graph_surface_code_17(self):
        r"""Initialize the QubitGraph's underlying qubits as the 17-qubit surface code graph from

            Tomita & Svore, 2014, Low-distance Surface Codes under Realistic Quantum Noise.
                https://arxiv.org/abs/1404.3747.

        This graph structure is commonly referred to as the "ninja star" surface code given its
        shape.

        The nodes are indexed as follows, where 'd' refers to data qubits and 'a' to ancilla qubits:

                          a9
                         /   \
                d0     d1     d2
               /  \   /  \   /
            a10    a11    a12
               \  /   \  /   \
                d3     d4     d5
                  \   /  \   /  \
                   a13    a14    a15
                  /   \  /   \  /
                d6     d7     d9
                  \   /
                   a16
        """
        if self.is_initialized:
            self._warn_reinitialization()
            return

        data_qubits = [("data", i) for i in range(9)]  # 9 data qubits, indexed 0, 1, ..., 8
        anci_qubits = [
            ("anci", i) for i in range(9, 17)
        ]  # 8 ancilla qubits, indexed 9, 10, ..., 16

        self._graph_qubits = nx.Graph()
        self._graph_qubits.add_nodes_from(data_qubits)
        self._graph_qubits.add_nodes_from(anci_qubits)

        # Adjacency list showing the connectivity of each ancilla qubit to its neighbouring data qubits
        anci_adjacency_list = {
            9: [1, 2],
            10: [0, 3],
            11: [0, 1, 3, 4],
            12: [1, 2, 4, 5],
            13: [3, 4, 6, 7],
            14: [4, 5, 7, 8],
            15: [5, 8],
            16: [6, 7],
        }

        for anci_node, data_nodes in anci_adjacency_list.items():
            for data_node in data_nodes:
                self._graph_qubits.add_edge(("anci", anci_node), ("data", data_node))

        for node in self._graph_qubits.nodes:
            # TODO: Having the QubitGraph stored in a dictionary under the key 'qubit' feels clunky
            self._graph_qubits.nodes[node]["qubits"] = QubitGraph()
