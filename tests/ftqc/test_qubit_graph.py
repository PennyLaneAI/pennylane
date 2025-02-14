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

"""Unit tests for the qubit_graph module"""

import networkx as nx
import pytest

from pennylane.ftqc.qubit_graph import QubitGraph


class TestQubitGraphsInitialization:
    """Tests for basic initialization of QubitGraphs."""

    def test_initialization_trivial(self):
        """Trivial case: test that we can initialize a QubitGraph."""
        qubit = QubitGraph()
        assert qubit

    def test_init_graph_user_defined(self):
        """Test that we can initialize a QubitGraph with a user-defined graph of underlying qubits."""
        g = nx.hexagonal_lattice_graph(3, 2)
        qubit = QubitGraph()
        qubit.init_graph(g)

        assert set(qubit.nodes) == set(g.nodes)
        assert set(qubit.edges) == set(g.edges)

        for node in qubit.nodes:
            assert isinstance(qubit.nodes[node]["qubits"], QubitGraph)

    def test_init_graph_2d_grid(self):
        """Test that we can initialize a QubitGraph with a 2D Cartesian grid of underlying qubits.

        For example, for a 2 x 3 grid, we expect a graph with the following structure:

            (0,0) ---- (0,1) ---- (0,2)
              |          |          |
              |          |          |
            (1,0) ---- (1,1) ---- (1,2)
        """
        qubit = QubitGraph()
        m, n = 2, 3
        qubit.init_graph_2d_grid(m, n)

        expected_graph = nx.grid_2d_graph(m, n)
        assert set(qubit.nodes) == set(expected_graph.nodes)
        assert set(qubit.edges) == set(expected_graph.edges)

        for node in qubit.nodes:
            assert isinstance(qubit.nodes[node]["qubits"], QubitGraph)

    def test_init_graph_2d_grid_nested_two_layers(self):
        """Test that we can initialize a QubitGraph with two layers, where each layer is a 2D grid
        of underlying qubits.

        For example, consider a 2 x 3 grid at the first layer, where each underlying qubit at the
        second layer is a 1 x 2 grid. The structure of this graph is:

            (0,0) --------------- (0,1) --------------- (0,2)
              -> (0,0) -- (0,1)     -> (0,0) -- (0,1)     -> (0,0) -- (0,1)
              |                     |                     |
              |                     |                     |
            (1,0) --------------- (1,1) --------------- (1,2)
              -> (0,0) -- (0,1)     -> (0,0) -- (0,1)     -> (0,0) -- (0,1)
        """
        m0, n0 = 2, 3
        m1, n1 = 1, 2

        # Initialize top-layer qubit (layer 0)
        qubit0 = QubitGraph()
        qubit0.init_graph_2d_grid(m0, n0)

        for node in qubit0.nodes:
            # Initialize each next-to-top-layer qubit (layer 1)
            qubit1 = QubitGraph()
            qubit1.init_graph_2d_grid(m1, n1)

            qubit0.nodes[node]["qubits"] = qubit1

        expected_graph0 = nx.grid_2d_graph(m0, n0)
        assert set(qubit0.nodes) == set(expected_graph0.nodes)
        assert set(qubit0.edges) == set(expected_graph0.edges)

        expected_graph1 = nx.grid_2d_graph(m1, n1)
        expected_graph1_nodes_set = set(expected_graph1.nodes)
        expected_graph1_edges_set = set(expected_graph1.edges)

        for node in qubit0.nodes:
            qubit1 = qubit0.nodes[node]["qubits"]
            assert set(qubit1.nodes) == expected_graph1_nodes_set
            assert set(qubit1.edges) == expected_graph1_edges_set

    def test_init_graph_3d_grid(self):
        """Test that we can initialize a QubitGraph with a 3D Cartesian grid of underlying qubits."""
        qubit = QubitGraph()
        n0, n1, n2 = 2, 3, 4
        qubit.init_graph_nd_grid((n0, n1, n2))

        expected_graph = nx.grid_graph((n0, n1, n2))
        assert set(qubit.nodes) == set(expected_graph.nodes)
        assert set(qubit.edges) == set(expected_graph.edges)

        for node in qubit.nodes:
            assert isinstance(qubit.nodes[node]["qubits"], QubitGraph)

    def init_graph_surface_code_17(self):
        """Test that we can initialize a QubitGraph with the underlying qubits following the
        structure of the 17-qubit surface code.
        """
        qubit = QubitGraph()
        qubit.init_graph_surface_code_17()

        # Create the expected graph structure for Surface Code 17
        # This is essentially duplicated from the QubitGraph implementation, but it ensures that
        # accidental changes to the production code will result in a test failure
        data_qubits = [("data", i) for i in range(9)]  # 9 data qubits, indexed 0, 1, ..., 8
        anci_qubits = [
            ("anci", i) for i in range(9, 17)
        ]  # 8 ancilla qubits, indexed 9, 10, ..., 16

        expected_graph = nx.Graph()
        expected_graph.add_nodes_from(data_qubits)
        expected_graph.add_nodes_from(anci_qubits)

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
                expected_graph.add_edge(("anci", anci_node), ("data", data_node))

        assert set(qubit.nodes) == set(expected_graph.nodes)
        assert set(qubit.edges) == set(expected_graph.edges)

        for node in qubit.nodes:
            assert isinstance(qubit.nodes[node]["qubits"], QubitGraph)


class TestQubitGraphsWarnings:
    """Tests for QubitGraph warning messages"""

    def test_access_uninitialized_nodes_warning(self):
        """Test that accessing the nodes property of an uninitialized graph emits a UserWarning."""
        q = QubitGraph()
        with pytest.warns(UserWarning, match="Attempting to access an uninitialized QubitGraph"):
            _ = q.nodes

    def test_access_uninitialized_edges_warning(self):
        """Test that accessing the edges property of an uninitialized graph emits a UserWarning."""
        q = QubitGraph()
        with pytest.warns(UserWarning, match="Attempting to access an uninitialized QubitGraph"):
            _ = q.edges

    def test_reinitialization_warning(self):
        """Test that re-initializing an already-initialized graph emits a UserWarning."""
        q = QubitGraph()
        q.init_graph_2d_grid(2, 2)
        with pytest.warns(UserWarning, match="Attempting to re-initialize a QubitGraph"):
            q.init_graph_2d_grid(2, 3)
