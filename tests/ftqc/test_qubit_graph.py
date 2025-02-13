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

import pytest
import networkx as nx

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

        assert len(qubit.nodes) == m * n
        assert len(qubit.edges) == m * (n - 1) + n * (m - 1)

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

        assert len(qubit0.nodes) == m0 * n0
        assert len(qubit0.edges) == m0 * (n0 - 1) + n0 * (m0 - 1)

        for node in qubit0.nodes:
            qubit1 = qubit0.nodes[node]["qubits"]
            assert len(qubit1.nodes) == m1 * n1
            assert len(qubit1.edges) == m1 * (n1 - 1) + n1 * (m1 - 1)

    def test_init_graph_3d_grid(self):
        """Test that we can initialize a QubitGraph with a 3D Cartesian grid of underlying qubits."""
        qubit = QubitGraph()
        n0, n1, n2 = 2, 3, 4
        qubit.init_graph_nd_grid((n0, n1, n2))

        assert len(qubit.nodes) == n0 * n1 * n2

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
