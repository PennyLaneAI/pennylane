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

    def test_initialization_constructor(self):
        """Test that we can initialize a QubitGraph with a user-defined graph of underlying qubits
        using the QubitGraph constructor."""
        g = nx.hexagonal_lattice_graph(3, 2)
        qubit = QubitGraph(g)

        assert set(qubit.nodes) == set(g.nodes)
        assert set(qubit.edges) == set(g.edges)

        for node in qubit.nodes:
            assert isinstance(qubit[node], QubitGraph)

    def test_init_graph(self):
        """Test that we can initialize a QubitGraph with a user-defined graph of underlying qubits."""
        g = nx.hexagonal_lattice_graph(3, 2)
        qubit = QubitGraph()
        qubit.init_graph(g)

        assert set(qubit.nodes) == set(g.nodes)
        assert set(qubit.edges) == set(g.edges)

        for node in qubit.nodes:
            assert isinstance(qubit[node], QubitGraph)

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
            assert isinstance(qubit[node], QubitGraph)

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

            qubit0[node] = qubit1

        expected_graph0 = nx.grid_2d_graph(m0, n0)
        assert set(qubit0.nodes) == set(expected_graph0.nodes)
        assert set(qubit0.edges) == set(expected_graph0.edges)

        expected_graph1 = nx.grid_2d_graph(m1, n1)
        expected_graph1_nodes_set = set(expected_graph1.nodes)
        expected_graph1_edges_set = set(expected_graph1.edges)

        for node in qubit0.nodes:
            qubit1 = qubit0[node]
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
            assert isinstance(qubit[node], QubitGraph)

    def test_init_graph_surface_code_17(self):
        """Test that we can initialize a QubitGraph with the underlying qubits following the
        structure of the 17-qubit surface code.
        """
        qubit = QubitGraph()
        qubit.init_graph_surface_code_17()

        # Create the expected graph structure for Surface Code 17
        # This is essentially duplicated from the QubitGraph implementation, but it ensures that
        # accidental changes to the production code will result in a test failure
        data_qubits = [("data", i) for i in range(9)]  # 9 data qubits, indexed 0, 1, ..., 8
        aux_qubits = [
            ("aux", i) for i in range(9, 17)
        ]  # 8 auxiliary qubits, indexed 9, 10, ..., 16

        expected_graph = nx.Graph()
        expected_graph.add_nodes_from(data_qubits)
        expected_graph.add_nodes_from(aux_qubits)

        # Adjacency list for connectivity of each auxiliary qubit to its neighbouring data qubits
        aux_adjacency_list = {
            9: [1, 2],
            10: [0, 3],
            11: [0, 1, 3, 4],
            12: [1, 2, 4, 5],
            13: [3, 4, 6, 7],
            14: [4, 5, 7, 8],
            15: [5, 8],
            16: [6, 7],
        }

        for aux_node, data_nodes in aux_adjacency_list.items():
            for data_node in data_nodes:
                expected_graph.add_edge(("aux", aux_node), ("data", data_node))

        assert set(qubit.nodes) == set(expected_graph.nodes)
        assert set(qubit.edges) == set(expected_graph.edges)

        for node in qubit.nodes:
            assert isinstance(qubit[node], QubitGraph)

    def test_init_graph_with_invalid_type_raises_type_error(self):
        """Test that attempting to initialize a graph with an invalid graph type raises a TypeError."""

        class NotAGraph:
            pass

        class SomethingWithOnlyNodes:
            def __init__(self):
                self.nodes = []

        class SomethingWithOnlyEdges:
            def __init__(self):
                self.edges = []

        # Test initialization with constructor
        with pytest.raises(TypeError, match="QubitGraph requires a graph-like input"):
            invalid_graph = NotAGraph()
            _ = QubitGraph(invalid_graph)

        with pytest.raises(TypeError, match="QubitGraph requires a graph-like input"):
            invalid_graph = SomethingWithOnlyNodes()
            _ = QubitGraph(invalid_graph)

        with pytest.raises(TypeError, match="QubitGraph requires a graph-like input"):
            invalid_graph = SomethingWithOnlyEdges()
            _ = QubitGraph(invalid_graph)

        # Test initialization with `init_graph` method
        with pytest.raises(TypeError, match="QubitGraph requires a graph-like input"):
            invalid_graph = NotAGraph()
            q = QubitGraph()
            q.init_graph(invalid_graph)

        with pytest.raises(TypeError, match="QubitGraph requires a graph-like input"):
            invalid_graph = SomethingWithOnlyNodes()
            q = QubitGraph()
            q.init_graph(invalid_graph)

        with pytest.raises(TypeError, match="QubitGraph requires a graph-like input"):
            invalid_graph = SomethingWithOnlyEdges()
            q = QubitGraph()
            q.init_graph(invalid_graph)

        with pytest.raises(TypeError, match="QubitGraph requires a graph-like input, got NoneType"):
            q = QubitGraph()
            q.init_graph(None)


class TestQubitGraphOperations:
    """Tests for operations on a QubitGraph."""

    def test_clear(self):
        """Test basic usage of the ``QubitGraph.clear`` method."""
        q = QubitGraph()
        assert q.graph is None

        q.init_graph(nx.grid_2d_graph(2, 1))
        assert q.graph is not None

        q.clear()
        assert q.graph is None

    def test_connected_qubits(self):
        """Test basic usage of the ``QubitGraph.connected_qubits`` method."""
        q = QubitGraph()
        q.init_graph(nx.grid_2d_graph(2, 2))

        assert set(q.connected_qubits((0, 0))) == set([q[(0, 1)], q[(1, 0)]])
        assert set(q.connected_qubits((0, 1))) == set([q[(0, 0)], q[(1, 1)]])
        assert set(q.connected_qubits((1, 0))) == set([q[(0, 0)], q[(1, 1)]])
        assert set(q.connected_qubits((1, 1))) == set([q[(0, 1)], q[(1, 0)]])


class TestQubitGraphIterationMethods:
    """Tests for iteration method on a QubitGraph."""

    def test_iterate_nodes(self):
        """Test that we can iterate over the nodes of a QubitGraph."""
        q = QubitGraph()
        q.init_graph_nd_grid((2,))

        for i, node in enumerate(q.nodes):
            assert node == i

    def test_iterate_edges(self):
        """Test that we can iterate over the edges of a QubitGraph."""
        q = QubitGraph()
        q.init_graph_nd_grid((2,))

        for i, edge in enumerate(q.edges):
            assert edge == (i, i + 1)

    def test_wrapping_in_container(self):
        """Test that wrapping a QubitGraph in a sequential container does not implicitly iterate
        over the underlying qubit graph.
        """
        q = QubitGraph()
        q.init_graph_nd_grid((2,))

        q_tuple = tuple(q)
        assert len(q_tuple) == 1
        assert q_tuple[0] is q


class TestQubitGraphIndexing:
    """Tests for indexing operations on a QubitGraph."""

    def test_linear_indexing(self):
        """Test basic linear indexing."""
        qubit = QubitGraph()

        n = 3
        qubit.init_graph_nd_grid((n,))

        for i in range(n):
            q = qubit[i]
            assert isinstance(q, QubitGraph)

    def test_linear_indexing_nested(self):
        """Test basic linear indexing in a nested QubitGraph."""
        qubit0 = QubitGraph()

        n0, n1 = 3, 2
        qubit0.init_graph_nd_grid((n0,))

        for node in qubit0.nodes:
            q1 = QubitGraph()
            q1.init_graph_nd_grid((n1,))

            qubit0[node] = q1

        for i in range(n0):
            for j in range(n1):
                q = qubit0[i][j]
                assert isinstance(q, QubitGraph)

    def test_linear_indexing_slice(self):
        """Test basic linear indexing using slices."""
        qubit = QubitGraph()
        qubit.init_graph_nd_grid((4,))

        qubit_slice_02 = qubit[0:2]
        assert len(qubit_slice_02) == 2
        assert qubit_slice_02[0] is qubit[0]
        assert qubit_slice_02[1] is qubit[1]

        qubit_slice_13 = qubit[1:3]
        assert len(qubit_slice_13) == 2
        assert qubit_slice_13[0] is qubit[1]
        assert qubit_slice_13[1] is qubit[2]

        qubit_slice_042 = qubit[0:4:2]
        assert len(qubit_slice_042) == 2
        assert qubit_slice_042[0] is qubit[0]
        assert qubit_slice_042[1] is qubit[2]

        qubit_slice_20m1 = qubit[2:0:-1]
        assert len(qubit_slice_20m1) == 2
        assert qubit_slice_20m1[0] is qubit[2]
        assert qubit_slice_20m1[1] is qubit[1]

    def test_assignment(self):
        """Test assignment of a new QubitGraph object at a given index."""
        qubit = QubitGraph()
        qubit.init_graph_nd_grid((2,))

        new_qubit = QubitGraph()
        new_qubit.init_graph_nd_grid((2, 2))

        qubit[0] = new_qubit
        assert qubit[0] is new_qubit
        assert qubit[0].is_initialized
        assert qubit[0].nodes is not None
        assert not qubit[1].is_initialized

    def test_invalid_index_raises_keyerror(self):
        """Test that accessing a QubitGraph with an invalid index raises a KeyError."""
        qubit = QubitGraph()
        qubit.init_graph_nd_grid((2,))

        invalid_index = 4
        with pytest.raises(KeyError, match=f"{invalid_index}"):
            _ = qubit[4]

    def test_invalid_assignment_raises_typeerror(self):
        """Test that attempting to assign a value that is not a QubitGraph to a node in the graph
        raises a TypeError."""
        qubit = QubitGraph()
        qubit.init_graph_nd_grid((2,))

        with pytest.raises(TypeError, match="item assignment type must also be a QubitGraph"):
            qubit[0] = 42

        with pytest.raises(TypeError, match="item assignment type must also be a QubitGraph"):
            qubit[0] = nx.Graph()


class TestQubitGraphRepresentation:
    """Test for representing a QubitGraph as a string."""

    def test_representation(self):
        """Test basic conversion of a QubitGraph to its string representation."""
        q = QubitGraph()
        assert str(q) == "QubitGraph"


class TestQubitGraphsWarnings:
    """Tests for QubitGraph warning messages."""

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

    def test_access_uninitialized_connected_qubits_warning(self):
        """Test that accessing the connected qubits of a qubit with an uninitialized graph emits a
        UserWarning.
        """
        q = QubitGraph()
        with pytest.warns(UserWarning, match="Attempting to access an uninitialized QubitGraph"):
            for connected_q in q.connected_qubits(0):
                _ = connected_q

    def test_access_uninitialized_subscript_warning(self):
        """Test that accessing an element of a qubit with the subscript operator with an
        uninitialized graph emits a UserWarning.
        """
        q = QubitGraph()
        with pytest.warns(UserWarning, match="Attempting to access an uninitialized QubitGraph"):
            _ = q[0]

    def test_assignment_uninitialized_subscript_warning(self):
        """Test that assigning an element of a qubit with an uninitialized graph emits a
        UserWarning.
        """
        q = QubitGraph()
        with pytest.warns(UserWarning, match="Attempting to access an uninitialized QubitGraph"):
            q[0] = QubitGraph()

    def test_reinitialization_warning(self):
        """Test that re-initializing an already-initialized graph emits a UserWarning."""
        q = QubitGraph()
        q.init_graph_2d_grid(2, 2)

        with pytest.warns(UserWarning, match="Attempting to re-initialize a QubitGraph"):
            g = nx.Graph()
            q.init_graph(g)

        with pytest.warns(UserWarning, match="Attempting to re-initialize a QubitGraph"):
            q.init_graph_2d_grid(2, 3)

        with pytest.warns(UserWarning, match="Attempting to re-initialize a QubitGraph"):
            q.init_graph_nd_grid((3,))

        with pytest.warns(UserWarning, match="Attempting to re-initialize a QubitGraph"):
            q.init_graph_surface_code_17()

    def test_unsupported_graph_type_warning(self):
        """Test that initializing a QubitGraph with a graph-like object that is not a networkx.Graph
        object emits a UserWarning.
        """

        class CustomGraph:
            def __init__(self):
                self.nodes = []
                self.edges = []

        g = CustomGraph()

        # Test that constructor raises warning
        with pytest.warns(
            UserWarning, match="QubitGraph expects an input graph of type 'networkx.Graph'"
        ):
            _ = QubitGraph(g)

        # Test that `init_graph` method raises warning
        q = QubitGraph()
        with pytest.warns(
            UserWarning, match="QubitGraph expects an input graph of type 'networkx.Graph'"
        ):
            q.init_graph(g)
