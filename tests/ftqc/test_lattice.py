# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the lattice module"""

import networkx as nx
import pytest

from pennylane.ftqc import Lattice, generate_lattice


class TestLattice:
    """Test for the qml.ftqc.Lattice class."""

    def test_lattice_creation_with_graph(self):
        """Test for Lattice object created by a nx.Graph object."""
        graph = nx.Graph([(0, 1), (1, 2)])
        lattice = Lattice("test", graph=graph)
        assert lattice.get_lattice_shape() == "test"
        assert len(lattice.get_nodes()) == 3
        assert len(lattice.get_edges()) == 2

    def test_lattice_creation_with_nodes_and_edges(self):
        """Test for Lattice object created by a list of nodes and edges."""
        nodes = [0, 1, 2]
        edges = [(0, 1), (1, 2)]
        lattice = Lattice("test", nodes=nodes, edges=edges)
        assert len(lattice.get_nodes()) == 3
        assert len(lattice.get_edges()) == 2

    def test_lattice_creation_invalid(self):
        """Test for Lattice object created with lattice name only. ValueError will be raised as neither graph nor nodes/edges provided"""
        with pytest.raises(ValueError):
            Lattice("test")

    def test_get_lattice_shape(self):
        """Test for get_lattice_shape()."""
        lattice = Lattice("test_shape", nx.Graph())
        assert lattice.get_lattice_shape() == "test_shape"

    def test_relabel_nodes(self):
        """Test for nodes relabelling."""
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2])
        graph.add_edges_from([(0, 1), (1, 2)])
        lattice = Lattice("test", graph)
        mapping = {0: "a", 1: "b", 2: "c"}
        lattice.relabel_nodes(mapping)
        assert list(lattice.get_nodes()) == ["a", "b", "c"]

    def test_set_node_attributes(self):
        """Test for adding attributes to nodes in a graph."""
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2])
        graph.add_edges_from([(0, 1), (1, 2)])
        lattice = Lattice("test", graph=graph)
        attributes = {0: {"color": "red"}, 1: {"color": "blue"}, 2: {"color": "black"}}
        lattice.set_node_attributes("color", attributes)
        assert lattice.get_node_attributes("color") == attributes

    def test_set_edge_attributes_dict(self):
        """Test for adding attributes to edges in a graph with provided dict."""
        graph = nx.Graph([(0, 1), (1, 2)])
        lattice = Lattice("test", graph=graph)
        attributes = {(0, 1): {"weight": 10}, (1, 2): {"weight": 20}}
        lattice.set_edge_attributes("weight", attributes)
        assert lattice.get_edge_attributes("weight") == attributes

    def test_set_edge_attributes_scalar(self):
        """Test for adding attributes to edges in a graph with provided scalar."""
        graph = nx.Graph([(0, 1), (1, 2)])
        lattice = Lattice("test", graph=graph)
        lattice.set_edge_attributes("weight", 10)
        expected_attributes = {(0, 1): 10, (1, 2): 10}
        assert nx.get_edge_attributes(lattice._graph, "weight") == expected_attributes

    def test_get_neighbors(self):
        """Test for getting the neighbors of a node."""
        graph = nx.grid_graph([3, 3])
        lattice = Lattice("rectangle", graph)
        neighbors = list(lattice.get_neighbors((1, 1)))
        assert len(neighbors) == 4

    def test_get_nodes(self):
        """Test for getting nodes."""
        nodes = [0, 1, 2]
        edges = [(0, 1), (1, 2)]
        lattice = Lattice("test", nodes=nodes, edges=edges)
        assert set(lattice.get_nodes()) == set(nodes)

    def test_get_edges(self):
        """Test for getting edges."""
        edges = [(0, 1), (1, 2)]
        lattice = Lattice("test", nodes=[0, 1, 2], edges=edges)
        assert set(lattice.get_edges()) == set(edges)
    def test_get_graph(self):
        """Test for getting graph."""
        graph = nx.Graph([(0, 1), (1, 2)])
        lattice = Lattice("test", graph=graph)
        assert lattice.get_graph() is graph


class TestGenerateLattice:
    """Test for generate_lattice method."""
    def test_generate_chain_lattice(self):
        """Test for generate a 1D chain lattice."""
        lattice = generate_lattice("chain", [5])
        assert isinstance(lattice, Lattice)
        assert len(lattice.get_nodes()) == 5

    def test_generate_rectangle_lattice(self):
        """Test for generate a 2D rectangle lattice."""
        lattice = generate_lattice("rectangle", [3, 4])
        assert isinstance(lattice, Lattice)
        assert len(lattice.get_nodes()) == 12

    def test_generate_cubic_lattice(self):
        """Test for generate a 3D cubic lattice."""
        lattice = generate_lattice("cubic", [2, 2, 2])
        assert isinstance(lattice, Lattice)
        assert len(lattice.get_nodes()) == 8

    def test_generate_triangle_lattice(self):
        """Test for generate a 2D triangle lattice."""
        lattice = generate_lattice("triangle", [3, 4])
        assert isinstance(lattice, Lattice)
        assert len(lattice.get_nodes()) > 0

    def test_generate_honeycomb_lattice(self):
        """Test for generate a 3D honeycomb lattice."""
        lattice = generate_lattice("honeycomb", [3, 4])
        assert isinstance(lattice, Lattice)
        assert len(lattice.get_nodes()) > 0

    def test_generate_invalid_lattice_shape(self):
        """Test for an unsupported lattice shape."""
        with pytest.raises(ValueError):
            generate_lattice("invalid_shape", [2, 2])

    def test_generate_invalid_dimensions(self):
        """Test for an incorrect dims input."""
        with pytest.raises(ValueError):
            generate_lattice("chain", [2, 2])
        with pytest.raises(ValueError):
            generate_lattice("rectangle", [2])
        with pytest.raises(ValueError):
            generate_lattice("cubic", [2, 2])
        with pytest.raises(ValueError):
            generate_lattice("triangle", [2, 2, 2])
        with pytest.raises(ValueError):
            generate_lattice("honeycomb", [2, 2, 2])
