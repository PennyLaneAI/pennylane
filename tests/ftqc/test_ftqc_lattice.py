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
# pylint: disable=no-name-in-module, no-self-use, protected-access
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
        assert lattice.shape == "test"
        assert len(lattice.nodes) == 3
        assert len(lattice.edges) == 2

    def test_lattice_creation_with_nodes_and_edges(self):
        """Test for Lattice object created by a list of nodes and edges."""
        nodes = [0, 1, 2]
        edges = [(0, 1), (1, 2)]
        lattice = Lattice("test", nodes=nodes, edges=edges)
        assert len(lattice.nodes) == 3
        assert len(lattice.edges) == 2

    def test_lattice_creation_invalid(self):
        """Test for Lattice object created with lattice name only. ValueError will be raised as neither graph nor nodes/edges provided"""
        with pytest.raises(ValueError, match="Neither a networkx Graph object nor nodes together"):
            Lattice("test")

    def test_get_lattice_shape(self):
        """Test for get_lattice_shape()."""
        lattice = Lattice("test_shape", nx.Graph())
        assert lattice.shape == "test_shape"

    def test_get_neighbors(self):
        """Test for getting the neighbors of a node."""
        graph = nx.grid_graph([3, 3])
        lattice = Lattice("rectangle", graph)
        assert set(lattice.get_neighbors((1, 1))) == set(graph.neighbors((1, 1)))

    def test_get_nodes(self):
        """Test for getting nodes."""
        nodes = [0, 1, 2]
        edges = [(0, 1), (1, 2)]
        lattice = Lattice("test", nodes=nodes, edges=edges)
        assert set(lattice.nodes) == set(nodes)

    def test_get_edges(self):
        """Test for getting edges."""
        edges = [(0, 1), (1, 2)]
        lattice = Lattice("test", nodes=[0, 1, 2], edges=edges)
        assert set(lattice.edges) == set(edges)

    def test_get_graph(self):
        """Test for getting graph."""
        graph = nx.Graph([(0, 1), (1, 2)])
        lattice = Lattice("test", graph=graph)
        assert lattice.graph is graph


class TestGenerateLattice:
    """Test for generate_lattice method."""

    def test_generate_chain_lattice(self):
        """Test to generate a 1D chain lattice."""
        lattice = generate_lattice([5], "chain")
        assert isinstance(lattice, Lattice)
        assert len(lattice.nodes) == 5

    def test_generate_rectangle_lattice(self):
        """Test to generate a 2D rectangle lattice."""
        lattice = generate_lattice([3, 4], "rectangle")
        assert isinstance(lattice, Lattice)
        assert len(lattice.nodes) == 12

    def test_generate_cubic_lattice(self):
        """Test to generate a 3D cubic lattice."""
        lattice = generate_lattice([2, 2, 2], "cubic")
        assert isinstance(lattice, Lattice)
        assert len(lattice.nodes) == 8

    def test_generate_triangle_lattice(self):
        """Test to generate a 2D triangle lattice."""
        lattice = generate_lattice([3, 4], "triangle")
        assert isinstance(lattice, Lattice)
        assert len(lattice.nodes) == 12

    def test_generate_honeycomb_lattice(self):
        """Test to generate a 2D honeycomb lattice."""
        lattice = generate_lattice([1, 4], "honeycomb")
        assert isinstance(lattice, Lattice)
        assert len(lattice.nodes) == 18

    def test_generate_invalid_lattice_shape(self):
        """Test for an unsupported lattice shape."""
        with pytest.raises(ValueError):
            generate_lattice([2, 2], "invalid_shape")

    @pytest.mark.parametrize(
        "dims, shape",
        [
            ([2, 2], "chain"),
            ([2], "rectangle"),
            ([2, 2], "cubic"),
            ([2, 2, 2], "triangle"),
            ([2], "honeycomb"),
        ],
    )
    def test_generate_invalid_dimensions(self, dims, shape):
        """Test for an incorrect dims input."""
        with pytest.raises(ValueError, match="the length of dims should be"):
            generate_lattice(dims, shape)
