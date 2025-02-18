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


def test_lattice_creation():
    graph = nx.grid_graph([2, 2])
    lattice = Lattice(graph)
    assert isinstance(lattice, Lattice)
    assert lattice._graph == graph


def test_get_neighbors():
    graph = nx.grid_graph([3, 3])
    lattice = Lattice(graph)
    neighbors = list(lattice.get_neighbors((1, 1)))
    assert len(neighbors) == 4


def test_get_nodes():
    graph = nx.grid_graph([2, 2])
    lattice = Lattice(graph)
    nodes = lattice.get_nodes()
    assert len(nodes) == 4


def test_get_edges():
    graph = nx.grid_graph([2, 2])
    lattice = Lattice(graph)
    edges = lattice.get_edges()
    assert len(edges) == 4


def test_get_graph():
    graph = nx.grid_graph([2, 2])
    lattice = Lattice(graph)
    assert lattice.get_graph() == graph


def test_generate_chain_lattice():
    lattice = generate_lattice("chain", [5])
    assert isinstance(lattice, Lattice)
    assert len(lattice.get_nodes()) == 5


def test_generate_rectangle_lattice():
    lattice = generate_lattice("rectangle", [3, 4])
    assert isinstance(lattice, Lattice)
    assert len(lattice.get_nodes()) == 12


def test_generate_cubic_lattice():
    lattice = generate_lattice("cubic", [2, 2, 2])
    assert isinstance(lattice, Lattice)
    assert len(lattice.get_nodes()) == 8


def test_generate_triangle_lattice():
    lattice = generate_lattice("triangle", [3, 4])
    assert isinstance(lattice, Lattice)
    assert len(lattice.get_nodes()) > 0  # Basic check


def test_generate_honeycomb_lattice():
    lattice = generate_lattice("honeycomb", [3, 4])
    assert isinstance(lattice, Lattice)
    assert len(lattice.get_nodes()) > 0  # Basic check


def test_generate_invalid_lattice_shape():
    with pytest.raises(ValueError):
        generate_lattice("invalid_shape", [2, 2])


def test_generate_invalid_dimensions():
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
