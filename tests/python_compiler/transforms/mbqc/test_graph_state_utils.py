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
"""Unit test module for the graph state utils"""

# pylint: disable=wrong-import-position

import pytest

pytestmark = pytest.mark.external

pytest.importorskip("xdsl")
pytest.importorskip("catalyst")

from pennylane.compiler.python_compiler.transforms.mbqc.graph_state_utils import (
    _adj_matrix_generation_helper,
    edge_iter,
    get_graph_state_edges,
    get_num_aux_wires,
    n_vertices_from_packed_adj_matrix,
)
from pennylane.exceptions import CompileError


@pytest.fixture(scope="module", name="mbqc_single_qubit_graph")
def fixture_mbqc_single_qubit_graph():
    """Fixture that returns the densely packed adjacency matrix for the graph state used for
    representing single-qubit gates in the MBQC formalism.

    The graph state is as follows:

        0 -- 1 -- 2 -- 3
    """
    # fmt: off
    packed_adj_matrix = [
    #   0  1  2
        1,       # 1
        0, 1,    # 2
        0, 0, 1  # 3
    ]
    return packed_adj_matrix


@pytest.fixture(scope="module", name="mbqc_cnot_graph")
def fixture_mbqc_cnot_graph():
    """Fixture that returns the densely packed adjacency matrix for the graph state used for
    representing a CNOT gate in the MBQC formalism.

    The graph state is as follows:

        0 -- 1 -- 2 -- 3 -- 4 -- 5
                  |
                  6
                  |
        7 -- 8 -- 9 -- 10 - 11 - 12
    """
    # fmt: off
    packed_adj_matrix = [
    #   0  1  2  3  4  5  6  7  8  9  10 11
        1,
        0, 1,
        0, 0, 1,
        0, 0, 0, 1,
        0, 0, 0, 0, 1,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 1, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
    ]
    return packed_adj_matrix


class TestGraphStateUtils:
    """Unit tests for graph state utils."""

    def test_unsupported_gate(self):
        """Test error raised for unsupported gates"""
        with pytest.raises(ValueError, match="rotxzx is not supported in the MBQC formalism."):
            get_graph_state_edges("rotxzx")

        with pytest.raises(ValueError, match="rotXzx is not supported in the MBQC formalism."):
            get_num_aux_wires("rotXzx")

    def test_adj_matrix_generation_helper(self):
        """Test that error raised for unsupported gates."""
        num_vertices = 4
        edges = [(0, 1), (1, 2), (2, 3)]
        adj_matrix = _adj_matrix_generation_helper(num_vertices, edges)
        assert adj_matrix == [1, 0, 1, 0, 0, 1]

    @pytest.mark.parametrize("n_vertices", range(1, 16))
    def test_n_vertices(self, n_vertices: int):
        """Test that the ``_n_vertices_from_packed_adj_matrix`` function returns correct results
        when given a valid densely packed adjacency matrix as input.

        This test performs the inverse operation of _n_vertices_from_packed_adj_matrix by computing
        the number of elements in the densely packed adjacency matrix from the given number of
        vertices, `n_vertices`, generates a null list of this length, and checks the function output
        given this list is the same as `n_vertices`.
        """
        n_elements = int(n_vertices * (n_vertices - 1) / 2)
        adj_matrix = [0] * n_elements
        n_observed = n_vertices_from_packed_adj_matrix(adj_matrix)

        assert n_observed == n_vertices

    @pytest.mark.parametrize("n", [2, 4, 5, 7, 8, 9])
    def test_n_vertices_raises_on_invalid(self, n):
        """Test that the ``_n_vertices_from_packed_adj_matrix`` function raises a CompileError when
        given an invalid densely packed adjacency matrix as input.
        """
        with pytest.raises(CompileError, match="densely packed adjacency matrix"):
            adj_matrix = [0] * n
            _ = n_vertices_from_packed_adj_matrix(adj_matrix)

    @pytest.mark.parametrize(
        "adj_matrix, expected_edges",
        [
            ([], []),
            ([0], []),
            ([1], [(0, 1)]),
            ([1, 0, 0], [(0, 1)]),
            ([1, 1, 0], [(0, 1), (0, 2)]),
            ([1, 1, 1], [(0, 1), (0, 2), (1, 2)]),
            ([False], []),
            ([True], [(0, 1)]),
        ],
    )
    def test_edge_iter(self, adj_matrix, expected_edges):
        """Test that the ``_edge_iter`` generator function yields correct results when given a valid
        densely packed adjacency matrix as input."""
        edges = list(edge_iter(adj_matrix))
        assert edges == expected_edges

    @pytest.mark.parametrize("n", [2, 4, 5, 7, 8, 9])
    def test_edge_iter_raises_on_invalid(self, n):
        """Test that the ``_edge_iter`` generator function raises a CompileError when given an
        invalid densely packed adjacency matrix as input.
        """
        with pytest.raises(CompileError, match="densely packed adjacency matrix"):
            adj_matrix = [0] * n
            _ = list(edge_iter(adj_matrix))

    def test_n_vertices_mbqc_single_qubit(self, mbqc_single_qubit_graph):
        """Test that the ``_n_vertices_from_packed_adj_matrix`` function correctly determines that
        the number of vertices in the densely packed adjacency matrix for the graph state used for
        representing single-qubit gates in the MBQC formalism is equal to 4.
        """
        n_observed = n_vertices_from_packed_adj_matrix(mbqc_single_qubit_graph)
        assert n_observed == 4

    def test_n_vertices_mbqc_cnot(self, mbqc_cnot_graph):
        """Test that the ``_n_vertices_from_packed_adj_matrix`` function correctly determines that
        the number of vertices in the densely packed adjacency matrix for the graph state used for
        representing a CNOT gate in the MBQC formalism is equal to 13.
        """
        n_observed = n_vertices_from_packed_adj_matrix(mbqc_cnot_graph)
        assert n_observed == 13

    def test_edge_iter_mbqc_single_qubit(self, mbqc_single_qubit_graph):
        """Test that the ``_edge_iter`` generator function applied to the densely packed adjacency
        matrix for the graph state used for representing single-qubit gates in the MBQC formalism
        yields the correct edges.

        For reference, the graph is:

            0 -- 1 -- 2 -- 3
        """
        edges_observed = list(edge_iter(mbqc_single_qubit_graph))

        assert edges_observed == get_graph_state_edges("RZ")

    def test_edge_iter_mbqc_cnot(self, mbqc_cnot_graph):
        """Test that the ``_edge_iter`` generator function applied to the densely packed adjacency
        matrix for the graph state used for representing a CNOT gate in the MBQC formalism yields
        the correct edges.

        For reference, the graph is:

            0 -- 1 -- 2 -- 3 -- 4 -- 5
                      |
                      6
                      |
            7 -- 8 -- 9 -- 10 - 11 - 12
        """
        edges_observed = list(edge_iter(mbqc_cnot_graph))
        assert edges_observed == get_graph_state_edges("CNOT")
