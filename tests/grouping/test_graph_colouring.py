# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for heuristic Pauli graph colouring functions in ``grouping/graph_colouring.py``.
"""
import pytest
import numpy as np
from pennylane.grouping.graph_colouring import largest_first, recursive_largest_first


class TestGraphcolouringFunctions:
    """Tests for graph colouring functions."""

    def verify_graph_colour_solution(self, adjacency_matrix, colouring):
        """Verifies if all vertices of the same colour are not connected."""

        for colour in colouring.keys():

            grouping = colouring[colour]
            size_grouping = len(grouping)

            for i in range(size_grouping):
                for j in range(i + 1, size_grouping):
                    vert_i = grouping[i][0]
                    vert_j = grouping[j][0]

                    if adjacency_matrix[vert_i][vert_j] == 1:
                        return False

        return True

    adjacency_matrices = [
        np.array([[0, 1], [1, 0]]),
        np.array(
            [[0, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [1, 0, 0, 0, 1], [0, 0, 1, 1, 0]]
        ),
        np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 1, 1, 0, 1, 1, 0],
                [0, 1, 0, 0, 1, 0, 1, 1],
                [0, 1, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 1, 1, 0],
            ],
        ),
    ]

    @pytest.mark.parametrize("adjacency_matrix", adjacency_matrices)
    def test_graph_colouring(self, adjacency_matrix):
        """Verify that random unweighted undirected graph's colour is a valid solution."""

        n_terms = np.shape(adjacency_matrix)[0]

        for i in range(n_terms):
            for j in range(i + 1, n_terms):
                adjacency_matrix[j, i] = adjacency_matrix[i, j]

        dummy_terms = np.reshape(list(range(n_terms)), (n_terms, 1))
        lf_colouring = largest_first(dummy_terms, adjacency_matrix)
        rlf_colouring = recursive_largest_first(dummy_terms, adjacency_matrix)

        assert self.verify_graph_colour_solution(adjacency_matrix, lf_colouring)
        assert self.verify_graph_colour_solution(adjacency_matrix, rlf_colouring)

    term_counts = list(range(10))

    @pytest.mark.parametrize("n_terms", term_counts)
    def test_trivial_graph_colouring(self, n_terms):
        """Tests validity of graph colouring solution for a graph with no edges."""

        adjacency_matrix = np.zeros((n_terms, n_terms))

        dummy_terms = np.reshape(list(range(n_terms)), (n_terms, 1))
        lf_colouring = largest_first(dummy_terms, adjacency_matrix)
        rlf_colouring = recursive_largest_first(dummy_terms, adjacency_matrix)

        assert self.verify_graph_colour_solution(adjacency_matrix, lf_colouring)
        assert self.verify_graph_colour_solution(adjacency_matrix, rlf_colouring)
