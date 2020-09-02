# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for heuristic Pauli graph colouring functions in `grouping/graph_colouring.py`.
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

    def test_graph_colouring(self):
        """Verify that random unweighted, undirected graph's colour is a valid solution."""

        n_qubits = 8
        adjacency_matrix = np.random.randint(2, size=(n_qubits, n_qubits))
        np.fill_diagonal(adjacency_matrix, 0)

        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                adjacency_matrix[j, i] = adjacency_matrix[i, j]

        lf_colouring = largest_first(np.asarray([list(range(n_qubits))]), adjacency_matrix)
        rlf_colouring = recursive_largest_first(
            np.asarray([list(range(n_qubits))]), adjacency_matrix
        )

        assert self.verify_graph_colour_solution(adjacency_matrix, lf_colouring)
        assert self.verify_graph_colour_solution(adjacency_matrix, rlf_colouring)

    def test_trivial_graph_colouring(self):
        """Tests validity of graph colouring solution for a graph with no edges."""

        n_qubits = 8
        adjacency_matrix = np.zeros((n_qubits, n_qubits))

        lf_colouring = largest_first(np.asarray([list(range(n_qubits))]), adjacency_matrix)
        rlf_colouring = recursive_largest_first(
            np.asarray([list(range(n_qubits))]), adjacency_matrix
        )

        assert self.verify_graph_colour_solution(adjacency_matrix, lf_colouring)
        assert self.verify_graph_colour_solution(adjacency_matrix, rlf_colouring)
