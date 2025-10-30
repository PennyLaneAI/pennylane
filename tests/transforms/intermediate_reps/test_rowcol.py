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
"""Tests for the CNOT routing algorithm ROWCOL."""
# pylint: disable=no-self-use
import networkx as nx
import numpy as np
import pytest

from pennylane import CNOT
from pennylane.tape import QuantumScript
from pennylane.transforms import parity_matrix, rowcol
from pennylane.transforms.intermediate_reps import postorder_traverse, preorder_traverse
from pennylane.transforms.intermediate_reps.rowcol import _rowcol_parity_matrix

path_graph_4 = nx.path_graph(4)
binary_graph_3 = nx.balanced_tree(2, 3)
ternary_graph_2 = nx.balanced_tree(3, 2)


class TestTreeTraversal:
    """Tests for tree-traversal methods."""

    @pytest.mark.parametrize(
        "tree, source, expected",
        [
            (path_graph_4, 0, [(3, 2), (2, 1), (1, 0)]),
            (path_graph_4, 1, [(0, 1), (3, 2), (2, 1)]),
            (path_graph_4, 2, [(0, 1), (1, 2), (3, 2)]),
            (path_graph_4, 3, [(0, 1), (1, 2), (2, 3)]),
            (
                binary_graph_3,
                0,
                # fmt: off
                [
                    (7, 3), (8, 3), (3, 1), (9, 4), (10, 4), (4, 1), (1, 0),
                    (11, 5), (12, 5), (5, 2), (13, 6), (14, 6), (6, 2), (2, 0),
                ],
                # fmt: on
            ),
            (
                binary_graph_3,
                4,
                # fmt: off
                [
                    (11, 5), (12, 5), (5, 2), (13, 6), (14, 6), (6, 2), (2, 0), (0, 1),
                    (7, 3), (8, 3), (3, 1), (1, 4), (9, 4), (10, 4),
                ],
                # fmt: on
            ),
            (
                ternary_graph_2,
                0,
                # fmt: off
                [
                    (4, 1), (5, 1), (6, 1), (1, 0), (7, 2), (8, 2), (9, 2),
                    (2, 0), (10, 3), (11, 3), (12, 3), (3, 0),
                ],
                # fmt: on
            ),
            (
                ternary_graph_2,
                5,
                # fmt: off
                [
                    (7, 2), (8, 2), (9, 2), (2, 0), (10, 3), (11, 3), (12, 3), (3, 0),
                    (0, 1), (4, 1), (6, 1), (1, 5),
                ],
                # fmt: on
            ),
        ],
    )
    def test_postorder_traverse(self, tree, source, expected):
        """Tests for postorder_traverse."""
        post = postorder_traverse(tree, source)
        assert post == expected

    @pytest.mark.parametrize(
        "tree, source, expected",
        [
            (path_graph_4, 0, [(1, 0), (2, 1), (3, 2)]),
            (path_graph_4, 1, [(0, 1), (2, 1), (3, 2)]),
            (path_graph_4, 2, [(1, 2), (0, 1), (3, 2)]),
            (path_graph_4, 3, [(2, 3), (1, 2), (0, 1)]),
            (
                binary_graph_3,
                0,
                # fmt: off
                [
                    (1, 0), (3, 1), (7, 3), (8, 3), (4, 1), (9, 4), (10, 4),
                    (2, 0), (5, 2), (11, 5), (12, 5), (6, 2), (13, 6), (14, 6),
                ],
                # fmt: on
            ),
            (
                binary_graph_3,
                4,
                # fmt: off
                [
                    (1, 4), (0, 1), (2, 0), (5, 2), (11, 5), (12, 5), (6, 2),
                    (13, 6), (14, 6), (3, 1), (7, 3), (8, 3), (9, 4), (10, 4),
                ],
                # fmt: on
            ),
            (
                ternary_graph_2,
                0,
                # fmt: off
                [
                    (1, 0), (4, 1), (5, 1), (6, 1), (2, 0), (7, 2), (8, 2), (9, 2),
                    (3, 0), (10, 3), (11, 3), (12, 3),
                ],
                # fmt: on
            ),
            (
                ternary_graph_2,
                5,
                # fmt: off
                [
                    (1, 5), (0, 1), (2, 0), (7, 2), (8, 2), (9, 2),
                    (3, 0), (10, 3), (11, 3), (12, 3), (4, 1), (6, 1),
                ],
                # fmt: on
            ),
        ],
    )
    def test_preorder_traverse(self, tree, source, expected):
        """Tests for preorder_traverse."""
        pre = preorder_traverse(tree, source)
        assert pre == expected


def assert_reproduces_parity_matrix(cnots, expected_P):
    """Helper function that compares a CNOT circuit to a given parity matrix."""
    tape = QuantumScript([CNOT(wires) for wires in cnots])
    new_P = parity_matrix(tape, wire_order=list(range(len(expected_P))))
    assert np.allclose(new_P, expected_P)


def assert_respects_connectivity(cnots, connectivity):
    """Helper function that asserts that only CNOTs allowed by a connectivity graph are used."""
    # Get sorted tuples representing connected qubits
    edges = {tuple(sorted(edge)) for edge in connectivity.edges()}
    # Get sorted tuples (a, b) representing CNOT(a,b) or CNOT(b,a) (there is no direction in the
    # connectivity graph)
    unique_undirected_cnots = {tuple(sorted(cnot)) for cnot in cnots}
    assert unique_undirected_cnots.issubset(edges)


@pytest.mark.external
class TestRowCol:
    """Tests for rowcol."""

    def test_connectivity_default(self):
        """Test the connectivity default is used correctly"""
        m1 = _rowcol_parity_matrix(np.array([[1, 0], [1, 1]]), connectivity=None)
        m2 = _rowcol_parity_matrix(np.array([[1, 0], [1, 1]]), connectivity=nx.complete_graph(2))
        assert np.allclose(m1, m2)

    @pytest.mark.parametrize("n", list(range(2, 13)))
    @pytest.mark.parametrize("connectivity_fn", [nx.path_graph, nx.complete_graph])
    def test_identity(self, n, connectivity_fn):
        """Test with the identity Parity matrix/circuit."""
        P = np.eye(n, dtype=int)
        connectivity = connectivity_fn(n)
        cnots = _rowcol_parity_matrix(P, connectivity)
        assert not cnots

    @pytest.mark.parametrize("n", list(range(2, 13)))
    @pytest.mark.parametrize("connectivity_fn", [nx.path_graph, nx.complete_graph])
    def test_few_commuting_cnots(self, n, connectivity_fn):
        """Test with a few commuting CNOTs."""
        P = np.eye(n, dtype=int)
        for i in range(0, n - 1, 2):
            P[i + 1] += P[i]
        for i in range(0, n - 2, 2):
            P[i + 1] += P[i + 2]
        input_P = P.copy()

        connectivity = connectivity_fn(n)
        input_connectivity = connectivity.copy()
        cnots = _rowcol_parity_matrix(P, connectivity)
        exp = sum(([(i, i + 1), (i + 2, i + 1)] for i in range(0, n - 2, 2)), start=[])
        if n % 2 == 0:
            exp.append((n - 2, n - 1))
        assert set(cnots) == set(exp)
        # Check that P and connectivity were not altered
        assert np.allclose(input_P, P)
        assert set(input_connectivity.nodes()) == set(connectivity.nodes())
        assert set(input_connectivity.edges()) == set(connectivity.edges())
        assert_reproduces_parity_matrix(cnots, input_P)
        assert_respects_connectivity(cnots, connectivity)

    @pytest.mark.parametrize("n", list(range(3, 13)))
    def test_long_range_cnot(self, n):
        """Test with a single long-ranged CNOT in linear connectivity."""
        P = np.eye(n, dtype=int)
        P[0] += P[-1]
        input_P = P.copy()

        connectivity = nx.path_graph(n)
        input_connectivity = connectivity.copy()
        cnots = _rowcol_parity_matrix(P, connectivity)
        assert len(cnots) == 4 * (n - 2)  # Minimal CNOT count for longe-range CNOT
        # Check that P and connectivity were not altered
        assert np.allclose(input_P, P)
        assert set(input_connectivity.nodes()) == set(connectivity.nodes())
        assert set(input_connectivity.edges()) == set(connectivity.edges())
        assert_reproduces_parity_matrix(cnots, input_P)
        assert_respects_connectivity(cnots, connectivity)

    @pytest.mark.parametrize("n", list(range(2, 13)))
    @pytest.mark.parametrize("connectivity_fn", [nx.path_graph, nx.complete_graph])
    @pytest.mark.parametrize("input_depth", [(lambda n: n), (lambda n: n**3)])
    def test_random_circuit(self, n, connectivity_fn, input_depth):
        """Test with a random CNOT circuit."""

        P = np.eye(n, dtype=int)
        for _ in range(input_depth(n)):
            i, j = np.random.choice(n, size=2, replace=False)
            P[i] += P[j]
        P %= 2
        input_P = P.copy()

        connectivity = connectivity_fn(n)
        input_connectivity = connectivity.copy()
        cnots = _rowcol_parity_matrix(P, connectivity)
        # Check that P and connectivity were not altered
        assert np.allclose(input_P, P)
        assert set(input_connectivity.nodes()) == set(connectivity.nodes())
        assert set(input_connectivity.edges()) == set(connectivity.edges())
        assert_reproduces_parity_matrix(cnots, input_P)
        assert_respects_connectivity(cnots, connectivity)

    @pytest.mark.parametrize("n", list(range(2, 14)))
    @pytest.mark.parametrize("connectivity_fn", [nx.path_graph, nx.complete_graph])
    def test_integration(self, n, connectivity_fn):
        """Test transform function on random CNOT circuits"""

        ops = []
        for _ in range(n**3):
            i, j = np.random.choice(n, size=2, replace=False)
            ops.append(CNOT((int(i), int(j))))

        in_tape = QuantumScript(ops)
        input_P = parity_matrix(in_tape, wire_order=range(n))

        connectivity = connectivity_fn(n)
        input_connectivity = connectivity.copy()
        res, processing_fn = rowcol(in_tape, connectivity)
        out_tape = processing_fn(res)
        output_P = parity_matrix(out_tape, wire_order=range(n))

        assert np.allclose(input_P, output_P)
        assert set(input_connectivity.nodes()) == set(connectivity.nodes())
        assert set(input_connectivity.edges()) == set(connectivity.edges())
