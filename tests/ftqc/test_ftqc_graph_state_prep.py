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
"""Unit tests for the GraphStatePrep module"""

import networkx as nx
import numpy as np
import pytest

import pennylane as qml
from pennylane.ftqc import GraphStatePrep, QubitGraph, generate_lattice, make_graph_state
from pennylane.ops.functions import assert_valid


class TestGraphStatePrep:
    """Test for graph state prep"""

    @pytest.mark.xfail(reason="Jax JIT cannot trace a graph object")
    def test_non_jaxjit_circuit_graph_state_prep(self):
        """Test if Jax JIT works with GraphStatePrep"""
        jax = pytest.importorskip("jax")

        lattice = generate_lattice([2, 2], "square")
        q = QubitGraph(lattice.graph)
        dev = qml.device("default.qubit")

        @jax.jit
        @qml.qnode(dev)
        def circuit(q):
            GraphStatePrep(graph=q)
            return qml.probs()

        circuit(q)

    def test_jaxjit_circuit_graph_state_prep(self):
        """Test if Jax JIT works with GraphStatePrep"""
        jax = pytest.importorskip("jax")

        lattice = generate_lattice([2, 2], "square")
        q = QubitGraph(lattice.graph)
        dev = qml.device("default.qubit")

        @jax.jit
        @qml.qnode(dev)
        def circuit(x):
            GraphStatePrep(q, wires=[0, 1, 2, 3])
            qml.RX(x, 3)
            return qml.probs()

        circuit(1.23)

    @pytest.mark.parametrize(
        "dims, shape, wires",
        [
            ([5], "chain", None),
            ([2, 2], "square", None),
            ([2, 3], "rectangle", None),
            ([2, 2, 2], "cubic", None),
            ([5], "chain", range(5)),
            ([2, 2], "square", range(4)),
            ([2, 2], "rectangle", range(4)),
            (
                [2, 2, 2],
                "cubic",
                range(8),
            ),
        ],
    )
    def test_circuit_accept_graph_state_prep(self, dims, shape, wires):
        """Test if a quantum function accepts GraphStatePrep."""
        lattice = generate_lattice(dims, shape)
        q = QubitGraph(lattice.graph)
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(q):
            GraphStatePrep(graph=q, wires=wires)
            return qml.probs()

        res = circuit(q)
        assert len(res) == 2 ** len(lattice.graph)
        assert np.isclose(np.sum(res), 1.0, rtol=0)

    @pytest.mark.parametrize(
        "dims, shape, wires",
        [
            ([5], "chain", None),
            ([2, 2], "square", None),
            ([2, 3], "rectangle", None),
            ([2, 2, 2], "cubic", None),
            ([5], "chain", range(5)),
            ([2, 2], "square", range(4)),
            ([2, 2], "rectangle", range(4)),
            (
                [2, 2, 2],
                "cubic",
                range(8),
            ),
        ],
    )
    def test_graph_state_prep_creation_with_qubit_graph(self, dims, shape, wires):
        lattice = generate_lattice(dims, shape)
        q = QubitGraph(lattice.graph)
        # Test for object construction via QubitGraph
        assert_valid(GraphStatePrep(graph=q, wires=wires), skip_deepcopy=True, skip_pickle=True)
        assert repr(GraphStatePrep(graph=q, wires=wires)) == "GraphStatePrep(Hadamard, CZ)"
        assert GraphStatePrep(graph=q, wires=wires).label() == "GraphStatePrep(Hadamard, CZ)"

    @pytest.mark.parametrize(
        "dims, shape, wires",
        [
            pytest.param(
                [5],
                "chain",
                None,
                marks=pytest.mark.xfail(reason="Wires must be specified with nx.Graph objects."),
            ),
            ([5], "chain", range(5)),
            ([2, 2], "square", range(4)),
            ([2, 2], "rectangle", range(4)),
            (
                [2, 2, 2],
                "cubic",
                range(8),
            ),
        ],
    )
    def test_graph_state_prep_creation_with_nx_graph(self, dims, shape, wires):
        lattice = generate_lattice(dims, shape)
        # Test for object construction via nx graph
        assert_valid(
            GraphStatePrep(graph=lattice.graph, wires=wires), skip_deepcopy=True, skip_pickle=True
        )
        assert (
            repr(GraphStatePrep(graph=lattice.graph, wires=wires)) == "GraphStatePrep(Hadamard, CZ)"
        )
        assert (
            GraphStatePrep(graph=lattice.graph, wires=wires).label()
            == "GraphStatePrep(Hadamard, CZ)"
        )

    @pytest.mark.parametrize(
        "dims, shape, wires",
        [
            ([5], "chain", [0, 1, 2, 3, 4]),
            pytest.param(
                [5],
                "chain",
                None,
                marks=pytest.mark.xfail(
                    reason="Wires must be specified when building circuit with nx.Graph objects."
                ),
            ),
            ([2, 2], "square", [0, 1, 2, 3]),
            ([2, 2], "rectangle", [0, 1, 2, 3]),
            ([2, 2, 2], "cubic", [0, 1, 2, 3, 4, 5, 6, 7]),
        ],
    )
    def test_circuit_accept_graph_state_prep_with_nx_wires(self, dims, shape, wires):
        """Test if GraphStatePrep can be created with nx.Graph and user provided wires."""
        dev = qml.device("default.qubit")
        lattice = generate_lattice(dims, shape).graph

        @qml.qnode(dev)
        def circuit(lattice, wires):
            GraphStatePrep(graph=lattice, wires=wires)
            return qml.probs()

        res = circuit(lattice, wires)
        assert len(res) == 2 ** len(lattice)
        assert np.isclose(np.sum(res), 1.0, rtol=0)

    @pytest.mark.parametrize(
        "dims, shape, expected",
        [
            ([5], "chain", 9),
            ([2, 2], "square", 8),
            ([2, 3], "rectangle", 13),
            ([1, 2], "triangle", 9),
            ([2, 1], "honeycomb", 21),
            ([2, 2, 2], "cubic", 20),
        ],
    )
    def test_compute_decompose(self, dims, shape, expected):
        """Test the compute_decomposition method of the GraphStatePrep class."""
        lattice = generate_lattice(dims, shape)
        q = QubitGraph(lattice.graph)
        wires = set(lattice.graph)
        queue = GraphStatePrep.compute_decomposition(wires=range(len(wires)), graph=q)
        assert len(queue) == expected

    @pytest.mark.parametrize(
        "one_qubit_ops, two_qubit_ops",
        [
            (qml.H, qml.CZ),
            (qml.X, qml.CNOT),
        ],
    )
    def test_decompose(self, one_qubit_ops, two_qubit_ops):
        """Test the decomposition method of the GraphStatePrep class."""
        lattice = generate_lattice([2, 2, 2], "cubic")
        q = QubitGraph(lattice.graph)
        op = GraphStatePrep(graph=q, one_qubit_ops=one_qubit_ops, two_qubit_ops=two_qubit_ops)
        queue = op.decomposition()
        assert len(queue) == 20  # 8 ops for |0> -> |+> and 12 ops to entangle nearest qubits
        for op in queue[:8]:
            assert op.name == one_qubit_ops(0).name
            assert isinstance(op.wires[0], QubitGraph)
        for op in queue[8:]:
            assert op.name == two_qubit_ops.name
            assert all(isinstance(w, QubitGraph) for w in op.wires)

    @pytest.mark.parametrize(
        "dims, shape, wires",
        [
            ([5], "chain", [0, 1, 2, 3, 4]),
            ([5], "chain", ["a", 1, 2, "c", 4]),
            ([2, 2], "square", [0, 1, 2, 3]),
            ([2, 2], "square", ["d", 1, 2, "f"]),
            ([2, 3], "rectangle", ["a", "b", "c", "d", "e", "f"]),
            ([2, 2, 2], "cubic", [0, 1, 2, 3, 4, 5, 6, 7]),
        ],
    )
    def test_decompose_wires(self, dims, shape, wires):
        """Test the decomposition method of the GraphStatePrep class when wires are provided."""
        lattice = generate_lattice(dims, shape)
        q = QubitGraph(lattice.graph)

        # GraphStatePrep built from qubit graph
        op = GraphStatePrep(graph=q, wires=wires)
        queue = op.decomposition()
        for idx, op in enumerate(queue[: len(wires)]):
            assert op.wires[0] == wires[idx]

        # GraphStatePrep built from nx graph
        op = GraphStatePrep(graph=lattice.graph, wires=wires)
        queue = op.decomposition()
        for idx, op in enumerate(queue[: len(wires)]):
            assert op.wires[0] == wires[idx]

    @pytest.mark.parametrize(
        "one_qubit_ops, two_qubit_ops",
        [
            (qml.H, qml.CZ),
            (qml.X, qml.CNOT),
        ],
    )
    def test_wires_graph_mismatch(self, one_qubit_ops, two_qubit_ops):
        """Test for wire-graph label mismatches."""
        wires = [0, 1, 2, 3]
        edges = [(0, 1), (1, 2), (2, 3)]
        lattice = nx.Graph()
        lattice.add_nodes_from(wires)
        lattice.add_edges_from(edges)
        wires.append(5)
        with pytest.raises(
            ValueError,
            match="Please ensure the length of wires objects match that of labels in graph",
        ):
            GraphStatePrep(
                wires=wires, graph=lattice, one_qubit_ops=one_qubit_ops, two_qubit_ops=two_qubit_ops
            )

        with pytest.raises(ValueError, match="Please ensure wires is specified."):
            GraphStatePrep(
                wires=None, graph=lattice, one_qubit_ops=one_qubit_ops, two_qubit_ops=two_qubit_ops
            )

        with pytest.raises(
            ValueError,
            match="Please ensure the length of wires objects match the number of children in QubitGraph.",
        ):
            GraphStatePrep(
                wires=wires,
                graph=QubitGraph(lattice),
                one_qubit_ops=one_qubit_ops,
                two_qubit_ops=two_qubit_ops,
            )


class TestMakeGraphState:
    """Test the helper-function make_graph_state that allows graph state preparation
    to be capture compatible"""

    def test_make_graph_state_no_capture(self, mocker):
        """Test that the default behaviour for make_graph_state works as expected"""

        lattice = generate_lattice([2, 2], "square")
        q_graph = QubitGraph(lattice.graph)

        spy = mocker.spy(qml.ftqc.graph_state_preparation, "GraphStatePrep")

        with qml.queuing.AnnotatedQueue() as q:
            make_graph_state(q_graph, [0, 1, 2, 3])

        assert len(q) == 1
        assert isinstance(q.queue[0], GraphStatePrep)
        spy.assert_called_with(
            graph=q_graph, one_qubit_ops=qml.H, two_qubit_ops=qml.CZ, wires=[0, 1, 2, 3]
        )

    def test_passing_operations_no_capture(self, mocker):
        """Test that gates for the graph state can be specified in make_graph_state,
        and are passed to GraphStatePrep"""

        lattice = generate_lattice([2, 2], "square")
        q_graph = QubitGraph(lattice.graph)

        spy = mocker.spy(qml.ftqc.graph_state_preparation, "GraphStatePrep")

        with qml.queuing.AnnotatedQueue() as q:
            make_graph_state(q_graph, [0, 1, 2, 3], one_qubit_ops=qml.X, two_qubit_ops=qml.CNOT)

        assert len(q) == 1
        assert isinstance(q.queue[0], GraphStatePrep)
        spy.assert_called_with(
            graph=q_graph, one_qubit_ops=qml.X, two_qubit_ops=qml.CNOT, wires=[0, 1, 2, 3]
        )

    @pytest.mark.capture
    def test_make_graph_state_with_capture(self, mocker):
        """Test that make_graph_state adds the decomposed graph state to the plxpr"""
        import jax

        spy = mocker.spy(qml.ftqc.graph_state_preparation.GraphStatePrep, "compute_decomposition")

        lattice = generate_lattice([2, 2], "square")
        q_graph = QubitGraph(lattice.graph)
        wires = [0, 1, 2, 3]

        def func():
            make_graph_state(q_graph, wires)

        plxpr = jax.make_jaxpr(func)()

        # the operator queue looks correct
        assert len(plxpr.eqns) == 8
        for eq in plxpr.eqns[:4]:
            assert "Hadamard" in str(eq)
        for eq in plxpr.eqns[4:]:
            assert "CZ" in str(eq)

        # the queue was generated by the expected method
        spy.assert_called_with([0, 1, 2, 3], q_graph, qml.H, qml.CZ)

    @pytest.mark.capture
    def test_passing_operations_with_capture(self, mocker):
        """Test that gates for the graph state can be specified in make_graph_state,
        and are passed to compute_decomposition when capture is enabled"""
        import jax

        lattice = generate_lattice([2, 2], "square")
        q_graph = QubitGraph(lattice.graph)

        spy = mocker.spy(qml.ftqc.graph_state_preparation.GraphStatePrep, "compute_decomposition")

        def func():
            make_graph_state(q_graph, [0, 1, 2, 3], one_qubit_ops=qml.X, two_qubit_ops=qml.CNOT)

        plxpr = jax.make_jaxpr(func)()

        assert len(plxpr.eqns) == 8
        spy.assert_called_with([0, 1, 2, 3], q_graph, qml.X, qml.CNOT)


class TestGraphStateInvariantUnderInternalGraphOrdering:
    """Additional tests of graph-state preparation to ensure that the resulting state is invariant
    under the internal ordering of the nodes and edges in the graph.

    In other words, identical graph structures, regardless of the order in which the nodes and edges
    are stored in memory, should result in the same state.

    At the time of writing, NetworkX graphs use built-in Python dictionaries to store a graph's
    adjacency list. As of Python 3.7, `dict` objects are insertion-ordered:

        https://docs.python.org/3/whatsnew/3.7.html

    Hence, the order in which nodes and edges are inserted into the NetworkX graph object will
    affect the access order when iterating over the set of nodes and the set of edges. Graph data
    structures are inherently unordered, therefore differences in the access order for the same
    graph structure should not affect the state returned by GraphStatePrep.
    """

    @pytest.mark.parametrize(
        "graph",
        [
            # Permute the node order in the adjacency list
            nx.Graph({1: {0, 2}, 2: {1, 3}, 3: {2}, 0: {1}}),
            nx.Graph({2: {1, 3}, 3: {2}, 0: {1}, 1: {0, 2}}),
            nx.Graph({3: {2}, 0: {1}, 1: {0, 2}, 2: {1, 3}}),
            nx.Graph({0: {1}, 1: {0, 2}, 3: {2}, 2: {1, 3}}),
            nx.Graph({0: {1}, 2: {1, 3}, 1: {0, 2}, 3: {2}}),
            nx.Graph({1: {0, 2}, 0: {1}, 2: {1, 3}, 3: {2}}),
        ],
    )
    @pytest.mark.parametrize("one_qubit_op", [qml.H, qml.X, qml.Y, qml.Z, qml.S, qml.SX])
    @pytest.mark.parametrize("two_qubit_op", [qml.CZ])
    def test_graph_state_invariant_under_internal_ordering_1d_chain_same_wires_indices(
        self, graph, one_qubit_op, two_qubit_op
    ):
        """Test that the state returned by GraphStatePrep is invariant under the internal ordering
        of the nodes and edges in the graph.

        The graph structure is a 1D chain and the sets of node and wire labels are the same.
        """
        # Graph structure: (0) -- (1) -- (2) -- (3)
        graph_ref = nx.Graph({0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}})

        # The sequence of wires is constant
        wires = [0, 1, 2, 3]

        self._check_graphs_yield_same_state(graph, graph_ref, wires, one_qubit_op, two_qubit_op)

    @pytest.mark.parametrize(
        "graph",
        [
            # Permute the node order in the adjacency list
            nx.Graph({"b": {"a", "c"}, "c": {"b", "d"}, "d": {"c"}, "a": {"b"}}),
            nx.Graph({"c": {"b", "d"}, "d": {"c"}, "a": {"b"}, "b": {"a", "c"}}),
            nx.Graph({"d": {"c"}, "a": {"b"}, "b": {"a", "c"}, "c": {"b", "d"}}),
            nx.Graph({"a": {"b"}, "b": {"a", "c"}, "d": {"c"}, "c": {"b", "d"}}),
            nx.Graph({"a": {"b"}, "c": {"b", "d"}, "b": {"a", "c"}, "d": {"c"}}),
            nx.Graph({"b": {"a", "c"}, "a": {"b"}, "c": {"b", "d"}, "d": {"c"}}),
        ],
    )
    @pytest.mark.parametrize("one_qubit_op", [qml.H, qml.X, qml.Y, qml.Z, qml.S, qml.SX])
    @pytest.mark.parametrize("two_qubit_op", [qml.CZ])
    def test_graph_state_invariant_under_internal_ordering_1d_chain_diff_wires_indices(
        self, graph, one_qubit_op, two_qubit_op
    ):
        """Test that the state returned by GraphStatePrep is invariant under the internal ordering
        of the nodes and edges in the graph.

        The graph structure is a 1D chain and the sets of node and wire labels are different.
        """
        # Graph structure: ("a") -- ("b") -- ("c") -- ("d")
        graph_ref = nx.Graph({"a": {"b"}, "b": {"a", "c"}, "c": {"b", "d"}, "d": {"c"}})

        # The sequence of wires is constant
        wires = [0, 1, 2, 3]

        self._check_graphs_yield_same_state(graph, graph_ref, wires, one_qubit_op, two_qubit_op)

    @pytest.mark.parametrize(
        "graph",
        [
            # Permute the node order in the adjacency list
            nx.Graph({1: {0, 3}, 2: {0, 3}, 3: {1, 2}, 0: {1, 2}}),
            nx.Graph({2: {0, 3}, 3: {1, 2}, 0: {1, 2}, 1: {0, 3}}),
            nx.Graph({3: {1, 2}, 0: {1, 2}, 1: {0, 3}, 2: {0, 3}}),
            nx.Graph({0: {1, 2}, 1: {0, 3}, 3: {1, 2}, 2: {0, 3}}),
            nx.Graph({0: {1, 2}, 2: {0, 3}, 1: {0, 3}, 3: {1, 2}}),
            nx.Graph({1: {0, 3}, 0: {1, 2}, 2: {0, 3}, 3: {1, 2}}),
        ],
    )
    @pytest.mark.parametrize("one_qubit_op", [qml.H, qml.X, qml.Y, qml.Z, qml.S, qml.SX])
    @pytest.mark.parametrize("two_qubit_op", [qml.CZ])
    def test_graph_state_invariant_under_internal_ordering_2d_grid_same_wires_indices(
        self, graph, one_qubit_op, two_qubit_op
    ):
        """Test that the state returned by GraphStatePrep is invariant under the internal ordering
        of the nodes and edges in the graph.

        The graph structure is a 2D grid and the sets of node and wire labels are the same.
        """
        # Graph structure: (0) -- (1)
        #                   |      |
        #                  (2) -- (3)
        graph_ref = nx.Graph({0: {1, 2}, 1: {0, 3}, 2: {0, 3}, 3: {1, 2}})

        # The sequence of wires is constant
        wires = [0, 1, 2, 3]

        self._check_graphs_yield_same_state(graph, graph_ref, wires, one_qubit_op, two_qubit_op)

    @pytest.mark.parametrize(
        "graph",
        [
            # Permute the node order in the adjacency list
            nx.Graph(
                {
                    (0, 1): {(0, 0), (1, 1)},
                    (1, 0): {(0, 0), (1, 1)},
                    (1, 1): {(0, 1), (1, 0)},
                    (0, 0): {(1, 0), (0, 1)},
                }
            ),
            nx.Graph(
                {
                    (1, 0): {(0, 0), (1, 1)},
                    (1, 1): {(0, 1), (1, 0)},
                    (0, 0): {(1, 0), (0, 1)},
                    (0, 1): {(0, 0), (1, 1)},
                }
            ),
            nx.Graph(
                {
                    (1, 1): {(0, 1), (1, 0)},
                    (0, 0): {(1, 0), (0, 1)},
                    (0, 1): {(0, 0), (1, 1)},
                    (1, 0): {(0, 0), (1, 1)},
                }
            ),
        ],
    )
    @pytest.mark.parametrize("one_qubit_op", [qml.H, qml.X, qml.Y, qml.Z, qml.S, qml.SX])
    @pytest.mark.parametrize("two_qubit_op", [qml.CZ])
    def test_graph_state_invariant_under_internal_ordering_2d_grid_diff_wires_indices(
        self, graph, one_qubit_op, two_qubit_op
    ):
        """Test that the state returned by GraphStatePrep is invariant under the internal ordering
        of the nodes and edges in the graph.

        The graph structure is a 2D grid and the sets of node and wire labels are different.
        """
        # Graph structure: (0,0) -- (0,1)
        #                    |        |
        #                  (1,0) -- (1,1)
        graph_ref = nx.Graph(
            {
                (0, 0): {(1, 0), (0, 1)},
                (0, 1): {(0, 0), (1, 1)},
                (1, 0): {(0, 0), (1, 1)},
                (1, 1): {(0, 1), (1, 0)},
            }
        )

        # The sequence of wires is constant
        wires = [0, 1, 2, 3]

        self._check_graphs_yield_same_state(graph, graph_ref, wires, one_qubit_op, two_qubit_op)

    @pytest.mark.parametrize(
        "graph",
        [
            # Permute the node order in the adjacency list
            nx.Graph(
                {
                    1: {0, 3, 5},
                    2: {0, 3, 6},
                    3: {1, 2, 7},
                    4: {0, 5, 6},
                    5: {1, 4, 7},
                    6: {2, 4, 7},
                    7: {3, 5, 6},
                    0: {1, 2, 4},
                }
            ),
            nx.Graph(
                {
                    2: {0, 3, 6},
                    3: {1, 2, 7},
                    4: {0, 5, 6},
                    5: {1, 4, 7},
                    6: {2, 4, 7},
                    7: {3, 5, 6},
                    0: {1, 2, 4},
                    1: {0, 3, 5},
                }
            ),
            nx.Graph(
                {
                    3: {1, 2, 7},
                    4: {0, 5, 6},
                    5: {1, 4, 7},
                    6: {2, 4, 7},
                    7: {3, 5, 6},
                    0: {1, 2, 4},
                    1: {0, 3, 5},
                    2: {0, 3, 6},
                }
            ),
            nx.Graph(
                {
                    4: {0, 5, 6},
                    5: {1, 4, 7},
                    6: {2, 4, 7},
                    7: {3, 5, 6},
                    0: {1, 2, 4},
                    1: {0, 3, 5},
                    2: {0, 3, 6},
                    3: {1, 2, 7},
                }
            ),
            nx.Graph(
                {
                    5: {1, 4, 7},
                    6: {2, 4, 7},
                    7: {3, 5, 6},
                    0: {1, 2, 4},
                    1: {0, 3, 5},
                    2: {0, 3, 6},
                    3: {1, 2, 7},
                    4: {0, 5, 6},
                }
            ),
            nx.Graph(
                {
                    6: {2, 4, 7},
                    7: {3, 5, 6},
                    0: {1, 2, 4},
                    1: {0, 3, 5},
                    2: {0, 3, 6},
                    3: {1, 2, 7},
                    4: {0, 5, 6},
                    5: {1, 4, 7},
                }
            ),
            nx.Graph(
                {
                    7: {3, 5, 6},
                    0: {1, 2, 4},
                    1: {0, 3, 5},
                    2: {0, 3, 6},
                    3: {1, 2, 7},
                    4: {0, 5, 6},
                    5: {1, 4, 7},
                    6: {2, 4, 7},
                }
            ),
        ],
    )
    @pytest.mark.parametrize("one_qubit_op", [qml.H, qml.X, qml.Y, qml.Z, qml.S, qml.SX])
    @pytest.mark.parametrize("two_qubit_op", [qml.CZ])
    def test_graph_state_invariant_under_internal_ordering_3d_grid_same_wires_indices(
        self, graph, one_qubit_op, two_qubit_op
    ):
        """Test that the state returned by GraphStatePrep is invariant under the internal ordering
        of the nodes and edges in the graph.

        The graph structure is a 3D chain and the sets of node and wire labels are the same.
        """
        # Graph structure:    (4)-----(5)
        #                     /|      /|
        #                  (0)-----(1) |
        #                   | (6)---|-(7)
        #                   | /     | /
        #                  (2)-----(3)
        graph_ref = nx.Graph(
            {
                0: {1, 2, 4},
                1: {0, 3, 5},
                2: {0, 3, 6},
                3: {1, 2, 7},
                4: {0, 5, 6},
                5: {1, 4, 7},
                6: {2, 4, 7},
                7: {3, 5, 6},
            }
        )

        # The sequence of wires is constant
        wires = [0, 1, 2, 3, 4, 5, 6, 7]

        self._check_graphs_yield_same_state(graph, graph_ref, wires, one_qubit_op, two_qubit_op)

    @staticmethod
    def _check_graphs_yield_same_state(graph, graph_ref, wires, one_qubit_op, two_qubit_op):
        """Helper function for the above tests.

        Creates a minimal circuit that returns the state resulting from GraphStatePrep. It then
        executes this circuit using the input graphs and asserts that the resulting states are
        equal. ``graph_ref`` is the assumed to be the reference graph that yields the state that is
        known to be correct.
        """
        # Sanity check: ensure that the graphs have the same adjacency list
        assert (
            graph.adj == graph_ref.adj
        ), "Internal error: all graphs defined in this test should have the same adjacency list"

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(graph):
            GraphStatePrep(
                graph=graph, wires=wires, one_qubit_ops=one_qubit_op, two_qubit_ops=two_qubit_op
            )
            return qml.state()

        result_ref = circuit(graph_ref)
        result = circuit(graph)
        assert np.all(result_ref == result)

    def test_unsortable_graph_node_labels_raise_type_error(self):
        """Test that using a graph with unsortable node labels (in this case, the node labels are a
        mix of integers and strings) as input to GraphStatePrep raises a TypeError.
        """
        # Graph structure: (0) -- (a) -- (1)
        graph = nx.Graph({0: {"a"}, "a": (0, 1), 1: {"a"}})
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            GraphStatePrep(graph=graph, wires=[0, 1, 2])
            return qml.state()

        with pytest.raises(
            TypeError,
            match="GraphStatePrep requires the node labels of the input graph to be sortable",
        ):
            circuit()
