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
from pennylane.ftqc import GraphStatePrep, QubitGraph, generate_lattice
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
