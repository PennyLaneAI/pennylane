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


class TestGraphStatePrep:
    """Test for graph state prep"""

    @pytest.mark.xfail(reason="Jax JIT requires wires to be integers.")
    def test_jaxjit_circuit_graph_state_prep(self):
        """Test if Jax JIT works with GraphStatePrep"""
        lattice = generate_lattice([2, 2], "square")
        q = QubitGraph(lattice.graph)
        dev = qml.device("default.qubit")

        jax = pytest.importorskip("jax")

        @jax.jit
        @qml.qnode(dev)
        def circuit(q):
            GraphStatePrep(graph=q)
            return qml.probs()

        circuit(q)

    def test_circuit_accept_graph_state_prep(self):
        """Test if a quantum function accepts GraphStatePrep."""
        lattice = generate_lattice([2, 2], "square")
        q = QubitGraph(lattice.graph)
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(q):
            GraphStatePrep(graph=q)
            return qml.probs()

        res = circuit(q)
        assert len(res) == 2 ** len(lattice.graph)
        assert np.isclose(np.sum(res), 1.0, rtol=0)
        assert repr(GraphStatePrep(graph=q)) == "GraphStatePrep(Hadamard, CZ)"
        assert GraphStatePrep(graph=q).label() == "GraphStatePrep(Hadamard, CZ)"

    def test_circuit_accept_graph_state_prep_with_nx_wires(self):
        """Test if a quantum function accepts GraphStatePrep."""
        dev = qml.device("default.qubit")
        lattice = nx.grid_graph((4,))
        wires = list(lattice.nodes)

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
        queue = GraphStatePrep.compute_decomposition(wires=wires, graph=q)
        assert len(queue) == expected

    @pytest.mark.parametrize(
        "qubit_ops, entanglement_ops",
        [
            (qml.H, qml.CZ),
            (qml.X, qml.CNOT),
        ],
    )
    def test_decompose(self, qubit_ops, entanglement_ops):
        """Test the decomposition method of the GraphStatePrep class."""
        lattice = generate_lattice([2, 2, 2], "cubic")
        q = QubitGraph(lattice.graph)
        op = GraphStatePrep(graph=q, qubit_ops=qubit_ops, entanglement_ops=entanglement_ops)
        queue = op.decomposition()
        assert len(queue) == 20  # 8 ops for |0> -> |+> and 12 ops to entangle nearest qubits
        for op in queue[:8]:
            assert op.name == qubit_ops(0).name
            assert isinstance(op.wires[0], QubitGraph)
        for op in queue[8:]:
            assert op.name == entanglement_ops.name
            assert all(isinstance(w, QubitGraph) for w in op.wires)

    @pytest.mark.parametrize(
        "qubit_ops, entanglement_ops",
        [
            (qml.H, qml.CZ),
            (qml.X, qml.CNOT),
        ],
    )
    def test_decompose_wires(self, qubit_ops, entanglement_ops):
        """Test the decomposition method of the GraphStatePrep class."""
        lattice = nx.grid_graph((4,))
        wires = list(lattice.nodes)

        op = GraphStatePrep(
            wires=wires, graph=lattice, qubit_ops=qubit_ops, entanglement_ops=entanglement_ops
        )
        queue = op.decomposition()
        assert len(queue) == 7  # 4 ops for |0> -> |+> and 3 ops to entangle nearest qubits
        for op in queue[:4]:
            assert op.name == qubit_ops(0).name
        for op in queue[4:]:
            assert op.name == entanglement_ops.name

    @pytest.mark.parametrize(
        "qubit_ops, entanglement_ops",
        [
            (qml.H, qml.CZ),
            (qml.X, qml.CNOT),
        ],
    )
    def test_wires_graph_mismatch(self, qubit_ops, entanglement_ops):
        """Test for wire-graph label mismatches."""
        wires = [0, 1, 2, 3]
        edges = [(0, 1), (1, 2), (2, 3)]
        lattice = nx.Graph()
        lattice.add_nodes_from(wires)
        lattice.add_edges_from(edges)
        wires.append(5)
        with pytest.raises(ValueError):
            GraphStatePrep(
                wires=wires, graph=lattice, qubit_ops=qubit_ops, entanglement_ops=entanglement_ops
            )

        with pytest.raises(ValueError):
            GraphStatePrep(
                wires=None, graph=lattice, qubit_ops=qubit_ops, entanglement_ops=entanglement_ops
            )

        with pytest.raises(ValueError):
            GraphStatePrep(
                wires=wires,
                graph=QubitGraph(lattice),
                qubit_ops=qubit_ops,
                entanglement_ops=entanglement_ops,
            )
