# Copyright 2022 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the `pennylane.qcut` package.
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.qcut.compiler import tape_to_graph

with qml.tape.QuantumTape() as tape:
    qml.RX(0.432, wires=0)
    qml.RY(0.543, wires="a")
    qml.CNOT(wires=[0, "a"])
    qml.RZ(0.240, wires=0)
    qml.RZ(0.133, wires="a")
    qml.expval(qml.PauliZ(wires=[0]))


class TestTapeToGraph:
    """
    Tests conversion of tapes to graph representations that are amenable to
    partitioning algorithms for circuit cutting
    """

    def test_converted_graph_nodes(self):
        """
        Tests that the conversion of a tape gives a graph containing the
        expected nodes
        """

        g = tape_to_graph(tape)
        nodes = list(g.nodes)

        ops = tape.operations

        assert len(nodes) == len(ops) + len(tape.observables)
        for op, node in zip(ops, nodes[:-1]):
            assert op == node
        assert tape.observables[0] == nodes[-1].obs

    def test_converted_graph_edges(self):
        """
        Tests that the conversion of a tape gives a graph containing the
        expected edges
        """
        g = tape_to_graph(tape)
        edges = list(g.edges)

        num_wires_connecting_gates = 5
        assert len(edges) == num_wires_connecting_gates

        ops = tape.operations
        expected_edge_connections = [
            (ops[0], ops[2]),
            (ops[1], ops[2]),
            (ops[2], ops[3]),
            (ops[2], ops[4]),
            (ops[3], tape.observables[0]),
        ]

        for edge, expected_edge in zip(edges[:-1], expected_edge_connections[:-1]):
            assert (edge[0], edge[1]) == (expected_edge[0], expected_edge[1])
        assert (edges[-1][0], edges[-1][1].obs) == (
            expected_edge_connections[-1][0],
            expected_edge_connections[-1][1],
        )

    def test_node_order_attribute(self):
        """
        Tests that the converted nodes contain the correct order attirbute
        """

        g = tape_to_graph(tape)
        node_data = list(g.nodes(data=True))

        expected_node_order = [
            {"order": 0},
            {"order": 1},
            {"order": 2},
            {"order": 3},
            {"order": 4},
            {"order": 5},
        ]

        for data, expected_order in zip(node_data, expected_node_order):
            assert data[-1] == expected_order

    def test_edge_wire_attribute(self):
        """
        Tests that the converted edges contain the correct wire attirbute
        """

        g = tape_to_graph(tape)
        edge_data = list(g.edges(data=True))

        expected_edge_wires = [{"wire": 0}, {"wire": "a"}, {"wire": 0}, {"wire": "a"}, {"wire": 0}]

        for data, expected_wire in zip(edge_data, expected_edge_wires):
            assert data[-1] == expected_wire

    @pytest.mark.parametrize(
        "obs",
        [
            qml.PauliZ(0) @ qml.PauliZ(2),  # Broken
            qml.Projector([0, 1], wires=[0, 1]),
            qml.Hamiltonian([1, 2], [qml.PauliZ(1), qml.PauliZ(2) @ qml.PauliX(0)]),  # Broken
            qml.Hermitian(np.array([[1, 0], [0, -1]]), wires=[0]),
            qml.Projector([0, 1], wires=[0, 1]) @ qml.Projector([1, 0], wires=[0, 2]),  # Broken
        ],
    )
    def test_observable_conversion(self, obs):
        """
        TODO: DocString
        """

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires=1)
            qml.expval(obs)

        g = tape_to_graph(tape)
        nodes = list(g.nodes)
        edges = list(g.edges)

        obs_node = nodes[-1]
        obs = tape.observables[0]
