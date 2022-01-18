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

import pennylane as qml
from pennylane.qcut.compiler import tape_to_graph

class TestTapeToGraph:
    """
    Tests conversion of tapes to graph representations that are amenable to
    partitioning algorithms for circuit cutting
    """

    def test_simple_tape_to_graph(self):
        """
        Tests that the conversion of a simple tape gives a graph containing the
        the expected nodes and edges
        """

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires='a')
            qml.CNOT(wires=[0, 'a'])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires='a')
            qml.expval(qml.PauliZ(wires=[0]))

        g = tape_to_graph(tape)
        nodes = list(g.nodes)

        ops = tape.operations

        assert len(nodes) == len(ops) + len(tape.observables)
        for op, node in zip(ops, nodes[:-1]):
            assert op == node
        assert tape.observables[0] == nodes[-1].obs

        edges = list(g.edges)
        num_wires_connecting_gates = 5
        assert len(edges) == num_wires_connecting_gates

        expected_edge_connections = [
            (ops[0],ops[2]),
            (ops[1], ops[2]),
            (ops[2], ops[3]),
            (ops[2], ops[4]),
            (ops[3], tape.observables[0])
        ]

        for edge, expected_edge in zip(edges[:-1], expected_edge_connections[:-1]):
            assert (edge[0], edge[1]) == (expected_edge[0], expected_edge[1])
        assert (edges[-1][0], edges[-1][1].obs) == (expected_edge_connections[-1][0], expected_edge_connections[-1][1])
