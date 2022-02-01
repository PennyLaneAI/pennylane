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
import pennylane as qml
import pytest
from pennylane.transforms import qcut

with qml.tape.QuantumTape() as tape:
    qml.RX(0.432, wires=0)
    qml.RY(0.543, wires="a")
    qml.CNOT(wires=[0, "a"])
    qml.CRZ(0.5, wires=["a", 0])
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

        g = qcut.tape_to_graph(tape)
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
        g = qcut.tape_to_graph(tape)
        edges = list(g.edges)

        num_wires_connecting_gates = 7
        assert len(edges) == num_wires_connecting_gates

        ops = tape.operations

        expected_edge_connections = [
            (ops[0], ops[2], 0),
            (ops[1], ops[2], 0),
            (ops[2], ops[3], 0),
            (ops[2], ops[3], 1),
            (ops[3], ops[4], 0),
            (ops[3], ops[5], 0),
            (ops[4], tape.measurements[0], 0),
        ]

        for edge, expected_edge in zip(edges, expected_edge_connections):
            assert edge == expected_edge

    def test_node_order_attribute(self):
        """
        Tests that the converted nodes contain the correct order attribute
        """

        g = qcut.tape_to_graph(tape)
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
        Tests that the converted edges contain the correct wire attribute
        """

        g = qcut.tape_to_graph(tape)
        edge_data = list(g.edges(data=True))

        expected_edge_wires = [
            {"wire": 0},
            {"wire": "a"},
            {"wire": "a"},
            {"wire": 0},
            {"wire": 0},
            {"wire": "a"},
            {"wire": 0},
        ]

        for data, expected_wire in zip(edge_data, expected_edge_wires):
            assert data[-1] == expected_wire

    @pytest.mark.parametrize(
        "obs,expected_obs",
        [
            (
                qml.PauliZ(0) @ qml.PauliZ(2),
                [qml.expval(qml.PauliZ(wires=[0])), qml.expval(qml.PauliZ(wires=[2]))],
            ),
            (
                qml.Projector([0, 1], wires=[0, 1]),
                [qml.expval(qml.Projector([0, 1], wires=[0, 1]))],
            ),
            (
                qml.Hamiltonian([1, 2], [qml.PauliZ(1), qml.PauliZ(2) @ qml.PauliX(0)]),
                [
                    qml.expval(
                        qml.Hamiltonian([1, 2], [qml.PauliZ(1), qml.PauliZ(2) @ qml.PauliX(0)])
                    )
                ],
            ),
            (
                qml.Hermitian(np.array([[1, 0], [0, -1]]), wires=[0]),
                [qml.expval(qml.Hermitian(np.array([[1, 0], [0, -1]]), wires=[0]))],
            ),
            (
                qml.Projector([0, 1], wires=[0, 1]) @ qml.Projector([1, 0], wires=[0, 2]),
                [
                    qml.expval(qml.Projector([0, 1], wires=[0, 1])),
                    qml.expval(qml.Projector([1, 0], wires=[0, 2])),
                ],
            ),
        ],
    )
    def test_observable_conversion(self, obs, expected_obs):
        """
        Tests that a variety of observables in a tape are correctly converted to
        observables contained within the graph nodes
        """

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires=1)
            qml.expval(obs)

        g = qcut.tape_to_graph(tape)
        nodes = list(g.nodes)

        node_observables = [node for node in nodes if hasattr(node, "return_type")]

        for node_obs, exp_obs in zip(node_observables, expected_obs):
            assert node_obs.wires == exp_obs.wires
            assert node_obs.obs.name == exp_obs.obs.name

    @pytest.mark.parametrize(
        "measurement,expected_measurement",
        [
            (qml.expval(qml.PauliZ(wires=[0])), "Expectation"),
            (qml.sample(qml.PauliZ(wires=[0])), "Sample"),
            (qml.var(qml.PauliZ(wires=[0])), "Variance"),
            (qml.probs(wires=0), "Probability"),
            (qml.state(), "State"),
            (qml.density_matrix([0]), "State"),
        ],
    )
    def test_measurement_conversion(self, measurement, expected_measurement):
        """
        Tests that measurements are correctly converted
        """

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires=1)
            qml.apply(measurement)

        g = qcut.tape_to_graph(tape)
        nodes = list(g.nodes)

        node_observables = [node for node in nodes if hasattr(node, "return_type")]

        assert node_observables[0].return_type.name == expected_measurement

    def test_multiple_observables(self):
        """
        Tests that a tape containing multiple measurements is correctly
        converted to a graph
        """

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires=1)
            qml.expval(qml.PauliZ(wires=[0]))
            qml.expval(qml.PauliX(wires=[1]) @ qml.PauliY(wires=[2]))

        expected_obs = [
            qml.expval(qml.PauliZ(wires=[0])),
            qml.expval(qml.PauliX(wires=[1])),
            qml.expval(qml.PauliY(wires=[2])),
        ]

        g = qcut.tape_to_graph(tape)
        nodes = list(g.nodes)

        node_observables = [node for node in nodes if hasattr(node, "return_type")]

        for node_obs, exp_obs in zip(node_observables, expected_obs):
            assert node_obs.wires == exp_obs.wires
            assert node_obs.obs.name == exp_obs.obs.name
