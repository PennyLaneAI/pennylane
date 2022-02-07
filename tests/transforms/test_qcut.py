# Copyright 2022 Xanadu Quantum Technologies Inc.

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

with qml.tape.QuantumTape() as multi_cut_tape:
    qml.RX(0.432, wires=0)
    qml.RY(0.543, wires="a")
    qml.WireCut(wires=0)
    qml.CNOT(wires=[0, "a"])
    qml.RZ(0.240, wires=0)
    qml.RZ(0.133, wires="a")
    qml.WireCut(wires="a")
    qml.CNOT(wires=["a", 2])
    qml.RX(0.432, wires="a")
    qml.WireCut(wires=2)
    qml.CNOT(wires=[2, 3])
    qml.RY(0.543, wires=2)
    qml.RZ(0.876, wires=3)
    qml.expval(qml.PauliZ(wires=[0]))


def compare_nodes(nodes, expected_wires, expected_names):
    """Helper function to compare nodes of directed multigraph"""

    for node, exp_wire in zip(nodes, expected_wires):
        assert node.wires.tolist() == exp_wire

    for node, exp_name in zip(nodes, expected_names):
        assert node.name == exp_name


def compare_fragment_nodes(node_data, expected_data):
    """Helper function to compare nodes of fragment graphs"""

    expected = [(exp_data[0].name, exp_data[0].wires, exp_data[1]) for exp_data in expected_data]

    for data in node_data:
        # The exact ordering of node_data varies on each call
        assert (data[0].name, data[0].wires, data[1]) in expected


def compare_fragment_edges(edge_data, expected_data):
    """Helper function to compare fragment edges"""

    expected = [(exp_data[0].name, exp_data[1].name, exp_data[2]) for exp_data in expected_data]

    for data in edge_data:
        # The exact ordering of edge_data varies on each call
        assert (data[0].name, data[1].name, data[2]) in expected


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


class TestReplaceWireCut:
    """
    Tests the replacement of WireCut operation with MeasureNode and
    PrepareNode
    """

    def test_single_wire_cut_replaced(self):
        """
        Tests that a single WireCut operator is replaced with a MeasureNode and
        a PrepareNode with the correct order
        """

        wire_cut_num = 1

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires=1)
            qml.WireCut(wires=wire_cut_num)
            qml.CNOT(wires=[1, 2])
            qml.RX(0.432, wires=1)
            qml.RY(0.543, wires=2)
            qml.expval(qml.PauliZ(wires=[0]))

        g = qcut.tape_to_graph(tape)
        node_data = list(g.nodes(data=True))

        wire_cut_order = {"order": 5}

        qcut.replace_wire_cut_nodes(g)
        new_node_data = list(g.nodes(data=True))
        op_names = [op.name for op, order in new_node_data]

        assert "WireCut" not in op_names
        assert "MeasureNode" in op_names
        assert "PrepareNode" in op_names

        for op, order in new_node_data:
            if op.name == "MeasureNode":
                assert op.wires.tolist() == [wire_cut_num]
                assert order == wire_cut_order
            elif op.name == "PrepareNode":
                assert op.wires.tolist() == [wire_cut_num]
                assert order == wire_cut_order

    def test_multiple_wire_cuts_replaced(self):
        """
        Tests that all WireCut operators are replaced with MeasureNodes and
        PrepareNodes with the correct order
        """

        wire_cut_1 = 0
        wire_cut_2 = "a"
        wire_cut_3 = 2
        wire_cut_num = [wire_cut_1, wire_cut_2, wire_cut_3]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires="a")
            qml.WireCut(wires=wire_cut_1)
            qml.CNOT(wires=[0, "a"])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires="a")
            qml.WireCut(wires=wire_cut_2)
            qml.CNOT(wires=["a", 2])
            qml.RX(0.432, wires="a")
            qml.WireCut(wires=wire_cut_3)
            qml.CNOT(wires=[2, 3])
            qml.RY(0.543, wires=2)
            qml.RZ(0.876, wires=3)
            qml.expval(qml.PauliZ(wires=[0]))

        g = qcut.tape_to_graph(tape)
        node_data = list(g.nodes(data=True))

        wire_cut_order = [order for op, order in node_data if op.name == "WireCut"]

        qcut.replace_wire_cut_nodes(g)
        new_node_data = list(g.nodes(data=True))
        op_names = [op.name for op, order in new_node_data]

        assert "WireCut" not in op_names
        assert op_names.count("MeasureNode") == 3
        assert op_names.count("PrepareNode") == 3

        measure_counter = prepare_counter = 0

        for op, order in new_node_data:
            if op.name == "MeasureNode":
                assert op.wires.tolist() == [wire_cut_num[measure_counter]]
                assert order == wire_cut_order[measure_counter]
                measure_counter += 1
            elif op.name == "PrepareNode":
                assert op.wires.tolist() == [wire_cut_num[prepare_counter]]
                assert order == wire_cut_order[prepare_counter]
                prepare_counter += 1

    def test_successor_and_predecessor(self):
        """
        Tests the successor of the MeasureNode is the PrepareNode and the
        predecessor of the PrepareNode is the MeasureNode
        """
        wire_cut_num = 1

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.133, wires=1)
            qml.WireCut(wires=wire_cut_num)
            qml.CNOT(wires=[1, 2])
            qml.RY(0.543, wires=2)
            qml.expval(qml.PauliZ(wires=[0]))

        g = qcut.tape_to_graph(tape)
        qcut.replace_wire_cut_nodes(g)

        nodes = list(g.nodes)

        for node in nodes:
            if node.name == "MeasureNode":
                succ = list(g.succ[node])[0]
                pred = list(g.pred[node])[0]
                assert succ.name == "PrepareNode"
                assert pred.name == "RZ"
            if node.name == "PrepareNode":
                succ = list(g.succ[node])[0]
                pred = list(g.pred[node])[0]
                assert succ.name == "CNOT"
                assert pred.name == "MeasureNode"

    def test_wirecut_has_no_predecessor(self):
        """
        Tests a wirecut is replaced if it is the first operation in the tape
        i.e if it has no predecessor
        """

        with qml.tape.QuantumTape() as tape:
            qml.WireCut(wires=0)
            qml.RX(0.432, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.133, wires=1)
            qml.expval(qml.PauliZ(wires=[0]))

        g = qcut.tape_to_graph(tape)
        node_data = list(g.nodes(data=True))

        wire_cut_order = {"order": 0}

        qcut.replace_wire_cut_nodes(g)
        new_node_data = list(g.nodes(data=True))
        op_names = [op.name for op, order in new_node_data]

        assert "WireCut" not in op_names
        assert "MeasureNode" in op_names
        assert "PrepareNode" in op_names

        for op, order in new_node_data:
            if op.name == "MeasureNode":
                assert order == {"order": 0}
                pred = list(g.pred[op])
                assert pred == []
            elif op.name == "PrepareNode":
                assert order == {"order": 0}

    def test_wirecut_has_no_successor(self):
        """
        Tests a wirecut is replaced if it is the last operation in the tape
        i.e if it has no successor
        """

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.133, wires=1)
            qml.WireCut(wires=0)

        g = qcut.tape_to_graph(tape)
        node_data = list(g.nodes(data=True))

        wire_cut_order = {"order": 3}

        qcut.replace_wire_cut_nodes(g)
        new_node_data = list(g.nodes(data=True))
        op_names = [op.name for op, order in new_node_data]

        assert "WireCut" not in op_names
        assert "MeasureNode" in op_names
        assert "PrepareNode" in op_names

        for op, order in new_node_data:
            if op.name == "MeasureNode":
                assert order == {"order": 3}
            elif op.name == "PrepareNode":
                assert order == {"order": 3}
                succ = list(g.succ[op])
                assert succ == []

    def test_multi_wire_wirecut_successor_and_predecessor(self):
        """
        Tests the successors and predecessors of a multi-wire wirecut are
        correct
        """
        wire_cut_num = 1

        with qml.tape.QuantumTape() as tape:
            qml.CNOT(wires=[0, "a"])
            qml.CNOT(wires=["a", 2])
            qml.WireCut(wires=[0, "a", 2])
            qml.CZ(wires=[0, 2])
            qml.Toffoli(wires=[0, "a", 2])

        g = qcut.tape_to_graph(tape)
        qcut.replace_wire_cut_nodes(g)

        nodes = list(g.nodes)
        measure_nodes = [node for node in nodes if node.name == "MeasureNode"]
        prepare_nodes = [node for node in nodes if node.name == "PrepareNode"]

        assert len(measure_nodes) == len(prepare_nodes) == 3

        expected_meas_pred_wires = [[0, "a"], ["a", 2], ["a", 2]]
        expected_meas_pred_name = [
            "CNOT",
            "CNOT",
        ] * len(measure_nodes)
        expected_meas_succ_wires = [[0], ["a"], [2]]
        expected_meas_succ_name = ["PrepareNode"] * len(measure_nodes)

        expected_prep_pred_wires = expected_meas_succ_wires
        exepeted_prep_pred_name = ["MeasureNode"] * len(measure_nodes)
        expected_prep_succ_wires = [[0, 2], [0, "a", 2], [0, 2]]
        expected_prep_succ_name = ["CZ", "Toffoli", "CZ"]

        measure_pred = [list(g.pred[node])[0] for node in measure_nodes]
        measure_succ = [list(g.succ[node])[0] for node in measure_nodes]
        prep_pred = [list(g.pred[node])[0] for node in prepare_nodes]
        prep_succ = [list(g.succ[node])[0] for node in prepare_nodes]

        assert len(measure_nodes) == len(prepare_nodes) == 3

        compare_nodes(measure_pred, expected_meas_pred_wires, expected_meas_pred_name)
        compare_nodes(measure_succ, expected_meas_succ_wires, expected_meas_succ_name)
        compare_nodes(prep_pred, expected_prep_pred_wires, exepeted_prep_pred_name)
        compare_nodes(prep_succ, expected_prep_succ_wires, expected_prep_succ_name)

        for node in measure_nodes + prepare_nodes:
            in_edges = list(g.in_edges(node, data="wire"))
            out_edges = list(g.out_edges(node, data="wire"))
            assert len(in_edges) == 1
            assert len(out_edges) == 1

            _, _, wire_label_in = in_edges[0]
            _, _, wire_label_out = out_edges[0]

            assert wire_label_in == wire_label_out == node.wires.tolist()[0]


class TestFragmentGraph:
    """
    Tests that a cut graph is fragmented into subgraphs correctly
    """

    def test_subgraphs_of_multi_wirecut(self):
        """
        Tests that the subgraphs of a graph with multiple wirecuts contain the
        correct nodes and edges
        """

        g = qcut.tape_to_graph(multi_cut_tape)
        qcut.replace_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)

        assert len(subgraphs) == 4

        sub_0_expected_nodes = [
            (qcut.MeasureNode(wires=[0]), {"order": 2}),
            (qml.RX(0.432, wires=[0]), {"order": 0}),
        ]
        sub_1_expected_nodes = [
            (qml.RY(0.543, wires=["a"]), {"order": 1}),
            (qcut.PrepareNode(wires=[0]), {"order": 2}),
            (qcut.MeasureNode(wires=["a"]), {"order": 6}),
            (qml.RZ(0.24, wires=[0]), {"order": 4}),
            (qml.CNOT(wires=[0, "a"]), {"order": 3}),
            (qml.expval(qml.PauliZ(wires=[0])), {"order": 13}),
            (qml.RZ(0.133, wires=["a"]), {"order": 5}),
        ]
        sub_2_expected_nodes = [
            (qml.RX(0.432, wires=["a"]), {"order": 8}),
            (qcut.MeasureNode(wires=[2]), {"order": 9}),
            (qcut.PrepareNode(wires=["a"]), {"order": 6}),
            (qml.CNOT(wires=["a", 2]), {"order": 7}),
        ]
        sub_3_expected_nodes = [
            (qml.CNOT(wires=[2, 3]), {"order": 10}),
            (qml.RY(0.543, wires=[2]), {"order": 11}),
            (qcut.PrepareNode(wires=[2]), {"order": 9}),
            (qml.RZ(0.876, wires=[3]), {"order": 12}),
        ]
        expected_nodes = [
            sub_0_expected_nodes,
            sub_1_expected_nodes,
            sub_2_expected_nodes,
            sub_3_expected_nodes,
        ]

        sub_0_expected_edges = [
            (qml.RX(0.432, wires=[0]), qcut.MeasureNode(wires=[0]), {"wire": 0})
        ]
        sub_1_expected_edges = [
            (qcut.PrepareNode(wires=[0]), qml.CNOT(wires=[0, "a"]), {"wire": 0}),
            (qml.RZ(0.24, wires=[0]), qml.expval(qml.PauliZ(wires=[0])), {"wire": 0}),
            (qml.RZ(0.133, wires=["a"]), qcut.MeasureNode(wires=["a"]), {"wire": "a"}),
            (qml.CNOT(wires=[0, "a"]), qml.RZ(0.24, wires=[0]), {"wire": 0}),
            (qml.CNOT(wires=[0, "a"]), qml.RZ(0.133, wires=["a"]), {"wire": "a"}),
            (qml.RY(0.543, wires=["a"]), qml.CNOT(wires=[0, "a"]), {"wire": "a"}),
        ]
        sub_2_expected_edges = [
            (qcut.PrepareNode(wires=["a"]), qml.CNOT(wires=["a", 2]), {"wire": "a"}),
            (qml.CNOT(wires=["a", 2]), qml.RX(0.432, wires=["a"]), {"wire": "a"}),
            (qml.CNOT(wires=["a", 2]), qcut.MeasureNode(wires=[2]), {"wire": 2}),
        ]
        sub_3_expected_edges = [
            (qcut.PrepareNode(wires=[2]), qml.CNOT(wires=[2, 3]), {"wire": 2}),
            (qml.CNOT(wires=[2, 3]), qml.RY(0.543, wires=[2]), {"wire": 2}),
            (qml.CNOT(wires=[2, 3]), qml.RZ(0.876, wires=[3]), {"wire": 3}),
        ]
        expected_edges = [
            sub_0_expected_edges,
            sub_1_expected_edges,
            sub_2_expected_edges,
            sub_3_expected_edges,
        ]

        for subgraph, expected_n in zip(subgraphs, expected_nodes):
            compare_fragment_nodes(list(subgraph.nodes(data=True)), expected_n)

        for subgraph, expected_e in zip(subgraphs, expected_edges):
            compare_fragment_edges(list(subgraph.edges(data=True)), expected_e)

    def test_communication_graph(self):
        """
        Tests that the communication graph contains the correct nodes and edges
        """

        g = qcut.tape_to_graph(multi_cut_tape)
        qcut.replace_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)

        assert list(communication_graph.nodes) == list(range(4))

        expected_edge_data = [
            (0, 1, {"pair": (qcut.MeasureNode(wires=[0]), qcut.PrepareNode(wires=[0]))}),
            (1, 2, {"pair": (qcut.MeasureNode(wires=["a"]), qcut.PrepareNode(wires=["a"]))}),
            (2, 3, {"pair": (qcut.MeasureNode(wires=[2]), qcut.PrepareNode(wires=[2]))}),
        ]
        edge_data = list(communication_graph.edges(data=True))

        for edge, exp_edge in zip(edge_data, expected_edge_data):
            assert edge[0] == exp_edge[0]
            assert edge[1] == exp_edge[1]

            for node, exp_node in zip(edge[2]["pair"], exp_edge[2]["pair"]):
                assert node.name == exp_node.name
                assert node.wires.tolist() == exp_node.wires.tolist()

    def test_fragment_wirecut_first_and_last(self):
        """
        Tests a circuit with wirecut at the start and end is fragmented
        correctly
        """

        with qml.tape.QuantumTape() as tape:
            qml.WireCut(wires=0)
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires="a")
            qml.WireCut(wires="a")

        g = qcut.tape_to_graph(tape)
        qcut.replace_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)

        sub_0_expected_nodes = [
            (qcut.PrepareNode(wires=[0]), {"order": 0}),
            (qml.RX(0.432, wires=[0]), {"order": 1}),
        ]
        sub_1_expected_nodes = [
            (qcut.MeasureNode(wires=["a"]), {"order": 3}),
            (qml.RY(0.543, wires=["a"]), {"order": 2}),
        ]
        sub_2_expected_nodes = [(qcut.MeasureNode(wires=[0]), {"order": 0})]
        sub_3_expected_nodes = [(qcut.PrepareNode(wires=["a"]), {"order": 3})]

        expected_nodes = [
            sub_0_expected_nodes,
            sub_1_expected_nodes,
            sub_2_expected_nodes,
            sub_3_expected_nodes,
        ]

        sub_0_expected_edges = [
            (qcut.PrepareNode(wires=[0]), qml.RX(0.432, wires=[0]), {"wire": 0})
        ]
        sub_1_expected_edges = [
            (qml.RY(0.543, wires=["a"]), qcut.MeasureNode(wires=["a"]), {"wire": "a"})
        ]
        sub_2_expected_edges = []
        sub_3_expected_edges = []

        expected_edges = [
            sub_0_expected_edges,
            sub_1_expected_edges,
            sub_2_expected_edges,
            sub_3_expected_edges,
        ]

        for subgraph, expected_n in zip(subgraphs, expected_nodes):
            compare_fragment_nodes(list(subgraph.nodes(data=True)), expected_n)

        for subgraph, expected_e in zip(subgraphs, expected_edges):
            compare_fragment_edges(list(subgraph.edges(data=True)), expected_e)

    def test_communication_graph_persistence(self):
        """
        Tests that when `fragment_graph` is repeatedly applied the
        communication graph is the same each time.
        """

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires=1)
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RX(0.432, wires=1)
            qml.RY(0.543, wires=2)
            qml.expval(qml.PauliZ(wires=[0]))

        g = qcut.tape_to_graph(tape)
        qcut.replace_wire_cut_nodes(g)
        subgraphs_0, communication_graph_0 = qcut.fragment_graph(g)
        subgraphs_1, communication_graph_1 = qcut.fragment_graph(g)

        assert communication_graph_0.nodes == communication_graph_1.nodes
        assert communication_graph_0.edges == communication_graph_1.edges
