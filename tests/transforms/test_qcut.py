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
import copy
import itertools
import string
import sys
from itertools import product

import pytest
from networkx import MultiDiGraph
from scipy.stats import unitary_group

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms import qcut
from pennylane.wires import Wires

I, X, Y, Z = (
    np.eye(2),
    qml.PauliX.compute_matrix(),
    qml.PauliY.compute_matrix(),
    qml.PauliZ.compute_matrix(),
)

states_pure = [
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1]) / np.sqrt(2),
    np.array([1, 1j]) / np.sqrt(2),
]

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


def kron(*args):
    """Multi-argument kronecker product"""
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        return np.kron(args[0], args[1])
    else:
        return np.kron(args[0], kron(*args[1:]))


# tape containing mid-circuit measurements
with qml.tape.QuantumTape() as mcm_tape:
    qml.Hadamard(wires=0)
    qml.RX(0.432, wires=0)
    qml.RY(0.543, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.WireCut(wires=1)
    qml.CNOT(wires=[1, 2])
    qml.WireCut(wires=1)
    qml.RZ(0.321, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=2)
    qml.WireCut(wires=1)
    qml.CNOT(wires=[1, 2])
    qml.WireCut(wires=1)
    qml.CNOT(wires=[0, 1])
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


def compare_tapes(tape, expected_tape):
    """
    Helper function to compare tapes
    """

    assert set(tape.wires) == set(expected_tape.wires)
    assert tape.get_parameters() == expected_tape.get_parameters()

    for op, exp_op in zip(tape.operations, expected_tape.operations):
        if (
            op.name == "PrepareNode"
        ):  # The exact ordering of PrepareNodes w.r.t wires varies on each call
            assert exp_op.name == "PrepareNode"
        else:
            assert op.name == exp_op.name
            assert op.wires.tolist() == exp_op.wires.tolist()

    for meas, exp_meas in zip(tape.measurements, expected_tape.measurements):
        assert meas.return_type.name == exp_meas.return_type.name
        assert meas.obs.name == exp_meas.obs.name
        assert meas.wires.tolist() == exp_meas.wires.tolist()


def compare_measurements(meas1, meas2):
    """
    Helper function to compare measurements
    """
    assert meas1.return_type.name == meas2.return_type.name
    obs1 = meas1.obs
    obs2 = meas2.obs
    assert np.array(obs1.name == obs2.name).all()
    assert obs1.wires.tolist() == obs2.wires.tolist()


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


class TestGraphToTape:
    """Tests that directed multigraphs are correctly converted to tapes"""

    def test_graph_to_tape(self):
        """
        Tests that directed multigraphs, containing MeasureNodes and
        PrepareNodes, are correctly converted to tapes
        """
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires="a")
            qml.WireCut(wires=[0, "a"])
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

        g = qcut.tape_to_graph(tape)
        qcut.replace_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)

        tapes = [qcut.graph_to_tape(sg) for sg in subgraphs]

        with qml.tape.QuantumTape() as tape_0:
            qml.RX(0.432, wires=[0])
            qcut.MeasureNode(wires=[0])

        with qml.tape.QuantumTape() as tape_1:
            qml.RY(0.543, wires=["a"])
            qcut.MeasureNode(wires=["a"])

        with qml.tape.QuantumTape() as tape_2:
            qcut.PrepareNode(wires=[0])
            qcut.PrepareNode(wires=["a"])
            qml.CNOT(wires=[0, "a"])
            qml.RZ(0.24, wires=[0])
            qml.RZ(0.133, wires=["a"])
            qcut.MeasureNode(wires=["a"])
            qml.expval(qml.PauliZ(wires=[0]))

        with qml.tape.QuantumTape() as tape_3:
            qcut.PrepareNode(wires=["a"])
            qml.CNOT(wires=["a", 2])
            qml.RX(0.432, wires=["a"])
            qcut.MeasureNode(wires=[2])

        with qml.tape.QuantumTape() as tape_4:
            qcut.PrepareNode(wires=[2])
            qml.CNOT(wires=[2, 3])
            qml.RY(0.543, wires=[2])
            qml.RZ(0.876, wires=[3])

        expected_tapes = [tape_0, tape_1, tape_2, tape_3, tape_4]

        for tape, expected_tape in zip(tapes, expected_tapes):
            compare_tapes(tape, expected_tape)

    def test_mid_circuit_measurement(self):
        """
        Tests a circuit that is fragmented into subgraphs that
        include mid-circuit measurements, ensuring that the
        generated circuits apply the deferred measurement principle.
        """
        g = qcut.tape_to_graph(mcm_tape)
        qcut.replace_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)

        tapes = [qcut.graph_to_tape(sg) for sg in subgraphs]

        assert tapes[0].wires == Wires([0, 1, 2, 3])
        assert tapes[1].wires == Wires([1, 2, 0])

        for tape in tapes:
            for i, op in enumerate(tape.operations):
                if isinstance(op, qcut.MeasureNode):
                    try:
                        next_op = tape.operations[i + 1]
                        if isinstance(next_op, qcut.PrepareNode):
                            assert op.wires != next_op.wires
                    except IndexError:
                        assert len(tape.operations) == i + 1

    def test_mid_circuit_measurements_fragments(self):
        """
        Considers a circuit that is fragmented into subgraphs that
        include mid-circuit measurements and tests that the subgraphs
        are correctly converted to the expected tapes
        """
        g = qcut.tape_to_graph(mcm_tape)
        qcut.replace_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)

        tapes = [qcut.graph_to_tape(sg) for sg in subgraphs]

        with qml.tape.QuantumTape() as tape_0:
            qml.Hadamard(wires=[0])
            qml.RX(0.432, wires=[0])
            qml.RY(0.543, wires=[1])
            qml.CNOT(wires=[0, 1])
            qcut.MeasureNode(wires=[1])
            qcut.PrepareNode(wires=[2])
            qml.RZ(0.321, wires=[2])
            qml.CNOT(wires=[0, 2])
            qcut.MeasureNode(wires=[2])
            qcut.PrepareNode(wires=[3])
            qml.CNOT(wires=[0, 3])
            qml.expval(qml.PauliZ(wires=[0]))

        with qml.tape.QuantumTape() as tape_1:
            qcut.PrepareNode(wires=[1])
            qml.CNOT(wires=[1, 2])
            qcut.MeasureNode(wires=[1])
            qml.Hadamard(wires=[2])
            qcut.PrepareNode(wires=[0])
            qml.CNOT(wires=[0, 2])
            qcut.MeasureNode(wires=[0])

        expected_tapes = [tape_0, tape_1]

        for tape, expected_tape in zip(tapes, expected_tapes):
            compare_tapes(tape, expected_tape)

    def test_multiple_conversions(self):
        """
        Tests that the original tape is unaffected by cutting pipeline and can
        be used multiple times to give a consistent output.
        """
        # preserve original tape data for later comparison
        copy_tape = copy.deepcopy(mcm_tape)

        g1 = qcut.tape_to_graph(mcm_tape)
        qcut.replace_wire_cut_nodes(g1)
        subgraphs1, communication_graph1 = qcut.fragment_graph(g1)

        tapes1 = [qcut.graph_to_tape(sg) for sg in subgraphs1]

        g2 = qcut.tape_to_graph(mcm_tape)
        qcut.replace_wire_cut_nodes(g2)
        subgraphs2, communication_graph2 = qcut.fragment_graph(g2)

        tapes2 = [qcut.graph_to_tape(sg) for sg in subgraphs2]

        compare_tapes(copy_tape, mcm_tape)

        for tape1, tape2 in zip(tapes1, tapes2):
            compare_tapes(tape1, tape2)


class TestGetMeasurements:
    """Tests for the _get_measurements function"""

    def test_multiple_measurements_raises(self):
        """Tests if the function raises a ValueError when more than 1 fixed measurement is
        specified"""
        group = [qml.Identity(0)]
        meas = [qml.expval(qml.PauliX(1)), qml.expval(qml.PauliY(2))]

        with pytest.raises(ValueError, match="with a single output measurement"):
            qcut._get_measurements(group, meas)

    def test_non_expectation_raises(self):
        """Tests if the function raises a ValueError when the fixed measurement is not an
        expectation value"""
        group = [qml.Identity(0)]
        meas = [qml.var(qml.PauliX(1))]

        with pytest.raises(ValueError, match="with expectation value measurements"):
            qcut._get_measurements(group, meas)

    def test_no_measurements(self):
        """Tests if the function simply processes ``group`` into expectation values when an empty
        list of fixed measurements is provided"""
        group = [qml.Identity(0)]
        meas = []

        out = qcut._get_measurements(group, meas)

        assert len(out) == len(group)
        assert out[0].return_type is qml.measure.Expectation
        assert out[0].obs.name == "Identity"
        assert out[0].obs.wires[0] == 0

    def test_single_measurement(self):
        """Tests if the function behaves as expected for a typical example"""
        group = [qml.PauliX(0) @ qml.PauliZ(2), qml.PauliX(0)]
        meas = [qml.expval(qml.PauliZ(1))]

        out = qcut._get_measurements(group, meas)

        assert len(out) == 2
        assert [o.return_type for o in out] == [qml.measure.Expectation, qml.measure.Expectation]

        obs = [o.obs for o in out]

        assert obs[0].wires.tolist() == [1, 0, 2]
        assert obs[1].wires.tolist() == [1, 0]

        assert [o.name for o in obs[0].obs] == ["PauliZ", "PauliX", "PauliZ"]
        assert [o.name for o in obs[1].obs] == ["PauliZ", "PauliX"]


class TestExpandFragmentTapes:
    """
    Tests that fragment tapes are correctly expanded to all configurations
    """

    def test_expand_fragment_tapes(self):
        """
        Tests that a fragment tape expands correctly
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
        subgraphs, communication_graph = qcut.fragment_graph(g)
        tapes = [qcut.graph_to_tape(sg) for sg in subgraphs]

        fragment_configurations = [qcut.expand_fragment_tapes(tape) for tape in tapes]
        frag_tapes_meas = fragment_configurations[0][0]
        frag_tapes_prep = fragment_configurations[1][0]

        assert len(frag_tapes_meas) == 3
        assert len(frag_tapes_prep) == 4

        frag_meas_ops = [
            qml.RX(0.432, wires=[0]),
            qml.RY(0.543, wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(0.24, wires=[0]),
            qml.RZ(0.133, wires=[1]),
        ]

        with qml.tape.QuantumTape() as tape_00:
            for op in frag_meas_ops:
                qml.apply(op)
            qml.expval(qml.PauliZ(wires=[0]) @ qml.Identity(wires=[1]))
            qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]))

        with qml.tape.QuantumTape() as tape_01:
            for op in frag_meas_ops:
                qml.apply(op)
            qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliX(wires=[1]))

        with qml.tape.QuantumTape() as tape_02:
            for op in frag_meas_ops:
                qml.apply(op)
            qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliY(wires=[1]))

        frag_meas_expected_tapes = [tape_00, tape_01, tape_02]

        frag_prep_ops = [qml.CNOT(wires=[1, 2]), qml.RX(0.432, wires=[1]), qml.RY(0.543, wires=[2])]

        with qml.tape.QuantumTape() as tape_10:
            qml.Identity(wires=[1])
            for op in frag_prep_ops:
                qml.apply(op)
            qml.expval(qml.Identity(wires=[1]))

        with qml.tape.QuantumTape() as tape_11:
            qml.PauliX(wires=[1])
            for op in frag_prep_ops:
                qml.apply(op)
            qml.expval(qml.Identity(wires=[1]))

        with qml.tape.QuantumTape() as tape_12:
            qml.Hadamard(wires=[1])
            for op in frag_prep_ops:
                qml.apply(op)
            qml.expval(qml.Identity(wires=[1]))

        with qml.tape.QuantumTape() as tape_13:
            qml.Hadamard(wires=[1])
            qml.S(wires=[1])
            for op in frag_prep_ops:
                qml.apply(op)
            qml.expval(qml.Identity(wires=[1]))

        frag_prep_expected_tapes = [tape_10, tape_11, tape_12, tape_13]

        for tape_meas, exp_tape_meas in zip(frag_tapes_meas, frag_meas_expected_tapes):
            compare_tapes(tape_meas, exp_tape_meas)

        for tape_prep, exp_tape_1 in zip(frag_tapes_prep, frag_prep_expected_tapes):
            compare_tapes(tape_prep, exp_tape_1)

    def test_multi_qubit_expansion_measurements(self):
        """
        Tests that a circuit with multiple MeasureNodes gives the correct
        measurements after expansion
        """

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=[0])
            qml.RX(0.432, wires=[0])
            qml.RY(0.543, wires=[1])
            qml.CNOT(wires=[0, 1])
            qcut.MeasureNode(wires=[1])
            qcut.PrepareNode(wires=[2])
            qml.RZ(0.321, wires=[2])
            qml.CNOT(wires=[0, 2])
            qcut.MeasureNode(wires=[2])
            qcut.PrepareNode(wires=[3])
            qml.CNOT(wires=[0, 3])
            qml.expval(qml.PauliZ(wires=[0]))

        # Here we have a fragment tape containing 2 MeasureNode and
        # PrepareNode pairs. This give 3**2 = 9 groups of Pauli measurements
        # and 4**2 = 16 preparations and thus 9*16 = 144 tapes.
        fragment_configurations = qcut.expand_fragment_tapes(tape)
        frag_tapes = fragment_configurations[0]

        assert len(frag_tapes) == 144

        all_expected_groups = [
            [
                qml.expval(qml.PauliZ(wires=[0]) @ qml.Identity(wires=[1])),
                qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[2])),
                qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1])),
                qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2])),
            ],
            [
                qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliX(wires=[2])),
                qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliX(wires=[2])),
            ],
            [
                qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliY(wires=[2])),
                qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliY(wires=[2])),
            ],
            [
                qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliX(wires=[1])),
                qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliZ(wires=[2])),
            ],
            [qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliX(wires=[2]))],
            [qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliY(wires=[2]))],
            [
                qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliY(wires=[1])),
                qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliZ(wires=[2])),
            ],
            [qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliX(wires=[2]))],
            [qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliY(wires=[2]))],
        ]

        all_measurements = [tape.measurements for tape in frag_tapes]

        # The 9 unique measurements are repeated 16 times since there are 4**2
        # preparations. This list prepares the indexing.
        index_list = []
        for i in range(len(frag_tapes)):
            if i % 9 == 0:
                c = 0
            index_list.append((c, i))
            c += 1

        for exp_i, i in index_list:
            expected_group = all_expected_groups[exp_i]
            group = all_measurements[i]
            for measurement, expected_measurement in zip(expected_group, group):
                compare_measurements(measurement, expected_measurement)

    def test_multi_qubit_expansion_preparation(self):
        """
        Tests that a circuit with multiple PrepareNodes gives the correct
        preparation after expansion
        """

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=[0])
            qml.RX(0.432, wires=[0])
            qml.RY(0.543, wires=[1])
            qml.CNOT(wires=[0, 1])
            qcut.MeasureNode(wires=[1])
            qcut.PrepareNode(wires=[2])
            qml.RZ(0.321, wires=[2])
            qml.CNOT(wires=[0, 2])
            qcut.MeasureNode(wires=[2])
            qcut.PrepareNode(wires=[3])
            qml.CNOT(wires=[0, 3])
            qml.expval(qml.PauliZ(wires=[0]))

        fragment_configurations = qcut.expand_fragment_tapes(tape)
        frag_tapes = fragment_configurations[0]

        prep_ops = [[qml.Identity], [qml.PauliX], [qml.Hadamard], [qml.Hadamard, qml.S]]
        prep_combos = list(product(prep_ops, prep_ops))
        expected_preps = [pc for pc in prep_combos for _ in range(9)]

        for ep, tape in zip(expected_preps, frag_tapes):
            wire2_ops = [op for op in tape.operations if op.wires == Wires(2)]
            wire3_ops = [op for op in tape.operations if op.wires == Wires(3)]

            wire2_exp = ep[0]
            wire3_exp = ep[1]

            wire2_prep_ops = wire2_ops[: len(wire2_exp)]
            wire3_prep_ops = wire3_ops[: len(wire2_exp)]

            for wire2_prep_op, wire2_exp_op in zip(wire2_prep_ops, wire2_exp):
                assert type(wire2_prep_op) == wire2_exp_op

            for wire3_prep_op, wire3_exp_op in zip(wire3_prep_ops, wire3_exp):
                assert type(wire3_prep_op) == wire3_exp_op


class TestContractTensors:
    """Tests for the contract_tensors function"""

    t = [np.arange(4), np.arange(4, 8)]
    m = [[qcut.MeasureNode(wires=0)], []]
    p = [[], [qcut.PrepareNode(wires=0)]]
    edge_dict = {"pair": (m[0][0], p[1][0])}
    g = MultiDiGraph([(0, 1, edge_dict)])
    expected_result = np.dot(*t)

    @pytest.mark.parametrize("use_opt_einsum", [False, True])
    def test_basic(self, use_opt_einsum):
        """Test that the correct answer is returned for a simple contraction scenario"""
        if use_opt_einsum:
            pytest.importorskip("opt_einsum")
        res = qcut.contract_tensors(self.t, self.g, self.p, self.m, use_opt_einsum=use_opt_einsum)

        assert np.allclose(res, self.expected_result)

    def test_fail_import(self, monkeypatch):
        """Test if an ImportError is raised when opt_einsum is requested but not installed"""

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "opt_einsum", None)

            with pytest.raises(ImportError, match="The opt_einsum package is required"):
                qcut.contract_tensors(self.t, self.g, self.p, self.m, use_opt_einsum=True)

    def test_run_out_of_symbols(self, monkeypatch):
        """Test if a ValueError is raised when there are not enough symbols in the
        use_opt_einsum = False setting"""

        with monkeypatch.context() as m:
            m.setattr(string, "ascii_letters", "")
            with pytest.raises(ValueError, match="Set the use_opt_einsum argument to True"):
                qcut.contract_tensors(self.t, self.g, self.p, self.m, use_opt_einsum=False)

    params = [0.3, 0.5]

    expected_grad_0 = (
        np.cos(params[0]) * np.cos(params[1])
        + 2 * np.cos(params[0]) * np.sin(params[0]) * np.cos(params[1]) ** 2
        + 3 * np.cos(params[0]) * np.sin(params[0]) ** 2 * np.cos(params[1]) ** 3
    )
    expected_grad_1 = (
        -np.sin(params[0]) * np.sin(params[1])
        - 2 * np.sin(params[0]) ** 2 * np.sin(params[1]) * np.cos(params[1])
        - 3 * np.sin(params[0]) ** 3 * np.sin(params[1]) * np.cos(params[1]) ** 2
    )
    expected_grad = np.array([expected_grad_0, expected_grad_1])

    @pytest.mark.parametrize("use_opt_einsum", [True, False])
    def test_basic_grad_autograd(self, use_opt_einsum):
        """Test if the basic contraction is differentiable using the autograd interface"""
        if use_opt_einsum:
            pytest.importorskip("opt_einsum")

        def contract(params):
            t1 = np.asarray([np.sin(params[0]) ** i for i in range(4)])
            t2 = np.asarray([np.cos(params[1]) ** i for i in range(4)])
            t = [t1, t2]
            r = qcut.contract_tensors(t, self.g, self.p, self.m, use_opt_einsum=use_opt_einsum)
            return r

        params = np.array(self.params, requires_grad=True)
        grad = qml.grad(contract)(params)

        assert np.allclose(grad, self.expected_grad)

    @pytest.mark.usefixtures("skip_if_no_torch_support")
    @pytest.mark.parametrize("use_opt_einsum", [True, False])
    def test_basic_grad_torch(self, use_opt_einsum):
        """Test if the basic contraction is differentiable using the torch interface"""
        if use_opt_einsum:
            pytest.importorskip("opt_einsum")
        import torch

        params = torch.tensor(self.params, requires_grad=True)

        t1 = torch.stack([torch.sin(params[0]) ** i for i in range(4)])
        t2 = torch.stack([torch.cos(params[1]) ** i for i in range(4)])
        t = [t1, t2]
        r = qcut.contract_tensors(t, self.g, self.p, self.m, use_opt_einsum=use_opt_einsum)

        r.backward()
        grad = params.grad

        assert np.allclose(grad, self.expected_grad)

    @pytest.mark.usefixtures("skip_if_no_tf_support")
    @pytest.mark.parametrize("use_opt_einsum", [True, False])
    def test_basic_grad_tf(self, use_opt_einsum):
        """Test if the basic contraction is differentiable using the tf interface"""
        if use_opt_einsum:
            pytest.importorskip("opt_einsum")
        import tensorflow as tf

        params = tf.Variable(self.params)

        with tf.GradientTape() as tape:
            t1 = tf.stack([tf.sin(params[0]) ** i for i in range(4)])
            t2 = tf.stack([tf.cos(params[1]) ** i for i in range(4)])
            t = [t1, t2]
            r = qcut.contract_tensors(t, self.g, self.p, self.m, use_opt_einsum=use_opt_einsum)

        grad = tape.gradient(r, params)

        assert np.allclose(grad, self.expected_grad)

    @pytest.mark.usefixtures("skip_if_no_jax_support")
    @pytest.mark.parametrize("use_opt_einsum", [True, False])
    def test_basic_grad_jax(self, use_opt_einsum):
        """Test if the basic contraction is differentiable using the jax interface"""
        if use_opt_einsum:
            pytest.importorskip("opt_einsum")
        import jax
        from jax import numpy as np

        params = np.array(self.params)

        def contract(params):
            t1 = np.stack([np.sin(params[0]) ** i for i in range(4)])
            t2 = np.stack([np.cos(params[1]) ** i for i in range(4)])
            t = [t1, t2]
            r = qcut.contract_tensors(t, self.g, self.p, self.m, use_opt_einsum=use_opt_einsum)
            return r

        grad = jax.grad(contract)(params)

        assert np.allclose(grad, self.expected_grad)

    @pytest.mark.parametrize("use_opt_einsum", [True, False])
    def test_advanced(self, mocker, use_opt_einsum):
        """Test if the contraction works as expected for a more complicated example based
        upon the circuit:

        with qml.tape.QuantumTape() as tape:
            qml.QubitUnitary(np.eye(2 ** 3), wires=[0, 1, 2])
            qml.QubitUnitary(np.eye(2 ** 2), wires=[3, 4])

            qml.Barrier(wires=0)
            qml.WireCut(wires=[1, 2, 3])

            qml.QubitUnitary(np.eye(2 ** 1), wires=[0])
            qml.QubitUnitary(np.eye(2 ** 4), wires=[1, 2, 3, 4])

            qml.WireCut(wires=[1, 2, 3, 4])

            qml.QubitUnitary(np.eye(2 ** 3), wires=[0, 1, 2])
            qml.QubitUnitary(np.eye(2 ** 2), wires=[3, 4])
        """
        if use_opt_einsum:
            opt_einsum = pytest.importorskip("opt_einsum")
            spy = mocker.spy(opt_einsum, "contract")
        else:
            spy = mocker.spy(qml.math, "einsum")

        t = [
            np.arange(4**8).reshape((4,) * 8),
            np.arange(4**4).reshape((4,) * 4),
            np.arange(4**2).reshape((4,) * 2),
        ]
        m = [
            [
                qcut.MeasureNode(wires=3),
                qcut.MeasureNode(wires=1),
                qcut.MeasureNode(wires=2),
                qcut.MeasureNode(wires=3),
                qcut.MeasureNode(wires=4),
            ],
            [
                qcut.MeasureNode(wires=1),
                qcut.MeasureNode(wires=2),
            ],
            [],
        ]
        p = [
            [
                qcut.PrepareNode(wires=1),
                qcut.PrepareNode(wires=2),
                qcut.PrepareNode(wires=3),
            ],
            [
                qcut.PrepareNode(wires=1),
                qcut.PrepareNode(wires=2),
            ],
            [
                qcut.PrepareNode(wires=4),
                qcut.PrepareNode(wires=3),
            ],
        ]
        edges = [
            (0, 0, 0, {"pair": (m[0][0], p[0][2])}),
            (0, 1, 0, {"pair": (m[0][1], p[1][0])}),
            (0, 1, 1, {"pair": (m[0][2], p[1][1])}),
            (0, 2, 0, {"pair": (m[0][4], p[2][0])}),
            (0, 2, 1, {"pair": (m[0][3], p[2][1])}),
            (1, 0, 0, {"pair": (m[1][0], p[0][0])}),
            (1, 0, 1, {"pair": (m[1][1], p[0][1])}),
        ]
        g = MultiDiGraph(edges)

        res = qcut.contract_tensors(t, g, p, m, use_opt_einsum=use_opt_einsum)

        eqn = spy.call_args[0][0]
        expected_eqn = "abccdegf,deab,fg"

        assert eqn == expected_eqn
        assert np.allclose(res, np.einsum(eqn, *t))


class TestQCutProcessingFn:
    """Tests for the qcut_processing_fn and contained functions"""

    def test_to_tensors(self, monkeypatch):
        """Test that _to_tensors correctly reshapes the flat list of results into the original
        tensors according to the supplied prepare_nodes and measure_nodes. Uses a mock function
        for _process_tensor since we do not need to process the tensors."""
        prepare_nodes = [[None] * 3, [None] * 2, [None] * 1, [None] * 4]
        measure_nodes = [[None] * 2, [None] * 2, [None] * 3, [None] * 3]
        tensors = [
            np.arange(4**5).reshape((4,) * 5),
            np.arange(4**4).reshape((4,) * 4),
            np.arange(4**4).reshape((4,) * 4),
            np.arange(4**7).reshape((4,) * 7),
        ]
        results = np.concatenate([t.flatten() for t in tensors])

        def mock_process_tensor(r, np, nm):
            return qml.math.reshape(r, (4,) * (np + nm))

        with monkeypatch.context() as m:
            m.setattr(qcut, "_process_tensor", mock_process_tensor)
            tensors_out = qcut._to_tensors(results, prepare_nodes, measure_nodes)

        for t1, t2 in zip(tensors, tensors_out):
            assert np.allclose(t1, t2)

    def test_to_tensors_raises(self):
        """Tests if a ValueError is raised when a results vector is passed to _to_tensors with a
        size that is incompatible with the prepare_nodes and measure_nodes arguments"""
        prepare_nodes = [[None] * 3]
        measure_nodes = [[None] * 2]
        tensors = [np.arange(4**5).reshape((4,) * 5), np.arange(4)]
        results = np.concatenate([t.flatten() for t in tensors])

        with pytest.raises(ValueError, match="should be a flat list of length 1024"):
            qcut._to_tensors(results, prepare_nodes, measure_nodes)

    @pytest.mark.parametrize("interface", ["autograd.numpy", "tensorflow", "torch", "jax.numpy"])
    @pytest.mark.parametrize("n", [1, 2])
    def test_process_tensor(self, n, interface):
        """Test if the tensor returned by _process_tensor is equal to the expected value"""
        lib = pytest.importorskip(interface)

        U = unitary_group.rvs(2**n, random_state=1967)

        # First, create target process tensor
        basis = np.array([I, X, Y, Z]) / np.sqrt(2)
        prod_inp = itertools.product(range(4), repeat=n)
        prod_out = itertools.product(range(4), repeat=n)

        results = []

        # Calculates U_{ijkl} = Tr((b[k] x b[l]) U (b[i] x b[j]) U*)
        # See Sec. II. A. of https://arxiv.org/abs/1909.07534, below Eq. (2).
        for inp, out in itertools.product(prod_inp, prod_out):
            input = kron(*[basis[i] for i in inp])
            output = kron(*[basis[i] for i in out])
            results.append(np.trace(output @ U @ input @ U.conj().T))

        target_tensor = np.array(results).reshape((4,) * (2 * n))

        # Now, create the input results vector found from executing over the product of |0>, |1>,
        # |+>, |+i> inputs and using the grouped Pauli terms for measurements
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev)
        def f(state, measurement):
            qml.QubitStateVector(state, wires=range(n))
            qml.QubitUnitary(U, wires=range(n))
            return [qml.expval(qml.grouping.string_to_pauli_word(m)) for m in measurement]

        prod_inp = itertools.product(range(4), repeat=n)
        prod_out = qml.grouping.partition_pauli_group(n)

        results = []

        for inp, out in itertools.product(prod_inp, prod_out):
            input = kron(*[states_pure[i] for i in inp])
            results.append(f(input, out))

        results = qml.math.cast_like(np.concatenate(results), lib.ones(1))

        # Now apply _process_tensor
        tensor = qcut._process_tensor(results, n, n)
        assert np.allclose(tensor, target_tensor)

    @pytest.mark.parametrize("use_opt_einsum", [True, False])
    def test_qcut_processing_fn(self, use_opt_einsum):
        """Test if qcut_processing_fn returns the expected answer when applied to a simple circuit
        that is cut up into three fragments:
        0: ──RX(0.5)─|─RY(0.6)─|─RX(0.8)──┤ ⟨Z⟩
        """
        if use_opt_einsum:
            pytest.importorskip("opt_einsum")

        ### Find the expected result
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def f(x, y, z):
            qml.RX(x, wires=0)
            ### CUT HERE
            qml.RY(y, wires=0)
            ### CUT HERE
            qml.RX(z, wires=0)
            return qml.expval(qml.PauliZ(0))

        x, y, z = 0.5, 0.6, 0.8
        expected_result = f(x, y, z)

        ### Find the result using qcut_processing_fn

        meas_basis = [I, Z, X, Y]
        states = [np.outer(s, s.conj()) for s in states_pure]
        zero_proj = states[0]

        u1 = qml.RX.compute_matrix(x)
        u2 = qml.RY.compute_matrix(y)
        u3 = qml.RX.compute_matrix(z)
        t1 = np.array([np.trace(b @ u1 @ zero_proj @ u1.conj().T) for b in meas_basis])
        t2 = np.array([[np.trace(b @ u2 @ s @ u2.conj().T) for b in meas_basis] for s in states])
        t3 = np.array([np.trace(Z @ u3 @ s @ u3.conj().T) for s in states])

        res = [t1, t2.flatten(), t3]
        p = [[], [qcut.PrepareNode(wires=0)], [qcut.PrepareNode(wires=0)]]
        m = [[qcut.MeasureNode(wires=0)], [qcut.MeasureNode(wires=0)], []]

        edges = [
            (0, 1, 0, {"pair": (m[0][0], p[1][0])}),
            (1, 2, 0, {"pair": (m[1][0], p[2][0])}),
        ]
        g = MultiDiGraph(edges)

        result = qcut.qcut_processing_fn(res, g, p, m, use_opt_einsum=use_opt_einsum)
        assert np.allclose(result, expected_result)

    @pytest.mark.parametrize("use_opt_einsum", [True, False])
    def test_qcut_processing_fn_autograd(self, use_opt_einsum):
        """Test if qcut_processing_fn handles the gradient as expected in the autograd interface
        using a simple example"""
        if use_opt_einsum:
            pytest.importorskip("opt_einsum")

        x = np.array(0.9, requires_grad=True)

        def f(x):
            t1 = x * np.arange(4)
            t2 = x**2 * np.arange(16).reshape((4, 4))
            t3 = np.sin(x * np.pi / 2) * np.arange(4)

            res = [t1, t2.flatten(), t3]
            p = [[], [qcut.PrepareNode(wires=0)], [qcut.PrepareNode(wires=0)]]
            m = [[qcut.MeasureNode(wires=0)], [qcut.MeasureNode(wires=0)], []]

            edges = [
                (0, 1, 0, {"pair": (m[0][0], p[1][0])}),
                (1, 2, 0, {"pair": (m[1][0], p[2][0])}),
            ]
            g = MultiDiGraph(edges)

            return qcut.qcut_processing_fn(res, g, p, m, use_opt_einsum=use_opt_einsum)

        grad = qml.grad(f)(x)
        expected_grad = (
            3 * x**2 * np.sin(x * np.pi / 2) + x**3 * np.cos(x * np.pi / 2) * np.pi / 2
        ) * f(1)

        assert np.allclose(grad, expected_grad)

    @pytest.mark.parametrize("use_opt_einsum", [True, False])
    def test_qcut_processing_fn_tf(self, use_opt_einsum):
        """Test if qcut_processing_fn handles the gradient as expected in the TF interface
        using a simple example"""
        if use_opt_einsum:
            pytest.importorskip("opt_einsum")
        tf = pytest.importorskip("tensorflow")

        x = tf.Variable(0.9, dtype=tf.float64)

        def f(x):
            x = tf.cast(x, dtype=tf.float64)
            t1 = x * tf.range(4, dtype=tf.float64)
            t2 = x**2 * tf.range(16, dtype=tf.float64)
            t3 = tf.sin(x * np.pi / 2) * tf.range(4, dtype=tf.float64)

            res = [t1, t2, t3]
            p = [[], [qcut.PrepareNode(wires=0)], [qcut.PrepareNode(wires=0)]]
            m = [[qcut.MeasureNode(wires=0)], [qcut.MeasureNode(wires=0)], []]

            edges = [
                (0, 1, 0, {"pair": (m[0][0], p[1][0])}),
                (1, 2, 0, {"pair": (m[1][0], p[2][0])}),
            ]
            g = MultiDiGraph(edges)

            return qcut.qcut_processing_fn(res, g, p, m, use_opt_einsum=use_opt_einsum)

        with tf.GradientTape() as tape:
            res = f(x)

        grad = tape.gradient(res, x)
        expected_grad = (
            3 * x**2 * np.sin(x * np.pi / 2) + x**3 * np.cos(x * np.pi / 2) * np.pi / 2
        ) * f(1)

        assert np.allclose(grad, expected_grad)

    @pytest.mark.parametrize("use_opt_einsum", [True, False])
    def test_qcut_processing_fn_torch(self, use_opt_einsum):
        """Test if qcut_processing_fn handles the gradient as expected in the torch interface
        using a simple example"""
        if use_opt_einsum:
            pytest.importorskip("opt_einsum")
        torch = pytest.importorskip("torch")

        x = torch.tensor(0.9, requires_grad=True, dtype=torch.float64)

        def f(x):
            t1 = x * torch.arange(4)
            t2 = x**2 * torch.arange(16)
            t3 = torch.sin(x * np.pi / 2) * torch.arange(4)

            res = [t1, t2, t3]
            p = [[], [qcut.PrepareNode(wires=0)], [qcut.PrepareNode(wires=0)]]
            m = [[qcut.MeasureNode(wires=0)], [qcut.MeasureNode(wires=0)], []]

            edges = [
                (0, 1, 0, {"pair": (m[0][0], p[1][0])}),
                (1, 2, 0, {"pair": (m[1][0], p[2][0])}),
            ]
            g = MultiDiGraph(edges)

            return qcut.qcut_processing_fn(res, g, p, m, use_opt_einsum=use_opt_einsum)

        res = f(x)
        res.backward()
        grad = x.grad

        x_ = x.detach().numpy()
        f1 = f(torch.tensor(1, dtype=torch.float64))
        expected_grad = (
            3 * x_**2 * np.sin(x_ * np.pi / 2) + x_**3 * np.cos(x_ * np.pi / 2) * np.pi / 2
        ) * f1

        assert np.allclose(grad.detach().numpy(), expected_grad)

    @pytest.mark.parametrize("use_opt_einsum", [True, False])
    def test_qcut_processing_fn_jax(self, use_opt_einsum):
        """Test if qcut_processing_fn handles the gradient as expected in the jax interface
        using a simple example"""
        if use_opt_einsum:
            pytest.importorskip("opt_einsum")
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        x = jnp.array(0.9)

        def f(x):
            t1 = x * jnp.arange(4)
            t2 = x**2 * jnp.arange(16).reshape((4, 4))
            t3 = jnp.sin(x * np.pi / 2) * jnp.arange(4)

            res = [t1, t2.flatten(), t3]
            p = [[], [qcut.PrepareNode(wires=0)], [qcut.PrepareNode(wires=0)]]
            m = [[qcut.MeasureNode(wires=0)], [qcut.MeasureNode(wires=0)], []]

            edges = [
                (0, 1, 0, {"pair": (m[0][0], p[1][0])}),
                (1, 2, 0, {"pair": (m[1][0], p[2][0])}),
            ]
            g = MultiDiGraph(edges)

            return qcut.qcut_processing_fn(res, g, p, m, use_opt_einsum=use_opt_einsum)

        grad = jax.grad(f)(x)
        expected_grad = (
            3 * x**2 * np.sin(x * np.pi / 2) + x**3 * np.cos(x * np.pi / 2) * np.pi / 2
        ) * f(1)

        assert np.allclose(grad, expected_grad)
