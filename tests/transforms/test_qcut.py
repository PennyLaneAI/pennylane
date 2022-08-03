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
from os import environ
from pathlib import Path

import pytest
from flaky import flaky
from networkx import MultiDiGraph, number_of_selfloops
from scipy.stats import unitary_group

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms import qcut
from pennylane.wires import Wires

pytestmark = pytest.mark.qcut

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
    qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[3]))


def kron(*args):
    """Multi-argument kronecker product"""
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        return np.kron(args[0], args[1])
    else:
        return np.kron(args[0], kron(*args[1:]))


def fn(x):
    """
    Classical processing function for MC circuit cutting
    """
    if x[0] == 0 and x[1] == 0:
        return 1
    if x[0] == 0 and x[1] == 1:
        return -1
    if x[0] == 1 and x[1] == 0:
        return -1
    if x[0] == 1 and x[1] == 1:
        return 1


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


with qml.tape.QuantumTape() as frag0:
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
    qml.sample(qml.Projector([1], wires=[0]))
    qml.sample(qml.Projector([1], wires=[3]))

with qml.tape.QuantumTape() as frag1:
    qcut.PrepareNode(wires=[0])
    qml.CNOT(wires=[0, 1])
    qcut.MeasureNode(wires=[0])
    qml.Hadamard(wires=[1])
    qcut.PrepareNode(wires=[2])
    qml.CNOT(wires=[2, 1])
    qcut.MeasureNode(wires=[2])
    qml.sample(qml.Projector([1], wires=[1]))


frag_edge_data = [
    (0, 1, {"pair": (frag0.operations[4], frag1.operations[0])}),
    (1, 0, {"pair": (frag1.operations[2], frag0.operations[5])}),
    (0, 1, {"pair": (frag0.operations[8], frag1.operations[4])}),
    (1, 0, {"pair": (frag1.operations[6], frag0.operations[9])}),
]


def compare_nodes(nodes, expected_wires, expected_names):
    """Helper function to compare nodes of directed multigraph"""

    for node, exp_wire in zip(nodes, expected_wires):
        assert node.wires.tolist() == exp_wire

    for node, exp_name in zip(nodes, expected_names):
        assert node.name == exp_name


def compare_fragment_nodes(node_data, expected_data):
    """Helper function to compare nodes of fragment graphs"""
    assert len(node_data) == len(expected_data)
    expected = [(exp_data[0].name, exp_data[0].wires, exp_data[1]) for exp_data in expected_data]

    for data in node_data:
        # The exact ordering of node_data varies on each call
        assert (data[0].name, data[0].wires, data[1]) in expected


def compare_fragment_edges(edge_data, expected_data):
    """Helper function to compare fragment edges"""
    assert len(edge_data) == len(expected_data)
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


def test_node_ids(monkeypatch):
    """
    Tests that the `MeasureNode` and `PrepareNode` return the correct id
    """
    with monkeypatch.context() as m:
        m.setattr("uuid.uuid4", lambda: "some_string")

        mn = qcut.MeasureNode(wires=0)
        pn = qcut.PrepareNode(wires=0)

        assert mn.id == "some_string"
        assert pn.id == "some_string"


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

    def test_split_sample_measurement(self):
        """
        Test that a circuit with a single sample measurement over all wires is
        correctly converted to a graph with a distinct node for each wire sampled
        """

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=1)
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.sample(wires=[0, 1, 2])

        g = qcut.tape_to_graph(tape)

        expected_nodes = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.PauliX(wires=[1]),
            qml.WireCut(wires=[1]),
            qml.CNOT(wires=[1, 2]),
            qml.sample(qml.Projector([1], wires=[0])),
            qml.sample(qml.Projector([1], wires=[1])),
            qml.sample(qml.Projector([1], wires=[2])),
        ]

        for node, expected_node in zip(g.nodes, expected_nodes):
            assert node.name == expected_node.name
            assert node.wires == expected_node.wires

            if getattr(node, "obs", None) is not None:
                assert node.return_type is qml.measurements.Sample
                assert node.obs.name == expected_node.obs.name

    def test_sample_tensor_obs(self):
        """
        Test that a circuit with a sample measurement of a tensor product of
        observables raises the correct error.
        """

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=1)
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.sample(qml.PauliX(0) @ qml.PauliY(1))

        with pytest.raises(ValueError, match="Sampling from tensor products of observables "):
            qcut.tape_to_graph(tape)

    def test_multiple_obs_samples(self):
        """
        Test that a circuit with multiple sample measurements of observables
        over individual wires is supported.
        """

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=1)
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.sample(qml.PauliZ(0))
            qml.sample(qml.PauliZ(1))
            qml.sample(qml.PauliZ(2))

        g = qcut.tape_to_graph(tape)

        expected_nodes = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.PauliX(wires=[1]),
            qml.WireCut(wires=[1]),
            qml.CNOT(wires=[1, 2]),
            qml.sample(qml.PauliZ(wires=[0])),
            qml.sample(qml.PauliZ(wires=[1])),
            qml.sample(qml.PauliZ(wires=[2])),
        ]

        for node, expected_node in zip(g.nodes, expected_nodes):
            assert node.name == expected_node.name
            assert node.wires == expected_node.wires

            if getattr(node, "obs", None) is not None:
                assert node.return_type is qml.measurements.Sample
                assert node.obs.name == expected_node.obs.name


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
            (qml.expval(qml.PauliZ(wires=[3])), {"order": 13}),
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
            (qml.RZ(0.876, wires=[3]), qml.expval(qml.PauliZ(wires=[3])), {"wire": 3}),
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

    def test_subgraphs_of_multi_wirecut_with_disconnected_components(self):
        """
        Tests that the subgraphs of a graph with multiple wirecuts contain the
        correct nodes and edges. Focuses on the case where fragmentation results in two fragments
        that are disconnected from the final measurements.
        """
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

        g = qcut.tape_to_graph(multi_cut_tape)
        qcut.replace_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)

        assert len(subgraphs) == 2

        sub_0_expected_nodes = [
            (qcut.MeasureNode(wires=[0]), {"order": 2}),
            (qml.RX(0.432, wires=[0]), {"order": 0}),
        ]
        sub_1_expected_nodes = [
            (qml.RY(0.543, wires=["a"]), {"order": 1}),
            (qcut.PrepareNode(wires=[0]), {"order": 2}),
            (qml.RZ(0.24, wires=[0]), {"order": 4}),
            (qml.CNOT(wires=[0, "a"]), {"order": 3}),
            (qml.expval(qml.PauliZ(wires=[0])), {"order": 13}),
            (qml.RZ(0.133, wires=["a"]), {"order": 5}),
        ]
        expected_nodes = [
            sub_0_expected_nodes,
            sub_1_expected_nodes,
        ]

        sub_0_expected_edges = [
            (qml.RX(0.432, wires=[0]), qcut.MeasureNode(wires=[0]), {"wire": 0})
        ]
        sub_1_expected_edges = [
            (qcut.PrepareNode(wires=[0]), qml.CNOT(wires=[0, "a"]), {"wire": 0}),
            (qml.RZ(0.24, wires=[0]), qml.expval(qml.PauliZ(wires=[0])), {"wire": 0}),
            (qml.CNOT(wires=[0, "a"]), qml.RZ(0.24, wires=[0]), {"wire": 0}),
            (qml.CNOT(wires=[0, "a"]), qml.RZ(0.133, wires=["a"]), {"wire": "a"}),
            (qml.RY(0.543, wires=["a"]), qml.CNOT(wires=[0, "a"]), {"wire": "a"}),
        ]
        expected_edges = [
            sub_0_expected_edges,
            sub_1_expected_edges,
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

    def test_contained_cut(self):
        """Tests that fragmentation ignores `MeasureNode` and `PrepareNode` pairs that do not
        result in a disconnection"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(0.4, wires=0)
            qml.expval(qml.PauliZ(0))

        g = qcut.tape_to_graph(tape)
        qcut.replace_wire_cut_nodes(g)
        fragments, communication_graph = qcut.fragment_graph(g)
        assert len(fragments) == 1
        assert number_of_selfloops(communication_graph) == 0

    def test_fragment_sample_circuit(self):
        """
        Test that a circuit containing a sample measurement is fragmented
        correctly
        """

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=1)
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.sample(wires=[0, 1, 2])

        g = qcut.tape_to_graph(tape)
        qcut.replace_wire_cut_nodes(g)
        fragments, communication_graph = qcut.fragment_graph(g)

        sub_0_expected_nodes = [
            (qml.Hadamard(wires=[0]), {"order": 0}),
            (qml.CNOT(wires=[0, 1]), {"order": 1}),
            (qml.PauliX(wires=[1]), {"order": 2}),
            (qml.sample(qml.Projector([1], wires=[0])), {"order": 5}),
            (qcut.MeasureNode(wires=[1]), {"order": 3}),
        ]

        sub_1_expected_nodes = [
            (qml.sample(qml.Projector([1], wires=[2])), {"order": 5}),
            (qml.sample(qml.Projector([1], wires=[1])), {"order": 5}),
            (qcut.PrepareNode(wires=[1]), {"order": 3}),
            (qml.CNOT(wires=[1, 2]), {"order": 4}),
        ]

        expected_nodes = [
            sub_0_expected_nodes,
            sub_1_expected_nodes,
        ]

        sub_0_expected_edges = [
            (qml.Hadamard(wires=[0]), qml.CNOT(wires=[0, 1]), {"wire": 0}),
            (qml.CNOT(wires=[0, 1]), qml.PauliX(wires=[1]), {"wire": 1}),
            (qml.CNOT(wires=[0, 1]), qml.sample(qml.Projector([1], wires=[0])), {"wire": 0}),
            (qml.PauliX(wires=[1]), qcut.MeasureNode(wires=[1]), {"wire": 1}),
        ]

        sub_1_expected_edges = [
            (qcut.PrepareNode(wires=[1]), qml.CNOT(wires=[1, 2]), {"wire": 1}),
            (qml.CNOT(wires=[1, 2]), qml.sample(qml.Projector([1], wires=[1])), {"wire": 1}),
            (qml.CNOT(wires=[1, 2]), qml.sample(qml.Projector([1], wires=[2])), {"wire": 2}),
        ]

        expected_edges = [
            sub_0_expected_edges,
            sub_1_expected_edges,
        ]

        for fragment, expected_n in zip(fragments, expected_nodes):
            compare_fragment_nodes(list(fragment.nodes(data=True)), expected_n)

        for fragment, expected_e in zip(fragments, expected_edges):
            compare_fragment_edges(list(fragment.edges(data=True)), expected_e)


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

    def test_identity(self):
        """Tests that the graph_to_tape function correctly performs the inverse of the tape_to_graph
        function, including converting a tensor product expectation value into separate nodes in the
        graph returned by tape_to_graph, and then combining those nodes again into a single tensor
        product in the circuit returned by graph_to_tape"""

        with qml.tape.QuantumTape() as tape:
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        graph = qcut.tape_to_graph(tape)
        tape_out = qcut.graph_to_tape(graph)

        compare_tapes(tape, tape_out)
        assert len(tape_out.measurements) == 1

    def test_change_obs_wires(self):
        """Tests that the graph_to_tape function correctly swaps the wires of observables when
        the tape contains mid-circuit measurements"""

        with qml.tape.QuantumTape() as tape:
            qml.CNOT(wires=[0, 1])
            qcut.MeasureNode(wires=1)
            qcut.PrepareNode(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(1))

        graph = qcut.tape_to_graph(tape)
        tape_out = qcut.graph_to_tape(graph)

        m = tape_out.measurements
        assert len(m) == 1
        assert m[0].wires == Wires([2])
        assert m[0].obs.name == "PauliZ"

    def test_single_sample_meas_conversion(self):
        """
        Tests that subgraphs with sample nodes are correctly converted to
        fragment tapes
        """

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=1)
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.sample(wires=[0, 1, 2])

        g = qcut.tape_to_graph(tape)
        qcut.replace_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)

        tapes = [qcut.graph_to_tape(sg) for sg in subgraphs]

        frag0_expected_meas = [qml.sample(qml.Projector([1], wires=[0]))]
        frag1_expected_meas = [
            qml.sample(qml.Projector([1], wires=[1])),
            qml.sample(qml.Projector([1], wires=[2])),
        ]

        for meas, expected_meas in zip(tapes[0].measurements, frag0_expected_meas):
            compare_measurements(meas, expected_meas)

        # For tapes with multiple measurements, the ordering varies
        # so we check the set of wires rather that the order
        for meas, expected_meas in zip(tapes[1].measurements, frag1_expected_meas):
            assert meas.return_type is qml.measurements.Sample
            assert isinstance(meas.obs, qml.Projector)
            assert meas.obs.wires in {Wires(1), Wires(2)}

    def test_sample_mid_circuit_meas(self):
        """
        Test that a circuit with sample measurements, that is partitioned into
        fragments requiring mid-circuit measurements, has its fragment subgraphs
        correctly converted into tapes
        """

        with qml.tape.QuantumTape() as tape:
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
            qml.sample(wires=[0, 1, 2])

        g = qcut.tape_to_graph(tape)
        qcut.replace_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)

        tapes = [qcut.graph_to_tape(sg) for sg in subgraphs]

        frag0_expected_meas = [
            qml.sample(qml.Projector([1], wires=[0])),
            qml.sample(qml.Projector([1], wires=[3])),
        ]
        frag1_expected_meas = [qml.sample(qml.Projector([1], wires=[2]))]

        for meas, expected_meas in zip(tapes[0].measurements, frag1_expected_meas):
            assert meas.return_type is qml.measurements.Sample
            assert isinstance(meas.obs, qml.Projector)
            assert meas.obs.wires in {Wires(0), Wires(3)}

        for meas, expected_meas in zip(tapes[1].measurements, frag1_expected_meas):
            compare_measurements(meas, expected_meas)

        # sample measurements should not exist on the same wire as MeasureNodes at this stage
        f0_sample_wires = [meas.wires for meas in tapes[0].measurements]
        f0_measurenode_wires = [
            op.wires for op in tapes[0].operations if isinstance(op, qcut.MeasureNode)
        ]

        f1_sample_wires = [meas.wires for meas in tapes[1].measurements]
        f1_measurenode_wires = [
            op.wires for op in tapes[1].operations if isinstance(op, qcut.MeasureNode)
        ]

        for f0_mn_wire in set(f0_measurenode_wires):
            assert f0_mn_wire not in set(f0_sample_wires)

        for f1_mn_wire in set(f1_measurenode_wires):
            assert f1_mn_wire not in set(f1_sample_wires)

    def test_mixed_measurements(self):
        """
        Tests thats a subgraph containing mixed measurements raises the correct
        error message.
        """

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=1)
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.sample(wires=[0, 1])
            qml.expval(qml.PauliZ(wires=2))

        g = qcut.tape_to_graph(tape)
        qcut.replace_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)

        with pytest.raises(
            ValueError, match="Only a single return type can be used for measurement "
        ):
            [qcut.graph_to_tape(sg) for sg in subgraphs]

    def test_unsupported_meas(self):
        """
        Tests that a subgraph containing an unsupported measurement raises the
        correct error message.
        """
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.var(qml.PauliZ(wires=0))

        g = qcut.tape_to_graph(tape)
        qcut.replace_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)

        with pytest.raises(
            ValueError,
            match="Invalid return type. Only expectation value and sampling measurements ",
        ):
            [qcut.graph_to_tape(sg) for sg in subgraphs]


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
        assert out[0].return_type is qml.measurements.Expectation
        assert out[0].obs.name == "Identity"
        assert out[0].obs.wires[0] == 0

    def test_single_measurement(self):
        """Tests if the function behaves as expected for a typical example"""
        group = [qml.PauliX(0) @ qml.PauliZ(2), qml.PauliX(0)]
        meas = [qml.expval(qml.PauliZ(1))]

        out = qcut._get_measurements(group, meas)

        assert len(out) == 2
        assert [o.return_type for o in out] == [
            qml.measurements.Expectation,
            qml.measurements.Expectation,
        ]

        obs = [o.obs for o in out]

        assert obs[0].wires.tolist() == [1, 0, 2]
        assert obs[1].wires.tolist() == [1, 0]

        assert [o.name for o in obs[0].obs] == ["PauliZ", "PauliX", "PauliZ"]
        assert [o.name for o in obs[1].obs] == ["PauliZ", "PauliX"]


class TestExpandFragmentTapes:
    """
    Tests that fragment tapes are correctly expanded to all configurations
    """

    def test_expand_fragment_tape(self):
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
            qml.expval(qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=2))

        g = qcut.tape_to_graph(tape)
        qcut.replace_wire_cut_nodes(g)
        subgraphs, communication_graph = qcut.fragment_graph(g)
        tapes = [qcut.graph_to_tape(sg) for sg in subgraphs]

        fragment_configurations = [qcut.expand_fragment_tape(tape) for tape in tapes]
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
            qml.expval(qml.PauliZ(wires=[2]))

        with qml.tape.QuantumTape() as tape_11:
            qml.PauliX(wires=[1])
            for op in frag_prep_ops:
                qml.apply(op)
            qml.expval(qml.PauliZ(wires=[2]))

        with qml.tape.QuantumTape() as tape_12:
            qml.Hadamard(wires=[1])
            for op in frag_prep_ops:
                qml.apply(op)
            qml.expval(qml.PauliZ(wires=[2]))

        with qml.tape.QuantumTape() as tape_13:
            qml.Hadamard(wires=[1])
            qml.S(wires=[1])
            for op in frag_prep_ops:
                qml.apply(op)
            qml.expval(qml.PauliZ(wires=[2]))

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
        fragment_configurations = qcut.expand_fragment_tape(tape)
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

        fragment_configurations = qcut.expand_fragment_tape(tape)
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

    def test_no_measure_node_observables(self):
        """
        Tests that a fragment with no MeasureNodes give the correct
        configurations
        """

        with qml.tape.QuantumTape() as frag:
            qml.RY(0.543, wires=[1])
            qcut.PrepareNode(wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.24, wires=[0])
            qml.RZ(0.133, wires=[1])
            qml.expval(qml.PauliZ(wires=[0]))

        expanded_tapes, prep_nodes, meas_nodes = qcut.expand_fragment_tape(frag)

        ops = [
            qml.CNOT(wires=[0, 1]),
            qml.RZ(0.24, wires=[0]),
            qml.RZ(0.133, wires=[1]),
            qml.expval(qml.PauliZ(wires=[0])),
        ]

        with qml.tape.QuantumTape() as config1:
            qml.RY(0.543, wires=[1])
            qml.Identity(wires=[0])
            for optr in ops:
                qml.apply(optr)

        with qml.tape.QuantumTape() as config2:
            qml.RY(0.543, wires=[1])
            qml.PauliX(wires=[0])
            for optr in ops:
                qml.apply(optr)

        with qml.tape.QuantumTape() as config3:
            qml.RY(0.543, wires=[1])
            qml.Hadamard(wires=[0])
            for optr in ops:
                qml.apply(optr)

        with qml.tape.QuantumTape() as config4:
            qml.RY(0.543, wires=[1])
            qml.Hadamard(wires=[0])
            qml.S(wires=[0])
            for optr in ops:
                qml.apply(optr)

        expected_configs = [config1, config2, config3, config4]

        for tape, config in zip(expanded_tapes, expected_configs):
            compare_tapes(tape, config)


class TestExpandFragmentTapesMC:
    """
    Tests that fragment tapes are correctly expanded to all random configurations
    for the Monte Carlo sampling technique.
    """

    def test_expand_mc(self, monkeypatch):
        """
        Tests that fragment configurations are generated correctly using the
        `expand_fragment_tapes_mc` function.
        """
        with qml.tape.QuantumTape() as tape0:
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            qcut.MeasureNode(wires=[1])
            qml.sample(qml.Projector([1], wires=[0]))

        with qml.tape.QuantumTape() as tape1:
            qcut.PrepareNode(wires=[1])
            qml.CNOT(wires=[1, 2])
            qml.sample(qml.Projector([1], wires=[1]))
            qml.sample(qml.Projector([1], wires=[2]))

        tapes = [tape0, tape1]

        edge_data = {"pair": (tape0.operations[2], tape1.operations[0])}
        communication_graph = MultiDiGraph([(0, 1, edge_data)])

        fixed_choice = np.array([[4, 0, 1]])
        with monkeypatch.context() as m:
            m.setattr(np.random, "choice", lambda a, size, replace: fixed_choice)
            fragment_configurations, settings = qcut.expand_fragment_tapes_mc(
                tapes, communication_graph, 3
            )

        assert np.allclose(settings, fixed_choice)

        frag_0_ops = [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])]
        frag_0_expected_meas = [
            [qml.sample(qml.Projector([1], wires=[0])), qml.sample(qml.PauliY(wires=[1]))],
            [qml.sample(qml.Projector([1], wires=[0])), qml.sample(qml.Identity(wires=[1]))],
            [qml.sample(qml.Projector([1], wires=[0])), qml.sample(qml.Identity(wires=[1]))],
        ]

        expected_tapes_0 = []
        for meas in frag_0_expected_meas:
            with qml.tape.QuantumTape() as expected_tape:
                for op in frag_0_ops:
                    qml.apply(op)
                for m in meas:
                    qml.apply(m)

            expected_tapes_0.append(expected_tape)

        for tape, exp_tape in zip(fragment_configurations[0], expected_tapes_0):
            compare_tapes(tape, exp_tape)

        frag_1_expected_preps = [
            [qml.Hadamard(wires=[1]), qml.S(wires=[1])],
            [qml.Identity(wires=[1])],
            [qml.PauliX(wires=[1])],
        ]

        frag_1_ops_and_meas = [
            qml.CNOT(wires=[1, 2]),
            qml.sample(qml.Projector([1], wires=[1])),
            qml.sample(qml.Projector([1], wires=[2])),
        ]

        expected_tapes_1 = []
        for preps in frag_1_expected_preps:
            with qml.tape.QuantumTape() as expected_tape:
                for prep in preps:
                    qml.apply(prep)
                for op in frag_1_ops_and_meas:
                    qml.apply(op)
            expected_tapes_1.append(expected_tape)

        for tape, exp_tape in zip(fragment_configurations[1], expected_tapes_1):
            compare_tapes(tape, exp_tape)

    def test_expand_multinode_frag(self, monkeypatch):
        """
        Tests that fragments with multiple measure and prepare nodes are
        expanded correctly.
        """

        frags = [frag0, frag1]

        communication_graph = MultiDiGraph(frag_edge_data)

        fixed_choice = np.array([[4, 6], [1, 2], [2, 3], [3, 0]])
        with monkeypatch.context() as m:
            m.setattr(
                np.random,
                "choice",
                lambda a, size, replace: fixed_choice,
            )
            fragment_configurations, settings = qcut.expand_fragment_tapes_mc(
                frags, communication_graph, 2
            )

        assert np.allclose(settings, fixed_choice)

        with qml.tape.QuantumTape() as config1:
            qml.Hadamard(wires=[0])
            qml.RX(0.432, wires=[0])
            qml.RY(0.543, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=[2])
            qml.RZ(0.321, wires=[2])
            qml.CNOT(wires=[0, 2])
            qml.PauliX(wires=[3])
            qml.Hadamard(wires=[3])
            qml.CNOT(wires=[0, 3])
            qml.sample(qml.Projector([1], wires=[0]))
            qml.sample(qml.Projector([1], wires=[3]))
            qml.sample(qml.PauliY(wires=[1]))
            qml.sample(qml.Identity(wires=[2]))

        with qml.tape.QuantumTape() as config2:
            qml.Hadamard(wires=[0])
            qml.RX(0.432, wires=[0])
            qml.RY(0.543, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=[2])
            qml.Hadamard(wires=[2])
            qml.RZ(0.321, wires=[2])
            qml.CNOT(wires=[0, 2])
            qml.Identity(wires=[3])
            qml.CNOT(wires=[0, 3])
            qml.sample(qml.Projector([1], wires=[0]))
            qml.sample(qml.Projector([1], wires=[3]))
            qml.sample(qml.PauliZ(wires=[1]))
            qml.sample(qml.PauliX(wires=[2]))

        expected_configs = [config1, config2]

        # check first fragment configs only for brevity
        for config, exp_config in zip(fragment_configurations[0], expected_configs):
            compare_tapes(config, exp_config)

    def test_mc_measurements(self):
        """
        Tests that the measurements functions used in MC configutations are
        correct
        """
        wire = "a"

        tapes = []
        for M in qcut.MC_MEASUREMENTS:
            with qml.tape.QuantumTape() as tape:
                M(wire)
            tapes.append(tape)

        expected_measurements = [
            qml.sample(qml.Identity(wires=["a"])),
            qml.sample(qml.Identity(wires=["a"])),
            qml.sample(qml.PauliX(wires=["a"])),
            qml.sample(qml.PauliX(wires=["a"])),
            qml.sample(qml.PauliY(wires=["a"])),
            qml.sample(qml.PauliY(wires=["a"])),
            qml.sample(qml.PauliZ(wires=["a"])),
            qml.sample(qml.PauliZ(wires=["a"])),
        ]

        measurements = [tape.measurements[0] for tape in tapes]

        for meas, exp_meas in zip(measurements, expected_measurements):
            compare_measurements(meas, exp_meas)

    def test_mc_state_prep(self):
        """
        Tests that the state preparation functions used in MC configutations are
        correct
        """

        wire = 3

        tapes = []
        for S in qcut.MC_STATES:
            with qml.tape.QuantumTape() as tape:
                S(wire)
            tapes.append(tape)

        expected_operations = [
            [qml.Identity(wires=[3])],
            [qml.PauliX(wires=[3])],
            [qml.Hadamard(wires=[3])],
            [qml.PauliX(wires=[3]), qml.Hadamard(wires=[3])],
            [qml.Hadamard(wires=[3]), qml.S(wires=[3])],
            [qml.PauliX(wires=[3]), qml.Hadamard(wires=[3]), qml.S(wires=[3])],
            [qml.Identity(wires=[3])],
            [qml.PauliX(wires=[3])],
        ]

        operations = [tape.operations for tape in tapes]

        for ops, expected_ops in zip(operations, expected_operations):
            for op, exp_op in zip(ops, expected_ops):
                assert op.name == exp_op.name
                assert op.wires == exp_op.wires


class TestMCPostprocessing:
    """
    Tests that the postprocessing for circuits containing sample measurements
    gives the correct results.
    """

    @pytest.mark.parametrize("interface", ["autograd.numpy", "tensorflow", "torch", "jax.numpy"])
    def test_sample_postprocess(self, interface):
        """
        Tests that the postprocessing for the generic sampling case gives the
        correct result
        """
        lib = pytest.importorskip(interface)
        fragment_tapes = [frag0, frag1]

        communication_graph = MultiDiGraph(frag_edge_data)
        shots = 3

        fixed_samples = [
            np.array([[1.0], [0.0], [1.0], [1.0]]),
            np.array([[0.0], [0.0], [1.0], [-1.0]]),
            np.array([[0.0], [1.0], [1.0], [-1.0]]),
            np.array([[0.0], [-1.0], [1.0]]),
            np.array([[0.0], [-1.0], [-1.0]]),
            np.array([[1.0], [1.0], [1.0]]),
        ]
        convert_fixed_samples = [qml.math.convert_like(fs, lib.ones(1)) for fs in fixed_samples]

        postprocessed = qcut.qcut_processing_fn_sample(
            convert_fixed_samples, communication_graph, shots
        )
        expected_postprocessed = [np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])]

        assert np.allclose(postprocessed[0], expected_postprocessed[0])
        assert type(convert_fixed_samples[0]) == type(postprocessed[0])

    @pytest.mark.parametrize("interface", ["autograd.numpy", "tensorflow", "torch", "jax.numpy"])
    def test_mc_sample_postprocess(self, interface, mocker):
        """
        Tests that the postprocessing for the generic sampling case gives the
        correct result
        """
        lib = pytest.importorskip(interface)
        fragment_tapes = [frag0, frag1]

        communication_graph = MultiDiGraph(frag_edge_data)
        shots = 3

        fixed_samples = [
            np.array([[1.0], [0.0], [1.0], [1.0]]),
            np.array([[0.0], [0.0], [1.0], [-1.0]]),
            np.array([[0.0], [1.0], [1.0], [-1.0]]),
            np.array([[0.0], [-1.0], [1.0]]),
            np.array([[0.0], [-1.0], [-1.0]]),
            np.array([[1.0], [1.0], [1.0]]),
        ]
        convert_fixed_samples = [qml.math.convert_like(fs, lib.ones(1)) for fs in fixed_samples]

        fixed_settings = np.array([[0, 7, 1], [5, 7, 2], [1, 0, 3], [5, 1, 1]])

        spy_prod = mocker.spy(np, "prod")
        spy_hstack = mocker.spy(np, "hstack")

        postprocessed = qcut.qcut_processing_fn_mc(
            convert_fixed_samples, communication_graph, fixed_settings, shots, fn
        )

        expected = -85.33333333333333

        prod_args = [
            np.array([1.0, 1.0, -1.0, 1.0]),
            [0.5, -0.5, 0.5, -0.5],
            np.array([1.0, -1.0, -1.0, -1.0]),
            [-0.5, -0.5, 0.5, 0.5],
            np.array([1.0, -1.0, 1.0, 1.0]),
            [0.5, 0.5, -0.5, 0.5],
        ]

        hstack_args = [
            [np.array([1.0, 0.0]), np.array([0.0])],
            [np.array([1.0, 1.0]), np.array([-1.0, 1.0])],
            [np.array([0.0, 0.0]), np.array([0.0])],
            [np.array([1.0, -1.0]), np.array([-1.0, -1.0])],
            [np.array([0.0, 1.0]), np.array([1.0])],
            [np.array([1.0, -1.0]), np.array([1.0, 1.0])],
        ]

        for arg, expected_arg in zip(spy_prod.call_args_list, prod_args):
            assert np.allclose(arg[0][0], expected_arg)

        for args, expected_args in zip(spy_hstack.call_args_list, hstack_args):
            for arg, expected_arg in zip(args[0][0], expected_args):
                assert np.allclose(arg, expected_arg)

        assert np.isclose(postprocessed, expected)
        assert type(convert_fixed_samples[0]) == type(postprocessed)

    def test_reshape_results(self):
        """
        Tests that results are reshaped correctly using the `_reshape_results`
        helper function
        """

        results = [
            np.array([[0.0], [-1.0]]),
            np.array([[1.0], [1.0]]),
            np.array([[1.0], [1.0]]),
            np.array([[1.0], [1.0]]),
            np.array([[1.0], [1.0]]),
            np.array([[0.0], [0.0]]),
        ]

        expected_reshaped = [
            [np.array([0.0, -1.0]), np.array([1.0, 1.0])],
            [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
            [np.array([1.0, 1.0]), np.array([0.0, 0.0])],
        ]

        shots = 3

        reshaped = qcut._reshape_results(results, shots)

        for resh, exp_resh in zip(reshaped, expected_reshaped):
            for arr, exp_arr in zip(resh, exp_resh):
                assert np.allclose(arr, exp_arr)

    def test_classical_processing_error(self):
        """
        Tests that the correct error is given if the classical processing
        function gives output outside of the interval [-1, 1]
        """

        fragment_tapes = [frag0, frag1]

        communication_graph = MultiDiGraph(frag_edge_data)
        shots = 3

        fixed_samples = [
            np.array([[1.0], [0.0], [1.0], [1.0]]),
            np.array([[0.0], [0.0], [1.0], [-1.0]]),
            np.array([[0.0], [1.0], [1.0], [-1.0]]),
            np.array([[0.0], [-1.0], [1.0]]),
            np.array([[0.0], [-1.0], [-1.0]]),
            np.array([[1.0], [1.0], [1.0]]),
        ]

        def fn(x):
            if x[0] == 0 and x[1] == 0:
                return 2
            if x[0] == 0 and x[1] == 1:
                return -2
            if x[0] == 1 and x[1] == 0:
                return -2
            if x[0] == 1 and x[1] == 1:
                return 2

        fixed_settings = np.array([[0, 7, 1], [5, 7, 2], [1, 0, 3], [5, 1, 1]])

        with pytest.raises(ValueError, match="The classical processing function supplied must "):
            qcut.qcut_processing_fn_mc(
                fixed_samples, communication_graph, fixed_settings, shots, fn
            )


class TestCutCircuitMCTransform:
    """
    Tests that the `cut_circuit_mc` transform gives the correct results.
    """

    @flaky(max_runs=3)
    def test_cut_circuit_mc_expval(self):
        """
        Tests that a circuit containing sampling measurements can be cut and
        recombined to give the correct expectation value
        """

        dev_sim = qml.device("default.qubit", wires=3)

        @qml.qnode(dev_sim)
        def target_circuit(v):
            qml.RX(v, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RX(v, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))

        dev = qml.device("default.qubit", wires=2, shots=10000)

        @qml.cut_circuit_mc(fn)
        @qml.qnode(dev)
        def circuit(v):
            qml.RX(v, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RX(v, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(wires=[0, 2])

        v = 0.319
        cut_res_mc = circuit(v)

        target = target_circuit(v)
        assert np.isclose(cut_res_mc, target, atol=0.1)  # not guaranteed to pass each time

    def test_cut_circuit_mc_sample(self):
        """
        Tests that a circuit containing sampling measurements can be cut and
        postprocessed to return bitstrings of the original circuit size.
        """

        dev = qml.device("default.qubit", wires=3, shots=100)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(wires=[0, 2])

        v = 0.319
        target = circuit(v)

        cut_circuit_bs = qcut.cut_circuit_mc(circuit)
        cut_res_bs = cut_circuit_bs(v)

        assert cut_res_bs.shape == target.shape
        assert type(cut_res_bs) == type(target)

    def test_override_samples(self):
        """
        Tests that the number of shots used on a device can be temporarily
        altered when executing the QNode
        """
        shots = 100
        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.cut_circuit_mc
        @qml.qnode(dev)
        def cut_circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(wires=[0, 2])

        v = 0.319

        temp_shots = 333
        cut_res = cut_circuit(v, shots=temp_shots)

        assert cut_res.shape == (temp_shots, 2)

        cut_res_original = cut_circuit(v)

        assert cut_res_original.shape == (shots, 2)

    def test_no_shots(self):
        """
        Tests that the correct error message is given if a device is provided
        without shots
        """

        dev = qml.device("default.qubit", wires=2)

        @qml.cut_circuit_mc
        @qml.qnode(dev)
        def cut_circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(wires=[0, 2])

        v = 0.319
        with pytest.raises(ValueError, match="A shots value must be provided in the device "):
            cut_circuit(v)

    def test_sample_obs_error(self):
        """
        Tests that a circuit with sample measurements containing observables
        gives the correct error
        """
        shots = 100
        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.cut_circuit_mc
        @qml.qnode(dev)
        def cut_circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(qml.PauliZ(0))

        v = 0.319
        with pytest.raises(ValueError, match="The Monte Carlo circuit cutting workflow only "):
            cut_circuit(v)

    def test_transform_shots_error(self):
        """
        Tests that the correct error is given when a `shots` argument is passed
        when transforming a qnode
        """

        dev = qml.device("default.qubit", wires=2)

        @qml.cut_circuit_mc(shots=456)
        @qml.qnode(dev)
        def cut_circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(wires=[0, 2])

        v = 0.319
        with pytest.raises(
            ValueError, match="Cannot provide a 'shots' value directly to the cut_circuit_mc "
        ):
            cut_circuit(v)

    def test_multiple_meas_error(self):
        """
        Tests that attempting to cut a circuit with multiple sample measurements
        using `cut_circuit_mc` gives the correct error
        """
        dev = qml.device("default.qubit", wires=3, shots=100)

        @qml.cut_circuit_mc
        @qml.qnode(dev)
        def cut_circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(wires=[0]), qml.sample(wires=[1]), qml.sample(wires=[2])

        v = 0.319
        with pytest.raises(
            ValueError, match="The Monte Carlo circuit cutting workflow only supports circuits "
        ):
            cut_circuit(v)

    def test_non_sample_meas_error(self):
        """
        Tests that attempting to cut a circuit with non-sample measurements
        using `cut_circuit_mc` gives the correct error
        """
        dev = qml.device("default.qubit", wires=2, shots=100)

        @qml.cut_circuit_mc
        @qml.qnode(dev)
        def cut_circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.expval(qml.PauliX(1))

        v = 0.319
        with pytest.raises(
            ValueError, match="The Monte Carlo circuit cutting workflow only supports circuits "
        ):
            cut_circuit(v)

    def test_qnode_shots_arg_error(self):
        """
        Tests that if a shots argument is passed directly to the qnode when using
        `cut_circuit_mc` the correct error is given
        """
        shots = 100
        dev = qml.device("default.qubit", wires=2, shots=shots)

        with pytest.raises(
            ValueError,
            match="Detected 'shots' as an argument of the quantum function to transform. ",
        ):

            @qml.cut_circuit_mc
            @qml.qnode(dev)
            def cut_circuit(x, shots=shots):
                qml.RX(x, wires=0)
                qml.RY(0.5, wires=1)
                qml.RX(1.3, wires=2)

                qml.CNOT(wires=[0, 1])
                qml.WireCut(wires=1)
                qml.CNOT(wires=[1, 2])

                qml.RX(x, wires=0)
                qml.RY(0.7, wires=1)
                qml.RX(2.3, wires=2)
                return qml.sample(wires=[0, 2])

    def test_no_interface(self):
        """
        Tests that if no interface is provided when using `cut_circuit_mc` the
        correct output is given
        """
        shots = 100
        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.cut_circuit_mc
        @qml.qnode(dev, interface=None)
        def cut_circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(wires=[0, 2])

        v = 0.319
        res = cut_circuit(v)
        assert res.shape == (shots, 2)
        assert type(res) == np.ndarray

    @pytest.mark.parametrize(
        "interface_import,interface",
        [
            ("autograd.numpy", "autograd"),
            ("tensorflow", "tensorflow"),
            ("torch", "torch"),
            ("jax.numpy", "jax"),
        ],
    )
    def test_all_interfaces_samples(self, interface_import, interface):
        """
        Tests that `cut_circuit_mc` returns the correct type of sample
        output value in all interfaces
        """
        lib = pytest.importorskip(interface_import)

        shots = 10
        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.cut_circuit_mc
        @qml.qnode(dev, interface=interface)
        def cut_circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(wires=[0, 2])

        v = 0.319
        convert_input = qml.math.convert_like(v, lib.ones(1))

        res = cut_circuit(convert_input)

        assert res.shape == (shots, 2)
        assert isinstance(res, type(convert_input))

    @pytest.mark.parametrize(
        "interface_import,interface",
        [
            ("autograd.numpy", "autograd"),
            ("tensorflow", "tensorflow"),
            ("torch", "torch"),
            ("jax.numpy", "jax"),
        ],
    )
    def test_all_interfaces_mc(self, interface_import, interface):
        """
        Tests that `cut_circuit_mc` returns the correct type of expectation
        value output in all interfaces
        """
        lib = pytest.importorskip(interface_import)

        shots = 10
        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.cut_circuit_mc(fn)
        @qml.qnode(dev, interface=interface)
        def cut_circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(wires=[0, 2])

        v = 0.319
        convert_input = qml.math.convert_like(v, lib.ones(1))
        res = cut_circuit(convert_input)

        assert isinstance(res, type(convert_input))

    def test_mc_with_mid_circuit_measurement(self, mocker):
        """Tests the full sample-based circuit cutting pipeline successfully returns a
        single value for a circuit that contains mid-circuit
        measurements and terminal sample measurements using the `cut_circuit_mc`
        transform."""

        shots = 10
        dev = qml.device("default.qubit", wires=3, shots=shots)

        @qml.cut_circuit_mc(fn)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.RX(np.sin(x) ** 2, wires=1)
            qml.CNOT(wires=[1, 2])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.sample(wires=[0, 1])

        spy = mocker.spy(qcut, "qcut_processing_fn_mc")
        x = np.array(0.531, requires_grad=True)
        res = circuit(x)

        spy.assert_called_once()
        assert res.size == 1

    def test_mc_circuit_with_disconnected_components(self, mocker):
        """Tests if a sample-based circuit that is fragmented into subcircuits such
        that some of the subcircuits are disconnected from the final terminal sample
        measurements is executed successfully"""
        shots = 10
        dev = qml.device("default.qubit", wires=3, shots=shots)

        @qml.cut_circuit_mc(fn)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RY(x**2, wires=2)
            return qml.sample(wires=[0, 1])

        x = 0.4
        res = circuit(x)
        assert res.size == 1

    def test_mc_circuit_with_trivial_wire_cut(self, mocker):
        """Tests that a sample-based circuit with a trivial wire cut (not
        separating the circuit into fragments) is executed successfully"""
        shots = 10
        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.cut_circuit_mc(fn)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(wires=[0, 1])

        spy = mocker.spy(qcut, "qcut_processing_fn_mc")

        x = 0.4
        res = circuit(x)
        assert res.size == 1
        assert len(spy.call_args[0][0]) == shots
        assert len(spy.call_args[0]) == 1

    def test_mc_complicated_circuit(self, mocker):
        """
        Tests that the full sample-based circuit cutting pipeline successfully returns a
        value for a complex circuit with multiple wire cut scenarios. The circuit is cut into
        fragments of at most 2 qubits and is drawn below:

        0: ──X──//─╭C───//──RX─╭C───//─╭C──╭//────────────────────┤
        1: ────────╰X──────────╰X───//─╰Z──╰//────────────────╭RX─┤ ╭Sample
        2: ──H─╭C──────────────────────╭RY────────────────╭RY─│───┤ ├Sample
        3: ────╰RY──//──H──╭C───//──H──╰C───//─╭RY──//──H─╰C──│───┤ ╰Sample
        4: ────────────────╰RY──H──────────────╰C─────────────╰C──┤
        """

        # We need a 4-qubit device to account for mid-circuit measurements
        shots = 10
        dev = qml.device("default.qubit", wires=4, shots=shots)

        def two_qubit_unitary(param, wires):
            qml.Hadamard(wires=[wires[0]])
            qml.CRY(param, wires=[wires[0], wires[1]])

        @qml.cut_circuit_mc(fn)
        @qml.qnode(dev)
        def circuit(params):
            qml.BasisState(np.array([1]), wires=[0])
            qml.WireCut(wires=0)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=0)
            qml.RX(params[0], wires=0)
            qml.CNOT(wires=[0, 1])

            qml.WireCut(wires=0)
            qml.WireCut(wires=1)

            qml.CZ(wires=[0, 1])
            qml.WireCut(wires=[0, 1])

            two_qubit_unitary(params[1], wires=[2, 3])
            qml.WireCut(wires=3)
            two_qubit_unitary(params[2] ** 2, wires=[3, 4])
            qml.WireCut(wires=3)
            two_qubit_unitary(np.sin(params[3]), wires=[3, 2])
            qml.WireCut(wires=3)
            two_qubit_unitary(np.sqrt(params[4]), wires=[4, 3])
            qml.WireCut(wires=3)
            two_qubit_unitary(np.cos(params[1]), wires=[3, 2])
            qml.CRX(params[2], wires=[4, 1])

            return qml.sample(wires=[1, 2, 3])

        spy = mocker.spy(qcut, "qcut_processing_fn_mc")

        params = np.array([0.4, 0.5, 0.6, 0.7, 0.8], requires_grad=True)
        res = circuit(params)

        spy.assert_called_once()
        assert res.size == 1


class TestContractTensors:
    """Tests for the contract_tensors function"""

    t = [np.arange(4), np.arange(4, 8)]
    # make copies of nodes to ensure id comparisons work correctly
    m = [[qcut.MeasureNode(wires=0)], []]
    p = [[], [qcut.PrepareNode(wires=0)]]
    edge_dict = {"pair": (copy.copy(m)[0][0], copy.copy(p)[1][0])}
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

    @pytest.mark.autograd
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

    @pytest.mark.torch
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

    @pytest.mark.tf
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

    @pytest.mark.jax
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
        # See Sec. II. A. of https://doi.org/10.1088/1367-2630/abd7bc, below Eq. (2).
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


@pytest.mark.parametrize("use_opt_einsum", [True, False])
class TestCutCircuitTransform:
    """
    Tests for the cut_circuit transform
    """

    @flaky(max_runs=3)
    @pytest.mark.parametrize("shots", [None, int(1e7)])
    def test_simple_cut_circuit(self, mocker, use_opt_einsum, shots):
        """
        Tests the full circuit cutting pipeline returns the correct value and
        gradient for a simple circuit using the `cut_circuit` transform.
        """

        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.543, wires=1)
            qml.WireCut(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires=1)
            return qml.expval(qml.PauliZ(wires=[0]))

        spy = mocker.spy(qcut, "qcut_processing_fn")
        x = np.array(0.531, requires_grad=True)
        cut_circuit = qcut.cut_circuit(circuit, use_opt_einsum=use_opt_einsum)

        atol = 1e-2 if shots else 1e-8
        assert np.isclose(cut_circuit(x), float(circuit(x)), atol=atol)
        spy.assert_called_once()

        gradient = qml.grad(circuit)(x)
        cut_gradient = qml.grad(cut_circuit)(x)

        assert np.isclose(gradient, cut_gradient, atol=atol)

    def test_simple_cut_circuit_torch(self, use_opt_einsum):
        """
        Tests the full circuit cutting pipeline returns the correct value and
        gradient for a simple circuit using the `cut_circuit` transform with the torch interface.
        """
        torch = pytest.importorskip("torch")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.543, wires=1)
            qml.WireCut(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires=1)
            return qml.expval(qml.PauliZ(wires=[0]))

        x = torch.tensor(0.531, requires_grad=True)
        cut_circuit = qcut.cut_circuit(circuit, use_opt_einsum=use_opt_einsum)

        res = cut_circuit(x)
        res_expected = circuit(x)
        assert np.isclose(res.detach().numpy(), res_expected.detach().numpy())

        res.backward()
        grad = x.grad.detach().numpy()

        x.grad = None
        res_expected.backward()
        grad_expected = x.grad.detach().numpy()

        assert np.isclose(grad, grad_expected)

    def test_simple_cut_circuit_tf(self, use_opt_einsum):
        """
        Tests the full circuit cutting pipeline returns the correct value and
        gradient for a simple circuit using the `cut_circuit` transform with the TF interface.
        """
        tf = pytest.importorskip("tensorflow")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="tf")
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.543, wires=1)
            qml.WireCut(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires=1)
            return qml.expval(qml.PauliZ(wires=[0]))

        x = tf.Variable(0.531)
        cut_circuit = qcut.cut_circuit(circuit, use_opt_einsum=use_opt_einsum)

        with tf.GradientTape() as tape:
            res = cut_circuit(x)

        grad = tape.gradient(res, x)

        with tf.GradientTape() as tape:
            res_expected = circuit(x)

        grad_expected = tape.gradient(res_expected, x)

        assert np.isclose(res, res_expected)
        assert np.isclose(grad, grad_expected)

    def test_simple_cut_circuit_jax(self, use_opt_einsum):
        """
        Tests the full circuit cutting pipeline returns the correct value and
        gradient for a simple circuit using the `cut_circuit` transform with the Jax interface.
        """
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.543, wires=1)
            qml.WireCut(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires=1)
            return qml.expval(qml.PauliZ(wires=[0]))

        x = jnp.array(0.531)
        cut_circuit = qcut.cut_circuit(circuit, use_opt_einsum=use_opt_einsum)

        res = cut_circuit(x)
        res_expected = circuit(x)

        grad = jax.grad(cut_circuit)(x)
        grad_expected = jax.grad(circuit)(x)

        assert np.isclose(res, res_expected)
        assert np.isclose(grad, grad_expected)

    def test_with_mid_circuit_measurement(self, mocker, use_opt_einsum):
        """Tests the full circuit cutting pipeline returns the correct value and gradient for a
        circuit that contains mid-circuit measurements, using the `cut_circuit` transform."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.RX(np.sin(x) ** 2, wires=1)
            qml.CNOT(wires=[1, 2])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        spy = mocker.spy(qcut, "qcut_processing_fn")
        x = np.array(0.531, requires_grad=True)
        cut_circuit = qcut.cut_circuit(circuit, use_opt_einsum=use_opt_einsum)

        assert np.isclose(cut_circuit(x), float(circuit(x)))
        spy.assert_called_once()

        gradient = qml.grad(circuit)(x)
        cut_gradient = qml.grad(cut_circuit)(x)

        assert np.isclose(gradient, cut_gradient)

    def test_simple_cut_circuit_torch_trace(self, mocker, use_opt_einsum):
        """
        Tests the full circuit cutting pipeline returns the correct value and
        gradient for a simple circuit using the `cut_circuit` transform with the torch interface and
        using torch tracing.
        """
        torch = pytest.importorskip("torch")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.543, wires=1)
            qml.WireCut(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires=1)
            return qml.expval(qml.PauliZ(wires=[0]))

        x = torch.tensor(0.531, requires_grad=True, dtype=torch.complex128)

        # Note that the jit.trace ends up calling qcut_processing_fn multiple times, so below we
        # delay introducing the spy until afterwards and then ensure that qcut_processing_fn is
        # not called again.
        cut_circuit_trace = torch.jit.trace(
            qcut.cut_circuit(circuit, use_opt_einsum=use_opt_einsum), x
        )

        # Run once with original value
        spy = mocker.spy(qcut, "qcut_processing_fn")
        res = cut_circuit_trace(x)

        spy.assert_not_called()

        res_expected = circuit(x)
        assert np.isclose(res.detach().numpy(), res_expected.detach().numpy())

        res.backward()
        grad = x.grad.detach().numpy()

        x.grad = None
        res_expected.backward()
        grad_expected = x.grad.detach().numpy()

        assert np.isclose(grad, grad_expected)

        # Run more times over a range of values
        for x in np.linspace(-1, 1, 10):
            x = torch.tensor(x, requires_grad=True)
            res = cut_circuit_trace(x)

            res_expected = circuit(x)
            assert np.isclose(res.detach().numpy(), res_expected.detach().numpy())

            res.backward()
            grad = x.grad.detach().numpy()

            x.grad = None
            res_expected.backward()
            grad_expected = x.grad.detach().numpy()

            assert np.isclose(grad, grad_expected)

        spy.assert_not_called()

    def test_simple_cut_circuit_tf_jit(self, mocker, use_opt_einsum):
        """
        Tests the full circuit cutting pipeline returns the correct value and
        gradient for a simple circuit using the `cut_circuit` transform with the TF interface and
        using JIT.
        """
        tf = pytest.importorskip("tensorflow")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="tf")
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.543, wires=1)
            qml.WireCut(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires=1)
            return qml.expval(qml.PauliZ(wires=[0]))

        x = tf.Variable(0.531)
        cut_circuit_jit = tf.function(
            qcut.cut_circuit(circuit, use_opt_einsum=use_opt_einsum),
            jit_compile=True,
            input_signature=(tf.TensorSpec(shape=None, dtype=tf.float32),),
        )

        # Run once with original value
        spy = mocker.spy(qcut, "qcut_processing_fn")

        # Note we call the function twice but assert qcut_processing_fn is called once. We expect
        # qcut_processing_fn to be called once during JIT compilation, with subsequent calls to
        # cut_circuit_jit using the compiled code.
        cut_circuit_jit(x)

        with tf.GradientTape() as tape:
            res = cut_circuit_jit(x)

        grad = tape.gradient(res, x)

        spy.assert_called_once()

        with tf.GradientTape() as tape:
            res_expected = circuit(x)

        grad_expected = tape.gradient(res_expected, x)

        assert np.isclose(res, res_expected)
        assert np.isclose(grad, grad_expected)

        # Run more times over a range of values
        for x in np.linspace(-1, 1, 10):
            x = tf.Variable(x, dtype=tf.float32)

            cut_circuit_jit(x)

            with tf.GradientTape() as tape:
                res = cut_circuit_jit(x)

            grad = tape.gradient(res, x)

            with tf.GradientTape() as tape:
                res_expected = circuit(x)

            grad_expected = tape.gradient(res_expected, x)

            assert np.isclose(res, res_expected)
            assert np.isclose(grad, grad_expected)

        spy.assert_called_once()

    def test_simple_cut_circuit_jax_jit(self, mocker, use_opt_einsum):
        """
        Tests the full circuit cutting pipeline returns the correct value and
        gradient for a simple circuit using the `cut_circuit` transform with the Jax interface and
        using JIT.
        """
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.543, wires=1)
            qml.WireCut(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires=1)
            return qml.expval(qml.PauliZ(wires=[0]))

        x = jnp.array(0.531)
        cut_circuit_jit = jax.jit(qcut.cut_circuit(circuit, use_opt_einsum=use_opt_einsum))

        # Run once with original value
        spy = mocker.spy(qcut, "qcut_processing_fn")

        # Note we call the function twice but assert qcut_processing_fn is called once. We expect
        # qcut_processing_fn to be called once during JIT compilation, with subsequent calls to
        # cut_circuit_jit using the compiled code.
        cut_circuit_jit(x)
        res = cut_circuit_jit(x)
        res_expected = circuit(x)

        spy.assert_called_once()
        assert np.isclose(res, res_expected)

        grad = jax.grad(cut_circuit_jit)(x)
        grad_expected = jax.grad(circuit)(x)

        assert np.isclose(grad, grad_expected)
        assert spy.call_count == 2

        # Run more times over a range of values
        for x in np.linspace(-1, 1, 10):
            x = jnp.array(x)

            cut_circuit_jit(x)
            res = cut_circuit_jit(x)
            res_expected = circuit(x)

            assert np.isclose(res, res_expected)

            grad = jax.grad(cut_circuit_jit)(x)
            grad_expected = jax.grad(circuit)(x)

            assert np.isclose(grad, grad_expected)

        assert spy.call_count == 4

    def test_device_wires(self, use_opt_einsum):
        """Tests that a 3-qubit circuit is cut into two 2-qubit fragments such that both fragments
        can be run on a 2-qubit device"""

        def circuit():
            qml.RX(0.4, wires=0)
            qml.RX(0.5, wires=1)
            qml.RX(0.6, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            return qml.expval(qml.PauliX(1) @ qml.PauliY(2))

        dev_uncut = qml.device("default.qubit", wires=3)
        dev_1 = qml.device("default.qubit", wires=2)
        dev_2 = qml.device("default.qubit", wires=["Alice", 3.14, "Bob"])

        uncut_circuit = qml.QNode(circuit, dev_uncut)
        cut_circuit_1 = qml.transforms.cut_circuit(
            qml.QNode(circuit, dev_1), use_opt_einsum=use_opt_einsum
        )
        cut_circuit_2 = qml.transforms.cut_circuit(
            qml.QNode(circuit, dev_2), use_opt_einsum=use_opt_einsum
        )

        res_expected = uncut_circuit()
        res_1 = cut_circuit_1()
        res_2 = cut_circuit_2()

        assert np.isclose(res_expected, res_1)
        assert np.isclose(res_expected, res_2)

    def test_circuit_with_disconnected_components(self, use_opt_einsum, mocker):
        """Tests if a circuit that is fragmented into subcircuits such that some of the subcircuits
        are disconnected from the final terminal measurements is executed correctly"""
        dev = qml.device("default.qubit", wires=3)

        @qml.transforms.cut_circuit(use_opt_einsum=use_opt_einsum)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RY(x**2, wires=2)
            return qml.expval(qml.PauliZ(wires=[0]))

        x = 0.4
        res = circuit(x)
        assert np.allclose(res, np.cos(x))

    def test_circuit_with_trivial_wire_cut(self, use_opt_einsum, mocker):
        """Tests that a circuit with a trivial wire cut (not separating the circuit into
        fragments) is executed correctly"""
        dev = qml.device("default.qubit", wires=2)

        @qml.transforms.cut_circuit(use_opt_einsum=use_opt_einsum)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=[0]))

        spy = mocker.spy(qcut, "contract_tensors")

        x = 0.4
        res = circuit(x)
        assert np.allclose(res, np.cos(x))
        assert len(spy.call_args[0][0]) == 1  # there should be 1 tensor for wire 0
        assert spy.call_args[0][0][0].shape == ()

    def test_complicated_circuit(self, mocker, use_opt_einsum):
        """
        Tests that the full circuit cutting pipeline returns the correct value and
        gradient for a complex circuit with multiple wire cut scenarios. The circuit is cut into
        fragments of at most 2 qubits and is drawn below:

        0: ──BasisState(M0)──//───────╭C───//──RX(0.40)─╭C───//─╭C────────╭//──────────────────
        1: ───────────────────────────╰X────────────────╰X───//─╰Z────────╰//──────────────────
        2: ──H──────────────╭C──────────────────────────────────╭RY(0.64)──────────────────────
        3: ─────────────────╰RY(0.50)──//──H──╭C─────────//──H──╰C─────────//─╭RY(0.89)──//──H─
        4: ───────────────────────────────────╰RY(0.36)──H────────────────────╰C───────────────

        ──────────────────────────┤
        ────────────────╭RX(0.60)─┤ ╭<Z@Z@Z>
        ╭RY(0.88)───────│─────────┤ ├<Z@Z@Z>
        ╰C──────────────│─────────┤ ╰<Z@Z@Z>
        ────────────────╰C────────┤
        """
        dev_original = qml.device("default.qubit", wires=5)

        # We need a 4-qubit device to account for mid-circuit measurements
        dev_cut = qml.device("default.qubit", wires=4)

        def two_qubit_unitary(param, wires):
            qml.Hadamard(wires=[wires[0]])
            qml.CRY(param, wires=[wires[0], wires[1]])

        def f(params):
            qml.BasisState(np.array([1]), wires=[0])
            qml.WireCut(wires=0)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=0)
            qml.RX(params[0], wires=0)
            qml.CNOT(wires=[0, 1])

            qml.WireCut(wires=0)
            qml.WireCut(wires=1)

            qml.CZ(wires=[0, 1])
            qml.WireCut(wires=[0, 1])

            two_qubit_unitary(params[1], wires=[2, 3])
            qml.WireCut(wires=3)
            two_qubit_unitary(params[2] ** 2, wires=[3, 4])
            qml.WireCut(wires=3)
            two_qubit_unitary(np.sin(params[3]), wires=[3, 2])
            qml.WireCut(wires=3)
            two_qubit_unitary(np.sqrt(params[4]), wires=[4, 3])
            qml.WireCut(wires=3)
            two_qubit_unitary(np.cos(params[1]), wires=[3, 2])
            qml.CRX(params[2], wires=[4, 1])

            return qml.expval(qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))

        params = np.array([0.4, 0.5, 0.6, 0.7, 0.8], requires_grad=True)

        circuit = qml.QNode(f, dev_original)
        cut_circuit = qcut.cut_circuit(qml.QNode(f, dev_cut), use_opt_einsum=use_opt_einsum)

        res_expected = circuit(params)
        grad_expected = qml.grad(circuit)(params)

        spy = mocker.spy(qcut, "qcut_processing_fn")
        res = cut_circuit(params)
        spy.assert_called_once()
        grad = qml.grad(cut_circuit)(params)

        assert np.isclose(res, res_expected)
        assert np.allclose(grad, grad_expected)

    @flaky(max_runs=3)
    @pytest.mark.parametrize("shots", [None, int(1e7)])
    def test_standard_circuit(self, mocker, use_opt_einsum, shots):
        """
        Tests that the full circuit cutting pipeline returns the correct value for a typical
        scenario. The circuit is drawn below:

        0: ─╭U(M1)───────────────────╭U(M4)─┤ ╭<Z@X>
        1: ─╰U(M1)──//─╭U(M2)──//────╰U(M4)─┤ │
        2: ─╭U(M0)─────╰U(M2)─╭U(M3)────────┤ │
        3: ─╰U(M0)────────────╰U(M3)────────┤ ╰<Z@X>
        """
        dev_original = qml.device("default.qubit", wires=4)

        # We need a 3-qubit device
        dev_cut = qml.device("default.qubit", wires=3, shots=shots)
        us = [unitary_group.rvs(2**2, random_state=i) for i in range(5)]

        def f():
            qml.QubitUnitary(us[0], wires=[0, 1])
            qml.QubitUnitary(us[1], wires=[2, 3])

            qml.WireCut(wires=[1])

            qml.QubitUnitary(us[2], wires=[1, 2])

            qml.WireCut(wires=[1])

            qml.QubitUnitary(us[3], wires=[0, 1])
            qml.QubitUnitary(us[4], wires=[2, 3])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(3))

        circuit = qml.QNode(f, dev_original)
        cut_circuit = qcut.cut_circuit(qml.QNode(f, dev_cut), use_opt_einsum=use_opt_einsum)

        res_expected = circuit()

        spy = mocker.spy(qcut, "qcut_processing_fn")
        res = cut_circuit()
        spy.assert_called_once()

        atol = 1e-2 if shots else 1e-8
        assert np.isclose(res, res_expected, atol=atol)


class TestRemapTapeWires:
    """Tests for the remap_tape_wires function"""

    def test_raises(self):
        """Test if a ValueError is raised when too few wires are provided"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.5, wires=2)
            qml.RY(0.6, wires=3)
            qml.CNOT(wires=[2, 3])
            qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))

        with pytest.raises(ValueError, match="a 2-wire circuit on a 1-wire device"):
            qcut.remap_tape_wires(tape, [0])

    def test_mapping(self):
        """Test if the function returns the expected tape when an observable measurement is
        used"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.5, wires=2)
            qml.RY(0.6, wires=3)
            qml.CNOT(wires=[2, 3])
            qml.expval(qml.PauliZ(2))

        with qml.tape.QuantumTape() as expected_tape:
            qml.RX(0.5, wires=0)
            qml.RY(0.6, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        new_tape = qcut.remap_tape_wires(tape, [0, 1])

        compare_tapes(expected_tape, new_tape)

    def test_mapping_tensor(self):
        """Test if the function returns the expected tape when a tensor product measurement is
        used"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.5, wires=2)
            qml.RY(0.6, wires=3)
            qml.CNOT(wires=[2, 3])
            qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))

        with qml.tape.QuantumTape() as expected_tape:
            qml.RX(0.5, wires=0)
            qml.RY(0.6, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        new_tape = qcut.remap_tape_wires(tape, [0, 1])

        compare_tapes(expected_tape, new_tape)


class TestCutCircuitTransformValidation:
    """Tests of validation checks in the cut_circuit function"""

    def test_multiple_measurements_raises(self):
        """Tests if a ValueError is raised when a tape with multiple measurements is requested
        to be cut"""

        with qml.tape.QuantumTape() as tape:
            qml.WireCut(wires=0)
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliZ(1))

        with pytest.raises(ValueError, match="The circuit cutting workflow only supports circuits"):
            qcut.cut_circuit(tape)

    def test_no_measurements_raises(self):
        """Tests if a ValueError is raised when a tape with no measurement is requested
        to be cut"""
        with qml.tape.QuantumTape() as tape:
            qml.WireCut(wires=0)

        with pytest.raises(ValueError, match="The circuit cutting workflow only supports circuits"):
            qcut.cut_circuit(tape)

    def test_non_expectation_raises(self):
        """Tests if a ValueError is raised when a tape with measurements that are not expectation
        values is requested to be cut"""

        with qml.tape.QuantumTape() as tape:
            qml.WireCut(wires=0)
            qml.var(qml.PauliZ(0))

        with pytest.raises(ValueError, match="workflow only supports circuits with expectation"):
            qcut.cut_circuit(tape)

    def test_fail_import(self, monkeypatch):
        """Test if an ImportError is raised when opt_einsum is requested but not installed"""
        with qml.tape.QuantumTape() as tape:
            qml.WireCut(wires=0)
            qml.expval(qml.PauliZ(0))

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "opt_einsum", None)

            with pytest.raises(ImportError, match="The opt_einsum package is required"):
                qcut.cut_circuit(tape, use_opt_einsum=True)

    def test_no_cuts_raises(self):
        """Tests if a ValueError is raised when circuit cutting is to be applied to a circuit
        without cuts"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="No WireCut operations found in the circuit."):
            qcut.cut_circuit(tape)


class TestCutCircuitExpansion:
    """Test of expansion in the cut_circuit and cut_circuit_mc functions"""

    transform_measurement_pairs = [
        (qcut.cut_circuit, qml.expval(qml.PauliZ(0))),
        (qcut.cut_circuit_mc, qml.sample(wires=[0])),
    ]

    @pytest.mark.parametrize("cut_transform, measurement", transform_measurement_pairs)
    def test_no_expansion(self, mocker, cut_transform, measurement):
        """Test if no/trivial expansion occurs if WireCut operations are already present in the
        tape"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.3, wires=0)
            qml.WireCut(wires=0)
            qml.RY(0.4, wires=0)
            qml.apply(measurement)

        spy = mocker.spy(qcut, "_qcut_expand_fn")

        kwargs = {"shots": 10} if measurement.return_type is qml.measurements.Sample else {}
        cut_transform(tape, device_wires=[0], **kwargs)

        spy.assert_called_once()

    @pytest.mark.parametrize("cut_transform, measurement", transform_measurement_pairs)
    def test_expansion(self, mocker, cut_transform, measurement):
        """Test if expansion occurs if WireCut operations are present in a nested tape"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.3, wires=0)
            with qml.tape.QuantumTape() as _:
                qml.WireCut(wires=0)
            qml.RY(0.4, wires=0)
            qml.apply(measurement)

        spy = mocker.spy(qcut, "_qcut_expand_fn")

        kwargs = {"shots": 10} if measurement.return_type is qml.measurements.Sample else {}
        cut_transform(tape, device_wires=[0], **kwargs)

        assert spy.call_count == 2

    @pytest.mark.parametrize("cut_transform, measurement", transform_measurement_pairs)
    def test_expansion_error(self, cut_transform, measurement):
        """Test if a ValueError is raised if expansion continues beyond the maximum depth"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.3, wires=0)
            with qml.tape.QuantumTape() as _:
                with qml.tape.QuantumTape() as __:
                    qml.WireCut(wires=0)
            qml.RY(0.4, wires=0)
            qml.apply(measurement)

        with pytest.raises(ValueError, match="No WireCut operations found in the circuit."):
            kwargs = {"shots": 10} if measurement.return_type is qml.measurements.Sample else {}
            cut_transform(tape, device_wires=[0], **kwargs)

    def test_expansion_ttn(self, mocker):
        """Test if wire cutting is compatible with the tree tensor network operation"""

        def block(weights, wires):
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RY(weights[0], wires=wires[0])
            qml.RY(weights[1], wires=wires[1])
            qml.WireCut(wires=wires[1])

        n_wires = 4
        n_block_wires = 2
        n_params_block = 2
        n_blocks = qml.TTN.get_n_blocks(range(n_wires), n_block_wires)
        template_weights = [[0.1, -0.3]] * n_blocks

        dev_cut = qml.device("default.qubit", wires=2)
        dev_big = qml.device("default.qubit", wires=4)

        def circuit(template_weights):
            qml.TTN(range(n_wires), n_block_wires, block, n_params_block, template_weights)
            return qml.expval(qml.PauliZ(wires=n_wires - 1))

        qnode = qml.QNode(circuit, dev_big)
        qnode_cut = qcut.cut_circuit(qml.QNode(circuit, dev_cut))

        spy = mocker.spy(qcut, "_qcut_expand_fn")
        res = qnode_cut(template_weights)
        assert spy.call_count == 2

        assert np.isclose(res, qnode(template_weights))

    def test_expansion_mc_ttn(self, mocker):
        """Test if wire cutting is compatible with the tree tensor network operation"""

        def block(weights, wires):
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RY(weights[0], wires=wires[0])
            qml.RY(weights[1], wires=wires[1])
            qml.WireCut(wires=wires[1])

        n_wires = 4
        n_block_wires = 2
        n_params_block = 2
        n_blocks = qml.TTN.get_n_blocks(range(n_wires), n_block_wires)
        template_weights = [[0.1, -0.3]] * n_blocks

        dev_cut = qml.device("default.qubit", wires=2, shots=10)

        def circuit(template_weights):
            qml.TTN(range(n_wires), n_block_wires, block, n_params_block, template_weights)
            return qml.sample(wires=[n_wires - 1])

        qnode_cut = qcut.cut_circuit_mc(qml.QNode(circuit, dev_cut))

        spy = mocker.spy(qcut, "_qcut_expand_fn")
        qnode_cut(template_weights)
        assert spy.call_count == 2


class TestCutStrategy:
    """Tests for class CutStrategy"""

    devs = [qml.device("default.qubit", wires=n) for n in [4, 6]]
    tape_dags = [qcut.tape_to_graph(t) for t in [tape, multi_cut_tape]]

    @pytest.mark.parametrize("devices", [None, 1, devs[0]])
    @pytest.mark.parametrize("imbalance_tolerance", [None, -1])
    @pytest.mark.parametrize("num_fragments_probed", [None, 0])
    def test_init_raises(self, devices, imbalance_tolerance, num_fragments_probed):
        """Test if ill-initialized instances throw errors."""

        if (
            isinstance(devices, qml.Device)
            and imbalance_tolerance is None
            and num_fragments_probed is None
        ):
            return  # skip the only valid combination

        with pytest.raises(ValueError):
            qcut.CutStrategy(
                devices=devices,
                num_fragments_probed=num_fragments_probed,
                imbalance_tolerance=imbalance_tolerance,
            )

    @pytest.mark.parametrize("devices", [devs[0], devs])
    @pytest.mark.parametrize("max_free_wires", [None, 3])
    @pytest.mark.parametrize("min_free_wires", [None, 2])
    @pytest.mark.parametrize("num_fragments_probed", [None, 2, (2, 4)])
    @pytest.mark.parametrize("imbalance_tolerance", [None, 0, 0.1])
    def test_init(
        self, devices, max_free_wires, min_free_wires, num_fragments_probed, imbalance_tolerance
    ):
        """Test the __post_init__ properly sets defaults based on provided info."""

        strategy = qcut.CutStrategy(
            devices=devices,
            max_free_wires=max_free_wires,
            num_fragments_probed=num_fragments_probed,
            imbalance_tolerance=imbalance_tolerance,
        )

        devices = [devices] if not isinstance(devices, list) else devices

        max_dev_wires = max((len(d.wires) for d in devices))
        assert strategy.max_free_wires == max_free_wires or max_dev_wires or min_free_wires
        assert strategy.min_free_wires == min_free_wires or max_free_wires or max_dev_wires
        assert strategy.imbalance_tolerance == imbalance_tolerance

        if num_fragments_probed is not None:
            assert (
                strategy.k_lower == num_fragments_probed
                if isinstance(num_fragments_probed, int)
                else min(num_fragments_probed)
            )
            assert (
                strategy.k_upper == num_fragments_probed
                if isinstance(num_fragments_probed, int)
                else max(num_fragments_probed)
            )
        else:
            assert strategy.k_lower is None
            assert strategy.k_upper is None

    @pytest.mark.parametrize("k", [4, 5, 6])
    @pytest.mark.parametrize("imbalance_tolerance", [None, 0, 0.1])
    def test_infer_wire_imbalance(self, k, imbalance_tolerance):
        """Test that the imbalance is correctly derived under simple circumstances."""

        wire_depths = dict(enumerate(range(10)))
        num_gates = int(sum(wire_depths.values()))
        free_wires = 3

        imbalance = qcut.CutStrategy._infer_imbalance(
            k=k,
            wire_depths=wire_depths,
            free_wires=free_wires,
            free_gates=1000,
            imbalance_tolerance=imbalance_tolerance,
        )

        depth_imbalance = max(wire_depths.values()) * len(wire_depths) / num_gates - 1
        if imbalance_tolerance is not None:
            assert imbalance <= imbalance_tolerance
        else:
            assert imbalance == depth_imbalance

    @pytest.mark.parametrize("num_wires", [50, 10])
    def test_infer_wire_imbalance_raises(
        self,
        num_wires,
    ):
        """Test that the imbalance correctly raises."""

        k = 2
        wire_depths = {k: 1 if num_wires > 40 else 5 for k in range(num_wires)}

        with pytest.raises(ValueError, match=f"`free_{'wires' if num_wires > 40 else 'gates'}`"):
            qcut.CutStrategy._infer_imbalance(
                k=k,
                wire_depths=wire_depths,
                free_wires=20,
                free_gates=20,
            )

    @pytest.mark.parametrize("devices", [devs[0], devs])
    @pytest.mark.parametrize("num_fragments_probed", [None, 4, (4, 6)])
    @pytest.mark.parametrize("imbalance_tolerance", [None, 0, 0.1])
    @pytest.mark.parametrize("tape_dag", tape_dags)
    @pytest.mark.parametrize("exhaustive", [True, False])
    def test_get_cut_kwargs(
        self, devices, num_fragments_probed, imbalance_tolerance, tape_dag, exhaustive
    ):
        """Test that the cut kwargs can be derived."""

        strategy = qcut.CutStrategy(
            devices=devices,
            num_fragments_probed=num_fragments_probed,
            imbalance_tolerance=imbalance_tolerance,
        )

        all_cut_kwargs = strategy.get_cut_kwargs(tape_dag=tape_dag, exhaustive=exhaustive)

        assert all_cut_kwargs
        assert all("imbalance" in kwargs and "num_fragments" in kwargs for kwargs in all_cut_kwargs)
        if imbalance_tolerance is not None:
            assert all([kwargs["imbalance"] <= imbalance_tolerance for kwargs in all_cut_kwargs])
        if num_fragments_probed is not None:
            assert {v["num_fragments"] for v in all_cut_kwargs} == (
                set(range(num_fragments_probed[0], num_fragments_probed[-1] + 1))
                if isinstance(num_fragments_probed, (list, tuple))
                else {num_fragments_probed}
            )
        elif exhaustive:
            num_tape_gates = sum(not isinstance(n, qcut.WireCut) for n in tape_dag.nodes)
            assert {v["num_fragments"] for v in all_cut_kwargs} == set(range(2, num_tape_gates + 1))

    @pytest.mark.parametrize(
        "num_fragments_probed", [1, qcut.CutStrategy.HIGH_NUM_FRAGMENTS + 1, (2, 100)]
    )
    def test_get_cut_kwargs_warnings(self, num_fragments_probed):
        """Test the 3 situations where the get_cut_kwargs pops out a warning."""
        strategy = qcut.CutStrategy(
            max_free_wires=2,
            num_fragments_probed=num_fragments_probed,
        )
        k = num_fragments_probed
        k_lower = k if isinstance(k, int) else k[0]
        assert strategy.k_lower == k_lower

        with pytest.warns(UserWarning):
            _ = strategy.get_cut_kwargs(self.tape_dags[1])

    @pytest.mark.parametrize("max_wires_by_fragment", [None, [2, 3]])
    @pytest.mark.parametrize("max_gates_by_fragment", [[20, 30], [20, 30, 40]])
    def test_by_fragment_sizes(self, max_wires_by_fragment, max_gates_by_fragment):
        """Test that the user provided by-fragment limits properly propagates."""
        strategy = qcut.CutStrategy(
            max_free_wires=2,
        )
        if (
            max_wires_by_fragment
            and max_gates_by_fragment
            and len(max_wires_by_fragment) != len(max_gates_by_fragment)
        ):
            with pytest.raises(ValueError):
                cut_kwargs = strategy.get_cut_kwargs(
                    self.tape_dags[1],
                    max_wires_by_fragment=max_wires_by_fragment,
                    max_gates_by_fragment=max_gates_by_fragment,
                )
            return

        cut_kwargs = strategy.get_cut_kwargs(
            self.tape_dags[1],
            max_wires_by_fragment=max_wires_by_fragment,
            max_gates_by_fragment=max_gates_by_fragment,
        )
        assert len(cut_kwargs) == 1

        cut_kwargs = cut_kwargs[0]
        assert cut_kwargs["num_fragments"] == len(max_wires_by_fragment or max_gates_by_fragment)

    @pytest.mark.parametrize("max_wires_by_fragment", [2, ["a", 3], [2, 3], None])
    @pytest.mark.parametrize("max_gates_by_fragment", [2, ["b", 30]])
    def test_validate_fragment_sizes(self, max_wires_by_fragment, max_gates_by_fragment):
        """Test that the user provided by-fragment limits has the right types."""
        with pytest.raises(ValueError):
            _ = qcut.CutStrategy._validate_input(
                max_wires_by_fragment=max_wires_by_fragment,
                max_gates_by_fragment=max_gates_by_fragment,
            )


def make_weakly_connected_tape(
    fragment_wire_sizes=[3, 5],
    single_gates_per_wire=1,
    double_gates_multiplier=1.5,
    inter_fragment_gate_wires={(0, 1): 1},
    repeats=2,
    seed=None,
):
    """Helper function for making random weakly connected tapes."""
    rng = np.random.default_rng(seed)
    inter_fragment_gate_wires = inter_fragment_gate_wires or {}
    with qml.tape.QuantumTape() as tape:
        for _ in range(repeats):
            for i, wire_size in enumerate(fragment_wire_sizes):
                double_gates_per_fragment = int(double_gates_multiplier * wire_size)
                for _ in range(double_gates_per_fragment):
                    j0, j1 = rng.choice(wire_size, size=2, replace=False)
                    qml.CNOT(wires=[f"{i}-{j0}", f"{i}-{j1}"])
                for _ in range(single_gates_per_wire):
                    for j in range(wire_size):
                        qml.RZ(0.5, wires=f"{i}-{j}")
            for frag_pair, num_gates in inter_fragment_gate_wires.items():
                f0, f1 = sorted(frag_pair)
                for _ in range(num_gates):
                    w0 = rng.choice(fragment_wire_sizes[f0])
                    w1 = rng.choice(fragment_wire_sizes[f1])
                    qml.CNOT(wires=[f"{f0}-{w0}", f"{f1}-{w1}"])
        for i, wire_size in enumerate(fragment_wire_sizes):
            if wire_size == 1:
                qml.expval(qml.PauliZ(wires=[f"{i}-0"]))
            else:
                qml.expval(qml.PauliZ(wires=[f"{i}-0"]) @ qml.PauliZ(wires=[f"{i}-{wire_size-1}"]))
    return tape


class TestKaHyPar:
    """Tests for the KaHyPar cutting function and utilities."""

    # Fixes seed for Github actions:
    seed = 11 if environ.get("CI") == "true" else None

    disjoint_tapes = [
        (
            2,
            0,
            make_weakly_connected_tape(
                single_gates_per_wire=2, inter_fragment_gate_wires=None, seed=seed
            ),
        ),
        (
            5,
            0,
            make_weakly_connected_tape(
                fragment_wire_sizes=[2] * 5, inter_fragment_gate_wires=None, seed=seed
            ),
        ),
    ]
    fragment_tapes = [
        (2, 2, make_weakly_connected_tape(inter_fragment_gate_wires={(0, 1): 1}, seed=seed)),
        (
            3,
            6,
            make_weakly_connected_tape(
                fragment_wire_sizes=[4, 5, 6],
                single_gates_per_wire=2,
                double_gates_multiplier=2,
                inter_fragment_gate_wires={(0, 1): 1, (1, 2): 1},
                seed=seed,
            ),
        ),
    ]
    config_path = str(
        Path(__file__).parent.parent.parent / "pennylane/transforms/_cut_kKaHyPar_sea20.ini"
    )

    def test_seed_in_ci(self):
        """Test if seed is properly set in github action CI"""
        if environ.get("CI") == "true":
            print(f"CI seed set to {self.seed}")
            assert self.seed == 11

    def test_import_raise(self, monkeypatch):
        """Test if import exception is properly raised for missing kahypar."""
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "kahypar", None)

            with pytest.raises(ImportError, match="KaHyPar must be installed"):
                qcut.kahypar_cut(None, 2)

    @pytest.mark.parametrize("tape", disjoint_tapes + fragment_tapes)
    @pytest.mark.parametrize("hyperwire_weight", [0, 2])
    @pytest.mark.parametrize("edge_weights", [None, 3])
    def test_graph_to_hmetis(self, tape, hyperwire_weight, edge_weights):
        """Test conversion to the hMETIS format."""

        num_frags, num_interfrag_gates, tape = tape
        graph = qcut.tape_to_graph(tape)
        edge_weights = [edge_weights] * len(graph.edges) if edge_weights is not None else None
        adj_nodes, edge_splits, all_edge_weights = qcut._graph_to_hmetis(
            graph, hyperwire_weight=hyperwire_weight, edge_weights=edge_weights
        )
        assert len(edge_splits) - 1 == len(graph.edges) + (hyperwire_weight > 0) * len(tape.wires)
        assert (all_edge_weights is not None) == (hyperwire_weight > 0) or (
            edge_weights is not None
        )
        assert max(adj_nodes) + 1 == len(graph.nodes)

    @pytest.mark.parametrize("tape", disjoint_tapes + fragment_tapes)
    @pytest.mark.parametrize("hyperwire_weight", [0, 1])
    def test_kahypar_cut(self, tape, hyperwire_weight):
        """Test vanilla cutting with kahypar"""
        pytest.importorskip("kahypar")

        num_frags, num_interfrag_gates, tape = tape
        graph = qcut.tape_to_graph(tape)

        cut_edges = qcut.kahypar_cut(
            graph=graph,
            num_fragments=num_frags,
            imbalance=0.5,
            hyperwire_weight=hyperwire_weight,
            seed=self.seed,
        )

        assert len(cut_edges) <= num_interfrag_gates * 2

        cut_graph = qcut.place_wire_cuts(graph=graph, cut_edges=cut_edges)
        qcut.replace_wire_cut_nodes(cut_graph)
        frags, comm_graph = qcut.fragment_graph(cut_graph)

        assert len(frags) == num_frags
        assert len(comm_graph.edges) == len(cut_edges)

    @pytest.mark.parametrize("config_path", [None, config_path])
    @pytest.mark.parametrize("fragment_weights", [None, [350, 210]])
    @pytest.mark.parametrize("imbalance", [None, 0.5])
    def test_kahypar_cut_options(self, imbalance, fragment_weights, config_path):
        """Test vanilla cutting with kahypar"""
        pytest.importorskip("kahypar")

        num_frags, num_interfrag_gates, tape = self.disjoint_tapes[0]
        graph = qcut.tape_to_graph(tape)

        cut_edges = qcut.kahypar_cut(
            graph=graph,
            num_fragments=num_frags,
            imbalance=imbalance,
            fragment_weights=fragment_weights,
            config_path=config_path,
            verbose=True,
            seed=self.seed,
        )

        assert len(cut_edges) <= num_interfrag_gates * 2

        cut_graph = qcut.place_wire_cuts(graph=graph, cut_edges=cut_edges)
        qcut.replace_wire_cut_nodes(cut_graph)
        frags, comm_graph = qcut.fragment_graph(cut_graph)

        assert len(frags) == num_frags
        assert len(comm_graph.edges) == len(cut_edges)

    @pytest.mark.parametrize("dangling_measure", [False, True])
    def test_remove_existing_cuts(self, dangling_measure):
        """Test if ``WireCut`` and ``MeasureNode``/``PrepareNode`` are correctly removed."""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires="a")
            qml.WireCut(wires="a")
            qml.RY(0.543, wires="a")
            qcut.MeasureNode(wires="a")
            qcut.PrepareNode(wires="a")
            if dangling_measure:
                qcut.MeasureNode(wires="a")
            qml.RX(0.678, wires=0)
            qml.expval(qml.PauliZ(wires=[0]))

        graph = qcut.tape_to_graph(tape)
        if dangling_measure:
            with pytest.warns(
                UserWarning,
                match="The circuit contains `MeasureNode` or `PrepareNode` operations",
            ):
                graph = qcut._remove_existing_cuts(graph)
        else:
            graph = qcut._remove_existing_cuts(graph)

        # Only dangling `MeasureNode` should be left:
        assert (
            len(
                [
                    n
                    for n in graph.nodes
                    if isinstance(n, (qml.WireCut, qcut.MeasureNode, qcut.PrepareNode))
                ]
            )
            == dangling_measure
        )

    def test_place_wire_cuts(self):
        """Test if ``WireCut`` are correctly placed with the correct order."""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires="a")
            qml.RY(0.543, wires="a")
            qml.CNOT(wires=[0, "a"])
            qml.RX(0.678, wires=0)
            qml.expval(qml.PauliZ(wires=[0]))

        graph = qcut.tape_to_graph(tape)
        op0, op1, op2 = tape.operations[0], tape.operations[1], tape.operations[2]
        cut_edges = [e for e in graph.edges if e[0] is op0 and e[1] is op1]
        cut_edges += [e for e in graph.edges if e[0] is op1 and e[1] is op2]

        cut_graph = qcut.place_wire_cuts(graph=graph, cut_edges=cut_edges)
        wire_cuts = [n for n in cut_graph.nodes if isinstance(n, qml.WireCut)]

        assert len(wire_cuts) == len(cut_edges)
        assert list(cut_graph.pred[wire_cuts[0]]) == [op0]
        assert list(cut_graph.succ[wire_cuts[0]]) == [op1]
        assert list(cut_graph.pred[wire_cuts[1]]) == [op1]
        assert list(cut_graph.succ[wire_cuts[1]]) == [op2]

        # check if order is unique and also if there's enough nodes.
        assert list({i for _, i in cut_graph.nodes.data("order")}) == list(
            range(len(graph.nodes) + len(cut_edges))
        )

    @pytest.mark.parametrize("local_measurement", [False, True])
    @pytest.mark.parametrize("with_manual_cut", [False, True])
    @pytest.mark.parametrize(
        "cut_strategy",
        [
            None,
            qcut.CutStrategy(qml.device("default.qubit", wires=3)),
            qcut.CutStrategy(max_free_wires=4),
            qcut.CutStrategy(max_free_wires=2),  # extreme constraint forcing exhaustive probing.
            qcut.CutStrategy(max_free_wires=2, num_fragments_probed=5),  # impossible to cut
        ],
    )
    def test_find_and_place_cuts(self, local_measurement, with_manual_cut, cut_strategy):
        """Integration tests for auto cutting pipeline."""
        pytest.importorskip("kahypar")

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.RY(0.2, wires=1)
            qml.RX(0.3, wires="a")
            qml.RY(0.4, wires="b")
            qml.CNOT(wires=[0, 1])
            if with_manual_cut:
                qml.WireCut(wires=1)
            qml.CNOT(wires=["a", "b"])
            qml.CNOT(wires=[1, "a"])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=["a", "b"])
            qml.RX(0.5, wires="a")
            qml.RY(0.6, wires="b")
            qml.expval(qml.PauliX(wires=[0]) @ qml.PauliY(wires=["a"]) @ qml.PauliZ(wires=["b"]))

        graph = qcut.tape_to_graph(tape)

        if cut_strategy is None:
            expected_num_cut_edges = 2
            num_frags = 2
            cut_graph = qcut.find_and_place_cuts(
                graph=graph,
                num_fragments=num_frags,
                imbalance=0.5,
                replace_wire_cuts=True,
                seed=self.seed,
                local_measurement=local_measurement,
            )

        elif cut_strategy.num_fragments_probed:
            with pytest.raises(ValueError):
                cut_graph = qcut.find_and_place_cuts(
                    graph=graph,
                    cut_strategy=cut_strategy,
                    local_measurement=local_measurement,
                )
            return

        else:
            cut_graph = qcut.find_and_place_cuts(
                graph=graph,
                cut_strategy=cut_strategy,
                replace_wire_cuts=True,
                seed=self.seed,
                local_measurement=local_measurement,
            )

            if cut_strategy.max_free_wires > 2:
                expected_num_cut_edges = 2
                num_frags = 2
            else:
                # There's some inherent randomness in Kahypar that's not fixable by seed.
                # Need to make this condition a bit relaxed for the extreme case.
                expected_num_cut_edges = [10, 11, 14, 15]
                num_frags = [9, 10, 13, 14]

        frags, comm_graph = qcut.fragment_graph(cut_graph)

        if num_frags == 2:

            assert len(frags) == num_frags
            assert len(comm_graph.edges) == expected_num_cut_edges

            assert (
                len([n for n in cut_graph.nodes if isinstance(n, qcut.MeasureNode)])
                == expected_num_cut_edges
            )
            assert (
                len([n for n in cut_graph.nodes if isinstance(n, qcut.PrepareNode)])
                == expected_num_cut_edges
            )

            # Cutting wire "a" is more balanced, thus will be cut if there's no manually placed cut on
            # wire 1:
            expected_cut_wire = 1 if with_manual_cut else "a"
            assert all(
                list(n.wires) == [expected_cut_wire]
                for n in cut_graph.nodes
                if isinstance(n, (qcut.MeasureNode, qcut.PrepareNode))
            )

            expected_fragment_sizes = [7, 11] if with_manual_cut else [8, 10]
            assert expected_fragment_sizes == [f.number_of_nodes() for f in frags]

        else:
            assert len(frags) in num_frags
            assert len(comm_graph.edges) in expected_num_cut_edges

            assert (
                len([n for n in cut_graph.nodes if isinstance(n, qcut.MeasureNode)])
                in expected_num_cut_edges
            )
            assert (
                len([n for n in cut_graph.nodes if isinstance(n, qcut.PrepareNode)])
                in expected_num_cut_edges
            )


class TestAutoCutCircuit:
    """Integration tests for automatic-cutting-enabled `cut_circuit` transform.
    Mostly borrowing tests cases from ``TestCutCircuitTransform``.
    """

    @pytest.mark.parametrize("max_depth", [0, 1])
    @pytest.mark.parametrize("free_wires", [3, 4])
    def test_complicated_circuit(self, max_depth, free_wires, mocker):
        """
        Tests that the full automatic circuit cutting pipeline returns the correct value and
        gradient for a complex circuit with multiple wire cut scenarios. The circuit is the
        uncut version of the circuit in ``TestCutCircuitTransform.test_complicated_circuit``.
        Note auto cut happens after expansion.

        0: ──BasisState(M0)─╭C───RX─╭C──╭C────────────────────┤
        1: ─────────────────╰X──────╰X──╰Z────────────────╭RX─┤ ╭<Z@Z@Z>
        2: ──H──────────────╭C─────────────╭RY────────╭RY─│───┤ ├<Z@Z@Z>
        3: ─────────────────╰RY──H──╭C───H─╰C──╭RY──H─╰C──│───┤ ╰<Z@Z@Z>
        4: ─────────────────────────╰RY──H─────╰C─────────╰C──┤
        """
        pytest.importorskip("kahypar")

        dev_original = qml.device("default.qubit", wires=5)

        dev_cut = qml.device("default.qubit", wires=free_wires)

        def two_qubit_unitary(param, wires):
            qml.Hadamard(wires=[wires[0]])
            qml.CRY(param, wires=[wires[0], wires[1]])

        def f(params):
            qml.BasisState(np.array([1]), wires=[0])

            qml.CNOT(wires=[0, 1])
            qml.RX(params[0], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.CZ(wires=[0, 1])

            two_qubit_unitary(params[1], wires=[2, 3])
            two_qubit_unitary(params[2] ** 2, wires=[3, 4])
            two_qubit_unitary(np.sin(params[3]), wires=[3, 2])
            two_qubit_unitary(np.sqrt(params[4]), wires=[4, 3])
            two_qubit_unitary(np.cos(params[1]), wires=[3, 2])
            qml.CRX(params[2], wires=[4, 1])

            return qml.expval(qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))

        params = np.array([0.4, 0.5, 0.6, 0.7, 0.8], requires_grad=True)

        circuit = qml.QNode(f, dev_original)
        cut_circuit = qcut.cut_circuit(qml.QNode(f, dev_cut), auto_cutter=True, max_depth=max_depth)

        res_expected = circuit(params)
        grad_expected = qml.grad(circuit)(params)

        spy = mocker.spy(qcut, "qcut_processing_fn")
        res = cut_circuit(params)
        spy.assert_called_once()
        grad = qml.grad(cut_circuit)(params)

        assert np.isclose(res, res_expected)
        assert np.allclose(grad, grad_expected)

    @pytest.mark.parametrize("shots", [None])  # using analytic mode only to save time.
    def test_standard_circuit(self, mocker, shots):
        """
        Tests that the full circuit cutting pipeline returns the correct value for a typical
        scenario. The circuit is the uncut version of the circuit in
        ``TestCutCircuitTransform.test_standard_circuit``:

        0: ─╭U(M1)───────────────────╭U(M4)─┤ ╭<Z@X>
        1: ─╰U(M1)─────╭U(M2)────────╰U(M4)─┤ │
        2: ─╭U(M0)─────╰U(M2)─╭U(M3)────────┤ │
        3: ─╰U(M0)────────────╰U(M3)────────┤ ╰<Z@X>
        """
        pytest.importorskip("kahypar")

        dev_original = qml.device("default.qubit", wires=4)

        # We need a 3-qubit device
        dev_cut = qml.device("default.qubit", wires=3, shots=shots)
        us = [unitary_group.rvs(2**2, random_state=i) for i in range(5)]

        def f():
            qml.QubitUnitary(us[0], wires=[0, 1])
            qml.QubitUnitary(us[1], wires=[2, 3])

            qml.QubitUnitary(us[2], wires=[1, 2])

            qml.QubitUnitary(us[3], wires=[0, 1])
            qml.QubitUnitary(us[4], wires=[2, 3])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(3))

        circuit = qml.QNode(f, dev_original)
        cut_circuit = qcut.cut_circuit(qml.QNode(f, dev_cut), auto_cutter=True)

        res_expected = circuit()

        spy = mocker.spy(qcut, "qcut_processing_fn")
        res = cut_circuit()
        spy.assert_called_once()

        atol = 1e-2 if shots else 1e-8
        assert np.isclose(res, res_expected, atol=atol)

    def test_circuit_with_disconnected_components(self):
        """Tests if a circuit that is fragmented into subcircuits such that some of the subcircuits
        are disconnected from the final terminal measurements is executed correctly after automatic
        cutting."""
        pytest.importorskip("kahypar")

        dev = qml.device("default.qubit", wires=3)

        @qml.transforms.cut_circuit(auto_cutter=True)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.RY(x**2, wires=2)
            return qml.expval(qml.PauliZ(wires=[0]))

        x = 0.4
        res = circuit(x)
        assert np.allclose(res, np.cos(x))

    def test_circuit_with_trivial_wire_cut(self):
        """Tests that a circuit with a trivial wire cut (not separating the circuit into
        fragments) is executed correctly after automatic cutting."""
        pytest.importorskip("kahypar")

        dev = qml.device("default.qubit", wires=2)

        @qml.transforms.cut_circuit(auto_cutter=True)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=[0]))

        x = 0.4
        res = circuit(x)
        assert np.allclose(res, np.cos(x))

    def test_cut_circuit_mc_sample(self):
        """
        Tests that a circuit containing sampling measurements can be cut and
        postprocessed to return bitstrings of the original circuit size.
        """
        pytest.importorskip("kahypar")

        dev = qml.device("default.qubit", wires=3, shots=100)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(wires=[0, 2])

        v = 0.319
        target = circuit(v)

        cut_circuit_bs = qcut.cut_circuit_mc(circuit, device_wires=Wires([1, 2]), auto_cutter=True)
        cut_res_bs = cut_circuit_bs(v)

        assert cut_res_bs.shape == target.shape
        assert type(cut_res_bs) == type(target)

    @pytest.mark.parametrize("measure_all_wires", [False, True])
    def test_cut_mps(self, measure_all_wires):
        """Test auto cut this circuit:
        0: ─╭C──RY───────────────────────────────────────────┤ ╭<Z@Z@Z@Z@Z@Z@Z@Z>
        1: ─╰X──RY─╭C──RY────────────────────────────────────┤ ├<Z@Z@Z@Z@Z@Z@Z@Z>
        2: ────────╰X──RY─╭C──RY─────────────────────────────┤ ├<Z@Z@Z@Z@Z@Z@Z@Z>
        3: ───────────────╰X──RY─╭C──RY──────────────────────┤ ├<Z@Z@Z@Z@Z@Z@Z@Z>
        4: ──────────────────────╰X──RY─╭C──RY───────────────┤ ├<Z@Z@Z@Z@Z@Z@Z@Z>
        5: ─────────────────────────────╰X──RY─╭C──RY────────┤ ├<Z@Z@Z@Z@Z@Z@Z@Z>
        6: ────────────────────────────────────╰X──RY─╭C──RY─┤ ├<Z@Z@Z@Z@Z@Z@Z@Z>
        7: ───────────────────────────────────────────╰X──RY─┤ ╰<Z@Z@Z@Z@Z@Z@Z@Z>

        into this:

        0: ─╭C──RY───────────────────────────────────────────────────────────────────┤ ╭<Z@Z@Z@Z@Z@Z@Z@Z>
        1: ─╰X──RY──//─╭C──RY────────────────────────────────────────────────────────┤ ├<Z@Z@Z@Z@Z@Z@Z@Z>
        2: ────────────╰X──RY──//─╭C──RY─────────────────────────────────────────────┤ ├<Z@Z@Z@Z@Z@Z@Z@Z>
        3: ───────────────────────╰X──RY──//─╭C──RY──────────────────────────────────┤ ├<Z@Z@Z@Z@Z@Z@Z@Z>
        4: ──────────────────────────────────╰X──RY──//─╭C──RY───────────────────────┤ ├<Z@Z@Z@Z@Z@Z@Z@Z>
        5: ─────────────────────────────────────────────╰X──RY──//─╭C──RY────────────┤ ├<Z@Z@Z@Z@Z@Z@Z@Z>
        6: ────────────────────────────────────────────────────────╰X──RY──//─╭C──RY─┤ ├<Z@Z@Z@Z@Z@Z@Z@Z>
        7: ───────────────────────────────────────────────────────────────────╰X──RY─┤ ╰<Z@Z@Z@Z@Z@Z@Z@Z>
        """

        pytest.importorskip("kahypar")

        def block(weights, wires):
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RY(weights[0], wires=wires[0])
            qml.RY(weights[1], wires=wires[1])

        n_wires = 8
        n_block_wires = 2
        n_params_block = 2
        n_blocks = qml.MPS.get_n_blocks(range(n_wires), n_block_wires)
        template_weights = [[0.1, -0.3]] * n_blocks

        cut_strategy = qml.transforms.qcut.CutStrategy(max_free_wires=2)

        with qml.tape.QuantumTape() as tape0:
            qml.MPS(range(n_wires), n_block_wires, block, n_params_block, template_weights)
            if measure_all_wires:
                qml.expval(qml.grouping.string_to_pauli_word("Z" * n_wires))
            else:
                qml.expval(qml.PauliZ(wires=n_wires - 1))

        tape = tape0.expand()
        graph = qcut.tape_to_graph(tape)
        cut_graph = qcut.find_and_place_cuts(
            graph=graph,
            cut_strategy=cut_strategy,
            replace_wire_cuts=True,
        )
        frags, _ = qcut.fragment_graph(cut_graph)
        assert len(frags) == 7

        if measure_all_wires:
            lower, upper = 5, 6
        else:
            lower, upper = 4, 5
        assert all(lower <= f.order() <= upper for f in frags)

        assert all(len(set(e[2] for e in f.edges.data('wire'))) == 2 for f in frags)
