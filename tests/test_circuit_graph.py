# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the :mod:`pennylane.circuit_graph` module.
"""
# pylint: disable=no-self-use,too-many-arguments,protected-access

from unittest.mock import MagicMock

import numpy as np

import pennylane as qml

from pennylane.operation import Expectation
from pennylane.circuit_graph import CircuitGraph

import pytest


@pytest.fixture()
def queue():
    """A fixture of a complex example of operations that depend on previous operations."""
    return [
        qml.RX(0.43, wires=0, do_queue=False),
        qml.RY(0.35, wires=1, do_queue=False),
        qml.RZ(0.35, wires=2, do_queue=False),
        qml.CNOT(wires=[0, 1], do_queue=False),
        qml.Hadamard(wires=2, do_queue=False),
        qml.CNOT(wires=[2, 0], do_queue=False),
        qml.PauliX(wires=1, do_queue=False),
    ]


@pytest.fixture()
def obs():
    """A fixture of observables to go after the queue fixture."""
    return [
        qml.expval(qml.PauliX(wires=0, do_queue=False)),
        qml.expval(qml.Hermitian(np.identity(4), wires=[1, 2], do_queue=False)),
    ]


@pytest.fixture()
def circuit(queue, obs):
    """A fixture of a circuit generated based on the queue and obs fixtures above."""
    circuit = CircuitGraph(queue, obs)
    return circuit


class TestCircuitGraph:
    """Test conversion of queues to DAGs"""

    def test_no_dependence(self):
        """Test case where operations do not depend on each other.
        This should result in a graph with no edges."""

        queue = [qml.RX(0.43, wires=0, do_queue=False), qml.RY(0.35, wires=1, do_queue=False)]

        obs = []

        res = CircuitGraph(queue, obs).graph
        assert len(res) == 2
        assert not res.edges()

    def test_dependence(self, queue, obs):
        """Test a more complex example containing operations
        that do depend on the result of previous operations"""

        circuit = CircuitGraph(queue, obs)
        res = circuit.graph
        assert len(res) == 9

        nodes = res.nodes().data()

        # the three rotations should be starting nodes in the graph
        assert nodes[0]["name"] == "RX"
        assert nodes[0]["op"] == queue[0]

        assert nodes[1]["name"] == "RY"
        assert nodes[1]["op"] == queue[1]

        assert nodes[2]["name"] == "RZ"
        assert nodes[2]["op"] == queue[2]

        # node 0 and node 1 should then connect to the CNOT gate
        assert nodes[3]["name"] == "CNOT"
        assert nodes[3]["op"] == queue[3]
        assert (0, 3) in res.edges()
        assert (1, 3) in res.edges()

        # RZ gate connects directly to a hadamard gate
        assert nodes[4]["name"] == "Hadamard"
        assert nodes[4]["op"] == queue[4]
        assert (2, 4) in res.edges()

        # hadamard gate and CNOT gate feed straight into another CNOT
        assert nodes[5]["name"] == "CNOT"
        assert nodes[5]["op"] == queue[5]
        assert (4, 5) in res.edges()
        assert (3, 5) in res.edges()

        # finally, the first CNOT also connects to a PauliX gate
        assert nodes[6]["name"] == "PauliX"
        assert nodes[6]["op"] == queue[6]
        assert (3, 6) in res.edges()

        # Measurements
        # PauliX is measured on the output of the second CNOT
        assert nodes[7]["name"] == "PauliX"
        assert nodes[7]["op"] == obs[0]
        assert nodes[7]["op"].return_type == Expectation
        assert (5, 7) in res.edges()

        # Hermitian is measured on the output of the second CNOT and the PauliX
        assert nodes[8]["name"] == "Hermitian"
        assert nodes[8]["op"] == obs[1]
        assert nodes[8]["op"].return_type == Expectation
        assert (5, 8) in res.edges()
        assert (6, 8) in res.edges()

        # Finally, checking the adjacency of the returned DAG:
        assert sorted(res.edges()) == [
            (0, 3),
            (1, 3),
            (2, 4),
            (3, 5),
            (3, 6),
            (4, 5),
            (5, 7),
            (5, 8),
            (6, 8),
        ]

    def test_ancestors_and_descendants_example(self, queue, obs):
        """
        Test that the `ancestors` and `descendants` methods return the expected result.
        """
        circuit = CircuitGraph(queue, obs)

        ancestors = circuit.ancestors([6])
        assert len(ancestors) == 3
        for o_idx in (0, 1, 3):
            assert queue[o_idx] in circuit.get_ops(ancestors)

        descendants = circuit.get_ops(circuit.descendants([6]))
        assert descendants == [obs[1]]

    def test_get_nodes_example(self, queue, obs):
        """
        Given a sample circuit, test that the `get_nodes` method returns the expected result.
        """
        circuit = CircuitGraph(queue, obs)

        o_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        nodes = circuit.get_nodes(o_idxs)

        assert nodes[0]["op"] == queue[0]
        assert nodes[1]["op"] == queue[1]
        assert nodes[2]["op"] == queue[2]
        assert nodes[3]["op"] == queue[3]
        assert nodes[4]["op"] == queue[4]
        assert nodes[5]["op"] == queue[5]
        assert nodes[6]["op"] == queue[6]
        assert nodes[7]["op"] == obs[0]
        assert nodes[8]["op"] == obs[1]

        assert nodes[0]["idx"] == 0
        assert nodes[1]["idx"] == 1
        assert nodes[2]["idx"] == 2
        assert nodes[3]["idx"] == 3
        assert nodes[4]["idx"] == 4
        assert nodes[5]["idx"] == 5
        assert nodes[6]["idx"] == 6
        assert nodes[7]["idx"] == 7
        assert nodes[8]["idx"] == 8

        assert nodes[0]["name"] == "RX"
        assert nodes[1]["name"] == "RY"
        assert nodes[2]["name"] == "RZ"
        assert nodes[3]["name"] == "CNOT"
        assert nodes[4]["name"] == "Hadamard"
        assert nodes[5]["name"] == "CNOT"
        assert nodes[6]["name"] == "PauliX"
        assert nodes[7]["name"] == "PauliX"
        assert nodes[8]["name"] == "Hermitian"

        assert nodes[0]["return_type"] is None
        assert nodes[1]["return_type"] is None
        assert nodes[2]["return_type"] is None
        assert nodes[3]["return_type"] is None
        assert nodes[4]["return_type"] is None
        assert nodes[5]["return_type"] is None
        assert nodes[6]["return_type"] is None
        assert nodes[7]["return_type"] == Expectation
        assert nodes[8]["return_type"] == Expectation

    def test_get_nodes(self, circuit, monkeypatch):
        """Test that `get_nodes` fetches the correct nodes from the graph"""
        mock_graph = MagicMock()
        monkeypatch.setattr(circuit, "_graph", mock_graph)

        o_idxs = list(range(10))
        result = circuit.get_nodes(o_idxs)
        assert result == [mock_graph.nodes[i] for i in o_idxs]

    def test_get_ops(self, circuit, monkeypatch):
        """Test that `get_ops` fetches the correct ops based on given nodes"""
        mock_nodes = MagicMock()
        mock_ops = MagicMock()
        mock_get_nodes = MagicMock()
        mock_get_nodes.return_value = [mock_nodes.a, mock_nodes.b]
        monkeypatch.setattr(circuit, "get_nodes", mock_get_nodes)

        result = circuit.get_ops(mock_ops)
        mock_get_nodes.assert_called_once_with(mock_ops)
        assert result == [mock_nodes.a["op"], mock_nodes.b["op"]]

    def test_update_node(self, circuit, monkeypatch):
        """Test that `nx.set_node_attributes` is called correctly when updating a node using
        `update_node` and a given node and op"""
        mock_command = MagicMock()
        mock_nx = MagicMock()

        monkeypatch.setattr("pennylane.circuit_graph.nx", mock_nx)
        monkeypatch.setattr("pennylane.circuit_graph.Command", mock_command)

        node = MagicMock()
        op = MagicMock()

        circuit.update_node(node, op)

        mock_command.assert_called_once_with(
            name=op.name, op=op, return_type=op.return_type, idx=node["idx"]
        )
        mock_nx.set_node_attributes.assert_called_once_with(
            circuit._graph,
            {
                node["idx"]: {
                    **mock_command(
                        name=op.name, op=op, return_type=op.return_type, idx=node["idx"]
                    )._asdict()
                }
            },
        )
