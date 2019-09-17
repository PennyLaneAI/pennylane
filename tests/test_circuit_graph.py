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

import numpy as np

import pennylane as qml

from pennylane.operation import Expectation
from pennylane.circuit_graph import CircuitGraph


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

    def test_dependence(self):
        """Test a more complex example containing operations
        that do depend on the result of previous operations"""

        queue = [
            qml.RX(0.43, wires=0, do_queue=False),
            qml.RY(0.35, wires=1, do_queue=False),
            qml.RZ(0.35, wires=2, do_queue=False),
            qml.CNOT(wires=[0, 1], do_queue=False),
            qml.Hadamard(wires=2, do_queue=False),
            qml.CNOT(wires=[2, 0], do_queue=False),
            qml.PauliX(wires=1, do_queue=False),
        ]

        obs = [
            qml.expval(qml.PauliX(wires=0, do_queue=False)),
            qml.expval(qml.Hermitian(np.identity(4), wires=[1, 2], do_queue=False)),
        ]

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
