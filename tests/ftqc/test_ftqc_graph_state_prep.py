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
"""Unit tests for the lattice module"""

import networkx as nx
import pytest

import pennylane as qml

from pennylane.ftqc import Lattice, generate_lattice, GraphStatePrep, QubitGraph

class TestGraphStatePrep:
    """Test for graph state prep"""

    def test_circuit_accept_graph_state_prep(self):
        lattice = generate_lattice([2,2], "square")
        q = []
        for i in range(len(lattice.nodes)):
            q.append(QubitGraph(i))
        dev = qml.device("default.qubit")
        @qml.qnode(dev)
        def circuit(lattice, q):
            GraphStatePrep(lattice, q)
            return qml.probs()
        circuit(lattice, q)
        assert True