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
from pennylane.transforms.decompose import decompose
from pennylane.ftqc import GraphStatePrep, Lattice, QubitGraph, generate_lattice


class TestGraphStatePrep:
    """Test for graph state prep"""

    def test_circuit_accept_graph_state_prep(self):
        lattice = generate_lattice([2, 2], "square")
        q = QubitGraph("test", lattice.graph)
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(q):
            GraphStatePrep(qubit_graph = q)
            return qml.probs()

        circuit(q)
        assert True

    def test_compute_decompose(self):
        lattice = generate_lattice([2, 2], "square")
        q = QubitGraph("test", lattice.graph)
        decomposed_tape = GraphStatePrep.compute_decomposition(q)
        assert len(decomposed_tape) == 8 # 4 ops for |0> -> |+> and 4 ops to entangle nearest qubits
    
    def test_decompose(self):
        lattice = generate_lattice([2, 2, 2], "cubic")
        q = QubitGraph("test", lattice.graph)
        graphstateobj = GraphStatePrep(qubit_graph = q)
        decomposed_tape = graphstateobj.decomposition()
        assert len(decomposed_tape) == 20 # 8 ops for |0> -> |+> and 12 ops to entangle nearest qubits

