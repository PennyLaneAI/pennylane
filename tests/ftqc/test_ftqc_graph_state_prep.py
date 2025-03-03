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
"""Unit tests for the GraphStatePrep module"""

import pennylane as qml
from pennylane.ftqc import GraphStatePrep, QubitGraph, generate_lattice
from pennylane.transforms.decompose import decompose


class TestGraphStatePrep:
    """Test for graph state prep"""

    def test_circuit_accept_graph_state_prep(self):
        """Test if a quantum function accepts GraphStatePrep."""
        lattice = generate_lattice([2, 2], "square")
        q = QubitGraph("test", lattice.graph)
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(q):
            GraphStatePrep(qubit_graph=q)
            return qml.probs()

        circuit(q)
        assert True

    def test_compute_decompose(self):
        """Test the compute_decomposition method of the GraphStatePrep class."""
        lattice = generate_lattice([2, 2], "square")
        q = QubitGraph("test", lattice.graph)
        queue = GraphStatePrep.compute_decomposition(q)
        assert len(queue) == 8  # 4 ops for |0> -> |+> and 4 ops to entangle nearest qubits

    def test_decompose(self):
        """Test the decomposition method of the GraphStatePrep class."""
        lattice = generate_lattice([2, 2, 2], "cubic")
        q = QubitGraph("test", lattice.graph)
        op = GraphStatePrep(qubit_graph=q)
        queue = op.decomposition()
        assert len(queue) == 20  # 8 ops for |0> -> |+> and 12 ops to entangle nearest qubits
        for op in queue[:8]:
            assert isinstance(op, qml.H)
            assert isinstance(op.wires[0], QubitGraph)
        for op in queue[8:]:
            assert isinstance(op, qml.CZ)
            assert all(isinstance(w, QubitGraph) for w in op.wires))

    def test_preprocess_decompose(self):
        """Test if pennylane.transforms.decompose work with the GraphStatePrep class."""
        lattice = generate_lattice([2, 2, 2], "cubic")
        q = QubitGraph("test", lattice.graph)
        ops = [GraphStatePrep(qubit_graph=q)]
        measurements = [qml.probs()]
        tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)
        expanded_tapes, _ = decompose(tape)
        expanded_tape = expanded_tapes[0]
        for i in range(len(expanded_tape) - 1):
            assert (
                expanded_tape[i].name == "Hadamard"
                if i < len(lattice.nodes)
                else expanded_tape[i].name == "CZ"
            )
            assert isinstance(expanded_tape[i].wires[0], QubitGraph)
            if i >= len(lattice.nodes):
                assert isinstance(expanded_tape[i].wires[1], QubitGraph)
        assert isinstance(expanded_tape, qml.tape.QuantumScript)
