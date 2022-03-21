# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import pennylane as qml
import pennylane.numpy as np
from pennylane.transforms.optimization.peephole import maximal_sequences, peephole_optimization


class TestPeepholeOptimization:
    """Sequences finding for peephole circuit optimization tests."""

    def test_simple_circuit_two_qubits(self):
        """Test peephole optimization (2 qubits) for a simple circuit."""

        def circuit():
            qml.PauliX(wires=2)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[2, 1])
            qml.PauliX(wires=1)
            return qml.expval(qml.PauliX(wires=0))

        dev = qml.device("default.qubit", wires=5)

        qnode = qml.QNode(circuit, dev)
        qnode()

        optimized_qfunc = peephole_optimization(size_qubits_subsets=[2])(circuit)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode()

        cnots_qnode = qml.specs(qnode)()["gate_types"]["CNOT"]
        cnots_optimized_qnode = qml.specs(optimized_qnode)()["gate_types"]["CNOT"]

        assert len(qnode.qtape.operations) == 12
        assert cnots_qnode == 6

        assert len(optimized_qnode.qtape.operations) == 10
        assert cnots_optimized_qnode == 4

        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    def test_simple_circuit_one_qubit(self):
        """Test peephole optimization (1 qubit) for a simple circuit."""

        def circuit():
            qml.PauliX(wires=0)
            qml.PauliY(wires=0)
            qml.PauliZ(wires=0)
            qml.PauliY(wires=0)
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliX(wires=0))

        dev = qml.device("default.qubit", wires=1)

        qnode = qml.QNode(circuit, dev)
        qnode()

        optimized_qfunc = peephole_optimization(size_qubits_subsets=[1])(circuit)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode()

        assert len(qnode.qtape.operations) == 12
        assert len(optimized_qnode.qtape.operations) == 10

        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    def test_simple_circuit_custom_dict(self):
        """Test peephole optimization (2 qubits) for a simple circuit with custom dictionary."""

        def circuit():
            qml.PauliX(wires=2)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[2, 1])
            qml.PauliX(wires=1)
            return qml.expval(qml.PauliX(wires=0))

        dev = qml.device("default.qubit", wires=5)

        qnode = qml.QNode(circuit, dev)
        qnode()

        custom_cost = {
            "PauliX": 1,
            "RX": 1,
            "RY": 1,
            "RZ": 1,
            "Rot": 1,
            "Hadamard": 1,
            "T": 1,
            "S": 1,
            "SX": 1,
            "CNOT": 10,
        }

        optimized_qfunc = peephole_optimization(
            size_qubits_subsets=[2], custom_quantum_cost=custom_cost
        )(circuit)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode()

        cnots_qnode = qml.specs(qnode)()["gate_types"]["CNOT"]
        cnots_optimized_qnode = qml.specs(optimized_qnode)()["gate_types"]["CNOT"]

        assert len(qnode.qtape.operations) == 12
        assert cnots_qnode == 6

        assert len(optimized_qnode.qtape.operations) == 10
        assert cnots_optimized_qnode == 4

        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    def test_two_sequences(self):
        """Test peephole optimization (2 qubits) with a more complicated circuit."""

        def circuit():
            qml.PauliX(wires=2)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[2, 1])
            qml.PauliX(wires=1)
            qml.PauliX(wires=5)
            qml.CNOT(wires=[4, 5])
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[4, 5])
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[3, 4])
            qml.CNOT(wires=[3, 5])
            qml.Hadamard(wires=5)
            qml.CNOT(wires=[5, 4])
            qml.Hadamard(wires=5)
            qml.CNOT(wires=[5, 4])
            qml.PauliX(wires=4)
            return qml.expval(qml.PauliX(wires=0))

        dev = qml.device("default.qubit", wires=6)

        qnode = qml.QNode(circuit, dev)
        qnode()

        optimized_qfunc = peephole_optimization(size_qubits_subsets=[2])(circuit)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode()

        cnots_qnode = qml.specs(qnode)()["gate_types"]["CNOT"]
        cnots_optimized_qnode = qml.specs(optimized_qnode)()["gate_types"]["CNOT"]

        assert len(qnode.qtape.operations) == 24
        assert cnots_qnode == 12

        assert len(optimized_qnode.qtape.operations) == 20
        assert cnots_optimized_qnode == 8

        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    def test_simple_circuit_wrong_size_input(self):
        """Test that an error is raised for wrong size input."""

        def circuit():
            qml.PauliX(wires=2)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[2, 1])
            qml.PauliX(wires=1)
            return qml.expval(qml.PauliX(wires=0))

        dev = qml.device("default.qubit", wires=5)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The list of number of qubits is not a valid. It should be a list containing 1 and or 2.",
        ):
            optimized_qfunc = peephole_optimization(size_qubits_subsets=[3])(circuit)
            optimized_qnode = qml.QNode(optimized_qfunc, dev)
            optimized_qnode()

    def test_size_subset(self):
        """Test that an error is raised for too large subset size compared to the number of qubits in the circuit."""

        def circuit():
            qml.PauliX(wires=2)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[2, 1])
            qml.PauliX(wires=1)
            return qml.expval(qml.PauliX(wires=0))

        circuit_dag = qml.commutation_dag(circuit)()

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The qubits subset considered must be smaller or equal than the number of qubits in the "
            "circuit.",
        ):
            sequence = maximal_sequences(circuit_dag, 4)

    def test_no_sequence(self):
        """Test that an empty gates sequence is returned when there is no sequence."""

        def circuit():
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 0])
            return qml.expval(qml.PauliX(wires=0))

        circuit_dag = qml.commutation_dag(circuit)()

        sequence = maximal_sequences(circuit_dag, 1)

        assert sequence == []

    def test_diamond_shape_dag(self):
        """Test that the right sequence is returned for a specific dag shape"""

        def circuit():
            qml.CNOT(wires=[0, 2])
            qml.CNOT(wires=[1, 0])
            qml.PauliZ(wires=2)
            qml.CNOT(wires=[0, 2])
            qml.PauliX(wires=2)
            return qml.expval(qml.PauliX(wires=0))

        circuit_dag = qml.commutation_dag(circuit)()

        sequence = maximal_sequences(circuit_dag, 2)

        assert sequence[0].sequence == [0, 2, 4]

    def test_multiple_qubits_conf(self):
        """Test that the right sequence is returned for a specific dag shape"""

        def circuit():
            qml.PauliZ(wires=0)
            qml.PauliZ(wires=0)
            return qml.expval(qml.PauliX(wires=1) @ qml.PauliX(wires=2))

        circuit_dag = qml.commutation_dag(circuit)()

        sequence = maximal_sequences(circuit_dag, 2)

        assert sequence[0].sequence == [0, 1]
        assert sequence[0].qubit == [set([0, 1]), set([0, 2])]
