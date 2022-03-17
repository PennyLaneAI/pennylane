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
from pennylane.transforms.optimization.sequences import maximal_sequences, sequences_optimization


class TestSequencesOptimization:
    """Sequences finding for circuit optimization tests."""

    def test_simple_circuit(self):
        """Test sequences finding for circuit optimization for a simple circuit."""

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

        optimized_qfunc = sequences_optimization(size_qubits_subsets=[2])(circuit)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode()

        cnots_qnode = qml.specs(qnode)()["gate_types"]["CNOT"]
        cnots_optimized_qnode = qml.specs(optimized_qnode)()["gate_types"]["CNOT"]

        assert len(qnode.qtape.operations) == 12
        assert cnots_qnode == 6

        assert len(optimized_qnode.qtape.operations) == 10
        assert cnots_optimized_qnode == 4

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
            optimized_qfunc = sequences_optimization(size_qubits_subsets=[3])(circuit)
            optimized_qnode = qml.QNode(optimized_qfunc, dev)
            optimized_qnode()
