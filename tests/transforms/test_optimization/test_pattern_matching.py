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

import pennylane as qml
from pennylane.transforms.optimization.pattern_matching import pattern_matching


class TestPatternMatching:
    """Pattern matching circuit optimization tests."""

    def test_quantum_function_pattern_matching(self):
        """Test pattern matching algorithmm for circuit optimization with a CNOTs template."""

        def circuit():
            qml.Toffoli(wires=[3, 4, 0])
            qml.CNOT(wires=[1, 4])
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=3)
            qml.PauliZ(wires=1)
            qml.CNOT(wires=[2, 3])
            qml.Toffoli(wires=[2, 3, 0])
            qml.CNOT(wires=[1, 4])
            return qml.expval(qml.PauliX(wires=0))

        with qml.tape.QuantumTape() as template:
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])

        dev = qml.device("default.qubit", wires=5)
        optimized_qfunc = pattern_matching(pattern_tapes=[template])(circuit)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode()
        assert len(optimized_qnode.qtape.operations) == 7
