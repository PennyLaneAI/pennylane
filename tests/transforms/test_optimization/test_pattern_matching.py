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
import timeit

import pennylane as qml
from pennylane.transforms.optimization.pattern_matching import pattern_matching


class TestPatternMatching:
    """Pattern matching circuit optimization tests."""

    def test_simple_quantum_function_pattern_matching(self):
        """Test pattern matching algorithm for circuit optimization with a CNOTs template."""

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

        qnode = qml.QNode(circuit, dev)
        qnode()

        optimized_qfunc = pattern_matching(pattern_tapes=[template])(circuit)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode()

        cnots_qnode = qml.specs(qnode)()["gate_types"]["CNOT"]
        cnots_optimized_qnode = qml.specs(optimized_qnode)()["gate_types"]["CNOT"]

        assert len(qnode.qtape.operations) == 8
        assert cnots_qnode == 4

        assert len(optimized_qnode.qtape.operations) == 7
        assert cnots_optimized_qnode == 3

    def test_mod_5_4_pattern_matching(self):
        """Test pattern matching algorithm for mod_5_4 with a CNOTs template."""

        def mod_5_4():
            qml.PauliX(wires=4)
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[3, 4])
            qml.CNOT(wires=[0, 4])
            qml.T(wires=4)
            qml.CNOT(wires=[3, 4])
            qml.adjoint(qml.T)(wires=4)
            qml.CNOT(wires=[0, 4])
            qml.CNOT(wires=[0, 3])
            qml.adjoint(qml.T)(wires=3)
            qml.CNOT(wires=[0, 3])
            qml.CNOT(wires=[3, 4])
            qml.CNOT(wires=[2, 4])
            qml.adjoint(qml.T)(wires=4)
            qml.CNOT(wires=[3, 4])
            qml.T(wires=4)
            qml.CNOT(wires=[2, 4])
            qml.CNOT(wires=[2, 3])
            qml.T(wires=3)
            qml.CNOT(wires=[2, 3])
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[3, 4])
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[2, 4])
            qml.adjoint(qml.T)(wires=4)
            qml.CNOT(wires=[1, 4])
            qml.T(wires=4)
            qml.CNOT(wires=[2, 4])
            qml.adjoint(qml.T)(wires=4)
            qml.CNOT(wires=[1, 4])
            qml.T(wires=4)
            qml.CNOT(wires=[1, 2])
            qml.adjoint(qml.T)(wires=2)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[2, 4])
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[1, 4])
            qml.T(wires=4)
            qml.CNOT(wires=[0, 4])
            qml.adjoint(qml.T)(wires=4)
            qml.CNOT(wires=[1, 4])
            qml.T(wires=4)
            qml.CNOT(wires=[0, 4])
            qml.adjoint(qml.T)(wires=4)
            qml.CNOT(wires=[0, 1])
            qml.T(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[1, 4])
            qml.CNOT(wires=[0, 4])
            return qml.expval(qml.PauliX(wires=0))

        with qml.tape.QuantumTape() as template:
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])

        dev = qml.device("default.qubit", wires=5)

        qnode = qml.QNode(mod_5_4, dev)
        qnode()

        optimized_qfunc = pattern_matching(pattern_tapes=[template])(mod_5_4)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode()

        cnots_qnode = qml.specs(qnode)()["gate_types"]["CNOT"]
        cnots_optimized_qnode = qml.specs(optimized_qnode)()["gate_types"]["CNOT"]

        assert len(qnode.qtape.operations) == 51
        assert cnots_qnode == 28

        assert len(optimized_qnode.qtape.operations) == 49
        assert cnots_optimized_qnode == 26

    def test_vbe_adder_3_pattern_matching(self):
        """Test pattern matching algorithm for vbe_adder_3 with a CNOTs template."""

        def vbe_adder_3():
            qml.T(wires=7)
            qml.T(wires=8)
            qml.Hadamard(wires=3)
            qml.CNOT(wires=[2, 3])
            qml.adjoint(qml.T)(wires=3)
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[2, 3])
            qml.adjoint(qml.T)(wires=3)
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[0, 3])
            qml.T(wires=3)
            qml.CNOT(wires=[2, 3])
            qml.adjoint(qml.T)(wires=3)
            qml.CNOT(wires=[0, 3])
            qml.S(wires=3)
            qml.Hadamard(wires=3)
            qml.Hadamard(wires=6)
            qml.CNOT(wires=[5, 6])
            qml.adjoint(qml.T)(wires=6)
            qml.CNOT(wires=[4, 6])
            qml.CNOT(wires=[5, 6])
            qml.adjoint(qml.T)(wires=6)
            qml.CNOT(wires=[4, 6])
            qml.CNOT(wires=[4, 5])
            qml.CNOT(wires=[4, 6])
            qml.CNOT(wires=[3, 6])
            qml.T(wires=6)
            qml.CNOT(wires=[5, 6])
            qml.adjoint(qml.T)(wires=6)
            qml.CNOT(wires=[3, 6])
            qml.S(wires=6)
            qml.Hadamard(wires=6)
            qml.T(wires=6)
            qml.Hadamard(wires=9)
            qml.CNOT(wires=[8, 9])
            qml.adjoint(qml.T)(wires=9)
            qml.CNOT(wires=[7, 9])
            qml.CNOT(wires=[8, 9])
            qml.adjoint(qml.T)(wires=9)
            qml.CNOT(wires=[7, 9])
            qml.CNOT(wires=[7, 8])
            qml.CNOT(wires=[8, 9])
            qml.CNOT(wires=[6, 9])
            qml.T(wires=9)
            qml.CNOT(wires=[8, 9])
            qml.adjoint(qml.T)(wires=9)
            qml.CNOT(wires=[6, 9])
            qml.T(wires=9)
            qml.CNOT(wires=[6, 8])
            qml.adjoint(qml.T)(wires=8)
            qml.Hadamard(wires=9)
            qml.Hadamard(wires=6)
            qml.adjoint(qml.T)(wires=6)
            qml.CNOT(wires=[5, 6])
            qml.CNOT(wires=[3, 6])
            qml.adjoint(qml.T)(wires=6)
            qml.CNOT(wires=[5, 6])
            qml.T(wires=6)
            qml.CNOT(wires=[3, 6])
            qml.CNOT(wires=[4, 5])
            qml.CNOT(wires=[5, 6])
            qml.T(wires=6)
            qml.CNOT(wires=[4, 6])
            qml.CNOT(wires=[5, 6])
            qml.T(wires=6)
            qml.CNOT(wires=[4, 6])
            qml.CNOT(wires=[3, 5])
            qml.CNOT(wires=[4, 5])
            qml.Hadamard(wires=6)
            qml.Hadamard(wires=3)
            qml.adjoint(qml.S)(wires=3)
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[0, 3])
            qml.adjoint(qml.T)(wires=3)
            qml.CNOT(wires=[2, 3])
            qml.T(wires=3)
            qml.CNOT(wires=[0, 3])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.T(wires=3)
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[2, 3])
            qml.T(wires=3)
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[0, 2])
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=3)
            return qml.expval(qml.PauliX(wires=0))

        with qml.tape.QuantumTape() as template:
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])

        dev = qml.device("default.qubit", wires=10)

        qnode = qml.QNode(vbe_adder_3, dev)
        qnode()

        optimized_qfunc = pattern_matching(pattern_tapes=[template])(vbe_adder_3)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode()

        cnots_qnode = qml.specs(qnode)()["gate_types"]["CNOT"]
        cnots_optimized_qnode = qml.specs(optimized_qnode)()["gate_types"]["CNOT"]

        assert len(qnode.qtape.operations) == 89
        assert cnots_qnode == 50

        assert len(optimized_qnode.qtape.operations) == 84
        assert cnots_optimized_qnode == 45
