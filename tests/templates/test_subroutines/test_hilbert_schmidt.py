# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the qft template.
"""
import pytest

import numpy as np
import pennylane as qml


class TestHilbertSchmidt:
    """Tests for the Hilbert Schmidt template."""

    def test_HS_decomposition_1_qubit(self):
        """Test if the HS operation is correctly decomposed"""
        with qml.tape.QuantumTape(do_queue=False) as U:
            qml.Hadamard(wires=0)

        def v_circuit(params):
            qml.RZ(params[0], wires=1)

        op = qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_tape=U)

        with qml.tape.QuantumTape() as tape_dec:
            op.decomposition()

        expected_operations = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
            qml.RZ(-0.1, wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]
        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.data == j.data
            assert i.wires == j.wires

    def test_HS_decomposition_2_qubit(self):
        """Test if the HS operation is correctly decomposed"""
        with qml.tape.QuantumTape(do_queue=False) as U:
            qml.SWAP(wires=[0, 1])

        def v_circuit(params):
            qml.RZ(params[0], wires=2)
            qml.CNOT(wires=[2, 3])

        op = qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[2, 3], u_tape=U)

        with qml.tape.QuantumTape() as tape_dec:
            op.decomposition()

        expected_operations = [
            qml.Hadamard(wires=[0]),
            qml.Hadamard(wires=[1]),
            qml.CNOT(wires=[0, 2]),
            qml.CNOT(wires=[1, 3]),
            qml.SWAP(wires=[0, 1]),
            qml.RZ(-0.1, wires=[2]),
            qml.CNOT(wires=[2, 3]),
            qml.CNOT(wires=[1, 3]),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=[0]),
            qml.Hadamard(wires=[1]),
        ]

        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.data == j.data
            assert i.wires == j.wires


class TestLocalHilbertSchmidt:
    """Tests for the Local Hilbert Schmidt template."""

    def test_lhs_decomposition_1_qubit(self):
        """Test if the LHS operation is correctly decomposed"""
        with qml.tape.QuantumTape(do_queue=False) as U:
            qml.Hadamard(wires=0)

        def v_circuit(params):
            qml.RZ(params[0], wires=1)

        op = qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_tape=U)

        with qml.tape.QuantumTape() as tape_dec:
            op.decomposition()

        expected_operations = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
            qml.RZ(-0.1, wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]

        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.data == j.data
            assert i.wires == j.wires

    def test_HS_decomposition_2_qubit(self):
        """Test if the HS operation is correctly decomposed"""
        with qml.tape.QuantumTape(do_queue=False) as U:
            qml.SWAP(wires=[0, 1])

        def v_circuit(params):
            qml.RZ(params[0], wires=2)
            qml.CNOT(wires=[2, 3])

        op = qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[2, 3], u_tape=U)

        with qml.tape.QuantumTape() as tape_dec:
            op.decomposition()

        expected_operations = [
            qml.Hadamard(wires=[0]),
            qml.Hadamard(wires=[1]),
            qml.CNOT(wires=[0, 2]),
            qml.CNOT(wires=[1, 3]),
            qml.SWAP(wires=[0, 1]),
            qml.RZ(-0.1, wires=[2]),
            qml.CNOT(wires=[2, 3]),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=[0]),
        ]

        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.data == j.data
            assert i.wires == j.wires
