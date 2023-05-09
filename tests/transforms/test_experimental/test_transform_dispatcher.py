# Copyright 2023 Xanadu Quantum Technologies Inc.

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
from pennylane.transforms.experimental import transform_dispatcher

with qml.tape.QuantumTape() as tape_circuit:
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=0)
    qml.RZ(0.42, wires=1)


def incorrect_transform(tape: qml.tape.QuantumTape) -> qml.tape.QuantumTape:
    tape.circuit.pop()
    return tape, lambda x: x


def transform(tape: qml.tape.QuantumTape) -> (qml.tape.QuantumTape, callable):
    tape.circuit.pop()
    return tape, lambda x: x


# def my_batch_transform():
#
# def my_informational_transform(tape)
#     def fn():
#         return tape
#     return [tape], fn

class TestTransformDispatcher:
    """Test that adjacent inverse gates are cancelled."""

    def test_dispatcher_signature(self):
        """Test that a single-qubit circuit with adjacent self-inverse gate cancels."""

        dispatched_transform = transform_dispatcher(incorrect_transform)

        dispatched_transform(tape_circuit)

# class TestTransformDispatcherTape:
#     """Test that adjacent inverse gates are cancelled."""
#
#     def test_one_qubit_cancel_adjacent_self_inverse(self):
#         """Test that a single-qubit circuit with adjacent self-inverse gate cancels."""
#
#         def qfunc():
#             qml.Hadamard(wires=0)
#             qml.Hadamard(wires=0)
#
#         transformed_qfunc = cancel_inverses(qfunc)
#
#         new_tape = qml.tape.make_qscript(transformed_qfunc)()
#
#         assert len(new_tape.operations) == 0
#
#
# class TestTransformDispatcherQfunc:
#     """Test that adjacent inverse gates are cancelled."""
#
#     def test_one_qubit_cancel_adjacent_self_inverse(self):
#         """Test that a single-qubit circuit with adjacent self-inverse gate cancels."""
#
#         def qfunc():
#             qml.Hadamard(wires=0)
#             qml.Hadamard(wires=0)
#
#         transformed_qfunc = cancel_inverses(qfunc)
#
#         new_tape = qml.tape.make_qscript(transformed_qfunc)()
#
#         assert len(new_tape.operations) == 0
#
#
# class TestTransformDispatcherQNode:
#     """Test that adjacent inverse gates are cancelled."""
#
#     def test_one_qubit_cancel_adjacent_self_inverse(self):
#         """Test that a single-qubit circuit with adjacent self-inverse gate cancels."""
#
#         def qfunc():
#             qml.Hadamard(wires=0)
#             qml.Hadamard(wires=0)
#
#         transformed_qfunc = cancel_inverses(qfunc)
#
#         new_tape = qml.tape.make_qscript(transformed_qfunc)()
#
#         assert len(new_tape.operations) == 0
