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
from pennylane.transforms.experimental import transform
from collections.abc import Sequence

# TODO: Replace with default qubit 2

dev = qml.device("default.qubit", wires=2)

with qml.tape.QuantumTape() as tape_circuit:
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=0)
    qml.RZ(0.42, wires=1)
    qml.expval(qml.PauliZ(wires=0))


def qfunc(a):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=0)
    qml.RZ(a, wires=1)
    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(device=dev)
def qfunc(a):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=0)
    qml.RZ(a, wires=1)
    return qml.expval(qml.PauliZ(wires=0))


##########################################
# Non-valid transforms

non_callable = tape_circuit


def no_processing_fn_transform(tape: qml.tape.QuantumTape) -> Sequence[qml.tape.QuantumTape]:
    tape_copy = tape.copy()
    return [tape, tape_copy]


def no_tape_sequence_transform(tape: qml.tape.QuantumTape) -> (qml.tape.QuantumTape, callable):
    return tape, lambda x: x


non_valid_transforms = [non_callable, no_processing_fn_transform, no_tape_sequence_transform]


##########################################
# Valid transforms


def a_valid_transform(
    tape: qml.tape.QuantumTape, index: int
) -> (Sequence[qml.tape.QuantumTape], callable):
    tape.circuit.pop()
    return [tape], lambda x: x


valid_transforms = [a_valid_transform]


##########################################
# Non-valid expand transforms


def no_processing_fn_expand_transform(tape: qml.tape.QuantumTape) -> qml.tape.QuantumTape:
    tape_copy = tape.copy()
    return [tape, tape_copy]


def no_tape_sequence_expand_transform(tape: qml.tape.QuantumTape) -> qml.tape.QuantumTape:
    return tape, lambda x: x


non_valid_expand_transforms = [
    non_callable,
    no_processing_fn_expand_transform,
    no_tape_sequence_expand_transform,
]


##########################################
# Valid expand transforms


def a_valid_expand_transform(tape: qml.tape.QuantumTape) -> (qml.tape.QuantumTape, callable):
    return [tape], lambda x: x


valid_expand_transforms = [a_valid_expand_transform]


class TestTransformDispatcher:
    """Test that adjacent inverse gates are cancelled."""

    def test_dispatcher_signature(self):
        """Test the signature"""

        dispatched_transform = transform(a_valid_transform)
        tapes, fn = dispatched_transform(tape_circuit, 0)

    def test_dispatcher_signature_non_valid_transform(self):
        """Test the signature"""

        dispatched_transform = transform(no_tape_sequence_transform)
