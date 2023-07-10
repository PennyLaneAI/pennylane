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
"""Unit and integration tests for execution of transform programs."""
from typing import Sequence, Callable

import pytest
import pennylane as qml
from pennylane.workflow import execute

dev = qml.device("default.qubit", wires=2)


@qml.qnode(device=dev)
def qnode_circuit(a):
    """QNode circuit."""
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=0)
    qml.RZ(a, wires=1)
    return qml.expval(qml.PauliZ(wires=0))


@qml.transforms.transform
def shift_transform(
        tape: qml.tape.QuantumTape, alpha: float
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """A valid (dummy) transform that shift all angles."""
    tape1 = tape.copy()
    parameters = tape1.get_parameters(trainable_only=False)
    parameters = [param + alpha for param in parameters]
    tape1.set_parameters(parameters, trainable_only=False)
    return [tape1], lambda x: x

@qml.transforms.transform
def sum_transform(
        tape: qml.tape.QuantumTape
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """A valid (dummy) transform that duplicates the tapes and sum the results."""
    tape1 = tape.copy()
    tape2 = tape.copy()

    def fn(results):
        return qml.math.sum(results)

    return [tape1, tape2], fn


class TestExecutionTransformPrograms:

    def test_shift_transform_execute(self):
        transformed_qnode = shift_transform(qnode_circuit, 0.1)(0.5)
        print(transformed_qnode)
