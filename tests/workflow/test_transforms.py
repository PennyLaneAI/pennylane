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
import numpy as np

import pytest
import pennylane as qml
import pennylane.math
from pennylane.workflow import execute

dev = qml.device("default.qubit", wires=2)


@qml.qnode(device=dev)
def qnode_circuit(a):
    """QNode circuit."""
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=1)
    qml.RX(a, wires=0)
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
    return [tape1], lambda x: x[0]

@qml.transforms.transform
def sum_transform(
        tape: qml.tape.QuantumTape
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """A valid (dummy) transform that duplicates the tapes and sum the results."""
    tape1 = tape.copy()
    tape2 = tape.copy()

    def fn(results):
        return qml.numpy.tensor(qml.math.sum(results))

    return [tape1, tape2], fn


class TestExecutionTransformPrograms:
    """Class to test the execution of transform programs."""

    def test_shift_transform_execute(self):
        """Test the shift transform on a qnode."""
        transformed_qnode = shift_transform(qnode_circuit, 0.1)
        assert isinstance(transformed_qnode(0.5), qml.numpy.tensor)

        # results

    def test_sum_transform_execute(self):
        """Test the sum transform on a qnode."""
        transformed_qnode = sum_transform(qnode_circuit)
        assert isinstance(transformed_qnode(0.5), qml.numpy.tensor)
        # results

    def test_sum_shift_transform_execute(self):
        """Test the sum and shift transforms on a qnode."""
        transformed_qnode = sum_transform(shift_transform(qnode_circuit, 0.1))
        assert isinstance(transformed_qnode(0.5), qml.numpy.tensor)
        # results

    def test_shift_sum_transform_execute(self):
        """Test the shift and sum transforms on a qnode."""
        transformed_qnode = shift_transform(sum_transform(qnode_circuit), 0.1)
        assert isinstance(transformed_qnode(0.5), qml.numpy.tensor)
        # results

    def test_sum_shift_results_transform_execute(self):
        """Test that the sum and shift transforms are not commuting"""
        transformed_qnode_1 = sum_transform(shift_transform(qnode_circuit, 0.1))
        transformed_qnode_2 = shift_transform(sum_transform(qnode_circuit), 0.1)
        assert isinstance(transformed_qnode_1(0.5), qml.numpy.tensor)
        assert isinstance(transformed_qnode_2(0.5), qml.numpy.tensor)
        # Transforms are not commuting
        assert not np.allclose(transformed_qnode_1(0.5), transformed_qnode_2(0.5))

