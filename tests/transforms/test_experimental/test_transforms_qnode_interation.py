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
# pylint: disable=not-callable
from typing import Sequence, Callable
from functools import partial
import copy
import numpy as np

import pennylane as qml

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
    tape1 = copy.deepcopy(tape)
    parameters = tape1.get_parameters(trainable_only=False)
    parameters = [param + alpha for param in parameters]
    tape1.set_parameters(parameters, trainable_only=False)

    def null_post_processing(results):
        return results[0]

    return (tape1,), null_post_processing


@qml.transforms.transform
def sum_transform(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
    """A valid (dummy) transform that duplicates the tapes and sum the results."""

    def fn(results):
        return qml.numpy.tensor(qml.math.sum(results))

    return (tape, tape), fn


@partial(qml.transforms.transform, is_informative=True)
def len_transform(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
    """A valid (dummy) informative transform that returns the length of each circuit."""

    def fn(results):
        return len(results[0])

    return (tape,), fn


class TestExecutionTransformPrograms:
    """Class to test the execution of different transform programs."""

    def test_shift_transform_execute(self):
        """Test the shift transform on a qnode."""
        transformed_qnode = shift_transform(qnode_circuit, 0.1)
        res = transformed_qnode(0.5)

        assert isinstance(res, qml.numpy.tensor)
        assert res == qnode_circuit(0.6)

    def test_sum_transform_execute(self):
        """Test the sum transform on a qnode."""
        transformed_qnode = sum_transform(qnode_circuit)
        res = transformed_qnode(0.5)

        assert isinstance(res, qml.numpy.tensor)
        assert res == 2 * qnode_circuit(0.5)

    def test_multiple_sum_transform_execute(self):
        """Test composition of sum transform on a qnode."""
        transformed_qnode = sum_transform(sum_transform(qnode_circuit))
        res = transformed_qnode(0.5)

        assert isinstance(res, qml.numpy.tensor)
        assert res == 4 * qnode_circuit(0.5)

        transformed_qnode = sum_transform(sum_transform(sum_transform(qnode_circuit)))
        res = transformed_qnode(0.5)

        assert isinstance(res, qml.numpy.tensor)
        assert res == 8 * qnode_circuit(0.5)

    def test_sum_shift_transform_execute(self):
        """Test the sum and shift transforms on a qnode."""
        transformed_qnode = sum_transform(shift_transform(qnode_circuit, 0.1))

        res = transformed_qnode(0.5)
        assert isinstance(res, qml.numpy.tensor)
        assert res == 2 * qnode_circuit(0.6)

    def test_multiple_shift_transform_execute(self):
        """Test multiple shift transformss on a qnode."""
        transformed_qnode = shift_transform(shift_transform(qnode_circuit, 0.1), 0.2)

        res = transformed_qnode(0.5)
        assert isinstance(res, qml.numpy.tensor)
        assert res == qnode_circuit(0.8)

    def test_shift_sum_transform_execute(self):
        """Test the shift and sum transforms on a qnode."""
        transformed_qnode = shift_transform(sum_transform(qnode_circuit), 0.1)

        res = transformed_qnode(0.5)
        assert isinstance(res, qml.numpy.tensor)
        assert res == 2 * qnode_circuit(0.6)
        # results

    def test_sum_shift_results_transform_execute(self):
        """Test that the sum and shift transforms are commuting"""
        transformed_qnode_1 = sum_transform(shift_transform(qnode_circuit, 0.1))
        transformed_qnode_2 = shift_transform(sum_transform(qnode_circuit), 0.1)
        assert isinstance(transformed_qnode_1(0.5), qml.numpy.tensor)
        assert isinstance(transformed_qnode_2(0.5), qml.numpy.tensor)
        # Transforms are not commuting
        assert np.allclose(transformed_qnode_1(0.5), transformed_qnode_2(0.5))

    def test_len_transform_informative(self):
        """Test the len informative transform."""
        transformed_qnode = len_transform(qnode_circuit)
        res = transformed_qnode(0.5)
        assert res == 4

    def test_len_sum_composition_informative(self):
        """Test the len informative transform with the sum transform."""
        transformed_qnode = len_transform(sum_transform(sum_transform(qnode_circuit)))
        res = transformed_qnode(0.5)
        assert res == 4 * 4

    def test_len_shift_composition_informative(self):
        """Test the len informative transform with the sum transform."""
        transformed_qnode = len_transform(shift_transform(qnode_circuit, 0.2))
        res = transformed_qnode(0.5)
        assert res == 4
