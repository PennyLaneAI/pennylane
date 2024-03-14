# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the batch transform.
"""
# pylint: disable=too-few-public-methods,not-callable

import pennylane as qml
from pennylane import numpy as np


class TestMapBatchTransform:
    """Tests for the map_batch_transform function"""

    def test_result(self, mocker):
        """Test that it correctly applies the transform to be mapped"""
        dev = qml.device("default.qubit.legacy", wires=2)
        H = qml.PauliZ(0) @ qml.PauliZ(1) - qml.PauliX(0)
        x = 0.6
        y = 0.7

        with qml.queuing.AnnotatedQueue() as q1:
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H)

        tape1 = qml.tape.QuantumScript.from_queue(q1)
        with qml.queuing.AnnotatedQueue() as q2:
            qml.Hadamard(wires=0)
            qml.CRX(x, wires=[0, 1])
            qml.CNOT(wires=[0, 1])
            qml.expval(H + 0.5 * qml.PauliY(0))

        tape2 = qml.tape.QuantumScript.from_queue(q2)
        spy = mocker.spy(qml.transforms, "hamiltonian_expand")
        tapes, fn = qml.transforms.map_batch_transform(
            qml.transforms.hamiltonian_expand, [tape1, tape2]
        )

        spy.assert_called()
        assert len(tapes) == 5

        res = qml.execute(tapes, dev, qml.gradients.param_shift, device_batch_transform=False)
        expected = [np.cos(y), 0.5 + 0.5 * np.cos(x) - 0.5 * np.sin(x / 2)]

        assert np.allclose(fn(res), expected)

    def test_differentiation(self):
        """Test that an execution using map_batch_transform can be differentiated"""
        dev = qml.device("default.qubit.legacy", wires=2)
        H = qml.PauliZ(0) @ qml.PauliZ(1) - qml.PauliX(0)

        weights = np.array([0.6, 0.8], requires_grad=True)

        def cost(weights):
            with qml.queuing.AnnotatedQueue() as q1:
                qml.RX(weights[0], wires=0)
                qml.RY(weights[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.expval(H)

            tape1 = qml.tape.QuantumScript.from_queue(q1)
            with qml.queuing.AnnotatedQueue() as q2:
                qml.Hadamard(wires=0)
                qml.CRX(weights[0], wires=[0, 1])
                qml.CNOT(wires=[0, 1])
                qml.expval(H + 0.5 * qml.PauliY(0))

            tape2 = qml.tape.QuantumScript.from_queue(q2)
            tapes, fn = qml.transforms.map_batch_transform(
                qml.transforms.hamiltonian_expand, [tape1, tape2]
            )
            res = qml.execute(tapes, dev, qml.gradients.param_shift, device_batch_transform=False)
            return np.sum(fn(res))

        res = cost(weights)
        x, y = weights
        expected = np.cos(y) + 0.5 + 0.5 * np.cos(x) - 0.5 * np.sin(x / 2)
        assert np.allclose(res, expected)

        res = qml.grad(cost)(weights)
        expected = [-0.5 * np.sin(x) - 0.25 * np.cos(x / 2), -np.sin(y)]
        assert np.allclose(res, expected)
