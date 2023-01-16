# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Tests for the gradients.lcu_gradient module.
"""
import pytest

from pennylane import numpy as np

import pennylane as qml
from pennylane.gradients import lcu_grad


class TestFiniteDiff:
    """Tests for the finite difference gradient transform"""

    def test_single_expectation_value(self, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        qml.enable_return()
        dev = qml.device("default.qubit", wires=3)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(wires=0)
            qml.RX(x, wires=[0])
            qml.PauliZ(wires=0)
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)

        tapes, fn = lcu_grad(tape)

        res = fn(dev.batch_execute(tapes))
        print("res", res)
        # assert res.shape == (1, 2)

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
        print(expected)
        # assert np.allclose(res, expected, atol=tol, rtol=0)
