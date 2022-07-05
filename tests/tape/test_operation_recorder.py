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

import pytest

import pennylane as qml
from pennylane.tape import OperationRecorder


class TestOperationRecorder:
    """Test the OperationRecorder class."""

    def test_circuit_integration(self):
        """Tests that the OperationRecorder integrates well with the
        core behaviour of PennyLane."""
        expected_output = (
            "Operations\n"
            + "==========\n"
            + "PauliY(wires=[0])\n"
            + "PauliY(wires=[1])\n"
            + "RZ(0.4, wires=[0])\n"
            + "RZ(0.4, wires=[1])\n"
            + "CNOT(wires=[0, 1])\n"
            + "\n"
            + "Observables\n"
            + "===========\n"
        )

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)

            with qml.tape.OperationRecorder() as recorder:
                ops = [
                    qml.PauliY(0),
                    qml.PauliY(1),
                    qml.RZ(c, wires=0),
                    qml.RZ(c, wires=1),
                    qml.CNOT(wires=[0, 1]),
                ]

            assert str(recorder) == expected_output
            assert recorder.queue == ops

            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        circuit(0.1, 0.2, 0.4)

    def test_template_integration(self):
        """Tests that the OperationRecorder integrates well with the
        core behaviour of PennyLane."""
        expected_output = (
            "Operations\n"
            + "==========\n"
            + "RZ(0, wires=[0])\n"
            + "RZ(3, wires=[0])\n"
            + "RZ(6, wires=[0])\n"
            + "RZ(9, wires=[0])\n"
            + "RZ(12, wires=[0])\n"
            + "\n"
            + "Observables\n"
            + "===========\n"
        )

        def template(x):
            for i in range(5):
                qml.RZ(i * x, wires=0)

        with qml.tape.OperationRecorder() as recorder:
            template(3)

        assert str(recorder) == expected_output

    def test_template_with_return_integration(self):
        """Tests that the OperationRecorder integrates well with the
        core behaviour of PennyLane."""
        expected_output = (
            "Operations\n"
            + "==========\n"
            + "RZ(0, wires=[0])\n"
            + "RZ(3, wires=[0])\n"
            + "RZ(6, wires=[0])\n"
            + "RZ(9, wires=[0])\n"
            + "RZ(12, wires=[0])\n"
            + "\n"
            + "Observables\n"
            + "===========\n"
            + "var(PauliZ(wires=[0]))\n"
            + "sample(PauliX(wires=[1]))\n"
        )

        def template(x):
            for i in range(5):
                qml.RZ(i * x, wires=0)

            return qml.var(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        with qml.tape.OperationRecorder() as recorder:
            template(3)

        assert str(recorder) == expected_output
