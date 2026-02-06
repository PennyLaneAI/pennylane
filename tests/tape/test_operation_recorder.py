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
Unit tests for the ``OperationRecorder`` in PennyLane.
"""
import pennylane as qp


class TestOperationRecorder:
    """Test the OperationRecorder class."""

    def test_circuit_integration(self):
        """Tests that the OperationRecorder integrates well with the
        core behaviour of PennyLane."""
        expected_output = (
            "Operations\n"
            + "==========\n"
            + "Y(0)\n"
            + "Y(1)\n"
            + "RZ(0.4, wires=[0])\n"
            + "RZ(0.4, wires=[1])\n"
            + "CNOT(wires=[0, 1])\n"
            + "\n"
            + "Observables\n"
            + "===========\n"
        )

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def circuit(a, b, c):
            qp.RX(a, wires=0)
            qp.RY(b, wires=1)

            with qp.tape.OperationRecorder() as recorder:
                ops = [
                    qp.PauliY(0),
                    qp.PauliY(1),
                    qp.RZ(c, wires=0),
                    qp.RZ(c, wires=1),
                    qp.CNOT(wires=[0, 1]),
                ]

            assert str(recorder) == expected_output
            assert recorder.queue == ops

            return qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliZ(1))

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
                qp.RZ(i * x, wires=0)

        with qp.tape.OperationRecorder() as recorder:
            template(3)

        assert str(recorder) == expected_output
        qp.assert_equal(recorder[0], qp.RZ(0, wires=0))

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
            + "Z(0)\n"
            + "X(1)\n"
        )

        def template(x):
            for i in range(5):
                qp.RZ(i * x, wires=0)

            return qp.var(qp.PauliZ(0)), qp.sample(qp.PauliX(1))

        with qp.tape.OperationRecorder() as recorder:
            template(3)

        assert str(recorder) == expected_output
