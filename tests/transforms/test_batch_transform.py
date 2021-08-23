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

import pytest

import pennylane as qml
from pennylane import numpy as np


class TestBatchTransform:
    """Unit tests for the batch_transform class"""

    def test_error_invalid_callable(self):
        """Test that an error is raised if the transform
        is applied to an invalid function"""

        with pytest.raises(ValueError, match="does not appear to be a valid Python function"):
            qml.batch_transform(5)

    def test_none_processing(self):
        """Test that a transform that return None for a processing function applies
        the identity as the processing function"""

        @qml.batch_transform
        def my_transform(tape, a, b):
            """Generates two tapes, one with all RX replaced with RY,
            and the other with all RX replaced with RZ."""
            tape1 = tape.copy()
            tape2 = tape.copy()
            return [tape1, tape2], None

        a = 0.1
        b = 0.4
        x = 0.543

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.expval(qml.PauliX(0))

        tapes, fn = my_transform(tape, a, b)
        assert fn(5) == 5

    def test_parametrized_transform_tape(self):
        """Test that a parametrized transform can be applied
        to a tape"""

        @qml.batch_transform
        def my_transform(tape, a, b):
            """Generates two tapes, one with all RX replaced with RY,
            and the other with all RX replaced with RZ."""

            tape1 = qml.tape.QuantumTape()
            tape2 = qml.tape.QuantumTape()

            # loop through all operations on the input tape
            for op in tape.operations + tape.measurements:
                if op.name == "RX":
                    wires = op.wires
                    param = op.parameters[0]

                    with tape1:
                        qml.RY(a * qml.math.abs(param), wires=wires)

                    with tape2:
                        qml.RZ(b * qml.math.sin(param), wires=wires)
                else:
                    for t in [tape1, tape2]:
                        with t:
                            qml.apply(op)

            def processing_fn(results):
                return qml.math.sum(qml.math.stack(results))

            return [tape1, tape2], processing_fn

        a = 0.1
        b = 0.4
        x = 0.543

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.RX(x, wires=0)
            qml.expval(qml.PauliX(0))

        tapes, fn = my_transform(tape, a, b)

        assert len(tapes[0].operations) == 2
        assert tapes[0].operations[0].name == "Hadamard"
        assert tapes[0].operations[1].name == "RY"
        assert tapes[0].operations[1].parameters == [a * np.abs(x)]

        assert len(tapes[1].operations) == 2
        assert tapes[1].operations[0].name == "Hadamard"
        assert tapes[1].operations[1].name == "RZ"
        assert tapes[1].operations[1].parameters == [b * np.sin(x)]

    def test_parametrized_transform_qnode(self, mocker):
        """Test that a parametrized transform can be applied
        to a QNode"""

        @qml.batch_transform
        def my_transform(tape, a, b):
            """Generates two tapes, one with all RX replaced with RY,
            and the other with all RX replaced with RZ."""

            tape1 = qml.tape.QuantumTape()
            tape2 = qml.tape.QuantumTape()

            # loop through all operations on the input tape
            for op in tape.operations + tape.measurements:
                if op.name == "RX":
                    wires = op.wires
                    param = op.parameters[0]

                    with tape1:
                        qml.RY(a * qml.math.abs(param), wires=wires)

                    with tape2:
                        qml.RZ(b * qml.math.sin(param), wires=wires)
                else:
                    for t in [tape1, tape2]:
                        with t:
                            qml.apply(op)

            def processing_fn(results):
                return qml.math.sum(qml.math.stack(results))

            return [tape1, tape2], processing_fn

        a = 0.1
        b = 0.4
        x = 0.543

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliX(0))

        transform_fn = my_transform(circuit, a, b)

        spy = mocker.spy(my_transform, "construct")
        res = transform_fn(x)

        spy.assert_called()
        tapes, fn = spy.spy_return

        assert len(tapes[0].operations) == 2
        assert tapes[0].operations[0].name == "Hadamard"
        assert tapes[0].operations[1].name == "RY"
        assert tapes[0].operations[1].parameters == [a * np.abs(x)]

        assert len(tapes[1].operations) == 2
        assert tapes[1].operations[0].name == "Hadamard"
        assert tapes[1].operations[1].name == "RZ"
        assert tapes[1].operations[1].parameters == [b * np.sin(x)]

        expected = fn(dev.batch_execute(tapes))
        assert res == expected

    def test_parametrized_transform_qnode_decorator(self, mocker):
        """Test that a parametrized transform can be applied
        to a QNode as a decorator"""

        @qml.batch_transform
        def my_transform(tape, a, b):
            """Generates two tapes, one with all RX replaced with RY,
            and the other with all RX replaced with RZ."""

            tape1 = qml.tape.QuantumTape()
            tape2 = qml.tape.QuantumTape()

            # loop through all operations on the input tape
            for op in tape.operations + tape.measurements:
                if op.name == "RX":
                    wires = op.wires
                    param = op.parameters[0]

                    with tape1:
                        qml.RY(a * qml.math.abs(param), wires=wires)

                    with tape2:
                        qml.RZ(b * qml.math.sin(param), wires=wires)
                else:
                    for t in [tape1, tape2]:
                        with t:
                            qml.apply(op)

            def processing_fn(results):
                return qml.math.sum(qml.math.stack(results))

            return [tape1, tape2], processing_fn

        a = 0.1
        b = 0.4
        x = 0.543

        dev = qml.device("default.qubit", wires=2)

        @my_transform(a, b)
        @qml.qnode(dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliX(0))

        spy = mocker.spy(my_transform, "construct")
        res = circuit(x)

        spy.assert_called()
        tapes, fn = spy.spy_return

        assert len(tapes[0].operations) == 2
        assert tapes[0].operations[0].name == "Hadamard"
        assert tapes[0].operations[1].name == "RY"
        assert tapes[0].operations[1].parameters == [a * np.abs(x)]

        assert len(tapes[1].operations) == 2
        assert tapes[1].operations[0].name == "Hadamard"
        assert tapes[1].operations[1].name == "RZ"
        assert tapes[1].operations[1].parameters == [b * np.sin(x)]

        expected = fn(dev.batch_execute(tapes))
        assert res == expected
