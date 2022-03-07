# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the cond transforms.

Note: integration tests use the defer_measurements transform and a QNode and
are located in the:

* tests/transforms/test_defer_measurements.py
* tests/test_qnode.py

files.
"""

import pytest

from pennylane import numpy as np

import pennylane as qml
from pennylane.transforms.condition import ConditionalTransformError


class TestCond:
    """Tests that verify that the cond transform works as expect."""

    def test_cond_queues(self):
        """Test that qml.cond queues Conditional operations as expected."""
        r = 1.234

        def f(x):
            qml.PauliX(1)
            qml.RY(x, wires=1)
            qml.PauliZ(1)

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.cond(m_0, f)(r)
            qml.probs(wires=1)

        ops = tape.queue
        target_wire = qml.wires.Wires(1)

        assert len(ops) == 5
        assert ops[0].return_type == qml.operation.MidMeasure

        assert isinstance(ops[1], qml.transforms.condition.Conditional)
        assert isinstance(ops[1].then_op, qml.PauliX)
        assert ops[1].then_op.wires == target_wire

        assert isinstance(ops[2], qml.transforms.condition.Conditional)
        assert isinstance(ops[2].then_op, qml.RY)
        assert ops[2].then_op.wires == target_wire
        assert ops[2].then_op.data == [r]

        assert isinstance(ops[3], qml.transforms.condition.Conditional)
        assert isinstance(ops[3].then_op, qml.PauliZ)
        assert ops[3].then_op.wires == target_wire

        assert ops[4].return_type == qml.operation.Probability

    def test_cond_queues_with_else(self):
        """Test that qml.cond queues Conditional operations as expected when an
        else qfunc is also provided."""
        r = 1.234

        def f(x):
            qml.PauliX(1)
            qml.RY(x, wires=1)
            qml.PauliZ(1)

        def g(x):
            qml.PauliY(1)

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.cond(m_0, f, g)(r)
            qml.probs(wires=1)

        ops = tape.queue
        target_wire = qml.wires.Wires(1)

        assert len(ops) == 6

        assert ops[0].return_type == qml.operation.MidMeasure

        assert isinstance(ops[1], qml.transforms.condition.Conditional)
        assert isinstance(ops[1].then_op, qml.PauliX)
        assert ops[1].then_op.wires == target_wire

        assert isinstance(ops[2], qml.transforms.condition.Conditional)
        assert isinstance(ops[2].then_op, qml.RY)
        assert ops[2].then_op.wires == target_wire
        assert ops[2].then_op.data == [r]

        assert isinstance(ops[3], qml.transforms.condition.Conditional)
        assert isinstance(ops[3].then_op, qml.PauliZ)
        assert ops[3].then_op.wires == target_wire

        assert isinstance(ops[4], qml.transforms.condition.Conditional)
        assert isinstance(ops[4].then_op, qml.PauliY)
        assert ops[4].then_op.wires == target_wire

        assert ops[5].return_type == qml.operation.Probability

    def test_cond_error(self):
        """Test that an error is raised when the qfunc has a measurement."""
        dev = qml.device("default.qubit", wires=3)

        def f():
            return qml.state()

        with pytest.raises(
            ConditionalTransformError, match="contain no measurements can be applied conditionally"
        ):
            m_0 = qml.measure(1)
            qml.cond(m_0, f)()

    def test_cond_error_else(self):
        """Test that an error is raised when one of the qfuncs has a
        measurement."""
        dev = qml.device("default.qubit", wires=3)

        def f():
            qml.PauliX(0)

        def g():
            return qml.state()

        with pytest.raises(
            ConditionalTransformError, match="contain no measurements can be applied conditionally"
        ):
            m_0 = qml.measure(1)
            qml.cond(m_0, f, g)()

        with pytest.raises(
            ConditionalTransformError, match="contain no measurements can be applied conditionally"
        ):
            m_0 = qml.measure(1)
            qml.cond(m_0, g, f)()  # Check that the same error is raised when f and g are swapped

    @pytest.mark.parametrize("inp", [1, "string", qml.PauliZ(0)])
    def test_cond_error_unrecognized_input(self, inp):
        """Test that an error is raised when the input is not recognized."""
        dev = qml.device("default.qubit", wires=3)

        with pytest.raises(
            ConditionalTransformError,
            match="Only operations and quantum functions with no measurements",
        ):
            m_0 = qml.measure(1)
            qml.cond(m_0, inp)()


class TestOtherTransforms:
    """Tests that qml.cond works correctly with other transforms."""

    def test_cond_queues_with_adjoint(self):
        """Test that qml.cond queues Conditional operations as expected with
        qml.adjoint."""
        r = 1.234

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.adjoint(qml.RX), qml.RX)(r, wires=1)
            qml.probs(wires=1)

        ops = tape.queue
        target_wire = qml.wires.Wires(1)

        assert len(ops) == 4
        assert ops[0].return_type == qml.operation.MidMeasure

        assert isinstance(ops[1], qml.transforms.condition.Conditional)
        assert isinstance(ops[1].then_op, qml.RX)
        assert ops[1].then_op.data == [-r]  # adjoint
        assert ops[1].then_op.wires == target_wire

        assert isinstance(ops[2], qml.transforms.condition.Conditional)
        assert isinstance(ops[2].then_op, qml.RX)
        assert ops[2].then_op.data == [r]
        assert ops[2].then_op.wires == target_wire

        assert ops[3].return_type == qml.operation.Probability

    def test_cond_queues_with_ctrl(self):
        """Test that qml.cond queues Conditional operations as expected with
        qml.ctrl."""
        r = 1.234

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.ctrl(qml.RX, 1), qml.ctrl(qml.RY, 1))(r, wires=2)
            qml.probs(wires=[1, 2])

        ops = tape.queue
        target_wire = qml.wires.Wires(2)

        assert len(ops) == 4
        assert ops[0].return_type == qml.operation.MidMeasure

        assert isinstance(ops[1], qml.transforms.condition.Conditional)
        assert isinstance(ops[1].then_op, qml.transforms.control.ControlledOperation)

        assert len(ops[1].then_op.subtape.queue) == 1
        controlled_op = ops[1].then_op.subtape.queue[0]
        assert isinstance(controlled_op, qml.RX)
        assert controlled_op.data == [r]
        assert controlled_op.wires == target_wire

        assert isinstance(ops[2], qml.transforms.condition.Conditional)
        assert isinstance(ops[2].then_op, qml.transforms.control.ControlledOperation)

        assert len(ops[2].then_op.subtape.queue) == 1
        controlled_op = ops[2].then_op.subtape.queue[0]
        assert isinstance(controlled_op, qml.RY)
        assert controlled_op.data == [r]
        assert controlled_op.wires == target_wire

        assert ops[3].return_type == qml.operation.Probability

    def test_ctrl_queues_with_cond(self):
        """Test that qml.cond queues Conditional operations as expected with
        qml.ctrl."""
        r = 1.234

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.ctrl(qml.cond(m_0, qml.RX, qml.RY), 1)(r, wires=0)
            qml.probs(wires=[1, 2])

        ops = tape.queue
        target_wire = qml.wires.Wires(2)

        assert len(ops) == 3
        assert ops[0].return_type == qml.operation.MidMeasure

        assert isinstance(ops[1], qml.transforms.control.ControlledOperation)
        assert len(ops[1].subtape.queue) == 2

        op1 = ops[1].subtape.queue[0]
        assert isinstance(op1, qml.transforms.condition.Conditional)
        assert isinstance(op1.then_op, qml.RX)
        assert op1.then_op.data == [r]

        op2 = ops[1].subtape.queue[1]
        assert isinstance(op2, qml.transforms.condition.Conditional)
        assert isinstance(op2.then_op, qml.RY)
        assert op1.then_op.data == [r]

        assert ops[2].return_type == qml.operation.Probability
