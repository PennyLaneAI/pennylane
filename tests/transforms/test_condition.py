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

terminal_meas = [
    qml.probs(wires=[1, 0]),
    qml.expval(qml.PauliZ(0)),
    qml.expval(qml.PauliZ("a") @ qml.PauliZ(3) @ qml.PauliZ(-1)),
    qml.var(qml.PauliX("b")),
    qml.state(),
    qml.density_matrix(wires=[2, 3]),
    qml.density_matrix(wires=["b", -231]),
]


@pytest.mark.parametrize("terminal_measurement", terminal_meas)
class TestCond:
    """Tests that verify that the cond transform works as expect."""

    def test_cond_ops(self, terminal_measurement):
        """Test that qml.cond creates conditional operations as expected."""
        r = 1.234

        def f(x):
            qml.PauliX(1)
            qml.RY(x, wires=1)
            qml.PauliZ(1)

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.cond(m_0, f)(r)
            qml.apply(terminal_measurement)

        ops = tape.operations
        target_wire = qml.wires.Wires(1)

        assert len(ops) == 4
        assert ops[0].return_type == qml.measurements.MidMeasure

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

        assert len(tape.measurements) == 1
        assert tape.measurements[0] is terminal_measurement

    def tape_with_else(f, g, r, meas):
        """Tape that uses cond by passing both a true and false func."""
        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.cond(m_0, f, g)(r)
            qml.apply(meas)

        return tape

    def tape_uses_cond_twice(f, g, r, meas):
        """Tape that uses cond twice such that it's equivalent to using cond
        with two functions being passed (tape_with_else)."""
        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.cond(m_0, f)(r)
            qml.cond(~m_0, g)(r)
            qml.apply(meas)

        return tape

    @pytest.mark.parametrize("tape", [tape_with_else, tape_uses_cond_twice])
    def test_cond_operationss_with_else(self, tape, terminal_measurement):
        """Test that qml.cond operationss Conditional operations as expected in two cases:
        1. When an else qfunc is provided;
        2. When qml.cond is used twice equivalent to using an else qfunc.
        """
        r = 1.234

        def f(x):
            qml.PauliX(1)
            qml.RY(x, wires=1)
            qml.PauliZ(1)

        def g(x):
            qml.PauliY(1)

        tape = tape(f, g, r, terminal_measurement)
        ops = tape.operations
        target_wire = qml.wires.Wires(1)

        assert len(ops) == 5

        assert ops[0].return_type == qml.measurements.MidMeasure

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

        # Check that: the measurement value is the same for true_fn conditional
        # ops
        assert ops[1].meas_val is ops[2].meas_val is ops[3].meas_val

        # However, it is not the same for the false_fn
        assert ops[3].meas_val is not ops[4].meas_val

        assert len(tape.measurements) == 1
        assert tape.measurements[0] is terminal_measurement

    def test_cond_error(self, terminal_measurement):
        """Test that an error is raised when the qfunc has a measurement."""
        dev = qml.device("default.qubit", wires=3)

        def f():
            return qml.apply(terminal_measurement)

        with pytest.raises(
            ConditionalTransformError, match="contain no measurements can be applied conditionally"
        ):
            m_0 = qml.measure(1)
            qml.cond(m_0, f)()

    def test_cond_error_else(self, terminal_measurement):
        """Test that an error is raised when one of the qfuncs has a
        measurement."""
        dev = qml.device("default.qubit", wires=3)

        def f():
            qml.PauliX(0)

        def g():
            return qml.apply(terminal_measurement)

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
    def test_cond_error_unrecognized_input(self, inp, terminal_measurement):
        """Test that an error is raised when the input is not recognized."""
        dev = qml.device("default.qubit", wires=3)

        with pytest.raises(
            ConditionalTransformError,
            match="Only operations and quantum functions with no measurements",
        ):
            m_0 = qml.measure(1)
            qml.cond(m_0, inp)()


@pytest.mark.parametrize("terminal_measurement", terminal_meas)
class TestOtherTransforms:
    """Tests that qml.cond works correctly with other transforms."""

    def test_cond_operationss_with_adjoint(self, terminal_measurement):
        """Test that qml.cond operationss Conditional operations as expected with
        qml.adjoint."""
        r = 1.234

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.adjoint(qml.RX), qml.RX)(r, wires=1)
            qml.apply(terminal_measurement)

        ops = tape.operations
        target_wire = qml.wires.Wires(1)

        assert len(ops) == 3
        assert ops[0].return_type == qml.measurements.MidMeasure

        assert isinstance(ops[1], qml.transforms.condition.Conditional)
        assert isinstance(ops[1].then_op, qml.ops.op_math.Adjoint)
        assert isinstance(ops[1].then_op.base, qml.RX)
        assert ops[1].then_op.wires == target_wire

        assert isinstance(ops[2], qml.transforms.condition.Conditional)
        assert isinstance(ops[2].then_op, qml.RX)
        assert ops[2].then_op.data == [r]
        assert ops[2].then_op.wires == target_wire

        assert len(tape.measurements) == 1
        assert tape.measurements[0] is terminal_measurement

    def test_cond_operationss_with_ctrl(self, terminal_measurement):
        """Test that qml.cond operations Conditional operations as expected with
        qml.ctrl."""
        r = 1.234

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.ctrl(qml.RX, 1), qml.ctrl(qml.RY, 1))(r, wires=2)
            qml.apply(terminal_measurement)

        ops = tape.operations
        target_wire = qml.wires.Wires(2)

        assert len(ops) == 3
        assert ops[0].return_type == qml.measurements.MidMeasure

        assert isinstance(ops[1], qml.transforms.condition.Conditional)
        assert isinstance(ops[1].then_op, qml.ops.op_math.Controlled)
        assert qml.equal(ops[1].then_op.base, qml.RX(r, wires=2))

        assert isinstance(ops[2], qml.transforms.condition.Conditional)
        assert isinstance(ops[2].then_op, qml.ops.op_math.Controlled)
        assert qml.equal(ops[2].then_op.base, qml.RY(r, wires=2))

        assert len(tape.measurements) == 1
        assert tape.measurements[0] is terminal_measurement

    def test_ctrl_operations_with_cond(self, terminal_measurement):
        """Test that qml.cond operationss Conditional operations as expected with
        qml.ctrl."""
        r = 1.234

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.ctrl(qml.cond(m_0, qml.RX, qml.RY), 1)(r, wires=0)
            qml.apply(terminal_measurement)

        ops = tape.operations
        target_wire = qml.wires.Wires(2)

        assert len(ops) == 3
        assert ops[0].return_type == qml.measurements.MidMeasure

        assert isinstance(ops[1], qml.ops.op_math.Controlled)
        assert isinstance(ops[1].base, qml.transforms.condition.Conditional)
        assert qml.equal(ops[1].base.then_op, qml.RX(1.234, wires=0))

        assert isinstance(ops[2], qml.ops.op_math.Controlled)
        assert isinstance(ops[2].base, qml.transforms.condition.Conditional)
        assert qml.equal(ops[2].base.then_op, qml.RY(r, wires=0))

        assert len(tape.measurements) == 1
        assert tape.measurements[0] is terminal_measurement
