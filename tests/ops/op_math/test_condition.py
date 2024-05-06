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

import pennylane as qml
from pennylane.ops.op_math.condition import ConditionalTransformError

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

        with qml.queuing.AnnotatedQueue() as q:
            m_0 = qml.measure(0)
            qml.cond(m_0, f)(r)
            qml.apply(terminal_measurement)

        tape = qml.tape.QuantumScript.from_queue(q)
        ops = tape.operations
        target_wire = qml.wires.Wires(1)

        assert len(ops) == 4
        assert ops[0].return_type == qml.measurements.MidMeasure

        assert isinstance(ops[1], qml.ops.Conditional)
        assert isinstance(ops[1].then_op, qml.PauliX)
        assert ops[1].then_op.wires == target_wire

        assert isinstance(ops[2], qml.ops.Conditional)
        assert isinstance(ops[2].then_op, qml.RY)
        assert ops[2].then_op.wires == target_wire
        assert ops[2].then_op.data == (r,)

        assert isinstance(ops[3], qml.ops.Conditional)
        assert isinstance(ops[3].then_op, qml.PauliZ)
        assert ops[3].then_op.wires == target_wire

        assert len(tape.measurements) == 1
        assert tape.measurements[0] is terminal_measurement

    @staticmethod
    def tape_with_else(f, g, r, meas):
        """Tape that uses cond by passing both a true and false func."""
        with qml.queuing.AnnotatedQueue() as q:
            m_0 = qml.measure(0)
            qml.cond(m_0, f, g)(r)
            qml.apply(meas)

        tape = qml.tape.QuantumScript.from_queue(q)
        return tape

    @staticmethod
    def tape_uses_cond_twice(f, g, r, meas):
        """Tape that uses cond twice such that it's equivalent to using cond
        with two functions being passed (tape_with_else)."""
        with qml.queuing.AnnotatedQueue() as q:
            m_0 = qml.measure(0)
            qml.cond(m_0, f)(r)
            qml.cond(~m_0, g)(r)
            qml.apply(meas)

        tape = qml.tape.QuantumScript.from_queue(q)
        return tape

    @pytest.mark.parametrize("tape", ["tape_with_else", "tape_uses_cond_twice"])
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
            # pylint: disable=unused-argument
            qml.PauliY(1)

        tape = getattr(self, tape)(f, g, r, terminal_measurement)
        ops = tape.operations
        target_wire = qml.wires.Wires(1)

        assert len(ops) == 5

        assert ops[0].return_type == qml.measurements.MidMeasure

        assert isinstance(ops[1], qml.ops.Conditional)
        assert isinstance(ops[1].then_op, qml.PauliX)
        assert ops[1].then_op.wires == target_wire

        assert isinstance(ops[2], qml.ops.Conditional)
        assert isinstance(ops[2].then_op, qml.RY)
        assert ops[2].then_op.wires == target_wire
        assert ops[2].then_op.data == (r,)

        assert isinstance(ops[3], qml.ops.Conditional)
        assert isinstance(ops[3].then_op, qml.PauliZ)
        assert ops[3].then_op.wires == target_wire

        assert isinstance(ops[4], qml.ops.Conditional)
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


class TestAdditionalCond:
    """Test additional/misc functionality relating to qml.cond"""

    @pytest.mark.parametrize("inp", [1, "string", qml.PauliZ(0)])
    def test_cond_error_unrecognized_input(self, inp):
        """Test that an error is raised when the input is not recognized."""

        with pytest.raises(
            ConditionalTransformError,
            match="Only operations and quantum functions with no measurements",
        ):
            m_0 = qml.measure(1)
            qml.cond(m_0, inp)()

    def test_map_wires(self):
        """Tests the cond.map_wires function."""
        with qml.queuing.AnnotatedQueue() as q:
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.PauliX)(1)

        meas, cond_op = q.queue
        mapped_cond = cond_op.map_wires({0: "a", 1: "b"})
        assert mapped_cond.meas_val.measurements == [meas.map_wires({0: "a"})]
        assert mapped_cond.then_op == qml.PauliX("b")

    @pytest.mark.parametrize(
        "op", [qml.RX(0.123, 0), qml.RX([0.123, 4.56], 0), qml.Rot(1.23, 4.56, 7.89, 0)]
    )
    def test_data_set_correctly(self, op):
        """Test that Conditional.data is the same as the data of the conditioned op."""
        cond_op = qml.ops.Conditional(qml.measure(0), op)
        assert cond_op.data == op.data
        assert cond_op.batch_size == op.batch_size
        assert cond_op.num_params == op.num_params
        assert cond_op.ndim_params == op.ndim_params


@pytest.mark.parametrize("op_class", [qml.PauliY, qml.Toffoli, qml.Hadamard, qml.CZ])
def test_conditional_label(op_class):
    """Test that the label for conditional oeprators is correct."""
    base_op = op_class(wires=range(op_class.num_wires))

    # Need to use queue because `qml.cond` doesn't return `Conditional` operators
    with qml.queuing.AnnotatedQueue() as q:
        m0 = qml.measure(0)
        qml.cond(m0, op_class)(wires=range(op_class.num_wires))

    cond_op = q.queue[1]

    assert base_op.label() == cond_op.label()


@pytest.mark.parametrize("terminal_measurement", terminal_meas)
class TestOtherTransforms:
    """Tests that qml.cond works correctly with other transforms."""

    def test_cond_operationss_with_adjoint(self, terminal_measurement):
        """Test that qml.cond operationss Conditional operations as expected with
        qml.adjoint."""
        r = 1.234

        with qml.queuing.AnnotatedQueue() as q:
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.adjoint(qml.RX), qml.RX)(r, wires=1)
            qml.apply(terminal_measurement)

        tape = qml.tape.QuantumScript.from_queue(q)
        ops = tape.operations
        target_wire = qml.wires.Wires(1)

        assert len(ops) == 3
        assert ops[0].return_type == qml.measurements.MidMeasure

        assert isinstance(ops[1], qml.ops.Conditional)
        assert isinstance(ops[1].then_op, qml.ops.op_math.Adjoint)
        assert isinstance(ops[1].then_op.base, qml.RX)
        assert ops[1].then_op.wires == target_wire

        assert isinstance(ops[2], qml.ops.Conditional)
        assert isinstance(ops[2].then_op, qml.RX)
        assert ops[2].then_op.data == (r,)
        assert ops[2].then_op.wires == target_wire

        assert len(tape.measurements) == 1
        assert tape.measurements[0] is terminal_measurement

    def test_cond_operationss_with_ctrl(self, terminal_measurement):
        """Test that qml.cond operations Conditional operations as expected with
        qml.ctrl."""
        r = 1.234

        with qml.queuing.AnnotatedQueue() as q:
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.ctrl(qml.RX, 1), qml.ctrl(qml.RY, 1))(r, wires=2)
            qml.apply(terminal_measurement)

        tape = qml.tape.QuantumScript.from_queue(q)
        ops = tape.operations

        assert len(ops) == 3
        assert ops[0].return_type == qml.measurements.MidMeasure

        assert isinstance(ops[1], qml.ops.Conditional)
        assert isinstance(ops[1].then_op, qml.ops.op_math.Controlled)
        assert qml.equal(ops[1].then_op.base, qml.RX(r, wires=2))

        assert isinstance(ops[2], qml.ops.Conditional)
        assert isinstance(ops[2].then_op, qml.ops.op_math.Controlled)
        assert qml.equal(ops[2].then_op.base, qml.RY(r, wires=2))

        assert len(tape.measurements) == 1
        assert tape.measurements[0] is terminal_measurement

    def test_ctrl_operations_with_cond(self, terminal_measurement):
        """Test that qml.cond operationss Conditional operations as expected with
        qml.ctrl."""
        r = 1.234

        with qml.queuing.AnnotatedQueue() as q:
            m_0 = qml.measure(0)
            qml.ctrl(qml.cond(m_0, qml.RX, qml.RY), 1)(r, wires=0)
            qml.apply(terminal_measurement)

        tape = qml.tape.QuantumScript.from_queue(q)
        ops = tape.operations

        assert len(ops) == 3
        assert ops[0].return_type == qml.measurements.MidMeasure

        assert isinstance(ops[1], qml.ops.op_math.Controlled)
        assert isinstance(ops[1].base, qml.ops.Conditional)
        assert qml.equal(ops[1].base.then_op, qml.RX(1.234, wires=0))

        assert isinstance(ops[2], qml.ops.op_math.Controlled)
        assert isinstance(ops[2].base, qml.ops.Conditional)
        assert qml.equal(ops[2].base.then_op, qml.RY(r, wires=0))

        assert len(tape.measurements) == 1
        assert tape.measurements[0] is terminal_measurement

    @pytest.mark.parametrize(
        "op_fn, fn_additional_args",
        [
            (qml.adjoint, ()),
            (qml.ctrl, ([1, 2],)),
            (qml.simplify, ()),
            (qml.evolve, (1.5,)),
            (qml.exp, (1.5,)),
            (qml.pow, (3,)),
            (qml.prod, (qml.prod(qml.PauliX(1), qml.PauliZ(1)),)),
        ],
    )
    def test_ops_as_args(self, op_fn, fn_additional_args, terminal_measurement):
        """Test that operations given are arguments to a conditioned function are not queued."""

        # Need to construct now so that id is not random
        mp = qml.measurements.MidMeasureMP(0, id="foo")
        mv = qml.measurements.MeasurementValue([mp], lambda v: v)

        def circuit():
            qml.Hadamard(0)
            qml.apply(mp)
            qml.cond(mv, op_fn)(qml.T(0), *fn_additional_args)
            return qml.apply(terminal_measurement)

        tape = qml.tape.make_qscript(circuit)()
        assert len(tape) == 4
        assert tape[0] == qml.Hadamard(0)
        assert tape[1] == mp
        assert isinstance(tape[2], qml.ops.Conditional)
        assert tape[3] == terminal_measurement
