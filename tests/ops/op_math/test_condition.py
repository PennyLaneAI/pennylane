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

import numpy as np
import pytest

import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops.op_math.condition import Conditional, ConditionalTransformError

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
    """Tests that the cond transform works as expect."""

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
        assert isinstance(ops[1].base, qml.PauliX)
        assert ops[1].base.wires == target_wire

        assert isinstance(ops[2], qml.ops.Conditional)
        assert isinstance(ops[2].base, qml.RY)
        assert ops[2].base.wires == target_wire
        assert ops[2].base.data == (r,)

        assert isinstance(ops[3], qml.ops.Conditional)
        assert isinstance(ops[3].base, qml.PauliZ)
        assert ops[3].base.wires == target_wire

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
        assert isinstance(ops[1].base, qml.PauliX)
        assert ops[1].base.wires == target_wire

        assert isinstance(ops[2], qml.ops.Conditional)
        assert isinstance(ops[2].base, qml.RY)
        assert ops[2].base.wires == target_wire
        assert ops[2].base.data == (r,)

        assert isinstance(ops[3], qml.ops.Conditional)
        assert isinstance(ops[3].base, qml.PauliZ)
        assert ops[3].base.wires == target_wire

        assert isinstance(ops[4], qml.ops.Conditional)
        assert isinstance(ops[4].base, qml.PauliY)
        assert ops[4].base.wires == target_wire

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
        assert mapped_cond.base == qml.PauliX("b")

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
        assert isinstance(ops[1].base, qml.ops.op_math.Adjoint)
        assert isinstance(ops[1].base.base, qml.RX)
        assert ops[1].base.wires == target_wire

        assert isinstance(ops[2], qml.ops.Conditional)
        assert isinstance(ops[2].base, qml.RX)
        assert ops[2].base.data == (r,)
        assert ops[2].base.wires == target_wire

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
        assert isinstance(ops[1].base, qml.ops.op_math.Controlled)
        assert qml.equal(ops[1].base.base, qml.RX(r, wires=2))

        assert isinstance(ops[2], qml.ops.Conditional)
        assert isinstance(ops[2].base, qml.ops.op_math.Controlled)
        assert qml.equal(ops[2].base.base, qml.RY(r, wires=2))

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
        assert qml.equal(ops[1].base.base, qml.RX(1.234, wires=0))

        assert isinstance(ops[2], qml.ops.op_math.Controlled)
        assert isinstance(ops[2].base, qml.ops.Conditional)
        assert qml.equal(ops[2].base.base, qml.RY(r, wires=0))

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


class TestProperties:
    """Test Conditional properties."""

    BASE_OP = [qml.RX(1.23, 0), qml.Rot(1.2, 2.3, 3.4, 0), qml.QubitUnitary([[0, 1], [1, 0]], 0)]

    def test_data(self):
        """Test base data can be get and set through Conditional class."""
        x = np.array(1.234)
        m = qml.measure("a")
        base = qml.RX(x, wires="a")
        cond_op = Conditional(m, base)

        assert cond_op.data == (x,)

        # update parameters through Conditional
        x_new = np.array(2.3456)
        cond_op.data = (x_new,)
        assert base.data == (x_new,)
        assert cond_op.data == (x_new,)

        # update base data updates Conditional data
        x_new2 = np.array(3.456)
        base.data = (x_new2,)
        assert cond_op.data == (x_new2,)

    @pytest.mark.parametrize("value", (True, False))
    def test_has_matrix(self, value):
        """Test that Conditional defers has_matrix to base operator."""

        # pylint:disable=too-few-public-methods
        class DummyOp(Operator):
            num_wires = 1
            has_matrix = value

        m = qml.measure(0)
        cond_op = Conditional(m, DummyOp(1))

        assert cond_op.has_matrix is value

    @pytest.mark.parametrize("value", (True, False))
    def test_has_adjoint(self, value):
        """Test that Conditional defers has_adjoint to base operator."""

        # pylint:disable=too-few-public-methods
        class DummyOp(Operator):
            num_wires = 1
            has_adjoint = value

        m = qml.measure(0)
        cond_op = Conditional(m, DummyOp(1))

        assert cond_op.has_adjoint is value

    @pytest.mark.parametrize("value", (True, False))
    def test_has_diagonalizing_gates(self, value):
        """Test that Conditional defers has_adjoint to base operator."""

        # pylint:disable=too-few-public-methods
        class DummyOp(Operator):
            num_wires = 1
            has_diagonalizing_gates = value

        m = qml.measure(0)
        cond_op = Conditional(m, DummyOp(1))

        assert cond_op.has_diagonalizing_gates is value

    @pytest.mark.parametrize("base", BASE_OP)
    def test_ndim_params(self, base):
        """Test that Conditional defers to base ndim_params."""
        m = qml.measure(0)
        op = Conditional(m, base)
        assert op.ndim_params == base.ndim_params

    @pytest.mark.parametrize("base", BASE_OP)
    def test_num_params(self, base):
        """Test that Conditional defers to base num_params."""
        m = qml.measure(0)
        op = Conditional(m, base)
        assert op.num_params == base.num_params


class TestMethods:
    """Test Conditional methods."""

    def test_diagonalizing_gates(self):
        """Test that Conditional defers to base diagonalizing_gates."""
        base = qml.PauliX(0)
        m = qml.measure(0)
        op = Conditional(m, base)

        assert op.diagonalizing_gates() == base.diagonalizing_gates()

    def test_eigvals(self):
        """Test that Conditional defers to base eigvals."""
        base = qml.PauliX(0)
        m = qml.measure(0)
        op = Conditional(m, base)

        assert qml.math.allclose(op.eigvals(), base.eigvals())

    def test_matrix_value(self):
        """Test that Conditional defers to base matrix."""
        base = qml.PauliX(0)
        m = qml.measure(0)
        op = Conditional(m, base)
        assert qml.math.allclose(op.matrix(), op.base.matrix())

    def test_matrix_wire_oder(self):
        """Test that `wire_order` in `matrix` method behaves as expected."""
        m = qml.measure(0)
        base = qml.RX(-4.432, wires=1)
        op = Conditional(m, base)

        method_order = op.matrix(wire_order=(1, 0))
        function_order = qml.math.expand_matrix(op.matrix(), op.wires, (1, 0))

        assert qml.math.allclose(method_order, function_order)

    def test_adjoint(self):
        """Test adjoint method for Conditional."""
        base = qml.RX(np.pi / 2, 0)
        m = qml.measure(0)
        op = Conditional(m, base)
        adj_op = op.adjoint()

        assert isinstance(adj_op, Conditional)
        assert adj_op.meas_val == op.meas_val
        assert adj_op.base == base.adjoint()
