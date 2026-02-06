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

import pennylane as qp
from pennylane.exceptions import ConditionalTransformError
from pennylane.operation import Operator
from pennylane.ops.op_math.condition import Conditional

terminal_meas = [
    qp.probs(wires=[1, 0]),
    qp.expval(qp.PauliZ(0)),
    qp.expval(qp.PauliZ("a") @ qp.PauliZ(3) @ qp.PauliZ(-1)),
    qp.var(qp.PauliX("b")),
    qp.state(),
    qp.density_matrix(wires=[2, 3]),
    qp.density_matrix(wires=["b", -231]),
]


@pytest.mark.parametrize("terminal_measurement", terminal_meas)
class TestCond:
    """Tests that the cond transform works as expect."""

    def test_cond_ops(self, terminal_measurement):
        """Test that qp.cond creates conditional operations as expected."""
        r = 1.234

        def f(x):
            qp.PauliX(1)
            qp.RY(x, wires=1)
            qp.PauliZ(1)

        with qp.queuing.AnnotatedQueue() as q:
            m_0 = qp.measure(0)
            qp.cond(m_0, f)(r)
            qp.apply(terminal_measurement)

        tape = qp.tape.QuantumScript.from_queue(q)
        ops = tape.operations
        target_wire = qp.wires.Wires(1)

        assert len(ops) == 4
        assert isinstance(ops[0], qp.ops.MidMeasure)

        assert isinstance(ops[1], qp.ops.Conditional)
        assert isinstance(ops[1].base, qp.PauliX)
        assert ops[1].base.wires == target_wire

        assert isinstance(ops[2], qp.ops.Conditional)
        assert isinstance(ops[2].base, qp.RY)
        assert ops[2].base.wires == target_wire
        assert ops[2].base.data == (r,)

        assert isinstance(ops[3], qp.ops.Conditional)
        assert isinstance(ops[3].base, qp.PauliZ)
        assert ops[3].base.wires == target_wire

        assert len(tape.measurements) == 1
        assert tape.measurements[0] == terminal_measurement

    @staticmethod
    def tape_with_else(f, g, r, meas):
        """Tape that uses cond by passing both a true and false func."""
        with qp.queuing.AnnotatedQueue() as q:
            m_0 = qp.measure(0)
            qp.cond(m_0, f, g)(r)
            qp.apply(meas)

        tape = qp.tape.QuantumScript.from_queue(q)
        return tape

    @staticmethod
    def tape_uses_cond_twice(f, g, r, meas):
        """Tape that uses cond twice such that it's equivalent to using cond
        with two functions being passed (tape_with_else)."""
        with qp.queuing.AnnotatedQueue() as q:
            m_0 = qp.measure(0)
            qp.cond(m_0, f)(r)
            qp.cond(~m_0, g)(r)
            qp.apply(meas)

        tape = qp.tape.QuantumScript.from_queue(q)
        return tape

    @pytest.mark.parametrize("tape", ["tape_with_else", "tape_uses_cond_twice"])
    def test_cond_operationss_with_else(self, tape, terminal_measurement):
        """Test that qp.cond operationss Conditional operations as expected in two cases:
        1. When an else qfunc is provided;
        2. When qp.cond is used twice equivalent to using an else qfunc.
        """
        r = 1.234

        def f(x):
            qp.PauliX(1)
            qp.RY(x, wires=1)
            qp.PauliZ(1)

        def g(x):
            # pylint: disable=unused-argument
            qp.PauliY(1)

        tape = getattr(self, tape)(f, g, r, terminal_measurement)
        ops = tape.operations
        target_wire = qp.wires.Wires(1)

        assert len(ops) == 5

        assert isinstance(ops[0], qp.ops.MidMeasure)

        assert isinstance(ops[1], qp.ops.Conditional)
        assert isinstance(ops[1].base, qp.PauliX)
        assert ops[1].base.wires == target_wire

        assert isinstance(ops[2], qp.ops.Conditional)
        assert isinstance(ops[2].base, qp.RY)
        assert ops[2].base.wires == target_wire
        assert ops[2].base.data == (r,)

        assert isinstance(ops[3], qp.ops.Conditional)
        assert isinstance(ops[3].base, qp.PauliZ)
        assert ops[3].base.wires == target_wire

        assert isinstance(ops[4], qp.ops.Conditional)
        assert isinstance(ops[4].base, qp.PauliY)
        assert ops[4].base.wires == target_wire

        # Check that: the measurement value is the same for true_fn conditional
        # ops
        assert ops[1].meas_val is ops[2].meas_val is ops[3].meas_val

        # However, it is not the same for the false_fn
        assert ops[3].meas_val is not ops[4].meas_val

        assert len(tape.measurements) == 1
        assert tape.measurements[0] == terminal_measurement

    def test_cond_error(self, terminal_measurement):
        """Test that an error is raised when the qfunc has a measurement."""

        def f():
            return qp.apply(terminal_measurement)

        with pytest.raises(
            ConditionalTransformError, match="contain no measurements can be applied conditionally"
        ):
            m_0 = qp.measure(1)
            qp.cond(m_0, f)()

    def test_cond_error_else(self, terminal_measurement):
        """Test that an error is raised when one of the qfuncs has a
        measurement."""

        def f():
            qp.PauliX(0)

        def g():
            return qp.apply(terminal_measurement)

        with pytest.raises(
            ConditionalTransformError, match="contain no measurements can be applied conditionally"
        ):
            m_0 = qp.measure(1)
            qp.cond(m_0, f, g)()

        with pytest.raises(
            ConditionalTransformError, match="contain no measurements can be applied conditionally"
        ):
            m_0 = qp.measure(1)
            qp.cond(m_0, g, f)()  # Check that the same error is raised when f and g are swapped


class TestAdditionalCond:
    """Test additional/misc functionality relating to qp.cond"""

    @pytest.mark.parametrize("inp", [1, "string", qp.PauliZ(0)])
    def test_cond_error_unrecognized_input(self, inp):
        """Test that an error is raised when the input is not recognized."""

        with pytest.raises(
            ConditionalTransformError,
            match="Only operations and quantum functions with no measurements",
        ):
            m_0 = qp.measure(1)
            qp.cond(m_0, inp)()

    def test_cond_error_for_mcms(self):
        """Test that an error is raised if a mid-circuit measurement is applied inside
        a Conditional"""

        # raises error in true_fn
        with pytest.raises(
            ConditionalTransformError,
            match="Only quantum functions that contain no measurements can be applied conditionally.",
        ):
            m_0 = qp.measure(1)
            qp.cond(m_0, qp.measure)(0)

        # raises error in false_fn
        with pytest.raises(
            ConditionalTransformError,
            match="Only quantum functions that contain no measurements can be applied conditionally.",
        ):
            m_0 = qp.measure(1)
            qp.cond(m_0, qp.X, qp.measure)(0)

    def test_cond_error_for_ppms(self):
        """Test that an error is raised if a pauli-product measurement is applied inside
        a Conditional"""

        with pytest.raises(
            ConditionalTransformError,
            match="Only quantum functions that contain no measurements can be applied conditionally.",
        ):
            m_0 = qp.measure(1)
            qp.cond(m_0, qp.pauli_measure)("X", wires=[0])

    def test_map_wires(self):
        """Tests the cond.map_wires function."""
        with qp.queuing.AnnotatedQueue() as q:
            m_0 = qp.measure(0)
            qp.cond(m_0, qp.PauliX)(1)

        meas, cond_op = q.queue
        mapped_cond = cond_op.map_wires({0: "a", 1: "b"})
        assert mapped_cond.meas_val.measurements == [meas.map_wires({0: "a"})]
        assert mapped_cond.base == qp.PauliX("b")

    @pytest.mark.parametrize(
        "op", [qp.RX(0.123, 0), qp.RX([0.123, 4.56], 0), qp.Rot(1.23, 4.56, 7.89, 0)]
    )
    def test_data_set_correctly(self, op):
        """Test that Conditional.data is the same as the data of the conditioned op."""
        cond_op = qp.ops.Conditional(qp.measure(0), op)
        assert cond_op.data == op.data
        assert cond_op.batch_size == op.batch_size
        assert cond_op.num_params == op.num_params
        assert cond_op.ndim_params == op.ndim_params

    def test_qfunc_arg_dequeued(self):
        """Tests that the operators in the quantum function arguments are dequeued."""

        def true_fn(op):
            qp.apply(op)

        def false_fn(op):
            qp.apply(op)

        def circuit(x):
            qp.cond(x > 0, true_fn, false_fn)(qp.X(0))

        with qp.queuing.AnnotatedQueue() as q:
            circuit(1)
            circuit(-1)

        assert len(q.queue) == 2
        assert q.queue == [qp.X(0), qp.X(0)]


@pytest.mark.parametrize("op_class", [qp.PauliY, qp.Toffoli, qp.Hadamard, qp.CZ])
def test_conditional_label(op_class):
    """Test that the label for conditional oeprators is correct."""
    base_op = op_class(wires=range(op_class.num_wires))

    # Need to use queue because `qp.cond` doesn't return `Conditional` operators
    with qp.queuing.AnnotatedQueue() as q:
        m0 = qp.measure(0)
        qp.cond(m0, op_class)(wires=range(op_class.num_wires))

    cond_op = q.queue[1]

    assert base_op.label() == cond_op.label()


@pytest.mark.parametrize("terminal_measurement", terminal_meas)
class TestOtherTransforms:
    """Tests that qp.cond works correctly with other transforms."""

    def test_cond_operationss_with_adjoint(self, terminal_measurement):
        """Test that qp.cond operationss Conditional operations as expected with
        qp.adjoint."""
        r = 1.234

        with qp.queuing.AnnotatedQueue() as q:
            m_0 = qp.measure(0)
            qp.cond(m_0, qp.adjoint(qp.RX), qp.RX)(r, wires=1)
            qp.apply(terminal_measurement)

        tape = qp.tape.QuantumScript.from_queue(q)
        ops = tape.operations
        target_wire = qp.wires.Wires(1)

        assert len(ops) == 3
        assert isinstance(ops[0], qp.ops.MidMeasure)

        assert isinstance(ops[1], qp.ops.Conditional)
        assert isinstance(ops[1].base, qp.ops.op_math.Adjoint)
        assert isinstance(ops[1].base.base, qp.RX)
        assert ops[1].base.wires == target_wire

        assert isinstance(ops[2], qp.ops.Conditional)
        assert isinstance(ops[2].base, qp.RX)
        assert ops[2].base.data == (r,)
        assert ops[2].base.wires == target_wire

        assert len(tape.measurements) == 1
        assert tape.measurements[0] == terminal_measurement

    def test_cond_operations_with_ctrl(self, terminal_measurement):
        """Test that qp.cond operations Conditional operations as expected with
        qp.ctrl."""
        r = 1.234

        with qp.queuing.AnnotatedQueue() as q:
            m_0 = qp.measure(0)
            qp.cond(m_0, qp.ctrl(qp.RX, 1), qp.ctrl(qp.RY, 1))(r, wires=2)
            qp.apply(terminal_measurement)

        tape = qp.tape.QuantumScript.from_queue(q)
        ops = tape.operations

        assert len(ops) == 3
        assert isinstance(ops[0], qp.ops.MidMeasure)

        assert isinstance(ops[1], qp.ops.Conditional)
        assert isinstance(ops[1].base, qp.ops.op_math.Controlled)
        qp.assert_equal(ops[1].base.base, qp.RX(r, wires=2))

        assert isinstance(ops[2], qp.ops.Conditional)
        assert isinstance(ops[2].base, qp.ops.op_math.Controlled)
        qp.assert_equal(ops[2].base.base, qp.RY(r, wires=2))

        assert len(tape.measurements) == 1
        assert tape.measurements[0] == terminal_measurement

    def test_ctrl_operations_with_cond(self, terminal_measurement):
        """Test that qp.cond operationss Conditional operations as expected with
        qp.ctrl."""
        r = 1.234

        with qp.queuing.AnnotatedQueue() as q:
            m_0 = qp.measure(0)
            qp.ctrl(qp.cond(m_0, qp.RX, qp.RY), 1)(r, wires=0)
            qp.apply(terminal_measurement)

        tape = qp.tape.QuantumScript.from_queue(q)
        ops = tape.operations

        assert len(ops) == 3
        assert isinstance(ops[0], qp.ops.MidMeasure)

        assert isinstance(ops[1], qp.ops.op_math.Controlled)
        assert isinstance(ops[1].base, qp.ops.Conditional)
        qp.assert_equal(ops[1].base.base, qp.RX(1.234, wires=0))

        assert isinstance(ops[2], qp.ops.op_math.Controlled)
        assert isinstance(ops[2].base, qp.ops.Conditional)
        qp.assert_equal(ops[2].base.base, qp.RY(r, wires=0))

        assert len(tape.measurements) == 1
        assert tape.measurements[0] == terminal_measurement

    @pytest.mark.parametrize(
        "op_fn, fn_additional_args",
        [
            (qp.adjoint, ()),
            (qp.ctrl, ([1, 2],)),
            (qp.simplify, ()),
            (qp.evolve, (1.5,)),
            (qp.exp, (1.5,)),
            (qp.pow, (3,)),
            (qp.prod, (qp.prod(qp.PauliX(1), qp.PauliZ(1)),)),
        ],
    )
    def test_ops_as_args(self, op_fn, fn_additional_args, terminal_measurement):
        """Test that operations given are arguments to a conditioned function are not queued."""

        # Need to construct now so that id is not random
        mp = qp.ops.MidMeasure(0, id="foo")
        mv = qp.ops.MeasurementValue([mp], lambda v: v)

        def circuit():
            qp.Hadamard(0)
            qp.apply(mp)
            qp.cond(mv, op_fn)(qp.T(0), *fn_additional_args)
            return qp.apply(terminal_measurement)

        tape = qp.tape.make_qscript(circuit)()
        assert len(tape) == 4
        assert tape[0] == qp.Hadamard(0)
        assert tape[1] == mp
        assert isinstance(tape[2], qp.ops.Conditional)
        assert tape[3] == terminal_measurement


class TestProperties:
    """Test Conditional properties."""

    BASE_OP = [qp.RX(1.23, 0), qp.Rot(1.2, 2.3, 3.4, 0), qp.QubitUnitary([[0, 1], [1, 0]], 0)]

    def test_data(self):
        """Test base data can be get and set through Conditional class."""
        x = np.array(1.234)
        m = qp.measure("a")
        base = qp.RX(x, wires="a")
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

        m = qp.measure(0)
        cond_op = Conditional(m, DummyOp(1))

        assert cond_op.has_matrix is value

    @pytest.mark.parametrize("value", (True, False))
    def test_has_adjoint(self, value):
        """Test that Conditional defers has_adjoint to base operator."""

        # pylint:disable=too-few-public-methods
        class DummyOp(Operator):
            num_wires = 1
            has_adjoint = value

        m = qp.measure(0)
        cond_op = Conditional(m, DummyOp(1))

        assert cond_op.has_adjoint is value

    @pytest.mark.parametrize("value", (True, False))
    def test_has_diagonalizing_gates(self, value):
        """Test that Conditional defers has_adjoint to base operator."""

        # pylint:disable=too-few-public-methods
        class DummyOp(Operator):
            num_wires = 1
            has_diagonalizing_gates = value

        m = qp.measure(0)
        cond_op = Conditional(m, DummyOp(1))

        assert cond_op.has_diagonalizing_gates is value

    @pytest.mark.parametrize("base", BASE_OP)
    def test_ndim_params(self, base):
        """Test that Conditional defers to base ndim_params."""
        m = qp.measure(0)
        op = Conditional(m, base)
        assert op.ndim_params == base.ndim_params

    @pytest.mark.parametrize("base", BASE_OP)
    def test_num_params(self, base):
        """Test that Conditional defers to base num_params."""
        m = qp.measure(0)
        op = Conditional(m, base)
        assert op.num_params == base.num_params


class TestMethods:
    """Test Conditional methods."""

    def test_diagonalizing_gates(self):
        """Test that Conditional defers to base diagonalizing_gates."""
        base = qp.PauliX(0)
        m = qp.measure(0)
        op = Conditional(m, base)

        assert op.diagonalizing_gates() == base.diagonalizing_gates()

    def test_eigvals(self):
        """Test that Conditional defers to base eigvals."""
        base = qp.PauliX(0)
        m = qp.measure(0)
        op = Conditional(m, base)

        assert qp.math.allclose(op.eigvals(), base.eigvals())

    def test_matrix_value(self):
        """Test that Conditional defers to base matrix."""
        base = qp.PauliX(0)
        m = qp.measure(0)
        op = Conditional(m, base)
        assert qp.math.allclose(op.matrix(), op.base.matrix())

    def test_matrix_wire_oder(self):
        """Test that `wire_order` in `matrix` method behaves as expected."""
        m = qp.measure(0)
        base = qp.RX(-4.432, wires=1)
        op = Conditional(m, base)

        method_order = op.matrix(wire_order=(1, 0))
        function_order = qp.math.expand_matrix(op.matrix(), op.wires, (1, 0))

        assert qp.math.allclose(method_order, function_order)

    def test_adjoint(self):
        """Test adjoint method for Conditional."""
        base = qp.RX(np.pi / 2, 0)
        m = qp.measure(0)
        op = Conditional(m, base)
        adj_op = op.adjoint()

        assert isinstance(adj_op, Conditional)
        assert adj_op.meas_val is op.meas_val
        assert adj_op.base == base.adjoint()


class TestPythonFallback:
    """Test python fallback"""

    def test_simple_if(self):
        """Test a simple if statement"""

        def f(x):
            c = qp.cond(x > 1, np.sin)
            assert c.true_fn is np.sin
            assert c.condition is (x > 1)
            return c(x)

        assert np.allclose(f(1.5), np.sin(1.5))
        assert f(0.5) is None

    def test_simple_if_else(self):
        """Test a simple if-else statement"""

        def f(x):
            c = qp.cond(x > 1, np.sin, np.cos)
            assert c.false_fn is np.cos
            return c(x)

        assert np.allclose(f(1.5), np.sin(1.5))
        assert np.allclose(f(0.5), np.cos(0.5))

    def test_simple_if_elif_else(self):
        """Test a simple if-elif-else statement"""

        def f(x):
            elifs = [(x >= -1, lambda y: y**2), (x > -10, lambda y: y**3)]
            c = qp.cond(x > 1, np.sin, np.cos, elifs)
            return c(x)

        assert np.allclose(f(1.5), np.sin(1.5))
        assert np.allclose(f(-0.5), (-0.5) ** 2)
        assert np.allclose(f(-5), (-5) ** 3)
        assert np.allclose(f(-10.5), np.cos(-10.5))

    def test_simple_if_elif_else_order(self):
        """Test a simple if-elif-else statement where the order of the elif
        statements matter"""

        def f(x):
            elifs = [(x > -10, lambda y: y**3), (x >= -1, lambda y: y**2)]
            c = qp.cond(x > 1, np.sin, np.cos, elifs)

            for i, j in zip(c.elifs, elifs):
                assert i[0] is j[0]
                assert i[1] is j[1]

            return c(x)

        assert np.allclose(f(1.5), np.sin(1.5))
        assert np.allclose(f(-0.5), (-0.5) ** 3)
        assert np.allclose(f(-5), (-5) ** 3)
        assert np.allclose(f(-10.5), np.cos(-10.5))

    def test_decorator_syntax_if(self):
        """test a decorator if statement"""

        def f(x):
            @qp.cond(x > 0)
            def conditional(y):
                return y**2

            return conditional(x + 1)

        assert np.allclose(f(0.5), (0.5 + 1) ** 2)
        assert f(-0.5) is None

    def test_decorator_syntax_if_else(self):
        """test a decorator if-else statement"""

        def f(x):
            @qp.cond(x > 0)
            def conditional(y):
                return y**2

            @conditional.otherwise
            def conditional_false_fn(y):  # pylint: disable=unused-variable
                return -y

            return conditional(x + 1)

        assert np.allclose(f(0.5), (0.5 + 1) ** 2)
        assert np.allclose(f(-0.5), -(-0.5 + 1))

    def test_decorator_syntax_if_elif_else(self):
        """test a decorator if-elif-else statement"""

        def f(x):
            @qp.cond(x > 0)
            def conditional(y):
                return y**2

            @conditional.else_if(x < -2)
            def conditional_elif(y):  # pylint: disable=unused-variable
                return y

            @conditional.otherwise
            def conditional_false_fn(y):  # pylint: disable=unused-variable
                return -y

            return conditional(x + 1)

        assert np.allclose(f(0.5), (0.5 + 1) ** 2)
        assert np.allclose(f(-0.5), -(-0.5 + 1))
        assert np.allclose(f(-2.5), (-2.5 + 1))

    def test_error_mcms_elif(self):
        """Test that an error is raised if elifs are provided
        when the conditional includes an MCM"""
        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit(x):
            qp.RX(x, wires=0)
            m = qp.measure(0)
            qp.cond(m, qp.RX, elifs=[(~m, qp.RY)])
            return qp.probs

        with pytest.raises(ConditionalTransformError, match="'elif' branches are not supported"):
            circuit(0.5)

    def test_error_no_true_fn(self):
        """Test that an error is raised if no true_fn is provided
        when the conditional includes an MCM"""
        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit(x):
            qp.RX(x, wires=0)
            m = qp.measure(0)

            @qp.cond(m)
            def conditional():
                qp.RZ(x**2)

            conditional()
            return qp.probs

        with pytest.raises(TypeError, match="cannot be used as a decorator"):
            circuit(0.5)

    def test_qnode(self):
        """Test that qp.cond falls back to Python when used
        within a QNode"""
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit(x):
            elifs = [(x > 1.4, lambda y, wires: qp.RY(y**2, wires=wires))]
            c = qp.cond(x > 2.7, qp.RX, qp.RZ, elifs)
            c(x, wires=0)
            return qp.probs(wires=0)

        tape = qp.workflow.construct_tape(circuit)(3)
        ops = tape.operations
        assert len(ops) == 1
        assert ops[0].name == "RX"

        tape = qp.workflow.construct_tape(circuit)(2)
        ops = tape.operations
        assert len(ops) == 1
        assert ops[0].name == "RY"
        assert np.allclose(ops[0].parameters[0], 2**2)

        tape = qp.workflow.construct_tape(circuit)(1)
        ops = tape.operations
        assert len(ops) == 1
        assert ops[0].name == "RZ"
