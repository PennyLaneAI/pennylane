# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Tests for CollectOpsandMeas and plxpr_to_tape
"""
import pytest

import pennylane as qml

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]

jax = pytest.importorskip("jax")


from pennylane.tape.plxpr_conversion import (  # pylint: disable=wrong-import-position
    CollectOpsandMeas,
)


class TestCollectOpsandMeas:
    """Tests for the CollectOpsandMeas class."""

    def test_flat_func(self):
        """Test a function without classical structure."""

        def f(x):
            qml.RX(x, 0)
            qml.CNOT((0, 1))
            qml.QFT(wires=(0, 1, 2))
            return qml.expval(qml.Z(0))

        obj = CollectOpsandMeas()
        obj(f)(1.2)
        qml.assert_equal(obj.state["ops"][0], qml.RX(1.2, 0))
        qml.assert_equal(obj.state["ops"][1], qml.CNOT((0, 1)))
        qml.assert_equal(obj.state["ops"][2], qml.QFT((0, 1, 2)))
        assert len(obj.state["ops"]) == 3

        qml.assert_equal(obj.state["measurements"][0], qml.expval(qml.Z(0)))

    def test_for_loop(self):
        """Test collecting the operations in a for loop."""

        def f(n):
            @qml.for_loop(n)
            def g(i):
                qml.X(i)

            g()

        obj = CollectOpsandMeas()
        obj(f)(3)
        assert len(obj.state["ops"]) == 3
        qml.assert_equal(obj.state["ops"][0], qml.X(0))
        qml.assert_equal(obj.state["ops"][1], qml.X(1))
        qml.assert_equal(obj.state["ops"][2], qml.X(2))

        assert len(obj.state["measurements"]) == 0

    def test_while_loop(self):
        """Test collecting the operations in a while loop."""

        def g(x):
            @qml.while_loop(lambda x, i: i < 3)
            def loop(x, i):
                qml.RX(x, i)
                return 2 * x, i + 1

            loop(x, 0)

        obj = CollectOpsandMeas()
        x = jax.numpy.array(1.2)
        obj(g)(x)

        assert len(obj.state["ops"]) == 3
        assert len(obj.state["measurements"]) == 0

        qml.assert_equal(obj.state["ops"][0], qml.RX(x, 0))
        qml.assert_equal(obj.state["ops"][1], qml.RX(2 * x, 1))
        qml.assert_equal(obj.state["ops"][2], qml.RX(4 * x, 2))

    def test_cond_bool(self):
        """Test applying a conditional of a classical vlaue."""

        def f(x, value):
            qml.cond(value, qml.RX, false_fn=qml.RY)(x, 0)

        obj1 = CollectOpsandMeas()
        x = jax.numpy.array(-0.5)
        obj1(f)(x, True)
        assert len(obj1.state["ops"]) == 1
        qml.assert_equal(obj1.state["ops"][0], qml.RX(x, 0))

        obj2 = CollectOpsandMeas()
        obj2(f)(x, False)
        assert len(obj2.state["ops"]) == 1
        qml.assert_equal(obj2.state["ops"][0], qml.RY(x, 0))

    def test_measure(self):
        """Test capturing measurements."""

        def f():
            m0 = qml.measure(0)
            return qml.sample(op=m0)

        obj = CollectOpsandMeas()
        obj(f)()

        assert len(obj.state["ops"]) == 1
        assert isinstance(obj.state["ops"][0], qml.measurements.MidMeasureMP)
        assert obj.state["ops"][0].wires == qml.wires.Wires(0)

        assert isinstance(obj.state["measurements"][0], qml.measurements.SampleMP)
        assert obj.state["measurements"][0].mv is not None

    def test_cond_mcm(self):
        """Test capturing a conditional of a mid circuit measurement."""

        def rx(x, w):
            qml.RX(x, w)

        def f(x):
            m0 = qml.measure(0)
            qml.cond(m0, rx)(x, 2)
            return m0

        x = jax.numpy.array(0.987)

        obj = CollectOpsandMeas()
        mv = obj(f)(x)

        assert len(obj.state["ops"]) == 2
        assert isinstance(obj.state["ops"][0], qml.measurements.MidMeasureMP)
        assert mv.measurements[0] is obj.state["ops"][0]

        qml.assert_equal(obj.state["ops"][1], qml.ops.Conditional(mv, qml.RX(x, 2)))

    def test_elif_mcm(self):
        """Test that an elif mcm can be caputured."""

        def rx(*args):
            qml.RX(*args)

        def ry(*args):
            qml.RY(*args)

        def rz(*args):
            qml.RZ(*args)

        def f(x):
            m0 = qml.measure(0)
            m1 = qml.measure(1)

            qml.cond(m0, rx, elifs=(m1, ry), false_fn=rz)(x, 0)

        x = jax.numpy.array(0.5)

        obj = CollectOpsandMeas()
        obj(f)(x)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_adjoint_transform(self, lazy):
        """Test capture the adjoint of a qfunc."""

        def qfunc(x):
            qml.RX(x, 0)
            qml.RY(2 * x, 0)
            qml.X(2)

        def f(x):
            qml.adjoint(qfunc, lazy=lazy)(x)

        obj = CollectOpsandMeas()
        x = jax.numpy.array(2.1)
        obj(f)(x)

        assert len(obj.state["ops"]) == 3
        qml.assert_equal(obj.state["ops"][0], qml.adjoint(qml.X(2), lazy=lazy))
        qml.assert_equal(obj.state["ops"][1], qml.adjoint(qml.RY(2 * x, 0), lazy=lazy))
        qml.assert_equal(obj.state["ops"][2], qml.adjoint(qml.RX(x, 0), lazy=lazy))

    def test_control_transform(self):
        """Test collecting the control of a qfunc."""

        def qfunc(x, wire):
            qml.RX(x, wire)
            qml.X(wire)

        def f(x):
            qml.ctrl(qfunc, control=[1, 2], control_values=[False, False])(x, 0)

        obj = CollectOpsandMeas()
        x = jax.numpy.array(-0.98)
        obj(f)(x)

        assert len(obj.state["ops"]) == 2
        expected0 = qml.ctrl(qml.RX(x, 0), [1, 2], control_values=[False, False])
        qml.assert_equal(obj.state["ops"][0], expected0)
        expected1 = qml.ctrl(qml.X(0), [1, 2], control_values=[False, False])
        qml.assert_equal(obj.state["ops"][1], expected1)

    def test_hybrid_cond_error(self):
        """Test an error is raised if a conditional contains both mcms and classical values."""

        def true_fn(x):
            qml.RX(x, 0)

        def elif_fn(x):
            qml.IsingXX(x, 0)

        def f(x, value):
            m0 = qml.measure(0)
            qml.cond(m0, true_fn, elifs=(value, elif_fn))(x)

        collector = CollectOpsandMeas()
        with pytest.raises(ValueError, match="Cannot use qml.cond with a combination"):
            collector(f)(0.5, False)

    def test_cond_fn_no_returns(self):
        """Test that an error is raised if a cond branch of a measurement value returns something"""

        def f():
            m0 = qml.measure(0)
            qml.cond(m0, qml.RX, false_fn=qml.RY)(0.5, 0)

        collector = CollectOpsandMeas()
        with pytest.raises(ValueError, match="Conditional branches of mid circuit measurements"):
            collector(f)()


class TestPlxprToTape:
    """Tests for the plxpr_to_tape function."""

    def test_flat_func(self):
        """Test a function without classical structure."""

        def f(x):
            qml.RX(x, 0)
            qml.CNOT((0, 1))
            qml.QFT(wires=(0, 1, 2))
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(-0.5)
        tape = qml.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 1.2, shots=100)
        qml.assert_equal(tape[0], qml.RX(1.2, 0))
        qml.assert_equal(tape[1], qml.CNOT((0, 1)))
        qml.assert_equal(tape[2], qml.QFT((0, 1, 2)))
        assert len(tape.operations) == 3

        qml.assert_equal(tape.measurements[0], qml.expval(qml.Z(0)))
        assert tape.shots == qml.measurements.Shots(100)

    def test_for_loop(self):
        """Test collecting the operations in a for loop."""

        def f(n):
            @qml.for_loop(n)
            def g(i):
                qml.X(i)

            g()

        jaxpr = jax.make_jaxpr(f)(5)
        tape = qml.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 3)
        assert len(tape.operations) == 3
        qml.assert_equal(tape[0], qml.X(0))
        qml.assert_equal(tape[1], qml.X(1))
        qml.assert_equal(tape[2], qml.X(2))

        assert len(tape.measurements) == 0

    def test_while_loop(self):
        """Test collecting the operations in a while loop."""

        def g(x):
            @qml.while_loop(lambda x, i: i < 3)
            def loop(x, i):
                qml.RX(x, i)
                return 2 * x, i + 1

            loop(x, 0)

        jaxpr = jax.make_jaxpr(g)(-0.8)
        x = jax.numpy.array(1.2)
        tape = qml.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, x)

        assert len(tape.operations) == 3
        assert len(tape.measurements) == 0

        qml.assert_equal(tape.operations[0], qml.RX(x, 0))
        qml.assert_equal(tape.operations[1], qml.RX(2 * x, 1))
        qml.assert_equal(tape.operations[2], qml.RX(4 * x, 2))

    def test_cond_bool(self):
        """Test applying a conditional of a classical vlaue."""

        def f(x, value):
            qml.cond(value, qml.RX, false_fn=qml.RY)(x, 0)

        x = jax.numpy.array(-0.5)
        jaxpr = jax.make_jaxpr(f)(x, False)
        tape = qml.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, x, True)
        assert len(tape.operations) == 1
        qml.assert_equal(tape.operations[0], qml.RX(x, 0))

        tape2 = qml.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, x, False)
        assert len(tape2.operations) == 1
        qml.assert_equal(tape2.operations[0], qml.RY(x, 0))

    def test_measure(self):
        """Test capturing measurements."""

        def f():
            m0 = qml.measure(0)
            return qml.sample(op=m0)

        jaxpr = jax.make_jaxpr(f)()
        tape = qml.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts)

        assert len(tape.operations) == 1
        assert isinstance(tape.operations[0], qml.measurements.MidMeasureMP)
        assert tape.operations[0].wires == qml.wires.Wires(0)

        assert isinstance(tape.measurements[0], qml.measurements.SampleMP)
        assert tape.measurements[0].mv is not None

    def test_cond_mcm(self):
        """Test capturing a conditional of a mid circuit measurement."""

        def rx(x, w):
            qml.RX(x, w)

        def f(x):
            m0 = qml.measure(0)
            qml.cond(m0, rx)(x, 2)
            return qml.sample(m0)

        x = jax.numpy.array(0.987)

        jaxpr = jax.make_jaxpr(f)(x)
        tape = qml.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, x)

        assert len(tape.operations) == 2
        assert isinstance(tape.operations[0], qml.measurements.MidMeasureMP)
        assert isinstance(tape.measurements[0], qml.measurements.SampleMP)
        mp = tape.measurements[0]
        assert mp.mv.measurements[0] is tape.operations[0]
        qml.assert_equal(tape.operations[1], qml.ops.Conditional(mp.mv, qml.RX(x, 2)))

    def test_elif_mcm(self):
        """Test that an elif mcm can be caputured."""

        def rx(*args):
            qml.RX(*args)

        def ry(*args):
            qml.RY(*args)

        def rz(*args):
            qml.RZ(*args)

        def f(x):
            m0 = qml.measure(0)
            m1 = qml.measure(1)

            qml.cond(m0, rx, elifs=(m1, ry), false_fn=rz)(x, 0)

        x = jax.numpy.array(0.5)
        jaxpr = jax.make_jaxpr(f)(x)
        tape = qml.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, x)
        assert tape  # TODO

    @pytest.mark.parametrize("lazy", (True, False))
    def test_adjoint_transform(self, lazy):
        """Test capture the adjoint of a qfunc."""

        def qfunc(x):
            qml.RX(x, 0)
            qml.RY(2 * x, 0)
            qml.X(2)

        def f(x):
            qml.adjoint(qfunc, lazy=lazy)(x)

        x = jax.numpy.array(2.1)
        jaxpr = jax.make_jaxpr(f)(0.6)
        tape = qml.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, x)

        assert len(tape.operations) == 3
        qml.assert_equal(tape.operations[0], qml.adjoint(qml.X(2), lazy=lazy))
        qml.assert_equal(tape.operations[1], qml.adjoint(qml.RY(2 * x, 0), lazy=lazy))
        qml.assert_equal(tape.operations[2], qml.adjoint(qml.RX(x, 0), lazy=lazy))

    def test_control_transform(self):
        """Test collecting the control of a qfunc."""

        def qfunc(x, wire):
            qml.RX(x, wire)
            qml.X(wire)

        def f(x):
            qml.ctrl(qfunc, control=[1, 2], control_values=[False, False])(x, 0)

        x = jax.numpy.array(-0.98)
        jaxpr = jax.make_jaxpr(f)(0.1)
        tape = qml.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, x)

        assert len(tape.operations) == 2
        expected0 = qml.ctrl(qml.RX(x, 0), [1, 2], control_values=[False, False])
        qml.assert_equal(tape.operations[0], expected0)
        expected1 = qml.ctrl(qml.X(0), [1, 2], control_values=[False, False])
        qml.assert_equal(tape.operations[1], expected1)

    def test_hybrid_cond_error(self):
        """Test an error is raised if a conditional contains both mcms and classical values."""

        def true_fn(x):
            qml.RX(x, 0)

        def elif_fn(x):
            qml.IsingXX(x, 0)

        def f(x, value):
            m0 = qml.measure(0)
            qml.cond(m0, true_fn, elifs=(value, elif_fn))(x)

        jaxpr = jax.make_jaxpr(f)(0.5, False)
        with pytest.raises(ValueError, match="Cannot use qml.cond with a combination"):
            qml.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 0.5, False)
