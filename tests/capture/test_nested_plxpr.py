# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for capturing nested plxpr.
"""
# pylint: disable=protected-access
import pytest

import pennylane as qml
from pennylane.ops.op_math.adjoint import _get_adjoint_qfunc_prim
from pennylane.ops.op_math.controlled import _get_ctrl_qfunc_prim

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")

adjoint_prim = _get_adjoint_qfunc_prim()
ctrl_prim = _get_ctrl_qfunc_prim()


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


class TestAdjointQfunc:
    """Tests for the adjoint transform."""

    def test_adjoint_qfunc(self):
        """Test that the adjoint qfunc transform can be captured."""

        def workflow(x):
            qml.adjoint(qml.PauliRot)(x, pauli_word="XY", wires=(0, 1))

        plxpr = jax.make_jaxpr(workflow)(0.5)

        assert len(plxpr.eqns) == 1
        assert plxpr.eqns[0].primitive == adjoint_prim

        nested_jaxpr = plxpr.eqns[0].params["jaxpr"]
        assert nested_jaxpr.eqns[0].primitive == qml.PauliRot._primitive
        assert nested_jaxpr.eqns[0].params == {"id": None, "n_wires": 2, "pauli_word": "XY"}

        assert plxpr.eqns[0].params["lazy"] is True

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 1.2)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.adjoint(qml.PauliRot(1.2, "XY", wires=(0, 1))))

    def test_adjoint_qfunc_eager(self):
        """Test eager execution can be captured by the qfunc transform."""

        def workflow(x, y, z):
            qml.adjoint(qml.Rot, lazy=False)(x, y, z, 0)

        plxpr = jax.make_jaxpr(workflow)(0.5, 0.7, 0.8)

        assert len(plxpr.eqns) == 1
        assert plxpr.eqns[0].primitive == adjoint_prim

        nested_jaxpr = plxpr.eqns[0].params["jaxpr"]
        assert nested_jaxpr.eqns[0].primitive == qml.Rot._primitive
        assert nested_jaxpr.eqns[0].params == {"n_wires": 1}

        assert plxpr.eqns[0].params["lazy"] is False

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, -1.0, -2.0, -3.0)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.Rot(3.0, 2.0, 1.0, 0))

    @pytest.mark.parametrize("eqn_out", (None, 2))
    def test_multiple_ops_and_classical_processing(self, eqn_out):
        """Tests applying the adjoint transform with multiple operations and classical processing."""

        # pylint: disable=inconsistent-return-statements
        def func(x, w):
            qml.X(w)
            qml.IsingXX(2 * x + 1, (w, w + 1))
            if eqn_out is None:
                return
            return eqn_out  # should be ignored by transform

        def workflow(x):
            return qml.adjoint(func)(x, 5)

        plxpr = jax.make_jaxpr(workflow)(0.5)

        with qml.queuing.AnnotatedQueue() as q:
            out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 1.2)

        assert out == []

        expected_op1 = qml.adjoint(qml.IsingXX(jax.numpy.array(2 * 1.2 + 1), wires=(5, 6)))
        qml.assert_equal(q.queue[0], expected_op1)
        expected_op2 = qml.adjoint(qml.X(5))
        qml.assert_equal(q.queue[1], expected_op2)

        assert len(q.queue) == 2

        assert plxpr.eqns[0].primitive == adjoint_prim
        assert plxpr.eqns[0].params["lazy"] is True

        inner_plxpr = plxpr.eqns[0].params["jaxpr"]
        assert len(inner_plxpr.eqns) == 5

    def test_nested_adjoint(self):
        """Test that adjoint can be nested multiple times."""

        def workflow(w):
            return qml.adjoint(qml.adjoint(qml.X))(w)

        plxpr = jax.make_jaxpr(workflow)(10)

        assert plxpr.eqns[0].primitive == adjoint_prim
        assert plxpr.eqns[0].params["jaxpr"].eqns[0].primitive == adjoint_prim
        assert (
            plxpr.eqns[0].params["jaxpr"].eqns[0].params["jaxpr"].eqns[0].primitive
            == qml.PauliX._primitive
        )

        with qml.queuing.AnnotatedQueue() as q:
            out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 10)

        assert out == []
        qml.assert_equal(q.queue[0], qml.adjoint(qml.adjoint(qml.X(10))))

       
    def test_qfunc_with_closure_tracer(self):
        """Test that we can take the adjoint of a qfunc with a closure variable tracer."""

        def workflow(x):
            def qfunc(wire):  # x is closure variable and a tracer
                qml.RX(x, wire)

            qml.adjoint(qfunc)(2)

        qml.assert_equal(out, qml.adjoint(qml.adjoint(qml.X(10))))
        qml.assert_equal(q.queue[0], out)


class TestCtrlQfunc:

    def test_operator_type_input(self):
        """Test that an operator type can be the callable."""

        def f(x, w):
            return qml.ctrl(qml.RX, 1)(x, w)

        plxpr = jax.make_jaxpr(f)(0.5, 0)

        with qml.queuing.AnnotatedQueue() as q:
            out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 1.2, 2)

        expected = qml.ctrl(qml.RX(1.2, 2), 1)
        qml.assert_equal(out[0], expected)
        qml.assert_equal(q.queue[0], expected)

        assert plxpr.eqns[0].primitive == ctrl_prim
        assert plxpr.eqns[0].params["control_values"] == [True]
        assert plxpr.eqns[0].params["n_control"] == 1
        assert plxpr.eqns[0].params["work_wires"] is None

    def test_dynamic_control_wires(self):
        """Test that control wires can be dynamic."""

        def f(w1, w2, w3):
            return qml.ctrl(qml.X, (w2, w3))(w1)

        plxpr = jax.make_jaxpr(f)(4, 5, 6)

        with qml.queuing.AnnotatedQueue() as q:
            out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 1, 2, 3)

        expected = qml.Toffoli(wires=(2, 3, 1))
        qml.assert_equal(q.queue[0], expected)
        qml.assert_equal(out[0], expected)
        assert len(out) == len(q) == 1

        assert plxpr.eqns[0].primitive == ctrl_prim
        assert plxpr.eqns[0].params["control_values"] == [True, True]
        assert plxpr.eqns[0].params["n_control"] == 2
        assert plxpr.eqns[0].params["work_wires"] is None

    def test_work_wires(self):
        """Test that work wires can be provided."""

        def f(w):
            return qml.ctrl(qml.S, (1, 2), work_wires="aux")(w)

        plxpr = jax.make_jaxpr(f)(6)

        with qml.queuing.AnnotatedQueue() as q:
            out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 5)

        expected = qml.ctrl(qml.S(5), (1, 2), work_wires="aux")
        qml.assert_equal(out[0], expected)
        qml.assert_equal(q.queue[0], expected)
        assert len(out) == len(q) == 1

        assert plxpr.eqns[0].params["work_wires"] == "aux"

    def test_control_values(self):
        """Test that control values can be provided."""

        def f(z):
            return qml.ctrl(qml.RZ, (3, 4), [False, True])(z, 0)

        plxpr = jax.make_jaxpr(f)(0.5)

        with qml.queuing.AnnotatedQueue() as q:
            out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 5.4)

        expected = qml.ctrl(qml.RZ(5.4, 0), (3, 4), [False, True])
        qml.assert_equal(out[0], expected)
        qml.assert_equal(q.queue[0], expected)
        assert len(q) == len(out) == 1

        assert plxpr.eqns[0].params["control_values"] == [False, True]
        assert plxpr.eqns[0].params["n_control"] == 2

    def test_nested_control(self):
        """Test that control can be nested."""

        def f(x, w1, w2):
            f1 = qml.ctrl(qml.Rot, w1)
            return qml.ctrl(f1, w2)(x, 0.5, 2 * x, 0)

        plxpr = jax.make_jaxpr(f)(-0.5, 1, 2)

        assert plxpr.eqns[0].params["n_consts"] == 1
        assert (
            plxpr.eqns[0].invars[0] is plxpr.jaxpr.invars[1]
        )  # first input is first control wire, const
        assert plxpr.eqns[0].invars[1] is plxpr.jaxpr.invars[0]  # second input is x, first arg
        assert plxpr.eqns[0].invars[-1] is plxpr.jaxpr.invars[2]  # second control wire
        assert len(plxpr.eqns[0].invars) == 6  # one const, 4 args, one control wire

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 1.2, 3, 4)

        target = qml.Rot(1.2, 0.5, jax.numpy.array(2 * 1.2), wires=0)
        expected = qml.ctrl(qml.ctrl(target, 3), 4)
        qml.assert_equal(q.queue[0], expected)
