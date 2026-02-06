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

import pennylane as qp

pytestmark = [pytest.mark.jax, pytest.mark.capture]

jax = pytest.importorskip("jax")

# pylint: disable=wrong-import-position
from pennylane.capture.primitives import adjoint_transform_prim, ctrl_transform_prim
from pennylane.tape.plxpr_conversion import CollectOpsandMeas


class TestAdjointQfunc:
    """Tests for the adjoint transform."""

    def test_adjoint_qfunc(self):
        """Test that the adjoint qfunc transform can be captured."""

        def workflow(x):
            qp.adjoint(qp.PauliRot)(x, pauli_word="XY", wires=(0, 1))

        plxpr = jax.make_jaxpr(workflow)(0.5)

        assert len(plxpr.eqns) == 1
        assert plxpr.eqns[0].primitive == adjoint_transform_prim

        nested_jaxpr = plxpr.eqns[0].params["jaxpr"]
        assert nested_jaxpr.eqns[0].primitive == qp.PauliRot._primitive
        assert nested_jaxpr.eqns[0].params == {"id": None, "n_wires": 2, "pauli_word": "XY"}

        assert plxpr.eqns[0].params["lazy"] is True

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 1.2)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.adjoint(qp.PauliRot(1.2, "XY", wires=(0, 1))))

    def test_adjoint_qfunc_eager(self):
        """Test eager execution can be captured by the qfunc transform."""

        def workflow(x, y, z):
            qp.adjoint(qp.Rot, lazy=False)(x, y, z, 0)

        plxpr = jax.make_jaxpr(workflow)(0.5, 0.7, 0.8)

        assert len(plxpr.eqns) == 1
        assert plxpr.eqns[0].primitive == adjoint_transform_prim

        nested_jaxpr = plxpr.eqns[0].params["jaxpr"]
        assert nested_jaxpr.eqns[0].primitive == qp.Rot._primitive
        assert nested_jaxpr.eqns[0].params == {"n_wires": 1}

        assert plxpr.eqns[0].params["lazy"] is False

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, -1.0, -2.0, -3.0)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.Rot(3.0, 2.0, 1.0, 0))

    @pytest.mark.parametrize("eqn_out", (None, 2))
    def test_multiple_ops_and_classical_processing(self, eqn_out):
        """Tests applying the adjoint transform with multiple operations and classical processing."""

        # pylint: disable=inconsistent-return-statements
        def func(x, w):
            qp.X(w)
            qp.IsingXX(2 * x + 1, (w, w + 1))
            if eqn_out is None:
                return
            return eqn_out  # should be ignored by transform

        def workflow(x):
            return qp.adjoint(func)(x, 5)

        plxpr = jax.make_jaxpr(workflow)(0.5)

        with qp.queuing.AnnotatedQueue() as q:
            out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 1.2)

        assert out == []
        assert workflow(0.5) is None

        expected_op1 = qp.adjoint(qp.IsingXX(jax.numpy.array(2 * 1.2 + 1), wires=(5, 6)))
        qp.assert_equal(q.queue[0], expected_op1)
        expected_op2 = qp.adjoint(qp.X(5))
        qp.assert_equal(q.queue[1], expected_op2)

        assert len(q.queue) == 2

        assert plxpr.eqns[0].primitive == adjoint_transform_prim
        assert plxpr.eqns[0].params["lazy"] is True

        inner_plxpr = plxpr.eqns[0].params["jaxpr"]
        assert len(inner_plxpr.eqns) == 5

    def test_nested_adjoint(self):
        """Test that adjoint can be nested multiple times."""

        def workflow(w):
            return qp.adjoint(qp.adjoint(qp.X))(w)

        plxpr = jax.make_jaxpr(workflow)(10)

        assert plxpr.eqns[0].primitive == adjoint_transform_prim
        assert plxpr.eqns[0].params["jaxpr"].eqns[0].primitive == adjoint_transform_prim
        assert (
            plxpr.eqns[0].params["jaxpr"].eqns[0].params["jaxpr"].eqns[0].primitive
            == qp.PauliX._primitive
        )

        with qp.queuing.AnnotatedQueue() as q:
            out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 10)

        assert out == []
        qp.assert_equal(q.queue[0], qp.adjoint(qp.adjoint(qp.X(10))))

    def test_qfunc_with_closure_tracer(self):
        """Test that we can take the adjoint of a qfunc with a closure variable tracer."""

        def workflow(x):
            def qfunc(wire):  # x is closure variable and a tracer
                qp.RX(x, wire)

            qp.adjoint(qfunc)(2)

        jaxpr = jax.make_jaxpr(workflow)(0.5)

        assert jaxpr.eqns[0].primitive == adjoint_transform_prim
        assert jaxpr.eqns[0].params["n_consts"] == 1
        assert len(jaxpr.eqns[0].invars) == 2  # one const, one arg

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2.5)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.adjoint(qp.RX(2.5, 2)))

    def test_adjoint_grad(self):
        """Test that adjoint differentiated with grad can be captured."""
        from pennylane.capture.primitives import jacobian_prim, qnode_prim

        @qp.grad
        @qp.qnode(qp.device("default.qubit", wires=1))
        def workflow(x):
            qp.adjoint(qp.RX)(x + 0.3, 0)
            return qp.expval(qp.Z(0))

        plxpr = jax.make_jaxpr(workflow)(0.5)

        assert len(plxpr.eqns) == 1
        grad_eqn = plxpr.eqns[0]
        assert grad_eqn.primitive == jacobian_prim
        assert set(grad_eqn.params.keys()) == {
            "argnums",
            "n_consts",
            "jaxpr",
            "method",
            "h",
            "fn",
            "scalar_out",
        }
        assert grad_eqn.params["argnums"] == (0,)
        assert grad_eqn.params["n_consts"] == 0
        assert grad_eqn.params["method"] == "auto"
        assert grad_eqn.params["h"] == 1e-6
        assert len(grad_eqn.params["jaxpr"].eqns) == 1

        qnode_eqn = grad_eqn.params["jaxpr"].eqns[0]
        assert qnode_eqn.primitive == qnode_prim
        adjoint_eqn = qnode_eqn.params["qfunc_jaxpr"].eqns[1]
        assert adjoint_eqn.primitive == adjoint_transform_prim
        assert adjoint_eqn.params["jaxpr"].eqns[0].primitive == qp.RX._primitive

        out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 0.5)
        assert qp.math.isclose(out, qp.math.sin(-(0.5 + 0.3)))


@pytest.mark.usefixtures("enable_disable_dynamic_shapes")
class TestAdjointDynamicShapes:

    def test_dynamic_shape_input(self):
        """Test that the adjoint transform can accept arrays with dynamic shapes."""

        def f(x):
            qp.adjoint(qp.RX)(x, 0)

        jaxpr = jax.make_jaxpr(f, abstracted_axes=("a",))(jax.numpy.arange(4))

        tape = qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 2, jax.numpy.arange(2))
        expected = qp.adjoint(qp.RX(jax.numpy.arange(2), 0))
        qp.assert_equal(tape[0], expected)

    def test_execution_of_dynamic_array_creation(self):
        """Test that the inner function can create a dynamic array."""

        def f(i):
            x = jax.numpy.arange(i)
            qp.RX(x, i)

        def w(i):
            qp.adjoint(f)(i)

        jaxpr = jax.make_jaxpr(w)(2)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 3)

        expected = qp.adjoint(qp.RX(jax.numpy.arange(3), 3))
        qp.assert_equal(expected, collector.state["ops"][0])

    def test_complicated_dynamic_shape_input(self):
        """Test a dynamic shape input with a more complicate shape."""

        def g(x, y):
            qp.RX(x["a"], 0)
            qp.RY(y, 0)

        def f(x, y):
            qp.adjoint(g)(x, y)

        x_a_axes = {0: "n"}
        y_axes = {0: "m"}
        x = {"a": jax.numpy.arange(2)}
        y = jax.numpy.arange(3)

        abstracted_axes = ({"a": x_a_axes}, y_axes)
        jaxpr = jax.make_jaxpr(f, abstracted_axes=abstracted_axes)(x, y)
        tape = qp.tape.plxpr_to_tape(
            jaxpr.jaxpr, jaxpr.consts, 3, 4, jax.numpy.arange(3), jax.numpy.arange(4)
        )

        op1 = qp.adjoint(qp.RY(jax.numpy.arange(4), 0))
        op2 = qp.adjoint(qp.RX(jax.numpy.arange(3), 0))
        qp.assert_equal(op1, tape[0])
        qp.assert_equal(op2, tape[1])

    def test_dynamic_shape_matches_arg(self):
        """Test that a dynamically shaped array can have a shape that matches another arg."""

        def f(i, x):
            return qp.RX(x, i)

        def workflow(i):
            return qp.adjoint(f)(i, jax.numpy.arange(i))

        jaxpr = jax.make_jaxpr(workflow)(3)
        tape = qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 4)
        assert len(tape) == 1
        op1 = qp.adjoint(qp.RX(jax.numpy.arange(4), wires=4))
        qp.assert_equal(op1, tape[0])

    def test_dynamic_shape_before_matching_arg(self):
        """Test that a dynamically shaped array can have a shape that matches another arg."""

        def workflow(i):
            return qp.adjoint(qp.RX)(jax.numpy.arange(i), i)

        jaxpr = jax.make_jaxpr(workflow)(3)
        tape = qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 4)
        assert len(tape) == 1
        op1 = qp.adjoint(qp.RX(jax.numpy.arange(4), wires=4))
        qp.assert_equal(op1, tape[0])


class TestCtrlQfunc:
    """Tests for the ctrl primitive."""

    def test_operator_type_input(self):
        """Test that an operator type can be the callable."""

        def f(x, w):
            return qp.ctrl(qp.RX, 1)(x, w)

        plxpr = jax.make_jaxpr(f)(0.5, 0)

        with qp.queuing.AnnotatedQueue() as q:
            out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 1.2, 2)

        assert f(0.5, 0) is None
        assert out == []
        expected = qp.ctrl(qp.RX(1.2, 2), 1)
        qp.assert_equal(q.queue[0], expected)

        assert plxpr.eqns[0].primitive == ctrl_transform_prim
        assert plxpr.eqns[0].params["control_values"] == (True,)
        assert plxpr.eqns[0].params["n_control"] == 1
        assert plxpr.eqns[0].params["work_wires"] is None
        assert plxpr.eqns[0].params["n_consts"] == 0

    def test_dynamic_control_wires(self):
        """Test that control wires can be dynamic."""

        def f(w1, w2, w3):
            return qp.ctrl(qp.X, (w2, w3))(w1)

        plxpr = jax.make_jaxpr(f)(4, 5, 6)

        with qp.queuing.AnnotatedQueue() as q:
            out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 1, 2, 3)

        assert out == []
        expected = qp.Toffoli(wires=(2, 3, 1))
        qp.assert_equal(q.queue[0], expected)
        assert len(q) == 1

        assert plxpr.eqns[0].primitive == ctrl_transform_prim
        assert plxpr.eqns[0].params["control_values"] == (True, True)
        assert plxpr.eqns[0].params["n_control"] == 2
        assert plxpr.eqns[0].params["work_wires"] is None

    def test_work_wires(self):
        """Test that work wires can be provided."""

        def f(w):
            return qp.ctrl(qp.S, (1, 2), work_wires="aux")(w)

        plxpr = jax.make_jaxpr(f)(6)

        with qp.queuing.AnnotatedQueue() as q:
            out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 5)

        assert out == []
        expected = qp.ctrl(qp.S(5), (1, 2), work_wires="aux")
        qp.assert_equal(q.queue[0], expected)
        assert len(q) == 1

        assert plxpr.eqns[0].params["work_wires"] == "aux"

    def test_control_values(self):
        """Test that control values can be provided."""

        def f(z):
            return qp.ctrl(qp.RZ, (3, 4), [False, True])(z, 0)

        plxpr = jax.make_jaxpr(f)(0.5)

        with qp.queuing.AnnotatedQueue() as q:
            out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 5.4)

        assert out == []
        expected = qp.ctrl(qp.RZ(5.4, 0), (3, 4), [False, True])
        qp.assert_equal(q.queue[0], expected)
        assert len(q) == 1

        assert plxpr.eqns[0].params["control_values"] == (False, True)
        assert plxpr.eqns[0].params["n_control"] == 2

    def test_nested_control(self):
        """Test that control can be nested."""

        def f(x, w1, w2):
            f1 = qp.ctrl(qp.Rot, w1)
            return qp.ctrl(f1, w2)(x, 0.5, 2 * x, 0)

        plxpr = jax.make_jaxpr(f)(-0.5, 1, 2)

        # First equation of plxpr is the multiplication of x by 2
        assert plxpr.eqns[1].params["n_consts"] == 1  # w1 is a const for the outer `ctrl`
        assert (
            plxpr.eqns[1].invars[0] is plxpr.jaxpr.invars[1]
        )  # first input is first control wire, const
        assert plxpr.eqns[1].invars[1] is plxpr.jaxpr.invars[0]  # second input is x, first arg
        assert plxpr.eqns[1].invars[-1] is plxpr.jaxpr.invars[2]  # second control wire
        assert len(plxpr.eqns[1].invars) == 6  # one const, 4 args, one control wire

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 1.2, 3, 4)

        target = qp.Rot(1.2, 0.5, jax.numpy.array(2 * 1.2), wires=0)
        expected = qp.ctrl(qp.ctrl(target, 3), 4)
        qp.assert_equal(q.queue[0], expected)

    @pytest.mark.parametrize("include_s", (True, False))
    def test_extended_qfunc(self, include_s):
        """Test that the qfunc can contain multiple operations and classical processing."""

        def qfunc(x, wire, include_s=True):
            qp.RX(2 * x, wire)
            qp.RY(x + 1, wire + 1)
            if include_s:
                qp.S(wire)

        def workflow(wire):
            qp.ctrl(qfunc, 0)(0.5, wire, include_s=include_s)

        jaxpr = jax.make_jaxpr(workflow)(1)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2)

        expected0 = qp.ctrl(qp.RX(jax.numpy.array(1.0), 2), 0)
        expected1 = qp.ctrl(qp.RY(jax.numpy.array(1.5), 3), 0)
        assert len(q.queue) == 2 + include_s
        qp.assert_equal(q.queue[0], expected0)
        qp.assert_equal(q.queue[1], expected1)
        if include_s:
            qp.assert_equal(q.queue[2], qp.ctrl(qp.S(2), 0))

        eqn = jaxpr.eqns[0]
        assert eqn.params["control_values"] == (True,)
        assert eqn.params["n_consts"] == 0
        assert eqn.params["n_control"] == 1
        assert eqn.params["work_wires"] is None

        assert len(eqn.params["jaxpr"].eqns) == 5 + include_s

    def test_ctrl_grad(self):
        """Test that ctrl differentiated with grad can be captured."""
        from pennylane.capture.primitives import jacobian_prim, qnode_prim

        @qp.grad
        @qp.qnode(qp.device("default.qubit", wires=2))
        def workflow(x):
            qp.Hadamard(1)
            qp.ctrl(qp.RX, control=1)(x + 0.3, 0)
            return qp.expval(qp.Z(0))

        plxpr = jax.make_jaxpr(workflow)(0.5)

        assert len(plxpr.eqns) == 1
        grad_eqn = plxpr.eqns[0]
        assert grad_eqn.primitive == jacobian_prim
        assert set(grad_eqn.params.keys()) == {
            "argnums",
            "n_consts",
            "jaxpr",
            "method",
            "h",
            "fn",
            "scalar_out",
        }
        assert grad_eqn.params["argnums"] == (0,)
        assert grad_eqn.params["n_consts"] == 0
        assert grad_eqn.params["method"] == "auto"
        assert grad_eqn.params["h"] == 1e-6
        assert len(grad_eqn.params["jaxpr"].eqns) == 1

        qnode_eqn = grad_eqn.params["jaxpr"].eqns[0]
        assert qnode_eqn.primitive == qnode_prim
        ctrl_eqn = qnode_eqn.params["qfunc_jaxpr"].eqns[2]
        assert ctrl_eqn.primitive == ctrl_transform_prim
        assert ctrl_eqn.params["jaxpr"].eqns[0].primitive == qp.RX._primitive

        out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 0.5)
        assert qp.math.isclose(out, -0.5 * qp.math.sin(0.5 + 0.3))

    def test_pytree_input(self):
        """Test that ctrl can accept pytree inputs."""

        def g(x):
            qp.RX(x["a"], x["wire"])

        def f(x):
            qp.ctrl(g, [1])(x)

        jaxpr = jax.make_jaxpr(f)({"a": 0.5, "wire": 0})
        tape = qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 0.5, 0)
        assert len(tape) == 1
        expected = qp.ctrl(qp.RX(0.5, 0), [1])
        qp.assert_equal(tape[0], expected)


@pytest.mark.usefixtures("enable_disable_dynamic_shapes")
class TestCtrlDynamicShapeInput:

    def test_dynamic_shape_input(self):
        """Test that ctrl can accept dynamic shape inputs."""

        def f(x):
            qp.ctrl(qp.RX, (2, 3))(x, 0)

        jaxpr = jax.make_jaxpr(f, abstracted_axes=("a",))(jax.numpy.arange(4))

        tape = qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 2, jax.numpy.arange(2))
        expected = qp.ctrl(qp.RX(jax.numpy.arange(2), 0), (2, 3))
        qp.assert_equal(tape[0], expected)

    def test_execution_of_dynamic_array_creation(self):
        """Test that the inner function can create a dynamic array."""

        def f(i):
            x = jax.numpy.arange(i)
            qp.RX(x, i)

        def w(i):
            qp.ctrl(f, 4)(i)

        jaxpr = jax.make_jaxpr(w)(2)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 3)

        expected = qp.ctrl(qp.RX(jax.numpy.arange(3), 3), 4)
        qp.assert_equal(expected, collector.state["ops"][0])

    def test_dynamic_shape_matches_arg(self):
        """Test that a dynamically shaped array can have a shape that matches another arg."""

        def f(i, x):
            return qp.RX(x, i)

        def workflow(i):
            return qp.ctrl(f, 2)(i, jax.numpy.arange(i))

        jaxpr = jax.make_jaxpr(workflow)(3)
        tape = qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 4)
        assert len(tape) == 1
        op1 = qp.ctrl(qp.RX(jax.numpy.arange(4), wires=4), 2)
        qp.assert_equal(op1, tape[0])

    def test_dynamic_shape_before_matching_arg(self):
        """Test that a dynamically shaped array can have a shape that matches another arg."""

        def workflow(i):
            return qp.ctrl(qp.RX, 6)(jax.numpy.arange(i), i)

        jaxpr = jax.make_jaxpr(workflow)(3)
        tape = qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 4)
        assert len(tape) == 1
        op1 = qp.ctrl(qp.RX(jax.numpy.arange(4), wires=4), 6)
        qp.assert_equal(op1, tape[0])
