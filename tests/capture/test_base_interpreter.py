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
This submodule tests strategy structure for defining custom plxpr interpreters
"""
# pylint: disable=protected-access
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

from pennylane.capture.base_interpreter import (  # pylint: disable=wrong-import-position
    PlxprInterpreter,
)
from pennylane.capture.primitives import (  # pylint: disable=wrong-import-position
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    for_loop_prim,
    qnode_prim,
    while_loop_prim,
)

pytestmark = pytest.mark.jax


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    """Enable and disable the PennyLane JAX capture context manager."""
    qml.capture.enable()
    yield
    qml.capture.disable()


class SimplifyInterpreter(PlxprInterpreter):

    def interpret_operation(self, op):
        new_op = op.simplify()
        if new_op is op:
            # if new op isn't queued, need to requeue op.
            data, struct = jax.tree_util.tree_flatten(new_op)
            new_op = jax.tree_util.tree_unflatten(struct, data)
        return new_op


def test_env_and_state_initialized():
    """Test that env and state are initialized at the start."""

    interpreter = SimplifyInterpreter()
    assert interpreter._env == {}  # pylint: disable=use-implicit-booleaness-not-comparison
    assert interpreter.state is None


def test_primitive_registrations():
    """Test that child primitive registrations dict's are not copied and do
    not effect PlxprInterpreeter."""

    class SimplifyInterpreterLocal(PlxprInterpreter):

        def interpret_operation(self, op):
            new_op = op.simplify()
            if new_op is op:
                # if new op isn't queued, need to requeue op.
                data, struct = jax.tree_util.tree_flatten(new_op)
                new_op = jax.tree_util.tree_unflatten(struct, data)
            return new_op

    assert (
        SimplifyInterpreterLocal._primitive_registrations
        is not PlxprInterpreter._primitive_registrations
    )

    @SimplifyInterpreterLocal.register_primitive(qml.X._primitive)
    def _(self, *invals, **params):  # pylint: disable=unused-argument
        print("in custom interpreter")
        return qml.Z(*invals)

    assert qml.X._primitive in SimplifyInterpreterLocal._primitive_registrations
    assert qml.X._primitive not in PlxprInterpreter._primitive_registrations

    @SimplifyInterpreterLocal()
    def f():
        qml.X(0)
        qml.Y(5)

    jaxpr = jax.make_jaxpr(f)()

    with qml.queuing.AnnotatedQueue() as q:
        jax.core.eval_jaxpr(jaxpr.jaxpr, [])

    qml.assert_equal(q.queue[0], qml.Z(0))  # turned into a Y
    qml.assert_equal(q.queue[1], qml.Y(5))  # mapped wire


def test_overriding_measurements():
    """Test usage of an interpreter with a custom way of handling measurements."""

    class MeasurementsToSample(PlxprInterpreter):

        def interpret_measurement_eqn(self, primitive, *invals, **params):
            temp_mp = primitive.impl(*invals, **params)
            return qml.sample(wires=temp_mp.wires)

    @MeasurementsToSample()
    @qml.qnode(qml.device("default.qubit", wires=2, shots=5))
    def circuit():
        return qml.expval(qml.Z(0)), qml.probs(wires=(0, 1))

    circuit()


class TestHigherOrderPrimitiveRegistrations:

    @pytest.mark.parametrize("lazy", (True, False))
    def test_adjoint_transform(self, lazy):
        """Test the higher order adjoint transform."""

        @SimplifyInterpreter()
        def f(x):
            def g(y):
                _ = qml.RX(y, 0) ** 3

            qml.adjoint(g, lazy=lazy)(x)

        jaxpr = jax.make_jaxpr(f)(0.5)

        assert jaxpr.eqns[0].params["lazy"] == lazy
        assert jaxpr.eqns[0].primitive == adjoint_transform_prim
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        # first eqn mul, second RX
        assert inner_jaxpr.eqns[1].primitive == qml.RX._primitive
        assert len(inner_jaxpr.eqns) == 2

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)

        if lazy:
            qml.assert_equal(q.queue[0], qml.adjoint(qml.RX(jax.numpy.array(1.5), 0)))
        else:
            qml.assert_equal(q.queue[0], qml.RX(jax.numpy.array(-1.5), 0))

    def test_ctrl_transform(self):
        """Test the higher order adjoint transform."""

        @SimplifyInterpreter()
        def f(x, control):
            def g(y):
                _ = qml.RY(y, 0) ** 3

            qml.ctrl(g, control)(x)

        jaxpr = jax.make_jaxpr(f)(0.5, 1)

        assert jaxpr.eqns[0].primitive == ctrl_transform_prim
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        # first eqn mul, second RY
        assert inner_jaxpr.eqns[1].primitive == qml.RY._primitive
        assert len(inner_jaxpr.eqns) == 2

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2.0, 1)

        qml.assert_equal(q.queue[0], qml.ctrl(qml.RY(jax.numpy.array(6.0), 0), 1))

    def test_cond(self):
        """Test the cond higher order primitive."""

        @SimplifyInterpreter()
        def f(x, control):

            def true_fn(y):
                _ = qml.RY(y, 0) ** 2

            def false_fn(y):
                _ = qml.adjoint(qml.RX(y, 0))

            qml.cond(control, true_fn, false_fn)(x)

        jaxpr = jax.make_jaxpr(f)(0.5, False)
        assert jaxpr.eqns[0].primitive == cond_prim

        branch1 = jaxpr.eqns[0].params["jaxpr_branches"][0]
        assert len(branch1.eqns) == 2
        assert branch1.eqns[1].primitive == qml.RY._primitive
        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(branch1, [], 0.5)
        qml.assert_equal(q.queue[0], qml.RY(2 * 0.5, 0))

        branch2 = jaxpr.eqns[0].params["jaxpr_branches"][1]
        assert len(branch2.eqns) == 2
        assert branch2.eqns[1].primitive == qml.RX._primitive
        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(branch2, [], 0.5)
        qml.assert_equal(q.queue[0], qml.RY(-0.5, 0))

        assert jaxpr.eqns[0].params["n_args"] == 1
        assert jaxpr.eqns[0].params["n_consts_per_branch"] == [0, 0]

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2.4, True)

        qml.assert_equal(q.queue[0], qml.RY(4.8, 0))

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.23, False)

        qml.assert_equal(q.queue[0], qml.RX(-1.23, 0))

    def test_cond_no_false_branch(self):
        """Test transforming a cond HOP when no false branch exists."""

        @SimplifyInterpreter()
        def f(control):

            @qml.cond(control)
            def f():
                qml.X(0) @ qml.X(0)

            f()

        jaxpr = jax.make_jaxpr(f)(True)

        assert jaxpr.eqns[0].params["jaxpr_branches"][-1] is None  # no false branch

        with qml.queuing.AnnotatedQueue() as q_true:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, True)

        qml.assert_equal(q_true.queue[0], qml.I(0))

        with qml.queuing.AnnotatedQueue() as q_false:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, False)

        assert len(q_false.queue) == 0

    def test_for_loop(self):
        """Test the higher order for loop registration."""

        @SimplifyInterpreter()
        def f(n):

            @qml.for_loop(n)
            def g(i):
                qml.adjoint(qml.X(i))

            g()

        jaxpr = jax.make_jaxpr(f)(3)
        assert jaxpr.eqns[0].primitive == for_loop_prim

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 1
        assert inner_jaxpr.eqns[0].primitive == qml.X._primitive  # no adjoint of x

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)

        qml.assert_equal(q.queue[0], qml.X(0))
        qml.assert_equal(q.queue[1], qml.X(1))
        qml.assert_equal(q.queue[2], qml.X(2))
        assert len(q) == 3

    def test_while_loop(self):
        """Test the higher order for loop registration."""

        @SimplifyInterpreter()
        def f(n):

            @qml.while_loop(lambda i: i < n)
            def g(i):
                qml.adjoint(qml.Z(i))
                return i + 1

            g(0)

        jaxpr = jax.make_jaxpr(f)(3)
        assert jaxpr.eqns[0].primitive == while_loop_prim

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 2
        assert inner_jaxpr.eqns[0].primitive == qml.Z._primitive  # no adjoint of x

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)

        qml.assert_equal(q.queue[0], qml.Z(0))
        qml.assert_equal(q.queue[1], qml.Z(1))
        qml.assert_equal(q.queue[2], qml.Z(2))
        assert len(q) == 3

    def test_qnode(self):
        """Test transforming qnodes."""

        class AddNoise(PlxprInterpreter):

            def interpret_operation(self, op):
                data, struct = jax.tree_util.tree_flatten(op)
                new_op = jax.tree_util.tree_unflatten(struct, data)
                _ = [qml.RX(0.1, w) for w in op.wires]
                return new_op

        dev = qml.device("default.qubit", wires=1)

        @AddNoise()
        @qml.qnode(dev, diff_method="adjoint", grad_on_execution=False)
        def f():
            qml.I(0)
            qml.I(0)
            return qml.probs(wires=0)

        jaxpr = jax.make_jaxpr(f)()
        assert jaxpr.eqns[0].primitive == qnode_prim
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

        assert len(inner_jaxpr.eqns) == 5
        assert inner_jaxpr.eqns[0].primitive == qml.I._primitive
        assert inner_jaxpr.eqns[2].primitive == qml.I._primitive
        assert inner_jaxpr.eqns[1].primitive == qml.RX._primitive
        assert inner_jaxpr.eqns[3].primitive == qml.RX._primitive

        assert jaxpr.eqns[0].params["qnode_kwargs"]["diff_method"] == "adjoint"
        assert jaxpr.eqns[0].params["qnode_kwargs"]["grad_on_execution"] is False
        assert jaxpr.eqns[0].params["device"] == dev

        res1 = f()
        # end up performing two rx gates with phase of 0.1 each on wire 0
        expected = jax.numpy.array([jax.numpy.cos(0.2 / 2) ** 2, jax.numpy.sin(0.2 / 2) ** 2])
        assert qml.math.allclose(res1, expected)
        res2 = jax.core.eval_jaxpr(jaxpr.jaxpr, [])
        assert qml.math.allclose(res2, expected)

    @pytest.mark.parametrize("grad_f", (qml.grad, qml.jacobian))
    def test_grad_and_jac(self, grad_f):
        """Test interpreters can handle grad and jacobian HOP's."""

        class DoubleAngle(PlxprInterpreter):

            def interpret_operation(self, op):
                leaves, struct = jax.tree_util.tree_flatten(op)
                return jax.tree_util.tree_unflatten(struct, [2 * l for l in leaves])

        @DoubleAngle()
        def f(x):
            @qml.qnode(qml.device("default.qubit", wires=2))
            def circuit(y):
                qml.RX(y, 0)
                return qml.expval(qml.Z(0))

            return grad_f(circuit)(x)

        out = f(0.5)
        expected = -2 * jax.numpy.sin(2 * 0.5)  # includes the factors of 2 from doubling the angle.
        assert qml.math.allclose(out, expected)
