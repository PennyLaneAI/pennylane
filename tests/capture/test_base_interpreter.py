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
# pylint: disable=protected-access, too-few-public-methods
import pytest

import pennylane as qp

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from pennylane.capture import PlxprInterpreter  # pylint: disable=wrong-import-position
from pennylane.capture.primitives import (  # pylint: disable=wrong-import-position
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    for_loop_prim,
    qnode_prim,
    while_loop_prim,
)

pytestmark = [pytest.mark.jax, pytest.mark.capture]


class SimplifyInterpreter(PlxprInterpreter):

    def interpret_operation(self, op):
        new_op = op.simplify()
        if new_op is op:
            new_op = new_op._unflatten(*op._flatten())
            # if new op isn't queued, need to requeue op.
        return new_op

    def interpret_measurement(self, measurement):
        new_mp = measurement.simplify()
        if new_mp is measurement:
            new_mp = new_mp._unflatten(*measurement._flatten())
            # if new op isn't queued, need to requeue op.
        return new_mp


# pylint: disable=use-implicit-booleaness-not-comparison
def test_env_and_initialized():
    """Test that env is initialized at the start."""

    interpreter = SimplifyInterpreter()
    assert interpreter._env == {}


def test_zip_length_validation():
    """Test that errors are raised if the input values isnt long enough for the needed variables."""

    def f(x):
        return x + 1

    jaxpr = jax.make_jaxpr(f)(0.5)
    with pytest.raises(ValueError):
        PlxprInterpreter().eval(jaxpr.jaxpr, [])

    y = jax.numpy.array([1.0])

    def g():
        return y + 2

    jaxpr = jax.make_jaxpr(g)()
    with pytest.raises(ValueError):
        PlxprInterpreter().eval(jaxpr.jaxpr, [])


def test_primitive_registrations():
    """Test that child primitive registrations dict's are not copied and do
    not affect PlxprInterpreter."""

    class SimplifyInterpreterLocal(SimplifyInterpreter):
        pass

    assert (
        SimplifyInterpreterLocal._primitive_registrations
        is not PlxprInterpreter._primitive_registrations
    )

    @SimplifyInterpreterLocal.register_primitive(qp.X._primitive)
    def _(self, *invals, **params):  # pylint: disable=unused-argument
        return qp.Z(*invals)

    assert qp.X._primitive in SimplifyInterpreterLocal._primitive_registrations
    assert qp.X._primitive not in PlxprInterpreter._primitive_registrations

    @SimplifyInterpreterLocal()
    def f():
        qp.X(0)
        qp.Y(5)

    jaxpr = jax.make_jaxpr(f)()

    with qp.queuing.AnnotatedQueue() as q:
        jax.core.eval_jaxpr(jaxpr.jaxpr, [])

    qp.assert_equal(q.queue[0], qp.Z(0))  # turned into a Z
    qp.assert_equal(q.queue[1], qp.Y(5))


def test_default_operator_handling():
    """Test that the PlxprInterpreter itself can handle operators and leaves them unchanged."""

    @PlxprInterpreter()
    def f(x):
        qp.adjoint(qp.RX(x, 0))
        qp.T(1)
        return qp.X(0) + qp.X(1)

    with qp.queuing.AnnotatedQueue() as q:
        out = f(0.5)

    qp.assert_equal(out, qp.X(0) + qp.X(1))
    qp.assert_equal(q.queue[0], qp.adjoint(qp.RX(0.5, 0)))
    qp.assert_equal(q.queue[1], qp.T(1))
    qp.assert_equal(q.queue[2], qp.X(0) + qp.X(1))

    jaxpr = jax.make_jaxpr(f)(1.2)

    assert jaxpr.eqns[0].primitive == qp.RX._primitive
    assert jaxpr.eqns[1].primitive == qp.ops.Adjoint._primitive
    assert jaxpr.eqns[2].primitive == qp.T._primitive
    assert jaxpr.eqns[3].primitive == qp.X._primitive
    assert jaxpr.eqns[4].primitive == qp.X._primitive
    assert jaxpr.eqns[5].primitive == qp.ops.Sum._primitive


@pytest.mark.parametrize(
    "op_class, args, kwargs",
    [
        (qp.CH, ([0, 1],), {}),
        (qp.CY, ([0, 1],), {}),
        (qp.CZ, ([0, 1],), {}),
        (qp.CSWAP, ([0, 1, 2],), {}),
        (qp.CCZ, ([0, 1, 2],), {}),
        (qp.CNOT, ([0, 1],), {}),
        (qp.Toffoli, ([0, 1, 2],), {}),
        (qp.MultiControlledX, (), {"wires": [0, 1, 2, 3]}),
        (qp.CRX, (1.5, [0, 1]), {}),
        (qp.CRY, (1.5, [0, 1]), {}),
        (qp.CRZ, (1.5, [0, 1]), {}),
        (qp.CRot, (1.5, 2.5, 3.5, [0, 1]), {}),
        (qp.ControlledPhaseShift, (1.5, [0, 1]), {}),
    ],
)
def test_controlled_operator_handling(op_class, args, kwargs):
    """Test that PlxprInterpreter can handle controlled operators"""

    @PlxprInterpreter()
    def f():
        op_class(*args, **kwargs)
        return qp.expval(qp.Z(0))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[0].primitive == op_class._primitive


def test_default_measurement_handling():
    """Test that measurements are simply re-queued by default."""

    def f():
        return qp.expval(qp.Z(0) + qp.Z(0)), qp.probs(wires=0)

    jaxpr = jax.make_jaxpr(f)()
    with qp.queuing.AnnotatedQueue() as q:
        res1, res2 = PlxprInterpreter().eval(jaxpr.jaxpr, jaxpr.consts)
    assert len(q.queue) == 2
    assert q.queue[0] is res1
    assert q.queue[1] is res2
    qp.assert_equal(res1, qp.expval(qp.Z(0) + qp.Z(0)))
    qp.assert_equal(res2, qp.probs(wires=0))


def test_measurement_handling():
    """Test that the default measurement handling works."""

    @SimplifyInterpreter()
    def f(w):
        return qp.expval(qp.X(w) + qp.X(w)), qp.probs(wires=w)

    m1, m2 = f(0)
    qp.assert_equal(m1, qp.expval(2 * qp.X(0)))
    qp.assert_equal(m2, qp.probs(wires=0))

    jaxpr = jax.make_jaxpr(f)(0)

    assert jaxpr.eqns[0].primitive == qp.X._primitive
    assert jaxpr.eqns[1].primitive == qp.ops.SProd._primitive
    assert jaxpr.eqns[2].primitive == qp.measurements.ExpectationMP._obs_primitive
    assert jaxpr.eqns[3].primitive == qp.measurements.ProbabilityMP._wires_primitive

    m1, m2 = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)
    qp.assert_equal(m1, qp.expval(2 * qp.X(0)))
    qp.assert_equal(m2, qp.probs(wires=0))


def test_call_with_pytree_arguments():
    """Test that pytree arguments are correctly flattened when calling a
    decorated function"""

    @PlxprInterpreter()
    def f(angles):
        qp.Rot(*angles["1"], 0)
        qp.Rot(*angles["2"], 0)
        return qp.state()

    args = ({"1": (1.0, 2.0, 3.0), "2": (4.0, 5.0, 6.0)},)
    jaxpr = jax.make_jaxpr(f)(*args)

    assert len(jaxpr.jaxpr.invars) == 6
    expected_primitives = [
        qp.Rot._primitive,
        qp.Rot._primitive,
        qp.measurements.StateMP._wires_primitive,
    ]
    assert all(eqn.primitive == ep for eqn, ep in zip(jaxpr.eqns, expected_primitives))

    assert jaxpr.eqns[0].invars[0:3] == jaxpr.jaxpr.invars[0:3]
    assert jaxpr.eqns[1].invars[0:3] == jaxpr.jaxpr.invars[3:]


def test_overriding_measurements():
    """Test usage of an interpreter with a custom way of handling measurements."""

    class MeasurementsToSample(PlxprInterpreter):

        def interpret_measurement(self, measurement):
            return qp.sample(wires=measurement.wires)

    @MeasurementsToSample()
    @qp.qnode(qp.device("default.qubit", wires=2), shots=5)
    def circuit():
        return qp.expval(qp.Z(0)), qp.probs(wires=(0, 1))

    res = circuit()
    assert qp.math.allclose(res[0], jax.numpy.zeros(5))
    assert qp.math.allclose(res[1], jax.numpy.zeros((5, 2)))

    jaxpr = jax.make_jaxpr(circuit)()
    assert (
        jaxpr.eqns[0].params["qfunc_jaxpr"].eqns[0].primitive
        == qp.measurements.SampleMP._wires_primitive
    )
    assert (
        jaxpr.eqns[0].params["qfunc_jaxpr"].eqns[1].primitive
        == qp.measurements.SampleMP._wires_primitive
    )


def test_setup_method():
    """Test that the setup method can be used to initialize variables at each call."""

    class CollectOps(PlxprInterpreter):

        ops = None

        def setup(self):
            self.ops = []

        def interpret_operation(self, op):
            self.ops.append(op)
            return op._unflatten(*op._flatten())

    def f(x):
        qp.RX(x, 0)
        qp.RY(2 * x, 0)

    jaxpr = jax.make_jaxpr(f)(0.5)
    inst = CollectOps()
    inst.eval(jaxpr.jaxpr, jaxpr.consts, 1.2)
    assert inst.ops
    assert len(inst.ops) == 2
    qp.assert_equal(inst.ops[0], qp.RX(1.2, 0))
    qp.assert_equal(inst.ops[1], qp.RY(jnp.array(2.4), 0))

    # refreshed if instance is re-used
    inst.eval(jaxpr.jaxpr, jaxpr.consts, -0.5)
    assert len(inst.ops) == 2
    qp.assert_equal(inst.ops[0], qp.RX(-0.5, 0))
    qp.assert_equal(inst.ops[1], qp.RY(jnp.array(-1.0), 0))


def test_cleanup_method():
    """Test that the cleanup method can be used to reset variables after evaluation."""

    class CleanupTester(PlxprInterpreter):

        state = "DEFAULT"

        def setup(self):
            self.state = "SOME LARGE MEMORY"

        def cleanup(self):
            self.state = None

    inst = CleanupTester()

    @inst
    def f(x):
        qp.RX(x, 0)

    f(0.5)
    assert inst.state is None


def test_returning_operators():
    """Test that operators that are returned are still processed by the interpreter."""

    @SimplifyInterpreter()
    def f():
        return qp.X(0) ** 2

    qp.assert_equal(f(), qp.I(0))


class ConstAdder(PlxprInterpreter):
    """This interpreter, along with the add_3 primitive below, will be used to test
    that consts propagate through higher order primitives correctly."""


add_3 = jax.extend.core.Primitive("add_3")
scalar = jnp.array(3)


@add_3.def_impl
def add_3_impl(x):
    return x + 3


@add_3.def_abstract_eval
def add_3_aval(_):
    return jax.core.ShapedArray((), int)


@ConstAdder.register_primitive(add_3)
def handle_add_3(self, x):  # pylint: disable=unused-argument
    """This custom registration adds a closure variable to the input to register it as
    a const rather than a jax.extend.core.Literal."""
    return x + scalar


class TestHigherOrderPrimitiveRegistrations:

    @pytest.mark.parametrize("lazy", (True, False))
    def test_adjoint_transform(self, lazy):
        """Test the higher order adjoint transform."""

        @SimplifyInterpreter()
        def f(x):
            def g(y):
                _ = qp.RX(y, 0) ** 3

            qp.adjoint(g, lazy=lazy)(x)

        jaxpr = jax.make_jaxpr(f)(0.5)

        assert jaxpr.eqns[0].params["lazy"] == lazy
        assert jaxpr.eqns[0].primitive == adjoint_transform_prim
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        # first eqn mul, second RX
        assert inner_jaxpr.eqns[1].primitive == qp.RX._primitive
        assert len(inner_jaxpr.eqns) == 2

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)

        if lazy:
            qp.assert_equal(q.queue[0], qp.adjoint(qp.RX(jax.numpy.array(1.5), 0)))
        else:
            qp.assert_equal(q.queue[0], qp.RX(jax.numpy.array(-1.5), 0))

    @pytest.mark.parametrize("lazy", (True, False))
    def test_adjoint_consts(self, lazy):
        """Test that consts propagate correctly when interpreting the adjoint primitive."""

        def f(x):
            def g(y):
                # One new const
                exponent = add_3.bind(0)
                _ = qp.RX(y, 0) ** exponent

            qp.adjoint(g, lazy=lazy)(x)

        jaxpr = jax.make_jaxpr(f)(0.5)
        assert len(jaxpr.consts) == 0
        assert len(jaxpr.eqns[0].params["jaxpr"].constvars) == 0

        jaxpr2 = jax.make_jaxpr(ConstAdder()(f))(0.5)
        assert jaxpr2.consts == [scalar]
        assert len(jaxpr2.eqns[0].params["jaxpr"].constvars) == 1

    def test_ctrl_transform(self):
        """Test the higher order ctrl transform."""

        @SimplifyInterpreter()
        def f(x, control):
            def g(y):
                _ = qp.RY(y, 0) ** 3

            qp.ctrl(g, control)(x)

        jaxpr = jax.make_jaxpr(f)(0.5, 1)

        assert jaxpr.eqns[0].primitive == ctrl_transform_prim
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        # first eqn mul, second RY
        assert inner_jaxpr.eqns[1].primitive == qp.RY._primitive
        assert len(inner_jaxpr.eqns) == 2

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2.0, 1)

        qp.assert_equal(q.queue[0], qp.ctrl(qp.RY(jax.numpy.array(6.0), 0), 1))

    def test_ctrl_consts(self):
        """Test that consts propagate correctly when interpreting the ctrl primitive."""

        def f(x, control):
            def g(y):
                # One new const
                exponent = add_3.bind(0)
                _ = qp.RX(y, 0) ** exponent

            qp.ctrl(g, control)(x)

        jaxpr = jax.make_jaxpr(f)(0.5, 1)
        assert len(jaxpr.consts) == 0
        assert len(jaxpr.eqns[0].params["jaxpr"].constvars) == 0

        jaxpr2 = jax.make_jaxpr(ConstAdder()(f))(0.5, 1)
        assert jaxpr2.consts == [scalar]
        assert len(jaxpr2.eqns[0].params["jaxpr"].constvars) == 1

    def test_cond(self):
        """Test the cond higher order primitive."""

        @SimplifyInterpreter()
        def f(x, control):

            def true_fn(y):
                _ = qp.RY(y, 0) ** 2

            def false_fn(y):
                _ = qp.adjoint(qp.RX(y, 0))

            qp.cond(control, true_fn, false_fn)(x)

        jaxpr = jax.make_jaxpr(f)(0.5, False)
        assert jaxpr.eqns[0].primitive == cond_prim

        branch1 = jaxpr.eqns[0].params["jaxpr_branches"][0]
        assert len(branch1.eqns) == 2
        assert branch1.eqns[1].primitive == qp.RY._primitive
        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(branch1, [], 0.5)
        qp.assert_equal(q.queue[0], qp.RY(2 * jax.numpy.array(0.5), 0))

        branch2 = jaxpr.eqns[0].params["jaxpr_branches"][1]
        assert len(branch2.eqns) == 2
        assert branch2.eqns[1].primitive == qp.RX._primitive
        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(branch2, [], 0.5)
        qp.assert_equal(q.queue[0], qp.RX(jax.numpy.array(-0.5), 0))

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2.4, True)

        qp.assert_equal(q.queue[0], qp.RY(jax.numpy.array(4.8), 0))

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.23, False)

        qp.assert_equal(q.queue[0], qp.RX(jax.numpy.array(-1.23), 0))

    def test_cond_no_false_branch(self):
        """Test transforming a cond HOP when no false branch exists."""

        @SimplifyInterpreter()
        def f(control):

            @qp.cond(control)
            def f():
                _ = qp.X(0) @ qp.X(0)

            f()

        jaxpr = jax.make_jaxpr(f)(True)

        false_branch = jaxpr.eqns[0].params["jaxpr_branches"][-1]
        assert len(false_branch.eqns) == 0

        with qp.queuing.AnnotatedQueue() as q_true:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, True)

        qp.assert_equal(q_true.queue[0], qp.I(0))

        with qp.queuing.AnnotatedQueue() as q_false:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, False)

        assert len(q_false.queue) == 0

    def test_cond_consts(self):
        """Test that consts propagate correctly when interpreting the cond primitive."""

        def f(x, control):

            @qp.cond(control)
            def cond_fn(y):
                # One new const
                exponent = add_3.bind(0)
                _ = qp.RY(y, 0) ** exponent

            @cond_fn.otherwise
            def _(y):
                # Zero consts
                _ = qp.RX(y, 0)

            cond_fn(x)

        jaxpr = jax.make_jaxpr(f)(0.5, 1)
        assert len(jaxpr.consts) == 0
        assert len(jaxpr.eqns[0].params["jaxpr_branches"][0].constvars) == 0
        assert len(jaxpr.eqns[0].params["jaxpr_branches"][1].constvars) == 0

        jaxpr2 = jax.make_jaxpr(ConstAdder()(f))(0.5, 1)
        assert jaxpr2.consts == [scalar]
        assert len(jaxpr2.eqns[0].params["jaxpr_branches"][0].constvars) == 1
        assert len(jaxpr2.eqns[0].params["jaxpr_branches"][1].constvars) == 0

    def test_for_loop(self):
        """Test the higher order for loop registration."""

        @SimplifyInterpreter()
        def f(n):

            @qp.for_loop(n)
            def g(i):
                qp.adjoint(qp.X(i))

            g()

        jaxpr = jax.make_jaxpr(f)(3)
        assert jaxpr.eqns[0].primitive == for_loop_prim

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 1
        assert inner_jaxpr.eqns[0].primitive == qp.X._primitive  # no adjoint of x

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)

        qp.assert_equal(q.queue[0], qp.X(0))
        qp.assert_equal(q.queue[1], qp.X(1))
        qp.assert_equal(q.queue[2], qp.X(2))
        assert len(q) == 3

    def test_for_loop_consts(self):
        """Test the higher order for loop registration propagates consts correctly."""

        def f(n):

            @qp.for_loop(n)
            def g(i):
                exponent = add_3.bind(0)
                _ = qp.adjoint(qp.X(i)) ** exponent

            g()

        jaxpr = jax.make_jaxpr(f)(4)
        assert len(jaxpr.consts) == 0
        assert len(jaxpr.eqns[0].params["jaxpr_body_fn"].constvars) == 0

        jaxpr2 = jax.make_jaxpr(ConstAdder()(f))(4)
        assert jaxpr2.consts == [scalar]
        assert len(jaxpr2.eqns[0].params["jaxpr_body_fn"].constvars) == 1

    def test_while_loop(self):
        """Test the higher order for loop registration."""

        @SimplifyInterpreter()
        def f(n):

            @qp.while_loop(lambda i: i < n)
            def g(i):
                qp.adjoint(qp.Z(i))
                return i + 1

            g(0)

        jaxpr = jax.make_jaxpr(f)(3)
        assert jaxpr.eqns[0].primitive == while_loop_prim

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 2
        assert inner_jaxpr.eqns[0].primitive == qp.Z._primitive  # no adjoint of x

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)

        qp.assert_equal(q.queue[0], qp.Z(0))
        qp.assert_equal(q.queue[1], qp.Z(1))
        qp.assert_equal(q.queue[2], qp.Z(2))
        assert len(q) == 3

    def test_while_loop_consts(self):
        """Test the higher order while loop registration propagates consts correctly."""

        def f(n):

            @qp.while_loop(lambda i: i < add_3.bind(n))
            def g(i):
                exponent = add_3.bind(0)
                _ = qp.adjoint(qp.Z(i)) ** exponent
                return i + 1

            g(0)

        jaxpr = jax.make_jaxpr(f)(4)
        assert len(jaxpr.consts) == 0
        assert len(jaxpr.eqns[0].params["jaxpr_cond_fn"].constvars) == 1
        assert len(jaxpr.eqns[0].params["jaxpr_body_fn"].constvars) == 0

        jaxpr2 = jax.make_jaxpr(ConstAdder()(f))(4)
        assert jaxpr2.consts == [scalar]
        assert len(jaxpr2.eqns[0].params["jaxpr_cond_fn"].constvars) == 2
        assert len(jaxpr2.eqns[0].params["jaxpr_body_fn"].constvars) == 1

    def test_qnode(self):
        """Test transforming qnodes."""

        class AddNoise(PlxprInterpreter):

            def interpret_operation(self, op):
                new_op = op._unflatten(*op._flatten())
                _ = [qp.RX(0.1, w) for w in op.wires]
                return new_op

        dev = qp.device("default.qubit", wires=1)

        @AddNoise()
        @qp.qnode(dev, diff_method="backprop", grad_on_execution=False)
        def f():
            qp.I(0)
            qp.I(0)
            return qp.probs(wires=0)

        jaxpr = jax.make_jaxpr(f)()
        assert jaxpr.eqns[0].primitive == qnode_prim
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

        assert len(inner_jaxpr.eqns) == 5
        assert inner_jaxpr.eqns[0].primitive == qp.I._primitive
        assert inner_jaxpr.eqns[2].primitive == qp.I._primitive
        assert inner_jaxpr.eqns[1].primitive == qp.RX._primitive
        assert inner_jaxpr.eqns[3].primitive == qp.RX._primitive

        assert jaxpr.eqns[0].params["execution_config"].gradient_method == "backprop"
        assert jaxpr.eqns[0].params["execution_config"].grad_on_execution is False
        assert jaxpr.eqns[0].params["device"] == dev

        res1 = f()
        # end up performing two rx gates with phase of 0.1 each on wire 0
        expected = jax.numpy.array([jax.numpy.cos(0.2 / 2) ** 2, jax.numpy.sin(0.2 / 2) ** 2])
        assert qp.math.allclose(res1, expected)
        res2 = jax.core.eval_jaxpr(jaxpr.jaxpr, [])
        assert qp.math.allclose(res2, expected)

    def test_qnode_consts(self):
        """Test the higher order qnode registration propagates consts correctly."""

        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev, diff_method="backprop", grad_on_execution=False)
        def f():
            exponent = add_3.bind(0)
            _ = qp.X(0) ** exponent
            _ = qp.I(0)
            return qp.probs(wires=0)

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.consts) == 0
        assert len(jaxpr.eqns[0].params["qfunc_jaxpr"].constvars) == 0

        jaxpr2 = jax.make_jaxpr(ConstAdder()(f))()
        assert jaxpr2.consts == [scalar]
        assert len(jaxpr2.eqns[0].params["qfunc_jaxpr"].constvars) == 1

    @pytest.mark.parametrize("grad_f", (qp.grad, qp.jacobian))
    def test_grad_and_jac(self, grad_f):
        """Test interpreters can handle grad and jacobian HOP's."""

        @SimplifyInterpreter()
        def f(x):
            @qp.qnode(qp.device("default.qubit", wires=2))
            def circuit(y):
                _ = qp.RX(y, 0) ** 2
                return qp.expval(qp.Z(0) + qp.Z(0))

            return grad_f(circuit)(x)

        jaxpr = jax.make_jaxpr(f)(0.5)

        assert jaxpr.eqns[0].primitive == qp.capture.primitives.jacobian_prim
        assert jaxpr.eqns[0].params["scalar_out"] == (grad_f == qp.grad)
        grad_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        qfunc_jaxpr = grad_jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[1].primitive == qp.RX._primitive  # eqn 0 is mul
        assert qfunc_jaxpr.eqns[2].primitive == qp.Z._primitive
        assert qfunc_jaxpr.eqns[3].primitive == qp.ops.SProd._primitive

    @pytest.mark.parametrize("grad_f", (qp.grad, qp.jacobian))
    def test_grad_and_jac_consts(self, grad_f):
        """Test interpreters can handle grad and jacobian HOP's and propagate consts correctly."""

        @SimplifyInterpreter()
        def f(x):
            @qp.qnode(qp.device("default.qubit", wires=2))
            def circuit(y):
                exponent = add_3.bind(0)
                _ = qp.RX(y, 0) ** exponent
                return qp.expval(qp.Z(0) + qp.Z(0))

            return grad_f(circuit)(x)

        jaxpr = jax.make_jaxpr(f)(0.5)
        assert len(jaxpr.consts) == 0
        assert len(jaxpr.eqns[0].params["jaxpr"].constvars) == 0

        jaxpr2 = jax.make_jaxpr(ConstAdder()(f))(0.5)
        assert jaxpr2.consts == [scalar]
        assert len(jaxpr2.eqns[0].params["jaxpr"].constvars) == 1


@pytest.mark.usefixtures("enable_disable_dynamic_shapes")
class TestDynamicShapes:
    """Test that our interpreters can handle dynamic array creation."""

    @pytest.mark.parametrize("reinterpret", (True, False))
    @pytest.mark.parametrize(
        "creation_fn", [jax.numpy.ones, jax.numpy.zeros, lambda s: jax.numpy.full(s, 0.5)]
    )
    def test_broadcast_in_dim(self, reinterpret, creation_fn):
        """Test that broadcast_in_dim can be executed with PlxprInterpreter."""

        def f(n):
            return 2 * creation_fn((2, n + 1))

        interpreter = PlxprInterpreter()

        if reinterpret:
            # can still capture it once again
            f = interpreter(f)

        jaxpr = jax.make_jaxpr(f)(3)

        output = interpreter.eval(jaxpr.jaxpr, jaxpr.consts, 4)

        assert len(output) == 2  # shape and array
        assert jax.numpy.allclose(output[0], 5)  # 4 + 1
        assert jax.numpy.allclose(output[1], 2 * creation_fn((2, 5)))

    @pytest.mark.parametrize("reinterpret", (True, False))
    def test_arange(self, reinterpret):
        """Test that broadcast_in_dim can be executed with PlxprInterpreter."""

        def f(n):
            return 2 * jax.numpy.arange(n + 1)

        interpreter = PlxprInterpreter()

        if reinterpret:
            # can still capture it once again
            f = interpreter(f)

        jaxpr = jax.make_jaxpr(f)(3)

        output = interpreter.eval(jaxpr.jaxpr, jaxpr.consts, 6)

        assert len(output) == 2  # shape and array
        assert jax.numpy.allclose(output[0], 7)  # 4 + 1
        assert jax.numpy.allclose(output[1], 2 * jax.numpy.arange(7))

    @pytest.mark.xfail  # v0.5.3 broke the ability to capture this
    def test_hstack(self):
        """Test that eval_jaxpr can handle the hstack primitive. hstack primitive produces a pjit equation,
        which currently does not work with dynamic shapes."""

        def f(i):
            x = jnp.zeros(i, int)
            y = jnp.ones(i, int)
            return jnp.hstack((x, y))

        jaxpr = jax.make_jaxpr(f)(2)
        [shape, res] = qp.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 4)
        assert qp.math.allclose(shape, 8)
        assert qp.math.allclose(res, jnp.hstack((jnp.zeros(4), jnp.ones(4))))
