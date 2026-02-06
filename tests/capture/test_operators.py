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
Integration tests for the capture of pennylane operations into jaxpr.
"""
import numpy as np

# pylint: disable=protected-access
import pytest

import pennylane as qp

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import AbstractOperator  # pylint: disable=wrong-import-position

pytestmark = [pytest.mark.jax, pytest.mark.capture]


def test_abstract_operator():
    """Perform unit tests on the abstract operator class."""

    ao1 = AbstractOperator()
    ao2 = AbstractOperator()
    assert ao1 == ao2
    assert hash(ao1) == hash(ao2)

    with pytest.raises(NotImplementedError):
        ao1.update()

    with pytest.raises(NotImplementedError):
        ao1.join(ao2)

    with pytest.raises(NotImplementedError):
        ao1.at_least_vspace()

    # arithmetic dunders integration tested


def test_operators_constructed_when_plxpr_enabled():
    """Test that normal operators can still be constructed when plxpr is enabled."""

    with qp.queuing.AnnotatedQueue() as q:
        op = qp.adjoint(qp.X(0) + qp.Y(1))

    assert len(q) == 1
    assert q.queue[0] is op
    assert isinstance(op, qp.ops.Adjoint)
    assert isinstance(op.base, qp.ops.Sum)
    assert op.base[0] == qp.X(0)
    assert op.base[1] == qp.Y(1)


def test_fallback_if_primitive_still_None():
    """Test that if the primitive is None (no jax or something went wrong) that the instance is simply created."""

    # pylint: disable=too-few-public-methods
    class MyOp(qp.operation.Operator):
        """A dummy operator."""

    MyOp._primitive = None

    op = MyOp(wires=0)
    assert isinstance(op, qp.operation.Operator)

    def f():
        MyOp(wires=0)

    jaxpr = jax.make_jaxpr(f)()
    assert len(jaxpr.eqns) == 0


def test_hybrid_capture_wires():
    """That a hybrid quantum-classical jaxpr can be captured with wire processing."""

    def f(a, b):
        qp.X(a + b)

    jaxpr = jax.make_jaxpr(f)(1, 2)
    assert len(jaxpr.eqns) == 2

    assert jaxpr.eqns[0].primitive.name == "add"

    assert jaxpr.eqns[0].outvars == jaxpr.eqns[1].invars
    assert jaxpr.eqns[1].primitive == qp.X._primitive

    with qp.queuing.AnnotatedQueue() as q:
        jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1, 2)

    assert len(q) == 1
    qp.assert_equal(q.queue[0], qp.X(3))


def test_hybrid_capture_parametrization():
    """Test a variety of classical processing with a parametrized operation."""

    def f(a):
        qp.Rot(2 * a, jax.numpy.sqrt(a), a**2, wires=2 * a)

    jaxpr = jax.make_jaxpr(f)(0.5)
    assert len(jaxpr.eqns) == 5

    in1 = jaxpr.jaxpr.invars[0]
    assert jaxpr.eqns[0].invars[1] == in1
    assert jaxpr.eqns[1].invars[0] == in1
    assert jaxpr.eqns[2].invars[0] == in1
    assert jaxpr.eqns[3].invars[-1] == in1  # the wire

    assert jaxpr.eqns[0].primitive.name == "mul"
    assert jaxpr.eqns[1].primitive.name == "sqrt"
    assert jaxpr.eqns[2].primitive.name == "integer_pow"
    assert jaxpr.eqns[3].primitive.name == "mul"
    assert jaxpr.eqns[4].primitive == qp.Rot._primitive

    with qp.queuing.AnnotatedQueue() as q:
        jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)

    assert len(q) == 1
    qp.assert_equal(
        q.queue[0], qp.Rot(1.0, jax.numpy.sqrt(0.5), 0.25, wires=1), check_interface=False
    )


@pytest.mark.parametrize("as_kwarg", (True, False))
@pytest.mark.parametrize(
    "w",
    (
        0,
        (0,),
        [0],
        range(1),
        qp.wires.Wires(0),
        {0},
        jax.numpy.array(0),
        jax.numpy.array([0]),
        np.array(0),
        np.array([0]),
    ),
)
@pytest.mark.parametrize("autograph", (True, False))
def test_different_wires(w, as_kwarg, autograph):
    """Test that wires can be passed positionally and as a keyword in a variety of different types."""

    if as_kwarg:

        def qfunc():
            qp.X(wires=w)

    else:

        def qfunc():
            qp.X(w)

    if autograph:
        qfunc = qp.capture.run_autograph(qfunc)

    jaxpr = jax.make_jaxpr(qfunc)()

    if isinstance(w, jax.numpy.ndarray) and w.shape != ():
        offset = 1
    else:
        offset = 0

    assert len(jaxpr.eqns) == 1 + offset

    eqn = jaxpr.eqns[offset + 0]
    assert eqn.primitive == qp.X._primitive
    assert len(eqn.invars) == 1
    if not isinstance(w, jax.numpy.ndarray):
        assert isinstance(eqn.invars[0], jax.extend.core.Literal)
        assert eqn.invars[0].val == 0

    assert isinstance(eqn.outvars[0].aval, AbstractOperator)
    assert isinstance(eqn.outvars[0], jax.core.DropVar)

    assert eqn.params == {"n_wires": 1}

    with qp.queuing.AnnotatedQueue() as q:
        jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

    assert len(q) == 1
    qp.assert_equal(q.queue[0], qp.X(0))


@pytest.mark.parametrize("as_kwarg", (True, False))
@pytest.mark.parametrize("interface", ("numpy", "jax"))
def test_ndarray_multiple_wires(as_kwarg, interface):
    """Test that wires can be provided as an ndarray."""

    def qfunc():
        if as_kwarg:
            qp.GroverOperator(wires=qp.math.arange(4, like=interface))
        else:
            qp.GroverOperator(qp.math.arange(4, like=interface))

    jaxpr = jax.make_jaxpr(qfunc)()

    assert jaxpr.eqns[-1].primitive == qp.GroverOperator._primitive
    assert jaxpr.eqns[-1].params == {"n_wires": 4}
    assert len(jaxpr.eqns[-1].invars) == 4

    with qp.queuing.AnnotatedQueue() as q:
        qp.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

    assert len(q) == 1
    qp.assert_equal(q.queue[0], qp.GroverOperator(wires=(0, 1, 2, 3)))


def test_parametrized_op():
    """Test capturing a parametrized operation."""

    jaxpr = jax.make_jaxpr(qp.Rot)(1.0, 2.0, 3.0, 10)

    assert len(jaxpr.eqns) == 1
    eqn = jaxpr.eqns[0]

    assert eqn.primitive == qp.Rot._primitive
    assert len(eqn.invars) == 4
    assert jaxpr.jaxpr.invars == jaxpr.eqns[0].invars

    assert isinstance(eqn.outvars[0].aval, AbstractOperator)
    assert eqn.params == {"n_wires": 1}

    with qp.queuing.AnnotatedQueue() as q:
        jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.0, 2.0, 3.0, 10)

    assert len(q) == 1
    qp.assert_equal(q.queue[0], qp.Rot(1.0, 2.0, 3.0, 10))


class TestSpecialOps:

    def test_pauli_rot(self):
        """Test a special operation that has positional metadata and overrides binding."""

        def qfunc(a, wire0, wire1):
            qp.PauliRot(a, "XY", (wire0, wire1))

        jaxpr = jax.make_jaxpr(qfunc)(0.5, 2, 3)
        assert len(jaxpr.eqns) == 1
        eqn = jaxpr.eqns[0]

        assert eqn.primitive == qp.PauliRot._primitive
        assert eqn.params == {"pauli_word": "XY", "id": None, "n_wires": 2}

        assert len(eqn.invars) == 3  # The rotation parameter and the two wires
        assert jaxpr.jaxpr.invars == eqn.invars

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2.5, 3, 4)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.PauliRot(2.5, "XY", (3, 4)))

    def test_GlobalPhase(self):
        """Test that a global phase on no wires can be captured."""

        def qfunc(phi):
            return qp.GlobalPhase(phi)

        jaxpr = jax.make_jaxpr(qfunc)(0.5)
        assert len(jaxpr.eqns) == 1

        assert jaxpr.eqns[0].primitive == qp.GlobalPhase._primitive
        assert len(jaxpr.eqns[0].invars) == 1
        assert jaxpr.eqns[0].params == {"n_wires": 0}

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.2)

        assert len(q.queue) == 1
        qp.assert_equal(q.queue[0], qp.GlobalPhase(1.2))

    def test_identity_no_wires(self):
        """Test that an identity on no wires can be captured."""

        jaxpr = jax.make_jaxpr(qp.I)()
        assert len(jaxpr.eqns) == 1

        assert jaxpr.eqns[0].primitive == qp.I._primitive
        assert len(jaxpr.eqns[0].invars) == 0
        assert jaxpr.eqns[0].params == {"n_wires": 0}

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q.queue) == 1
        qp.assert_equal(q.queue[0], qp.I())


class TestTemplates:

    def test_variable_wire_non_parametrized_template(self):
        """Test capturing a variable wire count, non-parametrized template like GroverOperator."""

        jaxpr = jax.make_jaxpr(qp.GroverOperator)(wires=(0, 1, 2, 3, 4, 5))

        assert len(jaxpr.eqns) == 1
        eqn = jaxpr.eqns[0]

        assert eqn.primitive == qp.GroverOperator._primitive
        assert eqn.params == {"n_wires": 6}
        assert eqn.invars == jaxpr.jaxpr.invars
        assert isinstance(eqn.outvars[0].aval, AbstractOperator)

    def test_nested_template(self):
        """Test capturing a template that depends on another operator."""

        def qfunc(coeffs):
            ops = [qp.X(0), qp.Z(0)]
            H = qp.dot(coeffs, ops)
            qp.TrotterProduct(H, time=2.4, order=2)

        coeffs = [0.25, 0.75]

        jaxpr = jax.make_jaxpr(qfunc)(coeffs)

        assert len(jaxpr.eqns) == 6

        assert jaxpr.eqns[0].primitive == qp.X._primitive
        assert jaxpr.eqns[1].primitive == qp.Z._primitive
        assert jaxpr.eqns[2].primitive == qp.ops.SProd._primitive
        assert jaxpr.eqns[3].primitive == qp.ops.SProd._primitive
        assert jaxpr.eqns[4].primitive == qp.ops.Sum._primitive
        assert not any(isinstance(eqn.outvars[0], jax.core.DropVar) for eqn in jaxpr.eqns[:5])

        eqn = jaxpr.eqns[5]
        assert eqn.primitive == qp.TrotterProduct._primitive
        assert eqn.invars == jaxpr.eqns[4].outvars  # the sum op

        assert eqn.params == {"order": 2, "time": 2.4}

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, coeffs[0], coeffs[1])

        assert len(q) == 1
        ops = [qp.X(0), qp.Z(0)]
        H = qp.dot(coeffs, ops)
        assert q.queue[0] == qp.TrotterProduct(H, time=2.4, order=2)


class TestOpmath:
    """Tests for capturing operator arithmetic."""

    def test_adjoint(self):
        """Test the adjoint on an op can be captured."""

        jaxpr = jax.make_jaxpr(qp.adjoint)(qp.X(0))

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qp.X._primitive

        eqn = jaxpr.eqns[1]
        assert eqn.primitive == qp.ops.Adjoint._primitive
        assert eqn.invars == jaxpr.eqns[0].outvars  # the pauli x op
        assert isinstance(eqn.outvars[0].aval, AbstractOperator)
        assert eqn.params == {}

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.adjoint(qp.X(0)))

    def test_adjoint_op_outside_qfunc(self):
        """Test that an op can be constructed outside a function and still be adjointed."""

        op = qp.X(0)

        def f():
            qp.adjoint(op)

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qp.X._primitive

        eqn = jaxpr.eqns[1]
        assert eqn.primitive == qp.ops.Adjoint._primitive
        assert eqn.invars == jaxpr.eqns[0].outvars  # the pauli x op
        assert isinstance(eqn.outvars[0].aval, AbstractOperator)
        assert eqn.params == {}

    def test_Controlled(self):
        """Test a nested control operation."""

        def qfunc(op):
            qp.ctrl(op, control=(3, 4), control_values=[0, 1])

        jaxpr = jax.make_jaxpr(qfunc)(qp.IsingXX(1.2, wires=(0, 1)))

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qp.IsingXX._primitive

        eqn = jaxpr.eqns[1]
        assert eqn.primitive == qp.ops.Controlled._primitive
        assert eqn.invars[0] == jaxpr.eqns[0].outvars[0]  # the isingxx
        assert eqn.invars[1].val == 3
        assert eqn.invars[2].val == 4

        assert isinstance(eqn.outvars[0].aval, AbstractOperator)
        assert eqn.params == {
            "control_values": (0, 1),
            "work_wires": None,
            "work_wire_type": "borrowed",
        }

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3.4)

        assert len(q) == 1
        expected = qp.ctrl(qp.IsingXX(3.4, wires=(0, 1)), control=(3, 4), control_values=[0, 1])
        qp.assert_equal(q.queue[0], expected)

    def test_ctrl_op_constructed_outside_qfunc(self):
        """Test an op constructed outside the qfunc can be controlled."""

        op = qp.IsingXX(1.2, wires=(0, 1))

        def f():
            qp.ctrl(op, control=(3, 4), control_values=[0, 1])

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qp.IsingXX._primitive

        eqn = jaxpr.eqns[1]
        assert eqn.primitive == qp.ops.Controlled._primitive
        assert eqn.invars[0] == jaxpr.eqns[0].outvars[0]  # the isingxx
        assert eqn.invars[1].val == 3
        assert eqn.invars[2].val == 4

        assert isinstance(eqn.outvars[0].aval, AbstractOperator)
        assert eqn.params == {
            "control_values": (0, 1),
            "work_wires": None,
            "work_wire_type": "borrowed",
        }


class TestAbstractDunders:
    """Test that operator dunders work when capturing."""

    def test_add(self):
        """Test that the add dunder works."""

        def qfunc():
            return qp.X(0) + qp.Y(1)

        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qp.X._primitive
        assert jaxpr.eqns[1].primitive == qp.Y._primitive

        eqn = jaxpr.eqns[2]

        assert eqn.primitive == qp.ops.Sum._primitive
        assert eqn.invars[0] == jaxpr.eqns[0].outvars[0]
        assert eqn.invars[1] == jaxpr.eqns[1].outvars[0]

        assert eqn.params == {"grouping_type": None, "id": None, "method": "lf"}

        assert isinstance(eqn.outvars[0].aval, AbstractOperator)

    def test_matmul(self):
        """Test that the matmul dunder works."""

        def qfunc():
            return qp.X(0) @ qp.Y(1)

        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qp.X._primitive
        assert jaxpr.eqns[1].primitive == qp.Y._primitive

        eqn = jaxpr.eqns[2]

        assert eqn.primitive == qp.ops.Prod._primitive
        assert eqn.invars[0] == jaxpr.eqns[0].outvars[0]
        assert eqn.invars[1] == jaxpr.eqns[1].outvars[0]

        assert eqn.params == {"id": None}

        assert isinstance(eqn.outvars[0].aval, AbstractOperator)

    def test_mul(self):
        """Test that the scalar multiplication dunder works."""

        def qfunc():
            return 2 * qp.Y(1) * 3

        jaxpr = jax.make_jaxpr(qfunc)()
        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[0].primitive == qp.Y._primitive

        assert jaxpr.eqns[1].primitive == qp.ops.SProd._primitive
        assert jaxpr.eqns[1].invars[0].val == 2
        assert jaxpr.eqns[1].invars[1] == jaxpr.eqns[0].outvars[0]  # the y from the previous step

        assert jaxpr.eqns[2].primitive == qp.ops.SProd._primitive
        assert jaxpr.eqns[2].invars[0].val == 3
        assert (
            jaxpr.eqns[2].invars[1] == jaxpr.eqns[1].outvars[0]
        )  # the sprod from the previous step

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        assert q.queue[0] == 3 * 2 * qp.Y(1)

    def test_pow(self):
        """Test that abstract operators can be raised to powers."""

        def qfunc(z):
            return qp.IsingZZ(z, (0, 1)) ** 2

        jaxpr = jax.make_jaxpr(qfunc)(1.2)

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qp.IsingZZ._primitive
        assert jaxpr.eqns[1].primitive == qp.ops.Pow._primitive

        assert jaxpr.eqns[1].invars[0] == jaxpr.eqns[0].outvars[0]
        assert jaxpr.eqns[1].invars[1].val == 2
