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

import pennylane as qml

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

    with qml.queuing.AnnotatedQueue() as q:
        op = qml.adjoint(qml.X(0) + qml.Y(1))

    assert len(q) == 1
    assert q.queue[0] is op
    assert isinstance(op, qml.ops.Adjoint)
    assert isinstance(op.base, qml.ops.Sum)
    assert op.base[0] == qml.X(0)
    assert op.base[1] == qml.Y(1)


def test_fallback_if_primitive_still_None():
    """Test that if the primitive is None (no jax or something went wrong) that the instance is simply created."""

    # pylint: disable=too-few-public-methods
    class MyOp(qml.operation.Operator):
        """A dummy operator."""

    MyOp._primitive = None

    op = MyOp(wires=0)
    assert isinstance(op, qml.operation.Operator)

    def f():
        MyOp(wires=0)

    jaxpr = jax.make_jaxpr(f)()
    assert len(jaxpr.eqns) == 0


def test_hybrid_capture_wires():
    """That a hybrid quantum-classical jaxpr can be captured with wire processing."""

    def f(a, b):
        qml.X(a + b)

    jaxpr = jax.make_jaxpr(f)(1, 2)
    assert len(jaxpr.eqns) == 2

    assert jaxpr.eqns[0].primitive.name == "add"

    assert jaxpr.eqns[0].outvars == jaxpr.eqns[1].invars
    assert jaxpr.eqns[1].primitive == qml.X._primitive

    with qml.queuing.AnnotatedQueue() as q:
        jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1, 2)

    assert len(q) == 1
    qml.assert_equal(q.queue[0], qml.X(3))


def test_hybrid_capture_parametrization():
    """Test a variety of classical processing with a parametrized operation."""

    def f(a):
        qml.Rot(2 * a, jax.numpy.sqrt(a), a**2, wires=2 * a)

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
    assert jaxpr.eqns[4].primitive == qml.Rot._primitive

    with qml.queuing.AnnotatedQueue() as q:
        jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)

    assert len(q) == 1
    qml.assert_equal(
        q.queue[0], qml.Rot(1.0, jax.numpy.sqrt(0.5), 0.25, wires=1), check_interface=False
    )


@pytest.mark.parametrize("as_kwarg", (True, False))
@pytest.mark.parametrize(
    "w",
    (
        0,
        (0,),
        [0],
        range(1),
        qml.wires.Wires(0),
        {0},
        jax.numpy.array(0),
        jax.numpy.array([0]),
        np.array(0),
        np.array([0]),
    ),
)
def test_different_wires(w, as_kwarg):
    """Test that wires can be passed positionally and as a keyword in a variety of different types."""

    def qfunc():
        if as_kwarg:
            qml.X(wires=w)
        else:
            qml.X(w)

    jaxpr = jax.make_jaxpr(qfunc)()

    if isinstance(w, jax.numpy.ndarray) and w.shape != ():
        offset = 1
    else:
        offset = 0

    assert len(jaxpr.eqns) == 1 + offset

    eqn = jaxpr.eqns[offset + 0]
    assert eqn.primitive == qml.X._primitive
    assert len(eqn.invars) == 1
    if not isinstance(w, jax.numpy.ndarray):
        assert isinstance(eqn.invars[0], jax.extend.core.Literal)
        assert eqn.invars[0].val == 0

    assert isinstance(eqn.outvars[0].aval, AbstractOperator)
    assert isinstance(eqn.outvars[0], jax.core.DropVar)

    assert eqn.params == {"n_wires": 1}

    with qml.queuing.AnnotatedQueue() as q:
        jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

    assert len(q) == 1
    qml.assert_equal(q.queue[0], qml.X(0))


@pytest.mark.parametrize("as_kwarg", (True, False))
@pytest.mark.parametrize("interface", ("numpy", "jax"))
def test_ndarray_multiple_wires(as_kwarg, interface):
    """Test that wires can be provided as an ndarray."""

    def qfunc():
        if as_kwarg:
            qml.GroverOperator(wires=qml.math.arange(4, like=interface))
        else:
            qml.GroverOperator(qml.math.arange(4, like=interface))

    jaxpr = jax.make_jaxpr(qfunc)()

    assert jaxpr.eqns[-1].primitive == qml.GroverOperator._primitive
    assert jaxpr.eqns[-1].params == {"n_wires": 4}
    assert len(jaxpr.eqns[-1].invars) == 4

    with qml.queuing.AnnotatedQueue() as q:
        qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

    assert len(q) == 1
    qml.assert_equal(q.queue[0], qml.GroverOperator(wires=(0, 1, 2, 3)))


def test_parametrized_op():
    """Test capturing a parametrized operation."""

    jaxpr = jax.make_jaxpr(qml.Rot)(1.0, 2.0, 3.0, 10)

    assert len(jaxpr.eqns) == 1
    eqn = jaxpr.eqns[0]

    assert eqn.primitive == qml.Rot._primitive
    assert len(eqn.invars) == 4
    assert jaxpr.jaxpr.invars == jaxpr.eqns[0].invars

    assert isinstance(eqn.outvars[0].aval, AbstractOperator)
    assert eqn.params == {"n_wires": 1}

    with qml.queuing.AnnotatedQueue() as q:
        jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.0, 2.0, 3.0, 10)

    assert len(q) == 1
    qml.assert_equal(q.queue[0], qml.Rot(1.0, 2.0, 3.0, 10))


class TestSpecialOps:

    def test_pauli_rot(self):
        """Test a special operation that has positional metadata and overrides binding."""

        def qfunc(a, wire0, wire1):
            qml.PauliRot(a, "XY", (wire0, wire1))

        jaxpr = jax.make_jaxpr(qfunc)(0.5, 2, 3)
        assert len(jaxpr.eqns) == 1
        eqn = jaxpr.eqns[0]

        assert eqn.primitive == qml.PauliRot._primitive
        assert eqn.params == {"pauli_word": "XY", "id": None, "n_wires": 2}

        assert len(eqn.invars) == 3  # The rotation parameter and the two wires
        assert jaxpr.jaxpr.invars == eqn.invars

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2.5, 3, 4)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.PauliRot(2.5, "XY", (3, 4)))

    def test_GlobalPhase(self):
        """Test that a global phase on no wires can be captured."""

        def qfunc(phi):
            return qml.GlobalPhase(phi)

        jaxpr = jax.make_jaxpr(qfunc)(0.5)
        assert len(jaxpr.eqns) == 1

        assert jaxpr.eqns[0].primitive == qml.GlobalPhase._primitive
        assert len(jaxpr.eqns[0].invars) == 1
        assert jaxpr.eqns[0].params == {"n_wires": 0}

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.2)

        assert len(q.queue) == 1
        qml.assert_equal(q.queue[0], qml.GlobalPhase(1.2))

    def test_identity_no_wires(self):
        """Test that an identity on no wires can be captured."""

        jaxpr = jax.make_jaxpr(qml.I)()
        assert len(jaxpr.eqns) == 1

        assert jaxpr.eqns[0].primitive == qml.I._primitive
        assert len(jaxpr.eqns[0].invars) == 0
        assert jaxpr.eqns[0].params == {"n_wires": 0}

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q.queue) == 1
        qml.assert_equal(q.queue[0], qml.I())


class TestTemplates:

    def test_variable_wire_non_parametrized_template(self):
        """Test capturing a variable wire count, non-parametrized template like GroverOperator."""

        jaxpr = jax.make_jaxpr(qml.GroverOperator)(wires=(0, 1, 2, 3, 4, 5))

        assert len(jaxpr.eqns) == 1
        eqn = jaxpr.eqns[0]

        assert eqn.primitive == qml.GroverOperator._primitive
        assert eqn.params == {"n_wires": 6}
        assert eqn.invars == jaxpr.jaxpr.invars
        assert isinstance(eqn.outvars[0].aval, AbstractOperator)

    def test_nested_template(self):
        """Test capturing a template that depends on another operator."""

        def qfunc(coeffs):
            ops = [qml.X(0), qml.Z(0)]
            H = qml.dot(coeffs, ops)
            qml.TrotterProduct(H, time=2.4, order=2)

        coeffs = [0.25, 0.75]

        jaxpr = jax.make_jaxpr(qfunc)(coeffs)

        assert len(jaxpr.eqns) == 6

        assert jaxpr.eqns[0].primitive == qml.X._primitive
        assert jaxpr.eqns[1].primitive == qml.Z._primitive
        assert jaxpr.eqns[2].primitive == qml.ops.SProd._primitive
        assert jaxpr.eqns[3].primitive == qml.ops.SProd._primitive
        assert jaxpr.eqns[4].primitive == qml.ops.Sum._primitive
        assert not any(isinstance(eqn.outvars[0], jax.core.DropVar) for eqn in jaxpr.eqns[:5])

        eqn = jaxpr.eqns[5]
        assert eqn.primitive == qml.TrotterProduct._primitive
        assert eqn.invars == jaxpr.eqns[4].outvars  # the sum op

        assert eqn.params == {"order": 2, "time": 2.4}

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, coeffs[0], coeffs[1])

        assert len(q) == 1
        ops = [qml.X(0), qml.Z(0)]
        H = qml.dot(coeffs, ops)
        assert q.queue[0] == qml.TrotterProduct(H, time=2.4, order=2)


class TestOpmath:
    """Tests for capturing operator arithmetic."""

    def test_adjoint(self):
        """Test the adjoint on an op can be captured."""

        jaxpr = jax.make_jaxpr(qml.adjoint)(qml.X(0))

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qml.X._primitive

        eqn = jaxpr.eqns[1]
        assert eqn.primitive == qml.ops.Adjoint._primitive
        assert eqn.invars == jaxpr.eqns[0].outvars  # the pauli x op
        assert isinstance(eqn.outvars[0].aval, AbstractOperator)
        assert eqn.params == {}

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.adjoint(qml.X(0)))

    def test_Controlled(self):
        """Test a nested control operation."""

        def qfunc(op):
            qml.ctrl(op, control=(3, 4), control_values=[0, 1])

        jaxpr = jax.make_jaxpr(qfunc)(qml.IsingXX(1.2, wires=(0, 1)))

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qml.IsingXX._primitive

        eqn = jaxpr.eqns[1]
        assert eqn.primitive == qml.ops.Controlled._primitive
        assert eqn.invars[0] == jaxpr.eqns[0].outvars[0]  # the isingxx
        assert eqn.invars[1].val == 3
        assert eqn.invars[2].val == 4

        assert isinstance(eqn.outvars[0].aval, AbstractOperator)
        assert eqn.params == {
            "control_values": [0, 1],
            "work_wires": None,
            "work_wire_type": "borrowed",
        }

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3.4)

        assert len(q) == 1
        expected = qml.ctrl(qml.IsingXX(3.4, wires=(0, 1)), control=(3, 4), control_values=[0, 1])
        qml.assert_equal(q.queue[0], expected)


class TestAbstractDunders:
    """Test that operator dunders work when capturing."""

    def test_add(self):
        """Test that the add dunder works."""

        def qfunc():
            return qml.X(0) + qml.Y(1)

        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qml.X._primitive
        assert jaxpr.eqns[1].primitive == qml.Y._primitive

        eqn = jaxpr.eqns[2]

        assert eqn.primitive == qml.ops.Sum._primitive
        assert eqn.invars[0] == jaxpr.eqns[0].outvars[0]
        assert eqn.invars[1] == jaxpr.eqns[1].outvars[0]

        assert eqn.params == {"grouping_type": None, "id": None, "method": "lf"}

        assert isinstance(eqn.outvars[0].aval, AbstractOperator)

    def test_matmul(self):
        """Test that the matmul dunder works."""

        def qfunc():
            return qml.X(0) @ qml.Y(1)

        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qml.X._primitive
        assert jaxpr.eqns[1].primitive == qml.Y._primitive

        eqn = jaxpr.eqns[2]

        assert eqn.primitive == qml.ops.Prod._primitive
        assert eqn.invars[0] == jaxpr.eqns[0].outvars[0]
        assert eqn.invars[1] == jaxpr.eqns[1].outvars[0]

        assert eqn.params == {"id": None}

        assert isinstance(eqn.outvars[0].aval, AbstractOperator)

    def test_mul(self):
        """Test that the scalar multiplication dunder works."""

        def qfunc():
            return 2 * qml.Y(1) * 3

        jaxpr = jax.make_jaxpr(qfunc)()
        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[0].primitive == qml.Y._primitive

        assert jaxpr.eqns[1].primitive == qml.ops.SProd._primitive
        assert jaxpr.eqns[1].invars[0].val == 2
        assert jaxpr.eqns[1].invars[1] == jaxpr.eqns[0].outvars[0]  # the y from the previous step

        assert jaxpr.eqns[2].primitive == qml.ops.SProd._primitive
        assert jaxpr.eqns[2].invars[0].val == 3
        assert (
            jaxpr.eqns[2].invars[1] == jaxpr.eqns[1].outvars[0]
        )  # the sprod from the previous step

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        assert q.queue[0] == 3 * 2 * qml.Y(1)

    def test_pow(self):
        """Test that abstract operators can be raised to powers."""

        def qfunc(z):
            return qml.IsingZZ(z, (0, 1)) ** 2

        jaxpr = jax.make_jaxpr(qfunc)(1.2)

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qml.IsingZZ._primitive
        assert jaxpr.eqns[1].primitive == qml.ops.Pow._primitive

        assert jaxpr.eqns[1].invars[0] == jaxpr.eqns[0].outvars[0]
        assert jaxpr.eqns[1].invars[1].val == 2
