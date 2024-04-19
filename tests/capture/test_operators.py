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
# pylint: disable=protected-access
import pytest

import pennylane as qml

from pennylane.capture.meta_type import _get_abstract_operator, PLXPRMeta

jax = pytest.importorskip("jax")

pytestmark = pytest.mark.jax

AbstractOperator = _get_abstract_operator()


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable_plxpr()
    yield
    qml.capture.disable_plxpr()


def test_custom_PLXPRMeta():
    """Test that we can capture custom classes with the PLXPRMeta metaclass by defining
    the _primitive_bind_call method."""

    p = jax.core.Primitive("p")

    @p.def_abstract_eval
    def _(a):
        return jax.core.ShapedArray(a.shape, a.dtype)

    # pylint: disable=too-few-public-methods
    class MyObj(metaclass=PLXPRMeta):
        """A PLXPRMeta class with a _primitive_bind_call class method."""

        @classmethod
        def _primitive_bind_call(cls, *args, **kwargs):
            return p.bind(*args, **kwargs)

    jaxpr = jax.make_jaxpr(MyObj)(0.5)

    assert len(jaxpr.eqns) == 1
    assert jaxpr.eqns[0].primitive == p


def test_custom_plxprmeta_no_bind_primitive_call():
    """Test that an NotImplementedError is raised if the type does not define _primitive_bind_call."""

    # pylint: disable=too-few-public-methods
    class MyObj(metaclass=PLXPRMeta):
        """A class that does not define _primitive_bind_call."""

        def __init__(self, a):
            self.a = a

    with pytest.raises(NotImplementedError, match="Types using PLXPRMeta must implement"):
        MyObj(0.5)


def test_operators_constructed_when_plxpr_enabled():
    """Test that normal operators can still be constructed when plxpr is enabled."""

    with qml.queuing.AnnotatedQueue() as q:
        op = qml.adjoint(qml.X(0) + qml.Y(1))

    assert q.queue[0] is op
    assert isinstance(op, qml.ops.Adjoint)
    assert isinstance(op.base, qml.ops.Sum)
    assert op.base[0] == qml.X(0)
    assert op.base[1] == qml.Y(1)


def test_hybrid_capture_wires():
    """That a hybrid quantum-classical jaxpr can be captured with wire processing."""

    def f(a, b):
        qml.X(a + b)

    jaxpr = jax.make_jaxpr(f)(1, 2)
    assert len(jaxpr.eqns) == 2

    assert jaxpr.eqns[0].primitive.name == "add"

    assert jaxpr.eqns[0].outvars == jaxpr.eqns[1].invars
    assert jaxpr.eqns[1].primitive == qml.X._primitive


def test_hybrid_capture_parametrization():
    """Test a variety of classical processing with a parametrized operation."""

    def f(a):
        qml.Rot(2 * a, jax.numpy.sqrt(a), a**2, wires=a)

    jaxpr = jax.make_jaxpr(f)(0.5)
    assert len(jaxpr.eqns) == 4

    in1 = jaxpr.eqns[0].invars[1]
    assert jaxpr.eqns[1].invars[0] == in1
    assert jaxpr.eqns[2].invars[0] == in1
    assert jaxpr.eqns[3].invars[-1] == in1  # the wire

    assert jaxpr.eqns[0].primitive.name == "mul"
    assert jaxpr.eqns[1].primitive.name == "sqrt"
    assert jaxpr.eqns[2].primitive.name == "integer_pow"
    assert jaxpr.eqns[3].primitive == qml.Rot._primitive


@pytest.mark.parametrize("as_kwarg", (True, False))
@pytest.mark.parametrize("w", (0, (0,), [0], range(1), qml.wires.Wires(0)))
def test_different_wires(w, as_kwarg):
    def qfunc():
        if as_kwarg:
            qml.X(wires=w)
        else:
            qml.X(w)

    jaxpr = jax.make_jaxpr(qfunc)()

    assert len(jaxpr.eqns) == 1

    eqn = jaxpr.eqns[0]
    assert eqn.primitive == qml.X._primitive
    assert len(eqn.invars) == 1
    assert isinstance(eqn.invars[0], jax.core.Literal)
    assert eqn.invars[0].val == 0

    assert isinstance(eqn.outvars[0].aval, AbstractOperator)
    assert isinstance(eqn.outvars[0], jax.core.DropVar)

    assert eqn.params == {"n_wires": 1}


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


def test_pauli_rot():
    """Test a special operation that has positional metadata and overrides binding."""

    def qfunc(a, wire0, wire1):
        qml.PauliRot(a, "XY", (wire0, wire1))

    jaxpr = jax.make_jaxpr(qfunc)(0.5, 2, 3)
    assert len(jaxpr.eqns) == 1
    eqn = jaxpr.eqns[0]

    assert eqn.primitive == qml.PauliRot._primitive
    assert eqn.params == {"pauli_word": "XY", "id": None, "n_wires": 2}

    assert len(eqn.invars) == 3
    assert jaxpr.jaxpr.invars == eqn.invars


class TestTemplates:
    def test_variable_wire_non_parametrized_template(self):
        """Test capturing a variable wire, non-parametrized template like GroverOperator."""

        jaxpr = jax.make_jaxpr(qml.GroverOperator)(wires=(0, 1, 2, 3, 4, 5))

        assert len(jaxpr.eqns) == 1
        eqn = jaxpr.eqns[0]

        assert eqn.primitive == qml.GroverOperator._primitive
        assert eqn.params == {"n_wires": 6}
        assert eqn.invars == jaxpr.jaxpr.invars

    def test_nested_template(self):
        """Test capturing a template that contains a nested opeartion defined outside the qfunc."""

        coeffs = [0.25, 0.75]
        ops = [qml.X(0), qml.Z(0)]
        H = qml.dot(coeffs, ops)

        def qfunc(Hi):
            qml.TrotterProduct(Hi, time=2.4, order=2)

        jaxpr = jax.make_jaxpr(qfunc)(H)

        assert len(jaxpr.eqns) == 6

        # due to flattening and unflattening H
        assert jaxpr.eqns[0].primitive == qml.X._primitive
        assert jaxpr.eqns[1].primitive == qml.ops.SProd._primitive
        assert jaxpr.eqns[2].primitive == qml.Z._primitive
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

    def test_control(self):
        """Test a nested control operation."""

        def qfunc(op):
            qml.ctrl(op, control=3, control_values=[0])

        jaxpr = jax.make_jaxpr(qfunc)(qml.IsingXX(1.2, wires=(0, 1)))

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qml.IsingXX._primitive

        eqn = jaxpr.eqns[1]
        assert eqn.primitive == qml.ops.Controlled._primitive
        assert eqn.invars[0] == jaxpr.eqns[0].outvars[0]  # the isingxx
        assert eqn.invars[1].val == 3

        assert isinstance(eqn.outvars[0].aval, AbstractOperator)
        assert eqn.params == {"control_values": [0], "work_wires": None}


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

        assert eqn.params == {"grouping_type": None, "id": None, "method": "rlf"}

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
