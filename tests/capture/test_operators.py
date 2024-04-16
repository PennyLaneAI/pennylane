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

from pennylane.capture.meta_type import _get_abstract_operator

jax = pytest.importorskip("jax")

pytestmark = pytest.mark.jax

AbstractOperator = _get_abstract_operator()


def test_operators_constructed_when_plxpr_enabled():
    """Test that normal operators can still be constructed when plxpr is enabled."""

    qml.capture.enable_plxpr()

    with qml.queuing.AnnotatedQueue() as q:
        op = qml.adjoint(qml.X(0) + qml.Y(1))

    assert q.queue[0] is op
    assert isinstance(op, qml.ops.Adjoint)
    assert isinstance(op.base, qml.ops.Sum)
    assert op.base[0] == qml.X(0)
    assert op.base[1] == qml.Y(1)

    qml.capture.disable_plxpr()


@pytest.mark.parametrize("as_kwarg", (True, False))
@pytest.mark.parametrize("w", (0, (0,), [0], range(1), qml.wires.Wires(0)))
def test_different_wires(w, as_kwarg):
    qml.capture.enable_plxpr()

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

    qml.capture.disable_plxpr()


def test_parametrized_op():
    """Test capturing a parametrized operation."""
    qml.capture.enable_plxpr()

    jaxpr = jax.make_jaxpr(qml.Rot)(1.0, 2.0, 3.0, 10)

    assert len(jaxpr.eqns) == 1
    eqn = jaxpr.eqns[0]

    assert eqn.primitive == qml.Rot._primitive
    assert len(eqn.invars) == 4
    assert jaxpr.jaxpr.invars == jaxpr.eqns[0].invars

    assert isinstance(eqn.outvars[0].aval, AbstractOperator)
    assert eqn.params == {"n_wires": 1}

    qml.capture.disable_plxpr()


def test_pauli_rot():
    """Test a special operation that has positional metadata and overrides binding."""

    qml.capture.enable_plxpr()

    def qfunc(a, wire0, wire1):
        qml.PauliRot(a, "XY", (wire0, wire1))

    jaxpr = jax.make_jaxpr(qfunc)(0.5, 2, 3)
    assert len(jaxpr.eqns) == 1
    eqn = jaxpr.eqns[0]

    assert eqn.primitive == qml.PauliRot._primitive
    assert eqn.params == {"pauli_word": "XY", "id": None, "n_wires": 2}

    assert len(eqn.invars) == 3
    assert jaxpr.jaxpr.invars == eqn.invars

    qml.capture.disable_plxpr()


class TestTemplates:
    def test_variable_wire_non_parametrized_template(self):
        """Test capturing a variable wire, non-parametrized template like GroverOperator."""

        qml.capture.enable_plxpr()

        jaxpr = jax.make_jaxpr(qml.GroverOperator)(wires=(0, 1, 2, 3, 4, 5))

        assert len(jaxpr.eqns) == 1
        eqn = jaxpr.eqns[0]

        assert eqn.primitive == qml.GroverOperator._primitive
        assert eqn.params == {"n_wires": 6}
        assert eqn.invars == jaxpr.jaxpr.invars

        qml.capture.disable_plxpr()

    def test_nested_template(self):
        """Test capturing a template that contains a nested opeartion defined outside the qfunc."""

        qml.capture.enable_plxpr()

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

        eqn = jaxpr.eqns[5]
        assert eqn.primitive == qml.TrotterProduct._primitive
        assert eqn.invars == jaxpr.eqns[4].outvars  # the sum op

        assert eqn.params == {"order": 2, "time": 2.4}

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, coeffs[0], coeffs[1])

        assert len(q) == 1
        assert q.queue[0] == qml.TrotterProduct(H, time=2.4, order=2)

        qml.capture.disable_plxpr()


class TestOpmath:
    """Tests for capturing operator arithmetic."""

    def test_adjoint(self):
        """Test the adjoint on an op can be captured."""

        qml.capture.enable_plxpr()

        jaxpr = jax.make_jaxpr(qml.adjoint)(qml.X(0))

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qml.X._primitive

        eqn = jaxpr.eqns[1]
        assert eqn.primitive == qml.ops.Adjoint._primitive
        assert eqn.invars == jaxpr.eqns[0].outvars  # the pauli x op
        assert isinstance(eqn.outvars[0].aval, AbstractOperator)
        assert eqn.params == {}

        qml.capture.disable_plxpr()

    def test_control(self):
        """Test a nested control operation."""

        qml.capture.enable_plxpr()

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

        qml.capture.disable_plxpr()


class TestAbstractDunders:
    """Test that operator dunders work when capturing."""

    def test_add(self):
        """Test that the add dunder works."""

        qml.capture.enable_plxpr()

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

        qml.capture.disable_plxpr()

    def test_matmul(self):
        """Test that the matmul dunder works."""

        qml.capture.enable_plxpr()

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

        qml.capture.disable_plxpr()

    def test_mul(self):
        """Test that the scalar multiplication dunder works."""

        qml.capture.enable_plxpr()

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

        qml.capture.disable_plxpr()

    def test_pow(self):
        """Test that abstract operators can be raised to powers."""

        qml.capture.enable_plxpr()

        def qfunc(z):
            return qml.IsingZZ(z, (0, 1)) ** 2

        jaxpr = jax.make_jaxpr(qfunc)(1.2)

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qml.IsingZZ._primitive
        assert jaxpr.eqns[1].primitive == qml.ops.Pow._primitive

        assert jaxpr.eqns[1].invars[0] == jaxpr.eqns[0].outvars[0]
        assert jaxpr.eqns[1].invars[1].val == 2

        qml.capture.disable_plxpr()
