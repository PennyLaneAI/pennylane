# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for capturing symbolic ``Operator2`` instances into plxpr."""

import pytest
from test_adjoint2 import RX2

import pennylane as qp
from pennylane.ops.op_math.adjoint2 import Adjoint2

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.capture]

# pylint: disable=wrong-import-position
from pennylane.capture.primitives import AbstractOperator, operator_p


def _single_op_eqn(jaxpr):
    """Return the only operator equation in a jaxpr, asserting there is just one."""
    op_eqns = [e for e in jaxpr.eqns if e.primitive is operator_p]
    assert len(op_eqns) == 1
    return op_eqns[0]


class TestAdjointCapture:
    """Tests for integration of Adjoint2 with program capture."""

    def test_single_equation_with_adjoint_flag(self):
        """Test that adjoint capture produces only produces one equation."""
        jaxpr = jax.make_jaxpr(lambda x: qp.adjoint(RX2(x, wires=0)))(0.5)
        eqn = _single_op_eqn(jaxpr)

        assert eqn.params["op_cls"] is RX2
        assert eqn.params["adjoint"] is True
        assert eqn.params["wire_lens"] == (1,)
        assert isinstance(eqn.outvars[0].aval, AbstractOperator)

    def test_dynamic_params_flow_to_inputs(self):
        """Test that dynamic base parameters are used correctly."""
        jaxpr = jax.make_jaxpr(lambda x: qp.adjoint(RX2(x, wires=0)))(0.5)
        eqn = _single_op_eqn(jaxpr)
        assert jaxpr.jaxpr.invars[0] in eqn.invars

    def test_double_adjoint_cancels(self):
        """Test that a double adjoint toggles the adjoint flag off."""
        jaxpr = jax.make_jaxpr(lambda x: qp.adjoint(qp.adjoint(RX2(x, wires=0))).tracer)(0.5)
        eqn = _single_op_eqn(jaxpr)
        assert eqn.params["adjoint"] is False

        [op] = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.7)
        assert op == RX2(0.7, 0)

    def test_construction_attaches_tracer(self):
        """Test that capture returns an Adjoint2 with an attached tracer."""

        def f(x):
            op = qp.adjoint(RX2(x, wires=0))
            assert isinstance(op, Adjoint2)
            assert op.tracer is not None
            assert isinstance(op.tracer.aval, AbstractOperator)
            assert op.base.tracer is None

        jax.make_jaxpr(f)(0.5)

    def test_preconstructed_base(self):
        """Test adjoint capture when the base operator is built in an outer scope."""

        base = RX2(0.5, wires=0)

        def f():
            return qp.adjoint(base).tracer

        jaxpr = jax.make_jaxpr(f)()
        eqn = _single_op_eqn(jaxpr)
        assert eqn.params["adjoint"] is True

        # pylint: disable=unbalanced-tuple-unpacking
        [op] = qp.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        assert op == Adjoint2(RX2(0.5, wires=0))

    def test_queued_during_capture(self):
        """Test that constructing an adjoint operator during capture still queues it."""
        with qp.queuing.AnnotatedQueue() as q:
            jax.make_jaxpr(lambda x: qp.adjoint(RX2(x, wires=0)))(0.5)

        assert len(q) == 1
        assert isinstance(list(q.keys())[0].obj, Adjoint2)

    def test_reconstructs_from_jaxpr(self):
        """Test that evaluating a captured adjoint jaxpr reconstructs the operator."""
        jaxpr = jax.make_jaxpr(lambda x: qp.adjoint(RX2(x, wires=0)).tracer)(0.5)
        expected = Adjoint2(RX2(0.7, wires=0))

        # pylint: disable=unbalanced-tuple-unpacking
        [op] = qp.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.7)
        qp.assert_equal(op, expected)

    def test_no_bind_without_capture(self):
        """Test that the operator primitive is not bound when program capture is disabled."""

        def fn(x):
            with qp.capture.pause():
                # pylint: disable=protected-access
                Adjoint2(RX2(x, wires=0))._bind_primitive()

        jaxpr = jax.make_jaxpr(fn)(1.5)
        assert len(jaxpr.eqns) == 0


if __name__ == "__main__":
    pytest.main(["-x", __file__])
