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

from functools import partial

import pytest
from test_adjoint2 import RX2

import pennylane as qp
from pennylane.ops.op_math.adjoint2 import Adjoint2
from pennylane.ops.op_math.controlled2 import ControlledOp2

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.capture]

# pylint: disable=wrong-import-position
from pennylane.capture.primitives import AbstractOperator, operator_p


def _single_op_eqn(jaxpr):
    """Return the only operator equation in a jaxpr, asserting there is just one."""
    op_eqns = [e for e in jaxpr.eqns if e.primitive is operator_p]
    assert len(op_eqns) == 1
    return op_eqns[0]


@pytest.mark.parametrize("adjoint_fn", [qp.adjoint, Adjoint2])
@pytest.mark.parametrize("lazy", [True, False])
class TestAdjointCapture:
    """Tests for integration of Adjoint2 with program capture."""

    def test_single_equation_with_adjoint_flag(self, adjoint_fn, lazy):
        """Test that adjoint capture produces only produces one equation."""
        if adjoint_fn is Adjoint2 and not lazy:
            pytest.skip("Adjoint2 is always assumed to be lazy.")
        elif adjoint_fn is qp.adjoint:
            adjoint_fn = partial(adjoint_fn, lazy=lazy)

        jaxpr = jax.make_jaxpr(lambda x: adjoint_fn(RX2(x, wires=0)).tracer)(0.5)
        eqn = _single_op_eqn(jaxpr)

        assert eqn.params["op_cls"] is RX2
        assert eqn.params["adjoint"] is lazy
        assert eqn.params["wire_lens"] == (1,)
        assert isinstance(eqn.outvars[0].aval, AbstractOperator)

        [op] = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.7)
        expected = Adjoint2(RX2(0.7, 0)) if lazy else RX2(-0.7, 0)
        qp.assert_equal(op, expected, check_interface=False)

    def test_dynamic_params_flow_to_inputs(self, adjoint_fn, lazy):
        """Test that dynamic base parameters are used correctly."""
        if not lazy:
            pytest.skip("This test is for Adjoint2, not eager adjoint.")

        jaxpr = jax.make_jaxpr(lambda x: adjoint_fn(RX2(x, wires=0)))(0.5)
        eqn = _single_op_eqn(jaxpr)
        assert jaxpr.jaxpr.invars[0] in eqn.invars

    def test_double_adjoint_cancels(self, adjoint_fn, lazy):
        """Test that a double adjoint toggles the adjoint flag off."""
        if adjoint_fn is Adjoint2 and not lazy:
            pytest.skip("Adjoint2 is always assumed to be lazy.")
        elif adjoint_fn is qp.adjoint:
            adjoint_fn = partial(adjoint_fn, lazy=lazy)

        jaxpr = jax.make_jaxpr(lambda x: adjoint_fn(adjoint_fn(RX2(x, wires=0))).tracer)(0.5)
        eqn = _single_op_eqn(jaxpr)
        assert eqn.params["adjoint"] is False

        [op] = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.7)
        qp.assert_equal(op, RX2(0.7, 0), check_interface=False)

    def test_multiple_adjoint_correct_capture(self, adjoint_fn, lazy):
        """Test that creating separate adjoints using the same base operator instance
        works as expected."""
        if adjoint_fn is Adjoint2 and not lazy:
            pytest.skip("Adjoint2 is always assumed to be lazy.")
        elif adjoint_fn is qp.adjoint:
            adjoint_fn = partial(adjoint_fn, lazy=lazy)

        def f(x):
            op = RX2(x, wires=0)
            return adjoint_fn(op).tracer, adjoint_fn(op).tracer

        jaxpr = jax.make_jaxpr(f)(0.5)
        eqns = [e for e in jaxpr.jaxpr.eqns if e.primitive == operator_p]
        assert len(eqns) == 2

        ops = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.7)

        expected = [Adjoint2(RX2(0.7, 0)) if lazy else RX2(-0.7, 0)] * 2
        for op, exp in zip(ops, expected, strict=True):
            qp.assert_equal(op, exp, check_interface=False)

    def test_construction_attaches_tracer(self, adjoint_fn, lazy):
        """Test that capture returns an Adjoint2 with an attached tracer."""
        if not lazy:
            pytest.skip("This test is for Adjoint2, not eager adjoint.")

        def f(x):
            op = adjoint_fn(RX2(x, wires=0))
            assert isinstance(op, Adjoint2)
            assert op.tracer is not None
            assert isinstance(op.tracer.aval, AbstractOperator)
            assert op.base.tracer is None

        jax.make_jaxpr(f)(0.5)

    def test_preconstructed_base(self, adjoint_fn, lazy):
        """Test adjoint capture when the base operator is built in an outer scope."""
        if adjoint_fn is Adjoint2 and not lazy:
            pytest.skip("Adjoint2 is always assumed to be lazy.")
        elif adjoint_fn is qp.adjoint:
            adjoint_fn = partial(adjoint_fn, lazy=lazy)

        base = RX2(0.5, wires=0)

        def f():
            return adjoint_fn(base).tracer

        jaxpr = jax.make_jaxpr(f)()
        eqn = _single_op_eqn(jaxpr)
        assert eqn.params["adjoint"] is lazy

        # pylint: disable=unbalanced-tuple-unpacking
        [op] = qp.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        expected = Adjoint2(RX2(0.5, wires=0)) if lazy else RX2(-0.5, wires=0)
        qp.assert_equal(op, expected, check_interface=False)

    def test_queued_during_capture(self, adjoint_fn, lazy):
        """Test that constructing an adjoint operator during capture still queues it."""
        if adjoint_fn is Adjoint2 and not lazy:
            pytest.skip("Adjoint2 is always assumed to be lazy.")
        elif adjoint_fn is qp.adjoint:
            adjoint_fn = partial(adjoint_fn, lazy=lazy)

        with qp.queuing.AnnotatedQueue() as q:
            jax.make_jaxpr(lambda x: adjoint_fn(RX2(x, wires=0)))(0.5)

        assert len(q) == 1
        assert isinstance(list(q.keys())[0].obj, Adjoint2 if lazy else RX2)

    def test_no_bind_without_capture(self, adjoint_fn, lazy):
        """Test that the operator primitive is not bound when program capture is disabled."""
        if adjoint_fn is Adjoint2 and not lazy:
            pytest.skip("Adjoint2 is always assumed to be lazy.")
        elif adjoint_fn is qp.adjoint:
            adjoint_fn = partial(adjoint_fn, lazy=lazy)

        def fn(x):
            with qp.capture.pause():
                # pylint: disable=protected-access
                adjoint_fn(RX2(x, wires=0))._bind_primitive()

        jaxpr = jax.make_jaxpr(fn)(1.5)
        op_eqns = tuple(eqn for eqn in jaxpr.eqns if eqn.primitive is operator_p)
        assert len(op_eqns) == 0


@pytest.mark.parametrize("ctrl_fn", [qp.ctrl, ControlledOp2])
class TestControlledCapture:
    """Tests for integration of ControlledOp2 with program capture."""

    def test_single_equation_records_control_metadata(self, ctrl_fn):
        """Test that controlled capture produces one equation on the base operator."""
        jaxpr = jax.make_jaxpr(lambda x: ctrl_fn(RX2(x, wires=1), [0]).tracer)(0.5)
        eqn = _single_op_eqn(jaxpr)

        assert eqn.params["op_cls"] is RX2
        assert eqn.params["n_ctrls"] == 1
        assert eqn.params["adjoint"] is False
        assert isinstance(eqn.outvars[0].aval, AbstractOperator)

        [op] = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.7)
        expected = ControlledOp2(RX2(0.7, 1), [0])
        qp.assert_equal(op, expected)

    def test_dynamic_params_flow_to_inputs(self, ctrl_fn):
        """Test that dynamic base parameters are passed as equation inputs."""
        jaxpr = jax.make_jaxpr(lambda x: ctrl_fn(RX2(x, wires=1), [0]))(0.5)
        eqn = _single_op_eqn(jaxpr)
        assert jaxpr.jaxpr.invars[0] in eqn.invars

    def test_multiple_control_wires(self, ctrl_fn):
        """Test that multiple control wires are recorded in ``n_ctrls`` and inputs."""
        jaxpr = jax.make_jaxpr(
            lambda x: ctrl_fn(RX2(x, wires=2), [0, 1], control_values=[True, False]).tracer
        )(0.5)
        eqn = _single_op_eqn(jaxpr)

        assert eqn.params["n_ctrls"] == 2
        assert eqn.invars[-4].val == 0
        assert eqn.invars[-3].val == 1
        assert eqn.invars[-2].val is True
        assert eqn.invars[-1].val is False

        [op] = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.7)
        expected = ControlledOp2(RX2(0.7, 2), [0, 1], [True, False])
        qp.assert_equal(op, expected)

    def test_construction_attaches_tracer(self, ctrl_fn):
        """Test that capture returns a ControlledOp2 with an attached tracer."""

        def f(x):
            op = ctrl_fn(RX2(x, wires=1), [0])
            assert isinstance(op, ControlledOp2)
            assert op.tracer is not None
            assert isinstance(op.tracer.aval, AbstractOperator)
            assert op.base.tracer is None

        jax.make_jaxpr(f)(0.5)

    def test_preconstructed_base(self, ctrl_fn):
        """Test controlled capture when the base operator is built in an outer scope."""
        base = RX2(0.5, wires=0)

        def f():
            return ctrl_fn(base, [1]).tracer

        jaxpr = jax.make_jaxpr(f)()
        eqn = _single_op_eqn(jaxpr)
        assert eqn.params["n_ctrls"] == 1

        # pylint: disable=unbalanced-tuple-unpacking
        [op] = qp.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        expected = ControlledOp2(RX2(0.5, wires=0), control_wires=[1])
        qp.assert_equal(op, expected)

    def test_queued_during_capture(self, ctrl_fn):
        """Test that constructing a controlled operator during capture still queues it."""
        with qp.queuing.AnnotatedQueue() as q:
            jax.make_jaxpr(lambda x: ctrl_fn(RX2(x, wires=1), [0]))(0.5)

        assert len(q) == 1
        assert isinstance(list(q.keys())[0].obj, ControlledOp2)

    def test_nested_controlled_single_equation(self, ctrl_fn):
        """Test that nested controlled operators collapse to a single equation."""
        jaxpr = jax.make_jaxpr(lambda x: ctrl_fn(ctrl_fn(RX2(x, wires=2), [1]), [0]).tracer)(0.5)
        eqn = _single_op_eqn(jaxpr)

        assert eqn.params["op_cls"] is RX2
        assert eqn.params["n_ctrls"] == 2
        assert jaxpr.jaxpr.invars[0] in eqn.invars

        # pylint: disable=unbalanced-tuple-unpacking
        [op] = qp.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.7)
        expected = ControlledOp2(RX2(0.7, wires=2), control_wires=[0, 1])
        qp.assert_equal(op, expected)

    def test_no_bind_without_capture(self, ctrl_fn):
        """Test that the operator primitive is not bound when program capture is disabled."""

        def fn(x):
            with qp.capture.pause():
                # pylint: disable=protected-access
                ctrl_fn(RX2(x, wires=1), [0])._bind_primitive()

        jaxpr = jax.make_jaxpr(fn)(1.5)
        op_eqns = tuple(eqn for eqn in jaxpr.eqns if eqn.primitive is operator_p)
        assert len(op_eqns) == 0


@pytest.mark.parametrize("adjoint_fn", [qp.adjoint, Adjoint2])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("ctrl_fn", [qp.ctrl, ControlledOp2])
class TestNestedSymbolicOpCapture:
    """Tests for capturing nested symbolic operators."""

    def test_controlled_of_adjoint(self, adjoint_fn, ctrl_fn, lazy):
        """Test capture of a controlled adjoint operator."""
        if adjoint_fn is Adjoint2 and not lazy:
            pytest.skip("Adjoint2 is always assumed to be lazy.")
        elif adjoint_fn is qp.adjoint:
            adjoint_fn = partial(adjoint_fn, lazy=lazy)

        jaxpr = jax.make_jaxpr(lambda x: ctrl_fn(adjoint_fn(RX2(x, wires=1)), [0]).tracer)(0.5)
        eqn = _single_op_eqn(jaxpr)

        assert eqn.params["op_cls"] is RX2
        assert eqn.params["n_ctrls"] == 1
        assert eqn.params["adjoint"] is lazy

        # pylint: disable=unbalanced-tuple-unpacking
        [op] = qp.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.7)
        inner_op = Adjoint2(RX2(0.7, wires=1)) if lazy else RX2(-0.7, wires=1)
        expected = ControlledOp2(inner_op, control_wires=[0])
        qp.assert_equal(op, expected, check_interface=False)

    def test_adjoint_of_controlled(self, adjoint_fn, ctrl_fn, lazy):
        """Test capture of an adjoint controlled operator."""
        if adjoint_fn is Adjoint2 and not lazy:
            pytest.skip("Adjoint2 is always assumed to be lazy.")
        elif adjoint_fn is qp.adjoint:
            adjoint_fn = partial(adjoint_fn, lazy=lazy)

        jaxpr = jax.make_jaxpr(lambda x: adjoint_fn(ctrl_fn(RX2(x, wires=1), [0])).tracer)(0.5)
        eqn = _single_op_eqn(jaxpr)

        assert eqn.params["op_cls"] is RX2
        assert eqn.params["n_ctrls"] == 1
        assert eqn.params["adjoint"] is lazy

        # pylint: disable=unbalanced-tuple-unpacking
        [op] = qp.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.7)
        inner_op = Adjoint2(RX2(0.7, wires=1)) if lazy else RX2(-0.7, wires=1)
        expected = ControlledOp2(inner_op, control_wires=[0])
        qp.assert_equal(op, expected, check_interface=False)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
