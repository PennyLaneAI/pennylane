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
"""Tests for capturing ``Operator2`` instances into plxpr."""

# pylint: disable=too-few-public-methods,protected-access,unbalanced-tuple-unpacking

import pytest
from operator2_utils import (
    CompOp,
    DynOp,
    FullOp,
    HybridOp,
    MixedHybridOp,
    MultiWireOp,
    StaticOp,
    TwoDynOp,
)

import pennylane as qp
from pennylane import apply
from pennylane.queuing import AnnotatedQueue

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.capture]

# pylint: disable=wrong-import-position
from pennylane.core.operator.operator2 import AbstractOperator, operator_p
from pennylane.pytrees import unflatten

# ---------------------- Helpers ----------------------


def _single_op_eqn(jaxpr):
    """Return the only operator equation in a jaxpr, asserting there is just one."""
    op_eqns = [e for e in jaxpr.eqns if e.primitive is operator_p]
    assert len(op_eqns) == 1
    return op_eqns[0]


def _eval(jaxpr, *args):
    """Evaluate a captured jaxpr and return the results.

    Capture is disabled afterwards so that any ``expected`` operators built for
    comparison are constructed normally rather than re-entering the capture
    machinery (operator reconstruction internally re-enables capture).
    """
    return qp.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)


# ---------------------- Tests ----------------------


class TestCaptureBasics:
    """Tests for capturing operators into a single primitive equation."""

    def test_tracer_none_without_capture(self):
        """Test that the tracer attribute is ``None`` when capture is disabled."""
        with qp.capture.pause():
            op = DynOp(0.5, wires=0)
            assert op.tracer is None

    def test_no_bind_without_capture(self):
        """Test that the operator primitive is not bound when program capture is disabled."""

        def fn(x):
            with qp.capture.pause():
                # pylint: disable=protected-access
                DynOp(x, wires=0)
                DynOp(x, wires=0)._bind_primitive()

        jaxpr = jax.make_jaxpr(fn)(1.5)
        assert len(jaxpr.eqns) == 0

    def test_queued_during_capture(self):
        """Test that constructing an operator during capture still queues it."""
        with qp.queuing.AnnotatedQueue() as q:
            jax.make_jaxpr(lambda x: DynOp(x, wires=0))(0.5)

        assert len(q) == 1
        assert isinstance(list(q.keys())[0].obj, DynOp)

    def test_unflatten_does_not_bind(self):
        """Test that reconstructing an operator via ``_unflatten`` while capture is
        enabled does not bind the operator primitive."""
        data, metadata = DynOp(0.5, wires=0)._flatten()

        def fn():
            _ = DynOp._unflatten(data, metadata)

        jaxpr = jax.make_jaxpr(fn)()
        assert len(jaxpr.eqns) == 0

    def test_simple_op_eqn(self):
        """Test that capturing an operator produces a single operator equation."""
        jaxpr = jax.make_jaxpr(lambda x: DynOp(x, wires=0))(0.5)

        assert len(jaxpr.eqns) == 1
        eqn = jaxpr.eqns[0]
        assert eqn.primitive is operator_p
        assert eqn.params["op_cls"] is DynOp
        assert eqn.params["wire_lens"] == (1,)
        assert isinstance(eqn.outvars[0].aval, AbstractOperator)

    def test_construction_returns_concrete_instance(self):
        """Test that constructing an operator during capture returns a concrete
        ``Operator2`` instance with a tracer attached (rather than the tracer itself)."""

        def f(x):
            op = DynOp(x, wires=0)
            assert isinstance(op, DynOp)
            assert op.tracer is not None
            assert isinstance(op.tracer.aval, AbstractOperator)

        _ = jax.make_jaxpr(f)(0.5)

    def test_dynamic_args_are_inputs(self):
        """Test that dynamic arguments are passed as equation inputs."""
        jaxpr = jax.make_jaxpr(lambda a, b: TwoDynOp(a, b, wires=0))(0.5, 0.6)
        eqn = _single_op_eqn(jaxpr)
        for invar in jaxpr.jaxpr.invars:
            assert invar in eqn.invars

    def test_wires_passed_as_inputs(self):
        """Test that wires are passed as equation inputs."""
        jaxpr = jax.make_jaxpr(lambda w: DynOp(0.5, wires=w))(0)
        eqn = _single_op_eqn(jaxpr)
        assert eqn.params["wire_lens"] == (1,)
        assert jaxpr.jaxpr.invars[0] in eqn.invars

    def test_multiple_wire_arguments(self):
        """Test that operators with multiple wire arguments record each length."""
        jaxpr = jax.make_jaxpr(lambda: MultiWireOp(wires=[0, 1], ctrl_wires=2))()
        eqn = _single_op_eqn(jaxpr)
        assert eqn.params["wire_lens"] == (2, 1)

    def test_static_arg_in_params(self):
        """Test that static arguments are stored as equation parameters."""
        jaxpr = jax.make_jaxpr(lambda: StaticOp("a", wires=0))()
        eqn = _single_op_eqn(jaxpr)
        assert unflatten(*eqn.params["label"]) == "a"

    def test_compilable_arg_in_params(self):
        """Test that compilable arguments are stored as equation parameters."""
        jaxpr = jax.make_jaxpr(lambda: CompOp(5, wires=0))()
        eqn = _single_op_eqn(jaxpr)
        assert unflatten(*eqn.params["n"]) == 5


class TestHybridCapture:
    """Tests for capturing hybrid arguments."""

    def test_numeric_hybrid_leaves_are_inputs(self):
        """Test that numeric leaves of a hybrid argument are passed as inputs."""
        jaxpr = jax.make_jaxpr(lambda x: HybridOp([x, 1.0], wires=0))(0.5)
        eqn = _single_op_eqn(jaxpr)
        assert eqn.params["hybrid_lens"] == (2,)
        assert jaxpr.jaxpr.invars[0] in eqn.invars

    def test_nested_operator_single_equation(self):
        """Test that a nested operator used as data does not leave a dead equation."""

        def f(x):
            inner = DynOp(x, wires=0)
            HybridOp([inner], wires=0)

        jaxpr = jax.make_jaxpr(f)(0.5)
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].params["op_cls"] is HybridOp

    def test_nested_operator_in_different_scope(self):
        """Test that operators of operators are captured correctly when the inner operator
        is initialized in a higher scope."""

        @qp.capture.run_autograph
        def f(pred):
            op = DynOp(0.5, 0)

            if pred:
                HybridOp([op], 2)

        jaxpr = jax.make_jaxpr(f)(True)
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive.name == "cond"
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_branches"][0]
        assert len(inner_jaxpr.eqns) == 1

        op_eqn = inner_jaxpr.eqns[0]
        assert op_eqn.primitive is operator_p
        assert op_eqn.params["op_cls"] is HybridOp

    def test_two_level_nested_single_equation(self):
        """Test that two levels of nested operators collapse to a single equation."""

        def f(x):
            inner = DynOp(x, wires=0)
            mid = HybridOp([inner], wires=0)
            HybridOp([mid], wires=0)

        jaxpr = jax.make_jaxpr(f)(0.5)
        assert len(jaxpr.eqns) == 1

    def test_nested_operator_data_flows(self):
        """Test that the dynamic data of a nested operator flows into the equation."""

        def f(x):
            inner = DynOp(x, wires=0)
            HybridOp([inner], wires=0)

        jaxpr = jax.make_jaxpr(f)(0.5)
        eqn = _single_op_eqn(jaxpr)
        assert jaxpr.jaxpr.invars[0] in eqn.invars

    def test_hybrid_wire_arg_with_other_hybrid(self):
        """Test that hybrid wire arguments mixed with other hybrid arguments are handled correctly."""
        jaxpr = jax.make_jaxpr(lambda x: MixedHybridOp(x, [x, 1.5], [[0], [1, 2]], wires=3))(0.5)
        eqn = _single_op_eqn(jaxpr)

        assert eqn.params["wire_lens"] == (1,)
        assert eqn.params["hybrid_lens"] == (2, 3)
        assert eqn.invars[0] == jaxpr.jaxpr.invars[0]
        assert eqn.invars[1].val == 3
        assert eqn.invars[2] == jaxpr.jaxpr.invars[0]
        assert eqn.invars[3].val == 1.5
        assert eqn.invars[4].val == 0
        assert eqn.invars[5].val == 1
        assert eqn.invars[6].val == 2

        ops_tree, wires_tree = eqn.params["hybrid_trees"]
        assert "list" in repr(ops_tree)
        assert "Wires" in repr(wires_tree)

    def test_mixed_hybrid_wire_roundtrip(self):
        """Test round-trip when there are both wire and non-wire hybrid arguments."""
        jaxpr = jax.make_jaxpr(lambda x: MixedHybridOp(x, [x, 1.0], [[0], [1, 2]], wires=3).tracer)(
            0.5
        )
        [op] = _eval(jaxpr, 0.7)
        qp.assert_equal(op, MixedHybridOp(0.7, [0.7, 1.0], [[0], [1, 2]], wires=3))


class TestReconstruction:
    """Tests that evaluating a captured jaxpr reconstructs the operator."""

    def test_simple_roundtrip(self):
        """Test that a simple operator round-trips through capture and evaluation."""
        jaxpr = jax.make_jaxpr(lambda x: DynOp(x, wires=0).tracer)(0.5)
        [op] = _eval(jaxpr, 0.7)
        qp.assert_equal(op, DynOp(0.7, wires=0))

    def test_dynamic_args_roundtrip(self):
        """Test that an operator with multiple dynamic args round-trips."""
        jaxpr = jax.make_jaxpr(lambda a, b: TwoDynOp(a, b, wires=0).tracer)(0.5, 0.6)
        [op] = _eval(jaxpr, 0.1, 0.2)
        qp.assert_equal(op, TwoDynOp(0.1, 0.2, wires=0))

    def test_static_roundtrip(self):
        """Test that a static argument round-trips through capture and evaluation."""
        jaxpr = jax.make_jaxpr(lambda x: StaticOp("a", wires=x).tracer)(0)
        [op] = _eval(jaxpr, 1)
        qp.assert_equal(op, StaticOp("a", wires=1))

    def test_compilable_roundtrip(self):
        """Test that a compilable argument round-trips through capture and evaluation."""
        jaxpr = jax.make_jaxpr(lambda x: CompOp(5, wires=x).tracer)(0)
        [op] = _eval(jaxpr, 1)
        qp.assert_equal(op, CompOp(5, wires=1))

    def test_multiwire_roundtrip(self):
        """Test that an operator with multiple wire arguments round-trips."""
        jaxpr = jax.make_jaxpr(lambda: MultiWireOp(wires=[0, 1], ctrl_wires=2).tracer)()
        [op] = _eval(jaxpr)
        qp.assert_equal(op, MultiWireOp(wires=[0, 1], ctrl_wires=2))

    def test_numeric_hybrid_roundtrip(self):
        """Test that a hybrid argument with numeric leaves round-trips."""
        jaxpr = jax.make_jaxpr(lambda x: HybridOp([x, 2.0], wires=0).tracer)(0.5)
        [op] = _eval(jaxpr, 0.7)
        qp.assert_equal(op, HybridOp([0.7, 2.0], wires=0))

    def test_nested_operator_roundtrip(self):
        """Test that a nested operator round-trips through capture and evaluation."""

        def f(x):
            inner = DynOp(x, wires=0)
            return HybridOp([inner], wires=0).tracer

        jaxpr = jax.make_jaxpr(f)(0.5)
        [op] = _eval(jaxpr, 0.7)
        qp.assert_equal(op, HybridOp([DynOp(0.7, wires=0)], wires=0))

    def test_two_level_nested_roundtrip(self):
        """Test that two levels of nesting round-trip through capture and evaluation."""

        def f(x):
            inner = DynOp(x, wires=0)
            mid = HybridOp([inner], wires=0)
            return HybridOp([mid], wires=0).tracer

        jaxpr = jax.make_jaxpr(f)(0.5)
        [op] = _eval(jaxpr, 0.9)
        expected = HybridOp([HybridOp([DynOp(0.9, wires=0)], wires=0)], wires=0)
        qp.assert_equal(op, expected)

    def test_full_operator_roundtrip(self):
        """Test that an operator using all argument groups round-trips."""
        jaxpr = jax.make_jaxpr(lambda x: FullOp(x, "lbl", [1.0, 2.0], wires=0).tracer)(0.5)
        [op] = _eval(jaxpr, 0.3)
        qp.assert_equal(op, FullOp(0.3, "lbl", [1.0, 2.0], wires=0))


class TestApply:

    @pytest.mark.parametrize("op2", [DynOp(1.0, wires=0), FullOp(0.3, "lbl", [1.0, 2.0], wires=0)])
    def test_apply_adds_eqn(self, op2):
        """Tests that when an Operator2 is applied, an equation is added for it."""

        def f(op):
            with AnnotatedQueue():
                apply(op)

        jaxpr = jax.make_jaxpr(f)(op2)
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].params["op_cls"] == type(op2)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
