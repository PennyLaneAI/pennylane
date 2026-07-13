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

import numpy as np
import pytest
from test_adjoint2 import RX2

import pennylane as qp
from pennylane.drawer.label import LabelledOp
from pennylane.fourier.mark import MarkedOp
from pennylane.ops.op_math.adjoint2 import Adjoint2
from pennylane.ops.op_math.controlled2 import ControlledOp2

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.capture]

# pylint: disable=wrong-import-position
from pennylane.capture.primitives import AbstractOperator, operator_p
from tests.core.operator.operator2_utils import NonParametricOp


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
        assert eqn.params.get("n_work_wires", 0) == 0
        assert eqn.params.get("work_wire_type", "borrowed") == "borrowed"
        assert eqn.params["adjoint"] is False
        assert isinstance(eqn.outvars[0].aval, AbstractOperator)

        [op] = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.7)
        expected = ControlledOp2(RX2(0.7, 1), [0])
        qp.assert_equal(op, expected)
        assert not op.work_wires
        assert op.work_wire_type == "borrowed"

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

    def test_work_wire_metadata_retraces_and_replays(self, ctrl_fn):
        """Control and work-wire metadata survive repeated capture and concrete replay."""
        base = RX2(0.5, wires=2)

        def f(dummy):
            del dummy
            return ctrl_fn(
                base,
                [0, 1],
                control_values=[False, True],
                work_wires=[3],
                work_wire_type="zeroed",
            ).tracer

        first = jax.make_jaxpr(f)(0.0)
        second = jax.make_jaxpr(f)(jax.numpy.zeros(1))

        for closed_jaxpr, arg in ((first, 0.0), (second, jax.numpy.zeros(1))):
            eqn = _single_op_eqn(closed_jaxpr)
            assert eqn.params["n_ctrls"] == 2
            assert eqn.params["n_work_wires"] == 1
            assert eqn.params["work_wire_type"] == "zeroed"
            assert [invar.val for invar in eqn.invars[-5:]] == [3, 0, 1, False, True]

            [op] = qp.capture.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, arg)
            expected = ControlledOp2(
                RX2(0.5, wires=2),
                [0, 1],
                control_values=[False, True],
                work_wires=[3],
                work_wire_type="zeroed",
            )
            qp.assert_equal(op, expected)
            assert op.work_wires == qp.wires.Wires([3])
            assert op.work_wire_type == "zeroed"

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


@pytest.mark.parametrize("defined_outside", (True, False))
def test_public_s_prod_binding(defined_outside):
    """Tests that the public API for symbolic op captures properly."""

    # Ensure you can construct the op outside a function that is being traced
    op = qp.s_prod(2.0, NonParametricOp(0))
    assert op.base == NonParametricOp(0)
    assert op.scalar == 2.0

    outside_op = NonParametricOp(0) if defined_outside else None

    def f():
        op = outside_op if defined_outside else NonParametricOp(0)
        qp.s_prod(2.0, op)

    cjaxpr = jax.make_jaxpr(f)()

    eqns = cjaxpr.eqns

    assert len(eqns) == 2  # operator and sprod
    assert eqns[0].primitive.name == "operator"
    assert eqns[0].params["op_cls"] is NonParametricOp

    assert eqns[1].primitive.name == "SProd"

    # SProd primitive consumes the op
    assert eqns[0].outvars[0] == eqns[1].invars[1]


def test_public_prod_binding():
    """Tests that the public API for symbolic op captures properly."""
    # Ensure you can construct the op outside a function that is being traced
    op = qp.prod(NonParametricOp(1), NonParametricOp(0))
    assert op == NonParametricOp(1) @ NonParametricOp(0)

    # NOTE: Have one op be outside trace context to
    # cover the tracer-is-none fallback
    outside_op = NonParametricOp(1)

    def f():
        qp.prod(outside_op, NonParametricOp(0))

    cjaxpr = jax.make_jaxpr(f)()

    eqns = cjaxpr.eqns

    assert len(eqns) == 3  # op, op and sprod
    assert eqns[0].primitive.name == "operator"
    assert eqns[0].params["op_cls"] is NonParametricOp
    assert eqns[1].primitive.name == "operator"
    assert eqns[1].params["op_cls"] is NonParametricOp

    assert eqns[2].primitive.name == "Prod"

    # Prod primitive consumes the ops
    assert eqns[1].outvars[0] == eqns[2].invars[0]
    assert eqns[0].outvars[0] == eqns[2].invars[1]


@pytest.mark.parametrize(
    "make_op",
    [
        pytest.param(lambda x, _: qp.s_prod(2.0, x), id="symbolic"),
        pytest.param(lambda x, y: qp.prod(x, y), id="composite"),
        pytest.param(
            lambda x, y: qp.ops.LinearCombination([1.0, 2.0], [x, y]),
            id="linear-combination",
        ),
    ],
)
def test_preconstructed_operator_data_retraces(make_op):
    """Operator2 inputs to legacy arithmetic primitives must not reuse a stale tracer."""
    x, y = qp.X(0), qp.Y(1)

    def f(dummy):
        del dummy
        return make_op(x, y)

    first = jax.make_jaxpr(f)(0.0)
    second = jax.make_jaxpr(f)(jax.numpy.zeros(1))

    with qp.capture.pause():
        expected = make_op(x, y)

    for closed_jaxpr, arg in ((first, 0.0), (second, jax.numpy.zeros(1))):
        assert not closed_jaxpr.consts
        [actual] = qp.capture.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, arg)
        qp.assert_equal(actual, expected)


@pytest.mark.parametrize(
    "make_op",
    [
        pytest.param(lambda base: qp.ops.SProd(scalar=2.0, base=base), id="sprod"),
        pytest.param(lambda base: qp.ops.Pow(base=base, z=2), id="pow"),
    ],
)
def test_keyword_operator_data_is_a_primitive_input(make_op):
    """Operator2 keyword arguments must not be stored as static primitive params."""
    base = qp.X(0)

    def f(dummy):
        del dummy
        return make_op(base)

    first = jax.make_jaxpr(f)(0.0)
    second = jax.make_jaxpr(f)(jax.numpy.zeros(1))

    with qp.capture.pause():
        expected = make_op(base)

    for closed_jaxpr, arg in ((first, 0.0), (second, jax.numpy.zeros(1))):
        base_eqn = _single_op_eqn(closed_jaxpr)
        [symbolic_eqn] = [eqn for eqn in closed_jaxpr.eqns if eqn is not base_eqn]
        assert base_eqn.outvars[0] in symbolic_eqn.invars
        assert "base" not in symbolic_eqn.params
        [actual] = qp.capture.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, arg)
        qp.assert_equal(actual, expected)


@pytest.mark.parametrize(
    "make_op,param_name",
    [
        pytest.param(lambda base, value: qp.ops.Pow(base=base, z=value), "z", id="pow"),
        pytest.param(lambda base, value: qp.ops.Exp(base=base, coeff=value), "coeff", id="exp"),
    ],
)
def test_keyword_dynamic_data_is_a_primitive_input(make_op, param_name):
    """Numeric keyword tracers must remain data when a symbolic operation is retraced."""
    base = qp.X(0)

    def f(value):
        return make_op(base, value)

    first = jax.make_jaxpr(f)(2.0)
    second = jax.make_jaxpr(f)(3.0)

    for closed_jaxpr, value in ((first, 4.0), (second, 5.0)):
        base_eqn = _single_op_eqn(closed_jaxpr)
        [symbolic_eqn] = [eqn for eqn in closed_jaxpr.eqns if eqn is not base_eqn]
        assert symbolic_eqn.invars == [base_eqn.outvars[0], closed_jaxpr.jaxpr.invars[0]]
        assert param_name not in symbolic_eqn.params

        [actual] = qp.capture.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, value)
        with qp.capture.pause():
            expected = make_op(base, value)
        qp.assert_equal(actual, expected)


@pytest.mark.parametrize(
    "make_op,param_name,value",
    [
        pytest.param(lambda base, value: qp.ops.Pow(base=base, z=value), "z", 2, id="pow"),
        pytest.param(
            lambda base, value: qp.ops.Exp(base=base, coeff=value), "coeff", 0.5, id="exp"
        ),
    ],
)
def test_closed_numpy_keyword_data_is_a_primitive_input(make_op, param_name, value):
    """Closed-over NumPy arrays must be primitive data rather than unhashable parameters."""
    base = qp.X(0)
    value = np.array(value)
    closed_jaxpr = jax.make_jaxpr(lambda: make_op(base, value))()

    base_eqn = _single_op_eqn(closed_jaxpr)
    [symbolic_eqn] = [eqn for eqn in closed_jaxpr.eqns if eqn is not base_eqn]
    assert symbolic_eqn.invars[0] == base_eqn.outvars[0]
    assert len(symbolic_eqn.invars) == 2
    assert param_name not in symbolic_eqn.params

    [actual] = qp.capture.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts)
    with qp.capture.pause():
        expected = make_op(base, value)
    qp.assert_equal(actual, expected)


@pytest.mark.parametrize(
    "make_op,param_name,value",
    [
        pytest.param(lambda base, value: MarkedOp(base, value), "marker", "mark", id="marked"),
        pytest.param(
            lambda base, value: LabelledOp(base, value),
            "custom_label",
            "label",
            id="labelled",
        ),
    ],
)
def test_positional_static_metadata_is_a_primitive_parameter(make_op, param_name, value):
    """Trailing positional metadata must not be treated as a JAX primitive input."""
    base = qp.X(0)

    def f(dummy):
        del dummy
        return make_op(base, value)

    first = jax.make_jaxpr(f)(0.0)
    second = jax.make_jaxpr(f)(jax.numpy.zeros(1))

    with qp.capture.pause():
        expected = make_op(base, value)

    for closed_jaxpr, arg in ((first, 0.0), (second, jax.numpy.zeros(1))):
        base_eqn = _single_op_eqn(closed_jaxpr)
        [symbolic_eqn] = [eqn for eqn in closed_jaxpr.eqns if eqn is not base_eqn]
        assert symbolic_eqn.invars == [base_eqn.outvars[0]]
        assert symbolic_eqn.params == {param_name: value}

        [actual] = qp.capture.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, arg)
        qp.assert_equal(actual, expected)


def test_adjoint_pop_does_not_mutate_a_finalized_jaxpr():
    """Popping an operator equation in a new trace must not edit the old trace."""
    adj = Adjoint2(qp.X(0))
    first = jax.make_jaxpr(lambda: qp.apply(adj))()
    first_eqn = _single_op_eqn(first)
    assert first_eqn.params["adjoint"] is True

    second = jax.make_jaxpr(lambda: Adjoint2(adj))()

    assert first_eqn.params["adjoint"] is True
    assert not first.consts
    assert not second.consts
    with qp.queuing.AnnotatedQueue() as first_queue:
        qp.capture.eval_jaxpr(first.jaxpr, first.consts)
    with qp.queuing.AnnotatedQueue() as second_queue:
        qp.capture.eval_jaxpr(second.jaxpr, second.consts)

    assert len(first_queue) == len(second_queue) == 1
    qp.assert_equal(first_queue.queue[0], Adjoint2(qp.X(0)))
    qp.assert_equal(second_queue.queue[0], qp.X(0))


def test_control_pop_does_not_mutate_a_finalized_jaxpr():
    """Controlling an operator in a new trace must not edit its old producer equation."""
    adj = Adjoint2(qp.X(0))
    first = jax.make_jaxpr(lambda: qp.apply(adj))()
    first_eqn = _single_op_eqn(first)
    assert first_eqn.params["n_ctrls"] == 0

    second = jax.make_jaxpr(lambda: qp.ctrl(adj, 1))()

    assert first_eqn.params["n_ctrls"] == 0
    assert not first.consts
    assert not second.consts
    with qp.queuing.AnnotatedQueue() as first_queue:
        qp.capture.eval_jaxpr(first.jaxpr, first.consts)
    with qp.queuing.AnnotatedQueue() as second_queue:
        qp.capture.eval_jaxpr(second.jaxpr, second.consts)

    assert len(first_queue) == len(second_queue) == 1
    qp.assert_equal(first_queue.queue[0], Adjoint2(qp.X(0)))
    qp.assert_equal(second_queue.queue[0], ControlledOp2(Adjoint2(qp.X(0)), [1]))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
