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
"""
Tests for the capture utilities in ``pennylane.testing``.
"""

from operator import eq, ne

import pytest

import pennylane as qp
from pennylane.ops.op_math.condition import Conditional

pytestmark = [pytest.mark.jax, pytest.mark.capture]

jax = pytest.importorskip("jax")

# pylint: disable=wrong-import-position
from pennylane.capture.primitives import operator_p
from pennylane.ops.mid_measure import MidMeasure


def test_plxpr_to_tape_aliases():
    """Test that the old locations of plxpr_to_tape still give the same function."""
    assert qp.tape.plxpr_to_tape is qp.testing.plxpr_to_tape
    assert qp.tape.plxpr_conversion.plxpr_to_tape is qp.testing.plxpr_to_tape


class TestPlxprToTape:
    """Tests for the plxpr_to_tape function."""

    def test_flat_func(self):
        """Test a function without classical structure."""

        def f(x):
            qp.RX(x, 0)
            qp.CNOT((0, 1))
            qp.QFT(wires=(0, 1, 2))
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)(-0.5)
        tape = qp.testing.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 1.2)
        qp.assert_equal(tape[0], qp.RX(1.2, 0))
        qp.assert_equal(tape[1], qp.CNOT((0, 1)))
        qp.assert_equal(tape[2], qp.QFT((0, 1, 2)))
        assert len(tape.operations) == 3

        qp.assert_equal(tape.measurements[0], qp.expval(qp.Z(0)))

    def test_qnode(self):
        """Test a qnode can be transformed into a tape."""
        dev = qp.device("default.qubit", wires=3)

        @qp.qnode(dev)
        def f(x):
            qp.RX(x, 0)
            qp.CNOT((0, 1))
            qp.QFT(wires=(0, 1, 2))
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)(-0.5)
        tape = qp.testing.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 1.2)
        qp.assert_equal(tape[0], qp.RX(1.2, 0))
        qp.assert_equal(tape[1], qp.CNOT((0, 1)))
        qp.assert_equal(tape[2], qp.QFT((0, 1, 2)))
        assert len(tape.operations) == 3

        qp.assert_equal(tape.measurements[0], qp.expval(qp.Z(0)))

    def test_for_loop(self):
        """Test collecting the operations in a for loop."""

        def f(n):
            @qp.for_loop(n)
            def g(i):
                qp.X(i)

            g()

        jaxpr = jax.make_jaxpr(f)(5)
        tape = qp.testing.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 3)
        assert len(tape.operations) == 3
        qp.assert_equal(tape[0], qp.X(0))
        qp.assert_equal(tape[1], qp.X(1))
        qp.assert_equal(tape[2], qp.X(2))

        assert len(tape.measurements) == 0

    def test_while_loop(self):
        """Test collecting the operations in a while loop."""

        def g(x):
            @qp.while_loop(lambda x, i: i < 3)
            def loop(x, i):
                qp.RX(x, i)
                return 2 * x, i + 1

            loop(x, 0)

        jaxpr = jax.make_jaxpr(g)(-0.8)
        x = jax.numpy.array(1.2)
        tape = qp.testing.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, x)

        assert len(tape.operations) == 3
        assert len(tape.measurements) == 0

        qp.assert_equal(tape.operations[0], qp.RX(x, 0))
        qp.assert_equal(tape.operations[1], qp.RX(2 * x, 1))
        qp.assert_equal(tape.operations[2], qp.RX(4 * x, 2))

    def test_cond_bool(self):
        """Test applying a conditional of a classical vlaue."""

        def f(x, value):
            qp.cond(value, qp.RX, false_fn=qp.RY)(x, 0)

        x = jax.numpy.array(-0.5)
        jaxpr = jax.make_jaxpr(f)(x, False)
        tape = qp.testing.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, x, True)
        assert len(tape.operations) == 1
        qp.assert_equal(tape.operations[0], qp.RX(x, 0))

        tape2 = qp.testing.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, x, False)
        assert len(tape2.operations) == 1
        qp.assert_equal(tape2.operations[0], qp.RY(x, 0))

    def test_measure(self):
        """Test capturing measurements."""

        def f():
            m0 = qp.measure(0)
            return qp.sample(op=m0)

        jaxpr = jax.make_jaxpr(f)()
        tape = qp.testing.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts)

        assert len(tape.operations) == 1
        assert isinstance(tape.operations[0], qp.ops.MidMeasure)
        assert tape.operations[0].wires == qp.wires.Wires(0)

        assert isinstance(tape.measurements[0], qp.measurements.SampleMP)
        assert tape.measurements[0].mv is not None

    def test_cond_mcm(self):
        """Test capturing a conditional of a mid circuit measurement."""

        def rx(x, w):
            qp.RX(x, w)

        def f(x):
            m0 = qp.measure(0)
            qp.cond(m0, rx)(x, 2)
            return qp.sample(m0)

        x = jax.numpy.array(0.987)

        jaxpr = jax.make_jaxpr(f)(x)
        tape = qp.testing.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, x)

        assert len(tape.operations) == 2
        assert isinstance(tape.operations[0], qp.ops.MidMeasure)
        assert isinstance(tape.measurements[0], qp.measurements.SampleMP)
        mp = tape.measurements[0]
        assert mp.mv.measurements[0] is tape.operations[0]
        qp.assert_equal(tape.operations[1], qp.ops.Conditional(mp.mv, qp.RX(x, 2)))

    @pytest.mark.parametrize(
        ("comparison", "expected_branches"),
        (
            pytest.param(
                eq,
                {(0, 0): True, (0, 1): False, (1, 0): False, (1, 1): True},
                id="equal",
            ),
            pytest.param(
                ne,
                {(0, 0): False, (0, 1): True, (1, 0): True, (1, 1): False},
                id="not-equal",
            ),
        ),
    )
    def test_cond_mcm_comparison(self, comparison, expected_branches):
        """Test capturing a conditional that compares two mid-circuit measurements."""

        def f():
            m0 = qp.measure(0)
            m1 = qp.measure(1)
            qp.cond(comparison(m0, m1), qp.X)(0)

        jaxpr = jax.make_jaxpr(f)()
        tape = qp.testing.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts)

        assert len(tape.operations) == 3
        assert isinstance(tape.operations[0], MidMeasure)
        assert isinstance(tape.operations[1], MidMeasure)
        conditional = tape.operations[2]
        assert isinstance(conditional, Conditional)
        assert conditional.meas_val.branches == expected_branches
        qp.assert_equal(conditional.base, qp.X(0))

    def test_elif_mcm(self):
        """Test that an elif mcm can be caputured."""

        def rx(*args):
            qp.RX(*args)

        def ry(*args):
            qp.RY(*args)

        def rz(*args):
            qp.RZ(*args)

        def f(x):
            m0 = qp.measure(0)
            m1 = qp.measure(1)

            qp.cond(m0, rx, elifs=(m1, ry), false_fn=rz)(x, 0)

        x = jax.numpy.array(0.5)
        jaxpr = jax.make_jaxpr(f)(x)
        tape = qp.testing.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, x)
        assert len(tape.operations) == 5
        assert isinstance(tape.operations[0], MidMeasure)
        assert isinstance(tape.operations[1], MidMeasure)
        for i in range(2, 5):
            assert isinstance(tape.operations[i], Conditional)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_adjoint_transform(self, lazy):
        """Test capture the adjoint of a qfunc."""

        def qfunc(x):
            qp.RX(x, 0)
            qp.RY(2 * x, 0)
            qp.X(2)

        def f(x):
            qp.adjoint(qfunc, lazy=lazy)(x)

        x = jax.numpy.array(2.1)
        jaxpr = jax.make_jaxpr(f)(0.6)
        tape = qp.testing.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, x)

        assert len(tape.operations) == 3
        qp.assert_equal(tape.operations[0], qp.adjoint(qp.X(2), lazy=lazy))
        qp.assert_equal(tape.operations[1], qp.adjoint(qp.RY(2 * x, 0), lazy=lazy))
        qp.assert_equal(tape.operations[2], qp.adjoint(qp.RX(x, 0), lazy=lazy))

    def test_control_transform(self):
        """Test collecting the control of a qfunc."""

        def qfunc(x, wire):
            qp.RX(x, wire)
            qp.X(wire)

        def f(x):
            qp.ctrl(qfunc, control=[1, 2], control_values=[False, False])(x, 0)

        x = jax.numpy.array(-0.98)
        jaxpr = jax.make_jaxpr(f)(0.1)
        tape = qp.testing.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, x)

        assert len(tape.operations) == 2
        expected0 = qp.ctrl(qp.RX(x, 0), [1, 2], control_values=[False, False])
        qp.assert_equal(tape.operations[0], expected0)
        expected1 = qp.ctrl(qp.X(0), [1, 2], control_values=[False, False])
        qp.assert_equal(tape.operations[1], expected1)

    def test_hybrid_cond_error(self):
        """Test an error is raised if a conditional contains both mcms and classical values."""

        def true_fn(x):
            qp.RX(x, 0)

        def elif_fn(x):
            qp.IsingXX(x, [0, 1])

        def f(x, value):
            m0 = qp.measure(0)
            qp.cond(m0, true_fn, elifs=(value, elif_fn))(x)

        jaxpr = jax.make_jaxpr(f)(0.5, False)
        with pytest.raises(ValueError, match="Cannot use qp.cond with a combination"):
            qp.testing.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 0.5, False)

    def test_dynamic_allocation(self):
        """Tests that circuits containing dynamic wire allocation can be converted."""

        def circuit():
            qp.H(0)
            with qp.allocation.allocate(2, state="zero", restored=True) as wires:
                qp.CNOT(wires)

        jaxpr = jax.make_jaxpr(circuit)()
        tape = qp.testing.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts)
        assert len(tape.operations) == 4
        qp.assert_equal(tape.operations[0], qp.H(0))
        assert isinstance(tape.operations[1], qp.allocation.Allocate)
        assert isinstance(tape.operations[2], qp.CNOT)
        assert tape.operations[2].wires == tape.operations[1].wires
        assert isinstance(tape.operations[3], qp.allocation.Deallocate)
        assert tape.operations[3].wires == tape.operations[1].wires


class TestExtractAllPrimitives:
    """Tests for the extract_all_primitives function."""

    def test_flat_jaxpr(self):
        """Test that the primitives of a jaxpr without nesting are collected."""

        def f(x):
            qp.RX(x, 0)
            qp.CNOT((0, 1))

        jaxpr = jax.make_jaxpr(f)(0.5)
        assert qp.testing.extract_all_primitives(jaxpr.jaxpr) == {operator_p}

    def test_nested_jaxpr(self):
        """Test that primitives inside a higher-order primitive are collected."""

        @qp.qnode(qp.device("default.qubit", wires=2))
        def circuit(x):
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(circuit)(0.5)
        names = {p.name for p in qp.testing.extract_all_primitives(jaxpr.jaxpr)}
        assert {"qnode", "operator", "expval_obs"} <= names

    def test_closed_jaxprs_in_a_tuple(self):
        """Test that primitives are collected from a tuple of closed jaxprs, as used by the
        branches of ``jax.lax.cond``."""

        def f(x):
            return jax.lax.cond(x > 0, jax.numpy.sin, jax.numpy.cos, x)

        jaxpr = jax.make_jaxpr(f)(0.5)
        primitives = qp.testing.extract_all_primitives(jaxpr.jaxpr)
        assert {jax.lax.sin_p, jax.lax.cos_p} <= primitives


class TestAssertEqnMatchesOp:
    """Tests for the assert_eqn_matches_op function."""

    def test_operator2(self):
        """Test matching an equation that creates an Operator2."""
        jaxpr = jax.make_jaxpr(lambda x: qp.RX(x, 0))(0.5)
        qp.testing.assert_eqn_matches_op(jaxpr.eqns[0], qp.RX)

        with pytest.raises(AssertionError):
            qp.testing.assert_eqn_matches_op(jaxpr.eqns[0], qp.RY)

    def test_legacy_operator(self):
        """Test matching an equation that creates an operator with its own primitive."""
        state = jax.numpy.array([1.0, 0.0])
        jaxpr = jax.make_jaxpr(lambda s: qp.StatePrep(s, 0))(state)
        qp.testing.assert_eqn_matches_op(jaxpr.eqns[0], qp.StatePrep)

        with pytest.raises(AssertionError):
            qp.testing.assert_eqn_matches_op(jaxpr.eqns[0], qp.RX)


class TestSingleOperatorEqn:
    """Tests for the single_operator_eqn function."""

    def test_returns_the_operator_eqn(self):
        """Test that the only operator equation is returned."""

        def f(x):
            qp.RX(2 * x, 0)

        jaxpr = jax.make_jaxpr(f)(0.5)
        eqn = qp.testing.single_operator_eqn(jaxpr.jaxpr)
        assert eqn.primitive is operator_p
        assert eqn.params["op_cls"] is qp.RX

    @pytest.mark.parametrize("num_ops", (0, 2))
    def test_error_if_not_exactly_one(self, num_ops):
        """Test that an error is raised unless there is exactly one operator equation."""

        def f(x):
            for _ in range(num_ops):
                qp.RX(x, 0)

        jaxpr = jax.make_jaxpr(f)(0.5)
        with pytest.raises(AssertionError):
            qp.testing.single_operator_eqn(jaxpr.jaxpr)
