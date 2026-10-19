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
"""Tests for ``pop_op_eqns``, which deletes the plxpr equation of an operator
that has been consumed by another operator (e.g. ``CNOT`` consuming ``X``)."""

import jax

# pylint: disable=protected-access, wrong-import-position, wrong-import-order
import pytest
from jax._src.interpreters.partial_eval import JaxprStackFrame

import pennylane as qp
from pennylane.capture.op_eqns import pop_op_eqns

pytestmark = [pytest.mark.jax, pytest.mark.capture, pytest.mark.usefixtures("enable_capture")]


def op_names(jaxpr):
    """Names of the operators left in a captured program."""
    return [eqn.params["op_cls"].__name__ for eqn in jaxpr.eqns]


@pytest.mark.parametrize(
    "idx, expected",
    [(0, ["PauliY", "PauliZ"]), (1, ["PauliX", "PauliZ"]), (2, ["PauliX", "PauliY"])],
)
def test_pops_only_the_target(idx, expected):
    """Only the target's equation is removed, wherever it sits in the list.

    idx=0 is the worst case for the backwards search: the target is at the head.
    """

    def f():
        ops = [qp.X(0), qp.Y(0), qp.Z(0)]
        popped = pop_op_eqns([ops[idx]])

        assert len(popped) == 1
        assert ops[idx].tracer is None

    assert op_names(jax.make_jaxpr(f)()) == expected


def test_pops_several_ops():
    """Every op passed in is removed."""

    def f():
        ops = [qp.X(0), qp.Y(0), qp.Z(0)]
        popped = pop_op_eqns([ops[0], ops[2]])

        assert len(popped) == 2

    assert op_names(jax.make_jaxpr(f)()) == ["PauliY"]


def test_skips_op_without_tracer():
    """An op that was already popped is ignored."""

    def f():
        x = qp.X(0)
        pop_op_eqns([x])
        popped = pop_op_eqns([x])

        assert popped == []

    assert op_names(jax.make_jaxpr(f)()) == []


@pytest.mark.parametrize(
    "fn, expected",
    [
        (lambda: qp.CNOT([0, 1]), ["CNOT"]),
        (lambda: qp.adjoint(qp.S(0)), ["S"]),
        (lambda: qp.ctrl(qp.RX(0.5, 0), [1, 2]), ["RX"]),
        (lambda: qp.adjoint(qp.ctrl(qp.Hadamard(0), 1)), ["CH"]),
        (lambda: [qp.CNOT([0, 1]) for _ in range(5)], ["CNOT"] * 5),
    ],
)
def test_symbolic_ops_leave_one_eqn(fn, expected):
    """A symbolic op leaves exactly one equation; its base is removed."""

    def f():
        fn()

    assert op_names(jax.make_jaxpr(f)()) == expected


def test_cost_is_linear(monkeypatch):
    """Tracing n CNOTs should look at O(n) equations, not O(n^2).

    Each CNOT traces an X, then removes it. The old implementation re-scanned the
    whole equation list on every removal. We count how many times an equation is
    looked up, which is deterministic (no timing).
    """
    lookups = 0

    def counting_add_eqn(self, eqn):
        def thunk():
            nonlocal lookups
            lookups += 1
            return eqn

        self.tracing_eqns.append(thunk)

    monkeypatch.setattr(JaxprStackFrame, "add_eqn", counting_add_eqn)

    def f():
        for _ in range(500):
            qp.CNOT([0, 1])

    jax.make_jaxpr(f)()

    # one lookup per removal + one per CNOT when the jaxpr is built = 1000.
    # The old full-list rebuild did ~125,000.
    assert lookups <= 1500
