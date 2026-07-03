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
Unit tests for the composite operator class of Operator2 instances.
"""

import pytest

import pennylane as qp

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.capture]

# pylint: disable=wrong-import-position
from tests.core.operator.operator2_utils import NonParametricOp


def test_public_dot_binding():
    """Tests that the public API for composite op captures properly."""

    # NOTE: Have one op be outside trace context to
    # cover the tracer-is-none fallback
    outside_op = NonParametricOp(0)

    def f():
        qp.dot([2.0, 3.0], [outside_op, NonParametricOp(1)])

    cjaxpr = jax.make_jaxpr(f)()

    eqns = cjaxpr.eqns

    assert len(eqns) == 5

    assert eqns[0].primitive.name == "operator"
    assert eqns[0].params["op_cls"] is NonParametricOp

    assert eqns[1].primitive.name == "operator"
    assert eqns[1].params["op_cls"] is NonParametricOp

    # Check sprods consume both operators
    assert eqns[2].primitive.name == "SProd"
    assert eqns[2].invars[1] == eqns[1].outvars[0]
    assert eqns[3].primitive.name == "SProd"
    assert eqns[3].invars[1] == eqns[0].outvars[0]

    # Check sum consumes both sprods
    assert eqns[4].primitive.name == "Sum"
    assert eqns[4].invars[0] == eqns[2].outvars[0]
    assert eqns[4].invars[1] == eqns[3].outvars[0]


def test_public_sum_binding():
    """Tests that the public API for composite op captures properly."""

    # NOTE: Have one op be outside trace context to
    # cover the tracer-is-none fallback
    outside_op = NonParametricOp(0)

    def f():
        qp.sum(outside_op, NonParametricOp(1))

    cjaxpr = jax.make_jaxpr(f)()

    eqns = cjaxpr.eqns

    assert len(eqns) == 3  # op, op and sum
    assert eqns[0].primitive.name == "operator"
    assert eqns[0].params["op_cls"] is NonParametricOp
    assert eqns[1].primitive.name == "operator"
    assert eqns[1].params["op_cls"] is NonParametricOp

    assert eqns[2].primitive.name == "Sum"

    # Sum primitive consumes the ops
    assert eqns[1].outvars[0] == eqns[2].invars[0]
    assert eqns[1].outvars[0] == eqns[2].invars[0]
    assert eqns[0].outvars[0] == eqns[2].invars[1]
    assert eqns[0].outvars[0] == eqns[2].invars[1]


def test_change_op_basis():
    """Tests that change_op_basis captures correctly."""

    def f():
        qp.change_op_basis(NonParametricOp(0), NonParametricOp(1))

    cjaxpr = jax.make_jaxpr(f)()

    eqns = cjaxpr.eqns

    assert len(eqns) == 3  # Op1 + Op2 + Adjoint(Op1)

    assert eqns[0].primitive.name == "operator"
    assert eqns[0].params["op_cls"] is NonParametricOp
    assert eqns[0].params["adjoint"] is False

    assert eqns[1].primitive.name == "operator"
    assert eqns[1].params["op_cls"] is NonParametricOp
    assert eqns[1].params["adjoint"] is False

    assert eqns[2].primitive.name == "operator"
    assert eqns[2].params["op_cls"] is NonParametricOp
    assert eqns[2].params["adjoint"] is True
