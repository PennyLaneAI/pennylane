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
"""Regression tests for sc-130866: arithmetic between Operator2 instances and
operator-valued tracers (AbstractOperator) under program capture."""

# pylint: disable=unnecessary-lambda
import pytest

import pennylane as qp

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.capture]

CASES = {
    "op2 + tracer": (lambda: (qp.Y(0) @ qp.X(1)) + 3 * qp.X(2)),
    "tracer + op2": (lambda: 3 * qp.X(2) + (qp.Y(0) @ qp.X(1))),
    "op2 - tracer": (lambda: (qp.Y(0) @ qp.X(1)) - 3 * qp.X(2)),
    "tracer - op2": (lambda: 3 * qp.X(2) - (qp.Y(0) @ qp.X(1))),
    "-tracer + op2": (lambda: -(3 * qp.X(2)) + (qp.Y(0) @ qp.X(1))),
    "op2 @ tracer": (lambda: (qp.Y(0) @ qp.X(1)) @ (3 * qp.X(2))),
    "tracer @ op2": (lambda: (3 * qp.X(2)) @ (qp.Y(0) @ qp.X(1))),
}


@pytest.mark.usefixtures("enable_capture")
@pytest.mark.parametrize("name", CASES)
def test_mixed_operator_arithmetic(name):
    """Mixed Operator2 / operator-tracer arithmetic must not treat the tracer as a scalar
    (no stray Identity), and the captured jaxpr must re-evaluate to the correct operator."""
    build = CASES[name]

    jaxpr = jax.make_jaxpr(build)()
    op_names = [
        eqn.params["op_cls"].__name__ if "op_cls" in eqn.params else eqn.primitive.name
        for eqn in jaxpr.eqns
    ]
    assert "Identity" not in op_names

    # re-evaluating the plxpr is what e.g. Catalyst's from_plxpr does
    with qp.queuing.AnnotatedQueue():
        (res,) = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

    qp.capture.disable()
    try:
        expected = build()
        assert qp.math.allclose(
            qp.matrix(res, wire_order=range(3)), qp.matrix(expected, wire_order=range(3))
        )
    finally:
        qp.capture.enable()
