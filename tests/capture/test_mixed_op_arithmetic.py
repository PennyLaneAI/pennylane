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
operator-valued tracers (AbstractOperator) under program capture.

AbstractOperator tracers are not ``TensorLike``, so Operator2 ``+/-`` defer to
AbstractOperator arithmetic instead of the scalar/Identity branch. Scalar
``+/-`` still uses the Identity embedding.
"""

# pylint: disable=unnecessary-lambda
import pytest

import pennylane as qp

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.capture]

# Mixed Operator2 / operator-tracer: the tracer must not be mistaken for a scalar.
MIXED_CASES = {
    "op2 + tracer": (lambda: (qp.Y(0) @ qp.X(1)) + 3 * qp.X(2)),
    "tracer + op2": (lambda: 3 * qp.X(2) + (qp.Y(0) @ qp.X(1))),
    "op2 - tracer": (lambda: (qp.Y(0) @ qp.X(1)) - 3 * qp.X(2)),
    "tracer - op2": (lambda: 3 * qp.X(2) - (qp.Y(0) @ qp.X(1))),
    "-tracer + op2": (lambda: -(3 * qp.X(2)) + (qp.Y(0) @ qp.X(1))),
    "op2 @ tracer": (lambda: (qp.Y(0) @ qp.X(1)) @ (3 * qp.X(2))),
    "tracer @ op2": (lambda: (3 * qp.X(2)) @ (qp.Y(0) @ qp.X(1))),
}

# Scalar arithmetic under capture must still go through the TensorLike path.
SCALAR_CASES = {
    "op2 + scalar": (lambda: (qp.Y(0) @ qp.X(1)) + 2),
    "op2 - scalar": (lambda: (qp.Y(0) @ qp.X(1)) - 2),
}


def _op_names(jaxpr):
    return [
        eqn.params["op_cls"].__name__ if "op_cls" in eqn.params else eqn.primitive.name
        for eqn in jaxpr.eqns
    ]


def _assert_eval_matches(build, wire_order):
    """Captured jaxpr must re-evaluate to the same operator as outside capture."""
    jaxpr = jax.make_jaxpr(build)()
    with qp.queuing.AnnotatedQueue():
        (res,) = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

    qp.capture.disable()
    try:
        expected = build()
        assert qp.math.allclose(
            qp.matrix(res, wire_order=wire_order),
            qp.matrix(expected, wire_order=wire_order),
        )
    finally:
        qp.capture.enable()

    return jaxpr


@pytest.mark.usefixtures("enable_capture")
@pytest.mark.parametrize("name", MIXED_CASES)
def test_mixed_operator_arithmetic(name):
    """Operator tracers must not be treated as scalars (no stray Identity)."""
    build = MIXED_CASES[name]
    jaxpr = _assert_eval_matches(build, wire_order=range(3))
    assert "Identity" not in _op_names(jaxpr)


@pytest.mark.usefixtures("enable_capture")
@pytest.mark.parametrize("name", SCALAR_CASES)
def test_operator2_scalar_arithmetic_under_capture(name):
    """Scalar +/- under capture still embeds via Identity and re-evaluates correctly."""
    build = SCALAR_CASES[name]
    jaxpr = _assert_eval_matches(build, wire_order=range(2))
    assert "Identity" in _op_names(jaxpr)


@pytest.mark.usefixtures("enable_capture")
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: 3 * qp.X(0) + 2, id="op_tracer + scalar"),
        pytest.param(lambda: 2 + 3 * qp.X(0), id="scalar + op_tracer"),
        pytest.param(lambda: 3 * qp.X(0) - 2, id="op_tracer - scalar"),
        pytest.param(lambda: 2 - 3 * qp.X(0), id="scalar - op_tracer"),
        pytest.param(lambda: (qp.Y(0) @ qp.X(1)) * (3 * qp.X(2)), id="op2 * op_tracer"),
        pytest.param(lambda: (3 * qp.X(2)) * (3 * qp.X(1)), id="op_tracer * op_tracer"),
    ],
)
def test_unsupported_operator_tracer_arithmetic_raises(build):
    """Unsupported operands fail at trace time with Python's standard TypeError."""
    with pytest.raises(TypeError, match="unsupported operand"):
        jax.make_jaxpr(build)()
