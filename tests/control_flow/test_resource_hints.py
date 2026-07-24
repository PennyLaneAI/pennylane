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
"""Tests for resource-estimation hints on control flow."""

import pytest

from pennylane.control_flow._resource_hints import (
    normalize_estimated_probabilities,
    validate_estimated_iterations,
    validate_estimated_probabilities,
    validate_estimated_probabilities_count,
    validate_estimated_probability,
)


class TestResourceHintValidation:
    def test_validate_estimated_iterations(self):
        assert validate_estimated_iterations(10) == 10.0
        assert validate_estimated_iterations(1.5) == 1.5

    @pytest.mark.parametrize("value", [-1, "10"])
    def test_validate_estimated_iterations_invalid(self, value):
        with pytest.raises((TypeError, ValueError)):
            validate_estimated_iterations(value)

    def test_validate_estimated_probability(self):
        assert validate_estimated_probability(0.75) == 0.75

    @pytest.mark.parametrize("value", [-0.1, 1.1, "0.5"])
    def test_validate_estimated_probability_invalid(self, value):
        with pytest.raises((TypeError, ValueError)):
            validate_estimated_probability(value)

    def test_validate_estimated_probabilities(self):
        assert validate_estimated_probabilities([0.2, 0.3]) == (0.2, 0.3)

    def test_normalize_estimated_probabilities_scalar(self):
        assert normalize_estimated_probabilities(0.75) == (0.75,)

    def test_validate_estimated_probabilities_sum_too_large(self):
        with pytest.raises(ValueError, match="sum to at most 1"):
            validate_estimated_probabilities([0.6, 0.6])

    def test_validate_estimated_probabilities_count(self):
        validate_estimated_probabilities_count((0.5,), 1)
        with pytest.raises(ValueError, match="one entry per non-default branch"):
            validate_estimated_probabilities_count((0.5, 0.2), 1)


@pytest.mark.capture
@pytest.mark.jax
def test_for_loop_hint_in_jaxpr():
    """The estimated_iterations hint is preserved in the for_loop primitive."""
    import jax

    import pennylane as qp

    qp.capture.enable()

    @qp.for_loop(0, 10, 1, estimated_iterations=10)
    def loop(_i, x):
        return x + 1

    jaxpr = jax.make_jaxpr(loop)(0)
    assert len(jaxpr.eqns) == 1
    assert jaxpr.eqns[0].params["estimated_iterations"] == 10


@pytest.mark.capture
@pytest.mark.jax
def test_for_loop_float_hint_in_jaxpr():
    """Float estimated_iterations hints are preserved in the for_loop primitive."""
    import jax

    import pennylane as qp

    qp.capture.enable()

    @qp.for_loop(0, 10, 1, estimated_iterations=2.5)
    def loop(_i, x):
        return x + 1

    jaxpr = jax.make_jaxpr(loop)(0)
    assert len(jaxpr.eqns) == 1
    assert jaxpr.eqns[0].params["estimated_iterations"] == 2.5


@pytest.mark.capture
@pytest.mark.jax
def test_cond_hint_in_jaxpr():
    """The estimated_probabilities hint is preserved in the cond primitive."""
    import jax

    import pennylane as qp

    qp.capture.enable()

    def workflow(x):
        @qp.cond(x > 0, estimated_probabilities=0.75)
        def branch():
            return x

        @branch.otherwise
        def otherwise():
            return -x

        return branch()

    jaxpr = jax.make_jaxpr(workflow)(1.0)
    cond_eqn = next(e for e in jaxpr.eqns if e.primitive.name == "cond")
    assert cond_eqn.params["estimated_probabilities"] == (0.75,)


@pytest.mark.capture
@pytest.mark.jax
def test_cond_elif_probabilities_in_jaxpr():
    """Multi-branch cond preserves estimated_probabilities in the cond primitive."""
    import jax

    import pennylane as qp

    qp.capture.enable()

    def workflow(x):
        @qp.cond(x > 2, estimated_probabilities=[0.2, 0.3])
        def branch():
            return 1

        @branch.else_if(x > 0)
        def elif_branch():
            return 2

        @branch.otherwise
        def otherwise():
            return 3

        return branch()

    jaxpr = jax.make_jaxpr(workflow)(1.0)
    cond_eqn = next(e for e in jaxpr.eqns if e.primitive.name == "cond")
    assert cond_eqn.params["estimated_probabilities"] == (0.2, 0.3)
