# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Unit tests for transforms with program capture
"""

import pytest

import pennylane as qml
from pennylane.capture import TransformTrace, TransformTracer
from pennylane.transforms.core import TransformProgram

jax = pytest.importorskip("jax")
jnp = jax.numpy

pytestmark = pytest.mark.jax


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    """enable and disable capture around each test."""
    qml.capture.enable()
    yield
    qml.capture.disable()


class TestTransformTrace:  # pylint: disable=too-few-public-methods
    """Unit tests for TransformTrace."""


class DummyTracer(jax.core.Tracer):
    # pylint: disable=too-few-public-methods
    def __init__(self, aval):
        self._aval = aval

    @property
    def aval(self):
        return self._aval


class TestTransformTracer:
    """Unit tests for TransformTracer."""

    def test_is_abstract(self):
        """Test that a TransformTracer is considered to be abstract."""
        dummy_program = TransformProgram()
        main = jax.core.MainTrace(
            level=0, trace_type=TransformTrace, transform_program=dummy_program
        )
        trace = main.with_cur_sublevel()
        tracer = TransformTracer(trace, 0, 0)

        assert qml.math.is_abstract(tracer)

    @pytest.mark.parametrize(
        "val, expected_aval",
        [
            (DummyTracer(1.23), 1.23),
            # Multiple by 2 below instead of creating another AbstractValue because both value
            # need to be same instance for equality operator to work how we want
            (jax.core.AbstractValue(),) * 2,
            (1, jax.core.ShapedArray((), int)),
            (1.0, jax.core.ShapedArray((), float)),
            (1 + 0j, jax.core.ShapedArray((), complex)),
            (True, jax.core.ShapedArray((), bool)),
            ([1, 2, 3], jax.core.ShapedArray((3,), int)),
            ((1, 2, 3), jax.core.ShapedArray((3,), int)),
            (
                jnp.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
                jax.core.ShapedArray((3,), complex),
            ),
        ],
    )
    def test_aval(self, val, expected_aval):
        """Test that the abstract evaluation of a TransformTracer is set correctly."""
        dummy_program = TransformProgram()
        main = jax.core.MainTrace(
            level=0, trace_type=TransformTrace, transform_program=dummy_program
        )
        trace = main.with_cur_sublevel()
        tracer = TransformTracer(trace, val, 0)

        assert tracer.aval == expected_aval

    def test_full_lower(self):
        """Test that TransformTracer.full_lower returns the same class."""
        dummy_program = TransformProgram()
        main = jax.core.MainTrace(
            level=0, trace_type=TransformTrace, transform_program=dummy_program
        )
        trace = main.with_cur_sublevel()
        tracer = TransformTracer(trace, 0, 0)

        assert tracer.full_lower() is tracer

    def test_repr(self):
        """Test that the repr(TransformTracer) is correct."""
        level = 0
        sublevel = 1
        val = 2
        idx = 3

        dummy_program = TransformProgram()
        main = jax.core.MainTrace(
            level=level, trace_type=TransformTrace, transform_program=dummy_program
        )
        trace = TransformTrace(main, sublevel, dummy_program)
        tracer = TransformTracer(trace, val, idx)

        expected_repr = (
            f"TransformTracer(TransformTrace(level={level}/{sublevel}), val={val}, idx={idx})"
        )
        assert repr(tracer) == expected_repr


class TestTransformInterpreter:  # pylint: disable=too-few-public-methods
    """Unit tests for TransformInterpreter."""
