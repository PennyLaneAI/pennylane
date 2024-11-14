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
# pylint: disable=protected-access, too-many-positional-arguments, too-many-arguments

from functools import partial

import numpy as np
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


class TestTracer(jax.core.Tracer):
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
            level=0, trace_type=TransformTrace, transform_program=dummy_program, state=None
        )
        trace = main.with_cur_sublevel()
        tracer = TransformTracer(trace, 0, 0)

        assert qml.math.is_abstract(tracer)

    @pytest.mark.parametrize(
        "val, expected_aval",
        [
            (TestTracer(1.23), 1.23),
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
            level=0, trace_type=TransformTrace, transform_program=dummy_program, state=None
        )
        trace = main.with_cur_sublevel()
        tracer = TransformTracer(trace, val, 0)

        assert tracer.aval == expected_aval

    def test_full_lower(self):
        """Test that TransformTracer.full_lower returns the same class."""
        dummy_program = TransformProgram()
        main = jax.core.MainTrace(
            level=0, trace_type=TransformTrace, transform_program=dummy_program, state=None
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
            level=level, trace_type=TransformTrace, transform_program=dummy_program, state=None
        )
        trace = TransformTrace(main, sublevel, dummy_program)
        tracer = TransformTracer(trace, val, idx)

        expected_repr = (
            f"TransformTracer(TransformTrace(level={level}/{sublevel}), val={val}, idx={idx})"
        )
        assert repr(tracer) == expected_repr


def _change_rotations_plxpr_transform(
    primitive, tracers, params, targs, tkwargs, state
):  # pylint: disable=unused-argument
    """Convert RX to RY, RY to RZ, and RZ to RX."""
    # Step 1: Transform primitive
    prim_map = {
        "RX": qml.RY._primitive,
        "RY": qml.RZ._primitive,
        "RZ": qml.RX._primitive,
    }
    primitive = prim_map.get(primitive.name, primitive)
    # Step 2: Update tracers
    tracers = [
        TransformTracer(t._trace, t.val, t.idx + 1) if isinstance(t, TransformTracer) else t
        for t in tracers
    ]
    # Step 3: Return the result of the transformation
    return primitive.bind(*tracers, **params)


@partial(qml.transforms.core.transform, plxpr_transform=_change_rotations_plxpr_transform)
def change_rotations(tape):
    op_map = {
        "RX": qml.RY,
        "RY": qml.RZ,
        "RZ": qml.RX,
    }
    new_ops = [
        op_map[op.name](op.data[0], op.wires) if op.name in op_map else op for op in tape.operations
    ]
    new_tape = qml.tape.QuantumScript(
        new_ops, tape.measurements, shots=tape.shots, trainable_params=tape.trainable_params
    )
    return [new_tape], lambda results: results[0]


class TestTransformTrace:
    """Unit tests for TransformTrace."""

    def test_pure(self):
        """Test that TransformTrace.pure returns the correct output."""
        dummy_program = TransformProgram()
        main = jax.core.MainTrace(
            level=0, trace_type=TransformTrace, transform_program=dummy_program, state=None
        )
        trace = TransformTrace(main, 0, dummy_program, state={})
        tracer = trace.pure(1.5)
        assert tracer._trace is trace
        assert tracer.aval == jax.core.ShapedArray((), float)
        assert tracer.idx == 0

    def test_process_primitive_skip_non_pennylane_primitive(self):
        """Test that non-PennyLane primitives are not processed."""
        add_prim = jax.core.Primitive("add")

        @add_prim.def_impl
        def _(x, y):
            return x + y

        @add_prim.def_abstract_eval
        def _(x, *_):
            return jax.core.ShapedArray((), type(x))

        program = TransformProgram()
        program.add_transform(change_rotations)
        main = jax.core.MainTrace(
            level=0, trace_type=TransformTrace, transform_program=program, state=None
        )
        trace = main.with_cur_sublevel()

        summands = [1.5, 2.5]
        tracers = [trace.pure(s) for s in summands]
        res = trace.process_primitive(add_prim, tracers, {})
        assert np.allclose(res, sum(summands))

    def test_process_primitive_skip_if_idx_out_of_range(self):
        """Test that primitives are not processed if the index of any of the tracers is
        more than the length of the transform program"""
        program = TransformProgram()
        program.add_transform(change_rotations)
        main = jax.core.MainTrace(
            level=0, trace_type=TransformTrace, transform_program=program, state=None
        )
        trace = main.with_cur_sublevel()

        # args are [RX rotation angle=0.5, wire=0]
        args = [0.5, 0]
        tracers = [TransformTracer(trace, a, idx=1) for a in args]
        params = {"n_wires": 1}
        res = trace.process_primitive(qml.RX._primitive, tracers, params)
        qml.assert_equal(res, qml.RX(*args))

    @pytest.mark.parametrize(
        "primitive, args, n_wires, expected_op",
        [
            (qml.RX._primitive, (1.5, 0), 1, qml.RY(1.5, 0)),
            (qml.RY._primitive, (2.5, 1), 1, qml.RZ(2.5, 1)),
            (qml.RZ._primitive, (3.5, 2), 1, qml.RX(3.5, 2)),
            (qml.CNOT._primitive, (0, 1), 2, qml.CNOT([0, 1])),
            (qml.Rot._primitive, (1.2, 3.4, 5.6, 0), 1, qml.Rot(1.2, 3.4, 5.6, 0)),
            (qml.MultiRZ._primitive, (0.5, 1, 2, 3, 4), 4, qml.MultiRZ(0.5, [1, 2, 3, 4])),
        ],
    )
    def test_process_primitive(self, primitive, args, n_wires, expected_op):
        """Test that primitives are processed correctly when they should not be skipped"""
        program = TransformProgram()
        program.add_transform(change_rotations)
        main = jax.core.MainTrace(
            level=0, trace_type=TransformTrace, transform_program=program, state=None
        )
        trace = main.with_cur_sublevel()

        tracers = [trace.pure(a) for a in args]
        params = {"n_wires": n_wires}
        res = trace.process_primitive(primitive, tracers, params)
        qml.assert_equal(res, expected_op)


class TestTransformInterpreter:  # pylint: disable=too-few-public-methods
    """Unit tests for TransformInterpreter."""
