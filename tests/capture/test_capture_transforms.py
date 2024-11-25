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

jax = pytest.importorskip("jax")
jnp = jax.numpy

# pylint: disable=wrong-import-position
from pennylane.capture import TransformInterpreter, TransformTrace, TransformTracer
from pennylane.capture.primitives import (
    cond_prim,
    for_loop_prim,
    grad_prim,
    jacobian_prim,
    qnode_prim,
    while_loop_prim,
)
from pennylane.transforms.core import TransformError, TransformProgram, transform

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


class DummyTrace(jax.core.Trace):
    """Dummy Trace for testing"""

    def pure(self, val):
        val1 = jnp.array(val)
        return DummyTracer(self, val, jax.core.ShapedArray(val1.shape, val1.dtype))

    lift = sublift = pure

    def process_primitive(self, primitive, tracers, params):
        return primitive.bind(*(t.val for t in tracers), **params)


class DummyTracer(jax.core.Tracer):
    """Dummy Tracer for testing"""

    # pylint: disable=too-few-public-methods
    def __init__(self, trace, val, aval):
        super().__init__(trace)
        self.val = val
        self._aval = aval

    @property
    def aval(self):
        return self._aval


dummy_main = jax.core.MainTrace(level=1, trace_type=DummyTrace)
dummy_trace = dummy_main.with_cur_sublevel()


def init_test_variables(level=0, transforms=()):
    """Create basic TransformProgram, MainTrace, and TransformTrace for unit testing."""
    program = TransformProgram()
    for tr in transforms:
        program.add_transform(tr)

    main = jax.core.MainTrace(
        level=level, trace_type=TransformTrace, transform_program=program, state=None
    )
    trace = main.with_cur_sublevel()
    return program, main, trace


@pytest.mark.unit
class TestTransformTracer:
    """Unit tests for TransformTracer."""

    def test_is_abstract(self):
        """Test that a TransformTracer is considered to be abstract."""
        _, _, trace = init_test_variables()
        tracer = TransformTracer(trace, 0, 0)
        assert qml.math.is_abstract(tracer)

    @pytest.mark.parametrize(
        "val, expected_aval",
        [
            (
                DummyTracer(dummy_trace, 1.23, jax.core.ShapedArray((), float)),
                jax.core.ShapedArray((), float),
            ),
            # Multiple by 2 below instead of creating another AbstractValue because both value
            # need to be same instance for equality operator to work how we want
            (jax.core.AbstractValue(),) * 2,
            (1, jax.core.ShapedArray((), int)),
            (1 + 0j, jax.core.ShapedArray((), complex)),
            (True, jax.core.ShapedArray((), bool)),
            ((1, 2, 3), jax.core.ShapedArray((3,), int)),
            (
                jnp.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
                jax.core.ShapedArray((3,), complex),
            ),
            (qml.PauliX(0), qml.capture.AbstractOperator()),
            (
                qml.expval(qml.Z(0)),  # Observable
                qml.capture.AbstractMeasurement(
                    qml.measurements.ExpectationMP._abstract_eval, n_wires=None
                ),
            ),
            (
                qml.probs(wires=0),  # wires
                qml.capture.AbstractMeasurement(
                    qml.measurements.ProbabilityMP._abstract_eval, n_wires=1, has_eigvals=False
                ),
            ),
            (
                qml.sample(op=jax.core.AbstractValue()),  # MCM
                qml.capture.AbstractMeasurement(
                    qml.measurements.SampleMP._abstract_eval, n_wires=1
                ),
            ),
        ],
    )
    def test_aval(self, val, expected_aval):
        """Test that the abstract evaluation of a TransformTracer is set correctly."""
        _, _, trace = init_test_variables()
        tracer = TransformTracer(trace, val, 0)
        assert tracer.aval == expected_aval

    def test_full_lower(self):
        """Test that TransformTracer.full_lower returns the same class."""
        _, _, trace = init_test_variables()
        tracer = TransformTracer(trace, 0, 0)
        assert tracer.full_lower() is tracer

    def test_repr(self):
        """Test that the repr(TransformTracer) is correct."""
        level = 0
        sublevel = 1
        val = 2
        idx = 3

        dummy_program, main, _ = init_test_variables(level=level)
        trace = TransformTrace(main, sublevel, dummy_program)
        tracer = TransformTracer(trace, val, idx)

        expected_repr = (
            f"TransformTracer(TransformTrace(level={level}/{sublevel}), val={val}, idx={idx})"
        )
        assert repr(tracer) == expected_repr


@transform
def non_plxpr_transform(tape):
    """Dummy transform that does not provide a way to transform PLxPR."""
    return [tape], lambda results: results[0]


def _change_rotations_plxpr_transform(
    primitive, tracers, params, targs, tkwargs, state
):  # pylint: disable=unused-argument
    """Convert RX to RY, RY to RZ, and RZ to RX PLxPR transform."""
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


@partial(transform, plxpr_transform=_change_rotations_plxpr_transform)
def change_rotations(tape):
    """Convert RX to RY, RY to RZ, and RZ to RX tape transform."""
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


def _expval_to_var_plxpr_transform(
    primitive, tracers, params, targs, tkwargs, state
):  # pylint: disable=unused-argument
    """Convert expval to var PLxPR transform."""
    # Step 1: Transform primitive
    if primitive.name == "expval_obs":
        primitive = qml.measurements.VarianceMP._obs_primitive
    # Step 2: Update tracers
    tracers = [
        TransformTracer(t._trace, t.val, t.idx + 1) if isinstance(t, TransformTracer) else t
        for t in tracers
    ]
    # Step 3: Return the result of the transformation
    return primitive.bind(*tracers, **params)


@partial(transform, plxpr_transform=_expval_to_var_plxpr_transform)
def expval_to_var(tape):
    """Covnert expval to var tape transform."""
    new_measurements = [
        qml.var(mp.obs) if isinstance(mp, qml.measurements.ExpectationMP) else mp
        for mp in tape.measurements
    ]
    new_tape = qml.tape.QuantumScript(
        tape.operations, new_measurements, shots=tape.shots, trainable_params=tape.trainable_params
    )
    return [new_tape], lambda results: results[0]


@pytest.mark.unit
class TestTransformTrace:
    """Unit tests for TransformTrace."""

    def test_pure(self):
        """Test that TransformTrace.pure returns the correct output."""
        _, _, trace = init_test_variables()
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

        _, _, trace = init_test_variables(level=0, transforms=[change_rotations])
        summands = [1.5, 2.5]
        tracers = [trace.pure(s) for s in summands]
        res = trace.process_primitive(add_prim, tracers, {})
        assert np.allclose(res, sum(summands))

    def test_process_primitive_skip_if_idx_out_of_range(self):
        """Test that primitives are not processed if the index of any of the tracers is
        more than the length of the transform program"""
        _, _, trace = init_test_variables(transforms=[change_rotations])

        # args are [RX rotation angle=0.5, wire=0]
        args = [0.5, 0]
        tracers = [TransformTracer(trace, a, idx=1) for a in args]
        params = {"n_wires": 1}
        res = trace.process_primitive(qml.RX._primitive, tracers, params)
        qml.assert_equal(res, qml.RX(*args))

    def test_process_primitive_non_plxpr_transform_error(self):
        """Test that an error is raised if attempting to process a primitive using a transform
        that does not provide a ``plxpr_transform`` attribute."""
        _, _, trace = init_test_variables(transforms=[non_plxpr_transform])

        # args are [RX rotation angle=0.5, wire=0]
        args = [0.5, 0]
        tracers = [TransformTracer(trace, a, idx=0) for a in args]
        params = {"n_wires": 1}
        with pytest.raises(
            TransformError, match="non_plxpr_transform cannot be used to transform PLxPR."
        ):
            _ = trace.process_primitive(qml.RX._primitive, tracers, params)

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
        _, _, trace = init_test_variables(transforms=[change_rotations])
        tracers = [trace.pure(a) for a in args]
        params = {"n_wires": n_wires}
        res = trace.process_primitive(primitive, tracers, params)
        qml.assert_equal(res, expected_op)

    @pytest.mark.parametrize(
        "primitive, args, n_wires, expected_op",
        [
            (qml.RX._primitive, (1.5, 0), 1, qml.RZ(1.5, 0)),
            (qml.RY._primitive, (2.5, 1), 1, qml.RX(2.5, 1)),
            (qml.RZ._primitive, (3.5, 2), 1, qml.RY(3.5, 2)),
        ],
    )
    def test_process_primitive_multiple_transforms(self, primitive, args, n_wires, expected_op):
        """Test that primitives are transformed correctly when the program
        has multiple transforms."""
        _, _, trace = init_test_variables(transforms=[change_rotations, change_rotations])
        tracers = [trace.pure(a) for a in args]
        params = {"n_wires": n_wires}
        res = trace.process_primitive(primitive, tracers, params)
        qml.assert_equal(res, expected_op)


@pytest.mark.unit
class TestTransformInterpreter:
    """Unit tests for TransformInterpreter."""

    def test_init(self):
        """Test that TransformInterpreter is initialized correctly."""
        program = TransformProgram()
        program.add_transform(change_rotations)
        program.add_transform(qml.devices.preprocess.decompose, lambda op: True)

        interpreter = TransformInterpreter(program)
        assert interpreter._trace is None
        assert interpreter._state == {}
        assert interpreter._transform_program == program

    def test_cleanup(self):
        """Test that TransformInterpreter.cleanup() correctly resets the state."""
        program = TransformProgram()
        interpreter = TransformInterpreter(program)
        # Using summy values for the sake of testing
        interpreter._trace = -1
        interpreter._state = -2

        interpreter.cleanup()
        assert interpreter._trace is None
        assert interpreter._state == {}

    def test_read_with_trace_already_boxed(self):
        """Test that values that are already boxed into tracers are not changed."""
        program, _, trace = init_test_variables()
        inval = trace.pure(1.5)
        interpreter = TransformInterpreter(program)
        interpreter._trace = trace

        outval = interpreter.read_with_trace(inval)
        assert outval is inval

    @pytest.mark.parametrize(
        "inval",
        [
            1.5,
            0,
            1 + 0j,
            jnp.array(1.0),
            DummyTracer(dummy_trace, -1, jax.core.ShapedArray((), int)),
        ],
    )
    def test_read_with_trace_unboxed(self, inval):
        """Test that values that are not boxed into tracers belonging to the interpreter
        are boxed in TransformTracers."""
        program, _, trace = init_test_variables(level=2)
        interpreter = TransformInterpreter(program)
        interpreter._trace = trace

        outval = interpreter.read_with_trace(inval)
        assert outval == trace.pure(inval)


@pytest.mark.integration
class TestTransformInterpreterIntegration:
    """Integration tests for TransformInterpreter."""

    def test_call(self):
        """Test that calling an interpreted function gives the correct results."""
        program, _, _ = init_test_variables(transforms=[change_rotations, expval_to_var])

        @TransformInterpreter(program)
        def f(x):
            qml.RX(x, 0)
            qml.RY(x, 0)
            qml.RZ(x, 0)
            qml.PhaseShift(x, 0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0])

        args = (1.5,)
        m1, m2 = f(*args)
        qml.assert_equal(m1, qml.var(qml.Z(0)))
        qml.assert_equal(m2, qml.probs(wires=[0]))

        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[0].primitive == qml.RY._primitive
        assert jaxpr.eqns[1].primitive == qml.RZ._primitive
        assert jaxpr.eqns[2].primitive == qml.RX._primitive
        assert jaxpr.eqns[3].primitive == qml.PhaseShift._primitive
        assert jaxpr.eqns[4].primitive == qml.Z._primitive
        assert jaxpr.eqns[5].primitive == qml.measurements.VarianceMP._obs_primitive
        assert jaxpr.eqns[6].primitive == qml.measurements.ProbabilityMP._wires_primitive

        m1, m2 = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        qml.assert_equal(m1, qml.var(qml.Z(0)))
        qml.assert_equal(m2, qml.probs(wires=[0]))

    @pytest.mark.parametrize("lazy", [True, False])
    def test_adjoint_transform(self, lazy):
        """Test that the correct error is raised if intepreted function contains Adjoint
        primitives."""
        program, _, _ = init_test_variables(transforms=[change_rotations, expval_to_var])

        @TransformInterpreter(program)
        def f(x):
            def adj_f(x):
                qml.RX(x, 0)

            qml.adjoint(adj_f, lazy=lazy)(x)
            return qml.expval(qml.Z(0))

        args = (1.5,)
        with pytest.raises(NotImplementedError):
            f(*args)

    def test_ctrl_transform(self):
        """Test that the correct error is raised if intepreted function contains Control
        primitives."""
        program, _, _ = init_test_variables(transforms=[change_rotations, expval_to_var])

        @TransformInterpreter(program)
        def f(x):
            def ctrl_f(x):
                qml.RX(x, 0)

            qml.ctrl(ctrl_f, control=[1, 2])(x)
            return qml.expval(qml.Z(0))

        args = (1.5,)
        with pytest.raises(NotImplementedError):
            f(*args)

    def test_cond(self):
        """Test that calling an interpreted function with cond primitives gives the correct
        results."""
        program, _, _ = init_test_variables(transforms=[change_rotations, expval_to_var])

        @TransformInterpreter(program)
        def f(x):
            @qml.cond(x > 2)
            def cond_f(x):
                qml.RX(x, 0)
                return qml.expval(qml.Z(0))

            @cond_f.else_if(x > 1)
            def _(x):
                qml.RY(x, 0)
                return qml.expval(qml.Z(0))

            @cond_f.else_if(x > 0)
            def _(x):
                qml.RZ(x, 0)
                return qml.expval(qml.Z(0))

            @cond_f.otherwise
            def _(x):
                qml.PhaseShift(x, 0)
                return qml.expval(qml.Z(0))

            out = cond_f(x)
            return out

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)
        # First 3 primitives are the conditions for each of the branches
        assert jaxpr.eqns[3].primitive == cond_prim

        def validate_branch(branch, args, expected_primitives, expected_queue):
            assert len(branch.eqns) == len(expected_primitives)
            for eqn, prim in zip(branch.eqns, expected_primitives):
                assert eqn.primitive == prim

            with qml.queuing.AnnotatedQueue() as q:
                jax.core.eval_jaxpr(branch, [], *args)

            assert len(q) == len(expected_queue)
            for actual, expected in zip(q.queue, expected_queue):
                qml.assert_equal(actual, expected)

            with qml.queuing.AnnotatedQueue() as q:
                jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)

            assert len(q) == len(expected_queue)
            for actual, expected in zip(q.queue, expected_queue):
                qml.assert_equal(actual, expected)

        # True branch
        branch1 = jaxpr.eqns[3].params["jaxpr_branches"][0]
        args = (2.5,)
        expected_primitives = [
            qml.RY._primitive,
            qml.Z._primitive,
            qml.measurements.VarianceMP._obs_primitive,
        ]
        expected_queue = [qml.RY(*args, 0), qml.var(qml.Z(0))]
        validate_branch(branch1, args, expected_primitives, expected_queue)

        # Elif branch 1
        branch2 = jaxpr.eqns[3].params["jaxpr_branches"][1]
        args = (1.5,)
        expected_primitives = [
            qml.RZ._primitive,
            qml.Z._primitive,
            qml.measurements.VarianceMP._obs_primitive,
        ]
        expected_queue = [qml.RZ(*args, 0), qml.var(qml.Z(0))]
        validate_branch(branch2, args, expected_primitives, expected_queue)

        # Elif branch 2
        branch3 = jaxpr.eqns[3].params["jaxpr_branches"][2]
        args = (0.5,)
        expected_primitives = [
            qml.RX._primitive,
            qml.Z._primitive,
            qml.measurements.VarianceMP._obs_primitive,
        ]
        expected_queue = [qml.RX(*args, 0), qml.var(qml.Z(0))]
        validate_branch(branch3, args, expected_primitives, expected_queue)

        # Else branch
        branch4 = jaxpr.eqns[3].params["jaxpr_branches"][3]
        args = (-0.5,)
        expected_primitives = [
            qml.PhaseShift._primitive,
            qml.Z._primitive,
            qml.measurements.VarianceMP._obs_primitive,
        ]
        expected_queue = [qml.PhaseShift(*args, 0), qml.var(qml.Z(0))]
        validate_branch(branch4, args, expected_primitives, expected_queue)

    def test_for_loop(self):
        """Test that calling an interpreted function with for_loop primitives gives the correct
        results."""
        program, _, _ = init_test_variables(transforms=[change_rotations])

        @TransformInterpreter(program)
        def f(x, n):
            @qml.for_loop(n)
            def g(i):
                qml.RX(x, i)

            g()

        args = (1.5, 5)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[0].primitive == for_loop_prim

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 1
        assert inner_jaxpr.eqns[0].primitive == qml.RY._primitive

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)

        assert len(q) == args[1]
        for i, op in enumerate(q.queue):
            qml.assert_equal(op, qml.RY(args[0], i))

    def test_while_loop(self):
        """Test that calling an interpreted function with while_loop primitives gives the correct
        results."""
        # pylint: disable=undefined-loop-variable
        program, _, _ = init_test_variables(transforms=[change_rotations])

        @TransformInterpreter(program)
        def f(x, n):
            @qml.while_loop(lambda i: i < 2 * n)
            def g(i):
                qml.RX(x, i)
                return i + 1

            g(0)

        args = (1.5, 5)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[0].primitive == while_loop_prim

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 2
        assert inner_jaxpr.eqns[0].primitive == qml.RY._primitive

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)

        assert len(q) == 2 * args[1]
        for i, op in enumerate(q.queue):
            qml.assert_equal(op, qml.RY(args[0], i))

    def test_qnode(self):
        """Test that calling an interpreted function with qnode primitives gives the correct
        results."""
        program, _, _ = init_test_variables(transforms=[change_rotations, expval_to_var])
        dev = qml.device("default.qubit", wires=2)

        @TransformInterpreter(program)
        @qml.qnode(dev, diff_method="adjoint", grad_on_execution=False)
        def f(x):
            # With RZ and <Z>, the expval will always be 1, so var will always be 0
            qml.RZ(x, 0)
            return qml.expval(qml.Z(0))

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[0].primitive == qnode_prim
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

        assert len(inner_jaxpr.eqns) == 3
        assert inner_jaxpr.eqns[0].primitive == qml.RX._primitive
        assert inner_jaxpr.eqns[1].primitive == qml.Z._primitive
        assert inner_jaxpr.eqns[2].primitive == qml.measurements.VarianceMP._obs_primitive

        assert jaxpr.eqns[0].params["qnode_kwargs"]["diff_method"] == "adjoint"
        assert jaxpr.eqns[0].params["qnode_kwargs"]["grad_on_execution"] is False
        assert jaxpr.eqns[0].params["device"] == dev

        res1 = f(*args)
        # We end up performing an RX gate and measuring var(Z)
        expected = jnp.sin(*args) ** 2
        assert qml.math.allclose(res1, expected)
        res2 = jax.core.eval_jaxpr(jaxpr.jaxpr, [], *args)
        assert qml.math.allclose(res2, expected)

    @pytest.mark.parametrize("grad_f", (qml.grad, qml.jacobian))
    def test_grad_and_jac(self, grad_f):
        """Test that calling an interpreted function with qnode primitives gives the correct
        results."""
        program, _, _ = init_test_variables(transforms=[change_rotations, expval_to_var])
        dev = qml.device("default.qubit", wires=2)

        @TransformInterpreter(program)
        def f(x):
            @qml.qnode(dev)
            def circuit(y):
                qml.RZ(y, 0)
                return qml.expval(qml.Z(0))

            return grad_f(circuit)(x)

        jaxpr = jax.make_jaxpr(f)(0.5)

        if grad_f == qml.grad:
            assert jaxpr.eqns[0].primitive == grad_prim
        else:
            assert jaxpr.eqns[0].primitive == jacobian_prim
        grad_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        qfunc_jaxpr = grad_jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.Z._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.measurements.VarianceMP._obs_primitive
