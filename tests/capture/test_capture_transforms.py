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
This submodule tests transforms with program capture
"""
# pylint: disable=protected-access
from functools import partial

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# pylint: disable=wrong-import-position
import pennylane as qml
from pennylane.capture.primitives import cond_prim, for_loop_prim, qnode_prim, while_loop_prim
from pennylane.tape.plxpr_conversion import CollectOpsandMeas
from pennylane.transforms.core import TransformError, TransformProgram, transform

pytestmark = [pytest.mark.jax, pytest.mark.capture]


@transform
def z_to_hadamard(
    tape, dummy_arg1, dummy_arg2, dummy_kwarg1=None, dummy_kwarg2=None
):  # pylint: disable=unused-argument
    """Transform that converts Z gates to H gates."""
    new_ops = [qml.H(wires=op.wires) if isinstance(op, qml.Z) else op for op in tape.operations]
    return [tape.copy(operations=new_ops)], lambda res: res[0]


@transform
def shift_rx_to_end(tape):
    """Transform that moves all RX gates to the end of the operations list."""
    new_ops, rxs = [], []

    for op in tape.operations:
        if isinstance(op, qml.RX):
            rxs.append(op)
        else:
            new_ops.append(op)

    operations = new_ops + rxs
    new_tape = tape.copy(operations=operations)
    return [new_tape], lambda res: res[0]


def expval_z_obs_to_x_obs_plxpr(
    jaxpr, consts, targs, tkwargs, *args
):  # pylint: disable=unused-argument

    class ExpvalZToX(qml.capture.PlxprInterpreter):  # pylint: disable=too-few-public-methods
        """Expval Z to X plxpr implementation."""

        def interpret_measurement(self, meas):
            """Interpret measurement."""
            if isinstance(meas, qml.measurements.ExpectationMP) and isinstance(
                meas.obs, qml.PauliZ
            ):
                return qml.expval(qml.X(meas.wires))

            return super().interpret_measurement(meas)

    def wrapper(*inner_args):
        return ExpvalZToX().eval(jaxpr, consts, *inner_args)

    return jax.make_jaxpr(wrapper)(*args)


@partial(transform, plxpr_transform=expval_z_obs_to_x_obs_plxpr)
def expval_z_obs_to_x_obs(
    tape, dummy_arg1, dummy_arg2, dummy_kwarg1=None, dummy_kwarg2=None
):  # pylint: disable=unused-argument
    """Transform that converts Z observables for expectation values to X observables.
    This transform works natively with plxpr."""
    new_measurements = [
        (
            qml.expval(qml.X(mp.wires))
            if isinstance(mp, qml.measurements.ExpectationMP) and isinstance(mp.obs, qml.Z)
            else mp
        )
        for mp in tape.measurements
    ]
    return [tape.copy(measurements=new_measurements)], lambda res: res[0]


@transform
def dummy_multi_tape_transform(tape):
    """Dummy transform that returns multiple tapes."""
    return [tape, tape], lambda res: res[0]


class TestCaptureTransforms:
    """Tests to verify that transforms are captured correctly."""

    def test_transform_primitive_capture(self):
        """Test that a transform's primitive is captured correctly."""

        def func(x):
            y = x * 5
            return y**0.5

        args = (1.5,)
        targs = [0, 1]
        tkwargs = {"dummy_kwarg1": "foo", "dummy_kwarg2": "bar"}

        transformed_func = z_to_hadamard(func, *targs, **tkwargs)
        jaxpr = jax.make_jaxpr(transformed_func)(*args)
        assert (transform_eqn := jaxpr.eqns[0]).primitive == z_to_hadamard._primitive

        params = transform_eqn.params
        assert params["args_slice"] == slice(0, 1)
        assert params["consts_slice"] == slice(1, 1)
        assert params["targs_slice"] == slice(1, None)
        assert params["tkwargs"] == tkwargs

        inner_jaxpr = params["inner_jaxpr"]
        expected_jaxpr = jax.make_jaxpr(func)(*args).jaxpr
        for eqn1, eqn2 in zip(inner_jaxpr.eqns, expected_jaxpr.eqns, strict=True):
            assert eqn1.primitive == eqn2.primitive

    def test_transform_qfunc_pytree_args(self):
        """Test that transforming an object that accepts pytree arguments is correct."""

        def func(x):
            y = x[0] * 5 + x[1]
            return y**0.5

        args = ([1.5, 2.5],)
        targs = [0, 1]
        tkwargs = {"dummy_kwarg1": "foo", "dummy_kwarg2": "bar"}

        transformed_func = z_to_hadamard(func, *targs, **tkwargs)

        jaxpr = jax.make_jaxpr(transformed_func)(*args)
        assert (transform_eqn := jaxpr.eqns[0]).primitive == z_to_hadamard._primitive

        params = transform_eqn.params
        assert params["args_slice"] == slice(0, 2)
        assert params["consts_slice"] == slice(2, 2)
        assert params["targs_slice"] == slice(2, None)
        assert params["tkwargs"] == tkwargs

        inner_jaxpr = params["inner_jaxpr"]
        expected_jaxpr = jax.make_jaxpr(func)(*args).jaxpr
        for eqn1, eqn2 in zip(inner_jaxpr.eqns, expected_jaxpr.eqns, strict=True):
            assert eqn1.primitive == eqn2.primitive

        # Verifying that transformed function can execute
        _ = transformed_func(*args)

    def test_transform_qnode_capture(self):
        """Test that a transformed QNode is captured correctly."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def func(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        args = (1.5,)
        targs = (0, 1)
        tkwargs = {"dummy_kwarg1": "foo", "dummy_kwarg2": "bar"}

        transformed_func = z_to_hadamard(func, *targs, **tkwargs)

        jaxpr = jax.make_jaxpr(transformed_func)(*args)
        assert (transform_eqn := jaxpr.eqns[0]).primitive == z_to_hadamard._primitive

        qnode_jaxpr = transform_eqn.params["inner_jaxpr"]
        assert qnode_jaxpr.eqns[0].primitive == qnode_prim

        qnode = qnode_jaxpr.eqns[0].params["qnode"]
        expected_program = TransformProgram()
        expected_program.add_transform(z_to_hadamard, *targs, **tkwargs)
        # Manually change targs from tuple to list
        expected_program[0]._args = tuple(targs)  # pylint: disable=protected-access
        assert qnode.transform_program == expected_program

        qfunc_jaxpr = qnode_jaxpr.eqns[0].params["qfunc_jaxpr"]
        expected_jaxpr = jax.make_jaxpr(func)(*args).eqns[0].params["qfunc_jaxpr"]
        for eqn1, eqn2 in zip(qfunc_jaxpr.eqns, expected_jaxpr.eqns, strict=True):
            assert eqn1.primitive == eqn2.primitive

    def test_transform_primitive_eval(self):
        """Test that evaluating jaxpr containing a transform primitive does not apply
        the transform."""

        def func():
            return qml.expval(qml.Z(0))

        targs = (2, 3)
        tkwargs = {"dummy_kwarg1": "hello", "dummy_kwarg2": "world"}

        transformed_func = expval_z_obs_to_x_obs(func, *targs, **tkwargs)
        jaxpr = jax.make_jaxpr(transformed_func)()

        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        # If the inner jaxpr was transformed, the output would be <X>, but it should
        # stay as <Z> if it wasn't transformed
        assert res == [qml.expval(qml.Z(0))]

    def test_multiple_transforms_capture(self):
        """Test that JAXPR containing a transformed qnode primitive is evaluated correctly."""

        def func(x):
            y = x * 5
            return y**0.5

        args = (1.5,)
        targs1 = (0, 1)
        tkwargs1 = {"dummy_kwarg1": "foo", "dummy_kwarg2": "bar"}
        targs2 = (2, 3)
        tkwargs2 = {"dummy_kwarg1": "hello", "dummy_kwarg2": "world"}

        transformed_func = z_to_hadamard(
            expval_z_obs_to_x_obs(func, *targs2, **tkwargs2), *targs1, **tkwargs1
        )
        jaxpr = jax.make_jaxpr(transformed_func)(*args)
        assert (transform_eqn1 := jaxpr.eqns[0]).primitive == z_to_hadamard._primitive

        params1 = transform_eqn1.params
        assert params1["args_slice"] == slice(0, 1)
        assert params1["consts_slice"] == slice(1, 1)
        assert params1["targs_slice"] == slice(1, None)
        assert params1["tkwargs"] == tkwargs1

        inner_jaxpr = params1["inner_jaxpr"]
        assert (transform_eqn2 := inner_jaxpr.eqns[0]).primitive == expval_z_obs_to_x_obs._primitive

        params2 = transform_eqn2.params
        assert params2["args_slice"] == slice(0, 1)
        assert params2["consts_slice"] == slice(1, 1)
        assert params2["targs_slice"] == slice(1, None)
        assert params2["tkwargs"] == tkwargs2

        inner_inner_jaxpr = params2["inner_jaxpr"]
        expected_jaxpr = jax.make_jaxpr(func)(*args).jaxpr
        for eqn1, eqn2 in zip(inner_inner_jaxpr.eqns, expected_jaxpr.eqns, strict=True):
            assert eqn1.primitive == eqn2.primitive

    def test_higher_order_primitives(self):
        """Test that transforms are captured correctly when used with higher-order primitives."""
        dev = qml.device("default.qubit", wires=5)

        targs = (0, 1)
        tkwargs = {"dummy_kwarg1": "foo", "dummy_kwarg2": "bar"}

        @qml.qnode(dev)
        def f():
            @partial(z_to_hadamard, dummy_arg1=targs[0], dummy_arg2=targs[1], **tkwargs)
            @qml.for_loop(3)
            def g(i):
                qml.X(i)
                qml.X(i)

            g()
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        assert jaxpr.eqns[0].primitive == qnode_prim

        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == z_to_hadamard._primitive

        loop_jaxpr = qfunc_jaxpr.eqns[0].params["inner_jaxpr"]
        assert loop_jaxpr.eqns[0].primitive == qml.capture.primitives.for_loop_prim

        loop_body_jaxpr = loop_jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert loop_body_jaxpr.eqns[0].primitive == qml.X._primitive
        assert loop_body_jaxpr.eqns[1].primitive == qml.X._primitive

        assert qfunc_jaxpr.eqns[1].primitive == qml.Z._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive


class TestTapeTransformFallback:
    """Unit tests for falling back to tape transforms."""

    def test_multi_tape_transform_error(self):
        """Test that a transform that returns multiple tapes raises an error."""

        def f(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(1.5)
        with pytest.raises(
            TransformError, match="Only transforms that return a single QuantumTape"
        ):
            dummy_multi_tape_transform.plxpr_transform(jaxpr.jaxpr, jaxpr.consts, (), {}, 1.5)

    def test_multi_tape_transform_integration_error(self):
        """Test that a transform that returns multiple tapes raises an error as a decorator."""

        @qml.capture.expand_plxpr_transforms
        @dummy_multi_tape_transform
        def f(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        with pytest.raises(
            TransformError, match="Only transforms that return a single QuantumTape"
        ):
            _ = f(1.5)

    def test_tape_transform(self):
        """Test that transforming plxpr by falling back to the tape implementation
        works correctly."""

        def f(x):
            qml.Z(0)
            qml.RX(x, 0)
            qml.Hadamard(0)
            return qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)

        jaxpr = jax.make_jaxpr(f)(1.5)
        dummy_targs = (0, 0)
        transformed_jaxpr = z_to_hadamard.plxpr_transform(
            jaxpr.jaxpr, jaxpr.consts, dummy_targs, {}, 1.5
        )

        # Manually checking jaxpr equations to verify correct order of equations
        expected_primitives = (
            qml.Hadamard._primitive,
            qml.RX._primitive,
            qml.Hadamard._primitive,
            qml.PauliZ._primitive,
            qml.measurements.ExpectationMP._obs_primitive,
            qml.measurements.StateMP._wires_primitive,
            qml.measurements.ProbabilityMP._wires_primitive,
        )
        actual_primitives = tuple(eqn.primitive for eqn in transformed_jaxpr.eqns)
        assert actual_primitives == expected_primitives

        collector = CollectOpsandMeas()
        collector.eval(transformed_jaxpr.jaxpr, transformed_jaxpr.consts, 1.5)

        ops = collector.state["ops"]
        assert ops == [qml.Hadamard(0), qml.RX(1.5, 0), qml.Hadamard(0)]
        meas = collector.state["measurements"]
        assert meas == [qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)]

    def test_tape_transform_integration(self):
        """Test that using tape transforms as decorators works correctly."""

        @qml.capture.expand_plxpr_transforms
        @partial(z_to_hadamard, dummy_arg1=0, dummy_arg2=0)
        def f(x):
            qml.Z(0)
            qml.RX(x, 0)
            qml.Hadamard(0)
            return qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)

        jaxpr = jax.make_jaxpr(f)(1.5)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 1.5)

        ops = collector.state["ops"]
        assert ops == [qml.Hadamard(0), qml.RX(1.5, 0), qml.Hadamard(0)]
        meas = collector.state["measurements"]
        assert meas == [qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)]

    def test_tape_transform_consts(self):
        """Test that using fallback transforms with constants works correctly."""
        x = jnp.array(1.5)

        def f():
            qml.Z(0)
            qml.RX(x, 0)
            qml.Hadamard(0)
            return qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)

        jaxpr = jax.make_jaxpr(f)()
        assert jaxpr.consts == [jnp.array(1.5)]

        dummy_targs = (0, 0)
        transformed_jaxpr = z_to_hadamard.plxpr_transform(
            jaxpr.jaxpr, jaxpr.consts, dummy_targs, {}
        )
        collector = CollectOpsandMeas()
        collector.eval(transformed_jaxpr.jaxpr, transformed_jaxpr.consts)

        ops = collector.state["ops"]
        assert ops == [qml.Hadamard(0), qml.RX(jnp.array(1.5), 0), qml.Hadamard(0)]
        meas = collector.state["measurements"]
        assert meas == [qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)]

    def test_tape_transform_dynamic_wires(self):
        """Test that using fallback transforms with dynamic wires works correctly."""

        def f(x, w):
            qml.Z(w)
            qml.RX(x, 0)
            qml.Hadamard(0)
            return qml.expval(qml.Z(w)), qml.state(), qml.probs(wires=0)

        jaxpr = jax.make_jaxpr(f)(1.5, 1)
        dummy_targs = (0, 0)
        transformed_jaxpr = z_to_hadamard.plxpr_transform(
            jaxpr.jaxpr, jaxpr.consts, dummy_targs, {}, 1.5, 1
        )
        collector = CollectOpsandMeas()
        collector.eval(transformed_jaxpr.jaxpr, transformed_jaxpr.consts, 1.5, 1)

        ops = collector.state["ops"]
        assert ops == [qml.Hadamard(1), qml.RX(1.5, 0), qml.Hadamard(0)]
        meas = collector.state["measurements"]
        assert meas == [qml.expval(qml.Z(1)), qml.state(), qml.probs(wires=0)]

    @pytest.mark.xfail(reason="dynamic shapes not supported [sc-85868]")
    def test_tape_transform_dynamic_shapes(
        self, enable_disable_dynamic_shapes
    ):  # pylint: disable=unused-argument
        """Test that using fallback transforms with dynamic shapes works correctly."""

        def f(x):
            qml.Z(0)
            dim = qml.math.shape(x)[0]
            n_wires = jnp.log2(dim).astype(int)
            qml.QubitUnitary(x, jnp.arange(n_wires))
            qml.Hadamard(0)
            return qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)

        jaxpr = jax.make_jaxpr(f, abstracted_axes=("a",))(jnp.eye(2))
        dummy_targs = (0, 0)
        transformed_jaxpr = z_to_hadamard.plxpr_transform(
            jaxpr.jaxpr, jaxpr.consts, dummy_targs, {}, jnp.eye(2)
        )
        collector = CollectOpsandMeas()
        collector.eval(transformed_jaxpr.jaxpr, transformed_jaxpr.consts, jnp.eye(2))

    def test_tape_transform_for_loop(self):
        """Test that transforming jaxpr with for_loops unrolls the loop and applies the
        transform correctly."""

        def f(x):
            qml.Z(0)
            qml.RX(x, 0)

            @qml.for_loop(3)
            def loop_fn(i):
                qml.Z(i)

            loop_fn()

            return qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)

        jaxpr = jax.make_jaxpr(f)(1.5)
        assert jaxpr.eqns[2].primitive == for_loop_prim
        dummy_targs = (0, 0)
        transformed_jaxpr = z_to_hadamard.plxpr_transform(
            jaxpr.jaxpr, jaxpr.consts, dummy_targs, {}, 1.5
        )
        assert all(eqn.primitive != for_loop_prim for eqn in transformed_jaxpr.eqns)
        collector = CollectOpsandMeas()
        collector.eval(transformed_jaxpr.jaxpr, transformed_jaxpr.consts, 1.5)

        ops = collector.state["ops"]
        assert ops == [
            qml.Hadamard(0),
            qml.RX(1.5, 0),
            qml.Hadamard(0),
            qml.Hadamard(1),
            qml.Hadamard(2),
        ]
        meas = collector.state["measurements"]
        assert meas == [qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)]

    @pytest.mark.xfail(reason="While loops are unsupported [sc-85869]")
    def test_tape_transform_while_loop(self):
        """Test that transforming jaxpr with while_loops unrolls the loop and applies the
        transform correctly."""

        def f(x, n):
            qml.Z(0)
            qml.RX(x, 0)
            w = 0

            @qml.while_loop(lambda i: i < n)
            def loop_fn(i):
                qml.Z(i)
                return i + 1

            loop_fn(w)

            return qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)

        jaxpr = jax.make_jaxpr(f, static_argnums=1)(1.5, 3)
        assert jaxpr.eqns[2].primitive == while_loop_prim
        dummy_targs = (0, 0)
        transformed_jaxpr = expval_z_obs_to_x_obs.plxpr_transform(
            jaxpr.jaxpr, jaxpr.consts, dummy_targs, {}, 1.5, 3
        )
        assert all(eqn.primitive != while_loop_prim for eqn in transformed_jaxpr.eqns)
        collector = CollectOpsandMeas()
        collector.eval(transformed_jaxpr.jaxpr, transformed_jaxpr.consts, 1.5, 3)

        ops = collector.state["ops"]
        assert ops == [
            qml.Hadamard(0),
            qml.RX(1.5, 0),
            qml.Hadamard(0),
            qml.Hadamard(1),
            qml.Hadamard(2),
        ]
        meas = collector.state["measurements"]
        assert meas == [qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)]

    def test_tape_transform_cond(self):
        """Test that transforming jaxpr with cond flattens the conditional and applies the
        transform correctly."""

        def f(x):

            @qml.cond(x > 3)
            def cond_fn():
                qml.Z(0)
                qml.X(0)

            @cond_fn.else_if(x > 2)
            def _():
                qml.Z(0)
                qml.Y(0)

            @cond_fn.otherwise
            def _():
                qml.Z(0)
                qml.T(0)

            cond_fn()

            return qml.expval(qml.Z(0)), qml.state()

        dummy_targs = (0, 0)

        # True branch
        jaxpr = jax.make_jaxpr(f, static_argnums=0)(3.5)
        assert jaxpr.eqns[0].primitive == cond_prim
        transformed_jaxpr = z_to_hadamard.plxpr_transform(
            jaxpr.jaxpr, jaxpr.consts, dummy_targs, {}
        )
        assert all(eqn.primitive != cond_prim for eqn in transformed_jaxpr.eqns)
        collector = CollectOpsandMeas()
        collector.eval(transformed_jaxpr.jaxpr, transformed_jaxpr.consts)
        ops = collector.state["ops"]
        assert ops == [qml.Hadamard(0), qml.X(0)]
        meas = collector.state["measurements"]
        assert meas == [qml.expval(qml.Z(0)), qml.state()]

        # Else if branch
        jaxpr = jax.make_jaxpr(f, static_argnums=0)(2.5)
        assert jaxpr.eqns[0].primitive == cond_prim
        transformed_jaxpr = z_to_hadamard.plxpr_transform(
            jaxpr.jaxpr, jaxpr.consts, dummy_targs, {}
        )
        assert all(eqn.primitive != cond_prim for eqn in transformed_jaxpr.eqns)
        collector = CollectOpsandMeas()
        collector.eval(transformed_jaxpr.jaxpr, transformed_jaxpr.consts)
        ops = collector.state["ops"]
        assert ops == [qml.Hadamard(0), qml.Y(0)]
        meas = collector.state["measurements"]
        assert meas == [qml.expval(qml.Z(0)), qml.state()]

        # Else branch
        jaxpr = jax.make_jaxpr(f, static_argnums=0)(1.5)
        assert jaxpr.eqns[0].primitive == cond_prim
        transformed_jaxpr = z_to_hadamard.plxpr_transform(
            jaxpr.jaxpr, jaxpr.consts, dummy_targs, {}
        )
        assert all(eqn.primitive != cond_prim for eqn in transformed_jaxpr.eqns)
        collector = CollectOpsandMeas()
        collector.eval(transformed_jaxpr.jaxpr, transformed_jaxpr.consts)
        ops = collector.state["ops"]
        assert ops == [qml.Hadamard(0), qml.T(0)]
        meas = collector.state["measurements"]
        assert meas == [qml.expval(qml.Z(0)), qml.state()]

    def test_tape_transform_multiple_transforms(self):
        """Test that multiple fallback transforms are applied correctly."""

        @qml.capture.expand_plxpr_transforms
        @shift_rx_to_end
        @partial(z_to_hadamard, dummy_arg1=0, dummy_arg2=0)
        def f(x):
            qml.Z(0)
            qml.RX(x, 0)
            qml.Hadamard(0)
            return qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)

        jaxpr = jax.make_jaxpr(f)(1.5)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 1.5)

        ops = collector.state["ops"]
        assert ops == [qml.Hadamard(0), qml.Hadamard(0), qml.RX(1.5, 0)]
        meas = collector.state["measurements"]
        assert meas == [qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)]

    def test_tape_transform_first_plxpr_transform_last(self):
        """Test that applying a plxpr transform after a fallback transform works as expected."""

        @qml.capture.expand_plxpr_transforms
        @partial(expval_z_obs_to_x_obs, dummy_arg1=0, dummy_arg2=0)
        @partial(z_to_hadamard, dummy_arg1=0, dummy_arg2=0)
        def f(x):
            qml.Z(0)
            qml.RX(x, 0)
            qml.Hadamard(0)
            return qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)

        jaxpr = jax.make_jaxpr(f)(1.5)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 1.5)

        ops = collector.state["ops"]
        assert ops == [qml.Hadamard(0), qml.RX(1.5, 0), qml.Hadamard(0)]
        meas = collector.state["measurements"]
        assert meas == [qml.expval(qml.X(0)), qml.state(), qml.probs(wires=0)]

    def test_plxpr_transform_first_tape_transform_last(self):
        """Test that applying a fallback transform after a plxpr transform works as expected."""

        @qml.capture.expand_plxpr_transforms
        @partial(z_to_hadamard, dummy_arg1=0, dummy_arg2=0)
        @partial(expval_z_obs_to_x_obs, dummy_arg1=0, dummy_arg2=0)
        def f(x):
            qml.Z(0)
            qml.RX(x, 0)
            qml.Hadamard(0)
            return qml.expval(qml.Z(0)), qml.state(), qml.probs(wires=0)

        jaxpr = jax.make_jaxpr(f)(1.5)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 1.5)

        ops = collector.state["ops"]
        assert ops == [qml.Hadamard(0), qml.RX(1.5, 0), qml.Hadamard(0)]
        meas = collector.state["measurements"]
        assert meas == [qml.expval(qml.X(0)), qml.state(), qml.probs(wires=0)]
