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

from pennylane.core.transforms import CompilePipeline

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# pylint: disable=wrong-import-position
import pennylane as qp
from pennylane.capture.primitives import (
    qnode_prim,
    transform_prim,
)
from pennylane.transforms.core import transform

pytestmark = [pytest.mark.jax, pytest.mark.capture]


@transform
def z_to_hadamard(
    tape, dummy_arg1, dummy_arg2, dummy_kwarg1=None, dummy_kwarg2=None
):  # pylint: disable=unused-argument
    """Transform that converts Z gates to H gates."""
    new_ops = [qp.H(wires=op.wires) if isinstance(op, qp.Z) else op for op in tape.operations]
    return [tape.copy(operations=new_ops)], lambda res: res[0]


@transform
def shift_rx_to_end(tape):
    """Transform that moves all RX gates to the end of the operations list."""
    new_ops, rxs = [], []

    for op in tape.operations:
        if isinstance(op, qp.RX):
            rxs.append(op)
        else:
            new_ops.append(op)

    operations = new_ops + rxs
    new_tape = tape.copy(operations=operations)
    return [new_tape], lambda res: res[0]


def expval_z_obs_to_x_obs_plxpr(
    jaxpr, consts, targs, tkwargs, *args
):  # pylint: disable=unused-argument
    class ExpvalZToX(qp.capture.PlxprInterpreter):  # pylint: disable=too-few-public-methods
        """Expval Z to X plxpr implementation."""

        def interpret_measurement(self, meas):  # pylint: disable=arguments-renamed
            """Interpret measurement."""
            if isinstance(meas, qp.measurements.ExpectationMP) and isinstance(meas.obs, qp.PauliZ):
                return qp.expval(qp.X(meas.wires))

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
            qp.expval(qp.X(mp.wires))
            if isinstance(mp, qp.measurements.ExpectationMP) and isinstance(mp.obs, qp.Z)
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

        assert (transform_eqn := jaxpr.eqns[0]).primitive == transform_prim

        params = transform_eqn.params
        assert params["args_slice"] == (0, 1, None)
        assert params["consts_slice"] == (1, 1, None)
        assert params["targs_slice"] == (1, None, None)

        assert dict(params["tkwargs"]) == tkwargs
        assert params["transform"] == z_to_hadamard

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
        assert (transform_eqn := jaxpr.eqns[0]).primitive == transform_prim
        assert transform_eqn.params["transform"] == z_to_hadamard

        params = transform_eqn.params
        assert params["args_slice"] == (0, 2, None)
        assert params["consts_slice"] == (2, 2, None)
        assert params["targs_slice"] == (2, None, None)
        # Dicts are also converted to tuples
        assert dict(params["tkwargs"]) == tkwargs

        inner_jaxpr = params["inner_jaxpr"]
        expected_jaxpr = jax.make_jaxpr(func)(*args).jaxpr
        for eqn1, eqn2 in zip(inner_jaxpr.eqns, expected_jaxpr.eqns, strict=True):
            assert eqn1.primitive == eqn2.primitive

        # Verifying that transformed function can execute
        _ = transformed_func(*args)

    def test_transform_qnode_capture(self):
        """Test that a transformed QNode is captured correctly."""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def func(x):
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        args = (1.5,)
        targs = (0, 1)
        tkwargs = {"dummy_kwarg1": "foo", "dummy_kwarg2": "bar"}

        transformed_func = z_to_hadamard(func, *targs, **tkwargs)

        jaxpr = jax.make_jaxpr(transformed_func)(*args)
        assert (transform_eqn := jaxpr.eqns[0]).primitive == transform_prim

        qnode_jaxpr = transform_eqn.params["inner_jaxpr"]
        assert transform_eqn.params["transform"] == z_to_hadamard
        assert qnode_jaxpr.eqns[0].primitive == qnode_prim

        qnode = qnode_jaxpr.eqns[0].params["qnode"]
        expected_program = CompilePipeline()
        expected_program.add_transform(z_to_hadamard, *targs, **tkwargs)
        # Manually change targs from tuple to list
        expected_program[0]._args = tuple(targs)  # pylint: disable=protected-access
        assert qnode.compile_pipeline == expected_program

        qfunc_jaxpr = qnode_jaxpr.eqns[0].params["qfunc_jaxpr"]
        expected_jaxpr = jax.make_jaxpr(func)(*args).eqns[0].params["qfunc_jaxpr"]
        for eqn1, eqn2 in zip(qfunc_jaxpr.eqns, expected_jaxpr.eqns, strict=True):
            assert eqn1.primitive == eqn2.primitive

    def test_transform_primitive_eval(self):
        """Test that evaluating jaxpr containing a transform primitive does not apply
        the transform."""

        def func():
            return qp.expval(qp.Z(0))

        targs = (2, 3)
        tkwargs = {"dummy_kwarg1": "hello", "dummy_kwarg2": "world"}

        transformed_func = expval_z_obs_to_x_obs(func, *targs, **tkwargs)
        jaxpr = jax.make_jaxpr(transformed_func)()

        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        # If the inner jaxpr was transformed, the output would be <X>, but it should
        # stay as <Z> if it wasn't transformed
        assert res == [qp.expval(qp.Z(0))]

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
        assert (transform_eqn1 := jaxpr.eqns[0]).primitive == transform_prim
        assert transform_eqn1.params["transform"] == z_to_hadamard

        params1 = transform_eqn1.params
        assert params1["args_slice"] == (0, 1, None)
        assert params1["consts_slice"] == (1, 1, None)
        assert params1["targs_slice"] == (1, None, None)
        # Dicts are also converted to tuples
        assert dict(params1["tkwargs"]) == tkwargs1

        inner_jaxpr = params1["inner_jaxpr"]
        assert (transform_eqn2 := inner_jaxpr.eqns[0]).primitive == transform_prim
        assert transform_eqn2.params["transform"] == expval_z_obs_to_x_obs

        params2 = transform_eqn2.params
        assert params2["args_slice"] == (0, 1, None)
        assert params2["consts_slice"] == (1, 1, None)
        assert params2["targs_slice"] == (1, None, None)
        assert dict(params2["tkwargs"]) == tkwargs2

        inner_inner_jaxpr = params2["inner_jaxpr"]
        expected_jaxpr = jax.make_jaxpr(func)(*args).jaxpr
        for eqn1, eqn2 in zip(inner_inner_jaxpr.eqns, expected_jaxpr.eqns, strict=True):
            assert eqn1.primitive == eqn2.primitive

    def test_higher_order_primitives(self):
        """Test that transforms are captured correctly when used with higher-order primitives."""
        dev = qp.device("default.qubit", wires=5)

        targs = (0, 1)
        tkwargs = {"dummy_kwarg1": "foo", "dummy_kwarg2": "bar"}

        @qp.qnode(dev)
        def f():
            @z_to_hadamard(dummy_arg1=targs[0], dummy_arg2=targs[1], **tkwargs)
            @qp.for_loop(3)
            def g(i):
                qp.X(i)
                qp.X(i)

            g()
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        assert jaxpr.eqns[0].primitive == qnode_prim

        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == transform_prim
        assert qfunc_jaxpr.eqns[0].params["transform"] == z_to_hadamard

        loop_jaxpr = qfunc_jaxpr.eqns[0].params["inner_jaxpr"]
        assert loop_jaxpr.eqns[0].primitive == qp.capture.primitives.for_loop_prim

        loop_body_jaxpr = loop_jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert loop_body_jaxpr.eqns[0].primitive == qp.X._primitive
        assert loop_body_jaxpr.eqns[1].primitive == qp.X._primitive

        assert qfunc_jaxpr.eqns[1].primitive == qp.Z._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qp.measurements.ExpectationMP._obs_primitive

    @pytest.mark.usefixtures("enable_disable_dynamic_shapes")
    def test_qnode_returns_dynamic_shape_consts(self):
        """ "Test that a qnode that returns dynamic shapes can be transformed."""

        def workflow(shots: int):
            @qp.transform(pass_name="my_other_pass_name")
            @qp.transform(pass_name="cancel-inverses")
            @qp.qnode(qp.device("lightning.qubit", wires=1), shots=shots)
            def aloha():
                qp.Hadamard(wires=0)
                return qp.sample()

            return aloha()

        jaxpr = jax.make_jaxpr(workflow)(3)

        j = jaxpr.jaxpr
        assert j.outvars[0].aval.shape[0] is j.invars[0]
        assert j.outvars[0].aval.shape[1] == 1

        j2 = j.eqns[0].params["inner_jaxpr"]
        assert j2.outvars[0].aval.shape[0] is j2.constvars[0]
        assert j2.outvars[0].aval.shape[1] == 1

        j3 = j2.eqns[0].params["inner_jaxpr"]
        assert j3.outvars[0].aval.shape[0] is j3.constvars[0]
        assert j3.outvars[0].aval.shape[1] == 1

    @pytest.mark.usefixtures("enable_disable_dynamic_shapes")
    def test_qnode_return_dynamic_shapes_args(self):
        """Test that a function that returns a dynamic shape that depends on an argument can be transformed."""

        def workflow(shots: int):
            @qp.transform(pass_name="my_other_pass_name")
            def f(inner_shots):
                @qp.qnode(qp.device("lightning.qubit", wires=1), shots=inner_shots)
                def aloha():
                    return qp.sample()

                return aloha()

            return f(shots)

        jaxpr = jax.make_jaxpr(workflow)(3)

        j = jaxpr.jaxpr
        assert j.outvars[0].aval.shape[0] is j.invars[0]
        assert j.outvars[0].aval.shape[1] == 1

        j2 = j.eqns[0].params["inner_jaxpr"]
        assert j2.outvars[0].aval.shape[0] is j2.invars[0]
        assert j2.outvars[0].aval.shape[1] == 1
