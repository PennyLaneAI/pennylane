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

import numpy as np
import pytest

import pennylane as qml
from pennylane.capture.transforms import MapWires
from pennylane.transforms.core import transform

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


@transform
def z_to_hadamard(
    tape, dummy_arg1, dummy_arg2, dummy_kwarg1=None, dummy_kwarg2=None
):  # pylint: disable=unused-argument
    """Transform that converts Z gates to H gates."""
    new_ops = [qml.H(wires=op.wires) if isinstance(op, qml.Z) else op for op in tape.operations]
    return [qml.tape.QuantumScript(new_ops, tape.measurements)], lambda res: res[0]


@transform
def expval_z_obs_to_x_obs(
    tape, dummy_arg1, dummy_arg2, dummy_kwarg1=None, dummy_kwarg2=None
):  # pylint: disable=unused-argument
    """Transform that converts Z observables for expectation values to X observables."""
    new_measurements = [
        (
            qml.expval(qml.X(mp.wires))
            if isinstance(mp, qml.measurements.ExpectationMP) and isinstance(mp.obs, qml.Z)
            else mp
        )
        for mp in tape.measurements
    ]
    return [qml.tape.QuantumScript(tape.operations, new_measurements)], lambda res: res[0]


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

    def test_transform_qnode_capture(self):
        """Test that a transformed QNode is captured correctly."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def func(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        args = (1.5,)
        targs = [0, 1]
        tkwargs = {"dummy_kwarg1": "foo", "dummy_kwarg2": "bar"}

        transformed_func = z_to_hadamard(func, *targs, **tkwargs)

        jaxpr = jax.make_jaxpr(transformed_func)(*args)
        assert (transform_eqn := jaxpr.eqns[0]).primitive == z_to_hadamard._primitive

        params = transform_eqn.params
        qnode_jaxpr = params["inner_jaxpr"]
        assert qnode_jaxpr.eqns[0].primitive == qml.capture.qnode_prim

        qfunc_jaxpr = qnode_jaxpr.eqns[0].params["qfunc_jaxpr"]
        expected_jaxpr = jax.make_jaxpr(func)(*args).eqns[0].params["qfunc_jaxpr"]
        for eqn1, eqn2 in zip(qfunc_jaxpr.eqns, expected_jaxpr.eqns, strict=True):
            assert eqn1.primitive == eqn2.primitive

    def test_transform_primitive_eval_not_implemented(self):
        """Test JAXPR containing a transform primitive cannot be evaluated due to a NotImplementedError."""

        def func(x):
            y = x * 5
            return y**0.5

        args = (1.5,)
        targs = (0, 1)
        tkwargs = {"dummy_kwarg1": "foo", "dummy_kwarg2": "bar"}

        transformed_func = z_to_hadamard(func, *targs, **tkwargs)
        jaxpr = jax.make_jaxpr(transformed_func)(*args)

        with pytest.raises(NotImplementedError):
            _ = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)[0]

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
        assert jaxpr.eqns[0].primitive == qml.capture.qnode_prim

        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == z_to_hadamard._primitive

        loop_jaxpr = qfunc_jaxpr.eqns[0].params["inner_jaxpr"]
        assert loop_jaxpr.eqns[0].primitive == qml.capture.primitives.for_loop_prim

        loop_body_jaxpr = loop_jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert loop_body_jaxpr.eqns[0].primitive == qml.X._primitive
        assert loop_body_jaxpr.eqns[1].primitive == qml.X._primitive

        assert qfunc_jaxpr.eqns[1].primitive == qml.Z._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive


# pylint: disable=protected-access, expression-not-assigned
class TestMapWiresTransform:
    """Unit tests for the MapWiresTransform class."""

    def test_qnode_simple(self):
        """Test that a simple qnode is transformed correctly."""

        const = 0.1

        @MapWires(wire_map={0: 1, 1: 2, 2: 3, 3: 2})
        @qml.qnode(qml.device("default.qubit", wires=4))
        def f(x):
            qml.RZ(x, 0)
            qml.PhaseShift(const, 1)
            qml.Hadamard(2)
            return qml.expval(qml.PauliZ(3)), qml.probs(wires=[1])

        jaxpr = jax.make_jaxpr(f)(0.1)
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

        assert len(inner_jaxpr.eqns) == 6

        assert inner_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert inner_jaxpr.eqns[0].invars[-1].val == 1

        assert inner_jaxpr.eqns[1].primitive == qml.PhaseShift._primitive
        assert inner_jaxpr.eqns[1].invars[-1].val == 2

        assert inner_jaxpr.eqns[2].primitive == qml.Hadamard._primitive
        assert inner_jaxpr.eqns[2].invars[-1].val == 3

        assert inner_jaxpr.eqns[3].primitive == qml.PauliZ._primitive
        assert inner_jaxpr.eqns[3].invars[-1].val == 2

        assert inner_jaxpr.eqns[4].primitive == qml.measurements.ExpectationMP._obs_primitive

        assert inner_jaxpr.eqns[5].primitive == qml.measurements.ProbabilityMP._wires_primitive
        assert inner_jaxpr.eqns[5].invars[-1].val == 2

    def test_qnode_nested_ops(self):
        """Test that a qnode with nested operations is transformed correctly."""

        const = 0.1

        @MapWires(wire_map={0: 1, 1: 2})
        @qml.qnode(qml.device("default.qubit", wires=4))
        def transformed_circuit(x):
            qml.RZ(const, 0) + qml.RX(x, 0)
            qml.PhaseShift(const, 1) @ qml.RX(x, 0) @ qml.X(0)
            qml.Hadamard(2) @ qml.RX(x, 0) + 4 * qml.RX(const, 0)
            return qml.state()

        jaxpr = jax.make_jaxpr(transformed_circuit)(0.1)
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

        # ensure every operation and measurement is captured
        assert len(inner_jaxpr.eqns) == 15

        assert inner_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert inner_jaxpr.eqns[0].invars[-1].val == 1

        assert inner_jaxpr.eqns[1].primitive == qml.RX._primitive
        assert inner_jaxpr.eqns[1].invars[-1].val == 1

        assert inner_jaxpr.eqns[2].primitive == qml.ops.Sum._primitive

        assert inner_jaxpr.eqns[3].primitive == qml.PhaseShift._primitive
        assert inner_jaxpr.eqns[3].invars[-1].val == 2

        assert inner_jaxpr.eqns[4].primitive == qml.RX._primitive
        assert inner_jaxpr.eqns[4].invars[-1].val == 1

        assert inner_jaxpr.eqns[5].primitive == qml.ops.Prod._primitive

        assert inner_jaxpr.eqns[6].primitive == qml.X._primitive
        assert inner_jaxpr.eqns[6].invars[-1].val == 1

        assert inner_jaxpr.eqns[7].primitive == qml.ops.Prod._primitive

        assert inner_jaxpr.eqns[8].primitive == qml.Hadamard._primitive
        assert inner_jaxpr.eqns[8].invars[-1].val == 2

        assert inner_jaxpr.eqns[9].primitive == qml.RX._primitive
        assert inner_jaxpr.eqns[9].invars[-1].val == 1

        assert inner_jaxpr.eqns[10].primitive == qml.ops.Prod._primitive

        assert inner_jaxpr.eqns[11].primitive == qml.RX._primitive
        assert inner_jaxpr.eqns[11].invars[-1].val == 1

        assert inner_jaxpr.eqns[12].primitive == qml.ops.SProd._primitive
        assert inner_jaxpr.eqns[13].primitive == qml.ops.Sum._primitive

        assert inner_jaxpr.eqns[14].primitive == qml.measurements.StateMP._wires_primitive

    def test_qnode_controlled_ops(self):
        """Test that a qnode with controlled operations is transformed correctly."""

        @MapWires(wire_map={0: 1, 1: 2, 2: 3, 3: 2})
        @qml.qnode(qml.device("default.qubit", wires=4))
        def f(x):
            qml.CNOT(wires=[0, 1])
            qml.CY(wires=[1, 2])
            qml.CZ(wires=[2, 3])
            qml.CRX(x, wires=[0, 1])
            qml.CRY(x, wires=[1, 2])
            qml.CRZ(x, wires=[2, 3])
            return qml.var(
                qml.QubitUnitary(
                    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]), wires=[0, 1]
                )
            )

        jaxpr = jax.make_jaxpr(f)(0.1)
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

        assert len(inner_jaxpr.eqns) == 8

        assert inner_jaxpr.eqns[0].primitive == qml.CNOT._primitive
        assert inner_jaxpr.eqns[0].invars[0].val == 1
        assert inner_jaxpr.eqns[0].invars[1].val == 2

        assert inner_jaxpr.eqns[1].primitive == qml.CY._primitive
        assert inner_jaxpr.eqns[1].invars[0].val == 2
        assert inner_jaxpr.eqns[1].invars[1].val == 3

        assert inner_jaxpr.eqns[2].primitive == qml.CZ._primitive
        assert inner_jaxpr.eqns[2].invars[0].val == 3
        assert inner_jaxpr.eqns[2].invars[1].val == 2

        assert inner_jaxpr.eqns[3].primitive == qml.CRX._primitive
        assert inner_jaxpr.eqns[3].invars[1].val == 1
        assert inner_jaxpr.eqns[3].invars[2].val == 2

        assert inner_jaxpr.eqns[4].primitive == qml.CRY._primitive
        assert inner_jaxpr.eqns[4].invars[1].val == 2
        assert inner_jaxpr.eqns[4].invars[2].val == 3

        assert inner_jaxpr.eqns[5].primitive == qml.CRZ._primitive
        assert inner_jaxpr.eqns[5].invars[1].val == 3
        assert inner_jaxpr.eqns[5].invars[2].val == 2

        assert inner_jaxpr.eqns[6].primitive == qml.QubitUnitary._primitive
        assert inner_jaxpr.eqns[6].invars[1].val == 1
        assert inner_jaxpr.eqns[6].invars[2].val == 2

        assert inner_jaxpr.eqns[7].primitive == qml.measurements.VarianceMP._obs_primitive

    def test_qnode_for_loop(self):
        """Test that a qnode with a for loop is transformed correctly."""

        @MapWires(wire_map={0: 1})
        @qml.qnode(qml.device("default.qubit", wires=4))
        def f(x):
            @qml.for_loop(3)
            def g(i):
                qml.RZ(x, 0)

            g()
            return qml.expval(qml.PauliZ(3))

        jaxpr = jax.make_jaxpr(f)(0.1)
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        loop_jaxpr = inner_jaxpr.eqns[0].params["jaxpr_body_fn"]

        assert len(loop_jaxpr.eqns) == 1

        assert loop_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert loop_jaxpr.eqns[0].invars[-1].val == 1

    def test_qnode_while_loop(self):
        """Test that a qnode with a while loop is transformed correctly."""

        @MapWires(wire_map={0: 1})
        @qml.qnode(qml.device("default.qubit", wires=4))
        def f(x):
            @qml.while_loop(lambda i: i < 3)
            def g(i):
                qml.RZ(x, 0)
                return i + 1

            g(0)
            return qml.expval(qml.PauliZ(3))

        jaxpr = jax.make_jaxpr(f)(0.1)
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        loop_jaxpr = inner_jaxpr.eqns[0].params["jaxpr_body_fn"]

        assert len(loop_jaxpr.eqns) == 2

        assert loop_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert loop_jaxpr.eqns[0].invars[-1].val == 1

    def test_qnode_conditional(self):
        """Test that a qnode with a conditional is transformed correctly."""

        @MapWires(wire_map={0: 1})
        @qml.qnode(qml.device("default.qubit", wires=4))
        def f(x):
            @qml.cond(x > 0.5)
            def g():
                qml.RZ(x, 0)

            g()
            return qml.expval(qml.PauliZ(3))

        jaxpr = jax.make_jaxpr(f)(0.1)
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        cond_jaxpr = inner_jaxpr.eqns[1].params["jaxpr_branches"][0]

        assert len(cond_jaxpr.eqns) == 1

        assert cond_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert cond_jaxpr.eqns[0].invars[-1].val == 1

    def test_invalid_wire_map_keys(self):
        """Test that invalid wire mappings raise an error."""
        with pytest.raises(ValueError, match="Wire map keys must be integer constants"):
            MapWires(wire_map={"a": 1, 1: 2})

    def test_invalid_wire_map_values(self):
        """Test that invalid wire mappings raise an error."""
        with pytest.raises(ValueError, match="Wire map values must be integer constants"):
            MapWires(wire_map={0: "a", 1: 2})

    def test_qnode_batched_parameters(self):
        """Test qnode transformation with batched parameters."""

        @MapWires(wire_map={0: 1, 1: 2})
        @qml.qnode(qml.device("default.qubit", wires=4))
        def f(x):
            qml.RZ(x, 0)
            qml.PhaseShift(0.1, 1)
            return qml.expval(qml.PauliZ(2))

        x = jax.numpy.array([0.1, 0.2, 0.3])
        jaxpr = jax.make_jaxpr(f)(x)
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

        assert len(inner_jaxpr.eqns) == 4

        assert inner_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert inner_jaxpr.eqns[0].invars[-1].val == 1

        assert inner_jaxpr.eqns[1].primitive == qml.PhaseShift._primitive
        assert inner_jaxpr.eqns[1].invars[-1].val == 2

        assert inner_jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert inner_jaxpr.eqns[2].invars[-1].val == 2

        assert inner_jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive
