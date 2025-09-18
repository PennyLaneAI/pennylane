# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for default.qubits adjoint execute_and_jvp method."""
from functools import partial

import pytest

import pennylane as qml

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# pylint: disable=wrong-import-position
from pennylane.devices.qubit.jaxpr_adjoint import execute_and_jvp

pytestmark = [pytest.mark.jax, pytest.mark.capture]


class TestErrors:
    """ "Test explicit errors for various unsupported cases."""

    def test_no_differentiable_op_math(self):
        """Test an error is raised if we have differentiable operator arithmetic."""

        def f(x):
            qml.adjoint(qml.RX(x, 0))
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5)

        with pytest.raises(NotImplementedError):
            execute_and_jvp(jaxpr.jaxpr, (0.5,), (1.0,), num_wires=1)

    def test_only_expvals(self):
        """Test that an error is raised for other measurements."""

        def f(x):
            qml.RX(x, 0)
            return qml.probs(wires=0)

        jaxpr = jax.make_jaxpr(f)(0.5).jaxpr

        with pytest.raises(NotImplementedError, match="expectations of observables"):
            execute_and_jvp(jaxpr, (0.5,), (1.0,), num_wires=1)

    def test_no_for_loop(self):
        """Generic test for a primitive without a registered jvp rule."""

        def f(x):
            @qml.for_loop(3)
            def g(i):
                qml.RX(x, i)

            g()
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5).jaxpr

        with pytest.raises(NotImplementedError, match="does not have a jvp rule."):
            execute_and_jvp(jaxpr, (0.5,), (1.0,), num_wires=1)

    def test_at_most_one_trainable_param(self):
        """Test that multiple arguments are not trainable."""

        def f(x, y, z):
            qml.Rot(x, y, z, 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5, 0.6, 0.7).jaxpr

        with pytest.raises(NotImplementedError, match="only differentiable parameters in the 0"):
            execute_and_jvp(jaxpr, (1.0, 2.0, 3.0), (1.0, 1.0, 1.0), num_wires=1)

    def test_capture_renabled_if_generator_failure(self):
        """Test that capture stays enabled if the generator is undefined."""

        def f(x):
            qml.Rot(x, 1.2, 2.3, wires=0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5).jaxpr

        with pytest.raises(ValueError):
            execute_and_jvp(jaxpr, (0.5,), (1.0,), num_wires=1)

        assert qml.capture.enabled()

    def test_bad_adjoint_op(self):
        """Test capture stays enabled if the adjoint of an operator throws an error."""

        class MyOp(qml.operation.Operator):

            def adjoint(self):
                raise ValueError

            def matrix(self):
                return qml.X.compute_matrix()

        def f():
            MyOp(wires=0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)().jaxpr
        with pytest.raises(ValueError):
            execute_and_jvp(jaxpr, (), (), num_wires=1)

        assert qml.capture.enabled()


class TestCorrectResults:

    def test_abstract_zero_tangent(self):
        """Test we get the derivatives will be an ad.Zero if the result is independent of the input."""

        def f(x):
            _ = x + 1
            qml.RX(0.5, 0)
            return qml.expval(qml.Z(0))

        args = (0.5,)
        tangents = (jax.interpreters.ad.Zero(jax.core.ShapedArray((), float)),)

        jaxpr = jax.make_jaxpr(f)(0.5)
        [results], [dresults] = execute_and_jvp(jaxpr.jaxpr, args, tangents, num_wires=1)
        assert qml.math.allclose(results, jnp.cos(0.5))
        assert isinstance(dresults, jax.interpreters.ad.Zero)

    @pytest.mark.parametrize("use_jit", (False, True))
    def test_basic_circuit(self, use_jit):
        """Test the calculation of results and jvp for a basic circuit."""

        def f(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5)

        args = (0.82,)
        tangents = (2.0,)

        executor = partial(execute_and_jvp, jaxpr.jaxpr, num_wires=1)
        if use_jit:
            executor = jax.jit(executor)

        results, dresults = executor(args, tangents)

        assert len(results) == 1
        assert qml.math.allclose(results, jnp.cos(args[0]))
        assert len(dresults) == 1
        assert qml.math.allclose(dresults[0], tangents[0] * -jnp.sin(args[0]))

    def test_multiple_in(self):
        """Test that we can differentiate multiple inputs."""

        def f(x, y):
            qml.RX(x, 0)
            qml.RY(y, 1)
            qml.CNOT((0, 1))
            return qml.expval(qml.Y(0))

        x = jnp.array(0.5)
        y = jnp.array(1.2)
        dx = jnp.array(2.0)
        dy = jnp.array(3.0)

        jaxpr = jax.make_jaxpr(f)(x, y).jaxpr

        [res], [dres] = execute_and_jvp(jaxpr, (x, y), (dx, dy), num_wires=2)

        expected = -jnp.sin(x) * jnp.sin(y)
        assert qml.math.allclose(res, expected)

        expected_dres = dx * -jnp.cos(x) * jnp.sin(y) + dy * -jnp.sin(x) * jnp.cos(y)
        assert qml.math.allclose(dres, expected_dres)

    def test_multiple_output(self):
        """Test we can compute the jvp with multiple outputs."""

        def f(x):
            qml.RX(x, 0)
            return qml.expval(qml.X(0)), qml.expval(qml.Y(0)), qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5).jaxpr

        x = -0.5
        res, dres = execute_and_jvp(jaxpr, (x,), (2.0,), num_wires=1)

        assert qml.math.allclose(res[0], 0)
        assert qml.math.allclose(res[1], -jnp.sin(x))
        assert qml.math.allclose(res[2], jnp.cos(x))

        assert qml.math.allclose(dres[0], 0)
        assert qml.math.allclose(dres[1], 2.0 * -jnp.cos(x))
        assert qml.math.allclose(dres[2], 2.0 * -jnp.sin(x))

    def test_classical_preprocessing(self):
        """Test that we can perform classical preprocessing of variables."""

        def f(x):
            y = x**2
            qml.RX(y[0], 0)
            qml.RX(y[1], 1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

        x = jnp.array([1.5, 2.5])
        dx = jnp.array([2.0, 3.0])
        jaxpr = jax.make_jaxpr(f)(x).jaxpr

        [res], [dres] = execute_and_jvp(jaxpr, (x,), (dx,), num_wires=2)

        expected = jnp.cos(x[0] ** 2) * jnp.cos(x[1] ** 2)
        assert qml.math.allclose(res, expected)
        dexpected = (
            -jnp.sin(x[0] ** 2) * 2 * x[0] * dx[0] * jnp.cos(x[1] ** 2)
            + jnp.cos(x[0] ** 2) * -jnp.sin(x[1] ** 2) * 2 * x[1] * dx[1]
        )
        assert qml.math.allclose(dres, dexpected)

    def test_jaxpr_consts(self):
        """Test that we can execute jaxpr with consts."""

        def f():
            x = jnp.array([1.0])
            qml.RX(x[0], 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)().jaxpr

        const = jnp.array([1.2])
        dconst = jnp.array([0.25])
        [res], [dres] = execute_and_jvp(jaxpr, (const,), (dconst,), num_wires=1)

        assert qml.math.allclose(res, jnp.cos(1.2))
        assert qml.math.allclose(dres, dconst[0] * -jnp.sin(1.2))
