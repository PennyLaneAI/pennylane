# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Integration tests for differentiation with capture.
"""

import pytest

import pennylane as qp

pytestmark = [pytest.mark.jax, pytest.mark.capture]

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.mark.parametrize("diff_method", ("finite-diff", "adjoint", "backprop"))
class TestJVPIntegration:

    @pytest.mark.parametrize("grad_f", (jax.grad, jax.jacobian))
    def test_simple_circuit(self, grad_f, diff_method):
        """Test accurate results for a simple, single parameter circuit."""

        @qp.qnode(qp.device("default.qubit", wires=1), diff_method=diff_method)
        def circuit(x):
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        x = 0.5
        result = grad_f(circuit)(x)

        assert qp.math.allclose(result, -jnp.sin(x))

    @pytest.mark.parametrize("argnums", ((0,), (1,), (0, 1)))
    def test_multi_inputs(self, diff_method, argnums):
        """Test gradients can be computed with multiple scalar inputs."""

        @qp.qnode(qp.device("default.qubit", wires=2), diff_method=diff_method)
        def circuit(x, y):
            qp.RX(x, 0)
            qp.RY(y, 1)
            qp.CNOT((0, 1))
            return qp.expval(qp.Z(1))

        x = 1.2
        y = jnp.array(2.0)
        grad = jax.grad(circuit, argnums=argnums)(x, y)

        grad_x = -jnp.sin(x) * jnp.cos(y)
        grad_y = -jnp.cos(x) * jnp.sin(y)
        g = [grad_x, grad_y]
        expected_grad = [g[i] for i in argnums]

        assert qp.math.allclose(grad, expected_grad)

    def test_array_input(self, diff_method):
        """Test that we can differentiate a circuit with an array input."""

        @qp.qnode(qp.device("default.qubit", wires=3), diff_method=diff_method)
        def circuit(x):
            qp.RX(x[0], 0)
            qp.RX(x[1], 1)
            qp.RX(x[2], 2)
            return qp.expval(qp.Z(0) @ qp.Z(1) @ qp.Z(2))

        x = jnp.array([0.5, 1.0, 1.5])
        grad = jax.grad(circuit)(x)
        assert grad.shape == (3,)

        grad0 = -jnp.sin(x[0]) * jnp.cos(x[1]) * jnp.cos(x[2])
        assert qp.math.allclose(grad[0], grad0)
        grad1 = jnp.cos(x[0]) * -jnp.sin(x[1]) * jnp.cos(x[2])
        assert qp.math.allclose(grad[1], grad1)
        grad2 = jnp.cos(x[0]) * jnp.cos(x[1]) * -jnp.sin(x[2])
        assert qp.math.allclose(grad[2], grad2)

    def test_jacobian_multiple_outputs(self, diff_method):
        """Test that finite diff can handle multiple outputs."""

        @qp.qnode(qp.device("default.qubit", wires=1), diff_method=diff_method)
        def circuit(x):
            qp.RX(x, 0)
            mps = [
                qp.expval(qp.Z(0)),
                qp.expval(qp.Y(0)),
                qp.expval(qp.X(0)),
            ]
            if diff_method == "adjoint":
                return mps
            return mps + [qp.probs(wires=0)]

        x = jnp.array(-0.65)
        jac = jax.jacobian(circuit)(x)

        assert qp.math.allclose(jac[0], -jnp.sin(x))
        assert qp.math.allclose(jac[1], -jnp.cos(x))
        assert qp.math.allclose(jac[2], 0)

        if diff_method != "adjoint":
            # probs = [cos(x/2)**2, sin(x/2)**2]
            probs_jac = [-jnp.cos(x / 2) * jnp.sin(x / 2), jnp.sin(x / 2) * jnp.cos(x / 2)]
            assert qp.math.allclose(jac[3], probs_jac)

    def test_classical_control_flow(self, diff_method):
        """Test that classical control flow can exist inside the circuit."""

        if diff_method == "adjoint":
            pytest.xfail("adjoint cannot handle control flow")

        @qp.qnode(qp.device("default.qubit", wires=4), diff_method=diff_method)
        def circuit(x):
            @qp.for_loop(3)
            def f(i):
                qp.cond(i < 2, qp.RX, false_fn=qp.RZ)(x[i], i)

            f()
            return [qp.expval(qp.Z(i)) for i in range(3)]

        x = jnp.array([0.2, 0.6, 1.0])
        jac = jax.jacobian(circuit)(x)

        assert qp.math.allclose(jac[0][0], -jnp.sin(x[0]))
        assert qp.math.allclose(jac[0][1:], 0)

        assert qp.math.allclose(jac[1][0], 0)
        assert qp.math.allclose(jac[1][1], -jnp.sin(x[1]))
        assert qp.math.allclose(jac[1][2], 0)

        # i = 2 applies RZ. grad should be zero
        assert qp.math.allclose(jac[2], jnp.zeros(3))

    def test_pre_and_postprocessing(self, diff_method):
        """Test that we can chain together pre and post processing."""

        @qp.qnode(qp.device("default.qubit", wires=4), diff_method=diff_method)
        def circuit(x):
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        def workflow(y):
            return 2 * circuit(y**2)

        x = jnp.array(-0.9)
        jac = jax.jacobian(workflow)(x)

        # res = 2*cos(y**2)
        # dres = 2 * -sin(y**2) * 2 *y
        expected = 2 * -jnp.sin(x**2) * 2 * x
        assert qp.math.allclose(jac, expected)
