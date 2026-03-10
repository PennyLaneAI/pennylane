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
This file contains tests for using finite difference derivatives
with program capture enabled.
"""

import pytest

import pennylane as qml

pytestmark = [pytest.mark.jax, pytest.mark.capture]

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def test_warning_float32():
    """Test that a warning is raised if trainable inputs are float32."""

    @qml.qnode(qml.device("default.qubit", wires=1), diff_method="finite-diff")
    def circuit(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0))

    with pytest.warns(
        UserWarning, match="Detected 32 bits precision parameter with finite differences."
    ):
        jax.grad(circuit)(jnp.array(0.5, dtype=jnp.float32))

    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.warns(UserWarning, match="Detected 32 bits precision with finite differences."):
            jax.grad(circuit)(0.5)
    finally:
        jax.config.update("jax_enable_x64", True)


class TestGradients:

    @pytest.mark.parametrize(
        "kwargs",
        ({"approx_order": 2}, {"strategy": "backward"}, {"approx_order": 2, "strategy": "center"}),
    )
    @pytest.mark.parametrize("grad_f", (jax.grad, jax.jacobian))
    def test_simple_circuit(self, grad_f, kwargs):
        """Test accurate results for a simple, single parameter circuit."""

        @qml.qnode(
            qml.device("default.qubit", wires=1), diff_method="finite-diff", gradient_kwargs=kwargs
        )
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        x = 0.5
        result = grad_f(circuit)(x)

        assert qml.math.allclose(result, -jnp.sin(x))

    def test_hessian(self):
        """Test that higher order derivatives like the hessian can be computed."""

        # h=1e-7 gets really noisy for some reason
        @qml.qnode(
            qml.device("default.qubit", wires=4),
            diff_method="finite-diff",
            gradient_kwargs={"h": 1e-6},
        )
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        x = jnp.array(0.5, dtype=jnp.float64)
        hess = jax.grad(jax.grad(circuit))(x)
        assert qml.math.allclose(hess, -jnp.cos(x), atol=5e-4)  # gets noisy

    @pytest.mark.parametrize("argnums", ((0,), (0, 1)))
    def test_jaxpr_contents(self, argnums):
        """Make some tests on the captured jaxpr to assert we are doing the correct thing."""

        @qml.qnode(
            qml.device("default.qubit", wires=1),
            diff_method="finite-diff",
            gradient_kwargs={"h": 1e-4},
        )
        def circuit(x, y):
            qml.RX(x, 0)
            qml.RY(y, 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(jax.grad(circuit, argnums=argnums))(0.5, 1.2)

        qnode_eqns = [eqn for eqn in jaxpr.eqns if eqn.primitive.name == "qnode"]
        assert len(qnode_eqns) == 1 + len(argnums)

        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "add":
                # only addition eqns are adding h to var
                assert eqn.invars[1].val == 1e-4
