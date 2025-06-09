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
"""Test Jax-based Catalyst-compatible QNG optimizer"""

import pytest

import pennylane as qml
import pennylane.numpy as pnp

dev_names = (
    "default.qubit",
    "lightning.qubit",
)


def circuit(params):
    """Simple circuit to use for testing."""
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    return qml.expval(qml.Z(wires=0))


class TestBasics:
    """Test basic properties of the QNGOptimizerJax."""

    def test_initialization_default(self):
        """Test that initializing QNGOptimizerJax with default values works."""
        opt = qml.QNGOptimizerJax()
        assert opt.stepsize == 0.01
        assert opt.approx == "block-diag"
        assert opt.lam == 0

    def test_initialization_custom(self):
        """Test that initializing QNGOptimizerJax with custom values works."""
        opt = qml.QNGOptimizerJax(stepsize=0.05, approx="diag", lam=1e-9)
        assert opt.stepsize == 0.05
        assert opt.approx == "diag"
        assert opt.lam == 1e-9

    def test_init_none_state(self):
        """Test that the QNGOptimizerJax state is initialized to `None`."""
        opt = qml.QNGOptimizerJax()
        state = opt.init([0.1, 0.2])
        assert state is None


class TestGradients:
    """Test gradient computation in the QNGOptimizerJax."""

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", dev_names)
    def test_get_grad_jax(self, dev_name):
        """Test `_get_grad` method with Jax interface."""

        import jax.numpy as jnp

        device = qml.device(dev_name, wires=2)
        qml_qnode = qml.QNode(circuit, device=device)

        opt = qml.QNGOptimizerJax()
        params = [0.1, 0.2]

        params_qml = pnp.array(params)
        grad_qml = qml.grad(qml_qnode)(params_qml)

        params_jax = jnp.array(params)
        grad_jax = opt._get_grad(qml_qnode, params_jax)

        assert qml.math.allclose(grad_qml, grad_jax)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", dev_names)
    def test_get_value_and_grad_jax(self, dev_name):
        """Test `_get_value_and_grad` method with Jax interface."""

        import jax.numpy as jnp

        device = qml.device(dev_name, wires=2)
        qml_qnode = qml.QNode(circuit, device=device)

        opt = qml.QNGOptimizerJax()
        params = [0.1, 0.2]

        params_qml = pnp.array(params)
        cost_qml = qml_qnode(params_qml)
        grad_qml = qml.grad(qml_qnode)(params_qml)

        params_jax = jnp.array(params)
        cost_jax, grad_jax = opt._get_value_and_grad(qml_qnode, params_jax)

        assert qml.math.allclose(cost_qml, cost_jax)
        assert qml.math.allclose(grad_qml, grad_jax)
