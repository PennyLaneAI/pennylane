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
"""Test Jax-based Catalyst-compatible Momentum-QNG optimizer"""

import numpy as np
import pytest

import pennylane as qml

dev_names = (
    "default.qubit",
    "lightning.qubit",
)


class TestBasics:
    """Test basic properties of the MomentumQNGOptimizerQJIT."""

    def test_initialization_default(self):
        """Test that initializing MomentumQNGOptimizerQJIT with default values works."""
        opt = qml.MomentumQNGOptimizerQJIT()
        assert opt.stepsize == 0.01
        assert opt.momentum == 0.9
        assert opt.approx == "block-diag"
        assert opt.lam == 0

    def test_initialization_custom(self):
        """Test that initializing MomentumQNGOptimizerQJIT with custom values works."""
        opt = qml.MomentumQNGOptimizerQJIT(stepsize=0.05, momentum=0.8, approx="diag", lam=1e-9)
        assert opt.stepsize == 0.05
        assert opt.momentum == 0.8
        assert opt.approx == "diag"
        assert opt.lam == 1e-9

    def test_init_zero_state(self):
        """Test that the MomentumQNGOptimizerQJIT state is initialized to an array of zeros."""
        opt = qml.MomentumQNGOptimizerQJIT()
        params = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        state = opt.init(params)
        assert np.all(state == np.zeros_like(params))


class TestOptimize:
    """Test basic optimization integration."""

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", dev_names)
    @pytest.mark.parametrize("rho", [0.9, 0.0])
    def test_step_and_cost(self, dev_name, rho):
        """Test that the step and step_and_cost methods are returning
        the correct result for a few optimization steps."""
        import jax.numpy as jnp

        @qml.qnode(qml.device(dev_name))
        def circ(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        params = [0.11, 0.412]
        params_qml = qml.numpy.array(params)
        params_jax = jnp.array(params)
        stepsize = 0.01

        opt = qml.MomentumQNGOptimizerQJIT(stepsize=stepsize, momentum=rho)
        state = opt.init(params_jax)

        for _ in range(4):
            new_params1, state1 = opt.step(circ, params_jax, state)
            new_params2, state2, cost = opt.step_and_cost(circ, params_jax, state)

            expected_mt = np.array([0.25, (np.cos(params[0]) ** 2) / 4])
            state = rho * state + stepsize * qml.grad(circ)(params_qml) / expected_mt
            expected_params = params_qml - state
            expected_cost = circ(params)

            assert np.allclose(new_params1, expected_params)
            assert np.allclose(new_params2, expected_params)
            assert np.allclose(state1, state)
            assert np.allclose(state2, state)
            assert np.allclose(cost, expected_cost)
