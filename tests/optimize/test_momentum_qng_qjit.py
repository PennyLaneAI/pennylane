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

from functools import partial

import numpy as np
import pytest

import pennylane as qml

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
    def test_step_and_cost(self, dev_name):
        """Test that the step and step_and_cost methods are returning the correct result."""
        import jax.numpy as jnp

        @qml.qnode(qml.device(dev_name))
        def circ(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        opt = qml.MomentumQNGOptimizerQJIT(stepsize=0.1, momentum=0.4)
        params = jnp.array([0.11, 0.412])
        state = jnp.array([[-0.31, 0.842]])

        new_params1, state1 = opt.step(circ, params, state)
        new_params2, state2, cost = opt.step_and_cost(circ, params, state)

        exp_params = qml.numpy.array(params)
        exp_state = np.array([-0.31, 0.842])
        exp_cost = circ(exp_params)
        exp_mt = np.array([0.25, (np.cos(exp_params[0]) ** 2) / 4])
        exp_state = opt.momentum * exp_state + opt.stepsize * qml.grad(circ)(exp_params) / exp_mt
        exp_params -= exp_state

        assert np.allclose(new_params1, exp_params)
        assert np.allclose(new_params2, exp_params)
        assert np.allclose(state1, exp_state)
        assert np.allclose(state2, exp_state)
        assert np.allclose(cost, exp_cost)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", dev_names)
    def test_step_and_cost_with_gen_hamiltonian(self, dev_name):
        """Test that the step and step_and_cost methods are returning the correct result
        when the generator of an operator is a Hamiltonian."""
        import jax.numpy as jnp

        @qml.qnode(qml.device(dev_name, wires=4))
        def circ(params):
            qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        opt = qml.MomentumQNGOptimizerQJIT(stepsize=0.1, momentum=0.4)
        params = jnp.array([0.11, 0.412])
        state = jnp.array([[-0.31, 0.842]])

        new_params1, state1 = opt.step(circ, params, state)
        new_params2, state2, cost = opt.step_and_cost(circ, params, state)

        exp_params = qml.numpy.array(params)
        exp_state = np.array([-0.31, 0.842])
        exp_cost = circ(exp_params)
        exp_mt = np.array([1 / 16, 1 / 4])
        exp_state = opt.momentum * exp_state + opt.stepsize * qml.grad(circ)(exp_params) / exp_mt
        exp_params -= exp_state

        assert np.allclose(new_params1, exp_params)
        assert np.allclose(new_params2, exp_params)
        assert np.allclose(state1, exp_state)
        assert np.allclose(state2, exp_state)
        assert np.allclose(cost, exp_cost)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", dev_names)
    def test_qubit_rotations_circuit(self, tol, dev_name):
        """Test that a simple qubit rotations circuit gets optimized correctly, checking params and cost at each step."""
        import jax.numpy as jnp

        @qml.qnode(qml.device(dev_name))
        def circ(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        def grad(params):
            """Returns the gradient of the above circuit."""
            da = -np.sin(params[0]) * np.cos(params[1])
            db = -np.cos(params[0]) * np.sin(params[1])
            return np.array([da, db])

        opt = qml.MomentumQNGOptimizerQJIT(stepsize=0.2, momentum=0.3)
        params = jnp.array([0.011, 0.012])
        state = opt.init(params)

        exp_params = np.array([0.011, 0.012])
        exp_state = np.zeros_like(exp_params)

        num_steps = 30
        for _ in range(num_steps):
            params, state, cost = opt.step_and_cost(circ, params, state)

            exp_cost = circ(exp_params)
            exp_mt = np.array([0.25, (np.cos(exp_params[0]) ** 2) / 4])
            exp_state = opt.momentum * exp_state + opt.stepsize * grad(exp_params) / exp_mt
            exp_params -= exp_state

            assert np.allclose(cost, exp_cost, atol=tol, rtol=0)
            assert np.allclose(state, exp_state, atol=tol, rtol=0)
            assert np.allclose(params, exp_params, atol=tol, rtol=0)

        assert np.allclose(circ(params), -1)

    @pytest.mark.jax
    def test_jit(self):
        """Test optimizer compatibility with jax.jit compilation."""
        import jax
        import jax.numpy as jnp

        device = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, device=device)

        opt = qml.MomentumQNGOptimizerQJIT()
        params = jnp.array([0.1, 0.2])
        state = opt.init(params)

        new_params1, state1 = opt.step(qnode, params, state)
        new_params2, state2, cost = opt.step_and_cost(qnode, params, state)

        step = jax.jit(partial(opt.step, qnode))
        step_and_cost = jax.jit(partial(opt.step_and_cost, qnode))
        new_params1_jit, state1_jit = step(params, state)
        new_params2_jit, state2_jit, cost_jit = step_and_cost(params, state)

        # check params have been updated
        assert not np.allclose(params, new_params1)
        assert not np.allclose(state, state1)
        assert not np.allclose(params, new_params2)
        assert not np.allclose(state, state2)

        # check jitted results match
        assert np.allclose(new_params1, new_params1_jit)
        assert np.allclose(state1, state1_jit)
        assert np.allclose(new_params2, new_params2_jit)
        assert np.allclose(state2, state2_jit)
        assert np.allclose(cost, cost_jit)

    @pytest.mark.jax
    @pytest.mark.catalyst
    @pytest.mark.external
    def test_qjit(self):
        """Test optimizer compatibility with qml.qjit compilation."""
        import jax.numpy as jnp

        pytest.importorskip("catalyst")

        device = qml.device("lightning.qubit", wires=2)
        qnode = qml.QNode(circuit, device=device)

        opt = qml.MomentumQNGOptimizerQJIT()
        params = jnp.array([0.1, 0.2])
        state = opt.init(params)

        new_params1, state1 = opt.step(qnode, params, state)
        new_params2, state2, cost = opt.step_and_cost(qnode, params, state)

        step = qml.qjit(partial(opt.step, qnode))
        step_and_cost = qml.qjit(partial(opt.step_and_cost, qnode))
        new_params1_qjit, state1_qjit = step(params, state)
        new_params2_qjit, state2_qjit, cost_qjit = step_and_cost(params, state)

        # check params and state have been updated
        assert not np.allclose(params, new_params1)
        assert not np.allclose(state, state1)
        assert not np.allclose(params, new_params2)
        assert not np.allclose(state, state2)

        # check qjitted results match
        assert np.allclose(new_params1, new_params1_qjit)
        assert np.allclose(state1, state1_qjit)
        assert np.allclose(new_params2, new_params2_qjit)
        assert np.allclose(state2, state2_qjit)
        assert np.allclose(cost, cost_qjit)
