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
# pylint: disable=assignment-from-none

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
    """Test basic properties of the QNGOptimizerQJIT."""

    def test_initialization_default(self):
        """Test that initializing QNGOptimizerQJIT with default values works."""
        opt = qml.QNGOptimizerQJIT()
        assert opt.stepsize == 0.01
        assert opt.approx == "block-diag"
        assert opt.lam == 0

    def test_initialization_custom(self):
        """Test that initializing QNGOptimizerQJIT with custom values works."""
        opt = qml.QNGOptimizerQJIT(stepsize=0.05, approx="diag", lam=1e-9)
        assert opt.stepsize == 0.05
        assert opt.approx == "diag"
        assert opt.lam == 1e-9

    def test_init_none_state(self):
        """Test that the QNGOptimizerQJIT state is initialized to `None`."""
        opt = qml.QNGOptimizerQJIT()
        assert opt.init([0.1, 0.2]) is None


class TestGradients:
    """Test gradient computation in the QNGOptimizerQJIT."""

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", dev_names)
    def test_get_grad_jax(self, dev_name):
        """Test `_get_grad` method with Jax interface."""
        # pylint:disable=protected-access
        import jax.numpy as jnp

        device = qml.device(dev_name, wires=2)
        qnode = qml.QNode(circuit, device=device)

        opt = qml.QNGOptimizerQJIT()
        params = [0.1, 0.2]

        params_jax = jnp.array(params)
        grad_jax = opt._get_grad(qnode, params_jax)

        grad_exact = jnp.array([-np.sin(params[0]), 0.0])

        assert np.allclose(grad_exact, grad_jax)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", dev_names)
    def test_get_value_and_grad_jax(self, dev_name):
        """Test `_get_value_and_grad` method with Jax interface."""
        # pylint:disable=protected-access
        import jax.numpy as jnp

        device = qml.device(dev_name, wires=2)
        qnode = qml.QNode(circuit, device=device)

        opt = qml.QNGOptimizerQJIT()
        params = [0.1, 0.2]

        params_jax = jnp.array(params)
        cost_jax, grad_jax = opt._get_value_and_grad(qnode, params_jax)

        cost_exact = qnode(params)
        grad_exact = jnp.array([-np.sin(params[0]), 0.0])

        assert np.allclose(cost_exact, cost_jax)
        assert np.allclose(grad_exact, grad_jax)


class TestMetricTensor:
    """Test that the metric tensor computation works with `approx` and `lam`."""

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", dev_names)
    def test_no_approx(self, dev_name):
        """Test that the full metric tensor is computed for `approx=None`."""
        # pylint:disable=protected-access
        import jax.numpy as jnp

        @qml.qnode(qml.device(dev_name))
        def circ(params):
            qml.RY(eta, wires=0)
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        eta = 0.7
        params = jnp.array([0.11, 0.412])

        opt = qml.QNGOptimizerQJIT(approx=None)
        mt = opt._get_metric_tensor(circ, params)

        # computing the expected metric tensor requires some manual calculation
        x = params[0]
        first_term = np.eye(2) / 4
        vec_potential = np.array([-0.5j * np.sin(eta), 0.5j * np.sin(x) * np.cos(eta)])
        second_term = np.real(np.outer(vec_potential.conj(), vec_potential))
        expected_mt = first_term - second_term

        assert np.allclose(mt, expected_mt)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", dev_names)
    def test_with_approx(self, dev_name):
        """Test that the approximated metric tensor is computed for `approx=block-diag` and `approx=diag`."""
        # pylint:disable=protected-access
        import jax.numpy as jnp

        @qml.qnode(qml.device(dev_name))
        def circ(params):
            qml.RY(eta, wires=0)
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        eta = 0.7
        params = jnp.array([0.11, 0.412])

        blockdiag_opt = qml.QNGOptimizerQJIT(approx="block-diag")
        blockdiag_mt = blockdiag_opt._get_metric_tensor(circ, params)

        diag_opt = qml.QNGOptimizerQJIT(approx="diag")
        diag_mt = diag_opt._get_metric_tensor(circ, params)

        # computing the expected metric tensor requires some manual calculation
        x = params[0]
        first_term = np.eye(2) / 4
        vec_potential = np.array([-0.5j * np.sin(eta), 0.5j * np.sin(x) * np.cos(eta)])
        second_term = np.real(np.outer(vec_potential.conj(), vec_potential))
        expected_mt = np.diag(np.diag(first_term - second_term))

        assert np.allclose(blockdiag_mt, expected_mt)
        assert np.allclose(diag_mt, expected_mt)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", dev_names)
    def test_lam(self, dev_name):
        """Test that the regularization `lam` is used correctly."""
        # pylint:disable=protected-access
        import jax.numpy as jnp

        @qml.qnode(qml.device(dev_name))
        def circ(params):
            qml.RY(eta, wires=0)
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        eta = np.pi
        params = jnp.array([eta / 2, 0.412])

        stepsize = 1.0
        lam = 1e-11

        opt_with_lam = qml.QNGOptimizerQJIT(stepsize=stepsize, approx=None, lam=lam)
        state_with_lam = opt_with_lam.init(params)
        new_params_with_lam, _ = opt_with_lam.step(circ, params, state_with_lam)
        mt_with_lam = opt_with_lam._get_metric_tensor(circ, params)

        opt = qml.QNGOptimizerQJIT(stepsize=stepsize, approx=None)
        state = opt.init(params)
        new_params, _ = opt.step(circ, params, state)
        mt = opt._get_metric_tensor(circ, params)

        # computing the expected metric tensor requires some manual calculation
        x, y = params
        first_term = np.eye(2) / 4
        vec_potential = np.array([-0.5j * np.sin(eta), 0.5j * np.sin(x) * np.cos(eta)])
        second_term = np.real(np.outer(vec_potential.conj(), vec_potential))
        expected_mt = first_term - second_term

        assert np.allclose(mt_with_lam, expected_mt + np.eye(2) * lam)
        assert np.allclose(mt, expected_mt)

        # with regularization y gets updated, without regularization it does not
        assert not np.isclose(new_params_with_lam[1], y)
        assert np.isclose(new_params[1], y)


# pylint:disable=too-few-public-methods
class TestExceptions:
    """Test exceptions are raised for incorrect usage."""

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", dev_names)
    def test_obj_func_not_a_qnode(self, dev_name):
        """Test that if the objective function is not a QNode, an error is raised."""
        import jax.numpy as jnp

        @qml.qnode(qml.device(dev_name))
        def circ(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        def cost(a):
            return circ(a)

        opt = qml.QNGOptimizerQJIT()
        params = jnp.array(0.5)
        state = opt.init(params)

        with pytest.raises(
            ValueError,
            match="The objective function must be encoded as a single QNode to use the Quantum Natural Gradient optimizer.",
        ):
            opt.step(cost, params, state)


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

        params = [0.31, 0.842]
        params_qml = qml.numpy.array(params)
        params_jax = jnp.array(params)

        opt = qml.QNGOptimizerQJIT(stepsize=0.05)
        state = opt.init(params_jax)

        new_params1, _ = opt.step(circ, params_jax, state)
        new_params2, _, cost = opt.step_and_cost(circ, params_jax, state)

        expected_mt = np.array([0.25, (np.cos(params[0]) ** 2) / 4])
        expected_params = params_qml - opt.stepsize * qml.grad(circ)(params_qml) / expected_mt
        expected_cost = circ(params)

        assert np.allclose(new_params1, expected_params)
        assert np.allclose(new_params2, expected_params)
        assert np.allclose(cost, expected_cost)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", dev_names)
    def test_step_and_cost_with_gen_hamiltonian(self, dev_name):
        """Test that the step and step_and_cost methods are returning the correct result
        when the generator of an operator is a Hamiltonian."""
        import jax.numpy as jnp

        @qml.qnode(qml.device(dev_name))
        def circ(params):
            qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        params = [0.31, 0.842]
        params_qml = qml.numpy.array(params)
        params_jax = jnp.array(params)

        opt = qml.QNGOptimizerQJIT(stepsize=0.05)
        state = opt.init(params_jax)

        new_params1, _ = opt.step(circ, params_jax, state)
        new_params2, _, cost = opt.step_and_cost(circ, params_jax, state)

        expected_cost = circ(params)
        expected_mt = np.array([1 / 16, 1 / 4])
        expected_params = params_qml - opt.stepsize * qml.grad(circ)(params_qml) / expected_mt

        assert np.allclose(cost, expected_cost)
        assert np.allclose(new_params1, expected_params)
        assert np.allclose(new_params2, expected_params)

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

        opt = qml.QNGOptimizerQJIT(stepsize=0.1)
        params = jnp.array([0.011, 0.012])
        expected_params = jnp.array([0.011, 0.012])
        state = opt.init(params)

        num_steps = 30
        for _ in range(num_steps):
            params, state, cost = opt.step_and_cost(circ, params, state)

            expected_cost = circ(expected_params)
            expected_mt = np.array([0.25, (np.cos(expected_params[0]) ** 2) / 4])
            expected_params -= opt.stepsize * grad(expected_params) / expected_mt

            assert np.allclose(cost, expected_cost, atol=tol, rtol=0)
            assert np.allclose(params, expected_params, atol=tol, rtol=0)

        assert np.allclose(circ(params), -1)

    @pytest.mark.jax
    def test_jit(self):
        """Test optimizer compatibility with jax.jit compilation."""
        import jax
        import jax.numpy as jnp

        device = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, device=device)

        params = [0.1, 0.2]
        params = jnp.array(params)

        opt = qml.QNGOptimizerQJIT()
        state = opt.init(params)

        new_params1, state1 = opt.step(qnode, params, state)
        new_params2, state2, cost = opt.step_and_cost(qnode, params, state)

        step = jax.jit(partial(opt.step, qnode))
        step_and_cost = jax.jit(partial(opt.step_and_cost, qnode))
        new_params1_jit, state1_jit = step(params, state)
        new_params2_jit, state2_jit, cost_jit = step_and_cost(params, state)

        # check params have been updated
        assert not np.allclose(params, new_params1)
        assert not np.allclose(params, new_params2)

        # check jitted results match
        assert np.allclose(new_params1, new_params1_jit)
        assert state1 is None
        assert state1_jit is None
        assert np.allclose(new_params2, new_params2_jit)
        assert state2 is None
        assert state2_jit is None
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

        params = [0.1, 0.2]
        params = jnp.array(params)

        opt = qml.QNGOptimizerQJIT()
        state = opt.init(params)

        new_params1, state1 = opt.step(qnode, params, state)
        new_params2, state2, cost = opt.step_and_cost(qnode, params, state)

        step = qml.qjit(partial(opt.step, qnode))
        step_and_cost = qml.qjit(partial(opt.step_and_cost, qnode))
        new_params1_qjit, state1_qjit = step(params, state)
        new_params2_qjit, state2_qjit, cost_qjit = step_and_cost(params, state)

        # check params have been updated
        assert not np.allclose(params, new_params1)
        assert not np.allclose(params, new_params2)

        # check qjitted results match
        assert np.allclose(new_params1, new_params1_qjit)
        assert state1 is None
        assert state1_qjit is None
        assert np.allclose(new_params2, new_params2_qjit)
        assert state2 is None
        assert state2_qjit is None
        assert np.allclose(cost, cost_qjit)
