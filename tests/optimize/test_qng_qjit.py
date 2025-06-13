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
        state = opt.init([0.1, 0.2])
        assert state is None


class TestGradients:
    """Test gradient computation in the QNGOptimizerQJIT."""

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", dev_names)
    def test_get_grad_jax(self, dev_name):
        """Test `_get_grad` method with Jax interface."""
        # pylint:disable=protected-access
        import jax.numpy as jnp

        device = qml.device(dev_name, wires=2)
        qml_qnode = qml.QNode(circuit, device=device)

        opt = qml.QNGOptimizerQJIT()
        params = [0.1, 0.2]

        params_qml = qml.numpy.array(params)
        grad_qml = qml.grad(qml_qnode)(params_qml)

        params_jax = jnp.array(params)
        grad_jax = opt._get_grad(qml_qnode, params_jax)

        assert np.allclose(grad_qml, grad_jax)

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", dev_names)
    def test_get_value_and_grad_jax(self, dev_name):
        """Test `_get_value_and_grad` method with Jax interface."""
        # pylint:disable=protected-access
        import jax.numpy as jnp

        device = qml.device(dev_name, wires=2)
        qml_qnode = qml.QNode(circuit, device=device)

        opt = qml.QNGOptimizerQJIT()
        params = [0.1, 0.2]

        params_qml = qml.numpy.array(params)
        cost_qml = qml_qnode(params_qml)
        grad_qml = qml.grad(qml_qnode)(params_qml)

        params_jax = jnp.array(params)
        cost_jax, grad_jax = opt._get_value_and_grad(qml_qnode, params_jax)

        assert np.allclose(cost_qml, cost_jax)
        assert np.allclose(grad_qml, grad_jax)


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
