# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the shot adaptive optimizer"""

import numpy as np
import pytest

import pennylane as qp
from pennylane import numpy as pnp


class TestExceptions:
    """Test exceptions are raised for incorrect usage"""

    def test_analytic_device_error(self):
        """Test that an exception is raised if an analytic device is used"""
        H = qp.Hamiltonian([0.3, 0.1], [qp.PauliX(0), qp.PauliZ(0)])
        dev = qp.device("default.qubit", wires=1)

        @qp.transforms.convert_to_numpy_parameters
        @qp.qnode(dev, interface="numpy")
        def expval_cost(x):
            qp.RX(x, wires=0)
            return qp.expval(H)

        opt = qp.ShotAdaptiveOptimizer(min_shots=10)

        x = pnp.array(0.5, requires_grad=True)

        with pytest.raises(ValueError, match="can only be used with qnodes that"):
            opt.step(expval_cost, x)

    def test_learning_error(self):
        """Test that an exception is raised if the learning rate is beyond the
        lipschitz bound"""
        coeffs = [0.3, 0.1]
        H = qp.Hamiltonian(coeffs, [qp.PauliX(0), qp.PauliZ(0)])
        dev = qp.device("default.qubit", wires=1)

        @qp.set_shots(100)
        @qp.transforms.convert_to_numpy_parameters
        @qp.qnode(dev, interface="numpy")
        def expval_cost(x):
            qp.RX(x, wires=0)
            return qp.expval(H)

        opt = qp.ShotAdaptiveOptimizer(min_shots=10, stepsize=100.0)

        # lipschitz constant is given by sum(|coeffs|)
        lipschitz = np.sum(np.abs(coeffs))

        assert opt.stepsize > 2 / lipschitz

        with pytest.raises(
            ValueError, match=f"The learning rate must be less than {2 / lipschitz}"
        ):
            opt.step(expval_cost, pnp.array(0.5, requires_grad=True))

    def test_compute_grad_no_qnode_error(self):
        """Test that an exception is raised if a cost_function is not encoded
        as a QNode Object for compute_grad()"""

        def cost_fn():
            return None

        opt = qp.ShotAdaptiveOptimizer(min_shots=10)
        x_init = pnp.array(0.5, requires_grad=True)

        with pytest.raises(
            ValueError, match="The objective function must be encoded as a single QNode object"
        ):
            opt.compute_grad(cost_fn, [x_init], {})

    def test_unknown_term_sampling_method(self):
        """Checks that an exception is raised if the term sampling method is unknown"""
        coeffs = [0.2, 0.1]
        dev = qp.device("default.qubit", wires=2)
        H = qp.Hamiltonian(coeffs, [qp.PauliZ(0), qp.PauliZ(0) @ qp.PauliZ(1)])

        @qp.set_shots(100)
        @qp.transforms.convert_to_numpy_parameters
        @qp.qnode(dev, interface="numpy")
        def circuit(weights):
            qp.StronglyEntanglingLayers(weights, wires=range(2))
            return qp.expval(H)

        weights = pnp.random.random(qp.templates.StronglyEntanglingLayers.shape(3, 2))

        opt = qp.ShotAdaptiveOptimizer(min_shots=10, term_sampling="uniform_random_sampling")

        with pytest.raises(ValueError, match="Unknown Hamiltonian term sampling method"):
            opt.step(circuit, weights)


class TestGradientStatistics:
    """Test native sample generation and backend-independent gradient aggregation."""

    @staticmethod
    def _qnode():
        @qp.qnode(qp.device("default.qubit"), interface="numpy", diff_method=None)
        def circuit(*params, offset=0.0):
            angle = sum(qp.math.reshape(param, (-1,))[0] for param in params) + offset
            qp.RX(angle, 0)
            return qp.expval(qp.Z(0))

        return circuit

    @pytest.mark.parametrize(
        "budget, expected_mean, expected_variance", [(5, 3.0, 2.5), (10, 5.5, 55 / 6)]
    )
    def test_scalar_shot_budget(self, mocker, budget, expected_mean, expected_variance):
        """The scalar shot budget selects the samples used for the mean and sample variance."""
        opt = qp.ShotAdaptiveOptimizer(min_shots=10)
        opt.s = [np.array(budget)]
        samples = np.arange(1.0, 11.0)
        sample_fn = mocker.patch.object(opt, "_single_shot_qnode_gradients", return_value=[samples])
        qnode = self._qnode()
        args = [np.array(0.5)]

        gradients, variances = opt.compute_grad(qnode, args, {})

        sample_fn.assert_called_once_with(qnode, args, {})
        assert len(gradients) == len(variances) == 1
        assert gradients[0].shape == variances[0].shape == ()
        assert np.allclose(gradients[0], expected_mean)
        assert np.allclose(variances[0], expected_variance)
        assert opt.s[0] == budget

    @pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3)])
    def test_per_element_shot_budgets(self, mocker, shape):
        """Matrix and padded parameter arrays preserve shape and distinct element budgets."""
        opt = qp.ShotAdaptiveOptimizer(min_shots=10)
        budget = np.full(shape, 5)
        budget.flat[0] = 8
        opt.s = [budget.copy()]
        coefficients = np.arange(1.0, 7.0).reshape(shape)
        samples = np.arange(1.0, 11.0).reshape((-1,) + (1,) * len(shape)) * coefficients
        sample_fn = mocker.patch.object(opt, "_single_shot_qnode_gradients", return_value=[samples])
        qnode = self._qnode()
        args = [np.zeros(shape)]

        gradients, variances = opt.compute_grad(qnode, args, {})

        sample_fn.assert_called_once_with(qnode, args, {})
        assert len(gradients) == len(variances) == 1
        assert gradients[0].shape == variances[0].shape == shape
        assert np.allclose(gradients[0], np.array([4.5, 6, 9, 12, 15, 18]).reshape(shape))
        assert np.allclose(variances[0], np.array([6, 10, 22.5, 40, 62.5, 90]).reshape(shape))
        assert np.array_equal(opt.s[0], budget)

    def test_multiple_argument_shot_budgets(self, mocker):
        """Each argument's scalar or vector budget selects its own gradient samples."""
        opt = qp.ShotAdaptiveOptimizer(min_shots=10)
        opt.s = [np.array(5), np.array([5, 8])]
        samples = np.arange(1.0, 11.0)
        sample_fn = mocker.patch.object(
            opt,
            "_single_shot_qnode_gradients",
            return_value=[samples, samples[:, None] * np.array([2.0, 3.0])],
        )
        qnode = self._qnode()
        args = [np.array(0.5), np.array([0.4, 0.2])]
        kwargs = {"offset": 1.0}

        gradients, variances = opt.compute_grad(qnode, args, kwargs)

        sample_fn.assert_called_once_with(qnode, args, kwargs)
        assert len(gradients) == len(variances) == 2
        assert gradients[0].shape == variances[0].shape == ()
        assert gradients[1].shape == variances[1].shape == (2,)
        assert np.allclose(gradients[0], 3.0)
        assert np.allclose(variances[0], 2.5)
        assert np.allclose(gradients[1], [6.0, 13.5])
        assert np.allclose(variances[1], [10.0, 54.0])

    @pytest.mark.jax
    @pytest.mark.parametrize("allocation, expected_calls", [([4, 0, 6], 2), ([4, 1, 5], 3)])
    @pytest.mark.parametrize("argnums", [(0,), (0, 1)])
    def test_zero_and_single_shot_term_allocations(
        self, mocker, allocation, expected_calls, argnums
    ):
        """Zero-shot terms skip differentiation and single-shot terms retain a sample axis."""
        import jax.numpy as jnp

        coeffs = [0.2, 0.1, 0.1]
        hamiltonian = qp.Hamiltonian(coeffs, [qp.Z(0), qp.X(1), qp.Z(0) @ qp.Z(1)])

        @qp.set_shots(100)
        @qp.qnode(
            qp.device("default.qubit", wires=2, seed=123),
            interface="jax",
            diff_method="parameter-shift",
        )
        def circuit(weights, angle):
            qp.StronglyEntanglingLayers(weights, wires=range(2))
            qp.RX(angle, 0)
            return qp.expval(hamiltonian)

        weights = jnp.array([[[0.2, 0.4, 0.3], [0.5, 0.7, 0.6]]])
        angle = jnp.array(1.1)
        opt = qp.ShotAdaptiveOptimizer(min_shots=10)
        derivative = mocker.spy(qp.optimize.shot_adaptive, "jacobian")
        mocker.patch(
            "scipy.stats._multivariate.multinomial_gen.rvs",
            return_value=np.array([allocation]),
        )

        gradients = opt.qnode_weighted_random_sampling(
            circuit, coeffs, hamiltonian.ops, 10, argnums, weights, angle
        )

        assert derivative.call_count == expected_calls
        assert len(gradients) == len(argnums)
        assert gradients[0].shape == (10, *weights.shape)
        if len(argnums) == 2:
            assert gradients[1].shape == (10,)
