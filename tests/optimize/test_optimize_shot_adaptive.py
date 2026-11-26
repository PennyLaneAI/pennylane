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
    def qnode():
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
        qnode = self.qnode()
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
        qnode = self.qnode()
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
        qnode = self.qnode()
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

    @pytest.mark.jax
    @pytest.mark.parametrize("term_sampling", [None, "weighted_random_sampling"])
    @pytest.mark.parametrize(
        "argnums, expected", [({0}, [-0.25]), ({1}, [-0.5]), ({0, 1}, [-0.25, -0.5])]
    )
    def test_native_single_shot_gradient_dispatch(self, mocker, term_sampling, argnums, expected):
        """The standalone sampling helper differentiates each explicit JAX argument."""
        import jax.numpy as jnp

        @qp.set_shots(100)
        @qp.qnode(
            qp.device("default.qubit", wires=1, seed=123),
            interface="jax",
            diff_method="parameter-shift",
        )
        def circuit(angle, offset, scale=2.0):
            qp.RX(angle + scale * offset, 0)
            return qp.expval(qp.Hamiltonian([0.25], [qp.Z(0)]))

        opt = qp.ShotAdaptiveOptimizer(min_shots=5, term_sampling=term_sampling)
        opt.max_shots = 5
        opt.trainable_args = argnums
        opt.s = [np.array(5) for _ in argnums]
        sample_fn = mocker.spy(opt, "_single_shot_qnode_gradients")

        gradients, variances = opt.compute_grad(
            circuit, [jnp.array(np.pi / 2), jnp.array(0.0)], {"scale": 2.0}
        )

        sample_fn.assert_called_once()
        assert len(sample_fn.spy_return) == len(argnums)
        for samples, value in zip(sample_fn.spy_return, expected, strict=True):
            assert samples.shape == (5,)
            assert np.allclose(samples, value)
        assert np.allclose(gradients, expected)
        assert np.allclose(variances, 0.0)
        assert opt.lipschitz == 0.25
        assert circuit.shots == qp.measurements.Shots(100)

    @pytest.mark.jax
    @pytest.mark.parametrize("term_sampling", [None, "weighted_random_sampling"])
    def test_nonconsecutive_single_shot_argument_order(self, term_sampling):
        """Native gradients follow positional input order for nonconsecutive arguments."""
        import jax.numpy as jnp

        @qp.set_shots(100)
        @qp.qnode(
            qp.device("default.qubit", wires=1, seed=123),
            interface="jax",
            diff_method="parameter-shift",
        )
        def circuit(*angles):
            qp.RX(angles[1] + 2 * angles[8], 0)
            return qp.expval(qp.Hamiltonian([0.25], [qp.Z(0)]))

        args = [jnp.array(0.0) for _ in range(9)]
        args[1] = jnp.array(np.pi / 2)
        opt = qp.ShotAdaptiveOptimizer(min_shots=5, term_sampling=term_sampling)
        opt.max_shots = 5
        opt.trainable_args = {1, 8}
        opt.s = [np.array(5), np.array(5)]

        gradients, variances = opt.compute_grad(circuit, args, {})

        assert np.allclose(gradients, [-0.25, -0.5])
        assert np.allclose(variances, 0.0)
        assert circuit.shots == qp.measurements.Shots(100)


class TestShotAllocation:
    """Test retained optimizer arithmetic with synthetic single-shot gradient samples."""

    @pytest.mark.parametrize("multiple_args", [False, True])
    def test_running_statistics_and_shot_budget(self, mocker, multiple_args):
        """Bias correction, budget clipping, and counters persist across optimization steps."""
        qnode = TestGradientStatistics.qnode()
        args = [pnp.array(1.0, requires_grad=True)]
        samples = [np.array([-1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0])]
        if multiple_args:
            args.extend([pnp.array([9.0, 8.0], requires_grad=False), pnp.array([2.0, 3.0])])
            samples.append(np.stack([samples[0], samples[0] - 0.5], axis=-1))

        opt = qp.ShotAdaptiveOptimizer(min_shots=3, stepsize=0.25, mu=0.5, b=0.25)
        opt.check_learning_rate([-1.0, 3.0])
        sample_fn = mocker.patch.object(opt, "_single_shot_qnode_gradients", return_value=samples)
        kwargs = {"offset": 0.2}

        first_args = opt.step(qnode, *args, **kwargs)

        sample_fn.assert_called_once_with(qnode, tuple(args), kwargs)
        assert opt.trainable_args == ({0, 2} if multiple_args else {0})
        assert opt.max_shots == 3
        assert opt.shots_used == opt.total_shots_used == (18 if multiple_args else 6)
        assert opt.k == 1
        assert np.allclose((opt.chi[0], opt.xi[0]), (0.5, 2.0))
        assert opt.s[0] == 7
        if multiple_args:
            assert isinstance(first_args, list)
            assert np.allclose(first_args[0], 0.75)
            assert first_args[1] is args[1]
            assert np.allclose(first_args[2], [1.75, 2.875])
            assert np.allclose(opt.chi[1], [0.5, 0.25])
            assert np.allclose(opt.xi[1], [2.0, 2.0])
            assert np.array_equal(opt.s[1], [7, 7])
        else:
            assert first_args.shape == ()
            assert np.allclose(first_args, 0.75)
            first_args = [first_args]

        second_args = opt.step(qnode, *first_args, **kwargs)

        assert sample_fn.call_count == 2
        assert opt.max_shots == 7
        assert (opt.shots_used, opt.total_shots_used) == ((42, 60) if multiple_args else (14, 20))
        assert opt.k == 2
        assert np.allclose((opt.chi[0], opt.xi[0]), (0.75, 5 / 3))
        assert opt.s[0] == 4
        assert opt.s[0].dtype == np.int64
        if multiple_args:
            assert np.allclose(second_args[0], 0.5)
            assert second_args[1] is args[1]
            assert np.allclose(second_args[2], [1.5, 2.75])
            assert np.allclose(opt.chi[1], [0.75, 0.375])
            assert np.allclose(opt.xi[1], [5 / 3, 5 / 3])
            assert np.array_equal(opt.s[1], [4, 4])
            assert opt.s[1].dtype == np.int64
        else:
            assert second_args.shape == ()
            assert np.allclose(second_args, 0.5)

    def test_step_and_cost_uses_old_parameters_and_current_budget(self, mocker):
        """Forward evaluation uses pre-update arguments and the current maximum shot budget."""

        @qp.set_shots(100)
        @qp.transforms.convert_to_numpy_parameters
        @qp.qnode(qp.device("default.qubit", wires=1, seed=123), interface="numpy")
        def circuit(angle):
            qp.RY(angle, 0)
            return qp.expval(qp.Z(0))

        opt = qp.ShotAdaptiveOptimizer(min_shots=3, stepsize=0.25, mu=0.5, b=0.25)
        opt.check_learning_rate([1.0])
        mocker.patch.object(
            opt, "_single_shot_qnode_gradients", return_value=[np.array([-1.0, 1.0, 3.0])]
        )
        shot_override = mocker.spy(qp.optimize.shot_adaptive, "set_shots")
        angle = pnp.array(0.0, requires_grad=True)

        updated, cost = opt.step_and_cost(circuit, angle)

        assert np.allclose(updated, -0.25)
        assert cost == 1.0
        shot_override.assert_called_once_with(circuit, shots=3)
        assert circuit.shots == qp.measurements.Shots(100)

    @pytest.mark.parametrize("dtype", ["int64", "float32", "float64"])
    def test_apply_grad_preserves_legacy_trainability(self, dtype):
        """Legacy flags, NumPy promotion, and fixed-argument identity are preserved."""
        opt = qp.ShotAdaptiveOptimizer(min_shots=5, stepsize=0.2)
        args = [
            pnp.array(2.0, dtype=dtype, requires_grad=True),
            pnp.array([3.0, 4.0], requires_grad=False),
        ]
        gradient = np.array(0.5)
        expected = args[0] - opt.stepsize * gradient

        updated = opt.apply_grad([gradient], args)

        assert isinstance(updated[0], pnp.tensor)
        assert updated[0].requires_grad
        assert updated[0].dtype == expected.dtype
        assert np.allclose(updated[0], expected)
        assert updated[1] is args[1]


class TestNativeOptimization:
    """Exercise JAX optimizer steps through real single-shot gradient sampling."""

    @pytest.mark.jax
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("term_sampling", [None, "weighted_random_sampling"])
    @pytest.mark.usefixtures("preserve_jax_x64")
    def test_step_and_cost_native_arguments(self, dtype, term_sampling):
        """Real sampled updates retain native types and frozen arguments."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", dtype == "float64")
        seen = []

        @qp.set_shots(100)
        @qp.qnode(
            qp.device("default.qubit", wires=1, seed=123),
            interface="jax",
            diff_method="parameter-shift",
        )
        def circuit(angle, frozen, weights):
            seen.append((angle, frozen, weights))
            qp.RX(angle + frozen + 2 * weights[0] + 3 * weights[1], 0)
            return qp.expval(qp.Hamiltonian([0.25], [qp.Z(0)]))

        angle = jnp.array(np.pi / 2, dtype=dtype)
        weights = jnp.array([0.0, 0.0], dtype=dtype)
        frozen = np.array(0.0, dtype=dtype)
        opt = qp.ShotAdaptiveOptimizer(min_shots=5, stepsize=0.1, term_sampling=term_sampling)

        updated, cost = opt.step_and_cost(circuit, angle, frozen, weights)

        assert opt.trainable_args == {0, 2}
        assert (opt.k, opt.max_shots, opt.shots_used, opt.total_shots_used) == (1, 5, 30, 30)
        assert np.allclose(opt.chi[0] / (1 - opt.mu), -0.25)
        assert np.allclose(opt.chi[1] / (1 - opt.mu), [-0.5, -0.75])
        assert np.allclose(opt.xi[0], 0.0)
        assert np.allclose(opt.xi[1], 0.0)
        assert opt.s[0] == 5
        assert np.array_equal(opt.s[1], [5, 5])
        assert updated[1] is frozen
        assert seen[-1][0] is angle
        assert seen[-1][1] is frozen
        assert seen[-1][2] is weights
        assert qp.math.get_interface(cost) == "jax"
        assert qp.math.shape(cost) == ()
        assert -0.25 <= qp.math.toarray(cost) <= 0.25
        for result, original, increment in zip(
            [updated[0], updated[2]],
            [angle, weights],
            [0.025, np.array([0.05, 0.075])],
            strict=True,
        ):
            assert qp.math.get_interface(result) == "jax"
            assert result.dtype == original.dtype
            assert qp.math.shape(result) == qp.math.shape(original)
            assert np.allclose(qp.math.toarray(result), qp.math.toarray(original) + increment)

        second = opt.step(circuit, *updated)

        assert opt.k == 2
        assert opt.total_shots_used == 60
        assert second[1] is frozen
        assert np.all(np.isfinite(opt.chi[0]))
        assert np.all(np.isfinite(opt.xi[0]))
        assert np.all(np.isfinite(opt.chi[1]))
        assert np.all(np.isfinite(opt.xi[1]))

    @pytest.mark.jax
    @pytest.mark.parametrize("min_shots", [3, 11])
    @pytest.mark.parametrize("gate, angle, gradient", [(qp.RZ, 0.4, 0.0), (qp.RX, np.pi / 2, -1.0)])
    def test_zero_variance_preserves_positive_shot_floor(self, min_shots, gate, angle, gradient):
        """Analytical zero-variance gradients retain the requested positive shot budget."""
        import jax.numpy as jnp

        @qp.set_shots(100)
        @qp.qnode(
            qp.device("default.qubit", wires=1, seed=123),
            interface="jax",
            diff_method="parameter-shift",
        )
        def circuit(angle):
            gate(angle, wires=0)
            return qp.expval(qp.Z(0))

        opt = qp.ShotAdaptiveOptimizer(min_shots=min_shots, stepsize=0.1)
        value = jnp.array(angle, dtype="float64")

        updated = opt.step(circuit, value)

        assert np.allclose(qp.math.toarray(updated), angle - 0.1 * gradient)
        assert qp.math.get_interface(updated) == "jax"
        assert qp.math.shape(updated) == ()
        assert updated.dtype == value.dtype
        assert np.allclose(opt.chi[0] / (1 - opt.mu), gradient)
        assert opt.xi[0] == 0.0
        assert opt.s[0] == min_shots
        assert opt.shots_used == 2 * min_shots

    @pytest.mark.parametrize(
        "value",
        [
            np.array(0.5),
            pnp.array(0.5, requires_grad=False),
        ],
    )
    def test_no_trainable_arguments(self, value):
        """Fixed arguments produce a useful error before shot allocation."""
        opt = qp.ShotAdaptiveOptimizer(min_shots=5)

        with pytest.raises(ValueError, match="No trainable arguments found"):
            opt.step(TestGradientStatistics.qnode(), value)
