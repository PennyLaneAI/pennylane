# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for the ``AdagradOptimizer``.
"""
import pytest

from pennylane import numpy as np
from pennylane.optimize import AdagradOptimizer


class TestAdagradOptimizer:
    """Test the AdaGrad (adaptive gradient) optimizer"""

    @pytest.mark.parametrize(
        "grad,args",
        [
            ([40.0, -4, 12, -17, 400], [0.0, 30, 6, -7, 800]),
            ([0.00033, 0.45e-5, 0.0], [1.3, -0.5, 8e3]),
            ([43.0], [0.8]),
        ],
    )
    def test_apply_grad(self, grad, args, tol):
        """
        Test that the gradient can be applied correctly to a set of parameters
        and that accumulation works correctly.
        """
        stepsize, eps = 0.1, 1e-8
        sgd_opt = AdagradOptimizer(stepsize, eps=eps)

        grad = (np.array(grad),)
        args = (np.array(args, requires_grad=True),)

        a1 = grad[0] ** 2
        expected = args[0] - stepsize / np.sqrt(a1 + eps) * grad[0]
        res = sgd_opt.apply_grad(grad, args)
        assert np.allclose(res, expected, atol=tol)

        # Simulate a new step
        grad = (grad[0] + args[0],)
        args = (expected,)

        res = sgd_opt.apply_grad(grad, args)

        a2 = a1 + grad[0] ** 2
        expected = args - stepsize / np.sqrt(a2 + eps) * grad[0]

        assert np.allclose(res, expected, atol=tol)

    @pytest.mark.parametrize("x_start", np.linspace(-10, 10, 16, endpoint=False))
    def test_adagrad_optimizer_univar(self, x_start, tol):
        """Tests that adagrad optimizer takes one and two steps correctly
        for univariate functions."""
        stepsize = 0.1
        adag_opt = AdagradOptimizer(stepsize)

        univariate_funcs = [np.sin, lambda x: np.exp(x / 10.0), lambda x: x**2]
        grad_uni_fns = [np.cos, lambda x: np.exp(x / 10.0) / 10.0, lambda x: 2 * x]

        for gradf, f in zip(grad_uni_fns, univariate_funcs):
            adag_opt.reset()

            x_onestep = adag_opt.step(f, x_start)
            past_grads = gradf(x_start) * gradf(x_start)
            adapt_stepsize = stepsize / np.sqrt(past_grads + 1e-8)
            x_onestep_target = x_start - gradf(x_start) * adapt_stepsize
            assert np.allclose(x_onestep, x_onestep_target, atol=tol)

            x_twosteps = adag_opt.step(f, x_onestep)
            past_grads = gradf(x_start) * gradf(x_start) + gradf(x_onestep) * gradf(x_onestep)
            adapt_stepsize = stepsize / np.sqrt(past_grads + 1e-8)
            x_twosteps_target = x_onestep - gradf(x_onestep) * adapt_stepsize
            assert np.allclose(x_twosteps, x_twosteps_target, atol=tol)

    def test_adagrad_optimizer_multivar(self, tol):
        """Tests that adagrad optimizer takes one and two steps correctly
        for multivariate functions."""
        stepsize = 0.1
        adag_opt = AdagradOptimizer(stepsize)

        multivariate_funcs = [
            lambda x: np.sin(x[0]) + np.cos(x[1]),
            lambda x: np.exp(x[0] / 3) * np.tanh(x[1]),
            lambda x: np.sum([x_**2 for x_ in x]),
        ]
        grad_multi_funcs = [
            lambda x: (np.array([np.cos(x[0]), -np.sin(x[1])]),),
            lambda x: (
                np.array(
                    [
                        np.exp(x[0] / 3) / 3 * np.tanh(x[1]),
                        np.exp(x[0] / 3) * (1 - np.tanh(x[1]) ** 2),
                    ]
                ),
            ),
            lambda x: (np.array([2 * x_ for x_ in x]),),
        ]

        x_vals = np.linspace(-10, 10, 16, endpoint=False)

        for gradf, f in zip(grad_multi_funcs, multivariate_funcs):
            for jdx in range(len(x_vals[:-1])):
                adag_opt.reset()

                x_vec = x_vals[jdx : jdx + 2]
                x_onestep = adag_opt.step(f, x_vec)
                past_grads = gradf(x_vec)[0] * gradf(x_vec)[0]
                adapt_stepsize = stepsize / np.sqrt(past_grads + 1e-8)
                x_onestep_target = x_vec - gradf(x_vec)[0] * adapt_stepsize
                assert np.allclose(x_onestep, x_onestep_target, atol=tol)

                x_twosteps = adag_opt.step(f, x_onestep)
                past_grads = (
                    gradf(x_vec)[0] * gradf(x_vec)[0] + gradf(x_onestep)[0] * gradf(x_onestep)[0]
                )
                adapt_stepsize = stepsize / np.sqrt(past_grads + 1e-8)
                x_twosteps_target = x_onestep - gradf(x_onestep)[0] * adapt_stepsize
                assert np.allclose(x_twosteps, x_twosteps_target, atol=tol)
