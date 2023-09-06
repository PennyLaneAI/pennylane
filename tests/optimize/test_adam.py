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
"""
Unit tests for the ``AdamOptimizer``.
"""
import pytest

from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer


class TestAdamOptimizer:
    """Test the Adam (adaptive moment estimation) optimizer"""

    @pytest.mark.parametrize(
        "grad,args",
        [
            ([40, -4, 12, -17, 400], [0, 30, 6, -7, 800]),
            ([0.00033, 0.45e-5, 0.0], [1.3, -0.5, 8e3]),
            ([43], [0.8]),
        ],
    )
    def test_apply_grad(self, grad, args, tol):
        """
        Test that the gradient can be applied correctly to a set of parameters
        and that accumulation works correctly.
        """
        stepsize, gamma, delta, eps = 0.1, 0.5, 0.8, 1e-8
        sgd_opt = AdamOptimizer(stepsize, beta1=gamma, beta2=delta, eps=eps)
        grad, args = np.array(grad), np.array(args, requires_grad=True)

        a1 = (1 - gamma) * grad
        b1 = (1 - delta) * grad**2
        a1_corrected = a1 / (1 - gamma)
        b1_corrected = b1 / (1 - delta)
        expected = args - stepsize * a1_corrected / (np.sqrt(b1_corrected) + eps)
        res = sgd_opt.apply_grad(grad, args)
        assert np.allclose(res, expected, atol=tol)

        # Simulate a new step
        grad = grad + args
        args = expected

        a2 = gamma * a1 + (1 - gamma) * grad
        b2 = delta * b1 + (1 - delta) * grad**2
        a2_corrected = a2 / (1 - gamma**2)
        b2_corrected = b2 / (1 - delta**2)
        expected = args - stepsize * a2_corrected / (np.sqrt(b2_corrected) + eps)
        res = sgd_opt.apply_grad(grad, args)
        assert np.allclose(res, expected, atol=tol)

    @pytest.mark.parametrize("x_start", np.linspace(-10, 10, 16, endpoint=False))
    def test_adam_optimizer_univar(self, x_start, tol):
        """Tests that adam optimizer takes one and two steps correctly
        for univariate functions."""
        stepsize, gamma, delta = 0.1, 0.5, 0.8
        adam_opt = AdamOptimizer(stepsize, beta1=gamma, beta2=delta)

        univariate_funcs = [np.sin, lambda x: np.exp(x / 10.0), lambda x: x**2]
        grad_uni_fns = [
            lambda x: (np.cos(x),),
            lambda x: (np.exp(x / 10.0) / 10.0,),
            lambda x: (2 * x,),
        ]

        for gradf, f in zip(grad_uni_fns, univariate_funcs):
            adam_opt.reset()

            x_onestep = adam_opt.step(f, x_start)
            adapted_stepsize = stepsize * np.sqrt(1 - delta) / (1 - gamma)
            firstmoment = (1 - gamma) * gradf(x_start)[0]
            secondmoment = (1 - delta) * gradf(x_start)[0] * gradf(x_start)[0]
            x_onestep_target = x_start - adapted_stepsize * firstmoment / (
                np.sqrt(secondmoment) + 1e-8
            )
            assert np.allclose(x_onestep, x_onestep_target, atol=tol)

            x_twosteps = adam_opt.step(f, x_onestep)
            adapted_stepsize = stepsize * np.sqrt(1 - delta**2) / (1 - gamma**2)
            firstmoment = gamma * firstmoment + (1 - gamma) * gradf(x_onestep)[0]
            secondmoment = (
                delta * secondmoment + (1 - delta) * gradf(x_onestep)[0] * gradf(x_onestep)[0]
            )
            x_twosteps_target = x_onestep - adapted_stepsize * firstmoment / (
                np.sqrt(secondmoment) + 1e-8
            )
            assert np.allclose(x_twosteps, x_twosteps_target, atol=tol)

    def test_adam_optimizer_multivar(self, tol):
        """Tests that adam optimizer takes one and two steps correctly
        for multivariate functions."""
        stepsize, gamma, delta = 0.1, 0.5, 0.8
        adam_opt = AdamOptimizer(stepsize, beta1=gamma, beta2=delta)

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
                adam_opt.reset()

                x_vec = x_vals[jdx : jdx + 2]
                x_onestep = adam_opt.step(f, x_vec)
                adapted_stepsize = stepsize * np.sqrt(1 - delta) / (1 - gamma)
                firstmoment = (1 - gamma) * gradf(x_vec)[0]
                secondmoment = (1 - delta) * gradf(x_vec)[0] * gradf(x_vec)[0]
                x_onestep_target = x_vec - adapted_stepsize * firstmoment / (
                    np.sqrt(secondmoment) + 1e-8
                )
                assert np.allclose(x_onestep, x_onestep_target, atol=tol)

                x_twosteps = adam_opt.step(f, x_onestep)
                adapted_stepsize = stepsize * np.sqrt(1 - delta**2) / (1 - gamma**2)
                firstmoment = gamma * firstmoment + (1 - gamma) * gradf(x_onestep)[0]
                secondmoment = (
                    delta * secondmoment + (1 - delta) * gradf(x_onestep)[0] * gradf(x_onestep)[0]
                )
                x_twosteps_target = x_onestep - adapted_stepsize * firstmoment / (
                    np.sqrt(secondmoment) + 1e-8
                )
                assert np.allclose(x_twosteps, x_twosteps_target, atol=tol)

    def test_adam_optimizer_properties(self):
        """Test the adam property interfaces"""
        stepsize, gamma, delta = 0.1, 0.5, 0.8
        adam_opt = AdamOptimizer(stepsize, beta1=gamma, beta2=delta)

        # check if None is returned when accumulation is empty
        assert adam_opt.fm is None
        assert adam_opt.sm is None
        assert adam_opt.t is None

        # Do some calculations to fill accumulation
        adam_opt.step(np.sin, np.random.rand(1))

        # Check the properties return the same values, stored in accumulation
        assert adam_opt.fm == adam_opt.accumulation["fm"]
        assert adam_opt.sm == adam_opt.accumulation["sm"]
        assert adam_opt.t == adam_opt.accumulation["t"]
