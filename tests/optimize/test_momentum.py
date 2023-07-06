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
Unit tests for the ``MomentumOptimizer``.
"""
import pytest

from pennylane import numpy as np
from pennylane.optimize import MomentumOptimizer


class TestMomentumOptimizer:
    """Test the Momentum optimizer"""

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
        and that momentum accumulation works correctly.
        """
        stepsize, gamma = 0.1, 0.5
        sgd_opt = MomentumOptimizer(stepsize, momentum=gamma)
        grad, args = np.array(grad), np.array(args, requires_grad=True)

        a1 = stepsize * grad
        expected = args - a1
        res = sgd_opt.apply_grad(grad, args)
        assert np.allclose(res, expected)

        # Simulate a new step
        grad = grad + args
        args = expected

        a2 = gamma * a1 + stepsize * grad
        expected = args - a2
        res = sgd_opt.apply_grad(grad, args)
        assert np.allclose(res, expected, atol=tol)

    @pytest.mark.parametrize("x_start", np.linspace(-10, 10, 16, endpoint=False))
    def test_momentum_optimizer_univar(self, x_start, tol):
        """Tests that momentum optimizer takes one and two steps correctly
        for univariate functions."""
        stepsize, gamma = 0.1, 0.5
        mom_opt = MomentumOptimizer(stepsize, momentum=gamma)

        univariate_funcs = [np.sin, lambda x: np.exp(x / 10.0), lambda x: x**2]
        grad_uni_fns = [
            lambda x: (np.cos(x),),
            lambda x: (np.exp(x / 10.0) / 10.0,),
            lambda x: (2 * x,),
        ]

        for gradf, f in zip(grad_uni_fns, univariate_funcs):
            mom_opt.reset()

            x_onestep = mom_opt.step(f, x_start)
            x_onestep_target = x_start - gradf(x_start)[0] * stepsize
            assert np.allclose(x_onestep, x_onestep_target, atol=tol)

            x_twosteps = mom_opt.step(f, x_onestep)
            momentum_term = gamma * gradf(x_start)[0]
            x_twosteps_target = x_onestep - (gradf(x_onestep)[0] + momentum_term) * stepsize
            assert np.allclose(x_twosteps, x_twosteps_target, atol=tol)

    def test_momentum_optimizer_multivar(self, tol):
        """Tests that momentum optimizer takes one and two steps correctly
        for multivariate functions."""
        stepsize, gamma = 0.1, 0.5
        mom_opt = MomentumOptimizer(stepsize, momentum=gamma)

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
                mom_opt.reset()

                x_vec = x_vals[jdx : jdx + 2]
                x_onestep = mom_opt.step(f, x_vec)
                x_onestep_target = x_vec - gradf(x_vec)[0] * stepsize
                assert np.allclose(x_onestep, x_onestep_target, atol=tol)

                x_twosteps = mom_opt.step(f, x_onestep)
                momentum_term = gamma * gradf(x_vec)[0]
                x_twosteps_target = x_onestep - (gradf(x_onestep)[0] + momentum_term) * stepsize
                assert np.allclose(x_twosteps, x_twosteps_target, atol=tol)
