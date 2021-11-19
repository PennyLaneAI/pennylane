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


x_vals = np.linspace(-10, 10, 16, endpoint=False)

# Hyperparameters for optimizers
stepsize = 0.1
gamma = 0.5

# functions and their gradients
fnames = ["test_function_1", "test_function_2", "test_function_3"]
univariate_funcs = [np.sin, lambda x: np.exp(x / 10.0), lambda x: x ** 2]
grad_uni_fns = [lambda x: (np.cos(x),), lambda x: (np.exp(x / 10.0) / 10.0,), lambda x: (2 * x,)]

multivariate_funcs = [
    lambda x: np.sin(x[0]) + np.cos(x[1]),
    lambda x: np.exp(x[0] / 3) * np.tanh(x[1]),
    lambda x: np.sum([x_ ** 2 for x_ in x]),
]
grad_multi_funcs = [
    lambda x: (np.array([np.cos(x[0]), -np.sin(x[1])]),),
    lambda x: (
        np.array(
            [np.exp(x[0] / 3) / 3 * np.tanh(x[1]), np.exp(x[0] / 3) * (1 - np.tanh(x[1]) ** 2)]
        ),
    ),
    lambda x: (np.array([2 * x_ for x_ in x]),),
]


class TestMomentumOptimizer:
    """Test the Momentum optimizer"""

    @pytest.mark.parametrize("x_start", x_vals)
    def test_momentum_optimizer_univar(self, x_start, tol):
        """Tests that momentum optimizer takes one and two steps correctly
        for univariate functions."""
        mom_opt = MomentumOptimizer(stepsize, momentum=gamma)

        for gradf, f, _ in zip(grad_uni_fns, univariate_funcs, fnames):
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
        mom_opt = MomentumOptimizer(stepsize, momentum=gamma)

        for gradf, f, _ in zip(grad_multi_funcs, multivariate_funcs, fnames):
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
