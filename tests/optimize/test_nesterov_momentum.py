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
Unit tests for the ``NesterovMomentumOptimizer``.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer


class TestNesterovMomentumOptimizer:
    """Test the Nesterov Momentum optimizer"""

    def test_step_and_cost_autograd_nesterov_multiple_inputs(self):
        """Test that the correct cost is returned via the step_and_cost method for the
        Nesterov momentum optimizer"""
        stepsize, gamma = 0.1, 0.5
        nesmom_opt = NesterovMomentumOptimizer(stepsize, momentum=gamma)

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun(*variables):
            qml.RX(variables[0][1], wires=[0])
            qml.RY(variables[1][2], wires=[0])
            qml.RY(variables[2], wires=[0])
            return qml.expval(qml.PauliZ(0))

        inputs = [
            np.array((0.2, 0.3), requires_grad=True),
            np.array([0.4, 0.2, 0.4], requires_grad=True),
            np.array(0.1, requires_grad=True),
        ]

        _, res = nesmom_opt.step_and_cost(quant_fun, *inputs)
        expected = quant_fun(*inputs)

        assert np.all(res == expected)

    def test_step_and_cost_autograd_nesterov_multid_array(self):
        """Test that the correct cost is returned via the step_and_cost method for the
        Nesterov momentum optimizer"""
        stepsize, gamma = 0.1, 0.5
        nesmom_opt = NesterovMomentumOptimizer(stepsize, momentum=gamma)
        multid_array = np.array([[0.1, 0.2], [-0.1, -0.4]])

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun_mdarr(var):
            qml.RX(var[0, 1], wires=[0])
            qml.RY(var[1, 0], wires=[0])
            qml.RY(var[1, 1], wires=[0])
            return qml.expval(qml.PauliZ(0))

        _, res = nesmom_opt.step_and_cost(quant_fun_mdarr, multid_array)
        expected = quant_fun_mdarr(multid_array)

        assert np.all(res == expected)

    @pytest.mark.parametrize("x_start", np.linspace(-10, 10, 16, endpoint=False))
    def test_nesterovmomentum_optimizer_univar(self, x_start, tol):
        """Tests that nesterov momentum optimizer takes one and two steps correctly
        for univariate functions."""
        stepsize, gamma = 0.1, 0.5
        nesmom_opt = NesterovMomentumOptimizer(stepsize, momentum=gamma)

        univariate_funcs = [np.sin, lambda x: np.exp(x / 10.0), lambda x: x**2]
        grad_uni_fns = [np.cos, lambda x: np.exp(x / 10.0) / 10.0, lambda x: 2 * x]

        for gradf, f in zip(grad_uni_fns, univariate_funcs):
            nesmom_opt.reset()

            x_onestep = nesmom_opt.step(f, x_start)
            x_onestep_target = x_start - gradf(x_start) * stepsize
            assert np.allclose(x_onestep, x_onestep_target, atol=tol)

            x_twosteps = nesmom_opt.step(f, x_onestep)
            momentum_term = gamma * gradf(x_start)
            shifted_grad_term = gradf(x_onestep - stepsize * momentum_term)
            x_twosteps_target = x_onestep - (shifted_grad_term + momentum_term) * stepsize
            assert np.allclose(x_twosteps, x_twosteps_target, atol=tol)

    def test_nesterovmomentum_optimizer_multivar(self, tol):
        """Tests that nesterov momentum optimizer takes one and two steps correctly
        for multivariate functions."""
        stepsize, gamma = 0.1, 0.5
        nesmom_opt = NesterovMomentumOptimizer(stepsize, momentum=gamma)

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
                nesmom_opt.reset()

                x_vec = x_vals[jdx : jdx + 2]
                x_onestep = nesmom_opt.step(f, x_vec)
                x_onestep_target = x_vec - gradf(x_vec)[0] * stepsize
                assert np.allclose(x_onestep, x_onestep_target, atol=tol)

                x_twosteps = nesmom_opt.step(f, x_onestep)
                momentum_term = gamma * gradf(x_vec)[0]
                shifted_grad_term = gradf(x_onestep - stepsize * momentum_term)[0]
                x_twosteps_target = x_onestep - (shifted_grad_term + momentum_term) * stepsize
                assert np.allclose(x_twosteps, x_twosteps_target, atol=tol)

    @pytest.mark.parametrize("x_start", np.linspace(-10, 10, 16, endpoint=False))
    def test_nesterovmomentum_optimizer_usergrad(self, x_start, tol):
        """Tests that nesterov momentum optimizer takes gradient-descent steps correctly
        using user-provided gradients."""
        stepsize, gamma = 0.1, 0.5
        nesmom_opt = NesterovMomentumOptimizer(stepsize, momentum=gamma)

        univariate_funcs = [np.sin, lambda x: np.exp(x / 10.0), lambda x: x**2]
        grad_uni_fns = [np.cos, lambda x: np.exp(x / 10.0) / 10.0, lambda x: 2 * x]

        for gradf, f in zip(grad_uni_fns[::-1], univariate_funcs):
            nesmom_opt.reset()

            x_onestep = nesmom_opt.step(f, x_start, grad_fn=gradf)
            x_onestep_target = x_start - gradf(x_start) * stepsize
            assert np.allclose(x_onestep, x_onestep_target, atol=tol)

            x_twosteps = nesmom_opt.step(f, x_onestep, grad_fn=gradf)
            momentum_term = gamma * gradf(x_start)
            shifted_grad_term = gradf(x_onestep - stepsize * momentum_term)
            x_twosteps_target = x_onestep - (shifted_grad_term + momentum_term) * stepsize
            assert np.allclose(x_twosteps, x_twosteps_target, atol=tol)
