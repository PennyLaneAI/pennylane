# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane` optimizers.
"""
# pylint: disable=redefined-outer-name
import itertools as it
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.utils import _flatten
from pennylane.optimize import (GradientDescentOptimizer,
                                MomentumOptimizer,
                                NesterovMomentumOptimizer,
                                AdagradOptimizer,
                                RMSPropOptimizer,
                                AdamOptimizer,
                                RotoselectOptimizer,
                                RotosolveOptimizer)

x_vals = np.linspace(-10, 10, 16, endpoint=False)

# Hyperparameters for optimizers
stepsize = 0.1
gamma = 0.5
delta = 0.8


# function arguments in various formats
mixed_list = [(0.2, 0.3), np.array([0.4, 0.2, 0.4]), 0.1]
mixed_tuple = (np.array([0.2, 0.3]), [0.4, 0.2, 0.4], 0.1)
nested_list = [[[0.2], 0.3], [0.1, [0.4]], -0.1]
flat_list = [0.2, 0.3, 0.1, 0.4, -0.1]
multid_array = np.array([[0.1, 0.2], [-0.1, -0.4]])
multid_list = [[0.1, 0.2], [-0.1, -0.4]]


# functions and their gradients
fnames = ['test_function_1', 'test_function_2', 'test_function_3']
univariate_funcs = [np.sin,
                    lambda x: np.exp(x / 10.),
                    lambda x: x ** 2]
grad_uni_fns = [np.cos,
                lambda x: np.exp(x / 10.) / 10.,
                lambda x: 2 * x]
multivariate_funcs = [lambda x: np.sin(x[0]) + np.cos(x[1]),
                      lambda x: np.exp(x[0] / 3) * np.tanh(x[1]),
                      lambda x: np.sum([x_ ** 2 for x_ in x])]
grad_multi_funcs = [lambda x: np.array([np.cos(x[0]), -np.sin(x[1])]),
                    lambda x: np.array([np.exp(x[0] / 3) / 3 * np.tanh(x[1]),
                                        np.exp(x[0] / 3) * (1 - np.tanh(x[1]) ** 2)]),
                    lambda x: np.array([2 * x_ for x_ in x])]
mvar_mdim_funcs = [lambda x: np.sin(x[0, 0]) + np.cos(x[1, 0]) - np.sin(x[0, 1]) + x[1, 1],
                   lambda x: np.exp(x[0, 0] / 3) * np.tanh(x[0, 1]),
                   lambda x: np.sum([x_[0] ** 2 for x_ in x])]
grad_mvar_mdim_funcs = [lambda x: np.array([[np.cos(x[0, 0]), -np.cos(x[0, 1])],
                                            [-np.sin(x[1, 0]), 1.]]),
                        lambda x: np.array([[np.exp(x[0, 0] / 3) / 3 * np.tanh(x[0, 1]),
                                             np.exp(x[0, 0] / 3) * (1 - np.tanh(x[0, 1]) ** 2)],
                                            [0., 0.]]),
                        lambda x: np.array([[2 * x_[0], 0.] for x_ in x])]



@qml.qnode(qml.device('default.qubit', wires=1))
def quant_fun(variables):
    qml.RX(variables[0][1], wires=[0])
    qml.RY(variables[1][2], wires=[0])
    qml.RY(variables[2], wires=[0])
    return qml.expval(qml.PauliZ(0))


@qml.qnode(qml.device('default.qubit', wires=1))
def quant_fun_nested(var):
    qml.RX(var[0][0][0], wires=[0])
    qml.RY(var[0][1], wires=[0])
    qml.RY(var[1][0], wires=[0])
    qml.RX(var[1][1][0], wires=[0])
    return qml.expval(qml.PauliZ(0))


@qml.qnode(qml.device('default.qubit', wires=1))
def quant_fun_flat(var):
    qml.RX(var[0], wires=[0])
    qml.RY(var[1], wires=[0])
    qml.RY(var[2], wires=[0])
    qml.RX(var[3], wires=[0])
    return qml.expval(qml.PauliZ(0))


@qml.qnode(qml.device('default.qubit', wires=1))
def quant_fun_mdarr(var):
    qml.RX(var[0, 1], wires=[0])
    qml.RY(var[1, 0], wires=[0])
    qml.RY(var[1, 1], wires=[0])
    return qml.expval(qml.PauliZ(0))


@qml.qnode(qml.device('default.qubit', wires=1))
def quant_fun_mdlist(var):
    qml.RX(var[0][1], wires=[0])
    qml.RY(var[1][0], wires=[0])
    qml.RY(var[1][1], wires=[0])
    return qml.expval(qml.PauliZ(0))


@pytest.fixture(scope="function")
def bunch():
    class A:
        sgd_opt = GradientDescentOptimizer(stepsize)
        mom_opt = MomentumOptimizer(stepsize, momentum=gamma)
        nesmom_opt = NesterovMomentumOptimizer(stepsize, momentum=gamma)
        adag_opt = AdagradOptimizer(stepsize)
        rms_opt = RMSPropOptimizer(stepsize, decay=gamma)
        adam_opt = AdamOptimizer(stepsize, beta1=gamma, beta2=delta)
        rotosolve_opt = RotosolveOptimizer()
        rotoselect_opt = RotoselectOptimizer()

    return A()


class TestOptimizer:
    """Basic optimizer tests.
    """
    def test_mixed_inputs_for_hybrid_optimization(self, bunch, tol):
        """Tests that gradient descent optimizer treats parameters of mixed types the same
        for hybrid optimization tasks."""

        def hybrid_fun(variables):
            return quant_fun(variables) + variables[0][1]

        hybrid_list = bunch.sgd_opt.step(hybrid_fun, mixed_list)
        hybrid_tuple = bunch.sgd_opt.step(hybrid_fun, mixed_tuple)

        assert hybrid_list[0] == pytest.approx(hybrid_tuple[0], abs=tol)
        assert hybrid_list[1] == pytest.approx(hybrid_tuple[1], abs=tol)
        assert hybrid_list[2] == pytest.approx(hybrid_tuple[2], abs=tol)

    def test_mixed_inputs_for_classical_optimization(self, bunch, tol):
        """Tests that gradient descent optimizer treats parameters of mixed types the same
        for purely classical optimization tasks."""

        def class_fun(var):
            return var[0][1] * 2. + var[1][2] + var[2]

        class_list = bunch.sgd_opt.step(class_fun, mixed_list)
        class_tuple = bunch.sgd_opt.step(class_fun, mixed_tuple)

        assert class_list[0] == pytest.approx(class_tuple[0], abs=tol)
        assert class_list[1] == pytest.approx(class_tuple[1], abs=tol)
        assert class_list[2] == pytest.approx(class_tuple[2], abs=tol)

    def test_mixed_inputs_for_quantum_optimization(self, bunch, tol):
        """Tests that gradient descent optimizer treats parameters of mixed types the same
        for purely quantum optimization tasks."""

        quant_list = bunch.sgd_opt.step(quant_fun, mixed_list)
        quant_tuple = bunch.sgd_opt.step(quant_fun, mixed_tuple)

        assert quant_list[0] == pytest.approx(quant_tuple[0], abs=tol)
        assert quant_list[1] == pytest.approx(quant_tuple[1], abs=tol)
        assert quant_list[2] == pytest.approx(quant_tuple[2], abs=tol)

    def test_nested_and_flat_returns_same_update(self, bunch, tol):
        """Tests that gradient descent optimizer has the same output for
         nested and flat lists."""

        def hybrid_fun_flat(var):
            return quant_fun_flat(var) + var[4]

        def hybrid_fun_nested(var):
            return quant_fun_nested(var) + var[2]

        nested = bunch.sgd_opt.step(hybrid_fun_nested, nested_list)
        flat = bunch.sgd_opt.step(hybrid_fun_flat, flat_list)

        assert flat == pytest.approx(list(_flatten(nested)), abs=tol)

    def test_array_and_list_return_same_update(self, bunch, tol):
        """Tests that gradient descent optimizer has the same output for
         lists and arrays."""

        def hybrid_fun_mdarr(var):
            return quant_fun_mdarr(var) + var[0, 0]

        def hybrid_fun_mdlist(var):
            return quant_fun_mdlist(var) + var[0][0]

        array = bunch.sgd_opt.step(hybrid_fun_mdarr, multid_array)
        list = bunch.sgd_opt.step(hybrid_fun_mdlist, multid_list)

        assert array == pytest.approx(np.asarray(list), abs=tol)

    @pytest.mark.parametrize('x_start', x_vals)
    def test_gradient_descent_optimizer_univar(self, x_start, bunch, tol):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        for uni-variate functions."""

        # TODO parametrize this for also
        for gradf, f, name in zip(grad_uni_fns, univariate_funcs, fnames):
            x_new = bunch.sgd_opt.step(f, x_start)
            x_correct = x_start - gradf(x_start) * stepsize
            assert x_new == pytest.approx(x_correct, abs=tol)

    def test_gradient_descent_optimizer_multivar(self, bunch, tol):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        for multi-variate functions."""

        for gradf, f, name in zip(grad_multi_funcs, multivariate_funcs, fnames):
            for jdx in range(len(x_vals[:-1])):
                x_vec = x_vals[jdx:jdx+2]
                x_new = bunch.sgd_opt.step(f, x_vec)
                x_correct = x_vec - gradf(x_vec) * stepsize
                assert x_new == pytest.approx(x_correct, abs=tol)

    def test_gradient_descent_optimizer_multivar_multidim(self, bunch, tol):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        for multi-variate functions and with higher dimensional inputs."""

        for gradf, f, name in zip(grad_mvar_mdim_funcs, mvar_mdim_funcs, fnames):
            for jdx in range(len(x_vals[:-3])):
                x_vec = x_vals[jdx:jdx+4]
                x_vec_multidim = np.reshape(x_vec, (2, 2))
                x_new = bunch.sgd_opt.step(f, x_vec_multidim)
                x_correct = x_vec_multidim - gradf(x_vec_multidim) * stepsize
                x_new_flat = x_new.flatten()
                x_correct_flat = x_correct.flatten()
                assert x_new_flat == pytest.approx(x_correct_flat, abs=tol)

    @pytest.mark.parametrize('x_start', x_vals)
    def test_gradient_descent_optimizer_usergrad(self, x_start, bunch, tol):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        using user-provided gradients."""

        for gradf, f, name in zip(grad_uni_fns[::-1], univariate_funcs, fnames):
            x_new = bunch.sgd_opt.step(f, x_start, grad_fn=gradf)
            x_correct = x_start - gradf(x_start) * stepsize
            assert x_new == pytest.approx(x_correct, abs=tol)

    @pytest.mark.parametrize('x_start', x_vals)
    def test_momentum_optimizer_univar(self, x_start, bunch, tol):
        """Tests that momentum optimizer takes one and two steps correctly
        for uni-variate functions."""

        for gradf, f, name in zip(grad_uni_fns, univariate_funcs, fnames):
            bunch.mom_opt.reset()

            x_onestep = bunch.mom_opt.step(f, x_start)
            x_onestep_target = x_start - gradf(x_start) * stepsize
            assert x_onestep == pytest.approx(x_onestep_target, abs=tol)

            x_twosteps = bunch.mom_opt.step(f, x_onestep)
            momentum_term = gamma * gradf(x_start)
            x_twosteps_target = x_onestep - (gradf(x_onestep) + momentum_term) * stepsize
            assert x_twosteps == pytest.approx(x_twosteps_target, abs=tol)

    def test_momentum_optimizer_multivar(self, bunch, tol):
        """Tests that momentum optimizer takes one and two steps correctly
        for multi-variate functions."""

        for gradf, f, name in zip(grad_multi_funcs, multivariate_funcs, fnames):
            for jdx in range(len(x_vals[:-1])):
                bunch.mom_opt.reset()

                x_vec = x_vals[jdx:jdx + 2]
                x_onestep = bunch.mom_opt.step(f, x_vec)
                x_onestep_target = x_vec - gradf(x_vec) * stepsize
                assert x_onestep == pytest.approx(x_onestep_target, abs=tol)

                x_twosteps = bunch.mom_opt.step(f, x_onestep)
                momentum_term = gamma * gradf(x_vec)
                x_twosteps_target = x_onestep - (gradf(x_onestep) + momentum_term) * stepsize
                assert x_twosteps == pytest.approx(x_twosteps_target, abs=tol)

    @pytest.mark.parametrize('x_start', x_vals)
    def test_nesterovmomentum_optimizer_univar(self, x_start, bunch, tol):
        """Tests that nesterov momentum optimizer takes one and two steps correctly
        for uni-variate functions."""

        for gradf, f, name in zip(grad_uni_fns, univariate_funcs, fnames):
            bunch.nesmom_opt.reset()

            x_onestep = bunch.nesmom_opt.step(f, x_start)
            x_onestep_target = x_start - gradf(x_start) * stepsize
            assert x_onestep == pytest.approx(x_onestep_target, abs=tol)

            x_twosteps = bunch.nesmom_opt.step(f, x_onestep)
            momentum_term = gamma * gradf(x_start)
            shifted_grad_term = gradf(x_onestep - stepsize * momentum_term)
            x_twosteps_target = x_onestep - (shifted_grad_term + momentum_term) * stepsize
            assert x_twosteps == pytest.approx(x_twosteps_target, abs=tol)

    def test_nesterovmomentum_optimizer_multivar(self, bunch, tol):
        """Tests that nesterov momentum optimizer takes one and two steps correctly
        for multi-variate functions."""

        for gradf, f, name in zip(grad_multi_funcs, multivariate_funcs, fnames):
            for jdx in range(len(x_vals[:-1])):
                bunch.nesmom_opt.reset()

                x_vec = x_vals[jdx:jdx + 2]
                x_onestep = bunch.nesmom_opt.step(f, x_vec)
                x_onestep_target = x_vec - gradf(x_vec) * stepsize
                assert x_onestep == pytest.approx(x_onestep_target, abs=tol)

                x_twosteps = bunch.nesmom_opt.step(f, x_onestep)
                momentum_term = gamma * gradf(x_vec)
                shifted_grad_term = gradf(x_onestep - stepsize * momentum_term)
                x_twosteps_target = x_onestep - (shifted_grad_term + momentum_term) * stepsize
                assert x_twosteps == pytest.approx(x_twosteps_target, abs=tol)

    @pytest.mark.parametrize('x_start', x_vals)
    def test_nesterovmomentum_optimizer_usergrad(self, x_start, bunch, tol):
        """Tests that nesterov momentum optimizer takes gradient-descent steps correctly
        using user-provided gradients."""

        for gradf, f, name in zip(grad_uni_fns[::-1], univariate_funcs, fnames):
            bunch.nesmom_opt.reset()

            x_onestep = bunch.nesmom_opt.step(f, x_start, grad_fn=gradf)
            x_onestep_target = x_start - gradf(x_start) * stepsize
            assert x_onestep == pytest.approx(x_onestep_target, abs=tol)

            x_twosteps = bunch.nesmom_opt.step(f, x_onestep, grad_fn=gradf)
            momentum_term = gamma * gradf(x_start)
            shifted_grad_term = gradf(x_onestep - stepsize * momentum_term)
            x_twosteps_target = x_onestep - (shifted_grad_term + momentum_term) * stepsize
            assert x_twosteps == pytest.approx(x_twosteps_target, abs=tol)

    @pytest.mark.parametrize('x_start', x_vals)
    def test_adagrad_optimizer_univar(self, x_start, bunch, tol):
        """Tests that adagrad optimizer takes one and two steps correctly
        for uni-variate functions."""

        for gradf, f, name in zip(grad_uni_fns, univariate_funcs, fnames):
            bunch.adag_opt.reset()

            x_onestep = bunch.adag_opt.step(f, x_start)
            past_grads = gradf(x_start)*gradf(x_start)
            adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
            x_onestep_target = x_start - gradf(x_start) * adapt_stepsize
            assert x_onestep == pytest.approx(x_onestep_target, abs=tol)

            x_twosteps = bunch.adag_opt.step(f, x_onestep)
            past_grads = gradf(x_start)*gradf(x_start) + gradf(x_onestep)*gradf(x_onestep)
            adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
            x_twosteps_target = x_onestep - gradf(x_onestep) * adapt_stepsize
            assert x_twosteps == pytest.approx(x_twosteps_target, abs=tol)

    def test_adagrad_optimizer_multivar(self, bunch, tol):
        """Tests that adagrad optimizer takes one and two steps correctly
        for multi-variate functions."""

        for gradf, f, name in zip(grad_multi_funcs, multivariate_funcs, fnames):
            for jdx in range(len(x_vals[:-1])):
                bunch.adag_opt.reset()

                x_vec = x_vals[jdx:jdx + 2]
                x_onestep = bunch.adag_opt.step(f, x_vec)
                past_grads = gradf(x_vec)*gradf(x_vec)
                adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
                x_onestep_target = x_vec - gradf(x_vec) * adapt_stepsize
                assert x_onestep == pytest.approx(x_onestep_target, abs=tol)

                x_twosteps = bunch.adag_opt.step(f, x_onestep)
                past_grads = gradf(x_vec) * gradf(x_vec) + gradf(x_onestep) * gradf(x_onestep)
                adapt_stepsize = stepsize / np.sqrt(past_grads + 1e-8)
                x_twosteps_target = x_onestep - gradf(x_onestep) * adapt_stepsize
                assert x_twosteps == pytest.approx(x_twosteps_target, abs=tol)

    @pytest.mark.parametrize('x_start', x_vals)
    def test_rmsprop_optimizer_univar(self, x_start, bunch, tol):
        """Tests that rmsprop optimizer takes one and two steps correctly
        for uni-variate functions."""

        for gradf, f, name in zip(grad_uni_fns, univariate_funcs, fnames):
            bunch.rms_opt.reset()

            x_onestep = bunch.rms_opt.step(f, x_start)
            past_grads = (1 - gamma) * gradf(x_start)*gradf(x_start)
            adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
            x_onestep_target = x_start - gradf(x_start) * adapt_stepsize
            assert x_onestep == pytest.approx(x_onestep_target, abs=tol)

            x_twosteps = bunch.rms_opt.step(f, x_onestep)
            past_grads = (1 - gamma) * gamma * gradf(x_start)*gradf(x_start) \
                         + (1 - gamma) * gradf(x_onestep)*gradf(x_onestep)
            adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
            x_twosteps_target = x_onestep - gradf(x_onestep) * adapt_stepsize
            assert x_twosteps == pytest.approx(x_twosteps_target, abs=tol)

    def test_rmsprop_optimizer_multivar(self, bunch, tol):
        """Tests that rmsprop optimizer takes one and two steps correctly
        for multi-variate functions."""

        for gradf, f, name in zip(grad_multi_funcs, multivariate_funcs, fnames):
            for jdx in range(len(x_vals[:-1])):
                bunch.rms_opt.reset()

                x_vec = x_vals[jdx:jdx + 2]
                x_onestep = bunch.rms_opt.step(f, x_vec)
                past_grads = (1 - gamma) * gradf(x_vec)*gradf(x_vec)
                adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
                x_onestep_target = x_vec - gradf(x_vec) * adapt_stepsize
                assert x_onestep == pytest.approx(x_onestep_target, abs=tol)

                x_twosteps = bunch.rms_opt.step(f, x_onestep)
                past_grads = (1 - gamma) * gamma * gradf(x_vec) * gradf(x_vec) \
                             + (1 - gamma) * gradf(x_onestep) * gradf(x_onestep)
                adapt_stepsize = stepsize / np.sqrt(past_grads + 1e-8)
                x_twosteps_target = x_onestep - gradf(x_onestep) * adapt_stepsize
                assert x_twosteps == pytest.approx(x_twosteps_target, abs=tol)

    @pytest.mark.parametrize('x_start', x_vals)
    def test_adam_optimizer_univar(self, x_start, bunch, tol):
        """Tests that adam optimizer takes one and two steps correctly
        for uni-variate functions."""

        for gradf, f, name in zip(grad_uni_fns, univariate_funcs, fnames):
            bunch.adam_opt.reset()

            x_onestep = bunch.adam_opt.step(f, x_start)
            adapted_stepsize = stepsize * np.sqrt(1 - delta)/(1 - gamma)
            firstmoment = gradf(x_start)
            secondmoment = gradf(x_start) * gradf(x_start)
            x_onestep_target = x_start - adapted_stepsize * firstmoment / (np.sqrt(secondmoment) + 1e-8)
            assert x_onestep == pytest.approx(x_onestep_target, abs=tol)

            x_twosteps = bunch.adam_opt.step(f, x_onestep)
            adapted_stepsize = stepsize * np.sqrt(1 - delta**2) / (1 - gamma**2)
            firstmoment = (gamma * gradf(x_start) + (1 - gamma) * gradf(x_onestep))
            secondmoment = (delta * gradf(x_start) * gradf(x_start) + (1 - delta) * gradf(x_onestep) * gradf(x_onestep))
            x_twosteps_target = x_onestep - adapted_stepsize * firstmoment / (np.sqrt(secondmoment) + 1e-8)
            assert x_twosteps == pytest.approx(x_twosteps_target, abs=tol)

    def test_adam_optimizer_multivar(self, bunch, tol):
        """Tests that adam optimizer takes one and two steps correctly
        for multi-variate functions."""

        for gradf, f, name in zip(grad_multi_funcs, multivariate_funcs, fnames):
            for jdx in range(len(x_vals[:-1])):
                bunch.adam_opt.reset()

                x_vec = x_vals[jdx:jdx + 2]
                x_onestep = bunch.adam_opt.step(f, x_vec)
                adapted_stepsize = stepsize * np.sqrt(1 - delta) / (1 - gamma)
                firstmoment = gradf(x_vec)
                secondmoment = gradf(x_vec) * gradf(x_vec)
                x_onestep_target = x_vec - adapted_stepsize * firstmoment / (np.sqrt(secondmoment) + 1e-8)
                assert x_onestep == pytest.approx(x_onestep_target, abs=tol)

                x_twosteps = bunch.adam_opt.step(f, x_onestep)
                adapted_stepsize = stepsize * np.sqrt(1 - delta**2) / (1 - gamma**2)
                firstmoment = (gamma * gradf(x_vec) + (1 - gamma) * gradf(x_onestep))
                secondmoment = (delta * gradf(x_vec) * gradf(x_vec) + (1 - delta) * gradf(x_onestep) * gradf(x_onestep))
                x_twosteps_target = x_onestep - adapted_stepsize * firstmoment / (np.sqrt(secondmoment) + 1e-8)
                assert x_twosteps == pytest.approx(x_twosteps_target, abs=tol)

    @staticmethod
    def rotosolve_step(f, x):
        """Helper function to test the Rotosolve and Rotoselect optimizers"""
        # make sure that x is an array
        if np.ndim(x) == 0:
            x = np.array([x])

        # helper function for x[d] = theta
        def insert(xf, d, theta):
            xf[d] = theta
            return xf

        for d, _ in enumerate(x):
            H_0 = float(f(insert(x, d, 0)))
            H_p = float(f(insert(x, d, np.pi / 2)))
            H_m = float(f(insert(x, d, -np.pi / 2)))

            a = np.arctan2(2 * H_0 - H_p - H_m, H_p - H_m)

            x[d] = -np.pi / 2 - a

            if x[d] <= -np.pi:
                x[d] += 2 * np.pi
        return x

    @pytest.mark.parametrize('x_start', x_vals)
    def test_rotosolve_optimizer_univar(self, x_start, bunch, tol):
        """Tests that rotosolve optimizer takes one and two steps correctly
        for uni-variate functions."""

        for f in univariate_funcs:
            x_onestep = bunch.rotosolve_opt.step(f, x_start)
            x_onestep_target = self.rotosolve_step(f, x_start)

            assert x_onestep == pytest.approx(x_onestep_target, abs=tol)

            x_twosteps = bunch.rotosolve_opt.step(f, x_onestep)
            x_twosteps_target = self.rotosolve_step(f, x_onestep_target)

            assert x_twosteps == pytest.approx(x_twosteps_target, abs=tol)

    @pytest.mark.parametrize('x_start', [[1.2, 0.2],
                                         [-0.62, -2.1],
                                         [0.05, 0.8],
                                         [[0.3], [0.25]],
                                         [[-0.6], [0.45]],
                                         [[1.3], [-0.9]]])
    def test_rotosolve_optimizer_multivar(self, x_start, bunch, tol):
        """Tests that rotosolve optimizer takes one and two steps correctly
        for multi-variate functions."""

        for func in multivariate_funcs:
            # alter multivariate_func to accept nested lists of parameters
            f = lambda x: func(np.ravel(x))

            x_onestep = bunch.rotosolve_opt.step(f, x_start)
            x_onestep_target = self.rotosolve_step(f, x_start)

            assert x_onestep == pytest.approx(x_onestep_target, abs=tol)

            x_twosteps = bunch.rotosolve_opt.step(f, x_onestep)
            x_twosteps_target = self.rotosolve_step(f, x_onestep_target)

            assert x_twosteps == pytest.approx(x_twosteps_target, abs=tol)

    @pytest.mark.parametrize('x_start', [[1.2, 0.2],
                                         [-0.62, -2.1],
                                         [0.05, 0.8]])
    @pytest.mark.parametrize('generators', [list(tup) for tup in it.product([qml.RX, qml.RY, qml.RZ], repeat=2)])
    def test_rotoselect_optimizer(self, x_start, generators, bunch, tol):
        """Tests that rotoselect optimizer finds the optimal generators and parameters for the VQE circuit
        defined in `this rotoselect tutorial <https://pennylane.ai/qml/demos/tutorial_rotoselect.html>`_."""

        # the optimal generators for the 2-qubit VQE circuit
        # H = 0.5 * Y_2 + 0.8 * Z_1 - 0.2 * X_1
        optimal_generators = [qml.RY, qml.RX]
        possible_generators = [qml.RX, qml.RY, qml.RZ]
        bunch.rotoselect_opt.possible_generators = possible_generators

        dev = qml.device("default.qubit", analytic=True, wires=2)

        def ansatz(params, generators):
            generators[0](params[0], wires=0)
            generators[1](params[1], wires=1)
            qml.CNOT(wires=[0, 1])

        @qml.qnode(dev)
        def circuit_1(params, generators=None):  # generators will be passed as a keyword arg
            ansatz(params, generators)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        @qml.qnode(dev)
        def circuit_2(params, generators=None):  # generators will be passed as a keyword arg
            ansatz(params, generators)
            return qml.expval(qml.PauliX(0))

        def cost_fn(params, generators):
            Z_1, Y_2 = circuit_1(params, generators=generators)
            X_1 = circuit_2(params, generators=generators)
            return 0.5 * Y_2 + 0.8 * Z_1 - 0.2 * X_1

        f_best_gen = lambda x: cost_fn(x, optimal_generators)
        optimal_x_start = x_start.copy()

        # after four steps the optimzer should find the optimal generators/x_start values
        for _ in range(4):
            x_start, generators = bunch.rotoselect_opt.step(cost_fn, x_start, generators)
            optimal_x_start = self.rotosolve_step(f_best_gen, optimal_x_start)

        assert x_start == pytest.approx(optimal_x_start, abs=tol)
        assert generators == optimal_generators


    def test_update_stepsize(self):
        """Tests that the stepsize correctly updates"""

        eta = 0.5
        opt = AdamOptimizer(eta)
        assert opt._stepsize == eta

        eta2 = 0.1
        opt.update_stepsize(eta2)
        assert opt._stepsize == eta2
