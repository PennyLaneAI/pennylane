# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane` :class:`GradientDescentOptimizer` subclasses.
"""

import unittest
import logging as log
log.getLogger('defaults')

from defaults import pennylane as qml, BaseTest

from pennylane import numpy as np
from pennylane.optimize import (GradientDescentOptimizer,
                              MomentumOptimizer,
                              NesterovMomentumOptimizer,
                              AdagradOptimizer,
                              RMSPropOptimizer,
                              AdamOptimizer)

x_vals = np.linspace(-10, 10, 16, endpoint=False)

# Hyperparameters for optimizers
stepsize = 0.1
gamma = 0.5
delta = 0.8

@qm.qnode(qm.device('default.qubit', wires=1))
def quant_fun(variables):
    qm.RX(variables[0][1], [0])
    qm.RY(variables[1][2], [0])
    qm.RY(variables[2], [0])
    return qm.expval.PauliZ(0)


def hybrid_fun(variables):
    return quant_fun(variables) + variables[0][1]


def class_fun(variables):
    return variables[0][1] * 2. + variables[1][2] + variables[2]


class BasicTest(BaseTest):
    """Basic optimizer tests.
    """
    def setUp(self):
        self.sgd_opt = GradientDescentOptimizer(stepsize)
        self.mom_opt = MomentumOptimizer(stepsize, momentum=gamma)
        self.nesmom_opt = NesterovMomentumOptimizer(stepsize, momentum=gamma)
        self.adag_opt = AdagradOptimizer(stepsize)
        self.rms_opt = RMSPropOptimizer(stepsize, decay=gamma)
        self.adam_opt = AdamOptimizer(stepsize, beta1=gamma, beta2=delta)

        self.fnames = ['test_function_1', 'test_function_2', 'test_function_3']
        self.univariate_funcs = [np.sin,
                                 lambda x: np.exp(x / 10.),
                                 lambda x: x ** 2]
        self.grad_uni_fns = [np.cos,
                             lambda x: np.exp(x / 10.) / 10.,
                             lambda x: 2 * x]
        self.multivariate_funcs = [lambda x: np.sin(x[0]) + np.cos(x[1]),
                                   lambda x: np.exp(x[0] / 3) * np.tanh(x[1]),
                                   lambda x: np.sum(x_ ** 2 for x_ in x)]
        self.grad_multi_funcs = [lambda x: np.array([np.cos(x[0]), -np.sin(x[1])]),
                                 lambda x: np.array([np.exp(x[0] / 3) / 3 * np.tanh(x[1]),
                                                     np.exp(x[0] / 3) * (1 - np.tanh(x[1]) ** 2)]),
                                 lambda x: np.array([2 * x_ for x_ in x])]
        self.mvar_mdim_funcs = [lambda x: np.sin(x[0, 0]) + np.cos(x[1, 0]) - np.sin(x[0, 1]) + x[1, 1],
                                lambda x: np.exp(x[0, 0] / 3) * np.tanh(x[0, 1]),
                                lambda x: np.sum(x_[0] ** 2 for x_ in x)]
        self.grad_mvar_mdim_funcs = [lambda x: np.array([[np.cos(x[0, 0]), -np.cos(x[0, 1])],
                                                         [-np.sin(x[1, 0]), 1.              ]]),
                                     lambda x: np.array([[np.exp(x[0, 0] / 3) / 3 * np.tanh(x[0, 1]),
                                                          np.exp(x[0, 0] / 3) * (1 - np.tanh(x[0, 1]) ** 2)],
                                                         [0., 0.]]),
                                     lambda x: np.array([[2 * x_[0], 0.] for x_ in x])]

        self.class_fun = class_fun
        self.quant_fun = quant_fun
        self.hybrid_fun = hybrid_fun

        self.mixed_list = [(0.2, 0.3), np.array([0.4, 0.2, 0.4]), 0.1]
        self.mixed_tuple = (np.array([0.2, 0.3]), [0.4, 0.2, 0.4], 0.1)

    def test_mixed_inputs_for_hybrid_optimization(self):
        """Tests that gradient descent optimizer treats parameters of mixed types the same
        for hybrid optimization tasks."""
        self.logTestName()

        hybrid_list = self.sgd_opt.step(self.hybrid_fun, self.mixed_list)
        hybrid_tuple = self.sgd_opt.step(self.hybrid_fun, self.mixed_tuple)

        self.assertAllAlmostEqual(hybrid_tuple[0], hybrid_list[0], delta=self.tol)
        self.assertAllAlmostEqual(hybrid_tuple[1], hybrid_list[1], delta=self.tol)
        self.assertAllAlmostEqual(hybrid_tuple[2], hybrid_list[2], delta=self.tol)

    def test_mixed_inputs_for_classical_optimization(self):
        """Tests that gradient descent optimizer treats parameters of mixed types the same
        for purely classical optimization tasks."""
        self.logTestName()

        class_list = self.sgd_opt.step(self.class_fun, self.mixed_list)
        class_tuple = self.sgd_opt.step(self.class_fun, self.mixed_tuple)

        self.assertAllAlmostEqual(class_list[0], class_tuple[0], delta=self.tol)
        self.assertAllAlmostEqual(class_list[1], class_tuple[1], delta=self.tol)
        self.assertAllAlmostEqual(class_list[2], class_tuple[2], delta=self.tol)

    def test_mixed_inputs_for_quantum_optimization(self):
        """Tests that gradient descent optimizer treats parameters of mixed types the same
        for purely quantum optimization tasks."""
        self.logTestName()

        quant_list = self.sgd_opt.step(self.quant_fun, self.mixed_list)
        quant_tuple = self.sgd_opt.step(self.quant_fun, self.mixed_tuple)

        self.assertAllAlmostEqual(quant_list[0], quant_tuple[0], delta=self.tol)
        self.assertAllAlmostEqual(quant_list[1], quant_tuple[1], delta=self.tol)
        self.assertAllAlmostEqual(quant_list[2], quant_tuple[2], delta=self.tol)

    def test_gradient_descent_optimizer_univar(self):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        for uni-variate functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_uni_fns, self.univariate_funcs, self.fnames):
            with self.subTest(i=name):
                for x_start in x_vals:
                    x_new = self.sgd_opt.step(f, x_start)
                    x_correct = x_start - gradf(x_start) * stepsize
                    self.assertAlmostEqual(x_new, x_correct, delta=self.tol)

    def test_gradient_descent_optimizer_multivar(self):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        for multi-variate functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_multi_funcs, self.multivariate_funcs, self.fnames):
            with self.subTest(i=name):
                for jdx in range(len(x_vals[:-1])):
                    x_vec = x_vals[jdx:jdx+2]
                    x_new = self.sgd_opt.step(f, x_vec)
                    x_correct = x_vec - gradf(x_vec) * stepsize
                    self.assertAllAlmostEqual(x_new, x_correct, delta=self.tol)

    def test_gradient_descent_optimizer_multivar_multidim(self):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        for multi-variate functions and with higher dimensional inputs."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_mvar_mdim_funcs, self.mvar_mdim_funcs, self.fnames):
            with self.subTest(i=name):
                for jdx in range(len(x_vals[:-3])):
                    x_vec = x_vals[jdx:jdx+4]
                    x_vec_multidim = np.reshape(x_vec, (2, 2))
                    x_new = self.sgd_opt.step(f, x_vec_multidim)
                    x_correct = x_vec_multidim - gradf(x_vec_multidim) * stepsize
                    x_new_flat = x_new.flatten()
                    x_correct_flat = x_correct.flatten()
                    self.assertAllAlmostEqual(x_new_flat, x_correct_flat, delta=self.tol)

    def test_gradient_descent_optimizer_usergrad(self):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        using user-provided gradients."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_uni_fns[::-1], self.univariate_funcs, self.fnames):
            with self.subTest(i=name):
                for x_start in x_vals:
                    x_new = self.sgd_opt.step(f, x_start, grad_fn=gradf)
                    x_correct = x_start - gradf(x_start) * stepsize
                    self.assertAlmostEqual(x_new, x_correct, delta=self.tol)

    def test_momentum_optimizer_univar(self):
        """Tests that momentum optimizer takes one and two steps correctly
        for uni-variate functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_uni_fns, self.univariate_funcs, self.fnames):
            with self.subTest(i=name):
                for x_start in x_vals:
                    self.mom_opt.reset()

                    x_onestep = self.mom_opt.step(f, x_start)
                    x_onestep_target = x_start - gradf(x_start) * stepsize
                    self.assertAlmostEqual(x_onestep, x_onestep_target, delta=self.tol)

                    x_twosteps = self.mom_opt.step(f, x_onestep)
                    momentum_term = gamma * gradf(x_start)
                    x_twosteps_target = x_onestep - (gradf(x_onestep) + momentum_term) * stepsize
                    self.assertAlmostEqual(x_twosteps, x_twosteps_target, delta=self.tol)

    def test_momentum_optimizer_multivar(self):
        """Tests that momentum optimizer takes one and two steps correctly
        for multi-variate functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_multi_funcs, self.multivariate_funcs, self.fnames):
            with self.subTest(i=name):
                for jdx in range(len(x_vals[:-1])):
                    self.mom_opt.reset()

                    x_vec = x_vals[jdx:jdx + 2]
                    x_onestep = self.mom_opt.step(f, x_vec)
                    x_onestep_target = x_vec - gradf(x_vec) * stepsize
                    self.assertAllAlmostEqual(x_onestep, x_onestep_target, delta=self.tol)

                    x_twosteps = self.mom_opt.step(f, x_onestep)
                    momentum_term = gamma * gradf(x_vec)
                    x_twosteps_target = x_onestep - (gradf(x_onestep) + momentum_term) * stepsize
                    self.assertAllAlmostEqual(x_twosteps, x_twosteps_target, delta=self.tol)

    def test_nesterovmomentum_optimizer_univar(self):
        """Tests that nesterov momentum optimizer takes one and two steps correctly
        for uni-variate functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_uni_fns, self.univariate_funcs, self.fnames):
            with self.subTest(i=name):
                for x_start in x_vals:
                    self.nesmom_opt.reset()

                    x_onestep = self.nesmom_opt.step(f, x_start)
                    x_onestep_target = x_start - gradf(x_start) * stepsize
                    self.assertAlmostEqual(x_onestep, x_onestep_target, delta=self.tol)

                    x_twosteps = self.nesmom_opt.step(f, x_onestep)
                    momentum_term = gamma * gradf(x_start)
                    shifted_grad_term = gradf(x_onestep - stepsize * momentum_term)
                    x_twosteps_target = x_onestep - (shifted_grad_term + momentum_term) * stepsize
                    self.assertAlmostEqual(x_twosteps, x_twosteps_target, delta=self.tol)

    def test_nesterovmomentum_optimizer_multivar(self):
        """Tests that nesterov momentum optimizer takes one and two steps correctly
        for multi-variate functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_multi_funcs, self.multivariate_funcs, self.fnames):
            with self.subTest(i=name):
                for jdx in range(len(x_vals[:-1])):
                    self.nesmom_opt.reset()

                    x_vec = x_vals[jdx:jdx + 2]
                    x_onestep = self.nesmom_opt.step(f, x_vec)
                    x_onestep_target = x_vec - gradf(x_vec) * stepsize
                    self.assertAllAlmostEqual(x_onestep, x_onestep_target, delta=self.tol)

                    x_twosteps = self.nesmom_opt.step(f, x_onestep)
                    momentum_term = gamma * gradf(x_vec)
                    shifted_grad_term = gradf(x_onestep - stepsize * momentum_term)
                    x_twosteps_target = x_onestep - (shifted_grad_term + momentum_term) * stepsize
                    self.assertAllAlmostEqual(x_twosteps, x_twosteps_target, delta=self.tol)

    def test_nesterovmomentum_optimizer_usergrad(self):
        """Tests that nesterov momentum optimizer takes gradient-descent steps correctly
        using user-provided gradients."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_uni_fns[::-1], self.univariate_funcs, self.fnames):
            with self.subTest(i=name):
                for x_start in x_vals:
                    self.nesmom_opt.reset()

                    x_onestep = self.nesmom_opt.step(f, x_start, grad_fn=gradf)
                    x_onestep_target = x_start - gradf(x_start) * stepsize
                    self.assertAlmostEqual(x_onestep, x_onestep_target, delta=self.tol)

                    x_twosteps = self.nesmom_opt.step(f, x_onestep, grad_fn=gradf)
                    momentum_term = gamma * gradf(x_start)
                    shifted_grad_term = gradf(x_onestep - stepsize * momentum_term)
                    x_twosteps_target = x_onestep - (shifted_grad_term + momentum_term) * stepsize
                    self.assertAlmostEqual(x_twosteps, x_twosteps_target, delta=self.tol)

    def test_adagrad_optimizer_univar(self):
        """Tests that adagrad optimizer takes one and two steps correctly
        for uni-variate functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_uni_fns, self.univariate_funcs, self.fnames):
            with self.subTest(i=name):
                for x_start in x_vals:
                    self.adag_opt.reset()

                    x_onestep = self.adag_opt.step(f, x_start)
                    past_grads = gradf(x_start)*gradf(x_start)
                    adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
                    x_onestep_target = x_start - gradf(x_start) * adapt_stepsize
                    self.assertAlmostEqual(x_onestep, x_onestep_target, delta=self.tol)

                    x_twosteps = self.adag_opt.step(f, x_onestep)
                    past_grads = gradf(x_start)*gradf(x_start) + gradf(x_onestep)*gradf(x_onestep)
                    adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
                    x_twosteps_target = x_onestep - gradf(x_onestep) * adapt_stepsize
                    self.assertAlmostEqual(x_twosteps, x_twosteps_target, delta=self.tol)

    def test_adagrad_optimizer_multivar(self):
        """Tests that adagrad optimizer takes one and two steps correctly
        for multi-variate functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_multi_funcs, self.multivariate_funcs, self.fnames):
            with self.subTest(i=name):
                for jdx in range(len(x_vals[:-1])):
                    self.adag_opt.reset()

                    x_vec = x_vals[jdx:jdx + 2]
                    x_onestep = self.adag_opt.step(f, x_vec)
                    past_grads = gradf(x_vec)*gradf(x_vec)
                    adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
                    x_onestep_target = x_vec - gradf(x_vec) * adapt_stepsize
                    self.assertAllAlmostEqual(x_onestep, x_onestep_target, delta=self.tol)

                    x_twosteps = self.adag_opt.step(f, x_onestep)
                    past_grads = gradf(x_vec) * gradf(x_vec) + gradf(x_onestep) * gradf(x_onestep)
                    adapt_stepsize = stepsize / np.sqrt(past_grads + 1e-8)
                    x_twosteps_target = x_onestep - gradf(x_onestep) * adapt_stepsize
                    self.assertAllAlmostEqual(x_twosteps, x_twosteps_target, delta=self.tol)

    def test_rmsprop_optimizer_univar(self):
        """Tests that rmsprop optimizer takes one and two steps correctly
        for uni-variate functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_uni_fns, self.univariate_funcs, self.fnames):
            with self.subTest(i=name):
                for x_start in x_vals:
                    self.rms_opt.reset()

                    x_onestep = self.rms_opt.step(f, x_start)
                    past_grads = (1 - gamma) * gradf(x_start)*gradf(x_start)
                    adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
                    x_onestep_target = x_start - gradf(x_start) * adapt_stepsize
                    self.assertAlmostEqual(x_onestep, x_onestep_target, delta=self.tol)

                    x_twosteps = self.rms_opt.step(f, x_onestep)
                    past_grads = (1 - gamma) * gamma * gradf(x_start)*gradf(x_start) \
                                 + (1 - gamma) * gradf(x_onestep)*gradf(x_onestep)
                    adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
                    x_twosteps_target = x_onestep - gradf(x_onestep) * adapt_stepsize
                    self.assertAlmostEqual(x_twosteps, x_twosteps_target, delta=self.tol)

    def test_rmsprop_optimizer_multivar(self):
        """Tests that rmsprop optimizer takes one and two steps correctly
        for multi-variate functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_multi_funcs, self.multivariate_funcs, self.fnames):
            with self.subTest(i=name):
                for jdx in range(len(x_vals[:-1])):
                    self.rms_opt.reset()

                    x_vec = x_vals[jdx:jdx + 2]
                    x_onestep = self.rms_opt.step(f, x_vec)
                    past_grads = (1 - gamma) * gradf(x_vec)*gradf(x_vec)
                    adapt_stepsize = stepsize/np.sqrt(past_grads + 1e-8)
                    x_onestep_target = x_vec - gradf(x_vec) * adapt_stepsize
                    self.assertAllAlmostEqual(x_onestep, x_onestep_target, delta=self.tol)

                    x_twosteps = self.rms_opt.step(f, x_onestep)
                    past_grads = (1 - gamma) * gamma * gradf(x_vec) * gradf(x_vec) \
                                 + (1 - gamma) * gradf(x_onestep) * gradf(x_onestep)
                    adapt_stepsize = stepsize / np.sqrt(past_grads + 1e-8)
                    x_twosteps_target = x_onestep - gradf(x_onestep) * adapt_stepsize
                    self.assertAllAlmostEqual(x_twosteps, x_twosteps_target, delta=self.tol)

    def test_adam_optimizer_univar(self):
        """Tests that adam optimizer takes one and two steps correctly
        for uni-variate functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_uni_fns, self.univariate_funcs, self.fnames):
            with self.subTest(i=name):
                for x_start in x_vals:
                    self.adam_opt.reset()

                    x_onestep = self.adam_opt.step(f, x_start)
                    adapted_stepsize = stepsize * np.sqrt(1 - delta)/(1 - gamma)
                    firstmoment = gradf(x_start)
                    secondmoment = gradf(x_start) * gradf(x_start)
                    x_onestep_target = x_start - adapted_stepsize * firstmoment / (np.sqrt(secondmoment) + 1e-8)
                    self.assertAlmostEqual(x_onestep, x_onestep_target, delta=self.tol)

                    x_twosteps = self.adam_opt.step(f, x_onestep)
                    adapted_stepsize = stepsize * np.sqrt(1 - delta**2) / (1 - gamma**2)
                    firstmoment = (gamma * gradf(x_start) + (1 - gamma) * gradf(x_onestep))
                    secondmoment = (delta * gradf(x_start) * gradf(x_start) + (1 - delta) * gradf(x_onestep) * gradf(x_onestep))
                    x_twosteps_target = x_onestep - adapted_stepsize * firstmoment / (np.sqrt(secondmoment) + 1e-8)
                    self.assertAlmostEqual(x_twosteps, x_twosteps_target, delta=self.tol)

    def test_adam_optimizer_multivar(self):
        """Tests that adam optimizer takes one and two steps correctly
        for multi-variate functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_multi_funcs, self.multivariate_funcs, self.fnames):
            with self.subTest(i=name):
                for jdx in range(len(x_vals[:-1])):
                    self.adam_opt.reset()

                    x_vec = x_vals[jdx:jdx + 2]
                    x_onestep = self.adam_opt.step(f, x_vec)
                    adapted_stepsize = stepsize * np.sqrt(1 - delta) / (1 - gamma)
                    firstmoment = gradf(x_vec)
                    secondmoment = gradf(x_vec) * gradf(x_vec)
                    x_onestep_target = x_vec - adapted_stepsize * firstmoment / (np.sqrt(secondmoment) + 1e-8)
                    self.assertAllAlmostEqual(x_onestep, x_onestep_target, delta=self.tol)

                    x_twosteps = self.adam_opt.step(f, x_onestep)
                    adapted_stepsize = stepsize * np.sqrt(1 - delta**2) / (1 - gamma**2)
                    firstmoment = (gamma * gradf(x_vec) + (1 - gamma) * gradf(x_onestep))
                    secondmoment = (delta * gradf(x_vec) * gradf(x_vec) + (1 - delta) * gradf(x_onestep) * gradf(x_onestep))
                    x_twosteps_target = x_onestep - adapted_stepsize * firstmoment / (np.sqrt(secondmoment) + 1e-8)
                    self.assertAllAlmostEqual(x_twosteps, x_twosteps_target, delta=self.tol)


if __name__ == '__main__':
    print('Testing PennyLane version ' + qml.version() + ', basic optimizers.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
