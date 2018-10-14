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
Sanity checks for classical automatic gradient formulas (without QNodes).
"""

import unittest
import logging as log
log.getLogger('defaults')

from defaults import openqml as qm, BaseTest

from openqml import numpy as np

x_vals = np.linspace(-10, 10, 16, endpoint=False)


class BasicTest(BaseTest):
    """Basic gradient tests.
    """
    def setUp(self):
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
        self.mvar_mdim_funcs = [lambda x: np.sin(x[0, 0]) + np.cos(x[1, 0]),
                                lambda x: np.exp(x[0, 0] / 3) * np.tanh(x[1, 0]),
                                lambda x: np.sum(x_[0] ** 2 for x_ in x)]
        self.grad_mvar_mdim_funcs = [lambda x: np.array([[np.cos(x[0, 0])], [-np.sin(x[[1]])]]),
                                     lambda x: np.array([[np.exp(x[0, 0] / 3) / 3 * np.tanh(x[1, 0])],
                                                         [np.exp(x[0, 0] / 3) * (1 - np.tanh(x[1, 0]) ** 2)]]),
                                     lambda x: np.array([[2 * x_[0]] for x_ in x])]

    def test_gradient_univar(self):
        """Tests gradients of univariate unidimensional functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_uni_fns, self.univariate_funcs, self.fnames):
            with self.subTest(i=name):
                for x in x_vals:
                    g = qm.grad(f)
                    auto_grad = g(x)
                    correct_grad = gradf(x)
                    self.assertAlmostEqual(auto_grad, correct_grad, delta=self.tol)

    def test_gradient_multivar(self):
        """Tests gradients of multivariate unidimensional functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_multi_funcs, self.multivariate_funcs, self.fnames):
            with self.subTest(i=name):
                for jdx in range(len(x_vals[:-1])):
                    x_vec = x_vals[jdx:jdx+2]
                    g = qm.grad(f)
                    auto_grad = g(x_vec)
                    correct_grad = gradf(x_vec)
                    self.assertAllAlmostEqual(auto_grad, correct_grad, delta=self.tol)

    def test_gradient_multivar_multidim(self):
        """Tests gradients of multivariate multidimensional functions."""
        self.logTestName()

        for gradf, f, name in zip(self.grad_mvar_mdim_funcs, self.mvar_mdim_funcs, self.fnames):
            with self.subTest(i=name):
                for jdx in range(len(x_vals[:-1])):
                    x_vec = x_vals[jdx:jdx+2]
                    x_vec_multidim = np.expand_dims(x_vec, axis=1)
                    g = qm.grad(f)
                    auto_grad = g(x_vec_multidim)
                    correct_grad = gradf(x_vec_multidim)
                    self.assertAllAlmostEqual(auto_grad, correct_grad, delta=self.tol)


if __name__ == '__main__':
    print('Testing OpenQML version ' + qm.version() + ', classical gradients.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
