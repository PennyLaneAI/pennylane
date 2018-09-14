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
Unit tests for the :mod:`openqml` :class:`QNode` class.
"""

import unittest
import logging as log
log.getLogger()

from defaults import openqml as qm, BaseTest

from openqml import numpy as np
from openqml._optimize import GradientDescentOptimizer

learning_rate = 0.1
x_vals = np.linspace(-10, 10, 16, endpoint=False)


class BasicTest(BaseTest):
    """Basic optimizer tests.
    """
    def setUp(self):
        self.sgd_opt = GradientDescentOptimizer(learning_rate)

    def test_gradient_descent_optimizer(self):
        "Tests that basic stochastic gradient descent takes gradient-descent steps correctly."
        log.info('test_gradient_descent_optimizer')
        univariate_funcs = [np.sin,
                            lambda x: np.exp(x / 10.),
                            lambda x: x ** 2]
        grad_uni_fns = [np.cos,
                    lambda x: np.exp(x / 10.) / 10.,
                    lambda x: 2 * x]

        for idx, f in enumerate(univariate_funcs):
            for x_start in x_vals:
                x_new = self.sgd_opt.minimize(f, x_start)
                x_correct = x_start - grad_uni_fns[idx](x_start) * learning_rate
                self.assertAlmostEqual(x_new, x_correct, delta=self.tol)

        multivariate_funcs = [lambda x: np.sin(x[0]) + np.cos(x[1]),
                              lambda x: np.exp(x[0] / 3) * np.tanh(x[1]),
                              lambda x: np.sum(x_ ** 2 for x_ in x)]
        grad_multi_funcs = [lambda x: np.array([np.cos(x[0]), -np.sin(x[1])]),
                            lambda x: np.array([np.exp(x[0] / 3) / 3 * np.tanh(x[1]),
                                                np.exp(x[0] / 3) * (1 - np.tanh(x[1]) ** 2)]),
                            lambda x: np.array([2 * x_ for x_ in x])]

        for idx, f in enumerate(multivariate_funcs):
            for jdx in range(len(x_vals[:-1])):
                x_vec = x_vals[jdx:jdx+2]
                x_new = self.sgd_opt.minimize(f, x_vec)
                x_correct = x_vec - grad_multi_funcs[idx](x_vec) * learning_rate
                self.assertAllAlmostEqual(x_new, x_correct, delta=self.tol)



if __name__ == '__main__':
    print('Testing OpenQML version ' + qm.version() + ', basic optimizers.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
