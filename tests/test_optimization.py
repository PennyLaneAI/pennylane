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
Unit tests for the :mod:`openqml` :class:`Optimizer` class.
"""

import unittest
import logging as log
log.getLogger()


import autograd
import autograd.numpy as np
from autograd.numpy.random import (randn,)

from matplotlib.pyplot import figure

from defaults import openqml as qm, BaseTest
from openqml import QNode
from openqml.optimizer import Optimizer, SCIPY_OPT_GRAD, SCIPY_OPT_NO_GRAD


def circuit(*args):
    qm.RX(args[0], 0)
    qm.CNOT([0, 1])
    qm.RX(args[1], 0)
    qm.RZ(2.7, 1)
    qm.CNOT([0, 1])
    qm.RX(-1.8, 0)
    qm.RZ(args[2], 1)
    qm.RX(args[6], 1)
    qm.RZ(args[7], 1)
    qm.CNOT([0, 1])
    qm.RX(args[3], 0)
    qm.RZ(args[4], 1)
    qm.CNOT([0, 1])
    qm.RX(args[5], 0)
    return qm.expectation.PauliZ(0)


class OptTest(BaseTest):
    """Optimizer tests.
    """
    def setUp(self):
        # arbitrary classification data
        self.data = np.array([[-0.8, -0.4],
                              [-0.5, 0.3],
                              [-0.2, -0.2],
                              [0.1, 0.1],
                              [0.4, 0.5],
                              [1.0, 0.9]], dtype=float)


    def map_data(self, data_in, weights):
        """Maps input data using a parametrized quantum circuit.

        Args:
          data_in (float): input data item
          weights (array[float]): optimization parameters
        Returns:
          float: mapped data
        """
        par = np.concatenate((np.array([0.5*np.pi * data_in]), weights))
        return self.qnode(*par)


    def cost(self, weights, data):
        """Cost (error) function to be minimized.

        Implements a quantum classifier, trying to map input data to output data.

        Args:
          weights (array[float]): optimization parameters
          data    (array[float]): data to use in the error calculation
        Returns:
          float: cost
        """
        cost = 0
        for d in data:
            temp = self.map_data(d[0], weights) -d[1]
            cost = cost +temp ** 2
        return cost


    def plot_result(self, weights):
        """Plot the classification function."""
        xx = np.linspace(-1, 1, 100)
        yy = np.array([self.map_data(x, weights) for x in xx], dtype=float)
        fig = figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(xx, yy, 'k-')
        ax.plot(self.data[:,0], self.data[:,1], 'rs')
        ax.legend(['classification', 'data'])
        ax.grid(True)
        ax.set_title('Quantum classifier')
        fig.show()


    def test_opt_errors(self):
        "Make sure faulty input raises an exception."
        weights = np.array([0])
        def f(p):
            return 0

        # initial weights must be given as an array
        self.assertRaises(TypeError, Optimizer, f,0,f, optimizer='SGD')
        self.assertRaises(TypeError, Optimizer, f,[0],f, optimizer='SGD')

        # optimizer has to be a callable or a known name
        self.assertRaises(ValueError, Optimizer, f,weights,f, optimizer='unknown')

        # some algorithms do not use a gradient function
        for opt in SCIPY_OPT_NO_GRAD:
            self.assertRaises(ValueError, Optimizer, f,weights,f, optimizer=opt)

        # only L2 regularization is supported for now
        temp = Optimizer(f,weights,f, regularizer='unknown')
        self.assertRaises(ValueError, temp.train)


    def test_opt(self):
        "Test all supported optimization algorithms on a simple optimization task."

        self.dev = qm.device('default.qubit', wires=2)
        self.qnode = QNode(circuit, self.dev)

        x0 = np.array([-0.71690972, -0.55632194,  0.74297438, -1.15401698,  0.62766983,  2.55008079, -0.27567698]) #fix "random" values to make the test pass/fail deterministically
        #grad = autograd.grad(self.cost, 0)  # gradient with respect to weights

        temp = 0.30288  # expected minimal cost
        tol = 0.001

        o = Optimizer(self.cost, x0, optimizer='SGD')
        res = o.train(100, error_goal=0.32, data=self.data, batch_size=4)
        self.plot_result(o.weights)
        self.assertAlmostLess(res.fun, temp, delta=0.2)  # SGD requires more iterations to converge well

        for opt in SCIPY_OPT_GRAD:
            print(80 * '-')
            o = Optimizer(self.cost, x0, optimizer=opt)
            res = o.train(data=self.data)
            self.assertAlmostEqual(res.fun, temp, delta=tol)

        for opt in SCIPY_OPT_NO_GRAD:
            print(80 * '-')
            o = Optimizer(self.cost, x0, optimizer=opt)
            res = o.train(200, data=self.data)
            self.assertAlmostEqual(res.fun, temp, delta=tol)


if __name__ == '__main__':
    print('Testing OpenQML version ' + qm.version() + ', Optimizer class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (OptTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
