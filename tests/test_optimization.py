"""
Unit tests for the :mod:`openqml` :class:`Optimizer` class.
"""

import unittest
from matplotlib.pyplot import figure

import autograd
import autograd.numpy as np
from autograd.numpy.random import (randn,)

from defaults import openqml, BaseTest
from openqml.plugin import (load_plugin,)
from openqml.circuit import (QNode,)
from openqml.optimize import (Optimizer,)



class OptTest(BaseTest):
    """ABC for tests.
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
        return self.q.evaluate(par)[0]


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
        self.assertRaises(TypeError, Optimizer, f,f,0, optimizer='SGD')
        self.assertRaises(TypeError, Optimizer, f,f,[0], optimizer='SGD')

        # optimizer has to be a callable or a known name
        self.assertRaises(ValueError, Optimizer, f,f,weights, optimizer='unknown')

        # some algorithms do not use a gradient function
        self.assertRaises(ValueError, Optimizer, f,f,weights, optimizer='Nelder-Mead')
        self.assertRaises(ValueError, Optimizer, f,f,weights, optimizer='Powell')

        # only L2 regularization is supported for now
        temp = Optimizer(f,f,weights, regularizer='unknown')
        self.assertRaises(ValueError, temp.train)


    def test_opt(self):
        "Test all supported optimization algorithms on a simple optimization task."

        self.plugin = load_plugin('dummy_plugin')
        self.circuit = self.plugin.get_circuit('opt_ev')
        p = self.plugin('test node')
        q = QNode(self.circuit, p)
        self.q = q
        #x0 = randn(q.circuit.n_par -1)  # one circuit param is used to encode the data
        x0 = np.array([1.11, -0.897, -0.929, -1.54, -0.865,  0.727, 0.140])
        grad = autograd.grad(self.cost, 0)  # gradient with respect to weights

        temp = 0.30288  # expected minimal cost
        tol = 0.001

        o = Optimizer(self.cost, grad, x0, optimizer='SGD')
        res = o.train(100, error_goal=0.32, data=self.data, batch_size=4)
        self.plot_result(o.weights)
        self.assertAlmostLess(res.fun, temp, delta=0.2)  # SGD requires more iterations to converge well

        opts = ['BFGS', 'CG', 'L-BFGS-B', 'TNC', 'SLSQP']
        opts_nograd = ['Nelder-Mead', 'Powell']

        for opt in opts:
            print(80 * '-')
            o = Optimizer(self.cost, grad, x0, optimizer=opt)
            res = o.train(data=self.data)
            self.assertAlmostEqual(res.fun, temp, delta=tol)

        for opt in opts_nograd:
            print(80 * '-')
            o = Optimizer(self.cost, None, x0, optimizer=opt)
            res = o.train(data=self.data)
            self.assertAlmostEqual(res.fun, temp, delta=tol)


if __name__ == '__main__':
    print('Testing OpenQML version ' + openqml.version() + ', Optimizer class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (OptTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
