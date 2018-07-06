"""
Unit tests for the :mod:`openqml` :class:`Optimizer` class.
"""

import unittest

import autograd
import autograd.numpy as np
from autograd.numpy.random import (randn,)

from defaults import openqml, BaseTest
from openqml.plugin import (load_plugin,)
from openqml.circuit import (QNode,)
from openqml.core import (Optimizer,)



class OptTest(BaseTest):
    """ABC for tests.
    """
    def setUp(self):
        # arbitrary classification data
        self.data = np.array([[1.0, 0.2],
                              [0.2, 0.1],
                              [-0.6, 0.2],
                              [1.7, 0.6],
                              [2.0, 0.8]], dtype=float)


    def cost(self, weights, data_sample=None):
        """Cost (error) function to be minimized.

        Implements a quantum classifier, trying to map input data to output data.

        Args:
          weights (array[float]): optimization parameters
          data_sample (array[int], None): For stochastic gradient methods, indices of the data samples to use in the error calculation.
            If None, all data samples are used.
        """
        cost = 0
        if data_sample is None:
            data = self.data
        else:
            data = self.data[data_sample]

        for d in data:
            par = np.concatenate((d[0:1], weights))
            temp = self.q.evaluate(par)[0] -d[1]
            cost = cost +temp ** 2
        return cost


    def cost_grad(self, weights, data_sample=None):
        """Gradient of cost (error) function.
        """
        grad = np.zeros(weights.shape)
        if data_sample is None:
            data = self.data
        else:
            data = self.data[data_sample]

        for d in data:
            par = np.r_[d[0], weights]
            temp = self.q.evaluate(par)[0] -d[1]
            grad += 2*temp * self.q.gradient_angle(par, [1])
        return grad


    def test_opt(self):
        "Test all supported optimization algorithms on a simple optimization task."

        self.plugin = load_plugin('dummy_plugin')
        self.circuit = self.plugin.get_circuit('opt_ev')
        p = self.plugin('test node')
        q = QNode(self.circuit, p)
        self.q = q
        x0 = randn(q.circuit.n_par -1)
        grad = autograd.grad(self.cost, 0)  # gradient with respect to weights

        temp = 0.404258  # expected minimal cost
        tol = 0.0001

        #o = Optimizer(self.cost, self.cost_grad, x0, n_data=self.data.shape[0], optimizer='SGD')
        o = Optimizer(self.cost, grad, x0, n_data=self.data.shape[0], optimizer='SGD')
        o.set_hp(batch_size=3)
        c = o.train(100)
        #self.assertAlmostEqual(c, temp, delta=tol)

        o = Optimizer(self.cost, None, x0, optimizer='Nelder-Mead')
        c = o.train()
        self.assertAlmostEqual(c, temp, delta=tol)

        o = Optimizer(self.cost, None, x0, optimizer='Powell')
        c = o.train()
        self.assertAlmostEqual(c, temp, delta=tol)

        o = Optimizer(self.cost, grad, x0, optimizer='CG')
        c = o.train()
        self.assertAlmostEqual(c, temp, delta=tol)

        o = Optimizer(self.cost, grad, x0, optimizer='BFGS')
        c = o.train()
        self.assertAlmostEqual(c, temp, delta=tol)

        o = Optimizer(self.cost, grad, x0, optimizer='L-BFGS-B')
        c = o.train()
        self.assertAlmostEqual(c, temp, delta=tol)

        o = Optimizer(self.cost, grad, x0, optimizer='TNC')
        c = o.train()
        self.assertAlmostEqual(c, temp, delta=tol)

        o = Optimizer(self.cost, grad, x0, optimizer='SLSQP')
        c = o.train()
        self.assertAlmostEqual(c, temp, delta=tol)


if __name__ == '__main__':
    print('Testing OpenQML version ' + openqml.version() + ', Optimizer class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (OptTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
