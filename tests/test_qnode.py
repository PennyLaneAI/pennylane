"""
Unit tests for the :mod:`openqml` :class:`QNode` class.
"""

import unittest

import autograd
import autograd.numpy as np
from autograd.numpy.random import (randn,)

from defaults import openqml, BaseTest
from openqml.plugin import (load_plugin,)
from openqml.circuit import (QNode,)


class BasicTest(BaseTest):
    """ABC for tests.
    """
    def setUp(self):
        #self.plugin = load_plugin('dummy_plugin')
        self.plugin = load_plugin('strawberryfields')
        self.circuit = self.plugin.get_circuit('demo_ev')


    def test_qnode(self):
        "Quantum node and node gradient evaluation."

        p = self.plugin('test node', backend='gaussian')  # use exact ev:s, gaussian backend so we get no truncation errors
        q = QNode(self.circuit, p)
        params = randn(q.circuit.n_par)
        x0 = q.evaluate(params)
        # manual gradients
        grad_fd1 = q.gradient(params, method='F', order=1)
        grad_fd2 = q.gradient(params, method='F', order=2)
        grad_angle = q.gradient(params, method='A')
        # automatic gradient (uses the default gradient method)
        grad = autograd.grad(q.evaluate)
        grad_auto = grad(params)

        #print('fd1:', grad_fd1)
        #print('fd2:', grad_fd2)
        #print('ang:', grad_angle)
        #print('aut:', grad_auto)

        # gradients computed with different methods must agree
        self.assertAllAlmostEqual(grad_fd1, grad_fd2, self.tol)
        self.assertAllAlmostEqual(grad_fd1, grad_angle, self.tol)
        self.assertAllAlmostEqual(grad_fd1, grad_auto, self.tol)


    def test_qnode_fail(self):
        "Expected failures."

        p = self.plugin('test node')
        q = QNode(self.circuit, p)
        params = randn(q.circuit.n_par)
        # only order-1 and order-2 finite diff methods are available
        self.assertRaises(ValueError, q.gradient, params, method='F', order=3)


    def test_autograd(self):
        "Automatic differentiation of a computational graph containing quantum nodes."

        p1 = self.plugin('node 1', backend='gaussian')
        q1 = QNode(self.circuit, p1)
        n_par = q1.circuit.n_par -1  # input data is the first parameter
        params = randn(n_par)
        data = randn(3, 2)

        def error(p):
            "Simple quantum classifier, trying to map inputs to outputs."
            ret = 0
            for d in data:
                x = np.concatenate((d[0:1], p))
                temp = q1.evaluate(x) -d[1]
                ret += temp ** 2
            return ret

        def d_error(p, grad_method):
            "Gradient of error, computed manually."
            ret = np.zeros(n_par, dtype=float)
            for d in data:
                x = np.concatenate((d[0:1], p))
                temp = q1.evaluate(x) -d[1]
                ret += 2 * temp * q1.gradient(x, which=range(1, n_par+1), method=grad_method).flatten()
            return ret

        y0 = error(params)
        grad = autograd.grad(error)
        grad_auto = grad(params)

        grad_fd1 = d_error(params, 'F')
        grad_angle = d_error(params, 'A')
        #print('fd1:', grad_fd1)
        #print('ang:', grad_angle)
        #print('aut:', grad_auto)

        tol = 1e-5  # avoid spurious assertion failures due to numerical errors
        self.assertAllAlmostEqual(grad_fd1, grad_auto, tol)
        self.assertAllAlmostEqual(grad_angle, grad_auto, tol)



if __name__ == '__main__':
    print('Testing OpenQML version ' + openqml.version() + ', QNode class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
