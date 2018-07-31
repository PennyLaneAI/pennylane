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
        self.plugin = load_plugin('dummy_plugin')

    def test_qnode(self):
        "Quantum node and node gradient evaluation."

        self.circuit = self.plugin.get_circuit('demo_ev')
        p = self.plugin('test node')
        q = QNode(self.circuit, p)
        params = randn(q.circuit.n_par)
        x0 = q.evaluate(params)
        # manual gradients
        grad_fd1 = q.gradient_finite_diff(params, order=1)
        grad_fd2 = q.gradient_finite_diff(params, order=2)
        grad_angle = q.gradient_angle(params)
        # automatic gradient
        grad = autograd.grad(q.evaluate)
        grad_auto = grad(params)

        # gradients computed with different methods must agree
        self.assertAllAlmostEqual(grad_fd1, grad_fd2, self.tol)
        self.assertAllAlmostEqual(grad_fd1, grad_angle, self.tol)
        self.assertAllAlmostEqual(grad_fd1, grad_auto, self.tol)


    def test_qnode_fail(self):
        "Expected failures."
        self.circuit = self.plugin.get_circuit('rubbish')

        p = self.plugin('test node')
        q = QNode(self.circuit, p)
        params = randn(q.circuit.n_par)
        # gradient_angle cannot handle more-than-one-parameter gates
        self.assertRaises(ValueError, q.gradient_angle, params)
        # only order-1 and order-2 methods are available
        self.assertRaises(ValueError, q.gradient_finite_diff, params, **{'order': 3})


    def test_autograd(self):
        "Automatic differentiation of a computational graph containing quantum nodes."

        self.circuit = self.plugin.get_circuit('demo_ev')
        self.assertEqual(self.circuit.n_par, 2)
        p1 = self.plugin('node 1')
        q1 = QNode(self.circuit, p1)
        params = randn(q1.circuit.n_par -1)  # input data is the first parameter
        data = randn(3, 2)

        def error(p):
            "Simple quantum classifier, trying to map inputs to outputs."
            ret = 0
            for d in data:
                #x = np.array([d[0], p[0]])
                x = np.concatenate((d[0:1], p[0:1]))
                temp = q1.evaluate(x) -d[1]
                ret += temp ** 2
            return ret

        def d_error(p, grad_func):
            "Gradient of error, computed manually."
            ret = 0
            for d in data:
                x = np.array([d[0], p[0]])
                temp = q1.evaluate(x) -d[1]
                ret += 2 * temp * grad_func(x, which=[1])
            return ret

        y0 = error(params)
        grad = autograd.grad(error)
        grad_auto = grad(params)

        grad_fd1 = d_error(params, q1.gradient_finite_diff)
        grad_angle = d_error(params, q1.gradient_angle)
        self.assertAllAlmostEqual(grad_fd1, grad_auto, self.tol)
        self.assertAllAlmostEqual(grad_angle, grad_auto, self.tol)



if __name__ == '__main__':
    print('Testing OpenQML version ' + openqml.version() + ', QNode class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
