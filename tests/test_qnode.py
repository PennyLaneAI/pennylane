"""
Unit tests for the :mod:`openqml` :class:`QNode` class.
"""

import unittest

import autograd
import autograd.numpy as np

from numpy.random import (randn,)

from defaults import openqml, BaseTest
from openqml.plugin import (load_plugin,)
from openqml.circuit import (QNode,)


class BasicTest(BaseTest):
    """ABC for tests.
    """
    def setUp(self):
        self.plugin = load_plugin('dummy_plugin')
        self.circuit = self.plugin.get_circuit('demo_ev')

    def test_qnode(self):
        "Quantum node and node gradient evaluation."
        p1 = self.plugin('node 1')
        q1 = QNode(self.circuit, p1)
        params = randn(2)
        #x0 = q1.evaluate(params)
        # manual gradients
        grad1 = q1.gradient_finite_diff(params)
        grad2 = q1.gradient_angle(params)
        # automatic gradient
        grad = autograd.grad(q1.evaluate)
        grad_auto = grad(params)

        # gradients computed with different methods must agree
        self.assertAllAlmostEqual(grad1, grad2, self.tol)
        self.assertAllAlmostEqual(grad_auto, grad2, self.tol)


    def test_autograd(self):
        "Automatic differentiation of a computational graph containing quantum nodes."
        p1 = self.plugin('node 1')
        q1 = QNode(self.circuit, p1)
        params = randn(1)
        data = randn(3, 2)

        def loss(p):
            "Simple quantum classifier, trying to map inputs to outputs."
            ret = 0
            for d in data:
                print(d[0])
                print(p[0])
                temp = np.r_[d[0], p[0]]
                print(temp)
                temp = q1.evaluate(temp) -d[1]
                ret += np.abs(temp) ** 2
            return ret

        grad = autograd.grad(loss)
        x0 = loss(params)
        print('loss:', x0)
        g = grad(params)
        print('autograd:', g)




if __name__ == '__main__':
    print('Testing OpenQML version ' + openqml.version() + ', QNode class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
