"""
Unit tests for the :mod:`openqml` :class:`QNode` class.
"""

import unittest
import logging as log
log.getLogger()

import autograd

from openqml import numpy as np
from defaults import openqml as qm, BaseTest


def rubbish_circuit(x):
    qm.Rot(x, 0.3, -0.2, [0])
    qm.SWAP([0, 1])
    qm.expectation.PauliZ(0)


def circuit_data(in_data, x):
    qm.RX(in_data, [0])
    qm.CNOT([0, 1])
    qm.RY(-1.6, [0])
    qm.RY(in_data, [1])
    qm.CNOT([1, 0])
    qm.RX(x, [0])
    qm.CNOT([0, 1])
    qm.expectation.PauliZ(0)


class BasicTest(BaseTest):
    """Qnode tests.
    """
    def setUp(self):
        self.dev = qm.device('default.qubit', wires=2)

    def test_qnode_fail(self):
        "Expected failures."
        log.info('test_qnode_fail')
        qnode = qm.QNode(rubbish_circuit, self.dev)
        params = np.random.randn(qnode.num_variables)
        qnode(params)

        # gradient_angle cannot handle more-than-one-parameter gates
        with self.assertRaisesRegex(ValueError, "only differentiate one-parameter gates"):
            res = qnode.gradient(params, method='A')

        # only order-1 and order-2 finite diff methods are available
        with self.assertRaisesRegex(ValueError, "Order must be 1 or 2"):
            qnode.gradient(params, method='F', order=3)


    def test_autograd(self):
        "Automatic differentiation of a computational graph containing quantum nodes."
        log.info('test_autograd')

        qnode = qm.QNode(circuit_data, self.dev)
        self.assertEqual(qnode.num_variables, 2)

        # input data is the first parameter
        params = np.random.randn(qnode.num_variables - 1)
        data = np.random.randn(3, 2)

        def error(p):
            "Simple quantum classifier, trying to map inputs to outputs."
            ret = 0
            for d in data:
                x = np.array([d[0], p[0]])
                temp = qnode(x) - d[1]
                ret += temp ** 2
            return ret

        def d_error(p, grad_method):
            "Gradient of error, computed manually."
            ret = 0
            for d in data:
                x = np.array([d[0], p[0]])
                temp = qnode(x) - d[1]
                ret += 2 * temp * qnode.gradient(x, which=[1], method=grad_method)
            return ret

        y0 = error(params)
        grad = autograd.grad(error)
        grad_auto = grad(params)
        # grad_auto = qm.grad(error, *params)

        grad_fd1 = d_error(params, 'F')
        grad_angle = d_error(params, 'A')
        self.assertAllAlmostEqual(grad_fd1, grad_angle, self.tol)
        self.assertAllAlmostEqual(grad_fd1, grad_auto, self.tol)
        self.assertAllAlmostEqual(grad_angle, grad_auto, self.tol)


if __name__ == '__main__':
    print('Testing OpenQML version ' + qm.version() + ', QNode class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
