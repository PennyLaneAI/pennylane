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
    return qm.expectation.PauliZ(0)

class BasicTest(BaseTest):
    """Qnode tests.
    """
    def setUp(self):
        self.dev = qm.device('default.qubit', wires=2)

    def test_qnode_fail(self):
        "Tests that expected failures correctly raise exceptions."
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

if __name__ == '__main__':
    print('Testing OpenQML version ' + qm.version() + ', QNode class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
