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
from openqml import Optimizer, QNode

thetas = np.linspace(-2*np.pi, 2*np.pi, 7)

class QubitGradientTest(BaseTest):
    """Tests of the automatic gradient method for qubit gates.
    """
    def setUp(self):
        self.dev = qm.device('default.qubit', wires=1)

    def test_RX_gradient(self):
        "Tests that the automatic gradient of a Pauli X-rotation is correct."

        @qm.qfunc(self.dev)
        def circuit(x):
            qm.RX(x, [0])
            return qm.expectation.PauliZ(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=1e-3)

    def test_RY_gradient(self):
        "Tests that the automatic gradient of a Pauli Y-rotation is correct."

        @qm.qfunc(self.dev)
        def circuit(x):
            qm.RY(x, [0])
            return qm.expectation.PauliZ(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=1e-3)

    def test_RZ_gradient(self):
        "Tests that the automatic gradient of a Pauli Z-rotation is correct."

        @qm.qfunc(self.dev)
        def circuit(x):
            qm.RZ(x, [0])
            return qm.expectation.PauliZ(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=1e-3)

    def test_Rot(self):
        "Tests that the automatic gradient of a arbitrary Euler-angle-parameterized gate is correct."

        @qm.qfunc(self.dev)
        def circuit(x,y,z):
            qm.Rot(x,y,z, [0])
            return qm.expectation.PauliZ(0)

        grad_fn = autograd.grad(circuit)

        eye = np.eye(3)
        for theta in thetas:
            angle_inputs = np.array([theta, theta ** 3, np.sqrt(2) * theta])
            autograd_val = grad_fn(angle_inputs)
            for idx in range(3):
                onehot_idx = eye[idx]
                manualgrad_val = (circuit(angle_inputs + np.pi / 2 * onehot_idx) - circuit(angle_inputs - np.pi / 2 * onehot_idx)) / 2
                self.assertAlmostEqual(autograd_val[idx], manualgrad_val, delta=1e-3)



if __name__ == '__main__':
    print('Testing OpenQML version ' + qm.version() + ', automatic gradients.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (QubitGradientTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
