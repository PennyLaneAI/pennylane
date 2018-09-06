"""
Unit tests for the :mod:`openqml` :class:`Optimizer` class.
"""

import unittest
import logging as log
log.getLogger()

import autograd
import autograd.numpy as np

from defaults import openqml as qm, BaseTest

thetas = np.linspace(-2*np.pi, 2*np.pi, 7)

class QubitGradientTest(BaseTest):
    """Tests of the automatic gradient method for qubit gates.
    """
    def setUp(self):
        self.dev = qm.device('default.qubit', wires=1)

    def test_RX_gradient(self):
        "Tests that the automatic gradient of a Pauli X-rotation is correct."
        log.info('test_RX_gradient')

        @qm.qfunc(self.dev)
        def circuit(x):
            qm.RX(x, [0])
            return qm.expectation.PauliZ(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_RY_gradient(self):
        "Tests that the automatic gradient of a Pauli Y-rotation is correct."
        log.info('test_RY_gradient')

        @qm.qfunc(self.dev)
        def circuit(x):
            qm.RY(x, [0])
            return qm.expectation.PauliZ(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_RZ_gradient(self):
        "Tests that the automatic gradient of a Pauli Z-rotation is correct."
        log.info('test_RZ_gradient')

        @qm.qfunc(self.dev)
        def circuit(x):
            qm.RZ(x, [0])
            return qm.expectation.PauliZ(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_Rot(self):
        "Tests that the automatic gradient of a arbitrary Euler-angle-parameterized gate is correct."
        log.info('test_Rot')

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
                self.assertAlmostEqual(autograd_val[idx], manualgrad_val, delta=self.tol)

    def test_gradient_functions_agree(self):
        "Tests that the various ways of computing the gradient of a qfunc all agree."
        log.info('test_gradient_functions_agree')

        def circuit(x, y, z):
            qm.RX(x, [0])
            qm.CNOT([0, 1])
            qm.RY(-1.6, [0])
            qm.RY(y, [1])
            qm.CNOT([1, 0])
            qm.RX(z, [0])
            qm.CNOT([0, 1])
            return qm.expectation.PauliZ(0)

        qnode = qm.QNode(circuit, qm.device('default.qubit', wires=2))
        params = np.array([0.1, -1.6, np.pi / 5])

        # manual gradients
        grad_fd1 = qnode.gradient(params, method='F', order=1)
        grad_fd2 = qnode.gradient(params, method='F', order=2)
        grad_angle = qnode.gradient(params, method='A')

        # automatic gradient
        grad_auto = qm.grad(qnode, params)

        # gradients computed with different methods must agree
        self.assertAllAlmostEqual(grad_fd1, grad_fd2, self.tol)
        self.assertAllAlmostEqual(grad_fd1, grad_angle, self.tol)
        self.assertAllAlmostEqual(grad_fd1, grad_auto, self.tol)


if __name__ == '__main__':
    print('Testing OpenQML version ' + qm.version() + ', automatic gradients.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (QubitGradientTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
