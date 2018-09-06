"""
Unit tests for the :mod:`openqml` :class:`Optimizer` class.
"""

import unittest
import logging as log
log.getLogger()

import autograd
import autograd.numpy as np

from defaults import openqml as qm, BaseTest
from openqml.plugins.default import frx as Rx, frz as Rz

def expZ(state):
    return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2

thetas = np.linspace(-2*np.pi, 2*np.pi, 7)

class QubitGradientTest(BaseTest):
    """Tests of the automatic gradient method for qubit gates.
    """
    def setUp(self):
        self.qubit_dev1 = qm.device('default.qubit', wires=1)
        self.qubit_dev2 = qm.device('default.qubit', wires=2)

    def test_RX_gradient(self):
        "Tests that the automatic gradient of a Pauli X-rotation is correct."
        log.info('test_RX_gradient')

        @qm.qfunc(self.qubit_dev1)
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

        @qm.qfunc(self.qubit_dev1)
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

        @qm.qfunc(self.qubit_dev1)
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

        @qm.qfunc(self.qubit_dev1)
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

    def test_qfunc_gradients(self):
        "Tests that the various ways of computing the gradient of a qfunc all agree."
        log.info('test_qfunc_gradients')

        def circuit(x, y, z):
            qm.RX(x, [0])
            qm.CNOT([0, 1])
            qm.RY(-1.6, [0])
            qm.RY(y, [1])
            qm.CNOT([1, 0])
            qm.RX(z, [0])
            qm.CNOT([0, 1])
            return qm.expectation.PauliZ(0)

        qnode = qm.QNode(circuit, self.qubit_dev2)
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

    def test_hybrid_gradients(self):
        "Tests that the various ways of computing the gradient of a hybrid computation all agree."
        log.info('test_hybrid_gradients')

        # input data is the first parameter
        def classifier_circuit(in_data, x):
            qm.RX(in_data, [0])
            qm.CNOT([0, 1])
            qm.RY(-1.6, [0])
            qm.RY(in_data, [1])
            qm.CNOT([1, 0])
            qm.RX(x, [0])
            qm.CNOT([0, 1])
            return qm.expectation.PauliZ(0)

        classifier = qm.QNode(classifier_circuit, self.qubit_dev2)

        param = -0.1259
        in_data = np.array([-0.1, -0.88, np.exp(0.5)])
        out_data = np.array([1.5, np.pi / 3, 0.0])

        def error(p):
            "Total square error of classifier predictions."
            ret = 0
            for d_in, d_out in zip(in_data, out_data):
                args = np.array([d_in, p])
                square_diff = (classifier(args) - d_out) ** 2
                ret = ret + square_diff
            return ret

        def d_error(p, grad_method):
            "Gradient of error, computed manually."
            ret = 0
            for d_in, d_out in zip(in_data, out_data):
                args = np.array([d_in, p])
                diff = (classifier(args) - d_out)
                ret = ret + 2 * diff * classifier.gradient(args, which=[1], method=grad_method)
            return ret

        y0 = error(param)
        grad = autograd.grad(error)
        grad_auto = grad(param)

        grad_fd1 = d_error(param, 'F')
        grad_angle = d_error(param, 'A')

        # gradients computed with different methods must agree
        self.assertAllAlmostEqual(grad_fd1, grad_angle, self.tol)
        self.assertAllAlmostEqual(grad_fd1, grad_auto, self.tol)
        self.assertAllAlmostEqual(grad_angle, grad_auto, self.tol)

    def test_qnode_gradient_fanout(self):
        "Tests that the correct gradient is computed for qnodes which use the same parameter in multiple gates."
        log.info('test_qnode_gradient_fanout')

        def circuit(reused_param, other_param):
            qm.RX(reused_param, [0])
            qm.RZ(other_param, [0])
            qm.RX(reused_param, [0])
            return qm.expectation.PauliZ(0)

        f = qm.QNode(circuit, self.qubit_dev1)
        zero_state = np.array([1., 0.])

        for reused_p in thetas:
            for theta in thetas:
                other_p = theta ** 2 / 11

                # autograd gradient
                grad = autograd.grad(f)
                grad_eval = grad(reused_p, other_p)

                # manual gradient
                grad_true0 = (expZ(Rx(reused_p + np.pi / 2) @ Rz(other_p) @ Rx(reused_p) @ zero_state) \
                    -expZ(Rx(reused_p - np.pi / 2) @ Rz(other_p) @ Rx(0.) @ zero_state)) / 2
                grad_true1 = (expZ(Rx(reused_p) @ Rz(other_p) @ Rx(reused_p + np.pi / 2) @ zero_state) \
                    -expZ(Rx(reused_p) @ Rz(other_p) @ Rx(reused_p - np.pi / 2) @ zero_state)) / 2
                grad_true = grad_true0 + grad_true1 # product rule

                self.assertAlmostEqual(grad_eval[0], grad_true, delta=self.tol)


if __name__ == '__main__':
    print('Testing OpenQML version ' + qm.version() + ', automatic gradients.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (QubitGradientTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
