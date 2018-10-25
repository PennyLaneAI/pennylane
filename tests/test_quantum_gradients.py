# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the computing gradients of quantum functions.
"""

import unittest
import logging as log
log.getLogger('defaults')

import autograd
import autograd.numpy as np

from defaults import openqml as qm, BaseTest
from openqml.plugins.default_qubit import frx as Rx, fry as Ry, frz as Rz

def expZ(state):
    return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2

hbar = 2
mag_alphas = np.linspace(0, 1.5, 5)
thetas = np.linspace(-2*np.pi, 2*np.pi, 8)
sqz_vals = np.linspace(0., 1., 5)

class CVGradientTest(BaseTest):
    """Tests of the automatic gradient method for CV circuits.
    """
    def setUp(self):
        self.gaussian_dev = qm.device('default.gaussian', wires=2)

    def test_rotation_gradient(self):
        "Tests that the automatic gradient of a phase space rotation is correct."
        self.logTestName()

        alpha = 0.5

        @qm.qnode(self.gaussian_dev)
        def circuit(y):
            qm.Displacement(alpha, 0., [0])
            qm.Rotation(y, [0])
            return qm.expval.X(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            # qfunc evalutes to hbar * alpha * cos(theta)
            manualgrad_val = - hbar * alpha * np.sin(theta)

            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_beamsplitter_gradient(self):
        "Tests that the automatic gradient of a beamsplitter is correct."
        self.logTestName()

        alpha = 0.5

        @qm.qnode(self.gaussian_dev)
        def circuit(y):
            qm.Displacement(alpha, 0., [0])
            qm.Beamsplitter(y, 0, [0, 1])
            return qm.expval.X(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            # qfunc evalutes to hbar * alpha * cos(theta)
            manualgrad_val = - hbar * alpha * np.sin(theta)

            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_displacement_gradient(self):
        "Tests that the automatic gradient of a phase space displacement is correct."
        self.logTestName()

        @qm.qnode(self.gaussian_dev)
        def circuit(r, phi):
            qm.Displacement(r, phi, [0])
            return qm.expval.X(0)

        grad_fn = autograd.grad(circuit)

        for mag in mag_alphas:
            for theta in thetas:
                #alpha = mag * np.exp(1j * theta)
                autograd_val = grad_fn(mag, theta)
                # qfunc evalutes to hbar * Re(alpha)
                manualgrad_val = hbar * np.cos(theta)

                self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_squeeze_gradient(self):
        "Tests that the automatic gradient of a phase space squeezing is correct."
        self.logTestName()

        alpha = 0.5

        @qm.qnode(self.gaussian_dev)
        def circuit(y, r=0.5):
            qm.Displacement(r, 0., [0])
            qm.Squeezing(y, 0., [0])
            return qm.expval.X(0)

        grad_fn = autograd.grad(circuit, 0)

        for r in sqz_vals:
            autograd_val = grad_fn(r)
            # qfunc evaluates to -exp(-r) * hbar * Re(alpha)
            manualgrad_val = -np.exp(-r) * hbar * alpha
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_cv_gradients_gaussian_circuit(self):
        """Tests that the gradients of circuits of gaussian gates match between the finite difference and analytic methods."""
        self.logTestName()

        class PolyN(qm.expval.PolyXP):
            "Mimics PhotonNumber using the arbitrary 2nd order observable interface. Results should be identical."
            def __init__(self, wires):
                hbar = 2
                q = np.diag([-0.5, 0.5/hbar, 0.5/hbar])
                super().__init__(q, wires=wires)
                self.name = 'PolyXP'

        gates = [cls for cls in qm.ops.cv.all_ops if cls.supports_analytic]
        obs   = [qm.expval.X, qm.expval.PhotonNumber, PolyN]
        par = [0.4]

        for G in reversed(gates):
            log.debug('Testing gate %s...', G.__name__[0])
            for O in obs:
                log.debug('Testing observable %s...', O.__name__[0])
                def circuit(x):
                    args = [0.3] * G.num_params
                    args[0] = x
                    qm.Displacement(0.5, 0, wires=0)
                    G(*args, wires=range(G.num_wires))
                    qm.Beamsplitter(1.3, -2.3, wires=[0, 1])
                    qm.Displacement(-0.5, 0, wires=0)
                    qm.Squeezing(0.5, -1.5, wires=0)
                    qm.Rotation(-1.1, wires=0)
                    return O(wires=0)

                q = qm.QNode(circuit, self.gaussian_dev)
                val = q.evaluate(par)
                # log.info('  value:', val)
                grad_F  = q.jacobian(par, method='F')
                grad_A2 = q.jacobian(par, method='A', force_order2=True)
                # log.info('  grad_F: ', grad_F)
                # log.info('  grad_A2: ', grad_A2)
                if O.ev_order == 1:
                    grad_A = q.jacobian(par, method='A')
                    # log.info('  grad_A: ', grad_A)
                    # the different methods agree
                    self.assertAllAlmostEqual(grad_A, grad_F, delta=self.tol)

                # analytic method works for every parameter
                self.assertTrue(q.grad_method_for_par == {0:'A'})
                # the different methods agree
                self.assertAllAlmostEqual(grad_A2, grad_F, delta=self.tol)


    def test_cv_gradients_multiple_gate_parameters(self):
        "Tests that gates with multiple free parameters yield correct gradients."
        self.logTestName()
        par = [0.4, -0.3, -0.7]

        def qf(x, y, z):
            qm.Displacement(x, 0.2, [0])
            qm.Squeezing(y, z, [0])
            qm.Rotation(-0.2, [0])
            return qm.expval.X(0)

        q = qm.QNode(qf, self.gaussian_dev)
        grad_F = q.jacobian(par, method='F')
        grad_A = q.jacobian(par, method='A')
        grad_A2 = q.jacobian(par, method='A', force_order2=True)

        # analytic method works for every parameter
        self.assertTrue(q.grad_method_for_par == {0:'A', 1:'A', 2:'A'})
        # the different methods agree
        self.assertAllAlmostEqual(grad_A, grad_F, delta=self.tol)
        self.assertAllAlmostEqual(grad_A2, grad_F, delta=self.tol)


    def test_cv_gradients_repeated_gate_parameters(self):
        "Tests that repeated use of a free parameter in a multi-parameter gate yield correct gradients."
        self.logTestName()
        par = [0.2, 0.3]

        def qf(x, y):
            qm.Displacement(x, 0, [0])
            qm.Squeezing(y, -1.3*y, [0])
            return qm.expval.X(0)

        q = qm.QNode(qf, self.gaussian_dev)
        grad_F = q.jacobian(par, method='F')
        grad_A = q.jacobian(par, method='A')
        grad_A2 = q.jacobian(par, method='A', force_order2=True)

        # analytic method works for every parameter
        self.assertTrue(q.grad_method_for_par == {0:'A', 1:'A'})
        # the different methods agree
        self.assertAllAlmostEqual(grad_A, grad_F, delta=self.tol)
        self.assertAllAlmostEqual(grad_A2, grad_F, delta=self.tol)


    def test_cv_gradients_parameters_inside_array(self):
        "Tests that free parameters inside an array passed to an Operation yield correct gradients."
        self.logTestName()
        par = [0.4, 1.3]

        def qf(x, y):
            qm.Displacement(0.5, 0, [0])
            qm.Squeezing(x, 0, [0])
            M = np.zeros((5, 5), dtype=object)
            M[1,1] = y
            M[1,2] = 1.0
            M[2,1] = 1.0
            return qm.expval.PolyXP(M, [0, 1])

        q = qm.QNode(qf, self.gaussian_dev)
        grad = q.jacobian(par)
        grad_F = q.jacobian(par, method='F')
        grad_A = q.jacobian(par, method='B')
        grad_A2 = q.jacobian(par, method='B', force_order2=True)

        # par[0] can use the 'A' method, par[1] cannot
        self.assertTrue(q.grad_method_for_par == {0:'A', 1:'F'})
        # the different methods agree
        self.assertAllAlmostEqual(grad, grad_F, delta=self.tol)


    def test_cv_gradient_fanout(self):
        "Tests that qnodes can compute the correct gradient when the same parameter is used in multiple gates."
        self.logTestName()
        par = [0.5, 1.3]

        def circuit(x, y):
            qm.Displacement(x, 0, [0])
            qm.Rotation(y, [0])
            qm.Displacement(0, x, [0])
            return qm.expval.X(0)

        q = qm.QNode(circuit, self.gaussian_dev)
        grad_F = q.jacobian(par, method='F')
        grad_A = q.jacobian(par, method='A')
        grad_A2 = q.jacobian(par, method='A', force_order2=True)

        # analytic method works for every parameter
        self.assertTrue(q.grad_method_for_par == {0:'A', 1:'A'})
        # the different methods agree
        self.assertAllAlmostEqual(grad_A, grad_F, delta=self.tol)
        self.assertAllAlmostEqual(grad_A2, grad_F, delta=self.tol)


class QubitGradientTest(BaseTest):
    """Tests of the automatic gradient method for qubit gates.
    """
    def setUp(self):
        self.qubit_dev1 = qm.device('default.qubit', wires=1)
        self.qubit_dev2 = qm.device('default.qubit', wires=2)

    def test_RX_gradient(self):
        "Tests that the automatic gradient of a Pauli X-rotation is correct."
        self.logTestName()

        @qm.qnode(self.qubit_dev1)
        def circuit(x):
            qm.RX(x, [0])
            return qm.expval.PauliZ(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_RY_gradient(self):
        "Tests that the automatic gradient of a Pauli Y-rotation is correct."
        self.logTestName()

        @qm.qnode(self.qubit_dev1)
        def circuit(x):
            qm.RY(x, [0])
            return qm.expval.PauliZ(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_RZ_gradient(self):
        "Tests that the automatic gradient of a Pauli Z-rotation is correct."
        self.logTestName()

        @qm.qnode(self.qubit_dev1)
        def circuit(x):
            qm.RZ(x, [0])
            return qm.expval.PauliZ(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_Rot(self):
        "Tests that the automatic gradient of a arbitrary Euler-angle-parameterized gate is correct."
        self.logTestName()

        @qm.qnode(self.qubit_dev1)
        def circuit(x,y,z):
            qm.Rot(x,y,z, [0])
            return qm.expval.PauliZ(0)

        grad_fn = autograd.grad(circuit, argnum=[0,1,2])

        eye = np.eye(3)
        for theta in thetas:
            angle_inputs = np.array([theta, theta ** 3, np.sqrt(2) * theta])
            autograd_val = grad_fn(*angle_inputs)
            for idx in range(3):
                onehot_idx = eye[idx]
                manualgrad_val = (circuit(angle_inputs + np.pi / 2 * onehot_idx) - circuit(angle_inputs - np.pi / 2 * onehot_idx)) / 2
                self.assertAlmostEqual(autograd_val[idx], manualgrad_val, delta=self.tol)

    def test_qfunc_gradients(self):
        "Tests that the various ways of computing the gradient of a qfunc all agree."
        self.logTestName()

        def circuit(x, y, z):
            qm.RX(x, [0])
            qm.CNOT([0, 1])
            qm.RY(-1.6, [0])
            qm.RY(y, [1])
            qm.CNOT([1, 0])
            qm.RX(z, [0])
            qm.CNOT([0, 1])
            return qm.expval.PauliZ(0)

        qnode = qm.QNode(circuit, self.qubit_dev2)
        params = np.array([0.1, -1.6, np.pi / 5])

        # manual gradients
        grad_fd1 = qnode.jacobian(params, method='F', order=1)
        grad_fd2 = qnode.jacobian(params, method='F', order=2)
        grad_angle = qnode.jacobian(params, method='A')

        # automatic gradient
        grad_fn = autograd.grad(qnode.evaluate)
        grad_auto = grad_fn(params)

        # gradients computed with different methods must agree
        self.assertAllAlmostEqual(grad_fd1, grad_fd2, self.tol)
        self.assertAllAlmostEqual(grad_fd1, grad_angle, self.tol)
        self.assertAllAlmostEqual(grad_fd1, grad_auto, self.tol)

    def test_hybrid_gradients(self):
        "Tests that the various ways of computing the gradient of a hybrid computation all agree."
        self.logTestName()

        # input data is the first parameter
        def classifier_circuit(in_data, x):
            qm.RX(in_data, [0])
            qm.CNOT([0, 1])
            qm.RY(-1.6, [0])
            qm.RY(in_data, [1])
            qm.CNOT([1, 0])
            qm.RX(x, [0])
            qm.CNOT([0, 1])
            return qm.expval.PauliZ(0)

        classifier = qm.QNode(classifier_circuit, self.qubit_dev2)

        param = -0.1259
        in_data = np.array([-0.1, -0.88, np.exp(0.5)])
        out_data = np.array([1.5, np.pi / 3, 0.0])

        def error(p):
            "Total square error of classifier predictions."
            ret = 0
            for d_in, d_out in zip(in_data, out_data):
                #args = np.array([d_in, p])
                square_diff = (classifier(d_in, p) - d_out) ** 2
                ret = ret + square_diff
            return ret

        def d_error(p, grad_method):
            "Gradient of error, computed manually."
            ret = 0
            for d_in, d_out in zip(in_data, out_data):
                args = np.array([d_in, p])
                diff = (classifier(args) - d_out)
                ret = ret + 2 * diff * classifier.jacobian(args, which=[1], method=grad_method)
            return ret

        y0 = error(param)
        grad = autograd.grad(error)
        grad_auto = grad([param])

        grad_fd1 = d_error(param, 'F')
        grad_angle = d_error(param, 'A')

        # gradients computed with different methods must agree
        self.assertAllAlmostEqual(grad_fd1, grad_angle, self.tol)
        self.assertAllAlmostEqual(grad_fd1, grad_auto, self.tol)
        self.assertAllAlmostEqual(grad_angle, grad_auto, self.tol)

    def test_qnode_gradient_fanout(self):
        "Tests that the correct gradient is computed for qnodes which use the same parameter in multiple gates."
        self.logTestName()

        extra_param = 0.31
        def circuit(reused_param, other_param):
            qm.RX(extra_param, [0])
            qm.RY(reused_param, [0])
            qm.RZ(other_param, [0])
            qm.RX(reused_param, [0])
            return qm.expval.PauliZ(0)

        f = qm.QNode(circuit, self.qubit_dev1)
        zero_state = np.array([1., 0.])

        for reused_p in thetas:
            reused_p = reused_p ** 3 / 19
            for other_p in thetas:
                other_p = other_p ** 2 / 11

                # autograd gradient
                grad = autograd.grad(f)
                grad_eval = grad(reused_p, other_p)

                # manual gradient
                grad_true0 = (expZ(Rx(reused_p) @ Rz(other_p) @ Ry(reused_p + np.pi / 2) @ Rx(extra_param) @ zero_state) \
                             -expZ(Rx(reused_p) @ Rz(other_p) @ Ry(reused_p - np.pi / 2) @ Rx(extra_param) @ zero_state)) / 2
                grad_true1 = (expZ(Rx(reused_p + np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state) \
                             -expZ(Rx(reused_p - np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state)) / 2
                grad_true = grad_true0 + grad_true1 # product rule

                self.assertAlmostEqual(grad_eval, grad_true, delta=self.tol)


if __name__ == '__main__':
    print('Testing OpenQML version ' + qm.version() + ', automatic gradients.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (CVGradientTest, QubitGradientTest):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
