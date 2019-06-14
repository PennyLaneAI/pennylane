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

from defaults import pennylane as qml, BaseTest
from pennylane.plugins.default_qubit import Rotx as Rx, Roty as Ry, Rotz as Rz

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
        self.gaussian_dev = qml.device('default.gaussian', wires=2)

    def test_rotation_gradient(self):
        "Tests that the automatic gradient of a phase space rotation is correct."
        self.logTestName()

        alpha = 0.5

        @qml.qnode(self.gaussian_dev)
        def circuit(y):
            qml.Displacement(alpha, 0., wires=[0])
            qml.Rotation(y, wires=[0])
            return qml.expval.X(0)

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

        @qml.qnode(self.gaussian_dev)
        def circuit(y):
            qml.Displacement(alpha, 0., wires=[0])
            qml.Beamsplitter(y, 0, wires=[0, 1])
            return qml.expval.X(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            # qfunc evalutes to hbar * alpha * cos(theta)
            manualgrad_val = - hbar * alpha * np.sin(theta)

            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_displacement_gradient(self):
        "Tests that the automatic gradient of a phase space displacement is correct."
        self.logTestName()

        @qml.qnode(self.gaussian_dev)
        def circuit(r, phi):
            qml.Displacement(r, phi, wires=[0])
            return qml.expval.X(0)

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

        @qml.qnode(self.gaussian_dev)
        def circuit(y, r=0.5):
            qml.Displacement(r, 0., wires=[0])
            qml.Squeezing(y, 0., wires=[0])
            return qml.expval.X(0)

        grad_fn = autograd.grad(circuit, 0)

        for r in sqz_vals:
            autograd_val = grad_fn(r)
            # qfunc evaluates to -exp(-r) * hbar * Re(alpha)
            manualgrad_val = -np.exp(-r) * hbar * alpha
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_number_state_gradient(self):
        "Tests that the automatic gradient of a squeezed state with number state expectation is correct."
        self.logTestName()

        @qml.qnode(self.gaussian_dev)
        def circuit(y):
            qml.Squeezing(y, 0., wires=[0])
            return qml.expval.NumberState(np.array([2, 0]), wires=[0, 1])

        grad_fn = autograd.grad(circuit, 0)

        # (d/dr) |<2|S(r)>|^2 = 0.5 tanh(r)^3 (2 csch(r)^2 - 1) sech(r)
        for r in sqz_vals[1:]: # formula above is not valid for r=0
            autograd_val = grad_fn(r)
            manualgrad_val = 0.5*np.tanh(r)**3 * (2/(np.sinh(r)**2)-1) / np.cosh(r)
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_cv_gradients_gaussian_circuit(self):
        """Tests that the gradients of circuits of gaussian gates match between the finite difference and analytic methods."""
        self.logTestName()

        class PolyN(qml.expval.PolyXP):
            "Mimics MeanPhoton using the arbitrary 2nd order observable interface. Results should be identical."
            def __init__(self, wires):
                hbar = 2
                q = np.diag([-0.5, 0.5/hbar, 0.5/hbar])
                super().__init__(q, wires=wires)
                self.name = 'PolyXP'

        gates = [cls for cls in qml.ops.cv.all_ops if cls.supports_analytic]
        obs   = [qml.expval.X, qml.expval.MeanPhoton, PolyN]
        par = [0.4]

        for G in reversed(gates):
            log.debug('Testing gate %s...', G.__name__[0])
            for O in obs:
                log.debug('Testing observable %s...', O.__name__[0])
                def circuit(x):
                    args = [0.3] * G.num_params
                    args[0] = x
                    qml.Displacement(0.5, 0, wires=0)
                    G(*args, wires=range(G.num_wires))
                    qml.Beamsplitter(1.3, -2.3, wires=[0, 1])
                    qml.Displacement(-0.5, 0, wires=0)
                    qml.Squeezing(0.5, -1.5, wires=0)
                    qml.Rotation(-1.1, wires=0)
                    return O(wires=0)

                q = qml.QNode(circuit, self.gaussian_dev)
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
            qml.Displacement(x, 0.2, wires=[0])
            qml.Squeezing(y, z, wires=[0])
            qml.Rotation(-0.2, wires=[0])
            return qml.expval.X(0)

        q = qml.QNode(qf, self.gaussian_dev)
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
            qml.Displacement(x, 0, wires=[0])
            qml.Squeezing(y, -1.3*y, wires=[0])
            return qml.expval.X(0)

        q = qml.QNode(qf, self.gaussian_dev)
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
            qml.Displacement(0.5, 0, wires=[0])
            qml.Squeezing(x, 0, wires=[0])
            M = np.zeros((5, 5), dtype=object)
            M[1,1] = y
            M[1,2] = 1.0
            M[2,1] = 1.0
            return qml.expval.PolyXP(M, [0, 1])

        q = qml.QNode(qf, self.gaussian_dev)
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
            qml.Displacement(x, 0, wires=[0])
            qml.Rotation(y, wires=[0])
            qml.Displacement(0, x, wires=[0])
            return qml.expval.X(0)

        q = qml.QNode(circuit, self.gaussian_dev)
        grad_F = q.jacobian(par, method='F')
        grad_A = q.jacobian(par, method='A')
        grad_A2 = q.jacobian(par, method='A', force_order2=True)

        # analytic method works for every parameter
        self.assertTrue(q.grad_method_for_par == {0:'A', 1:'A'})
        # the different methods agree
        self.assertAllAlmostEqual(grad_A, grad_F, delta=self.tol)
        self.assertAllAlmostEqual(grad_A2, grad_F, delta=self.tol)

    def test_CVOperation_with_heisenberg_and_no_params(self):
        """An integration test for CV gates that support analytic differentiation
        if succeeding the gate to be differentiated, but cannot be differentiated
        themselves (for example, they may be Gaussian but accept no parameters).

        This ensures that, assuming their _heisenberg_rep is defined, the quantum
        gradient analytic method can still be used, and returns the correct result."""
        self.logTestName()

        for cls in qml.ops.cv.all_ops:
            if cls.supports_heisenberg and (not cls.supports_analytic):
                dev = qml.device('default.gaussian', wires=2)

                U = np.array([[0.51310276+0.81702166j, 0.13649626+0.22487759j],
                              [0.26300233+0.00556194j, -0.96414101-0.03508489j]])

                if cls.num_wires == 0:
                    w = list(range(2))
                else:
                    w = list(range(cls.num_wires))

                def circuit(x):
                    qml.Displacement(x, 0, wires=0)

                    if cls.par_domain == 'A':
                        cls(U, wires=w)
                    else:
                        cls(wires=w)
                    return qml.expval.X(0)

                qnode = qml.QNode(circuit, dev)
                grad_F = qnode.jacobian(0.5, method='F')
                grad_A = qnode.jacobian(0.5, method='A')
                grad_A2 = qnode.jacobian(0.5, method='A', force_order2=True)

                # par[0] can use the 'A' method
                self.assertTrue(qnode.grad_method_for_par == {0: 'A'})

                # the different methods agree
                self.assertAllAlmostEqual(grad_A, grad_F, delta=self.tol)
                self.assertAllAlmostEqual(grad_A2, grad_F, delta=self.tol)


class QubitGradientTest(BaseTest):
    """Tests of the automatic gradient method for qubit gates.
    """
    def setUp(self):
        self.qubit_dev1 = qml.device('default.qubit', wires=1)
        self.qubit_dev2 = qml.device('default.qubit', wires=2)

    def test_RX_gradient(self):
        "Tests that the automatic gradient of a Pauli X-rotation is correct."
        self.logTestName()

        @qml.qnode(self.qubit_dev1)
        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval.PauliZ(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_RY_gradient(self):
        "Tests that the automatic gradient of a Pauli Y-rotation is correct."
        self.logTestName()

        @qml.qnode(self.qubit_dev1)
        def circuit(x):
            qml.RY(x, wires=[0])
            return qml.expval.PauliZ(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_RZ_gradient(self):
        "Tests that the automatic gradient of a Pauli Z-rotation is correct."
        self.logTestName()

        @qml.qnode(self.qubit_dev1)
        def circuit(x):
            qml.RZ(x, wires=[0])
            return qml.expval.PauliZ(0)

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            self.assertAlmostEqual(autograd_val, manualgrad_val, delta=self.tol)

    def test_Rot(self):
        "Tests that the automatic gradient of a arbitrary Euler-angle-parameterized gate is correct."
        self.logTestName()

        @qml.qnode(self.qubit_dev1)
        def circuit(x,y,z):
            qml.Rot(x,y,z, wires=[0])
            return qml.expval.PauliZ(0)

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
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(-1.6, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[1, 0])
            qml.RX(z, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval.PauliZ(0)

        qnode = qml.QNode(circuit, self.qubit_dev2)
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
            qml.RX(in_data, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(-1.6, wires=[0])
            qml.RY(in_data, wires=[1])
            qml.CNOT(wires=[1, 0])
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval.PauliZ(0)

        classifier = qml.QNode(classifier_circuit, self.qubit_dev2)

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
            qml.RX(extra_param, wires=[0])
            qml.RY(reused_param, wires=[0])
            qml.RZ(other_param, wires=[0])
            qml.RX(reused_param, wires=[0])
            return qml.expval.PauliZ(0)

        f = qml.QNode(circuit, self.qubit_dev1)
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
    print('Testing PennyLane version ' + qml.version() + ', automatic gradients.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (CVGradientTest, QubitGradientTest):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
