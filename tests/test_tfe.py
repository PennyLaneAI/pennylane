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
Unit tests for the :mod:`pennylane.interface.tfe` QNode interface.
"""

import unittest
import logging as log
log.getLogger('defaults')

import numpy as np

try:
    import tensorflow as tf
    import tensorflow.contrib.eager as tfe
    tf.enable_eager_execution()
    tf_support = True
except ImportError as e:
    tf_support = False


from defaults import pennylane as qml, BaseTest

from pennylane.qnode import _flatten, unflatten, QNode, QuantumFunctionError
from pennylane.plugins.default_qubit import CNOT, Rotx, Roty, Rotz, I, Y, Z
from pennylane._device import DeviceError


def expZ(state):
    return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2


class TFEQNodeTests(BaseTest):
    """TFEQNode basic tests."""
    def setUp(self):
        if not tf_support:
            self.skipTest('TFE interface not tested')

        self.dev1 = qml.device('default.qubit', wires=1)
        self.dev2 = qml.device('default.qubit', wires=2)

    def test_qnode_fail(self):
        """Tests that QNode initialization failures correctly raise exceptions."""
        self.logTestName()
        par = tfe.Variable(0.5)

        #---------------------------------------------------------
        ## faulty quantum functions

        # qfunc must return only Observables
        @qml.qnode(self.dev2, interface='tfe')
        def qf(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(wires=0)), 0.3

        with self.assertRaisesRegex(QuantumFunctionError, 'must return either'):
            qf(par)

        # all EVs must be returned...
        @qml.qnode(self.dev2, interface='tfe')
        def qf(x):
            qml.RX(x, wires=[0])
            ex = qml.expval(qml.PauliZ(wires=1))
            return qml.expval(qml.PauliZ(wires=0))

        with self.assertRaisesRegex(QuantumFunctionError, 'All measured expectation values'):
            qf(par)

        # ...in the correct order
        @qml.qnode(self.dev2, interface='tfe')
        def qf(x):
            qml.RX(x, wires=[0])
            ex = qml.expval(qml.PauliZ(wires=1))
            return qml.expval(qml.PauliZ(wires=0)), ex

        with self.assertRaisesRegex(QuantumFunctionError, 'All measured expectation values'):
            qf(par)

        # gates must precede EVs
        @qml.qnode(self.dev2, interface='tfe')
        def qf(x):
            qml.RX(x, wires=[0])
            ev = qml.expval(qml.PauliZ(wires=1))
            qml.RY(0.5, wires=[0])
            return ev

        with self.assertRaisesRegex(QuantumFunctionError, 'gates must precede'):
            qf(par)

        # a wire cannot be measured more than once
        @qml.qnode(self.dev2, interface='tfe')
        def qf(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1)), qml.expval(qml.PauliX(wires=0))

        with self.assertRaisesRegex(QuantumFunctionError, 'can only be measured once'):
            qf(par)

        # device must have enough wires for the qfunc
        @qml.qnode(self.dev2, interface='tfe')
        def qf(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 2])
            return qml.expval(qml.PauliZ(wires=0))

        with self.assertRaisesRegex(QuantumFunctionError, 'applied to invalid wire'):
            qf(par)

        # CV and discrete ops must not be mixed
        @qml.qnode(self.dev1, interface='tfe')
        def qf(x):
            qml.RX(x, wires=[0])
            qml.Displacement(0.5, 0, wires=[0])
            return qml.expval(qml.PauliZ(wires=0))

        with self.assertRaisesRegex(QuantumFunctionError, 'Continuous and discrete'):
            qf(par)

        # default plugin cannot execute CV operations, neither gates...
        @qml.qnode(self.dev1, interface='tfe')
        def qf(x):
            qml.Displacement(0.5, 0, wires=[0])
            return qml.expval(qml.X(wires=0))

        with self.assertRaisesRegex(DeviceError, 'Gate [a-zA-Z]+ not supported on device'):
            qf(par)

        # ...nor observables
        @qml.qnode(self.dev1, interface='tfe')
        def qf(x):
            return qml.expval(qml.X(wires=0))

        with self.assertRaisesRegex(DeviceError, 'Observable [a-zA-Z]+ not supported on device'):
            qf(par)

    def test_qnode_fanout(self):
        """Tests that qnodes can compute the correct function when the same parameter is used in multiple gates."""
        self.logTestName()

        @qml.qnode(self.dev1, interface='tfe')
        def circuit(reused_param, other_param):
            qml.RX(reused_param, wires=[0])
            qml.RZ(other_param, wires=[0])
            qml.RX(reused_param, wires=[0])
            return qml.expval(qml.PauliZ(wires=0))

        thetas = tf.linspace(-2*np.pi, 2*np.pi, 7)

        for reused_param in thetas:
            for theta in thetas:
                other_param = theta ** 2 / 11
                y_eval = circuit(reused_param, other_param)
                Rx = Rotx(reused_param.numpy())
                Rz = Rotz(other_param.numpy())
                zero_state = np.array([1.,0.])
                final_state = (Rx @ Rz @ Rx @ zero_state)
                y_true = expZ(final_state)
                self.assertAlmostEqual(y_eval, y_true, delta=self.tol)

    def test_qnode_array_parameters(self):
        "Test that QNode can take arrays as input arguments, and that they interact properly with TensorFlow."
        self.logTestName()

        @qml.qnode(self.dev1, interface='tfe')
        def circuit_n1s(dummy1, array, dummy2):
            qml.RY(0.5 * array[0,1], wires=0)
            qml.RY(-0.5 * array[1,1], wires=0)
            return qml.expval(qml.PauliX(wires=0))  # returns a scalar

        @qml.qnode(self.dev1, interface='tfe')
        def circuit_n1v(dummy1, array, dummy2):
            qml.RY(0.5 * array[0,1], wires=0)
            qml.RY(-0.5 * array[1,1], wires=0)
            return qml.expval(qml.PauliX(wires=0)),  # note the comma, returns a 1-vector

        @qml.qnode(self.dev2, interface='tfe')
        def circuit_nn(dummy1, array, dummy2):
            qml.RY(0.5 * array[0,1], wires=0)
            qml.RY(-0.5 * array[1,1], wires=0)
            qml.RY(array[1,0], wires=1)
            return qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliX(wires=1))  # returns a 2-vector

        grad_target = (np.array(1.), np.array([[0.5,  0.43879, 0], [0, -0.43879, 0]]), np.array(-0.4))
        cost_target = 1.03257

        for circuit in [circuit_n1s, circuit_n1v, circuit_nn]:

            args = (tfe.Variable(0.46), tfe.Variable([[2., 3., 0.3], [7., 4., 2.1]]), tfe.Variable(-0.13))

            def cost(x, array, y):
                c = tf.cast(circuit(tf.constant(0.111), array, tf.constant(4.5)), tf.float32)
                if c.shape != tf.TensorShape([]):
                    c = c[0]  # get a scalar
                return c +0.5*array[0,0] +x -0.4*y

            with tf.GradientTape() as tape:
                cost_res = cost(*args)
                grad_res = np.array([i.numpy() for i in tape.gradient(cost_res, [args[0], args[2]])])

            self.assertAllAlmostEqual(cost_res.numpy(), cost_target, delta=self.tol)
            self.assertAllAlmostEqual(grad_res, np.fromiter(grad_target[::2], dtype=np.float32), delta=self.tol)

    def test_array_parameters_evaluate(self):
        "Test that array parameters gives same result as positional arguments."
        self.logTestName()

        a, b, c = tf.constant(0.5), tf.constant(0.54), tf.constant(0.3)

        def ansatz(x, y, z):
            qml.QubitStateVector(np.array([1, 0, 1, 1])/np.sqrt(3), wires=[0, 1])
            qml.Rot(x, y, z, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliY(wires=1))

        @qml.qnode(self.dev2, interface='tfe')
        def circuit1(x, y, z):
            return ansatz(x, y, z)

        @qml.qnode(self.dev2, interface='tfe')
        def circuit2(x, array):
            return ansatz(x, array[0], array[1])

        @qml.qnode(self.dev2, interface='tfe')
        def circuit3(array):
            return ansatz(*array)

        positional_res = circuit1(a, b, c)
        array_res1 = circuit2(a, tfe.Variable([b, c]))
        array_res2 = circuit3(tfe.Variable([a, b, c]))
        self.assertAllAlmostEqual(positional_res, array_res1, delta=self.tol)
        self.assertAllAlmostEqual(positional_res, array_res2, delta=self.tol)

    def test_multiple_expectation_different_wires(self):
        "Tests that qnodes return multiple expectation values."
        self.logTestName()

        a, b, c = tf.constant(0.5), tf.constant(0.54), tf.constant(0.3)

        @qml.qnode(self.dev2, interface='tfe')
        def circuit(x, y, z):
            qml.RX(x, wires=[0])
            qml.RZ(y, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=[0])
            qml.RX(z, wires=[0])
            return qml.expval(qml.PauliY(wires=0)), qml.expval(qml.PauliZ(wires=1))

        res = circuit(a, b, c)

        out_state = np.kron(Rotx(c.numpy()), I) @ np.kron(Roty(b.numpy()), I) @ CNOT \
            @ np.kron(Rotz(b.numpy()), I) @ np.kron(Rotx(a.numpy()), I) @ np.array([1, 0, 0, 0])

        ex0 = np.vdot(out_state, np.kron(Y, I) @ out_state)
        ex1 = np.vdot(out_state, np.kron(I, Z) @ out_state)
        ex = np.array([ex0, ex1])
        self.assertAllAlmostEqual(ex, res.numpy(), delta=self.tol)

    def test_multiple_keywordargs_used(self):
        "Tests that qnodes use multiple keyword arguments."
        self.logTestName()

        def circuit(w, x=None, y=None):
            qml.RX(x, wires=[0])
            qml.RX(y, wires=[1])
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        circuit = qml.QNode(circuit, self.dev2).to_tfe()

        c = circuit(tf.constant(1.), x=np.pi, y=np.pi)
        self.assertAllAlmostEqual(c.numpy(), [-1., -1.], delta=self.tol)

    def test_multidimensional_keywordargs_used(self):
        "Tests that qnodes use multi-dimensional keyword arguments."
        self.logTestName()

        def circuit(w, x=None):
            qml.RX(x[0], wires=[0])
            qml.RX(x[1], wires=[1])
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        circuit = qml.QNode(circuit, self.dev2).to_tfe()

        c = circuit(tf.constant(1.), x=[np.pi, np.pi])
        self.assertAllAlmostEqual(c, [-1., -1.], delta=self.tol)

    def test_keywordargs_for_wires(self):
        "Tests that wires can be passed as keyword arguments."
        self.logTestName()

        default_q = 0

        def circuit(x, q=default_q):
            qml.RY(x, wires=0)
            return qml.expval.PauliZ(wires=q)

        circuit = qml.QNode(circuit, self.dev2).to_tfe()

        c = circuit(tf.constant(np.pi), q=1)
        self.assertAlmostEqual(c, 1., delta=self.tol)

        c = circuit(tf.constant(np.pi))
        self.assertAlmostEqual(c, -1., delta=self.tol)

    def test_keywordargs_used(self):
        "Tests that qnodes use keyword arguments."
        self.logTestName()

        def circuit(w, x=None):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(wires=0))

        circuit = qml.QNode(circuit, self.dev1).to_tfe()

        c = circuit(tf.constant(1.), x=np.pi)
        self.assertAlmostEqual(c, -1., delta=self.tol)

    def test_mixture_numpy_tensors(self):
        "Tests that qnodes work with python types and tensors."
        self.logTestName()

        @qml.qnode(self.dev2, interface='tfe')
        def circuit(w, x, y):
            qml.RX(x, wires=[0])
            qml.RX(y, wires=[1])
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        c = circuit(tf.constant(1.), np.pi, np.pi).numpy()
        self.assertAllAlmostEqual(c, [-1., -1.], delta=self.tol)

    def test_keywordarg_updated_in_multiple_calls(self):
        "Tests that qnodes update keyword arguments in consecutive calls."
        self.logTestName()

        def circuit(w, x=None):
            qml.RX(w, wires=[0])
            qml.RX(x, wires=[1])
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        circuit = qml.QNode(circuit, self.dev2).to_tfe()

        c1 = circuit(tf.constant(0.1), x=tf.constant(0.))
        c2 = circuit(tf.constant(0.1), x=np.pi)
        self.assertTrue(c1[1] != c2[1])

    def test_keywordarg_passes_through_classicalnode(self):
        "Tests that qnodes' keyword arguments pass through classical nodes."
        self.logTestName()

        def circuit(w, x=None):
            qml.RX(w, wires=[0])
            qml.RX(x, wires=[1])
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        circuit = qml.QNode(circuit, self.dev2).to_tfe()

        def classnode(w, x=None):
            return circuit(w, x=x)

        c = classnode(tf.constant(0.), x=np.pi)
        self.assertAllAlmostEqual(c, [1., -1.], delta=self.tol)

    def test_keywordarg_gradient(self):
        "Tests that qnodes' keyword arguments work with gradients"
        self.logTestName()

        def circuit(x, y, input_state=np.array([0, 0])):
            qml.BasisState(input_state, wires=[0, 1])
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[0])
            return qml.expval(qml.PauliZ(wires=0))

        circuit = qml.QNode(circuit, self.dev2).to_tfe()

        x = 0.543
        y = 0.45632
        expected_grad = np.array([np.sin(x)*np.cos(y), np.sin(y)*np.cos(x)])

        x_t = tfe.Variable(x)
        y_t = tfe.Variable(y)

        # test first basis state against analytic result
        with tf.GradientTape() as tape:
            c = circuit(x_t, y_t, input_state=np.array([0, 0]))
            grads = np.array(tape.gradient(c, [x_t, y_t]))

        self.assertAllAlmostEqual(grads, -expected_grad, delta=self.tol)

        # test third basis state against analytic result
        with tf.GradientTape() as tape:
            c = circuit(x_t, y_t, input_state=np.array([1, 0]))
            grads = np.array(tape.gradient(c, [x_t, y_t]))

        self.assertAllAlmostEqual(grads, expected_grad, delta=self.tol)

        # test first basis state via the default keyword argument against analytic result
        with tf.GradientTape() as tape:
            c = circuit(x_t, y_t)
            grads = np.array(tape.gradient(c, [x_t, y_t]))

        self.assertAllAlmostEqual(grads, -expected_grad, delta=self.tol)


class IntegrationTests(BaseTest):
    """Integration tests to ensure the TensorFlow QNode agrees with the NumPy QNode"""

    def setUp(self):
        if not tf_support:
            self.skipTest('TFE interface not tested')

    def test_qnode_evaluation_agrees(self):
        "Tests that simple example is consistent."
        self.logTestName()

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev, interface='autograd')
        def circuit(phi, theta):
            qml.RX(phi[0], wires=0)
            qml.RY(phi[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta[0], wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        @qml.qnode(dev, interface='tfe')
        def circuit_tfe(phi, theta):
            qml.RX(phi[0], wires=0)
            qml.RY(phi[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta[0], wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        phi = [0.5, 0.1]
        theta = [0.2]

        phi_t = tfe.Variable(phi)
        theta_t = tfe.Variable(theta)

        autograd_eval = circuit(phi, theta)
        tfe_eval = circuit_tfe(phi_t, theta_t)
        self.assertAllAlmostEqual(autograd_eval, tfe_eval.numpy(), delta=self.tol)

    def test_qnode_gradient_agrees(self):
        "Tests that simple gradient example is consistent."
        self.logTestName()

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev, interface='autograd')
        def circuit(phi, theta):
            qml.RX(phi[0], wires=0)
            qml.RY(phi[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta[0], wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        @qml.qnode(dev, interface='tfe')
        def circuit_tfe(phi, theta):
            qml.RX(phi[0], wires=0)
            qml.RY(phi[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta[0], wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        phi = [0.5, 0.1]
        theta = [0.2]

        phi_t = tfe.Variable(phi)
        theta_t = tfe.Variable(theta)

        dcircuit = qml.grad(circuit, [0, 1])
        autograd_grad = dcircuit(phi, theta)

        dcircuit = tfe.gradients_function(circuit_tfe)
        tfe_grad = dcircuit(phi_t, theta_t)

        self.assertAllAlmostEqual(autograd_grad[0], tfe_grad[0], delta=self.tol)
        self.assertAllAlmostEqual(autograd_grad[1], tfe_grad[1], delta=self.tol)


if __name__ == '__main__':
    print('Testing PennyLane version ' + qml.version() + ', QNode TFE interface.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (TFEQNodeTests, IntegrationTests):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
