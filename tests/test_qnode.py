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
Unit tests for the :mod:`openqml` :class:`QNode` class.
"""

import unittest
import logging as log
log.getLogger()

import autograd

from defaults import openqml as qm, BaseTest

from openqml import numpy as np
from openqml.qnode import _flatten
from openqml.plugins.default import CNOT, frx, fry, frz, I, Y, Z


def expZ(state):
    return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2


thetas = np.linspace(-2*np.pi, 2*np.pi, 7)


def rubbish_circuit(x):
    qm.Rot(x, 0.3, -0.2, [0])
    qm.SWAP([0, 1])
    return qm.expectation.PauliZ(0)


class BasicTest(BaseTest):
    """Qnode tests.
    """
    def setUp(self):
        self.qubit_dev1 = qm.device('default.qubit', wires=1)
        self.qubit_dev2 = qm.device('default.qubit', wires=2)

    def test_qnode_fail(self):
        "Tests that expected failures correctly raise exceptions."
        log.info('test_qnode_fail')
        qnode = qm.QNode(rubbish_circuit, self.qubit_dev2)
        params = np.random.randn(1)

        # only order-1 and order-2 finite diff methods are available
        with self.assertRaisesRegex(ValueError, "Order must be 1 or 2"):
            qnode.gradient(params, method='F', order=3)

    def test_qnode_fanout(self):
        "Tests that qnodes can compute the correct function when the same parameter is used in multiple gates."
        log.info('test_qnode_fanout')

        def circuit(reused_param, other_param):
            qm.RX(reused_param, [0])
            qm.RZ(other_param, [0])
            qm.RX(reused_param, [0])
            return qm.expectation.PauliZ(0)

        f = qm.QNode(circuit, self.qubit_dev1)

        for reused_param in thetas:
            for theta in thetas:
                other_param = theta ** 2 / 11
                y_eval = f(reused_param, other_param)
                Rx = frx(reused_param)
                Rz = frz(other_param)
                zero_state = np.array([1.,0.])
                final_state = (Rx @ Rz @ Rx @ zero_state)
                y_true = expZ(final_state)
                self.assertAlmostEqual(y_eval, y_true, delta=self.tol)

    def test_array_parameters_evaluate(self):
        "Test that array parameters gives same result as positional arguments."
        log.info('test_array_parameters')

        a, b, c = 0.5, 0.54, 0.3

        def ansatz(x, y, z):
            qm.QubitStateVector(np.array([1, 0, 1, 1])/np.sqrt(3), [0, 1])
            qm.Rot(x, y, z, 0)
            qm.CNOT([0, 1])
            return qm.expectation.PauliZ(0), qm.expectation.PauliY(1)

        def circuit1(x, y, z):
            return ansatz(x, y, z)

        def circuit2(x, array):
            return ansatz(x, array[0], array[1])

        def circuit3(array):
            return ansatz(*array)

        circuit1 = qm.QNode(circuit1, self.qubit_dev2)
        positional_res = circuit1(a, b, c)
        positional_grad = circuit1.gradient([a, b, c])

        circuit2 = qm.QNode(circuit2, self.qubit_dev2)
        array_res = circuit2(a, np.array([b, c]))
        array_grad = circuit2.gradient([a, np.array([b, c])])

        list_res = circuit2(a, [b, c])
        list_grad = circuit2.gradient([a, [b, c]])

        self.assertAllAlmostEqual(positional_res, array_res, delta=self.tol)
        self.assertAllAlmostEqual(positional_grad, array_grad, delta=self.tol)

        circuit3 = qm.QNode(circuit3, self.qubit_dev2)
        array_res = circuit2(np.array([a, b, c]))
        array_grad = circuit2.gradient([np.array([a, b, c])])

        list_res = circuit2([a, b, c])
        list_grad = circuit2.gradient([[a, b, c]])

        self.assertAllAlmostEqual(positional_res, array_res, delta=self.tol)
        self.assertAllAlmostEqual(positional_grad, array_grad, delta=self.tol)

    def test_array_parameters_autograd(self):
        "Test that array parameters autograd gives same result as positional arguments."
        log.info('test_array_parameters_autograd')

        a, b, c = 0.5, 0.54, 0.3

        def ansatz(x, y, z):
            qm.QubitStateVector(np.array([1, 0, 1, 1])/np.sqrt(3), [0, 1])
            qm.Rot(x, y, z, 0)
            qm.CNOT([0, 1])
            return qm.expectation.PauliZ(0)

        def circuit1(x, y, z):
            return ansatz(x, y, z)

        def circuit2(x, array):
            return ansatz(x, array[0], array[1])

        def circuit3(array):
            return ansatz(*array)

        def circuit1(x, y, z):
            qm.QubitStateVector(np.array([1, 0, 1, 1])/np.sqrt(3), [0, 1])
            qm.Rot(x, y, z, 0)
            qm.CNOT([0, 1])
            return qm.expectation.PauliZ(0)

        circuit1 = qm.QNode(circuit1, self.qubit_dev2)
        grad1 = autograd.grad(circuit1, argnum=[0, 1, 2])

        positional_grad = circuit1.gradient([a, b, c])
        positional_autograd = grad1(a, b, c)
        self.assertAllAlmostEqual(positional_grad, positional_autograd, delta=self.tol)

        circuit2 = qm.QNode(circuit2, self.qubit_dev2)
        grad2 = autograd.grad(circuit2, argnum=[0, 1])

        # NOTE: Mixing array and positional arguments doesn't seem to work with autograd!
        # array_grad = circuit2.gradient([a, np.array([b, c])])
        # array_autograd = grad2(a, np.array([b, c]))
        # array_autograd_flat = list(_flatten(array_autograd))
        # self.assertAllAlmostEqual(array_grad, array_autograd_flat, delta=self.tol)

        circuit3 = qm.QNode(circuit3, self.qubit_dev2)
        grad3 = autograd.grad(circuit3)

        array_grad = circuit3.gradient([np.array([a, b, c])])
        array_autograd = grad3(np.array([a, b, c]))
        self.assertAllAlmostEqual(array_grad, array_autograd, delta=self.tol)

    def test_multiple_expectation_same_wire(self):
        "Tests that qnodes raise error for multiple expectation values on the same wire."
        log.info('test_multiple_expectation_same_wire')

        @qm.qfunc(self.qubit_dev2)
        def circuit(x, y, z):
            qm.RX(x, [0])
            qm.RZ(y, [0])
            qm.RX(z, [0])
            qm.CNOT([0, 1])
            return qm.expectation.PauliY(1), qm.expectation.PauliZ(1)

        with self.assertRaisesRegex(qm.QuantumFunctionError, "can only be measured once"):
            circuit(0.5,0.54,0.3)

    @staticmethod
    def expected_jacobian(x, y, z):
        dw0dx = 2/3*np.sin(x)*np.sin(y)
        dw0dy = 1/3*(np.sin(y)-2*np.cos(x)*np.cos(y))
        dw0dz = 0

        dw1dx = -2/3*np.cos(x)*np.sin(y)
        dw1dy = -2/3*np.cos(y)*np.sin(x)
        dw1dz = 0

        return np.array([[dw0dx, dw0dy, dw0dz],
                         [dw1dx, dw1dy, dw1dz]])

    def test_multiple_expectation_different_wires(self):
        "Tests that qnodes return multiple expectation values."
        log.info('test_multiple_expectation_different_wires')

        a, b, c = 0.5, 0.54, 0.3

        @qm.qfunc(self.qubit_dev2)
        def circuit(x, y, z):
            qm.RX(x, [0])
            qm.RZ(y, [0])
            qm.CNOT([0, 1])
            qm.RY(y, [0])
            qm.RX(z, [0])
            return qm.expectation.PauliY(0), qm.expectation.PauliZ(1)

        res = circuit(a, b, c)

        out_state = np.kron(frx(c), I) @ np.kron(fry(b), I) @ CNOT \
            @ np.kron(frz(b), I) @ np.kron(frx(a), I) @ np.array([1, 0, 0, 0])

        ex0 = np.vdot(out_state, np.kron(Y, I) @ out_state)
        ex1 = np.vdot(out_state, np.kron(I, Z) @ out_state)
        ex = np.array([ex0, ex1])
        self.assertAllAlmostEqual(ex, res, delta=self.tol)

    def test_multiple_expectation_jacobian_positional(self):
        "Tests that qnodes using positional arguments return multiple expectation values."
        log.info('test_multiple_expectation_jacobian_positional')

        a, b, c = 0.5, 0.54, 0.3

        def circuit(x, y, z):
            qm.QubitStateVector(np.array([1, 0, 1, 1])/np.sqrt(3), [0, 1])
            qm.Rot(x, y, z, 0)
            qm.CNOT([0, 1])
            return qm.expectation.PauliZ(0), qm.expectation.PauliY(1)

        circuit = qm.QNode(circuit, self.qubit_dev2)

        # compare our manual Jacobian computation to theoretical result
        # Note: circuit.gradient actually returns a full jacobian in this case
        res = circuit.gradient(np.array([a, b, c]))
        self.assertAllAlmostEqual(self.expected_jacobian(a, b, c), res, delta=self.tol)

        # compare our manual Jacobian computation to autograd
        # not sure if this is the intended usage of jacobian
        jac0 = autograd.jacobian(circuit, 0)
        jac1 = autograd.jacobian(circuit, 1)
        jac2 = autograd.jacobian(circuit, 2)
        res = np.stack([jac0(a,b,c), jac1(a,b,c), jac2(a,b,c)]).T

        self.assertAllAlmostEqual(self.expected_jacobian(a, b, c), res, delta=self.tol)

    def test_multiple_expectation_jacobian_array(self):
        "Tests that qnodes using an array argument return multiple expectation values."
        log.info('test_multiple_expectation_jacobian_array')

        a, b, c = 0.5, 0.54, 0.3

        def circuit(weights):
           qm.QubitStateVector(np.array([1, 0, 1, 1])/np.sqrt(3), [0, 1])
           qm.Rot(weights[0], weights[1], weights[2], 0)
           qm.CNOT([0, 1])
           return qm.expectation.PauliZ(0), qm.expectation.PauliY(1)

        circuit = qm.QNode(circuit, self.qubit_dev2)

        res = circuit.gradient([np.array([a, b, c])])
        self.assertAllAlmostEqual(self.expected_jacobian(a, b, c), res, delta=self.tol)

        jac = autograd.jacobian(circuit, 0)
        res = jac(np.array([a, b, c]))
        self.assertAllAlmostEqual(self.expected_jacobian(a, b, c), res, delta=self.tol)


if __name__ == '__main__':
    print('Testing OpenQML version ' + qm.version() + ', QNode class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
