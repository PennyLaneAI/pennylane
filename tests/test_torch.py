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
Unit tests for the :mod:`pennylane.interface.torch` QNode interface.
"""

import unittest
import logging as log
log.getLogger('defaults')

import numpy as np

import torch
from torch.autograd import Variable

from defaults import pennylane as qml, BaseTest

from pennylane.qnode import _flatten, unflatten, QNode, QuantumFunctionError
from pennylane.plugins.default_qubit import CNOT, Rotx, Roty, Rotz, I, Y, Z
from pennylane._device import DeviceError


def expZ(state):
    return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2


class TorchQNodeTests(BaseTest):
    """TorchQNode basic tests."""
    def setUp(self):
        self.dev1 = qml.device('default.qubit', wires=1)
        self.dev2 = qml.device('default.qubit', wires=2)

    def test_qnode_fail(self):
        """Tests that QNode initialization failures correctly raise exceptions."""
        self.logTestName()
        par = torch.tensor(0.5)

        #---------------------------------------------------------
        ## faulty quantum functions

        # qfunc must return only Expectations
        @qml.qnode(self.dev2, interface='torch')
        def qf(x):
            qml.RX(x, [0])
            return qml.expval.PauliZ(0), 0.3

        with self.assertRaisesRegex(QuantumFunctionError, 'must return either'):
            qf(par)

        # all EVs must be returned...
        @qml.qnode(self.dev2, interface='torch')
        def qf(x):
            qml.RX(x, [0])
            ex = qml.expval.PauliZ(1)
            return qml.expval.PauliZ(0)

        with self.assertRaisesRegex(QuantumFunctionError, 'All measured expectation values'):
            qf(par)

        # ...in the correct order
        @qml.qnode(self.dev2, interface='torch')
        def qf(x):
            qml.RX(x, [0])
            ex = qml.expval.PauliZ(1)
            return qml.expval.PauliZ(0), ex

        with self.assertRaisesRegex(QuantumFunctionError, 'All measured expectation values'):
            qf(par)

        # gates must precede EVs
        @qml.qnode(self.dev2, interface='torch')
        def qf(x):
            qml.RX(x, [0])
            ev = qml.expval.PauliZ(1)
            qml.RY(0.5, [0])
            return ev

        with self.assertRaisesRegex(QuantumFunctionError, 'gates must precede'):
            qf(par)

        # a wire cannot be measured more than once
        @qml.qnode(self.dev2, interface='torch')
        def qf(x):
            qml.RX(x, [0])
            qml.CNOT([0, 1])
            return qml.expval.PauliZ(0), qml.expval.PauliZ(1), qml.expval.PauliX(0)

        with self.assertRaisesRegex(QuantumFunctionError, 'can only be measured once'):
            qf(par)

        # device must have enough wires for the qfunc
        @qml.qnode(self.dev2, interface='torch')
        def qf(x):
            qml.RX(x, [0])
            qml.CNOT([0, 2])
            return qml.expval.PauliZ(0)

        with self.assertRaisesRegex(QuantumFunctionError, 'applied to invalid wire'):
            qf(par)

        # CV and discrete ops must not be mixed
        @qml.qnode(self.dev1, interface='torch')
        def qf(x):
            qml.RX(x, [0])
            qml.Displacement(0.5, 0, [0])
            return qml.expval.PauliZ(0)

        with self.assertRaisesRegex(QuantumFunctionError, 'Continuous and discrete'):
            qf(par)

        # default plugin cannot execute CV operations, neither gates...
        @qml.qnode(self.dev1, interface='torch')
        def qf(x):
            qml.Displacement(0.5, 0, [0])
            return qml.expval.X(0)

        with self.assertRaisesRegex(DeviceError, 'Gate [a-zA-Z]+ not supported on device'):
            qf(par)

        # ...nor observables
        @qml.qnode(self.dev1, interface='torch')
        def qf(x):
            return qml.expval.X(0)

        with self.assertRaisesRegex(DeviceError, 'Expectation [a-zA-Z]+ not supported on device'):
            qf(par)

    def test_qnode_fanout(self):
        """Tests that qnodes can compute the correct function when the same parameter is used in multiple gates."""
        self.logTestName()

        @qml.qnode(self.dev1, interface='torch')
        def circuit(reused_param, other_param):
            qml.RX(reused_param, [0])
            qml.RZ(other_param, [0])
            qml.RX(reused_param, [0])
            return qml.expval.PauliZ(0)

        thetas = torch.linspace(-2*np.pi, 2*np.pi, 7)

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
        "Test that QNode can take arrays as input arguments, and that they interact properly with PyTorch."
        self.logTestName()

        @qml.qnode(self.dev1, interface='torch')
        def circuit_n1s(dummy1, array, dummy2):
            qml.RY(0.5 * array[0,1], 0)
            qml.RY(-0.5 * array[1,1], 0)
            return qml.expval.PauliX(0)  # returns a scalar

        @qml.qnode(self.dev1, interface='torch')
        def circuit_n1v(dummy1, array, dummy2):
            qml.RY(0.5 * array[0,1], 0)
            qml.RY(-0.5 * array[1,1], 0)
            return qml.expval.PauliX(0),  # note the comma, returns a 1-vector

        @qml.qnode(self.dev2, interface='torch')
        def circuit_nn(dummy1, array, dummy2):
            qml.RY(0.5 * array[0,1], 0)
            qml.RY(-0.5 * array[1,1], 0)
            qml.RY(array[1,0], 1)
            return qml.expval.PauliX(0), qml.expval.PauliX(1)  # returns a 2-vector

        grad_target = (np.array(1.), np.array([[0.5,  0.43879, 0], [0, -0.43879, 0]]), np.array(-0.4))
        cost_target = 1.03257

        for circuit in [circuit_n1s, circuit_n1v, circuit_nn]:

            args = (torch.tensor(0.46), torch.tensor([[2., 3., 0.3], [7., 4., 2.1]]), torch.tensor(-0.13))
            for i in args:
                i.requires_grad = True

            def cost(x, array, y):
                c = torch.as_tensor(circuit(torch.tensor(0.111), array, torch.tensor(4.5)), dtype=torch.float32)
                if c.shape:
                    c = c[0]  # get a scalar
                return c +0.5*array[0,0] +x -0.4*y

            cost_res = cost(*args)
            cost_res.backward()

            self.assertAllAlmostEqual(cost_res.detach().numpy(), cost_target, delta=self.tol)

            for i in range(3):
                self.assertAllAlmostEqual(args[i].grad.detach().numpy(), grad_target[i], delta=self.tol)

    def test_array_parameters_evaluate(self):
        "Test that array parameters gives same result as positional arguments."
        self.logTestName()

        a, b, c = torch.tensor(0.5), torch.tensor(0.54), torch.tensor(0.3)

        def ansatz(x, y, z):
            qml.QubitStateVector(np.array([1, 0, 1, 1])/np.sqrt(3), [0, 1])
            qml.Rot(x, y, z, 0)
            qml.CNOT([0, 1])
            return qml.expval.PauliZ(0), qml.expval.PauliY(1)

        @qml.qnode(self.dev2, interface='torch')
        def circuit1(x, y, z):
            return ansatz(x, y, z)

        @qml.qnode(self.dev2, interface='torch')
        def circuit2(x, array):
            return ansatz(x, array[0], array[1])

        @qml.qnode(self.dev2, interface='torch')
        def circuit3(array):
            return ansatz(*array)

        positional_res = circuit1(a, b, c)
        array_res1 = circuit2(a, torch.tensor([b, c]))
        array_res2 = circuit3(torch.tensor([a, b, c]))
        self.assertAllAlmostEqual(positional_res, array_res1, delta=self.tol)
        self.assertAllAlmostEqual(positional_res, array_res2, delta=self.tol)

    def test_multiple_expectation_different_wires(self):
        "Tests that qnodes return multiple expectation values."
        self.logTestName()

        a, b, c = torch.tensor(0.5), torch.tensor(0.54), torch.tensor(0.3)

        @qml.qnode(self.dev2, interface='torch')
        def circuit(x, y, z):
            qml.RX(x, [0])
            qml.RZ(y, [0])
            qml.CNOT([0, 1])
            qml.RY(y, [0])
            qml.RX(z, [0])
            return qml.expval.PauliY(0), qml.expval.PauliZ(1)

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

        @qml.qnode(self.dev2, interface='torch')
        def circuit(w, x=np.pi, y=np.pi):
            qml.RX(x, [0])
            qml.RX(y, [1])
            return qml.expval.PauliZ(0), qml.expval.PauliZ(1)

        c = circuit(torch.tensor(1.))
        self.assertAllAlmostEqual(c.numpy(), [-1., -1.], delta=self.tol)

    def test_mixture_numpy_tensors(self):
        "Tests that qnodes work with python types and tensors."
        self.logTestName()

        @qml.qnode(self.dev2, interface='torch')
        def circuit(w, x, y):
            qml.RX(x, [0])
            qml.RX(y, [1])
            return qml.expval.PauliZ(0), qml.expval.PauliZ(1)

        c = circuit(torch.tensor(1.), np.pi, np.pi).detach().numpy()
        self.assertAllAlmostEqual(c, [-1., -1.], delta=self.tol)


class IntegrationTests(BaseTest):
    """Integration tests to ensure the Torch QNode agrees with the NumPy QNode"""

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
            return qml.expval.PauliZ(0)

        @qml.qnode(dev, interface='torch')
        def circuit_torch(phi, theta):
            qml.RX(phi[0], wires=0)
            qml.RY(phi[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta[0], wires=0)
            return qml.expval.PauliZ(0)

        phi = [0.5, 0.1]
        theta = [0.2]

        phi_t = torch.tensor(phi)
        theta_t = torch.tensor(theta)

        autograd_eval = circuit(phi, theta)
        torch_eval = circuit_torch(phi_t, theta_t)
        self.assertAllAlmostEqual(autograd_eval, torch_eval.detach().numpy(), delta=self.tol)

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
            return qml.expval.PauliZ(0)

        @qml.qnode(dev, interface='torch')
        def circuit_torch(phi, theta):
            qml.RX(phi[0], wires=0)
            qml.RY(phi[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta[0], wires=0)
            return qml.expval.PauliZ(0)

        phi = [0.5, 0.1]
        theta = [0.2]

        phi_t = torch.autograd.Variable(torch.tensor(phi), requires_grad=True)
        theta_t = torch.autograd.Variable(torch.tensor(theta), requires_grad=True)

        dcircuit = qml.grad(circuit, [0, 1])
        autograd_grad = dcircuit(phi, theta)

        torch_eval = circuit_torch(phi_t, theta_t)
        torch_eval.backward()

        self.assertAllAlmostEqual(autograd_grad[0], phi_t.grad.detach().numpy(), delta=self.tol)
        self.assertAllAlmostEqual(autograd_grad[1], theta_t.grad.detach().numpy(), delta=self.tol)


if __name__ == '__main__':
    print('Testing PennyLane version ' + qml.version() + ', QNode Torch interface.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (TorchQNodeTests, IntegrationTests):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
