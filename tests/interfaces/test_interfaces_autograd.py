# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane` :class:`to_autograd` class.
"""

import autograd
import autograd.numpy as anp  # only to be used inside classical computational nodes
import pytest
import numpy as np

import pennylane as qml
from pennylane.qnodes.base import QuantumFunctionError
from pennylane.qnodes.qubit import QubitQNode
from pennylane.qnodes.cv import CVQNode

from pennylane.interfaces.autograd import to_autograd


alpha = 0.5  # displacement in tests
hbar = 2
mag_alphas = np.linspace(0, 1.5, 5)
thetas = np.linspace(-2*np.pi, 2*np.pi, 8)
sqz_vals = np.linspace(0., 1., 5)


class TestAutogradDetails:
    """Test configuration details of the autograd interface"""

    def test_interface_str(self, qubit_device_2_wires):
        """Test that the interface string is correctly identified
        as numpy"""
        def circuit(x, y, z):
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        circuit = to_autograd(QubitQNode(circuit, qubit_device_2_wires))
        assert circuit.interface == "autograd"


class TestAutogradJacobianCV:
    """Tests involving Autograd functions grad and jacobian for CV circuits."""

    @pytest.mark.parametrize('theta', thetas)
    def test_rotation_gradient(self, theta, tol):
        """Tests that the automatic gradient of a phase space rotation is correct."""

        def circuit(y):
            qml.Displacement(alpha, 0., wires=[0])
            qml.Rotation(y, wires=[0])
            return qml.expval(qml.X(0))

        dev = qml.device('default.gaussian', wires=1)
        circuit = to_autograd(QubitQNode(circuit, dev))
        grad_fn = autograd.grad(circuit)

        autograd_val = grad_fn(theta)
        # qfunc evalutes to hbar * alpha * cos(theta)
        manualgrad_val = - hbar * alpha * np.sin(theta)
        assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    @pytest.mark.parametrize('theta', thetas)
    def test_beamsplitter_gradient(self, theta, tol):
        """Tests that the automatic gradient of a beamsplitter is correct."""

        def circuit(y):
            qml.Displacement(alpha, 0., wires=[0])
            qml.Beamsplitter(y, 0, wires=[0, 1])
            return qml.expval(qml.X(0))

        dev = qml.device('default.gaussian', wires=2)
        circuit = to_autograd(CVQNode(circuit, dev))
        grad_fn = autograd.grad(circuit)

        autograd_val = grad_fn(theta)
        # qfunc evalutes to hbar * alpha * cos(theta)
        manualgrad_val = - hbar * alpha * np.sin(theta)
        assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    @pytest.mark.parametrize('mag', mag_alphas)
    @pytest.mark.parametrize('theta', thetas)
    def test_displacement_gradient(self, mag, theta, tol):
        """Tests that the automatic gradient of a phase space displacement is correct."""

        def circuit(r, phi):
            qml.Displacement(r, phi, wires=[0])
            return qml.expval(qml.X(0))

        dev = qml.device('default.gaussian', wires=1)
        circuit = to_autograd(CVQNode(circuit, dev))
        grad_fn = autograd.grad(circuit)

        #alpha = mag * np.exp(1j * theta)
        autograd_val = grad_fn(mag, theta)
        # qfunc evalutes to hbar * Re(alpha)
        manualgrad_val = hbar * np.cos(theta)
        assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    @pytest.mark.parametrize('r', sqz_vals)
    def test_squeeze_gradient(self, r, tol):
        """Tests that the automatic gradient of a phase space squeezing is correct."""

        def circuit(y):
            qml.Displacement(alpha, 0., wires=[0])
            qml.Squeezing(y, 0., wires=[0])
            return qml.expval(qml.X(0))

        dev = qml.device('default.gaussian', wires=1)
        circuit = to_autograd(CVQNode(circuit, dev))
        grad_fn = autograd.grad(circuit)

        autograd_val = grad_fn(r)
        # qfunc evaluates to -exp(-r) * hbar * Re(alpha)
        manualgrad_val = -np.exp(-r) * hbar * alpha
        assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    @pytest.mark.parametrize('r', sqz_vals[1:])  # formula is not valid for r=0
    def test_number_state_gradient(self, r, tol):
        """Tests that the automatic gradient of a squeezed state with number state expectation is correct."""

        def circuit(y):
            qml.Squeezing(y, 0., wires=[0])
            return qml.expval(qml.FockStateProjector(np.array([2, 0]), wires=[0, 1]))

        dev = qml.device('default.gaussian', wires=2)
        circuit = to_autograd(CVQNode(circuit, dev))
        grad_fn = autograd.grad(circuit)

        # (d/dr) |<2|S(r)>|^2 = 0.5 tanh(r)^3 (2 csch(r)^2 - 1) sech(r)
        autograd_val = grad_fn(r)
        manualgrad_val = 0.5*np.tanh(r)**3 * (2/(np.sinh(r)**2)-1) / np.cosh(r)
        assert autograd_val == pytest.approx(manualgrad_val, abs=tol)



class TestAutogradJacobianQubit:
    """Tests involving Autograd functions grad and jacobian for qubit circuits."""

    @staticmethod
    def expected_jacobian(x, y, z):
        dw0dx = 2 / 3 * np.sin(x) * np.sin(y)
        dw0dy = 1 / 3 * (np.sin(y) - 2 * np.cos(x) * np.cos(y))
        dw0dz = 0

        dw1dx = -2 / 3 * np.cos(x) * np.sin(y)
        dw1dy = -2 / 3 * np.cos(y) * np.sin(x)
        dw1dz = 0

        return np.array([[dw0dx, dw0dy, dw0dz], [dw1dx, dw1dy, dw1dz]])

    def test_multiple_expectation_jacobian_positional(self, tol, qubit_device_2_wires):
        """Tests that qnodes using positional arguments return
        correct gradients for multiple expectation values."""
        par = [0.5, 0.54, 0.3]

        def circuit(x, y, z):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(x, y, z, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        circuit = to_autograd(QubitQNode(circuit, qubit_device_2_wires))

        # compare our manual Jacobian computation to theoretical result
        expected_jac = self.expected_jacobian(*par)
        res = circuit.jacobian(par)
        assert expected_jac == pytest.approx(res, abs=tol)

        # compare our manual Jacobian computation to autograd
        # not sure if this is the intended usage of jacobian
        jac0 = autograd.jacobian(circuit, 0)
        jac1 = autograd.jacobian(circuit, 1)
        jac2 = autograd.jacobian(circuit, 2)
        res = np.stack([jac0(*par), jac1(*par), jac2(*par)]).T

        assert expected_jac == pytest.approx(res, abs=tol)

        #compare with what we get if argnum is a list
        jac = autograd.jacobian(circuit, argnum=[0, 1, 2])
        #res2 = jac(*par)  # FIXME this call gives a TypeError inside Autograd
        #assert res == pytest.approx(res2, abs=tol)

    def test_multiple_expectation_jacobian_array(self, tol, qubit_device_2_wires):
        """Tests that qnodes using an array argument return correct gradients
        for multiple expectation values."""
        par = np.array([0.5, 0.54, 0.3])

        def circuit(weights):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(weights[0], weights[1], weights[2], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        circuit = to_autograd(QubitQNode(circuit, qubit_device_2_wires))

        expected_jac = self.expected_jacobian(*par)
        res = circuit.jacobian([par])
        assert expected_jac == pytest.approx(res, abs=tol)

        jac = autograd.jacobian(circuit, 0)
        res = jac(par)
        assert expected_jac == pytest.approx(res, abs=tol)


    def test_array_parameters_autograd(self, tol, qubit_device_2_wires):
        """Test that gradients of array parameters give
        same results as positional arguments."""

        par = [0.5, 0.54, 0.3]

        def ansatz(x, y, z):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(x, y, z, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        def circuit1(x, y, z):
            return ansatz(x, y, z)

        def circuit2(x, array):
            return ansatz(x, array[0], array[1])

        def circuit3(array):
            return ansatz(*array)

        circuit1 = to_autograd(QubitQNode(circuit1, qubit_device_2_wires))
        grad1 = autograd.grad(circuit1, argnum=[0, 1, 2])

        # three positional parameters
        jac = circuit1.jacobian(par)
        ag = grad1(*par)
        ag = np.array([ag])
        assert jac == pytest.approx(ag, abs=tol)

        circuit2 = to_autograd(QubitQNode(circuit2, qubit_device_2_wires))
        grad2 = autograd.grad(circuit2, argnum=[0, 1])

        # one scalar, one array
        temp = [par[0], np.array(par[1:])]
        jac = circuit2.jacobian(temp)
        ag = grad2(*temp)
        ag = np.r_[ag][np.newaxis, :]
        assert jac == pytest.approx(ag, abs=tol)

        circuit3 = to_autograd(QubitQNode(circuit3, qubit_device_2_wires))
        grad3 = autograd.grad(circuit3, argnum=0)

        # one array
        temp = [np.array(par)]
        jac = circuit3.jacobian(temp)
        ag = grad3(*temp)[np.newaxis, :]
        assert jac == pytest.approx(ag, abs=tol)


    def test_array_parameters_scalar_return(self, qubit_device_1_wire, tol):
        """Test that QNode can take arrays as input arguments, and that they interact properly with Autograd.
           Test case for a circuit that returns a scalar."""

        def circuit(dummy1, array, dummy2):
            qml.RY(0.5 * array[0, 1], wires=0)
            qml.RY(-0.5 * array[1, 1], wires=0)
            return qml.expval(qml.PauliX(0))

        node = to_autograd(QubitQNode(circuit, qubit_device_1_wire))

        args = (0.46, np.array([[2.0, 3.0, 0.3], [7.0, 4.0, 2.1]]), -0.13)
        grad_target = (
            np.array(1.0),
            np.array([[0.5, 0.43879, 0], [0, -0.43879, 0]]),
            np.array(-0.4),
        )
        cost_target = 1.03257

        def cost(x, array, y):
            c = node(0.111, array, 4.5)
            return c + 0.5 * array[0, 0] + x - 0.4 * y

        cost_grad = autograd.grad(cost, argnum=[0, 1, 2])
        computed_grad = cost_grad(*args)

        assert cost(*args) == pytest.approx(cost_target, abs=tol)

        assert computed_grad[0] == pytest.approx(grad_target[0], abs=tol)
        assert computed_grad[1] == pytest.approx(grad_target[1], abs=tol)
        assert computed_grad[2] == pytest.approx(grad_target[2], abs=tol)

    def test_qnode_array_parameters_1_vector_return(self, qubit_device_1_wire, tol):
        """Test that QNode can take arrays as input arguments, and that they interact properly with Autograd.
           Test case for a circuit that returns a 1-vector."""

        def circuit(dummy1, array, dummy2):
            qml.RY(0.5 * array[0, 1], wires=0)
            qml.RY(-0.5 * array[1, 1], wires=0)
            return (qml.expval(qml.PauliX(0)),)

        node = to_autograd(QubitQNode(circuit, qubit_device_1_wire))

        args = (0.46, np.array([[2.0, 3.0, 0.3], [7.0, 4.0, 2.1]]), -0.13)
        grad_target = (
            np.array(1.0),
            np.array([[0.5, 0.43879, 0], [0, -0.43879, 0]]),
            np.array(-0.4),
        )
        cost_target = 1.03257

        def cost(x, array, y):
            c = node(0.111, array, 4.5)[0]
            return c + 0.5 * array[0, 0] + x - 0.4 * y

        cost_grad = autograd.grad(cost, argnum=[0, 1, 2])
        computed_grad = cost_grad(*args)

        assert cost(*args) == pytest.approx(cost_target, abs=tol)

        assert computed_grad[0] == pytest.approx(grad_target[0], abs=tol)
        assert computed_grad[1] == pytest.approx(grad_target[1], abs=tol)
        assert computed_grad[2] == pytest.approx(grad_target[2], abs=tol)

    def test_qnode_array_parameters_2_vector_return(self, qubit_device_2_wires, tol):
        """Test that QNode can take arrays as input arguments, and that they interact properly with Autograd.
           Test case for a circuit that returns a 2-vector."""

        def circuit(dummy1, array, dummy2):
            qml.RY(0.5 * array[0, 1], wires=0)
            qml.RY(-0.5 * array[1, 1], wires=0)
            qml.RY(array[1, 0], wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        node = to_autograd(QubitQNode(circuit, qubit_device_2_wires))

        args = (0.46, np.array([[2.0, 3.0, 0.3], [7.0, 4.0, 2.1]]), -0.13)
        grad_target = (
            np.array(1.0),
            np.array([[0.5, 0.43879, 0], [0, -0.43879, 0]]),
            np.array(-0.4),
        )
        cost_target = 1.03257

        def cost(x, array, y):
            c = node(0.111, array, 4.5)[0]
            return c + 0.5 * array[0, 0] + x - 0.4 * y

        cost_grad = autograd.grad(cost, argnum=[0, 1, 2])
        computed_grad = cost_grad(*args)

        assert cost(*args) == pytest.approx(cost_target, abs=tol)

        assert computed_grad[0] == pytest.approx(grad_target[0], abs=tol)
        assert computed_grad[1] == pytest.approx(grad_target[1], abs=tol)
        assert computed_grad[2] == pytest.approx(grad_target[2], abs=tol)

    def test_qfunc_gradients(self, qubit_device_2_wires, tol):
        "Tests that the various ways of computing the gradient of a qfunc all agree."

        def circuit(x, y, z):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(-1.6, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[1, 0])
            qml.RX(z, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        qnode = to_autograd(QubitQNode(circuit, qubit_device_2_wires))
        params = np.array([0.1, -1.6, np.pi / 5])

        # manual gradients
        grad_fd1 = qnode.jacobian(params, method='F', options={'order': 1})
        grad_fd2 = qnode.jacobian(params, method='F', options={'order': 2})
        grad_angle = qnode.jacobian(params, method='A')

        # automatic gradient
        grad_fn = autograd.grad(qnode, argnum=[0, 1, 2])
        grad_auto = np.array([grad_fn(*params)])

        # gradients computed with different methods must agree
        assert grad_fd1 == pytest.approx(grad_fd2, abs=tol)
        assert grad_fd1 == pytest.approx(grad_angle, abs=tol)
        assert grad_fd1 == pytest.approx(grad_auto, abs=tol)

    def test_hybrid_gradients(self, qubit_device_2_wires, tol):
        "Tests that the various ways of computing the gradient of a hybrid computation all agree."

        # input data is the first parameter
        def classifier_circuit(in_data, x):
            qml.RX(in_data, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(-1.6, wires=[0])
            qml.RY(in_data, wires=[1])
            qml.CNOT(wires=[1, 0])
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        classifier = to_autograd(QubitQNode(classifier_circuit, qubit_device_2_wires))

        param = -0.1259
        in_data = np.array([-0.1, -0.88, np.exp(0.5)])
        out_data = np.array([1.5, np.pi / 3, 0.0])

        def error(p):
            "Total square error of classifier predictions."
            ret = 0
            for d_in, d_out in zip(in_data, out_data):
                square_diff = (classifier(d_in, p) - d_out) ** 2
                ret = ret + square_diff
            return ret

        def d_error(p, grad_method):
            "Gradient of error, computed manually."
            ret = 0
            for d_in, d_out in zip(in_data, out_data):
                args = (d_in, p)
                diff = (classifier(*args) - d_out)
                ret = ret + 2 * diff * classifier.jacobian(args, wrt=[1], method=grad_method)
            return ret

        y0 = error(param)
        grad = autograd.grad(error)
        grad_auto = grad(param)

        grad_fd1 = d_error(param, 'F')
        grad_angle = d_error(param, 'A')

        # gradients computed with different methods must agree
        assert grad_fd1 == pytest.approx(grad_angle, abs=tol)
        assert grad_fd1 == pytest.approx(grad_auto, abs=tol)
        assert grad_angle == pytest.approx(grad_auto, abs=tol)


    def test_hybrid_gradients_autograd_numpy(self, qubit_device_2_wires, tol):
        "Test the gradient of a hybrid computation requiring autograd.numpy functions."

        def circuit(x, y):
            "Quantum node."
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        quantum = to_autograd(QubitQNode(circuit, qubit_device_2_wires))

        def classical(p):
            "Classical node, requires autograd.numpy functions."
            return anp.exp(anp.sum(quantum(p[0], anp.log(p[1]))))

        def d_classical(a, b, method):
            "Gradient of classical computed symbolically, can use normal numpy functions."
            val = classical((a, b))
            J = quantum.jacobian((a, np.log(b)), method=method)
            return val * np.array([J[0, 0] + J[1, 0], (J[0, 1] + J[1, 1]) / b])

        param = np.array([-0.1259, 1.53])
        y0 = classical(param)
        grad_classical = autograd.jacobian(classical)
        grad_auto = grad_classical(param)

        grad_fd1 = d_classical(*param, 'F')
        grad_angle = d_classical(*param, 'A')

        # gradients computed with different methods must agree
        assert grad_fd1 == pytest.approx(grad_angle, abs=tol)
        assert grad_fd1 == pytest.approx(grad_auto, abs=tol)
        assert grad_angle == pytest.approx(grad_auto, abs=tol)
