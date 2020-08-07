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
Unit tests for the :mod:`pennylane.interface.torch` QNode interface.
"""

import pytest

import numpy as np

torch = pytest.importorskip("torch", minversion="1.1")
from torch.autograd import Variable

import pennylane as qml

from pennylane.utils import _flatten, unflatten
from pennylane.qnodes import QNode, QuantumFunctionError
from pennylane._device import DeviceError
from pennylane.interfaces.torch import to_torch, unflatten_torch

from gate_data import CNOT, Rotx, Roty, Rotz, I, Y, Z


def expZ(state):
    return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2


class TestTorchQNodeExceptions():
    """TorchQNode basic tests."""

    def test_qnode_fails_on_wrong_return_type(self, qubit_device_2_wires):
        """The qfunc must return only Expectations"""
        @qml.qnode(qubit_device_2_wires, interface='torch')
        def qf(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0)), 0.3

        with pytest.raises(QuantumFunctionError, match='must return either'):
            qf(torch.tensor(0.5))

    def test_qnode_fails_on_expval_not_returned(self, qubit_device_2_wires):
        """All expectation values in the qfunc must be returned"""

        @qml.qnode(qubit_device_2_wires, interface='torch')
        def qf(x):
            qml.RX(x, wires=[0])
            ex = qml.expval(qml.PauliZ(1))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(QuantumFunctionError, match='All measured observables'):
            qf(torch.tensor(0.5))

    def test_qnode_fails_on_wrong_expval_order(self, qubit_device_2_wires):
        """Expvals must be returned in the order they were created in"""

        @qml.qnode(qubit_device_2_wires, interface='torch')
        def qf(x):
            qml.RX(x, wires=[0])
            ex = qml.expval(qml.PauliZ(1))
            return qml.expval(qml.PauliZ(0)), ex

        with pytest.raises(QuantumFunctionError, match='All measured observables'):
            qf(torch.tensor(0.5))

    def test_qnode_fails_on_gates_after_measurements(self, qubit_device_2_wires):
        """Gates have to precede measurements"""

        @qml.qnode(qubit_device_2_wires, interface='torch')
        def qf(x):
            qml.RX(x, wires=[0])
            ev = qml.expval(qml.PauliZ(1))
            qml.RY(0.5, wires=[0])
            return ev

        with pytest.raises(QuantumFunctionError, match='gates must precede'):
            qf(torch.tensor(0.5))

    def test_qnode_fails_on_multiple_measurements_of_same_wire(self, qubit_device_2_wires):
        """A wire can only be measured once"""
        
        @qml.qnode(qubit_device_2_wires, interface='torch')
        def qf(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliX(0))

        with pytest.raises(QuantumFunctionError, match='can only be measured once'):
            qf(torch.tensor(0.5))

    def test_qnode_fails_on_qfunc_with_too_many_wires(self, qubit_device_2_wires):
        """The device must have sufficient wires for the qfunc"""

        @qml.qnode(qubit_device_2_wires, interface='torch')
        def qf(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 2])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(QuantumFunctionError, match='applied to invalid wire'):
            qf(torch.tensor(0.5))

    def test_qnode_fails_on_combination_of_cv_and_qbit_ops(self, qubit_device_1_wire):
        """CV and discrete operations must not be mixed"""
        
        @qml.qnode(qubit_device_1_wire, interface='torch')
        def qf(x):
            qml.RX(x, wires=[0])
            qml.Displacement(0.5, 0, wires=[0])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(QuantumFunctionError, match='Continuous and discrete'):
            qf(torch.tensor(0.5))

    def test_qnode_fails_for_cv_ops_on_qubit_device(self, qubit_device_1_wire):
        """A qubit device cannot execute CV operations"""

        @qml.qnode(qubit_device_1_wire, interface='torch')
        def qf(x):
            qml.Displacement(0.5, 0, wires=[0])
            return qml.expval(qml.X(0))

        with pytest.raises(QuantumFunctionError, match='Device default.qubit is a qubit device; CV operations are not allowed.'):
            qf(torch.tensor(0.5))

    def test_qnode_fails_for_cv_observables_on_qubit_device(self, qubit_device_1_wire):
        """A qubit device cannot measure CV observables"""

        @qml.qnode(qubit_device_1_wire, interface='torch')
        def qf(x):
            return qml.expval(qml.X(0))

        with pytest.raises(QuantumFunctionError, match='Device default.qubit is a qubit device; CV operations are not allowed.'):
            qf(torch.tensor(0.5))

    def test_qnode_fails_for_return_state_with_bad_version(self, qubit_device_1_wire, monkeypatch):

        @qml.qnode(qubit_device_1_wire, interface='torch')
        def qf(x):
            return qml.state()

        with monkeypatch.context() as m:
            m.setattr("pennylane.interfaces.torch.MIN_VERSION_FOR_STATE", False)
            with pytest.raises(ImportError, match="Version 1.6.0 or above of PyTorch"):
                qf(torch.tensor(0.5))


class TestTorchQNodeParameterHandling:
    """Test that the TorchQNode properly handles the parameters of qfuncs"""

    def test_qnode_fanout(self, qubit_device_1_wire, tol):
        """Tests that qnodes can compute the correct function when the same parameter is used in multiple gates."""

        @qml.qnode(qubit_device_1_wire, interface='torch')
        def circuit(reused_param, other_param):
            qml.RX(reused_param, wires=[0])
            qml.RZ(other_param, wires=[0])
            qml.RX(reused_param, wires=[0])
            return qml.expval(qml.PauliZ(0))

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

                assert np.allclose(y_eval, y_true, atol=tol, rtol=0)

    def test_qnode_array_parameters_scalar_return(self, qubit_device_1_wire, tol):
        """Test that QNode can take arrays as input arguments, and that they interact properly with PyTorch.
           Test case for a circuit that returns a scalar."""

        # The objective of this test is not to check if the results are correctly calculated,
        # but to check that the interoperability of the different return types works.
        @qml.qnode(qubit_device_1_wire, interface='torch')
        def circuit(dummy1, array, dummy2):
            qml.RY(0.5 * array[0,1], wires=0)
            qml.RY(-0.5 * array[1,1], wires=0)
            return qml.expval(qml.PauliX(0))  # returns a scalar

        grad_target = (np.array(1.), np.array([[0.5,  0.43879, 0], [0, -0.43879, 0]]), np.array(-0.4))
        cost_target = 1.03257

        args = (torch.tensor(0.46), torch.tensor([[2., 3., 0.3], [7., 4., 2.1]]), torch.tensor(-0.13))
        for i in args:
            i.requires_grad = True

        def cost(x, array, y):
            c = torch.as_tensor(circuit(torch.tensor(0.111), array, torch.tensor(4.5)), dtype=torch.float32)
            return c +0.5*array[0,0] +x -0.4*y

        cost_res = cost(*args)
        cost_res.backward()

        assert np.allclose(cost_res.detach().numpy(), cost_target, atol=tol, rtol=0)

        for i in range(3):
            assert np.allclose(args[i].grad.detach().numpy(), grad_target[i], atol=tol, rtol=0)

    def test_qnode_array_parameters_1_vector_return(self, qubit_device_1_wire, tol):
        """Test that QNode can take arrays as input arguments, and that they interact properly with PyTorch.
           Test case for a circuit that returns a 1-vector."""

        # The objective of this test is not to check if the results are correctly calculated, 
        # but to check that the interoperability of the different return types works.
        @qml.qnode(qubit_device_1_wire, interface='torch')
        def circuit(dummy1, array, dummy2):
            qml.RY(0.5 * array[0,1], wires=0)
            qml.RY(-0.5 * array[1,1], wires=0)
            return qml.expval(qml.PauliX(0)),  # note the comma, returns a 1-vector

        grad_target = (np.array(1.), np.array([[0.5,  0.43879, 0], [0, -0.43879, 0]]), np.array(-0.4))
        cost_target = 1.03257

        args = (torch.tensor(0.46), torch.tensor([[2., 3., 0.3], [7., 4., 2.1]]), torch.tensor(-0.13))
        for i in args:
            i.requires_grad = True

        def cost(x, array, y):
            c = torch.as_tensor(circuit(torch.tensor(0.111), array, torch.tensor(4.5)), dtype=torch.float32)
            c = c[0]  # get a scalar
            return c +0.5*array[0,0] +x -0.4*y

        cost_res = cost(*args)
        cost_res.backward()

        assert np.allclose(cost_res.detach().numpy(), cost_target, atol=tol, rtol=0)

        for i in range(3):
            assert np.allclose(args[i].grad.detach().numpy(), grad_target[i], atol=tol, rtol=0)

    def test_qnode_array_parameters_2_vector_return(self, qubit_device_2_wires, tol):
        """Test that QNode can take arrays as input arguments, and that they interact properly with PyTorch.
           Test case for a circuit that returns a 2-vector."""

        # The objective of this test is not to check if the results are correctly calculated, 
        # but to check that the interoperability of the different return types works.
        @qml.qnode(qubit_device_2_wires, interface='torch')
        def circuit(dummy1, array, dummy2):
            qml.RY(0.5 * array[0,1], wires=0)
            qml.RY(-0.5 * array[1,1], wires=0)
            qml.RY(array[1,0], wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))  # returns a 2-vector

        grad_target = (np.array(1.), np.array([[0.5,  0.43879, 0], [0, -0.43879, 0]]), np.array(-0.4))
        cost_target = 1.03257

        args = (torch.tensor(0.46), torch.tensor([[2., 3., 0.3], [7., 4., 2.1]]), torch.tensor(-0.13))
        for i in args:
            i.requires_grad = True

        def cost(x, array, y):
            c = torch.as_tensor(circuit(torch.tensor(0.111), array, torch.tensor(4.5)), dtype=torch.float32)
            c = c[0]  # get a scalar
            return c +0.5*array[0,0] +x -0.4*y

        cost_res = cost(*args)
        cost_res.backward()

        assert np.allclose(cost_res.detach().numpy(), cost_target, atol=tol, rtol=0)

        for i in range(3):
            assert np.allclose(args[i].grad.detach().numpy(), grad_target[i], atol=tol, rtol=0)


    def test_array_parameters_evaluate(self, qubit_device_2_wires, tol):
        """Test that array parameters gives same result as positional arguments."""
        a, b, c = torch.tensor(0.5), torch.tensor(0.54), torch.tensor(0.3)

        def ansatz(x, y, z):
            qml.QubitStateVector(np.array([1, 0, 1, 1])/np.sqrt(3), wires=[0, 1])
            qml.Rot(x, y, z, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        @qml.qnode(qubit_device_2_wires, interface='torch')
        def circuit1(x, y, z):
            return ansatz(x, y, z)

        @qml.qnode(qubit_device_2_wires, interface='torch')
        def circuit2(x, array):
            return ansatz(x, array[0], array[1])

        @qml.qnode(qubit_device_2_wires, interface='torch')
        def circuit3(array):
            return ansatz(*array)

        positional_res = circuit1(a, b, c)
        array_res1 = circuit2(a, torch.tensor([b, c]))
        array_res2 = circuit3(torch.tensor([a, b, c]))

        assert np.allclose(positional_res.numpy(), array_res1.numpy(), atol=tol, rtol=0)
        assert np.allclose(positional_res.numpy(), array_res2.numpy(), atol=tol, rtol=0)

    def test_multiple_expectation_different_wires(self, qubit_device_2_wires, tol):
        """Tests that qnodes return multiple expectation values."""
        a, b, c = torch.tensor(0.5), torch.tensor(0.54), torch.tensor(0.3)

        @qml.qnode(qubit_device_2_wires, interface='torch')
        def circuit(x, y, z):
            qml.RX(x, wires=[0])
            qml.RZ(y, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=[0])
            qml.RX(z, wires=[0])
            return qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(1))

        res = circuit(a, b, c)

        out_state = np.kron(Rotx(c.numpy()), I) @ np.kron(Roty(b.numpy()), I) @ CNOT \
            @ np.kron(Rotz(b.numpy()), I) @ np.kron(Rotx(a.numpy()), I) @ np.array([1, 0, 0, 0])

        ex0 = np.vdot(out_state, np.kron(Y, I) @ out_state)
        ex1 = np.vdot(out_state, np.kron(I, Z) @ out_state)
        ex = np.array([ex0, ex1])

        assert np.allclose(ex, res.numpy(), atol=tol, rtol=0)

    def test_multiple_keywordargs_used(self, qubit_device_2_wires, tol):
        """Tests that qnodes use multiple keyword arguments."""

        @qml.qnode(qubit_device_2_wires, interface='torch')
        def circuit(w, x=None, y=None):
            qml.RX(x, wires=[0])
            qml.RX(y, wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        c = circuit(torch.tensor(1.), x=np.pi, y=np.pi)

        assert np.allclose(c.numpy(), [-1., -1.], atol=tol, rtol=0)

    def test_multidimensional_keywordargs_used(self, qubit_device_2_wires, tol):
        """Tests that qnodes use multi-dimensional keyword arguments."""
        def circuit(w, x=None):
            qml.RX(x[0], wires=[0])
            qml.RX(x[1], wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        circuit = qml.QNode(circuit, qubit_device_2_wires).to_torch()

        c = circuit(torch.tensor(1.), x=[np.pi, np.pi])
        assert np.allclose(c.numpy(), [-1., -1.], atol=tol, rtol=0)

    def test_keywordargs_for_wires(self, qubit_device_2_wires, tol):
        """Tests that wires can be passed as keyword arguments."""
        default_q = 0

        def circuit(x, q=default_q):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(q))

        circuit = qml.QNode(circuit, qubit_device_2_wires).to_torch()

        c = circuit(torch.tensor(np.pi), q=1)
        assert np.allclose(c, 1., atol=tol, rtol=0)

        c = circuit(torch.tensor(np.pi))
        assert np.allclose(c.numpy(), -1., atol=tol, rtol=0)

    def test_keywordargs_used(self, qubit_device_1_wire, tol):
        """Tests that qnodes use keyword arguments."""

        def circuit(w, x=None):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        circuit = qml.QNode(circuit, qubit_device_1_wire).to_torch()

        c = circuit(torch.tensor(1.), x=np.pi)
        assert np.allclose(c.numpy(), -1., atol=tol, rtol=0)

    def test_mixture_numpy_tensors(self, qubit_device_2_wires, tol):
        """Tests that qnodes work with python types and tensors."""

        @qml.qnode(qubit_device_2_wires, interface='torch')
        def circuit(w, x, y):
            qml.RX(x, wires=[0])
            qml.RX(y, wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        c = circuit(torch.tensor(1.), np.pi, np.pi).detach().numpy()
        assert np.allclose(c, [-1., -1.], atol=tol, rtol=0)

    def test_keywordarg_updated_in_multiple_calls(self, qubit_device_2_wires):
        """Tests that qnodes update keyword arguments in consecutive calls."""

        def circuit(w, x=None):
            qml.RX(w, wires=[0])
            qml.RX(x, wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        circuit = qml.QNode(circuit, qubit_device_2_wires).to_torch()

        c1 = circuit(torch.tensor(0.1), x=torch.tensor(0.))
        c2 = circuit(torch.tensor(0.1), x=np.pi)
        assert c1[1] != c2[1]

    def test_keywordarg_passes_through_classicalnode(self, qubit_device_2_wires, tol):
        """Tests that qnodes' keyword arguments pass through classical nodes."""

        def circuit(w, x=None):
            qml.RX(w, wires=[0])
            qml.RX(x, wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        circuit = qml.QNode(circuit, qubit_device_2_wires).to_torch()

        def classnode(w, x=None):
            return circuit(w, x=x)

        c = classnode(torch.tensor(0.), x=np.pi)
        assert np.allclose(c.numpy(), [1., -1.], atol=tol, rtol=0)

    def test_keywordarg_gradient(self, qubit_device_2_wires, tol):
        """Tests that qnodes' keyword arguments work with gradients"""

        def circuit(x, y, input_state=np.array([0, 0])):
            qml.BasisState(input_state, wires=[0, 1])
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[0])
            return qml.expval(qml.PauliZ(0))

        circuit = qml.QNode(circuit, qubit_device_2_wires).to_torch()

        x = 0.543
        y = 0.45632

        x_t = torch.autograd.Variable(torch.tensor(x), requires_grad=True)
        y_t = torch.autograd.Variable(torch.tensor(y), requires_grad=True)
        c = circuit(x_t, y_t, input_state=np.array([0, 0]))
        c.backward()
        assert np.allclose(x_t.grad.numpy(), [-np.sin(x)*np.cos(y)], atol=tol, rtol=0)
        assert np.allclose(y_t.grad.numpy(), [-np.sin(y)*np.cos(x)], atol=tol, rtol=0)

        x_t = torch.autograd.Variable(torch.tensor(x), requires_grad=True)
        y_t = torch.autograd.Variable(torch.tensor(y), requires_grad=True)
        c = circuit(x_t, y_t, input_state=np.array([1, 0]))
        c.backward()
        assert np.allclose(x_t.grad.numpy(), [np.sin(x)*np.cos(y)], atol=tol, rtol=0)
        assert np.allclose(y_t.grad.numpy(), [np.sin(y)*np.cos(x)], atol=tol, rtol=0)

        x_t = torch.autograd.Variable(torch.tensor(x), requires_grad=True)
        y_t = torch.autograd.Variable(torch.tensor(y), requires_grad=True)
        c = circuit(x_t, y_t)
        c.backward()
        assert np.allclose(x_t.grad.numpy(), [-np.sin(x)*np.cos(y)], atol=tol, rtol=0)
        assert np.allclose(y_t.grad.numpy(), [-np.sin(y)*np.cos(x)], atol=tol, rtol=0)


class TestIntegration():
    """Integration tests to ensure the Torch QNode agrees with the NumPy QNode"""

    def test_qnode_evaluation_agrees(self, qubit_device_2_wires, tol):
        """Tests that simple example is consistent."""

        @qml.qnode(qubit_device_2_wires, interface='autograd')
        def circuit(phi, theta):
            qml.RX(phi[0], wires=0)
            qml.RY(phi[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta[0], wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qubit_device_2_wires, interface='torch')
        def circuit_torch(phi, theta):
            qml.RX(phi[0], wires=0)
            qml.RY(phi[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta[0], wires=0)
            return qml.expval(qml.PauliZ(0))

        phi = [0.5, 0.1]
        theta = [0.2]

        phi_t = torch.tensor(phi)
        theta_t = torch.tensor(theta)

        autograd_eval = circuit(phi, theta)
        torch_eval = circuit_torch(phi_t, theta_t)
        assert np.allclose(autograd_eval, torch_eval.detach().numpy(), atol=tol, rtol=0)

    def test_qnode_gradient_agrees(self, qubit_device_2_wires, tol):
        """Tests that simple gradient example is consistent."""

        @qml.qnode(qubit_device_2_wires, interface='autograd')
        def circuit(phi, theta):
            qml.RX(phi[0], wires=0)
            qml.RY(phi[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta[0], wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qubit_device_2_wires, interface='torch')
        def circuit_torch(phi, theta):
            qml.RX(phi[0], wires=0)
            qml.RY(phi[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta[0], wires=0)
            return qml.expval(qml.PauliZ(0))

        phi = [0.5, 0.1]
        theta = [0.2]

        phi_t = torch.autograd.Variable(torch.tensor(phi), requires_grad=True)
        theta_t = torch.autograd.Variable(torch.tensor(theta), requires_grad=True)

        dcircuit = qml.grad(circuit, [0, 1])
        autograd_grad = dcircuit(phi, theta)

        torch_eval = circuit_torch(phi_t, theta_t)
        torch_eval.backward()

        assert np.allclose(autograd_grad[0], phi_t.grad.detach().numpy(), atol=tol, rtol=0)
        assert np.allclose(autograd_grad[1], theta_t.grad.detach().numpy(), atol=tol, rtol=0)


gradient_test_data = [
    (0.5, -0.1),
    (0.0, np.pi),
    (-3.6, -3.6),
    (1.0, 2.5),
]


class TestTorchGradients:
    """Integration tests involving gradients of QNodes and hybrid computations using the torch interface"""

    @pytest.fixture
    def qnodes(self):
        """Two QNodes to be used for the gradient tests"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def f(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev, interface="torch")
        def g(y):
            qml.RY(y, wires=0)
            return qml.expval(qml.PauliX(0))

        return f, g

    @pytest.mark.parametrize("x, y", gradient_test_data)
    def test_addition_qnodes_gradient(self, qnodes, x, y):
        """Test the gradient of addition of two QNode circuits"""
        f, g = qnodes

        def add(a, b):
            return a + b

        xt = torch.autograd.Variable(torch.tensor(x), requires_grad=True)
        yt = torch.autograd.Variable(torch.tensor(y), requires_grad=True)

        # addition
        a = f(xt)
        b = g(yt)
        a.retain_grad()
        b.retain_grad()

        add(a, b).backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

        # same tensor added to itself

        a = f(xt)
        a.retain_grad()

        add(a, a).backward()
        assert a.grad == 2.0

    @pytest.mark.parametrize("x, y", gradient_test_data)
    def test_subtraction_qnodes_gradient(self, qnodes, x, y):
        """Test the gradient of subtraction of two QNode circuits"""
        f, g = qnodes

        def subtract(a, b):
            return a - b

        xt = torch.autograd.Variable(torch.tensor(x), requires_grad=True)
        yt = torch.autograd.Variable(torch.tensor(y), requires_grad=True)

        # subtraction
        a = f(xt)
        b = g(yt)
        a.retain_grad()
        b.retain_grad()

        subtract(a, b).backward()
        assert a.grad == 1.0
        assert b.grad == -1.0

    @pytest.mark.parametrize("x, y", gradient_test_data)
    def test_multiplication_qnodes_gradient(self, qnodes, x, y):
        """Test the gradient of multiplication of two QNode circuits"""
        f, g = qnodes

        def mult(a, b):
            return a * b

        xt = torch.autograd.Variable(torch.tensor(x), requires_grad=True)
        yt = torch.autograd.Variable(torch.tensor(y), requires_grad=True)

        # multiplication
        a = f(xt)
        b = g(yt)
        a.retain_grad()
        b.retain_grad()

        mult(a, b).backward()
        assert a.grad == b
        assert b.grad == a

        a = f(xt)
        b = g(yt)
        a.retain_grad()
        b.retain_grad()

    @pytest.mark.parametrize("x, y", gradient_test_data)
    def test_division_qnodes_gradient(self, qnodes, x, y):
        """Test the gradient of division of two QNode circuits"""
        f, g = qnodes

        def div(a, b):
            return a / b

        xt = torch.autograd.Variable(torch.tensor(x), requires_grad=True)
        yt = torch.autograd.Variable(torch.tensor(y), requires_grad=True)

        # division
        # multiplication
        a = f(xt)
        b = g(yt)
        a.retain_grad()
        b.retain_grad()

        div(a, b).backward()
        assert a.grad == 1 / b
        assert b.grad == -a / b ** 2

    @pytest.mark.parametrize("x, y", gradient_test_data)
    def test_composition_qnodes_gradient(self, qnodes, x, y):
        """Test the gradient of composition of two QNode circuits"""
        f, g = qnodes

        def compose(f, x):
            return f(x)

        xt = torch.autograd.Variable(torch.tensor(x), requires_grad=True)
        yt = torch.autograd.Variable(torch.tensor(y), requires_grad=True)

        # compose function with xt as input
        compose(f, xt).backward()
        grad1 = xt.grad.detach().numpy()

        f(xt).backward()
        grad2 = xt.grad.detach().numpy()
        assert grad1 == grad2

        # compose function with a as input
        a = f(xt)
        a.retain_grad()

        compose(f, a).backward()
        grad1 = a.grad.detach().numpy()

        a = f(xt)
        a.retain_grad()

        f(a).backward()
        grad2 = a.grad.detach().numpy()
        assert grad1 == grad2

        # compose function with b as input
        b = g(yt)
        b.retain_grad()

        compose(f, b).backward()
        grad1 = b.grad.detach().numpy()

        b = g(yt)
        b.retain_grad()

        f(b).backward()
        grad2 = b.grad.detach().numpy()
        assert grad1 == grad2


class TestUnflatten:
    """Tests for pennylane.interfaces.torch.unflatten_torch"""

    flat = torch.tensor([i for i in range(12)])

    def test_unsupported_type_error(self):
        """Test that an unsupported type exception is raised if there is
        an unknown element in the model."""
        with pytest.raises(TypeError, match="Unsupported type in the model"):
            unflatten_torch(self.flat, [object()])

    def test_model_number(self):
        """Test that the function simply splits flat between its first and remaining elements
        when the model is a number"""
        unflattened = unflatten_torch(self.flat, 0)
        assert unflattened[0] == 0
        assert torch.all(unflattened[1] == self.flat[1:])

    def test_model_tensor(self):
        """Test that function correctly takes the first elements of flat and reshapes it into the
        model tensor, while leaving the remaining elements as a flat tensor"""
        model = torch.ones((3, 3))
        unflattened = unflatten_torch(self.flat, model)

        target = self.flat[:9].view((3, 3))
        remaining = self.flat[-3:]

        assert torch.all(unflattened[0] == target)
        assert torch.all(unflattened[1] == remaining)

    def test_model_iterable(self):
        """Test that the function correctly unflattens when the model is a list of numbers,
        which should result in unflatten_torch returning a list of tensors"""
        model = [1] * 12
        unflattened = unflatten_torch(self.flat, model)

        assert all([i.shape == () for i in unflattened[0]])
        assert unflattened[1].numel() == 0

    def test_model_nested_tensor(self):
        """Test that the function correctly unflattens when the model is a nested tensor,
        which should result in unflatten_torch returning a list of tensors of the same shape"""
        model = [torch.ones(3), torch.ones((2, 2)), torch.ones((3, 1)), torch.ones((1, 2))]
        unflattened = unflatten_torch(self.flat, model)

        assert all(
            [u.shape == model[i].shape for i, u in enumerate(unflattened[0])]
        )
        assert unflattened[1].numel() == 0


class TestParameterHandlingIntegration:
    """Test that the parameter handling for differentiable/non-differentiable
    parameters works correctly."""

    def test_differentiable_parameter_first(self):
        """Test that a differentiable parameter used as the first
        argument is correctly evaluated by QNode.jacobian, and that
        all other non-differentiable parameters are ignored"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(weights, data1, data2):
            # non-differentiable quantum function
            qml.templates.AmplitudeEmbedding(data1, wires=[0, 1])
            # differentiable quantum function
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            # non-differentiable quantum function
            qml.templates.AngleEmbedding(data2, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        # differentiating the circuit wrt the weights
        # input weights
        weights = qml.init.strong_ent_layers_normal(n_wires=2, n_layers=3)
        weights = torch.tensor(weights, requires_grad=True)

        # input data
        data1 = torch.tensor([0, 1, 1, 0], requires_grad=False) / np.sqrt(2)
        data2 = torch.tensor([1, 1], requires_grad=False)

        loss = circuit(weights, data1, data2)
        loss.backward()

        # check that weights is only once differentiable
        assert weights.grad.requires_grad == False

        res = weights.grad.detach().numpy()

        # we do not check for correctness, just that the output
        # is the correct shape
        assert res.shape == weights.shape

        # check that the first arg was marked as non-differentiable
        assert circuit.get_trainable_args() == {0}

        # Check that the gradient was not computed for the
        # non-differentiable elements of `data1` and `data2`.
        # First, extract the variable indices that the jacobian method
        # 'skipped' (those with grad_method="0"):
        non_diff_var_indices = sorted([k for k, v in circuit.par_to_grad_method.items() if v == "0"])

        # Check that these indices corresponds to the elements of data1 and data2
        # within the flattenened list [weights, data1, data2]
        assert non_diff_var_indices == [18, 19, 20, 21, 22, 23]

    def test_differentiable_parameter_middle(self):
        """Test that a differentiable parameter provided as the middle
        argument is correctly evaluated by QNode.jacobian, and that
        all other non-differentiable parameters are ignored"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(data1, weights, data2):
            # non-differentiable quantum function
            qml.templates.AmplitudeEmbedding(data1, wires=[0, 1])
            # differentiable quantum function
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            # non-differentiable quantum function
            qml.templates.AngleEmbedding(data2, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        # input weights
        weights = qml.init.strong_ent_layers_normal(n_wires=2, n_layers=3)
        weights = torch.tensor(weights, requires_grad=True)

        # input data
        data1 = torch.tensor([0, 1, 1, 0], requires_grad=False) / np.sqrt(2)
        data2 = torch.tensor([1, 1], requires_grad=False)

        loss = circuit(data1, weights, data2)
        loss.backward()
        res = weights.grad.detach().numpy()

        # we do not check for correctness, just that the output
        # is the correct shape
        assert res.shape == weights.shape

        # check that the second arg was marked as non-differentiable
        assert circuit.get_trainable_args() == {1}

        # Check that the gradient was not computed for the
        # non-differentiable elements of `data1` and `data2`.
        # First, extract the variable indices that the jacobian method
        # 'skipped' (those with grad_method="0"):
        non_diff_var_indices = sorted([k for k, v in circuit.par_to_grad_method.items() if v == "0"])

        # Check that these indices corresponds to the elements of data1 and data2
        # within the flattenened list [data1, weights, data2]
        assert non_diff_var_indices == [0, 1, 2, 3, 22, 23]

    def test_differentiable_parameter_last(self):
        """Test that a differentiable parameter used as the last
        argument is correctly evaluated by QNode.jacobian, and that
        all other non-differentiable parameters are ignored"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(data1, data2, weights):
            # non-differentiable quantum function
            qml.templates.AmplitudeEmbedding(data1, wires=[0, 1])
            # differentiable quantum function
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            # non-differentiable quantum function
            qml.templates.AngleEmbedding(data2, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        # input weights
        weights = qml.init.strong_ent_layers_normal(n_wires=2, n_layers=3)
        weights = torch.tensor(weights, requires_grad=True)

        # input data
        data1 = torch.tensor([0, 1, 1, 0], requires_grad=False) / np.sqrt(2)
        data2 = torch.tensor([1, 1], requires_grad=False)

        loss = circuit(data1, data2, weights)
        loss.backward()
        res = weights.grad.detach().numpy()

        # we do not check for correctness, just that the output
        # is the correct shape
        assert res.shape == weights.shape

        # check that the last arg was marked as non-differentiable
        assert circuit.get_trainable_args() == {2}

        # Check that the gradient was not computed for the
        # non-differentiable elements of `data1` and `data2`.
        # First, extract the variable indices that the jacobian method
        # 'skipped' (those with grad_method="0"):
        non_diff_var_indices = sorted([k for k, v in circuit.par_to_grad_method.items() if v == "0"])

        # Check that these indices corresponds to the elements of data1 and data2
        # within the flattenened list [data1, data2, weights]
        assert non_diff_var_indices == [0, 1, 2, 3, 4, 5]


    def test_multiple_differentiable_and_non_differentiable_parameters(self):
        """Test that multiple differentiable and non-differentiable parameters
        works as expected"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(data1, weights1, data2, weights2):
            qml.templates.AmplitudeEmbedding(data1, wires=[0, 1])
            qml.templates.StronglyEntanglingLayers(weights1, wires=[0, 1])
            qml.templates.AngleEmbedding(data2, wires=[0, 1])
            qml.templates.StronglyEntanglingLayers(weights2, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        # input weights
        weights1 = qml.init.strong_ent_layers_normal(n_wires=2, n_layers=3)
        weights2 = qml.init.strong_ent_layers_normal(n_wires=2, n_layers=4)

        weights1 = torch.tensor(weights1, requires_grad=True)
        weights2 = torch.tensor(weights2, requires_grad=True)

        # input data
        data1 = torch.tensor([0, 1, 1, 0], requires_grad=False) / np.sqrt(2)
        data2 = torch.tensor([1, 1], requires_grad=False)

        loss = circuit(data1, weights1, data2, weights2)
        loss.backward()
        res1 = weights1.grad.detach().numpy()
        res2 = weights2.grad.detach().numpy()

        # we do not check for correctness, just that the output
        # is the correct shape
        assert res1.shape == weights1.shape
        assert res2.shape == weights2.shape

        # check that the parameter shift was only performed for the
        # differentiable elements of `weights`, not the data input
        assert circuit.get_trainable_args() == {1, 3}

    def test_gradient_non_differentiable_exception(self):
        """Test that an exception is raised if non-differentiable data is
        differentiated"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(data1):
            qml.templates.AmplitudeEmbedding(data1, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        grad_fn = qml.grad(circuit, argnum=[0])
        data1 = torch.tensor([0, 1, 1, 0], requires_grad=False) / np.sqrt(2)

        loss = circuit(data1)
        assert circuit.get_trainable_args() == set()

        assert not loss.requires_grad

        with pytest.raises(RuntimeError, match="does not have a grad_fn"):
            loss.backward()

    def test_chained_qnodes(self):
        """Test that the gradient of chained QNodes works without error"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit1(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        @qml.qnode(dev, interface="torch")
        def circuit2(data, weights):
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        def cost(weights):
            w1, w2 = weights
            c1 = circuit1(w1)
            c2 = circuit2(c1, w2)
            return torch.sum(c2) ** 2

        w1 = qml.init.strong_ent_layers_normal(n_wires=2, n_layers=3)
        w2 = qml.init.strong_ent_layers_normal(n_wires=2, n_layers=4)

        w1 = torch.tensor(w1, requires_grad=True)
        w2 = torch.tensor(w2, requires_grad=True)

        weights = [w1, w2]

        loss = cost(weights)
        loss.backward()

        res = w1.grad.detach().numpy()
        assert res.shape == w1.shape

        res = w2.grad.detach().numpy()
        assert res.shape == w2.shape

    def test_gradient_value(self, tol):
        """Test that the returned gradient value for a qubit QNode is correct,
        when one of the arguments is non-differentiable."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="torch")
        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RX(b, wires=1)
            qml.RX(c, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliX(0) @ qml.PauliY(2))

        theta = torch.tensor(0.5, requires_grad=True)
        phi = torch.tensor(0.1, requires_grad=True)

        # varphi is non-differentiable
        varphi = torch.tensor(0.23)

        loss = circuit(theta, phi, varphi)
        loss.backward()

        res = [i.grad.detach().numpy() for i in [theta, phi]]
        expected = torch.tensor([
            torch.cos(theta) * torch.sin(phi) * torch.sin(varphi),
            torch.sin(theta) * torch.cos(phi) * torch.sin(varphi)
        ])

        assert np.allclose(res, expected, atol=tol, rtol=0)

        # check that the parameter-shift rule was not applied to varphi
        assert circuit.get_trainable_args() == {0, 1}

    def test_chained_gradient_value(self, mocker, tol):
        """Test that the returned gradient value for two chained qubit QNodes
        is correct."""
        spy = mocker.spy(qml.qnodes.JacobianQNode, "jacobian")
        dev1 = qml.device("default.qubit", wires=3)

        @qml.qnode(dev1, interface="torch")
        def circuit1(a, b, c):
            qml.RX(a, wires=0)
            qml.RX(b, wires=1)
            qml.RX(c, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(2))

        dev2 = qml.device("default.qubit", wires=2)

        @qml.qnode(dev2, interface="torch")
        def circuit2(data, weights):
            qml.RX(data[0], wires=0)
            qml.RX(data[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(weights[0], wires=0)
            qml.RZ(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(0) @ qml.PauliY(1))

        def cost(a, b, c, weights):
            return circuit2(circuit1(a, b, c), weights)

        # Set the first parameter of circuit1 as non-differentiable.
        a = torch.tensor(0.4, requires_grad=False)

        # the remaining free parameters are all differentiable
        b = torch.tensor(0.5, requires_grad=True)
        c = torch.tensor(0.1, requires_grad=True)
        weights = torch.tensor([0.2, 0.3], requires_grad=True)

        loss = cost(a, b, c, weights)
        loss.backward()
        res = [i.grad.detach().numpy() for i in [b, c, weights]]

        # Output should have shape [dcost/db, dcost/dc, dcost/dw],
        # where b,c are scalars, and w is a vector of length 2.
        assert len(res) == 3
        assert res[0].shape == tuple() # scalar
        assert res[1].shape == tuple() # scalar
        assert res[2].shape == (2,)    # vector

        cacbsc = torch.cos(a)*torch.cos(b)*torch.sin(c)

        expected = torch.tensor([
            # analytic expression for dcost/db
            -torch.cos(a)*torch.sin(b)*torch.sin(c)*torch.cos(cacbsc)*torch.sin(weights[0])*torch.sin(torch.cos(a)),
            # analytic expression for dcost/dc
            torch.cos(a)*torch.cos(b)*torch.cos(c)*torch.cos(cacbsc)*torch.sin(weights[0])*torch.sin(torch.cos(a)),
            # analytic expression for dcost/dw[0]
            torch.sin(cacbsc)*torch.cos(weights[0])*torch.sin(torch.cos(a)),
            # analytic expression for dcost/dw[1]
            0
        ])

        # np.hstack 'flattens' the ragged gradient array allowing it
        # to be compared with the expected result
        assert np.allclose(np.hstack(res), expected, atol=tol, rtol=0)

        # Check that the gradient was computed
        # for all parameters in circuit2
        assert circuit2.get_trainable_args() == {0, 1}

        # check that the gradient was not computed
        # for the first parameter of circuit1
        assert circuit1.get_trainable_args() == {1, 2}

    def test_non_diff_not_a_variable(self):
        """Test that an argument marked as non-differentiable
        is not wrapped as a variable."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="torch")
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.RZ(z, wires=0)

            assert isinstance(x, qml.variable.Variable)
            assert isinstance(y, float)
            assert isinstance(z, qml.variable.Variable)

            return qml.expval(qml.PauliZ(0))

        x = torch.tensor(1., requires_grad=True)
        y = torch.tensor(2., requires_grad=False)
        z = torch.tensor(3., requires_grad=True)

        res = circuit(x, y, z)

        assert circuit.get_trainable_args() == {0, 2}

        assert circuit.arg_vars[0] != x
        assert circuit.arg_vars[1] == y
        assert circuit.arg_vars[2] != z

    a = 0.6
    b = 0.2
    test_data = [
        ([0, 1], np.cos(2*a) * np.cos(b), [-2 * np.cos(b) * np.sin(2*a), -np.cos(2*a) * np.sin(b)]),
        ([1, 0], -np.cos(b) * np.sin(b), [0, -np.cos(b) ** 2 + np.sin(b) ** 2]),
    ]

    @pytest.mark.parametrize("w, expected_res, expected_grad", test_data)
    def test_non_diff_wires_argument(self, w, expected_res, expected_grad, tol):
        """Test that passing wires as a non-differentiable positional
        argument works correctly."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(wires, params):
            qml.Hadamard(wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RX(params[0], wires=wires[0])
            qml.RY(params[1], wires=wires[1])
            qml.CNOT(wires=[wires[1], wires[0]])
            qml.RX(params[0], wires=wires[0])
            qml.RY(params[1], wires=wires[1])
            return qml.expval(qml.PauliZ(0))

        params = torch.tensor([0.6, 0.2], requires_grad=True)
        wires = torch.tensor(w)

        res = circuit(wires, params)

        assert circuit.get_trainable_args() == {1}
        assert np.allclose(res.detach(), expected_res, atol=tol, rtol=0)

        res.backward()
        res_grad = params.grad

        assert circuit.get_trainable_args() == {1}
        assert np.allclose(res_grad.detach(), expected_grad, atol=tol, rtol=0)

    def test_call_changing_trainability(self):
        """Test that trainability properly changes between QNode calls"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.RZ(z, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = torch.tensor(1., requires_grad=True)
        y = torch.tensor(2., requires_grad=False)
        z = torch.tensor(3., requires_grad=True)

        res = circuit(x, y, z)

        assert circuit.get_trainable_args() == {0, 2}

        x.requires_grad = False
        y.requires_grad = True

        res = circuit(x, y, z)

        assert circuit.get_trainable_args() == {1, 2}

    def test_immutability(self):
        """Test that changing parameter differentiability raises an exception
        on immutable QNodes."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch", mutable=False)
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.RZ(z, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = torch.tensor(1., requires_grad=True)
        y = torch.tensor(2., requires_grad=False)
        z = torch.tensor(3., requires_grad=True)

        res = circuit(x, y, z)
        assert circuit.get_trainable_args() == {0, 2}

        # change values and compute the gradient again
        res = circuit(2*x, -y, z)
        assert circuit.get_trainable_args() == {0, 2}

        # attempting to change differentiability raises an error
        x.requires_grad = False
        y.requires_grad = True

        with pytest.raises(qml.QuantumFunctionError, match="cannot be modified"):
            circuit(x, y, z)


class TestConversion:
    """Integration tests to make sure that to_torch() correctly converts
    QNodes with/without pre-existing interfaces"""

    @pytest.fixture
    def qnode(self, interface, tf_support):
        """Returns a simple QNode corresponding to cos(x),
        with interface as determined by the interface fixture"""
        if interface == "tf" and not tf_support:
            pytest.skip("Skipped, no tf support")

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        return circuit

    @pytest.mark.parametrize("interface", ["torch"])
    def test_torch_conversion(self, qnode, tol):
        """Tests that the to_torch() function ignores QNodes that already
        have the torch interface."""
        converted_qnode = to_torch(qnode)
        assert converted_qnode is qnode

        x_val = 0.4
        x = torch.tensor(x_val, requires_grad=True)
        res = converted_qnode(x)
        res.backward()

        assert np.allclose(res.detach().numpy(), np.cos(x_val), atol=tol, rtol=0)
        assert np.allclose(x.grad, -np.sin(x_val), atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", [None, "autograd", "tf"])
    def test_other_conversion(self, qnode, tol):
        """Tests that the to_torch() function correctly converts both tf and autograd qnodes and
        QNodes with no interface."""
        converted_qnode = to_torch(qnode)
        assert converted_qnode is not qnode
        assert converted_qnode._qnode is getattr(qnode, "_qnode", qnode)

        x_val = 0.4
        x = torch.tensor(x_val, requires_grad=True)
        res = converted_qnode(x)
        res.backward()

        assert np.allclose(res.detach().numpy(), np.cos(x_val), atol=tol, rtol=0)
        assert np.allclose(x.grad, -np.sin(x_val), atol=tol, rtol=0)
