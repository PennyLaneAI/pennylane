# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the RewindTape tape"""
from pennylane import numpy as np
import pytest

import pennylane as qml
from pennylane.tape import QNode, qnode, JacobianTape
from pennylane.tape.measure import expval
from pennylane.tape.tapes.rewind import operation_derivative, RewindTape


class TestRewindTapeRaises:
    """Tests for the possible raises thrown by RewindTape"""

    def test_not_expval(self):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""

        dev = qml.device("default.qubit", wires=1)

        with RewindTape() as tape:
            qml.RX(0.1, wires=0)
            qml.var(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="The var return type is not supported"):
            tape.jacobian(dev)

    def test_no_apply_operation(self):
        """Test if a QuantumFunctionError is raised when using a device without an _apply_method
        method"""
        dev = qml.device("default.gaussian", wires=1)

        with RewindTape() as tape:
            qml.RX(0.1, wires=0)
            qml.expval(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="The rewind gradient method is only"):
            tape.jacobian(dev)

    def test_no_apply_unitary(self):
        """Test if a QuantumFunctionError is raised when using a device without an _apply_unitary
        method"""
        dev = qml.device("default.gaussian", wires=1)
        dev._apply_operation = None

        with RewindTape() as tape:
            qml.RX(0.1, wires=0)
            qml.expval(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="The rewind gradient method is only"):
            tape.jacobian(dev)

    def test_no_returns_state(self):
        """Test if a QuantumFunctionError is raised when using a device without an _apply_unitary
        method"""
        dev = qml.device("default.gaussian", wires=1)
        dev._apply_operation = None
        dev._apply_unitary = None

        with RewindTape() as tape:
            qml.RX(0.1, wires=0)
            qml.expval(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="The rewind gradient method is only"):
            tape.jacobian(dev)


class TestRewindTapeJacobian:
    """Tests for the jacobian method of RewindTape"""

    @pytest.fixture
    def dev(self):
        return qml.device('default.qubit', wires=2)


    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ])
    def test_pauli_rotation_gradient(self, G, theta, tol, dev):
        """Tests that the automatic gradients of Pauli rotations are correct."""

        with RewindTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1}

        autograd_val = tape.jacobian(dev, method="analytic")

        # compare to finite differences
        numeric_val = tape.jacobian(dev, method="numeric")
        assert np.allclose(autograd_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    def test_Rot_gradient(self, theta, tol, dev):
        """Tests that the automatic gradient of a arbitrary Euler-angle-parameterized gate is correct."""
        params = np.array([theta, theta ** 3, np.sqrt(2) * theta])

        with RewindTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        autograd_val = tape.jacobian(dev, method="analytic")

        # compare to finite differences
        numeric_val = tape.jacobian(dev, method="numeric")
        assert np.allclose(autograd_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("par", [1, -2, 1.623, -0.051, 0])  # integers, floats, zero
    def test_ry_gradient(self, par, tol, dev):
        """Test that the gradient of the RY gate matches the exact analytic formula."""

        with RewindTape() as tape:
            qml.RY(par, wires=[0])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {0}

        # gradients
        exact = np.cos(par)
        grad_F = tape.jacobian(dev, method="numeric")
        grad_A = tape.jacobian(dev, method="analytic")

        # different methods must agree
        assert np.allclose(grad_F, exact, atol=tol, rtol=0)
        assert np.allclose(grad_A, exact, atol=tol, rtol=0)

    def test_rx_gradient(self, tol, dev):
        """Test that the gradient of the RX gate matches the known formula."""
        a = 0.7418

        with RewindTape() as tape:
            qml.RX(a, wires=0)
            qml.expval(qml.PauliZ(0))

        circuit_output = tape.execute(dev)
        expected_output = np.cos(a)
        assert np.allclose(circuit_output, expected_output, atol=tol, rtol=0)

        # circuit jacobians
        circuit_jacobian = tape.jacobian(dev, method="analytic")
        expected_jacobian = -np.sin(a)
        assert np.allclose(circuit_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_multiple_rx_gradient(self, tol):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result."""
        dev = qml.device("default.qubit", wires=3)
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        with RewindTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            for idx in range(3):
                qml.expval(qml.PauliZ(idx))

        circuit_output = tape.execute(dev)
        expected_output = np.cos(params)
        assert np.allclose(circuit_output, expected_output, atol=tol, rtol=0)

        # circuit jacobians
        circuit_jacobian = tape.jacobian(dev, method="analytic")
        expected_jacobian = -np.diag(np.sin(params))
        assert np.allclose(circuit_jacobian, expected_jacobian, atol=tol, rtol=0)

    qubit_ops = [getattr(qml, name) for name in qml.ops._qubit__ops__]
    analytic_qubit_ops = {cls for cls in qubit_ops if cls.grad_method == "A"}
    analytic_qubit_ops -= {
        qml.CRot,  # not supported for RewindTape
        qml.PauliRot,  # not supported in test
        qml.MultiRZ,  # not supported in test
        qml.U1,  # not supported on device
        qml.U2,  # not supported on device
        qml.U3,  # not supported on device
    }

    @pytest.mark.parametrize("obs", [qml.PauliX, qml.PauliY])
    @pytest.mark.parametrize("op", analytic_qubit_ops)
    def test_gradients(self, op, obs, mocker, tol, dev):
        """Tests that the gradients of circuits match between the
        finite difference and analytic methods."""
        args = np.linspace(0.2, 0.5, op.num_params)

        with RewindTape() as tape:
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            op(*args, wires=range(op.num_wires))

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.RY(0.5, wires=1).inv()
            qml.CNOT(wires=[0, 1])

            qml.expval(obs(wires=0))
            qml.expval(qml.PauliZ(wires=1))

        tape.execute(dev)

        tape.trainable_params = set(range(1, 1 + op.num_params))

        grad_F = tape.jacobian(dev, method="numeric")

        spy = mocker.spy(RewindTape, "_rewind_jacobian")
        grad_A = tape.jacobian(dev, method="analytic")
        spy.assert_called()
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters(self, tol, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        with RewindTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        grad_A = tape.jacobian(dev, method="analytic")
        grad_F = tape.jacobian(dev, method="numeric")

        # gradient has the correct shape and every element is nonzero
        assert grad_A.shape == (1, 3)
        assert np.count_nonzero(grad_A) == 3
        # the different methods agree
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)


# class TestRewindTapeIntegration:
#     """Integration tests for RewindTape"""
#
#     def test_interface_tf(self):
#         tf = pytest.importorskip("tensorflow")
#         from pennylane.tape.interfaces.tf import TFInterface
#
#         dev = qml.device("default.qubit", wires=1)
#
#         with JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.Rot(0.1, 0.2, 0.3, wires=[0])
#             qml.RY(-0.2, wires=[0])
#             qml.expval(qml.PauliZ(0))
#
#         # with tf.GradientTape() as tape_tf:
#         TFInterface.apply(tape)
#         # result = tape.execute(dev)
#
#
#         jac = tape.jacobian(dev)
#         print(jac)
#         print("F")





class TestQNodeIntegration:
    """Test QNode integration with the rewind method"""

    @pytest.fixture
    def dev(self):
        return qml.device('default.qubit', wires=2)

    def test_qnode(self, mocker, tol, dev):
        """Test that specifying diff_method allows the rewind method to be selected"""
        args = np.array([0.54, 0.1, 0.5], requires_grad=True)

        def circuit(x, y, z):
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            qml.Rot(x, y, z, wires=0)

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.RY(0.5, wires=1)
            qml.CNOT(wires=[0, 1])

            return expval(qml.PauliX(0) @ qml.PauliZ(1))

        qnode1 = QNode(circuit, dev, diff_method="rewind")
        spy = mocker.spy(RewindTape, "_rewind_jacobian")

        grad_fn = qml.grad(qnode1)
        grad_A = grad_fn(*args)

        spy.assert_called()
        assert isinstance(qnode1.qtape, RewindTape)

        qnode2 = QNode(circuit, dev, diff_method="finite-diff")
        grad_fn = qml.grad(qnode2)
        grad_F = grad_fn(*args)

        assert not isinstance(qnode2.qtape, RewindTape)
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    thetas = np.linspace(-2 * np.pi, 2 * np.pi, 8)

    @pytest.mark.parametrize("reused_p", thetas ** 3 / 19)
    @pytest.mark.parametrize("other_p", thetas ** 2 / 1)
    def test_fanout_multiple_params(self, reused_p, other_p, tol, mocker, dev):
        """Tests that the correct gradient is computed for qnodes which
        use the same parameter in multiple gates."""

        from gate_data import Rotx as Rx, Roty as Ry, Rotz as Rz

        def expZ(state):
            return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2

        extra_param = np.array(0.31, requires_grad=False)

        @qnode(dev, diff_method="rewind")
        def cost(p1, p2):
            qml.RX(extra_param, wires=[0])
            qml.RY(p1, wires=[0])
            qml.RZ(p2, wires=[0])
            qml.RX(p1, wires=[0])
            return expval(qml.PauliZ(0))

        zero_state = np.array([1.0, 0.0])
        cost(reused_p, other_p)

        tape = cost.qtape

        assert isinstance(tape, RewindTape)
        spy = mocker.spy(RewindTape, "_rewind_jacobian")

        # analytic gradient
        grad_fn = qml.grad(cost)
        grad_A = grad_fn(reused_p, other_p)

        spy.assert_called_once()

        # manual gradient
        grad_true0 = (
            expZ(
                Rx(reused_p) @ Rz(other_p) @ Ry(reused_p + np.pi / 2) @ Rx(extra_param) @ zero_state
            )
            - expZ(
                Rx(reused_p) @ Rz(other_p) @ Ry(reused_p - np.pi / 2) @ Rx(extra_param) @ zero_state
            )
        ) / 2
        grad_true1 = (
            expZ(
                Rx(reused_p + np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state
            )
            - expZ(
                Rx(reused_p - np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state
            )
        ) / 2
        expected = grad_true0 + grad_true1  # product rule

        assert np.allclose(grad_A[0], expected, atol=tol, rtol=0)

    def test_gradient_repeated_gate_parameters(self, mocker, tol, dev):
        """Tests that repeated use of a free parameter in a multi-parameter gate yields correct
        gradients."""
        params = np.array([0.8, 1.3], requires_grad=True)

        def circuit(params):
            qml.RX(np.array(np.pi / 4, requires_grad=False), wires=[0])
            qml.Rot(params[1], params[0], 2 * params[0], wires=[0])
            return expval(qml.PauliX(0))

        spy_numeric = mocker.spy(JacobianTape, "numeric_pd")
        spy_analytic = mocker.spy(RewindTape, "_rewind_jacobian")

        cost = QNode(circuit, dev, diff_method="finite-diff")
        grad_fn = qml.grad(cost)
        grad_F = grad_fn(params)

        spy_numeric.assert_called()
        spy_analytic.assert_not_called()

        cost = QNode(circuit, dev, diff_method="rewind")
        grad_fn = qml.grad(cost)
        grad_A = grad_fn(params)

        spy_analytic.assert_called_once()

        # the different methods agree
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)


class TestOperationDerivative:
    """Tests for operation_derivative function"""

    def test_no_generator_raise(self):
        """Tests if the function raises a ValueError if the input operation has no generator"""
        op = qml.Rot(0.1, 0.2, 0.3, wires=0)

        with pytest.raises(ValueError, match="Operation Rot does not have a generator"):
            operation_derivative(op)

    def test_multiparam_raise(self):
        """Test if the function raises a ValueError if the input operation is composed of multiple
        parameters"""
        class RotWithGen(qml.Rot):
            generator = [np.zeros((2, 2)), 1]

        op = RotWithGen(0.1, 0.2, 0.3, wires=0)

        with pytest.raises(ValueError, match="Operation RotWithGen is not written in terms of"):
            operation_derivative(op)

    def test_rx(self):
        """Test if the function correctly returns the derivative of RX"""
        p = 0.3
        op = qml.RX(p, wires=0)

        derivative = operation_derivative(op)

        expected_derivative = 0.5 * np.array([[-np.sin(p / 2), -1j * np.cos(p / 2)],[-1j * np.cos(p / 2), - np.sin(p / 2)]])

        assert np.allclose(derivative, expected_derivative)

        op.inv()
        derivative_inv = operation_derivative(op)
        expected_derivative_inv = 0.5 * np.array([[-np.sin(p / 2), 1j * np.cos(p / 2)],[1j * np.cos(p / 2), -np.sin(p / 2)]])

        assert not np.allclose(derivative, derivative_inv)
        assert np.allclose(derivative_inv, expected_derivative_inv)

    def test_phase(self):
        """Test if the function correctly returns the derivative of PhaseShift"""
        p = 0.3
        op = qml.PhaseShift(p, wires=0)

        derivative = operation_derivative(op)
        expected_derivative = np.array([[0, 0], [0, 1j * np.exp(1j * p)]])
        assert np.allclose(derivative, expected_derivative)

    def test_cry(self):
        """Test if the function correctly returns the derivative of CRY"""
        p = 0.3
        op = qml.CRY(p, wires=[0, 1])

        derivative = operation_derivative(op)
        expected_derivative = 0.5 * np.array([[0, 0, 0, 0], [0, 0, 0, 0],
                                        [0, 0, -np.sin(p / 2), - np.cos(p / 2)],
                                        [0, 0, np.cos(p / 2), - np.sin(p / 2)],
                                        ])
        assert np.allclose(derivative, expected_derivative)
