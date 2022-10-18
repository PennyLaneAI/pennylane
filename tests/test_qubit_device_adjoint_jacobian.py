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
"""
Tests for the ``adjoint_jacobian`` method of the :mod:`pennylane` :class:`QubitDevice` class.
"""
import pytest

import pennylane as qml
from pennylane import QNode
from pennylane import numpy as np
from pennylane import qnode


class TestAdjointJacobian:
    """Tests for the adjoint_jacobian method"""

    @pytest.fixture
    def dev(self):
        return qml.device("default.qubit", wires=2)

    def test_not_expval(self, dev):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.var(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="Adjoint differentiation method does"):
            dev.adjoint_jacobian(tape)

    def test_finite_shots_warns(self):
        """Tests warning raised when finite shots specified"""

        dev = qml.device("default.qubit", wires=1, shots=1)

        with qml.tape.QuantumTape(shots=1) as tape:
            qml.expval(qml.PauliZ(0))

        with pytest.warns(
            UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
        ):
            dev.adjoint_jacobian(tape)

    def test_unsupported_op(self, dev):
        """Test if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qml.Rot"""

        with qml.tape.QuantumTape() as tape:
            qml.CRot(0.1, 0.2, 0.3, wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="The CRot operation is not"):
            dev.adjoint_jacobian(tape)

    @pytest.mark.autograd
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ])
    def test_pauli_rotation_gradient(self, G, theta, tol, dev):
        """Tests that the automatic gradients of Pauli rotations are correct."""

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1}

        calculated_val = dev.adjoint_jacobian(tape)

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(qml.execute(tapes, dev, None))
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.autograd
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    def test_Rot_gradient(self, theta, tol, dev):
        """Tests that the device gradient of an arbitrary Euler-angle-parameterized gate is
        correct."""
        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        calculated_val = dev.adjoint_jacobian(tape)

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(qml.execute(tapes, dev, None))
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)

    def test_ry_gradient(self, tol, dev):
        """Test that the gradient of the RY gate matches the exact analytic formula."""

        par = 0.23

        with qml.tape.QuantumTape() as tape:
            qml.RY(par, wires=[0])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {0}

        # gradients
        exact = np.cos(par)
        tapes, fn = qml.gradients.finite_diff(tape)
        grad_F = fn(qml.execute(tapes, dev, None))
        grad_A = dev.adjoint_jacobian(tape)

        # different methods must agree
        assert np.allclose(grad_F, exact, atol=tol, rtol=0)
        assert np.allclose(grad_A, exact, atol=tol, rtol=0)

    def test_rx_gradient(self, tol, dev):
        """Test that the gradient of the RX gate matches the known formula."""
        a = 0.7418

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.expval(qml.PauliZ(0))

        # circuit jacobians
        dev_jacobian = dev.adjoint_jacobian(tape)
        expected_jacobian = -np.sin(a)
        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_multiple_rx_gradient(self, tol):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result."""
        dev = qml.device("default.qubit", wires=3)
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            for idx in range(3):
                qml.expval(qml.PauliZ(idx))

        # circuit jacobians
        dev_jacobian = dev.adjoint_jacobian(tape)
        expected_jacobian = -np.diag(np.sin(params))
        assert np.allclose(dev_jacobian, expected_jacobian, atol=tol, rtol=0)

    ops = {qml.RX, qml.RY, qml.RZ, qml.PhaseShift, qml.CRX, qml.CRY, qml.CRZ, qml.Rot}

    @pytest.mark.autograd
    @pytest.mark.parametrize("obs", [qml.PauliY])
    @pytest.mark.parametrize(
        "op", [qml.RX(0.4, wires=0), qml.CRZ(1.0, wires=[0, 1]), qml.Rot(0.2, -0.1, 0.2, wires=0)]
    )
    def test_gradients(self, op, obs, tol, dev):
        """Tests that the gradients of circuits match between the finite difference and device
        methods."""

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            qml.apply(op)

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.RY(0.5, wires=1).inv()
            qml.CNOT(wires=[0, 1])

            qml.expval(obs(wires=0))
            qml.expval(qml.PauliZ(wires=1))

        tape.trainable_params = set(range(1, 1 + op.num_params))

        grad_F = (lambda t, fn: fn(qml.execute(t, dev, None)))(*qml.gradients.finite_diff(tape))
        grad_D = dev.adjoint_jacobian(tape)

        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_gradient_gate_with_multiple_parameters(self, tol, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        grad_D = dev.adjoint_jacobian(tape)
        grad_F = (lambda t, fn: fn(qml.execute(t, dev, None)))(*qml.gradients.finite_diff(tape))

        # gradient has the correct shape and every element is nonzero
        assert grad_D.shape == (1, 3)
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_use_device_state(self, tol, dev):
        """Tests that when using the device state, the correct answer is still returned."""

        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        dM1 = dev.adjoint_jacobian(tape)

        qml.execute([tape], dev, None)
        dM2 = dev.adjoint_jacobian(tape, use_device_state=True)

        assert np.allclose(dM1, dM2, atol=tol, rtol=0)

    def test_provide_starting_state(self, tol, dev):
        """Tests provides correct answer when provided starting state."""
        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        dM1 = dev.adjoint_jacobian(tape)

        qml.execute([tape], dev, None)
        dM2 = dev.adjoint_jacobian(tape, starting_state=dev._pre_rotated_state)

        assert np.allclose(dM1, dM2, atol=tol, rtol=0)

    def test_gradient_of_tape_with_hermitian(self, tol):
        """Test that computing the gradient of a tape that obtains the
        expectation value of a Hermitian operator works correctly."""
        dev = qml.device("default.qubit", wires=3)

        a, b, c = [0.5, 0.3, -0.7]

        def ansatz(a, b, c):
            qml.RX(a, wires=0)
            qml.RX(b, wires=1)
            qml.RX(c, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])

        mx = qml.matrix(qml.PauliX(0) @ qml.PauliY(2))
        with qml.tape.QuantumTape() as tape:
            ansatz(a, b, c)
            qml.RX(a, wires=0)
            qml.expval(qml.Hermitian(mx, wires=[0, 2]))

        tape.trainable_params = {0, 1, 2}
        res = dev.adjoint_jacobian(tape)

        expected = [
            np.cos(a) * np.sin(b) * np.sin(c),
            np.cos(b) * np.sin(a) * np.sin(c),
            np.cos(c) * np.sin(b) * np.sin(a),
        ]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_trainable_hermitian_warns(self, tol):
        """Test attempting to compute the gradient of a tape that obtains the
        expectation value of a Hermitian operator emits a warning if the
        parameters to Hermitian are trainable."""
        dev = qml.device("default.qubit", wires=3)

        mx = qml.matrix(qml.PauliX(0) @ qml.PauliY(2))
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(mx, wires=[0, 2]))

        tape.trainable_params = {0}
        with pytest.warns(
            UserWarning, match="Differentiating with respect to the input parameters of Hermitian"
        ):
            res = dev.adjoint_jacobian(tape)


class TestAdjointJacobianQNode:
    """Test QNode integration with the adjoint_jacobian method"""

    @pytest.fixture
    def dev(self):
        return qml.device("default.qubit", wires=2)

    @pytest.mark.autograd
    def test_finite_shots_warning(self):
        """Tests that a warning is raised when computing the adjoint diff on a device with finite shots"""

        dev = qml.device("default.qubit", wires=1, shots=1)

        with pytest.warns(
            UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
        ):

            @qml.qnode(dev, diff_method="adjoint")
            def circ(x):
                qml.RX(x, wires=0)
                return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
        ):
            qml.grad(circ)(0.1)

    @pytest.mark.autograd
    def test_qnode(self, mocker, tol, dev):
        """Test that specifying diff_method allows the adjoint method to be selected"""
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

            return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

        qnode1 = QNode(circuit, dev, diff_method="adjoint")
        spy = mocker.spy(dev, "adjoint_jacobian")

        grad_fn = qml.grad(qnode1)
        grad_A = grad_fn(*args)

        spy.assert_called()

        qnode2 = QNode(circuit, dev, diff_method="finite-diff")
        grad_fn = qml.grad(qnode2)
        grad_F = grad_fn(*args)

        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    thetas = np.linspace(-2 * np.pi, 2 * np.pi, 8)

    @pytest.mark.autograd
    @pytest.mark.parametrize("reused_p", thetas**3 / 19)
    @pytest.mark.parametrize("other_p", thetas**2 / 1)
    def test_fanout_multiple_params(self, reused_p, other_p, tol, mocker, dev):
        """Tests that the correct gradient is computed for qnodes which
        use the same parameter in multiple gates."""

        from gate_data import Rotx as Rx
        from gate_data import Roty as Ry
        from gate_data import Rotz as Rz

        def expZ(state):
            return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2

        extra_param = np.array(0.31, requires_grad=False)

        @qnode(dev, diff_method="adjoint")
        def cost(p1, p2):
            qml.RX(extra_param, wires=[0])
            qml.RY(p1, wires=[0])
            qml.RZ(p2, wires=[0])
            qml.RX(p1, wires=[0])
            return qml.expval(qml.PauliZ(0))

        zero_state = np.array([1.0, 0.0])
        cost(reused_p, other_p)

        spy = mocker.spy(dev, "adjoint_jacobian")

        # analytic gradient
        grad_fn = qml.grad(cost)
        grad_D = grad_fn(reused_p, other_p)

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

        assert np.allclose(grad_D[0], expected, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_gradient_repeated_gate_parameters(self, mocker, tol, dev):
        """Tests that repeated use of a free parameter in a multi-parameter gate yields correct
        gradients."""
        params = np.array([0.8, 1.3], requires_grad=True)

        def circuit(params):
            qml.RX(np.array(np.pi / 4, requires_grad=False), wires=[0])
            qml.Rot(params[1], params[0], 2 * params[0], wires=[0])
            return qml.expval(qml.PauliX(0))

        spy_analytic = mocker.spy(dev, "adjoint_jacobian")

        cost = QNode(circuit, dev, diff_method="finite-diff")

        grad_fn = qml.grad(cost)
        grad_F = grad_fn(params)

        spy_analytic.assert_not_called()

        cost = QNode(circuit, dev, diff_method="adjoint")
        grad_fn = qml.grad(cost)
        grad_D = grad_fn(params)

        spy_analytic.assert_called_once()

        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_interface_tf(self, dev):
        """Test if gradients agree between the adjoint and finite-diff methods when using the
        TensorFlow interface"""
        import tensorflow as tf

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * tf.sqrt(params2), wires=[0])
            qml.RY(tf.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        params1 = tf.Variable(0.3, dtype=tf.float64)
        params2 = tf.Variable(0.4, dtype=tf.float64)

        qnode1 = QNode(f, dev, interface="tf", diff_method="adjoint")
        qnode2 = QNode(f, dev, interface="tf", diff_method="finite-diff")

        with tf.GradientTape() as tape:
            res1 = qnode1(params1, params2)

        g1 = tape.gradient(res1, [params1, params2])

        with tf.GradientTape() as tape:
            res2 = qnode2(params1, params2)

        g2 = tape.gradient(res2, [params1, params2])

        assert np.allclose(g1, g2)

    @pytest.mark.torch
    def test_interface_torch(self, dev):
        """Test if gradients agree between the adjoint and finite-diff methods when using the
        Torch interface"""
        import torch

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * torch.sqrt(params2), wires=[0])
            qml.RY(torch.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        params1 = torch.tensor(0.3, requires_grad=True)
        params2 = torch.tensor(0.4, requires_grad=True)

        qnode1 = QNode(f, dev, interface="torch", diff_method="adjoint")
        qnode2 = QNode(f, dev, interface="torch", diff_method="finite-diff")

        res1 = qnode1(params1, params2)
        res1.backward()

        grad_adjoint = params1.grad, params2.grad

        res2 = qnode2(params1, params2)
        res2.backward()

        grad_fd = params1.grad, params2.grad

        assert np.allclose(grad_adjoint, grad_fd)

    @pytest.mark.jax
    def test_interface_jax(self, dev):
        """Test if the gradients agree between adjoint and backprop methods in the
        jax interface"""
        import jax

        def f(params1, params2):
            qml.RX(0.4, wires=[0])
            qml.RZ(params1 * jax.numpy.sqrt(params2), wires=[0])
            qml.RY(jax.numpy.cos(params2), wires=[0])
            return qml.expval(qml.PauliZ(0))

        params1 = jax.numpy.array(0.3)
        params2 = jax.numpy.array(0.4)

        qnode_adjoint = QNode(f, dev, interface="jax", diff_method="adjoint")
        qnode_backprop = QNode(f, dev, interface="jax", diff_method="backprop")

        grad_adjoint = jax.grad(qnode_adjoint)(params1, params2)
        grad_backprop = jax.grad(qnode_backprop)(params1, params2)

        assert np.allclose(grad_adjoint, grad_backprop)

    def test_gradient_of_qnode_with_hermitian(self, tol):
        """Test that computing the gradient of a QNode that obtains the
        expectation value of a Hermitian operator works correctly."""
        dev = qml.device("default.qubit", wires=3)

        def ansatz(a, b, c):
            qml.RX(a, wires=0)
            qml.RX(b, wires=1)
            qml.RX(c, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])

        mx = qml.matrix(qml.PauliX(0) @ qml.PauliY(2))

        @qml.qnode(dev, diff_method="adjoint")
        def circ(a, b, c):
            ansatz(a, b, c)
            qml.RX(a, wires=0)
            return qml.expval(qml.Hermitian(mx, wires=[0, 2]))

        a = np.array(0.5, requires_grad=True)
        b = np.array(0.3, requires_grad=True)
        c = np.array(-0.7, requires_grad=True)

        res = qml.grad(circ)(a, b, c)

        expected = [
            np.cos(a) * np.sin(b) * np.sin(c),
            np.cos(b) * np.sin(a) * np.sin(c),
            np.cos(c) * np.sin(b) * np.sin(a),
        ]
        assert np.allclose(res, expected, atol=tol, rtol=0)
