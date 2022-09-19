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
import warnings

import pennylane as qml
from pennylane import numpy as np
from pennylane import QNode, qnode
from pennylane._qubit_device import (
    _check_adjoint_diffability,
    _get_trainable_params_wo_obs,
    _check_gates_adjoint_hessian,
)


class TestCheckAdjointDiffability:
    """Tests for the helper method _check_adjoint_diffability in _qubit_device.py"""

    def test_not_expval(self):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.var(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1, shots=1)

        with pytest.raises(qml.QuantumFunctionError, match="Adjoint differentiation method does"):
            _check_adjoint_diffability(tape, dev)

    def test_finite_shots_warns(self):
        """Tests warning raised when finite shots specified"""

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1, shots=1)

        with pytest.warns(
            UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
        ):
            _check_adjoint_diffability(tape, dev)


class TestCheckGatesAdjointHessian:
    """Tests for the helper method _check_gates_adjoint_hessian in _qubit_device.py"""

    @pytest.mark.parametrize("trainable_params", [(0,), (2,), (0, 1, 2)])
    def test_multipar_op(self, trainable_params):
        """Test if a QuantumFunctionError is raised for a multi-parameter operation"""

        with qml.tape.QuantumTape() as tape:
            qml.Rot(0.1, 0.2, 0.3, wires=0)
            qml.expval(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="Only single-parameter gates"):
            _check_gates_adjoint_hessian(tape, trainable_params)

    def test_generator_not_hamiltonian(self):
        """Test if a QuantumFunctionError is raised for an operation that is not generated
        by a ``Hamiltonian``."""

        with qml.tape.QuantumTape() as tape:
            qml.PhaseShift(0.2, 0)
            qml.expval(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="Only gates that are generated"):
            _check_gates_adjoint_hessian(tape, (0,))

    def test_generator_multi_term_hamiltonian(self):
        """Test if a QuantumFunctionError is raised for an operation that is generated
        by a ``Hamiltonian`` but the Hamiltonian has multiple terms."""

        with qml.tape.QuantumTape() as tape:
            qml.IsingXY(0.2, [0, 1])
            qml.expval(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="Only gates that are generated"):
            _check_gates_adjoint_hessian(tape, (0,))

    def test_generator_does_not_square_to_id(self):
        """Test if a QuantumFunctionError is raised for an operation that is generated
        by a ``Hamiltonian`` but the operator in the Hamiltonian does not square to the identity."""

        with qml.tape.QuantumTape() as tape:
            qml.CRY(0.2, [0, 1])
            qml.expval(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="Only gates that are generated"):
            _check_gates_adjoint_hessian(tape, (0,))


class TestGetTrainableParamsWoObs:
    """Tests for the helper method _get_trainable_params_wo_obs in _qubit_device.py"""

    def test_trainable_params(self, tol):
        """Test that getting the trainable parameters of a tape that obtains the
        expectation value of a Hermitian operator emits a warning if the
        parameters to Hermitian are trainable, and skips the parameters in the output."""
        dev = qml.device("default.qubit", wires=3)

        with qml.tape.QuantumTape() as tape:
            qml.CRot(0.5, 2.1, -0.4, wires=[1, 0])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {0, 2}
        trainable = _get_trainable_params_wo_obs(tape)
        assert trainable == [0, 2]

    def test_trainable_hermitian_warns(self, tol):
        """Test that getting the trainable parameters of a tape that obtains the
        expectation value of a Hermitian operator emits a warning if the
        parameters to Hermitian are trainable, and skips the parameters in the output."""
        dev = qml.device("default.qubit", wires=3)

        mx = qml.matrix(qml.PauliX(0) @ qml.PauliY(2))
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.5, 0)
            qml.expval(qml.Hermitian(mx, wires=[0, 2]))

        tape.trainable_params = {0, 1}
        with pytest.warns(
            UserWarning, match="Differentiating with respect to the input parameters of Hermitian"
        ):
            trainable = _get_trainable_params_wo_obs(tape)
        assert trainable == [0]

    def test_nontrainable_hermitian_does_not_warn(self, tol):
        """Test that getting the trainable parameters of a tape that obtains the
        expectation value of a Hermitian operator does not emit a warning if the
        parameters to the Hermitian are not trainable, but skips the parameters in the output."""
        dev = qml.device("default.qubit", wires=3)

        mx = qml.matrix(qml.PauliX(0) @ qml.PauliY(2))
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(mx, wires=[0, 2]))

        tape.trainable_params = {}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            trainable = _get_trainable_params_wo_obs(tape)
        assert trainable == []


class TestAdjointJacobian:
    """Tests for the adjoint_jacobian method"""

    @pytest.fixture
    def dev(self):
        return qml.device("default.qubit", wires=2)

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

        from gate_data import Rotx as Rx, Roty as Ry, Rotz as Rz

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


class TestAdjointHessianDiag:
    """Tests for the adjoint_hessian_diagonal method"""

    @pytest.fixture
    def dev(self):
        return qml.device("default.qubit", wires=2)

    @pytest.mark.autograd
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("U", [qml.RX, qml.RY, qml.RZ, qml.IsingXX])
    def test_rotation_hessian(self, U, theta, tol, dev):
        """Tests that the automatic gradients of Pauli rotations are correct."""
        np.random.seed(214)
        init_state = np.random.random(4)
        init_state /= np.linalg.norm(init_state)

        with qml.tape.QuantumTape() as tape:
            # qml.QubitStateVector(init_state, wires=[0, 1])
            U(theta, wires=list(range(U.num_wires)))
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {0}

        calculated_val = dev.adjoint_hessian_diagonal(tape)

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape, n=2, h=1e-4)
        numeric_val = fn(qml.execute(tapes, dev, None))
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)

    def test_ry_gradient(self, tol, dev):
        """Test that the second derivative of the RY gate matches the exact analytic formula."""

        par = 0.23

        with qml.tape.QuantumTape() as tape:
            qml.RY(par, wires=[0])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {0}

        # gradients
        exact = -np.sin(par)
        tapes, fn = qml.gradients.finite_diff(tape, n=2, h=1e-4)
        hess_F = fn(qml.execute(tapes, dev, None))
        hess_A = dev.adjoint_hessian_diagonal(tape)

        # different methods must agree
        assert np.allclose(hess_F, exact, atol=tol, rtol=0)
        assert np.allclose(hess_A, exact, atol=tol, rtol=0)

    def test_rx_gradient(self, tol, dev):
        """Test that the secon derivative of the RX gate matches the known formula."""
        a = 0.7418

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.expval(qml.PauliZ(0))

        # circuit jacobians
        dev_hessian = dev.adjoint_hessian_diagonal(tape)
        expected_hessian = -np.cos(a)
        assert np.allclose(dev_hessian, expected_hessian, atol=tol, rtol=0)

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
        dev_hessian = dev.adjoint_hessian_diagonal(tape)
        expected_hessian = -np.diag(np.cos(params))
        assert np.allclose(dev_hessian, expected_hessian, atol=tol, rtol=0)

    @pytest.mark.autograd
    @pytest.mark.parametrize("obs", [qml.PauliY])
    @pytest.mark.parametrize(
        "op",
        [
            qml.RX(0.4, wires=0),
            qml.IsingZZ(1.0, wires=[0, 1]),
            qml.PauliRot(0.51, "YX", wires=[1, 0]),
        ],
    )
    def test_hessians(self, op, obs, tol, dev):
        """Tests that the hessians of circuits match between the
        finite difference and device methods."""

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

        tape.trainable_params = {1}

        hess_F = (lambda t, fn: fn(qml.execute(t, dev, None)))(
            *qml.gradients.finite_diff(tape, n=2, h=1e-4)
        )
        hess_D = dev.adjoint_hessian_diagonal(tape)

        assert np.allclose(hess_D, hess_F, atol=tol, rtol=0)

    def test_use_device_state(self, tol, dev):
        """Tests that when using the device state, the correct answer is still returned."""

        z = -0.8

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.RZ(z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1}

        hess_1 = dev.adjoint_hessian_diagonal(tape)

        qml.execute([tape], dev, None)
        hess_2 = dev.adjoint_hessian_diagonal(tape, use_device_state=True)

        assert np.allclose(hess_1, hess_2, atol=tol, rtol=0)

    def test_provide_starting_state(self, tol, dev):
        """Tests provides correct answer when provided starting state."""
        z = -0.8

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.PauliRot(z, "ZY", wires=[0, 1])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1}

        hess_1 = dev.adjoint_hessian_diagonal(tape)

        qml.execute([tape], dev, None)
        hess_2 = dev.adjoint_hessian_diagonal(tape, starting_state=dev._pre_rotated_state)

        assert np.allclose(hess_1, hess_2, atol=tol, rtol=0)

    def test_hessian_of_tape_with_hermitian(self, tol):
        """Test that computing the hessian of a tape that obtains the
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
        res = dev.adjoint_hessian_diagonal(tape)

        expected = [
            -np.sin(a) * np.sin(b) * np.sin(c),
            -np.sin(b) * np.sin(a) * np.sin(c),
            -np.sin(c) * np.sin(b) * np.sin(a),
        ]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_large_circuit(self, tol):
        """Test the hessian diagonal is correct for a more complex circuit."""
        dev = qml.device("default.qubit", wires=5)

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(0)
            qml.RZ(-0.4, 0)
            qml.IsingYY(1.24, [0, 1])
            qml.RX(0.3, 1)
            qml.Hadamard(3)
            qml.PauliRot(0.412, "XYXZ", wires=[0, 1, 2, 3])
            qml.CNOT([2, 1])
            qml.CNOT([1, 3])
            qml.IsingYY(-0.5123, [0, 2])
            qml.expval(qml.PauliX(0))
            qml.expval(
                qml.Hermitian(
                    qml.matrix(0.3 * qml.Hadamard(1) @ qml.Projector([1], wires=2)), wires=[1, 2]
                )
            )
            qml.expval(qml.PauliX(3))
            qml.expval(qml.PauliZ(4))

        tape.trainable_params = {0, 1, 2, 3, 4}

        hess_F = (lambda t, fn: fn(qml.execute(t, dev, None)))(
            *qml.gradients.finite_diff(tape, n=2, h=1e-4)
        )
        hess_D = dev.adjoint_hessian_diagonal(tape)

        assert np.allclose(hess_D, hess_F, atol=tol, rtol=0)
