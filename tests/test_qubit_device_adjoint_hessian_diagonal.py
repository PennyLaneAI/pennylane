# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Tests for the ``adjoint_hessian_diagonal`` method of the :mod:`pennylane`
:class:`QubitDevice` class.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np

# from pennylane import QNode, qnode


class TestAdjointHessianDiag:
    """Tests for the adjoint_hessian_diagonal method."""

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

        with qml.tape.QuantumTape() as tape:
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
