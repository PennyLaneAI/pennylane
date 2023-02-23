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


class TestAdjointHessianDiag:
    """Tests for the adjoint_hessian_diagonal method."""

    @pytest.fixture
    def dev(self):
        """Fixture that creates a two-qubit default qubit device."""
        return qml.device("default.qubit", wires=2)

    @pytest.mark.autograd
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("U", [qml.RX, qml.RY, qml.RZ, qml.IsingXX])
    def test_rotation_hessian(self, U, theta, tol, dev):
        """Tests that the Hessian of Pauli rotations is correct."""
        np.random.seed(214)
        init_state = np.random.random(4, requires_grad=False)
        init_state /= np.linalg.norm(init_state)

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(init_state, wires=[0, 1])
            U(theta, wires=list(range(U.num_wires)))
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1}

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
        """Test that the second derivative of the RX gate matches the known formula."""
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
            qml.adjoint(qml.RY)(0.5, wires=1)
            qml.CNOT(wires=[0, 1])

            qml.expval(obs(wires=0))
            qml.expval(qml.PauliZ(wires=1))

        tape.trainable_params = {1}

        tapes, fn = qml.gradients.finite_diff(tape, n=2, h=1e-4)
        hess_F = fn(qml.execute(tapes, dev, None))
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
        # pylint: disable=protected-access
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

        tapes, fn = qml.gradients.finite_diff(tape, n=2, h=1e-4)
        hess_F = fn(qml.execute(tapes, dev, None))
        hess_D = dev.adjoint_hessian_diagonal(tape)

        assert np.allclose(hess_D, hess_F, atol=tol, rtol=0)
