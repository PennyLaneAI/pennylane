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
"""Tests for the gradients.param_shift_hessian module."""

import pytest

import pennylane as qml
from pennylane import numpy as np


class TestParameterShiftHessian:
    """Test the general functionality of the param_shift_hessian method
    on the default interface (autograd)"""

    def test_single_two_term_gate(self):
        """Test that the correct hessian is calculated for a QNode with single RX operator
        and single expectation value output (0d -> 0d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = np.array(0.1, requires_grad=True)

        expected = qml.jacobian(qml.grad(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_single_two_term_gate_vector_output(self):
        """Test that the correct hessian is calculated for a QNode with single RY operator
        and probabilies as output (0d -> 1d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = np.array(0.1, requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_multiple_two_term_gates(self):
        """Test that the correct hessian is calculated for a QNode with two rotation operators
        and one expectation value output (1d -> 0d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        x = np.array([0.1, 0.2], requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_multiple_two_term_gates_vector_output(self):
        """Test that the correct hessian is calculated for a QNode with two rotation operators
        and probabilities output (1d -> 1d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=1)

        x = np.array([0.1, 0.2], requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_quantum_hessian_shape_vector_input_vector_output(self):
        """Test that the purely "quantum" hessian has the correct shape (1d -> 1d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(x[2], wires=1)
            qml.Rot(x[0], x[1], x[2], wires=1)
            return qml.probs(wires=0)

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)
        shape = (2, 6, 6)  # (num_output_vals, num_gate_args, num_gate_args)

        hessian = qml.gradients.param_shift_hessian(circuit, hybrid=False)(x)

        assert qml.math.shape(hessian) == shape

    def test_multiple_two_term_gates_reusing_parameters(self):
        """Test that the correct hessian is calculated when reusing parameters (1d -> 1d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(x[2], wires=1)
            qml.Rot(x[0], x[1], x[2], wires=1)
            return qml.probs(wires=0)

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_multiple_two_term_gates_classical_processing(self):
        """Test that the correct hessian is calculated when manipulating parameters (1d -> 1d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0] + x[1] + x[2], wires=0)
            qml.RY(x[1] - x[0] + 3 * x[2], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(x[2] / x[0] - x[1], wires=1)
            return qml.probs(wires=0)

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_multiple_two_term_gates_matrix_output(self):
        """Test that the correct hessian is calculated for higher dimensional QNode outputs
        (1d -> 2d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=0), qml.probs(wires=1)

        x = np.ones([2], requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_multiple_two_term_gates_matrix_input(self):
        """Test that the correct hessian is calculated for higher dimensional cl. jacobians
        (2d -> 2d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0, 0], wires=0)
            qml.RY(x[0, 1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x[0, 2], wires=0)
            qml.RY(x[0, 0], wires=0)
            return qml.probs(wires=0), qml.probs(wires=1)

        x = np.ones([1, 3], requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    @pytest.mark.slow
    def test_multiple_qnode_arguments_scalar(self):
        """Test that the correct Hessian is calculated with multiple QNode arguments (0D->1D)"""
        jax = pytest.importorskip("jax")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=1)
            qml.RY(y, wires=0)
            qml.RX(x, wires=0)
            return qml.probs(wires=[0, 1])

        x = np.array(0.1, requires_grad=True)
        y = np.array(0.5, requires_grad=True)
        z = np.array(0.3, requires_grad=True)

        expected = tuple(
            jax.jacobian(jax.jacobian(circuit, argnums=i), argnums=i)(
                jax.numpy.array(x), jax.numpy.array(y), jax.numpy.array(z)
            )
            for i in range(3)
        )
        circuit.interface = "autograd"
        hessian = qml.gradients.param_shift_hessian(circuit)(x, y, z)

        assert np.allclose(expected, hessian)

    @pytest.mark.slow
    def test_multiple_qnode_arguments_vector(self):
        """Test that the correct Hessian is calculated with multiple QNode arguments (1D->1D)"""
        jax = pytest.importorskip("jax")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(x, y, z):
            qml.RX(x[0], wires=1)
            qml.RY(y[0], wires=0)
            qml.RZ(z[0] + z[1], wires=1)
            qml.RY(y[1], wires=1)
            qml.RX(x[1], wires=0)
            return qml.probs(wires=[0, 1])

        x = np.array([0.1, 0.3], requires_grad=True)
        y = np.array([0.5, 0.7], requires_grad=True)
        z = np.array([0.3, 0.2], requires_grad=True)

        expected = tuple(
            jax.jacobian(jax.jacobian(circuit, argnums=i), argnums=i)(
                jax.numpy.array(x), jax.numpy.array(y), jax.numpy.array(z)
            )
            for i in range(3)
        )
        circuit.interface = "autograd"
        hessian = qml.gradients.param_shift_hessian(circuit)(x, y, z)

        assert np.allclose(expected, hessian)

    @pytest.mark.slow
    def test_multiple_qnode_arguments_matrix(self):
        """Test that the correct Hessian is calculated with multiple QNode arguments (2D->1D)"""
        jax = pytest.importorskip("jax")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(x, y, z):
            qml.RX(x[0, 0], wires=0)
            qml.RY(y[0, 0], wires=1)
            qml.RZ(z[0, 0] + z[1, 1], wires=1)
            qml.RY(y[1, 0], wires=0)
            qml.RX(x[1, 0], wires=1)
            return qml.probs(wires=[0, 1])

        x = np.array([[0.1, 0.3], [0.2, 0.4]], requires_grad=True)
        y = np.array([[0.5, 0.7], [0.2, 0.4]], requires_grad=True)
        z = np.array([[0.3, 0.2], [0.2, 0.4]], requires_grad=True)

        expected = tuple(
            jax.jacobian(jax.jacobian(circuit, argnums=i), argnums=i)(
                jax.numpy.array(x), jax.numpy.array(y), jax.numpy.array(z)
            )
            for i in range(3)
        )
        circuit.interface = "autograd"
        hessian = qml.gradients.param_shift_hessian(circuit)(x, y, z)

        assert np.allclose(expected, hessian)

    def test_multiple_qnode_arguments_mixed(self):
        """Test that the correct Hessian is calculated with multiple mixed-shape QNode arguments"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(z[0] + z[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(y[1, 0], wires=0)
            qml.RY(y[0, 1], wires=0)
            return qml.probs(wires=0), qml.probs(wires=1)

        x = np.array(0.1, requires_grad=True)
        y = np.array([[0.5, 0.6], [0.2, 0.1]], requires_grad=True)
        z = np.array([0.3, 0.4], requires_grad=True)

        expected = tuple(
            qml.jacobian(qml.jacobian(circuit, argnum=i), argnum=i)(x, y, z) for i in range(3)
        )
        hessian = qml.gradients.param_shift_hessian(circuit)(x, y, z)

        assert all(np.allclose(expected[i], hessian[i]) for i in range(3))

    def test_hessian_transform_is_differentiable(self):
        """Test that the 3rd derivate can be calculated via auto-differentiation (1d -> 1d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=3)
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.RY(x[0], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = np.array([0.1, 0.2], requires_grad=True)

        expected = qml.jacobian(qml.jacobian(qml.jacobian(circuit)))(x)
        derivative = qml.jacobian(qml.gradients.param_shift_hessian(circuit))(x)

        assert np.allclose(expected, derivative)

    # Some bounds on the efficiency (device executions) of the hessian for 2-term shift rules:
    # - < jacobian(jacobian())
    # - <= 2^d * (m+d-1)C(d)      see arXiv:2008.06517 p. 4
    # - <= 3^m                    see arXiv:2008.06517 p. 4
    # here d=2 is the derivative order, m is the number of variational parameters (w.r.t. gate args)

    def test_fewer_device_invocations_scalar_input(self):
        """Test that the hessian invokes less hardware executions than double differentiation
        (0d -> 0d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        x = np.array(0.1, requires_grad=True)

        with qml.Tracker(dev) as tracker:
            hessian = qml.gradients.param_shift_hessian(circuit)(x)
            hessian_qruns = tracker.totals["executions"]
            expected = qml.jacobian(qml.jacobian(circuit))(x)
            jacobian_qruns = tracker.totals["executions"] - hessian_qruns

        assert np.allclose(hessian, expected)
        assert hessian_qruns < jacobian_qruns
        assert hessian_qruns <= 2**2 * 1  # 1 = (1+2-1)C(2)
        assert hessian_qruns <= 3**1

    def test_fewer_device_invocations_vector_input(self):
        """Test that the hessian invokes less hardware executions than double differentiation
        (1d -> 0d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(x[1], wires=0)
            return qml.expval(qml.PauliZ(1))

        x = np.array([0.1, 0.2], requires_grad=True)

        with qml.Tracker(dev) as tracker:
            hessian = qml.gradients.param_shift_hessian(circuit)(x)
            hessian_qruns = tracker.totals["executions"]
            expected = qml.jacobian(qml.jacobian(circuit))(x)
            jacobian_qruns = tracker.totals["executions"] - hessian_qruns

        assert np.allclose(hessian, expected)
        assert hessian_qruns < jacobian_qruns
        assert hessian_qruns <= 2**2 * 3  # 3 = (2+2-1)C(2)
        assert hessian_qruns <= 3**2

    def test_fewer_device_invocations_vector_output(self):
        """Test that the hessian invokes less hardware executions than double differentiation
        (1d -> 1d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(x[1], wires=0)
            qml.RZ(x[2], wires=1)
            return qml.probs(wires=[0, 1])

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        with qml.Tracker(dev) as tracker:
            hessian = qml.gradients.param_shift_hessian(circuit)(x)
            hessian_qruns = tracker.totals["executions"]
            expected = qml.jacobian(qml.jacobian(circuit))(x)
            jacobian_qruns = tracker.totals["executions"] - hessian_qruns

        assert np.allclose(hessian, expected)
        assert hessian_qruns < jacobian_qruns
        assert hessian_qruns <= 2**2 * 6  # 6 = (3+2-1)C(2)
        assert hessian_qruns <= 3**3

    def test_error_unsupported_operation(self):
        """Test that the correct error is thrown for unsopperted operations"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CRZ(x[2], wires=[0, 1])
            return qml.probs(wires=1)

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        with pytest.raises(ValueError, match=r"The operation .+ is currently not supported"):
            qml.gradients.param_shift_hessian(circuit)(x)

    def test_error_unsupported_variance_measurement(self):
        """Test that the correct error is thrown for variance measurements"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CRZ(x[2], wires=[0, 1])
            return qml.var(qml.PauliZ(1))

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        with pytest.raises(
            ValueError,
            match="Computing the Hessian of circuits that return variances is currently not supported.",
        ):
            qml.gradients.param_shift_hessian(circuit)(x)

    def test_error_unsupported_state_measurement(self):
        """Test that the correct error is thrown for state measurements"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CRZ(x[2], wires=[0, 1])
            return qml.state()

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        with pytest.raises(
            ValueError,
            match="Computing the Hessian of circuits that return the state is not supported.",
        ):
            qml.gradients.param_shift_hessian(circuit)(x)

    def test_no_error_nondifferentiable_unsupported_operation(self):
        """Test that no error is thrown for operations that are not marked differentiable"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.CRZ(z, wires=[0, 1])
            return qml.probs(wires=1)

        x = np.array(0.1, requires_grad=True)
        y = np.array(0.2, requires_grad=True)
        z = np.array(0.3, requires_grad=False)

        qml.gradients.param_shift_hessian(circuit)(x, y, z)

    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch", "tensorflow"])
    def test_no_trainable_params_qnode(self, interface):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        if interface != "autograd":
            pytest.importorskip(interface)
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.warns(UserWarning, match="hessian of a QNode with no trainable parameters"):
            res = qml.gradients.param_shift_hessian(circuit)(weights)

        assert res == ()

    def test_no_trainable_params_tape(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        weights = [0.1, 0.2]
        with qml.tape.QuantumTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        # TODO: remove once #2155 is resolved
        tape.trainable_params = []

        with pytest.warns(UserWarning, match="hessian of a tape with no trainable parameters"):
            h_tapes, post_processing = qml.gradients.param_shift_hessian(tape)
        res = post_processing(qml.execute(h_tapes, dev, None))

        assert h_tapes == []
        assert res == ()

    def test_all_zero_diff_methods(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(params):
            qml.Rot(*params, wires=0)
            return qml.probs([2, 3])

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        result = qml.gradients.param_shift_hessian(circuit)(params)
        assert np.allclose(result, np.zeros((4, 3, 3)), atol=0, rtol=0)

        tapes, _ = qml.gradients.param_shift_hessian(circuit.tape)
        assert tapes == []

    def test_f0_argument(self):
        """Test that we can provide the results of a QNode to save on quantum invocations"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=1)

        x = np.array([0.1, 0.2], requires_grad=True)

        res = circuit(x)

        with qml.Tracker(dev) as tracker:
            hessian1 = qml.gradients.param_shift_hessian(circuit, f0=res)(x)
            qruns1 = tracker.totals["executions"]
            hessian2 = qml.gradients.param_shift_hessian(circuit)(x)
            qruns2 = tracker.totals["executions"] - qruns1

        assert np.allclose(hessian1, hessian2)
        assert qruns1 < qruns2


class TestInterfaces:
    """Test the param_shift_hessian method on different interfaces"""

    def test_hessian_transform_with_torch(self):
        """Test that the Hessian transform can be used with Torch (1d -> 1d)"""
        torch = pytest.importorskip("torch")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.RY(x[0], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x_np = np.array([0.1, 0.2], requires_grad=True)
        x_torch = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x_np)
        circuit.interface = "torch"
        hess = qml.gradients.param_shift_hessian(circuit)(x_torch)[0]

        assert np.allclose(expected, hess.detach())

    def test_hessian_transform_is_differentiable_torch(self):
        """Test that the 3rd derivate can be calculated via auto-differentiation in Torch
        (1d -> 1d)"""
        torch = pytest.importorskip("torch")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=3)
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.RY(x[0], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = np.array([0.1, 0.2], requires_grad=True)
        x_torch = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)

        expected = qml.jacobian(qml.jacobian(qml.jacobian(circuit)))(x)
        circuit.interface = "torch"
        jacobian_fn = torch.autograd.functional.jacobian
        torch_deriv = jacobian_fn(qml.gradients.param_shift_hessian(circuit), x_torch)[0]

        assert np.allclose(expected, torch_deriv)

    @pytest.mark.slow
    def test_hessian_transform_with_jax(self):
        """Test that the Hessian transform can be used with JAX (1d -> 1d)"""
        jax = pytest.importorskip("jax")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.RY(x[0], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x_np = np.array([0.1, 0.2], requires_grad=True)
        x_jax = jax.numpy.array([0.1, 0.2])

        expected = qml.jacobian(qml.jacobian(circuit))(x_np)
        circuit.interface = "jax"
        hess = qml.gradients.param_shift_hessian(circuit)(x_jax)

        assert np.allclose(expected, hess)

    @pytest.mark.slow
    def test_hessian_transform_is_differentiable_jax(self):
        """Test that the 3rd derivate can be calculated via auto-differentiation in JAX
        (1d -> 1d)"""
        jax = pytest.importorskip("jax")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="backprop", max_diff=3)
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.RY(x[0], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = np.array([0.1, 0.2], requires_grad=True)
        x_jax = jax.numpy.array([0.1, 0.2])

        expected = qml.jacobian(qml.jacobian(qml.jacobian(circuit)))(x)
        circuit.interface = "jax"
        jax_deriv = jax.jacobian(qml.gradients.param_shift_hessian(circuit))(x_jax)

        assert np.allclose(expected, jax_deriv)

    @pytest.mark.slow
    def test_hessian_transform_with_tensorflow(self):
        """Test that the Hessian transform can be used with TensorFlow (1d -> 1d)"""
        tf = pytest.importorskip("tensorflow")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.RY(x[0], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x_np = np.array([0.1, 0.2], requires_grad=True)
        x_tf = tf.Variable([0.1, 0.2], dtype=tf.float64)

        expected = qml.jacobian(qml.jacobian(circuit))(x_np)
        circuit.interface = "tf"
        with tf.GradientTape():
            hess = qml.gradients.param_shift_hessian(circuit)(x_tf)[0]

        assert np.allclose(expected, hess)

    @pytest.mark.slow
    def test_hessian_transform_is_differentiable_tensorflow(self):
        """Test that the 3rd derivate can be calculated via auto-differentiation in Tensorflow
        (1d -> 1d)"""
        tf = pytest.importorskip("tensorflow")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=3)
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.RY(x[0], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = np.array([0.1, 0.2], requires_grad=True)
        x_tf = tf.Variable([0.1, 0.2], dtype=tf.float64)

        expected = qml.jacobian(qml.jacobian(qml.jacobian(circuit)))(x)
        circuit.interface = "tf"
        with tf.GradientTape() as tf_tape:
            hessian = qml.gradients.param_shift_hessian(circuit)(x_tf)[0]
        tensorflow_deriv = tf_tape.jacobian(hessian, x_tf)

        assert np.allclose(expected, tensorflow_deriv)
