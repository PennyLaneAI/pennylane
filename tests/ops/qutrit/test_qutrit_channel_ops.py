# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Unit tests for the available built-in qutrit quantum channels.
"""
import numpy as np
import pytest
from numpy.linalg import matrix_power

import pennylane as qml
from pennylane import math
from pennylane import numpy as pnp
from pennylane.ops.qutrit import channel

QUDIT_DIM = 3


class TestQutritDepolarizingChannel:
    """Tests for the qutrit quantum channel QutritDepolarizingChannel"""

    @staticmethod
    def get_expected_kraus_matrices(p):
        """Gets the expected Kraus matrices given probability p."""
        w = np.exp(2j * np.pi / 3)

        X = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        Z = np.diag([1, w, w**2])

        Ks = [np.sqrt(1 - p) * np.eye(QUDIT_DIM)]
        for i in range(3):
            for j in range(3):
                if i == 0 and j == 0:
                    continue
                Ks.append(np.sqrt(p / 8) * matrix_power(X, i) @ matrix_power(Z, j))
        return Ks

    def test_p_zero(self, tol):
        """Test p=0 gives correct Kraus matrices"""
        op = qml.QutritDepolarizingChannel
        kraus_matrices = op(0, wires=0).kraus_matrices()

        assert len(kraus_matrices) == 9
        assert np.allclose(kraus_matrices[0], np.eye(QUDIT_DIM), atol=tol, rtol=0)
        assert np.allclose(np.array(kraus_matrices[1:]), 0, atol=tol, rtol=0)

    def test_p_arbitrary(self, tol):
        """Test p=0.1 gives correct Kraus matrices"""
        p = 0.1
        kraus_matrices = qml.QutritDepolarizingChannel(p, wires=0).kraus_matrices()
        expected_matrices = self.get_expected_kraus_matrices(p)
        for kraus_matrix, expected_matrix in zip(kraus_matrices, expected_matrices):
            assert np.allclose(kraus_matrix, expected_matrix, atol=tol, rtol=0)

    def test_p_invalid_parameter(self):
        """Test that error is raised given an inappropriate p value."""
        with pytest.raises(ValueError, match="p must be in the interval"):
            qml.QutritDepolarizingChannel(1.5, wires=0).kraus_matrices()

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    def test_grad_depolarizing(self, angle):
        """Test that analytical gradient is computed correctly for different states. Channel
        grad recipes are independent of channel parameter"""

        dev = qml.device("default.qutrit.mixed")
        prob = pnp.array(0.5, requires_grad=True)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(p):
            qml.TRX(angle, wires=0, subspace=(0, 1))
            qml.TRX(angle, wires=0, subspace=(1, 2))
            qml.QutritDepolarizingChannel(p, wires=0)
            return qml.expval(qml.GellMann(0, 3) + qml.GellMann(0, 8))

        expected_errorless = (
            (np.sqrt(3) - 3) * (1 - np.cos(2 * angle)) / 24
            - 2 / np.sqrt(3) * np.sin(angle / 2) ** 4
            + (np.sqrt(1 / 3) + 1) * np.cos(angle / 2) ** 2
        )

        assert np.allclose(circuit(prob), ((prob - (1 / 9)) / (8 / 9)) * expected_errorless)

        gradient = np.squeeze(qml.grad(circuit)(prob))
        assert np.allclose(gradient, circuit(1) - circuit(0))
        assert np.allclose(gradient, -(9 / 8) * expected_errorless)

    @staticmethod
    def expected_jac_fn(p):
        """Gets the expected Jacobian of Kraus matrices given probability p."""
        w = np.exp(2j * np.pi / 3)

        X = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        Z = np.diag([1, w, w**2])

        jacs = [-1 / (2 * np.sqrt(1 - p)) * np.eye(QUDIT_DIM)]
        for i in range(3):
            for j in range(3):
                if i == 0 and j == 0:
                    continue
                jacs.append(np.sqrt(1 / (32 * p)) * matrix_power(X, i) @ matrix_power(Z, j))

        return jacs

    @staticmethod
    def kraus_fn(p):
        """Gets a matrix of the Kraus matrices to be tested."""
        return math.stack(channel.QutritDepolarizingChannel(p, wires=0).kraus_matrices())

    @staticmethod
    def kraus_fn_real(p):
        """Gets a matrix of the real part of the Kraus matrices to be tested."""
        return math.real(math.stack(channel.QutritDepolarizingChannel(p, wires=0).kraus_matrices()))

    @staticmethod
    def kraus_fn_imag(p):
        """Gets a matrix of the imaginary part of the Kraus matrices to be tested."""
        return math.imag(math.stack(channel.QutritDepolarizingChannel(p, wires=0).kraus_matrices()))

    @pytest.mark.autograd
    def test_kraus_jac_autograd(self):
        """Tests Jacobian of Kraus matrices using autograd."""
        p = pnp.array(0.43, requires_grad=True)
        jac = qml.jacobian(self.kraus_fn_real)(p) + 1j * qml.jacobian(self.kraus_fn_imag)(p)
        assert math.allclose(jac, self.expected_jac_fn(p))

    @pytest.mark.torch
    def test_kraus_jac_torch(self):
        """Tests Jacobian of Kraus matrices using PyTorch."""
        import torch

        p = torch.tensor(0.43, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian
        jac = jacobian(self.kraus_fn_real, p) + 1j * jacobian(self.kraus_fn_imag, p)
        assert math.allclose(jac, self.expected_jac_fn(p.detach().numpy()))

    @pytest.mark.tf
    def test_kraus_jac_tf(self):
        """Tests Jacobian of Kraus matrices using TensorFlow."""
        import tensorflow as tf

        p = tf.Variable(0.43)
        with tf.GradientTape() as real_tape:
            real_out = self.kraus_fn_real(p)
        with tf.GradientTape() as imag_tape:
            imag_out = self.kraus_fn_imag(p)

        real_jac = math.cast(real_tape.jacobian(real_out, p), complex)
        imag_jac = math.cast(imag_tape.jacobian(imag_out, p), complex)
        jac = real_jac + 1j * imag_jac
        assert math.allclose(jac, self.expected_jac_fn(0.43))

    @pytest.mark.jax
    def test_kraus_jac_jax(self):
        """Tests Jacobian of Kraus matrices using JAX."""
        import jax

        jax.config.update("jax_enable_x64", True)

        p = jax.numpy.array(0.43, dtype=jax.numpy.complex128)
        jac = jax.jacobian(self.kraus_fn, holomorphic=True)(p)
        assert math.allclose(jac, self.expected_jac_fn(p))


class TestQutritAmplitudeDamping:
    """Tests for the qutrit quantum channel QutritAmplitudeDamping"""

    def test_gamma_zero(self, tol):
        """Test gamma_10=gamma_20=0 gives correct Kraus matrices"""
        kraus_mats = qml.QutritAmplitudeDamping(0, 0, 0, wires=0).kraus_matrices()
        assert np.allclose(kraus_mats[0], np.eye(3), atol=tol, rtol=0)
        for kraus_mat in kraus_mats[1:]:
            assert np.allclose(kraus_mat, np.zeros((3, 3)), atol=tol, rtol=0)

    @pytest.mark.parametrize("gamma_10,gamma_20,gamma_21", ((0.1, 0.2, 0.3), (0.75, 0.75, 0.25)))
    def test_gamma_arbitrary(self, gamma_10, gamma_20, gamma_21, tol):
        """Test the correct Kraus matrices are returned."""
        K_0 = np.diag((1, np.sqrt(1 - gamma_10), np.sqrt(1 - gamma_20 - gamma_21)))

        K_1 = np.zeros((3, 3))
        K_1[0, 1] = np.sqrt(gamma_10)

        K_2 = np.zeros((3, 3))
        K_2[0, 2] = np.sqrt(gamma_20)

        K_3 = np.zeros((3, 3))
        K_3[1, 2] = np.sqrt(gamma_21)

        expected = [K_0, K_1, K_2, K_3]
        damping_channel = qml.QutritAmplitudeDamping(gamma_10, gamma_20, gamma_21, wires=0)
        assert np.allclose(damping_channel.kraus_matrices(), expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "gamma_10,gamma_20,gamma_21",
        (
            (1.5, 0.0, 0.0),
            (0.0, 1.0 + math.eps, 0.0),
            (0.0, 0.0, 1.1),
            (0.0, 0.33, 0.67 + math.eps),
        ),
    )
    def test_gamma_invalid_parameter(self, gamma_10, gamma_20, gamma_21):
        """Ensures that error is thrown when gamma_10, gamma_20, gamma_21, or (gamma_20 + gamma_21) are outside [0,1]"""
        with pytest.raises(ValueError, match="must be in the interval"):
            channel.QutritAmplitudeDamping(gamma_10, gamma_20, gamma_21, wires=0).kraus_matrices()

    @staticmethod
    def expected_jac_fn(gamma_10, gamma_20, gamma_21):
        """Gets the expected Jacobian of Kraus matrices"""
        partial_1 = [math.zeros((3, 3)) for _ in range(4)]
        partial_1[0][1, 1] = -1 / (2 * math.sqrt(1 - gamma_10))
        partial_1[1][0, 1] = 1 / (2 * math.sqrt(gamma_10))

        partial_2 = [math.zeros((3, 3)) for _ in range(4)]
        partial_2[0][2, 2] = -1 / (2 * math.sqrt(1 - gamma_20 - gamma_21))
        partial_2[2][0, 2] = 1 / (2 * math.sqrt(gamma_20))

        partial_3 = [math.zeros((3, 3)) for _ in range(4)]
        partial_3[0][2, 2] = -1 / (2 * math.sqrt(1 - gamma_20 - gamma_21))
        partial_3[3][1, 2] = 1 / (2 * math.sqrt(gamma_21))

        return [partial_1, partial_2, partial_3]

    @staticmethod
    def kraus_fn(gamma_10, gamma_20, gamma_21):
        """Gets the Kraus matrices of QutritAmplitudeDamping channel, used for differentiation."""
        damping_channel = qml.QutritAmplitudeDamping(gamma_10, gamma_20, gamma_21, wires=0)
        return math.stack(damping_channel.kraus_matrices())

    @pytest.mark.autograd
    def test_kraus_jac_autograd(self):
        """Tests Jacobian of Kraus matrices using autograd."""
        gamma_10 = pnp.array(0.43, requires_grad=True)
        gamma_20 = pnp.array(0.12, requires_grad=True)
        gamma_21 = pnp.array(0.35, requires_grad=True)

        jac = qml.jacobian(self.kraus_fn)(gamma_10, gamma_20, gamma_21)
        assert math.allclose(jac, self.expected_jac_fn(gamma_10, gamma_20, gamma_21))

    @pytest.mark.torch
    def test_kraus_jac_torch(self):
        """Tests Jacobian of Kraus matrices using PyTorch."""
        import torch

        gamma_10 = torch.tensor(0.43, requires_grad=True)
        gamma_20 = torch.tensor(0.12, requires_grad=True)
        gamma_21 = torch.tensor(0.35, requires_grad=True)

        jac = torch.autograd.functional.jacobian(self.kraus_fn, (gamma_10, gamma_20, gamma_21))
        expected = self.expected_jac_fn(
            gamma_10.detach().numpy(), gamma_20.detach().numpy(), gamma_21.detach().numpy()
        )

        for res_partial, exp_partial in zip(jac, expected):
            assert math.allclose(res_partial.detach().numpy(), exp_partial)

    @pytest.mark.tf
    def test_kraus_jac_tf(self):
        """Tests Jacobian of Kraus matrices using TensorFlow."""
        import tensorflow as tf

        gamma_10 = tf.Variable(0.43)
        gamma_20 = tf.Variable(0.12)
        gamma_21 = tf.Variable(0.35)

        with tf.GradientTape() as tape:
            out = self.kraus_fn(gamma_10, gamma_20, gamma_21)
        jac = tape.jacobian(out, (gamma_10, gamma_20, gamma_21))
        assert math.allclose(jac, self.expected_jac_fn(gamma_10, gamma_20, gamma_21))

    @pytest.mark.jax
    def test_kraus_jac_jax(self):
        """Tests Jacobian of Kraus matrices using JAX."""
        import jax

        gamma_10 = jax.numpy.array(0.43)
        gamma_20 = jax.numpy.array(0.12)
        gamma_21 = jax.numpy.array(0.35)

        jac = jax.jacobian(self.kraus_fn, argnums=[0, 1, 2])(gamma_10, gamma_20, gamma_21)
        assert math.allclose(jac, self.expected_jac_fn(gamma_10, gamma_20, gamma_21))


class TestTritFlip:
    """Tests for the quantum channel TritFlip"""

    @pytest.mark.parametrize(
        "ps", [(0, 0, 0), (0.1, 0.12, 0.3), (0.5, 0.4, 0.1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    )
    def test_ps_arbitrary(self, ps, tol):
        """Test that various values of p give correct Kraus matrices"""
        kraus_mats = qml.TritFlip(*ps, wires=0).kraus_matrices()

        expected_K0 = np.sqrt(1 - sum(ps)) * np.eye(3)
        assert np.allclose(kraus_mats[0], expected_K0, atol=tol, rtol=0)

        Ks = [
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        ]

        for p, K, res in zip(ps, Ks, kraus_mats[1:]):
            expected_K = np.sqrt(p) * np.array(K)
            assert np.allclose(res, expected_K, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "p_01,p_02,p_12",
        [(1.2, 0, 0), (0, -0.3, 0.5), (0, 0, 1 + math.eps), (1, math.eps, 0), (0.3, 0.4, 0.4)],
    )
    def test_p_invalid_parameter(self, p_01, p_02, p_12):
        """Ensures that error is thrown when p_01, p_02, p_12, or their sum are outside [0,1]"""
        with pytest.raises(ValueError, match="must be in the interval"):
            qml.TritFlip(p_01, p_02, p_12, wires=0).kraus_matrices()

    @staticmethod
    def expected_jac_fn(p_01, p_02, p_12):
        """Gets the expected Jacobian of Kraus matrices"""
        # Set up the 3 partial derivatives of the 4 3x3 Kraus Matrices
        partials = math.zeros((3, 4, 3, 3))

        # All 3 partials have the same first Kraus Operator output
        partials[:, 0] = -1 / (2 * math.sqrt(1 - (p_01 + p_02 + p_12))) * math.eye(3)

        # Set the matrix defined by each partials parameter, the rest are 0
        partials[0, 1] = 1 / (2 * math.sqrt(p_01)) * math.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        partials[1, 2] = 1 / (2 * math.sqrt(p_02)) * math.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        partials[2, 3] = 1 / (2 * math.sqrt(p_12)) * math.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        return partials

    @staticmethod
    def kraus_fn(p_01, p_02, p_12):
        """Gets a matrix of the Kraus matrices to be tested."""
        return qml.math.stack(qml.TritFlip(p_01, p_02, p_12, wires=0).kraus_matrices())

    @pytest.mark.autograd
    def test_kraus_jac_autograd(self):
        """Tests Jacobian of Kraus matrices using autograd."""

        p_01 = pnp.array(0.14, requires_grad=True)
        p_02 = pnp.array(0.04, requires_grad=True)
        p_12 = pnp.array(0.23, requires_grad=True)
        jac = qml.jacobian(self.kraus_fn)(p_01, p_02, p_12)
        assert qml.math.allclose(jac, self.expected_jac_fn(p_01, p_02, p_12))

    @pytest.mark.torch
    def test_kraus_jac_torch(self):
        """Tests Jacobian of Kraus matrices using PyTorch."""
        import torch

        ps = [0.14, 0.04, 0.23]

        p_01 = torch.tensor(ps[0], requires_grad=True)
        p_02 = torch.tensor(ps[1], requires_grad=True)
        p_12 = torch.tensor(ps[2], requires_grad=True)

        jac = torch.autograd.functional.jacobian(self.kraus_fn, (p_01, p_02, p_12))
        expected_jac = self.expected_jac_fn(*ps)
        for j, exp in zip(jac, expected_jac):
            assert qml.math.allclose(j.detach().numpy(), exp)

    @pytest.mark.tf
    def test_kraus_jac_tf(self):
        """Tests Jacobian of Kraus matrices using TensorFlow."""
        import tensorflow as tf

        p_01 = tf.Variable(0.14)
        p_02 = tf.Variable(0.04)
        p_12 = tf.Variable(0.23)
        with tf.GradientTape() as tape:
            out = self.kraus_fn(p_01, p_02, p_12)
        jac = tape.jacobian(out, (p_01, p_02, p_12))
        assert qml.math.allclose(jac, self.expected_jac_fn(p_01, p_02, p_12))

    @pytest.mark.jax
    def test_kraus_jac_jax(self):
        """Tests Jacobian of Kraus matrices using JAX."""
        import jax

        p_01 = jax.numpy.array(0.14)
        p_02 = jax.numpy.array(0.04)
        p_12 = jax.numpy.array(0.23)
        jac = jax.jacobian(self.kraus_fn, argnums=[0, 1, 2])(p_01, p_02, p_12)
        assert qml.math.allclose(jac, self.expected_jac_fn(p_01, p_02, p_12))


class TestQutritChannel:
    """Tests for the quantum channel QubitChannel"""

    def test_input_correctly_handled(self, tol):
        """Test that Kraus matrices are correctly processed"""
        K_list = qml.QutritDepolarizingChannel(0.75, wires=0).kraus_matrices()
        out = qml.QutritChannel(K_list, wires=0).kraus_matrices()

        assert np.allclose(out, K_list, atol=tol, rtol=0)

    def test_kraus_matrices_are_square(self):
        """Tests that the given Kraus matrices are square"""
        K_list = [np.zeros((3, 3)), np.zeros((2, 3))]
        with pytest.raises(
            ValueError, match="Only channels with the same input and output Hilbert space"
        ):
            qml.QutritChannel(K_list, wires=0)

    def test_kraus_matrices_are_of_same_shape(self):
        """Tests that the given Kraus matrices are of same shape"""
        K_list = [np.eye(3), np.eye(4)]
        with pytest.raises(ValueError, match="All Kraus matrices must have the same shape."):
            qml.QutritChannel(K_list, wires=0)

    def test_kraus_matrices_are_dimensions(self):
        """Tests that the given Kraus matrices are of right dimension i.e (9,9)"""
        K_list = [np.eye(3), np.eye(3)]
        with pytest.raises(ValueError, match=r"Shape of all Kraus matrices must be \(9,9\)."):
            qml.QutritChannel(K_list, wires=[0, 1])

    def test_kraus_matrices_are_trace_preserved(self):
        """Tests that the channel represents a trace-preserving map"""
        K_list = [0.75 * np.eye(3), 0.35j * np.eye(3)]
        with pytest.raises(ValueError, match="Only trace preserving channels can be applied."):
            qml.QutritChannel(K_list, wires=0)

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "finite-diff", "backprop"])
    def test_integrations(self, diff_method):
        """Test integration"""
        kraus = [
            np.array([[1, 0, 0], [0, 0.70710678, 0], [0, 0, 0.8660254]]),
            np.array([[0, 0.70710678, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 0.5], [0, 0, 0], [0, 0, 0]]),
        ]

        dev = qml.device("default.qutrit.mixed", wires=1)

        @qml.qnode(dev, diff_method=diff_method)
        def func():
            qml.QutritChannel(kraus, 0)
            return qml.expval(qml.GellMann(wires=0, index=1))

        func()

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "finite-diff", "backprop"])
    def test_integration_grad(self, diff_method):
        """Test integration with grad"""
        dev = qml.device("default.qutrit.mixed", wires=1)

        @qml.qnode(dev, diff_method=diff_method)
        def func(p):
            kraus = qml.QutritDepolarizingChannel.compute_kraus_matrices(p)
            qml.QutritChannel(kraus, 0)
            return qml.expval(qml.GellMann(wires=0, index=1))

        qml.grad(func)(0.5)

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "finite-diff", "backprop"])
    def test_integration_jacobian(self, diff_method):
        """Test integration with grad"""
        dev = qml.device("default.qutrit.mixed", wires=1)

        @qml.qnode(dev, diff_method=diff_method)
        def func(p):
            kraus = qml.QutritDepolarizingChannel.compute_kraus_matrices(p)
            qml.QutritChannel(kraus, 0)
            return qml.expval(qml.GellMann(wires=0, index=1))

        qml.jacobian(func)(0.5)

    def test_flatten(self):
        """Test flatten method returns kraus matrices and wires"""
        kraus = [
            np.array([[1, 0, 0], [0, 0.70710678, 0], [0, 0, 0.8660254]]),
            np.array([[0, 0.70710678, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 0.5], [0, 0, 0], [0, 0, 0]]),
        ]

        qutrit_channel = qml.QutritChannel(kraus, 1, id="test")
        data, metadata = qutrit_channel._flatten()  # pylint: disable=protected-access
        new_op = qml.QutritChannel._unflatten(data, metadata)  # pylint: disable=protected-access
        qml.assert_equal(qutrit_channel, new_op)

        assert np.allclose(kraus, data)
        assert metadata == (qml.wires.Wires(1), ())
