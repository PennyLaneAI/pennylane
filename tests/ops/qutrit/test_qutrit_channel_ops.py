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
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.ops.qutrit import channel

QUDIT_DIM = 3


class TestQutritDepolarizingChannel:
    """Tests for the qutrit quantum channel QutritDepolarizingChannel"""

    @staticmethod
    def get_expected_kraus_matrices(p):
        """Gets the expected Kraus matrices given probability p."""
        w = np.exp(2j * np.pi / 3)

        Z0 = np.eye(QUDIT_DIM)
        Z1 = np.diag([1, w, w ** 2])
        Z2 = np.diag([1, w ** 2, w ** 4])

        X0 = np.eye(QUDIT_DIM)
        X1 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        X2 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

        Ks = [np.sqrt(1 - (8 * p / 9)) * np.eye(QUDIT_DIM)]
        for i, Z in enumerate((Z0, Z1, Z2)):
            for j, X in enumerate((X0, X1, X2)):
                if i == 0 and j == 0:
                    continue
                Ks.append(np.sqrt(p / 9) * X @ Z)

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
        for krause_matrix, expected_matrix in zip(kraus_matrices, expected_matrices):
            assert np.allclose(krause_matrix, expected_matrix, atol=tol, rtol=0)

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    def test_grad_depolarizing(self, angle):
        """Test that analytical gradient is computed correctly for different states. Channel
        grad recipes are independent of channel parameter"""

        dev = qml.device("default.qutrit.mixed")
        prob = pnp.array(0.5, requires_grad=True)

        @qml.qnode(dev)
        def circuit(p):
            qml.TRX(angle, wires=0)
            qml.QutritDepolarizingChannel(p, wires=0)
            return qml.expval(qml.GellMann(0, 3) + qml.GellMann(0, 8))

        gradient = np.squeeze(qml.grad(circuit)(prob))
        assert np.allclose(gradient, circuit(1) - circuit(0))
        assert np.allclose(gradient, -(4 / 3) * np.cos(angle))

    def test_p_invalid_parameter(self):
        """Test that error is raised given an inappropriate p value."""
        with pytest.raises(ValueError, match="p must be in the interval"):
            qml.QutritDepolarizingChannel(1.5, wires=0).kraus_matrices()

    @staticmethod
    def expected_jac_fn(p):
        """Gets the expected Jacobian of Kraus matrices given probability p."""
        w = np.exp(2j * np.pi / 3)

        Z0 = np.eye(QUDIT_DIM)
        Z1 = np.diag([1, w, w ** 2])
        Z2 = np.diag([1, w ** 2, w ** 4])

        X0 = np.eye(QUDIT_DIM)
        X1 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        X2 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

        jacs = [-(4 / 9) / np.sqrt(1 - (8 * p / 9)) * np.eye(QUDIT_DIM)]
        for i, Z in enumerate((Z0, Z1, Z2)):
            for j, X in enumerate((X0, X1, X2)):
                if i == 0 and j == 0:
                    continue
                jacs.append(1 / (18 * np.sqrt(p / 9)) * X @ Z)

        return jacs

    @staticmethod
    def kraus_fn(x):
        """Gets a matrix of the Kraus matrices to be tested."""
        return qml.math.stack(channel.QutritDepolarizingChannel(x, wires=0).kraus_matrices())

    @staticmethod
    def kraus_fn_real(x):
        """Gets a matrix of the real part of the Kraus matrices to be tested."""
        return qml.math.real(
            qml.math.stack(channel.QutritDepolarizingChannel(x, wires=0).kraus_matrices())
        )

    @staticmethod
    def kraus_fn_imag(x):
        """Gets a matrix of the imaginary part of the Kraus matrices to be tested."""
        return qml.math.imag(
            qml.math.stack(channel.QutritDepolarizingChannel(x, wires=0).kraus_matrices())
        )

    @pytest.mark.autograd
    def test_kraus_jac_autograd(self):
        """Tests Jacobian of Kraus matrices using autograd."""
        p = pnp.array(0.43, requires_grad=True)
        jac = qml.jacobian(self.kraus_fn_real)(p) + 1j * qml.jacobian(self.kraus_fn_imag)(p)
        assert qml.math.allclose(jac, self.expected_jac_fn(p))

    @pytest.mark.torch
    def test_kraus_jac_torch(self):
        """Tests Jacobian of Kraus matrices using torch."""
        import torch

        p = torch.tensor(0.43, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian
        jac = jacobian(self.kraus_fn_real, p) + 1j * jacobian(self.kraus_fn_imag, p)
        assert qml.math.allclose(jac, self.expected_jac_fn(p.detach().numpy()))

    @pytest.mark.tf
    def test_kraus_jac_tf(self):
        """Tests Jacobian of Kraus matrices using tensorflow."""
        import tensorflow as tf
        p = tf.Variable(0.43)
        with tf.GradientTape() as real_tape:
            real_out = self.kraus_fn_real(p)
        with tf.GradientTape() as imag_tape:
            imag_out = self.kraus_fn_imag(p)

        real_jac = qml.math.cast(real_tape.jacobian(real_out, p), complex)
        imaj_jac = qml.math.cast(imag_tape.jacobian(imag_out, p), complex)
        jac = real_jac + 1j * imaj_jac
        assert qml.math.allclose(jac, self.expected_jac_fn(0.43))

    @pytest.mark.jax
    def test_kraus_jac_jax(self):
        """Tests Jacobian of Kraus matrices using jax."""
        import jax

        jax.config.update("jax_enable_x64", True)

        p = jax.numpy.array(0.43, dtype=jax.numpy.complex128)
        jac = jax.jacobian(self.kraus_fn, holomorphic=True)(p)
        assert qml.math.allclose(jac, self.expected_jac_fn(p))
