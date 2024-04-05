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
from pennylane.ops import channel


class TestQutritDepolarizingChannel:
    """Tests for the qutrit quantum channel QutritDepolarizingChannel"""

    def test_p_zero(self, tol):
        """Test p=0 gives correct Kraus matrices"""
        op = qml.DepolarizingChannel
        assert np.allclose(op(0, wires=0).kraus_matrices()[0], np.eye(2), atol=tol, rtol=0)
        assert np.allclose(op(0, wires=0).kraus_matrices()[1], np.zeros((2, 2)), atol=tol, rtol=0)

    def test_p_arbitrary(self, tol):
        """Test p=0.1 gives correct Kraus matrices"""
        p = 0.1
        op = qml.DepolarizingChannel
        expected = np.sqrt(p / 3) * X
        assert np.allclose(op(0.1, wires=0).kraus_matrices()[1], expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    def test_grad_depolarizing(self, angle):
        """Test that analytical gradient is computed correctly for different states. Channel
        grad recipes are independent of channel parameter"""

        dev = qml.device("default.mixed", wires=1)
        prob = pnp.array(0.5, requires_grad=True)

        @qml.qnode(dev)
        def circuit(p):
            qml.RX(angle, wires=0)
            qml.DepolarizingChannel(p, wires=0)
            return qml.expval(qml.PauliZ(0))

        gradient = np.squeeze(qml.grad(circuit)(prob))
        assert np.allclose(gradient, circuit(1) - circuit(0))
        assert np.allclose(gradient, -(4 / 3) * np.cos(angle))

    def test_p_invalid_parameter(self):
        with pytest.raises(ValueError, match="p must be in the interval"):
            qml.DepolarizingChannel(1.5, wires=0).kraus_matrices()

    @staticmethod
    def expected_jac_fn(p):
        return [
            -1 / (2 * qml.math.sqrt(1 - p)) * qml.math.eye(2),
            1 / (6 * qml.math.sqrt(p / 3)) * qml.math.array([[0, 1], [1, 0]]),
            1 / (6 * qml.math.sqrt(p / 3)) * qml.math.array([[0, -1j], [1j, 0]]),
            1 / (6 * qml.math.sqrt(p / 3)) * qml.math.diag([1, -1]),
        ]

    @staticmethod
    def kraus_fn(x):
        return qml.math.stack(channel.DepolarizingChannel(x, wires=0).kraus_matrices())

    @staticmethod
    def kraus_fn_real(x):
        return qml.math.real(
            qml.math.stack(channel.DepolarizingChannel(x, wires=0).kraus_matrices())
        )

    @staticmethod
    def kraus_fn_imag(x):
        return qml.math.imag(
            qml.math.stack(channel.DepolarizingChannel(x, wires=0).kraus_matrices())
        )

    @pytest.mark.autograd
    def test_kraus_jac_autograd(self):
        p = pnp.array(0.43, requires_grad=True)
        jac = qml.jacobian(self.kraus_fn_real)(p) + 1j * qml.jacobian(self.kraus_fn_imag)(p)
        assert qml.math.allclose(jac, self.expected_jac_fn(p))

    @pytest.mark.torch
    def test_kraus_jac_torch(self):
        import torch

        p = torch.tensor(0.43, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian
        jac = jacobian(self.kraus_fn_real, p) + 1j * jacobian(self.kraus_fn_imag, p)
        assert qml.math.allclose(jac, self.expected_jac_fn(p.detach().numpy()))

    @pytest.mark.tf
    def test_kraus_jac_tf(self):
        import tensorflow as tf

        p = tf.Variable(0.43)
        with tf.GradientTape() as tape:
            out = self.kraus_fn(p)
        jac = tape.jacobian(out, p)
        assert qml.math.allclose(jac, self.expected_jac_fn(p))

    @pytest.mark.jax
    def test_kraus_jac_jax(self):
        import jax

        jax.config.update("jax_enable_x64", True)

        p = jax.numpy.array(0.43, dtype=jax.numpy.complex128)
        jac = jax.jacobian(self.kraus_fn, holomorphic=True)(p)
        assert qml.math.allclose(jac, self.expected_jac_fn(p))
