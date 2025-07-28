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
Unit tests for the available built-in quantum channels.
"""
# pylint: disable=too-few-public-methods
from itertools import product

import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.exceptions import WireError
from pennylane.ops import channel

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

one_arg_args = [(0.0,), (0.1,), (1.0,)]

channels_with_one_arg = [
    channel.AmplitudeDamping,
    channel.PhaseDamping,
    channel.BitFlip,
    channel.PhaseFlip,
    channel.DepolarizingChannel,
    channel.PhaseDamping,
]

two_arg_args = [
    (0.0, 0.0),
    (0.0, 0.1),
    (0.1, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (0.3, 0.2),
]

channels_with_two_args = [channel.GeneralizedAmplitudeDamping, channel.ResetError]

channels_and_args = (
    list(product(channels_with_one_arg, one_arg_args))
    + list(product(channels_with_two_args, two_arg_args))
    + [
        (channel.PauliError, ("X", 0.0)),
        (channel.PauliError, ("X", 0.41)),
        (channel.PauliError, ("X", 1.0)),
        (channel.PauliError, ("YY", 0.0)),
        (channel.PauliError, ("YX", 0.41)),
        (channel.PauliError, ("ZZ", 1.0)),
        (channel.ThermalRelaxationError, (0.0, 1e-4, 1e-4, 2e-8)),
        (channel.ThermalRelaxationError, (0.1, 1e-4, 1e-4, 2e-8)),
        (channel.ThermalRelaxationError, (1.0, 1e-4, 1e-4, 2e-8)),
        (channel.ThermalRelaxationError, (0.0, 1e-4, 1.2e-4, 2e-8)),
        (channel.ThermalRelaxationError, (0.1, 1e-4, 1.2e-4, 2e-8)),
        (channel.ThermalRelaxationError, (1.0, 1e-4, 1.2e-4, 2e-8)),
    ]
)


class TestChannels:
    """Tests for the quantum channels"""

    @pytest.mark.parametrize(
        "interface",
        [
            None,
            pytest.param("autograd", marks=pytest.mark.autograd),
            pytest.param("tensorflow", marks=pytest.mark.tf),
            pytest.param("jax", marks=pytest.mark.jax),
            pytest.param("torch", marks=pytest.mark.torch),
        ],
    )
    @pytest.mark.parametrize("ch, args", channels_and_args)
    def test_kraus_matrices_sum_identity(self, ch, args, interface, tol):
        """Test channels are trace-preserving"""
        if ch is channel.ResetError:
            args = (args[0] / 2, args[1] / 3)
        args = tuple(
            arg if isinstance(arg, str) else qml.math.array(arg, like=interface) for arg in args
        )
        if ch is channel.PauliError and len(args[0]) > 1:
            wires = [0, 1]
        else:
            wires = [0]
        op = ch(*args, wires=wires)
        K_list = op.kraus_matrices()
        K_arr = qml.math.stack(K_list)
        Kraus_sum = qml.math.einsum("ajk,ajl->kl", qml.math.conj(K_arr), K_arr)
        assert qml.math.allclose(Kraus_sum, np.eye(K_list[0].shape[0]), atol=tol, rtol=0)


class TestAmplitudeDamping:
    """Tests for the quantum channel AmplitudeDamping"""

    def test_gamma_zero(self, tol):
        """Test gamma=0 gives correct Kraus matrices"""
        op = channel.AmplitudeDamping
        assert np.allclose(op(0, wires=0).kraus_matrices()[0], np.eye(2), atol=tol, rtol=0)
        assert np.allclose(op(0, wires=0).kraus_matrices()[1], np.zeros((2, 2)), atol=tol, rtol=0)

    def test_gamma_arbitrary(self, tol):
        """Test gamma=0.1 gives correct Kraus matrices"""
        op = channel.AmplitudeDamping
        expected = [
            np.array([[1.0, 0.0], [0.0, 0.9486833]]),
            np.array([[0.0, 0.31622777], [0.0, 0.0]]),
        ]
        assert np.allclose(op(0.1, wires=0).kraus_matrices(), expected, atol=tol, rtol=0)

    def test_gamma_invalid_parameter(self):
        with pytest.raises(ValueError, match="gamma must be in the interval"):
            channel.AmplitudeDamping(1.5, wires=0).kraus_matrices()

    @staticmethod
    def expected_jac_fn(gamma):
        return [
            qml.math.array([[0, 0], [0, -1 / (2 * qml.math.sqrt(1 - gamma))]]),
            qml.math.array([[0, 1 / (2 * qml.math.sqrt(gamma))], [0, 0]]),
        ]

    @staticmethod
    def kraus_fn(x):
        return qml.math.stack(channel.AmplitudeDamping(x, wires=0).kraus_matrices())

    @pytest.mark.autograd
    def test_kraus_jac_autograd(self):
        gamma = pnp.array(0.43, requires_grad=True)
        jac = qml.jacobian(self.kraus_fn)(gamma)
        assert qml.math.allclose(jac, self.expected_jac_fn(gamma))

    @pytest.mark.torch
    def test_kraus_jac_torch(self):
        import torch

        gamma = torch.tensor(0.43, requires_grad=True)
        jac = torch.autograd.functional.jacobian(self.kraus_fn, gamma)
        assert qml.math.allclose(jac.detach().numpy(), self.expected_jac_fn(gamma.detach().numpy()))

    @pytest.mark.tf
    def test_kraus_jac_tf(self):
        import tensorflow as tf

        gamma = tf.Variable(0.43)
        with tf.GradientTape() as tape:
            out = self.kraus_fn(gamma)
        jac = tape.jacobian(out, gamma)
        assert qml.math.allclose(jac, self.expected_jac_fn(gamma))

    @pytest.mark.jax
    def test_kraus_jac_jax(self):
        import jax

        gamma = jax.numpy.array(0.43)
        jac = jax.jacobian(self.kraus_fn)(gamma)
        assert qml.math.allclose(jac, self.expected_jac_fn(gamma))


class TestGeneralizedAmplitudeDamping:
    """Tests for the quantum channel GeneralizedAmplitudeDamping"""

    def test_gamma_p_zero(self, tol):
        """Test p=0, gamma=0 gives correct Kraus matrices"""
        op = channel.GeneralizedAmplitudeDamping
        assert np.allclose(
            op(0, 0, wires=0).kraus_matrices()[0], np.zeros((2, 2)), atol=tol, rtol=0
        )
        assert np.allclose(op(0, 0, wires=0).kraus_matrices()[2], np.eye(2), atol=tol, rtol=0)

    def test_gamma_p_arbitrary(self, tol):
        """Test arbitrary p and gamma values give correct first Kraus matrix"""

        op = channel.GeneralizedAmplitudeDamping
        # check K0 for gamma=0.1, p =0.1
        expected_K0 = np.array([[0.31622777, 0.0], [0.0, 0.3]])
        assert np.allclose(op(0.1, 0.1, wires=0).kraus_matrices()[0], expected_K0, atol=tol, rtol=0)

        # check K3 for gamma=0.1, p=0.5
        expected_K3 = np.array([[0.0, 0.0], [0.2236068, 0.0]])
        assert np.allclose(op(0.1, 0.5, wires=0).kraus_matrices()[3], expected_K3, atol=tol, rtol=0)

    def test_gamma_invalid_parameter(self):
        with pytest.raises(ValueError, match="gamma must be in the interval"):
            channel.GeneralizedAmplitudeDamping(1.5, 0.0, wires=0).kraus_matrices()

    def test_p_invalid_parameter(self):
        with pytest.raises(ValueError, match="p must be in the interval"):
            channel.GeneralizedAmplitudeDamping(0.0, 1.5, wires=0).kraus_matrices()

    @staticmethod
    def expected_jac_fn(gamma, p):
        return (
            [
                qml.math.sqrt(p)
                * qml.math.array([[0, 0], [0, -1 / (2 * qml.math.sqrt(1 - gamma))]]),
                qml.math.sqrt(p) * qml.math.array([[0, 1 / (2 * qml.math.sqrt(gamma))], [0, 0]]),
                qml.math.sqrt(1 - p)
                * qml.math.array([[-1 / (2 * qml.math.sqrt(1 - gamma)), 0], [0, 0]]),
                qml.math.sqrt(1 - p)
                * qml.math.array([[0, 0], [1 / (2 * qml.math.sqrt(gamma)), 0]]),
            ],
            [
                1
                / (2 * qml.math.sqrt(p))
                * qml.math.array([[1, 0], [0, qml.math.sqrt(1 - gamma)]]),
                1 / (2 * qml.math.sqrt(p)) * qml.math.array([[0, qml.math.sqrt(gamma)], [0, 0]]),
                -1
                / (2 * qml.math.sqrt(1 - p))
                * qml.math.array([[qml.math.sqrt(1 - gamma), 0], [0, 1]]),
                -1
                / (2 * qml.math.sqrt(1 - p))
                * qml.math.array([[0, 0], [qml.math.sqrt(gamma), 0]]),
            ],
        )

    @staticmethod
    def kraus_fn(gamma, p):
        return qml.math.stack(
            channel.GeneralizedAmplitudeDamping(gamma, p, wires=0).kraus_matrices()
        )

    @pytest.mark.autograd
    def test_kraus_jac_autograd(self):
        gamma = pnp.array(0.43, requires_grad=True)
        p = pnp.array(0.3, requires_grad=True)
        jac = qml.jacobian(self.kraus_fn)(gamma, p)
        assert qml.math.allclose(jac, self.expected_jac_fn(gamma, p))

    @pytest.mark.torch
    def test_kraus_jac_torch(self):
        import torch

        gamma = torch.tensor(0.43, requires_grad=True)
        p = torch.tensor(0.3, requires_grad=True)
        jac = torch.autograd.functional.jacobian(self.kraus_fn, (gamma, p))
        exp_jac = self.expected_jac_fn(gamma.detach().numpy(), p.detach().numpy())
        assert len(jac) == len(exp_jac) == 2
        for j, exp_j in zip(jac, exp_jac):
            assert qml.math.allclose(j.detach().numpy(), exp_j)

    @pytest.mark.tf
    def test_kraus_jac_tf(self):
        import tensorflow as tf

        gamma = tf.Variable(0.43)
        p = tf.Variable(0.3)
        with tf.GradientTape() as tape:
            out = self.kraus_fn(gamma, p)
        jac = tape.jacobian(out, (gamma, p))
        assert qml.math.allclose(jac, self.expected_jac_fn(gamma, p))

    @pytest.mark.jax
    def test_kraus_jac_jax(self):
        import jax

        gamma = jax.numpy.array(0.43)
        p = jax.numpy.array(0.3)
        jac = jax.jacobian(self.kraus_fn, argnums=[0, 1])(gamma, p)
        assert qml.math.allclose(jac, self.expected_jac_fn(gamma, p))


class TestPhaseDamping:
    """Tests for the quantum channel PhaseDamping"""

    def test_gamma_zero(self, tol):
        """Test gamma=0 gives correct Kraus matrices"""
        op = channel.PhaseDamping
        assert np.allclose(op(0, wires=0).kraus_matrices()[0], np.eye(2), atol=tol, rtol=0)
        assert np.allclose(op(0, wires=0).kraus_matrices()[1], np.zeros((2, 2)), atol=tol, rtol=0)

    def test_gamma_arbitrary(self, tol):
        """Test gamma=0.1 gives correct Kraus matrices"""
        op = channel.PhaseDamping
        expected = [
            np.array([[1.0, 0.0], [0.0, 0.9486833]]),
            np.array([[0.0, 0.0], [0.0, 0.31622777]]),
        ]
        assert np.allclose(op(0.1, wires=0).kraus_matrices(), expected, atol=tol, rtol=0)

    def test_gamma_invalid_parameter(self):
        with pytest.raises(ValueError, match="gamma must be in the interval"):
            channel.PhaseDamping(1.5, wires=0).kraus_matrices()

    @staticmethod
    def expected_jac_fn(gamma):
        return [
            qml.math.array([[0, 0], [0, -1 / (2 * qml.math.sqrt(1 - gamma))]]),
            qml.math.array([[0, 0], [0, 1 / (2 * qml.math.sqrt(gamma))]]),
        ]

    @staticmethod
    def kraus_fn(x):
        return qml.math.stack(channel.PhaseDamping(x, wires=0).kraus_matrices())

    @pytest.mark.autograd
    def test_kraus_jac_autograd(self):
        gamma = pnp.array(0.43, requires_grad=True)
        jac = qml.jacobian(self.kraus_fn)(gamma)
        assert qml.math.allclose(jac, self.expected_jac_fn(gamma))

    @pytest.mark.torch
    def test_kraus_jac_torch(self):
        import torch

        gamma = torch.tensor(0.43, requires_grad=True)
        jac = torch.autograd.functional.jacobian(self.kraus_fn, gamma)
        assert qml.math.allclose(jac.detach().numpy(), self.expected_jac_fn(gamma.detach().numpy()))

    @pytest.mark.tf
    def test_kraus_jac_tf(self):
        import tensorflow as tf

        gamma = tf.Variable(0.43)
        with tf.GradientTape() as tape:
            out = self.kraus_fn(gamma)
        jac = tape.jacobian(out, gamma)
        assert qml.math.allclose(jac, self.expected_jac_fn(gamma))

    @pytest.mark.jax
    def test_kraus_jac_jax(self):
        import jax

        gamma = jax.numpy.array(0.43)
        jac = jax.jacobian(self.kraus_fn)(gamma)
        assert qml.math.allclose(jac, self.expected_jac_fn(gamma))


class TestBitFlip:
    """Tests for the quantum channel BitFlipChannel"""

    @pytest.mark.parametrize("p", [0, 0.1, 0.5, 1])
    def test_p_arbitrary(self, p, tol):
        """Test that various values of p give correct Kraus matrices"""
        op = channel.BitFlip

        expected_K0 = np.sqrt(1 - p) * np.eye(2)
        assert np.allclose(op(p, wires=0).kraus_matrices()[0], expected_K0, atol=tol, rtol=0)

        expected_K1 = np.sqrt(p) * X
        assert np.allclose(op(p, wires=0).kraus_matrices()[1], expected_K1, atol=tol, rtol=0)

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    def test_grad_bitflip(self, angle):
        """Test that analytical gradient is computed correctly for different states. Channel
        grad recipes are independent of channel parameter"""

        dev = qml.device("default.mixed", wires=1)
        prob = pnp.array(0.5, requires_grad=True)

        @qml.qnode(dev)
        def circuit(p):
            qml.RX(angle, wires=0)
            qml.BitFlip(p, wires=0)
            return qml.expval(qml.PauliZ(0))

        gradient = np.squeeze(qml.grad(circuit)(prob))
        assert np.allclose(gradient, circuit(1) - circuit(0))
        assert np.allclose(gradient, (-2 * np.cos(angle)))

    def test_p_invalid_parameter(self):
        with pytest.raises(ValueError, match="p must be in the interval"):
            channel.BitFlip(1.5, wires=0).kraus_matrices()

    @staticmethod
    def expected_jac_fn(p):
        return [
            -1 / (2 * qml.math.sqrt(1 - p)) * qml.math.eye(2),
            1 / (2 * qml.math.sqrt(p)) * qml.math.array([[0, 1], [1, 0]]),
        ]

    @staticmethod
    def kraus_fn(x):
        return qml.math.stack(channel.BitFlip(x, wires=0).kraus_matrices())

    @pytest.mark.autograd
    def test_kraus_jac_autograd(self):
        p = pnp.array(0.43, requires_grad=True)
        jac = qml.jacobian(self.kraus_fn)(p)
        assert qml.math.allclose(jac, self.expected_jac_fn(p))

    @pytest.mark.torch
    def test_kraus_jac_torch(self):
        import torch

        p = torch.tensor(0.43, requires_grad=True)
        jac = torch.autograd.functional.jacobian(self.kraus_fn, p)
        assert qml.math.allclose(jac.detach().numpy(), self.expected_jac_fn(p.detach().numpy()))

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

        p = jax.numpy.array(0.43)
        jac = jax.jacobian(self.kraus_fn)(p)
        assert qml.math.allclose(jac, self.expected_jac_fn(p))


class TestPhaseFlip:
    """Test that various values of p give correct Kraus matrices"""

    @pytest.mark.parametrize("p", [0, 0.1, 0.5, 1])
    def test_p_arbitrary(self, p, tol):
        """Test p=0.1 gives correct Kraus matrices"""
        op = channel.PhaseFlip

        expected_K0 = np.sqrt(1 - p) * np.eye(2)
        assert np.allclose(op(p, wires=0).kraus_matrices()[0], expected_K0, atol=tol, rtol=0)

        expected_K1 = np.sqrt(p) * Z
        assert np.allclose(op(p, wires=0).kraus_matrices()[1], expected_K1, atol=tol, rtol=0)

    # TODO: bring back angle 0 when the bug fixed https://github.com/PennyLaneAI/pennylane/pull/6684#issuecomment-2552123064
    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7)[1:])
    def test_grad_phaseflip(self, angle):
        """Test that analytical gradient is computed correctly for different states. Channel
        grad recipes are independent of channel parameter"""

        dev = qml.device("default.mixed", wires=1)
        prob = pnp.array(0.5, requires_grad=True)

        @qml.qnode(dev)
        def circuit(p):
            qml.RX(angle, wires=0)
            qml.PhaseFlip(p, wires=0)
            return qml.expval(qml.PauliZ(0))

        gradient = np.squeeze(qml.grad(circuit)(prob))
        assert gradient == circuit(1) - circuit(0)
        assert np.allclose(gradient, 0.0)

    def test_p_invalid_parameter(self):
        with pytest.raises(ValueError, match="p must be in the interval"):
            channel.PhaseFlip(1.5, wires=0).kraus_matrices()

    @staticmethod
    def expected_jac_fn(p):
        return [
            -1 / (2 * qml.math.sqrt(1 - p)) * qml.math.eye(2),
            1 / (2 * qml.math.sqrt(p)) * qml.math.diag([1, -1]),
        ]

    @staticmethod
    def kraus_fn(x):
        return qml.math.stack(channel.PhaseFlip(x, wires=0).kraus_matrices())

    @pytest.mark.autograd
    def test_kraus_jac_autograd(self):
        p = pnp.array(0.43, requires_grad=True)
        jac = qml.jacobian(self.kraus_fn)(p)
        assert qml.math.allclose(jac, self.expected_jac_fn(p))

    @pytest.mark.torch
    def test_kraus_jac_torch(self):
        import torch

        p = torch.tensor(0.43, requires_grad=True)
        jac = torch.autograd.functional.jacobian(self.kraus_fn, p)
        assert qml.math.allclose(jac.detach().numpy(), self.expected_jac_fn(p.detach().numpy()))

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

        p = jax.numpy.array(0.43)
        jac = jax.jacobian(self.kraus_fn)(p)
        assert qml.math.allclose(jac, self.expected_jac_fn(p))


class TestDepolarizingChannel:
    """Tests for the quantum channel DepolarizingChannel"""

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


class TestResetError:
    """Tests for the quantum channel ResetError"""

    @pytest.mark.parametrize("p_0,p_1", list(zip([0.5, 0.1, 0.0, 0.0], [0, 0.1, 0.5, 0.0])))
    def test_p0_p1_arbitrary(self, p_0, p_1, tol):
        """Test that various values of p_0 and p_1 give correct Kraus matrices"""
        op = channel.ResetError

        expected_K0 = np.sqrt(1 - p_0 - p_1) * np.eye(2)
        assert np.allclose(op(p_0, p_1, wires=0).kraus_matrices()[0], expected_K0, atol=tol, rtol=0)

        expected_K1 = np.sqrt(p_0) * np.array([[1, 0], [0, 0]])
        assert np.allclose(op(p_0, p_1, wires=0).kraus_matrices()[1], expected_K1, atol=tol, rtol=0)

        expected_K2 = np.sqrt(p_0) * np.array([[0, 1], [0, 0]])
        assert np.allclose(op(p_0, p_1, wires=0).kraus_matrices()[2], expected_K2, atol=tol, rtol=0)

        expected_K3 = np.sqrt(p_1) * np.array([[0, 0], [1, 0]])
        assert np.allclose(op(p_0, p_1, wires=0).kraus_matrices()[3], expected_K3, atol=tol, rtol=0)

        expected_K4 = np.sqrt(p_1) * np.array([[0, 0], [0, 1]])
        assert np.allclose(op(p_0, p_1, wires=0).kraus_matrices()[4], expected_K4, atol=tol, rtol=0)

    def test_p0_invalid_parameter(self):
        with pytest.raises(ValueError, match="p_0 must be in the interval"):
            channel.ResetError(1.5, 0.0, wires=0).kraus_matrices()

    def test_p1_invalid_parameter(self):
        with pytest.raises(ValueError, match="p_1 must be in the interval"):
            channel.ResetError(0.0, 1.5, wires=0).kraus_matrices()

    def test_p0_p1_sum_not_normalized(self):
        with pytest.raises(ValueError, match="must be in the interval"):
            channel.ResetError(1.0, 1.0, wires=0).kraus_matrices()

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    def test_grad_reset_error(self, angle):
        """Test that gradient is computed correctly for different states. Channel
        grad recipes are independent of channel parameter"""

        dev = qml.device("default.mixed", wires=1)
        p_0, p_1 = pnp.array([0.0, 0.5], requires_grad=True)

        @qml.qnode(dev)
        def circuit(p_0, p_1):
            qml.RX(angle, wires=0)
            qml.ResetError(p_0, p_1, wires=0)
            return qml.expval(qml.PauliZ(0))

        gradient = np.squeeze(qml.grad(circuit)(p_0, p_1))
        assert np.allclose(
            gradient,
            np.array(
                [
                    (1 / 0.1) * (circuit(0.1, p_1) - circuit(0.0, p_1)),
                    (1 / 0.1) * (circuit(p_0, 0.1) - circuit(p_0, 0.0)),
                ]
            ),
        )
        assert np.allclose(
            gradient,
            np.array(
                [
                    (2 * np.sin(angle / 2) * np.sin(angle / 2)),
                    (-2 * np.cos(angle / 2) * np.cos(angle / 2)),
                ]
            ),
        )

    @staticmethod
    def expected_jac_fn(p0, p1):
        return (
            [
                -1 / (2 * qml.math.sqrt(1 - p0 - p1)) * qml.math.eye(2),
                1 / (2 * qml.math.sqrt(p0)) * qml.math.array([[1, 0], [0, 0]]),
                1 / (2 * qml.math.sqrt(p0)) * qml.math.array([[0, 1], [0, 0]]),
                qml.math.zeros((2, 2)),
                qml.math.zeros((2, 2)),
            ],
            [
                -1 / (2 * qml.math.sqrt(1 - p0 - p1)) * qml.math.eye(2),
                qml.math.zeros((2, 2)),
                qml.math.zeros((2, 2)),
                1 / (2 * qml.math.sqrt(p1)) * qml.math.array([[0, 0], [1, 0]]),
                1 / (2 * qml.math.sqrt(p1)) * qml.math.array([[0, 0], [0, 1]]),
            ],
        )

    @staticmethod
    def kraus_fn(p0, p1):
        return qml.math.stack(channel.ResetError(p0, p1, wires=0).kraus_matrices())

    @pytest.mark.autograd
    def test_kraus_jac_autograd(self):
        p0 = pnp.array(0.43, requires_grad=True)
        p1 = pnp.array(0.12, requires_grad=True)

        jac = qml.jacobian(self.kraus_fn)(p0, p1)
        assert qml.math.allclose(jac, self.expected_jac_fn(p0, p1))

    @pytest.mark.torch
    def test_kraus_jac_torch(self):
        import torch

        p0 = torch.tensor(0.43, requires_grad=True)
        p1 = torch.tensor(0.12, requires_grad=True)
        jac = torch.autograd.functional.jacobian(self.kraus_fn, (p0, p1))
        exp_jac = self.expected_jac_fn(p0.detach().numpy(), p1.detach().numpy())
        assert len(jac) == len(exp_jac) == 2
        for j, exp_j in zip(jac, exp_jac):
            assert qml.math.allclose(j.detach().numpy(), exp_j)

    @pytest.mark.tf
    def test_kraus_jac_tf(self):
        import tensorflow as tf

        p0 = tf.Variable(0.43)
        p1 = tf.Variable(0.12)
        with tf.GradientTape() as tape:
            out = self.kraus_fn(p0, p1)
        jac = tape.jacobian(out, (p0, p1))
        assert qml.math.allclose(jac, self.expected_jac_fn(p0, p1))

    @pytest.mark.jax
    def test_kraus_jac_jax(self):
        import jax

        p0 = jax.numpy.array(0.43)
        p1 = jax.numpy.array(0.12)
        jac = jax.jacobian(self.kraus_fn, argnums=[0, 1])(p0, p1)
        assert qml.math.allclose(jac, self.expected_jac_fn(p0, p1))


class TestPauliError:
    """Tests for the quantum channel PauliError"""

    OPERATORS_WRONG_PARAMS = ["XXX", "XXX", "ABC", "XXX"]
    P_WRONG_PARAMS = [0.5, 1.5, 0.5, 0.5]
    WIRES_WRONG_PARAMS = [[0], [0, 1, 2], [0, 1, 2], [1, 1, 2]]
    EXPECTED_ERRORS = [ValueError, ValueError, ValueError, WireError]
    EXPECTED_MESSAGES = [
        "The number of operators must match the number of wires",
        "p must be in the interval \\[0,1\\]",
        "The specified operators need to be either of 'I', 'X', 'Y' or 'Z'.",
        "Wires must be unique",
    ]

    @pytest.mark.parametrize(
        "operators, p, wires, error, message",
        list(
            zip(
                OPERATORS_WRONG_PARAMS,
                P_WRONG_PARAMS,
                WIRES_WRONG_PARAMS,
                EXPECTED_ERRORS,
                EXPECTED_MESSAGES,
            )
        ),
    )
    def test_wrong_parameters(self, operators, p, wires, error, message):
        """Test wrong parametrizations of PauliError"""
        # pylint: disable=too-many-arguments
        with pytest.raises(error, match=message):
            channel.PauliError(operators, p, wires=wires)

    def test_warning_many_qubits(self):
        """Test if warning is thrown when huge matrix"""
        with pytest.warns(UserWarning):
            channel.PauliError("X" * 512, 0.5, wires=list(range(512)))

    def test_p_zero(self, tol):
        """Test resulting Kraus matrices for p=0"""
        expected_Ks = [np.eye(2**5), np.zeros((2**5, 2**5))]
        c = channel.PauliError("XXXXX", 0, wires=[0, 1, 2, 3, 4])

        assert np.allclose(c.kraus_matrices(), expected_Ks, atol=tol, rtol=0)

    def test_p_one(self, tol):
        """Test resulting Kraus matrices for p=1"""
        expected_Ks = [np.zeros((2**5, 2**5)), np.flip(np.eye(2**5), axis=1)]
        c = channel.PauliError("XXXXX", 1, wires=[0, 1, 2, 3, 4])

        assert np.allclose(c.kraus_matrices(), expected_Ks, atol=tol, rtol=0)

    OPERATORS = ["X", "XY", "ZX", "ZI"]
    WIRES = [[1], [0, 1], [3, 1], [1, 0]]
    EXPECTED_KS = [
        [
            np.sqrt(0.5) * np.eye(2),
            np.array(
                [
                    [0.0, 0.70710678],
                    [0.70710678, 0.0],
                ]
            ),
        ],
        [
            np.sqrt(0.5) * np.eye(4),
            np.array(
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 0.70710678j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.70710678j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 - 0.70710678j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.70710678j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ]
            ),
        ],
        [
            np.sqrt(0.5) * np.eye(4),
            np.array(
                [
                    [0.0, 0.70710678, 0.0, 0.0],
                    [0.70710678, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -0.0, -0.70710678],
                    [0.0, 0.0, -0.70710678, -0.0],
                ]
            ),
        ],
        [
            np.sqrt(0.5) * np.eye(4),
            np.sqrt(0.5)
            * np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0],
                ]
            ),
        ],
    ]

    @pytest.mark.parametrize(
        "operators, wires, expected_Ks", list(zip(OPERATORS, WIRES, EXPECTED_KS))
    )
    def test_kraus_matrix(self, tol, operators, wires, expected_Ks):
        """Test sevaral resulting kraus matrices for sevaral configurations"""
        c = channel.PauliError(operators, 0.5, wires=wires)

        assert np.allclose(c.kraus_matrices(), expected_Ks, atol=tol, rtol=0)

    expected_jac_fn = {
        "X": lambda p: [
            -1 / (2 * qml.math.sqrt(1 - p)) * qml.math.eye(2),
            1 / (2 * qml.math.sqrt(p)) * qml.math.array([[0, 1], [1, 0]]),
        ],
        "XY": lambda p: [
            -1 / (2 * qml.math.sqrt(1 - p)) * qml.math.eye(4),
            1 / (2 * qml.math.sqrt(p)) * (qml.math.diag([1j, -1j, 1j, -1j])[::-1]),
        ],
        "ZI": lambda p: [
            -1 / (2 * qml.math.sqrt(1 - p)) * qml.math.eye(4),
            1 / (2 * qml.math.sqrt(p)) * (qml.math.diag([1, 1, -1, -1])),
        ],
    }

    @pytest.mark.parametrize("ops", ["X", "XY", "ZI"])
    @pytest.mark.autograd
    def test_kraus_jac_autograd(self, ops):
        p = pnp.array(0.43, requires_grad=True)
        wires = list(range(len(ops)))

        def fn_real(x):
            return qml.math.real(
                qml.math.stack(channel.PauliError(ops, x, wires=wires).kraus_matrices())
            )

        def fn_imag(x):
            return qml.math.imag(
                qml.math.stack(channel.PauliError(ops, x, wires=wires).kraus_matrices())
            )

        jac_fn_real = qml.jacobian(fn_real)
        jac_fn_imag = qml.jacobian(fn_imag)
        jac = jac_fn_real(p) + 1j * jac_fn_imag(p)
        assert qml.math.allclose(jac, self.expected_jac_fn[ops](p))

    @pytest.mark.parametrize("ops", ["X", "XY", "ZI"])
    @pytest.mark.torch
    def test_kraus_jac_torch(self, ops):
        import torch

        p = torch.tensor(0.43, requires_grad=True)
        wires = list(range(len(ops)))

        def fn_real(x):
            return qml.math.real(
                qml.math.stack(channel.PauliError(ops, x, wires=wires).kraus_matrices())
            )

        def fn_imag(x):
            return qml.math.imag(
                qml.math.stack(channel.PauliError(ops, x, wires=wires).kraus_matrices())
            )

        jac_real = torch.autograd.functional.jacobian(fn_real, p).detach().numpy()
        jac_imag = torch.autograd.functional.jacobian(fn_imag, p).detach().numpy()
        assert qml.math.allclose(
            jac_real + 1j * jac_imag, self.expected_jac_fn[ops](p.detach().numpy())
        )

    @pytest.mark.parametrize("ops", ["X", "XY", "ZI"])
    @pytest.mark.tf
    def test_kraus_jac_tf(self, ops):
        import tensorflow as tf

        p = tf.Variable(0.43)
        wires = list(range(len(ops)))
        with tf.GradientTape() as tape:
            out = qml.math.stack(channel.PauliError(ops, p, wires=wires).kraus_matrices())
        jac = tape.jacobian(out, p)
        assert qml.math.allclose(jac, self.expected_jac_fn[ops](p))

    @pytest.mark.parametrize("ops", ["X", "XY", "ZI"])
    @pytest.mark.jax
    def test_kraus_jac_jax(self, ops):
        import jax

        p = jax.numpy.array(0.43, dtype=jax.numpy.complex128)
        wires = list(range(len(ops)))

        def fn(x):
            return qml.math.stack(channel.PauliError(ops, x, wires=wires).kraus_matrices())

        jac_fn = jax.jacobian(fn, holomorphic=True)
        jac = jac_fn(p)
        assert qml.math.allclose(jac, self.expected_jac_fn[ops](p))


class TestQubitChannel:
    """Tests for the quantum channel QubitChannel"""

    def test_input_correctly_handled(self, tol):
        """Test that Kraus matrices are correctly processed"""
        K_list1 = [
            np.array([[1.0, 0.0], [0.0, 0.9486833]]),
            np.array([[0.0, 0.31622777], [0.0, 0.0]]),
        ]
        out = channel.QubitChannel(K_list1, wires=0).kraus_matrices()

        # verify equivalent to input matrices
        assert np.allclose(out, K_list1, atol=tol, rtol=0)

    def test_kraus_matrices_valid(self):
        """Tests that the given Kraus matrices are valid"""

        # check all Kraus matrices are square matrices
        K_list1 = [np.zeros((2, 2)), np.zeros((2, 3))]
        with pytest.raises(
            ValueError, match="Only channels with the same input and output Hilbert space"
        ):
            channel.QubitChannel(K_list1, wires=0)

        # check all Kraus matrices have the same shape
        K_list2 = [np.eye(2), np.eye(4)]
        with pytest.raises(ValueError, match="All Kraus matrices must have the same shape."):
            channel.QubitChannel(K_list2, wires=0)

        # check the dimension of all Kraus matrices are valid
        K_list3 = [np.array([np.eye(2), np.eye(2)]), np.array([np.eye(2), np.eye(2)])]
        with pytest.raises(ValueError, match="Dimension of all Kraus matrices must be "):
            channel.QubitChannel(K_list3, wires=0)

    def test_channel_trace_preserving(self):
        """Tests that the channel represents a trace-preserving map"""

        # real Kraus matrices
        K_list1 = [
            np.array([[1.0, 0.0], [0.0, 0.9486833]]),
            np.array([[0.0, 0.31622777], [0.0, 0.0]]),
        ]
        with pytest.raises(ValueError, match="Only trace preserving channels can be applied."):
            channel.QubitChannel(K_list1 * 2, wires=0)

        # complex Kraus matrices
        p = 0.1
        K_list2 = [np.sqrt(p) * Y, np.sqrt(1 - p) * np.eye(2)]
        with pytest.raises(ValueError, match="Only trace preserving channels can be applied."):
            channel.QubitChannel(K_list2 * 2, wires=0)

    @pytest.mark.jax
    def test_jit_compatibility(self):
        """Test that QubitChannel can be jitted."""

        import jax

        dev = qml.device("default.mixed", wires=1)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def noise_channel(p):
            k0 = jax.numpy.sqrt(1 - p) * jax.numpy.eye(2)
            k1 = jax.numpy.sqrt(p) * jax.numpy.eye(2)
            qml.QubitChannel([k0, k1], wires=[0])
            return qml.expval(qml.PauliZ(0))

        # just checking it runs
        noise_channel(0.1)


class TestThermalRelaxationError:
    """Tests for the quantum channel ThermalRelaxationError"""

    @pytest.mark.parametrize(
        "pe,t1,t2,tg",
        list(
            zip(
                [0.2, 0.4, 0.6, 0.0],
                [100e-6, 50e-6, 80e-6, np.inf],
                [80e-6, 40e-6, 80e-6, 50e-6],
                [20e-9, 40e-9, 40e-6, 40e-9],
            )
        ),
    )
    def test_t2_le_t1_arbitrary(self, pe, t1, t2, tg, tol):
        """Test that various values of pe, t1, t2, and tg
        for t2 <= t1 give correct Kraus matrices"""
        # pylint: disable=too-many-arguments

        op = channel.ThermalRelaxationError

        eT1 = np.exp(-tg / t1)
        p_reset = 1 - eT1
        eT2 = np.exp(-tg / t2)
        pz = (1 - p_reset) * (1 - eT2 / eT1) / 2
        pr0 = (1 - pe) * p_reset
        pr1 = pe * p_reset
        pid = 1 - pz - pr0 - pr1

        expected_K0 = np.sqrt(pid) * np.eye(2)
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices()[0], expected_K0, atol=tol, rtol=0
        )

        expected_K1 = np.sqrt(pz) * np.array([[1, 0], [0, -1]])
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices()[1], expected_K1, atol=tol, rtol=0
        )

        expected_K2 = np.sqrt(pr0) * np.array([[1, 0], [0, 0]])
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices()[2], expected_K2, atol=tol, rtol=0
        )

        expected_K3 = np.sqrt(pr0) * np.array([[0, 1], [0, 0]])
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices()[3], expected_K3, atol=tol, rtol=0
        )

        expected_K4 = np.sqrt(pr1) * np.array([[0, 0], [1, 0]])
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices()[4], expected_K4, atol=tol, rtol=0
        )

        expected_K5 = np.sqrt(pr1) * np.array([[0, 0], [0, 1]])
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices()[5], expected_K5, atol=tol, rtol=0
        )

    @pytest.mark.parametrize(
        "pe,t1,t2,tg",
        list(
            zip(
                [0.8, 0.5, 0.0],
                [100e-6, 50e-6, 80e-6],
                [120e-6, 100e-6, 90e-6],
                [20e-9, 40e-9, 90e-6],
            )
        ),
    )
    def test_t2_g_t1_arbitrary(self, pe, t1, t2, tg, tol):
        """Test that various values of pe, t1, t2, and tg
        for t2 > t1 give correct Kraus matrices"""
        # pylint: disable=too-many-arguments

        op = channel.ThermalRelaxationError

        if t1 == np.inf:
            eT1 = 0
            p_reset = 0
        else:
            eT1 = np.exp(-tg / t1)
            p_reset = 1 - eT1
        if t2 == np.inf:
            eT2 = 1
        else:
            eT2 = np.exp(-tg / t2)

        e0 = p_reset * pe
        v0 = np.array([[0], [1], [0], [0]])
        e1 = -p_reset * pe + p_reset
        v1 = np.array([[0], [0], [1], [0]])
        common_term = np.sqrt(
            4 * eT2**2 + 4 * p_reset**2 * pe**2 - 4 * p_reset**2 * pe + p_reset**2
        )
        e2 = 1 - p_reset / 2 - common_term / 2
        term2 = 2 * eT2 / (2 * p_reset * pe - p_reset - common_term)
        v2 = np.array([[term2], [0], [0], [1]]) / np.sqrt(term2**2 + 1**2)
        term3 = 2 * eT2 / (2 * p_reset * pe - p_reset + common_term)
        e3 = 1 - p_reset / 2 + common_term / 2
        v3 = np.array([[term3], [0], [0], [1]]) / np.sqrt(term3**2 + 1**2)

        expected_K0 = np.sqrt(e0) * v0.reshape((2, 2), order="F")
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices()[0], expected_K0, atol=tol, rtol=0
        )

        expected_K1 = np.sqrt(e1) * v1.reshape((2, 2), order="F")
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices()[1], expected_K1, atol=tol, rtol=0
        )

        expected_K2 = np.sqrt(e2) * v2.reshape((2, 2), order="F")
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices()[2], expected_K2, atol=tol, rtol=0
        )

        expected_K3 = np.sqrt(e3) * v3.reshape((2, 2), order="F")
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices()[3], expected_K3, atol=tol, rtol=0
        )

    def test_pe_invalid_parameter(self):
        with pytest.raises(ValueError, match="pe must be between"):
            channel.ThermalRelaxationError(1.5, 100e-6, 100e-6, 20e-9, wires=0).kraus_matrices()

    def test_T2_g_2T1_invalid_parameter(self):
        with pytest.raises(ValueError, match="Invalid T_2 relaxation time parameter"):
            channel.ThermalRelaxationError(0.3, 100e-6, np.inf, 20e-9, wires=0).kraus_matrices()

    def test_T1_le_0_invalid_parameter(self):
        with pytest.raises(ValueError, match="Invalid T_1 relaxation time parameter"):
            channel.ThermalRelaxationError(0.3, -50e-6, np.inf, 20e-9, wires=0).kraus_matrices()

    def test_T2_le_0_invalid_parameter(self):
        with pytest.raises(ValueError, match="Invalid T_2 relaxation time parameter"):
            channel.ThermalRelaxationError(0.3, 100e-6, 0, 20e-9, wires=0).kraus_matrices()

    def test_tg_le_0_invalid_parameter(self):
        with pytest.raises(ValueError, match="Invalid gate_time"):
            channel.ThermalRelaxationError(0.3, 100e-6, 100e-6, -20e-9, wires=0).kraus_matrices()

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    def test_grad_thermal_relaxation_error(self, angle):
        """Test that gradient is computed correctly for different states. Channel
        grad recipes are independent of channel parameter"""

        dev = qml.device("default.mixed", wires=1)
        pe = pnp.array(0.0, requires_grad=True)

        @qml.qnode(dev)
        def circuit(pe):
            qml.RX(angle, wires=0)
            qml.ThermalRelaxationError(pe, 120e-6, 100e-6, 20e-9, wires=0)
            return qml.expval(qml.PauliZ(0))

        gradient = np.squeeze(qml.grad(circuit)(pe))
        assert np.allclose(
            gradient,
            np.array(
                [
                    (1 / 0.1) * (circuit(0.1) - circuit(0.0)),
                ]
            ),
        )
