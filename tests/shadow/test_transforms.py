# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the classical shadows transforms"""

import pytest

import pennylane as qml
from pennylane import numpy as np


def hadamard_circuit(wires, shots=10000, interface="autograd"):
    """Hadamard circuit to put all qubits in equal superposition (locally)"""
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit():
        for i in range(wires):
            qml.Hadamard(wires=i)
        return qml.classical_shadow(wires=range(wires))

    return circuit


def max_entangled_circuit(wires, shots=10000, interface="autograd"):
    """maximally entangled state preparation circuit"""
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.Hadamard(wires=0)
        for i in range(1, wires):
            qml.CNOT(wires=[0, i])
        return qml.classical_shadow(wires=range(wires))

    return circuit


def qft_circuit(wires, shots=10000, interface="autograd"):
    """Quantum Fourier Transform circuit"""
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    one_state = np.zeros(wires)
    one_state[-1] = 1

    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.BasisState(one_state, wires=range(wires))
        qml.QFT(wires=range(wires))
        return qml.classical_shadow(wires=range(wires))

    return circuit


def basic_entangler_circuit(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit(x):
        qml.BasicEntanglerLayers(weights=x, wires=range(wires))
        return qml.classical_shadow(wires=range(wires))

    return circuit


def basic_entangler_circuit_exact_state(wires, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev, interface=interface)
    def circuit(x):
        qml.BasicEntanglerLayers(weights=x, wires=range(wires))
        return qml.state()

    def state_to_dm(state):
        return qml.math.outer(state, qml.math.conj(state))

    return lambda x: state_to_dm(circuit(x))


def basic_entangler_circuit_exact_expval(wires, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev, interface=interface)
    def circuit(x, obs):
        qml.BasicEntanglerLayers(weights=x, wires=range(wires))
        return [qml.expval(ob) for ob in obs]

    return circuit


@pytest.mark.parametrize("diffable", [True, False])
@pytest.mark.autograd
class TestStateForward:
    """Test that the state reconstruction is correct for a variety of states"""

    @pytest.mark.parametrize("wires", [1, 3])
    def test_hadamard_state(self, wires, diffable):
        """Test that the state reconstruction is correct for a uniform
        superposition of qubits"""
        circuit = hadamard_circuit(wires)
        circuit = qml.shadows.state(wires=range(wires), diffable=diffable)(circuit)

        actual = circuit()
        expected = np.ones((2**wires, 2**wires)) / (2**wires)

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.parametrize("wires", [1, 3])
    def test_max_entangled_state(self, wires, diffable):
        """Test that the state reconstruction is correct for a maximally entangled state"""
        circuit = max_entangled_circuit(wires)
        circuit = qml.shadows.state(wires=range(wires), diffable=diffable)(circuit)

        actual = circuit()
        expected = np.zeros((2**wires, 2**wires))
        expected[np.array([0, 0, -1, -1]), np.array([0, -1, 0, -1])] = 0.5

        assert qml.math.allclose(actual, expected, atol=1e-1)

    def test_partial_state(self, diffable):
        """Test that the state reconstruction is correct for a subset
        of the qubits"""
        wires_list = [[0], [0, 1]]

        circuit = max_entangled_circuit(3)
        circuit = qml.shadows.state(wires=wires_list, diffable=diffable)(circuit)

        actual = circuit()

        expected = [
            np.array([[0.5, 0], [0, 0.5]]),
            np.array([[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]]),
        ]

        assert qml.math.allclose(actual[0], expected[0], atol=1e-1)
        assert qml.math.allclose(actual[1], expected[1], atol=1e-1)


@pytest.mark.all_interfaces
class TestStateForwardInterfaces:
    """Test that state reconstruction works for all interfaces"""

    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    @pytest.mark.parametrize("diffable", [True, False])
    def test_qft_state(self, interface, diffable):
        """Test that the state reconstruction is correct for a QFT state"""
        circuit = qft_circuit(3, interface=interface)
        circuit = qml.shadows.state(wires=[0, 1, 2], diffable=diffable)(circuit)

        actual = circuit()
        expected = np.exp(np.arange(8) * 2j * np.pi / 8) / np.sqrt(8)
        expected = np.outer(expected, np.conj(expected))

        assert qml.math.allclose(actual, expected, atol=1e-1)


class TestStateBackward:
    """Test that the gradient of the state reconstruction is correct"""

    @pytest.mark.autograd
    def test_backward_autograd(self):
        """Test the gradient of the state for the autograd interface"""
        shadow_circuit = basic_entangler_circuit(3, shots=20000, interface="autograd")
        shadow_circuit = qml.shadows.state(wires=[0, 1, 2], diffable=True)(shadow_circuit)
        exact_circuit = basic_entangler_circuit_exact_state(3, "autograd")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = np.random.uniform(0.8, 2, size=qml.BasicEntanglerLayers.shape(n_layers=1, n_wires=3))

        # for autograd in particular, take only the real part since it doesn't
        # support complex differentiation
        actual = qml.jacobian(lambda x: qml.math.real(shadow_circuit(x)))(x)
        expected = qml.jacobian(lambda x: qml.math.real(exact_circuit(x)))(x)

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.jax
    def test_backward_jax(self):
        """Test the gradient of the state for the JAX interface"""
        import jax
        from jax import numpy as jnp

        shadow_circuit = basic_entangler_circuit(3, shots=20000, interface="jax")
        shadow_circuit = qml.shadows.state(wires=[0, 1, 2], diffable=True)(shadow_circuit)
        exact_circuit = basic_entangler_circuit_exact_state(3, "jax")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = jnp.array(
            np.random.uniform(
                0.8, 2, size=qml.BasicEntanglerLayers.shape(n_layers=1, n_wires=3)
            ).astype(np.complex64)
        )

        actual = qml.math.real(jax.jacrev(shadow_circuit, holomorphic=True)(x))
        expected = qml.math.real(jax.jacrev(exact_circuit, holomorphic=True)(x))

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.tf
    def test_backward_tf(self):
        """Test the gradient of the state for the tensorflow interface"""
        import tensorflow as tf

        shadow_circuit = basic_entangler_circuit(3, shots=20000, interface="tf")
        shadow_circuit = qml.shadows.state(wires=[0, 1, 2], diffable=True)(shadow_circuit)
        exact_circuit = basic_entangler_circuit_exact_state(3, "tf")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = tf.Variable(
            np.random.uniform(0.8, 2, size=qml.BasicEntanglerLayers.shape(n_layers=1, n_wires=3))
        )

        with tf.GradientTape() as tape:
            out = shadow_circuit(x)

        actual = tape.jacobian(out, x)

        with tf.GradientTape() as tape2:
            out2 = exact_circuit(x)

        expected = tape2.jacobian(out2, x)

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.torch
    def test_backward_torch(self):
        """Test the gradient of the state for the torch interface"""
        import torch

        shadow_circuit = basic_entangler_circuit(3, shots=20000, interface="torch")
        shadow_circuit = qml.shadows.state(wires=[0, 1, 2], diffable=True)(shadow_circuit)
        exact_circuit = basic_entangler_circuit_exact_state(3, "torch")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = torch.tensor(
            np.random.uniform(0.8, 2, size=qml.BasicEntanglerLayers.shape(n_layers=1, n_wires=3)),
            requires_grad=True,
        )

        actual = torch.autograd.functional.jacobian(shadow_circuit, x)
        expected = torch.autograd.functional.jacobian(exact_circuit, x)

        assert qml.math.allclose(actual, expected, atol=1e-1)


@pytest.mark.autograd
class TestExpvalForward:
    """Test that the expval estimation is correct for a variety of states"""

    def test_hadamard_expval(self):
        """Test that the expval estimation is correct for a uniform
        superposition of qubits"""
        obs = [
            qml.PauliX(1),
            qml.PauliX(0) @ qml.PauliX(2),
            qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2),
            qml.PauliY(2),
            qml.PauliY(1) @ qml.PauliZ(2),
            qml.PauliX(0) @ qml.PauliY(1),
            qml.PauliX(0) @ qml.PauliY(1) @ qml.Identity(2),
        ]
        expected = [1, 1, 1, 0, 0, 0, 0]

        circuit = hadamard_circuit(3, shots=100000)
        circuit = qml.shadows.expval(obs)(circuit)
        actual = circuit()

        assert qml.math.allclose(actual, expected, atol=1e-1)

    def test_max_entangled_expval(self):
        """Test that the expval estimation is correct for a maximally entangled state"""
        obs = [
            qml.PauliX(1),
            qml.PauliX(0) @ qml.PauliX(2),
            qml.PauliZ(2),
            qml.Identity(1) @ qml.PauliZ(2),
            qml.PauliZ(1) @ qml.PauliZ(2),
            qml.PauliX(0) @ qml.PauliY(1),
            qml.PauliX(0) @ qml.PauliY(1) @ qml.Identity(2),
            qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliY(2),
        ]
        expected = [0, 0, 0, 0, 1, 0, 0, -1]

        circuit = max_entangled_circuit(3, shots=100000)
        circuit = qml.shadows.expval(obs)(circuit)
        actual = circuit()

        assert qml.math.allclose(actual, expected, atol=1e-1)


@pytest.mark.all_interfaces
class TestExpvalForwardInterfaces:
    """Test that expval estimation works for all interfaces"""

    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_qft_expval(self, interface):
        """Test that the expval estimation is correct for a QFT state"""
        obs = [
            qml.PauliX(0),
            qml.PauliX(0) @ qml.PauliX(1),
            qml.PauliX(0) @ qml.PauliX(2),
            qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2),
            qml.PauliZ(2),
            qml.PauliX(1) @ qml.PauliY(2),
            qml.PauliY(1) @ qml.PauliX(2),
            qml.Identity(0) @ qml.PauliY(1) @ qml.PauliX(2),
            qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2),
            qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2),
        ]
        expected = [
            -1,
            0,
            -1 / np.sqrt(2),
            -1 / np.sqrt(2),
            0,
            0,
            1 / np.sqrt(2),
            1 / np.sqrt(2),
            -1 / np.sqrt(2),
            0,
        ]

        circuit = qft_circuit(3, shots=100000, interface=interface)
        circuit = qml.shadows.expval(obs)(circuit)
        actual = circuit()

        assert qml.math.allclose(actual, expected, atol=1e-1)


obs_strongly_entangled = [
    qml.PauliX(1),
    qml.PauliX(0) @ qml.PauliX(2),
    qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2),
    qml.PauliY(2),
    qml.PauliY(1) @ qml.PauliZ(2),
    qml.PauliX(0) @ qml.PauliY(1),
    qml.PauliX(0) @ qml.PauliY(1) @ qml.Identity(2),
]


class TestExpvalBackward:
    """Test that the gradient of the expval estimation is correct"""

    @pytest.mark.autograd
    def test_backward_autograd(self):
        """Test the gradient of the expval for the autograd interface"""
        shadow_circuit = basic_entangler_circuit(3, shots=20000, interface="autograd")
        shadow_circuit = qml.shadows.expval(obs_strongly_entangled)(shadow_circuit)
        exact_circuit = basic_entangler_circuit_exact_expval(3, "autograd")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = np.random.uniform(0.8, 2, size=qml.BasicEntanglerLayers.shape(n_layers=1, n_wires=3))

        # for autograd in particular, take only the real part since it doesn't
        # support complex differentiation
        actual = qml.jacobian(shadow_circuit)(x)
        expected = qml.jacobian(exact_circuit)(x, obs_strongly_entangled)

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.jax
    def test_backward_jax(self):
        """Test the gradient of the expval for the JAX interface"""
        import jax
        from jax import numpy as jnp

        shadow_circuit = basic_entangler_circuit(3, shots=20000, interface="jax")
        shadow_circuit = qml.shadows.expval(obs_strongly_entangled)(shadow_circuit)
        exact_circuit = basic_entangler_circuit_exact_expval(3, "jax")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = jnp.array(
            np.random.uniform(0.8, 2, size=qml.BasicEntanglerLayers.shape(n_layers=1, n_wires=3))
        )

        actual = jax.jacrev(shadow_circuit)(x)
        expected = jax.jacrev(exact_circuit)(x, obs_strongly_entangled)

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.tf
    def test_backward_tf(self):
        """Test the gradient of the state for the tensorflow interface"""
        import tensorflow as tf

        shadow_circuit = basic_entangler_circuit(3, shots=20000, interface="tf")
        shadow_circuit = qml.shadows.expval(obs_strongly_entangled)(shadow_circuit)
        exact_circuit = basic_entangler_circuit_exact_expval(3, "tf")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = tf.Variable(
            np.random.uniform(0.8, 2, size=qml.BasicEntanglerLayers.shape(n_layers=1, n_wires=3))
        )

        with tf.GradientTape() as tape:
            out = shadow_circuit(x)

        actual = tape.jacobian(out, x)

        with tf.GradientTape() as tape2:
            out2 = exact_circuit(x, obs_strongly_entangled)

        expected = tape2.jacobian(out2, x)

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.torch
    def test_backward_torch(self):
        """Test the gradient of the state for the torch interface"""
        import torch

        shadow_circuit = basic_entangler_circuit(3, shots=20000, interface="torch")
        shadow_circuit = qml.shadows.expval(obs_strongly_entangled)(shadow_circuit)
        exact_circuit = basic_entangler_circuit_exact_expval(3, "torch")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = torch.tensor(
            np.random.uniform(0.8, 2, size=qml.BasicEntanglerLayers.shape(n_layers=1, n_wires=3)),
            requires_grad=True,
        )

        actual = torch.autograd.functional.jacobian(shadow_circuit, x)
        expected = torch.autograd.functional.jacobian(
            lambda x: exact_circuit(x, obs_strongly_entangled), x
        )

        assert qml.math.allclose(actual, expected, atol=1e-1)
