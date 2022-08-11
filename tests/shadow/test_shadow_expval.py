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
"""Unit tests for the classical shadows expval measurement process"""

import pytest
import copy

import pennylane as qml
from pennylane import numpy as np


def hadamard_circuit(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit(obs, k=1):
        for i in range(wires):
            qml.Hadamard(wires=i)
        return qml.classical_shadow_expval(obs, k=k)

    return circuit


def max_entangled_circuit(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit(obs, k=1):
        qml.Hadamard(wires=0)
        for i in range(1, wires):
            qml.CNOT(wires=[0, i])
        return qml.classical_shadow_expval(obs, k=k)

    return circuit


def qft_circuit(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    one_state = np.zeros(wires)
    one_state[-1] = 1

    @qml.qnode(dev, interface=interface)
    def circuit(obs, k=1):
        qml.BasisState(one_state, wires=range(wires))
        qml.QFT(wires=range(wires))
        return qml.classical_shadow_expval(obs, k=k)

    return circuit


def strongly_entangling_circuit(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit(x, obs, k):
        qml.StronglyEntanglingLayers(weights=x, wires=range(wires))
        return qml.classical_shadow_expval(obs, k=k)

    return circuit


def strongly_entangling_circuit_exact(wires, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev, interface=interface)
    def circuit(x, obs):
        qml.StronglyEntanglingLayers(weights=x, wires=range(wires))
        return qml.expval(obs)

    return circuit


@pytest.mark.autograd
class TestExpvalMeasurement:
    @pytest.mark.parametrize("wires", [1, 2, 3])
    @pytest.mark.parametrize("shots", [1, 10, 100])
    def test_measurement_process_numeric_type(self, wires, shots):
        """Test that the numeric type of the MeasurementProcess instance is correct"""
        dev = qml.device("default.qubit", wires=wires, shots=shots)
        H = qml.PauliZ(0)
        res = qml.classical_shadow_expval(H)
        assert res.numeric_type == float

    @pytest.mark.parametrize("wires", [1, 2, 3])
    @pytest.mark.parametrize("shots", [1, 10, 100])
    def test_measurement_process_shape(self, wires, shots):
        """Test that the shape of the MeasurementProcess instance is correct"""
        dev = qml.device("default.qubit", wires=wires, shots=shots)
        H = qml.PauliZ(0)
        res = qml.classical_shadow_expval(H)
        assert res.shape() == ()
        assert res.shape(dev) == ()

    @pytest.mark.parametrize("wires", [1, 2, 3])
    @pytest.mark.parametrize("shots", [1, 10, 100])
    def test_measurement_process_copy(self, wires, shots):
        """Test that the attributes of the MeasurementProcess instance are
        correctly copied"""
        dev = qml.device("default.qubit", wires=wires, shots=shots)
        H = qml.PauliZ(0)
        res = qml.classical_shadow_expval(H, k=10)

        copied_res = copy.copy(res)
        assert type(copied_res) == type(res)
        assert copied_res.return_type == res.return_type
        assert copied_res.obs == res.obs
        assert copied_res.seed == res.seed

    @pytest.mark.parametrize("wires", [1, 2, 3])
    def test_shots_none_error(self, wires):
        """Test that an error is raised when a device with shots=None is used
        to obtain classical shadows"""
        circuit = hadamard_circuit(wires, None)
        H = qml.PauliZ(0)

        msg = "The number of shots has to be explicitly set on the device when using sample-based measurements"
        with pytest.raises(qml.QuantumFunctionError, match=msg):
            shadow = circuit(H, k=10)

    @pytest.mark.parametrize("wires", [1, 2, 3])
    @pytest.mark.parametrize("shots", [1, 10, 100])
    def test_multi_measurement_error(self, wires, shots):
        """Test that an error is raised when classical shadows is returned
        with other measurement processes"""
        dev = qml.device("default.qubit", wires=wires, shots=shots)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)

            for target in range(1, wires):
                qml.CNOT(wires=[0, target])

            return qml.classical_shadow_expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0))

        msg = "Classical shadows cannot be returned in combination with other return types"
        with pytest.raises(qml.QuantumFunctionError, match=msg):
            shadow = circuit()


@pytest.mark.all_interfaces
class TestExpvalForward:
    """Test the classical_shadow_expval measurement process forward pass"""

    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.PauliX(1), 1),
            (qml.PauliX(0) @ qml.PauliX(2), 1),
            (qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2), 1),
            (qml.PauliY(2), 0),
            (qml.PauliY(1) @ qml.PauliZ(2), 0),
            (qml.PauliX(0) @ qml.PauliY(1), 0),
            (qml.PauliX(0) @ qml.PauliY(1) @ qml.Identity(2), 0),
        ],
    )
    @pytest.mark.parametrize("k", [1, 5, 10])
    def test_hadamard_expval(self, interface, obs, expected, k):
        """Test that the expval estimation is correct for a uniform
        superposition of qubits"""
        import torch

        circuit = hadamard_circuit(3, shots=100000, interface=interface)
        actual = circuit(obs, k=k)

        assert actual.shape == ()
        assert actual.dtype == torch.float64 if interface == "torch" else np.float64
        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.PauliX(1), 0),
            (qml.PauliX(0) @ qml.PauliX(2), 0),
            (qml.PauliZ(2), 0),
            (qml.Identity(1) @ qml.PauliZ(2), 0),
            (qml.PauliZ(1) @ qml.PauliZ(2), 1),
            (qml.PauliX(0) @ qml.PauliY(1), 0),
            (qml.PauliX(0) @ qml.PauliY(1) @ qml.Identity(2), 0),
            (qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliY(2), -1),
        ],
    )
    @pytest.mark.parametrize("k", [1, 5, 10])
    def test_max_entangled_expval(self, interface, obs, expected, k):
        """Test that the expval estimation is correct for a maximally
        entangled state"""
        import torch

        circuit = max_entangled_circuit(3, shots=100000, interface=interface)
        actual = circuit(obs, k=k)

        assert actual.shape == ()
        assert actual.dtype == torch.float64 if interface == "torch" else np.float64
        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.PauliX(0), -1),
            (qml.PauliX(0) @ qml.PauliX(1), 0),
            (qml.PauliX(0) @ qml.PauliX(2), -1 / np.sqrt(2)),
            (qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2), -1 / np.sqrt(2)),
            (qml.PauliZ(2), 0),
            (qml.PauliX(1) @ qml.PauliY(2), 0),
            (qml.PauliY(1) @ qml.PauliX(2), 1 / np.sqrt(2)),
            (qml.Identity(0) @ qml.PauliY(1) @ qml.PauliX(2), 1 / np.sqrt(2)),
            (qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2), -1 / np.sqrt(2)),
            (qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2), 0),
        ],
    )
    @pytest.mark.parametrize("k", [1, 5, 10])
    def test_qft_expval(self, interface, obs, expected, k):
        """Test that the expval estimation is correct for a QFT state"""
        import torch

        circuit = qft_circuit(3, shots=100000, interface=interface)
        actual = circuit(obs, k=k)

        assert actual.shape == ()
        assert actual.dtype == torch.float64 if interface == "torch" else np.float64
        assert qml.math.allclose(actual, expected, atol=1e-1)

    def test_non_pauli_error(self):
        """Test that an error is raised when a non-Pauli observable is passed"""
        circuit = hadamard_circuit(3)

        msg = "Observable must be a linear combination of Pauli observables"
        with pytest.raises(ValueError, match=msg):
            circuit(qml.Hadamard(0) @ qml.Hadamard(2))


class TestExpvalBackward:
    """Test the classical_shadow_expval measurement process backward pass"""

    @pytest.mark.autograd
    @pytest.mark.parametrize(
        "obs",
        [
            (qml.PauliX(1)),
            (qml.PauliX(0) @ qml.PauliX(2)),
            (qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2)),
            (qml.PauliY(2)),
            (qml.PauliY(1) @ qml.PauliZ(2)),
            (qml.PauliX(0) @ qml.PauliY(1)),
            (qml.PauliX(0) @ qml.PauliY(1) @ qml.Identity(2)),
        ],
    )
    def test_backward_autograd(self, obs):
        """Test that the gradient of the expval estimation is correct for
        the autograd interface"""
        shadow_circuit = strongly_entangling_circuit(3, shots=20000, interface="autograd")
        exact_circuit = strongly_entangling_circuit_exact(3, "autograd")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = np.random.uniform(
            0.8, 2, size=qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=3)
        )

        actual = qml.grad(shadow_circuit)(x, obs, k=10)
        expected = qml.grad(exact_circuit)(x, obs)

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "obs",
        [
            (qml.PauliX(1)),
            (qml.PauliX(0) @ qml.PauliX(2)),
            (qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2)),
            (qml.PauliY(2)),
            (qml.PauliY(1) @ qml.PauliZ(2)),
            (qml.PauliX(0) @ qml.PauliY(1)),
            (qml.PauliX(0) @ qml.PauliY(1) @ qml.Identity(2)),
        ],
    )
    def test_backward_jax(self, obs):
        """Test that the gradient of the expval estimation is correct for
        the jax interface"""
        import jax
        from jax import numpy as jnp

        shadow_circuit = strongly_entangling_circuit(3, shots=20000, interface="jax")
        exact_circuit = strongly_entangling_circuit_exact(3, "jax")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = jnp.array(
            np.random.uniform(
                0.8, 2, size=qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=3)
            )
        )

        actual = jax.grad(shadow_circuit)(x, obs, k=10)
        expected = jax.grad(exact_circuit)(x, obs)

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.tf
    @pytest.mark.parametrize(
        "obs",
        [
            (qml.PauliX(1)),
            (qml.PauliX(0) @ qml.PauliX(2)),
            (qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2)),
            (qml.PauliY(2)),
            (qml.PauliY(1) @ qml.PauliZ(2)),
            (qml.PauliX(0) @ qml.PauliY(1)),
            (qml.PauliX(0) @ qml.PauliY(1) @ qml.Identity(2)),
        ],
    )
    def test_backward_tf(self, obs):
        """Test that the gradient of the expval estimation is correct for
        the tensorflow interface"""
        import tensorflow as tf

        shadow_circuit = strongly_entangling_circuit(3, shots=20000, interface="tf")
        exact_circuit = strongly_entangling_circuit_exact(3, "tf")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = tf.Variable(
            np.random.uniform(
                0.8, 2, size=qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=3)
            )
        )

        with tf.GradientTape() as tape:
            out = shadow_circuit(x, obs, k=10)

        actual = tape.gradient(out, x)

        with tf.GradientTape() as tape2:
            out2 = exact_circuit(x, obs)

        expected = tape2.gradient(out2, x)

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.torch
    @pytest.mark.parametrize(
        "obs",
        [
            (qml.PauliX(1)),
            (qml.PauliX(0) @ qml.PauliX(2)),
            (qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2)),
            (qml.PauliY(2)),
            (qml.PauliY(1) @ qml.PauliZ(2)),
            (qml.PauliX(0) @ qml.PauliY(1)),
            (qml.PauliX(0) @ qml.PauliY(1) @ qml.Identity(2)),
        ],
    )
    def test_backward_torch(self, obs):
        """Test that the gradient of the expval estimation is correct for
        the pytorch interface"""
        import torch

        shadow_circuit = strongly_entangling_circuit(3, shots=20000, interface="torch")
        exact_circuit = strongly_entangling_circuit_exact(3, "torch")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = torch.tensor(
            np.random.uniform(
                0.8, 2, size=qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=3)
            ),
            requires_grad=True,
        )

        out = shadow_circuit(x, obs, k=10)
        out.backward()
        actual = x.grad

        out2 = exact_circuit(x, obs)
        out2.backward()
        expected = x.grad

        assert qml.math.allclose(actual, expected, atol=1e-1)
