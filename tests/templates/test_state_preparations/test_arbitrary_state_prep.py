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
Unit tests for the ArbitraryStatePreparation template.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates.state_preparations.arbitrary_state_preparation import (
    _state_preparation_pauli_words,
)


class TestHelpers:
    """Tests for the Pauli-word helper function."""

    @pytest.mark.parametrize(
        "num_wires,expected_pauli_words",
        [
            (1, ["X", "Y"]),
            (2, ["XI", "YI", "IX", "IY", "XX", "XY"]),
            (
                3,
                [
                    "XII",
                    "YII",
                    "IXI",
                    "IYI",
                    "IIX",
                    "IIY",
                    "IXX",
                    "IXY",
                    "XXI",
                    "XYI",
                    "XIX",
                    "XIY",
                    "XXX",
                    "XXY",
                ],
            ),
        ],
    )
    def test_state_preparation_pauli_words(self, num_wires, expected_pauli_words):
        """Test that the correct Pauli words are returned."""
        for idx, pauli_word in enumerate(_state_preparation_pauli_words(num_wires)):
            assert expected_pauli_words[idx] == pauli_word


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    def test_correct_gates_single_wire(self):
        """Test that the correct gates are applied on a single wire."""
        weights = np.array([0, 1], dtype=float)

        op = qml.ArbitraryStatePreparation(weights, wires=[0])
        queue = op.expand().operations

        assert queue[0].name == "PauliRot"

        assert queue[0].data[0] == weights[0]
        assert queue[0].hyperparameters["pauli_word"] == "X"
        assert queue[0].wires.labels == (0,)

        assert queue[1].name == "PauliRot"
        assert queue[1].data[0] == weights[1]
        assert queue[1].hyperparameters["pauli_word"] == "Y"
        assert queue[1].wires.labels == (0,)

    def test_correct_gates_two_wires(self):
        """Test that the correct gates are applied on on two wires."""
        weights = np.array([0, 1, 2, 3, 4, 5], dtype=float)

        op = qml.ArbitraryStatePreparation(weights, wires=[0, 1])
        queue = op.expand().operations

        assert queue[0].name == "PauliRot"

        assert queue[0].data[0] == weights[0]
        assert queue[0].hyperparameters["pauli_word"] == "XI"
        assert queue[0].wires.labels == (0, 1)

        assert queue[1].name == "PauliRot"
        assert queue[1].data[0] == weights[1]
        assert queue[1].hyperparameters["pauli_word"] == "YI"
        assert queue[1].wires.labels == (0, 1)

        assert queue[2].name == "PauliRot"
        assert queue[2].data[0] == weights[2]
        assert queue[2].hyperparameters["pauli_word"] == "IX"
        assert queue[2].wires.labels == (0, 1)

        assert queue[3].name == "PauliRot"
        assert queue[3].data[0] == weights[3]
        assert queue[3].hyperparameters["pauli_word"] == "IY"
        assert queue[3].wires.labels == (0, 1)

        assert queue[4].name == "PauliRot"
        assert queue[4].data[0] == weights[4]
        assert queue[4].hyperparameters["pauli_word"] == "XX"
        assert queue[4].wires.labels == (0, 1)

        assert queue[5].name == "PauliRot"
        assert queue[5].data[0] == weights[5]
        assert queue[5].hyperparameters["pauli_word"] == "XY"
        assert queue[5].wires.labels == (0, 1)

    def test_GHZ_generation(self, qubit_device_3_wires, tol):
        """Test that the template prepares a GHZ state."""
        GHZ_state = np.array([1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2)])

        weights = np.zeros(14)
        weights[13] = np.pi / 2

        @qml.qnode(qubit_device_3_wires)
        def circuit(weights):
            qml.ArbitraryStatePreparation(weights, [0, 1, 2])
            return qml.expval(qml.PauliZ(0))

        circuit(weights)

        assert np.allclose(circuit.device.state, GHZ_state, atol=tol, rtol=0)

    def test_even_superposition_generation(self, qubit_device_3_wires, tol):
        """Test that the template prepares an even superposition state."""
        even_superposition_state = np.ones(8) / np.sqrt(8)

        weights = np.zeros(14)
        weights[1] = np.pi / 2
        weights[3] = np.pi / 2
        weights[5] = np.pi / 2

        @qml.qnode(qubit_device_3_wires)
        def circuit(weights):
            qml.ArbitraryStatePreparation(weights, [0, 1, 2])

            return qml.expval(qml.PauliZ(0))

        circuit(weights)

        assert np.allclose(circuit.device.state, even_superposition_state, atol=tol, rtol=0)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        weights = np.random.random(size=(2**4 - 2))

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.ArbitraryStatePreparation(weights, wires=range(3))
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.ArbitraryStatePreparation(weights, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the
        number of dimensions of features is incorrect."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(weights):
            qml.ArbitraryStatePreparation(weights, wires=range(3))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Weights tensor must be of shape"):
            weights = np.zeros(12)
            circuit(weights)

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.ArbitraryStatePreparation(
            np.random.random(size=(2**4 - 2)), wires=[0, 1, 2], id="a"
        )
        assert template.id == "a"


class TestAttributes:
    """Tests additional methods and attributes"""

    @pytest.mark.parametrize(
        "n_wires, expected_shape",
        [
            (3, (14,)),
            (1, (2,)),
            (2, (6,)),
        ],
    )
    def test_shape(self, n_wires, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor"""

        shape = qml.ArbitraryStatePreparation.shape(n_wires)
        assert shape == expected_shape


def circuit_template(weights):
    qml.ArbitraryStatePreparation(weights, range(2))
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(weights):
    qml.PauliRot(weights[0], "XI", wires=[0, 1])
    qml.PauliRot(weights[1], "YI", wires=[0, 1])
    qml.PauliRot(weights[2], "IX", wires=[0, 1])
    qml.PauliRot(weights[3], "IY", wires=[0, 1])
    qml.PauliRot(weights[4], "XX", wires=[0, 1])
    qml.PauliRot(weights[5], "XY", wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    def test_list_and_tuples(self, tol):
        """Tests common iterables as inputs."""

        weights = [1] * 6

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        weights_tuple = tuple(weights)
        res = circuit(weights_tuple)
        res2 = circuit2(weights_tuple)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        weights = np.random.random(size=(6,))
        weights = pnp.array(weights, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = qml.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(6,)))

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev, interface="jax")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="jax")

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        weights = tf.Variable(np.random.random(size=(6,)))

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev, interface="tf")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="tf")

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            res = circuit(weights)
        grads = tape.gradient(res, [weights])

        with tf.GradientTape() as tape2:
            res2 = circuit2(weights)
        grads2 = tape2.gradient(res2, [weights])

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests the torch interface."""

        import torch

        weights = torch.tensor(np.random.random(size=(6,)), requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev, interface="torch")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="torch")

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(weights)
        res.backward()
        grads = [weights.grad]

        res2 = circuit2(weights)
        res2.backward()
        grads2 = [weights.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
