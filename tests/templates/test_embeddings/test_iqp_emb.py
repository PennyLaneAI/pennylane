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
Tests for the IQPEmbedding template.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    QUEUES = [
        (1, ["Hadamard", "RZ"], [[0], [0]]),
        (2, ["Hadamard", "RZ", "Hadamard", "RZ", "MultiRZ"], [[0], [0], [1], [1], [0, 1]]),
        (
            3,
            ["Hadamard", "RZ", "Hadamard", "RZ", "Hadamard", "RZ", "MultiRZ", "MultiRZ", "MultiRZ"],
            [[0], [0], [1], [1], [2], [2], [0, 1], [0, 2], [1, 2]],
        ),
    ]

    @pytest.mark.parametrize("n_wires, expected_names, expected_wires", QUEUES)
    def test_expansion(self, n_wires, expected_names, expected_wires):
        """Checks the queue for the default settings."""

        features = list(range(n_wires))

        op = qml.IQPEmbedding(features, wires=range(n_wires))
        tape = op.expand()

        for i, gate in enumerate(tape.operations):
            assert gate.name == expected_names[i]
            assert gate.wires.labels == tuple(expected_wires[i])

    def test_repeat(self):
        """Checks the queue for repetition of the template."""

        features = list(range(3))

        expected_names = self.QUEUES[2][1] + self.QUEUES[2][1]
        expected_wires = self.QUEUES[2][2] + self.QUEUES[2][2]

        op = qml.IQPEmbedding(features, wires=range(3), n_repeats=2)
        tape = op.expand()

        for i, gate in enumerate(tape.operations):
            assert gate.name == expected_names[i]
            assert gate.wires.labels == tuple(expected_wires[i])

    def test_custom_pattern(self):
        """Checks the queue for custom pattern for the entanglers."""

        features = list(range(3))
        pattern = [[0, 2], [0, 1]]
        expected_names = [
            "Hadamard",
            "RZ",
            "Hadamard",
            "RZ",
            "Hadamard",
            "RZ",
            "MultiRZ",
            "MultiRZ",
            "MultiRZ",
        ]
        expected_wires = [[0], [0], [1], [1], [2], [2], *pattern]

        op = qml.IQPEmbedding(features, wires=range(3), pattern=pattern)
        tape = op.expand()

        for i, gate in enumerate(tape.operations):
            assert gate.name == expected_names[i]
            assert gate.wires.labels == tuple(expected_wires[i])

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        features = np.random.random(size=(3,))

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.IQPEmbedding(features, wires=range(3))
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.IQPEmbedding(features, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        "features", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]
    )
    def test_exception_wrong_number_of_features(self, features):
        """Verifies that an exception is raised if 'features' has the wrong shape."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(f=None):
            qml.IQPEmbedding(features=f, wires=range(3))
            return [qml.expval(qml.PauliZ(w)) for w in range(3)]

        with pytest.raises(ValueError, match="Features must be"):
            circuit(f=features)

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.IQPEmbedding(np.array([1, 2]), wires=[0, 1], id="a")
        assert template.id == "a"


def circuit_template(features):
    qml.IQPEmbedding(features, range(2))
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(features):
    qml.Hadamard(wires=0)
    qml.RZ(features[0], wires=0)
    qml.Hadamard(wires=1)
    qml.RZ(features[1], wires=1)
    qml.MultiRZ(features[0] * features[1], wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    def test_list_and_tuples(self, tol):
        """Tests common iterables as inputs."""

        features = [0.1, -1.3]

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(tuple(features))
        res2 = circuit2(tuple(features))
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        features = np.random.random(size=(2,))
        features = pnp.array(features, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(features)

        grad_fn2 = qml.grad(circuit2)
        grads2 = grad_fn2(features)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        features = jnp.array(np.random.random(size=(2,)))

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev, interface="jax")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="jax")

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(features)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(features)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        features = tf.Variable(np.random.random(size=(2,)))

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev, interface="tf")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="tf")

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            res = circuit(features)
        grads = tape.gradient(res, [features])

        with tf.GradientTape() as tape2:
            res2 = circuit2(features)
        grads2 = tape2.gradient(res2, [features])

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests the torch interface."""

        import torch

        features = torch.tensor(np.random.random(size=(2,)), requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev, interface="torch")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="torch")

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(features)
        res.backward()
        grads = [features.grad]

        res2 = circuit2(features)
        res2.backward()
        grads2 = [features.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
