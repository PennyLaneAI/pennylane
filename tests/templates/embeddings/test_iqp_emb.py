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
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""
    features = (0.0, 1.0, 2.0)

    op = qml.IQPEmbedding(features, wires=(0, 1, 2))
    qml.ops.functions.assert_valid(op)


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
        tape = qml.tape.QuantumScript(op.decomposition())

        j = 0
        for i, gate in enumerate(tape.operations):
            assert gate.name == expected_names[i]
            assert gate.wires.labels == tuple(expected_wires[i])
            if gate.name == "RZ":
                assert gate.data[0] == features[j]
                j += 1

    @pytest.mark.parametrize("n_wires, expected_names, expected_wires", QUEUES)
    def test_expansion_broadcasted(self, n_wires, expected_names, expected_wires):
        """Checks the queue for the default settings."""

        features = np.arange(n_wires * 3).reshape((3, n_wires))

        op = qml.IQPEmbedding(features, wires=range(n_wires))
        assert op.batch_size == 3
        tape = qml.tape.QuantumScript(op.decomposition())

        j = 0
        for i, gate in enumerate(tape.operations):
            assert gate.name == expected_names[i]
            assert gate.wires.labels == tuple(expected_wires[i])
            if gate.name == "RZ":
                assert gate.batch_size == 3
                assert np.allclose(gate.data[0], features[:, j])
                j += 1

    def test_repeat(self):
        """Checks the queue for repetition of the template."""

        features = list(range(3))

        expected_names = self.QUEUES[2][1] + self.QUEUES[2][1]
        expected_wires = self.QUEUES[2][2] + self.QUEUES[2][2]

        op = qml.IQPEmbedding(features, wires=range(3), n_repeats=2)
        tape = qml.tape.QuantumScript(op.decomposition())

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
        tape = qml.tape.QuantumScript(op.decomposition())

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
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.IQPEmbedding(features, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        "features", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]
    )
    def test_exception_wrong_number_of_features(self, features):
        """Verifies that an exception is raised if 'features' has the wrong trailing dimension."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(f=None):
            qml.IQPEmbedding(features=f, wires=range(3))
            return [qml.expval(qml.PauliZ(w)) for w in range(3)]

        with pytest.raises(ValueError, match="Features must be of length"):
            circuit(f=features)

    @pytest.mark.parametrize("shape", [(2, 3, 4), ()])
    def test_exception_wrong_ndim(self, shape):
        """Verifies that an exception is raised if 'features' has the wrong number of dimensions."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(f=None):
            qml.IQPEmbedding(features=f, wires=range(3))
            return [qml.expval(qml.PauliZ(w)) for w in range(3)]

        features = np.ones(shape)

        with pytest.raises(ValueError, match="Features must be a one-dimensional"):
            circuit(f=features)

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.IQPEmbedding(np.array([1, 2]), wires=[0, 1], id="a")
        assert template.id == "a"


def circuit_template(features):
    qml.IQPEmbedding(features, range(2))
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed_lists(features):
    qml.Hadamard(wires=0)
    qml.RZ(features[0], wires=0)
    qml.Hadamard(wires=1)
    qml.RZ(features[1], wires=1)
    qml.MultiRZ(features[0] * features[1], wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(features):
    qml.Hadamard(wires=0)
    qml.RZ(features[..., 0], wires=0)
    qml.Hadamard(wires=1)
    qml.RZ(features[..., 1], wires=1)
    qml.MultiRZ(features[..., 0] * features[..., 1], wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    @pytest.mark.parametrize(
        "features",
        [[0.1, -1.3], [np.array([0.5, 2.0]), np.array([1.2, 0.6]), np.array([-0.7, 0.3])]],
    )
    def test_list_and_tuples(self, tol, features):
        """Tests common iterables as inputs."""

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        res = circuit(features)
        if isinstance(features[0], np.ndarray):
            circuit2 = qml.QNode(circuit_decomposed, dev)
            features = np.array(features)  # circuit_decomposed does not work with broadcasting
        else:
            circuit2 = qml.QNode(circuit_decomposed_lists, dev)

        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(tuple(features))
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.autograd
    @pytest.mark.parametrize("features", [[0.1, -1.3], [[0.5, 2.0], [1.2, 0.6], [-0.7, 0.3]]])
    def test_autograd(self, tol, features):
        """Tests the autograd interface."""

        features = pnp.array(features, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qml.jacobian(circuit)
        grads = grad_fn(features)

        grad_fn2 = qml.jacobian(circuit2)
        grads2 = grad_fn2(features)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("features", [[0.1, -1.3], [[0.5, 2.0], [1.2, 0.6], [-0.7, 0.3]]])
    def test_jax(self, tol, features):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        features = jnp.array(features)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.jacobian(circuit)
        grads = grad_fn(features)

        grad_fn2 = jax.jacobian(circuit2)
        grads2 = grad_fn2(features)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("features", [[0.1, -1.3], [[0.5, 2.0], [1.2, 0.6], [-0.7, 0.3]]])
    def test_jax_jit(self, tol, features):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        features = jnp.array(features)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = jax.jit(circuit)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.jacobian(circuit)
        grads = grad_fn(features)

        grad_fn2 = jax.jacobian(circuit2)
        grads2 = grad_fn2(features)

        assert qml.math.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("features", [[0.1, -1.3], [[0.5, 2.0], [1.2, 0.6], [-0.7, 0.3]]])
    def test_tf(self, tol, features):
        """Tests the tf interface."""

        import tensorflow as tf

        features = tf.Variable(features)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            res = circuit(features)
        grads = tape.jacobian(res, [features])

        with tf.GradientTape() as tape2:
            res2 = circuit2(features)
        grads2 = tape2.jacobian(res2, [features])

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.torch
    @pytest.mark.parametrize("features", [[0.1, -1.3], [[0.5, 2.0], [1.2, 0.6], [-0.7, 0.3]]])
    def test_torch(self, tol, features):
        """Tests the torch interface."""

        import torch

        features = torch.tensor(features, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grads = torch.autograd.functional.jacobian(circuit, features)

        grads2 = torch.autograd.functional.jacobian(circuit2, features)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
