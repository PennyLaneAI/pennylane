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
Tests for the QAOAEmbedding template.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    QUEUES = [
        (1, (1, 1), ["RX", "RY", "RX"]),
        (2, (1, 3), ["RX", "RX", "MultiRZ", "RY", "RY", "RX", "RX"]),
        (
            2,
            (2, 3),
            ["RX", "RX", "MultiRZ", "RY", "RY", "RX", "RX", "MultiRZ", "RY", "RY", "RX", "RX"],
        ),
        (
            3,
            (1, 6),
            ["RX", "RX", "RX", "MultiRZ", "MultiRZ", "MultiRZ", "RY", "RY", "RY", "RX", "RX", "RX"],
        ),
    ]

    @pytest.mark.parametrize("n_wires, weight_shape, expected_names", QUEUES)
    def test_expansion(self, n_wires, weight_shape, expected_names):
        """Checks the queue for the default settings."""

        features = list(range(n_wires))
        weights = np.zeros(shape=weight_shape)

        op = qml.QAOAEmbedding(features, weights, wires=range(n_wires))
        tape = op.expand()

        for i, gate in enumerate(tape.operations):
            assert gate.name == expected_names[i]

    @pytest.mark.parametrize("local_field", ["X", "Y", "Z"])
    def test_local_field(self, local_field):
        """Checks that custom local field is used."""

        get_name = {"X": "RX", "Y": "RY", "Z": "RZ"}

        features = list(range(2))
        weights = np.zeros(shape=(1, 3))

        op = qml.QAOAEmbedding(features, weights, wires=range(2), local_field=local_field)
        tape = op.expand()
        gate_names = [gate.name for gate in tape.operations]

        assert gate_names[3] == get_name[local_field]
        assert gate_names[4] == get_name[local_field]

    def test_exception_wrongrot(self):
        """Verifies exception raised if the
        rotation strategy is unknown."""

        n_wires = 1
        weights = np.zeros(shape=(1, 1))
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.QAOAEmbedding(features=x, weights=weights, wires=range(n_wires), local_field="A")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="did not recognize"):
            circuit(x=[1])

    def test_state_zero_weights(self, qubit_device, n_subsystems, tol):
        """Checks the state is correct if the weights are zero."""

        features = [np.pi, np.pi / 2, np.pi / 4, 0]
        if n_subsystems == 1:
            shp = (1, 1)
        elif n_subsystems == 2:
            shp = (1, 3)
        else:
            shp = (1, 2 * n_subsystems)

        weights = np.zeros(shape=shp)

        @qml.qnode(qubit_device)
        def circuit(x=None):
            qml.QAOAEmbedding(features=x, weights=weights, wires=range(n_subsystems))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [1, -1, 0, 1, 1]
        assert np.allclose(res, target[:n_subsystems], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "weights, target",
        [([[np.pi, 0, 0]], [1, 1]), ([[np.pi / 2, 0, 0]], [0, 0]), ([[0, 0, 0]], [-1, -1])],
    )
    def test_output_zz(self, weights, target, tol):
        """Checks the output if the features and entangler weights are nonzero,
        which makes the circuit only depend on the ZZ gate."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.QAOAEmbedding(features=x, weights=weights, wires=range(2))
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]

        res = circuit(x=[np.pi / 2, np.pi / 2])

        assert np.allclose(res, target, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "n_wires, features, weights, target",
        [
            (2, [0], [[0, 0, np.pi / 2]], [1, 0]),
            (3, [0, 0], [[0, 0, 0, 0, 0, np.pi / 2]], [1, 1, 0]),
        ],
    )
    def test_state_more_qubits_than_features(self, n_wires, features, weights, target, tol):
        """Checks the state is correct if there are more qubits than features."""

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.QAOAEmbedding(features=x, weights=weights, wires=range(n_wires), local_field="Z")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        res = circuit(x=features)
        assert np.allclose(res, target, atol=tol, rtol=0)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        weights = np.random.random(size=(1, 6))
        features = np.random.random(size=(3,))

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.QAOAEmbedding(features, weights, wires=range(3))
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.QAOAEmbedding(features, weights, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    def test_exception_fewer_qubits_than_features(
        self,
    ):
        """Verifies that exception raised if there are fewer
        wires than features."""

        features = [0, 0, 0, 0]
        n_wires = 1
        weights = np.zeros(shape=(1, 2 * n_wires))
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.QAOAEmbedding(features=x, weights=weights, wires=range(n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="Features must be of "):
            circuit(x=features)

    def test_exception_wrong_feature_shape(self):
        """Verifies that exception is raised if the shape of features is incorrect."""
        n_wires = 1
        weights = np.zeros(shape=(1, 1))
        features = np.zeros(shape=(2, 1))
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QAOAEmbedding(features, weights, wires=range(n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="Features must be a one-dimensional"):
            circuit()

    @pytest.mark.parametrize(
        "weights, n_wires",
        [(np.zeros(shape=(1, 2)), 1), (np.zeros(shape=(1, 4)), 2), (np.zeros(shape=(1, 3)), 3)],
    )
    def test_exception_wrong_weight_shape(self, weights, n_wires):
        """Verifies that exception is raised if the shape of weights is incorrect."""
        features = np.zeros(shape=(n_wires,))
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QAOAEmbedding(features, weights, wires=range(n_wires))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Weights tensor must be of shape"):
            circuit()

    @pytest.mark.parametrize(
        "n_layers, n_wires, expected_shape",
        [
            (2, 3, (2, 6)),
            (2, 1, (2, 1)),
            (2, 2, (2, 3)),
        ],
    )
    def test_shape(self, n_layers, n_wires, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor"""

        shape = qml.QAOAEmbedding.shape(n_layers, n_wires)
        assert shape == expected_shape

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.QAOAEmbedding(np.array([0]), weights=np.array([[0]]), wires=[0], id="a")
        assert template.id == "a"


def circuit_template(features, weights):
    qml.QAOAEmbedding(features, weights, range(2))
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(features, weights):
    qml.RX(features[0], wires=0)
    qml.RX(features[1], wires=1)
    qml.MultiRZ(weights[0, 0], wires=[0, 1])
    qml.RY(weights[0, 1], wires=0)
    qml.RY(weights[0, 2], wires=1)
    qml.RX(features[0], wires=0)
    qml.RX(features[1], wires=1)
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        features = np.random.random(size=(2,))
        features = pnp.array(features, requires_grad=True)

        weights = np.random.random(size=(1, 3))
        weights = pnp.array(weights, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features, weights)
        res2 = circuit2(features, weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(features, weights)

        grad_fn2 = qml.grad(circuit2)
        grads2 = grad_fn2(features, weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
        assert np.allclose(grads[1], grads2[1], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        features = jnp.array(np.random.random(size=(2,)))
        weights = jnp.array(np.random.random(size=(1, 3)))

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev, interface="jax")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="jax")

        res = circuit(features, weights)
        res2 = circuit2(features, weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(features, weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(features, weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
        assert np.allclose(grads[1], grads2[1], atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        features = tf.Variable(np.random.random(size=(2,)))
        weights = tf.Variable(np.random.random(size=(1, 3)))

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev, interface="tf")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="tf")

        res = circuit(features, weights)
        res2 = circuit2(features, weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            res = circuit(features, weights)
        grads = tape.gradient(res, [features, weights])

        with tf.GradientTape() as tape2:
            res2 = circuit2(features, weights)
        grads2 = tape2.gradient(res2, [features, weights])

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
        assert np.allclose(grads[1], grads2[1], atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests the torch interface."""

        import torch

        features = torch.tensor(np.random.random(size=(2,)), requires_grad=True)
        weights = torch.tensor(np.random.random(size=(1, 3)), requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev, interface="torch")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="torch")

        res = circuit(features, weights)
        res2 = circuit2(features, weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(features, weights)
        res.backward()
        grads = [features.grad, weights.grad]

        res2 = circuit2(features, weights)
        res2.backward()
        grads2 = [features.grad, weights.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
        assert np.allclose(grads[1], grads2[1], atol=tol, rtol=0)
