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
Unit tests for the IQPEmbedding template.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    QUEUES = [(1, ['Hadamard', 'RZ'], [[0], [0]]),
              (2, ['Hadamard', 'RZ', 'Hadamard', 'RZ', 'MultiRZ'], [[0], [0], [1], [1], [0, 1]]),
              (3, ['Hadamard', 'RZ', 'Hadamard', 'RZ', 'Hadamard', 'RZ',
                   'MultiRZ', 'MultiRZ', 'MultiRZ'], [[0], [0], [1], [1], [2], [2], [0, 1], [0, 2], [1, 2]])]

    @pytest.mark.parametrize("n_wires, expected_names, expected_wires", QUEUES)
    def test_expansion(self, n_wires, expected_names, expected_wires):
        """Checks the queue for the default settings."""

        features = list(range(n_wires))

        op = qml.templates.IQPEmbedding(features, wires=range(n_wires))
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
            qml.templates.IQPEmbedding(features, wires=range(3))
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.templates.IQPEmbedding(features, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


class TestParameters:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize('features', [[1., 2.],
                                          [1., 2., 3., 4.],
                                          [[1., 1.], [2., 2.], [3., 3.]]])
    def test_exception_wrong_number_of_features(self, features):
        """Verifies that an exception is raised if 'feature' has the wrong shape."""

        dev = qml.device('default.qubit', wires=3)

        @qml.qnode(dev)
        def circuit(f=None):
            qml.templates.IQPEmbedding(features=f, wires=range(3))
            return [qml.expval(qml.PauliZ(w)) for w in range(3)]

        with pytest.raises(ValueError, match="Features must be"):
            circuit(f=features)


def circuit_template(features):
    qml.templates.IQPEmbedding(features, range(2))
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(features):
    qml.Hadamard(wires=0)
    qml.RZ(features[0], wires=0)
    qml.Hadamard(wires=1)
    qml.RZ(features[1], wires=1)
    qml.MultiRZ(features[0]*features[1], wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


class TestGradients:
    """Tests that the gradient is computed correctly in all interfaces."""

    def test_autograd(self, tol):
        """Tests that gradients of template and decomposed circuit
        are the same in the autograd interface."""

        features = np.random.random(size=(2,))
        features = pnp.array(features, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(features)

        grad_fn2 = qml.grad(circuit2)
        grads2 = grad_fn2(features)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    def test_jax(self, tol, skip_if_no_jax_support):
        """Tests that gradients of template and decomposed circuit
        are the same in the jax interface."""

        import jax
        import jax.numpy as jnp

        features = jnp.array(np.random.random(size=(2,)))

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev, interface="jax")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="jax")

        grad_fn = jax.grad(circuit)
        grads = grad_fn(features)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(features)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    def test_tf(self, tol, skip_if_no_tf_support):
        """Tests that gradients of template and decomposed circuit
        are the same in the tf interface."""

        import tensorflow as tf

        features = tf.Variable(np.random.random(size=(2,)))

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev, interface="tf")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="tf")

        with tf.GradientTape() as tape:
            res = circuit(features)
        grads = tape.gradient(res, [features])

        with tf.GradientTape() as tape2:
            res2 = circuit2(features)
        grads2 = tape2.gradient(res2, [features])

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    def test_torch(self, tol, skip_if_no_torch_support):
        """Tests that gradients of template and decomposed circuit
        are the same in the torch interface."""

        import torch

        features = torch.tensor(np.random.random(size=(2,)), requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev, interface="torch")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="torch")

        res = circuit(features)
        res.backward()
        grads = [features.grad]

        res2 = circuit2(features)
        res2.backward()
        grads2 = [features.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
