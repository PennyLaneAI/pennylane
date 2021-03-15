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
Unit tests for QAOAEmbedding template.
"""
import pytest

import numpy as np
import pennylane as qml


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    QUEUES = [(1, (1, 1), [qml.RX, qml.RY, qml.RX]),
              (2, (1, 3), [qml.RX, qml.RX, qml.MultiRZ, qml.RY, qml.RY, qml.RX, qml.RX]),
              (2, (2, 3), [qml.RX, qml.RX, qml.MultiRZ, qml.RY, qml.RY, qml.RX, qml.RX,
                           qml.MultiRZ, qml.RY, qml.RY, qml.RX, qml.RX]),
              (3, (1, 6), [qml.RX, qml.RX, qml.RX, qml.MultiRZ, qml.MultiRZ, qml.MultiRZ,
                           qml.RY, qml.RY, qml.RY, qml.RX, qml.RX, qml.RX])]

    @pytest.mark.parametrize('n_wires, weight_shape, expected_types', QUEUES)
    def test_queue(self, n_wires, weight_shape, expected_types):
        """Checks the queue for the default settings."""

        features = list(range(n_wires))
        weights = np.zeros(shape=weight_shape)

        op = qml.templates.QAOAEmbedding(features, weights, wires=range(n_wires))
        tape = op.expand()

        for i, gate in enumerate(tape.operations):
            assert type(gate) == expected_types[i]

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
            qml.templates.QAOAEmbedding(features=x, weights=weights, wires=range(n_subsystems))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [1, -1, 0, 1, 1]
        assert np.allclose(res, target[:n_subsystems], atol=tol, rtol=0)

    @pytest.mark.parametrize('weights, target', [([[np.pi, 0, 0]], [1, 1]),
                                                 ([[np.pi / 2, 0, 0]], [0, 0]),
                                                 ([[0, 0, 0]], [-1, -1])])
    def test_output_zz(self, weights, target, tol):
        """Checks the output if the features and entangler weights are nonzero,
        which makes the circuit only depend on the ZZ gate."""

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.QAOAEmbedding(features=x, weights=weights, wires=range(2))
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]

        res = circuit(x=[np.pi/2, np.pi/2])

        assert np.allclose(res, target, atol=tol, rtol=0)


class TestHyperparameters:
    """Tests that the template's hyperparameters have the desired effect."""

    @pytest.mark.parametrize('n_subsystems, weights, target', [(1, [[np.pi / 2]], [0]),
                                                               (2, [[1, np.pi / 2, np.pi / 4]], [0, 1 / np.sqrt(2)]),
                                                               (3, [[0, 0, 0, np.pi, np.pi / 2, np.pi / 4]],
                                                                [-1, 0, 1 / np.sqrt(2)])])
    def test_output_local_field_ry(self, n_subsystems, weights, target, tol):
        """Checks the output if the features are zero for RY local fields."""

        features = np.zeros(shape=(n_subsystems,))
        dev = qml.device('default.qubit', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.QAOAEmbedding(features=x, weights=weights, wires=range(n_subsystems), local_field='Y')
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        assert np.allclose(res, target, atol=tol, rtol=0)

    @pytest.mark.parametrize('n_subsystems, weights, target', [(1, [[np.pi / 2]], [0]),
                                                               (2, [[1, np.pi / 2, np.pi / 4]], [0, 1 / np.sqrt(2)]),
                                                               (3, [[0, 0, 0, np.pi, np.pi / 2, np.pi / 4]],
                                                                [-1, 0, 1 / np.sqrt(2)])])
    def test_output_local_field_rx(self, n_subsystems, weights, target, tol):
        """Checks the output if the features are zero for RX local fields."""

        features = np.zeros(shape=(n_subsystems,))
        dev = qml.device('default.qubit', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.QAOAEmbedding(features=x, weights=weights, wires=range(n_subsystems), local_field='X')
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        assert np.allclose(res, target, atol=tol, rtol=0)

    @pytest.mark.parametrize('n_subsystems, weights, target', [(1, [[np.pi / 2]], [1]),
                                                               (2, [[1, np.pi / 2, np.pi / 4]], [1, 1]),
                                                               (3, [[0, 0, 0, np.pi, np.pi / 2, np.pi / 4]], [1, 1, 1])])
    def test_output_local_field_rz(self, n_subsystems, weights, target, tol):
        """Checks the output if the features are zero for RZ local fields."""

        features = np.zeros(shape=(n_subsystems,))
        dev = qml.device('default.qubit', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.QAOAEmbedding(features=x, weights=weights, wires=range(n_subsystems), local_field='Z')
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        assert np.allclose(res, target, atol=tol, rtol=0)

    @pytest.mark.parametrize('n_wires, features, weights, target', [(2, [0], [[0, 0, np.pi / 2]], [1, 0]),
                                                                    (3, [0, 0], [[0, 0, 0, 0, 0, np.pi / 2]],
                                                                     [1, 1, 0])])
    def test_state_more_qubits_than_features(self, n_wires, features, weights, target, tol):
        """Checks the state is correct if there are more qubits than features."""

        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.QAOAEmbedding(features=x, weights=weights, wires=range(n_wires), local_field='Z')
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        res = circuit(x=features)
        assert np.allclose(res, target, atol=tol, rtol=0)

    def test_exception_fewer_wires_than_features(self, ):
        """Verifies that exception raised if there are fewer
           wires than features."""

        features = [0, 0, 0, 0]
        n_wires = 1
        weights = np.zeros(shape=(1, 2 * n_wires))
        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.QAOAEmbedding(features=x, weights=weights, wires=range(n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="Features must be of "):
            circuit(x=features)

    def test_exception_wrongrot(self):
        """Verifies exception raised if the
        rotation strategy is unknown."""

        n_wires = 1
        weights = np.zeros(shape=(1, 1))
        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.QAOAEmbedding(features=x, weights=weights, wires=range(n_wires), local_field='A')
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="did not recognize"):
            circuit(x=[1])

    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the
        number of dimensions of features is incorrect."""
        n_wires = 1
        weights = np.zeros(shape=(1, 1))
        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.QAOAEmbedding(features=x, weights=weights, wires=range(n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="Features must be a one-dimensional"):
            circuit(x=[[1], [0]])


class TestGradients:
    """Tests that the gradient is computed correctly in all three interfaces."""

    def circuit(self, features, weights):
        qml.templates.QAOAEmbedding(features, weights, range(2))
        return qml.expval(qml.PauliZ(0))

    def circuit_decomposed(self, features, weights):
        qml.RX(features[0], wires=0)
        qml.RX(features[1], wires=1)
        qml.MultiRZ(weights[0, 0], wires=[0, 1])
        qml.RY(weights[0, 1], wires=0)
        qml.RY(weights[0, 2], wires=1)
        qml.RX(features[0], wires=0)
        qml.RX(features[1], wires=1)
        return qml.expval(qml.PauliZ(0))

    def test_autograd(self, tol):
        """Tests that gradients of template and decomposed circuit
        are the same in the autograd interface."""

        features = np.random.random(size=(2, ))
        weights = np.random.random(size=(1, 3))

        dev = qml.device('default.qubit', wires=2)

        circuit = qml.QNode(self.circuit, dev)
        circuit2 = qml.QNode(self.circuit_decomposed, dev)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(features, weights)

        grad_fn2 = qml.grad(circuit2)
        grads2 = grad_fn2(features, weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
        assert np.allclose(grads[1], grads2[1], atol=tol, rtol=0)

    def test_jax(self, tol, skip_if_no_jax_support):
        """Tests that gradients of template and decomposed circuit
        are the same in the jax interface."""

        import jax
        import jax.numpy as jnp

        features = jnp.array(np.random.random(size=(2, )))
        weights = jnp.array(np.random.random(size=(1, 3)))

        dev = qml.device('default.qubit', wires=2)

        circuit = qml.QNode(self.circuit, dev, interface='jax')
        circuit2 = qml.QNode(self.circuit_decomposed, dev, interface='jax')

        grad_fn = jax.grad(circuit)
        grads = grad_fn(features, weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(features, weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
        assert np.allclose(grads[1], grads2[1], atol=tol, rtol=0)

    def test_tf(self, tol, skip_if_no_tf_support):
        """Tests that gradients of template and decomposed circuit
        are the same in the tf interface."""

        import tensorflow as tf

        features = tf.Variable(np.random.random(size=(2, )))
        weights = tf.Variable(np.random.random(size=(1, 3)))

        dev = qml.device('default.qubit', wires=2)

        circuit = qml.QNode(self.circuit, dev, interface='tf')
        circuit2 = qml.QNode(self.circuit_decomposed, dev, interface='tf')

        with tf.GradientTape() as tape:
            res = circuit(features, weights)
        grads = tape.gradient(res, [features, weights])

        with tf.GradientTape() as tape2:
            res2 = circuit2(features, weights)
        grads2 = tape2.gradient(res2, [features, weights])

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
        assert np.allclose(grads[1], grads2[1], atol=tol, rtol=0)

    def test_torch(self, tol, skip_if_no_torch_support):
        """Tests that gradients of template and decomposed circuit
        are the same in the torch interface."""

        import torch

        features = torch.tensor(np.random.random(size=(2, )), requires_grad=True)
        weights = torch.tensor(np.random.random(size=(1, 3)), requires_grad=True)

        dev = qml.device('default.qubit', wires=2)

        circuit = qml.QNode(self.circuit, dev, interface='torch')
        circuit2 = qml.QNode(self.circuit_decomposed, dev, interface='torch')

        res = circuit(features, weights)
        res.backward()
        grads = [features.grad, weights.grad]

        res2 = circuit2(features, weights)
        res2.backward()
        grads2 = [features.grad, weights.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
        assert np.allclose(grads[1], grads2[1], atol=tol, rtol=0)
