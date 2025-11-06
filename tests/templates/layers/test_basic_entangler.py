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
Unit tests for the BasicEntanglerLayers template.
"""
import numpy as np

# pylint: disable=too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as pnp


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""

    weights = np.random.random((1, 3))

    op = qml.BasicEntanglerLayers(weights, wires=range(3))
    qml.ops.functions.assert_valid(op)


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    QUEUES = [
        (1, (1, 1), ["RX"], [[0]]),
        (2, (1, 2), ["RX", "RX", "CNOT"], [[0], [1], [0, 1]]),
        (2, (2, 2), ["RX", "RX", "CNOT", "RX", "RX", "CNOT"], [[0], [1], [0, 1], [0], [1], [0, 1]]),
        (
            3,
            (1, 3),
            ["RX", "RX", "RX", "CNOT", "CNOT", "CNOT"],
            [[0], [1], [2], [0, 1], [1, 2], [2, 0]],
        ),
    ]

    @pytest.mark.parametrize("n_wires, weight_shape, expected_names, expected_wires", QUEUES)
    def test_expansion(self, n_wires, weight_shape, expected_names, expected_wires):
        """Checks the queue for the default settings."""

        weights = np.random.random(size=weight_shape)

        op = qml.BasicEntanglerLayers(weights, wires=range(n_wires))
        tape = qml.tape.QuantumScript(op.decomposition())

        for i, gate in enumerate(tape.operations):
            assert gate.name == expected_names[i]
            assert gate.wires.labels == tuple(expected_wires[i])

    @pytest.mark.parametrize("rotation", [qml.RY, qml.RZ])
    def test_rotation(self, rotation):
        """Checks that custom rotation gate is used."""

        weights = np.zeros(shape=(1, 2))

        op = qml.BasicEntanglerLayers(weights, wires=range(2), rotation=rotation)
        queue = op.decomposition()

        assert rotation in [type(gate) for gate in queue]

    @pytest.mark.parametrize(
        "weights, n_wires, target",
        [
            ([[np.pi]], 1, [-1]),
            ([[np.pi] * 2], 2, [-1, 1]),
            ([[np.pi] * 3], 3, [1, 1, -1]),
            ([[np.pi] * 4], 4, [-1, 1, -1, 1]),
        ],
    )
    def test_simple_target_outputs(self, weights, n_wires, target, tol):
        """Tests the result of the template for simple cases."""

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(weights):
            qml.BasicEntanglerLayers(weights, wires=range(n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        expectations = circuit(weights)
        np.testing.assert_allclose(expectations, target, atol=tol, rtol=0)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        weights = np.random.random(size=(1, 3))

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.BasicEntanglerLayers(weights, wires=range(3))
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.BasicEntanglerLayers(weights, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the weights shape is incorrect."""

        n_wires = 1
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(weights):
            qml.BasicEntanglerLayers(weights=weights, wires=range(n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="Weights tensor must be 2-dimensional"):
            circuit([1, 0])

        with pytest.raises(ValueError, match="Weights tensor must have last dimension of length"):
            circuit([[1, 0], [1, 0]])

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.BasicEntanglerLayers(np.array([[1]]), wires=[0], id="a")
        assert template.id == "a"


class TestAttributes:
    """Tests additional methods and attributes"""

    @pytest.mark.parametrize(
        "n_layers, n_wires, expected_shape",
        [
            (2, 3, (2, 3)),
            (2, 1, (2, 1)),
            (2, 2, (2, 2)),
        ],
    )
    def test_shape(self, n_layers, n_wires, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor"""

        shape = qml.BasicEntanglerLayers.shape(n_layers, n_wires)
        assert shape == expected_shape


def circuit_template(weights):
    qml.BasicEntanglerLayers(weights, range(3))
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(weights):
    qml.RX(weights[0, 0], wires=0)
    qml.RX(weights[0, 1], wires=1)
    qml.RX(weights[0, 2], wires=2)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        weights = np.random.random(size=(1, 3))
        weights = pnp.array(weights, requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

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

        weights = jnp.array(np.random.random(size=(1, 3)))

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax_jit(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(1, 3)))

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = jax.jit(circuit)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert qml.math.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        weights = tf.Variable(np.random.random(size=(1, 3)))

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

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

        weights = torch.tensor(np.random.random(size=(1, 3)), requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

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
