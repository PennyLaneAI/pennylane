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
Unit tests for the StronglyEntanglingLayers template.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    QUEUES = [
        (1, (1, 1, 3), ["Rot"], [[0]]),
        (2, (1, 2, 3), ["Rot", "Rot", "CNOT", "CNOT"], [[0], [1], [0, 1], [1, 0]]),
        (
            2,
            (2, 2, 3),
            ["Rot", "Rot", "CNOT", "CNOT", "Rot", "Rot", "CNOT", "CNOT"],
            [[0], [1], [0, 1], [1, 0], [0], [1], [0, 1], [1, 0]],
        ),
        (
            3,
            (1, 3, 3),
            ["Rot", "Rot", "Rot", "CNOT", "CNOT", "CNOT"],
            [[0], [1], [2], [0, 1], [1, 2], [2, 0]],
        ),
    ]

    @pytest.mark.parametrize("n_wires, weight_shape, expected_names, expected_wires", QUEUES)
    def test_expansion(self, n_wires, weight_shape, expected_names, expected_wires):
        """Checks the queue for the default settings."""

        weights = np.random.random(size=weight_shape)

        op = qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
        tape = op.expand()

        for i, gate in enumerate(tape.operations):
            assert gate.name == expected_names[i]
            assert gate.wires.labels == tuple(expected_wires[i])

    @pytest.mark.parametrize("n_layers, n_wires", [(2, 2), (1, 3), (2, 4)])
    def test_uses_correct_imprimitive(self, n_layers, n_wires):
        """Test that correct number of entanglers are used in the circuit."""

        weights = np.random.randn(n_layers, n_wires, 3)

        op = qml.StronglyEntanglingLayers(weights=weights, wires=range(n_wires), imprimitive=qml.CZ)
        ops = op.expand().operations

        gate_names = [gate.name for gate in ops]
        assert gate_names.count("CZ") == n_wires * n_layers

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        weights = np.random.random(size=(1, 3, 3))

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.StronglyEntanglingLayers(weights, wires=range(3))
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.StronglyEntanglingLayers(weights, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "n_layers, n_wires, ranges", [(2, 2, [1, 1]), (1, 3, [2]), (4, 4, [2, 3, 1, 3])]
    )
    def test_custom_range_sequence(self, n_layers, n_wires, ranges):
        """Test that correct sequence of custom ranges are used in the circuit."""

        weights = np.random.randn(n_layers, n_wires, 3)

        op = qml.StronglyEntanglingLayers(weights=weights, wires=range(n_wires), ranges=ranges)
        ops = op.expand().operations

        gate_wires = [gate.wires.labels for gate in ops]
        range_idx = 0
        for idx, i in enumerate(gate_wires):
            if idx % (n_wires * 2) // n_wires == 1:
                expected_wire = (
                    idx % n_wires,
                    (ranges[range_idx % len(ranges)] + idx % n_wires) % n_wires,
                )
                assert i == expected_wire
                if idx % n_wires == n_wires - 1:
                    range_idx += 1


class TestInputs:
    """Test inputs and pre-processing."""

    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the
        number of dimensions of features is incorrect."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights, ranges=None):
            qml.StronglyEntanglingLayers(weights, wires=range(2), ranges=ranges)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Weights tensor must have second dimension"):
            weights = np.random.randn(2, 1, 3)
            circuit(weights)

        with pytest.raises(ValueError, match="Weights tensor must have third dimension"):
            weights = np.random.randn(2, 2, 1)
            circuit(weights)

        with pytest.raises(ValueError, match="Range sequence must be of length"):
            weights = np.random.randn(2, 2, 3)
            circuit(weights, ranges=[1])

    def test_exception_wrong_ranges(self):
        """Verifies that exception is raised if the
        value of ranges is incorrect."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights, ranges=None):
            qml.StronglyEntanglingLayers(weights, wires=range(2), ranges=ranges)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Ranges must not be zero nor"):
            weights = np.random.randn(1, 2, 3)
            circuit(weights, ranges=[0])

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.StronglyEntanglingLayers(np.array([[[1, 2, 3]]]), wires=[0], id="a")
        assert template.id == "a"


class TestAttributes:
    """Tests additional methods and attributes"""

    @pytest.mark.parametrize(
        "n_layers, n_wires, expected_shape",
        [
            (2, 3, (2, 3, 3)),
            (2, 1, (2, 1, 3)),
            (2, 2, (2, 2, 3)),
        ],
    )
    def test_shape(self, n_layers, n_wires, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor"""

        shape = qml.StronglyEntanglingLayers.shape(n_layers, n_wires)
        assert shape == expected_shape


def circuit_template(weights):
    qml.StronglyEntanglingLayers(weights, range(3))
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(weights):
    qml.Rot(weights[0, 0, 0], weights[0, 0, 1], weights[0, 0, 2], wires=0)
    qml.Rot(weights[0, 1, 0], weights[0, 1, 1], weights[0, 1, 2], wires=1)
    qml.Rot(weights[0, 2, 0], weights[0, 2, 1], weights[0, 2, 2], wires=2)
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

        weights = np.random.random(size=(1, 3, 3))
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

        weights = jnp.array(np.random.random(size=(1, 3, 3)))

        dev = qml.device("default.qubit", wires=3)

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

        weights = tf.Variable(np.random.random(size=(1, 3, 3)))

        dev = qml.device("default.qubit", wires=3)

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

        weights = torch.tensor(np.random.random(size=(1, 3, 3)), requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

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
