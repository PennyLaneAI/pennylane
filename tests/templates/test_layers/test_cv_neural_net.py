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
Unit tests for the CVNeuralNetLayers template.
"""
import numpy as np

# pylint: disable=too-few-public-methods,protected-access
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.devices import DefaultGaussian


class DummyDevice(DefaultGaussian):
    """Dummy Gaussian device to allow Kerr operations"""

    _operation_map = DefaultGaussian._operation_map.copy()
    _operation_map["Kerr"] = lambda *x, **y: np.identity(2)


def expected_shapes(n_layers, n_wires):
    # compute the expected shapes for a given number of wires
    n_if = n_wires * (n_wires - 1) // 2
    expected = (
        [(n_layers, n_if)] * 2
        + [(n_layers, n_wires)] * 3
        + [(n_layers, n_if)] * 2
        + [(n_layers, n_wires)] * 4
    )
    return expected


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    QUEUES = [
        (1, ["Rotation", "Squeezing", "Rotation", "Displacement", "Kerr"], [[0]] * 5),
        (
            2,
            [
                "Beamsplitter",  # Interferometer 1
                "Rotation",  # Interferometer 1
                "Rotation",  # Interferometer 1
                "Squeezing",
                "Squeezing",
                "Beamsplitter",  # Interferometer 2
                "Rotation",  # Interferometer 2
                "Rotation",  # Interferometer 2
                "Displacement",
                "Displacement",
                "Kerr",
                "Kerr",
            ],
            [[0, 1], [0], [1], [0], [1], [0, 1], [0], [1], [0], [1], [0], [1]],
        ),
    ]

    @pytest.mark.parametrize("n_wires, expected_names, expected_wires", QUEUES)
    def test_expansion(self, n_wires, expected_names, expected_wires):
        """Checks the queue for the default settings."""

        shapes = expected_shapes(1, n_wires)
        weights = [np.random.random(shape) for shape in shapes]

        op = qml.CVNeuralNetLayers(*weights, wires=range(n_wires))
        tape = qml.tape.QuantumScript(op.decomposition())

        i = 0
        for gate in tape.operations:
            if gate.name != "Interferometer":
                assert gate.name == expected_names[i]
                assert gate.wires.labels == tuple(expected_wires[i])
                i = i + 1
            else:
                for gate_inter in gate.decomposition():
                    assert gate_inter.name == expected_names[i]
                    assert gate_inter.wires.labels == tuple(expected_wires[i])
                    i = i + 1

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        shapes = expected_shapes(1, 3)
        weights = [np.random.random(shape) for shape in shapes]

        dev = DummyDevice(wires=3)
        dev2 = DummyDevice(wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.CVNeuralNetLayers(*weights, wires=range(3))
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.CVNeuralNetLayers(*weights, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev._state[0], dev2._state[0], atol=tol, rtol=0)
        assert np.allclose(dev._state[1], dev2._state[1], atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    def test_cvqnn_layers_exception_nlayers(self):
        """Check exception if inconsistent number of layers"""
        shapes = expected_shapes(1, 2)
        weights = [np.random.random(shape) for shape in shapes[:-1]]
        weights += [np.random.random((2, shapes[-1][1]))]

        dev = DummyDevice(wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.CVNeuralNetLayers(*weights, wires=range(2))
            return qml.expval(qml.QuadX(0))

        with pytest.raises(ValueError, match="The first dimension of all parameters"):
            circuit()

    def test_cvqnn_layers_exception_second_dim(self):
        """Check exception if wrong dimension of weights"""
        shapes = expected_shapes(1, 2)
        weights = [np.random.random(shape) for shape in shapes[:-1]]
        weights += [np.random.random((1, shapes[-1][1] - 1))]

        dev = DummyDevice(wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.CVNeuralNetLayers(*weights, wires=range(2))
            return qml.expval(qml.QuadX(0))

        with pytest.raises(ValueError, match="Got unexpected shape for one or more parameters"):
            circuit()

    def test_id(self):
        """Tests that the id attribute can be set."""
        shapes = expected_shapes(1, 2)
        weights = [np.random.random(shape) for shape in shapes]

        template = qml.CVNeuralNetLayers(*weights, wires=range(2), id="a")
        assert template.id == "a"


class TestAttributes:
    """Test methods and attributes."""

    @pytest.mark.parametrize(
        "n_layers, n_wires",
        [
            (2, 3),
            (2, 1),
            (2, 2),
        ],
    )
    def test_shapes(self, n_layers, n_wires, tol):
        """Test that the shape method returns the correct shapes for
        the weight tensors"""

        shapes = qml.CVNeuralNetLayers.shape(n_layers, n_wires)
        expected = expected_shapes(n_layers, n_wires)

        assert np.allclose(shapes, expected, atol=tol, rtol=0)


def circuit_template(*weights):
    qml.CVNeuralNetLayers(*weights, range(2))
    return qml.expval(qml.QuadX(0))


def circuit_decomposed(*weights):
    # Interferometer (replace with operation once this template is refactored)
    qml.Beamsplitter(weights[0][0, 0], weights[1][0, 0], wires=[0, 1])
    qml.Rotation(weights[2][0, 0], wires=0)
    qml.Rotation(weights[2][0, 1], wires=1)

    qml.Squeezing(weights[3][0, 0], weights[4][0, 0], wires=0)
    qml.Squeezing(weights[3][0, 1], weights[4][0, 1], wires=1)

    # Interferometer
    qml.Beamsplitter(weights[5][0, 0], weights[6][0, 0], wires=[0, 1])
    qml.Rotation(weights[7][0, 0], wires=0)
    qml.Rotation(weights[7][0, 1], wires=1)

    qml.Displacement(weights[8][0, 0], weights[9][0, 0], wires=0)
    qml.Displacement(weights[8][0, 1], weights[9][0, 1], wires=1)
    qml.Kerr(weights[10][0, 0], wires=0)
    qml.Kerr(weights[10][0, 1], wires=1)
    return qml.expval(qml.QuadX(0))


def test_adjoint():
    """Test that the adjoint method works"""
    dev = DummyDevice(wires=2)

    shapes = qml.CVNeuralNetLayers.shape(n_layers=1, n_wires=2)
    weights = [np.random.random(shape) for shape in shapes]

    @qml.qnode(dev)
    def circuit():
        qml.CVNeuralNetLayers(*weights, wires=[0, 1])
        qml.adjoint(qml.CVNeuralNetLayers)(*weights, wires=[0, 1])
        return qml.expval(qml.QuadX(0))

    assert qml.math.allclose(circuit(), 0)


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    def test_list_and_tuples(self, tol):
        """Tests common iterables as inputs."""

        shapes = expected_shapes(1, 2)
        weights = [np.random.random(shape) for shape in shapes]

        dev = DummyDevice(wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(*weights)
        res2 = circuit2(*weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        weights_tuple = tuple(w for w in weights)
        res = circuit(*weights_tuple)
        res2 = circuit2(*weights_tuple)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        shapes = expected_shapes(1, 2)
        weights = [np.random.random(shape) for shape in shapes]
        weights = [pnp.array(w, requires_grad=True) for w in weights]

        dev = DummyDevice(wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(*weights)
        res2 = circuit2(*weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(*weights)

        grad_fn2 = qml.grad(circuit2)
        grads2 = grad_fn2(*weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        shapes = expected_shapes(1, 2)
        weights = [np.random.random(shape) for shape in shapes]
        weights = [jnp.array(w) for w in weights]

        dev = DummyDevice(wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(*weights)
        res2 = circuit2(*weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(*weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(*weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax_jit(self, tol):
        """Tests jit within the jax interface."""

        import jax
        import jax.numpy as jnp

        shapes = expected_shapes(1, 2)
        weights = [np.random.random(shape) for shape in shapes]
        weights = [jnp.array(w) for w in weights]

        dev = DummyDevice(wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = jax.jit(circuit)

        res = circuit(*weights)
        res2 = circuit2(*weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(*weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(*weights)

        assert qml.math.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        shapes = expected_shapes(1, 2)
        weights = [np.random.random(shape) for shape in shapes]
        weights = [tf.Variable(w) for w in weights]

        dev = DummyDevice(wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(*weights)
        res2 = circuit2(*weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            res = circuit(*weights)
        grads = tape.gradient(res, [*weights])

        with tf.GradientTape() as tape2:
            res2 = circuit2(*weights)
        grads2 = tape2.gradient(res2, [*weights])

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests the torch interface."""

        import torch

        shapes = expected_shapes(1, 2)
        weights = [np.random.random(size=shape) for shape in shapes]
        weights = [torch.tensor(w, requires_grad=True) for w in weights]

        dev = DummyDevice(wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(*weights)
        res2 = circuit2(*weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(*weights)
        res.backward()
        grads = [w.grad for w in weights]

        res2 = circuit2(*weights)
        res2.backward()
        grads2 = [w.grad for w in weights]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
