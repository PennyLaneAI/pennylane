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
Tests for the AngleEmbedding template.
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""
    op = qml.AngleEmbedding(features=[1.0, 2.0, 3.0], wires=range(3), rotation="Z")
    qml.ops.functions.assert_valid(op)


def test_repr():
    """Test the custom repr for angle embedding."""
    op = qml.AngleEmbedding(features=[1, 2, 3], wires=range(3), rotation="Z")
    expected = "AngleEmbedding([1 2 3], wires=[0, 1, 2], rotation=Z)"
    assert repr(op) == expected


# pylint: disable=protected-access
def test_flatten_unflatten():
    """Test the _flatten and _unflatten methods."""
    wires = qml.wires.Wires((0, 1, 2))
    op = qml.AngleEmbedding(features=[1, 2, 3], wires=wires, rotation="Z")

    data, metadata = op._flatten()
    assert data == op.data
    assert len(metadata) == 2
    assert metadata[0] == wires
    assert metadata[1] == (("rotation", "Z"),)

    new_op = type(op)._unflatten(*op._flatten())
    qml.assert_equal(op, new_op)
    assert op is not new_op


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize("features", [[1, 1, 1], [1, 1]])
    def test_expansion(self, features):
        """Checks the queue for the default settings."""

        op = qml.AngleEmbedding(features=features, wires=range(4))
        tape = qml.tape.QuantumScript(op.decomposition())

        assert len(tape.operations) == len(features)
        for gate in tape.operations:
            assert gate.name == "RX"
            assert gate.parameters[0] == 1

    def test_expansion_broadcasted(self):
        """Checks the queue for the default settings."""

        features = np.ones((5, 3))

        op = qml.AngleEmbedding(features=features, wires=range(4))
        assert op.batch_size == 5
        tape = qml.tape.QuantumScript(op.decomposition())

        assert len(tape.operations) == 3
        for gate in tape.operations:
            assert gate.name == "RX"
            assert gate.batch_size == 5
            assert qml.math.allclose(gate.parameters[0], np.ones(5))

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_rotations(self, rotation):
        """Checks the queue for the specified rotation settings."""

        op = qml.AngleEmbedding(features=[1, 1, 1], wires=range(4), rotation=rotation)
        tape = qml.tape.QuantumScript(op.decomposition())

        for gate in tape.operations:
            assert gate.name == "R" + rotation

    def test_state(
        self,
    ):
        """Checks the state produced using the rotation='X' strategy."""

        features = [np.pi / 2, np.pi / 2, np.pi / 4, 0]
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.AngleEmbedding(features=x, wires=range(4), rotation="X")
            qml.PauliX(wires=0)
            qml.AngleEmbedding(features=x, wires=range(4), rotation="X")
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        res = circuit(x=features)
        target = [1, -1, 0, 1]

        assert np.allclose(res, target)

    def test_fewer_features(self):
        """Tests fewer features than rotation gates."""

        features = [np.pi / 2, np.pi / 2, 0, np.pi / 2]

        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.AngleEmbedding(features=x, wires=range(5))
            return [qml.expval(qml.PauliZ(i)) for i in range(5)]

        res = circuit(x=features)
        target = [0, 0, 1, 0, 1]
        assert np.allclose(res, target)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        features = np.random.random(size=(3,))

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.AngleEmbedding(features, wires=range(3))
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.AngleEmbedding(features, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    def test_exception_fewer_rotations(self):
        """Verifies that exception is raised if there are fewer
        rotation gates than features."""

        features = [0, 0, 1, 0]
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.AngleEmbedding(features=x, wires=range(3))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Features must be of"):
            circuit(x=features)

    def test_exception_wrongrot(self):
        """Verifies that exception is raised if the
        rotation strategy is unknown."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.AngleEmbedding(features=x, wires=range(1), rotation="A")
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Rotation option"):
            circuit(x=[1])

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.AngleEmbedding(np.array([1, 2]), wires=[0, 1], id="a")
        assert template.id == "a"


def circuit_template(features):
    qml.AngleEmbedding(features, range(3))
    qml.CNOT(wires=[2, 1])
    qml.CNOT(wires=[1, 0])
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(features):
    qml.RX(features[0], wires=0)
    qml.RX(features[1], wires=1)
    qml.RX(features[2], wires=2)
    qml.CNOT(wires=[2, 1])
    qml.CNOT(wires=[1, 0])
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    def test_list_and_tuples(self, tol):
        """Tests common iterables as inputs."""

        features = [1.0, 1.0, 1.0]

        dev = qml.device("default.qubit", wires=3)

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

        features = pnp.array([1.0, 1.0, 1.0], requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

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

        features = jnp.array([1.0, 1.0, 1.0])

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(features)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(features)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax_jit(self, tol):
        """Tests jit with the jax interface."""

        import jax
        import jax.numpy as jnp

        features = jnp.array([1.0, 1.0, 1.0])

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = jax.jit(circuit)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(features)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(features)

        assert qml.math.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        features = tf.Variable([1.0, 1.0, 1.0])

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

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

        features = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

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
