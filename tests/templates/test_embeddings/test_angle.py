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
Unit tests for the AngleEmbedding template.
"""
import pytest
import numpy as np
from pennylane import numpy as pnp
import pennylane as qml


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize("features", [[1, 1, 1], [1, 1]])
    def test_expansion(self, features):
        """Checks the queue for the default settings."""

        op = qml.templates.AngleEmbedding(features=features, wires=range(4))
        tape = op.expand()

        assert len(tape.operations) == len(features)
        for gate in tape.operations:
            assert gate.name == "RX"

    def test_state_rotx(self, qubit_device, n_subsystems):
        """Checks the state produced using the rotation='X' strategy."""

        features = [np.pi / 2, np.pi / 2, np.pi / 4, 0]

        @qml.qnode(qubit_device)
        def circuit(x=None):
            qml.templates.AngleEmbedding(features=x, wires=range(n_subsystems), rotation="X")
            qml.PauliX(wires=0)
            qml.templates.AngleEmbedding(features=x, wires=range(n_subsystems), rotation="X")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [1, -1, 0, 1, 1]

        assert np.allclose(res, target[:n_subsystems])

    def test_state_roty(self, qubit_device, n_subsystems):
        """Checks the state produced using the rotation='Y' strategy."""

        features = [np.pi / 2, np.pi / 2, np.pi / 4, 0]

        @qml.qnode(qubit_device)
        def circuit(x=None):
            qml.templates.AngleEmbedding(features=x, wires=range(n_subsystems), rotation="Y")
            qml.PauliX(wires=0)
            qml.templates.AngleEmbedding(features=x, wires=range(n_subsystems), rotation="Y")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [-1, -1, 0, 1, 1]
        assert np.allclose(res, target[:n_subsystems])

    def test_state_rotz(self, qubit_device, n_subsystems):
        """Checks the state using the rotation='Z' strategy."""

        features = [np.pi / 2, np.pi / 2, np.pi / 4, 0]

        @qml.qnode(qubit_device)
        def circuit(x=None):
            qml.templates.AngleEmbedding(features=x, wires=range(n_subsystems), rotation="Z")
            qml.PauliX(wires=0)
            qml.templates.AngleEmbedding(features=x, wires=range(n_subsystems), rotation="Z")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [-1, 1, 1, 1, 1]
        assert np.allclose(res, target[:n_subsystems])

    @pytest.mark.parametrize("strategy", ["X", "Y", "Z"])
    def test_angle_embedding_fewer_features(self, strategy):
        """Tests case with fewer features than rotation gates."""
        features = [np.pi / 2, np.pi / 2, np.pi / 4, 0]
        n_subsystems = 5
        dev = qml.device("default.qubit", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.AngleEmbedding(features=x, wires=range(n_subsystems), rotation="Z")
            qml.PauliX(wires=0)
            qml.templates.AngleEmbedding(features=x, wires=range(n_subsystems), rotation="Z")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features)
        target = [-1, 1, 1, 1, 1]
        assert np.allclose(res, target)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        features = np.random.random(size=(3,))

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.templates.AngleEmbedding(features, wires=range(3))
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.templates.AngleEmbedding(features, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


class TestParameters:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize("strategy", ["X", "Y", "Z"])
    def test_exception_fewer_rotations(self, strategy):
        """Verifies that exception is raised if there are fewer
        rotation gates than features."""

        features = [0, 0, 0, 0]
        n_subsystems = 1
        dev = qml.device("default.qubit", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.AngleEmbedding(features=x, wires=range(n_subsystems), rotation=strategy)
            qml.PauliX(wires=0)
            qml.templates.AngleEmbedding(features=x, wires=range(n_subsystems), rotation=strategy)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        with pytest.raises(ValueError, match="Features must be of"):
            circuit(x=features)

    def test_exception_wrongrot(self):
        """Verifies that exception is raised if the
        rotation strategy is unknown."""

        n_subsystems = 1
        dev = qml.device("default.qubit", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.AngleEmbedding(features=x, wires=range(n_subsystems), rotation="A")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        with pytest.raises(ValueError, match="Rotation option"):
            circuit(x=[1])

    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the
        number of dimensions of features is incorrect."""
        n_subsystems = 1
        dev = qml.device("default.qubit", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.AngleEmbedding(features=x, wires=range(n_subsystems))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        with pytest.raises(ValueError, match="Features must be a one-dimensional"):
            circuit(x=[[1], [0]])


def circuit_template(features):
    qml.templates.AngleEmbedding(features, range(3))
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


class TestGradients:
    """Tests that the gradient is computed correctly in all interfaces."""

    def test_autograd(self, tol):
        """Tests that gradients of template and decomposed circuit
        are the same in the autograd interface."""

        features = pnp.array([1.0, 1.0, 1.0], requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

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

        features = jnp.array([1.0, 1.0, 1.0])

        dev = qml.device("default.qubit", wires=3)

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

        features = tf.Variable([1.0, 1.0, 1.0])

        dev = qml.device("default.qubit", wires=3)

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

        features = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev, interface="torch")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="torch")

        res = circuit(features)
        res.backward()
        grads = [features.grad]

        res2 = circuit2(features)
        res2.backward()
        grads2 = [features.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
