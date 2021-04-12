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
Tests for the SqueezingEmbedding template.
"""
import pytest
import numpy as np
from pennylane import numpy as pnp
import pennylane as qml


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize("features", [[1, 2, 3], [-1, 1, -1]])
    def test_expansion(self, features):
        """Checks the queue for the default settings."""

        op = qml.templates.SqueezingEmbedding(features=features, wires=range(3))
        tape = op.expand()

        assert len(tape.operations) == len(features)
        for idx, gate in enumerate(tape.operations):
            assert gate.name == "Squeezing"
            assert gate.parameters[0] == features[idx]

    def test_state_execution_amplitude(self):
        """Checks the state using the amplitude execution method."""

        features = np.array([1.2, 0.3])
        n_wires = 2
        dev = qml.device("default.gaussian", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.SqueezingEmbedding(
                features=x, wires=range(n_wires), method="amplitude", c=1
            )
            return [
                qml.expval(qml.NumberOperator(wires=0)),
                qml.expval(qml.NumberOperator(wires=1)),
            ]

        # TODO: come up with better test case
        assert np.allclose(circuit(x=features), [2.2784, 0.09273], atol=0.001)

    def test_state_execution_phase(self):
        """Checks the state produced using the phase execution method."""

        features = np.array([1.2, 0.3])
        n_wires = 2
        dev = qml.device("default.gaussian", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.SqueezingEmbedding(features=x, wires=range(n_wires), method="phase", c=1)
            qml.Beamsplitter(np.pi / 2, 0, wires=[0, 1])
            qml.templates.SqueezingEmbedding(
                features=[0, 0], wires=range(n_wires), method="phase", c=1
            )
            return [
                qml.expval(qml.NumberOperator(wires=0)),
                qml.expval(qml.NumberOperator(wires=1)),
            ]

        # TODO: come up with better test case
        assert np.allclose(circuit(x=features), [12.86036, 8.960306], atol=0.001)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        features = np.random.random(size=(3,))

        dev = qml.device("default.gaussian", wires=3)
        dev2 = qml.device("default.gaussian", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.templates.SqueezingEmbedding(features, wires=range(3))
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.templates.SqueezingEmbedding(features, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev._state[0], dev2._state[0], atol=tol, rtol=0)
        assert np.allclose(dev._state[1], dev2._state[1], atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    def test_exception_for_wrong_num_wires(self):
        """Verifies that an exception is thrown if number of subsystems wrong."""

        n_wires = 2
        dev = qml.device("default.gaussian", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.SqueezingEmbedding(features=x, wires=range(n_wires), method="phase")
            return [qml.expval(qml.X(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="Features must be of"):
            circuit(x=[0.2, 0.3, 0.4])

    def test_strategy_not_recognized_exception(self):
        """Verifies that an exception is thrown if the method is unknown."""

        n_wires = 2
        dev = qml.device("default.gaussian", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.SqueezingEmbedding(features=x, wires=range(n_wires), method="A")
            return [qml.expval(qml.X(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="did not recognize"):
            circuit(x=[1, 2])

    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the
        number of dimensions of features is incorrect."""
        n_subsystems = 2
        dev = qml.device("default.gaussian", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.SqueezingEmbedding(features=x, wires=[0, 1])
            return qml.expval(qml.X(0))

        with pytest.raises(ValueError, match="Features must be a one-dimensional"):
            circuit(x=[[1], [0]])


def circuit_template(features):
    qml.templates.SqueezingEmbedding(features, range(3))
    qml.Beamsplitter(0.5, 0, wires=[2, 1])
    qml.Beamsplitter(0.5, 0, wires=[1, 0])
    return qml.expval(qml.X(0))


def circuit_decomposed(features):
    qml.Squeezing(features[0], 0.1, wires=0)
    qml.Squeezing(features[1], 0.1, wires=1)
    qml.Squeezing(features[2], 0.1, wires=2)
    qml.Beamsplitter(0.5, 0, wires=[2, 1])
    qml.Beamsplitter(0.5, 0, wires=[1, 0])
    return qml.expval(qml.X(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    def test_list_and_tuples(self, tol):
        """Tests common iterables as inputs."""

        features = [1.0, 1.0, 1.0]

        dev = qml.device("default.gaussian", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(tuple(features))
        res2 = circuit2(tuple(features))
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    def test_autograd(self, tol):
        """Tests the autograd interface."""

        features = pnp.array([1.0, 1.0, 1.0], requires_grad=True)

        dev = qml.device("default.gaussian", wires=3)

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

    def test_jax(self, tol):
        """Tests the jax interface."""

        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        features = jnp.array([1.0, 1.0, 1.0])

        dev = qml.device("default.gaussian", wires=3)

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

    def test_tf(self, tol):
        """Tests the tf interface."""

        tf = pytest.importorskip("tensorflow")

        features = tf.Variable([1.0, 1.0, 1.0])

        dev = qml.device("default.gaussian", wires=3)

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

    def test_torch(self, tol):
        """Tests the torch interface."""

        torch = pytest.importorskip("torch")

        features = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

        dev = qml.device("default.gaussian", wires=3)

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
