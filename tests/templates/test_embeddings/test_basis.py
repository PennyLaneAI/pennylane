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
Tests for the BasisEmbedding template.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize("features", [[1, 0, 1], [1, 1, 1], [0, 1, 0]])
    def test_expansion(self, features):
        """Checks the queue."""

        op = qml.BasisEmbedding(features=features, wires=range(3))
        tape = op.expand()

        assert len(tape.operations) == features.count(1)
        for gate in tape.operations:
            assert gate.name == "PauliX"

    @pytest.mark.parametrize("state", [[0, 1], [1, 1], [1, 0], [0, 0]])
    def test_state(self, state):
        """Checks that the correct state is prepared."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.BasisEmbedding(features=x, wires=range(2))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        res = circuit(x=state)
        expected = [1 if s == 0 else -1 for s in state]
        assert np.allclose(res, expected)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        features = [1, 0, 1]

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(features, wires=range(3))
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.BasisEmbedding(features, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        ("feat", "wires", "expected"),
        [(7, range(3), [1, 1, 1]), (2, range(4), [0, 0, 1, 0]), (8, range(5), [0, 1, 0, 0, 0])],
    )
    def test_features_as_int_conversion(self, feat, wires, expected):
        """checks conversion from features as int to a list of binary digits
        with length = len(wires)"""

        assert (
            qml.BasisEmbedding(features=feat, wires=wires).hyperparameters["basis_state"]
            == expected
        )

    @pytest.mark.parametrize("x", [[0], [0, 1, 1], 4])
    def test_wrong_input_bits_exception(self, x):
        """Checks exception if number of features is not same as number of qubits."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(features=x, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Features must be of length"):
            circuit()

    def test_input_not_binary_exception(self):
        """Checks exception if the features contain values other than zero and one."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.BasisEmbedding(features=x, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Basis state must only consist of"):
            circuit(x=[2, 3])

    def test_exception_wrong_dim(self):
        """Checks exception if the number of dimensions of features is incorrect."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.BasisEmbedding(features=x, wires=2)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Features must be one-dimensional"):
            circuit(x=[[1], [0]])

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.BasisEmbedding([0, 1], wires=[0, 1], id="a")
        assert template.id == "a"


def circuit_template(features):
    qml.BasisEmbedding(features, wires=range(3))
    return qml.state()


def circuit_decomposed(features):
    # convert tensor to list
    feats = list(qml.math.toarray(features))
    for i in range(len(feats)):
        if feats[i] == 1:
            qml.PauliX(wires=i)

    return qml.state()


class TestInterfaces:
    """Tests that the template is compatible with all interfaces."""

    def test_list_and_tuples(self, tol):
        """Tests common iterables as inputs."""

        features = [0, 1, 0]

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

        features = pnp.array([0, 1, 0])

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        features = jnp.array([0, 1, 0])

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev, interface="jax")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="jax")

        res = circuit(features)
        res2 = circuit2(features)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        features = tf.Variable([0, 1, 0])

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev, interface="tf")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="tf")

        res = circuit(features)
        res2 = circuit2(features)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests the torch interface."""

        import torch

        features = torch.tensor([0, 1, 0])

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev, interface="torch")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="torch")

        res = circuit(features)
        res2 = circuit2(features)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)
