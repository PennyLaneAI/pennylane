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
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""
    wires = qml.wires.Wires((0, 1, 2))
    op = qml.BasisEmbedding(features=np.array([1, 1, 1]), wires=wires)
    qml.ops.functions.assert_valid(op, skip_differentiation=True)


# pylint: disable=protected-access
def test_flatten_unflatten():
    """Test the _flatten and _unflatten methods."""
    wires = qml.wires.Wires((0, 1, 2))
    op = qml.BasisEmbedding(features=[1, 1, 1], wires=wires)
    data, metadata = op._flatten()
    assert np.allclose(data[0], [1, 1, 1])
    assert metadata[0] == wires

    # make sure metadata hashable
    assert hash(metadata)

    new_op = op._unflatten(*op._flatten())
    qml.assert_equal(op, new_op)
    assert op is not new_op


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize("features", [[1, 0, 1], [1, 1, 1], [0, 1, 0]])
    def test_expansion(self, features):
        """Checks the queue."""

        op = qml.BasisEmbedding(features=features, wires=range(3))
        tape = qml.tape.QuantumScript(op.decomposition())

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
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.BasisEmbedding(features, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        ("feat", "wires", "expected"),
        [(7, range(3), [1, 1, 1]), (2, range(4), [0, 0, 1, 0]), (8, range(5), [0, 1, 0, 0, 0])],
    )
    def test_features_as_int_conversion(self, feat, wires, expected):
        """checks conversion from features as int to a list of binary digits
        with length = len(wires)"""

        assert np.allclose(qml.BasisEmbedding(features=feat, wires=wires).parameters[0], expected)

    @pytest.mark.parametrize(
        "x, msg",
        [
            ([0], "State must be of length"),
            ([0, 1, 1], "State must be of length"),
            (4, "Integer state must be"),
        ],
    )
    def test_wrong_input_bits_exception(self, x, msg):
        """Checks exception if number of features is not same as number of qubits."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(features=x, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match=msg):
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

        with pytest.raises(ValueError, match="State must be one-dimensional"):
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
    feats = list(qml.math.array(features))
    _ = [qml.PauliX(wires=i) for i, feat in enumerate(feats) if feat == 1]

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

        res = circuit(2)
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

        res = circuit(pnp.array(2))
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax.numpy as jnp

        features = jnp.array([0, 1, 0])

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(jnp.array(2))
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax_jit(self, tol):
        """Tests compilation with JAX JIT."""

        import jax
        import jax.numpy as jnp

        features = jnp.array([0, 1, 0])

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = jax.jit(qml.QNode(circuit_template, dev))

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(2)
        res2 = circuit2(2)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        features = tf.Variable([0, 1, 0])

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(tf.Variable(2))
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf_autograph(self, tol):
        """Tests the tf interface with autograph"""

        import tensorflow as tf

        features = tf.Variable([0, 1, 0])

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(tf.Variable(2))
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests the torch interface."""

        import torch

        features = torch.tensor([0, 1, 0])

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(torch.tensor(2))
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)
