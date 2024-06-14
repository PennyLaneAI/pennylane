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
Unit tests for the RandomLayers template.
"""
import numpy as np

# pylint: disable=too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as pnp


def test_standard_validity():
    """Run standard checks with the assert_valid function."""

    weights = np.array([[0.1, -2.1, 1.4]])
    op = qml.RandomLayers(weights, wires=(0, 1))
    qml.ops.functions.assert_valid(op)


def test_hyperparameters():
    """Test that the hyperparmaeters are set as expected."""
    weights = np.array([[0.1, -2.1, 1.4]])
    op = qml.RandomLayers(weights, wires=(0, 1))

    assert op.hyperparameters == {
        "ratio_imprim": 0.3,
        "imprimitive": qml.CNOT,
        "rotations": (qml.RX, qml.RY, qml.RZ),
        "seed": 42,
    }

    op2 = qml.RandomLayers(
        weights, wires=(0, 1), ratio_imprim=1.0, imprimitive=qml.CZ, rotations=(qml.RX,), seed=None
    )

    assert op2.hyperparameters == {
        "ratio_imprim": 1.0,
        "imprimitive": qml.CZ,
        "rotations": (qml.RX,),
        "seed": None,
    }


# pylint: disable=protected-access
def test_flatten_unflatten():
    """Test the behavior of the flatten and unflatten methods."""
    weights = np.array([[0.1, -2.1, 1.4]])
    op = qml.RandomLayers(weights, wires=(0, 1))

    data, metadata = op._flatten()

    assert qml.math.allclose(data[0], weights)
    # check metadata hashable
    assert hash(metadata)

    new_op = type(op)._unflatten(*op._flatten())
    assert qml.equal(new_op, op)
    assert new_op is not op


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    def test_seed(self):
        """Test that the circuit is fixed by the seed."""
        weights = np.random.random((2, 3))

        op1 = qml.RandomLayers(weights, wires=range(2), seed=41)
        op2 = qml.RandomLayers(weights, wires=range(2), seed=42)
        op3 = qml.RandomLayers(weights, wires=range(2), seed=42)

        queue1 = op1.expand().operations
        decomp1 = op1.compute_decomposition(*op1.parameters, wires=op1.wires, **op1.hyperparameters)
        queue2 = op2.expand().operations
        decomp2 = op2.compute_decomposition(*op2.parameters, wires=op2.wires, **op2.hyperparameters)
        queue3 = op3.expand().operations
        decomp3 = op3.compute_decomposition(*op3.parameters, wires=op3.wires, **op3.hyperparameters)

        assert not all(g1.name == g2.name for g1, g2 in zip(queue1, queue2))
        assert all(g2.name == g3.name for g2, g3 in zip(queue2, queue3))

        assert all(qml.equal(op1, op2) for op1, op2 in zip(queue1, decomp1))
        assert all(qml.equal(op1, op2) for op1, op2 in zip(queue2, decomp2))
        assert all(qml.equal(op1, op2) for op1, op2 in zip(queue3, decomp3))

    @pytest.mark.parametrize("n_layers, n_rots", [(3, 4), (1, 2)])
    def test_number_gates(self, n_layers, n_rots):
        """Test that the number of gates is correct."""
        weights = np.random.randn(n_layers, n_rots)

        op = qml.RandomLayers(weights, wires=range(2))
        ops = op.expand().operations

        gate_names = [g.name for g in ops]
        assert len(gate_names) - gate_names.count("CNOT") == n_layers * n_rots

    @pytest.mark.parametrize("ratio", [0.2, 0.6])
    def test_ratio_imprim(self, ratio):
        """Test the ratio of imprimitive gates."""
        n_rots = 500
        weights = np.random.random(size=(1, n_rots))

        op = qml.RandomLayers(weights, wires=range(2), ratio_imprim=ratio)
        queue = op.decomposition()

        gate_names = [gate.name for gate in queue]
        ratio_impr = gate_names.count("CNOT") / len(gate_names)
        assert np.isclose(ratio_impr, ratio, atol=0.05)

    def test_random_wires(self):
        """Test that random wires are picked for the gates. This is done by
        taking the mean of all wires, which should be 1 for labels [0, 1, 2]"""
        n_rots = 5000
        weights = np.random.random(size=(2, n_rots))

        op = qml.RandomLayers(weights, wires=range(3))
        queue = op.expand().operations

        gate_wires = [gate.wires.labels for gate in queue]
        wires_flat = [item for w in gate_wires for item in w]
        mean_wire = np.mean(wires_flat)

        assert np.isclose(mean_wire, 1, atol=0.05)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        weights = np.random.random(size=(1, 3))

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.RandomLayers(weights, wires=range(3))
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.RandomLayers(weights, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the
        number of dimensions of features is incorrect."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(phi):
            qml.RandomLayers(phi, wires=list(range(4)))
            return qml.expval(qml.PauliZ(0))

        phi = np.array([0.04, 0.14, 3.29, 2.51])

        with pytest.raises(ValueError, match="Weights tensor must be 2-dimensional"):
            circuit(phi)

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.RandomLayers(np.random.random(size=(1, 3)), wires=range(3), id="a")
        assert template.id == "a"


class TestAttributes:
    """Tests additional methods and attributes"""

    @pytest.mark.parametrize(
        "n_layers, n_rots, expected_shape",
        [
            (2, 3, (2, 3)),
            (2, 1, (2, 1)),
            (2, 2, (2, 2)),
        ],
    )
    def test_shape(self, n_layers, n_rots, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor"""

        shape = qml.RandomLayers.shape(n_layers, n_rots)
        assert shape == expected_shape


def circuit_template(weights):
    qml.RandomLayers(weights, range(3), seed=42)
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(weights):
    # this structure is only true for a seed of 42 and 3 wires
    qml.RY(weights[0, 0], wires=1)
    qml.RX(weights[0][1], wires=2)
    qml.CNOT(wires=[1, 2])
    qml.RZ(weights[0, 2], wires=1)
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    def test_list_lists(self):
        """Tests the weights as a list of lists."""
        weights = [[0.1, -2.1, 1.4]]

        op = qml.RandomLayers(weights, wires=range(3), seed=42)

        decomp = op.decomposition()
        expected = [
            qml.RY(weights[0][0], wires=1),
            qml.RX(weights[0][1], wires=2),
            qml.CNOT(wires=[1, 2]),
            qml.RZ(weights[0][2], wires=1),
        ]

        for op1, op2 in zip(decomp, expected):
            assert qml.equal(op1, op2)

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
