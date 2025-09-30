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
Unit tests for the SimplifiedTwoDesign template.
"""
import numpy as np

# pylint: disable=too-many-arguments,too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as pnp


@pytest.mark.jax
def test_standard_validity():
    """Run standard checks with the assert_valid function."""

    n_wires = 2
    weight_shape = (2, 1, 2)

    weights = np.random.random(size=weight_shape)
    initial_layer = np.random.randn(n_wires)

    op = qml.SimplifiedTwoDesign(initial_layer, weights, wires=range(n_wires))
    qml.ops.functions.assert_valid(op)


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    QUEUES = [
        (1, (1,), ["RY"], [[0]]),
        (2, (1, 1, 2), ["RY", "RY", "CZ", "RY", "RY"], [[0], [1], [0, 1], [0], [1]]),
        (
            2,
            (2, 1, 2),
            ["RY", "RY", "CZ", "RY", "RY", "CZ", "RY", "RY"],
            [[0], [1], [0, 1], [0], [1], [0, 1], [0], [1]],
        ),
        (
            3,
            (1, 2, 2),
            ["RY", "RY", "RY", "CZ", "RY", "RY", "CZ", "RY", "RY"],
            [[0], [1], [2], [0, 1], [0], [1], [1, 2], [1], [2]],
        ),
    ]

    @pytest.mark.parametrize("n_wires, weight_shape, expected_names, expected_wires", QUEUES)
    def test_expansion(self, n_wires, weight_shape, expected_names, expected_wires):
        """Checks the queue for the default settings."""

        weights = np.random.random(size=weight_shape)
        initial_layer = np.random.randn(n_wires)

        op = qml.SimplifiedTwoDesign(initial_layer, weights, wires=range(n_wires))
        queue = op.decomposition()

        for i, gate in enumerate(queue):
            assert gate.name == expected_names[i]
            assert gate.wires.labels == tuple(expected_wires[i])

    @pytest.mark.parametrize(
        "n_wires, n_layers, shape_weights",
        [(1, 2, (0,)), (2, 2, (2, 1, 2)), (3, 2, (2, 2, 2)), (4, 2, (2, 3, 2))],
    )
    def test_circuit_parameters(self, n_wires, n_layers, shape_weights):
        """Tests the parameter values in the circuit."""

        initial_layer = np.random.randn(n_wires)
        weights = np.random.randn(*shape_weights)

        op = qml.SimplifiedTwoDesign(initial_layer, weights, wires=range(n_wires))
        queue = op.decomposition()

        # test the device parameters
        for _ in range(n_layers):
            # only select the rotation gates
            ops = [gate for gate in queue if isinstance(gate, qml.RY)]

            # check each initial_layer gate parameters
            for n in range(n_wires):
                res_param = ops[n].parameters[0]
                exp_param = initial_layer[n]
                assert res_param == exp_param

            # check layer gate parameters
            ops = ops[n_wires:]
            exp_params = weights.flatten()
            for o, exp_param in zip(ops, exp_params):
                res_param = o.parameters[0]
                assert res_param == exp_param

    @pytest.mark.parametrize(
        "initial_layer_weights, weights, n_wires, target",
        [
            ([np.pi], [], 1, [-1]),
            ([np.pi] * 2, [[[np.pi] * 2]], 2, [1, 1]),
            ([np.pi] * 3, [[[np.pi] * 2] * 2], 3, [1, -1, 1]),
            ([np.pi] * 4, [[[np.pi] * 2] * 3], 4, [1, -1, -1, 1]),
        ],
    )
    def test_correct_target_output(self, initial_layer_weights, weights, n_wires, target, tol):
        """Tests the result of the template for simple cases."""
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(initial_layer, weights):
            qml.SimplifiedTwoDesign(
                initial_layer_weights=initial_layer, weights=weights, wires=range(n_wires)
            )
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        expectations = circuit(np.array(initial_layer_weights), np.array(weights))
        for exp, target_exp in zip(expectations, target):
            assert np.allclose(exp, target_exp, atol=tol, rtol=0)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        weights = np.random.random(size=(1, 2, 2))
        initial_layer = np.random.randn(3)

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.SimplifiedTwoDesign(initial_layer, weights, wires=range(3))
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.SimplifiedTwoDesign(initial_layer, weights, wires=["z", "a", "k"])
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
        initial_layer = np.random.randn(2)

        @qml.qnode(dev)
        def circuit(initial_layer, weights):
            qml.SimplifiedTwoDesign(initial_layer, weights, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Weights tensor must have second dimension"):
            weights = np.random.randn(2, 2, 2)
            circuit(initial_layer, weights)

        with pytest.raises(ValueError, match="Weights tensor must have third dimension"):
            weights = np.random.randn(2, 1, 3)
            circuit(initial_layer, weights)

        with pytest.raises(ValueError, match="Initial layer weights must be of shape"):
            initial_layer = np.random.randn(3)
            weights = np.random.randn(2, 1, 2)
            circuit(initial_layer, weights)

    def test_id(self):
        """Tests that the id attribute can be set."""
        weights = np.random.random(size=(1, 2, 2))
        initial_layer = np.random.randn(3)
        template = qml.SimplifiedTwoDesign(initial_layer, weights, wires=range(3), id="a")
        assert template.id == "a"


class TestAttributes:
    """Tests additional methods and attributes"""

    @pytest.mark.parametrize(
        "n_layers, n_wires, expected_shape",
        [
            (2, 3, [(3,), (2, 2, 2)]),
            (2, 1, [(1,), (2,)]),
            (2, 2, [(2,), (2, 1, 2)]),
        ],
    )
    def test_shape(self, n_layers, n_wires, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor"""

        shape = qml.SimplifiedTwoDesign.shape(n_layers, n_wires)
        assert shape == expected_shape


def circuit_template(initial_weights, weights):
    qml.SimplifiedTwoDesign(initial_weights, weights, range(3))
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(initial_weights, weights):
    qml.RY(initial_weights[0], wires=0)
    qml.RY(initial_weights[1], wires=1)
    qml.RY(initial_weights[2], wires=2)

    qml.CZ(wires=[0, 1])
    qml.RY(weights[0, 0, 0], wires=0)
    qml.RY(weights[0, 0, 1], wires=1)

    qml.CZ(wires=[1, 2])
    qml.RY(weights[0, 1, 0], wires=1)
    qml.RY(weights[0, 1, 1], wires=2)

    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        weights = np.random.random(size=(1, 2, 2))
        weights = pnp.array(weights, requires_grad=True)
        initial_weights = np.random.random(size=(3,))
        initial_weights = pnp.array(initial_weights, requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(initial_weights, weights)
        res2 = circuit2(initial_weights, weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(initial_weights, weights)

        grad_fn2 = qml.grad(circuit2)
        grads2 = grad_fn2(initial_weights, weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
        assert np.allclose(grads[1], grads2[1], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(1, 2, 2)))
        initial_weights = jnp.array(np.random.random(size=(3,)))

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(initial_weights, weights)
        res2 = circuit2(initial_weights, weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(initial_weights, weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(initial_weights, weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
        assert np.allclose(grads[1], grads2[1], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax_jit(self, tol):
        """Tests jit within the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(1, 2, 2)))
        initial_weights = jnp.array(np.random.random(size=(3,)))

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = jax.jit(circuit)

        res = circuit(initial_weights, weights)
        res2 = circuit2(initial_weights, weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(initial_weights, weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(initial_weights, weights)

        assert qml.math.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        weights = tf.Variable(np.random.random(size=(1, 2, 2)))
        initial_weights = tf.Variable(np.random.random(size=(3,)))

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(initial_weights, weights)
        res2 = circuit2(initial_weights, weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            res = circuit(initial_weights, weights)
        grads = tape.gradient(res, [initial_weights, weights])

        with tf.GradientTape() as tape2:
            res2 = circuit2(initial_weights, weights)
        grads2 = tape2.gradient(res2, [initial_weights, weights])

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
        assert np.allclose(grads[1], grads2[1], atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests the torch interface."""

        import torch

        weights = torch.tensor(np.random.random(size=(1, 2, 2)), requires_grad=True)
        initial_weights = torch.tensor(np.random.random(size=(3,)), requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(initial_weights, weights)
        res2 = circuit2(initial_weights, weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(initial_weights, weights)
        res.backward()
        grads = [weights.grad, initial_weights.grad]

        res2 = circuit2(initial_weights, weights)
        res2.backward()
        grads2 = [weights.grad, initial_weights.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
        assert np.allclose(grads[1], grads2[1], atol=tol, rtol=0)
