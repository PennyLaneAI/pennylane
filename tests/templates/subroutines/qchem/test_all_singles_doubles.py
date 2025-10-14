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
Tests for the AllSinglesDoubles template.
"""
import numpy as np

# pylint: disable=too-many-arguments,too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as pnp


@pytest.mark.jax
def test_standard_validity():
    """Run standard tests of operation validity."""
    op = qml.AllSinglesDoubles(
        [1.0, 2.0],
        wires=range(4),
        hf_state=np.array([1, 1, 0, 0]),
        singles=[[0, 1]],
        doubles=[[0, 1, 2, 3]],
    )
    qml.ops.functions.assert_valid(op)


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        ("singles", "doubles", "weights", "ref_gates"),
        [
            (
                [[0, 2], [0, 4], [1, 3], [1, 5]],
                [[0, 1, 2, 3], [0, 1, 2, 5], [0, 1, 3, 4], [0, 1, 4, 5]],
                np.array(
                    [
                        -0.39926835,
                        2.24302631,
                        -1.87369867,
                        2.66863856,
                        0.08284505,
                        -1.90970947,
                        1.70143771,
                        -1.0404567,
                    ]
                ),
                [
                    [0, qml.BasisState, [0, 1, 2, 3, 4, 5], [np.array([1, 1, 0, 0, 0, 0])]],
                    [5, qml.SingleExcitation, [0, 2], [-0.39926835]],
                    [6, qml.SingleExcitation, [0, 4], [2.24302631]],
                    [7, qml.SingleExcitation, [1, 3], [-1.87369867]],
                    [8, qml.SingleExcitation, [1, 5], [2.66863856]],
                    [1, qml.DoubleExcitation, [0, 1, 2, 3], [0.08284505]],
                    [2, qml.DoubleExcitation, [0, 1, 2, 5], [-1.90970947]],
                    [3, qml.DoubleExcitation, [0, 1, 3, 4], [1.70143771]],
                    [4, qml.DoubleExcitation, [0, 1, 4, 5], [-1.0404567]],
                ],
            ),
            (
                [[1, 5]],
                [],
                np.array([3.815]),
                [
                    [0, qml.BasisState, [0, 1, 2, 3, 4, 5], [np.array([1, 1, 0, 0, 0, 0])]],
                    [1, qml.SingleExcitation, [1, 5], [3.815]],
                ],
            ),
            (
                [],
                [[0, 1, 4, 5]],
                np.array([4.866]),
                [
                    [0, qml.BasisState, [0, 1, 2, 3, 4, 5], [np.array([1, 1, 0, 0, 0, 0])]],
                    [1, qml.DoubleExcitation, [0, 1, 4, 5], [4.866]],
                ],
            ),
        ],
    )
    def test_allsinglesdoubles_operations(self, singles, doubles, weights, ref_gates):
        """Test the correctness of the AllSinglesDoubles template including the gate count
        and order, the wires the operation acts on and the correct use of parameters
        in the circuit."""

        N = 6
        wires = range(N)

        hf_state = np.array([1, 1, 0, 0, 0, 0])

        op = qml.AllSinglesDoubles(weights, wires, hf_state, singles=singles, doubles=doubles)
        queue = op.decomposition()

        assert len(queue) == len(singles) + len(doubles) + 1

        for gate in ref_gates:
            idx = gate[0]

            exp_gate = gate[1]
            res_gate = queue[idx]
            assert isinstance(res_gate, exp_gate)

            exp_wires = gate[2]
            res_wires = queue[idx].wires
            assert res_wires.tolist() == exp_wires

            exp_weight = gate[3]
            res_weight = queue[idx].parameters
            assert np.allclose(res_weight, exp_weight)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""

        weights = [0.1, 0.2]
        dev = qml.device("default.qubit", wires=4)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k", "e"])

        @qml.qnode(dev)
        def circuit():
            qml.AllSinglesDoubles(
                weights,
                wires=range(4),
                hf_state=np.array([1, 1, 0, 0]),
                singles=[[0, 1]],
                doubles=[[0, 1, 2, 3]],
            )
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.AllSinglesDoubles(
                weights,
                wires=["z", "a", "k", "e"],
                hf_state=np.array([1, 1, 0, 0]),
                singles=[["z", "a"]],
                doubles=[["z", "a", "k", "e"]],
            )
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        ("weights", "wires", "singles", "doubles", "hf_state", "msg_match"),
        [
            (
                np.array([-2.8]),
                range(4),
                [[0, 2]],
                None,
                np.array([1.2, 1, 0, 0]),
                "Elements of 'hf_state' must be integers",
            ),
            (
                np.array([-2.8]),
                range(4),
                None,
                None,
                np.array([1, 1, 0, 0]),
                "'singles' and 'doubles' lists can not be both empty",
            ),
            (
                np.array([-2.8]),
                range(4),
                [],
                [],
                np.array([1, 1, 0, 0]),
                "'singles' and 'doubles' lists can not be both empty",
            ),
            (
                np.array([-2.8]),
                range(4),
                None,
                [[0, 1, 2, 3, 4]],
                np.array([1, 1, 0, 0]),
                "Expected entries of 'doubles' to be of size 4",
            ),
            (
                np.array([-2.8]),
                range(4),
                [[0, 2, 3]],
                None,
                np.array([1, 1, 0, 0]),
                "Expected entries of 'singles' to be of size 2",
            ),
            (
                np.array([-2.8, 0.5]),
                range(4),
                [[0, 2]],
                [[0, 1, 2, 3]],
                np.array([1, 1, 0, 0, 0]),
                "State must be of length 4",
            ),
            (
                np.array([-2.8, 1.6]),
                range(4),
                [[0, 2]],
                None,
                np.array([1, 1, 0, 0]),
                "'weights' tensor must be of shape",
            ),
            (
                np.array([-2.8, 1.6]),
                range(4),
                None,
                [[0, 1, 2, 3]],
                np.array([1, 1, 0, 0]),
                "'weights' tensor must be of shape",
            ),
            (
                np.array([-2.8, 1.6]),
                range(4),
                [[0, 2], [1, 3]],
                [[0, 1, 2, 3]],
                np.array([1, 1, 0, 0]),
                "'weights' tensor must be of shape",
            ),
            (
                np.array([-2.8, 1.6]),
                range(4),
                None,
                [[0, 1, 2, 3]],
                np.array([1, 1, 0, 0]),
                "'weights' tensor must be of shape",
            ),
            (
                np.array([-2.8, 1.6]),
                range(4),
                [[0, 2]],
                None,
                np.array([1, 1, 0, 0]),
                "'weights' tensor must be of shape",
            ),
            (
                np.array([-2.8, 1.6]),
                range(1),
                [[0, 2]],
                None,
                np.array([1, 1, 0, 0]),
                "can not be less than 2",
            ),
        ],
    )
    def test_allsinglesdoubles_exceptions(
        self, weights, wires, singles, doubles, hf_state, msg_match
    ):
        """Test that AllSinglesDoubles throws an exception if the parameters have illegal
        shapes, types or values."""

        dev = qml.device("default.qubit", wires=len(wires))

        def circuit(
            weights=weights, wires=wires, hf_state=hf_state, singles=singles, doubles=doubles
        ):
            qml.AllSinglesDoubles(
                weights=weights,
                wires=wires,
                hf_state=hf_state,
                singles=singles,
                doubles=doubles,
            )
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match=msg_match):
            qnode(
                weights=weights,
                wires=wires,
                hf_state=hf_state,
                singles=singles,
                doubles=doubles,
            )

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.AllSinglesDoubles(
            [1, 2],
            wires=range(4),
            hf_state=np.array([1, 1, 0, 0]),
            singles=[[0, 1]],
            doubles=[[0, 1, 2, 3]],
            id="a",
        )
        assert template.id == "a"


class TestAttributes:
    """Tests additional methods and attributes"""

    @pytest.mark.parametrize(
        "singles, doubles, expected_shape",
        [
            ([[0, 2], [1, 3]], [[0, 1, 2, 3], [4, 5, 6, 7]], (4,)),
            ([[0, 2], [1, 3]], None, (2,)),
            ([[0, 2], [1, 3]], [], (2,)),
            (None, [[0, 1, 2, 3], [4, 5, 6, 7]], (2,)),
            ([], [[0, 1, 2, 3], [4, 5, 6, 7]], (2,)),
        ],
    )
    def test_shape(self, singles, doubles, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor"""

        shape = qml.AllSinglesDoubles.shape(singles, doubles)
        assert shape == expected_shape


def circuit_template(weights):
    qml.AllSinglesDoubles(
        weights,
        wires=range(4),
        hf_state=np.array([1, 1, 0, 0]),
        singles=[[0, 1]],
        doubles=[[0, 1, 2, 3]],
    )
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(weights):
    qml.BasisState(np.array([1, 1, 0, 0]), wires=range(4))
    qml.DoubleExcitation(weights[1], wires=[0, 1, 2, 3])
    qml.SingleExcitation(weights[0], wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    def test_list_and_tuples(self, tol):
        """Tests common iterables as inputs."""

        weights = list(range(2))

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        weights = pnp.array(np.random.random(size=(2,)), requires_grad=True)

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = qml.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(2,)))
        dev = qml.device("default.qubit", wires=4)

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
        """Tests jit within the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(2,)))
        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = jax.jit(circuit)

        res = circuit(weights)
        res2 = circuit2(weights)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert qml.math.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        weights = tf.Variable(np.random.random(size=(2,)))
        dev = qml.device("default.qubit", wires=4)

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

        weights = torch.tensor(np.random.random(size=(2,)), requires_grad=True)

        dev = qml.device("default.qubit", wires=4)

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
