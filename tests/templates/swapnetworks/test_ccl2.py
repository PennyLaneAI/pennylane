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
Tests for the TwoLocalSwapNetwork template.
"""

import pytest

import pennylane as qp
from pennylane import numpy as np


# pylint: disable=protected-access
def test_flatten_unflatten():
    """Test the flatten and unflatten methods."""

    def acquaintances(index, *_, use_CNOT=True, **__):
        return qp.CNOT(index) if use_CNOT else qp.CZ(index)

    weights = np.array([0.5, 0.6, 0.7])
    wires = qp.wires.Wires((0, 1, 2))

    op = qp.templates.TwoLocalSwapNetwork(
        wires, acquaintances, weights, fermionic=True, shift=False, use_CNOT=False
    )
    data, metadata = op._flatten()
    assert qp.math.allclose(data[0], weights)
    assert metadata[0] == wires
    assert metadata[1] == (
        ("acquaintances", acquaintances),
        ("fermionic", True),
        ("shift", False),
        ("use_CNOT", False),
    )

    # make sure metadata is hashable
    assert hash(metadata)

    new_op = type(op)._unflatten(*op._flatten())
    qp.assert_equal(new_op, op)
    assert new_op is not op


# pylint: disable=too-many-arguments
class TestDecomposition:
    """Test that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        ("wires", "acquaintances", "weights", "fermionic", "shift"),
        [
            (4, None, None, True, False),
            (5, lambda index, wires, param: qp.Identity(index), None, True, False),
            (5, lambda index, wires, param: qp.CNOT(index), None, False, False),
            (6, lambda index, wires, param: qp.CRX(param, index), np.random.rand(15), True, False),
            (6, lambda index, wires, param: qp.CRY(param, index), np.random.rand(15), True, True),
        ],
    )
    def test_ccl2_operations(self, wires, acquaintances, weights, fermionic, shift):
        """Test the correctness of the TwoLocalSwapNetwork template including the
        gate count and order, the wires the operation acts on and the correct use
        of parameters in the circuit."""

        wire_order = range(wires)
        itrweights = iter([]) if weights is None or acquaintances is None else iter(weights)
        qubit_pairs = [
            [[i, i + 1] for i in wire_order[(layer + shift) % 2 : -1 : 2]] for layer in range(wires)
        ]

        op = qp.templates.TwoLocalSwapNetwork(
            wire_order, acquaintances, weights, fermionic=fermionic, shift=shift
        )
        queue = op.decomposition()

        # number of gates
        assert len(queue) == sum(
            (
                2 * len(i)
                if acquaintances is not None and acquaintances([0, 1], [0, 1], 0.0)
                else len(i)
            )
            for i in qubit_pairs
        )

        # complete gate set
        gate_order = []
        ac_op = acquaintances if acquaintances is not None else ()
        for pairs in qubit_pairs:
            for pair in pairs:
                if ac_op and ac_op([0, 1], [0, 1], 0.0):
                    gate_order.append(ac_op(pair, pair, next(itrweights, 0.0)))
                sw_op = qp.FermionicSWAP(np.pi, pair) if fermionic else qp.SWAP(pair)
                gate_order.append(sw_op)

        for op1, op2 in zip(queue, gate_order):
            qp.assert_equal(op1, op2)

    def test_custom_wire_labels(self, tol=1e-8):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""

        def acquaintances(index, *_, **___):
            return qp.CNOT(index)

        weights = np.random.random(size=10)

        dev = qp.device("default.qubit", wires=5)
        dev2 = qp.device("default.qubit", wires=["z", "a", "k", "e", "y"])

        @qp.qnode(dev)
        def circuit():
            qp.templates.TwoLocalSwapNetwork(
                dev.wires, acquaintances, weights, fermionic=True, shift=False
            )
            return qp.state()

        @qp.qnode(dev2)
        def circuit2():
            qp.templates.TwoLocalSwapNetwork(
                dev2.wires, acquaintances, weights, fermionic=True, shift=False
            )
            return qp.state()

        assert np.allclose(circuit(), circuit2(), atol=tol, rtol=0)

    @pytest.mark.parametrize(
        ("num_wires", "acquaintances", "weights", "fermionic", "shift", "exp_state"),
        [
            (
                3,
                None,
                False,
                True,
                False,
                qp.math.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
            ),
            (
                3,
                lambda index, wires, param: qp.CNOT(index),
                False,
                True,
                False,
                qp.math.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            ),
            (
                4,
                lambda index, wires, param: qp.CNOT(index),
                False,
                False,
                False,
                qp.math.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                ),
            ),
            (
                4,
                lambda index, wires, param: qp.CRX(param, index),
                True,
                True,
                False,
                qp.math.array(
                    [
                        0.0,
                        -0.5j,
                        0.0,
                        0.0,
                        0.0,
                        -0.5,
                        0.0,
                        0.5j,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -0.5,
                    ]
                ),
            ),
        ],
    )
    def test_ccl2(self, num_wires, acquaintances, weights, fermionic, shift, exp_state):
        """Test that the TwoLocalSwapNetwork template works correctly by asserting the prepared state."""

        wires = range(num_wires)

        shape = qp.templates.TwoLocalSwapNetwork.shape(num_wires)
        weights = np.pi / 2 * qp.math.ones(shape) if weights else None

        dev = qp.device("default.qubit", wires=wires)

        @qp.qnode(dev)
        def circuit():
            qp.PauliX(wires=[0])
            qp.PauliX(wires=[2])
            qp.templates.TwoLocalSwapNetwork(wires, acquaintances, weights, fermionic, shift)
            return qp.state()

        assert qp.math.allclose(circuit(), exp_state, atol=1e-8)

    def test_decomposition_exception_not_enough_qubits(self):
        """Test that the decomposition function warns if there are not enough qubits."""

        with pytest.raises(ValueError, match="TwoLocalSwapNetwork requires at least 2 wires"):
            qp.templates.TwoLocalSwapNetwork.compute_decomposition(
                weights=None,
                wires=range(1),
                acquaintances=None,
            )


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        ("wires", "acquaintances", "weights", "fermionic", "shift", "msg_match"),
        [
            (
                1,
                None,
                None,
                True,
                False,
                "TwoLocalSwapNetwork requires at least 2 wires",
            ),
            (
                6,
                qp.CNOT(wires=[0, 1]),
                np.random.rand(18),
                True,
                False,
                "Acquaintances must either be a callable or None",
            ),
            (
                6,
                lambda index, wires, param: qp.CRX(param, index),
                np.random.rand(12),
                True,
                False,
                "Weight tensor must be of length",
            ),
        ],
    )
    def test_ccl2_exceptions(self, wires, acquaintances, weights, fermionic, shift, msg_match):
        """Test that TwoLocalSwapNetwork throws an exception if the parameters have illegal
        shapes, types or values."""

        dev = qp.device("default.qubit", wires=wires)

        @qp.qnode(dev)
        def circuit():
            qp.templates.TwoLocalSwapNetwork(
                dev.wires, acquaintances, weights, fermionic=fermionic, shift=shift
            )
            return qp.expval(qp.PauliZ(0))

        with pytest.raises(ValueError, match=msg_match):
            circuit()

    def test_ccl2_warnings(self):
        """Test that TwoLocalSwapNetwork throws correct warnings"""
        with pytest.warns(UserWarning, match="Weights are being provided without acquaintances"):
            qp.templates.TwoLocalSwapNetwork(
                wires=range(4),
                acquaintances=None,
                weights=np.array([1]),
                fermionic=True,
                shif=False,
            )

    def test_id(self):
        """Test that the id attribute can be set."""
        template = qp.templates.TwoLocalSwapNetwork(
            wires=range(4),
            acquaintances=None,
            weights=None,
            fermionic=True,
            shif=False,
            id="a",
        )
        assert template.id == "a"


class TestAttributes:
    """Test additional methods and attributes"""

    @pytest.mark.parametrize(
        "n_wires, expected_shape",
        [
            (2, (1,)),
            (4, (6,)),
            (5, (10,)),
            (6, (15,)),
        ],
    )
    def test_shape(self, n_wires, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor."""

        shape = qp.templates.TwoLocalSwapNetwork.shape(n_wires)
        assert shape == expected_shape

    def test_shape_exception_not_enough_qubits(self):
        """Test that the shape function warns if there are not enough qubits."""

        with pytest.raises(ValueError, match="TwoLocalSwapNetwork requires at least 2 wires"):
            qp.templates.TwoLocalSwapNetwork.shape(n_wires=1)


def circuit_template(weights):
    qp.templates.TwoLocalSwapNetwork(
        wires=range(4),
        acquaintances=lambda index, wires, param: qp.CRX(param, index),
        weights=weights,
        fermionic=True,
        shift=False,
    )
    return qp.expval(qp.PauliZ(0))


def circuit_decomposed(weights):
    for idx, pair in enumerate([[0, 1], [2, 3], [1, 2], [0, 1], [2, 3], [1, 2]]):
        qp.CRX(weights[idx], wires=pair)
    return qp.expval(qp.PauliZ(0))


class TestInterfaces:
    """Test that the template is compatible with all interfaces, including the computation
    of gradients."""

    def test_list_and_tuples(self, tol):
        """Test common iterables as inputs."""

        dev = qp.device("default.qubit", wires=4)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = qp.QNode(circuit_decomposed, dev)

        weights = [0.55, 0.72, 0.6, 0.54, 0.42, 0.65]
        res = circuit(weights)
        res2 = circuit2(weights)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

        weights_tuple = (0.55, 0.72, 0.6, 0.54, 0.42, 0.65)
        res = circuit(weights_tuple)
        res2 = circuit2(weights_tuple)

        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Test the autograd interface."""

        weights = qp.numpy.random.random(size=(6), requires_grad=True)

        dev = qp.device("default.qubit", wires=4)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = qp.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qp.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = qp.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Test the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=6))

        dev = qp.device("default.qubit", wires=4)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = qp.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax_jit(self, tol):
        """Test jit within the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=6))

        dev = qp.device("default.qubit", wires=4)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = jax.jit(circuit)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert qp.math.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Test the tf interface."""

        import tensorflow as tf

        weights = tf.Variable(np.random.random(size=6))

        dev = qp.device("default.qubit", wires=4)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = qp.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            res = circuit(weights)
        grads = tape.gradient(res, [weights])

        with tf.GradientTape() as tape2:
            res2 = circuit2(weights)
        grads2 = tape2.gradient(res2, [weights])

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Test the torch interface."""

        import torch

        weights = torch.tensor(np.random.random(size=6), requires_grad=True)

        dev = qp.device("default.qubit", wires=4)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = qp.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(weights)
        res.backward()
        grads = [weights.grad]

        res2 = circuit2(weights)
        res2.backward()
        grads2 = [weights.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)


# pylint: disable=too-few-public-methods
class TestGradient:
    """Test that the parameter-shift rule for this template matches that of backprop."""

    def test_ps_rule_gradient(self, tol):
        """Test parameter-shift rule gradient."""

        dev = qp.device("default.qubit", wires=4)

        backprop_grad = qp.grad(qp.QNode(circuit_template, dev, diff_method="backprop"))
        ps_rule_grad = qp.grad(qp.QNode(circuit_template, dev, diff_method="parameter-shift"))

        weights = qp.numpy.array([0.55, 0.72, 0.6, 0.54, 0.42, 0.65], requires_grad=True)
        res = backprop_grad(weights)
        res2 = ps_rule_grad(weights)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)
