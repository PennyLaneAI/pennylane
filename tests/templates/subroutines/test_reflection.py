# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for the Reflection Operator template
"""

import numpy as np
import pytest

import pennylane as qml
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@qml.prod
def hadamards(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)


@pytest.mark.jax
def test_standard_validity():
    """Test standard validity criteria using assert_valid."""
    op = qml.Reflection(qml.Hadamard(wires=0), 0.5, reflection_wires=[0])
    qml.ops.functions.assert_valid(op)


@pytest.mark.parametrize(
    ("prod", "reflection_wires"),
    [
        (qml.QFT([0, 1, 4]), [0, 1, 2]),
        (qml.QFT([0, 1, 2]), [3]),
        (qml.QFT([0, 1, 2]), [0, 1, 2, 3]),
    ],
)
def test_reflection_wires(prod, reflection_wires):
    """Assert that the reflection wires are a subset of the input operation wires"""
    with pytest.raises(
        ValueError, match="The reflection wires must be a subset of the operation wires."
    ):
        qml.Reflection(prod, 0.5, reflection_wires=reflection_wires)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (
            qml.Reflection(qml.Hadamard(wires=0), 0.5, reflection_wires=[0]),
            [
                qml.GlobalPhase(np.pi),
                qml.adjoint(qml.Hadamard(0)),
                qml.PauliX(wires=[0]),
                qml.PhaseShift(0.5, wires=[0]),
                qml.PauliX(wires=[0]),
                qml.Hadamard(0),
            ],
        ),
        (
            qml.Reflection(qml.QFT(wires=[0, 1]), 0.5),
            [
                qml.GlobalPhase(np.pi),
                qml.adjoint(qml.QFT(wires=[0, 1])),
                qml.PauliX(wires=[1]),
                qml.ctrl(qml.PhaseShift(0.5, wires=[1]), control=0, control_values=[0]),
                qml.PauliX(wires=[1]),
                qml.QFT(wires=[0, 1]),
            ],
        ),
    ],
)
def test_decomposition(op, expected):
    """Test that the decomposition of the Reflection operator is correct"""
    decomp = op.decomposition()
    assert decomp == expected


@pytest.mark.parametrize(
    ("op"),
    [
        qml.Reflection(qml.Hadamard(wires=0), 0.5, reflection_wires=[0]),
        qml.Reflection(qml.QFT(wires=[0, 1]), 0.5),
    ],
)
def test_decomposition_new(op):
    """Tests the decomposition rule implemented with the new system."""
    for rule in qml.list_decomps(qml.Reflection):
        _test_decomposition_rule(op, rule)


def test_default_values():
    """Test that the default values are correct"""

    U = qml.QFT(wires=[0, 1, 4])
    op = qml.Reflection(U)

    assert op.alpha == np.pi
    assert op.reflection_wires == U.wires


@pytest.mark.parametrize("n_wires", [3, 4, 5])
def test_grover_as_reflection(n_wires):
    """Test that the GroverOperator can be used as a Reflection operator"""

    grover_matrix = qml.matrix(qml.GroverOperator(wires=range(n_wires)))
    reflection_matrix = qml.matrix(qml.Reflection(hadamards(wires=range(n_wires))))

    assert np.allclose(grover_matrix, reflection_matrix)


class TestIntegration:
    """Tests that the Reflection is executable and differentiable in a QNode context"""

    @staticmethod
    def circuit(alpha):
        """Test circuit"""
        qml.RY(1.2, wires=0)
        qml.RY(-1.4, wires=1)
        qml.RX(-2, wires=0)
        qml.CRX(1, wires=[0, 1])
        qml.Reflection(hadamards(range(3)), alpha)
        return qml.probs(wires=range(3))

    x = np.array(0.25)

    # obtained with autograd, we are only ensuring that the results are consistent accross interfaces

    exp_result = np.array(
        [
            2.48209280e-01,
            2.79302360e-05,
            1.76417324e-01,
            2.79302360e-05,
            3.18107276e-01,
            2.79302360e-05,
            2.57154398e-01,
            2.79302360e-05,
        ]
    )
    exp_jac = np.array(
        [
            -0.00322306,
            0.00022228,
            0.00312425,
            0.00022228,
            0.01377867,
            0.00022228,
            -0.01456896,
            0.00022228,
        ]
    )

    def test_qnode_numpy(self):
        """Test that the QNode executes with Numpy."""
        dev = qml.device("default.qubit")
        qnode = qml.QNode(self.circuit, dev, interface=None)

        res = qnode(self.x)
        assert res.shape == (8,)
        assert np.allclose(res, self.exp_result, atol=0.002)

    def test_lightning_qubit(self):
        """Test that the QNode executes with the Lightning Qubit simulator."""
        dev = qml.device("lightning.qubit", wires=3)
        qnode = qml.QNode(self.circuit, dev)

        res = qnode(self.x)
        assert res.shape == (8,)
        assert np.allclose(res, self.exp_result, atol=0.002)

    # NOTE: the finite shot test of the results has a 3% chance to fail
    # due to the random nature of the sampling. Hence we just pin the salt
    @pytest.mark.autograd
    @pytest.mark.parametrize("shots", [None, 50000])
    def test_qnode_autograd(self, shots, seed):
        """Test that the QNode executes with Autograd."""

        dev = qml.device("default.qubit", wires=3, seed=seed)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.set_shots(
            qml.QNode(self.circuit, dev, interface="autograd", diff_method=diff_method), shots=shots
        )

        x = qml.numpy.array(self.x, requires_grad=True)
        res = qnode(x)
        assert qml.math.shape(res) == (8,)

        assert np.allclose(res, self.exp_result, atol=0.005)

        res = qml.jacobian(qnode)(x)
        assert np.shape(res) == (8,)

        assert np.allclose(res, self.exp_jac, atol=0.005)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("shots", [None, 50000])
    def test_qnode_jax(self, shots, use_jit, seed):
        """Test that the QNode executes and is differentiable with JAX. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import jax

        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit", seed=seed)

        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.set_shots(
            qml.QNode(self.circuit, dev, interface="jax", diff_method=diff_method), shots=shots
        )
        if use_jit:
            qnode = jax.jit(qnode)

        x = jax.numpy.array(self.x)
        res = qnode(x)
        assert qml.math.shape(res) == (8,)

        assert np.allclose(res, self.exp_result, atol=0.005)

        jac_fn = jax.jacobian(qnode)
        if use_jit:
            jac_fn = jax.jit(jac_fn)

        jac = jac_fn(x)
        assert jac.shape == (8,)

        assert np.allclose(jac, self.exp_jac, atol=0.005)

    @pytest.mark.torch
    @pytest.mark.parametrize("shots", [None, 50000])
    def test_qnode_torch(self, shots, seed):
        """Test that the QNode executes and is differentiable with Torch. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""

        import torch

        dev = qml.device("default.qubit", seed=seed)

        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.set_shots(
            qml.QNode(self.circuit, dev, interface="torch", diff_method=diff_method), shots=shots
        )

        x = torch.tensor(self.x, requires_grad=True)
        res = qnode(x)
        assert qml.math.shape(res) == (8,)

        assert qml.math.allclose(res, self.exp_result, atol=0.005)

        jac = torch.autograd.functional.jacobian(qnode, x)
        assert qml.math.shape(jac) == (8,)

        assert qml.math.allclose(jac, self.exp_jac, atol=0.005)

    @pytest.mark.tf
    @pytest.mark.parametrize("shots", [None, 50000])
    @pytest.mark.xfail(reason="tf gradient doesn't seem to be working, returns ()")
    def test_qnode_tf(self, shots, seed):
        """Test that the QNode executes and is differentiable with TensorFlow. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import tensorflow as tf

        dev = qml.device("default.qubit", seed=seed)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.set_shots(
            qml.QNode(self.circuit, dev, interface="tf", diff_method=diff_method), shots=shots
        )

        x = tf.Variable(self.x)
        with tf.GradientTape() as tape:
            res = qnode(x)

        assert qml.math.shape(res) == (8,)
        assert qml.math.allclose(res, self.exp_result, atol=0.002)

        jac = tape.gradient(res, x)
        assert qml.math.shape(jac) == (8,)


def test_correct_queueing():
    """Test that the Reflection operator is correctly queued in the circuit"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit1():
        qml.Hadamard(wires=0)
        qml.RY(2, wires=0)
        qml.CRY(1, wires=[0, 1])
        qml.Reflection(U=qml.Hadamard(wires=0), alpha=2.0)
        return qml.state()

    @qml.prod
    def generator(wires):
        qml.Hadamard(wires=wires)

    @qml.qnode(dev)
    def circuit2():
        generator(wires=0)
        qml.RY(2, wires=0)
        qml.CRY(1, wires=[0, 1])
        qml.Reflection(U=generator(wires=0), alpha=2.0)
        return qml.state()

    U = generator(0)

    @qml.qnode(dev)
    def circuit3():
        generator(wires=0)
        qml.RY(2, wires=0)
        qml.CRY(1, wires=[0, 1])
        qml.Reflection(U=U, alpha=2.0)
        return qml.state()

    assert np.allclose(circuit1(), circuit2())
    assert np.allclose(circuit1(), circuit3())


@pytest.mark.parametrize(
    "state", [[1 / np.sqrt(3), np.sqrt(2 / 3)], [1 / np.sqrt(4), np.sqrt(3 / 4)]]
)
def test_correct_reflection(state):
    """Test that the Reflection operator is correctly applied to the state."""

    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(state, wires=0)
        qml.Reflection(U=qml.Hadamard(wires=0))
        return qml.state()

    output = circuit()
    expected = np.array(output[::-1])

    assert np.allclose(state, expected)
