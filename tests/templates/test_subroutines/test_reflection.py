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
Tests for the Reflection Operator template
"""

import pytest
import numpy as np
import pennylane as qml


@pytest.mark.parametrize(
    ("prod", "reflection_wires"),
    [
        (qml.QFT([0, 1, 4]), [0, 1, 2]),
        (qml.QFT([0, 1, 2]), [3]),
        (qml.QFT([0, 1, 2]), [0, 1, 2, 3]),
    ],
)
def test_reflection_wires(prod, reflection_wires):
    """Assert reflection_wires is a subset of the U wires"""
    with pytest.raises(ValueError, match="The reflection_wires must be a subset of the U wires."):
        qml.Reflection(prod, 0.5, reflection_wires=reflection_wires)


def test_decomposition_one_wire():
    """Test that the decomposition of the Reflection operator is correct"""
    op = qml.Reflection(qml.Hadamard(wires=0), 0.5, reflection_wires=[0])
    decomp = op.decomposition()
    expected = [
        qml.GlobalPhase(np.pi),
        qml.adjoint(qml.Hadamard(0)),
        qml.PauliX(wires=[0]),
        qml.PhaseShift(0.5, wires=[0]),
        qml.PauliX(wires=[0]),
        qml.Hadamard(0),
    ]
    assert decomp == expected


def test_decomposition_two_wire():
    """Test that the decomposition of the Reflection operator is correct"""
    op = qml.Reflection(qml.QFT(wires=[0, 1]), 0.5)

    decomp = op.decomposition()
    expected = [
        qml.GlobalPhase(np.pi),
        qml.adjoint(qml.QFT(wires=[0, 1])),
        qml.PauliX(wires=[1]),
        qml.ctrl(qml.PhaseShift(0.5, wires=[1]), control=0, control_values=[0]),
        qml.PauliX(wires=[1]),
        qml.QFT(wires=[0, 1]),
    ]

    assert decomp == expected


def test_default_values():
    """Test that the default values are correct"""

    U = qml.QFT(wires=[0, 1, 4])
    op = qml.Reflection(U)

    assert op.alpha == np.pi
    assert op.reflection_wires == U.wires


@pytest.mark.parametrize("n_wires", [3, 4, 5])
def test_grover_as_reflection(n_wires):
    """Test that the GroverOperator can be used as a Reflection operator"""

    @qml.prod
    def hadamards(wires):
        for wire in wires:
            qml.Hadamard(wires=wire)

    grover_matrix = qml.matrix(qml.GroverOperator(wires=range(n_wires)))
    reflection_matrix = qml.matrix(qml.Reflection(hadamards(wires=range(n_wires))))

    assert np.allclose(grover_matrix, reflection_matrix)


@pytest.mark.tf
@pytest.mark.jax
@pytest.mark.torch
@pytest.mark.parametrize("value", [1.2, 2.1, 3.4])
def test_gradients_all_interfaces(value):
    import torch
    import jax
    import pennylane.numpy as pnp
    import tensorflow as tf

    n_wires = 3

    @qml.prod
    def hadamards(wires):
        for wire in wires:
            qml.Hadamard(wires=wire)

    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit(alpha):
        qml.RY(1.2, wires=0)
        qml.RY(-1.4, wires=1)
        qml.RX(-2, wires=0)
        qml.CRX(1, wires=[0, 1])
        qml.Reflection(hadamards(range(n_wires)), alpha)

        return qml.expval(qml.PauliZ(0))

    def cost(alpha):
        return circuit(alpha)

    alpha = pnp.array([value], requires_grad=True)
    grad_autograd = qml.grad(cost)(alpha)

    grad_jax = jax.grad(cost)(jax.numpy.array(value, dtype=jax.numpy.float32))

    x = torch.tensor([value], requires_grad=True)

    y = cost(x)
    y.backward()
    grad_torch = x.grad

    x = tf.Variable([value], dtype=tf.float32)

    with tf.GradientTape() as tape:
        y = cost(x)

    grad_tf = tape.gradient(y, x)

    assert np.allclose(grad_autograd, grad_jax, atol=1e-3)
    assert np.allclose(grad_autograd, grad_torch, atol=1e-3)
    assert np.allclose(grad_autograd, grad_tf, atol=1e-3)


def test_lightning_qubit():
    dev1 = qml.device("lightning.qubit", wires=2)

    @qml.qnode(dev1)
    def circuit1():
        qml.RX(2, wires=0)
        qml.CRY(1, wires=[0, 1])
        qml.Reflection(U=qml.PauliX(wires=0), alpha=2.0)
        return qml.probs(wires=[0, 1])

    dev2 = qml.device("default.qubit", wires=2)

    @qml.qnode(dev2)
    def circuit2():
        qml.RX(2, wires=0)
        qml.CRY(1, wires=[0, 1])
        qml.Reflection(U=qml.PauliX(wires=0), alpha=2.0)
        return qml.probs(wires=[0, 1])

    assert np.allclose(circuit1(), circuit2())


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
