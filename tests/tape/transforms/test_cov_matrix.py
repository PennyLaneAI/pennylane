# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the covariance matrix QNode transform"""
import pytest

import pennylane as qml
from pennylane import numpy as np

from pennylane.tape.transforms import functions as fn
from pennylane.tape.transforms.cov_matrix import cov_matrix


def ansatz(weights):
    qml.RX(weights[0, 0], wires=0)
    qml.RX(weights[0, 1], wires=1)
    qml.RX(weights[0, 2], wires=2)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])

    qml.RX(weights[1, 0], wires=0)
    qml.RX(weights[1, 1], wires=1)
    qml.RX(weights[1, 2], wires=2)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])

    qml.RX(weights[1, 0], wires=0)
    qml.RX(weights[1, 1], wires=1)
    qml.RX(weights[1, 2], wires=2)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])


def expected_covariance(weights):
    """calculate the expected covariance matrix and gradient manually"""
    dev = qml.device("default.qubit", wires=3)

    @qml.tape.qnode(dev, interface="autograd")
    def circuit_var(weights):
        ansatz(weights)
        return qml.var(qml.PauliX(0) @ qml.PauliZ(1)), qml.var(qml.PauliY(2))

    @qml.tape.qnode(dev, interface="autograd")
    def circuit_expval(weights):
        ansatz(weights)
        return qml.expval(qml.PauliX(0) @ qml.PauliZ(1)), qml.expval(qml.PauliY(2))

    @qml.tape.qnode(dev, interface="autograd")
    def circuit_expval2(weights):
        ansatz(weights)
        return qml.expval(qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliY(2))

    def cost(weights):
        off_diag = circuit_expval2(weights) - circuit_expval(weights)[0] * circuit_expval(weights)[1]
        res = np.diag(circuit_var(weights))
        res = res + np.array([[0, off_diag], [off_diag, 0]])
        return res

    grad_fn = qml.grad(lambda weights: cost(weights)[0, 1])
    return cost(weights), grad_fn(weights)


def test_autograd(tol):
    """Test that the covariance matrix evaluates correctly when using the
    autograd interface"""
    dev = qml.device("default.qubit", wires=3)
    weights = np.random.random(size=[3, 3], requires_grad=True)

    @qml.tape.qnode(dev, interface="autograd")
    def circuit(weights):
        ansatz(weights)
        return qml.expval(qml.PauliX(0) @ qml.PauliZ(1)), qml.expval(qml.PauliY(2))

    cov_circuit = cov_matrix(circuit)

    res = cov_circuit(weights)
    expected_res, expected_grad = expected_covariance(weights)
    assert np.allclose(res, expected_res, atol=tol, rtol=0)

    grad_fn = qml.grad(lambda weights: cov_circuit(weights)[0, 1])
    res = grad_fn(weights)
    assert np.allclose(res, expected_grad, atol=tol, rtol=0)


def test_torch(tol):
    """Test that the covariance matrix evaluates correctly when using the
    torch interface"""
    torch = pytest.importorskip("torch", minversion="1.3")

    dev = qml.device("default.qubit", wires=3)
    weights = torch.tensor(np.random.random(size=[3, 3]), requires_grad=True)

    @qml.tape.qnode(dev, interface="torch")
    def circuit(weights):
        ansatz(weights)
        return qml.expval(qml.PauliX(0) @ qml.PauliZ(1)), qml.expval(qml.PauliY(2))

    cov_circuit = cov_matrix(circuit)

    res = cov_circuit(weights)
    expected_res, expected_grad = expected_covariance(weights.detach().numpy())
    assert np.allclose(res.detach().numpy(), expected_res, atol=tol, rtol=0)

    loss = res[0, 1]
    loss.backward()
    assert np.allclose(weights.grad, expected_grad, atol=tol, rtol=0)


def test_tf(tol):
    """Test that the covariance matrix evaluates correctly when using the
    TF interface"""
    tf = pytest.importorskip("tensorflow", minversion="2.1")

    dev = qml.device("default.qubit", wires=3)
    weights = tf.Variable(np.random.random(size=[3, 3]))

    @qml.tape.qnode(dev, interface="tf")
    def circuit(weights):
        ansatz(weights)
        return qml.expval(qml.PauliX(0) @ qml.PauliZ(1)), qml.expval(qml.PauliY(2))

    cov_circuit = cov_matrix(circuit)

    with tf.GradientTape() as tape:
        res = cov_circuit(weights)
        loss = res[0, 1]

    expected_res, expected_grad = expected_covariance(weights.numpy())
    assert np.allclose(res, expected_res, atol=tol, rtol=0)

    grad = tape.gradient(loss, weights)
    assert np.allclose(grad, expected_grad, atol=tol, rtol=0)
