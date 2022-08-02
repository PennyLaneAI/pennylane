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
Unit tests for the batch inputs transform.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np


def test_simple_circuit():
    """Test that batching works for a simple circuit"""
    dev = qml.device("default.qubit", wires=2)

    @qml.batch_input(argnum=1)
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(inputs, weights):
        qml.RY(weights[0], wires=0)
        qml.AngleEmbedding(inputs, wires=range(2), rotation="Y")
        qml.RY(weights[1], wires=1)
        return qml.expval(qml.PauliZ(1))

    batch_size = 5
    inputs = np.random.uniform(0, np.pi, (batch_size, 2))
    inputs.requires_grad = False
    weights = np.random.uniform(-np.pi, np.pi, (2,))

    res = circuit(inputs, weights)
    assert res.shape == (batch_size,)


def test_value_error():
    """Test if the batch_input raises relevant errors correctly"""

    dev = qml.device("default.qubit", wires=2)

    @qml.batch_input(argnum=[0, 2])
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(input1, input2, weights):
        qml.AngleEmbedding(input1, wires=range(2), rotation="Y")
        qml.RY(weights[0], wires=0)
        qml.RY(input2[0], wires=0)
        qml.RY(weights[1], wires=1)
        return qml.expval(qml.PauliZ(1))

    batch_size = 5
    input1 = np.random.uniform(0, np.pi, (batch_size, 2))
    input1.requires_grad = False
    input2 = np.random.uniform(0, np.pi, (4, 1))
    input2.requires_grad = False
    weights = np.random.uniform(-np.pi, np.pi, (2,))

    with pytest.raises(ValueError, match="Batch dimension for all gate arguments"):
        res = circuit(input1, input2, weights)


def test_batch_input_with_trainable_parameters_raises_error():
    """Test that using the batch_input method with trainable parameters raises a ValueError."""
    dev = qml.device("default.qubit", wires=2)

    @qml.batch_input(argnum=0)
    @qml.qnode(dev)
    def circuit(input):
        qml.RY(input, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(0.1, wires=0)
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

    batch_size = 3

    input = np.linspace(0.1, 0.5, batch_size, requires_grad=True)

    with pytest.raises(
        ValueError,
        match="Batched inputs must be non-trainable."
        + " Please make sure that the parameters indexed by "
        + "'argnum' are not marked as trainable.",
    ):
        circuit(input)


def test_mottonenstate_preparation(mocker):
    """Test that batching works for MottonenStatePreparation"""
    dev = qml.device("default.qubit", wires=3)

    @qml.batch_input(argnum=0)
    @qml.qnode(dev)
    def circuit(data, weights):
        qml.templates.MottonenStatePreparation(data, wires=[0, 1, 2])
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 1, 2])

    batch_size = 3

    # create a batched input statevector
    data = np.random.random((batch_size, 2**3), requires_grad=False)
    data /= np.linalg.norm(data, axis=1).reshape(-1, 1)  # normalize

    # weights is not batched
    weights = np.random.random((10, 3, 3), requires_grad=True)

    spy = mocker.spy(circuit.device, "batch_execute")
    res = circuit(data, weights)
    assert res.shape == (batch_size, 2**3)
    assert len(spy.call_args[0][0]) == batch_size

    # check the results against individually executed circuits (no batching)
    @qml.qnode(dev)
    def circuit2(data, weights):
        qml.templates.MottonenStatePreparation(data, wires=[0, 1, 2])
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 1, 2])

    indiv_res = []
    for state in data:
        indiv_res.append(circuit2(state, weights))
    assert np.allclose(res, indiv_res)


@pytest.mark.autograd
@pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift"])
def test_autograd(diff_method, tol):
    """Test derivatives when using autograd"""
    dev = qml.device("default.qubit", wires=2)

    @qml.batch_input(argnum=0)
    @qml.qnode(dev, diff_method=diff_method)
    def circuit(input, x):
        qml.RY(input, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

    batch_size = 3

    def cost(input, x):
        return np.sum(circuit(input, x))

    input = np.linspace(0.1, 0.5, batch_size, requires_grad=False)
    x = np.array(0.1, requires_grad=True)

    res = qml.grad(cost)(input, x)
    expected = -np.sin(0.1) * sum(np.sin(input))
    assert np.allclose(res, expected, atol=tol, rtol=0)
