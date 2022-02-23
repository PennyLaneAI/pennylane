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
import functools
import pytest

import pennylane as qml
from pennylane import numpy as np


def test_simple_circuit():
    """Test that batching works for a simple circuit"""
    dev = qml.device("default.qubit", wires=2)

    @qml.batch_input(argnum=0)
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(2), rotation="Y")
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        return qml.expval(qml.PauliZ(1))

    batch_size = 5
    inputs = np.random.uniform(0, np.pi, (batch_size, 2))
    inputs.requires_grad = False
    weights = np.random.uniform(-np.pi, np.pi, (2,))

    res = circuit(inputs, weights)
    assert res.shape == (batch_size,)


def test_default():
    """Test that batching works for a simple circuit"""
    dev = qml.device("default.qubit", wires=2)

    @qml.batch_input(argnum=None)
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(2), rotation="Y")
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        return qml.expval(qml.PauliZ(1))

    batch_size = 5
    inputs = np.random.uniform(0, np.pi, (2,))
    inputs.requires_grad = False
    weights = np.random.uniform(-np.pi, np.pi, (2,))

    res = circuit(inputs, weights)
    assert res.shape == (1,)