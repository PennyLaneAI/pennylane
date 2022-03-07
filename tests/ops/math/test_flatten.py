# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the operator math utils.
"""
import pennylane as qml
from pennylane.ops.math.utils import flatten_decomposition, flatten_terms
from numpy import np


def test_flatten_decomposition():
    """Test that a multi-operator product is flattened correctly."""
    op = (
        qml.Hadamard(wires=1)
        @ qml.PauliX(wires=0)
        @ qml.PauliX(wires=5)
        @ qml.PauliX(wires=10)
        @ qml.PauliX(wires=1)
    )
    expected = [
        qml.Hadamard(wires=1),
        qml.PauliX(wires=0),
        qml.PauliX(wires=5),
        qml.PauliX(wires=10),
        qml.PauliX(wires=1),
    ]
    res = flatten_decomposition(op.decomposition())
    for op, op_expected in zip(res, expected):
        assert op.name == op_expected.name
        assert op.wires == op_expected.wires


def test_flatten_terms():
    """Test that a nested sum of terms is flattened correctly."""
    op = (
        qml.Hadamard(wires=1)
        + 2.0 * qml.PauliX(wires=0)
        + qml.PauliY(wires=5)
        + 3 * (1.0j * qml.PauliX(wires=10) + qml.PauliX(wires=1))
    )
    res = flatten_terms(*op.terms())
    assert np.allclose(res[0], [1.0, 2.0, 1.0, 1j, 3])
    assert res[1][0].name == "Hadamard"
    assert res[1][0].wires.tolist() == [1]
    assert res[1][0].name == "PauliX"
    assert res[1][0].wires.tolist() == [0]
    assert res[1][0].name == "PauliY"
    assert res[1][0].wires.tolist() == [5]
    assert res[1][0].name == "PauliX"
    assert res[1][0].wires.tolist() == [10]
    assert res[1][0].name == "PauliX"
    assert res[1][0].wires.tolist() == [1]
