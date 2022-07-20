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
Tests for classical shadows
"""
import pytest
import pennylane as qml
from pennylane.shadows import ClassicalShadow

def test_pauli_string_expval():
    """Testing the output of expectation values match those of exact evaluation"""
    wires = range(3)

    dev = qml.device("default.qubit", wires=wires, shots=10000)
    @qml.qnode(dev)
    def qnode():
        qml.Hadamard(0)
        qml.Hadamard(1)
        return qml.classical_shadow(wires=wires)

    bitstrings, recipes = qnode()
    shadow = ClassicalShadow(bitstrings, recipes)
    observable = qml.PauliZ(0) @ qml.PauliZ(1)
    res = shadow.expval_observable(observable, 2)

    res_exact = 1.
    assert qml.math.allclose(res, res_exact, atol=1e-1)