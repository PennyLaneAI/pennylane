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

wires = range(5)
shots = 10000
dev = qml.device("default.qubit", wires=wires, shots=shots)
@qml.qnode(dev)
def qnode(n_wires):
    for i in range(n_wires):
        qml.Hadamard(i)
    return qml.classical_shadow(wires=range(n_wires))

shadows = [ClassicalShadow(*qnode(n_wires)) for n_wires in range(1, 5)]


def test_unittest_local_snapshots(shadow):
    """Test the output shape of local_snapshots method"""
    T, n = shadow.bitstrings.shape
    assert all((T, n) == shadow.recipes.shape)
    assert all(shadow.local_snapshots().shape == (T, n, 2, 2))

def test_unittest_global_snapshots(shadow):
    """Test the output shape of global_snapshots method"""
    T, n = shadow.bitstrings.shape
    assert all((T, n) == shadow.recipes.shape)
    assert all(shadow.global_snapshots().shape == (T, 2**n, 2**n))



def test_pauli_string_expval(shadow):
    """Testing the output of expectation values match those of exact evaluation"""

    observable = qml.PauliZ(0) @ qml.PauliZ(1)
    res = shadow.expval_observable(observable, 2)

    res_exact = 1.
    assert qml.math.allclose(res, res_exact, atol=1e-1)

def test_expval_H():
    """Testing the output of expectation values match those of exact evaluation"""