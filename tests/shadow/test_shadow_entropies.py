# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for shadow entropies"""
# pylint:disable=no-self-use, import-outside-toplevel, redefined-outer-name, unpacking-non-sequence, too-few-public-methods, not-an-iterable, inconsistent-return-statements

import pytest

import pennylane as qml
import pennylane.numpy as np
from pennylane.shadows import ClassicalShadow

np.random.seed(777)

def max_entangled_circuit(wires, shots=10000, interface="autograd"):
    """maximally entangled state preparation circuit"""
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.Hadamard(wires=0)
        for i in range(1, wires):
            qml.CNOT(wires=[0, i])
        return qml.classical_shadow(wires=range(wires))

    return circuit

class TestShadowEntropies:
    """Tests for entropies in ClassicalShadow class"""
    @pytest.mark.parametrize("n_wires", [2, 3, 4])
    def test_constant_distribution(self, n_wires):
        """Test for state with constant eigenvalues of reduced state that all entropies are the same"""

        bits, recipes = max_entangled_circuit(wires=n_wires)()
        shadow = ClassicalShadow(bits, recipes)

        entropies = [shadow.entropy(wires=[0], alpha=alpha, atol=1e-2) for alpha in [1, 2, 3]]
        assert np.allclose(entropies, entropies[0], atol=1e-2)
