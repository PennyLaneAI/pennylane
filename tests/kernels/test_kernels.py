# Copyright 

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
Unit tests for the :mod:`pennylane` kernels module.
"""
import pennylane as qml
import pennylane.kernels as kern
import pytest
import numpy as np

@qml.template
def _simple_ansatz(x, params):
    qml.RX(params[0], wires=[0])
    qml.RZ(x, wires=[0])
    qml.RX(params[1], wires=[0])

class TestEmbeddingKernel:

    def test_construction(self):
        dev = qml.device("default.qubit", wires=1)
        k = kern.EmbeddingKernel(_simple_ansatz, dev)

        assert k.probs_qnode is not None

    @pytest.mark.parametrize("x1", np.linspace(0, 2*np.pi, 5))
    @pytest.mark.parametrize("x2", np.linspace(0, 2*np.pi, 5))
    def test_value_range(self, x1, x2):
        dev = qml.device("default.qubit", wires=1)
        k = kern.EmbeddingKernel(_simple_ansatz, dev)
        params = np.array([0.5, 0.9])

        val = k(x1, x2, params)

        assert 0 <= val
        assert val <= 1

    def test_known_values(self):
        dev = qml.device("default.qubit", wires=1)
        k = kern.EmbeddingKernel(_simple_ansatz, dev)
        params = np.array([0.5, 0.9])

        val = k(0.1, 0.1, params)

        assert val == pytest.approx(1.0)


