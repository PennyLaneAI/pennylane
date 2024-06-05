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
Unit tests for the DefaultTensor class.
"""


import numpy as np
import pytest

import pennylane as qml

quimb = pytest.importorskip("quimb")

pytestmark = pytest.mark.external


class TestMultiQubitGates:
    """Test that the DefaultTensor device can apply multi-qubit gates."""

    def test_multirz(self):
        """Test that the device can apply a 20-qubit MultiRZ gate."""

        wires = 20
        dev = qml.device("default.tensor", wires=wires, method="mps")

        np.random.seed(0)
        state = np.random.rand(2**wires) + 1j * np.random.rand(2**wires)
        state /= np.linalg.norm(state)

        def circuit():
            qml.StatePrep(state, range(wires))
            qml.MultiRZ(0.1, wires=range(1, wires - 1))
            return qml.state()

        _ = qml.QNode(circuit, dev)()

    def test_paulirot(self):
        """Test that the device can apply a 20-qubit PauliRot gate."""

        wires = 20
        dev = qml.device("default.tensor", wires=wires, method="mps")

        np.random.seed(0)
        state = np.random.rand(2**wires) + 1j * np.random.rand(2**wires)
        state /= np.linalg.norm(state)

        def circuit():
            qml.StatePrep(state, range(wires))
            qml.PauliRot(0.1, "XY" * (wires // 2 - 1), wires=range(1, wires - 1))
            return qml.state()

        _ = qml.QNode(circuit, dev)()

    def test_qft(self):
        """Test that the device can apply a 20-qubit QFT gate."""

        wires = 20
        dev = qml.device("default.tensor", wires=wires, method="mps")

        def circuit(basis_state):
            qml.BasisState(basis_state, wires=range(wires))
            qml.QFT(wires=range(wires))
            return qml.state()

        _ = qml.QNode(circuit, dev)(np.array([0, 1] * (wires // 2)))
