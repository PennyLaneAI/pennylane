# Copyright 2024 Xanadu Quantum Technologies Inc.

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


@pytest.mark.parametrize("method", ["mps", "tn"])
class TestMultiQubitGates:
    """Test that the DefaultTensor device can apply multi-qubit gates."""

    def test_multirz(self, method):
        """Test that the device can apply a multi-qubit MultiRZ gate."""
        wires = 16
        dev = qml.device("default.tensor", wires=wires, method=method)

        state = np.random.rand(2**wires) + 1j * np.random.rand(2**wires)
        state /= np.linalg.norm(state)

        def circuit():
            qml.StatePrep(state, range(wires))
            qml.MultiRZ(0.1, wires=range(1, wires - 1))
            return qml.state()

        _ = qml.QNode(circuit, dev)()

    def test_paulirot(self, method):
        """Test that the device can apply a multi-qubit PauliRot gate."""
        wires = 16
        dev = qml.device("default.tensor", wires=wires, method=method)

        state = np.random.rand(2**wires) + 1j * np.random.rand(2**wires)
        state /= np.linalg.norm(state)

        def circuit():
            qml.StatePrep(state, range(wires))
            qml.PauliRot(0.1, "XY" * (wires // 2 - 1), wires=range(1, wires - 1))
            return qml.state()

        _ = qml.QNode(circuit, dev)()

    def test_qft(self, method):
        """Test that the device can apply a multi-qubit QFT gate."""
        if method == "tn":
            pytest.skip("Test is too costly with the TN method.")
        wires = 16
        dev = qml.device("default.tensor", wires=wires, method=method, max_bond_dim=128)

        def circuit(basis_state):
            qml.BasisState(basis_state, wires=range(wires))
            qml.QFT(wires=range(wires))
            return qml.state()

        _ = qml.QNode(circuit, dev)(np.array([0, 1] * (wires // 2)))

    def test_trotter_product(self, method):
        """Test that the device can apply a multi-qubit TrotterProduct gate."""

        wires = 16
        dev = qml.device("default.tensor", wires=wires, method=method)

        coeffs = [0.25, 0.75]
        ops = [qml.X(0), qml.Z(0)]
        H = qml.dot(coeffs, ops)

        @qml.qnode(dev)
        def circuit():
            # Prepare some state
            qml.Hadamard(0)

            # Evolve according to H
            qml.TrotterProduct(H, time=2.4, order=2)

            # Measure some quantity
            return qml.state()

        _ = qml.QNode(circuit, dev)()


@pytest.mark.parametrize("method", ["mps", "tn"])
class TestMultiQubitMeasurements:
    """Test that the DefaultTensor device can compute multi-qubit measurements."""

    def test_prod(self, method):
        """Test that the device can compute the expval of a multi-qubit Prod."""

        wires = 30
        dev = qml.device("default.tensor", wires=wires, method=method)

        def circuit():
            return qml.expval(qml.ops.op_math.Prod(*(qml.PauliY(i) for i in range(wires))))

        _ = qml.QNode(circuit, dev)()

    def test_hamiltonian(self, method):
        """Test that the device can compute the expval of a multi-qubit Hamiltonian."""

        wires = 30
        dev = qml.device("default.tensor", wires=wires, method=method)

        def circuit():
            return qml.expval(
                qml.Hamiltonian(
                    [1.0], [qml.ops.op_math.Prod(*(qml.PauliY(i) for i in range(wires)))]
                )
            )

        _ = qml.QNode(circuit, dev)()

    def test_linear_combination(self, method):
        """Test that the device can compute the expval of a multi-qubit LinearCombination."""

        wires = 30
        dev = qml.device("default.tensor", wires=wires, method=method)

        def circuit():
            return qml.expval(
                qml.ops.LinearCombination(
                    [1.0], [qml.ops.op_math.Prod(*(qml.PauliY(i) for i in range(wires)))]
                )
            )

        _ = qml.QNode(circuit, dev)()
