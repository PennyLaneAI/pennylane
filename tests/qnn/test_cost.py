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
"""
Tests for the pennylane.qnn.cost module.
"""
import warnings

import numpy as np
import pytest

import pennylane as qml
from pennylane.exceptions import PennyLaneDeprecationWarning
from pennylane.qnn.cost import SquaredErrorLoss


@pytest.fixture(autouse=True)
def suppress_warnings():
    """
    A fixture that suppresses all PL deprecation warnings for the tests that use it.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=PennyLaneDeprecationWarning,
            message="qml.qnn.cost.SquaredErrorLoss is deprecated and will be removed",
        )
        yield


def test_deprecation_warning():
    """Test that the deprecation warning is raised when importing SquaredErrorLoss."""
    with pytest.warns(
        PennyLaneDeprecationWarning,
        match="qml.qnn.cost.SquaredErrorLoss is deprecated and will be removed",
    ):
        SquaredErrorLoss(rx_ansatz, [qml.PauliZ(0)], qml.device("default.qubit", wires=1))


# pylint: disable=unused-argument
def rx_ansatz(phis, **kwargs):
    for w, phi in enumerate(phis):
        qml.RX(phi, wires=w)


# pylint: disable=unused-argument
def layer_ansatz(weights, x=None, **kwargs):
    qml.templates.AngleEmbedding(x, wires=[0, 1, 2])
    qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])


@pytest.mark.autograd
class TestSquaredErrorLossAutograd:
    def test_no_target(self):
        with pytest.raises(ValueError, match="The target cannot be None"):
            num_qubits = 1

            dev = qml.device("default.qubit", wires=num_qubits)
            observables = [qml.PauliZ(0)]
            loss = SquaredErrorLoss(rx_ansatz, observables, dev)

            phis = np.ones(num_qubits)
            loss(phis)

    def test_invalid_target(self):
        with pytest.raises(ValueError, match="Input target of incorrect length 2 instead of 1"):
            num_qubits = 1

            dev = qml.device("default.qubit", wires=num_qubits)
            observables = [qml.PauliZ(0)]
            loss = SquaredErrorLoss(rx_ansatz, observables, dev)

            phis = np.ones(num_qubits)
            loss(phis, target=np.array([1.0, 2.0]))

    def test_layer_circuit(self):
        num_qubits = 3

        dev = qml.device("default.qubit", wires=num_qubits)
        observables = [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]
        loss = SquaredErrorLoss(layer_ansatz, observables, dev)

        weights = np.ones((num_qubits, 3, 3))
        res = loss(weights, x=np.array([1.0, 2.0, 1.0]), target=np.array([1.0, 0.5, 0.1]))

        assert np.allclose(res, np.array([0.88, 0.83, 0.05]), atol=0.01, rtol=0.01)

    def test_rx_circuit(self):
        num_qubits = 3

        dev = qml.device("default.qubit", wires=num_qubits)
        observables = [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]
        loss = SquaredErrorLoss(rx_ansatz, observables, dev)

        phis = np.ones(num_qubits)
        res = loss(phis, target=np.array([1.0, 0.5, 0.1]))

        assert np.allclose(res, np.array([0.21, 0.25, 0.03]), atol=0.01, rtol=0.01)


@pytest.mark.torch
class TestSquaredErrorLossTorch:
    def test_no_target(self):
        with pytest.raises(ValueError, match="The target cannot be None"):
            num_qubits = 1

            dev = qml.device("default.qubit", wires=num_qubits)
            observables = [qml.PauliZ(0)]
            loss = SquaredErrorLoss(rx_ansatz, observables, dev)

            phis = np.ones(num_qubits)
            loss(phis)

    def test_invalid_target(self):
        with pytest.raises(ValueError, match="Input target of incorrect length 2 instead of 1"):
            num_qubits = 1

            dev = qml.device("default.qubit", wires=num_qubits)
            observables = [qml.PauliZ(0)]
            loss = SquaredErrorLoss(rx_ansatz, observables, dev)

            phis = np.ones(num_qubits)
            loss(phis, target=np.array([1.0, 2.0]))

    def test_layer_circuit(self):
        num_qubits = 3

        dev = qml.device("default.qubit", wires=num_qubits)
        observables = [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]
        loss = SquaredErrorLoss(layer_ansatz, observables, dev)

        weights = np.ones((num_qubits, 3, 3))
        res = loss(weights, x=np.array([1.0, 2.0, 1.0]), target=np.array([1.0, 0.5, 0.1]))

        assert np.allclose(res, np.array([0.88, 0.83, 0.05]), atol=0.01, rtol=0.01)

    def test_rx_circuit(self):
        num_qubits = 3

        dev = qml.device("default.qubit", wires=num_qubits)
        observables = [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]
        loss = SquaredErrorLoss(rx_ansatz, observables, dev)

        phis = np.ones(num_qubits)
        res = loss(phis, target=np.array([1.0, 0.5, 0.1]))

        assert np.allclose(res, np.array([0.21, 0.25, 0.03]), atol=0.01, rtol=0.01)


@pytest.mark.tf
class TestSquaredErrorLossTf:
    def test_no_target(self):
        with pytest.raises(ValueError, match="The target cannot be None"):
            num_qubits = 1

            dev = qml.device("default.qubit", wires=num_qubits)
            observables = [qml.PauliZ(0)]
            loss = SquaredErrorLoss(rx_ansatz, observables, dev)

            phis = np.ones(num_qubits)
            loss(phis)

    def test_invalid_target(self):
        with pytest.raises(ValueError, match="Input target of incorrect length 2 instead of 1"):
            num_qubits = 1

            dev = qml.device("default.qubit", wires=num_qubits)
            observables = [qml.PauliZ(0)]
            loss = SquaredErrorLoss(rx_ansatz, observables, dev)

            phis = np.ones(num_qubits)
            loss(phis, target=np.array([1.0, 2.0]))

    def test_layer_circuit(self):
        num_qubits = 3

        dev = qml.device("default.qubit", wires=num_qubits)
        observables = [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]
        loss = SquaredErrorLoss(layer_ansatz, observables, dev)

        weights = np.ones((num_qubits, 3, 3))
        res = loss(weights, x=np.array([1.0, 2.0, 1.0]), target=np.array([1.0, 0.5, 0.1]))

        assert np.allclose(res, np.array([0.88, 0.83, 0.05]), atol=0.01, rtol=0.01)

    def test_rx_circuit(self):
        num_qubits = 3

        dev = qml.device("default.qubit", wires=num_qubits)
        observables = [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]
        loss = SquaredErrorLoss(rx_ansatz, observables, dev)

        phis = np.ones(num_qubits)
        res = loss(phis, target=np.array([1.0, 0.5, 0.1]))

        assert np.allclose(res, np.array([0.21, 0.25, 0.03]), atol=0.01, rtol=0.01)


@pytest.mark.jax
class TestSquaredErrorLossJax:
    def test_no_target(self):
        with pytest.raises(ValueError, match="The target cannot be None"):
            num_qubits = 1

            dev = qml.device("default.qubit", wires=num_qubits)
            observables = [qml.PauliZ(0)]
            loss = SquaredErrorLoss(rx_ansatz, observables, dev)

            phis = np.ones(num_qubits)
            loss(phis)

    def test_invalid_target(self):
        with pytest.raises(ValueError, match="Input target of incorrect length 2 instead of 1"):
            num_qubits = 1

            dev = qml.device("default.qubit", wires=num_qubits)
            observables = [qml.PauliZ(0)]
            loss = SquaredErrorLoss(rx_ansatz, observables, dev)

            phis = np.ones(num_qubits)
            loss(phis, target=np.array([1.0, 2.0]))

    def test_layer_circuit(self):
        num_qubits = 3

        dev = qml.device("default.qubit", wires=num_qubits)
        observables = [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]
        loss = SquaredErrorLoss(layer_ansatz, observables, dev)

        weights = np.ones((num_qubits, 3, 3))
        res = loss(weights, x=np.array([1.0, 2.0, 1.0]), target=np.array([1.0, 0.5, 0.1]))

        assert np.allclose(res, np.array([0.88, 0.83, 0.05]), atol=0.01, rtol=0.01)

    def test_rx_circuit(self):
        num_qubits = 3

        dev = qml.device("default.qubit", wires=num_qubits)
        observables = [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]
        loss = SquaredErrorLoss(rx_ansatz, observables, dev)

        phis = np.ones(num_qubits)
        res = loss(phis, target=np.array([1.0, 0.5, 0.1]))

        assert np.allclose(res, np.array([0.21, 0.25, 0.03]), atol=0.01, rtol=0.01)
