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
"""Tests that a device has the right attributes, arguments and methods."""
# pylint: disable=no-self-use
import numpy as np
import pytest
from flaky import flaky
from pennylane.templates.layers import RandomLayers

import pennylane as qml

pytestmark = pytest.mark.skip_unsupported

@flaky(max_runs=10)
class TestQubit:
    def test_easy_circuit(self, device, tol):
        """Test that arbitrary multi-mode Hermitian expectation values are correct"""
        n_wires = 2
        dev = device(n_wires)
        dev_def = qml.device("default.qubit", wires=n_wires)

        if "Hermitian" not in dev.observables:
            pytest.skip("Skipped because device does not support the Hermitian observable.")

        if dev.name == dev_def.name:
            pytest.skip("Device is the default.qubit.")

        theta = 0.432
        phi = 0.123
        A_ = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        def circuit(theta, phi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(A_, wires=[0, 1]))

        qnode_def = qml.QNode(circuit, dev_def)
        qnode = qml.QNode(circuit, dev)

        grad_def = qml.grad(qnode_def, argnum=[0, 1])
        grad = qml.grad(qnode, argnum=[0, 1])

        assert np.allclose(qnode(theta, phi), qnode_def(theta, phi), atol=tol(dev.analytic))
        assert np.allclose(grad(theta, phi), grad_def(theta, phi), atol=tol(dev.analytic))

    def test_pauliz_expectation(self, device, tol):
        """Test that PauliZ expectation value is correct"""
        n_wires = 2
        dev = device(n_wires)
        dev_def = qml.device("default.qubit", wires=n_wires)
        if dev.name == dev_def.name:
            pytest.skip("Device is the default.qubit.")

        if not dev.analytic:
            pytest.skip("Device is in non-analytical mode.")

        theta = 0.432
        phi = 0.123

        def circuit(theta, phi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

        qnode_def = qml.QNode(circuit, dev_def)
        qnode = qml.QNode(circuit, dev)

        grad_def = qml.grad(qnode_def, argnum=[0, 1])
        grad = qml.grad(qnode, argnum=[0, 1])

        assert np.allclose(qnode(theta, phi), qnode_def(theta, phi), atol=tol(dev.analytic))
        assert np.allclose(grad(theta, phi), grad_def(theta, phi), atol=tol(dev.analytic))

    def test_random_circuit(self, device, tol):
        """Test that random expectation value is correct"""
        n_wires = 2
        dev = device(n_wires)
        dev_def = qml.device("default.qubit", wires=n_wires)
        if dev.name == dev_def.name:
            pytest.skip("Device is the default.qubit.")

        if not dev.analytic:
            pytest.skip("Device is in non-analytical mode.")

        n_layers = np.random.randint(1, 5)
        weights = 2*np.pi*np.random.rand(n_layers, 1)

        def circuit(weights):
            RandomLayers(weights, wires=range(n_wires))
            return qml.expval(qml.PauliZ(wires=0) @ qml.PauliX(wires=1))

        qnode_def = qml.QNode(circuit, dev_def)
        qnode = qml.QNode(circuit, dev)

        grad_def = qml.grad(qnode_def, argnum=0)
        grad = qml.grad(qnode, argnum=0)

        assert np.allclose(qnode(weights), qnode_def(weights), atol=tol(dev.analytic))
        assert np.allclose(grad(weights), grad_def(weights), atol=tol(dev.analytic))
