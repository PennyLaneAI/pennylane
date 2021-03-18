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
"""Tests that a device gives the same output as the default device."""
# pylint: disable=no-self-use,no-member
import pytest
from flaky import flaky

import pennylane as qml
from pennylane import numpy as np  # Import from PennyLane to mirror the standard approach in demos
from pennylane.templates.layers import RandomLayers

pytestmark = [pytest.mark.skip_unsupported, pytest.mark.usefixtures("tape_mode")]


@flaky(max_runs=10)
class TestComparison:
    """Test that a device different to default.qubit gives the same result"""

    def test_hermitian_expectation(self, device, tol):
        """Test that arbitrary multi-mode Hermitian expectation values are correct"""
        n_wires = 2
        dev = device(n_wires)
        dev_def = qml.device("default.qubit", wires=n_wires)

        if not dev.analytic:
            pytest.skip("Device is in non-analytical mode.")

        if "Hermitian" not in dev.observables:
            pytest.skip("Skipped because device does not support the Hermitian observable.")

        if dev.name == dev_def.name:
            pytest.skip("Device is default.qubit.")

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

    def test_pauliz_expectation_analytic(self, device, tol):
        """Test that the tensor product of PauliZ expectation value is correct"""
        n_wires = 2
        dev = device(n_wires)
        dev_def = qml.device("default.qubit", wires=n_wires)

        if dev.name == dev_def.name:
            pytest.skip("Device is default.qubit.")

        supports_tensor = (
            "supports_tensor_observables" in dev.capabilities()
            and dev.capabilities()["supports_tensor_observables"]
        )

        if not supports_tensor:
            pytest.skip("Device does not support tensor observables.")

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

    @pytest.mark.parametrize("ret", ["expval", "var"])
    def test_random_circuit(self, device, tol, ret):
        """Test that the expectation value of a random circuit is correct"""
        n_wires = 2
        dev = device(n_wires)
        dev_def = qml.device("default.qubit", wires=n_wires)

        if dev.name == dev_def.name:
            pytest.skip("Device is default.qubit.")

        supports_tensor = (
            "supports_tensor_observables" in dev.capabilities()
            and dev.capabilities()["supports_tensor_observables"]
        )

        if not supports_tensor:
            pytest.skip("Device does not support tensor observables.")

        if not dev.analytic:
            pytest.skip("Device is in non-analytical mode.")

        n_layers = np.random.randint(1, 5)
        weights = 2 * np.pi * np.random.rand(n_layers, 1)

        ret_type = getattr(qml, ret)

        def circuit(weights):
            RandomLayers(weights, wires=range(n_wires))
            return ret_type(qml.PauliZ(wires=0) @ qml.PauliX(wires=1))

        qnode_def = qml.QNode(circuit, dev_def)
        qnode = qml.QNode(circuit, dev)

        grad_def = qml.grad(qnode_def, argnum=0)
        grad = qml.grad(qnode, argnum=0)

        assert np.allclose(qnode(weights), qnode_def(weights), atol=tol(dev.analytic))
        assert np.allclose(grad(weights), grad_def(weights), atol=tol(dev.analytic))

    def test_four_qubit_random_circuit(self, device, tol):
        """Test a four-qubit random circuit with the whole set of possible gates"""
        n_wires = 4
        dev = device(n_wires)
        dev_def = qml.device("default.qubit", wires=n_wires)

        if dev.name == dev_def.name:
            pytest.skip("Device is default.qubit.")

        if not dev.analytic:
            pytest.skip("Device is in non-analytical mode.")

        gates = [
            qml.PauliX,
            qml.PauliY,
            qml.PauliZ,
            qml.S,
            qml.T,
            qml.RX,
            qml.RY,
            qml.RZ,
            qml.Hadamard,
            qml.Rot,
            qml.CRot,
            qml.Toffoli,
            qml.SWAP,
            qml.CSWAP,
            qml.U1,
            qml.U2,
            qml.U3,
            qml.CRX,
            qml.CRY,
            qml.CRZ,
        ]

        layers = 3
        np.random.seed(1967)
        gates_per_layers = [np.random.permutation(gates).numpy() for _ in range(layers)]

        def circuit():
            """4-qubit circuit with layers of randomly selected gates and random connections for
            multi-qubit gates."""
            np.random.seed(1967)
            for gates in gates_per_layers:
                for gate in gates:
                    params = list(np.pi * np.random.rand(gate.num_params))
                    rnd_wires = np.random.choice(range(n_wires), size=gate.num_wires, replace=False)
                    gate(
                        *params,
                        wires=[
                            int(w) for w in rnd_wires
                        ]  # make sure we do not address wires as 0-d arrays
                    )
            return qml.expval(qml.PauliZ(0))

        qnode_def = qml.QNode(circuit, dev_def)
        qnode = qml.QNode(circuit, dev)

        assert np.allclose(qnode(), qnode_def(), atol=tol(dev.analytic))
