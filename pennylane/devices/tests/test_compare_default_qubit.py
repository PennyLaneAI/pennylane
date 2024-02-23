# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
from pennylane import numpy as pnp  # Import from PennyLane to mirror the standard approach in demos
from pennylane.templates.layers import RandomLayers

pytestmark = pytest.mark.skip_unsupported


@flaky(max_runs=10)
class TestComparison:
    """Test that a device different to default.qubit gives the same result"""

    def test_hermitian_expectation(self, device, tol, benchmark):
        """Test that arbitrary multi-mode Hermitian expectation values are correct"""
        n_wires = 2
        dev = device(n_wires)
        dev_def = qml.device("default.qubit")

        if dev.shots:
            pytest.skip("Device is in non-analytical mode.")

        if isinstance(dev, qml.Device) and "Hermitian" not in dev.observables:
            pytest.skip("Device does not support the Hermitian observable.")

        if dev.name == "default.qubit":
            pytest.skip("Device is default.qubit.")

        theta = 0.432
        phi = 0.123
        A_ = pnp.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ],
            requires_grad=False,
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

        def workload():
            return (
                qnode(theta, phi),
                qnode_def(theta, phi),
                grad(theta, phi),
                grad_def(theta, phi),
            )

        qnode_res, qnode_def_res, grad_res, grad_def_res = benchmark(workload)

        assert pnp.allclose(qnode_res, qnode_def_res, atol=tol(dev.shots))
        assert pnp.allclose(grad_res, grad_def_res, atol=tol(dev.shots))

    @pytest.mark.parametrize(
        "state",
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            pnp.array([1, 1, 0, 0]) / pnp.sqrt(2),
            pnp.array([0, 1, 0, 1]) / pnp.sqrt(2),
            pnp.array([1, 1, 1, 0]) / pnp.sqrt(3),
            pnp.array([1, 1, 1, 1]) / 2,
        ],
    )
    def test_projector_expectation(self, device, state, tol, benchmark):
        """Test that arbitrary multi-mode Projector expectation values are correct"""
        n_wires = 2
        dev = device(n_wires)
        dev_def = qml.device("default.qubit", wires=n_wires)

        if dev.shots:
            pytest.skip("Device is in non-analytical mode.")

        if isinstance(dev, qml.Device) and "Projector" not in dev.observables:
            pytest.skip("Device does not support the Projector observable.")

        if dev.name == "default.qubit":
            pytest.skip("Device is default.qubit.")

        theta = 0.432
        phi = 0.123

        def circuit(theta, phi, state):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Projector(state, wires=[0, 1]))

        qnode_def = qml.QNode(circuit, dev_def)
        qnode = qml.QNode(circuit, dev)

        grad_def = qml.grad(qnode_def, argnum=[0, 1])
        grad = qml.grad(qnode, argnum=[0, 1])

        def workload():
            return (
                qnode(theta, phi, state),
                qnode_def(theta, phi, state),
                grad(theta, phi, state),
                grad_def(theta, phi, state),
            )

        qnode_res, qnode_def_res, grad_res, grad_def_res = benchmark(workload)

        assert pnp.allclose(qnode_res, qnode_def_res, atol=tol(dev.shots))
        assert pnp.allclose(grad_res, grad_def_res, atol=tol(dev.shots))

    def test_pauliz_expectation_analytic(self, device, tol):
        """Test that the tensor product of PauliZ expectation value is correct"""
        n_wires = 2
        dev = device(n_wires)
        dev_def = qml.device("default.qubit", wires=n_wires)

        if dev.name == dev_def.name:
            pytest.skip("Device is default.qubit.")

        supports_tensor = isinstance(dev, qml.devices.Device) or (
            "supports_tensor_observables" in dev.capabilities()
            and dev.capabilities()["supports_tensor_observables"]
        )

        if not supports_tensor:
            pytest.skip("Device does not support tensor observables.")

        if dev.shots:
            pytest.skip("Device is in non-analytical mode.")

        theta = 0.432
        phi = 0.123

        def circuit(theta, phi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(0) @ qml.Z(1))

        qnode_def = qml.QNode(circuit, dev_def)
        qnode = qml.QNode(circuit, dev)

        grad_def = qml.grad(qnode_def, argnum=[0, 1])
        grad = qml.grad(qnode, argnum=[0, 1])

        assert pnp.allclose(qnode(theta, phi), qnode_def(theta, phi), atol=tol(dev.shots))
        assert pnp.allclose(grad(theta, phi), grad_def(theta, phi), atol=tol(dev.shots))

    @pytest.mark.parametrize("ret", ["expval", "var"])
    def test_random_circuit(self, device, tol, ret):
        """Compare the result of a random circuit to default.qubit"""
        n_wires = 2
        dev = device(n_wires)
        dev_def = qml.device("default.qubit", wires=n_wires)

        if dev.name == dev_def.name:
            pytest.skip("Device is default.qubit.")

        supports_tensor = isinstance(dev, qml.devices.Device) or (
            "supports_tensor_observables" in dev.capabilities()
            and dev.capabilities()["supports_tensor_observables"]
        )

        if not supports_tensor:
            pytest.skip("Device does not support tensor observables.")

        if dev.shots:
            pytest.skip("Device is in non-analytical mode.")

        n_layers = pnp.random.randint(1, 5)
        weights = 2 * pnp.pi * pnp.random.rand(n_layers, 1)

        ret_type = getattr(qml, ret)

        def circuit(weights):
            RandomLayers(weights, wires=range(n_wires))
            return ret_type(qml.Z(0) @ qml.X(1))

        qnode_def = qml.QNode(circuit, dev_def)
        qnode = qml.QNode(circuit, dev)

        grad_def = qml.grad(qnode_def, argnum=0)
        grad = qml.grad(qnode, argnum=0)

        assert pnp.allclose(qnode(weights), qnode_def(weights), atol=tol(dev.shots))
        assert pnp.allclose(grad(weights), grad_def(weights), atol=tol(dev.shots))

    def test_four_qubit_random_circuit(self, device, tol):
        """Compare a four-qubit random circuit with lots of different gates to default.qubit"""
        n_wires = 4
        dev = device(n_wires)
        dev_def = qml.device("default.qubit")

        if dev.name == dev_def.name:
            pytest.skip("Device is default.qubit.")

        if dev.shots:
            pytest.skip("Device is in non-analytical mode.")

        gates = [
            qml.X(0),
            qml.Y(1),
            qml.Z(2),
            qml.S(wires=3),
            qml.T(wires=0),
            qml.RX(2.3, wires=1),
            qml.RY(1.3, wires=2),
            qml.RZ(3.3, wires=3),
            qml.Hadamard(wires=0),
            qml.Rot(0.1, 0.2, 0.3, wires=1),
            qml.CRot(0.1, 0.2, 0.3, wires=[2, 3]),
            qml.Toffoli(wires=[0, 1, 2]),
            qml.SWAP(wires=[1, 2]),
            qml.CSWAP(wires=[1, 2, 3]),
            qml.U1(1.0, wires=0),
            qml.U2(1.0, 2.0, wires=2),
            qml.U3(1.0, 2.0, 3.0, wires=3),
            qml.CRX(0.1, wires=[1, 2]),
            qml.CRY(0.2, wires=[2, 3]),
            qml.CRZ(0.3, wires=[3, 1]),
        ]

        layers = 3
        rng = pnp.random.default_rng(1967)
        gates_per_layers = [rng.permutation(gates).numpy() for _ in range(layers)]

        def circuit():
            """4-qubit circuit with layers of randomly selected gates and random connections for
            multi-qubit gates."""
            for gates in gates_per_layers:
                for gate in gates:
                    qml.apply(gate)
            return qml.expval(qml.Z(0))

        qnode_def = qml.QNode(circuit, dev_def)
        qnode = qml.QNode(circuit, dev)

        assert pnp.allclose(qnode(), qnode_def(), atol=tol(dev.shots))
