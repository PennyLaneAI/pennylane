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
"""Unit tests for the autograd interface"""
import pytest
from pennylane import numpy as np

import pennylane as qml
from pennylane.beta.tapes import QuantumTape
from pennylane.beta.queuing import expval, var, sample, probs
from pennylane.beta.interfaces.autograd import AutogradInterface


class TestAutogradQuantumTape:
    """Test the autograd interface applied to a tape"""

    def test_interface_str(self):
        """Test that the interface string is correctly identified as autograd"""
        with AutogradInterface.apply(QuantumTape()) as tape:
            qml.RX(0.5, wires=0)
            expval(qml.PauliX(0))

        assert tape.interface == "autograd"
        assert isinstance(tape, AutogradInterface)

    def test_get_parameters(self):
        """Test that the get parameters function correctly sets and returns the
        trainable parameters"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=False)
        c = np.array(0.3, requires_grad=True)
        d = np.array(0.4, requires_grad=False)

        with AutogradInterface.apply(QuantumTape()) as tape:
            qml.Rot(a, b, c, wires=0)
            qml.RX(d, wires=1)
            qml.CNOT(wires=[0, 1])
            expval(qml.PauliX(0))

        assert tape.trainable_params == {0, 2}
        assert np.all(tape.get_parameters() == [a, c])

    def test_execution(self):
        """Test execution"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=False)

        def cost(a, b, device):
            with AutogradInterface.apply(QuantumTape()) as tape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                expval(qml.PauliZ(0))
            assert tape.trainable_params == {0}
            return tape.execute(device)

        dev = qml.device("default.qubit", wires=1)
        res = cost(a, b, device=dev)
        assert res.shape == (1,)

    def test_scalar_jacobian(self, tol):
        """Test scalar jacobian calculation"""
        a = np.array(0.1, requires_grad=True)

        def cost(a, device):
            with AutogradInterface.apply(QuantumTape()) as tape:
                qml.RY(a, wires=0)
                expval(qml.PauliZ(0))
            assert tape.trainable_params == {0}
            return tape.execute(device)

        dev = qml.device("default.qubit", wires=2)
        res = qml.jacobian(cost)(a, device=dev)
        assert res.shape == (1,)

        # compare to standard tape jacobian
        with QuantumTape() as tape:
            qml.RY(a, wires=0)
            expval(qml.PauliZ(0))

        tape.trainable_params = {0}
        expected = tape.jacobian(dev)
        assert expected.shape == (1, 1)
        assert np.allclose(res, np.squeeze(expected), atol=tol, rtol=0)

    def test_jacobian(self, tol):
        """Test jacobian calculation"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        def cost(a, b, device):
            with AutogradInterface.apply(QuantumTape()) as tape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                expval(qml.PauliZ(0))
                expval(qml.PauliY(0))
            assert tape.trainable_params == {0, 1}
            return tape.execute(device)

        dev = qml.device("default.qubit", wires=2)
        res = qml.jacobian(cost)(a, b, device=dev)
        assert res.shape == (2, 2)

        # compare to standard tape jacobian
        with QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            expval(qml.PauliZ(0))
            expval(qml.PauliY(0))

        tape.trainable_params = {0, 1}
        expected = tape.jacobian(dev)
        assert expected.shape == (2, 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_classical_processing(self, tol):
        """Test classical processing within the quantum tape"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=False)
        c = np.array(0.3, requires_grad=True)

        def cost(a, b, c, device):
            with AutogradInterface.apply(QuantumTape()) as tape:
                qml.RY(a * c, wires=0)
                qml.RZ(b, wires=0)
                qml.RX(c + c ** 2 + np.sin(a), wires=0)
                expval(qml.PauliZ(0))
            assert tape.trainable_params == {0, 2}
            return tape.execute(device)

        dev = qml.device("default.qubit", wires=2)
        res = qml.jacobian(cost)(a, b, c, device=dev)
        assert res.shape == (1, 2)

    def test_no_trainable_parameters(self, tol):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        a = np.array(0.1, requires_grad=False)
        b = np.array(0.2, requires_grad=False)

        def cost(a, b, device):
            with AutogradInterface.apply(QuantumTape()) as tape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliZ(0))
                expval(qml.PauliZ(1))
            assert tape.trainable_params == set()
            return tape.execute(device)

        dev = qml.device("default.qubit", wires=2)
        res = cost(a, b, device=dev)
        assert res.shape == (2,)

    def test_differentiable_expand(self, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""

        class U3(qml.U3):
            def expand(self):
                tape = QuantumTape()
                theta, phi, lam = self.data
                wires = self.wires
                tape._ops += [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]
                return tape

        def cost_fn(a, p, device):
            tape = QuantumTape()

            with tape:
                qml.RX(a, wires=0)
                U3(*p, wires=0)
                expval(qml.PauliX(0))

            tape = AutogradInterface.apply(tape.expand())

            assert tape.trainable_params == {1, 2, 3, 4}
            assert [i.name for i in tape.operations] == ["RX", "Rot", "PhaseShift"]
            assert np.all(tape.get_parameters() == [p[2], p[0], -p[2], p[1] + p[2]])

            return tape.execute(device=device)

        a = np.array(0.1, requires_grad=False)
        p = np.array([0.1, 0.2, 0.3], requires_grad=True)

        dev = qml.device("default.qubit", wires=1)
        res = cost_fn(a, p, device=dev)
        expected = np.cos(a) * np.cos(p[1]) * np.sin(p[0]) + np.sin(a) * (
            np.cos(p[2]) * np.sin(p[1]) + np.cos(p[0]) * np.cos(p[1]) * np.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac_fn = qml.jacobian(cost_fn)
        res = jac_fn(a, p, device=dev)
        expected = np.array(
            [
                np.cos(p[1]) * (np.cos(a) * np.cos(p[0]) - np.sin(a) * np.sin(p[0]) * np.sin(p[2])),
                np.cos(p[1]) * np.cos(p[2]) * np.sin(a)
                - np.sin(p[1])
                * (np.cos(a) * np.sin(p[0]) + np.cos(p[0]) * np.sin(a) * np.sin(p[2])),
                np.sin(a)
                * (np.cos(p[0]) * np.cos(p[1]) * np.cos(p[2]) - np.sin(p[1]) * np.sin(p[2])),
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)
