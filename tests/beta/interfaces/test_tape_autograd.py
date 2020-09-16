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
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliZ(0))
                expval(qml.PauliY(1))
            assert tape.trainable_params == {0, 1}
            return tape.execute(device)

        dev = qml.device("default.qubit", wires=2)

        res = cost(a, b, device=dev)
        expected = [np.cos(a), -np.cos(a) * np.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.jacobian(cost)(a, b, device=dev)
        assert res.shape == (2, 2)

        expected = [[-np.sin(a), 0], [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_options(self, mocker, tol):
        """Test setting jacobian options"""
        spy = mocker.spy(QuantumTape, "numeric_pd")

        a = np.array([0.1, 0.2], requires_grad=True)

        dev = qml.device("default.qubit", wires=1)

        def cost(a, device):
            with AutogradInterface.apply(QuantumTape()) as qtape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                expval(qml.PauliZ(0))

            qtape.jacobian_options = {"h": 1e-8, "order": 2}
            return qtape.execute(dev)

        res = qml.jacobian(cost)(a, device=dev)

        for args in spy.call_args_list:
            assert args[1]["order"] == 2
            assert args[1]["h"] == 1e-8

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

    def test_matrix_parameter(self, tol):
        """Test that the autograd interface works correctly
        with a matrix parameter"""
        U = np.array([[0, 1], [1, 0]], requires_grad=False)
        a = np.array(0.1, requires_grad=True)

        def cost(a, U, device):
            with AutogradInterface.apply(QuantumTape()) as tape:
                qml.QubitUnitary(U, wires=0)
                qml.RY(a, wires=0)
                expval(qml.PauliZ(0))
            assert tape.trainable_params == {1}
            return tape.execute(device)

        dev = qml.device("default.qubit", wires=2)
        res = cost(a, U, device=dev)
        assert np.allclose(res, -np.cos(a), atol=tol, rtol=0)

        jac_fn = qml.jacobian(cost)
        res = jac_fn(a, U, device=dev)
        assert np.allclose(res, np.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(self, mocker, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""
        mock = mocker.patch.object(qml.operation.Operation, "do_check_domain", False)

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
            assert np.all(np.array(tape.get_parameters()) == [p[2], p[0], -p[2], p[1] + p[2]])

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

    def test_probability_differentiation(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        def cost(x, y, device):
            with AutogradInterface.apply(QuantumTape()) as tape:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                probs(wires=[0])
                probs(wires=[1])

            return tape.execute(device)

        dev = qml.device("default.qubit", wires=2)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        res = cost(x, y, device=dev)
        expected = np.array(
            [
                [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2],
                [(1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac_fn = qml.jacobian(cost)
        res = jac_fn(x, y, device=dev)
        assert res.shape == (2, 2, 2)
        expected = np.array(
            [
                [[-np.sin(x) / 2, 0], [-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2]],
                [
                    [np.sin(x) / 2, 0],
                    [np.cos(y) * np.sin(x) / 2, np.cos(x) * np.sin(y) / 2],
                ],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_ragged_differentiation(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        def cost(x, y, device):
            with AutogradInterface.apply(QuantumTape()) as tape:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliZ(0))
                probs(wires=[1])

            return tape.execute(device)

        dev = qml.device("default.qubit", wires=2)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        res = cost(x, y, device=dev)
        expected = np.array(
            [np.cos(x), (1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac_fn = qml.jacobian(cost)
        res = jac_fn(x, y, device=dev)
        expected = np.array(
            [
                [-np.sin(x), 0],
                [-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2],
                [np.cos(y) * np.sin(x) / 2, np.cos(x) * np.sin(y) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_sampling(self):
        """Test sampling works as expected"""

        def cost(x, device):
            with AutogradInterface.apply(QuantumTape()) as tape:
                qml.Hadamard(wires=[0])
                qml.CNOT(wires=[0, 1])
                sample(qml.PauliZ(0))
                sample(qml.PauliX(1))

            return tape.execute(device)

        dev = qml.device("default.qubit", wires=2, shots=10)
        x = np.array(0.543, requires_grad=True)
        res = cost(x, device=dev)
        assert res.shape == (2, 10)


class TestAutogradPassthru:
    """Test that the quantum tape works with an autograd passthru
    device.

    These tests are very similar to the tests above, with three key differences:

    * We do **not** apply the autograd interface. These tapes simply use passthru
      backprop, no custom gradient registration needed.

    * We do not test the trainable_params attribute. Since these tapes have no
      autograd interface, the tape does not need to bookkeep which parameters
      are trainable; this is done by autograd internally.

    * We use mock.spy to ensure that the tape's Jacobian method is not being called.
    """

    def test_execution(self):
        """Test execution"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=False)

        def cost(a, b, device):
            with QuantumTape() as tape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                expval(qml.PauliZ(0))
            return tape.execute(device)

        dev = qml.device("default.qubit.autograd", wires=1)
        res = cost(a, b, device=dev)
        assert res.shape == (1,)

    def test_scalar_jacobian(self, tol, mocker):
        """Test scalar jacobian calculation"""
        spy = mocker.spy(QuantumTape, "jacobian")
        a = np.array(0.1, requires_grad=True)

        def cost(a, device):
            with QuantumTape() as tape:
                qml.RY(a, wires=0)
                expval(qml.PauliZ(0))
            return tape.execute(device)

        dev = qml.device("default.qubit.autograd", wires=2)
        res = qml.jacobian(cost)(a, device=dev)
        spy.assert_not_called()
        assert res.shape == (1,)

        # compare to standard tape jacobian
        with QuantumTape() as tape:
            qml.RY(a, wires=0)
            expval(qml.PauliZ(0))

        expected = tape.jacobian(dev)
        assert expected.shape == (1, 1)
        assert np.allclose(res, np.squeeze(expected), atol=tol, rtol=0)

    def test_jacobian(self, mocker, tol):
        """Test jacobian calculation"""
        spy = mocker.spy(QuantumTape, "jacobian")
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        def cost(a, b, device):
            with QuantumTape() as tape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                expval(qml.PauliZ(0))
                expval(qml.PauliY(0))
            return tape.execute(device)

        dev = qml.device("default.qubit.autograd", wires=2)
        res = qml.jacobian(cost)(a, b, device=dev)
        spy.assert_not_called()
        assert res.shape == (2, 2)

        # compare to standard tape jacobian
        with QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            expval(qml.PauliZ(0))
            expval(qml.PauliY(0))

        expected = tape.jacobian(dev)
        assert expected.shape == (2, 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_classical_processing(self, mocker, tol):
        """Test classical processing within the quantum tape"""
        spy = mocker.spy(QuantumTape, "jacobian")
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=False)
        c = np.array(0.3, requires_grad=True)

        def cost(a, b, c, device):
            with QuantumTape() as tape:
                qml.RY(a * c, wires=0)
                qml.RZ(b, wires=0)
                qml.RX(c + c ** 2 + np.sin(a), wires=0)
                expval(qml.PauliZ(0))
            return tape.execute(device)

        dev = qml.device("default.qubit.autograd", wires=2)
        res = qml.jacobian(cost)(a, b, c, device=dev)
        assert res.shape == (1, 2)
        spy.assert_not_called()

    def test_no_trainable_parameters(self, mocker, tol):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        spy = mocker.spy(QuantumTape, "jacobian")
        a = np.array(0.1, requires_grad=False)
        b = np.array(0.2, requires_grad=False)

        def cost(a, b, device):
            with QuantumTape() as tape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliZ(0))
                expval(qml.PauliZ(1))
            return tape.execute(device)

        dev = qml.device("default.qubit.autograd", wires=2)
        res = cost(a, b, device=dev)
        assert res.shape == (2,)
        spy.assert_not_called()

    def test_matrix_parameter(self, mocker, tol):
        """Test jacobian computation when the tape includes a matrix parameter"""
        spy = mocker.spy(QuantumTape, "jacobian")
        U = np.array([[0, 1], [1, 0]], requires_grad=False)
        a = np.array(0.1, requires_grad=True)

        def cost(a, U, device):
            with QuantumTape() as tape:
                qml.QubitUnitary(U, wires=0)
                qml.RY(a, wires=0)
                expval(qml.PauliZ(0))
            return tape.execute(device)

        dev = qml.device("default.qubit.autograd", wires=2)
        res = cost(a, U, device=dev)
        assert np.allclose(res, -np.cos(a), atol=tol, rtol=0)

        jac_fn = qml.jacobian(cost)
        res = jac_fn(a, U, device=dev)
        assert np.allclose(res, np.sin(a), atol=tol, rtol=0)
        spy.assert_not_called()

    def test_differentiable_expand(self, mocker, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""
        spy = mocker.spy(QuantumTape, "jacobian")
        mock = mocker.patch.object(qml.operation.Operation, "do_check_domain", False)

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

            tape = tape.expand()

            assert [i.name for i in tape.operations] == ["RX", "Rot", "PhaseShift"]
            assert np.all(tape.get_parameters() == [a, p[2], p[0], -p[2], p[1] + p[2]])

            return tape.execute(device=device)

        a = np.array(0.1, requires_grad=False)
        p = np.array([0.1, 0.2, 0.3], requires_grad=True)

        dev = qml.device("default.qubit.autograd", wires=1)

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
        spy.assert_not_called()

    def test_probability_differentiation(self, mocker, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        spy = mocker.spy(QuantumTape, "jacobian")

        def cost(x, y, device):
            with QuantumTape() as tape:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                probs(wires=[0])
                probs(wires=[1])

            return tape.execute(device)

        dev = qml.device("default.qubit.autograd", wires=2)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        res = cost(x, y, device=dev)
        expected = np.array(
            [
                [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2],
                [(1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2],
            ]
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac_fn = qml.jacobian(cost)
        res = jac_fn(x, y, device=dev)
        assert res.shape == (2, 2, 2)

        expected = np.array(
            [
                [[-np.sin(x) / 2, 0], [-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2]],
                [
                    [np.sin(x) / 2, 0],
                    [np.cos(y) * np.sin(x) / 2, np.cos(x) * np.sin(y) / 2],
                ],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)
        spy.assert_not_called()

    @pytest.mark.xfail
    def test_ragged_differentiation(self, mocker, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        spy = mocker.spy(QuantumTape, "jacobian")

        def cost(x, y, device):
            with QuantumTape() as tape:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliZ(0))
                probs(wires=[1])

            return np.hstack(tape.execute(device))

        dev = qml.device("default.qubit.autograd", wires=2)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        res = cost(x, y, device=dev)
        expected = np.array(
            [np.cos(x), (1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac_fn = qml.jacobian(cost)
        res = jac_fn(x, y, device=dev)
        expected = np.array(
            [
                [-np.sin(x), 0],
                [-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2],
                [np.cos(y) * np.sin(x) / 2, np.cos(x) * np.sin(y) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)
        spy.assert_not_called()

    def test_sampling(self):
        """Test sampling works as expected"""

        def cost(x, device):
            with QuantumTape() as tape:
                qml.Hadamard(wires=[0])
                qml.CNOT(wires=[0, 1])
                sample(qml.PauliZ(0))
                sample(qml.PauliX(1))

            return tape.execute(device)

        dev = qml.device("default.qubit.autograd", wires=2, shots=10)
        x = np.array(0.543, requires_grad=True)
        res = cost(x, device=dev)
        assert res.shape == (2, 10)
