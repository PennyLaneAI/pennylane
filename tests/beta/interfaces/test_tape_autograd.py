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
from pennylane.beta.tapes import QuantumTape, qnode
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


@pytest.mark.parametrize(
    "dev_name,diff_method",
    [["default.qubit", "finite-diff"], ["default.qubit.autograd", "backprop"]],
)
class TestQNode:
    """Same tests as above, but this time via the QNode interface!"""

    def test_execution_no_interface(self, dev_name, diff_method):
        """Test execution works without an interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface=None)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return expval(qml.PauliZ(0))

        a = np.array(0.1, requires_grad=True)

        res = circuit(a)

        assert circuit.qtape.interface == None

        # without the interface, the tape simply returns an array of results
        assert isinstance(res, np.ndarray)
        assert res.shape == (1,)
        assert isinstance(res[0], float)

        # without the interface, the tape is unable to deduce
        # trainable parameters
        assert circuit.qtape.trainable_params == {0, 1}

        # gradients should cause an error
        with pytest.raises(TypeError, match="must be real number, not ArrayBox"):
            qml.grad(circuit)(a)

    def test_execution_with_interface(self, dev_name, diff_method):
        """Test execution works with the interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface="autograd")
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(np.array(0.2, requires_grad=False), wires=0)
            return expval(qml.PauliZ(0))

        a = np.array(0.1, requires_grad=True)
        circuit(a)

        assert circuit.qtape.interface == "autograd"

        # the tape is able to deduce trainable parameters
        assert circuit.qtape.trainable_params == {0}

        # gradients should work
        grad = qml.grad(circuit)(a)
        assert len(grad) == 1
        assert isinstance(grad[0], np.ndarray)
        assert grad[0].shape == tuple()

    def test_interface_swap(self, dev_name, diff_method, tol):
        """Test that the autograd interface can be applied to a QNode
        with a pre-existing interface"""
        tf = pytest.importorskip("tensorflow", minversion="2.1")

        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface="tf")
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(np.array(0.2, requires_grad=False), wires=0)
            return expval(qml.PauliZ(0))

        a = tf.Variable(0.1, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res_tf = circuit(a)

        grad_tf = tape.gradient(res_tf, a)

        # switch to autograd interface
        circuit.to_autograd()

        a = np.array(0.1, requires_grad=True)

        res = circuit(a)
        grad = qml.grad(circuit)(a)
        assert len(grad) == 1

        assert np.allclose(res, res_tf, atol=tol, rtol=0)
        assert np.allclose(grad[0], grad_tf, atol=tol, rtol=0)

    def test_jacobian(self, dev_name, diff_method, mocker, tol):
        """Test jacobian calculation"""
        spy = mocker.spy(QuantumTape, "jacobian")
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [expval(qml.PauliZ(0)), expval(qml.PauliY(1))]

        res = circuit(a, b)

        assert circuit.qtape.trainable_params == {0, 1}
        assert res.shape == (2,)

        expected = [np.cos(a), -np.cos(a) * np.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.jacobian(circuit)(a, b)
        expected = [[-np.sin(a), 0], [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        if diff_method == "finite-diff":
            spy.assert_called()
        elif diff_method == "backprop":
            spy.assert_not_called()

    def test_jacobian_no_evaluate(self, dev_name, diff_method, mocker, tol):
        """Test jacobian calculation when no prior circuit evaluation has been performed"""
        spy = mocker.spy(QuantumTape, "jacobian")
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [expval(qml.PauliZ(0)), expval(qml.PauliY(1))]

        jac_fn = qml.jacobian(circuit)
        res = jac_fn(a, b)
        expected = [[-np.sin(a), 0], [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        if diff_method == "finite-diff":
            spy.assert_called()
        elif diff_method == "backprop":
            spy.assert_not_called()

        # call the Jacobian with new parameters
        a = np.array(0.6, requires_grad=True)
        b = np.array(0.832, requires_grad=True)

        res = jac_fn(a, b)
        expected = [[-np.sin(a), 0], [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_options(self, dev_name, diff_method, mocker, tol):
        """Test setting jacobian options"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        spy = mocker.spy(QuantumTape, "numeric_pd")

        a = np.array([0.1, 0.2], requires_grad=True)

        dev = qml.device("default.qubit", wires=1)

        @qnode(dev, interface="autograd", h=1e-8, order=2)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return expval(qml.PauliZ(0))

        qml.jacobian(circuit)(a)

        for args in spy.call_args_list:
            assert args[1]["order"] == 2
            assert args[1]["h"] == 1e-8

    def test_changing_trainability(self, dev_name, diff_method, mocker, tol):
        """Test changing the trainability of parameters changes the
        number of differentiation requests made"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="autograd")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return expval(qml.PauliZ(0)), expval(qml.PauliY(1))

        def loss(a, b):
            return np.sum(circuit(a, b))

        grad_fn = qml.grad(loss)
        spy = mocker.spy(QuantumTape, "numeric_pd")

        res = grad_fn(a, b)

        # the tape has reported both arguments as trainable
        assert circuit.qtape.trainable_params == {0, 1}

        expected = [-np.sin(a) + np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # QuantumTape.numeric_pd has been called for each argument
        assert len(spy.call_args_list) == 2

        # make the second QNode argument a constant
        a = np.array(0.54, requires_grad=True)
        b = np.array(0.8, requires_grad=False)

        spy.call_args_list = []
        res = grad_fn(a, b)

        # the tape has reported only the first argument as trainable
        assert circuit.qtape.trainable_params == {0}

        expected = [-np.sin(a) + np.sin(a) * np.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # QuantumTape.numeric_pd has been called only once
        assert len(spy.call_args_list) == 1

        # trainability also updates on evaluation
        a = np.array(0.54, requires_grad=False)
        b = np.array(0.8, requires_grad=True)
        circuit(a, b)
        assert circuit.qtape.trainable_params == {1}

    def test_classical_processing(self, dev_name, diff_method, tol):
        """Test classical processing within the quantum tape"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=False)
        c = np.array(0.3, requires_grad=True)

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(a, b, c):
            qml.RY(a * c, wires=0)
            qml.RZ(b, wires=0)
            qml.RX(c + c ** 2 + np.sin(a), wires=0)
            return expval(qml.PauliZ(0))

        res = qml.jacobian(circuit)(a, b, c)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == {0, 2}
            tape_params = np.array(circuit.qtape.get_parameters())
            assert np.all(tape_params == [a * c, c + c ** 2 + np.sin(a)])

        assert res.shape == (1, 2)

    def test_no_trainable_parameters(self, dev_name, diff_method, tol):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.CNOT(wires=[0, 1])
            return expval(qml.PauliZ(0)), expval(qml.PauliZ(1))

        a = np.array(0.1, requires_grad=False)
        b = np.array(0.2, requires_grad=False)

        res = circuit(a, b)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == set()

        assert res.shape == (2,)
        assert isinstance(res, np.ndarray)

        with pytest.raises(ValueError, match="need at least one array to stack"):
            # TODO: maybe qml.jacobian should return an empty array, or None in this case?
            qml.jacobian(circuit)(a, b)

        def cost(a, b):
            return np.sum(circuit(a, b))

        with pytest.warns(UserWarning, match="Output seems independent of input"):
            grad = qml.grad(cost)(a, b)

        assert grad == tuple()

    def test_matrix_parameter(self, dev_name, diff_method, tol):
        """Test that the TF interface works correctly
        with a matrix parameter"""
        U = np.array([[0, 1], [1, 0]], requires_grad=False)
        a = np.array(0.1, requires_grad=True)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(U, a):
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            return expval(qml.PauliZ(0))

        res = circuit(U, a)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == {1}

        res = qml.grad(circuit)(U, a)
        assert np.allclose(res, np.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(self, dev_name, diff_method, mocker, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""
        mock = mocker.patch.object(qml.operation.Operation, "do_check_domain", False)

        class U3(qml.U3):
            def expand(self):
                theta, phi, lam = self.data
                wires = self.wires

                with QuantumTape() as tape:
                    qml.Rot(lam, theta, -lam, wires=wires)
                    qml.PhaseShift(phi + lam, wires=wires)

                return tape

        dev = qml.device(dev_name, wires=1)
        a = np.array(0.1, requires_grad=False)
        p = np.array([0.1, 0.2, 0.3], requires_grad=True)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(a, p):
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            return expval(qml.PauliX(0))

        res = circuit(a, p)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == {1, 2, 3, 4}
        elif diff_method == "backprop":
            assert circuit.qtape.trainable_params == {0, 1, 2, 3, 4}

        assert [i.name for i in circuit.qtape.operations] == ["RX", "Rot", "PhaseShift"]

        if diff_method == "finite-diff":
            assert np.all(circuit.qtape.get_parameters() == [p[2], p[0], -p[2], p[1] + p[2]])
        elif diff_method == "backprop":
            assert np.all(circuit.qtape.get_parameters() == [a, p[2], p[0], -p[2], p[1] + p[2]])

        expected = np.cos(a) * np.cos(p[1]) * np.sin(p[0]) + np.sin(a) * (
            np.cos(p[2]) * np.sin(p[1]) + np.cos(p[0]) * np.cos(p[1]) * np.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.grad(circuit)(a, p)
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

    def test_probability_differentiation(self, dev_name, diff_method, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        dev = qml.device(dev_name, wires=2)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return probs(wires=[0]), probs(wires=[1])

        res = circuit(x, y)

        expected = np.array(
            [
                [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2],
                [(1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.jacobian(circuit)(x, y)
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

    def test_ragged_differentiation(self, dev_name, diff_method, monkeypatch, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device(dev_name, wires=2)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        def _asarray(args, dtype=np.float64):
            if len({i.shape[0] if i.shape else 0 for i in args}):
                res = [np.reshape(i, [-1]) for i in args]
                return np.hstack(res)
            return np.tensor(args, dtype=dtype)

        if dev_name == "default.qubit.autograd":
            # we need to patch the asarray method on the device
            monkeypatch.setattr(dev, "_asarray", _asarray)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return [expval(qml.PauliZ(0)), probs(wires=[1])]

        res = circuit(x, y)

        expected = np.array(
            [np.cos(x), (1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.jacobian(circuit)(x, y)
        expected = np.array(
            [
                [-np.sin(x), 0],
                [-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2],
                [np.cos(y) * np.sin(x) / 2, np.cos(x) * np.sin(y) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_sampling(self, dev_name, diff_method):
        """Test sampling works as expected"""
        dev = qml.device(dev_name, wires=2, shots=10)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return [sample(qml.PauliZ(0)), sample(qml.PauliX(1))]

        res = circuit()

        assert res.shape == (2, 10)
        assert isinstance(res, np.ndarray)
