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
"""Unit tests for the torch interface"""
import pytest

torch = pytest.importorskip("torch", minversion="1.3")

import numpy as np

import pennylane as qml
from pennylane.beta.tapes import QuantumTape, qnode
from pennylane.beta.queuing import expval, var, sample, probs
from pennylane.beta.interfaces.torch import TorchInterface


class TestTorchQuantumTape:
    """Test the autograd interface applied to a tape"""

    def test_interface_construction(self):
        """Test that the interface is correctly applied"""
        with TorchInterface.apply(QuantumTape()) as tape:
            qml.RX(0.5, wires=0)
            expval(qml.PauliX(0))

        assert tape.interface == "torch"
        assert isinstance(tape, TorchInterface)
        assert tape.__bare__ == QuantumTape

    def test_repeated_interface_construction(self):
        """Test that the interface is correctly applied multiple times"""
        with TorchInterface.apply(QuantumTape()) as tape:
            qml.RX(0.5, wires=0)
            expval(qml.PauliX(0))

        assert tape.interface == "torch"
        assert isinstance(tape, TorchInterface)
        assert tape.__bare__ == QuantumTape

        TorchInterface.apply(QuantumTape())
        assert tape.interface == "torch"
        assert isinstance(tape, TorchInterface)
        assert tape.__bare__ == QuantumTape

    def test_get_parameters(self):
        """Test that the get parameters function correctly sets and returns the
        trainable parameters"""
        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2)
        c = torch.tensor(0.3, requires_grad=True)
        d = 0.4

        with TorchInterface.apply(QuantumTape()) as tape:
            qml.Rot(a, b, c, wires=0)
            qml.RX(d, wires=1)
            qml.CNOT(wires=[0, 1])
            expval(qml.PauliX(0))

        assert tape.trainable_params == {0, 2}
        assert np.all(tape.get_parameters() == [a, c])

    def test_execution(self):
        """Test execution"""
        a = torch.tensor(0.1, requires_grad=True)
        dev = qml.device("default.qubit", wires=1)

        with TorchInterface.apply(QuantumTape()) as tape:
            qml.RY(a, wires=0)
            qml.RX(torch.tensor(0.2), wires=0)
            expval(qml.PauliZ(0))

        assert tape.trainable_params == {0}
        res = tape.execute(dev)

        assert isinstance(res, torch.Tensor)
        assert res.shape == (1,)

    def test_jacobian(self, mocker, tol):
        """Test jacobian calculation"""
        spy = mocker.spy(QuantumTape, "jacobian")

        a_val = 0.1
        b_val = 0.2

        a = torch.tensor(a_val, requires_grad=True)
        b = torch.tensor(b_val, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        with TorchInterface.apply(QuantumTape()) as tape:
            qml.RZ(torch.tensor(0.543), wires=0)
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            expval(qml.PauliZ(0))
            expval(qml.PauliY(1))

        assert tape.trainable_params == {1, 2}
        res = tape.execute(dev)

        assert isinstance(res, torch.Tensor)
        assert res.shape == (2,)

        expected = [np.cos(a_val), -np.cos(a_val) * np.sin(b_val)]
        assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

        loss = torch.sum(res)

        loss.backward()
        expected = [-np.sin(a_val) + np.sin(a_val) * np.sin(b_val), -np.cos(a_val) * np.cos(b_val)]
        assert np.allclose(a.grad, expected[0], atol=tol, rtol=0)
        assert np.allclose(b.grad, expected[1], atol=tol, rtol=0)

        spy.assert_called()

    def test_jacobian_options(self, mocker, tol):
        """Test setting jacobian options"""
        spy = mocker.spy(QuantumTape, "numeric_pd")

        a = torch.tensor([0.1, 0.2], requires_grad=True)

        dev = qml.device("default.qubit", wires=1)

        with TorchInterface.apply(QuantumTape()) as tape:
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            expval(qml.PauliZ(0))

        tape.jacobian_options = {"h": 1e-8, "order": 2}
        res = tape.execute(dev)
        res.backward()

        for args in spy.call_args_list:
            assert args[1]["order"] == 2
            assert args[1]["h"] == 1e-8

    def test_jacobian_dtype(self, tol):
        """Test calculating the jacobian with a different datatype"""
        a_val = 0.1
        b_val = 0.2

        a = torch.tensor(a_val, requires_grad=True, dtype=torch.float32)
        b = torch.tensor(b_val, requires_grad=True, dtype=torch.float32)

        dev = qml.device("default.qubit", wires=2)

        with TorchInterface.apply(QuantumTape(), dtype=torch.float32) as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            expval(qml.PauliZ(0))
            expval(qml.PauliY(1))

        assert tape.trainable_params == {0, 1}
        res = tape.execute(dev)

        assert isinstance(res, torch.Tensor)
        assert res.shape == (2,)
        assert res.dtype is torch.float32

        loss = torch.sum(res)
        loss.backward()
        assert a.grad.dtype is torch.float32
        assert b.grad.dtype is torch.float32

    def test_reusing_quantum_tape(self, tol):
        """Test re-using a quantum tape by passing new parameters"""
        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        with TorchInterface.apply(QuantumTape()) as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            expval(qml.PauliZ(0))
            expval(qml.PauliY(1))

        assert tape.trainable_params == {0, 1}

        loss = torch.sum(tape.execute(dev))
        loss.backward()

        a_val = 0.54
        b_val = 0.8
        a = torch.tensor(a_val, requires_grad=True)
        b = torch.tensor(b_val, requires_grad=True)
        res2 = tape.execute(dev, params=[2 * a, b])

        expected = [np.cos(2 * a_val), -np.cos(2 * a_val) * np.sin(b_val)]
        assert np.allclose(res2.detach().numpy(), expected, atol=tol, rtol=0)

        loss = torch.sum(res2)
        loss.backward()

        expected = [
            -2 * np.sin(2 * a_val) + 2 * np.sin(2 * a_val) * np.sin(b_val),
            -np.cos(2 * a_val) * np.cos(b_val),
        ]

        assert np.allclose(a.grad, expected[0], atol=tol, rtol=0)
        assert np.allclose(b.grad, expected[1], atol=tol, rtol=0)

    def test_classical_processing(self, tol):
        """Test classical processing within the quantum tape"""
        p_val = [0.1, 0.2]
        params = torch.tensor(p_val, requires_grad=True)

        dev = qml.device("default.qubit", wires=1)

        with TorchInterface.apply(QuantumTape()) as tape:
            qml.RY(params[0] * params[1], wires=0)
            qml.RZ(0.2, wires=0)
            qml.RX(params[1] + params[1] ** 2 + torch.sin(params[0]), wires=0)
            expval(qml.PauliZ(0))

        assert tape.trainable_params == {0, 2}

        tape_params = [i.detach().numpy() for i in tape.get_parameters()]
        assert np.allclose(
            tape_params,
            [p_val[0] * p_val[1], p_val[1] + p_val[1] ** 2 + np.sin(p_val[0])],
            atol=tol,
            rtol=0,
        )

        res = tape.execute(dev)
        res.backward()

        assert isinstance(params.grad, torch.Tensor)
        assert params.shape == (2,)

    def test_no_trainable_parameters(self, tol):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        dev = qml.device("default.qubit", wires=2)

        with TorchInterface.apply(QuantumTape()) as tape:
            qml.RY(0.2, wires=0)
            qml.RX(torch.tensor(0.1), wires=0)
            qml.CNOT(wires=[0, 1])
            expval(qml.PauliZ(0))
            expval(qml.PauliZ(1))

        assert tape.trainable_params == set()

        res = tape.execute(dev)

        assert res.shape == (2,)
        assert isinstance(res, torch.Tensor)

        with pytest.raises(
            RuntimeError,
            match="element 0 of tensors does not require grad and does not have a grad_fn",
        ):
            res.backward()

    @pytest.mark.parametrize("U", [torch.tensor([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])])
    def test_matrix_parameter(self, U, tol):
        """Test that the Torch interface works correctly
        with a matrix parameter"""
        a_val = 0.1
        a = torch.tensor(a_val, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        with TorchInterface.apply(QuantumTape()) as tape:
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            expval(qml.PauliZ(0))

        assert tape.trainable_params == {1}
        res = tape.execute(dev)

        assert np.allclose(res.detach().numpy(), -np.cos(a_val), atol=tol, rtol=0)

        res.backward()
        assert np.allclose(a.grad, np.sin(a_val), atol=tol, rtol=0)

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

        tape = QuantumTape()

        dev = qml.device("default.qubit", wires=1)
        a = np.array(0.1)
        p_val = [0.1, 0.2, 0.3]
        p = torch.tensor(p_val, requires_grad=True)

        with tape:
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            expval(qml.PauliX(0))

        tape = TorchInterface.apply(tape.expand())

        assert tape.trainable_params == {1, 2, 3, 4}
        assert [i.name for i in tape.operations] == ["RX", "Rot", "PhaseShift"]

        tape_params = [i.detach().numpy() for i in tape.get_parameters()]
        assert np.allclose(
            tape_params, [p_val[2], p_val[0], -p_val[2], p_val[1] + p_val[2]], atol=tol, rtol=0
        )

        res = tape.execute(device=dev)

        expected = np.cos(a) * np.cos(p_val[1]) * np.sin(p_val[0]) + np.sin(a) * (
            np.cos(p_val[2]) * np.sin(p_val[1])
            + np.cos(p_val[0]) * np.cos(p_val[1]) * np.sin(p_val[2])
        )
        assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

        res.backward()
        expected = np.array(
            [
                np.cos(p_val[1])
                * (np.cos(a) * np.cos(p_val[0]) - np.sin(a) * np.sin(p_val[0]) * np.sin(p_val[2])),
                np.cos(p_val[1]) * np.cos(p_val[2]) * np.sin(a)
                - np.sin(p_val[1])
                * (np.cos(a) * np.sin(p_val[0]) + np.cos(p_val[0]) * np.sin(a) * np.sin(p_val[2])),
                np.sin(a)
                * (
                    np.cos(p_val[0]) * np.cos(p_val[1]) * np.cos(p_val[2])
                    - np.sin(p_val[1]) * np.sin(p_val[2])
                ),
            ]
        )
        assert np.allclose(p.grad, expected, atol=tol, rtol=0)

    def test_probability_differentiation(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        dev = qml.device("default.qubit", wires=2)
        x_val = 0.543
        y_val = -0.654
        x = torch.tensor(x_val, requires_grad=True)
        y = torch.tensor(y_val, requires_grad=True)

        with TorchInterface.apply(QuantumTape()) as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            probs(wires=[0])
            probs(wires=[1])

        res = tape.execute(dev)

        expected = np.array(
            [
                [np.cos(x_val / 2) ** 2, np.sin(x_val / 2) ** 2],
                [(1 + np.cos(x_val) * np.cos(y_val)) / 2, (1 - np.cos(x_val) * np.cos(y_val)) / 2],
            ]
        )
        assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

        loss = torch.sum(res)
        loss.backward()
        expected = np.array(
            [
                -np.sin(x_val) / 2
                + np.sin(x_val) / 2
                - np.sin(x_val) * np.cos(y_val) / 2
                + np.cos(y_val) * np.sin(x_val) / 2,
                -np.cos(x_val) * np.sin(y_val) / 2 + np.cos(x_val) * np.sin(y_val) / 2,
            ]
        )
        assert np.allclose(x.grad, expected[0], atol=tol, rtol=0)
        assert np.allclose(y.grad, expected[1], atol=tol, rtol=0)

    def test_ragged_differentiation(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x_val = 0.543
        y_val = -0.654
        x = torch.tensor(x_val, requires_grad=True)
        y = torch.tensor(y_val, requires_grad=True)

        with TorchInterface.apply(QuantumTape()) as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            expval(qml.PauliZ(0))
            probs(wires=[1])

        res = tape.execute(dev)

        expected = np.array(
            [
                np.cos(x_val),
                (1 + np.cos(x_val) * np.cos(y_val)) / 2,
                (1 - np.cos(x_val) * np.cos(y_val)) / 2,
            ]
        )
        assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

        loss = torch.sum(res)
        loss.backward()
        expected = np.array(
            [
                -np.sin(x_val)
                + -np.sin(x_val) * np.cos(y_val) / 2
                + np.cos(y_val) * np.sin(x_val) / 2,
                -np.cos(x_val) * np.sin(y_val) / 2 + np.cos(x_val) * np.sin(y_val) / 2,
            ]
        )
        assert np.allclose(x.grad, expected[0], atol=tol, rtol=0)
        assert np.allclose(y.grad, expected[1], atol=tol, rtol=0)

    def test_sampling(self):
        """Test sampling works as expected"""
        dev = qml.device("default.qubit", wires=2, shots=10)

        with TorchInterface.apply(QuantumTape()) as tape:
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            sample(qml.PauliZ(0))
            sample(qml.PauliX(1))

        res = tape.execute(dev)

        assert res.shape == (2, 10)
        assert isinstance(res, torch.Tensor)


class TestQNode:
    """Same tests as above, but this time via the QNode interface!"""

    def test_execution_no_interface(self):
        """Test execution works without an interface"""
        dev = qml.device("default.qubit", wires=1)

        @qnode(dev)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return expval(qml.PauliZ(0))

        a = torch.tensor(0.1, requires_grad=True)

        res = circuit(a)

        assert circuit.qtape.interface == "autograd"

        # without the interface, the tape simply returns an array of results
        assert isinstance(res, np.ndarray)
        assert res.shape == (1,)
        assert isinstance(res[0], float)

        # without the interface, the tape is unable to deduce
        # trainable parameters
        assert circuit.qtape.trainable_params == {0, 1}

    def test_execution_with_interface(self):
        """Test execution works with the interface"""
        dev = qml.device("default.qubit", wires=1)

        @qnode(dev, interface="torch")
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return expval(qml.PauliZ(0))

        a = torch.tensor(0.1, requires_grad=True)
        res = circuit(a)

        assert circuit.qtape.interface == "torch"

        # with the interface, the tape returns tensorflow tensors
        assert isinstance(res, torch.Tensor)
        assert res.shape == (1,)

        # the tape is able to deduce trainable parameters
        assert circuit.qtape.trainable_params == {0}

        # gradients should work
        res.backward()
        grad = a.grad
        assert isinstance(grad, torch.Tensor)
        assert grad.shape == tuple()

    def test_interface_swap(self, tol):
        """Test that the TF interface can be applied to a QNode
        with a pre-existing interface"""
        dev = qml.device("default.qubit", wires=1)

        @qnode(dev, interface="autograd")
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return expval(qml.PauliZ(0))

        from pennylane import numpy as anp

        a = anp.array(0.1, requires_grad=True)

        res1 = circuit(a)
        grad_fn = qml.grad(circuit)
        grad1 = grad_fn(a)

        # switch to TF interface
        circuit.to_torch()

        a = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

        res2 = circuit(a)
        res2.backward()
        grad2 = a.grad
        assert np.allclose(res1, res2.detach().numpy(), atol=tol, rtol=0)
        assert np.allclose(grad1, grad2, atol=tol, rtol=0)

    def test_jacobian(self, mocker, tol):
        """Test jacobian calculation"""
        spy = mocker.spy(QuantumTape, "jacobian")

        a_val = 0.1
        b_val = 0.2

        a = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(b_val, dtype=torch.float64, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="torch")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [expval(qml.PauliZ(0)), expval(qml.PauliY(1))]

        res = circuit(a, b)

        assert circuit.qtape.trainable_params == {0, 1}

        assert isinstance(res, torch.Tensor)
        assert res.shape == (2,)

        expected = [np.cos(a_val), -np.cos(a_val) * np.sin(b_val)]
        assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

        loss = torch.sum(res)

        loss.backward()
        expected = [-np.sin(a_val) + np.sin(a_val) * np.sin(b_val), -np.cos(a_val) * np.cos(b_val)]
        assert np.allclose(a.grad, expected[0], atol=tol, rtol=0)
        assert np.allclose(b.grad, expected[1], atol=tol, rtol=0)

        spy.assert_called()

    def test_jacobian_dtype(self, tol):
        """Test calculating the jacobian with a different datatype"""
        a = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [expval(qml.PauliZ(0)), expval(qml.PauliY(1))]

        circuit.to_torch(dtype=torch.float32)
        assert circuit.dtype is torch.float32

        res = circuit(a, b)

        assert circuit.qtape.interface == "torch"
        assert circuit.qtape.trainable_params == {0, 1}

        assert isinstance(res, torch.Tensor)
        assert res.shape == (2,)
        assert res.dtype is torch.float32

        loss = torch.sum(res)
        loss.backward()
        assert a.grad.dtype is torch.float32
        assert b.grad.dtype is torch.float32

    def test_jacobian_options(self, mocker, tol):
        """Test setting jacobian options"""
        spy = mocker.spy(QuantumTape, "numeric_pd")

        a = torch.tensor([0.1, 0.2], requires_grad=True)

        dev = qml.device("default.qubit", wires=1)

        @qnode(dev, interface="torch", h=1e-8, order=2)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return expval(qml.PauliZ(0))

        res = circuit(a)
        res.backward()

        for args in spy.call_args_list:
            assert args[1]["order"] == 2
            assert args[1]["h"] == 1e-8

    def test_changing_trainability(self, mocker, tol):
        """Test changing the trainability of parameters changes the
        number of differentiation requests made"""
        a_val = 0.1
        b_val = 0.2

        a = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(b_val, dtype=torch.float64, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="torch")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return expval(qml.PauliZ(0)), expval(qml.PauliY(1))

        res = circuit(a, b)

        # the tape has reported both gate arguments as trainable
        assert circuit.qtape.trainable_params == {0, 1}

        expected = [np.cos(a_val), -np.cos(a_val) * np.sin(b_val)]
        assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

        spy = mocker.spy(QuantumTape, "numeric_pd")

        loss = torch.sum(res)
        loss.backward()

        expected = [-np.sin(a_val) + np.sin(a_val) * np.sin(b_val), -np.cos(a_val) * np.cos(b_val)]
        assert np.allclose([a.grad, b.grad], expected, atol=tol, rtol=0)

        # QuantumTape.numeric_pd has been called for each argument
        assert len(spy.call_args_list) == 2

        # make the second QNode argument a constant
        a_val = 0.54
        b_val = 0.8

        a = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(b_val, dtype=torch.float64, requires_grad=False)

        res = circuit(a, b)

        # the tape has reported only the first argument as trainable
        assert circuit.qtape.trainable_params == {0}

        expected = [np.cos(a_val), -np.cos(a_val) * np.sin(b_val)]
        assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

        spy.call_args_list = []
        loss = torch.sum(res)
        loss.backward()
        expected = -np.sin(a_val) + np.sin(a_val) * np.sin(b_val)
        assert np.allclose(a.grad, expected, atol=tol, rtol=0)

        # QuantumTape.numeric_pd has been called only once
        assert len(spy.call_args_list) == 1

    def test_classical_processing(self, tol):
        """Test classical processing within the quantum tape"""
        a = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(0.2, dtype=torch.float64, requires_grad=False)
        c = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

        dev = qml.device("default.qubit", wires=1)

        @qnode(dev, interface="torch")
        def circuit(a, b, c):
            qml.RY(a * c, wires=0)
            qml.RZ(b, wires=0)
            qml.RX(c + c ** 2 + torch.sin(a), wires=0)
            return expval(qml.PauliZ(0))

        res = circuit(a, b, c)

        assert circuit.qtape.trainable_params == {0, 2}
        assert circuit.qtape.get_parameters() == [a * c, c + c ** 2 + torch.sin(a)]

        res.backward()

        assert isinstance(a.grad, torch.Tensor)
        assert b.grad is None
        assert isinstance(c.grad, torch.Tensor)

    def test_no_trainable_parameters(self, tol):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="torch")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.CNOT(wires=[0, 1])
            return expval(qml.PauliZ(0)), expval(qml.PauliZ(1))

        a = 0.1
        b = torch.tensor(0.2, dtype=torch.float64, requires_grad=False)

        res = circuit(a, b)

        assert circuit.qtape.trainable_params == set()

        assert res.shape == (2,)
        assert isinstance(res, torch.Tensor)

    @pytest.mark.parametrize(
        "U", [torch.tensor([[0, 1], [1, 0]], requires_grad=False), np.array([[0, 1], [1, 0]])]
    )
    def test_matrix_parameter(self, U, tol):
        """Test that the TF interface works correctly
        with a matrix parameter"""
        a_val = 0.1
        a = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="torch")
        def circuit(U, a):
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            return expval(qml.PauliZ(0))

        res = circuit(U, a)

        assert circuit.qtape.trainable_params == {1}

        assert np.allclose(res.detach(), -np.cos(a_val), atol=tol, rtol=0)

        res.backward()
        assert np.allclose(a.grad, np.sin(a_val), atol=tol, rtol=0)

    def test_differentiable_expand(self, mocker, tol):
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

        dev = qml.device("default.qubit", wires=1)
        a = np.array(0.1)
        p_val = [0.1, 0.2, 0.3]
        p = torch.tensor(p_val, dtype=torch.float64, requires_grad=True)

        @qnode(dev, interface="torch")
        def circuit(a, p):
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            return expval(qml.PauliX(0))

        res = circuit(a, p)

        tape_params = [i.detach().numpy() for i in circuit.qtape.get_parameters()]
        assert np.allclose(
            tape_params, [p_val[2], p_val[0], -p_val[2], p_val[1] + p_val[2]], atol=tol, rtol=0
        )

        expected = np.cos(a) * np.cos(p_val[1]) * np.sin(p_val[0]) + np.sin(a) * (
            np.cos(p_val[2]) * np.sin(p_val[1])
            + np.cos(p_val[0]) * np.cos(p_val[1]) * np.sin(p_val[2])
        )
        assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

        res.backward()
        expected = np.array(
            [
                np.cos(p_val[1])
                * (np.cos(a) * np.cos(p_val[0]) - np.sin(a) * np.sin(p_val[0]) * np.sin(p_val[2])),
                np.cos(p_val[1]) * np.cos(p_val[2]) * np.sin(a)
                - np.sin(p_val[1])
                * (np.cos(a) * np.sin(p_val[0]) + np.cos(p_val[0]) * np.sin(a) * np.sin(p_val[2])),
                np.sin(a)
                * (
                    np.cos(p_val[0]) * np.cos(p_val[1]) * np.cos(p_val[2])
                    - np.sin(p_val[1]) * np.sin(p_val[2])
                ),
            ]
        )
        assert np.allclose(p.grad, expected, atol=tol, rtol=0)

    def test_probability_differentiation(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        dev = qml.device("default.qubit", wires=2)
        x_val = 0.543
        y_val = -0.654
        x = torch.tensor(x_val, requires_grad=True)
        y = torch.tensor(y_val, requires_grad=True)

        @qnode(dev, interface="torch")
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return probs(wires=[0]), probs(wires=[1])

        res = circuit(x, y)

        expected = np.array(
            [
                [np.cos(x_val / 2) ** 2, np.sin(x_val / 2) ** 2],
                [(1 + np.cos(x_val) * np.cos(y_val)) / 2, (1 - np.cos(x_val) * np.cos(y_val)) / 2],
            ]
        )
        assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

        loss = torch.sum(res)
        loss.backward()
        expected = np.array(
            [
                -np.sin(x_val) / 2
                + np.sin(x_val) / 2
                - np.sin(x_val) * np.cos(y_val) / 2
                + np.cos(y_val) * np.sin(x_val) / 2,
                -np.cos(x_val) * np.sin(y_val) / 2 + np.cos(x_val) * np.sin(y_val) / 2,
            ]
        )
        assert np.allclose(x.grad, expected[0], atol=tol, rtol=0)
        assert np.allclose(y.grad, expected[1], atol=tol, rtol=0)

    def test_ragged_differentiation(self, monkeypatch, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x_val = 0.543
        y_val = -0.654
        x = torch.tensor(x_val, requires_grad=True)
        y = torch.tensor(y_val, requires_grad=True)

        @qnode(dev, interface="torch")
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return [expval(qml.PauliZ(0)), probs(wires=[1])]

        res = circuit(x, y)

        expected = np.array(
            [
                np.cos(x_val),
                (1 + np.cos(x_val) * np.cos(y_val)) / 2,
                (1 - np.cos(x_val) * np.cos(y_val)) / 2,
            ]
        )
        assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

        loss = torch.sum(res)
        loss.backward()
        expected = np.array(
            [
                -np.sin(x_val)
                + -np.sin(x_val) * np.cos(y_val) / 2
                + np.cos(y_val) * np.sin(x_val) / 2,
                -np.cos(x_val) * np.sin(y_val) / 2 + np.cos(x_val) * np.sin(y_val) / 2,
            ]
        )
        assert np.allclose(x.grad, expected[0], atol=tol, rtol=0)
        assert np.allclose(y.grad, expected[1], atol=tol, rtol=0)

    def test_sampling(self):
        """Test sampling works as expected"""
        dev = qml.device("default.qubit", wires=2, shots=10)

        @qnode(dev, interface="torch")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return [sample(qml.PauliZ(0)), sample(qml.PauliX(1))]

        res = circuit()

        assert res.shape == (2, 10)
        assert isinstance(res, torch.Tensor)
