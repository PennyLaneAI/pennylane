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
from pennylane import qnode, QNode
from pennylane.tape import JacobianTape
from torch.autograd.functional import hessian, jacobian


@pytest.mark.parametrize(
    "dev_name,diff_method",
    [
        ["default.qubit", "finite-diff"],
        ["default.qubit", "parameter-shift"],
        ["default.qubit", "adjoint"],
    ],
)
class TestQNode:
    """Same tests as above, but this time via the QNode interface!"""

    def test_import_error(self, dev_name, diff_method, mocker):
        """Test that an exception is caught on import error"""
        mock = mocker.patch("pennylane.interfaces.torch.TorchInterface.apply")
        mock.side_effect = ImportError()

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device(dev_name, wires=2)
        qn = QNode(func, dev, interface="torch", diff_method=diff_method)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="PyTorch not found. Please install the latest version of PyTorch to enable the 'torch' interface",
        ):
            qn(0.1, 0.1)

    def test_execution_no_interface(self, dev_name, diff_method):
        """Test execution works without an interface"""
        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = torch.tensor(0.1, requires_grad=True)

        res = circuit(a)

        assert circuit.qtape.interface == "autograd"

        # without the interface, the tape simply returns an array of results
        assert isinstance(res, np.ndarray)
        assert res.shape == tuple()

        # without the interface, the tape is unable to deduce
        # trainable parameters
        assert circuit.qtape.trainable_params == {0}

    def test_execution_with_interface(self, dev_name, diff_method):
        """Test execution works with the interface"""
        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = torch.tensor(0.1, requires_grad=True)
        res = circuit(a)

        assert circuit.qtape.interface == "torch"

        # with the interface, the tape returns torch tensors

        assert isinstance(res, torch.Tensor)
        assert res.shape == tuple()

        # the tape is able to deduce trainable parameters
        assert circuit.qtape.trainable_params == {0}

        # gradients should work
        res.backward()
        grad = a.grad
        assert isinstance(grad, torch.Tensor)
        assert grad.shape == tuple()

    def test_interface_swap(self, dev_name, diff_method, tol):
        """Test that the Torch interface can be applied to a QNode
        with a pre-existing interface"""
        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        from pennylane import numpy as anp

        a = anp.array(0.1, requires_grad=True)

        res1 = circuit(a)
        grad_fn = qml.grad(circuit)
        grad1 = grad_fn(a)

        # switch to Torch interface
        circuit.to_torch()

        a = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

        res2 = circuit(a)
        res2.backward()
        grad2 = a.grad
        assert np.allclose(res1, res2.detach().numpy(), atol=tol, rtol=0)
        assert np.allclose(grad1, grad2, atol=tol, rtol=0)

    def test_drawing(self, dev_name, diff_method):
        """Test circuit drawing when using the torch interface"""

        x = torch.tensor(0.1, requires_grad=True)
        y = torch.tensor([0.2, 0.3], requires_grad=True)
        z = torch.tensor(0.4, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="torch")
        def circuit(p1, p2=y, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=0), qml.var(qml.PauliZ(1))

        circuit(p1=x, p3=z)

        result = circuit.draw()
        expected = """\
 0: ──RX(0.1)───RX(0.4)──╭C──┤ Probs  
 1: ──RY(0.06)───────────╰X──┤ Var[Z] 
"""

        assert result == expected

    def test_jacobian(self, dev_name, diff_method, mocker, tol):
        """Test jacobian calculation"""
        spy = mocker.spy(JacobianTape, "jacobian")

        a_val = 0.1
        b_val = 0.2

        a = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(b_val, dtype=torch.float64, requires_grad=True)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]

        res = circuit(a, b)

        assert circuit.qtape.trainable_params == {0, 1}

        assert isinstance(res, torch.Tensor)
        assert res.shape == (2,)

        expected = [np.cos(a_val), -np.cos(a_val) * np.sin(b_val)]
        assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

        loss = torch.sum(res)

        loss.backward()
        expected = [
            -np.sin(a_val) + np.sin(a_val) * np.sin(b_val),
            -np.cos(a_val) * np.cos(b_val),
        ]
        assert np.allclose(a.grad, expected[0], atol=tol, rtol=0)
        assert np.allclose(b.grad, expected[1], atol=tol, rtol=0)

        spy.assert_called()

    def test_jacobian_dtype(self, dev_name, diff_method, tol):
        """Test calculating the jacobian with a different datatype"""
        a = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]

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

    def test_jacobian_options(self, dev_name, diff_method, mocker, tol):
        """Test setting jacobian options"""
        if diff_method != "finite-diff":
            pytest.skip("Test only works with finite-diff")

        spy = mocker.spy(JacobianTape, "numeric_pd")

        a = torch.tensor([0.1, 0.2], requires_grad=True)

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="torch", h=1e-8, order=2)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(a)
        res.backward()

        for args in spy.call_args_list:
            assert args[1]["order"] == 2
            assert args[1]["h"] == 1e-8

    def test_changing_trainability(self, dev_name, diff_method, mocker, tol):
        """Test that changing the trainability of parameters changes the
        number of differentiation requests made"""
        if diff_method != "finite-diff":
            pytest.skip("Test only works with finite-diff")

        a_val = 0.1
        b_val = 0.2

        a = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(b_val, dtype=torch.float64, requires_grad=True)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, interface="torch", diff_method="finite-diff")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        res = circuit(a, b)

        # the tape has reported both gate arguments as trainable
        assert circuit.qtape.trainable_params == {0, 1}

        expected = [np.cos(a_val), -np.cos(a_val) * np.sin(b_val)]
        assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

        spy = mocker.spy(JacobianTape, "numeric_pd")

        loss = torch.sum(res)
        loss.backward()

        expected = [
            -np.sin(a_val) + np.sin(a_val) * np.sin(b_val),
            -np.cos(a_val) * np.cos(b_val),
        ]
        assert np.allclose([a.grad, b.grad], expected, atol=tol, rtol=0)

        # JacobianTape.numeric_pd has been called for each argument
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

        # JacobianTape.numeric_pd has been called only once
        assert len(spy.call_args_list) == 1

    def test_classical_processing(self, dev_name, diff_method, tol):
        """Test classical processing within the quantum tape"""
        a = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(0.2, dtype=torch.float64, requires_grad=False)
        c = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(a, b, c):
            qml.RY(a * c, wires=0)
            qml.RZ(b, wires=0)
            qml.RX(c + c ** 2 + torch.sin(a), wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(a, b, c)

        assert circuit.qtape.trainable_params == {0, 2}
        assert circuit.qtape.get_parameters() == [a * c, c + c ** 2 + torch.sin(a)]

        res.backward()

        assert isinstance(a.grad, torch.Tensor)
        assert b.grad is None
        assert isinstance(c.grad, torch.Tensor)

    def test_no_trainable_parameters(self, dev_name, diff_method, tol):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        a = 0.1
        b = torch.tensor(0.2, dtype=torch.float64, requires_grad=False)

        res = circuit(a, b)

        assert circuit.qtape.trainable_params == set()

        assert res.shape == (2,)
        assert isinstance(res, torch.Tensor)

        with pytest.raises(
            RuntimeError,
            match="element 0 of tensors does not require grad and does not have a grad_fn",
        ):
            res.backward()

    @pytest.mark.parametrize(
        "U",
        [
            torch.tensor([[0, 1], [1, 0]], requires_grad=False),
            np.array([[0, 1], [1, 0]]),
        ],
    )
    def test_matrix_parameter(self, dev_name, diff_method, U, tol):
        """Test that the Torch interface works correctly
        with a matrix parameter"""
        a_val = 0.1
        a = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(U, a):
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(U, a)
        assert circuit.qtape.trainable_params == {1}

        assert np.allclose(res.detach(), -np.cos(a_val), atol=tol, rtol=0)

        res.backward()
        assert np.allclose(a.grad, np.sin(a_val), atol=tol, rtol=0)

    def test_differentiable_expand(self, dev_name, diff_method, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""

        class U3(qml.U3):
            def expand(self):
                theta, phi, lam = self.data
                wires = self.wires

                with JacobianTape() as tape:
                    qml.Rot(lam, theta, -lam, wires=wires)
                    qml.PhaseShift(phi + lam, wires=wires)

                return tape

        dev = qml.device(dev_name, wires=1)
        a = np.array(0.1)
        p_val = [0.1, 0.2, 0.3]
        p = torch.tensor(p_val, dtype=torch.float64, requires_grad=True)

        @qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(a, p):
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            return qml.expval(qml.PauliX(0))

        res = circuit(a, p)

        tape_params = [i.detach().numpy() for i in circuit.qtape.get_parameters()]
        assert np.allclose(
            tape_params,
            [p_val[2], p_val[0], -p_val[2], p_val[1] + p_val[2]],
            atol=tol,
            rtol=0,
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

    def test_probability_differentiation(self, dev_name, diff_method, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        dev = qml.device(dev_name, wires=2)
        x_val = 0.543
        y_val = -0.654
        x = torch.tensor(x_val, requires_grad=True)
        y = torch.tensor(y_val, requires_grad=True)

        @qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0]), qml.probs(wires=[1])

        res = circuit(x, y)

        expected = np.array(
            [
                [np.cos(x_val / 2) ** 2, np.sin(x_val / 2) ** 2],
                [
                    (1 + np.cos(x_val) * np.cos(y_val)) / 2,
                    (1 - np.cos(x_val) * np.cos(y_val)) / 2,
                ],
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

    def test_ragged_differentiation(self, dev_name, diff_method, monkeypatch, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        dev = qml.device(dev_name, wires=2)
        x_val = 0.543
        y_val = -0.654
        x = torch.tensor(x_val, requires_grad=True)
        y = torch.tensor(y_val, requires_grad=True)

        @qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])]

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

    def test_sampling(self, dev_name, diff_method):
        """Test sampling works as expected"""
        dev = qml.device(dev_name, wires=2, shots=10)

        @qnode(dev, diff_method=diff_method, interface="torch")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return [qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))]

        res = circuit()

        assert res.shape == (2, 10)
        assert isinstance(res, torch.Tensor)

    def test_sampling_expval(self, dev_name, diff_method):
        """Test sampling works as expected if combined with expectation values"""
        dev = qml.device(dev_name, wires=2, shots=10)

        @qnode(dev, diff_method=diff_method, interface="torch")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        res = circuit()

        assert len(res) == 2
        assert isinstance(res, tuple)
        assert res[0].shape == (10,)
        assert isinstance(res[0], torch.Tensor)
        assert isinstance(res[1], torch.Tensor)

    def test_hessian(self, dev_name, diff_method, mocker, tol):
        """Test hessian calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        res = circuit(x)

        res.backward()
        g = x.grad

        spy = mocker.spy(JacobianTape, "hessian")
        hess = hessian(circuit, x)
        spy.assert_called_once()

        a, b = x.detach().numpy()

        expected_res = np.cos(a) * np.cos(b)
        assert np.allclose(res.detach(), expected_res, atol=tol, rtol=0)

        expected_g = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]
        assert np.allclose(g.detach(), expected_g, atol=tol, rtol=0)

        expected_hess = [
            [-np.cos(a) * np.cos(b), np.sin(a) * np.sin(b)],
            [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)],
        ]
        assert np.allclose(hess.detach(), expected_hess, atol=tol, rtol=0)

    def test_hessian_vector_valued(self, dev_name, diff_method, mocker, tol):
        """Test hessian calculation of a vector valued QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.probs(wires=0)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        res = circuit(x)
        jac_fn = lambda x: jacobian(circuit, x, create_graph=True)

        g = jac_fn(x)

        spy = mocker.spy(JacobianTape, "hessian")
        hess = jacobian(jac_fn, x)
        spy.assert_called_once()

        a, b = x.detach().numpy()

        expected_res = [
            0.5 + 0.5 * np.cos(a) * np.cos(b),
            0.5 - 0.5 * np.cos(a) * np.cos(b),
        ]
        assert np.allclose(res.detach(), expected_res, atol=tol, rtol=0)

        expected_g = [
            [-0.5 * np.sin(a) * np.cos(b), -0.5 * np.cos(a) * np.sin(b)],
            [0.5 * np.sin(a) * np.cos(b), 0.5 * np.cos(a) * np.sin(b)],
        ]
        assert np.allclose(g.detach(), expected_g, atol=tol, rtol=0)

        expected_hess = [
            [
                [-0.5 * np.cos(a) * np.cos(b), 0.5 * np.sin(a) * np.sin(b)],
                [0.5 * np.sin(a) * np.sin(b), -0.5 * np.cos(a) * np.cos(b)],
            ],
            [
                [0.5 * np.cos(a) * np.cos(b), -0.5 * np.sin(a) * np.sin(b)],
                [-0.5 * np.sin(a) * np.sin(b), 0.5 * np.cos(a) * np.cos(b)],
            ],
        ]
        assert np.allclose(hess.detach(), expected_hess, atol=tol, rtol=0)

    def test_hessian_ragged(self, dev_name, diff_method, mocker, tol):
        """Test hessian calculation of a ragged QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            qml.RY(x[0], wires=1)
            qml.RX(x[1], wires=1)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=1)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        res = circuit(x)
        jac_fn = lambda x: jacobian(circuit, x, create_graph=True)

        g = jac_fn(x)

        spy = mocker.spy(JacobianTape, "hessian")
        hess = jacobian(jac_fn, x)
        spy.assert_called_once()

        a, b = x.detach().numpy()

        expected_res = [
            np.cos(a) * np.cos(b),
            0.5 + 0.5 * np.cos(a) * np.cos(b),
            0.5 - 0.5 * np.cos(a) * np.cos(b),
        ]
        assert np.allclose(res.detach(), expected_res, atol=tol, rtol=0)

        expected_g = [
            [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)],
            [-0.5 * np.sin(a) * np.cos(b), -0.5 * np.cos(a) * np.sin(b)],
            [0.5 * np.sin(a) * np.cos(b), 0.5 * np.cos(a) * np.sin(b)],
        ]
        assert np.allclose(g.detach(), expected_g, atol=tol, rtol=0)

        expected_hess = [
            [
                [-np.cos(a) * np.cos(b), np.sin(a) * np.sin(b)],
                [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)],
            ],
            [
                [-0.5 * np.cos(a) * np.cos(b), 0.5 * np.sin(a) * np.sin(b)],
                [0.5 * np.sin(a) * np.sin(b), -0.5 * np.cos(a) * np.cos(b)],
            ],
            [
                [0.5 * np.cos(a) * np.cos(b), -0.5 * np.sin(a) * np.sin(b)],
                [-0.5 * np.sin(a) * np.sin(b), 0.5 * np.cos(a) * np.cos(b)],
            ],
        ]
        assert np.allclose(hess.detach(), expected_hess, atol=tol, rtol=0)

    def test_hessian_vector_valued_postprocessing(self, dev_name, diff_method, mocker, tol):
        """Test hessian calculation of a vector valued QNode with post-processing"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0))]

        x = torch.tensor([0.76, -0.87], requires_grad=True, dtype=torch.float64)

        def cost_fn(x):
            return x @ circuit(x)

        a, b = x.detach().numpy()

        res = cost_fn(x)
        expected_res = np.array([a, b]) @ [np.cos(a) * np.cos(b), np.cos(a) * np.cos(b)]
        assert np.allclose(res.detach(), expected_res, atol=tol, rtol=0)

        res.backward()

        g = x.grad
        expected_g = [
            np.cos(b) * (np.cos(a) - (a + b) * np.sin(a)),
            np.cos(a) * (np.cos(b) - (a + b) * np.sin(b)),
        ]
        assert np.allclose(g.detach(), expected_g, atol=tol, rtol=0)

        spy = mocker.spy(JacobianTape, "hessian")
        hess = hessian(cost_fn, x)

        if diff_method == "backprop":
            spy.assert_not_called()
        elif diff_method == "parameter-shift":
            spy.assert_called_once()

        expected_hess = [
            [
                -(np.cos(b) * ((a + b) * np.cos(a) + 2 * np.sin(a))),
                -(np.cos(b) * np.sin(a)) + (-np.cos(a) + (a + b) * np.sin(a)) * np.sin(b),
            ],
            [
                -(np.cos(b) * np.sin(a)) + (-np.cos(a) + (a + b) * np.sin(a)) * np.sin(b),
                -(np.cos(a) * ((a + b) * np.cos(b) + 2 * np.sin(b))),
            ],
        ]

        assert np.allclose(hess.detach(), expected_hess, atol=tol, rtol=0)


class Test_adjoint:
    def test_adjoint_default_save_state(self, mocker):
        """tests that the state will be saved by default"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="adjoint", interface="torch")
        def circ(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=(0, 1))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        expected_grad = lambda x: torch.tensor([-torch.sin(x[0]), torch.cos(x[1])])

        spy = mocker.spy(dev, "adjoint_jacobian")

        x1 = torch.tensor([0.1, 0.2], requires_grad=True)
        x2 = torch.tensor([0.3, 0.4], requires_grad=True)

        res1 = circ(x1)
        res2 = circ(x2)

        res1.backward(torch.Tensor([1, 1]))
        res2.backward(torch.Tensor([1, 1]))

        assert np.allclose(x1.grad, expected_grad(x1))
        assert np.allclose(x2.grad, expected_grad(x2))

        assert circ.device.num_executions == 2

        spy.assert_called_with(mocker.ANY, starting_state=mocker.ANY)

    def test_adjoint_save_state(self, mocker):
        """Tests that the torch interface reuses device state when prompted by `cache_state=True`.
        Also tests a second execution before backward pass does not alter gradient.
        """

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="adjoint", interface="torch", adjoint_cache=True)
        def circ(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=(0, 1))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        expected_grad = lambda x: torch.tensor([-torch.sin(x[0]), torch.cos(x[1])])

        spy = mocker.spy(dev, "adjoint_jacobian")

        x1 = torch.tensor([0.1, 0.2], requires_grad=True)
        x2 = torch.tensor([0.3, 0.4], requires_grad=True)

        res1 = circ(x1)
        res2 = circ(x2)

        res1.backward(torch.Tensor([1, 1]))
        res2.backward(torch.Tensor([1, 1]))

        assert np.allclose(x1.grad, expected_grad(x1))
        assert np.allclose(x2.grad, expected_grad(x2))

        assert circ.device.num_executions == 2

        spy.assert_called_with(mocker.ANY, starting_state=mocker.ANY)

        assert circ.qtape.jacobian_options["adjoint_cache"] == True

    def test_adjoint_no_save_state(self, mocker):
        """Tests that with `adjoint_cache=False`, the state is not cached"""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, diff_method="adjoint", interface="torch", adjoint_cache=False)
        def circ(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        spy = mocker.spy(dev, "adjoint_jacobian")

        x = torch.tensor(0.1, requires_grad=True)
        res = circ(x)
        res.backward()

        assert circ.device.num_executions == 2

        spy.assert_called_with(mocker.ANY)

        assert circ.qtape.jacobian_options.get("adjoint_cache", False) == False


def qtransform(qnode, a, framework=torch):
    """Transforms every RY(y) gate in a circuit to RX(-a*cos(y))"""

    def construct(self, args, kwargs):
        """New quantum tape construct method, that performs
        the transform on the tape in a define-by-run manner"""

        # the following global variable is defined simply for testing
        # purposes, so that we can easily extract the transformed operations
        # for verification.
        global t_op

        t_op = []

        QNode.construct(self, args, kwargs)

        new_ops = []
        for o in self.qtape.operations:
            # here, we loop through all tape operations, and make
            # the transformation if a RY gate is encountered.
            if isinstance(o, qml.RY):
                t_op.append(qml.RX(-a * framework.cos(o.data[0]), wires=o.wires))
                new_ops.append(t_op[-1])
            else:
                new_ops.append(o)

        self.qtape._ops = new_ops
        self.qtape._update()

    import copy

    new_qnode = copy.deepcopy(qnode)
    new_qnode.construct = construct.__get__(new_qnode, QNode)
    return new_qnode


def test_transform(tol):
    """Test an example transform"""

    dev = qml.device("default.qubit", wires=1)

    @qnode(dev, interface="torch")
    def circuit(weights):
        # the following global variables are defined simply for testing
        # purposes, so that we can easily extract the operations for verification.
        global op1, op2
        op1 = qml.RY(weights[0], wires=0)
        op2 = qml.RX(weights[1], wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    weights = torch.tensor([0.32, 0.543], requires_grad=True)
    a = torch.tensor(0.5, requires_grad=True)

    # transform the circuit QNode with trainable weight 'a'
    new_qnode = qtransform(circuit, a)

    # evaluate the transformed QNode
    res = new_qnode(weights)

    # evaluate the original QNode with pre-processed parameters
    res2 = circuit(torch.sin(weights))

    # the loss is the sum of the two QNode evaluations
    loss = res + res2

    # verify that the transformed QNode has the expected operations
    assert circuit.qtape.operations == [op1, op2]
    assert new_qnode.qtape.operations[0] == t_op[0]
    assert new_qnode.qtape.operations[1].name == op2.name
    assert new_qnode.qtape.operations[1].wires == op2.wires

    # check that the incident gate arguments of both QNode tapes are correct
    tape_params = [p.detach() for p in circuit.qtape.get_parameters()]
    assert np.all(tape_params == [torch.sin(w) for w in weights])

    tape_params = [p.detach() for p in new_qnode.qtape.get_parameters()]
    assert np.all(tape_params == [-a * torch.cos(weights[0]), weights[1]])

    # verify that the gradient has the correct shape
    loss.backward()
    assert weights.grad.shape == weights.shape
    assert a.grad.shape == a.shape

    # compare against the expected values
    assert np.allclose(loss.detach(), 1.8244501889992706, atol=tol, rtol=0)
    assert np.allclose(weights.grad, [-0.26610258, -0.47053553], atol=tol, rtol=0)
    assert np.allclose(a.grad, 0.06486032, atol=tol, rtol=0)
