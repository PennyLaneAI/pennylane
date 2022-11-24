# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for using the Torch interface with a QNode"""
import numpy as np
import pytest

pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch", minversion="1.3")
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian

import pennylane as qml
from pennylane import QNode, qnode
from pennylane.tape import QuantumTape

qubit_device_and_diff_method_and_mode = [
    ["default.qubit", "backprop", "forward"],
    ["default.qubit", "finite-diff", "backward"],
    ["default.qubit", "parameter-shift", "backward"],
    ["default.qubit", "adjoint", "forward"],
    ["default.qubit", "adjoint", "backward"],
]

@pytest.mark.parametrize("dev_name,diff_method,mode", qubit_device_and_diff_method_and_mode)
@pytest.mark.parametrize("shots", [None, 10000])
class TestReturn:
    """Class to test the shape of the Grad/Jacobian/Hessian with different return types."""

    def test_grad_single_measurement_param(self, dev_name, diff_method, mode, shots):
        """For one measurement and one param, the gradient is a float."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=1, shots=shots)

        @qnode(dev, interface="torch", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = torch.tensor(0.1, requires_grad=True)

        res = circuit(a)

        # gradient
        res.backward()
        grad = a.grad

        assert isinstance(grad, torch.Tensor)
        assert grad.shape == ()

    def test_grad_single_measurement_multiple_param(self, dev_name, diff_method, mode, shots):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=1, shots=shots)

        @qnode(dev, interface="torch", diff_method=diff_method, mode=mode)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=True)

        res = circuit(a, b)

        # gradient
        res.backward()
        grad_a = a.grad
        grad_b = b.grad
        print(grad_a, grad_b)
        assert grad_a.shape == ()
        assert grad_b.shape == ()

    def test_grad_single_measurement_multiple_param_array(self, dev_name, diff_method, mode, shots):
        """For one measurement and multiple param as a single array params, the gradient is an array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=1, shots=shots)

        @qnode(dev, interface="torch", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        a = torch.tensor([0.1, 0.2], requires_grad=True)

        res = circuit(a)

        # gradient
        res.backward()
        grad = a.grad
        print(grad)
        assert isinstance(grad, torch.Tensor)
        assert grad.shape == (2,)

    def test_jacobian_single_measurement_param_probs(self, dev_name, diff_method, mode, shots):
        """For a multi dimensional measurement (probs), check that a single array is returned with the correct
        dimension"""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, interface="torch", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.probs(wires=[0, 1])

        a = torch.tensor(0.1, requires_grad=True)

        jac = jacobian(circuit, a)
        print(jac)
        assert isinstance(jac, torch.Tensor)
        assert jac.shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param(
        self, dev_name, diff_method, mode, shots
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, interface="torch", diff_method=diff_method, mode=mode)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=[0, 1])

        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=True)

        jac = jacobian(circuit, (a, b))
        print(jac)
        assert isinstance(jac, tuple)

        assert isinstance(jac[0], torch.Tensor)
        assert jac[0].shape == (4,)

        assert isinstance(jac[1], torch.Tensor)
        assert jac[1].shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param_single_array(
        self, dev_name, diff_method, mode, shots
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, interface="torch", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.probs(wires=[0, 1])

        a = torch.tensor([0.1, 0.2], requires_grad=True)
        jac = jacobian(circuit, a)
        print(jac)
        assert isinstance(jac, torch.Tensor)
        assert jac.shape == (4, 2)

    def test_jacobian_expval_expval_multiple_params(self, dev_name, diff_method, mode, shots):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")
        dev = qml.device(dev_name, wires=2, shots=shots)

        par_0 = torch.tensor(0.1, requires_grad=True)
        par_1 = torch.tensor(0.2, requires_grad=True)

        @qnode(dev, interface="torch", diff_method=diff_method, max_diff=1, mode=mode)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        jac = jacobian(circuit, (par_0, par_1))
        print(jac)
        assert isinstance(jac, tuple)

        assert isinstance(jac[0], tuple)
        assert len(jac[0]) == 2
        assert isinstance(jac[0][0], torch.Tensor)
        assert jac[0][0].shape == ()
        assert isinstance(jac[0][1], torch.Tensor)
        assert jac[0][1].shape == ()

        assert isinstance(jac[1], tuple)
        assert len(jac[1]) == 2
        assert isinstance(jac[1][0], torch.Tensor)
        assert jac[1][0].shape == ()
        assert isinstance(jac[1][1], torch.Tensor)
        assert jac[1][1].shape == ()

    def test_jacobian_expval_expval_multiple_params_array(self, dev_name, diff_method, mode, shots):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")
        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, interface="torch", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        a = torch.tensor([0.1, 0.2], requires_grad=True)

        jac = jacobian(circuit, a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], torch.Tensor)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], torch.Tensor)
        assert jac[1].shape == (2,)

    def test_jacobian_var_var_multiple_params(self, dev_name, diff_method, mode, shots):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=2, shots=shots)

        par_0 = torch.tensor(0.1, requires_grad=True)
        par_1 = torch.tensor(0.2, requires_grad=True)

        @qnode(dev, interface="torch", diff_method=diff_method, max_diff=1, mode=mode)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.var(qml.PauliZ(0))

        jac = jacobian(circuit, (par_0, par_1))

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], tuple)
        assert len(jac[0]) == 2
        assert isinstance(jac[0][0], torch.Tensor)
        assert jac[0][0].shape == ()
        assert isinstance(jac[0][1], torch.Tensor)
        assert jac[0][1].shape == ()

        assert isinstance(jac[1], tuple)
        assert len(jac[1]) == 2
        assert isinstance(jac[1][0], torch.Tensor)
        assert jac[1][0].shape == ()
        assert isinstance(jac[1][1], torch.Tensor)
        assert jac[1][1].shape == ()

    def test_jacobian_var_var_multiple_params_array(self, dev_name, diff_method, mode, shots):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, interface="torch", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.var(qml.PauliZ(0))

        a = torch.tensor([0.1, 0.2], requires_grad=True)

        jac = jacobian(circuit, a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], torch.Tensor)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], torch.Tensor)
        assert jac[1].shape == (2,)

    def test_jacobian_multiple_measurement_single_param(self, dev_name, diff_method, mode, shots):
        """The jacobian of multiple measurements with a single params return an array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")
        dev = qml.device(dev_name, wires=2, shots=shots)

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        @qnode(dev, interface="torch", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = torch.tensor(0.1, requires_grad=True)

        jac = jacobian(circuit, a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], torch.Tensor)
        assert jac[0].shape == ()

        assert isinstance(jac[1], torch.Tensor)
        assert jac[1].shape == (4,)

    def test_jacobian_multiple_measurement_multiple_param(self, dev_name, diff_method, mode, shots):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, interface="torch", diff_method=diff_method, mode=mode)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=True)

        jac = jacobian(circuit, (a, b))

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], tuple)
        assert len(jac[0]) == 2
        assert isinstance(jac[0][0], torch.Tensor)
        assert jac[0][0].shape == ()
        assert isinstance(jac[0][1], torch.Tensor)
        assert jac[0][1].shape == ()

        assert isinstance(jac[1], tuple)
        assert len(jac[1]) == 2
        assert isinstance(jac[1][0], torch.Tensor)
        assert jac[1][0].shape == (4,)
        assert isinstance(jac[1][1], torch.Tensor)
        assert jac[1][1].shape == (4,)

    def test_jacobian_multiple_measurement_multiple_param_array(
        self, dev_name, diff_method, mode, shots
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, interface="torch", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = torch.tensor([0.1, 0.2], requires_grad=True)

        jac = jacobian(circuit, a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], torch.Tensor)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], torch.Tensor)
        assert jac[1].shape == (4, 2)

    def test_hessian_expval_multiple_params(self, dev_name, diff_method, mode, shots):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")
        dev = qml.device(dev_name, wires=2, shots=shots)

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        par_0 = torch.tensor(0.1, requires_grad=True)
        par_1 = torch.tensor(0.2, requires_grad=True)

        @qnode(dev, interface="torch", diff_method=diff_method, max_diff=2, mode=mode)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        hess = hessian(circuit, (par_0, par_1))

        print(hess)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], torch.Tensor)
        assert hess[0][0].shape == ()
        assert hess[0][1].shape == ()

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], torch.Tensor)
        assert hess[1][0].shape == ()
        assert hess[1][1].shape == ()

    def test_hessian_expval_multiple_param_array(self, dev_name, diff_method, mode, shots):
        """The hessian of single measurement with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=2, shots=shots)

        params = torch.tensor([0.1, 0.2], requires_grad=True)

        @qnode(dev, interface="torch", diff_method=diff_method, max_diff=2, mode=mode)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        hess = hessian(circuit, params)

        assert isinstance(hess, torch.Tensor)
        assert hess.shape == (2, 2)

    def test_hessian_var_multiple_params(self, dev_name, diff_method, mode, shots):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")
        dev = qml.device(dev_name, wires=2, shots=shots)

        par_0 = torch.tensor(0.1, requires_grad=True)
        par_1 = torch.tensor(0.2, requires_grad=True)

        @qnode(dev, interface="torch", diff_method=diff_method, max_diff=2, mode=mode)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        hess = hessian(circuit, (par_0, par_1))

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], torch.Tensor)
        assert hess[0][0].shape == ()
        assert hess[0][1].shape == ()

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], torch.Tensor)
        assert hess[1][0].shape == ()
        assert hess[1][1].shape == ()

    def test_hessian_var_multiple_param_array(self, dev_name, diff_method, mode, shots):
        """The hessian of single measurement with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=2, shots=shots)

        params = torch.tensor([0.1, 0.2], requires_grad=True)

        @qnode(dev, interface="torch", diff_method=diff_method, max_diff=2, mode=mode)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        hess = hessian(circuit, params)

        assert isinstance(hess, torch.Tensor)
        assert hess.shape == (2, 2)

    def test_hessian_probs_expval_multiple_params(self, dev_name, diff_method, mode, shots):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2, shots=shots)
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par_0 = torch.tensor(0.1, requires_grad=True)
        par_1 = torch.tensor(0.2, requires_grad=True)

        @qnode(dev, interface="torch", diff_method=diff_method, max_diff=2, mode=mode)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0, 1])

        def circuit_stack(x, y):
            return torch.hstack(circuit(x, y))

        jac_fn = lambda x, y: jacobian(circuit_stack, (x, y), create_graph=True)

        hess = jacobian(jac_fn, (par_0, par_1))

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], torch.Tensor)
        assert tuple(hess[0][0].shape) == (5,)
        assert isinstance(hess[0][1], torch.Tensor)
        assert tuple(hess[0][1].shape) == (5,)

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], torch.Tensor)
        assert tuple(hess[1][0].shape) == (5,)
        assert isinstance(hess[1][1], torch.Tensor)
        assert tuple(hess[1][1].shape) == (5,)

    def test_hessian_expval_probs_multiple_param_array(self, dev_name, diff_method, mode, shots):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        dev = qml.device(dev_name, wires=2, shots=shots)
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par = torch.tensor([0.1, 0.2], requires_grad=True)

        @qnode(dev, interface="torch", diff_method=diff_method, max_diff=2, mode=mode)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0, 1])

        def circuit_stack(x):
            return torch.hstack(circuit(x))

        jac_fn = lambda x: jacobian(circuit_stack, x, create_graph=True)

        hess = jacobian(jac_fn, par)

        assert isinstance(hess, torch.Tensor)
        assert tuple(hess.shape) == (5, 2, 2)

    def test_hessian_probs_var_multiple_params(self, dev_name, diff_method, mode, shots):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2, shots=shots)
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par_0 = torch.tensor(0.1, requires_grad=True)
        par_1 = torch.tensor(0.2, requires_grad=True)

        @qnode(dev, interface="torch", diff_method=diff_method, max_diff=2, mode=mode)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0, 1])

        def circuit_stack(x, y):
            return torch.hstack(circuit(x, y))

        jac_fn = lambda x, y: jacobian(circuit_stack, (x, y), create_graph=True)

        hess = jacobian(jac_fn, (par_0, par_1))

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], torch.Tensor)
        assert tuple(hess[0][0].shape) == (5,)
        assert isinstance(hess[0][1], torch.Tensor)
        assert tuple(hess[0][1].shape) == (5,)

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], torch.Tensor)
        assert tuple(hess[1][0].shape) == (5,)
        assert isinstance(hess[1][1], torch.Tensor)
        assert tuple(hess[1][1].shape) == (5,)

    def test_hessian_var_probs_multiple_param_array(self, dev_name, diff_method, mode, shots):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        dev = qml.device(dev_name, wires=2, shots=shots)
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par = torch.tensor([0.1, 0.2], requires_grad=True)

        @qnode(dev, interface="torch", diff_method=diff_method, max_diff=2, mode=mode)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0, 1])

        def circuit_stack(x):
            return torch.hstack(circuit(x))

        jac_fn = lambda x: jacobian(circuit_stack, x, create_graph=True)

        hess = jacobian(jac_fn, par)
        print(hess)
        assert isinstance(hess, torch.Tensor)
        assert tuple(hess.shape) == (5, 2, 2)