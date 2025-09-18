# Copyright 2023 Xanadu Quantum Technologies Inc.

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

# pylint: disable=too-many-arguments,unexpected-keyword-arg,no-member,comparison-with-callable, no-name-in-module
# pylint: disable=use-implicit-booleaness-not-comparison, unnecessary-lambda-assignment, use-dict-literal
import numpy as np
import pytest
from param_shift_dev import ParamShiftDerivativesDevice

import pennylane as qml
from pennylane import qnode
from pennylane.devices import DefaultQubit
from pennylane.exceptions import DeviceError

pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch", minversion="1.3")
jacobian = torch.autograd.functional.jacobian
hessian = torch.autograd.functional.hessian

# device, diff_method, grad_on_execution, device_vjp
qubit_device_and_diff_method = [
    [DefaultQubit(), "finite-diff", False, False],
    [DefaultQubit(), "parameter-shift", False, False],
    [DefaultQubit(), "backprop", True, False],
    [DefaultQubit(), "adjoint", True, False],
    [DefaultQubit(), "adjoint", False, False],
    [DefaultQubit(), "adjoint", True, True],
    [DefaultQubit(), "adjoint", False, True],
    [DefaultQubit(), "spsa", False, False],
    [DefaultQubit(), "hadamard", False, False],
    [qml.device("lightning.qubit", wires=5), "adjoint", False, True],
    [qml.device("lightning.qubit", wires=5), "adjoint", True, True],
    [qml.device("lightning.qubit", wires=5), "adjoint", False, False],
    [qml.device("lightning.qubit", wires=5), "adjoint", True, False],
    [ParamShiftDerivativesDevice(), "parameter-shift", False, False],
    [ParamShiftDerivativesDevice(), "best", False, False],
    [ParamShiftDerivativesDevice(), "parameter-shift", True, False],
    [ParamShiftDerivativesDevice(), "parameter-shift", False, True],
    [qml.device("reference.qubit"), "parameter-shift", False, False],
]

interface_and_qubit_device_and_diff_method = [
    ["auto"] + inner_list for inner_list in qubit_device_and_diff_method
] + [["torch"] + inner_list for inner_list in qubit_device_and_diff_method]

TOL_FOR_SPSA = 1.0
H_FOR_SPSA = 0.01


@pytest.mark.parametrize(
    "interface, dev,diff_method,grad_on_execution, device_vjp",
    interface_and_qubit_device_and_diff_method,
)
class TestQNode:
    """Test that using the QNode with Torch integrates with the PennyLane stack"""

    def test_execution_with_interface(
        self, interface, dev, diff_method, grad_on_execution, device_vjp
    ):
        """Test execution works with the interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            interface=interface,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = torch.tensor(0.1, requires_grad=True)
        res = circuit(a)

        assert circuit.interface == interface

        # with the interface, the tape returns torch tensors

        assert isinstance(res, torch.Tensor)
        assert res.shape == ()

        # gradients should work
        res.backward()
        grad = a.grad

        assert isinstance(grad, torch.Tensor)
        assert grad.shape == ()

    def test_interface_swap(self, interface, dev, diff_method, grad_on_execution, device_vjp, tol):
        """Test that the Torch interface can be applied to a QNode
        with a pre-existing interface"""

        @qnode(
            dev,
            diff_method=diff_method,
            interface="autograd",
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
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
        circuit.interface = interface

        a = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

        res2 = circuit(a)
        res2.backward()
        grad2 = a.grad
        assert np.allclose(res1, res2.detach().numpy(), atol=tol, rtol=0)
        assert np.allclose(grad1, grad2, atol=tol, rtol=0)

    def test_drawing(self, interface, dev, diff_method, grad_on_execution, device_vjp):
        """Test circuit drawing when using the torch interface"""

        x = torch.tensor(0.1, requires_grad=True)
        y = torch.tensor([0.2, 0.3], requires_grad=True)
        z = torch.tensor(0.4, requires_grad=True)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(p1, p2=y, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        circuit(p1=x, p3=z)

        result = qml.draw(circuit)(p1=x, p3=z)
        expected = "0: ──RX(0.10)──RX(0.40)─╭●─┤  <Z>\n1: ──RY(0.06)───────────╰X─┤  <Z>"

        assert result == expected

    def test_jacobian(self, interface, dev, diff_method, grad_on_execution, device_vjp, tol, seed):
        """Test jacobian calculation"""
        kwargs = dict(
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            interface=interface,
            device_vjp=device_vjp,
        )
        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        a_val = 0.1
        b_val = 0.2

        a = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(b_val, dtype=torch.float64, requires_grad=True)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        res = circuit(a, b)

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], torch.Tensor)
        assert res[0].shape == ()

        assert isinstance(res[1], torch.Tensor)
        assert res[1].shape == ()

        expected = [np.cos(a_val), -np.cos(a_val) * np.sin(b_val)]
        assert np.allclose(res[0].detach().numpy(), expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1].detach().numpy(), expected[1], atol=tol, rtol=0)

        loss = res[0] + res[1]

        loss.backward()
        expected = [
            -np.sin(a_val) + np.sin(a_val) * np.sin(b_val),
            -np.cos(a_val) * np.cos(b_val),
        ]
        assert np.allclose(a.grad, expected[0], atol=tol, rtol=0)
        assert np.allclose(b.grad, expected[1], atol=tol, rtol=0)

    # TODO: fix this behavior with float: already present before return type.
    def test_jacobian_dtype(
        self,
        interface,
        dev,
        diff_method,
        grad_on_execution,
        device_vjp,
    ):
        """Test calculating the jacobian with a different datatype"""
        if not "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("Failing unless lightning.qubit")
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        a = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        res = circuit(a, b)

        assert circuit.interface == interface

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert res[0].dtype is torch.float32
        assert res[1].dtype is torch.float32

        loss = res[0] + res[1]
        loss.backward()
        assert a.grad.dtype is torch.float32
        assert b.grad.dtype is torch.float32

    def test_jacobian_options(
        self,
        interface,
        dev,
        diff_method,
        grad_on_execution,
        device_vjp,
    ):
        """Test setting jacobian options"""
        if diff_method not in {"finite-diff", "spsa"}:
            pytest.skip("Test only works with finite-diff and spsa")

        a = torch.tensor([0.1, 0.2], requires_grad=True)

        gradient_kwargs = {"h": 1e-8, "approx_order": 2}

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            interface=interface,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(a)
        res.backward()

    def test_changing_trainability(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol
    ):
        """Test that changing the trainability of parameters changes the
        number of differentiation requests made"""
        if diff_method != "parameter-shift":
            pytest.skip("Test only supports parameter-shift")

        a_val = 0.1
        b_val = 0.2

        a = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(b_val, dtype=torch.float64, requires_grad=True)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        res = circuit(a, b)

        expected = [np.cos(a_val), -np.cos(a_val) * np.sin(b_val)]

        assert np.allclose(res[0].detach().numpy(), expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1].detach().numpy(), expected[1], atol=tol, rtol=0)

        loss = res[0] + res[1]
        loss.backward()

        expected = [
            -np.sin(a_val) + np.sin(a_val) * np.sin(b_val),
            -np.cos(a_val) * np.cos(b_val),
        ]
        assert np.allclose([a.grad, b.grad], expected, atol=tol, rtol=0)

        # make the second QNode argument a constant
        a_val = 0.54
        b_val = 0.8

        a = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(b_val, dtype=torch.float64, requires_grad=False)

        res = circuit(a, b)

        expected = [np.cos(a_val), -np.cos(a_val) * np.sin(b_val)]

        assert np.allclose(res[0].detach().numpy(), expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1].detach().numpy(), expected[1], atol=tol, rtol=0)

        loss = res[0] + res[1]
        loss.backward()
        expected = -np.sin(a_val) + np.sin(a_val) * np.sin(b_val)
        assert np.allclose(a.grad, expected, atol=tol, rtol=0)

    def test_classical_processing(
        self,
        interface,
        dev,
        diff_method,
        grad_on_execution,
        device_vjp,
    ):
        """Test classical processing within the quantum tape"""
        a = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(0.2, dtype=torch.float64, requires_grad=False)
        c = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            interface=interface,
            device_vjp=device_vjp,
        )
        def circuit(a, b, c):
            qml.RY(a * c, wires=0)
            qml.RZ(b, wires=0)
            qml.RX(c + c**2 + torch.sin(a), wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(a, b, c)

        res.backward()

        assert isinstance(a.grad, torch.Tensor)
        assert b.grad is None
        assert isinstance(c.grad, torch.Tensor)

    def test_no_trainable_parameters(
        self,
        interface,
        dev,
        diff_method,
        grad_on_execution,
        device_vjp,
    ):
        """Test evaluation and Jacobian if there are no trainable parameters"""

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            interface=interface,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        a = 0.1
        b = torch.tensor(0.2, dtype=torch.float64, requires_grad=False)

        res = circuit(a, b)

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert res[0].shape == ()
        assert isinstance(res[0], torch.Tensor)

        assert res[1].shape == ()
        assert isinstance(res[1], torch.Tensor)

        with pytest.raises(
            RuntimeError,
            match="element 0 of tensors does not require grad and does not have a grad_fn",
        ):
            res[0].backward()

        with pytest.raises(
            RuntimeError,
            match="element 0 of tensors does not require grad and does not have a grad_fn",
        ):
            res[1].backward()

    @pytest.mark.parametrize(
        "U",
        [
            torch.tensor([[0, 1], [1, 0]], requires_grad=False),
            np.array([[0, 1], [1, 0]]),
        ],
    )
    def test_matrix_parameter(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, U, tol
    ):
        """Test that the Torch interface works correctly
        with a matrix parameter"""
        a_val = 0.1
        a = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            interface=interface,
            device_vjp=device_vjp,
        )
        def circuit(U, a):
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(U, a)

        assert np.allclose(res.detach(), -np.cos(a_val), atol=tol, rtol=0)

        res.backward()
        assert np.allclose(a.grad, np.sin(a_val), atol=tol, rtol=0)

    def test_differentiable_expand(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol, seed
    ):
        """Test that operation and nested tapes expansion
        is differentiable"""
        kwargs = dict(
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            interface=interface,
            device_vjp=device_vjp,
        )
        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        class U3(qml.U3):  # pylint:disable=too-few-public-methods
            def decomposition(self):
                theta, phi, lam = self.data
                wires = self.wires
                return [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]

        a = np.array(0.1)
        p_val = [0.1, 0.2, 0.3]
        p = torch.tensor(p_val, dtype=torch.float64, requires_grad=True)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(a, p):
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            return qml.expval(qml.PauliX(0))

        res = circuit(a, p)

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


class TestShotsIntegration:
    """Test that the QNode correctly changes shot value, and
    differentiates it."""

    @pytest.mark.xfail(reason="deprecated. To be removed in 0.44")
    def test_changing_shots(self):
        """Test that changing shots works on execution"""
        dev = DefaultQubit()
        a, b = torch.tensor([0.543, -0.654], requires_grad=True, dtype=torch.float64)

        @qnode(dev, interface="torch", diff_method=qml.gradients.param_shift)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.sample(wires=(0, 1))

        # execute with device default shots (None)
        with pytest.raises(DeviceError):
            circuit(a, b)

        # execute with shots=100
        res = circuit(a, b, shots=100)
        assert res.shape == (100, 2)

    # TODO: add this test after shot vectors addition
    @pytest.mark.xfail
    def test_gradient_integration(self):
        """Test that temporarily setting the shots works
        for gradient computations"""
        dev = DefaultQubit()
        a, b = torch.tensor([0.543, -0.654], requires_grad=True)

        @qnode(dev, interface="torch", diff_method=qml.gradients.param_shift)
        def cost_fn(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        res = jacobian(qml.set_shots(shots=[10000, 10000, 10000])(cost_fn), (a, b))
        res = qml.math.transpose(torch.stack(res))
        assert dev.shots is None
        assert len(res) == 3

        expected = torch.tensor([torch.sin(a) * torch.sin(b), -torch.cos(a) * torch.cos(b)])
        assert torch.allclose(torch.mean(res, axis=0), expected, atol=0.1, rtol=0)

    def test_multiple_gradient_integration(self, tol):
        """Test that temporarily setting the shots works
        for gradient computations, even if the QNode has been re-evaluated
        with a different number of shots in the meantime."""
        weights = torch.tensor([0.543, -0.654], requires_grad=True)
        a, b = weights

        @qnode(DefaultQubit(), interface="torch", diff_method=qml.gradients.param_shift)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        res1 = circuit(*weights)
        assert qml.math.shape(res1) == ()

        circuit_shots = qml.set_shots(shots=[(1, 1000)])(circuit)
        res2 = circuit_shots(*weights)
        assert len(res2) == 1000

        res1.backward()

        expected = torch.tensor([torch.sin(a) * torch.sin(b), -torch.cos(a) * torch.cos(b)])
        assert torch.allclose(weights.grad, expected, atol=tol, rtol=0)

    def test_update_diff_method(self):
        """Test that temporarily setting the shots updates the diff method"""
        a, b = torch.tensor([0.543, -0.654], requires_grad=True)
        dev = DefaultQubit()

        @qnode(dev, interface="torch")
        def cost_fn(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        with dev.tracker:
            cost_fn100 = qml.set_shots(shots=100)(cost_fn)
            res = cost_fn100(a, b)
            res.backward()
        # since we are using finite shots, use parameter shift
        assert dev.tracker.totals["executions"] == 5

        # if we use the default shots value of None, backprop can now be used
        with dev.tracker:
            res = cost_fn(a, b)
            res.backward()
        assert dev.tracker.totals["executions"] == 1


@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution, device_vjp",
    interface_and_qubit_device_and_diff_method,
)
class TestQubitIntegration:
    """Tests that ensure various qubit circuits integrate correctly"""

    def test_probability_differentiation(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol, seed
    ):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("lightning does not support measureing probabilities with adjoint.")
        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA

        x_val = 0.543
        y_val = -0.654
        x = torch.tensor(x_val, requires_grad=True, dtype=torch.float64)
        y = torch.tensor(y_val, requires_grad=True, dtype=torch.float64)

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            interface=interface,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
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

        assert np.allclose(res[0].detach().numpy(), expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1].detach().numpy(), expected[1], atol=tol, rtol=0)

        jac = jacobian(circuit, (x, y))

        res_0 = np.array([-np.sin(x_val) / 2, np.sin(x_val) / 2])
        res_1 = np.array([0.0, 0.0])
        res_2 = np.array([-np.sin(x_val) * np.cos(y_val) / 2, np.cos(y_val) * np.sin(x_val) / 2])
        res_3 = np.array([-np.cos(x_val) * np.sin(y_val) / 2, +np.cos(x_val) * np.sin(y_val) / 2])

        assert np.allclose(jac[0][0], res_0, atol=tol, rtol=0)
        assert np.allclose(jac[0][1], res_1, atol=tol, rtol=0)
        assert np.allclose(jac[1][0], res_2, atol=tol, rtol=0)
        assert np.allclose(jac[1][1], res_3, atol=tol, rtol=0)

    def test_ragged_differentiation(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol, seed
    ):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("lightning does not support measureing probabilities with adjoint.")
        kwargs = dict(
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            interface=interface,
            device_vjp=device_vjp,
        )
        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        x_val = 0.543
        y_val = -0.654
        x = torch.tensor(x_val, requires_grad=True, dtype=torch.float64)
        y = torch.tensor(y_val, requires_grad=True, dtype=torch.float64)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])

        res = circuit(x, y)

        res_0 = np.array(np.cos(x_val))
        res_1 = np.array(
            [(1 + np.cos(x_val) * np.cos(y_val)) / 2, (1 - np.cos(x_val) * np.cos(y_val)) / 2]
        )

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert np.allclose(res[0].detach().numpy(), res_0, atol=tol, rtol=0)
        assert np.allclose(res[1].detach().numpy(), res_1, atol=tol, rtol=0)

        jac = jacobian(circuit, (x, y))

        res_0 = -np.sin(x_val)
        res_1 = np.array(0.0)
        res_2 = np.array([-np.sin(x_val) * np.cos(y_val) / 2, np.cos(y_val) * np.sin(x_val) / 2])
        res_3 = np.array([-np.cos(x_val) * np.sin(y_val) / 2, +np.cos(x_val) * np.sin(y_val) / 2])

        assert np.allclose(jac[0][0], res_0, atol=tol, rtol=0)
        assert np.allclose(jac[0][1], res_1, atol=tol, rtol=0)
        assert np.allclose(jac[1][0], res_2, atol=tol, rtol=0)
        assert np.allclose(jac[1][1], res_3, atol=tol, rtol=0)

    def test_chained_qnodes(
        self,
        interface,
        dev,
        diff_method,
        grad_on_execution,
        device_vjp,
    ):
        """Test that the gradient of chained QNodes works without error"""

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit1(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit2(data, weights):
            data = qml.math.hstack(data)
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        def cost(weights):
            w1, w2 = weights
            c1 = circuit1(w1)
            c2 = circuit2(c1, w2)
            return torch.sum(c2) ** 2

        w1 = np.random.random(qml.templates.StronglyEntanglingLayers.shape(3, 2))
        w2 = np.random.random(qml.templates.StronglyEntanglingLayers.shape(4, 2))

        w1 = torch.tensor(w1, requires_grad=True)
        w2 = torch.tensor(w2, requires_grad=True)

        weights = [w1, w2]

        loss = cost(weights)
        loss.backward()

    def test_hessian(self, interface, dev, diff_method, grad_on_execution, device_vjp, tol):
        """Test hessian calculation of a scalar valued QNode"""
        if diff_method in {"adjoint", "spsa"} or dev.name == "param_shift.qubit":
            pytest.skip("Adjoint and SPSA do not support second derivative.")

        options = {}
        if diff_method == "finite-diff":
            options = {"h": 1e-6}

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
            device_vjp=device_vjp,
            gradient_kwargs=options,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = torch.tensor([1.0, 2.0], requires_grad=True, dtype=torch.float64)
        res = circuit(x)

        res.backward()
        g = x.grad

        hess = hessian(circuit, x)
        a, b = x.detach().numpy()

        assert isinstance(hess, torch.Tensor)
        assert tuple(hess.shape) == (2, 2)

        expected_res = np.cos(a) * np.cos(b)
        assert np.allclose(res.detach(), expected_res, atol=tol, rtol=0)

        expected_g = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]
        assert np.allclose(g.detach(), expected_g, atol=tol, rtol=0)

        expected_hess = [
            [-np.cos(a) * np.cos(b), np.sin(a) * np.sin(b)],
            [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)],
        ]

        if diff_method == "finite-diff":
            assert np.allclose(hess.detach(), expected_hess, atol=10e-2, rtol=0)
        else:
            assert np.allclose(hess.detach(), expected_hess, atol=tol, rtol=0)

    def test_hessian_vector_valued(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol
    ):
        """Test hessian calculation of a vector valued QNode"""
        if diff_method in {"adjoint", "spsa"} or dev.name == "param_shift.qubit":
            pytest.skip("Adjoint and SPSA do not support second derivative.")

        options = {}
        if diff_method == "finite-diff":
            options = {"h": 1e-6}

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
            device_vjp=device_vjp,
            gradient_kwargs=options,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.probs(wires=0)

        x = torch.tensor([1.0, 2.0], requires_grad=True, dtype=torch.float64)
        res = circuit(x)
        jac_fn = lambda x: jacobian(circuit, x, create_graph=True)

        g = jac_fn(x)
        hess = jacobian(jac_fn, x)

        a, b = x.detach().numpy()

        assert isinstance(hess, torch.Tensor)
        assert tuple(hess.shape) == (2, 2, 2)

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
        if diff_method == "finite-diff":
            assert np.allclose(hess.detach(), expected_hess, atol=10e-2, rtol=0)
        else:
            assert np.allclose(hess.detach(), expected_hess, atol=tol, rtol=0)

    def test_hessian_ragged(self, interface, dev, diff_method, grad_on_execution, device_vjp, tol):
        """Test hessian calculation of a ragged QNode"""
        if diff_method in {"adjoint", "spsa"} or dev.name == "param_shift.qubit":
            pytest.skip("Adjoint and SPSA do not support second derivative.")

        options = {}
        if diff_method == "finite-diff":
            options = {"h": 1e-6}

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
            device_vjp=device_vjp,
            gradient_kwargs=options,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            qml.RY(x[0], wires=1)
            qml.RX(x[1], wires=1)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=1)

        def circuit_stack(x):
            return torch.hstack(circuit(x))

        x = torch.tensor([1.0, 2.0], requires_grad=True, dtype=torch.float64)
        res = circuit_stack(x)

        jac_fn = lambda x: jacobian(circuit_stack, x, create_graph=True)

        g = jac_fn(x)
        hess = jacobian(jac_fn, x)
        a, b = x.detach().numpy()

        assert isinstance(hess, torch.Tensor)
        assert tuple(hess.shape) == (3, 2, 2)

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
        if diff_method == "finite-diff":
            assert np.allclose(hess.detach(), expected_hess, atol=10e-2, rtol=0)
        else:
            assert np.allclose(hess.detach(), expected_hess, atol=tol, rtol=0)

    def test_hessian_vector_valued_postprocessing(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol
    ):
        """Test hessian calculation of a vector valued QNode with post-processing"""
        if diff_method in {"adjoint", "spsa"} or dev.name == "param_shift.qubit":
            pytest.skip("Adjoint and SPSA do not support second derivative.")

        options = {}
        if diff_method == "finite-diff":
            options = {"h": 1e-6}

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
            device_vjp=device_vjp,
            gradient_kwargs=options,
        )
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0))

        x = torch.tensor([0.76, -0.87], requires_grad=True, dtype=torch.float64)

        def cost_fn(x):
            return x @ torch.hstack(circuit(x))

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

        hess = hessian(cost_fn, x)
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

        if diff_method == "finite-diff":
            assert np.allclose(hess.detach(), expected_hess, atol=10e-2, rtol=0)
        else:
            assert np.allclose(hess.detach(), expected_hess, atol=tol, rtol=0)

    def test_state(self, interface, dev, diff_method, grad_on_execution, device_vjp, tol):
        """Test that the state can be returned and differentiated"""

        if dev.name == "param_shift.qubit":
            pytest.skip("parameter shift does not support measuring the state.")
        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("Lightning devices do not support state with adjoint diff.")

        x = torch.tensor(0.543, requires_grad=True)
        y = torch.tensor(-0.654, requires_grad=True)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.state()

        def cost_fn(x, y):
            res = circuit(x, y)
            assert torch.is_complex(res)
            probs = torch.abs(res) ** 2
            return probs[0] + probs[2]

        res = cost_fn(x, y)

        if diff_method not in {"backprop"}:
            pytest.skip("Test only supports backprop")

        res.backward()
        res = torch.tensor([x.grad, y.grad])
        expected = torch.tensor(
            [-torch.sin(x) * torch.cos(y) / 2, -torch.cos(x) * torch.sin(y) / 2]
        )
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("state", [[1], [0, 1]])  # Basis state and state vector
    def test_projector(
        self, state, interface, dev, diff_method, grad_on_execution, device_vjp, tol, seed
    ):
        """Test that the variance of a projector is correctly returned"""
        kwargs = dict(
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("adjoint supports either all expvals or all diagonal measurements")
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA
        elif diff_method == "hadamard":
            pytest.skip("Hadamard does not support variances.")
        if dev.name == "reference.qubit":
            pytest.xfail("diagonalize_measurements do not support projectors (sc-72911)")

        P = torch.tensor(state, requires_grad=False)

        x, y = 0.765, -0.654
        weights = torch.tensor([x, y], requires_grad=True, dtype=torch.float64)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.Projector(P, wires=0) @ qml.PauliX(1))

        res = circuit(*weights)
        expected = 0.25 * np.sin(x / 2) ** 2 * (3 + np.cos(2 * y) + 2 * np.cos(x) * np.sin(y) ** 2)
        assert np.allclose(res.detach(), expected, atol=tol, rtol=0)

        res.backward()
        expected = np.array(
            [
                [
                    0.5 * np.sin(x) * (np.cos(x / 2) ** 2 + np.cos(2 * y) * np.sin(x / 2) ** 2),
                    -2 * np.cos(y) * np.sin(x / 2) ** 4 * np.sin(y),
                ]
            ]
        )
        assert np.allclose(weights.grad.detach(), expected, atol=tol, rtol=0)

    def test_postselection_differentiation(
        self, interface, dev, diff_method, grad_on_execution, device_vjp
    ):
        """Test that when postselecting with default.qubit, differentiation works correctly."""

        if diff_method in ["adjoint", "spsa", "hadamard"]:
            pytest.skip("Diff method does not support postselection.")

        if dev.name == "reference.qubit":
            pytest.skip("reference.qubit does not support postselection.")

        @qml.qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(phi, theta):
            qml.RX(phi, wires=0)
            qml.CNOT([0, 1])
            qml.measure(wires=0, postselect=1)
            qml.RX(theta, wires=1)
            return qml.expval(qml.PauliZ(1))

        @qml.qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def expected_circuit(theta):
            qml.PauliX(1)
            qml.RX(theta, wires=1)
            return qml.expval(qml.PauliZ(1))

        phi = torch.tensor(1.23, requires_grad=True)
        theta = torch.tensor(4.56, requires_grad=True)

        assert qml.math.allclose(circuit(phi, theta), expected_circuit(theta))

        gradient = torch.autograd.grad(circuit(phi, theta), [phi, theta])
        exp_theta_grad = torch.autograd.grad(expected_circuit(theta), theta)[0]
        assert qml.math.allclose(gradient, [0.0, exp_theta_grad])


@pytest.mark.parametrize(
    "dev,diff_method,grad_on_execution, device_vjp", qubit_device_and_diff_method
)
class TestTapeExpansion:
    """Test that tape expansion within the QNode integrates correctly
    with the Torch interface"""

    def test_gradient_expansion(self, dev, diff_method, grad_on_execution, device_vjp):
        """Test that a *supported* operation with no gradient recipe is
        expanded for both parameter-shift and finite-differences, but not for execution."""
        if diff_method not in ("parameter-shift", "finite-diff", "spsa", "hadamard"):
            pytest.skip("Only supports gradient transforms")

        class PhaseShift(qml.PhaseShift):  # pylint:disable=too-few-public-methods
            grad_method = None
            has_generator = False

            def decomposition(self):
                return [qml.RY(3 * self.data[0], wires=self.wires)]

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            device_vjp=device_vjp,
            interface="torch",
        )
        def circuit(x):
            qml.Hadamard(wires=0)
            PhaseShift(x, wires=0)
            return qml.expval(qml.PauliX(0))

        x = torch.tensor(0.5, requires_grad=True, dtype=torch.float64)

        loss = circuit(x)

        loss.backward()
        res = x.grad

        assert torch.allclose(res, -3 * torch.sin(3 * x))

        if diff_method == "parameter-shift" and dev.name != "param_shift.qubit":
            # test second order derivatives
            res = torch.autograd.functional.hessian(circuit, x)
            assert torch.allclose(res, -9 * torch.cos(3 * x))

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_gradient_expansion_trainable_only(
        self,
        dev,
        diff_method,
        grad_on_execution,
        device_vjp,
        max_diff,
    ):
        """Test that a *supported* operation with no gradient recipe is only
        expanded for parameter-shift and finite-differences when it is trainable."""
        if diff_method not in ("parameter-shift", "finite-diff", "spsa", "hadamard"):
            pytest.skip("Only supports gradient transforms")

        class PhaseShift(qml.PhaseShift):  # pylint:disable=too-few-public-methods
            grad_method = None

            def decomposition(self):
                return [qml.RY(3 * self.data[0], wires=self.wires)]

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            interface="torch",
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.Hadamard(wires=0)
            PhaseShift(x, wires=0)
            PhaseShift(2 * y, wires=0)
            return qml.expval(qml.PauliX(0))

        x = torch.tensor(0.5, requires_grad=True)
        y = torch.tensor(0.7, requires_grad=False)

        loss = circuit(x, y)
        loss.backward()

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_hamiltonian_expansion_analytic(
        self, dev, diff_method, grad_on_execution, max_diff, device_vjp, tol, seed
    ):
        """Test that if there
        are non-commuting groups and the number of shots is None
        the first and second order gradients are correctly evaluated"""
        kwargs = dict(
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            interface="torch",
            device_vjp=device_vjp,
        )
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not yet support Hamiltonians")
        elif diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA
        elif diff_method == "hadamard":
            pytest.skip("The hadamard method does not yet support Hamiltonians")

        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(data, weights, coeffs):
            weights = torch.reshape(weights, [1, -1])
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            qml.templates.BasicEntanglerLayers(weights, wires=[0, 1])
            return qml.expval(qml.Hamiltonian(coeffs, obs))

        d = torch.tensor([0.1, 0.2], requires_grad=False, dtype=torch.float64)
        w = torch.tensor([0.654, -0.734], requires_grad=True, dtype=torch.float64)
        c = torch.tensor([-0.6543, 0.24, 0.54], requires_grad=True, dtype=torch.float64)

        # test output
        res = circuit(d, w, c)

        expected = c[2] * torch.cos(d[1] + w[1]) - c[1] * torch.sin(d[0] + w[0]) * torch.sin(
            d[1] + w[1]
        )
        assert torch.allclose(res, expected, atol=tol)

        # test gradients
        res.backward()
        grad = (w.grad, c.grad)

        expected_w = torch.tensor(
            [
                -c[1] * torch.cos(d[0] + w[0]) * torch.sin(d[1] + w[1]),
                -c[1] * torch.cos(d[1] + w[1]) * torch.sin(d[0] + w[0])
                - c[2] * torch.sin(d[1] + w[1]),
            ]
        )
        expected_c = torch.tensor(
            [0, -torch.sin(d[0] + w[0]) * torch.sin(d[1] + w[1]), torch.cos(d[1] + w[1])]
        )
        assert torch.allclose(grad[0], expected_w, atol=tol)
        assert torch.allclose(grad[1], expected_c, atol=tol)

        # test second-order derivatives
        if (
            diff_method in ("parameter-shift", "backprop")
            and max_diff == 2
            and dev.name != "param_shift.qubit"
        ):
            hessians = torch.autograd.functional.hessian(circuit, (d, w, c))

            grad2_c = hessians[2][2]
            assert torch.allclose(grad2_c, torch.zeros([3, 3], dtype=torch.float64), atol=tol)

            grad2_w_c = hessians[1][2]
            expected = torch.tensor(
                [
                    [0, -torch.cos(d[0] + w[0]) * torch.sin(d[1] + w[1]), 0],
                    [
                        0,
                        -torch.cos(d[1] + w[1]) * torch.sin(d[0] + w[0]),
                        -torch.sin(d[1] + w[1]),
                    ],
                ]
            )
            assert torch.allclose(grad2_w_c, expected, atol=tol)

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_hamiltonian_finite_shots(
        self, dev, diff_method, device_vjp, grad_on_execution, max_diff, seed
    ):
        """Test that the Hamiltonian is correctly measured if there
        are non-commuting groups and the number of shots is finite
        and the first and second order gradients are correctly evaluated"""
        gradient_kwargs = {}
        tol = 0.1
        if diff_method in ("adjoint", "backprop"):
            pytest.skip("The adjoint and backprop methods do not yet support sampling")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "sampler_rng": np.random.default_rng(seed),
                "num_directions": 20,
            }
            tol = TOL_FOR_SPSA
        elif diff_method == "finite-diff":
            gradient_kwargs = {"h": 0.05}
            tol = 0.15
        elif diff_method == "hadamard":
            pytest.skip("The hadamard method does not yet support Hamiltonians")

        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            interface="torch",
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(data, weights, coeffs):
            weights = torch.reshape(weights, [1, -1])
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            qml.templates.BasicEntanglerLayers(weights, wires=[0, 1])
            H = qml.Hamiltonian(coeffs, obs)
            return qml.expval(H)

        d = torch.tensor([0.1, 0.2], requires_grad=False, dtype=torch.float64)
        w = torch.tensor([0.654, -0.734], requires_grad=True, dtype=torch.float64)
        c = torch.tensor([-0.6543, 0.24, 0.54], requires_grad=True, dtype=torch.float64)

        # wrap circuit with fixed shots
        circuit_shots = qml.set_shots(shots=50000)(circuit)

        # test output
        res = circuit_shots(d, w, c)

        expected = c[2] * torch.cos(d[1] + w[1]) - c[1] * torch.sin(d[0] + w[0]) * torch.sin(
            d[1] + w[1]
        )
        assert torch.allclose(res, expected, atol=tol)

        # test gradients
        if diff_method in ["finite-diff", "spsa"]:
            pytest.skip(f"{diff_method} not compatible")

        res.backward()
        grad = (w.grad, c.grad)

        expected_w = torch.tensor(
            [
                -c[1] * torch.cos(d[0] + w[0]) * torch.sin(d[1] + w[1]),
                -c[1] * torch.cos(d[1] + w[1]) * torch.sin(d[0] + w[0])
                - c[2] * torch.sin(d[1] + w[1]),
            ]
        )
        expected_c = torch.tensor(
            [0, -torch.sin(d[0] + w[0]) * torch.sin(d[1] + w[1]), torch.cos(d[1] + w[1])]
        )

        assert torch.allclose(grad[0], expected_w, atol=tol)
        assert torch.allclose(grad[1], expected_c, atol=tol)

        # test second-order derivatives
        if diff_method == "parameter-shift" and max_diff == 2 and dev.name != "param_shift.qubit":
            hessians = torch.autograd.functional.hessian(circuit_shots, (d, w, c))

            grad2_c = hessians[2][2]
            assert torch.allclose(grad2_c, torch.zeros([3, 3], dtype=torch.float64), atol=tol)

            grad2_w_c = hessians[1][2]
            expected = torch.tensor(
                [
                    [0, -torch.cos(d[0] + w[0]) * torch.sin(d[1] + w[1]), 0],
                    [
                        0,
                        -torch.cos(d[1] + w[1]) * torch.sin(d[0] + w[0]),
                        -torch.sin(d[1] + w[1]),
                    ],
                ]
            )
            assert torch.allclose(grad2_w_c, expected, atol=tol)


class TestSample:
    """Tests for the sample integration"""

    def test_sample_dimension(self):
        """Test sampling works as expected"""

        @qml.set_shots(shots=10)
        @qnode(DefaultQubit(), diff_method="parameter-shift", interface="torch")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        res = circuit()

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert tuple(res[0].shape) == (10,)
        assert isinstance(res[0], torch.Tensor)

        assert tuple(res[1].shape) == (10,)
        assert isinstance(res[1], torch.Tensor)

    def test_sampling_expval(self):
        """Test sampling works as expected if combined with expectation values"""

        @qml.set_shots(shots=10)
        @qnode(DefaultQubit(), diff_method="parameter-shift", interface="torch")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        res = circuit()

        assert len(res) == 2
        assert isinstance(res, tuple)

        assert isinstance(res[0], torch.Tensor)
        assert res[0].shape == (10,)
        assert isinstance(res[1], torch.Tensor)
        assert res[1].shape == ()

    def test_counts_expval(self):
        """Test counts works as expected if combined with expectation values"""

        @qml.set_shots(shots=10)
        @qnode(qml.device("default.qubit"), diff_method="parameter-shift", interface="torch")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.counts(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        res = circuit()

        assert len(res) == 2
        assert isinstance(res, tuple)

        assert isinstance(res[0], dict)
        assert isinstance(res[1], torch.Tensor)
        assert res[1].shape == ()

    def test_sample_combination(self):
        """Test the output of combining expval, var and sample"""

        @qml.set_shots(shots=10)
        @qnode(DefaultQubit(), diff_method="parameter-shift", interface="torch")
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1)), qml.var(qml.PauliY(2))

        result = circuit()

        assert isinstance(result, tuple)
        assert len(result) == 3

        assert np.array_equal(result[0].shape, (10,))
        assert result[1].shape == ()
        assert isinstance(result[1], torch.Tensor)
        assert result[2].shape == ()
        assert isinstance(result[2], torch.Tensor)
        assert result[0].dtype is torch.float64

    def test_single_wire_sample(self):
        """Test the return type and shape of sampling a single wire"""

        @qnode(DefaultQubit(), diff_method="parameter-shift", interface="torch")
        def circuit():
            qml.RX(0.54, wires=0)
            return qml.sample(qml.PauliZ(0))

        circuit10 = qml.set_shots(shots=10)(circuit)
        result = circuit10()

        assert isinstance(result, torch.Tensor)
        assert np.array_equal(result.shape, (10,))

    def test_multi_wire_sample_regular_shape(self):
        """Test the return type and shape of sampling multiple wires
        where a rectangular array is expected"""

        @qml.set_shots(shots=10)
        @qnode(DefaultQubit(), diff_method="parameter-shift", interface="torch")
        def circuit():
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1)), qml.sample(qml.PauliZ(2))

        result = circuit()

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert isinstance(result, tuple)
        assert tuple(result[0].shape) == (10,)
        assert tuple(result[1].shape) == (10,)
        assert tuple(result[2].shape) == (10,)
        assert result[0].dtype == torch.float64
        assert result[1].dtype == torch.float64
        assert result[2].dtype == torch.float64


qubit_device_and_diff_method_and_grad_on_execution = [
    [DefaultQubit(), "backprop", True, False],
    [DefaultQubit(), "finite-diff", False, False],
    [DefaultQubit(), "parameter-shift", False, False],
    [DefaultQubit(), "adjoint", True, False],
    [DefaultQubit(), "adjoint", False, False],
    [DefaultQubit(), "adjoint", True, True],
    [DefaultQubit(), "adjoint", False, True],
    [DefaultQubit(), "hadamard", False, False],
]


@pytest.mark.parametrize(
    "dev,diff_method,grad_on_execution, device_vjp",
    qubit_device_and_diff_method_and_grad_on_execution,
)
@pytest.mark.parametrize("shots", [None, 10000])
class TestReturn:
    """Class to test the shape of the Grad/Jacobian/Hessian with different return types."""

    # pylint:disable=too-many-public-methods

    def test_grad_single_measurement_param(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """For one measurement and one param, the gradient is a float."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = torch.tensor(0.1, requires_grad=True)

        circuit_shots = qml.set_shots(shots=shots)(circuit)
        res = circuit_shots(a)

        assert isinstance(res, torch.Tensor)
        assert res.shape == ()
        # gradient
        res.backward()
        grad = a.grad

        assert isinstance(grad, torch.Tensor)
        assert grad.shape == ()

    def test_grad_single_measurement_multiple_param(
        self, dev, diff_method, grad_on_execution, shots, device_vjp
    ):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=True)

        circuit_shots = qml.set_shots(shots=shots)(circuit)
        res = circuit_shots(a, b)

        # gradient
        res.backward()
        grad_a = a.grad
        grad_b = b.grad

        assert grad_a.shape == ()
        assert grad_b.shape == ()

    def test_grad_single_measurement_multiple_param_array(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """For one measurement and multiple param as a single array params, the gradient is an array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            device_vjp=device_vjp,
            grad_on_execution=grad_on_execution,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        a = torch.tensor([0.1, 0.2], requires_grad=True)

        circuit_shots = qml.set_shots(shots=shots)(circuit)
        res = circuit_shots(a)

        # gradient
        res.backward()
        grad = a.grad

        assert isinstance(grad, torch.Tensor)
        assert grad.shape == (2,)

    def test_jacobian_single_measurement_param_probs(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned with the correct
        dimension"""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            device_vjp=device_vjp,
            grad_on_execution=grad_on_execution,
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.probs(wires=[0, 1])

        a = torch.tensor(0.1, requires_grad=True)

        jac = jacobian(circuit, a)

        assert isinstance(jac, torch.Tensor)
        assert jac.shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            device_vjp=device_vjp,
            grad_on_execution=grad_on_execution,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=[0, 1])

        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=True)

        circuit_shots = qml.set_shots(shots=shots)(circuit)
        jac = jacobian(circuit_shots, (a, b))

        assert isinstance(jac, tuple)

        assert isinstance(jac[0], torch.Tensor)
        assert jac[0].shape == (4,)

        assert isinstance(jac[1], torch.Tensor)
        assert jac[1].shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param_single_array(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qml.set_shots(shots=shots)
        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            device_vjp=device_vjp,
            grad_on_execution=grad_on_execution,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.probs(wires=[0, 1])

        a = torch.tensor([0.1, 0.2], requires_grad=True)
        jac = jacobian(circuit, a)

        assert isinstance(jac, torch.Tensor)
        assert jac.shape == (4, 2)

    def test_jacobian_expval_expval_multiple_params(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par_0 = torch.tensor(0.1, requires_grad=True)
        par_1 = torch.tensor(0.2, requires_grad=True)

        @qml.set_shots(shots=shots)
        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            max_diff=1,
            device_vjp=device_vjp,
            grad_on_execution=grad_on_execution,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        jac = jacobian(circuit, (par_0, par_1))

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

    def test_jacobian_expval_expval_multiple_params_array(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qml.set_shots(shots=shots)
        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            device_vjp=device_vjp,
            grad_on_execution=grad_on_execution,
        )
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

    def test_jacobian_var_var_multiple_params(
        self, dev, diff_method, device_vjp, grad_on_execution, shots
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of var.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports Hadamard because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par_0 = torch.tensor(0.1, requires_grad=True)
        par_1 = torch.tensor(0.2, requires_grad=True)

        @qml.set_shots(shots=shots)
        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            device_vjp=device_vjp,
            max_diff=1,
            grad_on_execution=grad_on_execution,
        )
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

    def test_jacobian_var_var_multiple_params_array(
        self, dev, diff_method, device_vjp, grad_on_execution, shots
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of var.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports Hadamard because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qml.set_shots(shots=shots)
        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            device_vjp=device_vjp,
            grad_on_execution=grad_on_execution,
        )
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

    def test_jacobian_multiple_measurement_single_param(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """The jacobian of multiple measurements with a single params return an array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        @qml.set_shots(shots=shots)
        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            device_vjp=device_vjp,
            grad_on_execution=grad_on_execution,
        )
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

    def test_jacobian_multiple_measurement_multiple_param(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qml.set_shots(shots=shots)
        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            device_vjp=device_vjp,
            grad_on_execution=grad_on_execution,
        )
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
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qml.set_shots(shots=shots)
        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
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

    def test_hessian_expval_multiple_params(
        self, dev, diff_method, grad_on_execution, shots, device_vjp
    ):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        par_0 = torch.tensor(0.1, requires_grad=True, dtype=torch.float64)
        par_1 = torch.tensor(0.2, requires_grad=True, dtype=torch.float64)

        @qml.set_shots(shots=shots)
        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        hess = hessian(circuit, (par_0, par_1))

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], torch.Tensor)
        assert isinstance(hess[0][1], torch.Tensor)
        assert hess[0][0].shape == ()
        assert hess[0][1].shape == ()

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], torch.Tensor)
        assert isinstance(hess[1][1], torch.Tensor)
        assert hess[1][0].shape == ()
        assert hess[1][1].shape == ()

    def test_hessian_expval_multiple_param_array(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """The hessian of single measurement with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        params = torch.tensor([0.1, 0.2], requires_grad=True)

        @qml.set_shots(shots=shots)
        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        hess = hessian(circuit, params)

        assert isinstance(hess, torch.Tensor)
        assert hess.shape == (2, 2)

    def test_hessian_var_multiple_params(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """The hessian of a single measurement with multiple params returns a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports Hadamard because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par_0 = torch.tensor(0.1, requires_grad=True, dtype=torch.float64)
        par_1 = torch.tensor(0.2, requires_grad=True, dtype=torch.float64)

        @qml.set_shots(shots=shots)
        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
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
        assert isinstance(hess[0][1], torch.Tensor)
        assert hess[0][1].shape == ()

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], torch.Tensor)
        assert hess[1][0].shape == ()
        assert isinstance(hess[1][1], torch.Tensor)
        assert hess[1][1].shape == ()

    def test_hessian_var_multiple_param_array(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """The hessian of single measurement with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports Hadamard because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        params = torch.tensor([0.1, 0.2], requires_grad=True, dtype=torch.float64)

        @qml.set_shots(shots=shots)
        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        hess = hessian(circuit, params)

        assert isinstance(hess, torch.Tensor)
        assert hess.shape == (2, 2)

    def test_hessian_probs_expval_multiple_params(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports non commuting measurement.")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par_0 = torch.tensor(0.1, requires_grad=True, dtype=torch.float64)
        par_1 = torch.tensor(0.2, requires_grad=True, dtype=torch.float64)

        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        def circuit_stack(x, y):
            return torch.hstack(qml.set_shots(shots=shots)(circuit)(x, y))

        jac_fn = lambda x, y: jacobian(circuit_stack, (x, y), create_graph=True)

        hess = jacobian(jac_fn, (par_0, par_1))

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], torch.Tensor)
        assert tuple(hess[0][0].shape) == (3,)
        assert isinstance(hess[0][1], torch.Tensor)
        assert tuple(hess[0][1].shape) == (3,)

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], torch.Tensor)
        assert tuple(hess[1][0].shape) == (3,)
        assert isinstance(hess[1][1], torch.Tensor)
        assert tuple(hess[1][1].shape) == (3,)

    def test_hessian_expval_probs_multiple_param_array(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports non commuting measurement.")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par = torch.tensor([0.1, 0.2], requires_grad=True, dtype=torch.float64)

        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        def circuit_stack(x):
            return torch.hstack(qml.set_shots(shots=shots)(circuit)(x))

        jac_fn = lambda x: jacobian(circuit_stack, x, create_graph=True)

        hess = jacobian(jac_fn, par)

        assert isinstance(hess, torch.Tensor)
        assert tuple(hess.shape) == (3, 2, 2)

    def test_hessian_probs_var_multiple_params(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports Hadamard because of var.")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par_0 = torch.tensor(0.1, requires_grad=True)
        par_1 = torch.tensor(0.2, requires_grad=True)

        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        def circuit_stack(x, y):
            return torch.hstack(qml.set_shots(shots=shots)(circuit)(x, y))

        jac_fn = lambda x, y: jacobian(circuit_stack, (x, y), create_graph=True)

        hess = jacobian(jac_fn, (par_0, par_1))

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], torch.Tensor)
        assert tuple(hess[0][0].shape) == (3,)
        assert isinstance(hess[0][1], torch.Tensor)
        assert tuple(hess[0][1].shape) == (3,)

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], torch.Tensor)
        assert tuple(hess[1][0].shape) == (3,)
        assert isinstance(hess[1][1], torch.Tensor)
        assert tuple(hess[1][1].shape) == (3,)

    def test_hessian_var_probs_multiple_param_array(
        self, dev, diff_method, grad_on_execution, device_vjp, shots
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports Hadamard because of var.")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par = torch.tensor([0.1, 0.2], requires_grad=True)

        @qnode(
            dev,
            interface="torch",
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        def circuit_stack(x):
            return torch.hstack(qml.set_shots(shots=shots)(circuit)(x))

        jac_fn = lambda x: jacobian(circuit_stack, x, create_graph=True)

        hess = jacobian(jac_fn, par)

        assert isinstance(hess, torch.Tensor)
        assert tuple(hess.shape) == (3, 2, 2)


def test_no_ops():
    """Test that the return value of the QNode matches in the interface
    even if there are no ops"""

    @qml.qnode(DefaultQubit(), interface="torch")
    def circuit():
        qml.Hadamard(wires=0)
        return qml.state()

    res = circuit()
    assert isinstance(res, torch.Tensor)
