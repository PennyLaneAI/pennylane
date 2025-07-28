# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for using the autograd interface with a QNode"""
# pylint: disable=no-member, too-many-arguments, unexpected-keyword-arg, use-dict-literal, no-name-in-module


import autograd
import autograd.numpy as anp
import pytest
from param_shift_dev import ParamShiftDerivativesDevice

import pennylane as qml
from pennylane import numpy as np
from pennylane import qnode
from pennylane.devices import DefaultQubit
from pennylane.exceptions import DeviceError, QuantumFunctionError

# dev, diff_method, grad_on_execution, device_vjp
qubit_device_and_diff_method = [
    [qml.device("default.qubit"), "finite-diff", False, False],
    [qml.device("default.qubit"), "parameter-shift", False, False],
    [qml.device("default.qubit"), "backprop", True, False],
    [qml.device("default.qubit"), "adjoint", True, False],
    [qml.device("default.qubit"), "adjoint", False, False],
    [qml.device("default.qubit"), "spsa", False, False],
    [qml.device("default.qubit"), "hadamard", False, False],
    [ParamShiftDerivativesDevice(), "parameter-shift", False, False],
    [ParamShiftDerivativesDevice(), "best", False, False],
    [ParamShiftDerivativesDevice(), "parameter-shift", True, False],
    [ParamShiftDerivativesDevice(), "parameter-shift", False, True],
    [qml.device("reference.qubit"), "parameter-shift", False, False],
]

interface_qubit_device_and_diff_method = [
    ["autograd", DefaultQubit(), "finite-diff", False, False],
    ["autograd", DefaultQubit(), "parameter-shift", False, False],
    ["autograd", DefaultQubit(), "backprop", True, False],
    ["autograd", DefaultQubit(), "adjoint", True, False],
    ["autograd", DefaultQubit(), "adjoint", False, False],
    ["autograd", DefaultQubit(), "adjoint", True, True],
    ["autograd", DefaultQubit(), "adjoint", False, True],
    ["autograd", DefaultQubit(), "spsa", False, False],
    ["autograd", DefaultQubit(), "hadamard", False, False],
    ["autograd", qml.device("lightning.qubit", wires=5), "adjoint", False, True],
    ["autograd", qml.device("lightning.qubit", wires=5), "adjoint", True, True],
    ["autograd", qml.device("lightning.qubit", wires=5), "adjoint", False, False],
    ["autograd", qml.device("lightning.qubit", wires=5), "adjoint", True, False],
    ["auto", DefaultQubit(), "finite-diff", False, False],
    ["auto", DefaultQubit(), "parameter-shift", False, False],
    ["auto", DefaultQubit(), "backprop", True, False],
    ["auto", DefaultQubit(), "adjoint", True, False],
    ["auto", DefaultQubit(), "adjoint", False, False],
    ["auto", DefaultQubit(), "spsa", False, False],
    ["auto", DefaultQubit(), "hadamard", False, False],
    ["auto", qml.device("lightning.qubit", wires=5), "adjoint", False, False],
    ["auto", qml.device("lightning.qubit", wires=5), "adjoint", True, False],
    ["auto", qml.device("reference.qubit"), "parameter-shift", False, False],
]

pytestmark = pytest.mark.autograd

TOL_FOR_SPSA = 1.0
H_FOR_SPSA = 0.01


@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution, device_vjp",
    interface_qubit_device_and_diff_method,
)
class TestQNode:
    """Test that using the QNode with Autograd integrates with the PennyLane stack"""

    def test_execution_with_interface(
        self, interface, dev, diff_method, grad_on_execution, device_vjp
    ):
        """Test execution works with the interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array(0.1, requires_grad=True)
        assert circuit.interface == interface
        # gradients should work
        grad = qml.grad(circuit)(a)

        assert grad.shape == tuple()

    def test_jacobian(self, interface, dev, diff_method, grad_on_execution, tol, device_vjp, seed):
        """Test jacobian calculation"""
        kwargs = dict(
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )

        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        res = circuit(a, b)

        def cost(x, y):
            return autograd.numpy.hstack(circuit(x, y))

        assert isinstance(res, tuple)
        assert len(res) == 2

        expected = [np.cos(a), -np.cos(a) * np.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.jacobian(cost)(a, b)
        assert isinstance(res, tuple) and len(res) == 2
        expected = ([-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)])
        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == (2,)
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == (2,)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_jacobian_no_evaluate(
        self, interface, dev, diff_method, grad_on_execution, tol, device_vjp, seed
    ):
        """Test jacobian calculation when no prior circuit evaluation has been performed"""
        kwargs = dict(
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        def cost(x, y):
            return autograd.numpy.hstack(circuit(x, y))

        jac_fn = qml.jacobian(cost)
        res = jac_fn(a, b)
        expected = ([-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)])
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

        # call the Jacobian with new parameters
        a = np.array(0.6, requires_grad=True)
        b = np.array(0.832, requires_grad=True)

        res = jac_fn(a, b)
        expected = ([-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)])
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_jacobian_options(self, interface, dev, diff_method, grad_on_execution, device_vjp):
        """Test setting jacobian options"""
        if diff_method != "finite-diff":
            pytest.skip("Test only supports finite diff.")

        a = np.array([0.1, 0.2], requires_grad=True)

        gradient_kwargs = {"h": 1e-8, "approx_order": 2}

        @qnode(
            dev,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        qml.jacobian(circuit)(a)

    def test_changing_trainability(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol
    ):
        """Test changing the trainability of parameters changes the
        number of differentiation requests made"""
        if diff_method != "parameter-shift":
            pytest.skip("Test only supports parameter-shift")

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        @qnode(
            dev,
            interface=interface,
            diff_method="parameter-shift",
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        def loss(a, b):
            return np.sum(autograd.numpy.hstack(circuit(a, b)))

        grad_fn = qml.grad(loss)
        res = grad_fn(a, b)
        expected = [-np.sin(a) + np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # make the second QNode argument a constant
        a = np.array(0.54, requires_grad=True)
        b = np.array(0.8, requires_grad=False)

        res = grad_fn(a, b)
        expected = [-np.sin(a) + np.sin(a) * np.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # trainability also updates on evaluation
        a = np.array(0.54, requires_grad=False)
        b = np.array(0.8, requires_grad=True)
        res = grad_fn(a, b)
        expected = [-np.cos(a) * np.cos(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_classical_processing(self, interface, dev, diff_method, grad_on_execution, device_vjp):
        """Test classical processing within the quantum tape"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=False)
        c = np.array(0.3, requires_grad=True)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b, c):
            qml.RY(a * c, wires=0)
            qml.RZ(b, wires=0)
            qml.RX(c + c**2 + np.sin(a), wires=0)
            return qml.expval(qml.PauliZ(0))

        res = qml.jacobian(circuit)(a, b, c)

        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

    def test_no_trainable_parameters(
        self, interface, dev, diff_method, grad_on_execution, device_vjp
    ):
        """Test evaluation and Jacobian if there are no trainable parameters"""

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        a = np.array(0.1, requires_grad=False)
        b = np.array(0.2, requires_grad=False)

        res = circuit(a, b)

        assert len(res) == 2
        assert isinstance(res, tuple)

        def cost(x, y):
            return autograd.numpy.hstack(circuit(x, y))

        with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
            assert not qml.jacobian(cost)(a, b)

        def cost2(a, b):
            return np.sum(circuit(a, b))

        with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
            grad = qml.grad(cost2)(a, b)

        assert grad == tuple()

    def test_matrix_parameter(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol
    ):
        """Test that the autograd interface works correctly
        with a matrix parameter"""
        U = np.array([[0, 1], [1, 0]], requires_grad=False)
        a = np.array(0.1, requires_grad=True)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(U, a):
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(U, a)

        res = qml.grad(circuit)(U, a)
        assert np.allclose(res, np.sin(a), atol=tol, rtol=0)

    def test_gradient_non_differentiable_exception(
        self, interface, dev, diff_method, grad_on_execution, device_vjp
    ):
        """Test that an exception is raised if non-differentiable data is
        differentiated"""

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(data1):
            qml.templates.AmplitudeEmbedding(data1, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        grad_fn = qml.grad(circuit, argnum=0)
        data1 = np.array([0, 1, 1, 0], requires_grad=False) / np.sqrt(2)

        with pytest.raises(qml.numpy.NonDifferentiableError, match="is non-differentiable"):
            grad_fn(data1)

    def test_differentiable_expand(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol, seed
    ):
        """Test that operation and nested tape expansion
        is differentiable"""
        kwargs = dict(
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 10
            tol = TOL_FOR_SPSA

        # pylint: disable=too-few-public-methods
        class U3(qml.U3):
            """Custom U3."""

            def decomposition(self):
                theta, phi, lam = self.data
                wires = self.wires
                return [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]

        a = np.array(0.1, requires_grad=False)
        p = np.array([0.1, 0.2, 0.3], requires_grad=True)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(a, p):
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            return qml.expval(qml.PauliX(0))

        res = circuit(a, p)
        expected = np.cos(a) * np.cos(p[1]) * np.sin(p[0]) + np.sin(a) * (
            np.cos(p[2]) * np.sin(p[1]) + np.cos(p[0]) * np.cos(p[1]) * np.sin(p[2])
        )
        # assert isinstance(res, np.ndarray)
        # assert res.shape == ()
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.grad(circuit)(a, p)

        # assert isinstance(res, np.ndarray)
        # assert len(res) == 3

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


class TestShotsIntegration:
    """Test that the QNode correctly changes shot value, and
    remains differentiable."""

    @pytest.mark.xfail(reason="deprecated. To be removed in 0.44")
    def test_changing_shots(self):
        """Test that changing shots works on execution"""
        dev = DefaultQubit()
        a, b = np.array([0.543, -0.654], requires_grad=True)

        @qnode(dev, diff_method=qml.gradients.param_shift)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.sample(wires=(0, 1))

        # execute with device default shots (None)
        with pytest.raises(DeviceError):
            res = circuit(a, b)

        # execute with shots=100
        res = circuit(a, b, shots=100)
        assert res.shape == (100, 2)  # pylint: disable=comparison-with-callable

    @pytest.mark.xfail(reason="Param shift and shot vectors.")
    def test_gradient_integration(self):
        """Test that temporarily setting the shots works
        for gradient computations"""
        dev = DefaultQubit()
        a, b = np.array([0.543, -0.654], requires_grad=True)

        @qnode(dev, diff_method=qml.gradients.param_shift)
        def cost_fn(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        # TODO: fix the shot vectors issue
        # wrap cost_fn with shot vector for gradient computation
        cost_fn_shots = qml.set_shots(shots=[10000, 10000, 10000])(cost_fn)
        res = qml.jacobian(cost_fn_shots)(a, b)
        assert dev.shots is None
        assert isinstance(res, tuple) and len(res) == 2
        assert all(r.shape == (3,) for r in res)

        expected = [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert all(
            np.allclose(np.mean(r, axis=0), e, atol=0.1, rtol=0) for r, e in zip(res, expected)
        )

    def test_update_diff_method(self):
        """Test that temporarily setting the shots updates the diff method"""
        a, b = np.array([0.543, -0.654], requires_grad=True)

        dev = DefaultQubit()

        @qnode(dev)
        def cost_fn(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        with dev.tracker:
            cost_fn100 = qml.set_shots(shots=100)(cost_fn)
            qml.grad(cost_fn100)(a, b)
        # since we are using finite shots, use parameter shift
        assert dev.tracker.totals["executions"] == 5

        # if we use the default shots value of None, backprop can now be used
        with dev.tracker:
            qml.grad(cost_fn)(a, b)
        assert dev.tracker.totals["executions"] == 1


@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution, device_vjp",
    interface_qubit_device_and_diff_method,
)
class TestQubitIntegration:
    """Tests that ensure various qubit circuits integrate correctly"""

    def test_probability_differentiation(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol, seed
    ):
        """Tests correct output shape and evaluation for a tape
        with a single prob output"""
        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("lightning does not support measuring probabilities with adjoint.")

        kwargs = dict(
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA

        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        res = qml.jacobian(circuit)(x, y)
        assert isinstance(res, tuple) and len(res) == 2

        expected = (
            np.array([-np.sin(x) * np.cos(y) / 2, np.cos(y) * np.sin(x) / 2]),
            np.array([-np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2]),
        )
        assert all(np.allclose(r, e, atol=tol, rtol=0) for r, e in zip(res, expected))

    def test_multiple_probability_differentiation(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol, seed
    ):
        """Tests correct output shape and evaluation for a tape
        with multiple prob outputs"""
        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("lightning does not support measuring probabilities with adjoint.")
        kwargs = dict(
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )

        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA

        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0]), qml.probs(wires=[1])

        res = circuit(x, y)

        expected = np.array(
            [
                [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2],
                [(1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        def cost(x, y):
            return autograd.numpy.hstack(circuit(x, y))

        res = qml.jacobian(cost)(x, y)

        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (4,)
        assert res[1].shape == (4,)

        expected = (
            np.array(
                [
                    [
                        -np.sin(x) / 2,
                        np.sin(x) / 2,
                        -np.sin(x) * np.cos(y) / 2,
                        np.sin(x) * np.cos(y) / 2,
                    ],
                ]
            ),
            np.array(
                [
                    [0, 0, -np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2],
                ]
            ),
        )
        assert all(np.allclose(r, e, atol=tol, rtol=0) for r, e in zip(res, expected))

    def test_ragged_differentiation(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol, seed
    ):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("lightning does not support measuring probabilities with adjoint.")

        kwargs = dict(
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )

        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA

        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])

        res = circuit(x, y)
        assert isinstance(res, tuple)
        expected = [np.cos(x), [(1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2]]
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

        def cost(x, y):
            return autograd.numpy.hstack(circuit(x, y))

        res = qml.jacobian(cost)(x, y)
        assert isinstance(res, tuple)
        assert len(res) == 2

        assert res[0].shape == (3,)
        assert res[1].shape == (3,)

        expected = (
            np.array([-np.sin(x), -np.sin(x) * np.cos(y) / 2, np.sin(x) * np.cos(y) / 2]),
            np.array([0, -np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2]),
        )
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_ragged_differentiation_variance(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol, seed
    ):
        """Tests correct output shape and evaluation for a tape
        with prob and variance outputs"""
        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("lightning does not support measuring probabilities with adjoint.")
        kwargs = dict(
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )

        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA
        elif diff_method == "hadamard":
            pytest.skip("Hadamard gradient does not support variances.")

        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0)), qml.probs(wires=[1])

        res = circuit(x, y)

        expected_var = np.array(np.sin(x) ** 2)
        expected_probs = np.array(
            [(1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2]
        )

        assert isinstance(res, tuple)
        assert len(res) == 2

        # assert isinstance(res[0], np.ndarray)
        # assert res[0].shape == ()
        assert np.allclose(res[0], expected_var, atol=tol, rtol=0)

        # assert isinstance(res[1], np.ndarray)
        # assert res[1].shape == (2,)
        assert np.allclose(res[1], expected_probs, atol=tol, rtol=0)

        def cost(x, y):
            return autograd.numpy.hstack(circuit(x, y))

        jac = qml.jacobian(cost)(x, y)
        assert isinstance(res, tuple) and len(res) == 2

        expected = (
            np.array([np.sin(2 * x), -np.sin(x) * np.cos(y) / 2, np.sin(x) * np.cos(y) / 2]),
            np.array([0, -np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2]),
        )

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        # assert isinstance(jac[0], np.ndarray)
        # assert jac[0].shape == (3,)
        assert np.allclose(jac[0], expected[0], atol=tol, rtol=0)

        # assert isinstance(jac[1], np.ndarray)
        # assert jac[1].shape == (3,)
        assert np.allclose(jac[1], expected[1], atol=tol, rtol=0)

    def test_chained_qnodes(self, interface, dev, diff_method, grad_on_execution, device_vjp):
        """Test that the gradient of chained QNodes works without error"""

        # pylint: disable=too-few-public-methods
        class Template(qml.templates.StronglyEntanglingLayers):
            """Custom template."""

            def decomposition(self):
                return [qml.templates.StronglyEntanglingLayers(*self.parameters, self.wires)]

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit1(weights):
            Template(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit2(data, weights):
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            Template(weights, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        def cost(w1, w2):
            c1 = circuit1(w1)
            c2 = circuit2(c1, w2)
            return np.sum(c2) ** 2

        w1 = qml.templates.StronglyEntanglingLayers.shape(n_wires=2, n_layers=3)
        w2 = qml.templates.StronglyEntanglingLayers.shape(n_wires=2, n_layers=4)

        weights = [
            np.random.random(w1, requires_grad=True),
            np.random.random(w2, requires_grad=True),
        ]

        grad_fn = qml.grad(cost)
        res = grad_fn(*weights)

        assert len(res) == 2

    def test_chained_gradient_value(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol, seed
    ):
        """Test that the returned gradient value for two chained qubit QNodes
        is correct."""
        kwargs = dict(
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )

        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA
        dev1 = qml.device("default.qubit")

        @qnode(dev1, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit1(a, b, c):
            qml.RX(a, wires=0)
            qml.RX(b, wires=1)
            qml.RX(c, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(2))

        dev2 = dev

        @qnode(
            dev2, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit2(data, weights):
            qml.RX(data[0], wires=0)
            qml.RX(data[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(weights[0], wires=0)
            qml.RZ(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(0) @ qml.PauliY(1))

        def cost(a, b, c, weights):
            return circuit2(circuit1(a, b, c), weights)

        grad_fn = qml.grad(cost)

        # Set the first parameter of circuit1 as non-differentiable.
        a = np.array(0.4, requires_grad=False)

        # The remaining free parameters are all differentiable.
        b = np.array(0.5, requires_grad=True)
        c = np.array(0.1, requires_grad=True)
        weights = np.array([0.2, 0.3], requires_grad=True)

        res = grad_fn(a, b, c, weights)

        # Output should have shape [dcost/db, dcost/dc, dcost/dw],
        # where b,c are scalars, and w is a vector of length 2.
        assert len(res) == 3
        assert res[0].shape == tuple()  # scalar
        assert res[1].shape == tuple()  # scalar
        assert res[2].shape == (2,)  # vector

        cacbsc = np.cos(a) * np.cos(b) * np.sin(c)

        expected = np.array(
            [
                # analytic expression for dcost/db
                -np.cos(a)
                * np.sin(b)
                * np.sin(c)
                * np.cos(cacbsc)
                * np.sin(weights[0])
                * np.sin(np.cos(a)),
                # analytic expression for dcost/dc
                np.cos(a)
                * np.cos(b)
                * np.cos(c)
                * np.cos(cacbsc)
                * np.sin(weights[0])
                * np.sin(np.cos(a)),
                # analytic expression for dcost/dw[0]
                np.sin(cacbsc) * np.cos(weights[0]) * np.sin(np.cos(a)),
                # analytic expression for dcost/dw[1]
                0,
            ]
        )

        # np.hstack 'flattens' the ragged gradient array allowing it
        # to be compared with the expected result
        assert np.allclose(np.hstack(res), expected, atol=tol, rtol=0)

    def test_second_derivative(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol
    ):
        """Test second derivative calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            device_vjp=device_vjp,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array([1.0, 2.0], requires_grad=True)
        res = circuit(x)
        g = qml.grad(circuit)(x)
        g2 = qml.grad(lambda x: np.sum(qml.grad(circuit)(x)))(x)

        a, b = x

        expected_res = np.cos(a) * np.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        expected_g2 = [
            -np.cos(a) * np.cos(b) + np.sin(a) * np.sin(b),
            np.sin(a) * np.sin(b) - np.cos(a) * np.cos(b),
        ]

        if diff_method in {"finite-diff"}:
            tol = 10e-2

        assert np.allclose(g2, expected_g2, atol=tol, rtol=0)

    def test_hessian(self, interface, dev, diff_method, grad_on_execution, device_vjp, tol):
        """Test hessian calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            max_diff=2,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array([1.0, 2.0], requires_grad=True)
        res = circuit(x)

        a, b = x

        expected_res = np.cos(a) * np.cos(b)

        # assert isinstance(res, np.ndarray)
        # assert res.shape == ()
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        g = grad_fn(x)

        expected_g = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]

        assert isinstance(g, np.ndarray)
        assert g.shape == (2,)
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        hess = qml.jacobian(grad_fn)(x)

        expected_hess = [
            [-np.cos(a) * np.cos(b), np.sin(a) * np.sin(b)],
            [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)],
        ]

        assert isinstance(hess, np.ndarray)
        assert hess.shape == (2, 2)

        if diff_method in {"finite-diff"}:
            tol = 10e-2

        assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_hessian_unused_parameter(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol
    ):
        """Test hessian calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            max_diff=2,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array([1.0, 2.0], requires_grad=True)
        res = circuit(x)

        a, _ = x

        expected_res = np.cos(a)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)

        hess = qml.jacobian(grad_fn)(x)

        expected_hess = [
            [-np.cos(a), 0],
            [0, 0],
        ]

        if diff_method in {"finite-diff"}:
            tol = 10e-2

        assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_hessian_vector_valued(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol
    ):
        """Test hessian calculation of a vector valued QNode"""

        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            max_diff=2,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.probs(wires=0)

        x = np.array([1.0, 2.0], requires_grad=True)
        res = circuit(x)

        a, b = x

        expected_res = [0.5 + 0.5 * np.cos(a) * np.cos(b), 0.5 - 0.5 * np.cos(a) * np.cos(b)]

        assert isinstance(res, np.ndarray)
        assert res.shape == (2,)  # pylint: disable=comparison-with-callable
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        jac_fn = qml.jacobian(circuit)
        jac = jac_fn(x)

        expected_res = [
            [-0.5 * np.sin(a) * np.cos(b), -0.5 * np.cos(a) * np.sin(b)],
            [0.5 * np.sin(a) * np.cos(b), 0.5 * np.cos(a) * np.sin(b)],
        ]

        assert isinstance(jac, np.ndarray)
        assert jac.shape == (2, 2)
        assert np.allclose(jac, expected_res, atol=tol, rtol=0)

        hess = qml.jacobian(jac_fn)(x)

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

        assert isinstance(hess, np.ndarray)
        assert hess.shape == (2, 2, 2)

        if diff_method in {"finite-diff"}:
            tol = 10e-2

        assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_hessian_vector_valued_postprocessing(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol
    ):
        """Test hessian calculation of a vector valued QNode with post-processing"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            max_diff=2,
        )
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0))

        def cost_fn(x):
            return x @ autograd.numpy.hstack(circuit(x))

        x = np.array([0.76, -0.87], requires_grad=True)
        res = cost_fn(x)

        a, b = x

        expected_res = x @ [np.cos(a) * np.cos(b), np.cos(a) * np.cos(b)]

        assert isinstance(res, np.ndarray)
        assert res.shape == ()
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        hess = qml.jacobian(qml.grad(cost_fn))(x)

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

        assert hess.shape == (2, 2)

        if diff_method in {"finite-diff"}:
            tol = 10e-2

        assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_hessian_vector_valued_separate_args(
        self, interface, dev, diff_method, grad_on_execution, device_vjp, tol
    ):
        """Test hessian calculation of a vector valued QNode that has separate input arguments"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            max_diff=2,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=0)

        a = np.array(1.0, requires_grad=True)
        b = np.array(2.0, requires_grad=True)
        res = circuit(a, b)

        expected_res = [0.5 + 0.5 * np.cos(a) * np.cos(b), 0.5 - 0.5 * np.cos(a) * np.cos(b)]
        assert isinstance(res, np.ndarray)
        assert res.shape == (2,)  # pylint: disable=comparison-with-callable
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        jac_fn = qml.jacobian(circuit)
        g = jac_fn(a, b)
        assert isinstance(g, tuple) and len(g) == 2

        expected_g = (
            [-0.5 * np.sin(a) * np.cos(b), 0.5 * np.sin(a) * np.cos(b)],
            [-0.5 * np.cos(a) * np.sin(b), 0.5 * np.cos(a) * np.sin(b)],
        )
        assert g[0].shape == (2,)
        assert np.allclose(g[0], expected_g[0], atol=tol, rtol=0)

        assert g[1].shape == (2,)
        assert np.allclose(g[1], expected_g[1], atol=tol, rtol=0)

        def jac_fn_a(*args):
            return jac_fn(*args)[0]

        def jac_fn_b(*args):
            return jac_fn(*args)[1]

        hess_a = qml.jacobian(jac_fn_a)(a, b)
        hess_b = qml.jacobian(jac_fn_b)(a, b)
        assert isinstance(hess_a, tuple) and len(hess_a) == 2
        assert isinstance(hess_b, tuple) and len(hess_b) == 2

        exp_hess_a = (
            [-0.5 * np.cos(a) * np.cos(b), 0.5 * np.cos(a) * np.cos(b)],
            [0.5 * np.sin(a) * np.sin(b), -0.5 * np.sin(a) * np.sin(b)],
        )
        exp_hess_b = (
            [0.5 * np.sin(a) * np.sin(b), -0.5 * np.sin(a) * np.sin(b)],
            [-0.5 * np.cos(a) * np.cos(b), 0.5 * np.cos(a) * np.cos(b)],
        )

        if diff_method in {"finite-diff"}:
            tol = 10e-2

        for hess, exp_hess in zip([hess_a, hess_b], [exp_hess_a, exp_hess_b]):
            assert np.allclose(hess[0], exp_hess[0], atol=tol, rtol=0)
            assert np.allclose(hess[1], exp_hess[1], atol=tol, rtol=0)

    def test_hessian_ragged(self, interface, dev, diff_method, grad_on_execution, device_vjp, tol):
        """Test hessian calculation of a ragged QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            max_diff=2,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            qml.RY(x[0], wires=1)
            qml.RX(x[1], wires=1)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=1)

        x = np.array([1.0, 2.0], requires_grad=True)

        a, b = x

        expected_res = [
            np.cos(a) * np.cos(b),
            0.5 + 0.5 * np.cos(a) * np.cos(b),
            0.5 - 0.5 * np.cos(a) * np.cos(b),
        ]

        def cost_fn(x):
            return autograd.numpy.hstack(circuit(x))

        res = cost_fn(x)
        assert qml.math.allclose(res, expected_res)

        jac_fn = qml.jacobian(cost_fn)

        hess = qml.jacobian(jac_fn)(x)
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

        if diff_method in {"finite-diff"}:
            tol = 10e-2

        assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_state(self, interface, dev, diff_method, grad_on_execution, device_vjp, tol):
        """Test that the state can be returned and differentiated"""

        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("Lightning does not support state adjoint diff.")

        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

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
            assert res.dtype is np.dtype("complex128")
            probs = np.abs(res) ** 2
            return probs[0] + probs[2]

        res = cost_fn(x, y)

        if diff_method not in {"backprop"}:
            pytest.skip("Test only supports backprop")

        res = qml.jacobian(cost_fn)(x, y)
        expected = np.array([-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2])
        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()
        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == ()
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("state", [[1], [0, 1]])  # Basis state and state vector
    def test_projector(
        self, state, interface, dev, diff_method, grad_on_execution, device_vjp, tol, seed
    ):
        """Test that the variance of a projector is correctly returned"""
        if diff_method == "adjoint":
            pytest.skip("adjoint supports either expvals or diagonal measurements.")
        if dev.name == "reference.qubit":
            pytest.xfail("diagonalize_measurements do not support projectors (sc-72911)")
        kwargs = dict(
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )

        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA
        elif diff_method == "hadamard":
            pytest.skip("Hadamard gradient does not support variances.")

        P = np.array(state, requires_grad=False)
        x, y = np.array([0.765, -0.654], requires_grad=True)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.Projector(P, wires=0) @ qml.PauliX(1))

        res = circuit(x, y)
        expected = 0.25 * np.sin(x / 2) ** 2 * (3 + np.cos(2 * y) + 2 * np.cos(x) * np.sin(y) ** 2)
        # assert isinstance(res, np.ndarray)
        # assert res.shape == ()
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac = qml.jacobian(circuit)(x, y)
        expected = np.array(
            [
                [
                    0.5 * np.sin(x) * (np.cos(x / 2) ** 2 + np.cos(2 * y) * np.sin(x / 2) ** 2),
                    -2 * np.cos(y) * np.sin(x / 2) ** 4 * np.sin(y),
                ]
            ]
        )

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], np.ndarray)
        assert jac[0].shape == ()

        assert isinstance(jac[1], np.ndarray)
        assert jac[1].shape == ()

        assert np.allclose(jac, expected, atol=tol, rtol=0)

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

        phi = np.array(1.23, requires_grad=True)
        theta = np.array(4.56, requires_grad=True)

        assert np.allclose(circuit(phi, theta), expected_circuit(theta))

        gradient = qml.grad(circuit)(phi, theta)
        exp_theta_grad = qml.grad(expected_circuit)(theta)
        assert np.allclose(gradient, [0.0, exp_theta_grad])


@pytest.mark.parametrize(
    "dev,diff_method,grad_on_execution,device_vjp", qubit_device_and_diff_method
)
class TestTapeExpansion:
    """Test that tape expansion within the QNode integrates correctly
    with the Autograd interface"""

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_gradient_expansion_trainable_only(
        self, dev, diff_method, grad_on_execution, max_diff, device_vjp
    ):
        """Test that a *supported* operation with no gradient recipe is only
        expanded for parameter-shift and finite-differences when it is trainable."""
        if diff_method not in ("parameter-shift", "finite-diff", "spsa", "hadamard"):
            pytest.skip("Only supports gradient transforms")
        if max_diff == 2 and diff_method == "hadamard":
            pytest.skip("Max diff > 1 not supported for Hadamard gradient.")

        # pylint: disable=too-few-public-methods
        class PhaseShift(qml.PhaseShift):
            """dummy phase shift."""

            grad_method = None

            def decomposition(self):
                return [qml.RY(3 * self.data[0], wires=self.wires)]

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.Hadamard(wires=0)
            PhaseShift(x, wires=0)
            PhaseShift(2 * y, wires=0)
            return qml.expval(qml.PauliX(0))

        x = np.array(0.5, requires_grad=True)
        y = np.array(0.7, requires_grad=False)
        circuit(x, y)

        _ = qml.grad(circuit)(x, y)

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_hamiltonian_expansion_analytic(
        self, dev, diff_method, grad_on_execution, max_diff, tol, device_vjp, seed
    ):
        """Test that if there are non-commuting groups and the number of shots is None
        the first and second order gradients are correctly evaluated"""
        kwargs = dict(
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            device_vjp=device_vjp,
        )

        gradient_kwargs = {}
        if diff_method in ["adjoint", "hadamard"]:
            pytest.skip("The diff method requested does not yet support Hamiltonians")
        elif diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 10
            tol = TOL_FOR_SPSA

        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(data, weights, coeffs):
            weights = weights.reshape(1, -1)
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            qml.templates.BasicEntanglerLayers(weights, wires=[0, 1])
            return qml.expval(qml.Hamiltonian(coeffs, obs))

        d = np.array([0.1, 0.2], requires_grad=False)
        w = np.array([0.654, -0.734], requires_grad=True)
        c = np.array([-0.6543, 0.24, 0.54], requires_grad=True)

        # test output
        res = circuit(d, w, c)
        expected = c[2] * np.cos(d[1] + w[1]) - c[1] * np.sin(d[0] + w[0]) * np.sin(d[1] + w[1])
        assert np.allclose(res, expected, atol=tol)

        # test gradients
        grad = qml.grad(circuit)(d, w, c)
        expected_w = [
            -c[1] * np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]),
            -c[1] * np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]) - c[2] * np.sin(d[1] + w[1]),
        ]
        expected_c = [0, -np.sin(d[0] + w[0]) * np.sin(d[1] + w[1]), np.cos(d[1] + w[1])]
        assert np.allclose(grad[0], expected_w, atol=tol)
        assert np.allclose(grad[1], expected_c, atol=tol)

        # test second-order derivatives
        if (
            diff_method in ("parameter-shift", "backprop")
            and max_diff == 2
            and dev.name != "param_shift.qubit"
        ):
            if diff_method == "backprop":
                with pytest.warns(UserWarning, match=r"Output seems independent of input."):
                    grad2_c = qml.jacobian(qml.grad(circuit, argnum=2), argnum=2)(d, w, c)
            else:
                grad2_c = qml.jacobian(qml.grad(circuit, argnum=2), argnum=2)(d, w, c)
            assert np.allclose(grad2_c, 0, atol=tol)

            grad2_w_c = qml.jacobian(qml.grad(circuit, argnum=1), argnum=2)(d, w, c)
            expected = [0, -np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]), 0], [
                0,
                -np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]),
                -np.sin(d[1] + w[1]),
            ]
            assert np.allclose(grad2_w_c, expected, atol=tol)

    @pytest.mark.slow
    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_hamiltonian_finite_shots(
        self, dev, diff_method, grad_on_execution, max_diff, device_vjp, seed
    ):
        """Test that the Hamiltonian is correctly measured if there
        are non-commuting groups and the number of shots is finite
        and the first and second order gradients are correctly evaluated"""
        gradient_kwargs = {}
        tol = 0.1
        if diff_method in ("adjoint", "backprop", "hadamard"):
            pytest.skip("The adjoint and backprop methods do not yet support sampling")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "sampler_rng": np.random.default_rng(seed),
                "num_directions": 10,
            }
            tol = TOL_FOR_SPSA
        elif diff_method == "finite-diff":
            gradient_kwargs = {"h": 0.05}

        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qml.set_shots(shots=50000)
        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(data, weights, coeffs):
            weights = weights.reshape(1, -1)
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            qml.templates.BasicEntanglerLayers(weights, wires=[0, 1])
            H = qml.Hamiltonian(coeffs, obs)
            H.compute_grouping()
            return qml.expval(H)

        d = np.array([0.1, 0.2], requires_grad=False)
        w = np.array([0.654, -0.734], requires_grad=True)
        c = np.array([-0.6543, 0.24, 0.54], requires_grad=True)

        # test output
        res = circuit(d, w, c)
        expected = c[2] * np.cos(d[1] + w[1]) - c[1] * np.sin(d[0] + w[0]) * np.sin(d[1] + w[1])
        assert np.allclose(res, expected, atol=tol)

        # test gradients
        if diff_method in ["finite-diff", "spsa"]:
            pytest.skip(f"{diff_method} not compatible")

        grad = qml.grad(circuit)(d, w, c)
        expected_w = [
            -c[1] * np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]),
            -c[1] * np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]) - c[2] * np.sin(d[1] + w[1]),
        ]
        expected_c = [0, -np.sin(d[0] + w[0]) * np.sin(d[1] + w[1]), np.cos(d[1] + w[1])]
        assert np.allclose(grad[0], expected_w, atol=tol)
        assert np.allclose(grad[1], expected_c, atol=tol)

        # test second-order derivatives
        if diff_method == "parameter-shift" and max_diff == 2 and dev.name != "param_shift.qubit":
            grad2_c = qml.jacobian(qml.grad(circuit, argnum=2), argnum=2)(d, w, c)
            assert np.allclose(grad2_c, 0, atol=tol)

            grad2_w_c = qml.jacobian(qml.grad(circuit, argnum=1), argnum=2)(d, w, c)
            expected = [0, -np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]), 0], [
                0,
                -np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]),
                -np.sin(d[1] + w[1]),
            ]
            assert np.allclose(grad2_w_c, expected, atol=tol)


class TestSample:
    """Tests for the sample integration"""

    def test_backprop_error(self):
        """Test that sampling in backpropagation grad_on_execution raises an error"""
        dev = DefaultQubit()

        @qml.set_shots(shots=10)
        @qnode(dev, diff_method="backprop")
        def circuit():
            qml.RX(0.54, wires=0)
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        with pytest.raises(QuantumFunctionError, match="does not support backprop with requested"):
            circuit()

    def test_sample_dimension(self):
        """Test that the sample function outputs samples of the right size"""
        n_sample = 10

        dev = DefaultQubit()

        @qml.set_shots(shots=n_sample)
        @qnode(dev, diff_method=None)
        def circuit():
            qml.RX(0.54, wires=0)
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        res = circuit()

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert res[0].shape == (10,)  # pylint: disable=comparison-with-callable
        assert isinstance(res[0], np.ndarray)

        assert res[1].shape == (10,)  # pylint: disable=comparison-with-callable
        assert isinstance(res[1], np.ndarray)

    def test_sample_combination(self):
        """Test the output of combining expval, var and sample"""

        n_sample = 10

        dev = DefaultQubit()

        @qml.set_shots(shots=n_sample)
        @qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1)), qml.var(qml.PauliY(2))

        result = circuit()

        assert isinstance(result, tuple)
        assert len(result) == 3

        assert np.array_equal(result[0].shape, (n_sample,))
        assert isinstance(result[1], (float, np.ndarray))
        assert isinstance(result[2], (float, np.ndarray))
        assert result[0].dtype == np.dtype("float")

    def test_single_wire_sample(self):
        """Test the return type and shape of sampling a single wire"""
        n_sample = 10

        dev = DefaultQubit()

        @qml.set_shots(shots=n_sample)
        @qnode(dev, diff_method=None)
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0))

        result = circuit()

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result.shape, (n_sample,))

    def test_multi_wire_sample_regular_shape(self):
        """Test the return type and shape of sampling multiple wires
        where a rectangular array is expected"""
        n_sample = 10

        dev = DefaultQubit()

        @qml.set_shots(shots=n_sample)
        @qnode(dev, diff_method=None)
        def circuit():
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1)), qml.sample(qml.PauliZ(2))

        result = circuit()

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert isinstance(result, tuple)
        assert len(result) == 3

        assert result[0].shape == (10,)  # pylint: disable=comparison-with-callable
        assert isinstance(result[0], np.ndarray)

        assert result[1].shape == (10,)  # pylint: disable=comparison-with-callable
        assert isinstance(result[1], np.ndarray)

        assert result[2].shape == (10,)  # pylint: disable=comparison-with-callable
        assert isinstance(result[2], np.ndarray)


@pytest.mark.parametrize(
    "dev,diff_method,grad_on_execution,device_vjp", qubit_device_and_diff_method
)
class TestReturn:
    """Class to test the shape of the Grad/Jacobian/Hessian with different return types."""

    def test_grad_single_measurement_param(self, dev, diff_method, grad_on_execution, device_vjp):
        """For one measurement and one param, the gradient is a float."""

        @qnode(
            dev,
            interface="autograd",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array(0.1, requires_grad=True)

        grad = qml.grad(circuit)(a)

        assert isinstance(grad, np.tensor if diff_method == "backprop" else float)

    def test_grad_single_measurement_multiple_param(
        self, dev, diff_method, grad_on_execution, device_vjp
    ):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""

        @qnode(
            dev,
            interface="autograd",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        grad = qml.grad(circuit)(a, b)

        assert isinstance(grad, tuple)
        assert len(grad) == 2
        assert grad[0].shape == ()
        assert grad[1].shape == ()

    def test_grad_single_measurement_multiple_param_array(
        self, dev, diff_method, grad_on_execution, device_vjp
    ):
        """For one measurement and multiple param as a single array params, the gradient is an array."""

        @qnode(
            dev,
            interface="autograd",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array([0.1, 0.2], requires_grad=True)

        grad = qml.grad(circuit)(a)

        assert isinstance(grad, np.ndarray)
        assert len(grad) == 2
        assert grad.shape == (2,)

    def test_jacobian_single_measurement_param_probs(
        self, dev, diff_method, grad_on_execution, device_vjp
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned with the correct
        dimension"""

        @qnode(
            dev,
            interface="autograd",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.probs(wires=[0, 1])

        a = np.array(0.1, requires_grad=True)

        jac = qml.jacobian(circuit)(a)

        assert isinstance(jac, np.ndarray)
        assert jac.shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param(
        self, dev, diff_method, grad_on_execution, device_vjp
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""

        @qnode(
            dev,
            interface="autograd",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=[0, 1])

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        jac = qml.jacobian(circuit)(a, b)

        assert isinstance(jac, tuple)

        assert isinstance(jac[0], np.ndarray)
        assert jac[0].shape == (4,)

        assert isinstance(jac[1], np.ndarray)
        assert jac[1].shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param_single_array(
        self, dev, diff_method, grad_on_execution, device_vjp
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned."""

        @qnode(
            dev,
            interface="autograd",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.probs(wires=[0, 1])

        a = np.array([0.1, 0.2], requires_grad=True)
        jac = qml.jacobian(circuit)(a)

        assert isinstance(jac, np.ndarray)
        assert jac.shape == (4, 2)

    def test_jacobian_multiple_measurement_single_param(
        self, dev, diff_method, grad_on_execution, device_vjp
    ):
        """The jacobian of multiple measurements with a single params return an array."""

        @qnode(
            dev,
            interface="autograd",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = np.array(0.1, requires_grad=True)

        def cost(x):
            return anp.hstack(circuit(x))

        jac = qml.jacobian(cost)(a)

        assert isinstance(jac, np.ndarray)
        assert jac.shape == (5,)

    def test_jacobian_multiple_measurement_multiple_param(
        self, dev, diff_method, grad_on_execution, device_vjp
    ):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""

        @qnode(
            dev,
            interface="autograd",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        def cost(x, y):
            return anp.hstack(circuit(x, y))

        jac = qml.jacobian(cost)(a, b)

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], np.ndarray)
        assert jac[0].shape == (5,)

        assert isinstance(jac[1], np.ndarray)
        assert jac[1].shape == (5,)

    def test_jacobian_multiple_measurement_multiple_param_array(
        self, dev, diff_method, grad_on_execution, device_vjp
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""

        @qnode(
            dev,
            interface="autograd",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = np.array([0.1, 0.2], requires_grad=True)

        def cost(x):
            return anp.hstack(circuit(x))

        jac = qml.jacobian(cost)(a)

        assert isinstance(jac, np.ndarray)
        assert jac.shape == (5, 2)

    def test_hessian_expval_multiple_params(self, dev, diff_method, grad_on_execution, device_vjp):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support second-order diff.")

        par_0 = qml.numpy.array(0.1, requires_grad=True)
        par_1 = qml.numpy.array(0.2, requires_grad=True)

        @qnode(
            dev,
            interface="autograd",
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

        def cost(x, y):
            return anp.hstack(qml.grad(circuit)(x, y))

        hess = qml.jacobian(cost)(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], np.ndarray)
        assert hess[0].shape == (2,)

        assert isinstance(hess[1], np.ndarray)
        assert hess[1].shape == (2,)

    def test_hessian_expval_multiple_param_array(
        self, dev, diff_method, grad_on_execution, device_vjp
    ):
        """The hessian of single measurement with a multiple params array return a single array."""

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support second-order diff.")

        params = qml.numpy.array([0.1, 0.2], requires_grad=True)

        @qnode(
            dev,
            interface="autograd",
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

        hess = qml.jacobian(qml.grad(circuit))(params)

        assert isinstance(hess, np.ndarray)
        assert hess.shape == (2, 2)

    def test_hessian_var_multiple_params(self, dev, diff_method, grad_on_execution, device_vjp):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support second-order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Hadamard gradient does not support variances.")

        par_0 = qml.numpy.array(0.1, requires_grad=True)
        par_1 = qml.numpy.array(0.2, requires_grad=True)

        @qnode(
            dev,
            interface="autograd",
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

        def cost(x, y):
            return anp.hstack(qml.grad(circuit)(x, y))

        hess = qml.jacobian(cost)(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], np.ndarray)
        assert hess[0].shape == (2,)

        assert isinstance(hess[1], np.ndarray)
        assert hess[1].shape == (2,)

    def test_hessian_var_multiple_param_array(
        self, dev, diff_method, grad_on_execution, device_vjp
    ):
        """The hessian of single measurement with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support second-order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Hadamard gradient does not support variances.")

        params = qml.numpy.array([0.1, 0.2], requires_grad=True)

        @qnode(
            dev,
            interface="autograd",
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

        hess = qml.jacobian(qml.grad(circuit))(params)

        assert isinstance(hess, np.ndarray)
        assert hess.shape == (2, 2)

    def test_hessian_probs_expval_multiple_params(
        self, dev, diff_method, grad_on_execution, device_vjp
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        if diff_method in ["adjoint", "hadamard"]:
            pytest.skip("The adjoint method does not currently support second-order diff.")

        par_0 = qml.numpy.array(0.1, requires_grad=True)
        par_1 = qml.numpy.array(0.2, requires_grad=True)

        @qnode(
            dev,
            interface="autograd",
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
            return anp.hstack(circuit(x, y))

        def cost(x, y):
            return anp.hstack(qml.jacobian(circuit_stack)(x, y))

        hess = qml.jacobian(cost)(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], np.ndarray)
        assert hess[0].shape == (6,)

        assert isinstance(hess[1], np.ndarray)
        assert hess[1].shape == (6,)

    def test_hessian_expval_probs_multiple_param_array(
        self, dev, diff_method, grad_on_execution, device_vjp
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""

        if diff_method in ["adjoint", "hadamard"]:
            pytest.skip("The adjoint method does not currently support second-order diff.")

        params = qml.numpy.array([0.1, 0.2], requires_grad=True)

        @qnode(
            dev,
            interface="autograd",
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

        def cost(x):
            return anp.hstack(circuit(x))

        hess = qml.jacobian(qml.jacobian(cost))(params)

        assert isinstance(hess, np.ndarray)
        assert hess.shape == (3, 2, 2)  # pylint: disable=no-member

    def test_hessian_probs_var_multiple_params(
        self, dev, diff_method, grad_on_execution, device_vjp
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support second-order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Hadamard gradient does not support variances.")

        par_0 = qml.numpy.array(0.1, requires_grad=True)
        par_1 = qml.numpy.array(0.2, requires_grad=True)

        @qnode(
            dev,
            interface="autograd",
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
            return anp.hstack(circuit(x, y))

        def cost(x, y):
            return anp.hstack(qml.jacobian(circuit_stack)(x, y))

        hess = qml.jacobian(cost)(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], np.ndarray)
        assert hess[0].shape == (6,)

        assert isinstance(hess[1], np.ndarray)
        assert hess[1].shape == (6,)

    def test_hessian_var_multiple_param_array2(
        self, dev, diff_method, grad_on_execution, device_vjp
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support second-order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Hadamard gradient does not support variances.")

        params = qml.numpy.array([0.1, 0.2], requires_grad=True)

        @qnode(
            dev,
            interface="autograd",
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

        def cost(x):
            return anp.hstack(circuit(x))

        hess = qml.jacobian(qml.jacobian(cost))(params)

        assert isinstance(hess, np.ndarray)
        assert hess.shape == (3, 2, 2)  # pylint: disable=no-member


def test_no_ops():
    """Test that the return value of the QNode matches in the interface
    even if there are no ops"""

    dev = DefaultQubit()

    @qml.qnode(dev, interface="autograd")
    def circuit():
        qml.Hadamard(wires=0)
        return qml.state()

    res = circuit()
    assert isinstance(res, np.tensor)
