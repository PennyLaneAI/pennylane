# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests trainable circuits using the Autograd interface."""
# pylint:disable=no-self-use
import pytest

import numpy as np

import pennylane as qml
from pennylane import numpy as pnp


@pytest.mark.usefixtures("validate_diff_method")
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "hadamard"])
class TestGradients:
    """Test various gradient computations."""

    def test_basic_grad(self, diff_method, device, tol):
        """Test a basic function with one RX and one expectation."""
        wires = 2 if diff_method == "hadamard" else 1
        dev = device(wires=wires)
        tol = tol(dev.shots)
        if diff_method == "hadamard":
            tol += 0.01

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        x = pnp.array(0.5)
        res = qml.grad(circuit)(x)
        assert np.isclose(res, -pnp.sin(x), atol=tol, rtol=0)

    def test_backprop_state(self, diff_method, device, tol):
        """Test the trainability of parameters in a circuit returning the state."""
        if diff_method != "backprop":
            pytest.skip(reason="test only works with backprop")
        dev = device(2)
        if dev.shots:
            pytest.skip("test uses backprop, must be in analytic mode")
        if "mixed" in dev.name:
            pytest.skip("mixed-state simulator will wrongly use grad on non-scalar results")
        tol = tol(dev.shots)

        x = pnp.array(0.543)
        y = pnp.array(-0.654)

        @qml.qnode(dev, diff_method=diff_method, grad_on_execution=True)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.state()

        def cost_fn(x, y):
            res = circuit(x, y)
            probs = pnp.abs(res) ** 2
            return probs[0] + probs[2]

        res = qml.grad(cost_fn)(x, y)
        expected = np.array([-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        y = pnp.array(-0.654, requires_grad=False)
        res = qml.grad(cost_fn)(x, y)
        assert np.allclose(res, expected[0], atol=tol, rtol=0)

    def test_parameter_shift(self, diff_method, device, tol):
        """Test a multi-parameter circuit with parameter-shift."""
        if diff_method != "parameter-shift":
            pytest.skip(reason="test only works with parameter-shift")

        a = pnp.array(0.1)
        b = pnp.array(0.2)

        dev = device(2)
        tol = tol(dev.shots)

        @qml.qnode(dev, diff_method="parameter-shift", grad_on_execution=False)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hamiltonian([1, 1], [qml.Z(0), qml.Y(1)]))

        res = qml.grad(circuit)(a, b)
        expected = [-np.sin(a) + np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # make the second QNode argument a constant
        b = pnp.array(0.2, requires_grad=False)
        res = qml.grad(circuit)(a, b)
        assert np.allclose(res, expected[0], atol=tol, rtol=0)

    def test_probs(self, diff_method, device, tol):
        """Test differentiation of a circuit returning probs()."""
        wires = 3 if diff_method == "hadamard" else 2
        dev = device(wires=wires)
        tol = tol(dev.shots)
        x = pnp.array(0.543)
        y = pnp.array(-0.654)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        res = qml.jacobian(circuit)(x, y)

        expected = np.array(
            [
                [-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2],
                [np.cos(y) * np.sin(x) / 2, np.cos(x) * np.sin(y) / 2],
            ]
        )

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], pnp.ndarray)
        assert res[0].shape == (2,)

        assert isinstance(res[1], pnp.ndarray)
        assert res[1].shape == (2,)

        if diff_method == "hadamard" and "raket" in dev.name:
            pytest.xfail(reason="braket gets wrong results for hadamard here")
        assert np.allclose(res[0], expected.T[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected.T[1], atol=tol, rtol=0)

    def test_multi_meas(self, diff_method, device, tol):
        """Test differentiation of a circuit with both scalar and array-like returns."""
        wires = 3 if diff_method == "hadamard" else 2
        dev = device(wires=wires)
        tol = tol(dev.shots)
        x = pnp.array(0.543)
        y = pnp.array(-0.654, requires_grad=False)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(0)), qml.probs(wires=[1])

        def cost_fn(x, y):
            return pnp.hstack(circuit(x, y))

        jac = qml.jacobian(cost_fn)(x, y)

        expected = [-np.sin(x), -np.sin(x) * np.cos(y) / 2, np.cos(y) * np.sin(x) / 2]
        assert isinstance(jac, pnp.ndarray)
        assert np.allclose(jac, expected, atol=tol, rtol=0)

    def test_hessian(self, diff_method, device, tol):
        """Test hessian computation."""
        wires = 3 if diff_method == "hadamard" else 1
        dev = device(wires=wires)
        tol = tol(dev.shots)

        @qml.qnode(dev, diff_method=diff_method, max_diff=2)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.Z(0))

        x = pnp.array([1.0, 2.0])
        res = circuit(x)

        a, b = x

        expected_res = np.cos(a) * np.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        g = grad_fn(x)

        expected_g = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        hess = qml.jacobian(grad_fn)(x)

        expected_hess = [
            [-np.cos(a) * np.cos(b), np.sin(a) * np.sin(b)],
            [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)],
        ]
        assert np.allclose(hess, expected_hess, atol=tol, rtol=0)
