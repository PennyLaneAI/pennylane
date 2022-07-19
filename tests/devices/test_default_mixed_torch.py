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
"""
Tests for the ``default.mixed`` device for the Torch interface.
"""
import re
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.devices.default_mixed import DefaultMixed
from pennylane import DeviceError

pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")


class TestQNodeIntegration:
    """Integration tests for default.mixed with Torch. This test ensures it integrates
    properly with the PennyLane UI, in particular the QNode."""

    def test_load_device(self):
        """Test that the plugin device loads correctly"""
        dev = qml.device("default.mixed", wires=2)
        assert dev.num_wires == 2
        assert dev.shots == None
        assert dev.short_name == "default.mixed"
        assert dev.capabilities()["passthru_devices"]["torch"] == "default.mixed"

    def test_qubit_circuit(self, tol):
        """Test that the device provides the correct
        result for a simple circuit."""
        p = torch.tensor(0.543)

        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -np.sin(p)

        assert circuit.gradient_fn == "backprop"
        assert np.isclose(circuit(p), expected, atol=tol, rtol=0)

    def test_correct_state(self, tol):
        """Test that the device state is correct after evaluating a
        quantum function on the device"""

        dev = qml.device("default.mixed", wires=2)

        state = dev.state
        expected = np.zeros((4, 4))
        expected[0, 0] = 1
        assert np.allclose(state, expected, atol=tol, rtol=0)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(a):
            qml.Hadamard(wires=0)
            qml.RZ(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit(torch.tensor(np.pi / 4))
        state = dev.state

        amplitude = np.exp(-1j * np.pi / 4) / 2
        expected = np.array(
            [[0.5, 0, amplitude, 0], [0, 0, 0, 0], [np.conj(amplitude), 0, 0.5, 0], [0, 0, 0, 0]]
        )

        assert np.allclose(state, expected, atol=tol, rtol=0)


class TestDtypePreserved:
    """Test that the user-defined dtype of the device is preserved for QNode
    evaluation"""

    @pytest.mark.parametrize(
        "r_dtype, r_dtype_torch", [(np.float32, torch.float32), (np.float64, torch.float64)]
    )
    @pytest.mark.parametrize(
        "measurement",
        [
            qml.expval(qml.PauliY(0)),
            qml.var(qml.PauliY(0)),
            qml.probs(wires=[1]),
            qml.probs(wires=[2, 0]),
        ],
    )
    def test_real_dtype(self, r_dtype, r_dtype_torch, measurement, tol):
        """Test that the user-defined dtype of the device is preserved
        for QNodes with real-valued outputs"""
        p = torch.tensor(0.543)

        dev = qml.device("default.mixed", wires=3, r_dtype=r_dtype)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.apply(measurement)

        res = circuit(p)
        assert res.dtype == r_dtype_torch

    @pytest.mark.parametrize(
        "c_dtype, c_dtype_torch",
        [(np.complex64, torch.complex64), (np.complex128, torch.complex128)],
    )
    @pytest.mark.parametrize(
        "measurement",
        [qml.state(), qml.density_matrix(wires=[1]), qml.density_matrix(wires=[2, 0])],
    )
    def test_complex_dtype(self, c_dtype, c_dtype_torch, measurement, tol):
        """Test that the user-defined dtype of the device is preserved
        for QNodes with complex-valued outputs"""
        p = torch.tensor(0.543)

        dev = qml.device("default.mixed", wires=3, c_dtype=c_dtype)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.apply(measurement)

        res = circuit(p)
        assert res.dtype == c_dtype_torch


class TestOps:
    """Unit tests for operations supported by the default.mixed device with Torch"""

    def test_multirz_jacobian(self):
        """Test that the patched numpy functions are used for the MultiRZ
        operation and the jacobian can be computed."""
        wires = 4
        dev = qml.device("default.mixed", wires=wires)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(param):
            qml.MultiRZ(param, wires=[0, 1])
            return qml.probs(wires=list(range(wires)))

        param = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        res = torch.autograd.functional.jacobian(circuit, param)

        assert np.allclose(res, np.zeros(wires**2))

    def test_full_subsystem(self, mocker):
        """Test applying a state vector to the full subsystem"""
        dev = DefaultMixed(wires=["a", "b", "c"])
        state = torch.tensor([1, 0, 0, 0, 1, 0, 1, 1], dtype=torch.complex128) / 2.0
        state_wires = qml.wires.Wires(["a", "b", "c"])

        spy = mocker.spy(qml.math, "scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        state = torch.outer(state, torch.conj(state))

        assert torch.allclose(torch.reshape(dev._state, (-1,)), torch.reshape(state, (-1,)))
        spy.assert_not_called()

    def test_partial_subsystem(self, mocker):
        """Test applying a state vector to a subset of wires of the full subsystem"""

        dev = DefaultMixed(wires=["a", "b", "c"])
        state = torch.tensor([1, 0, 1, 0], dtype=torch.complex128) / np.sqrt(2.0)
        state_wires = qml.wires.Wires(["a", "c"])

        spy = mocker.spy(qml.math, "scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        state = torch.kron(torch.outer(state, torch.conj(state)), torch.tensor([[1, 0], [0, 0]]))

        assert torch.allclose(torch.reshape(dev._state, (8, 8)), state)
        spy.assert_called()


class TestPassthruIntegration:
    """Tests for integration with the PassthruQNode"""

    def test_jacobian_variable_multiply(self, tol):
        """Test that jacobian of a QNode with an attached default.mixed.torch device
        gives the correct result in the case of parameters multiplied by scalars"""
        x = torch.tensor(0.43316321, dtype=torch.float64)
        y = torch.tensor(0.2162158, dtype=torch.float64)
        z = torch.tensor(0.75110998, dtype=torch.float64)
        weights = torch.tensor([x, y, z], dtype=torch.float64, requires_grad=True)

        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(p):
            qml.RX(3 * p[0], wires=0)
            qml.RY(p[1], wires=0)
            qml.RX(p[2] / 2, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit.gradient_fn == "backprop"
        res = circuit(weights)

        expected = np.cos(3 * x) * np.cos(y) * np.cos(z / 2) - np.sin(3 * x) * np.sin(z / 2)
        assert qml.math.allclose(res, expected, atol=tol, rtol=0)

        res.backward()
        res = weights.grad

        expected = np.array(
            [
                -3 * (np.sin(3 * x) * np.cos(y) * np.cos(z / 2) + np.cos(3 * x) * np.sin(z / 2)),
                -np.cos(3 * x) * np.sin(y) * np.cos(z / 2),
                -0.5 * (np.sin(3 * x) * np.cos(z / 2) + np.cos(3 * x) * np.cos(y) * np.sin(z / 2)),
            ]
        )

        assert qml.math.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_repeated(self, tol):
        """Test that the jacobian of a QNode with an attached default.mixed.torch device
        gives the correct result in the case of repeated parameters"""
        x = torch.tensor(0.43316321, dtype=torch.float64)
        y = torch.tensor(0.2162158, dtype=torch.float64)
        z = torch.tensor(0.75110998, dtype=torch.float64)
        p = torch.tensor([x, y, z], dtype=torch.float64, requires_grad=True)
        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(p)
        res.backward()

        expected = torch.cos(y) ** 2 - torch.sin(x) * torch.sin(y) ** 2
        assert torch.allclose(res, expected, atol=tol, rtol=0)

        expected = torch.tensor(
            [
                -torch.cos(x) * torch.sin(y) ** 2,
                -2 * (torch.sin(x) + 1) * torch.sin(y) * torch.cos(y),
                0,
            ]
        )
        assert torch.allclose(p.grad, expected, atol=tol, rtol=0)

    def test_backprop_jacobian_agrees_parameter_shift(self, tol):
        """Test that jacobian of a QNode with an attached default.mixed.torch device
        gives the correct result with respect to the parameter-shift method"""
        p = np.array([0.43316321, 0.2162158, 0.75110998, 0.94714242])
        p_torch = torch.tensor(p, dtype=torch.float64, requires_grad=True)

        def circuit(x):
            for i in range(0, len(p), 2):
                qml.RX(x[i], wires=0)
                qml.RY(x[i + 1], wires=1)
            for i in range(2):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))

        dev1 = qml.device("default.mixed", wires=3)
        dev2 = qml.device("default.mixed", wires=3)

        circuit1 = qml.QNode(circuit, dev1, diff_method="backprop", interface="torch")
        circuit2 = qml.QNode(circuit, dev2, diff_method="parameter-shift")

        assert circuit1.gradient_fn == "backprop"
        assert circuit2.gradient_fn is qml.gradients.param_shift

        res = circuit1(p_torch)
        assert qml.math.allclose(res, circuit2(p), atol=tol, rtol=0)

        grad = torch.autograd.functional.jacobian(circuit1, p_torch)
        assert qml.math.allclose(grad, qml.jacobian(circuit2)(p), atol=tol, rtol=0)

    @pytest.mark.parametrize("wires", [[0], ["abc"]])
    def test_state_differentiability(self, wires, tol):
        """Test that the device state can be differentiated"""
        dev = qml.device("default.mixed", wires=wires)

        @qml.qnode(dev, diff_method="backprop", interface="torch")
        def circuit(a):
            qml.RY(a, wires=wires[0])
            return qml.state()

        a = torch.tensor(0.54, dtype=torch.float64, requires_grad=True)

        state = circuit(a)
        res = torch.abs(state) ** 2
        res = res[1][1] - res[0][0]
        res.backward()

        expected = torch.sin(a)
        assert torch.allclose(a.grad, expected, atol=tol, rtol=0)

    def test_state_vector_differentiability(self, tol):
        """Test that the device state vector can be differentiated directly"""
        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(a):
            qml.RY(a, wires=0)
            return qml.state()

        a = torch.tensor(0.54, dtype=torch.complex128, requires_grad=True)

        grad = torch.autograd.functional.jacobian(circuit, a)
        expected = 0.5 * torch.tensor([[-torch.sin(a), torch.cos(a)], [torch.cos(a), torch.sin(a)]])

        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("wires", [range(2), [-12.32, "abc"]])
    def test_density_matrix_differentiability(self, wires, tol):
        """Test that the density matrix can be differentiated"""
        dev = qml.device("default.mixed", wires=wires)

        @qml.qnode(dev, diff_method="backprop", interface="torch")
        def circuit(a):
            qml.RY(a, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            return qml.density_matrix(wires=wires[1])

        a = torch.tensor(0.54, dtype=torch.float64, requires_grad=True)

        state = circuit(a)
        res = torch.abs(state) ** 2
        res = res[1][1] - res[0][0]
        res.backward()

        expected = torch.sin(a)
        assert torch.allclose(a.grad, expected, atol=tol, rtol=0)

    def test_prob_differentiability(self, tol):
        """Test that the device probability can be differentiated"""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="torch")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = torch.tensor(0.54, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(0.12, dtype=torch.float64, requires_grad=True)

        probs = circuit(a, b)
        res = probs[1] - probs[0]
        res.backward()

        expected = -torch.cos(a) * torch.cos(b)
        assert torch.allclose(res, expected, atol=tol, rtol=0)

        assert torch.allclose(a.grad, torch.sin(a) * torch.cos(b), atol=tol, rtol=0)
        assert torch.allclose(b.grad, torch.cos(a) * torch.sin(b), atol=tol, rtol=0)

    def test_prob_vector_differentiability(self, tol):
        """Test that the device probability vector can be differentiated directly"""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="torch")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = torch.tensor(0.54, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(0.12, dtype=torch.float64, requires_grad=True)

        res = circuit(a, b)

        expected = torch.tensor(
            [
                torch.cos(a / 2) ** 2 * torch.cos(b / 2) ** 2
                + torch.sin(a / 2) ** 2 * torch.sin(b / 2) ** 2,
                torch.cos(a / 2) ** 2 * torch.sin(b / 2) ** 2
                + torch.sin(a / 2) ** 2 * torch.cos(b / 2) ** 2,
            ]
        )
        assert torch.allclose(res, expected, atol=tol, rtol=0)

        grad_a, grad_b = torch.autograd.functional.jacobian(circuit, (a, b))

        assert torch.allclose(
            grad_a, 0.5 * torch.tensor([-torch.sin(a) * torch.cos(b), torch.sin(a) * torch.cos(b)])
        )
        assert torch.allclose(
            grad_b, 0.5 * torch.tensor([-torch.cos(a) * torch.sin(b), torch.cos(a) * torch.sin(b)])
        )

    def test_sample_backprop_error(self):
        """Test that sampling in backpropagation mode raises an error"""
        dev = qml.device("default.mixed", wires=1, shots=100)

        msg = "Backpropagation is only supported when shots=None"

        with pytest.raises(qml.QuantumFunctionError, match=msg):

            @qml.qnode(dev, diff_method="backprop", interface="torch")
            def circuit(a):
                qml.RY(a, wires=0)
                return qml.sample(qml.PauliZ(0))

    def test_expval_gradient(self, tol):
        """Tests that the gradient of expval is correct"""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="torch")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = torch.tensor(-0.234, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(0.654, dtype=torch.float64, requires_grad=True)

        res = circuit(a, b)
        res.backward()

        expected_cost = 0.5 * (torch.cos(a) * torch.cos(b) + torch.cos(a) - torch.cos(b) + 1)
        assert torch.allclose(res, expected_cost, atol=tol, rtol=0)

        assert torch.allclose(a.grad, -0.5 * torch.sin(a) * (torch.cos(b) + 1), atol=tol, rtol=0)
        assert torch.allclose(b.grad, 0.5 * torch.sin(b) * (1 - torch.cos(a)), atol=tol, rtol=0)

    @pytest.mark.parametrize("x, shift", [(0.0, 0.0), (0.5, -0.5)])
    def test_hessian_at_zero(self, x, shift):
        """Tests that the Hessian at vanishing state vector amplitudes
        is correct."""
        dev = qml.device("default.mixed", wires=1)

        shift = torch.tensor(shift)
        x = torch.tensor(x, requires_grad=True)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x):
            qml.RY(shift, wires=0)
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        grad = torch.autograd.functional.jacobian(circuit, x)
        hess = torch.autograd.functional.hessian(circuit, x)

        assert qml.math.isclose(grad, torch.tensor(0.0))
        assert qml.math.isclose(hess, torch.tensor(-1.0))

    @pytest.mark.parametrize("operation", [qml.U3, qml.U3.compute_decomposition])
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "finite-diff"])
    def test_torch_interface_gradient(self, operation, diff_method, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the TF interface, using a variety of differentiation methods."""
        dev = qml.device("default.mixed", wires=1)
        state = torch.tensor(1j * np.array([1, -1]) / np.sqrt(2), requires_grad=False)

        @qml.qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(x, weights, w):
            """In this example, a mixture of scalar
            arguments, array arguments, and keyword arguments are used."""
            qml.QubitStateVector(state, wires=w)
            operation(x, weights[0], weights[1], wires=w)
            return qml.expval(qml.PauliX(w))

        def cost(params):
            """Perform some classical processing"""
            return circuit(params[0], params[1:], w=0) ** 2

        theta = 0.543
        phi = -0.234
        lam = 0.654

        params = torch.tensor([theta, phi, lam], dtype=torch.float64, requires_grad=True)

        res = cost(params)
        expected_cost = (np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi)) ** 2
        assert torch.allclose(res, torch.tensor(expected_cost), atol=tol, rtol=0)

        # Check that the correct differentiation method is being used.
        if diff_method == "backprop":
            assert circuit.gradient_fn == "backprop"
        elif diff_method == "parameter-shift":
            assert circuit.gradient_fn is qml.gradients.param_shift
        else:
            assert circuit.gradient_fn is qml.gradients.finite_diff

        res.backward()
        res = params.grad

        expected_grad = (
            np.array(
                [
                    np.sin(theta) * np.cos(lam) * np.cos(phi),
                    np.cos(theta) * np.cos(lam) * np.sin(phi) + np.sin(lam) * np.cos(phi),
                    np.cos(theta) * np.sin(lam) * np.cos(phi) + np.cos(lam) * np.sin(phi),
                ]
            )
            * 2
            * (np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi))
        )
        assert torch.allclose(res, torch.tensor(expected_grad), atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "dev_name,diff_method,mode",
        [
            ["default.mixed", "finite-diff", "backward"],
            ["default.mixed", "parameter-shift", "backward"],
            ["default.mixed", "backprop", "forward"],
        ],
    )
    def test_ragged_differentiation(self, dev_name, diff_method, mode, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        dev = qml.device(dev_name, wires=2)
        x = torch.tensor(0.543, dtype=torch.float64)
        y = torch.tensor(-0.654, dtype=torch.float64)

        @qml.qnode(dev, diff_method=diff_method, mode=mode, interface="torch")
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])]

        res = circuit(x, y)

        expected = torch.tensor(
            [
                torch.cos(x),
                (1 + torch.cos(x) * torch.cos(y)) / 2,
                (1 - torch.cos(x) * torch.cos(y)) / 2,
            ]
        )
        assert torch.allclose(res, expected, atol=tol, rtol=0)

        res_x, res_y = torch.autograd.functional.jacobian(circuit, (x, y))
        expected_x = torch.tensor(
            [-torch.sin(x), -torch.sin(x) * torch.cos(y) / 2, torch.cos(y) * torch.sin(x) / 2]
        )
        expected_y = torch.tensor(
            [0, -torch.cos(x) * torch.sin(y) / 2, torch.cos(x) * torch.sin(y) / 2]
        )

        assert torch.allclose(res_x, expected_x, atol=tol, rtol=0)
        assert torch.allclose(res_y, expected_y, atol=tol, rtol=0)

    def test_batching(self, tol):
        """Tests that the gradient of the qnode is correct with batching parameters"""
        dev = qml.device("default.mixed", wires=2)

        @qml.batch_params
        @qml.qnode(dev, diff_method="backprop", interface="torch")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = torch.tensor([-0.234, 0.678], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([0.654, 1.236], dtype=torch.float64, requires_grad=True)

        res = circuit(a, b)

        expected_cost = 0.5 * (torch.cos(a) * torch.cos(b) + torch.cos(a) - torch.cos(b) + 1)
        assert qml.math.allclose(res, expected_cost, atol=tol, rtol=0)

        res_a, res_b = torch.autograd.functional.jacobian(circuit, (a, b))
        expected_a, expected_b = [
            -0.5 * torch.sin(a) * (torch.cos(b) + 1),
            0.5 * torch.sin(b) * (1 - torch.cos(a)),
        ]

        assert qml.math.allclose(torch.diagonal(res_a), expected_a, atol=tol, rtol=0)
        assert qml.math.allclose(torch.diagonal(res_b), expected_b, atol=tol, rtol=0)


class TestHighLevelIntegration:
    """Tests for integration with higher level components of PennyLane."""

    def test_template_integration(self):
        """Test that a PassthruQNode default.mixed.torch works with templates."""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
        weights = torch.tensor(np.random.random(shape), dtype=torch.float64, requires_grad=True)

        res = circuit(weights)
        res.backward()

        assert isinstance(weights.grad, torch.Tensor)
        assert weights.grad.shape == weights.shape

    def test_qnode_collection_integration(self):
        """Test that a PassthruQNode default.mixed.torch works with QNodeCollections."""
        dev = qml.device("default.mixed", wires=2)

        obs_list = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1)]
        qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev, interface="torch")

        assert qnodes.interface == "torch"

        torch.manual_seed(42)
        weights = torch.rand(
            qml.templates.StronglyEntanglingLayers.shape(n_wires=2, n_layers=2), requires_grad=True
        )

        def cost(weights):
            return torch.sum(qnodes(weights))

        res = cost(weights)
        res.backward()

        grad = weights.grad

        assert torch.is_tensor(res)
        assert grad.shape == weights.shape
