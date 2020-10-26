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
"""
Integration tests for the ``default.qubit.autograd`` device.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.devices.default_qubit_autograd import DefaultQubitAutograd


class TestQNodeIntegration:
    """Integration tests for default.qubit.autograd. This test ensures it integrates
    properly with the PennyLane UI, in particular the new QNode."""

    def test_defines_correct_capabilities(self):
        """Test that the device defines the right capabilities"""

        dev = qml.device("default.qubit.autograd", wires=1)
        cap = dev.capabilities()
        capabilities = {"model": "qubit",
                        "supports_finite_shots": True,
                        "supports_tensor_observables": True,
                        "returns_probs": True,
                        "returns_state": True,
                        "supports_reversible_diff": False,
                        "supports_inverse_operations": True,
                        "supports_analytic_computation": True,
                        "passthru_interface": 'autograd',
                        }
        assert cap == capabilities

    def test_load_device(self):
        """Test that the plugin device loads correctly"""
        dev = qml.device("default.qubit.autograd", wires=2)
        assert dev.num_wires == 2
        assert dev.shots == 1000
        assert dev.analytic
        assert dev.short_name == "default.qubit.autograd"
        assert dev.capabilities()["passthru_interface"] == "autograd"

    def test_qubit_circuit(self, tol):
        """Test that the device provides the correct
        result for a simple circuit."""
        p = np.array(0.543)

        dev = qml.device("default.qubit.autograd", wires=1)

        @qml.qnode(dev, interface="autograd")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -np.sin(p)

        assert isinstance(circuit, qml.qnodes.PassthruQNode)
        assert np.isclose(circuit(p), expected, atol=tol, rtol=0)

    def test_correct_state(self, tol):
        """Test that the device state is correct after applying a
        quantum function on the device"""

        dev = qml.device("default.qubit.autograd", wires=2)

        state = dev.state
        expected = np.array([1, 0, 0, 0])
        assert np.allclose(state, expected, atol=tol, rtol=0)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit():
            qml.Hadamard(wires=0)
            qml.RZ(np.pi / 4, wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit()
        state = dev.state

        amplitude = np.exp(-1j * np.pi / 8) / np.sqrt(2)

        expected = np.array([amplitude, 0, np.conj(amplitude), 0])
        assert np.allclose(state, expected, atol=tol, rtol=0)


class TestPassthruIntegration:
    """Tests for integration with the PassthruQNode"""

    def test_jacobian_variable_multiply(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.autograd device
        gives the correct result in the case of parameters multiplied by scalars"""
        x = 0.43316321
        y = 0.2162158
        z = 0.75110998
        weights = np.array([x, y, z], requires_grad=True)

        dev = qml.device("default.qubit.autograd", wires=1)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit(p):
            qml.RX(3 * p[0], wires=0)
            qml.RY(p[1], wires=0)
            qml.RX(p[2] / 2, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert isinstance(circuit, qml.qnodes.PassthruQNode)
        res = circuit(weights)

        expected = np.cos(3 * x) * np.cos(y) * np.cos(z / 2) - np.sin(3 * x) * np.sin(z / 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad_fn = qml.jacobian(circuit, 0)
        res = grad_fn(np.array(weights))

        expected = np.array(
            [
                -3 * (np.sin(3 * x) * np.cos(y) * np.cos(z / 2) + np.cos(3 * x) * np.sin(z / 2)),
                -np.cos(3 * x) * np.sin(y) * np.cos(z / 2),
                -0.5 * (np.sin(3 * x) * np.cos(z / 2) + np.cos(3 * x) * np.cos(y) * np.sin(z / 2)),
            ]
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_repeated(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.autograd device
        gives the correct result in the case of repeated parameters"""
        x = 0.43316321
        y = 0.2162158
        z = 0.75110998
        p = np.array([x, y, z], requires_grad=True)
        dev = qml.device("default.qubit.autograd", wires=1)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(p)

        expected = np.cos(y) ** 2 - np.sin(x) * np.sin(y) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad_fn = qml.jacobian(circuit, 0)
        res = grad_fn(p)

        expected = np.array(
            [-np.cos(x) * np.sin(y) ** 2, -2 * (np.sin(x) + 1) * np.sin(y) * np.cos(y), 0]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_agrees_backprop_parameter_shift(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.autograd device
        gives the correct result with respect to the parameter-shift method"""
        p = np.array([0.43316321, 0.2162158, 0.75110998, 0.94714242], requires_grad=True)

        def circuit(x):
            for i in range(0, len(p), 2):
                qml.RX(x[i], wires=0)
                qml.RY(x[i + 1], wires=1)
            for i in range(2):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))

        dev1 = qml.device("default.qubit.autograd", wires=3)
        dev2 = qml.device("default.qubit.autograd", wires=3)

        circuit1 = qml.QNode(circuit, dev1, diff_method="backprop", interface="autograd")
        circuit2 = qml.QNode(circuit, dev2, diff_method="parameter-shift")

        assert isinstance(circuit1, qml.qnodes.PassthruQNode)
        assert isinstance(circuit2, qml.qnodes.QubitQNode)

        res = circuit1(p)

        assert np.allclose(res, circuit2(p), atol=tol, rtol=0)

        grad_fn = qml.jacobian(circuit1, 0)
        res = grad_fn(p)
        assert np.allclose(res, circuit2.jacobian([p]), atol=tol, rtol=0)

    def test_state_differentiability(self, tol):
        """Test that the device state can be differentiated"""
        dev = qml.device("default.qubit.autograd", wires=1)

        @qml.qnode(dev, diff_method="backprop", interface="autograd")
        def circuit(a):
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array(0.54, requires_grad=True)

        def cost(a):
            """A function of the device quantum state, as a function
            of input QNode parameters."""
            circuit(a)
            res = np.abs(dev.state) ** 2
            return res[1] - res[0]

        grad = qml.grad(cost)(a)
        expected = np.sin(a)
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_prob_differentiability(self, tol):
        """Test that the device probability can be differentiated"""
        dev = qml.device("default.qubit.autograd", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="autograd")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = np.array(0.54, requires_grad=True)
        b = np.array(0.12, requires_grad=True)

        def cost(a, b):
            prob_wire_1 = circuit(a, b)[0]
            return prob_wire_1[1] - prob_wire_1[0]

        res = cost(a, b)
        expected = -np.cos(a) * np.cos(b)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad = qml.grad(cost)(a, b)
        expected = [np.sin(a) * np.cos(b), np.cos(a) * np.sin(b)]
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_backprop_gradient(self, tol):
        """Tests that the gradient of the qnode is correct"""
        dev = qml.device("default.qubit.autograd", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="autograd")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = np.array(-0.234, requires_grad=True)
        b = np.array(0.654, requires_grad=True)

        res = circuit(a, b)
        expected_cost = 0.5 * (np.cos(a) * np.cos(b) + np.cos(a) - np.cos(b) + 1)
        assert np.allclose(res, expected_cost, atol=tol, rtol=0)

        res = qml.grad(circuit)(a, b)
        expected_grad = np.array(
            [-0.5 * np.sin(a) * (np.cos(b) + 1), 0.5 * np.sin(b) * (1 - np.cos(a))]
        )
        assert np.allclose(res, expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("operation", [qml.U3, qml.U3.decomposition])
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "finite-diff"])
    def test_autograd_interface_gradient(self, operation, diff_method, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the Autograd interface, using a variety of differentiation methods."""
        dev = qml.device("default.qubit.autograd", wires=1)

        @qml.qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(x, weights, w=None):
            """In this example, a mixture of scalar
            arguments, array arguments, and keyword arguments are used."""
            qml.QubitStateVector(1j * np.array([1, -1]) / np.sqrt(2), wires=w)
            operation(x, weights[0], weights[1], wires=w)
            return qml.expval(qml.PauliX(w))

        # Check that the correct QNode type is being used.
        if diff_method == "backprop":
            assert isinstance(circuit, qml.qnodes.PassthruQNode)
            assert not hasattr(circuit, "jacobian")
        else:
            assert not isinstance(circuit, qml.qnodes.PassthruQNode)
            assert hasattr(circuit, "jacobian")

        def cost(params):
            """Perform some classical processing"""
            return circuit(params[0], params[1:], w=0) ** 2

        theta = 0.543
        phi = -0.234
        lam = 0.654

        params = np.array([theta, phi, lam], requires_grad=True)

        res = cost(params)
        expected_cost = (np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi)) ** 2
        assert np.allclose(res, expected_cost, atol=tol, rtol=0)

        res = qml.grad(cost)(params)
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
        assert np.allclose(res, expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["tf", "torch"])
    def test_error_backprop_wrong_interface(self, interface, tol):
        """Tests that an error is raised if diff_method='backprop' but not using
        the Autograd interface"""
        dev = qml.device("default.qubit.autograd", wires=1)

        def circuit(x, w=None):
            qml.RZ(x, wires=w)
            return qml.expval(qml.PauliX(w))

        with pytest.raises(
            ValueError,
            match="default.qubit.autograd only supports diff_method='backprop' when using the autograd interface",
        ):
            qml.qnode(dev, diff_method="backprop", interface=interface)(circuit)


class TestHighLevelIntegration:
    """Tests for integration with higher level components of PennyLane."""

    def test_template_integration(self):
        """Test that a PassthruQNode default.qubit.autograd works with templates."""
        dev = qml.device("default.qubit.autograd", wires=2)

        @qml.qnode(dev, diff_method="backprop")
        def circuit(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        weights = np.array(
            qml.init.strong_ent_layers_normal(n_wires=2, n_layers=2), requires_grad=True
        )

        grad = qml.grad(circuit)(weights)[0]
        assert grad.shape == weights.shape

    def test_qnode_collection_integration(self):
        """Test that a PassthruQNode default.qubit.autograd works with QNodeCollections."""
        dev = qml.device("default.qubit.autograd", wires=2)

        def ansatz(weights, **kwargs):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])

        obs_list = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1)]
        qnodes = qml.map(ansatz, obs_list, dev, interface="autograd")

        assert qnodes.interface == "autograd"

        weights = np.array([0.1, 0.2], requires_grad=True)

        def cost(weights):
            return np.sum(qnodes(weights))

        grad = qml.grad(cost)(weights)[0]
        assert grad.shape == weights.shape

class TestOps:
    """Unit tests for operations supported by the default.qubit.autograd device"""

    def test_multirz_jacobian(self):
        """Test that the patched numpy functions are used for the MultiRZ
        operation and the jacobian can be computed."""
        wires = 4
        dev = qml.device('default.qubit.autograd', wires=wires)

        @qml.qnode(dev, diff_method="backprop")
        def circuit(param):
            qml.MultiRZ(param, wires=[0,1])
            return qml.probs(wires = list(range(wires)))

        param = 0.3
        res = qml.jacobian(circuit)(param)
        assert np.allclose(res, np.zeros(wires **2))

    def test_full_subsystem(self, mocker):
        """Test applying a state vector to the full subsystem"""
        dev = DefaultQubitAutograd(wires=['a', 'b', 'c'])
        state = np.array([1, 0, 0, 0, 1, 0, 1, 1]) / 2.
        state_wires = qml.wires.Wires(['a', 'b', 'c'])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        assert np.all(dev._state.flatten() == state)
        spy.assert_not_called()

    def test_partial_subsystem(self, mocker):
        """Test applying a state vector to a subset of wires of the full subsystem"""

        dev = DefaultQubitAutograd(wires=['a', 'b', 'c'])
        state = np.array([1, 0, 1, 0]) / np.sqrt(2.)
        state_wires = qml.wires.Wires(['a', 'c'])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)
        res = np.sum(dev._state, axis=(1,)).flatten()

        assert np.all(res == state)
        spy.assert_called()
