# Copyright 2019 Xanadu Quantum Technologies Inc.

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
Unit tests and integration tests for the :mod:`pennylane.plugin.Tensornet.tf` device.
"""
from itertools import product

import numpy
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.qnode_new import qnode, QNode
from pennylane.qnode_new.decorator import ALLOWED_INTERFACES, ALLOWED_DIFF_METHODS


tensornetwork = pytest.importorskip("tensornetwork", minversion="0.1")
tensorflow = pytest.importorskip("tensorflow", minversion="2.0")


class TestQNodeIntegration:
    """Integration tests for expt.tensornet.tf. This test ensures it integrates
    properly with the PennyLane UI, in particular the new QNode."""

    def test_load_tensornet_tf_device(self):
        """Test that the tensor network plugin loads correctly"""
        dev = qml.device("expt.tensornet.tf", wires=2)
        assert dev.num_wires == 2
        assert dev.shots == 1000
        assert dev.analytic
        assert dev.short_name == "expt.tensornet.tf"
        assert dev.capabilities()["provides_jacobian"]

    @pytest.mark.parametrize("decorator", [qml.qnode, qnode])
    def test_qubit_circuit(self, decorator, tol):
        """Test that the tensor network plugin provides correct
        result for a simple circuit using the old QNode.
        This test is parametrized for both the new and old QNode decorator."""
        p = 0.543

        dev = qml.device("expt.tensornet.tf", wires=1)

        @decorator(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -np.sin(p)

        assert np.isclose(circuit(p), expected, atol=tol, rtol=0)

    def test_cannot_overwrite_state(self):
        """Tests that _state is a property and cannot be overwritten."""
        dev = qml.device("expt.tensornet.tf", wires=2)

        with pytest.raises(AttributeError, match="can't set attribute"):
            dev._state = np.array([[1, 0], [0, 0]])

    def test_correct_state(self, tol):
        """Test that the device state is correct after applying a
        quantum function on the device"""

        dev = qml.device("expt.tensornet.tf", wires=2)

        state = dev._state

        expected = np.array([[1, 0], [0, 0]])
        assert np.allclose(state, expected, atol=tol, rtol=0)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit()
        state = dev._state

        expected = np.array([[1, 0], [1, 0]]) / np.sqrt(2)
        assert np.allclose(state, expected, atol=tol, rtol=0)

    def test_jacobian_fanout(self, torch_support, tol):
        """Test that qnode.jacobian applied to the tensornet.tf device
        returns the same result as default.qubit, in the case of repeated parameters"""
        p = np.array([0.43316321, 0.2162158, 0.75110998])

        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.RY(x[0], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.Rot(*x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))

        dev1 = qml.device("expt.tensornet.tf", wires=3)
        dev2 = qml.device("default.qubit", wires=3)

        circuit1 = QNode(circuit, dev1, diff_method="best")
        circuit2 = QNode(circuit, dev2, diff_method="best")

        assert np.allclose(circuit1(p), circuit2(p), atol=tol, rtol=0)
        assert np.allclose(circuit1.jacobian([p]), circuit2.jacobian([p]), atol=tol, rtol=0)

    def test_jacobian_agrees(self, torch_support, tol):
        """Test that qnode.jacobian applied to the tensornet.tf device
        returns the same result as default.qubit."""
        p = np.array([0.43316321, 0.2162158, 0.75110998, 0.94714242])

        def circuit(x):
            for i in range(0, len(p), 2):
                qml.RX(x[i], wires=0)
                qml.RY(x[i + 1], wires=1)
            for i in range(2):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))

        dev1 = qml.device("expt.tensornet.tf", wires=3)
        dev2 = qml.device("default.qubit", wires=3)

        circuit1 = QNode(circuit, dev1, diff_method="best")
        circuit2 = QNode(circuit, dev2, diff_method="best")

        assert np.allclose(circuit1(p), circuit2(p), atol=tol, rtol=0)
        assert np.allclose(circuit1.jacobian([p]), circuit2.jacobian([p]), atol=tol, rtol=0)


class TestGradientInterfaceIntegration:
    """Integration tests for expt.tensornet.tf. This test class ensures it integrates
    properly with the PennyLane UI, in particular the classical machine learning
    interfaces."""

    a = -0.234
    b = 0.654
    p = [a, b]

    # the analytic result of evaluating circuit(a, b)
    expected_cost = 0.5 * (np.cos(a)*np.cos(b) + np.cos(a) - np.cos(b) + 1)

    # the analytic result of evaluating grad(circuit(a, b))
    expected_grad = np.array([
        -0.5 * np.sin(a) * (np.cos(b) + 1),
        0.5 * np.sin(b) * (1 - np.cos(a))
    ])

    @pytest.fixture
    def circuit(self, interface, torch_support):
        """Fixture to create cost function for the test class"""
        dev = qml.device("expt.tensornet.tf", wires=2)

        if interface == "torch" and not torch_support:
            pytest.skip("Skipped, no torch support")

        @qnode(dev, diff_method="best", interface=interface)
        def circuit_fn(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        return circuit_fn

    @pytest.mark.parametrize("interface", ["autograd"])
    def test_autograd_interface(self, circuit, interface, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the autograd interface"""
        res = circuit(*self.p)
        assert np.allclose(res, self.expected_cost, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit, argnum=[0, 1])
        res = np.asarray(grad_fn(*self.p))
        assert np.allclose(res, self.expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["torch"])
    def test_torch_interface(self, circuit, interface, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the Torch interface"""
        import torch
        from torch.autograd import Variable

        params = Variable(torch.tensor(self.p), requires_grad=True)
        res = circuit(*params)
        assert np.allclose(res.detach().numpy(), self.expected_cost, atol=tol, rtol=0)

        res.backward()
        res = params.grad
        assert np.allclose(res.detach().numpy(), self.expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["tf"])
    def test_tf_interface(self, circuit, interface, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the TensorFlow interface"""
        import tensorflow as tf

        a = tf.Variable(self.a, dtype=tf.float64)
        b = tf.Variable(self.b, dtype=tf.float64)

        with tf.GradientTape() as tape:
            tape.watch([a, b])
            res = circuit(a, b)

        assert np.allclose(res.numpy(), self.expected_cost, atol=tol, rtol=0)

        res = tape.gradient(res, [a, b])
        assert np.allclose(res.numpy(), self.expected_grad, atol=tol, rtol=0)


class TestHybridInterfaceIntegration:
    """Integration tests for expt.tensornet.tf. This test class ensures it integrates
    properly with the PennyLane UI, in particular the classical machine learning
    interfaces in the case of hybrid-classical computation."""

    theta = 0.543
    phi = -0.234
    lam = 0.654
    p = [theta, phi, lam]

    # the analytic result of evaluating cost(p)
    expected_cost = (np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi)) ** 2

    # the analytic result of evaluating grad(cost(p))
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

    @pytest.fixture
    def cost(self, interface, torch_support):
        """Fixture to create cost function for the test class"""
        dev = qml.device("expt.tensornet.tf", wires=1)

        if interface == "torch" and not torch_support:
            pytest.skip("Skipped, no torch support")

        @qnode(dev, diff_method="best", interface=interface)
        def circuit(x, weights, w=None):
            """In this example, a mixture of scalar
            arguments, array arguments, and keyword arguments are used."""
            qml.QubitStateVector(1j * np.array([1, -1]) / np.sqrt(2), wires=w)
            # the parameterized gate is one that gets decomposed
            # via a template
            qml.U3(x, weights[0], weights[1], wires=w)
            return qml.expval(qml.PauliX(w))

        def cost_fn(params):
            """Perform some classical processing"""
            return circuit(params[0], params[1:], w=0) ** 2

        return cost_fn

    @pytest.mark.parametrize("interface", ["autograd"])
    def test_autograd_interface(self, cost, interface, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the autograd interface"""
        res = cost(self.p)
        assert np.allclose(res, self.expected_cost, atol=tol, rtol=0)

        grad_fn = qml.grad(cost, argnum=[0])
        res = np.asarray(grad_fn(self.p))
        assert np.allclose(res, self.expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["torch"])
    def test_torch_interface(self, cost, interface, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the Torch interface"""
        import torch
        from torch.autograd import Variable

        params = Variable(torch.tensor(self.p), requires_grad=True)
        res = cost(params)
        assert np.allclose(res.detach().numpy(), self.expected_cost, atol=tol, rtol=0)

        res.backward()
        res = params.grad
        assert np.allclose(res.detach().numpy(), self.expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["tf"])
    def test_tf_interface(self, cost, interface, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the TensorFlow interface"""
        import tensorflow as tf

        params = tf.Variable(self.p, dtype=tf.float64)

        with tf.GradientTape() as tape:
            tape.watch(params)
            res = cost(params)

        assert np.allclose(res.numpy(), self.expected_cost, atol=tol, rtol=0)

        res = tape.gradient(res, params)
        assert np.allclose(res.numpy(), self.expected_grad, atol=tol, rtol=0)
