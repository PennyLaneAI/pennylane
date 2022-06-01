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
"""Unit tests for differentiable quantum entropies.
"""
import numpy
import pytest

import pennylane as qml
from pennylane import numpy as np

pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")


class TestVonNeumannEntropy:
    """Tests for creating a density matrix from state vectors."""

    single_wires_list = [
        [0],
        [1],
    ]

    base = [2, np.exp(1), 10]

    check_state = [True, False]

    parameters = np.linspace(0, 2 * np.pi, 50)
    devices = ["default.qubit", "default.mixed"]

    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    def test_IsingXX_qnode_transform_entropy(self, param, wires, device):
        """Test entropy for a QNode numpy."""

        dev = qml.device(device, wires=2)

        @qml.qnode(dev)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = qml.qinfo.vn_entropy_transform(circuit_state, indices=wires)(param)

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]

        expected_entropy = eigs * np.log(eigs)

        expected_entropy = -np.sum(expected_entropy)
        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.autograd
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    def test_IsingXX_qnode_transform_entropy_grad(self, param, wires):
        """Test entropy for a QNode gradient with autograd."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="backprop")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        grad_entropy = qml.grad(qml.qinfo.vn_entropy_transform(circuit_state, indices=wires))(param)

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = numpy.maximum(eigs, 1e-08)

        grad_expected_entropy = -(
            (np.log(eigs[0]) + 1)
            * (
                np.sin(param / 2) ** 3 * np.cos(param / 2)
                - np.sin(param / 2) * np.cos(param / 2) ** 3
            )
            / np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)
        ) - (
            (np.log(eigs[1]) + 1)
            * (
                np.sin(param / 2)
                * np.cos(param / 2)
                * (np.cos(param / 2) ** 2 - np.sin(param / 2) ** 2)
            )
            / np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)
        )

        assert qml.math.allclose(grad_entropy, grad_expected_entropy)

    @pytest.mark.torch
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    def test_IsingXX_qnode_transform_torch_entropy(self, param, wires, device):
        """Test entropy for a QNode with torch interface."""
        import torch

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = qml.qinfo.vn_entropy_transform(circuit_state, indices=wires)(torch.tensor(param))

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]

        expected_entropy = eigs * np.log(eigs)

        expected_entropy = -np.sum(expected_entropy)

        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.torch
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    def test_IsingXX_qnode_transform_entropy_grad_torch(self, param, wires):
        """Test entropy for a QNode gradient with torch."""
        import torch

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = numpy.maximum(eigs, 1e-08)

        grad_expected_entropy = -(
            (np.log(eigs[0]) + 1)
            * (
                np.sin(param / 2) ** 3 * np.cos(param / 2)
                - np.sin(param / 2) * np.cos(param / 2) ** 3
            )
            / np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)
        ) - (
            (np.log(eigs[1]) + 1)
            * (
                np.sin(param / 2)
                * np.cos(param / 2)
                * (np.cos(param / 2) ** 2 - np.sin(param / 2) ** 2)
            )
            / np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)
        )

        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        torch.autograd.set_detect_anomaly(True)
        entropy = qml.qinfo.vn_entropy_transform(circuit_state, indices=wires)(param)
        entropy.backward()
        grad_entropy = param.grad

        assert qml.math.allclose(grad_entropy, grad_expected_entropy)

    @pytest.mark.tf
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    def test_IsingXX_qnode_transform_tf_entropy(self, param, wires, device):
        """Test entropy for a QNode with tf interface."""
        import tensorflow as tf

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface="tf")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = qml.qinfo.vn_entropy_transform(circuit_state, indices=wires)(tf.Variable(param))

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]

        expected_entropy = eigs * np.log(eigs)

        expected_entropy = -np.sum(expected_entropy)

        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.tf
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    def test_IsingXX_qnode_transform_entropy_grad_tf(self, param, wires):
        """Test entropy for a QNode gradient with tf."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        param = tf.Variable(param)
        with tf.GradientTape() as tape:
            entropy = qml.qinfo.vn_entropy_transform(circuit_state, indices=wires)(param)

        grad_entropy = tape.gradient(entropy, param)

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = numpy.maximum(eigs, 1e-08)

        grad_expected_entropy = -(
            (np.log(eigs[0]) + 1)
            * (
                np.sin(param / 2) ** 3 * np.cos(param / 2)
                - np.sin(param / 2) * np.cos(param / 2) ** 3
            )
            / np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)
        ) - (
            (np.log(eigs[1]) + 1)
            * (
                np.sin(param / 2)
                * np.cos(param / 2)
                * (np.cos(param / 2) ** 2 - np.sin(param / 2) ** 2)
            )
            / np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)
        )
        print(grad_entropy, grad_expected_entropy)
        assert qml.math.allclose(grad_entropy, grad_expected_entropy)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    def test_IsingXX_qnode_transform_jax_entropy(self, param, wires, device):
        """Test entropy for a QNode with jax interface."""
        import jax.numpy as jnp

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = qml.qinfo.vn_entropy_transform(circuit_state, indices=wires)(jnp.array(param))

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]

        expected_entropy = eigs * np.log(eigs)

        expected_entropy = -np.sum(expected_entropy)

        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    def test_IsingXX_qnode_transform_entropy_grad_jax(self, param, wires):
        """Test entropy for a QNode gradient with Jax."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        grad_entropy = jax.grad(qml.qinfo.vn_entropy_transform(circuit_state, indices=wires))(
            jax.numpy.array(param)
        )

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = numpy.maximum(eigs, 1e-08)

        grad_expected_entropy = -(
            (np.log(eigs[0]) + 1)
            * (
                np.sin(param / 2) ** 3 * np.cos(param / 2)
                - np.sin(param / 2) * np.cos(param / 2) ** 3
            )
            / np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)
        ) - (
            (np.log(eigs[1]) + 1)
            * (
                np.sin(param / 2)
                * np.cos(param / 2)
                * (np.cos(param / 2) ** 2 - np.sin(param / 2) ** 2)
            )
            / np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)
        )

        assert qml.math.allclose(grad_entropy, grad_expected_entropy, rtol=1e-04, atol=1e-05)


class TestMutualInformation:
    """Tests for the mutual information functions"""

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed"])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    @pytest.mark.parametrize(
        "params", [np.array([0, 0]), np.array([0.3, 0.4]), np.array([0.6, 0.8])]
    )
    def test_qnode_state(self, device, interface, params):
        """Test that mutual information works for QNodes that return the state"""
        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = qml.qinfo.mutual_info_transform(circuit, wires0=[0], wires1=[1])(params)

        # compare QNode results with the results of computing directly from the state
        state = circuit(params)
        expected = qml.math.to_mutual_info(state, wires0=[0], wires1=[1])

        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed"])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    @pytest.mark.parametrize(
        "params", [np.array([0, 0]), np.array([0.3, 0.4]), np.array([0.6, 0.8])]
    )
    def test_qnode_mutual_info(self, device, interface, params):
        """Test that mutual information works for QNodes that directly return it"""
        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit_mutual_info(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        @qml.qnode(dev, interface=interface)
        def circuit_state(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = circuit_mutual_info(params)

        # compare QNode results with the results of computing directly from the state
        state = circuit_state(params)
        expected = qml.math.to_mutual_info(state, wires0=[0], wires1=[1])

        assert np.allclose(actual, expected)

    def test_grad_qnode(self):
        """Test that the gradient of mutual information works for QNodes"""
