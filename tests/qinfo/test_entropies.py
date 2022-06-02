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
"""Unit tests for differentiable quantum entropies.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np


def expected_entropy_ising_xx(param):
    """
    Return the analytical entropy for the IsingXX.
    """
    eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    eigs = [eig_1, eig_2]
    eigs = [eig for eig in eigs if eig > 0]

    expected_entropy = eigs * np.log(eigs)

    expected_entropy = -np.sum(expected_entropy)
    return expected_entropy


def expected_entropy_grad_ising_xx(param):
    """
    Return the analytical gradient entropy for the IsingXX.
    """
    eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    eigs = [eig_1, eig_2]
    eigs = np.maximum(eigs, 1e-08)

    grad_expected_entropy = -(
        (np.log(eigs[0]) + 1)
        * (np.sin(param / 2) ** 3 * np.cos(param / 2) - np.sin(param / 2) * np.cos(param / 2) ** 3)
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
    return grad_expected_entropy


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
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_transform_entropy(self, param, wires, device, base):
        """Test entropy for a QNode numpy."""

        dev = qml.device(device, wires=2)

        @qml.qnode(dev)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = qml.qinfo.vn_entropy_transform(circuit_state, indices=wires, base=base)(param)

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)
        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.autograd
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_transform_entropy_grad(self, param, wires, base):
        """Test entropy for a QNode gradient with autograd."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="backprop")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        grad_entropy = qml.grad(
            qml.qinfo.vn_entropy_transform(circuit_state, indices=wires, base=base)
        )(param)

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)
        assert qml.math.allclose(grad_entropy, grad_expected_entropy)

    @pytest.mark.torch
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_transform_torch_entropy(self, param, wires, device, base):
        """Test entropy for a QNode with torch interface."""
        import torch

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = qml.qinfo.vn_entropy_transform(circuit_state, indices=wires, base=base)(
            torch.tensor(param)
        )

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)
        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.torch
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_transform_entropy_grad_torch(self, param, wires, base):
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
        eigs = np.maximum(eigs, 1e-08)

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        entropy = qml.qinfo.vn_entropy_transform(circuit_state, indices=wires, base=base)(param)
        entropy.backward()
        grad_entropy = param.grad

        assert qml.math.allclose(grad_entropy, grad_expected_entropy)

    @pytest.mark.tf
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_transform_tf_entropy(self, param, wires, device, base):
        """Test entropy for a QNode with tf interface."""
        import tensorflow as tf

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface="tf")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = qml.qinfo.vn_entropy_transform(circuit_state, indices=wires, base=base)(
            tf.Variable(param)
        )

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.tf
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_transform_entropy_grad_tf(self, param, wires, base):
        """Test entropy for a QNode gradient with tf."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        param = tf.Variable(param)
        with tf.GradientTape() as tape:
            entropy = qml.qinfo.vn_entropy_transform(circuit_state, indices=wires, base=base)(param)

        grad_entropy = tape.gradient(entropy, param)

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        assert qml.math.allclose(grad_entropy, grad_expected_entropy)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_transform_jax_entropy(self, param, wires, device, base):
        """Test entropy for a QNode with jax interface."""
        import jax.numpy as jnp

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = qml.qinfo.vn_entropy_transform(circuit_state, indices=wires, base=base)(
            jnp.array(param)
        )

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_transform_entropy_grad_jax(self, param, wires, base):
        """Test entropy for a QNode gradient with Jax."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        grad_entropy = jax.grad(
            qml.qinfo.vn_entropy_transform(circuit_state, indices=wires, base=base)
        )(jax.numpy.array(param))

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        assert qml.math.allclose(grad_entropy, grad_expected_entropy, rtol=1e-04, atol=1e-05)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_transform_jax_jit_entropy(self, param, wires, base):
        """Test entropy for a QNode with jax-jit interface."""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax-jit")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = jax.jit(qml.qinfo.vn_entropy_transform(circuit_state, indices=wires, base=base))(
            jnp.array(param)
        )

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_transform_entropy_grad_jax_jit(self, param, wires, base):
        """Test entropy for a QNode gradient with Jax-jit."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax-jit", diff_method="backprop")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        grad_entropy = jax.jit(
            jax.grad(qml.qinfo.vn_entropy_transform(circuit_state, indices=wires, base=base))
        )(jax.numpy.array(param))

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        assert qml.math.allclose(grad_entropy, grad_expected_entropy, rtol=1e-04, atol=1e-05)
