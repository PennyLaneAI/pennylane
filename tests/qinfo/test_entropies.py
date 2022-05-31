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


class TestVonNeumannEntropy:
    """Tests for creating a density matrix from state vectors."""

    state_vector = [([1, 0, 0, 1] / np.sqrt(2), False), ([1, 0, 0, 0], True)]

    single_wires_list = [
        [0],
        [1],
    ]

    base = [2, np.exp(1), 10]

    check_state = [True, False]

    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("state_vector,pure", state_vector)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_state)
    def test_state_vector_entropy(self, state_vector, wires, base, check_state, pure):
        """Test entropy for different state vectors."""
        entropy = qml.qinfo.entropies.to_vn_entropy(state_vector, wires, base, check_state)

        if pure:
            expected_entropy = 0
        else:
            expected_entropy = np.log(2) / np.log(base)

        assert qml.math.allclose(entropy, expected_entropy)

    density_matrices = [
        ([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]], False),
        ([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], True),
    ]

    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("density_matrix,pure", density_matrices)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_state)
    def test_density_matrices_entropy(self, density_matrix, wires, base, check_state, pure):
        """Test entropy for different density matrices."""
        entropy = qml.qinfo.entropies.to_vn_entropy(density_matrix, wires, base, check_state)

        if pure:
            expected_entropy = 0
        else:
            expected_entropy = np.log(2) / np.log(base)

        assert qml.math.allclose(entropy, expected_entropy)

    parameters = np.linspace(0, 2 * np.pi, 50)
    devices = ["default.qubit", "default.mixed"]

    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    def test_IsingXX_qnode_entropy(self, param, wires, device):
        """Test entropy for a QNode numpy."""

        dev = qml.device(device, wires=2)

        @qml.qnode(dev)
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires)

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]

        expected_entropy = eigs * np.log(eigs)

        expected_entropy = -np.sum(expected_entropy)

        entropy = circuit_entropy(param)
        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.torch
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    def test_IsingXX_qnode_torch_entropy(self, param, wires, device):
        """Test entropy for a QNode with torch interface."""
        import torch

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires)

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]

        expected_entropy = eigs * np.log(eigs)

        expected_entropy = -np.sum(expected_entropy)

        entropy = circuit_entropy(torch.tensor(param))
        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.tf
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    def test_IsingXX_qnode_tf_entropy(self, param, wires, device):
        """Test entropy for a QNode with tf interface."""
        import tensorflow as tf

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface="tf")
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires)

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]

        expected_entropy = eigs * np.log(eigs)

        expected_entropy = -np.sum(expected_entropy)

        entropy = circuit_entropy(tf.Variable(param))
        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    def test_IsingXX_qnode_jax_entropy(self, param, wires, device):
        """Test entropy for a QNode with jax interface."""
        import jax.numpy as jnp

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires)

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]

        expected_entropy = eigs * np.log(eigs)

        expected_entropy = -np.sum(expected_entropy)

        entropy = circuit_entropy(jnp.array(param))
        assert qml.math.allclose(entropy, expected_entropy)
