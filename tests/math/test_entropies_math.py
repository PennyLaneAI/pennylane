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

import pytest

import pennylane as qml
from pennylane import numpy as np


pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")


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
        entropy = qml.math.to_vn_entropy(state_vector, wires, base, check_state)

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
        entropy = qml.math.to_vn_entropy(density_matrix, wires, base, check_state)

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


class TestMutualInformation:
    """Tests for the mutual information functions"""

    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize(
        "state, expected",
        [
            ([1, 0, 0, 0], 0),
            ([np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0], 0),
            ([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2], 2 * np.log(2)),
            (np.ones(4) * 0.5, 0),
        ],
    )
    def test_state(self, interface, state, expected):
        """Test that mutual information works for states"""
        state = qml.math.asarray(state, like=interface)
        actual = qml.math.to_mutual_info(state, wires0=[0], wires1=[1])
        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize(
        "state, expected",
        [
            ([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 0),
            ([[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]], 0),
            ([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]], 2 * np.log(2)),
            (np.ones((4, 4)) * 0.25, 0),
        ],
    )
    def test_density_matrix(self, interface, state, expected):
        """Test that mutual information works for density matrices"""
        state = qml.math.asarray(state, like=interface)
        actual = qml.math.to_mutual_info(state, wires0=[0], wires1=[1])
        assert np.allclose(actual, expected)

    def test_grad_state(self):
        """Test that the gradient of mutual information works for states"""

    def test_grad_density_matrix(self):
        """Test that the gradient of mutual information works for density matrices"""

    @pytest.mark.parametrize(
        "state, wires0, wires1",
        [
            (np.array([1, 0, 0, 0]), [0], [0]),
            (np.array([1, 0, 0, 0]), [0], [0, 1]),
            (np.array([1, 0, 0, 0, 0, 0, 0, 0]), [0, 1], [1]),
            (np.array([1, 0, 0, 0, 0, 0, 0, 0]), [0, 1], [1, 2]),
        ],
    )
    def test_subsystem_overlap(self, state, wires0, wires1):
        """Test that an error is raised when the subsystems overlap"""
        with pytest.raises(
            ValueError, match="Subsystems for computing mutual information must not overlap"
        ):
            qml.math.to_mutual_info(state, wires0=wires0, wires1=wires1)

    @pytest.mark.parametrize("state", [np.array(5), np.ones((3, 4)), np.ones((2, 2, 2))])
    def test_invalid_type(self, state):
        """Test that an error is raised when an unsupported type is passed"""
        with pytest.raises(
            ValueError, match="The state is not a state vector or a density matrix."
        ):
            qml.math.to_mutual_info(state, wires0=[0], wires1=[1])
