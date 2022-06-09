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

import numpy as onp
import pytest

import pennylane as qml
from pennylane import numpy as np

pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class TestFidelityMath:
    """Tests for Fidelity function between two states (state vectors or density matrix)."""

    state0_state1_fid = [
        ([1, 0], [0, 1], 0),
        ([0, 1], [0, 1], 1.0),
        ([1, 0], [1, 1] / np.sqrt(2), 0.5),
        ([1, 0], [[0, 0], [0, 1]], 0),
        ([1, 0], [[1, 0], [0, 0]], 1.0),
        ([1, 0], [[0.5, 0], [0, 0.5]], 0.5),
        ([0, 1], [[0.5, 0], [0, 0.5]], 0.5),
        ([1, 0], [[0.5, 0.5], [0.5, 0.5]], 0.5),
        ([0, 1], [[0.5, 0.5], [0.5, 0.5]], 0.5),
        ([[0.5, 0], [0, 0.5]], [1, 0], 0.5),
        ([[0.5, 0], [0, 0.5]], [0, 1], 0.5),
        ([[0.5, 0.5], [0.5, 0.5]], [1, 0], 0.5),
        ([[0.5, 0.5], [0.5, 0.5]], [0, 1], 0.5),
        ([[1, 0], [0, 0]], [[0.5, 0], [0, 0.5]], 0.5),
        ([[0, 0], [0, 1]], [[0.5, 0], [0, 0.5]], 0.5),
        ([[1, 0], [0, 0]], [[0.5, 0.5], [0.5, 0.5]], 0.5),
        ([[0, 0], [0, 1]], [[0.5, 0.5], [0.5, 0.5]], 0.5),
    ]

    array_funcs = [
        lambda x: x,
        onp.array,
        np.array,
        jnp.array,
        torch.tensor,
        tf.Variable,
        tf.constant,
    ]

    check_state = [True, False]

    @pytest.mark.parametrize("state0,state1,fid", state0_state1_fid)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("func", array_funcs)
    def test_state_vector_entropy(self, state0, state1, check_state, fid, func):
        """Test fidelity between different quantum states."""
        state0 = func(state0)
        state1 = func(state1)
        fidelity = qml.math.fidelity(state0, state1, check_state)

        assert qml.math.allclose(fid, fidelity)

    def test_state_vector_0_amplitudes(self):
        """Test that a message is raised when the state 0does not have right norm"""
        state0 = [0.5, 0]
        state1 = [0, 1]
        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            qml.math.fidelity(state0, state1, check_state=True)

    def test_state_vector_1_amplitudes(self):
        """Test that a message is raised when the state 1 does not have right norm"""
        state0 = [0, 1]
        state1 = [0.5, 1]
        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            qml.math.fidelity(state0, state1, check_state=True)

    def test_state_vector_0_wrong_shape(self):
        """Test that a message is raised when the state 0does not have right norm"""
        state0 = [0, 1, 1]
        state1 = [0, 1]
        with pytest.raises(ValueError, match="State vector must be of length"):
            qml.math.fidelity(state0, state1, check_state=True)

    def test_state_vector_1_wrong_shape(self):
        """Test that a message is raised when the state 1 does not have right norm"""
        state0 = [0, 1]
        state1 = [0, 1, 1]
        with pytest.raises(ValueError, match="State vector must be of length"):
            qml.math.fidelity(state0, state1, check_state=True)

    def test_density_matrix_0_wrong_shape(self):
        """Test that a message is raised when the state0 does not have the right shape."""
        state0 = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        state1 = [0, 1]
        with pytest.raises(ValueError, match="Density matrix must be of shape"):
            qml.math.fidelity(state0, state1, check_state=True)

    def test_density_matrix_1_wrong_shape(self):
        """Test that a message is raised when the state 1 does not have the right shape."""
        state0 = [0, 1]
        state1 = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        with pytest.raises(ValueError, match="Density matrix must be of shape"):
            qml.math.fidelity(state0, state1, check_state=True)

    def test_density_matrix_0_wrong_trace(self):
        """Test that a message is raised when the state0 does not have the right trace."""
        state0 = [[1, 0], [0, -1]]
        state1 = [0, 1]
        with pytest.raises(ValueError, match="The trace of the density matrix should be one"):
            qml.math.fidelity(state0, state1, check_state=True)

    def test_density_matrix_1_wrong_trace(self):
        """Test that a message is raised when the state 1 does not have the right trace."""
        state0 = [0, 1]
        state1 = [[1, 0], [0, -1]]
        with pytest.raises(ValueError, match="The trace of the density matrix should be one"):
            qml.math.fidelity(state0, state1, check_state=True)

    def test_density_matrix_0_not_hermitian(self):
        """Test that a message is raised when the state0 is not hermitian."""
        state0 = [[1, 1], [0, 0]]
        state1 = [0, 1]
        with pytest.raises(ValueError, match="The matrix is not hermitian"):
            qml.math.fidelity(state0, state1, check_state=True)

    def test_density_matrix_1_not_hermitian(self):
        """Test that a message is raised when the state 1 is not hermitian."""
        state0 = [0, 1]
        state1 = [[1, 1], [0, 0]]
        with pytest.raises(ValueError, match="The matrix is not hermitian"):
            qml.math.fidelity(state0, state1, check_state=True)

    def test_density_matrix_0_not_positive_semi_def(self):
        """Test that a message is raised when the state0 is not positive semi def."""
        state0 = [[2, 0], [0, -1]]
        state1 = [0, 1]
        with pytest.raises(ValueError, match="The matrix is not positive semi"):
            qml.math.fidelity(state0, state1, check_state=True)

    def test_density_matrix_1_not_positive_semi_def(self):
        """Test that a message is raised when the state 1 is not positive semi def."""
        state0 = [0, 1]
        state1 = [[2, 0], [0, -1]]
        with pytest.raises(ValueError, match="The matrix is not positive semi"):
            qml.math.fidelity(state0, state1, check_state=True)

    def test_same_number_wires(self):
        """Test that the two states must act on the same number of wires"""
        state0 = [0, 1, 0, 0]
        state1 = [[1, 0], [0, 0]]
        with pytest.raises(
            qml.QuantumFunctionError, match="The two states must have the same number of wires"
        ):
            qml.math.fidelity(state0, state1, check_state=True)
