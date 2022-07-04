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
        # Vector-Vector-Fid
        ([1, 0], [0, 1], 0),
        ([0, 1], [0, 1], 1.0),
        ([1, 0], [1, 1] / np.sqrt(2), 0.5),
        # Vector-Density mat-Fid
        ([1, 0], [[0, 0], [0, 1]], 0),
        ([1, 0], [[1, 0], [0, 0]], 1.0),
        ([1, 0], [[0.5, 0], [0, 0.5]], 0.5),
        ([0, 1], [[0.5, 0], [0, 0.5]], 0.5),
        ([1, 0], [[0.5, 0.5], [0.5, 0.5]], 0.5),
        ([0, 1], [[0.5, 0.5], [0.5, 0.5]], 0.5),
        # Density mat-Vector-Fid
        ([[0.5, 0], [0, 0.5]], [1, 0], 0.5),
        ([[0.5, 0], [0, 0.5]], [0, 1], 0.5),
        ([[0.5, 0.5], [0.5, 0.5]], [1, 0], 0.5),
        ([[0.5, 0.5], [0.5, 0.5]], [0, 1], 0.5),
        # Density mat-Density mat-Fid
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
    def test_state_vector_fidelity(self, state0, state1, check_state, fid, func):
        """Test fidelity between different quantum states."""
        state0 = func(state0)
        state1 = func(state1)
        fidelity = qml.math.fidelity(state0, state1, check_state)

        assert qml.math.allclose(fid, fidelity)

    state_wrong_amp = [([0.5, 0], [0, 1]), ([0, 1], [0.5, 0])]

    @pytest.mark.parametrize("state0,state1", state_wrong_amp)
    def test_state_vector_wrong_amplitudes(self, state0, state1):
        """Test that a message is raised when a state does not have right norm"""
        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            qml.math.fidelity(state0, state1, check_state=True)

    state_wrong_shape = [([0, 1, 1], [0, 1]), ([0, 1], [0, 1, 1])]

    @pytest.mark.parametrize("state0,state1", state_wrong_shape)
    def test_state_vector_wrong_shape(self, state0, state1):
        """Test that a message is raised when the state does not have the right shape."""
        with pytest.raises(ValueError, match="State vector must be of length"):
            qml.math.fidelity(state0, state1, check_state=True)

    d_mat_wrong_shape = [
        ([[1, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 1]),
        ([0, 1], [[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
    ]

    @pytest.mark.parametrize("state0,state1", d_mat_wrong_shape)
    def test_density_matrix_wrong_shape(self, state0, state1):
        """Test that a message is raised when the density matrix does not have the right shape."""
        with pytest.raises(ValueError, match="Density matrix must be of shape"):
            qml.math.fidelity(state0, state1, check_state=True)

    d_mat_wrong_trace = [([[1, 0], [0, -1]], [0, 1]), ([0, 1], [[1, 0], [0, -1]])]

    @pytest.mark.parametrize("state0,state1", d_mat_wrong_trace)
    def test_density_matrix_wrong_trace(self, state0, state1):
        """Test that a message is raised when the density matrix does not have the right trace."""
        with pytest.raises(ValueError, match="The trace of the density matrix should be one"):
            qml.math.fidelity(state0, state1, check_state=True)

    d_mat_not_hermitian = [([[1, 1], [0, 0]], [0, 1]), ([0, 1], [[1, 1], [0, 0]])]

    @pytest.mark.parametrize("state0,state1", d_mat_not_hermitian)
    def test_density_matrix_not_hermitian(self, state0, state1):
        """Test that a message is raised when the density matrix is not Hermitian."""
        with pytest.raises(ValueError, match="The matrix is not Hermitian"):
            qml.math.fidelity(state0, state1, check_state=True)

    d_mat_not_positive = [([[2, 0], [0, -1]], [0, 1]), ([0, 1], [[2, 0], [0, -1]])]

    @pytest.mark.parametrize("state0,state1", d_mat_not_positive)
    def test_density_matrix_not_positive_semi_def(self, state0, state1):
        """Test that a message is raised when the density matrix is not positive semi def."""
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
