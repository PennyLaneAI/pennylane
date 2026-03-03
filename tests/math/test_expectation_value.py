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
"""Unit tests for differentiable matrix-vector expectation values."""

import numpy as onp
import pytest

import pennylane as qml
from pennylane import numpy as np

pytestmark = pytest.mark.all_interfaces

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class TestExpectationValueMath:
    """Tests for Expectation value of a single operator for a state vector."""

    ops_vs_vecstates = [
        ([[1, 0], [0, 0]], [1, 0], 1),
        ([[0, 1], [1, 0]], [0, 1], 0),
        ([[0.5, 0.5], [0.5, 0.5]], [1, 1] / np.sqrt(2), 1),
        (
            [[0.40975111, 0.40751457], [0.40751457, 0.59024889]],
            [0.8660254, 0.5],
            0.8077935208042251,
        ),
        (
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
            [1, 0, 1, 0] / np.sqrt(2),
            0,
        ),
    ]

    array_funcs = [
        lambda x: x,
        onp.array,
        np.array,
        jnp.array,
        torch.tensor,
    ]

    @pytest.mark.parametrize("operator_and_states", ops_vs_vecstates)
    @pytest.mark.parametrize("func", array_funcs)
    def test_mat_expectation_value(self, operator_and_states, func):
        """Test the expectation value of a single operator for a vector state."""
        ops, state_vectors, expected = operator_and_states
        ops = func(ops)
        state_vectors = func(state_vectors)
        overlap = qml.math.expectation_value(ops, state_vectors)
        assert qml.math.allclose(expected, overlap)

    state_wrong_amp = [
        ([[1, 0], [0, 1]], [0.5, 0]),
        ([[1, 0], [0, 1]], [26, 70]),
        (
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
            [1, 2, 0, 0],
        ),
        (
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            [1, 1, 1, 1],
        ),
    ]

    @pytest.mark.parametrize("ops,state_vectors", state_wrong_amp)
    def test_state_vector_wrong_amplitudes(self, ops, state_vectors):
        """Test that a message is raised when a state does not have right norm"""
        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            qml.math.expectation_value(ops, state_vectors, check_state=True)

    operator_wrong_shape = [
        ([[1, 0, 0], [0, 0, 1]], [0, 1]),
        (
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
            [1, 0, 0, 0],
        ),
        ([[0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]], [1, 0]),
    ]

    @pytest.mark.parametrize("ops,state_vectors", operator_wrong_shape)
    def test_operator_wrong_shape(self, ops, state_vectors):
        """Test that a message is raised when the state does not have the right shape."""
        with pytest.raises(ValueError, match="Operator matrix must be of shape"):
            qml.math.expectation_value(ops, state_vectors, check_operator=True)

    operator_non_hermitian = [
        ([[1, 1], [0, 0]], [0, 1]),
        (
            [[0, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1]],
            [1, 0, 0, 0],
        ),
    ]

    @pytest.mark.parametrize("ops,state_vectors", operator_non_hermitian)
    def test_operator_non_hermitian(self, ops, state_vectors):
        """Test that a message is raised when the state does not have the right shape."""
        with pytest.raises(ValueError, match="The matrix is not Hermitian"):
            qml.math.expectation_value(ops, state_vectors, check_operator=True)

    state_wrong_shape = [
        ([[1, 0], [0, 1]], [0, 1, 0]),
        ([[1, 0], [0, 1]], [0, 0, 0, 0, 1]),
        (
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
            [1, 0, 0, 0, 0],
        ),
        ([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], [1, 0, 0]),
    ]

    @pytest.mark.parametrize("ops,state_vectors", state_wrong_shape)
    def test_state_vector_wrong_shape(self, ops, state_vectors):
        """Test that a message is raised when the state does not have the right shape."""
        with pytest.raises(ValueError, match="State vector must be of shape"):
            qml.math.expectation_value(ops, state_vectors, check_state=True)

    def test_same_number_wires_dm(self):
        """Test that the two states must act on the same number of wires"""
        ops = np.diag([0, 1, 0, 0])
        state_vectors = [1, 0]
        with pytest.raises(
            ValueError,
            match="The operator and the state vector must have the same number of wires.",
        ):
            qml.math.expectation_value(ops, state_vectors, check_state=True, check_operator=True)

    @pytest.mark.parametrize("func", array_funcs)
    def test_broadcast_op_sv(self, func):
        """Test simultaneous broadcasting of operators and state vectors works."""
        ops = qml.math.stack(
            (
                func([[1, 0], [0, 0]]),
                func([[0.5, 0.5], [0.5, 0.5]]),
                func([[0, 0], [0, 1]]),
                func([[0.40975111, 0.40751457], [0.40751457, 0.59024889]]),
            )
        )
        state_vectors = qml.math.stack(
            [
                func([0, 1]),
                func([1, 0]),
                func([1, 1] / np.sqrt(2)),
                func([0.8660254, 0.5]),
            ]
        )
        expected = [0, 0.5, 0.5, 0.8077935208042251]

        overlap = qml.math.expectation_value(ops, state_vectors)
        assert qml.math.allclose(overlap, expected)

    @pytest.mark.parametrize("func", array_funcs)
    def test_broadcast_op_unbatched(self, func):
        """Test broadcasting works for expectation values when the operators input is unbatched"""
        ops = func([[1, 0], [0, 0]])
        state_vectors = qml.math.stack(
            [
                func([0, 1]),
                func([1, 0]),
                func([1, 1] / np.sqrt(2)),
                func([0.8660254, 0.5]),
            ]
        )
        expected = [0, 1, 0.5, 0.7499999934451599]

        overlap = qml.math.expectation_value(ops, state_vectors)
        assert qml.math.allclose(overlap, expected)

    @pytest.mark.parametrize("func", array_funcs)
    def test_broadcast_sv_unbatched(self, func):
        """Test broadcasting works for expectation values when the state vector input is unbatched"""
        ops = qml.math.stack(
            (
                func([[1, 0], [0, 0]]),
                func([[0.5, 0.5], [0.5, 0.5]]),
                func([[0, 0], [0, 1]]),
                func([[0.40975111, 0.40751457], [0.40751457, 0.59024889]]),
            )
        )
        state_vectors = func([1, 0])
        expected = [1, 0.5, 0, 0.40975111]

        overlap = qml.math.expectation_value(ops, state_vectors)
        assert qml.math.allclose(overlap, expected)
