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
"""Unit tests for differentiable trace distance."""

import numpy as onp
import pytest

import pennylane as qml
from pennylane import numpy as np

pytestmark = pytest.mark.all_interfaces

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class TestTraceDistanceMath:
    """Tests for the trace distance function between two states (state vectors or density matrix)."""

    state0_state1_td = [
        (
            (
                [[1, 0], [0, 0]],
                qml.math.reduce_statevector([x, np.sqrt(1 - x**2)], indices=[0]),
            ),
            np.sqrt(1 - x**2),
        )
        for x in np.linspace(0, 1, 10)
    ]
    state0_state1_td += [
        (
            (
                [[0.5, 0.5], [0.5, 0.5]],
                qml.math.reduce_statevector([x, np.sqrt(1 - x**2) * 1j], indices=[0]),
            ),
            np.sqrt(2) / 2,
        )
        for x in np.linspace(0, 1, 10)
    ]
    state0_state1_td.append(
        (
            (
                [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ],
            ),
            np.sqrt(2) / 2,
        )
    )

    array_funcs = [
        lambda x: x,
        onp.array,
        np.array,
        jnp.array,
        torch.tensor,
    ]

    check_state = [True, False]

    @pytest.mark.parametrize("states,td", state0_state1_td)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("func", array_funcs)
    def test_trace_distance(self, states, check_state, td, func):
        """Test trace distance between different quantum states."""
        state0, state1 = states
        state0 = func(state0)
        state1 = func(state1)

        assert qml.math.allclose(td, qml.math.trace_distance(state0, state1, check_state))
        assert qml.math.allclose(td, qml.math.trace_distance(state1, state0, check_state))

    state0_state1_td_batched = [
        # Batch-Mat-TD
        (
            (
                [
                    [[1, 0], [0, 0]],
                    [[0.5, 0], [0, 0.5]],
                    [[0.5, -0.5], [-0.5, 0.5]],
                    [[0.5, 0.5], [0.5, 0.5]],
                ],
                [[0.5, 0.5], [0.5, 0.5]],
            ),
            [np.sqrt(2) / 2, 0.5, 1, 0],
        ),
        (
            (
                [
                    [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
                    [
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                    ],
                    [[0.25, 0, 0, 0], [0, 0.25, 0, 0], [0, 0, 0.25, 0], [0, 0, 0, 0.25]],
                ],
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ],
            ),
            [np.sqrt(2) / 2, 0, 0.75],
        ),
        # Batch-Batch-TD
        (
            (
                [[[1, 0], [0, 0]], [[0.5, 0], [0, 0.5]], [[0.3, 0], [0, 0.7]], [[1, 0], [0, 0]]],
                [
                    [[0.5, 0], [0, 0.5]],
                    [[0.5, 0.5], [0.5, 0.5]],
                    [[1, 0], [0, 0]],
                    [[0, 0], [0, 1]],
                ],
            ),
            [0.5, 0.5, 0.7, 1],
        ),
        (
            (
                [
                    [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
                    [
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                    ],
                    [[0, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0]],
                ],
                [
                    [
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                    ],
                    [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
                ],
            ),
            [np.sqrt(2) / 2, np.sqrt(3) / 2, 1],
        ),
    ]

    @pytest.mark.parametrize("states,td", state0_state1_td_batched)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("func", array_funcs)
    def test_broadcast_trace_distance(self, states, check_state, td, func):
        """Test trace distance between different quantum states in batches."""
        state0, state1 = states
        state0 = func(state0)
        state1 = func(state1)

        assert qml.math.allclose(td, qml.math.trace_distance(state0, state1, check_state))
        assert qml.math.allclose(td, qml.math.trace_distance(state1, state0, check_state))

    d_mat_wrong_shape = [
        # Shape that is not a power of 2
        (([[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 0], [0, 0]])),
        # Len of shape that is not in (2, 3)
        (([1, 0], [[1, 0], [0, 0]])),
        # Shapes with different dimensions
        (([[1, 0, 0], [0, 0, 0]], [[1, 0], [0, 0]])),
        (([[[1, 0, 0], [0, 0, 0]]], [[1, 0], [0, 0]])),
    ]

    @pytest.mark.parametrize("states", d_mat_wrong_shape)
    def test_density_matrix_wrong_shape(self, states):
        """Test that a message is raised when the density matrix does not have the right shape."""
        state0, state1 = states

        with pytest.raises(ValueError, match="Density matrix must be of shape"):
            qml.math.trace_distance(state0, state1, check_state=True)

        with pytest.raises(ValueError, match="Density matrix must be of shape"):
            qml.math.trace_distance(state1, state0, check_state=True)

    d_mat_wrong_trace = [
        (([[1, 0], [0, -1]], [[1, 0], [0, 0]])),
        (([[[1, 0], [0, -1]], [[1, 0], [0, 0]]], [[1, 0], [0, 0]])),
    ]

    @pytest.mark.parametrize("states", d_mat_wrong_trace)
    def test_density_matrix_wrong_trace(self, states):
        """Test that a message is raised when the density matrix does not have the right trace."""
        state0, state1 = states

        with pytest.raises(ValueError, match="The trace of the density matrix should be one"):
            qml.math.trace_distance(state0, state1, check_state=True)

        with pytest.raises(ValueError, match="The trace of the density matrix should be one"):
            qml.math.trace_distance(state1, state0, check_state=True)

    d_mat_not_hermitian = [
        (([[1, 1], [0, 0]], [[1, 0], [0, 0]])),
        (([[1, 0], [0, 0]], [[1, 1], [0, 0]])),
        (([[[1, 1], [0, 0]], [[1, 0], [0, 0]]], [[1, 0], [0, 0]])),
        (([[1, 0], [0, 0]], [[[1, 1], [0, 0]], [[1, 0], [0, 0]]])),
    ]

    @pytest.mark.parametrize("states", d_mat_not_hermitian)
    def test_density_matrix_not_hermitian(self, states):
        """Test that a message is raised when the density matrix is not Hermitian."""
        state0, state1 = states

        with pytest.raises(ValueError, match="The matrix is not Hermitian"):
            qml.math.trace_distance(state0, state1, check_state=True)

        with pytest.raises(ValueError, match="The matrix is not Hermitian"):
            qml.math.trace_distance(state1, state0, check_state=True)

    d_mat_not_positive = [
        (([[2, 0], [0, -1]], [[1, 0], [0, 0]])),
        (([[1, 0], [0, 0]], [[2, 0], [0, -1]])),
        (([[[2, 0], [0, -1]], [[1, 0], [0, 0]]], [[1, 0], [0, 0]])),
        (([[1, 0], [0, 0]], [[[2, 0], [0, -1]], [[1, 0], [0, 0]]])),
    ]

    @pytest.mark.parametrize("states", d_mat_not_positive)
    def test_density_matrix_not_positive_semi_def(self, states):
        """Test that a message is raised when the density matrix is not positive semi def."""
        state0, state1 = states

        with pytest.raises(ValueError, match="The matrix is not positive semi"):
            qml.math.trace_distance(state0, state1, check_state=True)

        with pytest.raises(ValueError, match="The matrix is not positive semi"):
            qml.math.trace_distance(state1, state0, check_state=True)

    d_mat_different_wires = [
        (([[1, 0], [0, 0]], [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])),
        (([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[1, 0], [0, 0]])),
        (([[[1, 0], [0, 0]]], [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])),
        (([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[[1, 0], [0, 0]]])),
    ]

    @pytest.mark.parametrize("states", d_mat_different_wires)
    def test_same_number_wires(self, states):
        """Test that the two states must act on the same number of wires"""
        state0, state1 = states

        with pytest.raises(ValueError, match="The two states must have the same number of wires"):
            qml.math.trace_distance(state0, state1, check_state=True)

        with pytest.raises(ValueError, match="The two states must have the same number of wires"):
            qml.math.trace_distance(state1, state0, check_state=True)

    d_mat_different_batch_sizes = [
        (
            (
                [[[1, 0], [0, 0]], [[1, 0], [0, 0]]],
                [[[1, 0], [0, 0]], [[1, 0], [0, 0]], [[1, 0], [0, 0]]],
            )
        ),
        (
            (
                [[[1, 0], [0, 0]]],
                [[[1, 0], [0, 0]], [[1, 0], [0, 0]]],
            )
        ),
    ]

    @pytest.mark.parametrize("states", d_mat_different_batch_sizes)
    def test_different_batch_sizes(self, states):
        """Test that the two states have batches size that are compatible"""
        state0, state1 = states

        with pytest.raises(ValueError, match="The two states must be batches of the same size"):
            qml.math.trace_distance(state0, state1, check_state=True)

        with pytest.raises(ValueError, match="The two states must be batches of the same size"):
            qml.math.trace_distance(state1, state0, check_state=True)
