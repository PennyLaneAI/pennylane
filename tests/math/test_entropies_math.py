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
import pennylane
from pennylane.math.utils import cast

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
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("interface", [None, "autograd", "jax", "tensorflow", "torch"])
    def test_state_vector_entropy_without_base(
        self, state_vector, wires, check_state, pure, interface
    ):
        """Test entropy for different state vectors without base for log."""
        if interface:
            state_vector = qml.math.asarray(state_vector, like=interface)

        entropy = qml.math.vn_entropy(state_vector, wires, check_state=check_state)

        if pure:
            expected_entropy = 0
        else:
            expected_entropy = np.log(2)
        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("state_vector,pure", state_vector)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("interface", [None, "autograd", "jax", "tensorflow", "torch"])
    def test_state_vector_entropy(self, state_vector, wires, base, check_state, pure, interface):
        """Test entropy for different state vectors."""
        if interface:
            state_vector = qml.math.asarray(state_vector, like=interface)
        entropy = qml.math.vn_entropy(state_vector, wires, base, check_state)

        if pure:
            expected_entropy = 0
        else:
            expected_entropy = np.log(2) / np.log(base)

        assert qml.math.allclose(entropy, expected_entropy)

    density_matrices = [
        ([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]], False),
        ([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], True),
    ]

    @pytest.mark.parametrize("density_matrix,pure", density_matrices)
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("interface", [None, "autograd", "jax", "tensorflow", "torch"])
    def test_density_matrices_entropy(
        self, density_matrix, pure, wires, base, check_state, interface
    ):
        """Test entropy for different density matrices."""
        if interface:
            density_matrix = qml.math.asarray(density_matrix, like=interface)
        entropy = qml.math.vn_entropy(density_matrix, wires, base, check_state)

        if pure:
            expected_entropy = 0
        else:
            expected_entropy = np.log(2) / np.log(base)

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
        actual = qml.math.mutual_info(state, indices0=[0], indices1=[1])
        assert np.allclose(actual, expected, rtol=1e-06, atol=1e-07)

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
        actual = qml.math.mutual_info(state, indices0=[0], indices1=[1])

        assert np.allclose(actual, expected)

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
            qml.math.mutual_info(state, indices0=wires0, indices1=wires1)

    @pytest.mark.parametrize("state", [np.array(5), np.ones((3, 4)), np.ones((2, 2, 2))])
    def test_invalid_type(self, state):
        """Test that an error is raised when an unsupported type is passed"""
        with pytest.raises(
            ValueError, match="The state is not a state vector or a density matrix."
        ):
            qml.math.mutual_info(state, indices0=[0], indices1=[1])


class TestRelativeEntropy:
    """Tests for the relative entropy qml.math function"""

    bases = [None, 2]
    check_states = [True, False]

    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize(
        "state0, state1, expected",
        [([1, 0], [0, 1], np.inf), ([1, 0], [1, 1] / np.sqrt(2), np.inf), ([0, 1], [0, 1], 0)],
    )
    @pytest.mark.parametrize("base", bases)
    @pytest.mark.parametrize("check_state", check_states)
    def test_state(self, interface, state0, state1, expected, base, check_state):
        """Test that mutual information works for states"""
        state0 = qml.math.asarray(state0, like=interface)
        state1 = qml.math.asarray(state1, like=interface)
        actual = qml.math.relative_entropy(state0, state1, base=base, check_state=check_state)

        div = 1 if base is None else np.log(base)
        assert np.allclose(actual, expected / div, rtol=1e-06, atol=1e-07)

    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize(
        "state0, state1, expected",
        [
            ([[1, 0], [0, 0]], [[0, 0], [0, 1]], np.inf),
            ([[1, 0], [0, 0]], np.ones((2, 2)) / 2, np.inf),
            ([[0, 0], [0, 1]], [[0, 0], [0, 1]], 0),
            ([[1, 0], [0, 0]], np.eye(2) / 2, np.log(2)),
        ],
    )
    @pytest.mark.parametrize("base", bases)
    @pytest.mark.parametrize("check_state", check_states)
    def test_density_matrix(self, interface, state0, state1, expected, base, check_state):
        """Test that mutual information works for density matrices"""
        state0 = qml.math.asarray(state0, like=interface)
        state1 = qml.math.asarray(state1, like=interface)
        actual = qml.math.relative_entropy(state0, state1, base=base, check_state=check_state)

        div = 1 if base is None else np.log(base)
        assert np.allclose(actual, expected / div, rtol=1e-06, atol=1e-07)

    @pytest.mark.parametrize(
        "state0, state1",
        [
            (np.array([1, 0, 0, 0]), np.array([1, 0])),
            (np.array([[1, 0], [0, 0]]), np.array([0, 1, 0, 0])),
            (
                np.array([[1, 0], [0, 0]]),
                np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            ),
        ],
    )
    def test_size_mismatch(self, state0, state1):
        """Test that an error is raised when the dimensions do not match"""
        msg = "The two states must have the same number of wires"

        with pytest.raises(qml.QuantumFunctionError, match=msg):
            qml.math.relative_entropy(state0, state1)


class TestMinEntropy:
    """Tests for the min entropy qml.math function"""

    base = [None, 2]
    check_states = [True, False]

    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize(
        "state, expected",
        [
            ([1, 0], -np.log(1)),
            ([0, 1], -np.log(1)),
            ([1, 1] / np.sqrt(2), -np.log(1)),
            ([1, -1] / np.sqrt(2), -np.log(1)),
            # ([1 / np.sqrt(3), np.sqrt(2 / 3)], -(np.sqrt(56249041) + 7500) / 15000),
        ],
    )
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_states)
    def test_state(self, interface, state, expected, base, check_state):
        """Test min_entropy function"""
        state = qml.math.asarray(state, like=interface)
        actual = qml.math.min_entropy(state, base=base, check_state=check_state)

        div = 1 if base is None else np.log(base)
        assert np.allclose(actual, expected / div, rtol=1e-06, atol=1e-07)

    # Testing Differentiabilty

    @pytest.mark.autograd
    @pytest.mark.parametrize(
        "state, expected",
        [
            ([1, 0], -1),
            ([0, 1], -1),
            ([1, 1] / np.sqrt(2), -1),
            ([1, -1] / np.sqrt(2), -1),
            # (
            #     [1 / np.sqrt(3), np.sqrt(2 / 3)],
            #     -1 / ((np.sqrt(56249041) + 7500) / 15000),
            # ),
        ],
    )
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_states)
    def test_grad(self, state, expected, base, check_state):
        """Test that the gradient of min_entropy works
        with the autograd interface"""

        c_dtype = "complex128"
        state = qml.math.asarray(state, like="autograd")
        state = cast(state, dtype=c_dtype)
        grad = qml.grad(qml.math.min_entropy)(state, base=base, check_state=check_state)

        div = 1 if base is None else np.log(base)
        assert np.allclose(grad, expected / div, rtol=1e-06, atol=1e-07)

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "state, expected",
        [
            ([1, 0], -1),
            ([0, 1], -1),
            ([1, 1] / np.sqrt(2), -1),
            ([1, -1] / np.sqrt(2), -1),
            # (
            #     [1 / np.sqrt(3), np.sqrt(2 / 3)],
            #     -1 / ((np.sqrt(56249041) + 7500) / 15000),
            # ),
        ],
    )
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_states)
    def test_grad_jax(self, state, expected, base, check_state):
        """Test that the gradient of min_entropy works
        with the JAX interface"""

        import jax

        c_dtype = "complex128"
        state = qml.math.asarray(state, like="jax")
        state = cast(state, dtype=c_dtype)
        grad = jax.grad(qml.math.min_entropy)(state, base=base, check_state=check_state)

        div = 1 if base is None else np.log(base)
        assert np.allclose(grad, expected / div, rtol=1e-06, atol=1e-07)

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "state, expected",
        [
            ([1, 0], -1),
            ([0, 1], -1),
            ([1, 1] / np.sqrt(2), -1),
            ([1, -1] / np.sqrt(2), -1),
            # (
            #     [1 / np.sqrt(3), np.sqrt(2 / 3)],
            #     -1 / ((np.sqrt(56249041) + 7500) / 15000),
            # ),
        ],
    )
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_states)
    def test_grad_jaxjit(self, state, expected, base, check_state):
        """Test that the gradient of min_entropy works
        with the JAX-jit interface"""

        import jax

        c_dtype = "complex128"
        state = qml.math.asarray(state, like="jax")
        state = cast(state, dtype=c_dtype)
        grad = jax.jit(jax.grad(qml.math.min_entropy))(state, base=base, check_state=check_state)

        div = 1 if base is None else np.log(base)
        assert np.allclose(grad, expected / div, rtol=1e-06, atol=1e-07)

    @pytest.mark.tf
    @pytest.mark.parametrize(
        "state, expected",
        [
            ([1, 0], -1),
            ([0, 1], -1),
            ([1, 1] / np.sqrt(2), -1),
            ([1, -1] / np.sqrt(2), -1),
            # (
            #     [1 / np.sqrt(3), np.sqrt(2 / 3)],
            #     -1 / ((np.sqrt(56249041) + 7500) / 15000),
            # ),
        ],
    )
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_states)
    def test_grad_tensorflow(self, state, expected, base, check_state):
        """Test that the gradient of min_entropy works
        with the tensorflow interface"""

        import tensorflow as tf

        state = qml.math.asarray(state, like="tensorflow")
        entropy = qml.math.min_entropy(state, base=base, check_state=check_state)

        state = tf.Variable(state)
        with tf.GradientTape() as tape:
            grad_entropy = tape.gradient(entropy, state)

        div = 1 if base is None else np.log(base)
        assert np.allclose(grad_entropy, expected / div, rtol=1e-06, atol=1e-07)

    @pytest.mark.torch
    @pytest.mark.parametrize(
        "state, expected",
        [
            ([1, 0], -1),
            ([0, 1], -1),
            ([1, 1] / np.sqrt(2), -1),
            ([1, -1] / np.sqrt(2), -1),
            # (
            #     [1 / np.sqrt(3), np.sqrt(2 / 3)],
            #     -1 / ((np.sqrt(56249041) + 7500) / 15000),
            # ),
        ],
    )
    @pytest.mark.parametrize("base", base)
    def test_grad_torch(self, state, expected, base):
        """Test that the gradient of min_entropy works
        with the torch interface"""

        import torch

        c_dtype = "complex128"
        state = qml.math.asarray(state, like="torch")

        state = torch.tensor(state, dtype=torch.float64, requires_grad=True)
        state = cast(state, dtype=c_dtype)
        grad_entropy = state.grad

        div = 1 if base is None else np.log(base)
        assert np.allclose(grad_entropy, expected / div, rtol=1e-06, atol=1e-07)
