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
"""Unit tests for differentiable quantum entropies."""
# pylint: disable=too-many-arguments
import pytest

import pennylane as qml
from pennylane import numpy as np

pytestmark = pytest.mark.all_interfaces

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")


class TestPurity:
    """Tests for computing the purity of a given state"""

    density_matrices = [
        ([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]], 1 / 2, 1),
        ([[1 / 2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1 / 2]], 1 / 2, 1 / 2),
        ([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 1, 1),
    ]

    single_wires_list = [
        [0],
        [1],
    ]

    full_wires_list = [[0, 1]]

    check_state = [True, False]

    @pytest.mark.parametrize("density_matrix,subsystems_purity,_", density_matrices)
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("interface", [None, "autograd", "jax", "torch"])
    def test_density_matrices_purity_single_wire(
        self, density_matrix, wires, check_state, subsystems_purity, _, interface
    ):
        """Test purity for different density matrices."""

        if interface:
            density_matrix = qml.math.asarray(density_matrix, like=interface)

        purity = qml.math.purity(density_matrix, wires, check_state=check_state)
        assert qml.math.allclose(purity, subsystems_purity)

    @pytest.mark.parametrize("density_matrix,_,full_purity", density_matrices)
    @pytest.mark.parametrize("wires", full_wires_list)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("interface", [None, "autograd", "jax", "torch"])
    def test_density_matrices_purity_full_wire(
        self, density_matrix, wires, check_state, _, full_purity, interface
    ):
        """Test purity for different density matrices."""

        if interface:
            density_matrix = qml.math.asarray(density_matrix, like=interface)

        purity = qml.math.purity(density_matrix, wires, check_state=check_state)
        assert qml.math.allclose(purity, full_purity)


class TestVonNeumannEntropy:  # pylint: disable=too-few-public-methods
    """Tests for creating a density matrix from state vectors."""

    single_wires_list = [
        [0],
        [1],
    ]

    base = [2, np.exp(1), 10]

    check_state = [True, False]

    density_matrices = [
        ([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]], False),
        ([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], True),
    ]

    @pytest.mark.parametrize("density_matrix,pure", density_matrices)
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("interface", [None, "autograd", "jax", "torch"])
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

    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
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
            (np.diag([1, 0, 0, 0]), [0], [0]),
            (np.diag([1, 0, 0, 0]), [0], [0, 1]),
            (np.diag([1, 0, 0, 0, 0, 0, 0, 0]), [0, 1], [1]),
            (np.diag([1, 0, 0, 0, 0, 0, 0, 0]), [0, 1], [1, 2]),
        ],
    )
    def test_subsystem_overlap(self, state, wires0, wires1):
        """Test that an error is raised when the subsystems overlap"""
        with pytest.raises(
            ValueError, match="Subsystems for computing mutual information must not overlap"
        ):
            qml.math.mutual_info(state, indices0=wires0, indices1=wires1)

    @pytest.mark.parametrize("state", [np.array([5])])
    def test_invalid_type(self, state):
        """Test that an error is raised when an unsupported type is passed"""
        with pytest.raises(ValueError, match="Density matrix must be of shape"):
            qml.math.mutual_info(state, indices0=[0], indices1=[1], check_state=True)


class TestEntanglementEntropy:
    """Tests for the vn entanglement entropy function"""

    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    @pytest.mark.parametrize(
        "state, expected",
        [
            (
                [
                    [0.25, 0.25, -0.25, -0.25],
                    [0.25, 0.25, -0.25, -0.25],
                    [-0.25, -0.25, 0.25, 0.25],
                    [-0.25, -0.25, 0.25, 0.25],
                ],
                0,
            ),
            ([[0, 0, 0, 0], [0, 0.5, -0.5, 0], [0, -0.5, 0.5, 0], [0, 0, 0, 0]], np.log(2)),
            (
                [
                    [1 / 1.25, 0, 0, 0.5 / 1.25],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0.5 / 1.25, 0, 0, 0.25 / 1.25],
                ],
                0.500402,
            ),
        ],
    )
    def test_density_matrix(self, interface, state, expected):
        """Test that mutual information works for density matrices"""
        state = qml.math.asarray(state, like=interface)
        actual = qml.math.vn_entanglement_entropy(state, indices0=[0], indices1=[1])
        assert qml.math.allclose(actual, expected)

    @pytest.mark.parametrize(
        "state, wires0, wires1",
        [
            (np.diag([1, 0, 0, 0]), [0], [0]),
            (np.diag([1, 0, 0, 0]), [0], [0, 1]),
            (np.diag([1, 0, 0, 0, 0, 0, 0, 0]), [0, 1], [1]),
            (np.diag([1, 0, 0, 0, 0, 0, 0, 0]), [0, 1], [1, 2]),
        ],
    )
    def test_subsystem_overlap(self, state, wires0, wires1):
        """Test that an error is raised when the subsystems overlap"""
        with pytest.raises(
            ValueError, match="Subsystems for computing the entanglement entropy must not overlap"
        ):
            qml.math.vn_entanglement_entropy(state, indices0=wires0, indices1=wires1)

    @pytest.mark.parametrize("state", [np.array([5])])
    def test_invalid_type(self, state):
        """Test that an error is raised when an unsupported type is passed"""
        with pytest.raises(ValueError, match="Density matrix must be of shape"):
            qml.math.vn_entanglement_entropy(state, indices0=[0], indices1=[1], check_state=True)


class TestRelativeEntropy:
    """Tests for the relative entropy qml.math function"""

    bases = [None, 2]
    check_states = [True, False]

    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
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
            (np.array([[1, 0], [0, 0]]), np.diag([0, 1, 0, 0])),
            (
                np.array([[1, 0], [0, 0]]),
                np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            ),
        ],
    )
    def test_size_mismatch(self, state0, state1):
        """Test that an error is raised when the dimensions do not match"""
        msg = "The two states must have the same number of wires"

        with pytest.raises(ValueError, match=msg):
            qml.math.relative_entropy(state0, state1)


class TestMaxEntropy:
    """Tests for computing the maximum entropy of a given state."""

    single_wires_list = [
        [0],
        [1],
    ]

    base = [2, np.exp(1), 10]

    check_state = [True, False]

    density_matrices = [
        ([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]], False),
        ([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], True),
    ]

    @pytest.mark.parametrize("density_matrix,pure", density_matrices)
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("interface", [None, "autograd", "jax", "torch"])
    def test_density_matrices_max_entropy(
        self, density_matrix, pure, wires, base, check_state, interface
    ):
        """Test maximum entropy for different density matrices."""
        if interface:
            density_matrix = qml.math.asarray(density_matrix, like=interface)

        entropy = qml.math.max_entropy(density_matrix, wires, base, check_state)

        if pure:
            expected_max_entropy = 0
        else:
            expected_max_entropy = np.log(2) / np.log(base)

        assert qml.math.allclose(entropy, expected_max_entropy)

    parameters = [
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]],
    ]

    @pytest.mark.autograd
    @pytest.mark.parametrize("params", parameters)
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_state)
    def test_max_entropy_grad(self, params, wires, base, check_state):
        """Test `max_entropy` differentiability with autograd."""
        params = np.tensor(params)

        gradient = qml.grad(qml.math.max_entropy)(params, wires, base, check_state)
        assert qml.math.allclose(gradient, 0.0)

    @pytest.mark.torch
    @pytest.mark.parametrize("params", parameters)
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_state)
    def test_max_entropy_grad_torch(self, params, wires, base, check_state):
        """Test `max_entropy` differentiability with torch interface."""
        params = torch.tensor(params, requires_grad=True)

        max_entropy = qml.math.max_entropy(params, wires, base, check_state)
        max_entropy.backward()
        gradient = params.grad

        assert qml.math.allclose(gradient, 0.0)

    @pytest.mark.jax
    @pytest.mark.parametrize("params", parameters)
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("jit", [False, True])
    def test_max_entropy_grad_jax(self, params, wires, base, check_state, jit):
        """Test `max_entropy` differentiability with jax."""

        jnp = jax.numpy
        jax.config.update("jax_enable_x64", True)

        params = jnp.array(params)

        max_entropy_grad = jax.grad(qml.math.max_entropy)

        if jit:
            gradient = jax.jit(max_entropy_grad, static_argnums=[1, 2, 3])(
                params, tuple(wires), base, check_state
            )
        else:
            gradient = max_entropy_grad(params, wires, base, check_state)

        assert qml.math.allclose(gradient, 0.0)


class TestMinEntropy:
    """Test for computing the minimum entropy of a given state."""

    single_wires_list = [
        [0],
        [1],
    ]

    base = [2, np.exp(1), 10]

    check_state = [True, False]

    density_matrices = [
        ([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]], False),
        ([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], True),
    ]

    @pytest.mark.parametrize("density_matrix,pure", density_matrices)
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("interface", [None, "autograd", "jax", "torch"])
    def test_density_matrices_min_entropy(
        self, density_matrix, pure, wires, base, check_state, interface
    ):
        """Test minimum entropy for different density matrices."""
        if interface:
            density_matrix = qml.math.asarray(density_matrix, like=interface)

        entropy = qml.math.min_entropy(density_matrix, wires, base, check_state)

        if pure:
            expected_min_entropy = 0
        else:
            expected_min_entropy = np.log(2) / np.log(base)

        assert qml.math.allclose(entropy, expected_min_entropy)

    parameters = [
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]],
    ]

    @pytest.mark.autograd
    @pytest.mark.parametrize("params", parameters)
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_state)
    def test_min_entropy_grad(self, params, wires, base, check_state):
        """Test `min_entropy` differentiability with autograd."""

        params = np.tensor(params)

        gradient = qml.grad(qml.math.min_entropy)(params, wires, base, check_state)
        assert qml.math.allclose(gradient, -np.eye(4) / np.log(base))

    @pytest.mark.torch
    @pytest.mark.parametrize("params", parameters)
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_state)
    def test_min_entropy_grad_torch(self, params, wires, base, check_state):
        """Test `min_entropy` differentiability with torch interface."""

        params = torch.tensor(params, requires_grad=True)

        min_entropy = qml.math.min_entropy(params, wires, base, check_state)
        min_entropy.backward()
        gradient = params.grad

        assert qml.math.allclose(gradient, -np.eye(4) / np.log(base))

    @pytest.mark.jax
    @pytest.mark.parametrize("params", parameters)
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("jit", [False, True])
    def test_min_entropy_grad_jax(self, params, wires, base, check_state, jit):
        """Test `min_entropy` differentiability with jax."""

        jnp = jax.numpy

        jax.config.update("jax_enable_x64", True)  # Enabling complex128 datatypes for jax

        params = jnp.array(params)

        min_entropy_grad = jax.grad(qml.math.min_entropy)

        if jit:
            gradient = jax.jit(min_entropy_grad, static_argnums=[1, 2, 3])(
                params, tuple(wires), base, check_state
            )
        else:
            gradient = min_entropy_grad(params, wires, base, check_state)

        assert qml.math.allclose(gradient, -np.eye(4) / np.log(base))


class TestEntropyBroadcasting:
    """Test that broadcasting works as expected for the entropy functions"""

    single_wires_list = [[0], [1]]
    full_wires_list = [[0, 1]]
    check_state = [True, False]

    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("interface", [None, "autograd", "jax", "torch"])
    def test_purity_broadcast_dm(self, wires, check_state, interface):
        """Test broadcasting for purity and density matrices"""
        density_matrix = [
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]],
            [[1 / 2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1 / 2]],
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
        expected = [0.5, 0.5, 1]

        if interface:
            density_matrix = qml.math.asarray(density_matrix, like=interface)

        purity = qml.math.purity(density_matrix, wires, check_state=check_state)
        assert qml.math.allclose(purity, expected)

    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("interface", [None, "autograd", "jax", "torch"])
    def test_vn_entropy_broadcast_dm(self, wires, check_state, interface):
        """Test broadcasting for vn_entropy and density matrices"""
        density_matrix = [
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]],
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
        expected = [np.log(2), 0]

        if interface:
            density_matrix = qml.math.asarray(density_matrix, like=interface)

        entropy = qml.math.vn_entropy(density_matrix, wires, check_state=check_state)
        assert qml.math.allclose(entropy, expected)

    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    def test_mutual_info_broadcast_dm(self, interface):
        """Test broadcasting for mutual_info and density matrices"""
        state = [
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]],
            [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
            np.ones((4, 4)) * 0.25,
        ]
        expected = [0, 0, 2 * np.log(2), 0]

        state = qml.math.asarray(state, like=interface)

        actual = qml.math.mutual_info(state, indices0=[0], indices1=[1])
        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    @pytest.mark.parametrize("check_state", check_state)
    def test_relative_entropy_broadcast_dm(self, interface, check_state):
        """Test broadcasting for relative entropy and density matrices"""
        state0 = [[[1, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 1]], [[1, 0], [0, 0]]]
        state1 = [[[0, 0], [0, 1]], np.ones((2, 2)) / 2, [[0, 0], [0, 1]], np.eye(2) / 2]
        expected = [np.inf, np.inf, 0, np.log(2)]

        state0 = qml.math.asarray(state0, like=interface)
        state1 = qml.math.asarray(state1, like=interface)

        actual = qml.math.relative_entropy(state0, state1, check_state=check_state)
        assert np.allclose(actual, expected, rtol=1e-06, atol=1e-07)

    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    @pytest.mark.parametrize("check_state", check_state)
    def test_relative_entropy_broadcast_dm_unbatched(self, interface, check_state):
        """Test broadcasting for relative entropy and density matrices when one input is unbatched"""
        state0 = [[[0, 0], [0, 1]], np.ones((2, 2)) / 2, [[1, 0], [0, 0]], np.eye(2) / 2]
        state1 = np.eye(2) / 2
        expected = [np.log(2), np.log(2), np.log(2), 0]

        state0 = qml.math.asarray(state0, like=interface)
        state1 = qml.math.asarray(state1, like=interface)

        actual = qml.math.relative_entropy(state0, state1, check_state=check_state)
        assert np.allclose(actual, expected, rtol=1e-06, atol=1e-07)

    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("check_state", check_state)
    @pytest.mark.parametrize("interface", [None, "autograd", "jax", "torch"])
    def test_max_entropy_broadcast_dm(self, wires, check_state, interface):
        """Test broadcasting for max entropy and density matrices"""
        density_matrix = [
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]],
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
        expected = [np.log(2), 0]

        if interface:
            density_matrix = qml.math.asarray(density_matrix, like=interface)

        entropy = qml.math.max_entropy(density_matrix, wires, check_state=check_state)
        assert qml.math.allclose(entropy, expected)
