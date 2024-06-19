# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the ParametrizedHamiltonianPytree class
"""
# pylint: disable=import-outside-toplevel
import numpy as np
import pytest

import pennylane as qml

try:
    from pennylane.pulse.hardware_hamiltonian import _reorder_parameters, amplitude_and_phase, drive
    from pennylane.pulse.parametrized_hamiltonian_pytree import (
        LazyDotPytree,
        ParametrizedHamiltonianPytree,
    )
except ImportError:
    pass

# if this fails, test file will be skipped
jnp = pytest.importorskip("jax.numpy")


def f1(p, t):
    """Compute the function p * sin(t) * (t - 1)."""
    return p * np.sin(t) * (t - 1)


def f2(p, t):
    """Compute the function p * cos(t**2)."""
    return np.cos(p * t**2)


PH = qml.dot([1, f1, f2], [qml.PauliX(0), qml.PauliY(1), qml.Hadamard(3)])

RH = drive(amplitude=f1, phase=f2, wires=[0, 1, 2])
RH += qml.dot([1.0], [qml.PauliZ(0)])

# Hamiltonians and the parameters for the individual coefficients
HAMILTONIANS_WITH_COEFF_PARAMETERS = [
    (PH, None, [f1, f2], [1.2, 2.3]),
    (
        RH,
        _reorder_parameters,
        [amplitude_and_phase(jnp.cos, f1, f2), amplitude_and_phase(jnp.sin, f1, f2)],
        [[1.2, 2.3], [1.2, 2.3]],
    ),
]


@pytest.mark.jax
class TestParametrizedHamiltonianPytree:
    """Unit tests for the ParametrizedHamiltonianPytree class."""

    @pytest.mark.parametrize("H, fn, coeffs_callable, params", HAMILTONIANS_WITH_COEFF_PARAMETERS)
    def test_attributes(self, H, fn, coeffs_callable, params):
        """Test that the attributes of the ParametrizedHamiltonianPytree class are initialized
        correctly."""
        from jax.experimental import sparse

        H_pytree = ParametrizedHamiltonianPytree.from_hamiltonian(
            H, dense=False, wire_order=[2, 3, 1, 0]
        )

        assert isinstance(H_pytree.mat_fixed, sparse.BCSR)
        assert isinstance(H_pytree.mats_parametrized, tuple)
        assert qml.math.allclose(
            [c(p, 2) for c, p in zip(H_pytree.coeffs_parametrized, params)],
            [c(p, 2) for c, p in zip(coeffs_callable, params)],
            atol=1e-7,
        )
        assert H_pytree.reorder_fn == fn

    def test_call_method_parametrized_hamiltonian(self):
        """Test that the call method works correctly."""
        H_pytree = ParametrizedHamiltonianPytree.from_hamiltonian(
            PH, dense=False, wire_order=[2, 3, 1, 0]
        )
        params = [1.2, 2.3]
        time = 4
        res = H_pytree(params, t=time)

        assert isinstance(res, LazyDotPytree)
        assert qml.math.allclose(res.coeffs, (1, f1(params[0], time), f2(params[1], time)))

    def test_call_method_rydberg_hamiltonian(self):
        """Test that the call method works correctly."""
        H_pytree = ParametrizedHamiltonianPytree.from_hamiltonian(
            RH, dense=False, wire_order=[2, 3, 1, 0]
        )
        params = [1.2, 2.3]
        time = 4
        res = H_pytree(params, t=time)

        assert isinstance(res, LazyDotPytree)
        assert qml.math.allclose(
            res.coeffs,
            (
                1.0,
                amplitude_and_phase(jnp.cos, f1, f2)(params, time),
                amplitude_and_phase(jnp.sin, f1, f2)(params, time),
            ),
        )

    @pytest.mark.parametrize("H, fn", [(PH, None), (RH, _reorder_parameters)])
    def test_flatten_method(self, H, fn):
        """Test the tree_flatten method."""
        H_pytree = ParametrizedHamiltonianPytree.from_hamiltonian(
            H, dense=False, wire_order=[2, 3, 1, 0]
        )

        flat_tree = H_pytree.tree_flatten()

        assert isinstance(flat_tree, tuple)
        assert flat_tree == (
            (H_pytree.mat_fixed, H_pytree.mats_parametrized),
            H_pytree.coeffs_parametrized,
            fn,
        )

    @pytest.mark.parametrize("H, fn", [(PH, None), (RH, _reorder_parameters)])
    def test_unflatten_method(self, H, fn):
        """Test the tree_unflatten method."""
        H_pytree = ParametrizedHamiltonianPytree.from_hamiltonian(
            H, dense=False, wire_order=[2, 3, 1, 0]
        )

        flat_tree = H_pytree.tree_flatten()

        new_H_pytree = H_pytree.tree_unflatten(flat_tree[1], flat_tree[0], flat_tree[2])

        assert new_H_pytree.mat_fixed == H_pytree.mat_fixed
        assert new_H_pytree.mats_parametrized == H_pytree.mats_parametrized
        assert new_H_pytree.coeffs_parametrized == H_pytree.coeffs_parametrized
        assert new_H_pytree.reorder_fn == fn


@pytest.mark.jax
class TestLazyDotPytree:
    """Unit tests for the LazyDotPytree class."""

    def test_initialization(self):
        """Test that a LazyDotPytree is initialized correctly."""
        coeffs = [1, 2, 3]
        ops = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
        mats = [qml.matrix(o) for o in ops]
        D = LazyDotPytree(coeffs=coeffs, mats=mats)

        assert D.coeffs == coeffs
        assert D.mats == mats

    def test_matmul(self):
        """Test the __matmul__ method."""
        coeffs = [1, 2, 3]
        ops = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
        mats = [qml.matrix(o) for o in ops]
        D = LazyDotPytree(coeffs=coeffs, mats=mats)

        another_matrix = qml.matrix(qml.PauliX(0))
        res = D @ another_matrix

        assert qml.math.allclose(res, qml.matrix(qml.dot(coeffs, ops)) @ another_matrix)

    def test_rmul(self):
        """Test the __rmul__ method"""

        coeffs = [1, 2, 3]
        ops = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
        mats = [qml.matrix(o) for o in ops]
        D = LazyDotPytree(coeffs=coeffs, mats=mats)

        assert isinstance(3 * D, LazyDotPytree)
        assert isinstance(D * 3, LazyDotPytree)

        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = jnp.array([[1], [2]]) * D

    def test_flatten_method(self):
        """Test the tree_flatten method."""
        coeffs = [1, 2, 3]
        ops = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
        mats = [qml.matrix(o) for o in ops]
        D = LazyDotPytree(coeffs=coeffs, mats=mats)

        flat_tree = D.tree_flatten()

        assert isinstance(flat_tree, tuple)
        assert flat_tree[0] == (D.coeffs, D.mats)
        assert flat_tree[1] is None

    def test_unflatten_method(self):
        """Test the tree_unflatten method."""
        coeffs = [1, 2, 3]
        ops = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
        mats = [qml.matrix(o) for o in ops]
        D = LazyDotPytree(coeffs=coeffs, mats=mats)

        flat_tree = D.tree_flatten()

        new_D = D.tree_unflatten(flat_tree[1], flat_tree[0])

        assert new_D.coeffs == D.coeffs
        assert new_D.mats == D.mats
