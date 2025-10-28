# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test vibronic norm functions."""


import numpy as np

from pennylane.labs.trotter_error.realspace import (
    RealspaceCoeffs,
    RealspaceMatrix,
    RealspaceOperator,
    RealspaceSum,
)

from pennylane.labs.trotter_error.vibronic_norm import (
    vibronic_norm,
    _block_norm,
    _get_eigenvalue,
    build_mat,
)


def test_vibronic_norm():
    """Test that the vibronic_norm function computes the correct norm."""
    n_states = 1
    n_modes = 5
    gridpoints = 2
    op1 = RealspaceOperator(n_modes, (), RealspaceCoeffs(np.array(1)))
    op2 = RealspaceOperator(n_modes, ("Q"), RealspaceCoeffs(np.array([1, 2, 3, 4, 5]), label="phi"))
    rs_sum = RealspaceSum(n_modes, [op1, op2])
    rs_mat = RealspaceMatrix(n_states, n_modes, {(0, 0): rs_sum})

    norm_computed = vibronic_norm(rs_mat, gridpoints)
    norm_expected = np.array(
        max(np.linalg.eigvals(rs_mat.matrix(basis="harmonic", gridpoints=gridpoints)))
    )

    assert np.allclose(norm_computed, norm_expected)


def test_block_norm():
    """Test that the _block_norm function computes the correct norm."""
    n_modes = 5
    gridpoints = 2
    op1 = RealspaceOperator(n_modes, (), RealspaceCoeffs(np.array(1)))
    op2 = RealspaceOperator(n_modes, ("Q"), RealspaceCoeffs(np.array([1, 2, 3, 4, 5]), label="phi"))
    rs_sum = RealspaceSum(n_modes, [op1, op2])

    norm_computed = _block_norm(rs_sum, gridpoints)
    norm_expected = np.array(
        max(np.linalg.eigvals(rs_sum.matrix(basis="harmonic", gridpoints=gridpoints)))
    )

    assert np.allclose(norm_computed, norm_expected)


def test_get_eigenvalue():
    """Test that the _get_eigenvalue function computes the correct eigenvalue."""
    group_ops = {
        "ops": [("QPP",), ("QQ",), ("Q",), ("PPQ",)],
        "coeffs": [(0.01), (0.02j), (0.03), (0.04j)],
    }

    computed = _get_eigenvalue(group_ops, 4)
    expected = 0.15442961739479524

    assert np.allclose(computed, expected)


def test_build_mat():
    """Test that the build_mat function computes the correct matrix."""
    ops = ("QQ", "QQPP")
    gridpoints = 4

    computed = build_mat(ops, gridpoints)
    expected = 0.15442961739479524

    assert np.allclose(computed, expected)
