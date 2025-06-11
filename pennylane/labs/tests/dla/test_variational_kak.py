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
"""Tests for pennylane/labs/dla/variational_kak.py functionality"""
import numpy as np

# pylint: disable=too-few-public-methods, protected-access, no-self-use, import-outside-toplevel
import pytest

import pennylane as qml
from pennylane import X, Y, Z
from pennylane.labs.dla import orthonormalize, run_opt, validate_kak, variational_kak_adj
from pennylane.liealg import (
    cartan_decomp,
    check_cartan_decomp,
    concurrence_involution,
    horizontal_cartan_subalgebra,
)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_kak_Ising(n, dense):
    """Basic test for kak decomposition on Ising model"""
    gens = [X(i) @ X(i + 1) for i in range(n - 1)]
    gens += [Z(i) for i in range(n)]
    H = qml.sum(*gens)

    g = qml.lie_closure(gens, matrix=dense)
    if not dense:
        g = [op.pauli_rep for op in g]

    involution = concurrence_involution

    assert not involution(H)
    k, m = cartan_decomp(g, involution=involution)
    assert check_cartan_decomp(k, m)

    if not dense:
        g = k + m
        adj = qml.structure_constants(g)
    else:
        g = np.vstack([k, m])
        adj = qml.structure_constants(g, matrix=True)

    g, k, mtilde, h, adj = horizontal_cartan_subalgebra(k, m, adj, tol=1e-10, start_idx=0)

    dims = (len(k), len(mtilde), len(h))
    kak_res = variational_kak_adj(H, g, dims, adj, verbose=False)
    # Both the adjvec of the CSA element and the optimized parameters should be real
    assert kak_res[0].shape == (len(mtilde) + len(h),)
    assert kak_res[0].dtype == np.float64
    assert kak_res[1].shape == (len(k),)
    assert kak_res[1].dtype == np.float64
    assert validate_kak(H, g, k, kak_res, n, 1e-6)


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("dense", [False, True])
def test_kak_Heisenberg(n, dense):
    """Basic test for kak decomposition on Heisenberg model"""
    gens = [X(i) @ X(i + 1) for i in range(n - 1)]
    gens += [Y(i) @ Y(i + 1) for i in range(n - 1)]
    gens += [Z(i) @ Z(i + 1) for i in range(n - 1)]
    H = qml.sum(*gens)

    g = qml.lie_closure(gens, matrix=dense)
    if not dense:
        g = [op.pauli_rep for op in g]

    involution = concurrence_involution

    assert not involution(H)
    k, m = cartan_decomp(g, involution=involution)
    assert check_cartan_decomp(k, m)

    if not dense:
        g = k + m
        adj = qml.structure_constants(g)
    else:
        g = np.vstack([k, m])
        adj = qml.structure_constants(g, matrix=True)

    g, k, mtilde, h, adj = horizontal_cartan_subalgebra(k, m, adj, tol=1e-10, start_idx=0)

    dims = (len(k), len(mtilde), len(h))
    kak_res = variational_kak_adj(H, g, dims, adj, verbose=False)
    # Both the adjvec of the CSA element and the optimized parameters should be real
    assert kak_res[0].shape == (len(mtilde) + len(h),)
    assert kak_res[0].dtype == np.float64
    assert kak_res[1].shape == (len(k),)
    assert kak_res[1].dtype == np.float64
    assert validate_kak(H, g, k, kak_res, n, 1e-6)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("is_orthogonal", [True, False])
def test_kak_Heisenberg_summed(is_orthogonal, dense):
    """Basic test for kak decomposition on summed Heisenberg model"""
    n = 4
    gens = [X(i) @ X(i + 1) + Y(i) @ Y(i + 1) + Z(i) @ Z(i + 1) for i in range(n - 1)]
    H = qml.sum(*gens)

    g = qml.lie_closure(gens, matrix=dense)
    if not dense:
        g = [op.pauli_rep for op in g]

    if is_orthogonal:
        g = orthonormalize(g)

    involution = concurrence_involution

    assert not involution(H)
    k, m = cartan_decomp(g, involution=involution)
    assert check_cartan_decomp(k, m)

    if not dense:
        g = k + m
        adj = qml.structure_constants(g, is_orthogonal=is_orthogonal)
    else:
        g = np.vstack([k, m])
        adj = qml.structure_constants(g, matrix=True, is_orthogonal=is_orthogonal)

    g, k, mtilde, h, adj = horizontal_cartan_subalgebra(
        k, m, adj, tol=1e-10, start_idx=0, is_orthogonal=is_orthogonal
    )

    dims = (len(k), len(mtilde), len(h))
    kak_res = variational_kak_adj(H, g, dims, adj, verbose=False)
    # Both the adjvec of the CSA element and the optimized parameters should be real
    assert kak_res[0].shape == (len(mtilde) + len(h),)
    assert kak_res[0].dtype == np.float64
    assert kak_res[1].shape == (len(k),)
    assert kak_res[1].dtype == np.float64
    assert validate_kak(H, g, k, kak_res, n, 1e-6)


@pytest.mark.jax
def test_run_opt_with_other_optimizer():
    """Test that run_opt works with alternate optimizer"""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import optax

    def cost(x):
        return x**2

    x0 = jnp.array(0.4)

    optimizer = optax.lbfgs(learning_rate=0.1, memory_size=1000)
    thetas, energy, _ = run_opt(cost, x0, optimizer=optimizer)

    assert np.isclose(thetas[-1], 0)
    assert np.isclose(energy[-1], 0)
