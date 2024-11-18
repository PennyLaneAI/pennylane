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

# pylint: disable=too-few-public-methods, protected-access, no-self-use
import pytest

import pennylane as qml
from pennylane import X, Y, Z
from pennylane.labs.dla import (
    cartan_decomposition,
    cartan_subalgebra,
    check_cartan_decomp,
    concurrence_involution,
    lie_closure_dense,
    orthonormalize,
    structure_constants_dense,
    validate_kak,
    variational_kak,
)


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("dense", [False, True])
def test_kak_Ising(n, dense):
    """Basic test for khk decomposition on Ising model"""
    gens = [X(i) @ X(i + 1) for i in range(n - 1)]
    gens += [Z(i) for i in range(n)]
    H = qml.sum(*gens)

    if not dense:
        g = qml.lie_closure(gens)
        g = [op.pauli_rep for op in g]
    else:
        g = lie_closure_dense(gens)

    involution = concurrence_involution

    assert not involution(H)
    k, m = cartan_decomposition(g, involution=involution)
    assert check_cartan_decomp(k, m)

    if not dense:
        g = k + m
        adj = qml.structure_constants(g)
    else:
        g = np.vstack([k, m])
        adj = structure_constants_dense(g)

    g, k, mtilde, h, adj = cartan_subalgebra(g, k, m, adj, tol=1e-10, start_idx=0)

    dims = (len(k), len(mtilde), len(h))
    khk_res = variational_kak(H, g, dims, adj, verbose=False)
    assert validate_kak(H, g, k, khk_res, n, 1e-6)


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("dense", [False, True])
def test_kak_Heisenberg(n, dense):
    """Basic test for khk decomposition on Heisenberg model"""
    gens = [X(i) @ X(i + 1) for i in range(n - 1)]
    gens += [Y(i) @ Y(i + 1) for i in range(n - 1)]
    gens += [Z(i) @ Z(i + 1) for i in range(n - 1)]
    H = qml.sum(*gens)

    H = qml.sum(*gens)

    if not dense:
        g = qml.lie_closure(gens)
        g = [op.pauli_rep for op in g]
    else:
        g = lie_closure_dense(gens)

    involution = concurrence_involution

    assert not involution(H)
    k, m = cartan_decomposition(g, involution=involution)
    assert check_cartan_decomp(k, m)

    if not dense:
        g = k + m
        adj = qml.structure_constants(g)
    else:
        g = np.vstack([k, m])
        adj = structure_constants_dense(g)

    g, k, mtilde, h, adj = cartan_subalgebra(g, k, m, adj, tol=1e-10, start_idx=0)

    dims = (len(k), len(mtilde), len(h))
    khk_res = variational_kak(H, g, dims, adj, verbose=False)
    assert validate_kak(H, g, k, khk_res, n, 1e-6)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("is_orthogonal", [True, False])
def test_kak_Heisenberg_summed(is_orthogonal, dense):
    """Basic test for khk decomposition on summed Heisenberg model"""
    n = 4
    gens = [X(i) @ X(i + 1) + Y(i) @ Y(i + 1) + Z(i) @ Z(i + 1) for i in range(n - 1)]
    H = qml.sum(*gens)

    if not dense:
        g = qml.lie_closure(gens)
        g = [op.pauli_rep for op in g]
    else:
        g = lie_closure_dense(gens)

    if is_orthogonal:
        g = orthonormalize(g)

    involution = concurrence_involution

    assert not involution(H)
    k, m = cartan_decomposition(g, involution=involution)
    assert check_cartan_decomp(k, m)

    if not dense:
        g = k + m
        adj = qml.structure_constants(g, is_orthogonal=is_orthogonal)
    else:
        g = np.vstack([k, m])
        adj = structure_constants_dense(g, is_orthonormal=is_orthogonal)

    g, k, mtilde, h, adj = cartan_subalgebra(
        g, k, m, adj, tol=1e-10, start_idx=0, is_orthogonal=is_orthogonal
    )

    dims = (len(k), len(mtilde), len(h))
    khk_res = variational_kak(H, g, dims, adj, verbose=False)
    assert validate_kak(H, g, k, khk_res, n, 1e-6)
