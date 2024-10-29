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
"""Tests for pennylane/labs/dla/lie_closure_dense.py functionality"""
# pylint: disable=too-few-public-methods, protected-access, no-self-use
import pytest

import pennylane as qml
from pennylane import X, Z
from pennylane.labs.dla import (
    cartan_decomposition,
    cartan_subalgebra,
    check_cartan_decomp,
    concurrence_involution,
    validate_kak,
    variational_kak,
)


@pytest.mark.parametrize("n", [2, 3])
def test_khk_Ising2(n):
    """Basic test for khk decomposition on Ising model with two qubits"""
    gens = [X(i) @ X(i + 1) for i in range(n - 1)]
    gens += [Z(i) for i in range(n)]
    H = qml.sum(*gens)

    g = qml.lie_closure(gens)
    g = [op.pauli_rep for op in g]

    involution = concurrence_involution

    assert not involution(H)
    k, m = cartan_decomposition(g, involution=involution)
    assert check_cartan_decomp(k, m)

    g = k + m
    adj = qml.structure_constants(g)

    g, k, mtilde, h, adj = cartan_subalgebra(g, k, m, adj, tol=1e-14, start_idx=0)

    dims = (len(k), len(mtilde), len(h))
    khk_res = variational_kak(H, g, dims, adj, verbose=False)
    assert validate_kak(H, g, k, khk_res, n, 1e-6)
