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
"""Tests for pennylane/dla/lie_closure_dense.py functionality"""
import numpy as np

# pylint: disable=no-self-use,too-few-public-methods,missing-class-docstring
import pytest

import pennylane as qml
from pennylane import X, Z
from pennylane.labs.dla import (
    cartan_decomp,
    cartan_subalgebra,
    check_cartan_decomp,
    even_odd_involution,
)


@pytest.mark.parametrize("n, len_g, len_h, len_mtilde", [(2, 6, 2, 2), (3, 15, 2, 6)])
def test_Ising(n, len_g, len_h, len_mtilde):
    """Test Cartan subalgebra of 2 qubit Ising model"""
    gens = [X(w) @ X(w + 1) for w in range(n - 1)] + [Z(w) for w in range(n)]
    gens = [op.pauli_rep for op in gens]
    g = qml.lie_closure(gens, pauli=True)

    k, m = cartan_decomp(g, even_odd_involution)
    assert check_cartan_decomp(k, m)

    g = k + m
    assert len(g) == len_g

    adj = qml.structure_constants(g)

    newg, k, mtilde, h, new_adj = cartan_subalgebra(g, k, m, adj, start_idx=0)
    assert len(h) == len_h
    assert len(mtilde) == len_mtilde
    assert len(h) + len(mtilde) == len(m)

    new_adj_re = qml.structure_constants(newg)

    assert np.allclose(new_adj_re, new_adj)
