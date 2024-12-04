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
# pylint: disable=no-self-use,too-few-public-methods,missing-class-docstring
import numpy as np

import pennylane as qml
from pennylane import X, Z
from pennylane.labs.dla import (
    cartan_decomp,
    cartan_subalgebra,
    check_cartan_decomp,
    even_odd_involution,
)


def test_Ising2():
    """Test Cartan subalgebra of 2 qubit Ising model"""
    gens = [X(0) @ X(1), Z(0), Z(1)]
    gens = [op.pauli_rep for op in gens]
    g = qml.lie_closure(gens, pauli=True)

    k, m = cartan_decomp(g, even_odd_involution)
    assert check_cartan_decomp(k, m)

    g = k + m

    adj = qml.structure_constants(g)

    newg, k, mtilde, h, new_adj = cartan_subalgebra(g, k, m, adj, start_idx=0)
    assert len(h) == 2
    assert len(mtilde) == 2
    assert len(h) + len(mtilde) == len(m)

    new_adj_re = qml.structure_constants(newg)

    assert np.allclose(new_adj_re, new_adj)
