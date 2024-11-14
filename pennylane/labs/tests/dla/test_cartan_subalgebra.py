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
    cartan_decomposition,
    cartan_subalgebra,
    check_cartan_decomp,
    even_odd_involution,
    op_to_adjvec,
)
from pennylane.labs.dla.cartan_subalgebra import _change_basis_adj


class TestChangeBasisAdj:

    def test_permutation(self):
        """Test that a permutation is accounted for correctly."""
        ops = [qml.X(0), qml.Y(1), qml.Y(0) @ qml.Z(1), qml.X(1)]
        dla = qml.lie_closure(ops)
        adj = qml.structure_constants(dla)
        perm = np.random.permutation(len(dla))
        permuted_dla = [dla[i] for i in perm]
        permuted_adj = qml.structure_constants(permuted_dla)

        new_adj = _change_basis_adj(adj, op_to_adjvec(dla, dla), op_to_adjvec(permuted_dla, dla))
        assert np.allclose(new_adj, permuted_adj)

    def test_tiny_skewed_basis(self):
        """Test that changing from a tiny orthonormal basis to a skewed basis works."""
        dla = [qml.X(0), qml.Y(0), qml.Z(0)]
        adj = qml.structure_constants(dla)
        coeffs = np.random.random((len(dla), len(dla)))
        skewed_dla = [qml.sum(*(c * op for c, op in zip(_coeffs, dla))) for _coeffs in coeffs]
        skewed_adj = qml.structure_constants(skewed_dla)

        new_adj = _change_basis_adj(adj, op_to_adjvec(dla, dla), op_to_adjvec(skewed_dla, dla))
        assert np.allclose(new_adj, skewed_adj)

    def test_tiny_skewed_basis_from_non_ortho(self):
        """Test that changing from a tiny non-orthonormal basis to a skewed basis works."""
        ortho_dla = [qml.X(0), qml.Y(0), qml.Z(0)]  # only used to create adj rep.
        dla = [0.2 * qml.X(0) - 0.6 * qml.Y(0), 0.4 * qml.Y(0) + 0.9 * qml.Z(0), qml.Z(0)]

        adj = qml.structure_constants(dla)
        coeffs = np.random.random((len(dla), len(dla)))
        skewed_dla = [qml.sum(*(c * op for c, op in zip(_coeffs, dla))) for _coeffs in coeffs]
        skewed_adj = qml.structure_constants(skewed_dla)

        new_adj = _change_basis_adj(
            adj, op_to_adjvec(dla, ortho_dla), op_to_adjvec(skewed_dla, ortho_dla)
        )
        assert np.allclose(new_adj, skewed_adj)

    def test_skewed_basis(self):
        """Test that changing from an orthonormal basis to a skewed basis works."""
        ops = [qml.X(0), qml.Y(1), qml.Y(0) @ qml.Z(1)]
        dla = qml.lie_closure(ops)
        adj = qml.structure_constants(dla)
        coeffs = np.random.random((len(dla), len(dla)))
        skewed_dla = [qml.sum(*(c * op for c, op in zip(_coeffs, dla))) for _coeffs in coeffs]
        skewed_adj = qml.structure_constants(skewed_dla)

        new_adj = _change_basis_adj(adj, op_to_adjvec(dla, dla), op_to_adjvec(skewed_dla, dla))
        assert np.allclose(new_adj, skewed_adj)

    def test_skewed_basis_from_non_ortho(self):
        """Test that changing from a non-orthonormal basis to a skewed basis works."""
        ops = [qml.X(0), qml.Y(1), qml.Y(0) @ qml.Z(1)]
        ortho_dla = qml.lie_closure(ops)  # only used to create adj rep.

        coeffs = np.random.random((len(ortho_dla), len(ortho_dla)))
        dla = [qml.sum(*(c * op for c, op in zip(_coeffs, ortho_dla))) for _coeffs in coeffs]
        adj = qml.structure_constants(dla)

        coeffs = np.random.random((len(dla), len(dla)))
        skewed_dla = [qml.sum(*(c * op for c, op in zip(_coeffs, dla))) for _coeffs in coeffs]
        skewed_adj = qml.structure_constants(skewed_dla)

        new_adj = _change_basis_adj(adj, op_to_adjvec(dla, dla), op_to_adjvec(skewed_dla, dla))
        assert np.allclose(new_adj, skewed_adj)


def test_Ising2():
    """Test Cartan subalgebra of 2 qubit Ising model"""
    gens = [X(0) @ X(1), Z(0), Z(1)]
    gens = [op.pauli_rep for op in gens]
    g = qml.lie_closure(gens, pauli=True)

    k, m = cartan_decomposition(g, even_odd_involution)
    assert check_cartan_decomp(k, m)

    g = k + m

    adj = qml.structure_constants(g)

    newg, k, mtilde, h, new_adj = cartan_subalgebra(g, k, m, adj, start_idx=0)
    assert len(h) == 2
    assert len(mtilde) == 2
    assert len(h) + len(mtilde) == len(m)

    new_adj_re = qml.structure_constants(newg)

    assert np.allclose(new_adj_re, new_adj)


def test_op_to_adjvec_dense():
    """Basic test of op_to_adjvec with dense matrices"""
    basis = [qml.matrix(op, wire_order=range(2)) for op in [X(0), X(1)]]
    ops = [X(0), X(1)]
    res = op_to_adjvec(ops, basis)
    assert np.allclose(res, np.eye(2))
