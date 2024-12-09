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
"""Tests for pennylane/labs/dla/dense_util.py functionality"""
# pylint: disable=no-self-use, unnecessary-lambda-assignment
import numpy as np
import pytest
from scipy.linalg import sqrtm

import pennylane as qml
from pennylane import I, X, Y, Z
from pennylane.labs.dla import (
    adjvec_to_op,
    batched_pauli_decompose,
    change_basis_ad_rep,
    check_cartan_decomp,
    check_orthonormal,
    lie_closure_dense,
    op_to_adjvec,
    orthonormalize,
    pauli_coefficients,
    trace_inner_product,
)
from pennylane.pauli import PauliSentence, PauliVSpace

# Make an operator matrix on given wire and total wire count
I_ = lambda w, n: I(w).matrix(wire_order=range(n))
X_ = lambda w, n: X(w).matrix(wire_order=range(n))
Y_ = lambda w, n: Y(w).matrix(wire_order=range(n))
Z_ = lambda w, n: Z(w).matrix(wire_order=range(n))

# Make the i-th 4**n-dimensional unit vector
uv = lambda i, n: np.eye(4**n)[i]


class TestPauliCoefficients:
    """Tests for ``pauli_coefficients`` utility function."""

    # I  X  Y  Z
    # 0  1  2  3

    @pytest.mark.parametrize(
        "H, expected",
        [
            (0.3 * X_(0, 1), [0, 0.3, 0, 0]),
            (-2.3 * Y_(0, 1), [0, 0, -2.3, 0]),
            (Z_(0, 1), [0, 0, 0, 1.0]),
            (0.3 * X_(0, 1) - 0.6 * Z_(0, 1), [0, 0.3, 0, -0.6]),
            (X_(0, 1) - Z_(0, 1) + Y_(0, 1), [0, 1, 1, -1]),
            (I_(0, 1) * 0.7, [0.7, 0, 0, 0]),
            (Y_(0, 1) * 0, [0, 0, 0, 0]),
        ],
    )
    def test_one_qubit(self, H, expected):
        """Test that Pauli coefficients for a one-qubit matrix are correct."""
        coeffs = pauli_coefficients(H)
        assert coeffs.shape == (4,)
        assert coeffs.dtype == np.float64
        assert np.allclose(coeffs, expected)

    def test_one_qubit_batched(self):
        """Test that batched Pauli coefficients for a one-qubit matrix are correct."""
        H = np.stack(
            [
                0.3 * X_(0, 1),
                -2.3 * Y_(0, 1),
                Z_(0, 1),
                0.3 * X_(0, 1) - 0.6 * Z_(0, 1),
                X_(0, 1) - Z_(0, 1) + Y_(0, 1),
                I_(0, 1) * 0.7,
                Y_(0, 1) * 0,
            ]
        )
        expected = [
            [0, 0.3, 0, 0],
            [0, 0, -2.3, 0],
            [0, 0, 0, 1.0],
            [0, 0.3, 0, -0.6],
            [0, 1, 1, -1],
            [0.7, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        coeffs = pauli_coefficients(H)
        assert coeffs.shape == (7, 4)
        assert coeffs.dtype == np.float64
        assert np.allclose(coeffs, expected)

    #  qubit
    #  0 \ 1    I    X    Y    Z
    #         -------------------
    #   I    |  0    1    2    3
    #   X    |  4    5    6    7
    #   Y    |  8    9   10   11
    #   Z    | 12   13   14   15

    @pytest.mark.parametrize(
        "H, expected",
        [
            (I_(0, 2) + I_(1, 2), 2 * uv(0, 2)),
            (0.3 * X_(0, 2), 0.3 * uv(4, 2)),
            (-2.3 * Y_(0, 2), -2.3 * uv(8, 2)),
            (Z_(0, 2) - Z_(1, 2), uv(12, 2) - uv(3, 2)),
            (0.3 * X_(0, 2) - 0.6 * Z_(0, 2), 0.3 * uv(4, 2) - 0.6 * uv(12, 2)),
            (X_(1, 2) - Z_(1, 2) + Y_(0, 2) @ X_(1, 2), uv(1, 2) - uv(3, 2) + uv(9, 2)),
            (X_(1, 2) @ Z_(0, 2) - Y_(0, 2) @ Z_(1, 2), uv(13, 2) - uv(11, 2)),
        ],
    )
    def test_two_qubits(self, H, expected):
        """Test that Pauli coefficients for a two-qubit matrix are correct."""
        coeffs = pauli_coefficients(H)
        assert coeffs.shape == (16,)
        assert coeffs.dtype == np.float64
        assert np.allclose(coeffs, expected)

    def test_two_qubits_batched(self):
        """Test that batched Pauli coefficients for a two-qubit matrix are correct."""
        H = np.stack(
            [
                I_(0, 2) + I_(1, 2),
                0.3 * X_(0, 2),
                -2.3 * Y_(0, 2),
                Z_(0, 2) - Z_(1, 2),
                0.3 * X_(0, 2) - 0.6 * Z_(0, 2),
                X_(1, 2) - Z_(1, 2) + Y_(0, 2) @ X_(1, 2),
                X_(1, 2) @ Z_(0, 2) - Y_(0, 2) @ Z_(1, 2),
            ]
        )
        expected = np.array(
            [
                2 * uv(0, 2),
                0.3 * uv(4, 2),
                -2.3 * uv(8, 2),
                uv(12, 2) - uv(3, 2),
                0.3 * uv(4, 2) - 0.6 * uv(12, 2),
                uv(1, 2) - uv(3, 2) + uv(9, 2),
                uv(13, 2) - uv(11, 2),
            ]
        )
        coeffs = pauli_coefficients(H)
        assert coeffs.shape == (7, 16)
        assert coeffs.dtype == np.float64
        assert np.allclose(coeffs, expected)


class TestPauliDecompose:
    """Tests for ``batched_pauli_decompose`` utility function."""

    # I  X  Y  Z
    # 0  1  2  3

    @pytest.mark.parametrize(
        "H, expected",
        [
            (0.3 * X_(0, 1), 0.3 * X(0)),
            (-2.3 * Y_(0, 1), -2.3 * Y(0)),
            (Z_(0, 1), Z(0)),
            (0.3 * X_(0, 1) - 0.6 * Z_(0, 1), 0.3 * X(0) - 0.6 * Z(0)),
            (X_(0, 1) - Z_(0, 1) + Y_(0, 1), X(0) - Z(0) + Y(0)),
            (I_(0, 1) * 0.7, I(0) * 0.7),
            (Y_(0, 1) * 0, 0 * I(0)),  # Zero operator is mapped to zero times identity, not Y
        ],
    )
    @pytest.mark.parametrize("pauli", [False, True])
    def test_one_qubit(self, H, expected, pauli):
        """Test that Pauli decomposition for a one-qubit matrix is correct."""
        op = batched_pauli_decompose(H, pauli=pauli)
        if pauli:
            expected = expected.pauli_rep
            expected.simplify()
            assert isinstance(op, PauliSentence)
            assert all(c.dtype == np.float64 for c in op.values())
            assert set(op.keys()) == set(expected.keys())
            assert all(np.isclose(op[k], expected[k]) for k in op.keys())
        else:
            assert isinstance(op, qml.operation.Operator)
            assert qml.equal(op, expected)

    @pytest.mark.parametrize("pauli", [False, True])
    def test_one_qubit_batched(self, pauli):
        """Test that batched Pauli decomposition for a one-qubit matrix is correct."""
        H = np.stack(
            [
                0.3 * X_(0, 1),
                -2.3 * Y_(0, 1),
                Z_(0, 1),
                0.3 * X_(0, 1) - 0.6 * Z_(0, 1),
                X_(0, 1) - Z_(0, 1) + Y_(0, 1),
                I_(0, 1) * 0.7,
                Y_(0, 1) * 0,
            ]
        )
        expected = [
            0.3 * X(0),
            -2.3 * Y(0),
            Z(0),
            0.3 * X(0) - 0.6 * Z(0),
            X(0) - Z(0) + Y(0),
            I(0) * 0.7,
            I(0) * 0,
        ]
        op = batched_pauli_decompose(H, pauli=pauli)
        assert isinstance(op, list)
        assert len(op) == len(expected)
        if pauli:
            for _op, e in zip(op, expected):
                e = e.pauli_rep
                e.simplify()
                assert isinstance(_op, PauliSentence)
                assert all(c.dtype == np.float64 for c in _op.values())
                assert set(_op.keys()) == set(e.keys())
                assert all(np.isclose(_op[k], e[k]) for k in _op.keys())
        else:
            for _op, e in zip(op, expected):
                assert isinstance(_op, qml.operation.Operator)
                assert qml.equal(_op, e)

    @pytest.mark.parametrize(
        "H, expected",
        [
            (I_(0, 2) + I_(1, 2), 2 * I()),
            (0.3 * X_(0, 2), 0.3 * X(0)),
            (-2.3 * Y_(0, 2), -2.3 * Y(0)),
            (Z_(0, 2) - Z_(1, 2), Z(0) - Z(1)),
            (0.3 * X_(0, 2) - 0.6 * Z_(0, 2), 0.3 * X(0) - 0.6 * Z(0)),
            (X_(1, 2) - Z_(1, 2) + Y_(0, 2) @ X_(1, 2), X(1) - Z(1) + Y(0) @ X(1)),
            (X_(1, 2) @ Z_(0, 2) - Y_(0, 2) @ Z_(1, 2), X(1) @ Z(0) - Y(0) @ Z(1)),
        ],
    )
    @pytest.mark.parametrize("pauli", [False, True])
    def test_two_qubits(self, H, expected, pauli):
        """Test that Pauli decomposition for a two-qubit matrix is correct."""
        op = batched_pauli_decompose(H, pauli=pauli)
        if pauli:
            expected = expected.pauli_rep
            expected.simplify()
            assert isinstance(op, PauliSentence)
            assert all(c.dtype == np.float64 for c in op.values())
            assert set(op.keys()) == set(expected.keys())
            assert all(np.isclose(op[k], expected[k]) for k in op.keys())
        else:
            assert isinstance(op, qml.operation.Operator)
            assert qml.equal(op, expected)

    @pytest.mark.parametrize("pauli", [False, True])
    def test_two_qubits_batched(self, pauli):
        """Test that batched Pauli decomposition for a two-qubit matrix is correct."""
        H = np.stack(
            [
                I_(0, 2) + I_(1, 2),
                0.3 * X_(0, 2),
                -2.3 * Y_(0, 2),
                Z_(0, 2) - Z_(1, 2),
                0 * X_(0, 2),
                0.3 * X_(0, 2) - 0.6 * Z_(0, 2),
                X_(1, 2) - Z_(1, 2) + Y_(0, 2) @ X_(1, 2),
                X_(1, 2) @ Z_(0, 2) - Y_(0, 2) @ Z_(1, 2),
            ]
        )
        expected = [
            2 * I(),
            0.3 * X(0),
            -2.3 * Y(0),
            Z(0) - Z(1),
            0 * I(0),
            0.3 * X(0) - 0.6 * Z(0),
            X(1) - Z(1) + Y(0) @ X(1),
            X(1) @ Z(0) - Y(0) @ Z(1),
        ]
        op = batched_pauli_decompose(H, pauli=pauli)
        assert isinstance(op, list)
        assert len(op) == len(expected)
        if pauli:
            for _op, e in zip(op, expected):
                e = e.pauli_rep
                e.simplify()
                assert isinstance(_op, PauliSentence)
                assert all(c.dtype == np.float64 for c in _op.values())
                assert set(_op.keys()) == set(e.keys())
                assert all(np.isclose(_op[k], e[k]) for k in _op.keys())
        else:
            for _op, e in zip(op, expected):
                assert isinstance(_op, qml.operation.Operator)
                assert qml.equal(_op, e)


@pytest.mark.parametrize("op1", [X(0), -0.8 * X(0) @ X(1), X(0) @ Y(2), X(0) @ Z(1) + X(1) @ X(2)])
@pytest.mark.parametrize(
    "op2", [X(0), X(0) + X(0) @ X(1), 0.2 * X(0) @ Y(2), X(0) @ Z(1) + X(1) @ X(2)]
)
def test_trace_inner_product_consistency(op1, op2):
    """Test that the trace inner product norm for different operators is consistent"""
    res1 = trace_inner_product(
        qml.matrix(op1, wire_order=range(3)), qml.matrix(op2, wire_order=range(3))
    )
    res2 = trace_inner_product(op1.pauli_rep, op2.pauli_rep)
    res3 = trace_inner_product(op1, op2)
    assert np.allclose(res1, res2)
    assert np.allclose(res1, res3)


id_pw = qml.pauli.PauliWord({})


@pytest.mark.parametrize(
    "g, inner_product",
    [
        (qml.ops.qubit.special_unitary.pauli_basis_matrices(3), trace_inner_product),
        (qml.pauli.pauli_group(4), lambda A, B: (A @ B).pauli_rep.trace()),
        (qml.pauli.pauli_group(4), lambda A, B: (A.pauli_rep @ B.pauli_rep).get(id_pw, 0.0)),
        (list("abcdefghi"), lambda A, B: int(A == B)),
    ],
)
def test_check_orthonormal_True(g, inner_product):
    """Test check_orthonormal"""
    assert check_orthonormal(g, inner_product)


# The reasons the following are not orthonormal are:
# Non-normalized ops
# Non-orthogonal ops
# Inner product is non-normalized trace inner product


@pytest.mark.parametrize(
    "g, inner_product",
    [
        ([np.eye(2), qml.X(0).matrix() + qml.Z(0).matrix()], trace_inner_product),
        ([qml.X(0).matrix(), qml.X(0).matrix() + qml.Z(0).matrix()], trace_inner_product),
        (qml.pauli.pauli_group(2), lambda A, B: np.trace((A @ B).matrix())),
    ],
)
def test_check_orthonormal_False(g, inner_product):
    """Test check_orthonormal"""
    assert not check_orthonormal(g, inner_product)


gens1 = [X(i) @ X(i + 1) + Y(i) @ Y(i + 1) + Z(i) @ Z(i + 1) for i in range(3)]
Heisenberg4_sum_op = qml.lie_closure(gens1)
Heisenberg4_sum_ps = [op.pauli_rep for op in Heisenberg4_sum_op]
Heisenberg4_sum_vspace = PauliVSpace(Heisenberg4_sum_ps)
Heisenberg4_sum_dense = [qml.matrix(op, wire_order=range(4)) for op in Heisenberg4_sum_op]


@pytest.mark.parametrize(
    "g", [Heisenberg4_sum_ps, Heisenberg4_sum_vspace, Heisenberg4_sum_op, Heisenberg4_sum_dense]
)
def test_orthonormalize(g):
    """Test orthonormalize"""

    g = orthonormalize(g)

    assert check_orthonormal(g, trace_inner_product)


def test_check_cartan_decomp():
    """Test that check_cartan_decomp correctly checks Ising cartan decomp from fdhs paper (https://arxiv.org/abs/2104.00728)"""
    k = [Z(0) @ Y(1), Y(0) @ Z(1)]
    m = [Z(0) @ Z(1), Y(0) @ Y(1), X(0), X(1)]

    assert check_cartan_decomp(k, m)


class TestChangeBasisAdRep:
    """Tests for ``change_basis_ad_rep`` to change the adjoint representation into a new basis."""

    def test_permutation(self):
        """Test that a permutation is accounted for correctly."""
        ops = [qml.X(0), qml.Y(1), qml.Y(0) @ qml.Z(1), qml.X(1)]
        dla = qml.lie_closure(ops)
        adj = qml.structure_constants(dla)
        perm = np.random.permutation(len(dla))
        permuted_dla = [dla[i] for i in perm]
        permuted_adj = qml.structure_constants(permuted_dla)

        basis_change = op_to_adjvec(permuted_dla, dla) @ np.linalg.pinv(op_to_adjvec(dla, dla))
        new_adj = change_basis_ad_rep(adj, basis_change)
        assert np.allclose(new_adj, permuted_adj)

    def test_tiny_skewed_basis(self):
        """Test that changing from a tiny orthonormal basis to a skewed basis works."""
        dla = [qml.X(0), qml.Y(0), qml.Z(0)]
        adj = qml.structure_constants(dla)
        coeffs = np.random.random((len(dla), len(dla)))
        skewed_dla = [qml.sum(*(c * op for c, op in zip(_coeffs, dla))) for _coeffs in coeffs]
        skewed_adj = qml.structure_constants(skewed_dla, is_orthogonal=False)

        basis_change = op_to_adjvec(skewed_dla, dla) @ np.linalg.pinv(op_to_adjvec(dla, dla))
        new_adj = change_basis_ad_rep(adj, basis_change)
        assert np.allclose(new_adj, skewed_adj)

    def test_tiny_skewed_basis_from_non_ortho(self):
        """Test that changing from a tiny non-orthonormal basis to a skewed basis works."""
        ortho_dla = [qml.X(0), qml.Y(0), qml.Z(0)]  # only used to create adj rep.
        dla = [0.2 * qml.X(0) - 0.6 * qml.Y(0), 0.4 * qml.Y(0) + 0.9 * qml.Z(0), qml.Z(0)]

        adj = qml.structure_constants(dla, is_orthogonal=False)
        coeffs = np.random.random((len(dla), len(dla)))
        skewed_dla = [qml.sum(*(c * op for c, op in zip(_coeffs, dla))) for _coeffs in coeffs]
        skewed_adj = qml.structure_constants(skewed_dla, is_orthogonal=False)

        basis_change = op_to_adjvec(skewed_dla, ortho_dla) @ np.linalg.pinv(
            op_to_adjvec(dla, ortho_dla)
        )
        new_adj = change_basis_ad_rep(adj, basis_change)
        assert np.allclose(new_adj, skewed_adj)

    def test_skewed_basis(self):
        """Test that changing from an orthonormal basis to a skewed basis works."""
        ops = [qml.X(0), qml.Y(1), qml.Y(0) @ qml.Z(1)]
        dla = qml.lie_closure(ops)
        adj = qml.structure_constants(dla)
        coeffs = np.random.random((len(dla), len(dla)))
        skewed_dla = [qml.sum(*(c * op for c, op in zip(_coeffs, dla))) for _coeffs in coeffs]
        skewed_adj = qml.structure_constants(skewed_dla, is_orthogonal=False)

        basis_change = op_to_adjvec(skewed_dla, dla) @ np.linalg.pinv(op_to_adjvec(dla, dla))
        new_adj = change_basis_ad_rep(adj, basis_change)
        assert np.allclose(new_adj, skewed_adj)

    def test_skewed_basis_from_non_ortho(self):
        """Test that changing from a non-orthonormal basis to a skewed basis works."""
        ops = [qml.X(0), qml.Y(1), qml.Y(0) @ qml.Z(1)]
        ortho_dla = qml.lie_closure(ops)  # only used to create adj rep.

        coeffs = np.random.random((len(ortho_dla), len(ortho_dla)))
        dla = [qml.sum(*(c * op for c, op in zip(_coeffs, ortho_dla))) for _coeffs in coeffs]
        adj = qml.structure_constants(dla, is_orthogonal=False)

        coeffs = np.random.random((len(dla), len(dla)))
        skewed_dla = [qml.sum(*(c * op for c, op in zip(_coeffs, dla))) for _coeffs in coeffs]
        skewed_adj = qml.structure_constants(skewed_dla, is_orthogonal=False)

        basis_change = op_to_adjvec(skewed_dla, dla) @ np.linalg.pinv(op_to_adjvec(dla, dla))
        new_adj = change_basis_ad_rep(adj, basis_change)
        assert np.allclose(new_adj, skewed_adj)


### Single-qubit test cases
# Create Pauli bases on 1 and 2 qubits
paulis_1_qubit = [op.pauli_rep for op in qml.pauli.pauli_group(1)]
paulis_2_qubits = [op.pauli_rep for op in qml.pauli.pauli_group(2)]

# Create a non-orthonormal basis on 1 qubit
basis_coeffs_0 = np.random.random((4, 4))
basis_0 = [qml.dot(c, paulis_1_qubit).pauli_rep for c in basis_coeffs_0]

# Create an adjvec for three operators, which we understand to be in basis_0
adjvec_0 = np.random.random((3, 4))

# To express the adjvec given in basis_0 in the standard Pauli basis, we need the inv sqrt Gram.
# Compute the operator given by adjvec_0 via the new coefficient matrix adjvec_0 @ inv_sqrt_Gram_0
inv_sqrt_Gram_0 = np.linalg.pinv(sqrtm(basis_coeffs_0 @ basis_coeffs_0.T))
expected_0 = [qml.dot(c, basis_0).pauli_rep for c in adjvec_0 @ inv_sqrt_Gram_0]

# Use the adjvec_0 for another 3 operators, now understood in the (orthonormal) basis
# paulis_1_qubits. The expected output does not require a Gram matrix to be computed.
expected_1 = [qml.dot(c, paulis_1_qubit).pauli_rep for c in adjvec_0]

### Two-qubit test cases

# Create a non-orthonormal basis on 2 qubits
basis_coeffs_1 = np.random.random((16, 16))
basis_1 = [qml.dot(c, paulis_2_qubits).pauli_rep for c in basis_coeffs_1]

# Create an adjvec for 13 operators, which we understand to be in basis_1
adjvec_1 = np.random.random((13, 16))

# To express the adjvec given in basis_1 in the standard Pauli basis, we need the inv sqrt Gram.
# Compute the operator given by adjvec_1 via the new coefficient matrix adjvec_1 @ inv_sqrt_Gram_1
inv_sqrt_Gram_1 = np.linalg.pinv(sqrtm(basis_coeffs_1 @ basis_coeffs_1.T))
expected_2 = [qml.dot(c, basis_1).pauli_rep for c in adjvec_1 @ inv_sqrt_Gram_1]

# Use the adjvec_1 for another 13 operators, now understood in the (orthonormal) basis
# paulis_2_qubits. The expected output does not require a Gram matrix to be computed.
expected_3 = [qml.dot(c, paulis_2_qubits).pauli_rep for c in adjvec_1]

# Collect all test cases. All operators are formatted as qml.pauli.PauliSentence
ps_test_cases = [
    (adjvec_0, basis_0, expected_0, False),  # Non-orthogonal basis
    (adjvec_0, paulis_1_qubit, expected_1, True),  # Orthonormal basis
    (adjvec_1, basis_1, expected_2, False),  # Non-orthogonal basis
    (adjvec_1, paulis_2_qubits, expected_3, True),  # Orthonormal basis
]

# Translate test cases to qml.operation.Operations
op_test_cases = [
    (adj_vecs, [ps.operation() for ps in basis], [ps.operation() for ps in expected], is_ortho)
    for adj_vecs, basis, expected, is_ortho in ps_test_cases
]

# Translate test cases to dense matrices
dense_test_cases = [
    (
        adj_vecs,
        [qml.matrix(ps, wire_order=[0, 1]) for ps in basis],
        [qml.matrix(ps, wire_order=[0, 1]) for ps in expected],
        is_ortho,
    )
    for adj_vecs, basis, expected, is_ortho in ps_test_cases
]


class TestAdjvecToOp:
    """Test adjvec_to_op."""

    @pytest.mark.parametrize("adj_vecs, basis, expected, is_ortho", ps_test_cases)
    def test_with_ps(self, adj_vecs, basis, expected, is_ortho):
        """Test ``adjvec_to_op`` with a basis of ``PauliSentence`` operators."""
        out = adjvec_to_op(adj_vecs, basis, is_orthogonal=False)
        for out_op, exp_op in zip(out, expected):
            assert isinstance(out_op, PauliSentence)
            assert all(c.dtype == np.float64 for c in out_op.values())
            assert set(out_op) == set(exp_op)
            assert all(np.isclose(out_op[k], exp_op[k]) for k in out_op)
        if is_ortho:
            out = adjvec_to_op(adj_vecs, basis, is_orthogonal=True)
            for out_op, exp_op in zip(out, expected):
                assert isinstance(out_op, PauliSentence)
                assert all(c.dtype == np.float64 for c in out_op.values())
                assert set(out_op) == set(exp_op)
                assert all(np.isclose(out_op[k], exp_op[k]) for k in out_op)

    @pytest.mark.parametrize("adj_vecs, basis, expected, is_ortho", op_test_cases)
    def test_with_op(self, adj_vecs, basis, expected, is_ortho):
        """Test ``adjvec_to_op`` with a basis of ``Operator`` operators."""
        out = adjvec_to_op(adj_vecs, basis, is_orthogonal=False)
        for out_op, exp_op in zip(out, expected):
            assert qml.equal(out_op.simplify(), exp_op.simplify())
        if is_ortho:
            out = adjvec_to_op(adj_vecs, basis, is_orthogonal=True)
            for out_op, exp_op in zip(out, expected):
                assert qml.equal(out_op.simplify(), exp_op.simplify())

    @pytest.mark.parametrize("adj_vecs, basis, expected, is_ortho", dense_test_cases)
    def test_with_dense(self, adj_vecs, basis, expected, is_ortho):
        """Test ``adjvec_to_op`` with a basis of dense operators."""
        out = adjvec_to_op(adj_vecs, basis, is_orthogonal=False)
        assert qml.math.shape(out) == qml.math.shape(expected)
        assert np.allclose(out, expected)
        if is_ortho:
            out = adjvec_to_op(adj_vecs, basis, is_orthogonal=True)
            assert qml.math.shape(out) == qml.math.shape(expected)
            assert np.allclose(out, expected)


class TestOpToAdjvec:
    """Test op_to_adjvec. We reuse the test cases from adjvec_to_op and simply re-interpret which
    part is passed to the function, and which represents the expected output."""

    @pytest.mark.parametrize("expected, basis, ops, is_ortho", ps_test_cases)
    def test_with_ps(self, ops, basis, expected, is_ortho):
        """Test ``op_to_adjvec`` with a basis of ``PauliSentence`` operators."""
        out = op_to_adjvec(ops, basis, is_orthogonal=False)
        assert out.dtype == np.float64
        assert qml.math.shape(out) == qml.math.shape(expected)
        assert np.allclose(out, expected)
        if is_ortho:
            out = op_to_adjvec(ops, basis, is_orthogonal=True)
            assert out.dtype == np.float64
            assert qml.math.shape(out) == qml.math.shape(expected)
            assert np.allclose(out, expected)

    @pytest.mark.parametrize("expected, basis, ops, is_ortho", op_test_cases)
    def test_with_op(self, ops, basis, expected, is_ortho):
        """Test ``op_to_adjvec`` with a basis of ``Operator`` operators."""
        out = op_to_adjvec(ops, basis, is_orthogonal=False)
        assert out.dtype == np.float64
        assert qml.math.shape(out) == qml.math.shape(expected)
        assert np.allclose(out, expected)
        if is_ortho:
            out = op_to_adjvec(ops, basis, is_orthogonal=True)
            assert out.dtype == np.float64
            assert qml.math.shape(out) == qml.math.shape(expected)
            assert np.allclose(out, expected)

    @pytest.mark.parametrize("expected, basis, ops, is_ortho", dense_test_cases)
    def test_with_dense(self, ops, basis, expected, is_ortho):
        """Test ``op_to_adjvec`` with a basis of dense operators."""
        out = op_to_adjvec(ops, basis, is_orthogonal=False)
        assert out.dtype == np.float64
        assert qml.math.shape(out) == qml.math.shape(expected)
        assert np.allclose(out, expected)
        if is_ortho:
            out = op_to_adjvec(ops, basis, is_orthogonal=True)
            assert out.dtype == np.float64
            assert qml.math.shape(out) == qml.math.shape(expected)
            assert np.allclose(out, expected)

    def test_consistent_with_input_types(self):
        """Test that op_to_adjvec yields the same results independently of the input type"""

        g = list(qml.pauli.pauli_group(3))  # su(8)
        g = lie_closure_dense(g)

        m = g[:32]

        res1 = op_to_adjvec(m, g)

        g = list(qml.pauli.pauli_group(3))  # su(8)
        g = qml.lie_closure(g)
        g = [_.pauli_rep for _ in g]

        m = g[:32]

        res2 = np.array(op_to_adjvec(m, g))
        assert res1.shape == res2.shape
        assert np.allclose(res1, res2)
