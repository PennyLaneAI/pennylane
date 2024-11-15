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

import pennylane as qml
from pennylane import I, X, Y, Z
from pennylane.labs.dla import (
    adjvec_to_op,
    check_orthonormal,
    op_to_adjvec,
    orthonormalize,
    pauli_coefficients,
    pauli_decompose,
    trace_inner_product,
)
from pennylane.pauli import PauliSentence, PauliWord

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
    """Tests for ``pauli_decompose`` utility function."""

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
        op = pauli_decompose(H, pauli=pauli)
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
        op = pauli_decompose(H, pauli=pauli)
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
        op = pauli_decompose(H, pauli=pauli)
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
        op = pauli_decompose(H, pauli=pauli)
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


@pytest.mark.parametrize("op1", [X(0), X(0) @ X(1), X(0) @ Y(2), X(0) @ Z(1) + X(1) @ X(2)])
@pytest.mark.parametrize("op2", [X(0), X(0) @ X(1), X(0) @ Y(2), X(0) @ Z(1) + X(1) @ X(2)])
def test_trace_inner_product_consistency(op1, op2):
    """Test that the trace inner product norm for different operators is consistent"""
    res1 = trace_inner_product(
        qml.matrix(op1, wire_order=range(3)), qml.matrix(op2, wire_order=range(3))
    )
    res2 = trace_inner_product(op1.pauli_rep, op2.pauli_rep)
    assert np.allclose(res1, res2)


gens1 = [X(i) @ X(i + 1) + Y(i) @ Y(i + 1) + Z(i) @ Z(i + 1) for i in range(3)]
Heisenberg4_sum_op = qml.lie_closure(gens1)
Heisenberg4_sum_ps = [op.pauli_rep for op in Heisenberg4_sum_op]
Heisenberg4_sum_dense = [qml.matrix(op, wire_order=range(4)) for op in Heisenberg4_sum_op]


@pytest.mark.parametrize("g", [Heisenberg4_sum_ps, Heisenberg4_sum_dense])
def test_orthonormalize_ps(g):
    """Test orthonormalize on pauli sentences"""

    g = orthonormalize(g)

    assert check_orthonormal(g, trace_inner_product)


some_ps = PauliSentence({PauliWord({0: "X", 23: "Y"}): 0.5, PauliWord({3: "Z", 2: "X"}): -1.3})
x = np.arange(1, 151).reshape((10, 15))
paulis_2_qubits = [op.pauli_rep for op in qml.pauli.pauli_group(2)]
ps_basis_0_rot = np.random.random((16, 16))
ps_basis_0 = [sum(c * op for c, op in zip(_coeffs, paulis_2_qubits)) for _coeffs in ps_basis_0_rot]
coeffs_0 = np.random.random((13, 16))
expected_0 = [
    sum(c * op for c, op in zip(coeffs, paulis_2_qubits)) for coeffs in coeffs_0 @ ps_basis_0_rot
]
expected_1 = [sum(c * op for c, op in zip(coeffs, paulis_2_qubits)) for coeffs in coeffs_0]

ps_test_cases = [
    (x, [some_ps] * 15, [c * some_ps for c in np.sum(x, axis=1)], False),  # linearly dep. "basis"
    (coeffs_0, ps_basis_0, expected_0, False),  # Non-orthogonal basis
    (coeffs_0, paulis_2_qubits, expected_1, True),  # Orthonormal basis
]

op_test_cases = [
    (adj_vecs, [ps.operation() for ps in basis], [ps.operation() for ps in expected], is_ortho)
    for adj_vecs, basis, expected, is_ortho in ps_test_cases
]

wire_orders = ([0, 2, 3, 23], [0, 1])
dense_test_cases = [
    (
        adj_vecs,
        [qml.matrix(ps, wire_order=wo) for ps in basis],
        [qml.matrix(ps, wire_order=wo) for ps in expected],
        is_ortho,
    )
    for (adj_vecs, basis, expected, is_ortho), wo in zip(ps_test_cases, wire_orders)
]


class TestAdjvecToOp:
    """Test adjvec_to_op."""

    @pytest.mark.parametrize("adj_vecs, basis, expected, _", ps_test_cases)
    def test_with_ps(self, adj_vecs, basis, expected, _):
        """Test ``adjvec_to_op`` with a basis of ``PauliSentence`` operators."""
        out = adjvec_to_op(adj_vecs, basis)
        for out_op, exp_op in zip(out, expected):
            assert isinstance(out_op, PauliSentence)
            assert all(c.dtype == np.float64 for c in out_op.values())
            assert set(out_op) == set(exp_op)
            assert all(np.isclose(out_op[k], exp_op[k]) for k in out_op)

    @pytest.mark.parametrize("adj_vecs, basis, expected, _", op_test_cases)
    def test_with_op(self, adj_vecs, basis, expected, _):
        """Test ``adjvec_to_op`` with a basis of ``Operator`` operators."""
        out = adjvec_to_op(adj_vecs, basis)
        for out_op, exp_op in zip(out, expected):
            assert qml.equal(out_op.simplify(), exp_op.simplify())

    @pytest.mark.parametrize("adj_vecs, basis, expected, _", dense_test_cases)
    def test_with_dense(self, adj_vecs, basis, expected, _):
        """Test ``adjvec_to_op`` with a basis of dense operators."""
        out = adjvec_to_op(adj_vecs, basis)
        assert qml.math.shape(out) == qml.math.shape(expected)
        assert np.allclose(out, expected)


class TestOpToAdjvec:
    """Test op_to_adjvec. We reuse the test cases from adjvec_to_op, except
    for the first one, which uses a linearly dependent "basis"."""

    @pytest.mark.parametrize("expected, basis, ops, is_ortho", ps_test_cases[1:])
    def test_with_ps(self, ops, basis, expected, is_ortho):
        """Test ``op_to_adjvec`` with a basis of ``PauliSentence`` operators."""
        out = op_to_adjvec(ops, basis, is_orthogonal=False)
        assert qml.math.shape(out) == qml.math.shape(expected)
        assert np.allclose(out, expected)
        if is_ortho:
            out = op_to_adjvec(ops, basis, is_orthogonal=True)
            assert qml.math.shape(out) == qml.math.shape(expected)
            assert np.allclose(out, expected)

    @pytest.mark.parametrize("expected, basis, ops, is_ortho", op_test_cases[1:])
    def test_with_op(self, ops, basis, expected, is_ortho):
        """Test ``op_to_adjvec`` with a basis of ``Operator`` operators."""
        out = op_to_adjvec(ops, basis, is_orthogonal=False)
        assert qml.math.shape(out) == qml.math.shape(expected)
        assert np.allclose(out, expected)
        if is_ortho:
            out = op_to_adjvec(ops, basis, is_orthogonal=True)
            assert qml.math.shape(out) == qml.math.shape(expected)
            assert np.allclose(out, expected)

    @pytest.mark.parametrize("expected, basis, ops, is_ortho", dense_test_cases[1:])
    def test_with_dense(self, ops, basis, expected, is_ortho):
        """Test ``op_to_adjvec`` with a basis of dense operators."""
        out = op_to_adjvec(ops, basis, is_orthogonal=False)
        assert qml.math.shape(out) == qml.math.shape(expected)
        assert np.allclose(out, expected)

        if is_ortho:
            out = op_to_adjvec(ops, basis, is_orthogonal=True)
            assert qml.math.shape(out) == qml.math.shape(expected)
            assert np.allclose(out, expected)
