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
"""Tests for pennylane/labs/dla/horizontal_cartan_subalgebra.py functionality"""
import numpy as np

# pylint: disable=no-self-use,too-few-public-methods,missing-class-docstring, too-many-positional-arguments, too-many-arguments
import pytest
from scipy.linalg import sqrtm

import pennylane as qml
from pennylane import X, Y, Z
from pennylane.liealg import (
    adjvec_to_op,
    cartan_decomp,
    change_basis_ad_rep,
    check_cartan_decomp,
    even_odd_involution,
    horizontal_cartan_subalgebra,
    op_to_adjvec,
)
from pennylane.pauli import PauliSentence


class TestCartanSubalgebra:
    """Tests for qml.liealg.horizontal_cartan_subalgebra"""

    @pytest.mark.parametrize("n, len_g, len_h, len_mtilde", [(2, 6, 2, 2), (3, 15, 2, 6)])
    @pytest.mark.parametrize("provide_adj", [True, False])
    def test_horizontal_cartan_subalgebra_Ising(self, n, len_g, len_h, len_mtilde, provide_adj):
        """Test Cartan subalgebra of 2 qubit Ising model"""
        gens = [X(w) @ X(w + 1) for w in range(n - 1)] + [Z(w) for w in range(n)]
        gens = [op.pauli_rep for op in gens]
        g = qml.lie_closure(gens, pauli=True)

        k, m = cartan_decomp(g, even_odd_involution)
        assert check_cartan_decomp(k, m)

        g = k + m
        assert len(g) == len_g

        if provide_adj:
            adj = qml.structure_constants(g)
        else:
            adj = None

        newg, k, mtilde, h, new_adj = horizontal_cartan_subalgebra(k, m, adj, start_idx=0)
        assert len(h) == len_h
        assert len(mtilde) == len_mtilde
        assert len(h) + len(mtilde) == len(m)

        new_adj_re = qml.structure_constants(newg)

        assert np.allclose(new_adj_re, new_adj)

    @pytest.mark.parametrize("start_idx", [0, 2])
    def test_horizontal_cartan_subalgebra_matrix_input(self, start_idx):
        """Test horizontal_cartan_subalgebra with matrix inputs"""
        k = [1.0 * Z(0), 1.0 * Z(1)]
        m = [1.0 * X(0) @ X(1), -1.0 * Y(0) @ X(1), -1.0 * X(0) @ Y(1), 1.0 * Y(0) @ Y(1)]
        k = np.array([qml.matrix(op, wire_order=range(2)) for op in k])
        m = np.array([qml.matrix(op, wire_order=range(2)) for op in m])

        newg, k, mtilde, h, new_adj = horizontal_cartan_subalgebra(k, m, start_idx=start_idx)
        assert len(h) + len(mtilde) == len(m)

        new_adj_re = qml.structure_constants(newg, matrix=True)

        assert np.allclose(new_adj_re, new_adj)
        assert qml.liealg.check_abelian(h)

    def test_horizontal_cartan_subalgebra_adjvec_output(self):
        """Test horizontal_cartan_subalgebra with adjvec outputs"""
        k = [1.0 * Z(0), 1.0 * Z(1)]
        m = [1.0 * X(0) @ X(1), -1.0 * Y(0) @ X(1), -1.0 * X(0) @ Y(1), 1.0 * Y(0) @ Y(1)]

        np_newg, _, np_mtilde, np_h, new_adj = qml.liealg.horizontal_cartan_subalgebra(
            k, m, return_adjvec=True, start_idx=0
        )
        assert len(np_h) + len(np_mtilde) == len(m)
        newg = adjvec_to_op(np_newg, k + m)

        new_adj_re = qml.structure_constants(newg)

        assert np.allclose(new_adj_re, new_adj)
        h = adjvec_to_op(np_h, k + m)
        assert qml.liealg.check_abelian(h)

    def test_horizontal_cartan_subalgebra_verbose(self, capsys):
        """Test verbose outputs during horizontal_cartan_subalgebra computation"""
        k = [1.0 * Z(0), 1.0 * Z(1)]
        m = [1.0 * X(0) @ X(1), -1.0 * Y(0) @ X(1), -1.0 * X(0) @ Y(1), 1.0 * Y(0) @ Y(1)]
        _ = qml.liealg.horizontal_cartan_subalgebra(k, m, verbose=True)
        captured = capsys.readouterr()
        assert "iteration 1: Found 1 independent Abelian operators." in captured.out
        assert "iteration 2: Found 2 independent Abelian operators." in captured.out


class TestChangeBasisAdRep:
    """Tests for ``change_basis_ad_rep`` to change the adjoint representation into a new basis."""

    def test_permutation(self, seed):
        """Test that a permutation is accounted for correctly."""
        rng = np.random.default_rng(seed)
        ops = [qml.X(0), qml.Y(1), qml.Y(0) @ qml.Z(1), qml.X(1)]
        dla = qml.lie_closure(ops)
        adj = qml.structure_constants(dla)
        perm = rng.permutation(len(dla))
        permuted_dla = [dla[i] for i in perm]
        permuted_adj = qml.structure_constants(permuted_dla)

        basis_change = op_to_adjvec(permuted_dla, dla) @ np.linalg.pinv(op_to_adjvec(dla, dla))
        new_adj = change_basis_ad_rep(adj, basis_change)
        assert np.allclose(new_adj, permuted_adj)

    def test_tiny_skewed_basis(self, seed):
        """Test that changing from a tiny orthonormal basis to a skewed basis works."""
        rng = np.random.default_rng(seed)
        dla = [qml.X(0), qml.Y(0), qml.Z(0)]
        adj = qml.structure_constants(dla)
        coeffs = rng.random((len(dla), len(dla)))
        skewed_dla = [qml.sum(*(c * op for c, op in zip(_coeffs, dla))) for _coeffs in coeffs]
        skewed_adj = qml.structure_constants(skewed_dla, is_orthogonal=False)

        basis_change = op_to_adjvec(skewed_dla, dla) @ np.linalg.pinv(op_to_adjvec(dla, dla))
        new_adj = change_basis_ad_rep(adj, basis_change)
        assert np.allclose(new_adj, skewed_adj)

    def test_tiny_skewed_basis_from_non_ortho(self, seed):
        """Test that changing from a tiny non-orthonormal basis to a skewed basis works."""
        rng = np.random.default_rng(seed)
        ortho_dla = [qml.X(0), qml.Y(0), qml.Z(0)]  # only used to create adj rep.
        dla = [0.2 * qml.X(0) - 0.6 * qml.Y(0), 0.4 * qml.Y(0) + 0.9 * qml.Z(0), qml.Z(0)]

        adj = qml.structure_constants(dla, is_orthogonal=False)
        coeffs = rng.random((len(dla), len(dla)))
        skewed_dla = [qml.sum(*(c * op for c, op in zip(_coeffs, dla))) for _coeffs in coeffs]
        skewed_adj = qml.structure_constants(skewed_dla, is_orthogonal=False)

        basis_change = op_to_adjvec(skewed_dla, ortho_dla) @ np.linalg.pinv(
            op_to_adjvec(dla, ortho_dla)
        )
        new_adj = change_basis_ad_rep(adj, basis_change)
        assert np.allclose(new_adj, skewed_adj)

    def test_skewed_basis(self, seed):
        """Test that changing from an orthonormal basis to a skewed basis works."""
        rng = np.random.default_rng(seed)
        ops = [qml.X(0), qml.Y(1), qml.Y(0) @ qml.Z(1)]
        dla = qml.lie_closure(ops)
        adj = qml.structure_constants(dla)
        coeffs = rng.random((len(dla), len(dla)))
        skewed_dla = [qml.sum(*(c * op for c, op in zip(_coeffs, dla))) for _coeffs in coeffs]
        skewed_adj = qml.structure_constants(skewed_dla, is_orthogonal=False)

        basis_change = op_to_adjvec(skewed_dla, dla) @ np.linalg.pinv(op_to_adjvec(dla, dla))
        new_adj = change_basis_ad_rep(adj, basis_change)
        assert np.allclose(new_adj, skewed_adj)

    def test_skewed_basis_from_non_ortho(self, seed):
        """Test that changing from a non-orthonormal basis to a skewed basis works."""
        rng = np.random.default_rng(seed)
        ops = [qml.X(0), qml.Y(1), qml.Y(0) @ qml.Z(1)]
        ortho_dla = qml.lie_closure(ops)  # only used to create adj rep.

        coeffs = rng.random((len(ortho_dla), len(ortho_dla)))
        dla = [qml.sum(*(c * op for c, op in zip(_coeffs, ortho_dla))) for _coeffs in coeffs]
        adj = qml.structure_constants(dla, is_orthogonal=False)

        coeffs = rng.random((len(dla), len(dla)))
        skewed_dla = [qml.sum(*(c * op for c, op in zip(_coeffs, dla))) for _coeffs in coeffs]
        skewed_adj = qml.structure_constants(skewed_dla, is_orthogonal=False)

        basis_change = op_to_adjvec(skewed_dla, dla) @ np.linalg.pinv(op_to_adjvec(dla, dla))
        new_adj = change_basis_ad_rep(adj, basis_change)
        assert np.allclose(new_adj, skewed_adj)


# Test case generation using seed fixture
def _generate_test_cases(seed_value):
    """Generate test cases with a specific seed value."""
    rng = np.random.RandomState(seed_value)

    ### Single-qubit test cases
    # Create Pauli bases on 1 and 2 qubits
    paulis_1_qubit = [op.pauli_rep for op in qml.pauli.pauli_group(1)]
    paulis_2_qubits = [op.pauli_rep for op in qml.pauli.pauli_group(2)]

    # Create a non-orthonormal basis on 1 qubit
    basis_coeffs_0 = rng.random((4, 4))
    basis_0 = [qml.dot(c, paulis_1_qubit).pauli_rep for c in basis_coeffs_0]

    # Create an adjvec for three operators, which we understand to be in basis_0
    adjvec_0 = rng.random((3, 4))

    # To express the adjvec given in basis_0 in the standard Pauli basis, we need the inv sqrt Gram.
    # Compute the operator given by adjvec_0 via the new coefficient matrix adjvec_0 @ inv_sqrt_Gram_0
    gram_0 = basis_coeffs_0 @ basis_coeffs_0.T
    inv_sqrt_Gram_0 = np.linalg.pinv(sqrtm(gram_0))
    expected_0 = [qml.dot(c, basis_0).pauli_rep for c in adjvec_0 @ inv_sqrt_Gram_0]

    # Use the adjvec_0 for another 3 operators, now understood in the (orthonormal) basis
    # paulis_1_qubits. The expected output does not require a Gram matrix to be computed.
    expected_1 = [qml.dot(c, paulis_1_qubit).pauli_rep for c in adjvec_0]

    # Two-qubit test cases
    # Create a non-orthonormal basis on 2 qubits
    basis_coeffs_1 = rng.random((16, 16))
    basis_1 = [qml.dot(c, paulis_2_qubits).pauli_rep for c in basis_coeffs_1]

    # Create an adjvec for 13 operators, which we understand to be in basis_1
    adjvec_1 = rng.random((13, 16))

    # To express the adjvec given in basis_1 in the standard Pauli basis, we need the inv sqrt Gram.
    # Compute the operator given by adjvec_1 via the new coefficient matrix adjvec_1 @ inv_sqrt_Gram_1
    gram_1 = basis_coeffs_1 @ basis_coeffs_1.T
    inv_sqrt_Gram_1 = np.linalg.pinv(sqrtm(gram_1))
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

    return ps_test_cases, op_test_cases, dense_test_cases


class TestData:  # pylint: disable=attribute-defined-outside-init
    """A base class to provide shared, seed-dependent test case fixtures."""

    @pytest.fixture(autouse=True)
    def setup_test_cases(self, seed):
        """Generate test cases for the class instance."""
        self.ps_cases, self.op_cases, self.dense_cases = _generate_test_cases(seed)

    @pytest.fixture(params=[0, 1, 2, 3], ids=["case0", "case1", "case2", "case3"])
    def ps_test_case(self, request):
        """Fixture to provide individual PS test cases dynamically."""
        return self.ps_cases[request.param]

    @pytest.fixture(params=[0, 1, 2, 3], ids=["case0", "case1", "case2", "case3"])
    def op_test_case(self, request):
        """Fixture to provide individual Operation test cases dynamically."""
        return self.op_cases[request.param]

    @pytest.fixture(params=[0, 1, 2, 3], ids=["case0", "case1", "case2", "case3"])
    def dense_test_case(self, request):
        """Fixture to provide individual Dense test cases dynamically."""
        return self.dense_cases[request.param]


class TestAdjvecToOp(TestData):
    """Test adjvec_to_op."""

    def test_NotImplementedError(self):
        """Test that NotImplementedError is raised"""
        with pytest.raises(
            NotImplementedError, match="At least one operator in the specified basis"
        ):
            _ = adjvec_to_op([[0.0]], [qml.pauli.PauliWord({0: "X"})])

    @pytest.mark.parametrize("vspace", [True, False])
    def test_with_ps(self, ps_test_case, vspace):
        """Test ``adjvec_to_op`` with a basis of ``PauliSentence`` operators."""
        adj_vecs, basis, expected, is_ortho = ps_test_case

        if vspace:
            basis = qml.pauli.PauliVSpace(basis)
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

    def test_with_op(self, op_test_case):
        """Test ``adjvec_to_op`` with a basis of ``Operator`` operators."""
        adj_vecs, basis, expected, is_ortho = op_test_case

        out = adjvec_to_op(adj_vecs, basis, is_orthogonal=False)
        for out_op, exp_op in zip(out, expected):
            assert qml.equal(out_op.simplify(), exp_op.simplify())
        if is_ortho:
            out = adjvec_to_op(adj_vecs, basis, is_orthogonal=True)
            for out_op, exp_op in zip(out, expected):
                assert qml.equal(out_op.simplify(), exp_op.simplify())

    def test_with_dense(self, dense_test_case):
        """Test ``adjvec_to_op`` with a basis of dense operators."""
        adj_vecs, basis, expected, is_ortho = dense_test_case

        out = adjvec_to_op(adj_vecs, basis, is_orthogonal=False)
        assert qml.math.shape(out) == qml.math.shape(expected)
        assert np.allclose(out, expected)
        if is_ortho:
            out = adjvec_to_op(adj_vecs, basis, is_orthogonal=True)
            assert qml.math.shape(out) == qml.math.shape(expected)
            assert np.allclose(out, expected)


class TestOpToAdjvec(TestData):
    """Test op_to_adjvec. We reuse the test cases from adjvec_to_op and simply re-interpret which
    part is passed to the function, and which represents the expected output."""

    def test_NotImplementedError(self):
        """Test that NotImplementedError is raised"""
        from fractions import Fraction

        with pytest.raises(
            NotImplementedError, match="At least one operator in the specified basis"
        ):
            _ = op_to_adjvec([Fraction(2)], [Fraction(2)])

        with pytest.raises(
            NotImplementedError, match="At least one operator in the specified basis"
        ):
            _ = op_to_adjvec([Fraction(2)], [1.0])

    def test_op_and_dense(self):
        """Test that an operator is correctly turned into an adjvec when the basis is provided as tensors"""
        H = X(0) + Y(0)
        basis = [X(0), Y(0), Z(0)]
        basis = [qml.matrix(op) for op in basis]

        adjvec = qml.liealg.op_to_adjvec([H], basis)
        assert np.allclose(adjvec, [1, 1, 0])

    @pytest.mark.parametrize("vspace", [True, False])
    def test_with_ps(self, ps_test_case, vspace):
        """Test ``op_to_adjvec`` with a basis of ``PauliSentence`` operators."""
        expected, basis, ops, is_ortho = ps_test_case

        if vspace:
            basis = qml.pauli.PauliVSpace(basis)

        out = op_to_adjvec(ops, basis, is_orthogonal=False)
        assert out.dtype == np.float64
        assert qml.math.shape(out) == qml.math.shape(expected)
        assert np.allclose(out, expected)
        if is_ortho:
            out = op_to_adjvec(ops, basis, is_orthogonal=True)
            assert out.dtype == np.float64
            assert qml.math.shape(out) == qml.math.shape(expected)
            assert np.allclose(out, expected)

    def test_with_op(self, op_test_case):
        """Test ``op_to_adjvec`` with a basis of ``Operator`` operators."""
        expected, basis, ops, is_ortho = op_test_case

        out = op_to_adjvec(ops, basis, is_orthogonal=False)
        assert out.dtype == np.float64
        assert qml.math.shape(out) == qml.math.shape(expected)
        assert np.allclose(out, expected)
        if is_ortho:
            out = op_to_adjvec(ops, basis, is_orthogonal=True)
            assert out.dtype == np.float64
            assert qml.math.shape(out) == qml.math.shape(expected)
            assert np.allclose(out, expected)

    def test_with_dense(self, dense_test_case):
        """Test ``op_to_adjvec`` with a basis of dense operators."""
        expected, basis, ops, is_ortho = dense_test_case

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
        g = qml.lie_closure(g, matrix=True)

        m = g[:32]

        res1 = op_to_adjvec(m, g)

        g = list(qml.pauli.pauli_group(3))  # su(8)
        g = qml.lie_closure(g)
        g = [_.pauli_rep for _ in g]

        m = g[:32]

        res2 = np.array(op_to_adjvec(m, g))
        assert res1.shape == res2.shape
        assert np.allclose(res1, res2)


abelian_group0 = [X(i) for i in range(3)]
non_abelian_group0 = [X(0), Y(0), Z(0)]


def test_check_abelian_ops():
    """Test check_abelian with ops as inputs"""
    assert qml.liealg.check_abelian(abelian_group0)
    assert not qml.liealg.check_abelian(non_abelian_group0)


def test_check_abelian_matrix():
    """Test check_abelian with matrices as inputs"""
    abelian_group = [qml.matrix(op, wire_order=range(3)) for op in abelian_group0]
    assert qml.liealg.check_abelian(abelian_group)
    non_abelian_group = [qml.matrix(op, wire_order=range(3)) for op in non_abelian_group0]
    assert not qml.liealg.check_abelian(non_abelian_group)


def test_check_abelian_ps():
    """Test check_abelian with ps as inputs"""
    abelian_group = [op.pauli_rep for op in abelian_group0]
    assert qml.liealg.check_abelian(abelian_group)
    non_abelian_group = [op.pauli_rep for op in non_abelian_group0]
    assert not qml.liealg.check_abelian(non_abelian_group)


def test_check_abelian_NotImplemented():
    """Test that check_abelian raises NotImplementedError"""
    with pytest.raises(NotImplementedError, match="At least one operator in the"):
        _ = qml.liealg.check_abelian([qml.pauli.PauliWord({0: "X"})])
