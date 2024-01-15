# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for non-parametric Operators inheriting from ControlledOp.
"""

import copy

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from gate_data import (
    CY,
    CZ,
)

import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import AnyWires

# Non-parametrized operations and their matrix representation
OPERATIONS = [
    (qml.CY, CY),
    (qml.CZ, CZ),
]

SPARSE_MATRIX_SUPPORTED_OPERATIONS = (
    (qml.CY(wires=[0, 1]), CY),
    (qml.CZ(wires=[0, 1]), CZ),
)

DECOMPOSITIONS = (
    (qml.CY, [qml.CRY(np.pi, wires=[0, 1]), qml.S(0)]),
    (qml.CZ, [qml.ctrl(qml.PhaseShift(np.pi, wires=1), 0)]),
)

X = np.array([[0, 1], [1, 0]])
X_broadcasted = np.array([X] * 3)


class TestControlledOperations:
    @pytest.mark.parametrize("op_cls, _", OPERATIONS)
    def test_non_parametrized_op_copy(self, op_cls, _, tol):
        """Tests that copied non-parametrized ops function as expected"""
        op = op_cls(wires=0 if op_cls.num_wires is AnyWires else range(op_cls.num_wires))
        copied_op = copy.copy(op)
        assert qml.equal(copied_op, op, atol=tol, rtol=0)
        assert copied_op is not op
        assert qml.equal(copied_op, op, atol=tol, rtol=0)

    @pytest.mark.parametrize("ops, mat", OPERATIONS)
    def test_matrices(self, ops, mat, tol):
        """Test matrices of non-parametrized operations are correct"""
        op = ops(wires=0 if ops.num_wires is AnyWires else range(ops.num_wires))
        res_static = op.compute_matrix()
        res_dynamic = op.matrix()
        assert np.allclose(res_static, mat, atol=tol, rtol=0)
        assert np.allclose(res_dynamic, mat, atol=tol, rtol=0)

    @pytest.mark.parameterize("ops, decomp_ops")
    def test_decompositions(self, ops, expected_ops, tol):
        """Tests that decompositions of non-parametrized operations are correct"""
        op = ops(wires=[0, 1])
        decomps = op.decomposition()
        decomposed_matrix = qml.matrix(op.decomposition)()

        for gate, expected in zip(decomps, expected_ops):
            assert qml.equal(gate, expected, atol=tol, rtol=0)

        assert np.allclose(decomposed_matrix, op.matrix(), atol=tol, rtol=0)

    @pytest.mark.parametrize("op, _", OPERATIONS)
    def test_eigenval(self, op, _):
        """Tests that the CZ eigenvalue matches the numpy eigenvalues of the CZ matrix"""
        op = qml.CZ(wires=[0, 1])
        exp = np.linalg.eigvals(op.matrix())
        res = op.eigvals()
        assert np.allclose(res, exp)


period_two_ops = (
    qml.CY(wires=(0, 1)),
    qml.CZ(wires=(0, 1)),
)


class TestPowMethod:
    @pytest.mark.parametrize("op", period_two_ops)
    @pytest.mark.parametrize("n", (1, 5, -1, -5))
    def test_period_two_pow_odd(self, op, n):
        """Test that ops with a period of 2 raised to an odd power are the same as the original op."""
        assert qml.equal(op.pow(n)[0], op)
        assert np.allclose(op.pow(n)[0].matrix(), op.matrix())
        assert op.pow(n)[0].name == op.name

    @pytest.mark.parametrize("op", period_two_ops)
    @pytest.mark.parametrize("n", (2, 6, 0, -2))
    def test_period_two_pow_even(self, op, n):
        """Test that ops with a period of 2 raised to an even power are empty lists."""
        assert len(op.pow(n)) == 0

    @pytest.mark.parametrize("op", period_two_ops)
    def test_period_two_noninteger_power(self, op):
        """Test that ops with a period of 2 raised to a non-integer power raise an error."""
        if isinstance(op, (qml.PauliZ, qml.CZ)):
            pytest.skip("PauliZ can be raised to any power.")
        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(1.234)

        if op.__class__ is qml.CZ:
            pytest.skip("CZ can be raised to any power.")
        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(1.234)

    @pytest.mark.parametrize("n", (0.12, -3.462, 3.693))
    def test_cz_general_power(self, n):
        """Check that CZ raised to an non-integer power that's not the square root
        results in a controlled PhaseShift."""
        op_pow = qml.CZ(wires=[0, 1]).pow(n)

        assert len(op_pow) == 1
        assert isinstance(op_pow[0], qml.ops.ControlledOp)
        assert isinstance(op_pow[0].base, qml.PhaseShift)
        assert qml.math.allclose(op_pow[0].data[0], np.pi * (n % 2))


class TestControlledMethod:  # pylint: disable=too-few-public-methods
    """Tests for the _controlled method of non-parametric operations."""

    def test_CZ(self):
        """Test the PauliZ _controlled method."""
        out = qml.CZ(wires=[0, 1])._controlled("a")  # pylint: disable=protected-access
        assert qml.equal(out, qml.CCZ(("a", 0, 1)))


class TestSparseMatrix:  # pylint: disable=too-few-public-methods
    @pytest.mark.parametrize("op, mat", SPARSE_MATRIX_SUPPORTED_OPERATIONS)
    def test_sparse_matrix(self, op, mat):
        """Tests the sparse matrix method for operations which support it."""
        expected_sparse_mat = csr_matrix(mat)
        sparse_mat = op.sparse_matrix()

        assert isinstance(sparse_mat, csr_matrix)
        assert isinstance(expected_sparse_mat, csr_matrix)
        assert all(sparse_mat.data == expected_sparse_mat.data)
        assert all(sparse_mat.indices == expected_sparse_mat.indices)


label_data = [
    (qml.CY(wires=(0, 1)), "Y"),
    (qml.CZ(wires=(0, 1)), "Z"),
]


@pytest.mark.parametrize("op, label", label_data)
def test_label_method(op, label):
    """Tests that the label method gives the expected result."""
    assert op.label() == label
    assert op.label(decimals=2) == label


control_data = [
    (qml.CY(wires=(0, 1)), Wires(0)),
    (qml.CZ(wires=(0, 1)), Wires(0)),
]


@pytest.mark.parametrize("op, control_wires", control_data)
def test_control_wires(op, control_wires):
    """Test ``control_wires`` attribute for non-parametrized operations."""

    assert op.control_wires == control_wires


involution_ops = [  # ops who are their own inverses
    qml.CY((0, 1)),
    qml.CZ(wires=(0, 1)),
]


@pytest.mark.parametrize("op", involution_ops)
def test_adjoint_method(op):
    """Tests the adjoint method for operations that are their own adjoint."""
    adj_op = copy.copy(op)
    for _ in range(4):
        adj_op = adj_op.adjoint()

        assert qml.equal(adj_op, op)


@pytest.mark.parametrize("op_cls, _", OPERATIONS)
def test_map_wires(op_cls, _):
    """Test that we can get and set private wires in all operations."""

    op = op_cls(wires=[0, 1])
    assert op.wires == Wires((0, 1))

    op = op.map_wires(wire_map={0: "a", 1: "b"})
    assert op.base.wires == Wires(("b"))
    assert op.control_wires == Wires(("a"))
