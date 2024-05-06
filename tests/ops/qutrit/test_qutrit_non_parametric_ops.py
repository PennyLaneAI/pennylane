# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the available non-parametric qutrit operations
"""
import copy

import numpy as np
import pytest
from gate_data import TADD, TCLOCK, TH, TSHIFT, TSWAP

import pennylane as qml
from pennylane.wires import Wires

NON_PARAMETRIZED_OPERATIONS = [
    (qml.TShift, TSHIFT, None),
    (qml.TClock, TCLOCK, None),
    (qml.TAdd, TADD, None),
    (qml.TSWAP, TSWAP, None),
    (qml.THadamard, TH, None),
    (qml.THadamard, np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2), (0, 1)),
    (qml.THadamard, np.array([[1, 0, 1], [0, np.sqrt(2), 0], [1, 0, -1]]) / np.sqrt(2), (0, 2)),
    (qml.THadamard, np.array([[np.sqrt(2), 0, 0], [0, 1, 1], [0, 1, -1]]) / np.sqrt(2), (1, 2)),
]


class TestOperations:
    @pytest.mark.parametrize("op_cls, _, subspace", NON_PARAMETRIZED_OPERATIONS)
    def test_nonparametrized_op_copy(self, op_cls, _, subspace, tol):
        """Tests that copied nonparametrized ops function as expected"""
        op = (
            op_cls(wires=range(op_cls.num_wires))
            if subspace is None
            else op_cls(wires=range(op_cls.num_wires), subspace=subspace)
        )
        copied_op = copy.copy(op)
        np.testing.assert_allclose(op.matrix(), copied_op.matrix(), atol=tol)

    @pytest.mark.parametrize("ops, mat, subspace", NON_PARAMETRIZED_OPERATIONS)
    def test_matrices(self, ops, mat, subspace, tol):
        """Test matrices of non-parametrized operations are correct"""
        op = (
            ops(wires=range(ops.num_wires))
            if subspace is None
            else ops(wires=range(ops.num_wires), subspace=subspace)
        )
        res_static = (
            op.compute_matrix() if subspace is None else op.compute_matrix(subspace=subspace)
        )
        res_dynamic = op.matrix()
        assert np.allclose(res_static, mat, atol=tol, rtol=0)
        assert np.allclose(res_dynamic, mat, atol=tol, rtol=0)


class TestEigenval:
    def test_tshift_eigenval(self):
        """Tests that the TShift eigenvalue matches the numpy eigenvalues of the TShift matrix"""
        op = qml.TShift(wires=0)
        exp = np.linalg.eigvals(op.matrix())
        res = op.eigvals()
        assert np.allclose(res, exp)

    def test_tclock_eigenval(self):
        """Tests that the TClock eigenvalue matches the numpy eigenvalues of the TClock matrix"""
        op = qml.TClock(wires=0)
        exp = np.linalg.eigvals(op.matrix())
        res = op.eigvals()
        assert np.allclose(res, exp)

    def test_tadd_eigenval(self):
        """Tests that the TAdd eigenvalue matches the numpy eigenvalues of the TAdd matrix"""
        op = qml.TAdd(wires=[0, 1])
        exp = np.linalg.eigvals(op.matrix())
        res = op.eigvals()
        assert np.allclose(res, exp)

    def test_tswap_eigenval(self):
        """Tests that the TSWAP eigenvalue matches the numpy eigenvalues of the TSWAP matrix"""
        op = qml.TSWAP(wires=[0, 1])
        exp = np.linalg.eigvals(op.matrix())
        res = op.eigvals()
        assert np.allclose(res, exp)


period_three_ops = [
    qml.TShift(wires=0),
    qml.TClock(wires=0),
    qml.TAdd(wires=[0, 1]),
]

period_two_ops = [
    qml.TSWAP(wires=[0, 1]),
    qml.THadamard(wires=0, subspace=(0, 1)),
    qml.THadamard(wires=0, subspace=(0, 2)),
    qml.THadamard(wires=0, subspace=(1, 2)),
]

no_pow_method_ops = [
    qml.THadamard(wires=0, subspace=None),
]


class TestPowMethod:
    """Tests for the pow method of non-parametric qutrit operations."""

    @pytest.mark.parametrize("op", period_three_ops)
    @pytest.mark.parametrize("offset", (-6, -3, 0, 3, 6))
    def test_period_three_ops_pow_multiple_of_3(self, op, offset):
        """Tests that ops with period == 3 return an empty list when raised to a
        power that's a multiple of 3."""

        assert len(op.pow(0 + offset)) == 0

    @pytest.mark.parametrize("op", period_three_ops)
    @pytest.mark.parametrize("offset", (-6, -3, 0, 3, 6))
    def test_period_three_ops_pow_offset_1(self, op, offset):
        """Tests that ops with a period == 3 return a queued copy of themselves when
        raised to a power that is 1+multiple of three.
        """
        # When raising to power == 1 mod 3
        with qml.queuing.AnnotatedQueue() as q:
            op_pow_1 = op.pow(1 + offset)[0]

        assert q.queue[0] is op_pow_1
        assert qml.equal(op_pow_1, op)

    @pytest.mark.parametrize("op", period_three_ops)
    @pytest.mark.parametrize("offset", (0, 3))
    def test_period_three_ops_pow_offset_2(self, op, offset):
        """Tests that ops with a period ==3 raise a PowUndefinedError when raised
        to a power that is 2+multiple of three."""

        # When raising to power == 2 mod 3
        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(2 + offset)

    @pytest.mark.parametrize("op", period_three_ops + period_two_ops)
    def test_period_two_three_noninteger_power(self, op):
        """Test that ops with a period of 2 or 3 raised to a non-integer power raise an error"""
        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(1.234)

    @pytest.mark.parametrize("offset", [0, 2, -2, 4, -4])
    @pytest.mark.parametrize("op", period_two_ops)
    def test_period_two_pow(self, offset, op):
        """Tests that ops with period == 2 behave correctly when raised to various
        integer powers"""

        assert len(op.pow(0 + offset)) == 0
        assert qml.equal(op.pow(1 + offset)[0], op)

    @pytest.mark.parametrize("op", no_pow_method_ops)
    def test_no_pow_ops(self, op):
        assert len(op.pow(0)) == 0

        op_pow = op.pow(1)
        assert len(op_pow) == 1
        assert op_pow[0].__class__ == op.__class__

        pows = [0.1, 2, -2, -2.5]

        for pow in pows:
            with pytest.raises(qml.operation.PowUndefinedError):
                op.pow(pow)


label_data = [
    (qml.TShift(0), "TShift"),
    (qml.TClock(0), "TClock"),
    (qml.TAdd([0, 1]), "TAdd"),
    (qml.TSWAP([0, 1]), "TSWAP"),
    (qml.THadamard(0), "TH"),
    (qml.THadamard(0, subspace=(0, 1)), "TH"),
]


@pytest.mark.parametrize("op, label", label_data)
def test_label_method(op, label):
    assert op.label() == label
    assert op.label(decimals=2) == label


control_data = [
    (qml.TShift(0), Wires([])),
    (qml.TClock(0), Wires([])),
    (qml.TAdd([0, 1]), Wires([0])),
    (qml.TSWAP([0, 1]), Wires([])),
    (qml.THadamard(wires=0), Wires([])),
]


@pytest.mark.parametrize("op, control_wires", control_data)
def test_control_wires(op, control_wires):
    """Test ``control_wires`` attribute for non-parametrized operations."""

    assert op.control_wires == control_wires


no_adjoint_ops = [  # ops that are not their own inverses
    qml.TShift(wires=0),
    qml.TClock(wires=0),
    qml.TAdd(wires=[0, 1]),
    qml.THadamard(wires=0, subspace=None),
]

involution_ops = [  # ops that are their own inverses
    qml.TSWAP(wires=[0, 1]),
    qml.THadamard(wires=0, subspace=(0, 1)),
    qml.THadamard(wires=0, subspace=(0, 2)),
    qml.THadamard(wires=0, subspace=(1, 2)),
]


@pytest.mark.parametrize("op", no_adjoint_ops)
def test_adjoint_method(op):
    """Assert that ops that are not their own inverses do not have a defined adjoint."""
    assert not op.has_adjoint

    with pytest.raises(qml.operation.AdjointUndefinedError):
        op.adjoint()


@pytest.mark.parametrize("op", involution_ops)
def test_adjoint_method_involution(op):
    """Assert that involution ops are their own adjoint."""
    assert op.has_adjoint

    adj_op = op.adjoint()
    assert qml.equal(adj_op, op)
    assert adj_op is not op
