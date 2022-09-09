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
import pytest
import copy
import numpy as np
from scipy.stats import unitary_group

import pennylane as qml
from pennylane.wires import Wires

from gate_data import TSHIFT, TCLOCK, TADD, TSWAP

NON_PARAMETRIZED_OPERATIONS = [
    (qml.TShift, TSHIFT),
    (qml.TClock, TCLOCK),
    (qml.TAdd, TADD),
    (qml.TSWAP, TSWAP),
]


# TODO: Add tests for testing that the decomposition of non-parametric ops is correct


class TestOperations:
    @pytest.mark.parametrize("op_cls, mat", NON_PARAMETRIZED_OPERATIONS)
    def test_nonparametrized_op_copy(self, op_cls, mat, tol):
        """Tests that copied nonparametrized ops function as expected"""
        op = op_cls(wires=range(op_cls.num_wires))
        copied_op = copy.copy(op)
        np.testing.assert_allclose(op.matrix(), copied_op.matrix(), atol=tol)

        op._inverse = True
        copied_op2 = copy.copy(op)
        np.testing.assert_allclose(op.matrix(), copied_op2.matrix(), atol=tol)

    @pytest.mark.parametrize("ops, mat", NON_PARAMETRIZED_OPERATIONS)
    def test_matrices(self, ops, mat, tol):
        """Test matrices of non-parametrized operations are correct"""
        op = ops(wires=range(ops.num_wires))
        res_static = op.compute_matrix()
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


class TestPowMethod:
    @pytest.mark.parametrize("op", period_three_ops)
    @pytest.mark.parametrize("n", (-5, -2, 1, 4, 7))
    def test_period_three_1_mod_3(self, op, n):
        """Tests that ops raised to an integer power == 1 mod 3 are the same as the original op"""
        assert op.pow(n)[0].__class__ is op.__class__

    @pytest.mark.parametrize("op", period_three_ops)
    @pytest.mark.parametrize("n", (-4, -1, 2, 5, 8))
    def test_period_three_2_mod_3(self, op, n):
        """Tests that ops raised to an integer power == 2 mod 3 are the adjoint of the original op"""
        op_pow = op.pow(n)[0]
        assert op_pow.__class__ is op.__class__
        assert np.allclose(op.matrix().conj().T, op_pow.matrix())
        assert op_pow.inverse == True

    @pytest.mark.parametrize("op", period_three_ops)
    @pytest.mark.parametrize("n", (-6, -3, 0, 3, 6))
    def test_period_three_0_mod_3(self, op, n):
        """Tests that ops raised to an integer power == 0 mod 3 are empty lists"""
        assert len(op.pow(n)) == 0

    @pytest.mark.parametrize("op", period_three_ops)
    def test_period_three_noninteger_power(self, op):
        """Test that ops with a period of 3 raised to a non-integer power raise an error."""
        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(1.234)

    @pytest.mark.parametrize("offset", [0, 2, -2, 4, -4])
    def test_tswap_pow(self, offset):
        """Test powers of the TSWAP operator"""
        op = qml.TSWAP(wires=[0, 1])

        assert len(op.pow(0 + offset)) == 0
        assert op.pow(1 + offset)[0].__class__ is qml.TSWAP

        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(1.234)  # Expect error raised for non-integer power


label_data = [
    (qml.TShift(0), "TShift", "TShift⁻¹"),
    (qml.TClock(0), "TClock", "TClock⁻¹"),
    (qml.TAdd([0, 1]), "TAdd", "TAdd⁻¹"),
    (qml.TSWAP([0, 1]), "TSWAP", "TSWAP"),
]


@pytest.mark.parametrize("op, label1, label2", label_data)
def test_label_method(op, label1, label2):
    assert op.label() == label1
    assert op.label(decimals=2) == label1

    op.inv()
    assert op.label() == label2


control_data = [
    (qml.TShift(0), Wires([])),
    (qml.TClock(0), Wires([])),
    (qml.TAdd([0, 1]), Wires([0])),
    (qml.TSWAP([0, 1]), Wires([])),
]


@pytest.mark.parametrize("op, control_wires", control_data)
def test_control_wires(op, control_wires):
    """Test ``control_wires`` attribute for non-parametrized operations."""

    assert op.control_wires == control_wires


adjoint_ops = [  # ops that are not their own inverses
    qml.TShift(wires=0),
    qml.TClock(wires=0),
    qml.TAdd(wires=[0, 1]),
]

involution_ops = [  # ops that are their own inverses
    qml.TSWAP(wires=[0, 1]),
]


@pytest.mark.parametrize("op", adjoint_ops)
def test_adjoint_method(op, tol):
    adj_op = copy.copy(op)
    adj_op = adj_op.adjoint()

    assert adj_op.name == op.name + ".inv"
    assert np.allclose(adj_op.matrix(), op.matrix().conj().T)


@pytest.mark.parametrize("op", involution_ops)
def test_adjoint_method_involution(op, tol):
    adj_op = copy.copy(op)
    adj_op = adj_op.adjoint()

    assert adj_op.name == op.name
    assert np.allclose(adj_op.matrix(), op.matrix())
