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
Unit tests for the Sum arithmetic class of qubit operations
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import math

from pennylane.ops.op_math import Sum, sum
from pennylane.operation import MatrixUndefinedError
import gate_data as gd  # a file containing matrix rep of each gate

no_mat_ops = (
    qml.Barrier,
    qml.WireCut,
)

non_param_ops = (
    (qml.Identity, gd.I),
    (qml.Hadamard, gd.H),
    (qml.PauliX, gd.X),
    (qml.PauliY, gd.Y),
    (qml.PauliZ, gd.Z),
    (qml.S, gd.S),
    (qml.T, gd.T),
    (qml.SX, gd.SX),
    (qml.CNOT, gd.CNOT),
    (qml.CZ, gd.CZ),
    (qml.CY, gd.CY),
    (qml.SWAP, gd.SWAP),
    (qml.ISWAP, gd.ISWAP),
    (qml.SISWAP, gd.SISWAP),
    (qml.CSWAP, gd.CSWAP),
    (qml.Toffoli, gd.Toffoli),
)

param_ops = (
    (qml.RX, gd.Rotx),
    (qml.RY, gd.Roty),
    (qml.RZ, gd.Rotz),
    (qml.PhaseShift, gd.Rphi),
    (qml.Rot, gd.Rot3),
    (qml.U1, gd.U1),
    (qml.U2, gd.U2),
    (qml.U3, gd.U3),
    (qml.CRX, gd.CRotx),
    (qml.CRY, gd.CRoty),
    (qml.CRZ, gd.CRotz),
    (qml.CRot, gd.CRot3),
    (qml.IsingXX, gd.IsingXX),
    (qml.IsingYY, gd.IsingYY),
    (qml.IsingZZ, gd.IsingZZ),
)


def compare_and_expand_mat(mat1, mat2):
    """helper function which takes two square matrices (of potentially different sizes)
    and expands the smaller matrix until their shapes match."""

    if mat1.size == mat2.size:
        return mat1, mat2

    (smaller_mat, larger_mat, flip_order) = (mat1,  mat2, 0) if mat1.size < mat2.size else (mat2,  mat1, 1)

    while smaller_mat.size < larger_mat.size:
        smaller_mat = math.cast_like(math.kron(smaller_mat,  math.eye(2)), smaller_mat)

    if flip_order:
        return larger_mat, smaller_mat

    return smaller_mat, larger_mat


class TestMatrix:

    @pytest.mark.parametrize("op_and_mat1", non_param_ops)
    @pytest.mark.parametrize("op_and_mat2", non_param_ops)
    def test_non_parametric_ops_two_terms(self, op_and_mat1, op_and_mat2):
        op1, mat1 = op_and_mat1
        op2, mat2 = op_and_mat2
        mat1, mat2 = compare_and_expand_mat(mat1, mat2)

        sum_op = Sum(op1(wires=range(op1.num_wires)), op2(wires=range(op2.num_wires)))
        sum_mat = sum_op.matrix()

        true_mat = mat1 + mat2
        assert(np.allclose(sum_mat, true_mat))

    @pytest.mark.parametrize("op_mat1", param_ops)
    @pytest.mark.parametrize("op_mat2", param_ops)
    def test_parametric_ops_two_terms(self, op_mat1, op_mat2):
        op1, mat1 = op_mat1
        op2, mat2 = op_mat2

        par1 = tuple(range(op1.num_params))
        par2 = tuple(range(op2.num_params))
        mat1, mat2 = compare_and_expand_mat(mat1(*par1), mat2(*par2))

        sum_op = Sum(op1(*par1, wires=range(op1.num_wires)), op2(*par2, wires=range(op2.num_wires)))
        sum_mat = sum_op.matrix()

        true_mat = mat1 + mat2
        assert(np.allclose(sum_mat, true_mat))

    @pytest.mark.parametrize("op", no_mat_ops)
    def test_error_no_mat(self, op):
        sum_op = Sum(op(0), qml.PauliX(2), qml.PauliZ(1))
        with pytest.raises(MatrixUndefinedError):
            sum_op.matrix()

    def test_sum_ops_multi_terms(self):
        return

    def test_sum_ops_multi_wires(self):
        return

    def test_sum_templates(self):
        return

    def test_sum_qchem_ops(self):
        return

    def test_sum_observables(self):
        return

    def test_sum_qubit_unitary(self):
        return


class TestProperties:

    ops = (
        (qml.PauliX(wires=0), qml.PauliZ(wires=0), qml.Hadamard(wires=0)),
        (qml.CNOT(wires=[0, 1]), qml.RX(1.23, wires=1), qml.Identity(wires=0)),
        (qml.IsingXX(4.56, wires=[2, 3]), qml.Toffoli(wires=[1, 2, 3]), qml.Rot(0.34, 1.0, 0, wires=0)),
    )

    @pytest.mark.parametrize("ops_lst", ops)
    def test_num_params(self, ops_lst):
        sum_op = Sum(*ops_lst)
        true_num_params = 0

        for op in ops_lst:
            true_num_params += op.num_params

        assert sum_op.num_params == true_num_params

    @pytest.mark.parametrize("ops_lst", ops)
    def test_num_wires(self, ops_lst):
        sum_op = Sum(*ops_lst)
        true_wires = set()

        for op in ops_lst:
            true_wires = true_wires.union(op.wires.toset())

        assert sum_op.num_wires == len(true_wires)

    @pytest.mark.parametrize("ops_lst", ops)
    def test_is_hermitian(self, ops_lst):
        sum_op = Sum(*ops_lst)
        true_hermitian_state = True

        for op in ops_lst:
            true_hermitian_state = true_hermitian_state and op.is_hermitian

        assert sum_op.is_hermitian == true_hermitian_state

    @pytest.mark.parametrize("ops_lst", ops)
    def test_queue_catagory(self, ops_lst):
        sum_op = Sum(*ops_lst)
        assert sum_op._queue_category is None

    @pytest.mark.parametrize("ops_lst", ops)
    def test_eigendecompostion(self, ops_lst):
        return

    @pytest.mark.parametrize("ops_lst", ops)
    def test_eigen_caching(self, ops_lst):
        return

    @pytest.mark.parametrize("ops_lst", ops)
    def test_diagonalizing_gates(self, ops_lst):
        return

    @pytest.mark.parametrize("ops_lst", ops)
    def test_eigenvals(self, ops_lst):
        return

    # def test_decomposition(self, ops_lst):
    #     sum_op = Sum(*ops_lst)
    #     return


class TestWrapperFunc:

    def test_sum_top_level(self):
        """Test that the top level function constructs an identical instance to one
        created using the class."""

        summands = (qml.PauliX(1), qml.RX(1.23, wires=0), qml.CNOT(wires=[0, 1]))
        id = 'sum_op'
        do_queue = False

        sum_func_op = sum(*summands, id=id, do_queue=do_queue)
        sum_class_op = Sum(*summands, id=id, do_queue=do_queue)

        assert(sum_class_op.summands == sum_func_op.summands)
        assert(sum_class_op.matrix() == sum_func_op.matrix())
        assert(sum_class_op.id == sum_func_op.id)
        assert(sum_class_op.wires == sum_func_op.wires)
        assert(sum_class_op.parameters == sum_func_op.parameters)


class TestPrivateSum:

    def test_sum_private(self):
        return

    def test_error_raised_no_mat(self):
        return

    def test_dtype(self):
        return

    def test_cast_like(self):
        return


class TestIntegration:

    def test_measurement_process(self):
        return

    def test_diff_measurement_process(self):
        return

    def test_non_hermitian_op_in_measurement_process(self):
        return
