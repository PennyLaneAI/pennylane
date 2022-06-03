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

from pennylane.ops.op_math import Sum, sum
from pennylane.operation import MatrixUndefinedError
import gate_data as gd  # a file containing matrix rep of each gate

no_mat_ops = (
    qml.Barrier,
    qml.WireCut,
)

single_qubit_non_param_ops = (
    (qml.Identity, gd.I),
    (qml.Hadamard, gd.H),
    (qml.PauliX, gd.X),
    (qml.PauliY, gd.Y),
    (qml.PauliZ, gd.Z),
    (qml.S, gd.S),
    (qml.T, gd.T),
    (qml.SX, gd.SX),
)

single_qubit_parametric_ops = (
    (qml.RX, gd.Rotx),
    (qml.RY, gd.Roty),
    (qml.RZ, gd.Rotz),
    (qml.PhaseShift, gd.Rphi),
    (qml.Rot, gd.Rot3),
    (qml.U1, gd.U1),
    (qml.U2, gd.U2),
)

double_qubit_non_param_ops = (
    (qml.CNOT, gd.CNOT),
    (qml.CZ, gd.CZ),
    (qml.CY, gd.CY),
    (qml.SWAP, gd.SWAP),
    (qml.ISWAP, gd.ISWAP),
    (qml.SISWAP, gd.SISWAP),
)

double_qubit_parametric_ops = (
    (qml.CRX, gd.CRotx),
    (qml.CRY, gd.CRoty),
    (qml.CRZ, gd.CRotz),
    (qml.CRot, gd.CRot3),
    (qml.IsingXX, gd.IsingXX),
    (qml.IsingYY, gd.IsingYY),
    (qml.IsingZZ, gd.IsingZZ),
)

triple_qubit_non_param_ops = (
    (qml.CSWAP, gd.CSWAP),
    (qml.Toffoli, gd.Toffoli),
)


arithmetic_ops = (
    (qml.QubitCarry, _),
    (qml.QubitSum, _),
)

matrix_ops = (
    (qml.QubitUnitary, _),
    (qml.ControlledQubitUnitary, _),
    (qml.DiagonalQubitUnitary, _),
)

observable_ops = (
    (qml.Hermitian, _),
    (qml.Projector, _),
)

templates = (

)


class TestMatrix:

    @pytest.mark.parametrize("op_and_mat1", single_qubit_non_param_ops)
    @pytest.mark.parametrize("op_and_mat2", single_qubit_non_param_ops)
    def test_single_qubit_matrix_non_parametric_ops_two_terms(self, op_and_mat1, op_and_mat2):
        op1, mat1 = op_and_mat1
        op2, mat2 = op_and_mat2
        wires = 0
        sum_op = qml.ops.arithmetic.Sum(op1(wires), op2(wires))

        sum_mat = sum_op.matrix()
        true_mat = mat1 + mat2

        assert(np.allclose(sum_mat, true_mat))

    @pytest.mark.parametrize("op", no_mat_ops)
    def test_error_no_mat(self, op):
        sum_op = sum(op(0), qml.PauliX(2), qml.PauliZ(1))
        with pytest.raises(MatrixUndefinedError):
            mat = sum_op.matrix()

    def test_sparse_hamiltonian(self):
        return


class TestProperties:

    ops = (
        (qml.PauliX(0), qml.PauliZ(0), qml.Hadamard(0)),
        (qml.CNOT(wires=[0,1]), qml.RX(1.23, wires=1), qml.Identity(0)),
        (qml.IsingXX(4.56, wires=[2, 3]), qml.Toffoli(wires=[1, 2, 3]), qml.Rot(0.34, 1.0, 0, wires=0)),
    )

    @pytest.mark.parametrize("ops_lst", ops)
    def test_num_params(self, ops_lst):
        sum_op = sum(ops_lst)
        true_num_params = 0

        for op in ops_lst:
            true_num_params += op.num_params

        assert sum_op.num_params == true_num_params

    @pytest.mark.parametrize("ops_lst", ops)
    def test_num_wires(self, ops_lst):
        sum_op = sum(ops_lst)
        true_num_wires = 0

        for op in ops_lst:
            true_num_wires += op.num_wires

        assert sum_op.num_params == true_num_wires

    @pytest.mark.parametrize("ops_lst", ops)
    def test_is_hermitian(self, ops_lst):
        sum_op = sum(ops_lst)
        true_hermitian_state = True

        for op in ops_lst:
            true_hermitian_state = true_hermitian_state and op.is_hermitian

        assert sum_op.is_hermitian == true_hermitian_state

    @pytest.mark.parametrize("ops_lst", ops)
    def test_queue_catagory(self, ops_lst):
        sum_op = sum(ops_lst)
        assert sum_op._queue_category is None


class TestWrapperFunc:

    def test_sum_private(self):
        return

    def test_sum_top_level(self):
        return


class TestIntegration:

    def test_measurement_process(self):
        return

    def test_diff_measurement_process(self):
        return

    def test_non_hermitian_op_in_measurement_process(self):
        return
