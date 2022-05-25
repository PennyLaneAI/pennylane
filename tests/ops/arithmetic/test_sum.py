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
    (qml.U3, gd.U3),
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
    (),
)

triple_qubit_non_param_ops = (
    (qml.CSWAP, gd.CSWAP),
    (qml.Toffoli, gd.Toffoli),
)


class TestMatrix:

    @pytest.mark.parametrize("op_and_mat1", single_qubit_non_param_ops)
    @pytest.mark.parametrize("op_and_mat2", single_qubit_non_param_ops)
    def test_single_qubit_matrix_non_parametric_ops(self, op_and_mat1, op_and_mat2):
        op1, mat1 = op_and_mat1
        op2, mat2 = op_and_mat2
        wires = 0
        sum_op = qml.ops.arithmetic.Sum(op1(wires), op2(wires))

        sum_mat = sum_op.matrix()
        true_mat = mat1 + mat2

        assert(np.allclose(sum_mat, true_mat))

