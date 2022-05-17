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

from gate_data import *

single_qubit_non_param_ops = (
    (qml.Identity(0), I),
    (qml.Hadamard(0), H),
    (qml.PauliX(0), X),
    (qml.PauliY(0), Y),
    (qml.PauliZ(0), Z),
    (qml.S(0), S),
    (qml.T(0), T),
    (qml.SX(0), SX),
)


class TestMatrix:

    @pytest.mark.parametrize("op_and_mat1", single_qubit_non_param_ops)
    @pytest.mark.parametrize("op_and_mat2", single_qubit_non_param_ops)
    def test_matrix_non_parametric_ops(self, op_and_mat1, op_and_mat2):
        op1, mat1 = op_and_mat1
        op2, mat2 = op_and_mat2
        sum_op = qml.ops.arithmetic.Sum(op1, op2)

        sum_mat = sum_op.matrix()
        true_mat = mat1 + mat2

        assert(np.allclose(sum_mat, true_mat))
