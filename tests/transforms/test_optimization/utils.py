# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convenient utility functions for testing optimization transforms."""

import pennylane as qml
from pennylane import numpy as np
from gate_data import I, SWAP


def compute_matrix_from_ops_one_qubit(ops):
    """Given a list of single-qubit operations, construct its matrix representation."""

    mat = I

    for op in ops:
        mat = qml.math.dot(op.matrix, mat)
    return mat


def compute_matrix_from_ops_two_qubit(ops, wire_order):
    """Given a list of two-qubit operations, construct its matrix representation."""

    mat = np.eye(4)

    wire_order = qml.wires.Wires(wire_order)

    for op in ops:
        op_wires = qml.wires.Wires(op.wires)

        if len(op_wires) == 1:
            # These first two cases are to cover tensorflow quirks
            if op.name == "RZ":
                op_mat = qml.RZ(*qml.math.unwrap(op.parameters), wires=0).matrix
            elif op.name == "Rot":
                op_mat = qml.Rot(*qml.math.unwrap(op.parameters), wires=0).matrix
            else:
                op_mat = qml.math.unwrap(op.matrix)

            if op_wires[0] == wire_order[0]:
                tensor_prod = np.kron(op_mat, I)
            else:
                tensor_prod = np.kron(I, op_mat)

            mat = np.dot(tensor_prod, mat)

        else:
            if op_wires == wire_order:
                mat = np.dot(op.matrix, mat)
            else:
                mat = np.linalg.multi_dot([SWAP, op.matrix, SWAP, mat])

    return mat


def check_matrix_equivalence(matrix_expected, matrix_obtained, atol=1e-8):
    """Takes two matrices and checks if multiplying one by the conjugate
    transpose of the other gives the identity."""

    mat_product = qml.math.dot(qml.math.conj(qml.math.T(matrix_obtained)), matrix_expected)
    mat_product = mat_product / mat_product[0, 0]

    return qml.math.allclose(mat_product, qml.math.eye(matrix_expected.shape[0]), atol=atol)


def compare_operation_lists(ops_obtained, names_expected, wires_expected):
    """Compare two lists of operations."""
    assert len(ops_obtained) == len(names_expected)
    assert all([op.name == exp_name for (op, exp_name) in zip(ops_obtained, names_expected)])
    assert all([op.wires == exp_wires for (op, exp_wires) in zip(ops_obtained, wires_expected)])
