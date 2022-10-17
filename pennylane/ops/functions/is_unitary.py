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
"""
This module contains the qml.is_unitary function.
"""
import pennylane as qml
from pennylane.operation import Operator


def is_unitary(op: Operator):
    r"""Check if the operation is unitary.

    A matrix is unitary if its adjoint is also its inverse, that is, if

    .. math:: O^\dagger O = OO^\dagger = I

    Args:
        op (~.operation.Operator): the operator to check against

    Returns:
        bool: True if the operation is unitary, False otherwise

    .. note::
        This check might be expensive for large operators.

    **Example**

    >>> op = qml.RX(0.54, wires=0)
    >>> qml.is_unitary(op)
    True
    >>> op2 = op + op
    >>> qml.is_unitary(op2)
    False
    """
    identity_mat = qml.math.eye(2 ** len(op.wires))
    adj_op = qml.adjoint(op)
    op_prod_adjoint_matrix = qml.matrix(qml.prod(op, adj_op))
    return qml.math.allclose(op_prod_adjoint_matrix, identity_mat)
