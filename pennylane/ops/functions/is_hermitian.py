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
This module contains the qml.is_hermitian function.
"""
import pennylane as qml
from pennylane.operation import Operator


def is_hermitian(op: Operator):
    r"""Check if the operation is hermitian.

    A hermitian matrix is a complex square matrix that is equal to its own adjoint

    .. math:: O^\dagger = O

    Args:
        op (~.operation.Operator): the operator to check against

    Returns:
        bool: True if the operation is hermitian, False otherwise

    .. note::
        This check might be expensive for large operators.

    **Example**

    >>> op = qml.X(0)
    >>> qml.is_hermitian(op)
    True
    >>> op2 = qml.RX(0.54, wires=0)
    >>> qml.is_hermitian(op2)
    False
    """
    if op.is_hermitian is True:
        return True
    return qml.math.allclose(qml.matrix(op), qml.matrix(qml.adjoint(op)))
