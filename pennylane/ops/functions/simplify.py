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
This module contains the qml.simplify function.
"""
from copy import copy

import pennylane as qml
from pennylane.operation import Operator


@qml.op_transform
def simplify(op: Operator):
    """Simplifies the operator by reducing arithmetic depth or number of rotation
    parameters.


    Args:
        op (.Operator): an operator

    Returns:
        .Operator: simplified operator

    **Example**

    Given an instantiated operator, ``qml.simplify`` reduces the operator's arithmetic depth:

    >>> op = qml.adjoint(qml.RX(0.54, wires=0) + qml.PauliX(0) + qml.PauliZ(1))
    >>> op.arithmetic_depth
    3
    >>> sim_op = qml.simplify(op)
    >>> sim_op.arithmetic_depth
    2
    >>> type(sim_op)
    pennylane.ops.op_math.sum.Sum
    >>> sim_op.summands
    (Adjoint(RX)(0.54, wires=[0]),
    Adjoint(PauliX)(wires=[0]),
    Adjoint(PauliZ)(wires=[1]))

    This function also can simplify the number of rotation gate parameters:

    >>> qml.simplify(qml.Rot(np.pi / 2, 0.1, -np.pi / 2, wires=0))
    RX(0.1, wires=[0])

    """
    if qml.queuing.QueuingContext.recording():
        with qml.tape.stop_recording():
            new_op = copy(op.simplify())
        qml.queuing.QueuingContext.safe_update_info(op, owner=new_op)
        return qml.apply(new_op)
    return op.simplify()
