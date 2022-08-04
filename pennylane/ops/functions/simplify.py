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
import pennylane as qml
from pennylane.operation import Operator
from pennylane.tape import QuantumTape


@qml.op_transform
def simplify(op: Operator, depth=-1):
    """Reduces the depth of nested operators.

    If ``depth`` is not provided or negative, then the operator is reduced to the maximum.

    Args:
        op (.Operator): an operator

    Keyword Args:
        depth (int): Reduced depth. Default is -1.

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

    You can also specify the reduction depth:

    >>> sum_op = qml.op_sum(qml.op_sum(qml.op_sum(qml.PauliX(0), qml.PauliY(0)), qml.PauliZ(0)),
    qml.PauliX(0))
    >>> sum_op.arithmetic_depth
    3
    >>> sim_op = qml.simplify(sum_op, depth=1)
    >>> sim_op.arithmetic_depth
    2
    >>> sim_op.summands
    (PauliX(wires=[0]) + PauliY(wires=[0]), PauliZ(wires=[0]), PauliX(wires=[0]))
    """
    return op.simplify(depth=depth)


@simplify.tape_transform
def _simplify_tape(tape: QuantumTape, depth=-1):
    """Simplify all operators of a quantum tape and return a simplified quantum function."""
    if depth == 0:
        return tape
    return None
