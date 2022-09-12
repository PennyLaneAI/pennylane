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
from functools import wraps
from typing import Callable, Union

import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.qnode import QNode
from pennylane.queuing import QueuingContext
from pennylane.tape import QuantumTape, stop_recording


def simplify(input: Union[Operator, MeasurementProcess, QuantumTape, QNode, Callable]):
    """Simplifies an operator, tape, qnode or quantum function by reducing its arithmetic depth
    or number of rotation parameters.

    Args:
        input (.Operator, pennylane.QNode, .QuantumTape, or Callable): an operator, quantum node,
            tape or function that applies quantum operations

    Returns:
        (.Operator, pennylane.QNode, .QuantumTape, or Callable): Simplified input.

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
    >>> sim_op.operands
    (Adjoint(RX)(0.54, wires=[0]),
    Adjoint(PauliX)(wires=[0]),
    Adjoint(PauliZ)(wires=[1]))

    This function can also simplify the number of rotation gate parameters:

    >>> qml.simplify(qml.Rot(np.pi / 2, 0.1, -np.pi / 2, wires=0))
    RX(0.1, wires=[0])

    Both types of simplification occur together:

    >>> op = qml.adjoint(qml.U2(-np.pi/2, np.pi/2, wires=0) + qml.PauliX(0))
    >>> op
    Adjoint(Sum)([-1.5707963267948966, 1.5707963267948966], [], wires=[0])
    >>> qml.simplify(op)
    Adjoint(RX)(1.5707963267948966, wires=[0]) + Adjoint(PauliX)(wires=[0])

    Moreover, ``qml.simplify`` can be used to simplify QNodes or quantum functions:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.simplify
        @qml.qnode(dev)
        def circuit():
            qml.adjoint(qml.prod(qml.RX(1, 0) ** 1, qml.RY(1, 0), qml.RZ(1, 0)))
            return qml.probs(wires=0)
    >>> circuit()
    tensor([[0.64596329, 0.35403671]], requires_grad=True)
    >>> list(circuit.tape)
    [RZ(-1, wires=[0]) @ RY(-1, wires=[0]) @ RX(-1, wires=[0]), probs(wires=[0])]
    """
    if isinstance(input, (Operator, MeasurementProcess)):
        if QueuingContext.recording():
            with stop_recording():
                new_op = copy(input.simplify())
            QueuingContext.safe_update_info(input, owner=new_op)
            return qml.apply(new_op)
        return input.simplify()

    if isinstance(input, QuantumTape):
        with QuantumTape() as new_tape:
            for op in list(input):
                _ = qml.simplify(op)

        return new_tape

    if callable(input):

        func = input.func if isinstance(input, QNode) else input

        @wraps(func)
        def qfunc(*args, **kwargs):
            tape = QuantumTape()
            with stop_recording(), tape:
                func(*args, **kwargs)

            _ = [qml.simplify(op) for op in tape.operations]
            m = tuple(qml.simplify(m) for m in tape.measurements)
            return m[0] if len(m) == 1 else m

        if isinstance(input, QNode):
            return QNode(
                func=qfunc,
                device=input.device,
                interface=input.interface,
                diff_method=input.diff_method,
                expansion_strategy=input.expansion_strategy,
                **input.execute_kwargs,
                **input.gradient_kwargs,
            )
        return qfunc

    raise ValueError(f"Cannot simplify the object {input} of type {type(input)}.")
