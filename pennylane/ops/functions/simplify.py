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
from typing import Callable, Union, Sequence

import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.workflow import QNode
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumTape


def simplify(input: Union[Operator, MeasurementProcess, QuantumTape, QNode, Callable]):
    """Simplifies an operator, tape, qnode or quantum function by reducing its arithmetic depth
    or number of rotation parameters.

    Args:
        input (.Operator, .MeasurementProcess, pennylane.QNode, .QuantumTape, or Callable): an
            operator, quantum node, tape or function that applies quantum operations

    Returns:
        (Operator or MeasurementProcess or qnode (QNode) or quantum function (Callable)
        or tuple[List[QuantumTape], function]): Simplified input. If an operator or measurement
        process is provided as input, the simplified input is returned directly. Otherwise, the
        transformed circuit is returned as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    Given an instantiated operator, ``qml.simplify`` reduces the operator's arithmetic depth:

    >>> op = qml.adjoint(qml.RX(0.54, wires=0) + qml.X(0) + qml.Z(1))
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

    >>> op = qml.adjoint(qml.U2(-np.pi/2, np.pi/2, wires=0) + qml.X(0))
    >>> op
    Adjoint(Sum)([-1.5707963267948966, 1.5707963267948966], [], wires=[0])
    >>> qml.simplify(op)
    Adjoint(RX)(1.5707963267948966, wires=[0]) + Adjoint(PauliX)(wires=[0])

    Moreover, ``qml.simplify`` can be used to simplify QNodes or quantum functions:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev)
    ... @qml.simplify
    ... def circuit():
    ...     qml.adjoint(qml.prod(qml.RX(1, 0) ** 1, qml.RY(1, 0), qml.RZ(1, 0)))
    ...     return qml.probs(wires=0)
    >>> circuit()
    tensor([0.64596329, 0.35403671], requires_grad=True)
    >>> list(circuit.tape)
    [RZ(11.566370614359172, wires=[0]) @ RY(11.566370614359172, wires=[0]) @ RX(11.566370614359172, wires=[0]),
     probs(wires=[0])]
    """
    if isinstance(input, (Operator, MeasurementProcess)):
        if QueuingManager.recording():
            with QueuingManager.stop_recording():
                new_op = copy(input.simplify())
            QueuingManager.remove(input)
            return qml.apply(new_op)
        return input.simplify()

    if isinstance(input, QuantumScript) or callable(input):
        return _simplify_transform(input)

    raise ValueError(f"Cannot simplify the object {input} of type {type(input)}.")


@qml.transform
def _simplify_transform(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):
    with qml.QueuingManager.stop_recording():
        new_operations = [op.simplify() for op in tape.operations]
        new_measurements = [m.simplify() for m in tape.measurements]

    new_tape = type(tape)(new_operations, new_measurements, shots=tape.shots)

    def null_processing_fn(res):
        return res[0]

    return [new_tape], null_processing_fn
