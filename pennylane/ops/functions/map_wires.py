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
This module contains the qml.map_wires function.
"""
from functools import partial
from typing import Callable, Union, Sequence

import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.workflow import QNode
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumTape
from pennylane import transform


def map_wires(
    input: Union[Operator, MeasurementProcess, QuantumTape, QNode, Callable],
    wire_map: dict,
    queue=False,
    replace=False,
):
    """Changes the wires of an operator, tape, qnode or quantum function according to the given
    wire map.

    Args:
        input (Operator or QNode or QuantumTape or Callable): an operator or a quantum circuit.
        wire_map (dict): dictionary containing the old wires as keys and the new wires as values
        queue (bool): Whether or not to queue the object when recording. Defaults to False.
        replace (bool): When ``queue=True``, if ``replace=True`` the input operators will be
            replaced by its mapped version. Defaults to False.

    Returns:
        operator (Operator) or qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:

        The transformed circuit or operator with updated wires in :func:`qml.transform <pennylane.transform>`.

    .. note::

        ``qml.map_wires`` can be used as a decorator with the help of the ``functools`` module:

        >>> dev = qml.device("default.qubit", wires=1)
        >>> wire_map = {0: 10}
        >>>
        >>> @functools.partial(qml.map_wires, wire_map=wire_map)
        ... @qml.qnode(dev)
        ... def func(x):
        ...     qml.RX(x, wires=0)
        ...     return qml.expval(qml.Z(0))
        ...
        >>> print(qml.draw(func)(0.1))
        10: ──RX(0.10)─┤  <Z>


    **Example**

    Given an operator, ``qml.map_wires`` returns a copy of the operator with its wires changed:

    >>> op = qml.RX(0.54, wires=0) + qml.X(1) + (qml.Z(2) @ qml.RY(1.23, wires=3))
    >>> op
    (RX(0.54, wires=[0]) + X(1)) + (Z(2) @ RY(1.23, wires=[3]))
    >>> wire_map = {0: 3, 1: 2, 2: 1, 3: 0}
    >>> qml.map_wires(op, wire_map)
    (RX(0.54, wires=[3]) + X(2)) + (Z(1) @ RY(1.23, wires=[0]))

    Moreover, ``qml.map_wires`` can be used to change the wires of QNodes or quantum functions:

    >>> dev = qml.device("default.qubit", wires=4)
    >>> @qml.qnode(dev)
    ... def circuit():
    ...    qml.RX(0.54, wires=0) @ qml.X(1) @ qml.Z(2) @ qml.RY(1.23, wires=3)
    ...    return qml.probs(wires=0)
    ...
    >>> mapped_circuit = qml.map_wires(circuit, wire_map)
    >>> mapped_circuit()
    tensor([0.92885434, 0.07114566], requires_grad=True)
    >>> list(mapped_circuit.tape)
    [((RX(0.54, wires=[3]) @ X(2)) @ Z(1)) @ RY(1.23, wires=[0]), probs(wires=[3])]
    """
    if isinstance(input, (Operator, MeasurementProcess)):
        if QueuingManager.recording():
            with QueuingManager.stop_recording():
                new_op = input.map_wires(wire_map=wire_map)
            if replace:
                QueuingManager.remove(input)
            if queue:
                qml.apply(new_op)
            return new_op
        return input.map_wires(wire_map=wire_map)
    if isinstance(input, (QuantumScript, QNode)) or callable(input):
        return _map_wires_transform(input, wire_map=wire_map, queue=queue)

    raise ValueError(f"Cannot map wires of object {input} of type {type(input)}.")


@partial(transform)
def _map_wires_transform(
    tape: qml.tape.QuantumTape, wire_map=None, queue=False
) -> (Sequence[qml.tape.QuantumTape], Callable):
    ops = [
        (
            map_wires(op, wire_map, queue=queue)
            if not isinstance(op, QuantumScript)
            else map_wires(op, wire_map, queue=queue)[0][0]
        )
        for op in tape.operations
    ]
    measurements = [map_wires(m, wire_map, queue=queue) for m in tape.measurements]

    out = tape.__class__(
        ops=ops, measurements=measurements, shots=tape.shots, trainable_params=tape.trainable_params
    )

    def processing_fn(res):
        """Defines how matrix works if applied to a tape containing multiple operations."""
        return res[0]

    return [out], processing_fn
