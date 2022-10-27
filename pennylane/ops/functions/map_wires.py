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
from functools import wraps
from typing import Callable, Union

import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.qnode import QNode
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumTape


def map_wires(
    input: Union[Operator, MeasurementProcess, QuantumScript, QNode, Callable],
    wire_map: dict,
    queue=False,
    replace=False,
):
    """Changes the wires of an operator, tape, qnode or quantum function according to the given
    wire map.

    Args:
        wire_map (dict): dictionary containing the old wires as keys and the new wires as values
        queue (bool): Whether or not to queue the object when recording. Defaults to False.
        replace (bool): When ``queue=True``, if ``replace=True`` the input operators will be
            replaced by its mapped version. Defaults to False.

    Returns:
        (.Operator, pennylane.QNode, .QuantumScript, or Callable): input with changed wires

    .. note::

        ``qml.map_wires`` can be used as a decorator with the help of the ``functools`` module:

        >>> @functools.partial(qml.map_wires, wire_map=wire_map)
        ... @qml.qnode(dev)
        ... def func(x):
        ...     qml.RX(x, wires=0)
        ...     return qml.expval(qml.PauliZ(0))
        >>> print(qml.draw(func)(0.1))
        10: ──RX(0.10)─┤  <Z>


    **Example**

    Given an operator, ``qml.map_wires`` returns a copy of the operator with its wires changed:

    >>> op = qml.RX(0.54, wires=0) + qml.PauliX(1) + (qml.PauliZ(2) @ qml.RY(1.23, wires=3))
    >>> op
    (RX(0.54, wires=[0]) + PauliX(wires=[1])) + (PauliZ(wires=[2]) @ RY(1.23, wires=[3]))
    >>> wire_map = {0: 3, 1: 2, 2: 1, 3: 0}
    >>> qml.map_wires(op, wire_map)
    (RX(0.54, wires=[3]) + PauliX(wires=[2])) + (PauliZ(wires=[1]) @ RY(1.23, wires=[0]))

    Moreover, ``qml.map_wires`` can be used to change the wires of QNodes or quantum functions:

    >>> dev = qml.device("default.qubit", wires=4)
    >>> @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0) @ qml.PauliX(1) @ qml.PauliZ(2) @ qml.RY(1.23, wires=3)
            return qml.probs(wires=0)
    >>> mapped_circuit = qml.map_wires(circuit, wire_map)
    >>> mapped_circuit()
    tensor([0.92885434, 0.07114566], requires_grad=True)
    >>> list(mapped_circuit.tape)
    [((RX(0.54, wires=[3]) @ PauliX(wires=[2])) @ PauliZ(wires=[1])) @ RY(1.23, wires=[0]),
    probs(wires=[3])]
    """
    if isinstance(input, (Operator, MeasurementProcess)):
        if QueuingManager.recording():
            with QueuingManager.stop_recording():
                new_op = input.map_wires(wire_map=wire_map)
            if replace:
                QueuingManager.update_info(input, owner=new_op)
            if queue:
                qml.apply(new_op)
            return new_op
        return input.map_wires(wire_map=wire_map)

    if isinstance(input, QuantumScript):
        ops = [qml.map_wires(op, wire_map) for op in input._ops]  # pylint: disable=protected-access
        measurements = [qml.map_wires(m, wire_map) for m in input.measurements]
        prep = [qml.map_wires(p, wire_map) for p in input._prep]  # pylint: disable=protected-access

        return QuantumScript(ops=ops, measurements=measurements, prep=prep)

    if callable(input):

        func = input.func if isinstance(input, QNode) else input

        @wraps(func)
        def qfunc(*args, **kwargs):
            tape = QuantumTape()
            with QueuingManager.stop_recording(), tape:
                func(*args, **kwargs)

            _ = [qml.map_wires(op, wire_map=wire_map, queue=True) for op in tape.operations]
            m = tuple(qml.map_wires(m, wire_map=wire_map, queue=True) for m in tape.measurements)
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

    raise ValueError(f"Cannot map wires of object {input} of type {type(input)}.")
