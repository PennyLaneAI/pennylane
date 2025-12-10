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
from __future__ import annotations

from collections.abc import Callable
from typing import overload

import pennylane as qml
from pennylane import transform
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn
from pennylane.workflow import QNode


@overload
def map_wires(
    input: Operator, wire_map: dict, queue: bool = False, replace: bool = False
) -> Operator: ...
@overload
def map_wires(
    input: MeasurementProcess, wire_map: dict, queue: bool = False, replace: bool = False
) -> MeasurementProcess: ...
@overload
def map_wires(
    input: QuantumScript, wire_map: dict, queue: bool = False, replace: bool = False
) -> tuple[QuantumScriptBatch, PostprocessingFn]: ...
@overload
def map_wires(
    input: QNode, wire_map: dict, queue: bool = False, replace: bool = False
) -> QNode: ...
@overload
def map_wires(
    input: Callable, wire_map: dict, queue: bool = False, replace: bool = False
) -> Callable: ...
@overload
def map_wires(
    input: QuantumScriptBatch, wire_map: dict, queue: bool = False, replace: bool = False
) -> tuple[QuantumScriptBatch, PostprocessingFn]: ...
@transform
def map_wires(
    input: Operator | MeasurementProcess | QuantumScript | QNode | Callable | QuantumScriptBatch,
    wire_map: dict,
    queue=False,
    replace=False,
):  # pylint: disable=unused-argument
    """Changes the wires of an operator, tape, qnode or quantum function according
    to the given wire map.

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

        .. code-block:: python

            dev = qml.device("default.qubit")
            wire_map = {0: 10}

            @qml.map_wires(wire_map=wire_map)
            @qml.qnode(dev)
            def func(x):
                qml.RX(x, wires=0)
                return qml.expval(qml.Z(0))

        >>> print(qml.draw(func)(0.1))
        10: ──RX(0.10)─┤  <Z>


    **Example**

    Given an operator, ``qml.map_wires`` returns a copy of the operator with its wires changed:

    >>> op = qml.RX(0.54, wires=0) + qml.X(1) + (qml.Z(2) @ qml.RY(1.23, wires=3))
    >>> op
    (
        RX(0.54, wires=[0])
      + X(1)
      + Z(2) @ RY(1.23, wires=[3])
    )
    >>> wire_map = {0: 3, 1: 2, 2: 1, 3: 0}
    >>> qml.map_wires(op, wire_map)
    (
        RX(0.54, wires=[3])
      + X(2)
      + Z(1) @ RY(1.23, wires=[0])
    )

    Moreover, ``qml.map_wires`` can be used to change the wires of QNodes or quantum functions:

    >>> dev = qml.device("default.qubit", wires=4)
    >>> @qml.qnode(dev)
    ... def circuit():
    ...    qml.RX(0.54, wires=0) @ qml.X(1) @ qml.Z(2) @ qml.RY(1.23, wires=3)
    ...    return qml.probs(wires=0)
    ...
    >>> mapped_circuit = qml.map_wires(circuit, wire_map)
    >>> mapped_circuit()
    array([0.92885434, 0.07114566])
    >>> tape = qml.workflow.construct_tape(mapped_circuit)()
    >>> list(tape)
    [RX(0.54, wires=[3]) @ X(2) @ Z(1) @ RY(1.23, wires=[0]), probs(wires=[3])]
    """
    assert isinstance(input, QuantumScript)
    ops = [map_wires(op, wire_map, queue=queue) for op in input.operations]
    measurements = [map_wires(m, wire_map, queue=queue) for m in input.measurements]
    out = input.copy(ops=ops, measurements=measurements, trainable_params=input.trainable_params)

    def null_processing(res):
        """An empty postprocessing function that leaves the results unchanged."""
        return res[0]

    return (out,), null_processing


@map_wires.register
def _map_op_meas_wires(
    input: Operator | MeasurementProcess, wire_map: dict, queue: bool = False, replace: bool = False
):
    if QueuingManager.recording():
        with QueuingManager.stop_recording():
            new_op = input.map_wires(wire_map=wire_map)
        if replace:
            QueuingManager.remove(input)
        if queue:
            new_op = qml.apply(new_op)
        return new_op
    return input.map_wires(wire_map=wire_map)
