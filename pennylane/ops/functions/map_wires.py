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
from collections.abc import Callable
from typing import Union, overload

from functools import partial

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
def map_wires(
    input: Union[Operator, MeasurementProcess, QuantumScript, QNode, Callable, QuantumScriptBatch],
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
    >>> tape = qml.workflow.construct_tape(mapped_circuit)()
    >>> list(tape)
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
    return _map_wires_transform(input, wire_map=wire_map, queue=queue)


def processing_fn(res):
    """An empty postprocessing function that leaves the results unchanged."""
    return res[0]


# This transform does not work with dynamic wires
def _map_wires_plxpr_transform(
    primitive, tracers, params, targs, tkwargs, state=None
):  # pylint: disable=unused-argument, too-many-arguments, too-many-positional-arguments
    """Implementation for ``_map_wires_transform`` with program capture"""
    # pylint: disable=import-outside-toplevel
    from pennylane.capture import TransformTracer

    if "queue" in tkwargs:
        raise ValueError("The 'queue' argument is not supported in the plxpr transformation.")

    print(
        f"\n\n_map_wires_plxpr_transform called with \nprimitive: {primitive}, \ntracers: {tracers}, \nparams: {params}, \ntargs: {targs}, \ntkwargs: {tkwargs}, \nstate: {state}\n"
    )

    # With this, only the first operator is mapped
    # wire_map = tkwargs.pop("wire_map", None)
    wire_map = tkwargs.get("wire_map", None)

    # This is None if 'n_wires' is not in the params
    n_wires = params.get("n_wires")

    with qml.QueuingManager.stop_recording():
        # With this, the arguments of operators do not have TransformTracers
        tracers_in = [t.val for t in tracers]
        op = primitive.impl(*tracers_in, **params)

    if isinstance(op, MeasurementProcess):

        meas_tracers = []
        for t in tracers:
            if isinstance(t, TransformTracer):

                if len(op.wires) > 0:

                    meas_tracers.append(
                        TransformTracer(t._trace, *op.map_wires(wire_map=wire_map).wires, t.idx + 1)
                    )

                else:
                    meas_tracers.append(TransformTracer(t._trace, t.val, t.idx + 1))

            else:
                meas_tracers.append(t)

        return primitive.bind(*meas_tracers, **params)

    # TODO: complete this
    # This is for nested operations
    if n_wires is None:

        print(f"op: {op}")
        mapped_op = op.map_wires(wire_map=wire_map)
        print(f"mapped_op: {mapped_op}")

        inner_abs_ops = []

        for idx, single_mapped_op in enumerate(mapped_op.operands):
            roba = single_mapped_op._primitive.bind(
                *single_mapped_op.parameters, *single_mapped_op.wires
            )
            inner_abs_ops.append(roba)

        op_tracers = []
        for idx, t in enumerate(tracers):
            if isinstance(t, TransformTracer):

                op_tracers.append(
                    TransformTracer(
                        t._trace,
                        inner_abs_ops[idx],
                        # *t.val.map_wires(wire_map=wire_map).wires,
                        t.idx + 1,
                    )
                )
            else:
                op_tracers.append(t)

        return primitive.bind(*op_tracers, **params)

    split = None if n_wires == 0 else -n_wires
    tracers_no_wires = tracers[:split]
    tracers_wires = tracers[split:]

    print(f"split: {split}")
    print(f"tracers_no_wires: {tracers_no_wires}")
    print(f"tracers_wires: {tracers_wires}")

    op_tracers = []
    for t in tracers_no_wires:
        if isinstance(t, TransformTracer):
            op_tracers.append(
                TransformTracer(
                    t._trace,
                    t.val,
                    t.idx + 1,
                )
            )
        else:
            op_tracers.append(t)

    # For each wire, there is a separate TransformTracer,
    # and the wire is the value of the TransformTracer

    print(f"op: {op}")

    # We disable the capture to call the map_wires method of the operator.
    # The reason is that for ControlledOp the map_wires method binds the base operation
    # with capture enabled
    qml.capture.disable()
    try:
        mapped_wires = op.map_wires(wire_map=wire_map).wires
    finally:
        qml.capture.enable()

    print(f"mapped_wires: {mapped_wires}")

    for idx, t in enumerate(tracers_wires):
        if isinstance(t, TransformTracer):
            # For this to work as expected, the wires should not be TransformTracers
            op_tracers.append(
                TransformTracer(
                    t._trace,
                    mapped_wires[idx],
                    # or
                    # wire_map.get(t.val, t.val),
                    t.idx + 1,
                )
            )
        else:
            op_tracers.append(t)

    return primitive.bind(*op_tracers, **params)


@partial(transform, plxpr_transform=_map_wires_plxpr_transform)
def _map_wires_transform(
    tape: QuantumScript, wire_map=None, queue=False
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
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

    return (out,), processing_fn
