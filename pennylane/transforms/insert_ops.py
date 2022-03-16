# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Provides transforms for inserting operations into quantum circuits.
"""
from collections.abc import Sequence
from types import FunctionType
from typing import Type, Union

from pennylane import Device, apply
from pennylane.operation import Operation
from pennylane.tape import QuantumTape
from pennylane.tape.tape import STATE_PREP_OPS
from pennylane.transforms.qfunc_transforms import qfunc_transform

# pylint: disable=too-many-branches


def _check_position(position):
    """Checks the position argument to determine if an operation or list of operations was provided."""
    not_op = False
    req_ops = False
    if isinstance(position, list):
        req_ops = position.copy()
        for operation in req_ops:
            try:
                if Operation not in operation.__bases__:
                    not_op = True
            except AttributeError:
                not_op = True
    elif not isinstance(position, list):
        try:
            if Operation in position.__bases__:
                req_ops = [position]
            else:
                not_op = True
        except AttributeError:
            not_op = True
    return not_op, req_ops


@qfunc_transform
def insert(
    circuit: Union[callable, QuantumTape, Device],
    op: Union[callable, Type[Operation]],
    op_args: Union[tuple, float],
    position: Union[str, list, Type[Operation]] = "all",
    before: bool = False,
) -> Union[callable, QuantumTape]:
    """Insert an operation into specified points in an input circuit.

    Circuits passed through this transform will be updated to have the operation, specified by the
    ``op`` argument, added according to the positioning specified in the ``position`` argument. Only
    single qubit operations are permitted to be inserted.

    The type of ``op`` can be either a single operation or a quantum
    function acting on a single wire. A quantum function can be used
    to specify a sequence of operations acting on a single qubit (see the usage details
    for more information).

    Args:
        circuit (callable or QuantumTape or Device): the input circuit to be transformed, or a
            device
        op (callable or Type[Operation]): the single-qubit operation, or sequence of operations
            acting on a single qubit, to be inserted into the circuit
        op_args (tuple or float): the arguments fed to the operation, either as a tuple or a single
            float
        position (str or PennyLane operation or list of operations): Specification of where to add the operation.
            Should be one of: ``"all"`` to add the operation after all gates (except state preparations);
            ``"start"`` to add the operation to all wires at the start of the circuit (but after state preparations);
            ``"end"`` to add the operation to all wires at the end of the circuit;
            list of operations to add the operation before or after depending on ``before``.
        before (bool): Whether to add the operation before the given operation(s) in ``position``.
            Default is ``False`` and the operation is inserted after.

    Returns:
        callable or QuantumTape or Device: the updated version of the input circuit or an updated
        device which will transform circuits before execution

    Raises:
        ValueError: if a single operation acting on multiple wires is passed to ``op``
        ValueError: if the requested ``position`` argument is not ``'start'``, ``'end'`` or
            ``'all'`` OR PennyLane Operation

    **Example:**

    The following QNode can be transformed to add noise to the circuit:

    .. code-block:: python3

        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev)
        @qml.transforms.insert(qml.AmplitudeDamping, 0.2, position="end")
        def f(w, x, y, z):
            qml.RX(w, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=0)
            qml.RX(z, wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    Executions of this circuit will differ from the noise-free value:

    >>> f(0.9, 0.4, 0.5, 0.6)
    tensor(0.754847, requires_grad=True)
    >>> print(qml.draw(f)(0.9, 0.4, 0.5, 0.6))
    0: ──RX(0.90)─╭C──RY(0.50)──AmplitudeDamping(0.20)─┤ ╭<Z@Z>
    1: ──RY(0.40)─╰X──RX(0.60)──AmplitudeDamping(0.20)─┤ ╰<Z@Z>

    .. UsageDetails::

        **Specifying the operation as a quantum function:**

        Instead of specifying ``op`` as a single :class:`~.Operation`, we can instead define a
        quantum function. For example:

        .. code-block:: python3

            def op(x, y, wires):
                qml.RX(x, wires=wires)
                qml.PhaseShift(y, wires=wires)

        This operation can be inserted into the following circuit:

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            @qml.transforms.insert(op, [0.2, 0.3], position="end")
            def f(w, x, y, z):
                qml.RX(w, wires=0)
                qml.RY(x, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RY(y, wires=0)
                qml.RX(z, wires=1)
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        To check this, let's print out the circuit:

        >>> print(qml.draw(f)(0.9, 0.4, 0.5, 0.6))
        0: ──RX(0.90)─╭C──RY(0.50)──RX(0.20)──Rϕ(0.30)─┤ ╭<Z@Z>
        1: ──RY(0.40)─╰X──RX(0.60)──RX(0.20)──Rϕ(0.30)─┤ ╰<Z@Z>

        **Transforming tapes:**

        Consider the following tape:

        .. code-block:: python3

            with qml.tape.QuantumTape() as tape:
                qml.RX(0.9, wires=0)
                qml.RY(0.4, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RY(0.5, wires=0)
                qml.RX(0.6, wires=1)
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        We can add the :class:`~.AmplitudeDamping` channel to the end of the circuit using:

        >>> from pennylane.transforms import insert
        >>> noisy_tape = insert(qml.AmplitudeDamping, 0.05, position="end")(tape)
        >>> print(qml.drawer.tape_text(noisy_tape, decimals=2))
        0: ──RX(0.90)─╭C──RY(0.50)──AmplitudeDamping(0.05)─┤ ╭<Z@Z>
        1: ──RY(0.40)─╰X──RX(0.60)──AmplitudeDamping(0.05)─┤ ╰<Z@Z>

        **Transforming devices:**

        Consider the following QNode:

        .. code-block:: python3

            dev = qml.device("default.mixed", wires=2)

            def f(w, x, y, z):
                qml.RX(w, wires=0)
                qml.RY(x, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RY(y, wires=0)
                qml.RX(z, wires=1)
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

            qnode = qml.QNode(f, dev)

        Execution of the circuit on ``dev`` will be noise-free:

        >>> qnode(0.9, 0.4, 0.5, 0.6)
        tensor(0.86243536, requires_grad=True)

        However, noise can be easily added to the device:

        >>> dev_noisy = qml.transforms.insert(qml.AmplitudeDamping, 0.2)(dev)
        >>> qnode_noisy = qml.QNode(f, dev_noisy)
        >>> qnode_noisy(0.9, 0.4, 0.5, 0.6)
        tensor(0.72945434, requires_grad=True)
    """
    circuit = circuit.expand(stop_at=lambda op: not isinstance(op, QuantumTape))

    if not isinstance(op, FunctionType) and op.num_wires != 1:
        raise ValueError("Only single-qubit operations can be inserted into the circuit")

    not_op, req_ops = _check_position(position)

    if position not in ("start", "end", "all") and not_op:
        raise ValueError(
            "Position must be either 'start', 'end', or 'all' (default) OR a PennyLane operation or list of operations."
        )

    if not isinstance(op_args, Sequence):
        op_args = [op_args]

    num_preps = sum(isinstance(o, STATE_PREP_OPS) for o in circuit.operations)

    for i in range(num_preps):
        apply(circuit.operations[i])

    if position == "start":
        for w in circuit.wires:
            op(*op_args, wires=w)

    for circuit_op in circuit.operations[num_preps:]:
        if not before:
            apply(circuit_op)

        if position == "all":
            for w in circuit_op.wires:
                op(*op_args, wires=w)

        if req_ops:
            for operation in req_ops:
                if operation == type(circuit_op):
                    for w in circuit_op.wires:
                        op(*op_args, wires=w)

        if before:
            apply(circuit_op)

    if position == "end":
        for w in circuit.wires:
            op(*op_args, wires=w)

    for m in circuit.measurements:
        apply(m)
