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
from collections.abc import Callable, Sequence
from types import FunctionType

from pennylane import templates
from pennylane.devices.preprocess import decompose
from pennylane.operation import DecompositionUndefinedError, Operation, Operator
from pennylane.ops.op_math import Adjoint
from pennylane.tape import QuantumScript, QuantumScriptBatch, make_qscript
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

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


@transform
def insert(
    tape: QuantumScript,
    op: Callable | type[Operation],
    op_args: tuple | float,
    position: str | list | type[Operation] = "all",
    before: bool = False,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Insert an operation into specified points in an input circuit.

    Circuits passed through this transform will be updated to have the operation, specified by the
    ``op`` argument, added according to the positioning specified in the ``position`` argument. Only
    single qubit operations are permitted to be inserted.

    The type of ``op`` can be either a single operation or a quantum
    function acting on a single wire. A quantum function can be used
    to specify a sequence of operations acting on a single qubit (see the usage details
    for more information).

    Args:
        tape (QNode or QuantumTape or Callable or pennylane.devices.Device): the input circuit to be transformed.
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
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function] or device (pennylane.devices.Device):

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.


    Raises:
        ValueError: if a single operation acting on multiple wires is passed to ``op``
        ValueError: if the requested ``position`` argument is not ``'start'``, ``'end'`` or
            ``'all'`` OR PennyLane Operation

    **Example:**

    The following QNode can be transformed to add noise to the circuit:

    .. code-block:: python

        from functools import partial

        dev = qml.device("default.mixed", wires=2)

        @partial(qml.noise.insert, op=qml.AmplitudeDamping, op_args=0.2, position="end")
        @qml.qnode(dev)
        def f(w, x, y, z):
            qml.RX(w, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=0)
            qml.RX(z, wires=1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

    Executions of this circuit will differ from the noise-free value:

    >>> f(0.9, 0.4, 0.5, 0.6)
    np.float64(0.7548469968854761)
    >>> print(qml.draw(f)(0.9, 0.4, 0.5, 0.6))
    0: ──RX(0.90)─╭●──RY(0.50)──AmplitudeDamping(0.20)─┤ ╭<Z@Z>
    1: ──RY(0.40)─╰X──RX(0.60)──AmplitudeDamping(0.20)─┤ ╰<Z@Z>

    .. details::
        :title: Usage Details

        **Specifying the operation as a quantum function:**

        Instead of specifying ``op`` as a single :class:`~.Operation`, we can instead define a
        quantum function. For example:

        .. code-block:: python

            def op(x, y, wires):
                qml.RX(x, wires=wires)
                qml.PhaseShift(y, wires=wires)

        This operation can be inserted into the following circuit:

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            @partial(qml.noise.insert, op=op, op_args=[0.2, 0.3], position="end")
            def f(w, x, y, z):
                qml.RX(w, wires=0)
                qml.RY(x, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RY(y, wires=0)
                qml.RX(z, wires=1)
                return qml.expval(qml.Z(0) @ qml.Z(1))

        To check this, let's print out the circuit:

        >>> print(qml.draw(f)(0.9, 0.4, 0.5, 0.6))
        0: ──RX(0.90)─╭●──RY(0.50)──RX(0.20)──Rϕ(0.30)─┤ ╭<Z@Z>
        1: ──RY(0.40)─╰X──RX(0.60)──RX(0.20)──Rϕ(0.30)─┤ ╰<Z@Z>

        **Transforming tapes:**

        Consider the following tape:

        .. code-block:: python

            ops = [
                qml.RX(0.9, wires=0),
                qml.RY(0.4, wires=1),
                qml.CNOT((0,1)),
                qml.RY(0.5, wires=0),
                qml.RX(0.6, wires=1)
            ]
            measurements = [qml.expval(qml.Z(0) @ qml.Z(1))]
            tape = qml.tape.QuantumTape(ops, measurements)

        We can add the :class:`~.AmplitudeDamping` channel to the end of the circuit using:

        >>> from pennylane.noise import insert
        >>> [noisy_tape], _ = insert(tape, qml.AmplitudeDamping, 0.05, position="end")
        >>> print(qml.drawer.tape_text(noisy_tape, decimals=2))
        0: ──RX(0.90)─╭●──RY(0.50)──AmplitudeDamping(0.05)─┤ ╭<Z@Z>
        1: ──RY(0.40)─╰X──RX(0.60)──AmplitudeDamping(0.05)─┤ ╰<Z@Z>

        **Transforming devices:**

        Consider the following QNode:

        .. code-block:: python

            dev = qml.device("default.mixed", wires=2)

            def f(w, x, y, z):
                qml.RX(w, wires=0)
                qml.RY(x, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RY(y, wires=0)
                qml.RX(z, wires=1)
                return qml.expval(qml.Z(0) @ qml.Z(1))

            qnode = qml.QNode(f, dev)

        Execution of the circuit on ``dev`` will be noise-free:

        >>> qnode(0.9, 0.4, 0.5, 0.6)
        np.float64(0.8624353588253786)

        However, noise can be easily added to the device:

        >>> dev_noisy = qml.noise.insert(dev, qml.AmplitudeDamping, 0.2)
        >>> qnode_noisy = qml.QNode(f, dev_noisy)
        >>> qnode_noisy(0.9, 0.4, 0.5, 0.6)
        np.float64(0.7294543367428854)
    """

    # decompose templates and their adjoints to fix a bug in the tutorial_error_mitigation demo
    def stop_at(obj):
        if not isinstance(obj, Operator):
            return True
        if not obj.has_decomposition:
            return True
        return not (hasattr(templates, obj.name) or isinstance(obj, Adjoint))

    [tape], _ = decompose(
        tape, stopping_condition=stop_at, name="insert", error=DecompositionUndefinedError
    )

    if not isinstance(op, FunctionType) and op.num_wires != 1:
        raise ValueError("Only single-qubit operations can be inserted into the circuit")

    not_op, req_ops = _check_position(position)

    if position not in ("start", "end", "all") and not_op:
        raise ValueError(
            "Position must be either 'start', 'end', or 'all' (default) OR a PennyLane operation or list of operations."
        )

    if not isinstance(op_args, Sequence):
        op_args = [op_args]
    new_operations = []
    for prep_op in tape.operations[: tape.num_preps]:
        new_operations.append(prep_op)

    if position == "start":
        for w in tape.wires:
            sub_tape = make_qscript(op)(*op_args, wires=w)
            new_operations.extend(sub_tape.operations)

    for circuit_op in tape.operations[tape.num_preps :]:
        if not before:
            new_operations.append(circuit_op)

        if position == "all":
            for w in circuit_op.wires:
                sub_tape = make_qscript(op)(*op_args, wires=w)
                new_operations.extend(sub_tape.operations)

        if req_ops:
            for operation in req_ops:
                if operation == type(circuit_op):
                    for w in circuit_op.wires:
                        sub_tape = make_qscript(op)(*op_args, wires=w)
                        new_operations.extend(sub_tape.operations)

        if before:
            new_operations.append(circuit_op)

    if position == "end":
        for w in tape.wires:
            sub_tape = make_qscript(op)(*op_args, wires=w)
            new_operations.extend(sub_tape.operations)

    new_tape = tape.copy(operations=new_operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
