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
"""Transform for cancelling adjacent inverse gates in quantum circuits."""
# pylint: disable=too-many-branches
from typing import Sequence, Callable

from pennylane.ops.op_math import Adjoint
from pennylane.wires import Wires
from pennylane.tape import QuantumTape
from pennylane.transforms import transform

from pennylane.ops.qubit.attributes import (
    self_inverses,
    symmetric_over_all_wires,
    symmetric_over_control_wires,
)
from .optimization_utils import find_next_gate


def _ops_equal(op1, op2):
    """Checks if two operators are equal up to class, data, hyperparameters, and wires"""
    return (
        op1.__class__ is op2.__class__
        and (op1.data == op2.data)
        and (op1.hyperparameters == op2.hyperparameters)
        and (op1.wires == op2.wires)
    )


def _are_inverses(op1, op2):
    """Checks if two operators are inverses of each other

    Args:
        op1 (~.Operator)
        op2 (~.Operator)

    Returns:
        Bool
    """
    # op1 is self-inverse and the next gate is also op1
    if op1 in self_inverses and op1.name == op2.name:
        return True

    # op1 is an `Adjoint` class and its base is equal to op2
    if isinstance(op1, Adjoint) and _ops_equal(op1.base, op2):
        return True

    # op2 is an `Adjoint` class and its base is equal to op1
    if isinstance(op2, Adjoint) and _ops_equal(op2.base, op1):
        return True

    return False


@transform
def cancel_inverses(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):
    """Quantum function transform to remove any operations that are applied next to their
    (self-)inverses or adjoint.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.


    **Example**

    You can apply the cancel inverses transform directly on :class:`~.QNode`.

    >>> dev = qml.device('default.qubit', wires=3)

    .. code-block:: python

        @cancel_inverses
        @qml.qnode(device=dev)
        def circuit(x, y, z):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.Hadamard(wires=0)
            qml.RX(x, wires=2)
            qml.RY(y, wires=1)
            qml.X(1)
            qml.RZ(z, wires=0)
            qml.RX(y, wires=2)
            qml.CNOT(wires=[0, 2])
            qml.X(1)
            return qml.expval(qml.Z(0))

    >>> circuit(0.1, 0.2, 0.3)
    0.999999999999999

    .. details::
        :title: Usage Details

        You can also apply it on quantum functions:

        .. code-block:: python

            def qfunc(x, y, z):
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=1)
                qml.Hadamard(wires=0)
                qml.RX(x, wires=2)
                qml.RY(y, wires=1)
                qml.X(1)
                qml.RZ(z, wires=0)
                qml.RX(y, wires=2)
                qml.CNOT(wires=[0, 2])
                qml.X(1)
                return qml.expval(qml.Z(0))

        The circuit before optimization:

        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)(1, 2, 3))
        0: ──H─────────H─────────RZ(3.00)─╭●────┤  <Z>
        1: ──H─────────RY(2.00)──X────────│───X─┤
        2: ──RX(1.00)──RX(2.00)───────────╰X────┤

        We can see that there are two adjacent Hadamards on the first qubit that
        should cancel each other out. Similarly, there are two Pauli-X gates on the
        second qubit that should cancel. We can obtain a simplified circuit by running
        the ``cancel_inverses`` transform:

        >>> optimized_qfunc = cancel_inverses(qfunc)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)(1, 2, 3))
        0: ──RZ(3.00)───────────╭●─┤  <Z>
        1: ──H─────────RY(2.00)─│──┤
        2: ──RX(1.00)──RX(2.00)─╰X─┤

    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()
    operations = []

    while len(list_copy) > 0:
        current_gate = list_copy[0]
        list_copy.pop(0)

        # Find the next gate that acts on at least one of the same wires
        next_gate_idx = find_next_gate(current_gate.wires, list_copy)

        # If no such gate is found queue the operation and move on
        if next_gate_idx is None:
            operations.append(current_gate)
            continue

        # Otherwise, get the next gate
        next_gate = list_copy[next_gate_idx]

        # If either of the two flags is true, we can potentially cancel the gates
        if _are_inverses(current_gate, next_gate):
            # If the wires are the same, then we can safely remove both
            if current_gate.wires == next_gate.wires:
                list_copy.pop(next_gate_idx)
                continue
            # If wires are not equal, there are two things that can happen.
            # 1. There is not full overlap in the wires; we cannot cancel
            if len(Wires.shared_wires([current_gate.wires, next_gate.wires])) != len(
                current_gate.wires
            ):
                operations.append(current_gate)
                continue

            # 2. There is full overlap, but the wires are in a different order.
            # If the wires are in a different order, gates that are "symmetric"
            # over all wires (e.g., CZ), can be cancelled.
            if current_gate in symmetric_over_all_wires:
                list_copy.pop(next_gate_idx)
                continue
            # For other gates, as long as the control wires are the same, we can still
            # cancel (e.g., the Toffoli gate).
            if current_gate in symmetric_over_control_wires:
                # TODO[David Wierichs]: This assumes single-qubit targets of controlled gates
                if (
                    len(Wires.shared_wires([current_gate.wires[:-1], next_gate.wires[:-1]]))
                    == len(current_gate.wires) - 1
                ):
                    list_copy.pop(next_gate_idx)
                    continue
        # Apply gate any cases where
        # - there is no wire symmetry
        # - the control wire symmetry does not apply because the control wires are not the same
        # - neither of the flags are_self_inverses and are_inverses are true
        operations.append(current_gate)
        continue

    new_tape = type(tape)(operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
