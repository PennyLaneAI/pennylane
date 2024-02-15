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
"""Transforms for pushing commuting gates through targets/control qubits."""
from typing import Sequence, Callable

from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.wires import Wires

from .optimization_utils import find_next_gate


def _commute_controlled_right(op_list):
    """Push commuting single qubit gates to the right of controlled gates.

    Args:
        op_list (list[Operation]): The initial list of operations.

    Returns:
        list[Operation]: The modified list of operations with all single-qubit
        gates as far right as possible.
    """
    # We will go through the list backwards; whenever we find a single-qubit
    # gate, we will extract it and push it through 2-qubit gates as far as
    # possible to the right.
    current_location = len(op_list) - 1

    while current_location >= 0:
        current_gate = op_list[current_location]

        # We are looking only at the gates that can be pushed through
        # controls/targets; these are single-qubit gates with the basis
        # property specified.
        if current_gate.basis is None or len(current_gate.wires) != 1:
            current_location -= 1
            continue

        # Find the next gate that contains an overlapping wire
        next_gate_idx = find_next_gate(current_gate.wires, op_list[current_location + 1 :])

        new_location = current_location

        # Loop as long as a valid next gate exists
        while next_gate_idx is not None:
            next_gate = op_list[new_location + next_gate_idx + 1]

            # Only go ahead if information is available
            if next_gate.basis is None:
                break

            # If the next gate does not have control_wires defined, it is not
            # controlled so we can't push through.
            if len(next_gate.control_wires) == 0:
                break
            shared_controls = Wires.shared_wires(
                [Wires(current_gate.wires), next_gate.control_wires]
            )

            # Case 1: overlap is on the control wires. Only Z-type gates go through
            if len(shared_controls) > 0:
                if current_gate.basis == "Z":
                    new_location += next_gate_idx + 1
                else:
                    break

            # Case 2: since we know the gates overlap somewhere, and it's a
            # single-qubit gate, if it wasn't on a control it's the target.
            else:
                if current_gate.basis == next_gate.basis:
                    new_location += next_gate_idx + 1
                else:
                    break

            next_gate_idx = find_next_gate(current_gate.wires, op_list[new_location + 1 :])

        # After we have gone as far as possible, move the gate to new location
        op_list.insert(new_location + 1, current_gate)
        op_list.pop(current_location)
        current_location -= 1

    return op_list


def _commute_controlled_left(op_list):
    """Push commuting single qubit gates to the left of controlled gates.

    Args:
        op_list (list[Operation]): The initial list of operations.

    Returns:
        list[Operation]: The modified list of operations with all single-qubit
        gates as far left as possible.
    """
    # We will go through the list forwards; whenever we find a single-qubit
    # gate, we will extract it and push it through 2-qubit gates as far as
    # possible back to the left.
    current_location = 0

    while current_location < len(op_list):
        current_gate = op_list[current_location]

        if current_gate.basis is None or len(current_gate.wires) != 1:
            current_location += 1
            continue

        # Pass a backwards copy of the list
        prev_gate_idx = find_next_gate(current_gate.wires, op_list[:current_location][::-1])

        new_location = current_location

        while prev_gate_idx is not None:
            prev_gate = op_list[new_location - prev_gate_idx - 1]

            if prev_gate.basis is None:
                break

            if len(prev_gate.control_wires) == 0:
                break
            shared_controls = Wires.shared_wires(
                [Wires(current_gate.wires), prev_gate.control_wires]
            )

            if len(shared_controls) > 0:
                if current_gate.basis == "Z":
                    new_location = new_location - prev_gate_idx - 1
                else:
                    break

            else:
                if current_gate.basis == prev_gate.basis:
                    new_location = new_location - prev_gate_idx - 1
                else:
                    break

            prev_gate_idx = find_next_gate(current_gate.wires, op_list[:new_location][::-1])

        op_list.pop(current_location)
        op_list.insert(new_location, current_gate)
        current_location += 1

    return op_list


@transform
def commute_controlled(tape: QuantumTape, direction="right") -> (Sequence[QuantumTape], Callable):
    """Quantum transform to move commuting gates past control and target qubits of controlled operations.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        direction (str): The direction in which to move single-qubit gates.
            Options are "right" (default), or "left". Single-qubit gates will
            be pushed through controlled operations as far as possible in the
            specified direction.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.


    **Example**

    >>> dev = qml.device('default.qubit', wires=3)

    You can apply the transform directly on :class:`QNode`:

    .. code-block:: python

        @partial(commute_controlled, direction="right")
        @qml.qnode(device=dev)
        def circuit(theta):
            qml.CZ(wires=[0, 2])
            qml.X(2)
            qml.S(wires=0)

            qml.CNOT(wires=[0, 1])

            qml.Y(1)
            qml.CRY(theta, wires=[0, 1])
            qml.PhaseShift(theta/2, wires=0)

            qml.Toffoli(wires=[0, 1, 2])
            qml.T(wires=0)
            qml.RZ(theta/2, wires=1)

            return qml.expval(qml.Z(0))

    >>> circuit(0.5)
    0.9999999999999999

    .. details::
        :title: Usage Details

        You can also apply it on quantum function.

        .. code-block:: python

            def qfunc(theta):
                qml.CZ(wires=[0, 2])
                qml.X(2)
                qml.S(wires=0)

                qml.CNOT(wires=[0, 1])

                qml.Y(1)
                qml.CRY(theta, wires=[0, 1])
                qml.PhaseShift(theta/2, wires=0)

                qml.Toffoli(wires=[0, 1, 2])
                qml.T(wires=0)
                qml.RZ(theta/2, wires=1)

                return qml.expval(qml.Z(0))

        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)(0.5))
        0: ─╭●──S─╭●────╭●─────────Rϕ(0.25)─╭●──T────────┤  <Z>
        1: ─│─────╰X──Y─╰RY(0.50)───────────├●──RZ(0.25)─┤
        2: ─╰Z──X───────────────────────────╰X───────────┤

        Diagonal gates on either side of control qubits do not affect the outcome
        of controlled gates; thus we can push all the single-qubit gates on the
        first qubit together on the right (and fuse them if desired). Similarly, X
        gates commute with the target of ``CNOT`` and ``Toffoli`` (and ``PauliY``
        with ``CRY``). We can use the transform to push single-qubit gates as
        far as possible through the controlled operations:

        >>> optimized_qfunc = commute_controlled(qfunc, direction="right")
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)(0.5))
        0: ─╭●─╭●─╭●───────────╭●──S─────────Rϕ(0.25)──T─┤  <Z>
        1: ─│──╰X─╰RY(0.50)──Y─├●──RZ(0.25)──────────────┤
        2: ─╰Z─────────────────╰X──X─────────────────────┤

    """
    if direction not in ("left", "right"):
        raise ValueError("Direction for commute_controlled must be 'left' or 'right'")

    if direction == "right":
        op_list = _commute_controlled_right(tape.operations)
    else:
        op_list = _commute_controlled_left(tape.operations)

    new_tape = type(tape)(op_list, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
