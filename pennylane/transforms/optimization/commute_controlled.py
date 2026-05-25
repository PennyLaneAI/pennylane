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

import pennylane as qp  # is_commuting circular import problems
from pennylane.ops.op_math import Controlled
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

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

        if len(current_gate.wires) != 1:
            current_location -= 1
            continue

        # Find the next gate that contains an overlapping wire
        next_gate_idx = find_next_gate(current_gate.wires, op_list[current_location + 1 :])

        new_location = current_location

        # Loop as long as a valid next gate exists
        while next_gate_idx is not None:
            next_gate = op_list[new_location + next_gate_idx + 1]

            if not isinstance(next_gate, Controlled) or not qp.is_commuting(
                current_gate, next_gate
            ):
                break

            new_location += next_gate_idx + 1

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

        if len(current_gate.wires) != 1:
            current_location += 1
            continue

        # Pass a backwards copy of the list
        prev_gate_idx = find_next_gate(current_gate.wires, op_list[:current_location][::-1])

        new_location = current_location

        while prev_gate_idx is not None:
            prev_gate = op_list[new_location - prev_gate_idx - 1]

            if not isinstance(prev_gate, Controlled) or not qp.is_commuting(
                current_gate, prev_gate
            ):
                break

            new_location -= prev_gate_idx + 1

            prev_gate_idx = find_next_gate(current_gate.wires, op_list[:new_location][::-1])

        op_list.pop(current_location)
        op_list.insert(new_location, current_gate)
        current_location += 1

    return op_list


@transform
def commute_controlled(
    tape: QuantumScript, direction="right"
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Quantum transform to move commuting gates past control and target qubits of controlled operations.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        direction (str): The direction in which to move single-qubit gates.
            Options are "right" (default), or "left". Single-qubit gates will
            be pushed through controlled operations as far as possible in the
            specified direction.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qp.transform <pennylane.transform>`.


    **Example**

    You can apply the transform directly on :class:`QNode`:

    .. code-block:: python

        import pennylane as qp

        dev = qp.device('default.qubit')

        @qp.transforms.commute_controlled(direction="right")
        @qp.qnode(device=dev)
        def circuit(theta):
            qp.CZ(wires=[0, 2])
            qp.X(2)
            qp.S(wires=0)

            qp.CNOT(wires=[0, 1])

            qp.Y(1)
            qp.CRY(theta, wires=[0, 1])
            qp.PhaseShift(theta/2, wires=0)

            qp.Toffoli(wires=[0, 1, 2])
            qp.T(wires=0)
            qp.RZ(theta/2, wires=1)

            return qp.expval(qp.Z(0))

    >>> print(qp.draw(circuit)(0.5))
    0: ─╭●─╭●─╭●───────────╭●──S─────────Rϕ(0.25)──T─┤  <Z>
    1: ─│──╰X─╰RY(0.50)──Y─├●──RZ(0.25)──────────────┤
    2: ─╰Z─────────────────╰X──X─────────────────────┤

    .. details::
        :title: Usage Details

        You can also apply this transform to quantum functions.

        .. code-block:: python

            def qfunc(theta):
                qp.CZ(wires=[0, 2])
                qp.X(2)
                qp.S(wires=0)

                qp.CNOT(wires=[0, 1])

                qp.Y(1)
                qp.CRY(theta, wires=[0, 1])
                qp.PhaseShift(theta/2, wires=0)

                qp.Toffoli(wires=[0, 1, 2])
                qp.T(wires=0)
                qp.RZ(theta/2, wires=1)

                return qp.expval(qp.Z(0))

        >>> qnode = qp.QNode(qfunc, dev)
        >>> print(qp.draw(qnode)(0.5))
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
        >>> optimized_qnode = qp.QNode(optimized_qfunc, dev)
        >>> print(qp.draw(optimized_qnode)(0.5))
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

    new_tape = tape.copy(operations=op_list)

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
