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
from pennylane import apply
from pennylane.wires import Wires
from pennylane.transforms import qfunc_transform

from pennylane.ops.qubit.attributes import (
    self_inverses,
    symmetric_over_all_wires,
    symmetric_over_control_wires,
)
from .optimization_utils import find_next_gate


@qfunc_transform
def cancel_inverses(tape):
    """Quantum function transform to remove any operations that are applied next to their
    (self-)inverse.

    Args:
        qfunc (function): A quantum function.

    Returns:
        function: the transformed quantum function

    **Example**

    Consider the following quantum function:

    .. code-block:: python

        def qfunc(x, y, z):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.Hadamard(wires=0)
            qml.RX(x, wires=2)
            qml.RY(y, wires=1)
            qml.PauliX(wires=1)
            qml.RZ(z, wires=0)
            qml.RX(y, wires=2)
            qml.CNOT(wires=[0, 2])
            qml.PauliX(wires=1)
            return qml.expval(qml.PauliZ(0))

    The circuit before optimization:

    >>> dev = qml.device('default.qubit', wires=3)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> print(qml.draw(qnode)(1, 2, 3))
    0: ──H──────H──────RZ(3)─────╭C──┤ ⟨Z⟩
    1: ──H──────RY(2)──X──────X──│───┤
    2: ──RX(1)──RX(2)────────────╰X──┤

    We can see that there are two adjacent Hadamards on the first qubit that
    should cancel each other out. Similarly, there are two Pauli-X gates on the
    second qubit that should cancel. We can obtain a simplified circuit by running
    the ``cancel_inverses`` transform:

    >>> optimized_qfunc = cancel_inverses(qfunc)
    >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
    >>> print(qml.draw(optimized_qnode)(1, 2, 3))
    0: ──RZ(3)─────────╭C──┤ ⟨Z⟩
    1: ──H──────RY(2)──│───┤
    2: ──RX(1)──RX(2)──╰X──┤
    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()

    while len(list_copy) > 0:
        current_gate = list_copy[0]
        list_copy.pop(0)

        # Find the next gate that acts on at least one of the same wires
        next_gate_idx = find_next_gate(current_gate.wires, list_copy)

        # If no such gate is found queue the operation and move on
        if next_gate_idx is None:
            apply(current_gate)
            continue

        # Otherwise, get the next gate
        next_gate = list_copy[next_gate_idx]

        # There are then three possibilities that may lead to inverse cancellation. For a gate U,
        # 1. U is self-inverse, and the next gate is also U
        # 2. The current gate is U.inv and the next gate is U
        # 3. The current gate is U and the next gate is U.inv

        # Case 1
        are_self_inverses = (current_gate in self_inverses) and current_gate.name == next_gate.name

        # Cases 2 and 3
        are_inverses = False
        name_set = set([current_gate.name, next_gate.name])
        shortest_name = min(name_set)
        if {shortest_name, shortest_name + ".inv"} == name_set:
            are_inverses = True

        # If either of the two flags is true, we can potentially cancel the gates
        if are_self_inverses or are_inverses:
            # If the wires are the same, then we can safely remove both
            if current_gate.wires == next_gate.wires:
                list_copy.pop(next_gate_idx)
                continue
            # If wires are not equal, there are two things that can happen.
            # 1. There is not full overlap in the wires; we cannot cancel
            if len(Wires.shared_wires([current_gate.wires, next_gate.wires])) != len(
                current_gate.wires
            ):
                apply(current_gate)
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
        apply(current_gate)
        continue

    # Queue the measurements normally
    for m in tape.measurements:
        apply(m)
