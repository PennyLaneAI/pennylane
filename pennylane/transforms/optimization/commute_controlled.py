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

from itertools import chain

from pennylane import apply
from pennylane.wires import Wires
from pennylane.transforms import qfunc_transform

from .optimization_utils import find_next_gate


@qfunc_transform
def commute_controlled(tape):
    """Quantum function transform to move commuting gates past
    control and target qubits of controlled operations.

    Args:
        qfunc (function): A quantum function.

    **Example**

    Consider the following quantum function.

    .. code-block:: python

        def qfunc(theta):
            qml.CZ(wires=[0, 2])
            qml.PauliX(wires=2)
            qml.S(wires=0)

            qml.CNOT(wires=[0, 1])

            qml.PauliY(wires=1)
            qml.CRY(theta, wires=[0, 1])
            qml.PhaseShift(theta/2, wires=0)

            qml.Toffoli(wires=[0, 1, 2])
            qml.T(wires=0)
            qml.RZ(theta/2, wires=1)

            return qml.expval(qml.PauliZ(0))

    >>> dev = qml.device('default.qubit', wires=3)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> print(qml.draw(qnode)(0.5))
     0: ──╭C──S──╭C─────╭C────────Rϕ(0.25)──╭C──T─────────┤ ⟨Z⟩
     1: ──│──────╰X──Y──╰RY(0.5)────────────├C──RZ(0.25)──┤
     2: ──╰Z──X─────────────────────────────╰X────────────┤

    Diagonal gates on either side of control qubits do not affect the outcome
    of controlled gates; thus we can push all the single-qubit gates on the
    first qubit together on the right (and fuse them if desired). Similarly, X
    gates commute with the target of ``CNOT`` and ``Toffoli`` (and ``PauliY``
    with ``CRY``). We can use the transform to push single-qubit gates as
    far as possible through the controlled operations:

    >>> optimized_qfunc = commute_controlled(qfunc)
    >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
    >>> print(qml.draw(optimized_qnode)(0.5))
     0: ──╭C──╭C──╭C───────────╭C──S─────────Rϕ(0.25)──T──┤ ⟨Z⟩
     1: ──│───╰X──╰RY(0.5)──Y──├C──RZ(0.25)───────────────┤
     2: ──╰Z───────────────────╰X──X──────────────────────┤
    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()

    # We will go through the list backwards; whenever we find a single-qubit
    # gate, we will extract it and push it through 2-qubit gates as far as
    # possible to the right.
    current_location = len(list_copy) - 1

    while current_location >= 0:

        current_gate = list_copy[current_location]

        # We are looking only at the gates that can be pushed through
        # controls/targets; these are single-qubit gates with the basis
        # property specified.
        if current_gate.basis is None or len(current_gate.wires) != 1:
            current_location -= 1
            continue

        # Find the next gate that contains an overlapping wire
        next_gate_idx = find_next_gate(current_gate.wires, list_copy[current_location + 1 :])

        new_location = current_location

        # Loop as long as a valid next gate exists
        while next_gate_idx is not None:
            # Get the next gate
            next_gate = list_copy[new_location + next_gate_idx + 1]

            # Only go ahead if information is available
            if next_gate.basis is None:
                break

            # If the next gate does not have comp_control_wires defined, it is not
            # controlled so we can't push through.
            try:
                shared_controls = Wires.shared_wires(
                    [Wires(current_gate.wires), next_gate.comp_control_wires]
                )
            except (NotImplementedError, AttributeError):
                break

            # Case 1: the overlap is on the control wires. Only Z-type gates go through
            if len(shared_controls) > 0:
                # If the gate is a Z-basis gate, it can be pushed through
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

            next_gate_idx = find_next_gate(current_gate.wires, list_copy[new_location + 1 :])

        # After we have gone as far as possible, move the gate to new location
        list_copy.insert(new_location + 1, current_gate)
        list_copy.pop(current_location)
        current_location -= 1

    # Once the list is rearranged, queue all the operations
    for op in list_copy + tape.measurements:
        apply(op)
