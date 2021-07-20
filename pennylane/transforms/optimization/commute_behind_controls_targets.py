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

from pennylane import apply
from pennylane.wires import Wires
from pennylane.transforms import qfunc_transform

from .optimization_utils import find_next_gate

# Gates that commute among each particular basis
commuting_gates = {
    "X": ["PauliX", "RX", "SX"],
    "Y": ["PauliY", "RY"],
    "Z": ["PauliZ", "RZ", "PhaseShift", "S", "T"],
}


@qfunc_transform
def commute_behind_controls_targets(tape):
    """Quantum function transform to move commuting gates behind
    control and target qubits of controlled operations.

    Args:
        tape (.QuantumTape): A quantum tape.

    **Example**

    Consider the following quantum function.

    .. code-block:: python

        def qfunc_with_many_rots():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.T(wires=0)
            return qml.expval(qml.PauliX(0))

    In this circuit, the ``PauliZ`` and the ``CNOT`` commute; this means
    that the they can swap places, and then the ``PauliZ`` can be fused with the
    ``Hadamard`` gate if desired:

    >>> optimized_qfunc = qml.compile(pipeline=[diag_behind_controls, single_qubit_fusion])(qfunc)
    >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
    >>> print(qml.draw(optimized_qnode)())
    0: ──Rot(3.14, 1.57, 0.785)──╭C──┤ ⟨X⟩
    1: ──────────────────────────╰X──┤

    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()

    while len(list_copy) > 0:
        current_gate = list_copy[0]

        # Consider only gates that have is_controlled set to True
        if current_gate.is_controlled is None:
            apply(current_gate)
            list_copy.pop(0)
            continue

        # Find the next gate that acts on any subset of the wires
        next_gate_idx = find_next_gate(current_gate.wires, list_copy[1:])

        if next_gate_idx is None:
            apply(current_gate)
            list_copy.pop(0)
            continue

        # For this transform, we need to check through all the wires this gate has.
        # It might be that the adjacent gate cannot be moved, but a subsequent one can be.
        wire_count = 0

        # Loop as long as a valid next gate exists
        while next_gate_idx is not None and wire_count <= len(current_gate.wires):
            # Get the next gate
            next_gate = list_copy[next_gate_idx + 1 + wire_count]

            # This transform only works for pushing single-qubit operations
            if len(next_gate.wires) != 1:
                break

            shared_controls = Wires.shared_wires(
                [Wires(current_gate.control_wires), Wires(next_gate.wires)]
            )

            # Case 1: the overlap is on the control wires
            if len(shared_controls) > 0:
                # If the gate is a Z-basis gate, it can be pushed through
                if next_gate.name in commuting_gates["Z"]:
                    list_copy.pop(next_gate_idx + 1 + wire_count)
                    apply(next_gate)
                # Otherwise, we increment the wire count to indicate we checked this wire
                else:
                    wire_count += 1

            # Case 2: since we know the gates overlap somewhere, and it's a
            # single-qubit gate, if it wasn't on a control it's the target.
            else:
                if next_gate.name in commuting_gates[current_gate.target_gate_basis]:
                    list_copy.pop(next_gate_idx + 1 + wire_count)
                    apply(next_gate)
                else:
                    wire_count += 1

            next_gate_idx = find_next_gate(current_gate.wires, list_copy[1 + wire_count :])

        # After we have found all possible  gates to push through,
        # apply the original gate
        apply(current_gate)
        list_copy.pop(0)

    # Queue the measurements normally
    for m in tape.measurements:
        apply(m)
