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
"""Transform for simple template optimization."""

from pennylane import apply
from pennylane.tape import get_active_tape
from pennylane.wires import Wires
from pennylane.transforms import qfunc_transform
import pennylane.ops.qubit as ops

from .optimization_utils import find_next_gate

# Instead of using the circuit DAG, I tried doing this with string matching.
# It works, but its maybe not the best solution.
templates = {
    2: {"S0 S0": [(ops.PauliZ, 0)], "T0 T0": [(ops.S, 0)],},
    3: {
        "Hadamard0 PauliX0 Hadamard0": [(ops.PauliZ, 0)],
        "Hadamard0 PauliZ0 Hadamard0": [(ops.PauliX, 0)],
        "CNOT01 PauliX0 CNOT01": [(ops.PauliX, 0), (ops.PauliX, 1)],
        "CNOT01 Toffoli012 CNOT01": [(ops.Toffoli, [0, 1, 2]), (ops.CNOT, [0, 2])],
    },
    4: {"CNOT01 Hadamard0 Hadamard1 CNOT10": [(ops.Hadamard, 0), (ops.Hadamard, 1)]},
}


def _gates_to_string(gate_list, wire_map):
    """Convert a list of gates into string format suitable for a template."""
    gate_string_list = []

    # TODO: sort the gates based on wire order

    for gate in gate_list:
        this_gate_string = gate.name

        for wire_label in gate.wires.labels:
            this_gate_string += str(wire_map[wire_label])

        gate_string_list.append(this_gate_string)

    return " ".join(gate_string_list)


def _replace_with_template(gate_string, wire_map):
    """Convert a string indicating a list of gates to the actual list 
    of gates to apply, given a particular wire map.

    Returns: None if no valid template is found, otherwise the set of 
    operations in the template.
    """

    reverse_wire_map = {val: key for key, val in wire_map.items()}

    blob = []

    # Count the number of spaces to indicate how many gates were in the template
    template_set_idx = gate_string.count(" ") + 1

    if template_set_idx in templates.keys():
        # Check if this sequence is a defined template
        if gate_string in templates[template_set_idx].keys():
            op_list = templates[template_set_idx][gate_string]

            for op, which_wires in op_list:
                if not isinstance(which_wires, list):
                    which_wires = [which_wires]

                wires_with_labels = [reverse_wire_map[w] for w in which_wires]

                blob.append(op(wires=Wires(wires_with_labels)))
        else:
            # If it isn't, return None
            return None
    else:
        return None

    return blob


def _replace_patterns_in_tape(op_list):
    """Loop through the operations on a tape and replace any templates  
    that are found."""

    global_something_was_changed = False

    idx = 0

    # Go through the list once
    while idx < len(op_list):
        current_gate = op_list[idx]

        # Keep track of whether anything actually changed
        something_was_changed = False

        # Go through the templates in increasing size
        for template_size in list(templates.keys()):
            # Only look if there is even a chance that we'll have enough gates
            if len(op_list) - idx < template_size:
                continue

            # Get the block of upcoming gates that act on this gates wires
            gate_list = [current_gate]
            gate_list_idxs = [idx]
            gate_offset = 1

            # Get the next number of gates needed to potentially have a template of this size
            for _ in range(template_size - 1):
                wires_in_block = Wires.all_wires([g.wires for g in gate_list])

                next_gate_idx = find_next_gate(wires_in_block, op_list[idx + gate_offset :])

                if next_gate_idx is not None:
                    gate_list.append(op_list[idx + gate_offset + next_gate_idx])
                    gate_list_idxs.append(idx + gate_offset + next_gate_idx)
                    gate_offset += next_gate_idx + 1
                else:
                    break

            # If there is a block of gates, test whether it's a template of this size
            used_wires = Wires.all_wires([g.wires for g in gate_list])
            wire_map = {used_wires[x]: x for x in range(len(used_wires))}
            gate_string = _gates_to_string(gate_list, wire_map)

            # Check for a match
            replacement = _replace_with_template(gate_string, wire_map)

            # If we found a replacement, replace applicable gates with gates from the template
            if replacement is not None:
                for gate_idx in gate_list_idxs[::-1]:
                    op_list.pop(gate_idx)

                    # Add any replacements that can be added
                    if len(replacement) > 0:
                        op_list.insert(gate_idx, replacement.pop())

                while len(replacement) > 0:
                    op_list.insert(gate_list_idxs[0], replacement.pop())

                something_was_changed = True
                global_something_was_changed = True

        # If anything was changed, the gate at the current index has been replaced,
        # so loop back over the same index and see if we can do anything new. If not,
        # we move to the next gate.
        if not something_was_changed:
            idx += 1

    return op_list, global_something_was_changed


@qfunc_transform
def pattern_match(tape):
    """Quantum function transform to find and replace small patterns in a circuit.

    Args:
        qfunc (function): A quantum function.

    Returns:
        function: A quantum function with its templates replaced.

    **Example**

    Consider the following quantum function:

    .. code-block:: python3
        def my_circuit():
            qml.S(wires=0)
            qml.S(wires=0)
            qml.T(wires=0)
            qml.T(wires=1)
            qml.T(wires=1)
            qml.CNOT(wires=[1, 0])
            qml.Hadamard(wires=1)
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0))

    The two S gates on the first qubit can be replaced by a Z; the two T gates on
    the second qubits can be replaced with an S. Furthermore, the last four gates
    can be replaced by simply a Hadamard on each of the two qubits. If we express
    these relationships as templates, the ``pattern_match`` transform will replace
    them by the more efficient versions.

    >>> dev = qml.device('default.qubit', wires=2)
    >>> qfunc = qml.transforms.pattern_match(my_circuit)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> print(qml.draw(qnode)())
     0: ──Z──T──H──┤ ⟨Z⟩ 
     1: ──S──H─────┤     
    """

    list_copy = tape.operations.copy()

    current_tape = get_active_tape()

    something_was_changed = True

    # Loop through the tape until we find no more matching patterns
    with current_tape.stop_recording():
        while something_was_changed is True:
            list_copy, something_was_changed = _replace_patterns_in_tape(list_copy)

    for op in list_copy + tape.measurements:
        apply(op)
