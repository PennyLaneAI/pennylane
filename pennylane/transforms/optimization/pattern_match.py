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
from pennylane.wires import Wires
from pennylane.transforms import qfunc_transform
import pennylane.ops.qubit as ops

from .optimization_utils import find_next_gate

templates = {
    2 : {
        "S0 S0" : [ops.PauliZ],
        "T0 T0" : [ops.S]
    },
    4: {
        "CNOT01 H0 H1 CNOT10": [ops.Hadamard, ops.Hadamard]
    }
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
                
                blob.append(op(wires=qml.wires.Wires(wires_with_labels)))

        # If it isn't, return None
        return None

    return blob
        
@qfunc_transform
def pattern_match(tape):
    """Quantum function transform to remove any operations that are applied next to their
    (self-)inverse.

    Args:
        qfunc (function): A quantum function.

    Returns:
    """

    list_copy = tape.operations.copy()

    something_was_changed = True
    
    # Loop through the list of operations and find templates until there aren't any more
    while something_was_changed:

        something_was_changed = False

        op_list = []
        
        while len(list_copy) > 0:
            current_gate = list_copy[0]

            # Get the largest block of upcoming gates that act on this gates wires
            gate_list = []
            gate_list_idxs = []
            gate_offset = 1

            while True:
                next_gate_idx = find_next_gate(current_gate.wires, list_copy[gate_offset:])
                if next_gate_idx is not None:
                    print(f"Next gate is {list_copy[next_gate_idx+1]}")
                    gate_list.append(list_copy[gate_offset + next_gate_idx])
                    gate_list_idxs.append(gate_offset + next_gate_idx)
                    print(gate_list)
                    gate_offset += next_gate_idx + 1
                else:
                    break

            # If there is no block of gates, simply add this gate and move on
            if len(gate_list) == 0:
                op_list.append(list_copy[0])
                list_copy.pop(0)
                continue

            # If there is a block of gates, test whether it's a template
            used_wires = Wires.all_wires([g.wires for g in gate_list])
            wire_map = {used_wires[x] : x for x in range(len(used_wires))}

            # It could be that the block we found is larger than any of the
            # template sizes; we have to check the first two elements, see if
            # there is a match; if not, then second and third elements, etc.
            # up to max template size
            for template_size in list(templates.keys()):
                # Don't look if we don't have enough gates for a possible match
                if len(gate_list) >= template_size:
                    window_start_idx = 0

                    while window_start_idx < len(gate_list) - template_size:
                        gates_in_window = gate_list[window_start_idx:window_start_idx+template_size]
                        gate_string = _gates_to_string(gates_in_window, wire_map)

                        # Check for a match
                        replacement = _replace_with_template(gate_string, wire_map)

                        # If we found a replacement, add it to the list, then remove all the
                        # other gates in the template
                        if replacement is not None:
                            print(f"Found replacement for template {gate_string}")

                            # Remove the old gates from the current gate list
                            for gate_idx in range(window_start_idx, template_size+1):
                                gate_list.pop(gate_idx)

                            # Add the new gates
                            for idx, gate in replacement:
                                gate_list.insert(window_start_idx + idx, gate)
                            something_was_changed = True

            # Add our completed template to the op list
            op_list.extend(gate_list)
            
            # After finding all the templates in this section, we need to remove those original
            # gates from the list copy, and replace them with the gates in the modified template
            for gate_idx in gate_list_idxs[::-1]:
                list_copy.pop(gate_idx)

            for gate in gate_list[::-1]:
                list_copy.insert(0, gate)
            
            # In either case, remove the first element from the list and move on
            list_copy.pop(0)            

        # For the next iteration, copy the current op_list into the list_copy
        list_copy = op_list.copy()
        
    for op in op_list + tape.measurements:
        apply(op)
