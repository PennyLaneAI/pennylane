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
"""
This module contains a helper function to sort operations into layers.

Currently, circuit drawing performs:
* construction of circuit graph
* construction of greedy layers
* expansion of greedy layers to prevent collisions

This function aims to replace those three steps with a single function.
"""

def _default_wire_order(ops):
    """This helper function may be moved elsewhere as integration of circuit
    drawing component progresses.

    Args:
        ops Iterable[Operation]
    
    Returns:
        dict: map from wires to sequential positive integers
    """

    wire_order  = dict()
    highest_number=0
    for op in ops:
        for wire in op.wires:
            if wire not in wire_order.keys():
                wire_order[wire] = highest_number
                highest_number+=1
    return wire_order

def drawable_layers(ops, wire_order=None):
    """Determine non-overlapping yet dense placement of operations for drawing.

    Args:
        ops Iterable[Operation]

    Returns:
        list[list[Operation]] : Each index is a set of operations 
            for the corresponding layer
    """

    if wire_order is None:
        wire_order = _default_wire_order(ops)

    # initialize
    max_layer = 0

    occupied_wires_per_layer = [set()]
    ops_per_layer = [[]]
        
    def recursive_find_layer(checking_layer, op_occupied_wires):
        # function uses outer-scope `occupied_wires_per_layer`

        if occupied_wires_per_layer[checking_layer] & op_occupied_wires:
            # this layer is occupied, use higher one
            return checking_layer+1
        elif checking_layer == 0:
            # reached first layer, so stop
            return 0
        else:
            return recursive_find_layer(checking_layer-1, op_occupied_wires)
    
    # loop over operations
    for op in ops:
        mapped_wires = {wire_order[wire] for wire in op.wires}
        op_occupied_wires = set( range(min(mapped_wires), max(mapped_wires)+1))

        op_layer = recursive_find_layer(max_layer, op_occupied_wires)

        # see if need to add new layer
        if op_layer > max_layer:
            max_layer += 1
            occupied_wires_per_layer.append([])
            ops_per_layer.append({})

        # Add to op_layer
        ops_per_layer[op_layer].append(op)
        occupied_wires_per_layer[op_layer].update(op_occupied_wires)
        
    return ops_per_layer