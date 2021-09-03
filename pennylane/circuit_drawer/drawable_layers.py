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
"""

def _default_wire_map(ops):
    """This helper function may be moved elsewhere as integration of circuit
    drawing component progresses.

    Args:
        ops Iterable[Operation]
    
    Returns:
        dict: map from wires to sequential positive integers
    """

    wire_map  = dict()
    highest_number=0
    for op in ops:
        for wire in op.wires:
            if wire not in wire_map.keys():
                wire_map[wire] = highest_number
                highest_number+=1
    return wire_map

def drawable_grid(ops, wire_map=None):
    """Determine non-overlapping yet dense placement of operations for drawing.  Returns
    structure compatible with ``qml.circuit_drawer.Grid``. 
    
    Args:
        ops Iterable[~.Operator]: a list of operations

    Keyword Args:
        wire_map=None dict: dictionary mapping wire labels to sucessive positive integers.

    Returns:
        List[List[~.Operator]] : layers compatible with grid objects

    """

    if len(ops) == 0:
        return [ [] for _ in range(len(wire_map))]

    if wire_map is None:
        wire_map = _default_wire_map(ops)

    layers = drawable_layers(ops, wire_map=wire_map)

    n_wires = len(wire_map)
    n_layers = len(layers)

    grid = [[None for _ in range(n_layers)] for _ in range(n_wires)]

    for layer, ops in enumerate(layers):
        for op in ops:
            for wire in op.wires:
                grid[wire_map[wire]][layer] = op
    return grid

def drawable_layers(ops, wire_map=None):
    """Determine non-overlapping yet dense placement of operations for drawing.

    Args:
        ops Iterable[~.Operator]: a list of operations

    Keyword Args:
        wire_map=None dict: dictionary mapping wire labels to sucessive positive integers.

    Returns:
        list[set[~.Operator]] : Each index is a set of operations 
            for the corresponding layer
    """

    if wire_map is None:
        wire_map = _default_wire_map(ops)

    # initialize
    max_layer = 0

    occupied_wires_per_layer = [set()]
    ops_per_layer = [set()]
        
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
        mapped_wires = {wire_map[wire] for wire in op.wires}
        op_occupied_wires = set( range(min(mapped_wires), max(mapped_wires)+1))

        op_layer = recursive_find_layer(max_layer, op_occupied_wires)

        # see if need to add new layer
        if op_layer > max_layer:
            max_layer += 1
            occupied_wires_per_layer.append(set())
            ops_per_layer.append(set())

        # Add to op_layer
        ops_per_layer[op_layer].add(op)
        occupied_wires_per_layer[op_layer].update(op_occupied_wires)
        
    return ops_per_layer