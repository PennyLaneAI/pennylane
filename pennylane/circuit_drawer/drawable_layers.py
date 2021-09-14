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

from .utils import default_wire_map


def _recursive_find_layer(checking_layer, op_occupied_wires, occupied_wires_per_layer):
    """Determine correct layer for operation with ``op_occupied_wires``

    Args:
        checking_layer (int): function determines if operation fits on this layer
        op_occupied_wires (set(int)): wires covered the drawn operation.  Includes everything
            between used wires in a multi-wire gate.
        occupied_wires_per_layer (list[set[int]]): which wires will already have something drawn
            on them

    Returns:
        int: layer to place relevant operation in
    """

    if occupied_wires_per_layer[checking_layer] & op_occupied_wires:
        # this layer is occupied, use higher one
        return checking_layer + 1
    if checking_layer == 0:
        # reached first layer, so stop
        return 0
    # keep pushing the operation back
    return _recursive_find_layer(checking_layer - 1, op_occupied_wires, occupied_wires_per_layer)


def drawable_layers(ops, wire_map=None):
    """Determine non-overlapping yet dense placement of operations for drawing.

    Args:
        ops Iterable[~.Operator]: a list of operations

    Keyword Args:
        wire_map=None (dict): a map from wire label to non-negative integers

    Returns:
        list[set[~.Operator]] : Each index is a set of operations
            for the corresponding layer
    """

    if wire_map is None:
        wire_map = default_wire_map(ops)

    # initialize
    max_layer = 0
    occupied_wires_per_layer = [set()]
    ops_per_layer = [set()]

    # loop over operations
    for op in ops:
        if len(op.wires) == 0:
            # if no wires, then it acts of all wires
            # for example, state and sample
            mapped_wires = set(wire_map.values())
            op_occupied_wires = mapped_wires
        else:
            mapped_wires = {wire_map[wire] for wire in op.wires}
            op_occupied_wires = set(range(min(mapped_wires), max(mapped_wires) + 1))

        op_layer = _recursive_find_layer(max_layer, op_occupied_wires, occupied_wires_per_layer)

        # see if need to add new layer
        if op_layer > max_layer:
            max_layer += 1
            occupied_wires_per_layer.append(set())
            ops_per_layer.append(set())

        # Add to op_layer
        ops_per_layer[op_layer].add(op)
        occupied_wires_per_layer[op_layer].update(op_occupied_wires)

    return ops_per_layer


def drawable_grid(ops, wire_map=None):
    """Determine non-overlapping yet dense placement of operations for drawing.  Returns
    structure compatible with ``qml.circuit_drawer.Grid``.

    Args:
        ops Iterable[~.Operator]: a list of operations

    Keyword Args:
        wire_map=None (dict): a map from wire label to non-negative integers

    Returns:
        List[List[~.Operator]] : layers compatible with grid objects
    """
    if wire_map is None:
        wire_map = default_wire_map(ops)

    if len(ops) == 0:
        if len(wire_map) == 0:
            return [[]]
        return [[] for _ in range(len(wire_map))]

    ops_per_layer = drawable_layers(ops, wire_map=wire_map)

    n_wires = len(wire_map)
    n_layers = len(ops_per_layer)

    grid = [[None for _ in range(n_layers)] for _ in range(n_wires)]

    for layer, layer_ops in enumerate(ops_per_layer):
        for op in layer_ops:
            if len(op.wires) == 0:
                # apply to all wires, like state and sample
                for wire in range(n_wires):
                    grid[wire][layer] = op

            for wire in op.wires:
                grid[wire_map[wire]][layer] = op
    return grid
