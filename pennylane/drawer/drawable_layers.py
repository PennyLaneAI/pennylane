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

from pennylane.ops import Conditional
from pennylane.measurements import MidMeasureMP, MeasurementProcess
from .utils import default_wire_map


def _recursive_find_layer(layer_to_check, op_occupied_wires, occupied_wires_per_layer):
    """Determine correct layer for an operation drawn over ``op_occupied_wires``.

    An "occupied wire" will have something on top of it in the final drawing.  This could
    be a wire used by an operation or a wire between those used in a multi-qubit gate.

    In this function, we work with wires that are ordered, sequential integers, not the general
    hashable label that operations act on. ``drawable_layers`` performs this conversion.

    Args:
        layer_to_check (int): the function determines if the operation fits on this layer
        op_occupied_wires (set(int)): wires covered the drawn operation.  Includes everything
            between used wires in a multi-wire gate.
        occupied_wires_per_layer (list[set[int]]): which wires already have something drawn
            on them. Each set is a different layer.

    Returns:
        int: layer to place operation in
    """

    if occupied_wires_per_layer[layer_to_check] & op_occupied_wires:
        # this layer is occupied, use higher one
        return layer_to_check + 1
    if layer_to_check == 0:
        # reached first layer, so stop
        return 0
    # keep pushing the operation to lower layers
    return _recursive_find_layer(layer_to_check - 1, op_occupied_wires, occupied_wires_per_layer)


def _recursive_find_mcm_stats_layer(
    layer_to_check, op_occupied_cwires, used_cwires_per_layer, bit_map
):
    """Determine correct layer for a terminal measurement that is collectings statistics
    for mid-circuit measurement values.

    Args:
        layer_to_check (int): the function determines if the operation fits on this layer
        op_occupied_cwires (set(int)): classical wires occupied by measurement
        occupied_cwires_per_layer (list[set[int]]): which classical wires are already
            in use for collecting statistics. Each set is a different layer.
        bit_map (dict): A map containing mid-circuit measurements used for classical conditions
            or collecting statistics as keys

    Returns:
        int: layer to place measurement process in
    """

    if op_occupied_cwires & used_cwires_per_layer[layer_to_check]:
        # this layer is occupied, use higher one
        return layer_to_check + 1

    if layer_to_check == 0:
        # reached first layer, so stop
        return 0
    # keep pushing the operation to lower layers
    return _recursive_find_mcm_stats_layer(
        layer_to_check - 1, op_occupied_cwires, used_cwires_per_layer, bit_map
    )


def _get_op_occupied_wires(op, wire_map, bit_map):
    """Helper function to find wires that would be used by an operator in a drawable layer."""
    if isinstance(op, MidMeasureMP):
        mapped_wire = wire_map[op.wires[0]]

        if op in bit_map:
            min_wire = mapped_wire
            max_wire = max(wire_map.values())
            return set(range(min_wire, max_wire + 1))

        return {mapped_wire}

    if isinstance(op, Conditional):
        mapped_wires = [wire_map[wire] for wire in op.then_op.wires]
        min_wire = min(mapped_wires)
        max_wire = max(wire_map.values())
        return set(range(min_wire, max_wire + 1))

    if len(op.wires) == 0:
        # if no wires, then it acts on all wires
        # for example, qml.state and qml.sample
        mapped_wires = set(wire_map.values())
        return mapped_wires

    mapped_wires = {wire_map[wire] for wire in op.wires}
    # get all integers from the minimum to the maximum
    min_wire = min(mapped_wires)
    max_wire = max(mapped_wires)

    return set(range(min_wire, max_wire + 1))


def drawable_layers(operations, wire_map=None, bit_map=None):
    """Determine non-overlapping yet dense placement of operations into layers for drawing.

    Args:
        operations (Iterable[~.Operator]): A list of operations.

    Keyword Args:
        wire_map (dict): A map from wire label to non-negative integers. Defaults to None.
        bit_map (dict): A map containing mid-circuit measurements used for classical conditions
            or collecting statistics as keys. Defaults to None.

    Returns:
        (list[set[~.Operator]], list[set[~.MeasurementProcess]]) : Each index is a set of operations
        for the corresponding layer in both lists. The first list corresponds to the operation layers,
        and the second corresponds to the measurement layers.

    **Details**

    The function recursively pushes operations as far to the left (lowest layer) possible
    *without* altering order.

    From the start, the function cares about the locations the operation altered
    during a drawing, not just the wires the operation acts on. An "occupied" wire
    refers to a wire that will be altered in the drawing of an operation.
    Assuming wire ``1`` is between ``0`` and ``2`` in the ordering, ``qml.CNOT(wires=(0,2))``
    will also "occupy" wire ``1``.  In this scenario, an operation on wire ``1``, like
    ``qml.X(1)``, will not be pushed to the left
    of the ``qml.CNOT(wires=(0,2))`` gate, but be blocked by the occupied wire. This preserves
    ordering and makes placement more intuitive.

    The ``wire_order`` keyword argument used by user facing functions like :func:`~.draw` maps position
    to wire label.   The ``wire_map`` keyword argument used here maps label to position.
    The utility function :func:`~.circuit_drawer.utils.convert_wire_order` can perform this
    transformation.

    """

    wire_map = wire_map or default_wire_map(operations)
    bit_map = bit_map or {}

    # initialize for operation layers
    max_layer = 0
    occupied_wires_per_layer = [set()]
    ops_in_layer = [[]]
    used_cwires_per_layer = [set()]

    # loop over operations
    for op in operations:
        if isinstance(op, MidMeasureMP):
            if len(op.wires) > 1:
                raise ValueError("Cannot draw mid-circuit measurements with more than one wire.")

        if isinstance(op, MeasurementProcess) and op.mv is not None:
            # Only terminal measurements that collect mid-circuit measurement statistics have
            # op.mv != None.
            # Get the occupied classical wires of the measurement process and find which layer
            # to put it in.
            op_occupied_wires = set()
            mapped_cwires = (
                [bit_map[m.measurements[0]] for m in op.mv]
                if isinstance(op.mv, list)
                else [bit_map[m] for m in op.mv.measurements]
            )
            op_occupied_cwires = set(range(min(mapped_cwires), max(mapped_cwires) + 1))
            op_layer = _recursive_find_mcm_stats_layer(
                max_layer, op_occupied_cwires, used_cwires_per_layer, bit_map
            )

        else:
            # Find occupied wires of the operator/measurement process and find which layer to
            # put it in.
            op_occupied_wires = _get_op_occupied_wires(op, wire_map, bit_map)
            op_layer = _recursive_find_layer(max_layer, op_occupied_wires, occupied_wires_per_layer)
            op_occupied_cwires = set()

        # see if need to add new layer
        if op_layer > max_layer:
            max_layer += 1
            occupied_wires_per_layer.append(set())
            ops_in_layer.append([])
            used_cwires_per_layer.append(set())

        # add to op_layer
        ops_in_layer[op_layer].append(op)
        occupied_wires_per_layer[op_layer].update(op_occupied_wires)
        used_cwires_per_layer[op_layer].update(op_occupied_cwires)

    return list(filter(None, ops_in_layer[:-1])) + ops_in_layer[-1:]
