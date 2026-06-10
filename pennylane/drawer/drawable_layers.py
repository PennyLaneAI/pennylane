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

from dataclasses import dataclass, field
from functools import singledispatch

from pennylane.allocation import Allocate, Deallocate, DynamicWire
from pennylane.measurements import MeasurementProcess
from pennylane.ops import (
    Conditional,
    GlobalPhase,
    Identity,
    MeasurementValue,
    MidMeasure,
    PauliMeasure,
)
from pennylane.pytrees import flatten
from pennylane.templates import SubroutineOp

from .utils import default_wire_map, unwrap_controls


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


def _recursive_find_mcm_stats_layer(layer_to_check, op_occupied_cwires, used_cwires_per_layer):
    """Determine correct layer for a terminal measurement that is collecting statistics
    for mid-circuit measurement values.

    Args:
        layer_to_check (int): the function determines if the operation fits on this layer
        op_occupied_cwires (set(int)): classical wires occupied by measurement
        used_cwires_per_layer (list[set[int]]): which classical wires are already
            in use for collecting statistics. Each set is a different layer.

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
        layer_to_check - 1, op_occupied_cwires, used_cwires_per_layer
    )


# pylint: disable=unused-argument
@singledispatch
def _get_op_occupied_wires(op, wire_map, bit_map):
    """Helper function to find wires that would be used by an operator in a drawable layer."""
    *_, base = unwrap_controls(op)

    if len(op.wires) == 0 or isinstance(base, (GlobalPhase, Identity)):
        # if no wires, then it acts on all wires. For example qp.state and qp.sample or
        # (controlled) GlobalPhase or (controlled) Identity
        mapped_wires = set(wire_map.values())
        return mapped_wires

    mapped_wires = {wire_map[wire] for wire in op.wires}
    # get all integers from the minimum to the maximum
    min_wire = min(mapped_wires)
    max_wire = max(mapped_wires)

    return set(range(min_wire, max_wire + 1))


@_get_op_occupied_wires.register
def _occupied_subroutine_op_wires(op: SubroutineOp, wire_map, bit_map):
    mapped_wires = [wire_map[wire] for wire in op.wires]

    mvs = (v for v in flatten(op.output)[0] if isinstance(v, MeasurementValue))
    if any(m in bit_map for mv in mvs for m in mv.measurements):
        min_wire = min(mapped_wires)
        max_wire = max(wire_map.values())
        return set(range(min_wire, max_wire + 1))

    min_wire = min(mapped_wires)
    max_wire = max(mapped_wires)
    return set(range(min_wire, max_wire + 1))


@_get_op_occupied_wires.register(MidMeasure)
@_get_op_occupied_wires.register(PauliMeasure)
def _handle_mid_measure(op: MidMeasure | PauliMeasure, wire_map, bit_map):
    mapped_wires = [wire_map[wire] for wire in op.wires]

    if op in bit_map:
        min_wire = min(mapped_wires)
        max_wire = max(wire_map.values())
        return set(range(min_wire, max_wire + 1))

    min_wire = min(mapped_wires)
    max_wire = max(mapped_wires)
    return set(range(min_wire, max_wire + 1))


@_get_op_occupied_wires.register
def _handle_cond(op: Conditional, wire_map, bit_map):
    mapped_wires = [wire_map[wire] for wire in op.base.wires]
    min_wire = min(mapped_wires)
    max_wire = max(wire_map.values())
    return set(range(min_wire, max_wire + 1))


@dataclass
class _LayersData:
    """Data for putting operations into layers."""

    ops_in_layer: list[list] = field(default_factory=lambda: [[]])
    """Lists of which operators should be placed in each layer."""

    occupied_wires_per_layer: list[set[int]] = field(default_factory=lambda: [set()])
    """The mapped wires that will be occupied in the drawing at each layer."""

    used_cwires_per_layer: list[set[int]] = field(default_factory=lambda: [set()])
    """The classical wires that will be occupied in each layer."""

    waiting_dynamic_wires: list[DynamicWire] = field(default_factory=list)
    """DynamicWires that are waiting for the first interaction between
    the dynamic wires and an algorithmic wire. 
    
    This allows us to push the allocation and initial setup to the right and conserve
    drawer space, keeping things together.  See ``insert_waiting_ops``.
    """

    waiting_dynamic_ops: list = field(default_factory=list)
    """The Allocate instructions and operators that are waiting on the first interaction between
    the waiting_dynamic_wires and an algorithmic wire.
    """

    @property
    def max_layer(self) -> int:
        """The maximum number of layers present."""
        return len(self.ops_in_layer) - 1

    def add_to_layer(self, op, op_layer, occupied_wires, used_cwires):
        """Adds an operation to a layer."""
        while op_layer > self.max_layer:
            # when in insert_waiting_ops, may need multiple new layers
            self.occupied_wires_per_layer.append(set())
            self.ops_in_layer.append([])
            self.used_cwires_per_layer.append(set())

        self.ops_in_layer[op_layer].append(op)
        self.occupied_wires_per_layer[op_layer].update(occupied_wires)
        self.used_cwires_per_layer[op_layer].update(used_cwires)

    def insert_waiting_ops(self, op_layer, wire_map, bit_map):
        """Insert ops in ``waiting_dynamic_ops`` before ``op_layer``."""
        inner_layers = drawable_layers(
            self.waiting_dynamic_ops,
            wire_map=wire_map,
            bit_map=bit_map,
            _dynamic_wires=False,
        )
        # if not enough space, bump op_layer
        op_layer = max(op_layer, len(inner_layers))
        # if any barrier/ global phase/ etc we need to shift forward where
        # we add things even more
        # bit of complicated logic for a rare edge case, but might as well support it
        #
        # qp.Barrier()
        # with qp.allocate(1, state="any") as wires: # cant add this to layer 0 that has a barrier
        # .    qp.CNOT((0, wires[0]))
        for offset in range(1, len(inner_layers) + 1):
            check_layer = op_layer - offset
            if check_layer <= self.max_layer and any(
                not op.wires for op in self.ops_in_layer[check_layer]
            ):
                op_layer = check_layer + len(inner_layers)
                break
        for i, layer in enumerate(inner_layers):
            insert_layer = op_layer - len(inner_layers) + i
            for op in layer:
                self.add_to_layer(op, insert_layer, {}, {})

        self.waiting_dynamic_wires = []
        self.waiting_dynamic_ops = []
        return op_layer


def drawable_layers(operations, wire_map=None, bit_map=None, _dynamic_wires=True):
    """Determine non-overlapping yet dense placement of operations into layers for drawing.

    Args:
        operations (Iterable[~.Operator]): A list of operations.

    Keyword Args:
        wire_map (dict): A map from wire label to non-negative integers. Defaults to None.
        bit_map (dict): A map containing mid-circuit measurements used for classical conditions
            or collecting statistics as keys. Defaults to None.
        _dynamic_wires (bool): **Internal**.  Whether or not to push allocations and ops
            on only dynamic wires to the right next to the first time the dynamic wires are used.

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
    Assuming wire ``1`` is between ``0`` and ``2`` in the ordering, ``qp.CNOT(wires=(0,2))``
    will also "occupy" wire ``1``.  In this scenario, an operation on wire ``1``, like
    ``qp.X(1)``, will not be pushed to the left
    of the ``qp.CNOT(wires=(0,2))`` gate, but be blocked by the occupied wire. This preserves
    ordering and makes placement more intuitive.

    The ``wire_order`` keyword argument used by user facing functions like :func:`~.draw` maps position
    to wire label.   The ``wire_map`` keyword argument used here maps label to position.
    The utility function :func:`~.circuit_drawer.utils.convert_wire_order` can perform this
    transformation.

    """

    wire_map = wire_map or default_wire_map(operations)[1]
    bit_map = bit_map or {}

    # initialize for operation layers
    data = _LayersData()

    # loop over operations
    for op in operations:
        if _dynamic_wires and isinstance(op, Allocate):
            data.waiting_dynamic_ops.append(op)
            data.waiting_dynamic_wires.extend(op.wires)

        elif (
            _dynamic_wires
            and all(w in data.waiting_dynamic_wires for w in op.wires)
            and op.wires  # if no wires (GlobalPhase) then do not put into waiting_dynamic_ops
            and not isinstance(op, Deallocate)  # deallocate should force putting into circuit
        ):
            data.waiting_dynamic_ops.append(op)
        elif isinstance(op, MeasurementProcess) and op.mv is not None:
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
                data.max_layer, op_occupied_cwires, data.used_cwires_per_layer
            )

            data.add_to_layer(op, op_layer, op_occupied_wires, op_occupied_cwires)

        else:
            # Find occupied wires of the operator/measurement process and find which layer to
            # put it in.
            op_occupied_wires = _get_op_occupied_wires(op, wire_map, bit_map)
            try:
                op_layer = _recursive_find_layer(
                    data.max_layer, op_occupied_wires, data.occupied_wires_per_layer
                )
            except RecursionError as e:
                raise RecursionError(
                    f"Drawer is currently at depth {data.max_layer}, which is too deep to handle. "
                    "Try drawing a smaller subset of your circuit instead."
                ) from e
            op_occupied_cwires = set()

            # barrier should also trigger insertion of waitin gops
            if _dynamic_wires and (
                any(w in data.waiting_dynamic_wires for w in op.wires) or not op.wires
            ):
                op_layer = data.insert_waiting_ops(op_layer, wire_map, bit_map)

            data.add_to_layer(op, op_layer, op_occupied_wires, op_occupied_cwires)

    return list(filter(None, data.ops_in_layer[:-1])) + data.ops_in_layer[-1:]
