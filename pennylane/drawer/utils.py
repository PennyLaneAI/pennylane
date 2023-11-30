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
This module contains some useful utility functions for circuit drawing.
"""
from dataclasses import dataclass
from collections import namedtuple
from typing import Optional

from pennylane.ops import Controlled, Conditional
from pennylane.measurements import MidMeasureMP


@dataclass
class DrawingConfig:
    """Dataclass containing attributes needed for updating the strings to be drawn for each layer"""

    wire_map: dict
    """Map between wire labels and their place in order"""

    bit_map: dict
    """Map between mid-circuit measurements and their corresponding bit in order"""

    cur_layer: Optional[int] = None
    """Current layer index that is being updated"""

    final_cond_layers: Optional[list] = None
    """List mapping bits to the last layer where they are used"""

    decimals: Optional[int] = None
    """Specifies how to round the parameters of operators"""

    cache: Optional[dict] = None
    """dictionary that carries information between label calls in the same drawing"""


MidMeasureProperties = namedtuple(
    "MidMeasureProperties",
    [
        # mid_measure op for current bit [MidMeasureMP]
        "m_op",
        # layer of mid_measure op [int]
        "m_layer",
        # mapped wire of mid_measure op [int]
        "m_wire",
        # List of layers of qml.conds that use current bit [List[int]]
        "cond_layers",
        # List of bottom-most mapped wire that uses current bit [List[int]]
        "cond_wires",
        # Index of last operation layer where the current bit should be drawn [int]
        "last_layer",
    ],
)


def default_wire_map(tape):
    """Create a dictionary mapping used wire labels to non-negative integers

    Args:
        tape [~.tape.QuantumTape): the QuantumTape containing operations and measurements

    Returns:
        dict: map from wires to sequential positive integers
    """

    # Use dictionary to preserve ordering, sets break order
    used_wires = {wire: None for op in tape for wire in op.wires}
    return {wire: ind for ind, wire in enumerate(used_wires)}


def convert_wire_order(tape, wire_order=None, show_all_wires=False):
    """Creates the mapping between wire labels and place in order.

    Args:
        tape (~.tape.QuantumTape): the Quantum Tape containing operations and measurements
        wire_order Sequence[Any]: the order (from top to bottom) to print the wires

    Keyword Args:
        show_all_wires=False (bool): whether to display all wires in ``wire_order``
            or only include ones used by operations in ``ops``

    Returns:
        dict: map from wire labels to sequential positive integers
    """
    default = default_wire_map(tape)

    if wire_order is None:
        return default

    wire_order = list(wire_order) + [wire for wire in default if wire not in wire_order]

    if not show_all_wires:
        used_wires = {wire for op in tape for wire in op.wires}
        wire_order = [wire for wire in wire_order if wire in used_wires]

    return {wire: ind for ind, wire in enumerate(wire_order)}


def unwrap_controls(op):
    """Unwraps nested controlled operations for drawing.

    Controlled operations may themselves contain controlled operations; check
    for any nesting of operators when drawing so that we correctly identify
    and label _all_ control and target qubits.

    Args:
        op (.Operation): A PennyLane operation.

    Returns:
        Wires, List: The control wires of the operation, along with any associated
        control values.
    """
    # Get wires and control values of base operation; need to make a copy of
    # control values, otherwise it will modify the list in the operation itself.
    control_wires = getattr(op, "control_wires", [])
    control_values = getattr(op, "hyperparameters", {}).get("control_values", None)

    if isinstance(control_values, list):
        control_values = control_values.copy()

    if isinstance(op, Controlled):
        next_ctrl = op

        while hasattr(next_ctrl, "base"):
            base_control_wires = getattr(next_ctrl.base, "control_wires", [])
            control_wires += base_control_wires

            base_control_values = next_ctrl.base.hyperparameters.get(
                "control_values", [True] * len(base_control_wires)
            )

            if control_values is not None:
                control_values.extend(base_control_values)

            next_ctrl = next_ctrl.base

    control_values = [bool(int(i)) for i in control_values] if control_values else control_values
    return control_wires, control_values


def find_mid_measure_cond_connections(operations, layers):
    """Collect and return information about connections between mid-circuit measurements
    and classical conditions.

    This utility function returns three items needed for processing mid-circuit measurements
    and classical conditions for drawing:

    * A dictionary mapping each mid-circuit measurement to a corresponding bit index.
        This map only contains mid-circuit measurements that are used for classical conditioning.
    * A list where each index is a bit and the values are the indices  of the layers containing
        the mid-circuit measurement corresponding to the bits.
    * A list where each index is a bit and the values are the indices of the last layers that
        use those bits for classical conditions.

    Args:
        operations (list[~.Operation]): List of operations on the tape
        layers (list[list[~.Operation]]): List of drawable layers containing list of operations
            for each layer

    Returns:
        tuple[dict, list, list]: Data structures needed for correctly drawing classical conditions
        as described above.
    """

    # Map between mid-circuit measurements and their position on the drawing
    # The bit map only contains mid-circuit measurements that are used in
    # classical conditions.
    bit_map = {}

    # Map between classical bit positions and the layer of their corresponding mid-circuit
    # measurements.
    measurement_layers = []

    # Map between classical bit positions and the final layer where the bit is used.
    # This is needed to know when to stop drawing a bit line. The bit is the index,
    # so each of the two lists must have the same length as the number of bits
    final_cond_layers = []

    measurements_for_conds = set()
    conditional_ops = []
    for op in operations:
        if isinstance(op, Conditional):
            measurements_for_conds.update(op.meas_val.measurements)
            conditional_ops.append(op)

    if len(measurements_for_conds) > 0:
        cond_mid_measures = [op for op in operations if op in measurements_for_conds]
        cond_mid_measures.sort(key=operations.index)

        bit_map = dict(zip(cond_mid_measures, range(len(cond_mid_measures))))

        n_bits = len(bit_map)

        # Set lists to correct size
        measurement_layers = [None] * n_bits
        final_cond_layers = [None] * n_bits

        for i, layer in enumerate(layers):
            for op in layer:
                if isinstance(op, MidMeasureMP) and op in bit_map:
                    measurement_layers[bit_map[op]] = i

                if isinstance(op, Conditional):
                    for mid_measure in op.meas_val.measurements:
                        final_cond_layers[bit_map[mid_measure]] = i

    return bit_map, measurement_layers, final_cond_layers


def _new_find_mid_measure_connections(
    tape, layers, wire_map, measurment_layers_index
):  # pylint: disable=too-many-branches
    """Collect and return information about connections between mid-circuit measurements
    and classical conditions and statistics that they connect to.

    This utility function returns three items needed for processing mid-circuit measurements
    and classical conditions and statistics for drawing:

    * A dictionary mapping each mid-circuit measurement to a corresponding bit index.
        This map only contains mid-circuit measurements that are used for classical conditioning
        or collecting statistics.
    * A list where each index is a bit and the values are the indices  of the layers containing
        the mid-circuit measurement corresponding to the bits.
    * A list where each index is a bit and the values are the indices of the last layers where the
        bits should be drawn.

    Args:
        tape (~.QuantumScript): Tape to be drawn
        layers (list[list[~.Operation]]): List of drawable layers containing list of operations
            for each layer
        measurement_layres_index (int): index for the first measurement layer

    Returns:
        tuple[dict, list, list]: Data structures needed for correctly drawing classical conditions
        as described above.
    """

    used_mid_measures = set()
    conditional_ops = []
    for op in tape.operations:
        if isinstance(op, Conditional):
            used_mid_measures.update(op.meas_val.measurements)
            conditional_ops.append(op)

    for mp in tape.measurements:
        if mp.mv is not None:
            used_mid_measures.update(mp.mv.measurements)

    if len(used_mid_measures) == 0:
        return {}, []

    # Map between mid-circuit measurements and their position on the drawing
    # The bit map only contains mid-circuit measurements that are used in
    # classical conditions. Order is decided by order of mid circuit measurements
    # in the quantum tape.
    used_mid_measures = list(used_mid_measures)
    used_mid_measures.sort(key=tape.operations.index)

    bit_map = dict(zip(used_mid_measures, range(len(used_mid_measures))))
    n_bits = len(bit_map)

    # Map between classical bit positions and properties about its mid-circuit measurement
    # conditions depending on it. See MidMeasureProperties to learn about which properties
    # are stored.
    bit_properties = []

    # Create lists of properties
    mid_measure_ops = sorted(used_mid_measures, key=bit_map.get)
    mid_measure_layers = [None] * n_bits
    mid_measure_wires = [None] * n_bits
    cond_layers = [[] for _ in range(n_bits)]
    cond_wires = [[] for _ in range(n_bits)]
    last_layers = [None] * n_bits

    for i, layer in enumerate(layers[:measurment_layers_index]):
        for op in layer:
            if isinstance(op, MidMeasureMP) and op in bit_map:
                mid_measure_layers[bit_map[op]] = i
                mid_measure_wires[bit_map[op]] = wire_map[op.wires[0]]

            elif isinstance(op, Conditional):
                mapped_wires = [wire_map[w] for w in op.then_op.wires]
                max_wire = max(mapped_wires)

                for mid_measure in op.meas_val.measurements:
                    cond_layers[bit_map[mid_measure]].append(op)
                    cond_wires[bit_map[mid_measure]].append(max_wire)
                    last_layers[bit_map[mid_measure]] = i

    for layer in layers[measurment_layers_index:]:
        for mp in layer:
            if mp.mv is not None:
                for mid_measure in mp.mv.measurements:
                    last_layers[bit_map[mid_measure]] = measurment_layers_index - 1

    for mo, ml, mw, cl, cw, ll in zip(
        mid_measure_ops, mid_measure_layers, mid_measure_wires, cond_layers, cond_wires, last_layers
    ):
        mid_measure_properties = MidMeasureProperties(mo, ml, mw, cl, cw, ll)
        bit_properties.append(mid_measure_properties)

    return bit_map, bit_properties
