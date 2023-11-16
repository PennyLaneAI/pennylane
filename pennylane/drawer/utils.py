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
from pennylane.ops import Controlled
from pennylane.measurements import MidMeasureMP


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
    """Create dictionaries with various mappings needed for drawing
    mid-circuit measurements and classically controlled operators
    correctly."""

    # Map between mid-circuit measurements and their position on the drawing
    # The bit map only contains mid-circuit measurements that are used in
    # classical conditions.
    bit_map = {}

    # Map between classical bit positions and whether we have reached the mid-circuit
    # measurement of that bit. This is needed to know when to start drawing the bit line.
    bit_measurements_reached = []

    # Map between classical bit positions and the final layer where the bit is used.
    # This is needed to know when to stop drawing a bit line. The bit is the index,
    # so each of the two lists must have the same length as the number of bits
    all_bit_terminal_layers = [[], []]

    measurements_for_conds = set()
    conditional_ops = []
    for op in operations:
        if op.__class__.__name__ == "Conditional":
            measurements_for_conds.update(op.meas_val.measurements)
            conditional_ops.append(op)

    if len(measurements_for_conds) > 0:
        cond_mid_measures = [
            op for op in operations if isinstance(op, MidMeasureMP) if op in measurements_for_conds
        ]
        cond_mid_measures.sort(
            key=lambda mp: operations.index(mp)  # pylint: disable=unnecessary-lambda
        )

        bit_map = dict(zip(cond_mid_measures, range(len(cond_mid_measures))))

        n_bits = len(bit_map)
        bit_measurements_reached = [False for _ in range(n_bits)]

        # Terminal layers in ops
        all_bit_terminal_layers[0] = [None for _ in range(n_bits)]
        # Terminal layers in meas
        all_bit_terminal_layers[1] = [-1 for _ in range(n_bits)]
        # Only iterating through operation layers because bits are only determined
        # using those layers
        for i, layer in enumerate(layers[0]):
            for op in layer:
                if op.__class__.__name__ == "Conditional":
                    for mid_measure in op.meas_val.measurements:
                        all_bit_terminal_layers[0][bit_map[mid_measure]] = i

    return bit_map, bit_measurements_reached, all_bit_terminal_layers
