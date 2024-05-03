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
from pennylane.ops import Controlled, Conditional
from pennylane.measurements import MeasurementProcess, MidMeasureMP, MeasurementValue
from pennylane.tape import QuantumScript


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


def default_bit_map(tape):
    """Create a dictionary mapping ``MidMeasureMP``'s to indices corresponding to classical
    wires. We only add mid-circuit measurements that are used for classical conditions and for
    collecting statistics to this dictionary.

    Args:
        tape [~.tape.QuantumTape]: the QuantumTape containing operations and measurements

    Returns:
        dict: map from mid-circuit measurements to classical wires."""
    bit_map = {}

    for op in tape:
        if isinstance(op, Conditional):
            for m in op.meas_val.measurements:
                bit_map[m] = None

        if isinstance(op, MeasurementProcess) and op.mv is not None:
            if isinstance(op.mv, MeasurementValue):
                for m in op.mv.measurements:
                    bit_map[m] = None
            else:
                for m in op.mv:
                    bit_map[m.measurements[0]] = None

    cur_cwire = 0
    for op in tape:
        if isinstance(op, MidMeasureMP) and op in bit_map:
            bit_map[op] = cur_cwire
            cur_cwire += 1

    return bit_map


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

            if isinstance(next_ctrl.base, Controlled):
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


def cwire_connections(layers, bit_map):
    """Extract the information required for classical control wires.

    Args:
        layers (List[List[.Operator, .MeasurementProcess]]): the operations and measurements sorted
            into layers via ``drawable_layers``. Measurement layers may be appended to operation layers.
        bit_map (Dict): Dictionary containing mid-circuit measurements that are used for
            classical conditions or measurement statistics as keys.

    Returns:
        list, list: list of list of accessed layers for each classical wire, and largest wire
        corresponding to the accessed layers in the list above.

    >>> with qml.queuing.AnnotatedQueue() as q:
    ...     m0 = qml.measure(0)
    ...     m1 = qml.measure(1)
    ...     qml.cond(m0 & m1, qml.Y)(0)
    ...     qml.cond(m0, qml.S)(3)
    >>> tape = qml.tape.QuantumScript.from_queue(q)
    >>> layers = drawable_layers(tape)
    >>> bit_map, cwire_layers, cwire_wires = cwire_connections(layers)
    >>> bit_map
    {measure(wires=[0]): 0, measure(wires=[1]): 1}
    >>> cwire_layers
    [[0, 2, 3], [1, 2]]
    >>> cwire_wires
    [[0, 0, 3], [1, 0]]

    From this information, we can see that the first classical wire is active in layers
    0, 2, and 3 while the second classical wire is active in layers 1 and 2.  The first "active"
    layer will always be the one with the mid circuit measurement.

    """
    if len(bit_map) == 0:
        return [], []

    connected_layers = [[] for _ in bit_map]
    connected_wires = [[] for _ in bit_map]

    for layer_idx, layer in enumerate(layers):
        for op in layer:
            if isinstance(op, MidMeasureMP) and op in bit_map:
                connected_layers[bit_map[op]].append(layer_idx)
                connected_wires[bit_map[op]].append(op.wires[0])

            elif isinstance(op, Conditional):
                for m in op.meas_val.measurements:
                    cwire = bit_map[m]
                    connected_layers[cwire].append(layer_idx)
                    connected_wires[cwire].append(max(op.wires))

            elif isinstance(op, MeasurementProcess) and op.mv is not None:
                if isinstance(op.mv, MeasurementValue):
                    for m in op.mv.measurements:
                        cwire = bit_map[m]
                        connected_layers[cwire].append(layer_idx)
                else:
                    for m in op.mv:
                        cwire = bit_map[m.measurements[0]]
                        connected_layers[cwire].append(layer_idx)

    return connected_layers, connected_wires


def transform_deferred_measurements_tape(tape):
    """Helper function to replace MeasurementValues with wires for tapes using
    deferred measurements."""
    if not any(isinstance(op, MidMeasureMP) for op in tape.operations) and any(
        m.mv is not None for m in tape.measurements
    ):
        new_measurements = []
        for m in tape.measurements:
            if m.mv is not None:
                new_m = type(m)(wires=m.wires)
                new_measurements.append(new_m)
            else:
                new_measurements.append(m)
        new_tape = QuantumScript(tape.operations, new_measurements)
        return new_tape

    return tape
