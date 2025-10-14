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
import numpy as np

from pennylane.measurements import MeasurementProcess, MeasurementValue, MidMeasureMP
from pennylane.ops import Conditional, Controlled


def default_wire_map(tape):
    """Create a dictionary mapping used wire labels to non-negative integers

    Args:
        tape [~.tape.QuantumTape): the QuantumTape containing operations and measurements

    Returns:
        tuple[dict]: A tuple of maps from wires to sequential positive integers. The first map
        includes work wires whereas the second map excludes work wires.
    """

    # Use dictionary to preserve ordering, sets break order
    used_wires = {wire: None for op in tape for wire in op.wires}
    used_wire_map = {wire: ind for ind, wire in enumerate(used_wires)}
    # Will only add wires that are not present in used_wires yet, and to the end of used_wires
    used_and_work_wires = used_wires | {
        wire: None for op in tape for wire in getattr(op, "work_wires", [])
    }
    full_wire_map = {wire: ind for ind, wire in enumerate(used_and_work_wires)}
    return full_wire_map, used_wire_map


def default_bit_map(tape):
    """Create a dictionary mapping ``MidMeasureMP``'s to indices corresponding to classical
    wires. We only add mid-circuit measurements that are used for classical conditions and for
    collecting statistics to this dictionary.

    Args:
        tape [~.tape.QuantumTape]: the QuantumTape containing operations and measurements

    Returns:
        dict: map from mid-circuit measurements to classical wires."""
    bit_map = {}
    mcms = {}

    mcm_idx = 0
    for op in tape:
        if isinstance(op, MidMeasureMP):
            mcms[op] = mcm_idx
            mcm_idx += 1

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

    bit_map = {mcm: i for i, mcm in enumerate(sorted(bit_map, key=mcms.get))}

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
        tuple[dict]: Two maps from wire labels to sequential positive integers. The first map
        includes work wires, the second map excludes work wires.
    """
    full_wire_map, used_wire_map = default_wire_map(tape)

    if wire_order is None:
        # If no external wire order is dictated, the tape ordering is all we need to consider
        return full_wire_map, used_wire_map

    # Create wire order complemented by all wires in the tape mapping that are not in the order yet
    full_wire_order = list(wire_order) + [wire for wire in full_wire_map if wire not in wire_order]
    used_wire_order = list(wire_order) + [wire for wire in used_wire_map if wire not in wire_order]

    if not show_all_wires:
        # Filter out wires that are in wire_order but not in full_wire_map/used_wire_map
        full_wire_order = [wire for wire in full_wire_order if wire in full_wire_map]
        used_wire_order = [wire for wire in used_wire_order if wire in used_wire_map]

    # Create consecutive integer mapping from ordered list
    full_wire_map = {wire: ind for ind, wire in enumerate(full_wire_order)}
    used_wire_map = {wire: ind for ind, wire in enumerate(used_wire_order)}

    return full_wire_map, used_wire_map


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

    next_ctrl = op
    if isinstance(op, Controlled):

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
    return control_wires, control_values, next_ctrl


def cwire_connections(layers, bit_map):
    """Extract the information required for classical control wires.

    Args:
        layers (List[List[.Operator, .MeasurementProcess]]): the operations and measurements sorted
            into layers via ``drawable_layers``. Measurement layers may be appended to operation layers.
        bit_map (Dict): Dictionary containing mid-circuit measurements that are used for
            classical conditions or measurement statistics as keys.

    Returns:
        dict, dict, dict: The first dictionary is the updated ``bit_map``, potentially with
        some mid-circuit measurements mapped to new (smaller) classical wires. The second and third
        dictionaries have the classical wires as keys and lists of lists as values, with the outer
        list running over different (re)usages of the classical wire. For the second dictionary,
        the inner lists contain the indices of the accessed layers, for the third dictionary,
        they contain the measured quantum wires and the largest quantum wire of conditionally
        applied operations (no entries for terminal statistics of mid-circuit measurements).

    >>> from pennylane.drawer.utils import cwire_connections
    >>> from pennylane.drawer.drawable_layers import drawable_layers
    >>> with qml.queuing.AnnotatedQueue() as q:
    ...     m0 = qml.measure(0)
    ...     m1 = qml.measure(1)
    ...     qml.cond(m0 & m1, qml.Y)(0)
    ...     qml.cond(m0, qml.S)(3)
    >>> tape = qml.tape.QuantumScript.from_queue(q)
    >>> bit_map = {m0.measurements[0]: 0, m1.measurements[0]: 1}
    >>> layers = drawable_layers(tape, bit_map=bit_map)
    >>> new_bit_map, cwire_layers, cwire_wires = cwire_connections(layers, bit_map)
    >>> new_bit_map == bit_map # No reusage happening
    True
    >>> cwire_layers
    {0: [[0, 2, 3]], 1: [[1, 2]]}
    >>> cwire_wires
    {0: [[0, 0, 3]], 1: [[1, 0]]}

    From this information, we can see that classical wire ``0`` is active in layers
    0, 2, and 3 while classical wire ``1`` is active in layers 1 and 2, with both classical
    wires being used only once (the outer lists all have length 1). The first "active"
    layer will always be the one with the mid circuit measurement.
    """
    if len(bit_map) == 0:
        return bit_map, {}, {}

    old_cwires = list(bit_map.values())
    connected_layers = {cwire: [] for cwire in old_cwires}
    connected_wires = {cwire: [] for cwire in old_cwires}

    for layer_idx, layer in enumerate(layers):
        for op in layer:
            if isinstance(op, MidMeasureMP) and op in bit_map:
                _meas = [op]
                con_wire = op.wires[0]

            elif isinstance(op, Conditional):
                _meas = op.meas_val.measurements
                con_wire = max(op.wires)

            elif isinstance(op, MeasurementProcess) and op.mv is not None:
                if isinstance(op.mv, MeasurementValue):
                    _meas = op.mv.measurements
                else:
                    _meas = [m.measurements[0] for m in op.mv]
                con_wire = None

            else:
                continue

            for m in _meas:
                cwire = bit_map[m]
                connected_layers[cwire].append(layer_idx)
                if con_wire is not None:
                    connected_wires[cwire].append(con_wire)

    bit_map, connected_layers, connected_wires = _try_reusing_cwires(
        bit_map, connected_layers, connected_wires
    )

    return bit_map, connected_layers, connected_wires


def _try_reusing_cwires(bit_map, connected_layers, connected_wires):
    # Extract (start, end) tuples (incl end) where each cwire is occupied with old bit map
    occupation = {
        cwire: (min(con_layer), max(con_layer)) for cwire, con_layer in connected_layers.items()
    }
    # Mark until where each cwire is currently occupied during the following loop.
    # Start with -1 for each cwire
    occ_ends = -np.ones(len(bit_map))
    # Write a map from old cwires to new cwires
    cwire_map = {}
    for cwire, occ in occupation.items():
        # Find the first cwire that is currently not occupied, i.e. that has its occupation end
        # before the current occ starts (first entry of occ)
        new_cwire = int(np.where(occ_ends < occ[0])[0][0])
        # allocate a new (or the old) cwire based on the first one that was free above
        cwire_map[cwire] = new_cwire
        # Update the occupation end of the newly allocated cwire
        occ_ends[new_cwire] = occ[1]
    # Create an inverted cwire map that maps new cwires to all old cwires that are mapped to it
    inv_cwire_map = {new_cwire: [] for new_cwire in cwire_map.values()}
    for old_cwire in bit_map.values():
        inv_cwire_map[cwire_map[old_cwire]].append(old_cwire)

    # Collect the connected layers from all old cwires that are being mapped to the same new cwire
    connected_layers = {
        new_cwire: [connected_layers[w] for w in old_cwires]
        for new_cwire, old_cwires in inv_cwire_map.items()
    }
    # Collect the connected wires from all old cwires that are being mapped to the same new cwire
    connected_wires = {
        new_cwire: [connected_wires[w] for w in old_cwires]
        for new_cwire, old_cwires in inv_cwire_map.items()
    }
    # Update bit map according to the condensed/reused cwires
    bit_map = {op: cwire_map[cwire] for op, cwire in bit_map.items()}
    return bit_map, connected_layers, connected_wires


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
        new_tape = tape.copy(measurements=new_measurements)
        return new_tape

    return tape
