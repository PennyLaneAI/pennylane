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
