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

from pennylane.wires import Wires

from .drawable_layers import drawable_grid
from .utils import convert_wire_order
from .circuit_drawer import CircuitDrawer

def draw(tape, charset="unicode", wire_order=None, show_all_wires=False):
    """Draw a QuantumTape
    
    Args:
        tape (QuantumTape): tape to draw

    Keyword Args:
        charset (str, optional): The charset that should be used. Currently, "unicode" and
            "ascii" are supported.
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
        show_all_wires (bool): If True, all wires, including empty wires, are printed.

    Returns:
        str
    
    """

    if wire_order is None:
        wire_order = tape.wires
    else:
        wire_order = Wires.all_wires([Wires(wire_order), tape.wires])

    wire_map = convert_wire_order(tape.operations+tape.measurements, wire_order=wire_order,
        show_all_wires=show_all_wires)

    ops_grid = drawable_grid(tape.operations, wire_map=wire_map)
    meas_grid = drawable_grid(tape.measurements, wire_map=wire_map)

    drawer = CircuitDrawer(ops_grid, meas_grid, wires=wire_order,
        charset=charset, show_all_wires=show_all_wires)

    return drawer.draw()