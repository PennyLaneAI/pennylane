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
This module contains integration functions for drawing tapes.
"""

# cant import pennylane as a whole because of circular imports with circuit graph
from pennylane import ops
from pennylane.wires import Wires
from .mpldrawer import MPLDrawer
from .drawable_layers import drawable_layers
from .utils import convert_wire_order

def _add_swap(drawer, layer, mapped_wires):
    drawer.SWAP(layer, mapped_wires)

def _add_cswap(drawer, layer, mapped_wires):
    drawer.ctrl(layer, wires=mapped_wires[0], wires_target=mapped_wires[1:])
    drawer.SWAP(layer, wires=mapped_wires[1:])

def _add_cx(drawer, layer, mapped_wires):
    drawer.CNOT(layer, mapped_wires)

def _add_cz(drawer, layer, mapped_wires):
    drawer.ctrl(layer, mapped_wires)

special_cases = {
    ops.SWAP: _add_swap,
    ops.CSWAP: _add_cswap,
    ops.CNOT: _add_cx,
    ops.Toffoli: _add_cx,
    ops.MultiControlledX: _add_cx,
    ops.CZ: _add_cz
}

def draw(tape, wire_order=None, show_all_wires=False, decimals=None):
    """docstring

    Args:
        tape

    Keyword Args:
        wire_order=None
        show_all_wires=False

    Returns:
        fig, ax

    """

    if wire_order is None:
        wire_order = tape.wires
    else:
        wire_order = Wires.all_wires([Wires(wire_order, tape.wires)])

    wire_map = convert_wire_order(tape.operations+tape.measurements, wire_order=wire_order,
        show_all_wires=show_all_wires)

    layers = drawable_layers(tape.operations, wire_map=wire_map)

    n_layers = len(layers)
    n_wires = len(wire_map)

    drawer = MPLDrawer(n_layers=n_layers, n_wires=n_wires)
    drawer.label([label for label in wire_map])

    for layer, layer_ops in enumerate(layers):
        for op in layer_ops:
            mapped_wires = [wire_map[w] for w in op.wires]
            try:
                control_wires = [wire_map[w] for w in op.control_wires]
                target_wires = [wire_map[w] for w in op.wires if w not in op.control_wires]
            except (NotImplementedError, AttributeError):
                control_wires = None
                target_wires = None

            specialfunc = special_cases.get(op.__class__, None)
            if specialfunc is not None:
                specialfunc(drawer, layer, mapped_wires)

            elif control_wires is not None:
                drawer.ctrl(layer, control_wires, wires_target=target_wires)
                drawer.box_gate(layer, target_wires, op.label(decimals=decimals) )

            else:
                drawer.box_gate(layer, mapped_wires, op.label(decimals=decimals) )

    m_wires = Wires([])
    for m in tape.measurements:
        if len(m.wires) == 0:
            m_wires = -1
            break
        else:
            m_wires += m.wires

    if m_wires == -1:
        for wire in range(n_wires):
            drawer.measure(n_layers, wire)
    else:
        for wire in m_wires:
            drawer.measure(n_layers, wire_map[wire])

    return drawer.fig, drawer.ax