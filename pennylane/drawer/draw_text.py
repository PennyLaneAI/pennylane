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
This module contains the ``draw_text`` function
"""

from pennylane.operation import Expectation, Probability, Sample, Variance, State

from .drawable_layers import drawable_layers
from .utils import convert_wire_order


measurement_label_map = {
    Expectation: lambda label: f"<{label}>",
    Probability: lambda label: "Probs",
    Sample: lambda label: "Sample",
    Variance: lambda label: f"Var[{label}]",
    State: lambda label: "State"
}


def _add_op(op, layer_str, wire_map, decimals):
    """Updates ``layer_str`` with ``op`` operation."""
    if len(op.wires) > 1:
        mapped_wires = [wire_map[w] for w in op.wires]
        min_w, max_w = min(mapped_wires),  max(mapped_wires)
        layer_str[min_w] = "╭"
        layer_str[max_w] = "╰"

        for w in range(min_w+1, max_w):
            layer_str[w] = "├" if w in mapped_wires else "│"

    for w in op.control_wires:
        layer_str[wire_map[w]] += "C"

    for w in op.wires:
        if w not in op.control_wires:
            layer_str[wire_map[w]] += op.label(decimals=decimals).replace("\n", "")

    return layer_str


def _add_measurement(m, layer_str, wire_map, decimals):
    """Updates ``layer_str`` with the ``m`` measurement. """
    obs_label = "" if m.obs is None else m.obs.label(decimals=decimals).replace("\n", "")
    meas_label = measurement_label_map[m.return_type](obs_label)

    if len(m.wires) == 0: # state or probability across all wires
        for _, w in wire_map.items():
            layer_str[w] = meas_label

    for w in m.wires:
        layer_str[wire_map[w]] = meas_label
    return layer_str


def draw_text(tape, wire_order=None, show_all_wires=False, decimals=None, max_length=100):
    """Text based diagram for a Quantum Tape.
    
    Args:
        tape (QuantumTape)

    Keyword Args:
        wire_order (Iterable)
        show_all_wires (Bool)
        decimals (Int) : how many decimal points to display in the operation label.  If ``None``,
            no parameters will be displayed.
        max_length (Int) : Maximum length of a individual line.  After this length, the diagram will
        begin anew beneath the previous lines.

    Returns:
        str : String based graphic of the circuit.
    
    **Example:**

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            qml.QFT(wires=(0,1,2))
            qml.RX(1.234, wires=0)
            qml.RY(1.234, wires=1)
            qml.RZ(1.234, wires=2)
            qml.Toffoli(wires=(0,1,"aux"))
            
            qml.expval(qml.PauliZ("aux"))
            qml.state()

    >>> print(draw_text(tape))
      0: ─╭QFT──RX─╭C─┤     State
      1: ─├QFT──RY─├C─┤     State
      2: ─╰QFT──RZ─│──┤     State
    aux: ──────────╰X─┤ <Z> State

    .. UsageDetails::

    **Decimals:**

    >>> print(draw_text(tape, decimals=2))
      0: ─╭QFT──RX(1.23)─╭C─┤     State
      1: ─├QFT──RY(1.23)─├C─┤     State
      2: ─╰QFT──RZ(1.23)─│──┤     State
    aux: ────────────────╰X─┤ <Z> State

    **Max Length:**

    >>> rng = np.random.default_rng(seed=42)
    >>> shape = qml.StronglyEntanglingLayers.shape(n_wires=5, n_layers=5)
    >>> params = rng.random(shape)
    >>> tape2 = qml.StronglyEntanglingLayers(params, wires=range(5)).expand()
    >>> print(draw_text(tape2, max_length=60))
    0: ──Rot─╭C──────────╭X──Rot─╭C───────╭X──Rot──────╭C────╭X
    1: ──Rot─╰X─╭C───────│───Rot─│──╭C────│──╭X────Rot─│──╭C─│─
    2: ──Rot────╰X─╭C────│───Rot─╰X─│──╭C─│──│─────Rot─│──│──╰C
    3: ──Rot───────╰X─╭C─│───Rot────╰X─│──╰C─│─────Rot─╰X─│────
    4: ──Rot──────────╰X─╰C──Rot───────╰X────╰C────Rot────╰X───

    ───Rot───────────╭C─╭X──Rot──────╭C──────────────╭X─┤  
    ──╭X────Rot──────│──╰C─╭X────Rot─╰X───╭C─────────│──┤  
    ──│────╭X────Rot─│─────╰C───╭X────Rot─╰X───╭C────│──┤  
    ──╰C───│─────Rot─│──────────╰C───╭X────Rot─╰X─╭C─│──┤  
    ───────╰C────Rot─╰X──────────────╰C────Rot────╰X─╰C─┤  

    **Wire Order:**

    >>> print(draw_text(tape, wire_order=["aux", 2, 1, 0]))
    aux: ──────────╭X─┤ <Z> State
      2: ─╭QFT──RZ─│──┤     State
      1: ─├QFT──RY─├C─┤     State
      0: ─╰QFT──RX─╰C─┤     State

    **Show all wires:**

    >>> print(draw_text(tape, wire_order=["a", "b", "aux", 0,1,2], show_all_wires=True))
      a: ─────────────┤     State
      b: ─────────────┤     State
    aux: ──────────╭X─┤ <Z> State
      0: ─╭QFT──RX─├C─┤     State
      1: ─├QFT──RY─╰C─┤     State
      2: ─╰QFT──RZ────┤     State

    """

    wire_map = convert_wire_order(tape.operations+tape.measurements, wire_order=wire_order,
        show_all_wires=show_all_wires)
    n_wires = len(wire_map)
    if n_wires == 0:
        return ""

    totals = [f"{wire}: " for wire in wire_map]
    line_length = max(len(s) for s in totals)
    totals = [s.rjust(line_length, " ") for s in totals]

    # Used to store lines that are hitting the maximum length
    finished_lines = []

    layers_list = [drawable_layers(tape.operations, wire_map=wire_map),
        drawable_layers(tape.measurements, wire_map=wire_map)]
    add_list = [_add_op, _add_measurement]
    fillers = ["─", " "]
    enders = [True, False] # add "─┤" after all operations

    for layers, add, filler, ender in zip(layers_list, add_list, fillers, enders):

        for layer in layers:
            layer_str = [filler] * n_wires
            for op in layer:
                layer_str = add(op, layer_str, wire_map, decimals)

            max_label_len = max(len(s) for s in layer_str)
            layer_str = [s.ljust(max_label_len, filler) for s in layer_str]
            
            line_length += max_label_len + 1 # one for the filler character
            if line_length > max_length:
                # move totals into finished_lines and reset totals
                finished_lines += totals
                finished_lines[-1] += "\n"
                totals = [filler] * n_wires
                line_length = 1

            totals = [filler.join([t, s]) for t, s in zip(totals, layer_str)]
        
        if ender:
            totals = [s + "─┤" for s in totals]

    return "\n".join(finished_lines+totals)
