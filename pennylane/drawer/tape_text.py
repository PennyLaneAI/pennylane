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
This module contains logic for the text based circuit drawer through the ``tape_text`` function.
"""

import pennylane as qml
from pennylane.measurements import Expectation, Probability, Sample, Variance, State, MidMeasureMP

from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls


def _add_grouping_symbols(op, layer_str, wire_map, bit_map):
    """Adds symbols indicating the extent of a given object."""
    if op.__class__.__name__ == "Conditional":
        return _add_cond_grouping_symbols(op, layer_str, wire_map, bit_map)
    if isinstance(op, MidMeasureMP):
        _add_mid_measure_grouping_symbols(op, layer_str, wire_map, bit_map)

    if len(op.wires) > 1:
        mapped_wires = [wire_map[w] for w in op.wires]
        min_w, max_w = min(mapped_wires), max(mapped_wires)
        layer_str[min_w] = "╭"
        layer_str[max_w] = "╰"

        for w in range(min_w + 1, max_w):
            layer_str[w] = "├" if w in mapped_wires else "│"

    return layer_str


def _add_cond_grouping_symbols(op, layer_str, wire_map, bit_map):
    """Adds symbols indicating the extent of a given object for conditional
    operators"""
    n_wires = len(wire_map)
    mapped_wires = [wire_map[w] for w in op.wires]
    mapped_bits = [bit_map[m] for m in op.meas_val.measurements]
    min_w = min(mapped_wires)
    max_b = max(mapped_bits)
    layer_str[min_w] = "╓"
    layer_str[max_b] = "╚"

    for w in range(min_w + 1, n_wires):
        layer_str[w] = "╟" if w in mapped_wires else "║"

    for b in range(n_wires, max_b):
        layer_str[b] = "╠" if b in mapped_bits else "║"

    return layer_str


def _add_mid_measure_grouping_symbols(op, layer_str, wire_map, bit_map):
    """Adds symbols indicating the extent of a given object for conditional
    operators"""
    mapped_wires = [wire_map[w] for w in op.wires]
    bit = bit_map[op]
    min_w = min(mapped_wires)
    layer_str[min_w] = "╔"
    layer_str[bit] = "╚"

    for w in range(min_w + 1, bit):
        layer_str[w] = "║"


def _add_op(op, layer_str, wire_map, bit_map, decimals, cache):
    """Updates ``layer_str`` with ``op`` operation."""
    if op.__class__.__name__ == "Conditional":
        return _add_cond_op(op, layer_str, wire_map, bit_map, decimals, cache)
    if isinstance(op, MidMeasureMP):
        _add_mid_measure_op(op, layer_str, wire_map, bit_map, decimals, cache)

    layer_str = _add_grouping_symbols(op, layer_str, wire_map, bit_map)

    control_wires, control_values = unwrap_controls(op)

    if control_values:
        for w, val in zip(control_wires, control_values):
            layer_str[wire_map[w]] += "●" if val else "○"
    else:
        for w in control_wires:
            layer_str[wire_map[w]] += "●"

    label = op.label(decimals=decimals, cache=cache).replace("\n", "")
    if len(op.wires) == 0:  # operation (e.g. barrier, snapshot) across all wires
        for i, s in enumerate(layer_str):
            layer_str[i] = s + label
    else:
        for w in op.wires:
            if w not in control_wires:
                layer_str[wire_map[w]] += label

    return layer_str


def _add_cond_op(op, layer_str, wire_map, bit_map, decimals, cache):
    """Updates ``layer_str`` with ``op`` operation when ``op`` is a
    ``qml.transforms.Conditional``."""
    layer_str = _add_grouping_symbols(op, layer_str, wire_map, bit_map)

    control_bits = [bit_map[m] for m in op.meas_val.measurements]

    for b in control_bits:
        layer_str[b] += "●"

    label = op.label(decimals=decimals, cache=cache).replace("\n", "")
    if len(op.wires) == 0:  # operation (e.g. barrier, snapshot) across all wires
        for i, s in enumerate(layer_str):
            layer_str[i] = s + label
    else:
        for w in op.wires:
            layer_str[wire_map[w]] += label

    return layer_str


def _add_mid_measure_op(op, layer_str, wire_map, bit_map, decimals, cache):
    """Updates ``layer_str`` with ``op`` operation when ``op`` is a
    ``qml.measurements.MidMeasureMP``."""
    layer_str = _add_grouping_symbols(op, layer_str, wire_map, bit_map)
    label = op.label(decimals=decimals, cache=cache).replace("\n", "")

    for w in op.wires:
        layer_str[wire_map[w]] += label

    return layer_str


measurement_label_map = {
    Expectation: lambda label: f"<{label}>",
    Probability: lambda label: f"Probs[{label}]" if label else "Probs",
    Sample: lambda label: f"Sample[{label}]" if label else "Sample",
    Variance: lambda label: f"Var[{label}]",
    State: lambda label: "State",
}


def _add_measurement(m, layer_str, wire_map, bit_map, decimals, cache):
    """Updates ``layer_str`` with the ``m`` measurement."""
    layer_str = _add_grouping_symbols(m, layer_str, wire_map, bit_map)

    if m.obs is None:
        obs_label = None
    else:
        obs_label = m.obs.label(decimals=decimals, cache=cache).replace("\n", "")
    meas_label = measurement_label_map[m.return_type](obs_label)

    if len(m.wires) == 0:  # state or probability across all wires
        for i, s in enumerate(layer_str):
            layer_str[i] = s + meas_label

    for w in m.wires:
        layer_str[wire_map[w]] += meas_label
    return layer_str


# pylint: disable=too-many-arguments
def tape_text(
    tape,
    wire_order=None,
    show_all_wires=False,
    decimals=None,
    max_length=100,
    show_matrices=True,
    cache=None,
):
    """Text based diagram for a Quantum Tape.

    Args:
        tape (QuantumTape): the operations and measurements to draw

    Keyword Args:
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        decimals (int): How many decimal points to include when formatting operation parameters.
            Default ``None`` will omit parameters from operation labels.
        max_length (Int) : Maximum length of a individual line.  After this length, the diagram will
            begin anew beneath the previous lines.
        show_matrices=True (bool): show matrix valued parameters below all circuit diagrams
        cache (dict): Used to store information between recursive calls. Necessary keys are ``'tape_offset'``
            and ``'matrices'``.

    Returns:
        str : String based graphic of the circuit.

    **Example:**

    .. code-block:: python

        ops = [
            qml.QFT(wires=(0, 1, 2)),
            qml.RX(1.234, wires=0),
            qml.RY(1.234, wires=1),
            qml.RZ(1.234, wires=2),
            qml.Toffoli(wires=(0, 1, "aux"))
        ]
        measurements = [
            qml.expval(qml.PauliZ("aux")),
            qml.var(qml.PauliZ(0) @ qml.PauliZ(1)),
            qml.probs(wires=(0, 1, 2, "aux"))
        ]
        tape = qml.tape.QuantumTape(ops, measurements)

    >>> print(qml.drawer.tape_text(tape))
      0: ─╭QFT──RX─╭●─┤ ╭Var[Z@Z] ╭Probs
      1: ─├QFT──RY─├●─┤ ╰Var[Z@Z] ├Probs
      2: ─╰QFT──RZ─│──┤           ├Probs
    aux: ──────────╰X─┤  <Z>      ╰Probs

    .. details::
        :title: Usage Details

    By default, parameters are omitted. By specifying the ``decimals`` keyword, parameters
    are displayed to the specified precision. Matrix-valued parameters are never displayed.

    >>> print(qml.drawer.tape_text(tape, decimals=2))
      0: ─╭QFT──RX(1.23)─╭●─┤ ╭Var[Z@Z] ╭Probs
      1: ─├QFT──RY(1.23)─├●─┤ ╰Var[Z@Z] ├Probs
      2: ─╰QFT──RZ(1.23)─│──┤           ├Probs
    aux: ────────────────╰X─┤  <Z>      ╰Probs


    The ``max_length`` keyword wraps long circuits:

    .. code-block:: python

        rng = np.random.default_rng(seed=42)
        shape = qml.StronglyEntanglingLayers.shape(n_wires=5, n_layers=5)
        params = rng.random(shape)
        tape2 = qml.StronglyEntanglingLayers(params, wires=range(5)).expand()
        print(qml.drawer.tape_text(tape2, max_length=60))


    .. code-block:: none

        0: ──Rot─╭●──────────╭X──Rot─╭●───────╭X──Rot──────╭●────╭X
        1: ──Rot─╰X─╭●───────│───Rot─│──╭●────│──╭X────Rot─│──╭●─│─
        2: ──Rot────╰X─╭●────│───Rot─╰X─│──╭●─│──│─────Rot─│──│──╰●
        3: ──Rot───────╰X─╭●─│───Rot────╰X─│──╰●─│─────Rot─╰X─│────
        4: ──Rot──────────╰X─╰●──Rot───────╰X────╰●────Rot────╰X───

        ───Rot───────────╭●─╭X──Rot──────╭●──────────────╭X─┤
        ──╭X────Rot──────│──╰●─╭X────Rot─╰X───╭●─────────│──┤
        ──│────╭X────Rot─│─────╰●───╭X────Rot─╰X───╭●────│──┤
        ──╰●───│─────Rot─│──────────╰●───╭X────Rot─╰X─╭●─│──┤
        ───────╰●────Rot─╰X──────────────╰●────Rot────╰X─╰●─┤


    The ``wire_order`` keyword specifies the order of the wires from
    top to bottom:

    >>> print(qml.drawer.tape_text(tape, wire_order=["aux", 2, 1, 0]))
    aux: ──────────╭X─┤  <Z>      ╭Probs
      2: ─╭QFT──RZ─│──┤           ├Probs
      1: ─├QFT──RY─├●─┤ ╭Var[Z@Z] ├Probs
      0: ─╰QFT──RX─╰●─┤ ╰Var[Z@Z] ╰Probs

    If the wire order contains empty wires, they are only shown if the ``show_all_wires=True``.

    >>> print(qml.drawer.tape_text(tape, wire_order=["a", "b", "aux", 0, 1, 2], show_all_wires=True))
      a: ─────────────┤
      b: ─────────────┤
    aux: ──────────╭X─┤  <Z>      ╭Probs
      0: ─╭QFT──RX─├●─┤ ╭Var[Z@Z] ├Probs
      1: ─├QFT──RY─╰●─┤ ╰Var[Z@Z] ├Probs
      2: ─╰QFT──RZ────┤           ╰Probs

    Matrix valued parameters are always denoted by ``M`` followed by an integer corresponding to
    unique matrices.  The list of unique matrices can be printed at the end of the diagram by
    selecting ``show_matrices=True`` (the default):

    .. code-block:: python

        ops = [
            qml.QubitUnitary(np.eye(2), wires=0),
            qml.QubitUnitary(np.eye(2), wires=1)
        ]
        measurements = [qml.expval(qml.Hermitian(np.eye(4), wires=(0,1)))]
        tape = qml.tape.QuantumTape(ops, measurements)

    >>> print(qml.drawer.tape_text(tape))
    0: ──U(M0)─┤ ╭<𝓗(M1)>
    1: ──U(M0)─┤ ╰<𝓗(M1)>
    M0 =
    [[1. 0.]
    [0. 1.]]
    M1 =
    [[1. 0. 0. 0.]
    [0. 1. 0. 0.]
    [0. 0. 1. 0.]
    [0. 0. 0. 1.]]

    An existing matrix cache can be passed via the ``cache`` keyword. Note that the dictionary
    passed to ``cache`` will be modified during execution to contain any new matrices and the
    tape offset.

    >>> cache = {'matrices': [-np.eye(3)]}
    >>> print(qml.drawer.tape_text(tape, cache=cache))
    0: ──U(M1)─┤ ╭<𝓗(M2)>
    1: ──U(M1)─┤ ╰<𝓗(M2)>
    M0 =
    [[-1. -0. -0.]
    [-0. -1. -0.]
    [-0. -0. -1.]]
    M1 =
    [[1. 0.]
    [0. 1.]]
    M2 =
    [[1. 0. 0. 0.]
    [0. 1. 0. 0.]
    [0. 0. 1. 0.]
    [0. 0. 0. 1.]]
    >>> cache
    {'matrices': [tensor([[-1., -0., -0.],
        [-0., -1., -0.],
        [-0., -0., -1.]], requires_grad=True), tensor([[1., 0.],
        [0., 1.]], requires_grad=True), tensor([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]], requires_grad=True)], 'tape_offset': 0}

    When the provided tape has nested tapes inside, this function is called recursively.
    To maintain numbering of tapes to arbitrary levels of nesting, the ``cache`` keyword
    uses the ``"tape_offset"`` value to determine numbering. Note that the value is updated
    during the call.

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            with qml.tape.QuantumTape() as tape_inner:
                qml.PauliX(0)

        cache = {'tape_offset': 3}
        print(qml.drawer.tape_text(tape, cache=cache))
        print("New tape offset: ", cache['tape_offset'])


    .. code-block:: none

        0: ──Tape:3─┤

        Tape:3
        0: ──X─┤
        New tape offset:  4

    """
    cache = cache or {}
    cache.setdefault("tape_offset", 0)
    cache.setdefault("matrices", [])
    tape_cache = []

    wire_map = convert_wire_order(tape, wire_order=wire_order, show_all_wires=show_all_wires)
    n_wires = len(wire_map)
    if n_wires == 0:
        return ""

    conditional_measurements = set()
    for op in tape.operations:
        if op.__class__.__name__ == "Conditional":
            conditional_measurements.update(op.meas_val.measurements)

    if len(conditional_measurements) > 0:
        mid_measures = [
            op
            for op in tape.operations
            if isinstance(op, MidMeasureMP)
            if op in conditional_measurements
        ]
        bit_map = dict(zip(mid_measures, range(n_wires, n_wires + len(mid_measures))))
    else:
        bit_map = {}
    n_bits = len(bit_map)
    bit_measurements_reached = dict(zip(bit_map.values(), [False] * n_bits))

    wire_totals = [f"{wire}: " for wire in wire_map]
    bit_totals = ["" for _ in range(n_bits)]
    line_length = max(len(s) for s in wire_totals)

    wire_totals = [s.rjust(line_length, " ") for s in wire_totals]
    bit_totals = [s.rjust(line_length, " ") for s in bit_totals]
    totals = wire_totals + bit_totals

    # Used to store lines that are hitting the maximum length
    finished_lines = []

    layers_list = [
        drawable_layers(tape.operations, wire_map=wire_map),
        drawable_layers(tape.measurements, wire_map=wire_map),
    ]
    add_list = [_add_op, _add_measurement]
    wire_fillers = ["─", " "]
    bit_fillers = [[" ", "═"], [" ", " "]]
    enders = [True, False]  # add "─┤" after all operations

    for layers, add, w_filler, b_filler, ender in zip(
        layers_list, add_list, wire_fillers, bit_fillers, enders
    ):
        for layer in layers:
            layer_str = [w_filler] * n_wires + [""] * n_bits
            for b in bit_map.values():
                # layer_str[b] = b_filler[1] if bit_measurements_reached[b] else b_filler[0]
                layer_str[b] = b_filler[0]

            for op in layer:
                layer_mid_measure = None

                if isinstance(op, qml.tape.QuantumScript):
                    layer_str = _add_grouping_symbols(op, layer_str, wire_map, bit_map)
                    label = f"Tape:{cache['tape_offset']+len(tape_cache)}"
                    for w in op.wires:
                        layer_str[wire_map[w]] += label
                    tape_cache.append(op)
                else:
                    layer_str = add(op, layer_str, wire_map, bit_map, decimals, cache)

                    if isinstance(op, MidMeasureMP) and bit_map.get(op, None) is not None:
                        layer_mid_measure = op

            # Adjust width of layer for wires
            max_label_len = max(len(s) for s in layer_str)
            for w in range(n_wires):
                layer_str[w] = layer_str[w].ljust(max_label_len, w_filler)

            # Adjust width of layer for bits
            for b in range(n_wires, n_wires + n_bits):
                cur_b_filler = b_filler[1] if bit_measurements_reached[b] else b_filler[0]
                layer_str[b] = layer_str[b].ljust(max_label_len, cur_b_filler)

            line_length += max_label_len + 1  # one for the filler character
            if line_length > max_length:
                # move totals into finished_lines and reset totals
                finished_lines += totals
                finished_lines[-1] += "\n"
                wire_totals = [w_filler] * n_wires
                bit_totals = [b_filler] * n_bits
                totals = wire_totals + bit_totals
                line_length = 2 + max_label_len

            # Join current layer with lines for previous layers
            wire_totals = [w_filler.join([t, s]) for t, s in zip(wire_totals, layer_str[:n_wires])]
            new_bit_totals = []
            for i, (bt, s) in enumerate(zip(bit_totals, layer_str[n_wires : n_wires + n_bits])):
                cur_b_filler = b_filler[1] if bit_measurements_reached[i + n_wires] else b_filler[0]
                new_bit_totals.append(cur_b_filler.join([bt, s]))
            bit_totals = new_bit_totals

            # Update classical bit filler
            if layer_mid_measure is not None:
                bit_measurements_reached[bit_map[layer_mid_measure]] = True

            totals = wire_totals + bit_totals
        if ender:
            totals = [s + "─┤" for s in totals]
            line_length += 2

    # Recursively handle nested tapes #
    tape_totals = "\n".join(finished_lines + totals)
    current_tape_offset = cache["tape_offset"]
    cache["tape_offset"] += len(tape_cache)
    for i, nested_tape in enumerate(tape_cache):
        label = f"\nTape:{i+current_tape_offset}"
        tape_str = tape_text(
            nested_tape,
            wire_order,
            show_all_wires,
            decimals,
            max_length,
            show_matrices=False,
            cache=cache,
        )
        tape_totals = "\n".join([tape_totals, label, tape_str])

    if show_matrices:
        mat_str = ""
        for i, mat in enumerate(cache["matrices"]):
            mat_str += f"\nM{i} = \n{mat}"
        return tape_totals + mat_str

    return tape_totals
