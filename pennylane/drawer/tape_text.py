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
# TODO: Fix the latter two pylint warnings
# pylint: disable=too-many-arguments, too-many-branches, too-many-statements

from dataclasses import dataclass
from typing import Optional
import pennylane as qml
from pennylane.measurements import (
    Expectation,
    Probability,
    Sample,
    Variance,
    State,
    Counts,
    MidMeasureMP,
)

from .drawable_layers import drawable_layers
from .utils import (
    convert_wire_order,
    cwire_connections,
    default_bit_map,
    transform_deferred_measurements_tape,
    unwrap_controls,
)


@dataclass
class _Config:
    """Dataclass containing attributes needed for updating the strings to be drawn for each layer"""

    wire_map: dict
    """Map between wire labels and their place in order"""

    bit_map: dict
    """Map between mid-circuit measurements and their corresponding bit in order"""

    cur_layer: Optional[int] = None
    """Current layer index that is being updated"""

    cwire_layers: Optional[list] = None
    """A list of layers used (mid measure or conditional) for each classical wire."""

    decimals: Optional[int] = None
    """Specifies how to round the parameters of operators"""

    cache: Optional[dict] = None
    """dictionary that carries information between label calls in the same drawing"""


def _add_grouping_symbols(op, layer_str, config):  # pylint: disable=unused-argument
    """Adds symbols indicating the extent of a given object."""

    if len(op.wires) > 1:
        mapped_wires = [config.wire_map[w] for w in op.wires]
        min_w, max_w = min(mapped_wires), max(mapped_wires)
        layer_str[min_w] = "╭"
        layer_str[max_w] = "╰"

        for w in range(min_w + 1, max_w):
            layer_str[w] = "├" if w in mapped_wires else "│"

    return layer_str


def _add_cond_grouping_symbols(op, layer_str, config):
    """Adds symbols indicating the extent of a given object for conditional
    operators"""
    n_wires = len(config.wire_map)

    mapped_wires = [config.wire_map[w] for w in op.wires]
    mapped_bits = [config.bit_map[m] for m in op.meas_val.measurements]
    max_w = max(mapped_wires)
    max_b = max(mapped_bits) + n_wires

    ctrl_symbol = "╩" if config.cur_layer != config.cwire_layers[max(mapped_bits)][-1] else "╝"
    layer_str[max_b] = f"═{ctrl_symbol}"

    for w in range(max_w + 1, max(config.wire_map.values()) + 1):
        layer_str[w] = "─║"

    for b in range(n_wires, max_b):
        if b - n_wires in mapped_bits:
            intersection = "╣" if config.cur_layer == config.cwire_layers[b - n_wires][-1] else "╬"
            layer_str[b] = f"═{intersection}"
        else:
            filler = " " if layer_str[b][-1] == " " else "═"
            layer_str[b] = f"{filler}║"

    return layer_str


def _add_mid_measure_grouping_symbols(op, layer_str, config):
    """Adds symbols indicating the extent of a given object for mid-measure
    operators"""
    if op not in config.bit_map:
        return layer_str

    n_wires = len(config.wire_map)
    mapped_wire = config.wire_map[op.wires[0]]
    bit = config.bit_map[op] + n_wires
    layer_str[bit] += " ╚"

    for w in range(mapped_wire + 1, n_wires):
        layer_str[w] += "─║"

    for b in range(n_wires, bit):
        filler = " " if layer_str[b][-1] == " " else "═"
        layer_str[b] += f"{filler}║"

    return layer_str


def _add_op(op, layer_str, config):
    """Updates ``layer_str`` with ``op`` operation."""
    if isinstance(op, qml.ops.Conditional):  # pylint: disable=no-member
        layer_str = _add_cond_grouping_symbols(op, layer_str, config)
        return _add_op(op.then_op, layer_str, config)

    if isinstance(op, MidMeasureMP):
        return _add_mid_measure_op(op, layer_str, config)

    layer_str = _add_grouping_symbols(op, layer_str, config)

    control_wires, control_values = unwrap_controls(op)

    if control_values:
        for w, val in zip(control_wires, control_values):
            layer_str[config.wire_map[w]] += "●" if val else "○"
    else:
        for w in control_wires:
            layer_str[config.wire_map[w]] += "●"

    label = op.label(decimals=config.decimals, cache=config.cache).replace("\n", "")
    if len(op.wires) == 0:  # operation (e.g. barrier, snapshot) across all wires
        n_wires = len(config.wire_map)
        for i, s in enumerate(layer_str[:n_wires]):
            layer_str[i] = s + label
    else:
        for w in op.wires:
            if w not in control_wires:
                layer_str[config.wire_map[w]] += label

    return layer_str


def _add_mid_measure_op(op, layer_str, config):
    """Updates ``layer_str`` with ``op`` operation when ``op`` is a
    ``qml.measurements.MidMeasureMP``."""
    layer_str = _add_mid_measure_grouping_symbols(op, layer_str, config)
    label = op.label(decimals=config.decimals, cache=config.cache).replace("\n", "")

    for w in op.wires:
        layer_str[config.wire_map[w]] += label

    return layer_str


measurement_label_map = {
    Expectation: lambda label: f"<{label}>",
    Probability: lambda label: f"Probs[{label}]" if label else "Probs",
    Sample: lambda label: f"Sample[{label}]" if label else "Sample",
    Counts: lambda label: f"Counts[{label}]" if label else "Counts",
    Variance: lambda label: f"Var[{label}]",
    State: lambda label: "State",
}


def _add_cwire_measurement_grouping_symbols(mcms, layer_str, config):
    """Adds symbols indicating the extent of a given object for mid-circuit measurement
    statistics."""
    if len(mcms) > 1:
        n_wires = len(config.wire_map)
        mapped_bits = [config.bit_map[m] for m in mcms]
        min_b, max_b = min(mapped_bits) + n_wires, max(mapped_bits) + n_wires

        layer_str[min_b] = "╭"
        layer_str[max_b] = "╰"

        for b in range(min_b + 1, max_b):
            layer_str[b] = "├" if b - n_wires in mapped_bits else "│"

    return layer_str


def _add_cwire_measurement(m, layer_str, config):
    """Updates ``layer_str`` with the ``m`` measurement when it is used
    for collecting mid-circuit measurement statistics."""
    mcms = [v.measurements[0] for v in m.mv] if isinstance(m.mv, list) else m.mv.measurements
    layer_str = _add_cwire_measurement_grouping_symbols(mcms, layer_str, config)

    mv_label = "MCM"
    meas_label = measurement_label_map[m.return_type](mv_label)

    n_wires = len(config.wire_map)
    for mcm in mcms:
        ind = config.bit_map[mcm] + n_wires
        layer_str[ind] += meas_label

    return layer_str


def _add_measurement(m, layer_str, config):
    """Updates ``layer_str`` with the ``m`` measurement."""
    if m.mv is not None:
        return _add_cwire_measurement(m, layer_str, config)

    layer_str = _add_grouping_symbols(m, layer_str, config)

    if m.obs is None:
        obs_label = None
    else:
        obs_label = m.obs.label(decimals=config.decimals, cache=config.cache).replace("\n", "")
    if m.return_type in measurement_label_map:
        meas_label = measurement_label_map[m.return_type](obs_label)
    else:
        meas_label = m.return_type.value

    if len(m.wires) == 0:  # state or probability across all wires
        n_wires = len(config.wire_map)
        for i, s in enumerate(layer_str[:n_wires]):
            layer_str[i] = s + meas_label

    for w in m.wires:
        layer_str[config.wire_map[w]] += meas_label
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
            qml.expval(qml.Z("aux")),
            qml.var(qml.Z(0) @ qml.Z(1)),
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
                qml.X(0)

        cache = {'tape_offset': 3}
        print(qml.drawer.tape_text(tape, cache=cache))
        print("New tape offset: ", cache['tape_offset'])


    .. code-block:: none

        0: ──Tape:3─┤

        Tape:3
        0: ──X─┤
        New tape offset:  4

    """
    tape = transform_deferred_measurements_tape(tape)
    cache = cache or {}
    cache.setdefault("tape_offset", 0)
    cache.setdefault("matrices", [])
    tape_cache = []

    wire_map = convert_wire_order(tape, wire_order=wire_order, show_all_wires=show_all_wires)
    bit_map = default_bit_map(tape)
    n_wires = len(wire_map)
    n_bits = len(bit_map)
    if n_wires == 0:
        return ""

    # Used to store lines that are hitting the maximum length
    finished_lines = []

    layers = drawable_layers(tape.operations, wire_map=wire_map, bit_map=bit_map)
    final_operations_layer = len(layers) - 1
    layers += drawable_layers(tape.measurements, wire_map=wire_map, bit_map=bit_map)

    # Update bit map and collect information about connections between mid-circuit measurements,
    # classical conditions, and terminal measurements for processing mid-circuit measurements.
    cwire_layers, _ = cwire_connections(layers, bit_map)

    wire_totals = [f"{wire}: " for wire in wire_map]
    bit_totals = ["" for _ in range(n_bits)]
    line_length = max(len(s) for s in wire_totals)

    wire_totals = [s.rjust(line_length, " ") for s in wire_totals]
    bit_totals = [s.rjust(line_length, " ") for s in bit_totals]

    # Collect information needed for drawing layers
    config = _Config(
        wire_map=wire_map,
        bit_map=bit_map,
        cur_layer=-1,
        cwire_layers=cwire_layers,
        decimals=decimals,
        cache=cache,
    )

    for i, layer in enumerate(layers):
        # Update fillers and helper function
        w_filler = "─" if i <= final_operations_layer else " "
        b_filler = "═" if i <= final_operations_layer else " "
        add_fn = _add_op if i <= final_operations_layer else _add_measurement

        # Create initial strings for the current layer using wire and cwire fillers
        layer_str = [w_filler] * n_wires + [" "] * n_bits
        for b in bit_map.values():
            cur_b_filler = b_filler if min(cwire_layers[b]) < i < max(cwire_layers[b]) else " "
            layer_str[b + n_wires] = cur_b_filler

        config.cur_layer = i

        ##########################################
        # Update current layer strings with labels
        ##########################################
        for op in layer:
            if isinstance(op, qml.tape.QuantumScript):
                layer_str = _add_grouping_symbols(op, layer_str, config)
                label = f"Tape:{cache['tape_offset']+len(tape_cache)}"
                for w in op.wires:
                    layer_str[wire_map[w]] += label
                tape_cache.append(op)
            else:
                layer_str = add_fn(op, layer_str, config)

        #################################################
        # Left justify layer strings and pad on the right
        #################################################
        # Adjust width for wire filler on unused wires
        max_label_len = max(len(s) for s in layer_str)
        for w in range(n_wires):
            layer_str[w] = layer_str[w].ljust(max_label_len, w_filler)

        # Adjust width for bit filler on unused bits
        for b in range(n_bits):
            cur_b_filler = b_filler if cwire_layers[b][0] <= i < cwire_layers[b][-1] else " "
            layer_str[b + n_wires] = layer_str[b + n_wires].ljust(max_label_len, cur_b_filler)

        line_length += max_label_len + 1  # one for the filler character

        ##################
        # Create new lines
        ##################
        if line_length > max_length:
            # move totals into finished_lines and reset totals
            finished_lines += wire_totals + bit_totals
            finished_lines[-1] += "\n"
            wire_totals = [w_filler] * n_wires

            # Bit totals for new lines for warped drawings need to be consistent with the
            # current bit filler
            bit_totals = []
            for b in range(n_bits):
                cur_b_filler = b_filler if cwire_layers[b][0] < i <= cwire_layers[b][-1] else " "
                bit_totals.append(cur_b_filler)

            line_length = 2 + max_label_len

        ###################################################
        # Join current layer with lines for previous layers
        ###################################################
        # Joining is done by adding a filler at the end of the previous layer
        wire_totals = [w_filler.join([t, s]) for t, s in zip(wire_totals, layer_str[:n_wires])]

        for j, (bt, s) in enumerate(zip(bit_totals, layer_str[n_wires : n_wires + n_bits])):
            cur_b_filler = b_filler if cwire_layers[j][0] < i <= cwire_layers[j][-1] else " "
            bit_totals[j] = cur_b_filler.join([bt, s])

        ################################################
        # Add ender characters to final operations layer
        ################################################
        if i == final_operations_layer:
            wire_totals = [f"{s}─┤" for s in wire_totals]
            for b in range(n_bits):
                if cwire_layers[b][-1] > final_operations_layer:
                    bit_totals[b] += "═╡"
                else:
                    bit_totals[b] += "  "

            line_length += 2

    # Recursively handle nested tapes #
    tape_totals = "\n".join(finished_lines + wire_totals + bit_totals)
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
