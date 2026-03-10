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


from dataclasses import dataclass, field

from ._add_obj import _add_obj
from .drawable_layers import drawable_layers
from .utils import (
    convert_wire_order,
    cwire_connections,
    default_bit_map,
    transform_deferred_measurements_tape,
)


@dataclass
class _CurrentTotals:
    finished_lines: list[str]
    wire_totals: list[str]
    bit_totals: list[str]


@dataclass
class _Config:
    """Dataclass containing attributes needed for updating the strings to be drawn for each layer"""

    wire_map: dict
    """Map between wire labels and their place in order"""

    bit_map: dict
    """Map between mid-circuit measurements and their corresponding bit in order"""

    num_op_layers: int

    cur_layer: int = -1
    """Current layer index that is being updated"""

    cwire_layers: list = field(default_factory=list)
    """A list of layers used (mid measure or conditional) for each classical wire."""

    decimals: int | None = None
    """Specifies how to round the parameters of operators"""

    cache: dict | None = None
    """dictionary that carries information between label calls in the same drawing"""

    @property
    def wire_filler(self) -> str:
        """The filler character for wires at the current layer."""
        return "─" if self.cur_layer < self.num_op_layers else " "

    def bit_filler(self, bit, next_layer: bool = False) -> str:
        """The filler character for bits at the current layer and the designated bit."""
        layer = self.cur_layer + 1 if next_layer else self.cur_layer
        if self.cur_layer >= self.num_op_layers:
            return " "
        for layer_stretch in self.cwire_layers[bit]:
            if layer_stretch[0] < layer <= layer_stretch[-1]:
                return "═"
        return " "

    @property
    def n_bits(self) -> int:
        """The number of bits."""
        return len(set(self.bit_map.values()))

    @property
    def n_wires(self) -> int:
        """The number of wires."""
        return len(self.wire_map)


def _initialize_wire_and_bit_totals(
    config: _Config, show_wire_labels: bool, continuation: bool = False
) -> tuple[list[str], list[str]]:
    """Initialize the wire totals and bit_totals with the required wire labels."""

    prefix = "··· " if continuation else ""

    if show_wire_labels:
        wire_totals = [f"{wire}: " + prefix for wire in config.wire_map]
        line_length = max(len(s) for s in wire_totals)
        wire_totals = [s.rjust(line_length, " ") for s in wire_totals]
        bit_totals = [" " * line_length] * config.n_bits
    else:
        wire_totals = [prefix] * config.n_wires
        bit_totals = [prefix] * config.n_bits

    return wire_totals, bit_totals


def _initialize_layer_str(config: _Config) -> list[str]:
    """Initialize the list of strings for a new layer.

    For example, if we have three wires and two classical wires, we will get:

    .. code-block::

        ['─', '─', '─', ' ', ' ']

    """
    return [config.wire_filler] * config.n_wires + [
        config.bit_filler(b) for b in range(config.n_bits)
    ]


def _left_justify(layer_str: list[str], config: _Config) -> list[str]:
    """Add filler characters to layer_str so that everything has the same length.

    If we initialize with:

    .. code-block::

        ['─Rot', '─', '─', ' ']

    We will get out:

    .. code-block::

        ['─Rot', '────', '────', '    ']

    where every entry in the layer now has the same length.

    """
    max_label_len = max(len(s) for s in layer_str)

    for w in range(config.n_wires):
        layer_str[w] = layer_str[w].ljust(max_label_len, config.wire_filler)

    # Adjust width for bit filler on unused bits
    for b in range(config.n_bits):
        # needs filler for next layer, as adding to the right of this one
        cur_b_filler = config.bit_filler(b, next_layer=True)
        layer_str[b + config.n_wires] = layer_str[b + config.n_wires].ljust(
            max_label_len, cur_b_filler
        )

    return layer_str


def _add_to_finished_lines(
    totals: _CurrentTotals, config: _Config, show_wire_labels: bool
) -> _CurrentTotals:
    """Add current totals to the finished lines and initialize new totals."""

    suffix = " ···"

    totals.finished_lines += [line + suffix for line in totals.wire_totals]
    totals.finished_lines += totals.bit_totals
    totals.finished_lines[-1] += "\n"

    # Reset wire and bit totals. Bit totals for new lines for warped drawings
    # need to be consistent with the current bit filler
    totals.wire_totals, totals.bit_totals = _initialize_wire_and_bit_totals(
        config, show_wire_labels, continuation=True
    )

    return totals


def _add_layer_str_to_totals(totals: _CurrentTotals, layer_str, config) -> _CurrentTotals:
    """Combine the current layer's string representation with the accumulated circuit representation.

    This function joins each wire and bit string in the current layer with the corresponding
    accumulated string in totals, using appropriate filler characters.

    Args:
        totals: Object containing the current state of the circuit representation
        layer_str: List of strings representing the current layer to be added
        config: Configuration object with drawing settings and current state

    Returns:
        Updated totals with the current layer added
    """
    # Process quantum wires - join accumulated wire strings with current layer strings
    totals.wire_totals = [
        config.wire_filler.join([t, s])
        for t, s in zip(totals.wire_totals, layer_str[: config.n_wires])
    ]

    # Process classical bits - join accumulated bit strings with current layer strings
    for j, (bt, s) in enumerate(
        zip(totals.bit_totals, layer_str[config.n_wires : config.n_wires + config.n_bits])
    ):
        totals.bit_totals[j] = config.bit_filler(j).join([bt, s])

    return totals


def _finalize_layers(totals: _CurrentTotals, config: _Config) -> _CurrentTotals:
    """Add ending characters to separate the operation layers from the measurement layers"""
    totals.wire_totals = [f"{s}─┤" for s in totals.wire_totals]
    for b in range(config.n_bits):
        if config.cwire_layers[b][-1][-1] >= config.num_op_layers:
            totals.bit_totals[b] += "═╡"
        else:
            totals.bit_totals[b] += "  "

    return totals


# pylint: disable=too-many-arguments
def tape_text(
    tape,
    wire_order=None,
    *,
    show_all_wires=False,
    decimals=None,
    max_length=100,
    show_matrices=True,
    show_wire_labels=True,
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
        show_wire_labels (bool): Whether or not to show the wire labels.
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
        op = qml.StronglyEntanglingLayers(params, wires=range(5))
        tape2 = qml.tape.QuantumScript(op.decomposition())
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
    {'matrices': [array([[-1., -0., -0.],
           [-0., -1., -0.],
           [-0., -0., -1.]]), array([[1., 0.],
           [0., 1.]]), array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])], 'tape_offset': 0}

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

    _, wire_map = convert_wire_order(tape, wire_order=wire_order, show_all_wires=show_all_wires)
    bit_map = default_bit_map(tape)
    n_wires = len(wire_map)
    if n_wires == 0:
        return ""

    layers = drawable_layers(tape.operations, wire_map=wire_map, bit_map=bit_map)
    num_op_layers = len(layers)
    layers += drawable_layers(tape.measurements, wire_map=wire_map, bit_map=bit_map)
    # Update bit map and collect information about connections between mid-circuit measurements,
    # classical conditions, and terminal measurements for processing mid-circuit measurements.
    bit_map, cwire_layers, _ = cwire_connections(layers, bit_map)
    # Collect information needed for drawing layers
    config = _Config(
        wire_map=wire_map,
        bit_map=bit_map,
        num_op_layers=num_op_layers,
        cwire_layers=cwire_layers,
        decimals=decimals,
        cache=cache,
    )

    totals = _CurrentTotals([], *(_initialize_wire_and_bit_totals(config, show_wire_labels)))
    len_suffix = 4  # Suffix dots at then of a partitioned circuit (' ...') have length 4

    for cur_layer, layer in enumerate(layers):
        config.cur_layer = cur_layer
        layer_str = _initialize_layer_str(config)

        for op in layer:
            layer_str = _add_obj(op, layer_str, config, tape_cache)
        layer_str = _left_justify(layer_str, config)

        cur_max_length = max_length - len_suffix if cur_layer < len(layers) - 1 else max_length
        if len(totals.wire_totals[0]) + len(layer_str[0]) > cur_max_length - 1:
            totals = _add_to_finished_lines(totals, config, show_wire_labels)

        totals = _add_layer_str_to_totals(totals, layer_str, config)

        if config.cur_layer == config.num_op_layers - 1:
            totals = _finalize_layers(totals, config)

    # Recursively handle nested tapes #
    tape_totals = "\n".join(totals.finished_lines + totals.wire_totals + totals.bit_totals)
    current_tape_offset = cache["tape_offset"]
    cache["tape_offset"] += len(tape_cache)
    for i, nested_tape in enumerate(tape_cache):
        label = f"\nTape:{i+current_tape_offset}"
        tape_str = tape_text(
            nested_tape,
            wire_order,
            show_all_wires=show_all_wires,
            decimals=decimals,
            max_length=max_length,
            show_matrices=False,
            cache=cache,
        )
        tape_totals = "\n".join([tape_totals, label, tape_str])

    if show_matrices:
        mat_str = "".join(f"\nM{i} = \n{mat}" for i, mat in enumerate(cache["matrices"]))
        return tape_totals + mat_str

    return tape_totals
