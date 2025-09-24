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
This module a function for generating matplotlib images from a tape.

Developer note: when making changes to this file, you can run
`pennylane/doc/_static/tape_mpl/tape_mpl_examples.py` to generate docstring
images.  If you change the docstring examples, please update this file.
"""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence
from functools import singledispatch
from typing import TYPE_CHECKING

from pennylane import ops
from pennylane.measurements import MidMeasureMP

from .drawable_layers import drawable_layers
from .mpldrawer import MPLDrawer
from .style import _set_style
from .utils import (
    convert_wire_order,
    cwire_connections,
    default_bit_map,
    transform_deferred_measurements_tape,
    unwrap_controls,
)

if TYPE_CHECKING:
    from pennylane.operation import Operator
    from pennylane.tape import QuantumScript


has_mpl = True
try:
    import matplotlib as mpl
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    has_mpl = False

_Config = namedtuple(
    "_Config", ("decimals", "active_wire_notches", "bit_map", "cwire_layers", "cwire_wires")
)


@singledispatch
def _add_operation_to_drawer(op: Operator, drawer: MPLDrawer, layer: int, config: _Config) -> None:
    """Adds the ``op`` to an ``MPLDrawer`` at the designated location.

    Args:
        op (.Operator): An operator to add to the drawer
        drawer (.MPLDrawer): A matplotlib drawer
        layer (int): The layer to place the operator in.
        config (_Config): named tuple containing ``wire_map``, ``decimals``, and ``active_wire_notches``.

    Side Effects:
        Adds a depiction of ``op`` to ``drawer``

    """
    op_control_wires, control_values, base = unwrap_controls(op)
    is_global_op = isinstance(base, (ops.GlobalPhase, ops.Identity))
    if len(op.wires) == 0 or is_global_op:
        op_wires = list(range(drawer.n_wires))
    else:
        op_wires = op.wires
    target_wires = [w for w in op_wires if w not in op_control_wires]
    if is_global_op and len(target_wires) == 0:
        raise ValueError("Can't draw controlled global gate with unknown non-control wires.")

    if control_values is None:
        control_values = [True for _ in op_control_wires]

    if op_control_wires:
        drawer.ctrl(
            layer,
            op_control_wires,
            wires_target=target_wires,
            control_values=control_values,
        )
    drawer.box_gate(
        layer,
        target_wires,
        op.label(decimals=config.decimals),
        box_options={"zorder": 4},  # make sure box and text above control wires if controlled
        text_options={"zorder": 5},
        active_wire_notches=config.active_wire_notches,
    )


@_add_operation_to_drawer.register
def _(op: ops.SWAP, drawer, layer, _) -> None:
    drawer.SWAP(layer, list(op.wires))


@_add_operation_to_drawer.register
def _(op: ops.CSWAP, drawer, layer, _):
    drawer.ctrl(layer, wires=op.wires[0], wires_target=op.wires[1:])
    drawer.SWAP(layer, wires=list(op.wires[1:]))


@_add_operation_to_drawer.register
def _(op: ops.CNOT, drawer, layer, _):
    drawer.CNOT(layer, op.wires)


@_add_operation_to_drawer.register
def _(op: ops.Toffoli, drawer, layer, _):
    drawer.CNOT(layer, op.wires)


@_add_operation_to_drawer.register
def _(op: ops.MultiControlledX, drawer, layer, _):
    drawer.CNOT(layer, op.wires, control_values=op.control_values)


@_add_operation_to_drawer.register
def _(op: ops.CZ, drawer, layer, _):
    drawer.ctrl(layer, op.wires)


@_add_operation_to_drawer.register
def _(op: ops.CCZ, drawer, layer, _):
    drawer.ctrl(layer, op.wires)


@_add_operation_to_drawer.register
def _(op: ops.Barrier, drawer, layer, _):
    mapped_wires = op.wires if len(op.wires) != 0 else list(range(drawer.n_wires))
    ymin = min(mapped_wires) - 0.5
    ymax = max(mapped_wires) + 0.5
    # by default, uses rcParams['lines.color'] at time when displayed, not at time when added to figure
    # so we have to force it to use the value at the time the line was added to the figure
    drawer.ax.vlines(layer - 0.05, ymin=ymin, ymax=ymax, color=mpl.pyplot.rcParams["lines.color"])
    drawer.ax.vlines(layer + 0.05, ymin=ymin, ymax=ymax, color=mpl.pyplot.rcParams["lines.color"])


@_add_operation_to_drawer.register
def _(op: ops.WireCut, drawer, layer, _):
    ymin = min(op.wires) - 0.5
    ymax = max(op.wires) + 0.5
    drawer.ax.text(layer - 0.35, y=max(op.wires), s="✂", fontsize=40)
    drawer.ax.vlines(layer, ymin=ymin, ymax=ymax, linestyle="--")


@_add_operation_to_drawer.register
def _(op: MidMeasureMP, drawer, layer, _):
    text = None if op.postselect is None else str(int(op.postselect))
    drawer.measure(layer, op.wires[0], text=text)  # assume one wire

    if op.reset:
        drawer.erase_wire(layer, op.wires[0], 1)
        drawer.box_gate(
            layer + 1,
            op.wires[0],
            "|0⟩",
            box_options={"zorder": 4},
            text_options={"zorder": 5},
        )


@_add_operation_to_drawer.register
def _(op: ops.Conditional, drawer, layer, config) -> None:
    drawer.box_gate(
        layer,
        list(op.wires),
        op.base.label(decimals=config.decimals),
        box_options={"zorder": 4},
        text_options={"zorder": 5},
    )
    sorted_bits = sorted([config.bit_map[m] for m in op.meas_val.measurements])
    for b in sorted_bits[:-1]:
        erase_right = layer < config.cwire_layers[b][-1][-1]
        drawer.cwire_join(layer, b + drawer.n_wires, erase_right=erase_right)


def _get_measured_wires(measurements, wires) -> set:
    measured_wires = set()
    for m in measurements:
        if m.mv is None:
            # state and probs
            if len(m.wires) == 0:
                return wires

            for wire in m.wires:
                measured_wires.add(wire)
    return measured_wires


def _add_classical_wires(drawer, cwire_layers, cwire_wires):

    for cwire, layer_ids_per_cwire in cwire_layers.items():
        for layer_ids, layer_wires in zip(layer_ids_per_cwire, cwire_wires[cwire], strict=True):
            xs, ys = [], []

            len_diff = len(layer_ids) - len(layer_wires)
            if len_diff > 0:
                layer_wires += [cwire + drawer.n_wires] * len_diff
            for l, w in zip(layer_ids, layer_wires, strict=True):
                xs.extend([l, l, l])
                ys.extend([cwire + drawer.n_wires, w, cwire + drawer.n_wires])

            drawer.classical_wire(xs, ys)


def _get_measured_bits(measurements, bit_map, offset):
    measured_bits = []
    for m in measurements:
        if isinstance(m.mv, list):
            for mv in m.mv:
                measured_bits += [bit_map[mcm] + offset for mcm in mv.measurements]
        elif m.mv is not None:
            measured_bits += [bit_map[mcm] + offset for mcm in m.mv.measurements]
    return measured_bits


def _draw_layers(layers, measurements, wire_map, config, starting_dots=False, **kwargs):
    """Private function drawing the given layers."""
    n_layers = len(layers)
    n_wires = len(wire_map)

    drawer = MPLDrawer(
        n_layers=n_layers,
        wire_map=wire_map,
        c_wires=len(config.bit_map),
        wire_options=kwargs.get("wire_options", None),
        fig=kwargs.get("fig", None),
        starting_dots=starting_dots,
    )

    if n_wires == 0:
        return drawer.fig, drawer.ax

    fontsize = kwargs.get("fontsize", None)
    if fontsize is not None:
        drawer.fontsize = fontsize

    if kwargs.get("show_wire_labels", True):
        drawer.label(list(wire_map), text_options=kwargs.get("label_options", None))
    else:
        drawer.crop_wire_labels()

    _add_classical_wires(drawer, config.cwire_layers, config.cwire_wires)

    for layer, layer_ops in enumerate(layers):
        for op in layer_ops:
            _add_operation_to_drawer(op, drawer, layer, config)

    if measurements == "dots":
        for wire in wire_map.values():
            drawer.ax.text(
                n_layers + 0.25,
                wire,
                s="···",
                ha="center",
                va="center_baseline",
                fontsize=21,
            )
    else:
        for wire in _get_measured_wires(measurements, list(range(n_wires))):
            drawer.measure(n_layers, wire)

        measured_bits = _get_measured_bits(measurements, config.bit_map, drawer.n_wires)
        if measured_bits:
            drawer.measure(n_layers, measured_bits)

    return drawer.fig, drawer.ax


def _tape_mpl(tape, wire_order=None, show_all_wires=False, max_length=None, **kwargs):
    """Internal implementation for matplotlib circuit visualization.

    Handles the core drawing logic while the public tape_mpl wrapper manages styling.

    Returns:
        When max_length is None: tuple(Figure, Axes) for the complete circuit
        Otherwise: list[tuple(Figure, Axes)] with one pair per circuit segment
    """
    full_wire_map, used_wire_map = convert_wire_order(
        tape, wire_order=wire_order, show_all_wires=show_all_wires
    )
    tape = transform_deferred_measurements_tape(tape)
    tape = ops.functions.map_wires(tape, wire_map=full_wire_map)[0][0]
    bit_map = default_bit_map(tape)

    layers = drawable_layers(tape.operations, wire_map={i: i for i in tape.wires}, bit_map=bit_map)

    for i, layer in enumerate(layers):
        if any(isinstance(o, MidMeasureMP) and o.reset for o in layer):
            layers.insert(i + 1, [])

    bit_map, cwire_layers, cwire_wires = cwire_connections(layers + [tape.measurements], bit_map)

    config = _Config(
        decimals=kwargs.get("decimals", None),
        active_wire_notches=kwargs.get("active_wire_notches", True),
        bit_map=bit_map,
        cwire_layers=cwire_layers,
        cwire_wires=cwire_wires,
    )

    if max_length is None:
        return _draw_layers(
            layers, tape.measurements, config=config, wire_map=used_wire_map, **kwargs
        )

    layer_count = len(layers)
    return [
        _draw_layers(
            layers=layers[i : i + max_length],
            measurements=tape.measurements if i + max_length >= layer_count else "dots",
            config=config,
            wire_map=used_wire_map,
            starting_dots=i > 0,
            **kwargs,
        )
        for i in range(0, layer_count, max_length)
    ]


# pylint: disable=too-many-arguments
def tape_mpl(
    tape: QuantumScript,
    wire_order: Sequence | None = None,
    show_all_wires: bool = False,
    decimals: int | None = None,
    style: str | None = None,
    *,
    fig: mpl.figure.Figure | None = None,
    max_length: int | None = None,
    **kwargs,
):
    """Produces matplotlib graphic objects (``fig`` and ``ax``) from a tape.

    Args:
        tape (QuantumTape): the operations and measurements to draw

    Keyword Args:
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        decimals (int): How many decimal points to include when formatting operation parameters.
            Default ``None`` will omit parameters from operation labels.
        style (str): visual style of plot. Valid strings are ``{'black_white', 'black_white_dark', 'sketch',
            'sketch_dark', 'solarized_light', 'solarized_dark', 'default'}``. If no style is specified, the
            global style set with :func:`~.use_style` will be used, and the initial default is 'black_white'.
            If you would like to use your environment's current rcParams, set `style` to "rcParams".
            Setting style does not modify matplotlib global plotting settings.
        fontsize (float or str): fontsize for text. Valid strings are
            ``{'xx-small', 'x-small', 'small', 'medium', large', 'x-large', 'xx-large'}``.
            Default is ``14``.
        wire_options (dict): matplotlib formatting options for the wire lines. In addition to
            standard options, options per wire can be specified with ``wire_label: options``
            pairs, also see examples below.
        label_options (dict): matplotlib formatting options for the wire labels
        show_wire_labels (bool): Whether or not to show the wire labels.
        active_wire_notches (bool): whether or not to add notches indicating active wires.
            Defaults to ``True``.
        fig (None or matplotlib Figure): Matplotlib figure to plot onto. If None, then create a new figure.
        max_length (Optional[int]): When there is more than ``max_length`` layers, additional plots
            will be produced with at most ``max_length`` individual layers.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes._axes.Axes: The key elements for matplotlib's object oriented interface.
        If ``max_length`` is less than the required number of layers, a list of tuples containing the figures
        and axes will be returned instead.

    **Example:**

    .. code-block:: python

        ops = [
            qml.QFT(wires=(0,1,2,3)),
            qml.IsingXX(1.234, wires=(0,2)),
            qml.Toffoli(wires=(0,1,2)),
            qml.CSWAP(wires=(0,2,3)),
            qml.RX(1.2345, wires=0),
            qml.CRZ(1.2345, wires=(3,0))
        ]
        measurements = [qml.expval(qml.Z(0))]
        tape = qml.tape.QuantumTape(ops, measurements)

        fig, ax = qml.drawer.tape_mpl(tape)
        fig.show()

    .. figure:: ../../_static/tape_mpl/default.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    .. details::
        :title: Usage Details

    **Decimals:**

    The keyword ``decimals`` controls how many decimal points to include when labelling the operations.
    The default value ``None`` omits parameters for brevity.

    .. code-block:: python

        ops = [qml.RX(1.23456, wires=0), qml.Rot(1.2345,2.3456, 3.456, wires=0)]
        measurements = [qml.expval(qml.Z(0))]
        tape2 = qml.tape.QuantumTape(ops, measurements)

        fig, ax = qml.drawer.tape_mpl(tape2, decimals=2)

    .. figure:: ../../_static/tape_mpl/decimals.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    **Wires:**

    The keywords ``wire_order`` and ``show_all_wires`` control the location of wires from top to bottom.

    .. code-block:: python

        fig, ax = qml.drawer.tape_mpl(tape, wire_order=[3,2,1,0])

    .. figure:: ../../_static/tape_mpl/wire_order.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    If a wire is in ``wire_order``, but not in the ``tape``, it will be omitted by default.  Only by selecting
    ``show_all_wires=True`` will empty wires be diplayed.

    .. code-block:: python

        fig, ax = qml.drawer.tape_mpl(tape, wire_order=["aux"], show_all_wires=True)

    .. figure:: ../../_static/tape_mpl/show_all_wires.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    **Integration with matplotlib:**

    This function returns matplotlib figure and axes objects. Using these objects,
    users can perform further customization of the graphic.

    .. code-block:: python

        import matplotlib.pyplot as plt

        fig, ax = qml.drawer.tape_mpl(tape)
        fig.suptitle("My Circuit", fontsize="xx-large")

        options = {'facecolor': "white", 'edgecolor': "#f57e7e", "linewidth": 6, "zorder": -1}
        box1 = plt.Rectangle((-0.5, -0.5), width=3.0, height=4.0, **options)
        ax.add_patch(box1)

        ax.annotate("CSWAP", xy=(3, 2.5), xycoords='data', xytext=(3.8,1.5), textcoords='data',
                    arrowprops={'facecolor': 'black'}, fontsize=14)

    .. figure:: ../../_static/tape_mpl/postprocessing.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    **Formatting:**

    PennyLane has inbuilt styles for controlling the appearance of the circuit drawings.
    All available styles can be determined by evaluating ``qml.drawer.available_styles()``.
    Any available string can then be passed via the kwarg ``style`` to change the settings for
    that plot. This will not affect style settings for subsequent matplotlib plots.

    .. code-block:: python

        fig, ax = qml.drawer.tape_mpl(tape, style='sketch')

    .. figure:: ../../_static/tape_mpl/sketch_style.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    You can also control the appearance with matplotlib's provided tools, see the
    `matplotlib docs <https://matplotlib.org/stable/tutorials/introductory/customizing.html>`_ .
    For example, we can customize ``plt.rcParams``. To use a customized appearance based on matplotlib's
    ``plt.rcParams``, ``qml.drawer.tape_mpl`` must be run with ``style="rcParams"``:

    .. code-block:: python

        plt.rcParams['patch.facecolor'] = 'mistyrose'
        plt.rcParams['patch.edgecolor'] = 'maroon'
        plt.rcParams['text.color'] = 'maroon'
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['patch.linewidth'] = 4
        plt.rcParams['patch.force_edgecolor'] = True
        plt.rcParams['lines.color'] = 'indigo'
        plt.rcParams['lines.linewidth'] = 5
        plt.rcParams['figure.facecolor'] = 'ghostwhite'

        fig, ax = qml.drawer.tape_mpl(tape, style="rcParams")

    .. figure:: ../../_static/tape_mpl/rcparams.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    The wires and wire labels can be manually formatted by passing in dictionaries of
    keyword-value pairs of matplotlib options. ``wire_options`` accepts options for lines,
    and ``label_options`` accepts text options.

    .. code-block:: python

        fig, ax = qml.drawer.tape_mpl(tape, wire_options={'color':'teal', 'linewidth': 5},
                    label_options={'size': 20})

    .. figure:: ../../_static/tape_mpl/wires_labels.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    Additionally, ``wire_options`` may contain sub-dictionaries of matplotlib options assigned
    to separate wire labels, which will control the line style for the respective individual wires.

    .. code-block:: python

        wire_options = {
            'color': 'teal', # all wires but wire 2 will be teal
            'linewidth': 5, # all wires but wire 2 will be bold
            2: {'color': 'orange', 'linestyle': '--'}, # wire 2 will be orange and dashed
        }
        fig, ax = qml.drawer.tape_mpl(tape, wire_options=wire_options)

    .. figure:: ../../_static/tape_mpl/per_wire_options.png
            :align: center
            :width: 60%
            :target: javascript:void(0);
    """

    restore_params = {}
    if update_style := (has_mpl and style != "rcParams"):
        restore_params = mpl.rcParams.copy()
        _set_style(style)

    try:
        return _tape_mpl(
            tape,
            wire_order=wire_order,
            show_all_wires=show_all_wires,
            decimals=decimals,
            fig=fig,
            max_length=max_length,
            **kwargs,
        )
    finally:
        if update_style:
            # we don't want to mess with how it modifies whether the interface is interactive
            # but we want to restore everything else
            restore_params["interactive"] = mpl.rcParams["interactive"]
            mpl.rcParams.update(restore_params)
