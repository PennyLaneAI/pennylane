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
from functools import singledispatch

import pennylane as qml
from pennylane import ops
from pennylane.measurements import MidMeasureMP
from pennylane.wires import Wires
from .mpldrawer import MPLDrawer
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls
from .style import _set_style

has_mpl = True
try:
    import matplotlib as mpl
except (ModuleNotFoundError, ImportError) as e:  # pragma: no cover
    has_mpl = False


_measured_layers = {}


@singledispatch
def add_operation_to_drawer(op, drawer, layer, wire_map, decimals, active_wire_notches):
    op_control_wires, control_values = unwrap_controls(op)

    control_wires = [wire_map[w] for w in op_control_wires]
    target_wires = (
        [wire_map[w] for w in op.wires if w not in op_control_wires]
        if len(op.wires) != 0
        else wire_map.values()
    )

    if control_values is None:
        control_values = [True for _ in control_wires]

    if control_wires:
        drawer.ctrl(
            layer,
            control_wires,
            wires_target=target_wires,
            control_values=control_values,
        )
    drawer.box_gate(
        layer,
        target_wires,
        op.label(decimals=decimals),
        box_options={"zorder": 4},  # make sure box and text above control wires if controlled
        text_options={"zorder": 5},
        active_wire_notches=active_wire_notches,
    )


@add_operation_to_drawer.register
def add_SWAP(op: ops.SWAP, drawer, layer, wire_map, decimals, active_wire_notches):
    mapped_wires = [wire_map[w] for w in op.wires] if len(op.wires) != 0 else wire_map.values()
    drawer.SWAP(layer, mapped_wires)


@add_operation_to_drawer.register
def add_CSWAP(op: ops.CSWAP, drawer, layer, wire_map, decimals, active_wire_notches):
    mapped_wires = [wire_map[w] for w in op.wires] if len(op.wires) != 0 else wire_map.values()
    drawer.ctrl(layer, wires=mapped_wires[0], wires_target=mapped_wires[1:])
    drawer.SWAP(layer, wires=mapped_wires[1:])


@add_operation_to_drawer.register
def add_CNOT(op: ops.CNOT, drawer, layer, wire_map, decimals, active_wire_notches):
    mapped_wires = [wire_map[w] for w in op.wires] if len(op.wires) != 0 else wire_map.values()
    drawer.CNOT(layer, mapped_wires)


@add_operation_to_drawer.register
def add_Toffoli(op: ops.Toffoli, drawer, layer, wire_map, decimals, active_wire_notches):
    mapped_wires = [wire_map[w] for w in op.wires] if len(op.wires) != 0 else wire_map.values()
    drawer.CNOT(layer, mapped_wires)


@add_operation_to_drawer.register
def add_MultiControlledX(
    op: ops.MultiControlledX, drawer, layer, wire_map, decimals, active_wire_notches
):
    mapped_wires = [wire_map[w] for w in op.wires] if len(op.wires) != 0 else wire_map.values()
    control_values = [(i == "1") for i in op.hyperparameters["control_values"]]
    drawer.CNOT(layer, mapped_wires, control_values=control_values)


@add_operation_to_drawer.register
def add_CZ(op: ops.CZ, drawer, layer, wire_map, decimals, active_wire_notches):
    mapped_wires = [wire_map[w] for w in op.wires] if len(op.wires) != 0 else wire_map.values()
    drawer.ctrl(layer, mapped_wires)


@add_operation_to_drawer.register
def add_CZ(op: ops.CCZ, drawer, layer, wire_map, decimals, active_wire_notches):
    mapped_wires = [wire_map[w] for w in op.wires] if len(op.wires) != 0 else wire_map.values()
    drawer.ctrl(layer, mapped_wires)


@add_operation_to_drawer.register
def add_barrier(op: ops.Barrier, drawer, layer, wire_map, decimals, active_wire_notches):
    mapped_wires = [wire_map[w] for w in op.wires] if len(op.wires) != 0 else wire_map.values()
    ymin = min(mapped_wires) - 0.5
    ymax = max(mapped_wires) + 0.5
    drawer.ax.vlines(layer - 0.05, ymin=ymin, ymax=ymax)
    drawer.ax.vlines(layer + 0.05, ymin=ymin, ymax=ymax)


@add_operation_to_drawer.register
def add_wirecut(op: ops.WireCut, drawer, layer, wire_map, decimals, active_wire_notches):
    mapped_wires = [wire_map[w] for w in op.wires] if len(op.wires) != 0 else wire_map.values()
    ymin = min(mapped_wires) - 0.5
    ymax = max(mapped_wires) + 0.5
    drawer.ax.text(layer - 0.35, y=max(mapped_wires), s="✂", fontsize=40)
    drawer.ax.vlines(layer, ymin=ymin, ymax=ymax, linestyle="--")


@add_operation_to_drawer.register
def add_mcm(op: MidMeasureMP, drawer, layer, wire_map, decimals, active_wire_notches):
    mapped_wires = [wire_map[w] for w in op.wires] if len(op.wires) != 0 else wire_map.values()
    _measured_layers[op.id] = layer
    text = None if op.postselect is None else str(int(op.postselect))
    drawer.measure(layer, mapped_wires[0], text=text)  # assume one wire

    if op.reset:
        drawer.block_wire(layer, mapped_wires[0], 1)
        drawer.box_gate(
            layer + 1,
            mapped_wires[0],
            "|0⟩",
            box_options={"zorder": 4},
            text_options={"zorder": 5},
        )


@add_operation_to_drawer.register
def add_conditional(
    op: qml.ops.op_math.Conditional, drawer, layer, wire_map, decimals, active_wire_notches
):
    target_wires = [wire_map[w] for w in op.wires]
    drawer.box_gate(
        layer,
        target_wires,
        op.then_op.label(decimals=decimals),
        box_options={"zorder": 4, "linestyle": "dashed"},
        text_options={"zorder": 5},
    )


# pylint: disable=too-many-branches
def _tape_mpl(tape, wire_order=None, show_all_wires=False, decimals=None, **kwargs):
    """Private function wrapped with styling."""
    wire_options = kwargs.get("wire_options", None)
    label_options = kwargs.get("label_options", None)
    active_wire_notches = kwargs.get("active_wire_notches", True)
    fontsize = kwargs.get("fontsize", None)

    wire_map = convert_wire_order(tape, wire_order=wire_order, show_all_wires=show_all_wires)

    layers = drawable_layers(tape.operations, wire_map=wire_map)
    for i, layer in enumerate(layers):
        if any(isinstance(o, qml.measurements.MidMeasureMP) and o.reset for o in layer):
            layers.insert(i + 1, [])

    n_layers = len(layers)
    n_wires = len(wire_map)

    drawer = MPLDrawer(n_layers=n_layers, n_wires=n_wires, wire_options=wire_options)

    if n_wires == 0:
        return drawer.fig, drawer.ax

    if fontsize is not None:
        drawer.fontsize = fontsize

    drawer.label(list(wire_map), text_options=label_options)

    for layer, layer_ops in enumerate(layers):
        for op in layer_ops:
            if type(op).__name__ == "Conditional":
                m_ids = [m.id for m in op.meas_val.measurements]
                measured_layer = {_measured_layers[m_id] for m_id in m_ids}.pop()
                control_wires = [
                    wire_map[o.wires[0]] for o in layers[measured_layer] if o.id in m_ids
                ]
                target_wires = [wire_map[w] for w in op.wires]
                # drawer.cond(layer, measured_layer, control_wires, target_wires)
            add_operation_to_drawer(op, drawer, layer, wire_map, decimals, active_wire_notches)

    measured_wires = set()
    for m in tape.measurements:
        # state and probs
        if len(m.wires) == 0:
            measured_wires = set(tape.wires)
            break

        for wire in m.wires:
            measured_wires += wire

    for wire in measured_wires:
        drawer.measure(n_layers, wire_map[wire])

    return drawer.fig, drawer.ax


def tape_mpl(tape, wire_order=None, show_all_wires=False, decimals=None, style=None, **kwargs):
    """Produces a matplotlib graphic from a tape.

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
        wire_options (dict): matplotlib formatting options for the wire lines
        label_options (dict): matplotlib formatting options for the wire labels
        active_wire_notches (bool): whether or not to add notches indicating active wires.
            Defaults to ``True``.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes._axes.Axes: The key elements for matplotlib's object oriented interface.

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
        measurements = [qml.expval(qml.PauliZ(0))]
        tape = qml.tape.QuantumTape(ops, measurements)

        fig, ax = tape_mpl(tape)
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
        measurements = [qml.expval(qml.PauliZ(0))]
        tape2 = qml.tape.QuantumTape(ops, measurements)

        fig, ax = tape_mpl(tape2, decimals=2)

    .. figure:: ../../_static/tape_mpl/decimals.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    **Wires:**

    The keywords ``wire_order`` and ``show_all_wires`` control the location of wires from top to bottom.

    .. code-block:: python

        fig, ax = tape_mpl(tape, wire_order=[3,2,1,0])

    .. figure:: ../../_static/tape_mpl/wire_order.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    If a wire is in ``wire_order``, but not in the ``tape``, it will be omitted by default.  Only by selecting
    ``show_all_wires=True`` will empty wires be diplayed.

    .. code-block:: python

        fig, ax = tape_mpl(tape, wire_order=["aux"], show_all_wires=True)

    .. figure:: ../../_static/tape_mpl/show_all_wires.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    **Integration with matplotlib:**

    This function returns matplotlib figure and axes objects. Using these objects,
    users can perform further customization of the graphic.

    .. code-block:: python

        fig, ax = tape_mpl(tape)
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

        fig, ax = tape_mpl(tape, style='sketch')

    .. figure:: ../../_static/tape_mpl/sketch_style.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    You can also control the appearance with matplotlib's provided tools, see the
    `matplotlib docs <https://matplotlib.org/stable/tutorials/introductory/customizing.html>`_ .
    For example, we can customize ``plt.rcParams``. To use a customized appearance based on matplotlib's
    ``plt.rcParams``, ``qml.drawer.tape_mpl`` must be run with ``style=None``:

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

        fig, ax = tape_mpl(tape, style=None)

    .. figure:: ../../_static/tape_mpl/rcparams.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    The wires and wire labels can be manually formatted by passing in dictionaries of
    keyword-value pairs of matplotlib options. ``wire_options`` accepts options for lines,
    and ``label_options`` accepts text options.

    .. code-block:: python

        fig, ax = tape_mpl(tape, wire_options={'color':'teal', 'linewidth': 5},
                    label_options={'size': 20})

    .. figure:: ../../_static/tape_mpl/wires_labels.png
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
            tape, wire_order=wire_order, show_all_wires=show_all_wires, decimals=decimals, **kwargs
        )
    finally:
        if update_style:
            # we don't want to mess with how it modifies whether the interface is interactive
            # but we want to restore everything else
            restore_params["interactive"] = mpl.rcParams["interactive"]
            mpl.rcParams.update(restore_params)
