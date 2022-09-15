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

# cant `import pennylane as qml` because of circular imports with circuit graph
from pennylane import ops
from pennylane.wires import Wires
from .mpldrawer import MPLDrawer
from .drawable_layers import drawable_layers
from .utils import convert_wire_order

############################ Special Gate Methods #########################
# If an operation is drawn differently than the standard box/ ctrl+box style
# create a private method here and add it to the ``special_cases`` dictionary
# These methods should accept arguments in the order of ``drawer, layer, mapped_wires, op``

# pylint: disable=unused-argument,no-member
def _add_swap(drawer, layer, mapped_wires, op):
    drawer.SWAP(layer, mapped_wires)


# pylint: disable=unused-argument
def _add_cswap(drawer, layer, mapped_wires, op):
    drawer.ctrl(layer, wires=mapped_wires[0], wires_target=mapped_wires[1:])
    drawer.SWAP(layer, wires=mapped_wires[1:])


# pylint: disable=unused-argument
def _add_cx(drawer, layer, mapped_wires, op):
    drawer.CNOT(layer, mapped_wires)


def _add_multicontrolledx(drawer, layer, mapped_wires, op):
    # convert control values
    control_values = [(i == "1") for i in op.hyperparameters["control_values"]]
    drawer.CNOT(layer, mapped_wires, control_values=control_values)


# pylint: disable=unused-argument
def _add_cz(drawer, layer, mapped_wires, op):
    drawer.ctrl(layer, mapped_wires)


# pylint: disable=unused-argument
def _add_barrier(drawer, layer, mapped_wires, op):
    ymin = min(mapped_wires) - 0.5
    ymax = max(mapped_wires) + 0.5
    drawer.ax.vlines(layer - 0.05, ymin=ymin, ymax=ymax)
    drawer.ax.vlines(layer + 0.05, ymin=ymin, ymax=ymax)


# pylint: disable=unused-argument
def _add_wirecut(drawer, layer, mapped_wires, op):
    ymin = min(mapped_wires) - 0.5
    ymax = max(mapped_wires) + 0.5
    drawer.ax.vlines(layer, ymin=ymin, ymax=ymax, linestyle="--")


special_cases = {
    ops.SWAP: _add_swap,
    ops.CSWAP: _add_cswap,
    ops.CNOT: _add_cx,
    ops.Toffoli: _add_cx,
    ops.MultiControlledX: _add_multicontrolledx,
    ops.CZ: _add_cz,
    ops.Barrier: _add_barrier,
    ops.WireCut: _add_wirecut,
}
"""Dictionary mapping special case classes to functions for drawing them."""

# pylint: disable=too-many-branches
def tape_mpl(tape, wire_order=None, show_all_wires=False, decimals=None, **kwargs):
    """Produces a matplotlib graphic from a tape.

    Args:
        tape (QuantumTape): the operations and measurements to draw

    Keyword Args:
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        decimals (int): How many decimal points to include when formatting operation parameters.
            Default ``None`` will omit parameters from operation labels.
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

        with qml.tape.QuantumTape() as tape:
            qml.QFT(wires=(0,1,2,3))
            qml.IsingXX(1.234, wires=(0,2))
            qml.Toffoli(wires=(0,1,2))
            qml.CSWAP(wires=(0,2,3))
            qml.RX(1.2345, wires=0)
            qml.CRZ(1.2345, wires=(3,0))
            qml.expval(qml.PauliZ(0))

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

        with qml.tape.QuantumTape() as tape2:
            qml.RX(1.23456, wires=0)
            qml.Rot(1.2345,2.3456, 3.456, wires=0)
            qml.expval(qml.PauliZ(0))

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
    Any available string can then be passed to ``qml.drawer.use_style``.

    .. code-block:: python

        qml.drawer.use_style('black_white')
        fig, ax = tape_mpl(tape)

    .. figure:: ../../_static/tape_mpl/black_white_style.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    You can also control the appearance with matplotlib's provided tools, see the
    `matplotlib docs <https://matplotlib.org/stable/tutorials/introductory/customizing.html>`_ .
    For example, we can customize ``plt.rcParams``:

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

        fig, ax = tape_mpl(tape)

    .. figure:: ../../_static/tape_mpl/rcparams.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    The wires and wire labels can be manually formatted by passing in dictionaries of
    keyword-value pairs of matplotlib options. ``wire_options`` accepts options for lines,
    and ``label_options`` accepts text options.

    .. code-block:: python

        fig, ax = tape_mpl(tape, wire_options={'color':'black', 'linewidth': 5},
                    label_options={'size': 20})

    .. figure:: ../../_static/tape_mpl/wires_labels.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    """
    wire_options = kwargs.get("wire_options", None)
    label_options = kwargs.get("label_options", None)
    active_wire_notches = kwargs.get("active_wire_notches", True)
    fontsize = kwargs.get("fontsize", None)

    wire_map = convert_wire_order(tape, wire_order=wire_order, show_all_wires=show_all_wires)

    layers = drawable_layers(tape.operations, wire_map=wire_map)

    n_layers = len(layers)
    n_wires = len(wire_map)

    drawer = MPLDrawer(n_layers=n_layers, n_wires=n_wires, wire_options=wire_options)

    if fontsize is not None:
        drawer.fontsize = fontsize

    drawer.label(list(wire_map), text_options=label_options)

    for layer, layer_ops in enumerate(layers):
        for op in layer_ops:
            specialfunc = special_cases.get(op.__class__, None)
            if specialfunc is not None:
                mapped_wires = [wire_map[w] for w in op.wires]
                specialfunc(drawer, layer, mapped_wires, op)

            else:
                op_control_wires = getattr(op, "control_wires", [])
                control_wires = [wire_map[w] for w in op_control_wires]
                target_wires = [wire_map[w] for w in op.wires if w not in op_control_wires]
                control_values = op.hyperparameters.get("control_values", None)

                if control_values is None:
                    control_values = [True for _ in control_wires]
                elif isinstance(control_values[0], str):
                    control_values = [(i == "1") for i in control_values]

                if len(control_wires) != 0:
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
                    box_options={
                        "zorder": 4
                    },  # make sure box and text above control wires if controlled
                    text_options={"zorder": 5},
                    active_wire_notches=active_wire_notches,
                )

    # store wires we've already drawn on
    # max one measurement symbol per wire
    measured_wires = Wires([])

    for m in tape.measurements:
        # state and probs
        if len(m.wires) == 0:
            for wire in range(n_wires):
                if wire not in measured_wires:
                    drawer.measure(n_layers, wire)
            break

        for wire in m.wires:
            if wire not in measured_wires:
                drawer.measure(n_layers, wire_map[wire])
                measured_wires += wire

    return drawer.fig, drawer.ax
