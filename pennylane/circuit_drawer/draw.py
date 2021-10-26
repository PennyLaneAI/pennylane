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


def _add_swap(drawer, op, layer, mapped_wires):
    drawer.SWAP(layer, mapped_wires)


def _add_cswap(drawer, op, layer, mapped_wires):
    drawer.ctrl(layer, wires=mapped_wires[0], wires_target=mapped_wires[1:])
    drawer.SWAP(layer, wires=mapped_wires[1:])


def _add_cx(drawer, op, layer, mapped_wires):
    drawer.CNOT(layer, mapped_wires)


def _add_multicontrolledx(drawer, op, layer, mapped_wires):
    # convert control values
    control_values = [(i == "1") for i in op.control_values]
    drawer.ctrl(layer, mapped_wires[:-1], mapped_wires[-1], control_values=control_values)
    drawer._target_x(layer, mapped_wires[-1])


def _add_cz(drawer, op, layer, mapped_wires):
    drawer.ctrl(layer, mapped_wires)


special_cases = {
    ops.SWAP: _add_swap,
    ops.CSWAP: _add_cswap,
    ops.CNOT: _add_cx,
    ops.Toffoli: _add_cx,
    ops.MultiControlledX: _add_multicontrolledx,
    ops.CZ: _add_cz,
}


def draw_mpl(tape, wire_order=None, show_all_wires=False, decimals=None, **kwargs):
    """Produces a matplotlib graphic from a tape.

    Args:
        tape (QuantumTape): the operations and measurements to draw

    Keyword Args:
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        wire_options (dict): matplotlib formatting options for the wire lines
        label_options (dict): matplotlib formatting options for the wire labels

    Returns:
        matplotlib.figure.Figure, matplotlib.axes._axes.Axes

    **Example:**

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            qml.templates.GroverOperator(wires=(0,1,2,3))
            qml.Toffoli(wires=(0,1,2))
            qml.CSWAP(wires=(0,2,3))
            qml.RX(1.2345, wires=0)

            qml.CRZ(1.2345, wires=(3,0))
            qml.expval(qml.PauliZ(0))

        fig, ax = draw_mpl(tape)

    .. figure:: ../../_static/draw_mpl/default.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    .. UsageDetails::

    **Decimals:**

    The keyword ``decimals`` controls how many decimal points to include when labelling the operations.
    The default value ``None`` omits parameters for brevity.

    .. code-block:: python

        with qml.tape.QuantumTape() as tape2:
            qml.RX(1.23456, wires=0)
            qml.Rot(1.2345,2.3456, 3.456, wires=0)
            qml.expval(qml.PauliZ(0))

        fig, ax = draw_mpl(tape2, decimals=2)

    .. figure:: ../../_static/draw_mpl/decimals.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    **Wires:**

    The keywords ``wire_order`` and ``show_all_wires`` control the location of wires from top to bottom.

    .. code-block:: python

        fig, ax = draw_mpl(tape, wire_order=[3,2,1,0])

    .. figure:: ../../_static/draw_mpl/wire_order.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    If a wire is in ``wire_order``, but not in the ``tape``, it will be omitted by default.  Only by selecting
    ``show_all_wires=True`` will empty wires be diplayed.

    .. code-block:: python

        fig, ax = draw_mpl(tape, wire_order=["aux"], show_all_wires=True)

    .. figure:: ../../_static/draw_mpl/show_all_wires.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    **Integration with matplotlib:**

    This function returns matplotlib figure and axes objects. Using these objects,
    users can perform further customization of the graphic.

    .. code-block:: python

        fig, ax = draw_mpl(tape)
        fig.suptitle("My Circuit", fontsize="xx-large")

        options = {'facecolor': "white", 'edgecolor': "#f57e7e", "linewidth": 6, "zorder": -1}
        box1 = plt.Rectangle((-0.5, -0.5), width=3.0, height=4.0, **options)
        ax.add_patch(box1)

        ax.annotate("CSWAP", xy=(2, 2.5), xycoords='data', xytext=(2.8,1.5), textcoords='data',
                    arrowprops={'facecolor': 'black'}, fontsize=14)

    .. figure:: ../../_static/draw_mpl/postprocessing.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    **Formatting:**

    You can globally control the style with ``plt.rcParams`` and styles, see the
    `matplotlib docs <https://matplotlib.org/stable/tutorials/introductory/customizing.html>`_ .
    If we customize ``plt.rcParams``, we get a
    different style:

    .. code-block:: python

        plt.rcParams['patch.facecolor'] = 'white'
        plt.rcParams['patch.edgecolor'] = 'black'
        plt.rcParams['patch.linewidth'] = 2
        plt.rcParams['patch.force_edgecolor'] = True
        plt.rcParams['lines.color'] = 'black'

        fig, ax = draw_mpl(tape)

    .. figure:: ../../_static/draw_mpl/rcparams.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    Instead of manually customizing everything, you can choose one of
    the provided styles. You can see available styles with ``plt.style.available``.
    We can set the ``'Solarize_Light2'`` style and instead get:

    .. code-block:: python

        with plt.style.context("Solarize_Light2):
            fig, ax = draw_mpl(tape)

    .. figure:: ../../_static/draw_mpl/Solarize_Light2.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    The wires and wire labels can be manually formatted by passing in dictionaries of
    keyword-value pairs of matplotlib options. ``wire_options`` accepts options for lines,
    and ``label_options`` accepts text options.

    .. code-block:: python

        fig, ax = draw_mpl(tape, wire_options={'color':'black', 'linewidth': 5},
                    label_options={'size': 20})

    .. figure:: ../../_static/draw_mpl/wires_labels.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    """
    wire_options = kwargs.get("wire_options", None)
    label_options = kwargs.get("label_options", None)

    wire_map = convert_wire_order(
        tape.operations + tape.measurements, wire_order=wire_order, show_all_wires=show_all_wires
    )

    layers = drawable_layers(tape.operations, wire_map=wire_map)

    n_layers = len(layers)
    n_wires = len(wire_map)

    drawer = MPLDrawer(n_layers=n_layers, n_wires=n_wires, wire_options=wire_options)
    drawer.label(list(wire_map), text_options=label_options)

    for layer, layer_ops in enumerate(layers):
        for op in layer_ops:
            mapped_wires = [wire_map[w] for w in op.wires]
            try:
                control_wires = [wire_map[w] for w in op.control_wires]
            except (NotImplementedError, AttributeError):
                control_wires = None

            specialfunc = special_cases.get(op.__class__, None)
            if specialfunc is not None:
                specialfunc(drawer, op, layer, mapped_wires)

            elif control_wires is not None:
                target_wires = [wire_map[w] for w in op.wires if w not in op.control_wires]
                drawer.ctrl(layer, control_wires, wires_target=target_wires)
                drawer.box_gate(
                    layer,
                    target_wires,
                    op.label(decimals=decimals),
                    box_options={"zorder": 4},
                    text_options={"zorder": 5},
                )

            else:
                drawer.box_gate(layer, mapped_wires, op.label(decimals=decimals))

    m_wires = Wires([])
    for m in tape.measurements:
        if len(m.wires) == 0:
            m_wires = -1
            break
        m_wires += m.wires

    if m_wires == -1:
        for wire in range(n_wires):
            drawer.measure(n_layers, wire)
    else:
        for wire in m_wires:
            drawer.measure(n_layers, wire_map[wire])

    return drawer.fig, drawer.ax
