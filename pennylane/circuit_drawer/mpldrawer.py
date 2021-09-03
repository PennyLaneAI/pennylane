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
This module contains the MPLDrawer class for creating circuit diagrams with matplotlib
"""
from collections.abc import Iterable

try:
    import matplotlib.pyplot as plt
    from matplotlib import patches
except (ModuleNotFoundError, ImportError) as e:
    raise ImportError(
        "Module matplotlib is required for ``MPLDrawer`` class. "
        "You can install matplotlib via \n\n   pip install matplotlib"
    ) from e


def _to_tuple(a):
    """converts int or iterable to always tuple"""
    if a is None:
        return tuple()
    if isinstance(a, Iterable):
        return tuple(a)
    return (a,)


# pylint: disable=too-many-instance-attributes, too-many-arguments
class MPLDrawer:
    r"""Allows easy creation of graphics representing circuits with matplotlib

    Args:
        n_layers (int): the number of layers
        n_wires (int): the number of wires

    Keyword Args:
        wire_kwargs=None (dict): matplotlib configuration options for drawing the wire lines
        figsize=None (Iterable): Allows users to specify the size of the figure manually.  Defaults
            to scale with the size of the circuit via ``n_layers`` and ``n_wires``.

    **Example**

    .. code-block:: python

        drawer = MPLDrawer(n_wires=5,n_layers=5)

        drawer.label(["0","a",r"$|\Psi\rangle$",r"$|\theta\rangle$", "aux"])

        drawer.box_gate(layer=0, wires=[0,1,2,3,4], text="Entangling Layers", text_kwargs={'rotation':'vertical'})
        drawer.box_gate(layer=1, wires=[0, 1], text="U(θ)")

        drawer.box_gate(layer=1, wires=4, text="Z")

        drawer.SWAP(layer=1, wires=(2, 3))
        drawer.CNOT(layer=2, wires=(0,2))

        drawer.ctrl(layer=3, wires=[1,3], control_values = [True, False])
        drawer.box_gate(layer=3, wires=2, text="H", zorder=2)

        drawer.ctrl(layer=4, wires=[1,2])

        drawer.measure(layer=5, wires=0)

        drawer.fig.suptitle('My Circuit', fontsize='xx-large')

        return drawer

    .. figure:: ../../_static/drawer/example_basic.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    .. UsageDetails::

    **Matplotlib Integration**

    This class relies on matplotlib.  As such, users can extend this class via interacting with the figure
    ``drawer.fig`` and axes ``drawer.ax`` objects manually.  For example, the example circuit manipulates the
    figure to set a title using ``drawer.fig.suptitle``. Users can also save the image using ``plt.savefig``.
    As descrived in the next section, the figure supports both global styling and individual styling of
    elements with matplotlib styles, configuration, and keywords.

    **Formatting**

    You can globally control the style with ``plt.rcParams``  and styles, see the
    `matplotlib docs <https://matplotlib.org/stable/tutorials/introductory/customizing.html>`_ .
    If we customize ``plt.rcParams`` before executing our example function, we get a
    different style:

    .. code-block:: python

        plt.rcParams['patch.facecolor'] = 'white'
        plt.rcParams['patch.edgecolor'] = 'black'
        plt.rcParams['patch.linewidth'] = 2
        plt.rcParams['patch.force_edgecolor'] = True
        plt.rcParams['lines.color'] = 'black'

    .. figure:: ../../_static/drawer/example_rcParams.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    Instead of manually customizing everything, you can choose one of
    the provided styles. You can see available styles with ``plt.style.available``.
    We can set the ``'Solarize_Light2'`` style with the same graph as drawn above and instead get:

    .. code-block:: python

        plt.style.use('Solarize_light2')

    .. figure:: ../../_static/drawer/example_Solarize_Light2.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    You can also manually control the styles of individual plot elements via the drawer class.
    Any control-type method, ``ctrl``, ``CNOT``, ``_ctrl_circ``, ``_ctrlo_circ``, and ``_target_x``, only
    accept a color keyword.  All other gates accept dictionaries of keyword-values pairs for matplotlib object
    components.  Acceptable keywords differ based on what's being drawn. For example, you cannot pass ``"fontsize"``
    to the dictionary controlling how to format a rectangle.

    This example demonstrates the different ways you can format the individual elements:

    .. code-block:: python

        wire_kwargs = {"color": "indigo", "linewidth": 4}
        drawer = MPLDrawer(n_wires=2,n_layers=4, wire_kwargs=wire_kwargs)

        label_kwargs = {"fontsize": "x-large", 'color': 'indigo'}
        drawer.label(["0","a"], text_kwargs=label_kwargs)

        box_kwargs = {'facecolor':'lightcoral', 'edgecolor': 'maroon', 'linewidth': 5}
        text_kwargs = {'fontsize': 'xx-large', 'color':'maroon'}
        drawer.box_gate(layer=0, wires=0, text="Z", box_kwargs=box_kwargs, text_kwargs=text_kwargs)

        swap_kwargs = {'linewidth': 4, 'color': 'darkgreen'}
        drawer.SWAP(layer=1, wires=(0,1), kwargs=swap_kwargs)

        drawer.CNOT(layer=2, wires=(0,1), color='teal')

        drawer.ctrl(layer=3, wires=(0,1), color='black')


        measure_box = {'facecolor': 'white', 'edgecolor': 'indigo'}
        measure_lines = {'edgecolor': 'indigo', 'facecolor': 'plum', 'linewidth': 2}
        for wire in range(2):
            drawer.measure(layer=4, wires=wire, box_kwargs=measure_box, lines_kwargs=measure_lines)

        drawer.fig.suptitle('My Circuit', fontsize='xx-large')

    .. figure:: ../../_static/drawer/example_formatted.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    **Positioning**

    Each gate takes arguments in order of ``layer`` followed by ``wires``.  These translate to ``x`` and
    ``y`` coordinates in the graph. Layer number (``x``) increases as you go right, and wire number
    (``y``) increases as you go down as the y-axis is inverted.  You can pass non-integer values to either keyword.
    If you have a long label, the gate can span multiple layers and have extra width:

    .. code-block:: python

        drawer = MPLDrawer(2,2)
        drawer.box_gate(layer=0, wires=1, text="X")
        drawer.box_gate(layer=1, wires=1, text="Y")

        # Gate between two layers
        drawer.box_gate(layer=0.5, wires=0, text="Big Gate", extra_width=0.5)

    .. figure:: ../../_static/drawer/float_layer.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    Many methods accept a ``zorder`` keyword. Higher ``zorder`` objects are drawn
    on top of lower ``zorder`` objects. In top example, we have to set a ``zorder``
    to a value of ``2`` in order to draw it *on top* of the control wires, instead of below them.
    """

    def __init__(self, n_layers, n_wires, wire_kwargs=None, figsize=None):

        self.n_layers = n_layers
        self.n_wires = n_wires

        # half the width of a box
        # is difference between center and edge
        self._box_dx = 0.4

        self._circ_rad = 0.3
        self._ctrl_rad = 0.1
        self._octrl_rad = 0.15
        self._swap_dx = 0.2

        ## Creating figure and ax

        if figsize is None:
            figsize = (self.n_layers + 3, self.n_wires + 1)

        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_axes(
            [0, 0, 1, 1],
            xlim=(-2, self.n_layers + 1),
            ylim=(-1, self.n_wires),
            xticks=[],
            yticks=[],
        )

        self.ax.invert_yaxis()

        if wire_kwargs is None:
            wire_kwargs = {}

        # adding wire lines
        for wire in range(self.n_wires):
            line = plt.Line2D((-1, self.n_layers), (wire, wire), zorder=1, **wire_kwargs)
            self.ax.add_line(line)

    def label(self, labels, text_kwargs=None):
        """Label each wire.

        Args:
            labels [Iterable[str]]: Iterable of labels for the wires

        Keyword Args:
            text_kwargs (dict): any matplotlib keywords for a text object, such as font or size.

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=1)
            drawer.label(["a", "b"])

        .. figure:: ../../_static/drawer/labels.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        You can also pass any
        `Matplotlib Text keywords <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html>`_
        as a dictionary to the ``text_kwargs`` keyword:

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=1)
            drawer.label(["a", "b"], text_kwargs={"color": "indigo", "fontsize": "xx-large"})

        .. figure:: ../../_static/drawer/labels_formatted.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """
        if text_kwargs is None:
            text_kwargs = {}

        for wire, ii_label in enumerate(labels):
            self.ax.text(-1.5, wire, ii_label, **text_kwargs)

    def box_gate(
        self, layer, wires, text="", extra_width=0, zorder=0, box_kwargs=None, text_kwargs=None
    ):
        """Draws a box and adds label text to its center.

        Args:
            layer (Int): x coordinate for the box center
            wires (Union[Int, Iterable[Int]]): y locations to include inside the box.  Only min and max
                of an Iterable affect the output
            text (str): string to print at the box's center

        Kwargs:
            extra_width (float): Extra box width
            zorder_base=0 (Int): increase number to draw on top of other objects, like control wires
            box_kwargs=None (dict): Any matplotlib keywords for the Rectangle patch
            text_kwargs=None (dict): Any matplotlib keywords for the text

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=1)

            drawer.box_gate(layer=0, wires=(0, 1), text="CY")

        .. figure:: ../../_static/drawer/box_gates.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        This method can accept two different sets of design keywords.  ``box_kwargs`` takes
        `Rectangle keywords <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html>`_
        , and ``text_kwargs`` accepts
        `Matplotlib Text keywords <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html>`_ .

        .. code-block:: python

            box_kwargs = {'facecolor':'lightcoral', 'edgecolor': 'maroon', 'linewidth': 5}
            text_kwargs = {'fontsize': 'xx-large', 'color':'maroon'}

            drawer = MPLDrawer(n_wires=2, n_layers=1)

            drawer.box_gate(layer=0, wires=(0, 1), text="CY",
                box_kwargs=box_kwargs, text_kwargs=text_kwargs)

        .. figure:: ../../_static/drawer/box_gates_formatted.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """
        if box_kwargs is None:
            box_kwargs = {}
        if text_kwargs is None:
            text_kwargs = {}

        wires = _to_tuple(wires)

        box_min = min(wires)
        box_max = max(wires)
        box_len = box_max - box_min
        box_center = (box_max + box_min) / 2.0

        box = plt.Rectangle(
            (layer - self._box_dx - extra_width / 2, box_min - self._box_dx),
            2 * self._box_dx + extra_width,
            (box_len + 2 * self._box_dx),
            zorder=2 + zorder,
            **box_kwargs,
        )
        self.ax.add_patch(box)

        default_text_kwargs = {"ha": "center", "va": "center", "fontsize": "x-large"}
        default_text_kwargs.update(text_kwargs)

        self.ax.text(
            layer,
            box_center,
            text,
            zorder=3 + zorder,
            # rotation=rotation,
            **default_text_kwargs,
        )

    def ctrl(self, layer, wires, wires_target=None, control_values=None, color=None):
        """Add an arbitrary number of control wires

        Args:
            layer (Int): the layer to draw the object in
            wires (Union[Int, Iterable[Int]]): set of wires to control on

        Keyword Args:
            wires_target=None (Union[Int, Iterable[Int]]): target wires. Used to determine min
                and max wires for the vertical line
            control_values=None (Union[Bool, Iterable[Bool]]): for each control wire, denotes whether to control
                on ``False=0`` or ``True=1``.
            kwargs=None (dict): mpl line keywords

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=3)

            drawer.ctrl(layer=0, wires=0, wires_target=1)
            drawer.ctrl(layer=1, wires=(0,1), control_values=[0,1])
            drawer.ctrl(layer=2, wires=(0,1), color="indigo")

        .. figure:: ../../_static/drawer/ctrl.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """
        wires_ctrl = _to_tuple(wires)
        wires_target = _to_tuple(wires_target)
        if control_values is not None:
            control_values = _to_tuple(control_values)

        wires_all = wires_ctrl + wires_target
        min_wire = min(wires_all)
        max_wire = max(wires_all)

        line = plt.Line2D((layer, layer), (min_wire, max_wire), zorder=2, color=color)
        self.ax.add_line(line)

        if control_values is None:
            for wire in wires_ctrl:
                self._ctrl_circ(layer, wire, zorder=3, color=color)
        else:
            if len(control_values) != len(wires_ctrl):
                raise ValueError("`control_values` must be the same length as `wires`")
            for wire, control_on in zip(wires_ctrl, control_values):
                if control_on:
                    self._ctrl_circ(layer, wire, zorder=3, color=color)
                else:
                    self._ctrlo_circ(layer, wire, zorder=3, color=color)

    def _ctrl_circ(self, layer, wires, zorder=3, color=None):
        """Draw a solid circle that indicates control on one"""

        if color is None:
            kwargs = {"facecolor": plt.rcParams["lines.color"]}
        else:
            kwargs = {"color": color}

        circ_ctrl = plt.Circle((layer, wires), radius=self._ctrl_rad, zorder=zorder, **kwargs)
        self.ax.add_patch(circ_ctrl)

    def _ctrlo_circ(self, layer, wires, zorder=3, color=None):
        """Draw an open circle that indicates control on zero."""

        kwargs = {
            "edgecolor": plt.rcParams["lines.color"],
            "facecolor": plt.rcParams["axes.facecolor"],
            "linewidth": plt.rcParams["lines.linewidth"],
        }
        if color is not None:
            kwargs["edgecolor"] = color

        circ_ctrlo = plt.Circle((layer, wires), radius=(self._octrl_rad), zorder=zorder, **kwargs)

        self.ax.add_patch(circ_ctrlo)

    def CNOT(self, layer, wires, color=None):
        """Draws a CNOT gate.

        Args:
            layer (Int): layer to draw in
            wires (Int, Int): tuple of (control, target)

        Keyword Args:
            color (None or str): mpl compatible color designation

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=2)

            drawer.CNOT(0, (0,1))
            drawer.CNOT(1, (1,0), color='indigo')

        .. figure:: ../../_static/drawer/cnot.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """
        target = wires[1]

        self.ctrl(layer, *wires, color=color)
        self._target_x(layer, target, color=color)

    def _target_x(self, layer, wires, color=None):
        """Draws the circle used to represent a CNOT's target

        Args:
            layer (Int): layer to draw on
            wires (Int): wire to draw on

        Keyword Args:
            color=None: mpl compatible color designation
        """

        default_kwargs = {
            "edgecolor": plt.rcParams["lines.color"],
            "linewidth": plt.rcParams["lines.linewidth"],
            "facecolor": plt.rcParams["axes.facecolor"],
        }
        if color is not None:
            default_kwargs["edgecolor"] = color

        target_circ = plt.Circle((layer, wires), radius=self._circ_rad, zorder=3, **default_kwargs)
        target_v = plt.Line2D(
            (layer, layer), (wires - self._circ_rad, wires + self._circ_rad), zorder=4, color=color
        )
        target_h = plt.Line2D(
            (layer - self._circ_rad, layer + self._circ_rad), (wires, wires), zorder=4, color=color
        )
        self.ax.add_patch(target_circ)
        self.ax.add_line(target_v)
        self.ax.add_line(target_h)

    def SWAP(self, layer, wires, kwargs=None):
        """Draws a SWAP gate

        Args:
            layer (Int): layer to draw on
            wires (Int, Int): Two wires the SWAP acts on

        Keyword Args:
            kwargs=None (dict): matplotlib keywords for ``Line2D`` objects

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=1)

            drawer.SWAP(0, (0,1))

        .. figure:: ../../_static/drawer/SWAP.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        The ``kwargs`` keyword can accept any
        `Line2D compatible keywords <https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_
        in a dictionary.

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=1)

            swap_keywords = {"linewidth":2, "color":"indigo"}
            drawer.SWAP(0, (0, 1), kwargs=swap_keywords)

        .. figure:: ../../_static/drawer/SWAP_formatted.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """
        if kwargs is None:
            kwargs = {}

        line = plt.Line2D((layer, layer), wires, zorder=2, **kwargs)
        self.ax.add_line(line)

        for wire in wires:
            self._swap_x(layer, wire, kwargs)

    def _swap_x(self, layer, wire, kwargs=None):
        """Draw an x such as used in drawing a swap gate

        Args:
            layer (Int): the layer
            wire (Int): the wire

        Keyword Args:
            kwargs=None (dict): matplotlib keywords for ``Line2D`` objects

        """
        if kwargs is None:
            kwargs = {}

        l1 = plt.Line2D(
            (layer - self._swap_dx, layer + self._swap_dx),
            (wire - self._swap_dx, wire + self._swap_dx),
            zorder=2,
            **kwargs,
        )
        l2 = plt.Line2D(
            (layer - self._swap_dx, layer + self._swap_dx),
            (wire + self._swap_dx, wire - self._swap_dx),
            zorder=2,
            **kwargs,
        )

        self.ax.add_line(l1)
        self.ax.add_line(l2)

    def measure(self, layer, wires, zorder_base=0, box_kwargs=None, lines_kwargs=None):
        """Draw a Measurement graphic at designated layer, wire combination.

        Args:
            layer (Int): the layer
            wires (Int): the wire

        Keyword Args:
            zorder_base=0 (Int): amount to shift in zorder from the default
            box_kwargs=None (dict): dictionary to format a matplotlib rectangle
            lines_kwargs=None (dict): dictionary to format matplotlib arc and arrow

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=1, n_layers=1)
            drawer.measure(0, 0)

        .. figure:: ../../_static/drawer/measure.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        This method accepts two different formatting dictionaries.  ``box_kwargs`` edits the rectangle
        while ``lines_kwargs`` edits the arc and arrow.

        .. code-block:: python

            drawer = MPLDrawer(n_wires=1, n_layers=1)

            measure_box = {'facecolor': 'white', 'edgecolor': 'indigo'}
            measure_lines = {'edgecolor': 'indigo', 'facecolor': 'plum', 'linewidth': 2}
            drawer.measure(0, 0, box_kwargs=measure_box, lines_kwargs=measure_lines)

        .. figure:: ../../_static/drawer/measure_formatted.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """
        if box_kwargs is None:
            box_kwargs = {}

        if lines_kwargs is None:
            lines_kwargs = {}

        box = plt.Rectangle(
            (layer - self._box_dx, wires - self._box_dx),
            2 * self._box_dx,
            2 * self._box_dx,
            zorder=2 + zorder_base,
            **box_kwargs,
        )
        self.ax.add_patch(box)

        arc = patches.Arc(
            (layer, wires + self._box_dx / 8),
            1.2 * self._box_dx,
            1.1 * self._box_dx,
            theta1=180,
            theta2=0,
            zorder=3 + zorder_base,
            **lines_kwargs,
        )
        self.ax.add_patch(arc)

        # can experiment with the specific numbers to make it look decent
        arrow_start_x = layer - 0.33 * self._box_dx
        arrow_start_y = wires + 0.5 * self._box_dx
        arrow_width = 0.6 * self._box_dx
        arrow_height = -1.0 * self._box_dx

        arrow = plt.arrow(
            arrow_start_x,
            arrow_start_y,
            arrow_width,
            arrow_height,
            head_width=self._box_dx / 4,
            zorder=4 + zorder_base,
            **lines_kwargs,
        )
        self.ax.add_line(arrow)
