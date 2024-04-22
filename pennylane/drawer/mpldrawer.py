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
import warnings
from typing import Sequence

has_mpl = True
try:
    import matplotlib.pyplot as plt
    from matplotlib import patches
    import matplotlib.patheffects as path_effects
except (ModuleNotFoundError, ImportError) as e:  # pragma: no cover
    has_mpl = False


def _to_tuple(a):
    """Converts int or iterable to tuple"""
    if a is None:
        return tuple()
    if isinstance(a, Iterable):
        return tuple(a)
    return (a,)


def _open_circ_options_process(options):
    """For use in both ``_ctrlo_circ`` and ``_target_x``."""
    if options is None:
        options = {}

    new_options = options.copy()
    if "color" in new_options:
        new_options["facecolor"] = plt.rcParams["axes.facecolor"]
        new_options["edgecolor"] = options["color"]
        new_options["color"] = None
    else:
        new_options["edgecolor"] = plt.rcParams["lines.color"]
        new_options["facecolor"] = plt.rcParams["axes.facecolor"]

    if "linewidth" not in new_options:
        new_options["linewidth"] = plt.rcParams["lines.linewidth"]
    if "zorder" not in new_options:
        new_options["zorder"] = 3

    return new_options


# pylint: disable=too-many-instance-attributes, too-many-arguments
class MPLDrawer:
    r"""Allows easy creation of graphics representing circuits with matplotlib

    Args:
        n_layers (int): the number of layers
        n_wires (int): the number of wires

    Keyword Args:
        c_wires=0 (int): the number of classical wires to leave space for.
        wire_options=None (dict): matplotlib configuration options for drawing the wire lines
        figsize=None (Iterable): Allows users to specify the size of the figure manually. Defaults
            to scale with the size of the circuit via ``n_layers`` and ``n_wires``.
        fig=None (matplotlib Figure): Allows users to specify the figure window to plot to.

    **Example**

    .. code-block:: python

        drawer = qml.drawer.MPLDrawer(n_wires=5, n_layers=6)

        drawer.label(["0", "a", r"$|\Psi\rangle$", r"$|\theta\rangle$", "aux"])

        drawer.box_gate(layer=0, wires=[0, 1, 2, 3, 4], text="Entangling Layers")
        drawer.box_gate(layer=1, wires=[0, 2, 3], text="U(θ)")

        drawer.box_gate(layer=1, wires=4, text="Z")

        drawer.SWAP(layer=2, wires=(3,4))
        drawer.CNOT(layer=2, wires=(0, 2))

        drawer.ctrl(layer=3, wires=[1, 3], control_values=[True, False])
        drawer.box_gate(
            layer=3, wires=2, text="H", box_options={"zorder": 4}, text_options={"zorder": 5}
        )

        drawer.ctrl(layer=4, wires=[1, 2])

        drawer.measure(layer=5, wires=0)

        drawer.fig.suptitle('My Circuit', fontsize='xx-large')

    .. figure:: ../../_static/drawer/example_basic.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    .. details::
        :title: Usage Details

    **Matplotlib Integration**

    This class relies on matplotlib. As such, users can extend this class via interacting with the figure
    ``drawer.fig`` and axes ``drawer.ax`` objects manually. For instance, the example circuit manipulates the
    figure to set a title using ``drawer.fig.suptitle``. Users can save the image using ``plt.savefig`` or via
    the figure method ``drawer.fig.savefig``.

    As described in the next section, the figure supports both global styling and individual styling of
    elements with matplotlib styles, configuration, and keywords.

    **Formatting**

    PennyLane has inbuilt styles for controlling the appearance of the circuit drawings.
    All available styles can be determined by evaluating ``qml.drawer.available_styles()``.
    Any available string can then be passed to ``qml.drawer.use_style``.

    .. code-block:: python

        qml.drawer.use_style('black_white')

    .. figure:: ../../_static/drawer/black_white_style.png
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


    .. figure:: ../../_static/drawer/example_rcParams.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    You can also manually control the styles of individual plot elements via the drawer class.
    All accept dictionaries of keyword-values pairs for matplotlib object
    components. Acceptable keywords differ based on what's being drawn. For example, you cannot pass ``"fontsize"``
    to the dictionary controlling how to format a rectangle. For the control-type gates ``CNOT`` and
    ``ctrl`` the options dictionary can only contain ``'linewidth'``, ``'color'``, or ``'zorder'`` keys.

    This example demonstrates the different ways you can format the individual elements:

    .. code-block:: python

        wire_options = {"color": "indigo", "linewidth": 4}
        drawer = MPLDrawer(n_wires=2, n_layers=4, wire_options=wire_options)

        label_options = {"fontsize": "x-large", 'color': 'indigo'}
        drawer.label(["0", "a"], text_options=label_options)

        box_options = {'facecolor': 'lightcoral', 'edgecolor': 'maroon', 'linewidth': 5}
        text_options = {'fontsize': 'xx-large', 'color': 'maroon'}
        drawer.box_gate(layer=0, wires=0, text="Z", box_options=box_options, text_options=text_options)

        swap_options = {'linewidth': 4, 'color': 'darkgreen'}
        drawer.SWAP(layer=1, wires=(0, 1), options=swap_options)

        ctrl_options = {'linewidth': 4, 'color': 'teal'}
        drawer.CNOT(layer=2, wires=(0, 1), options=ctrl_options)
        drawer.ctrl(layer=3, wires=(0, 1), options=ctrl_options)


        measure_box = {'facecolor': 'white', 'edgecolor': 'indigo'}
        measure_lines = {'edgecolor': 'indigo', 'facecolor': 'plum', 'linewidth': 2}
        for wire in range(2):
            drawer.measure(layer=4, wires=wire, box_options=measure_box, lines_options=measure_lines)

        drawer.fig.suptitle('My Circuit', fontsize='xx-large')

    .. figure:: ../../_static/drawer/example_formatted.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    **Positioning**

    Each gate takes arguments in order of ``layer`` followed by ``wires``. These translate to ``x`` and
    ``y`` coordinates in the graph. Layer number (``x``) increases as you go right, and wire number
    (``y``) increases as you go down; the y-axis is inverted. You can pass non-integer values to either keyword.
    If you have a long label, the gate can span multiple layers and have extra width:

    .. code-block:: python

        drawer = MPLDrawer(2, 2)
        drawer.box_gate(layer=0, wires=1, text="X")
        drawer.box_gate(layer=1, wires=1, text="Y")

        # Gate between two layers
        drawer.box_gate(layer=0.5, wires=0, text="Big Gate", extra_width=0.5)

    .. figure:: ../../_static/drawer/float_layer.png
            :align: center
            :width: 60%
            :target: javascript:void(0);
    """

    _box_length = 0.75
    """The width/height of the rectangle drawn by ``box_gate``"""

    _circ_rad = 0.3
    """The radius of CNOT's target symbol."""

    _ctrl_rad = 0.1
    """The radius of the control-on-one solid circle."""

    _octrl_rad = 0.1
    """The radius of the control-on-zero open circle."""

    _swap_dx = 0.2
    """Half the width/height of the SWAP X-symbol."""

    _fontsize = 14
    """The default fontsize."""

    _pad = 0.2
    """Padding for FancyBboxPatch objects."""

    _boxstyle = "round, pad=0.2"
    """Style for FancyBboxPatch objects."""

    _notch_width = 0.04
    """The width of active wire notches."""

    _notch_height = 0.25
    """The height of active wire notches."""

    _notch_style = "round, pad=0.05"
    """Box style for active wire notches."""

    _cond_shift = 0.03
    """The shift value from the centre axis for classical double-lines."""

    _cwire_scaling = 0.25
    """The distance between successive control wires."""

    def __init__(self, n_layers, n_wires, c_wires=0, wire_options=None, figsize=None, fig=None):
        if not has_mpl:  # pragma: no cover
            raise ImportError(
                "Module matplotlib is required for ``MPLDrawer`` class. "
                "You can install matplotlib via \n\n   pip install matplotlib"
            )

        self.n_layers = n_layers
        self.n_wires = n_wires

        ## Creating figure and ax

        if figsize is None:
            figheight = self.n_wires + self._cwire_scaling * c_wires + 1 + 0.5 * (c_wires > 0)
            figsize = (self.n_layers + 3, figheight)

        if fig is None:
            self._fig = plt.figure(figsize=figsize)
        else:
            fig.clear()
            fig.set_figwidth(figsize[0])
            fig.set_figheight(figsize[1])
            self._fig = fig

        self._ax = self._fig.add_axes(
            [0, 0, 1, 1],
            xlim=(-2, self.n_layers + 1),
            ylim=(-1, self.n_wires + self._cwire_scaling * c_wires + 0.5 * (c_wires > 0)),
            xticks=[],
            yticks=[],
        )

        self._ax.axis("off")

        self._ax.invert_yaxis()

        if wire_options is None:
            wire_options = {}

        # adding wire lines
        self._wire_lines = [
            plt.Line2D((-1, self.n_layers), (wire, wire), zorder=1, **wire_options)
            for wire in range(self.n_wires)
        ]
        for line in self._wire_lines:
            self._ax.add_line(line)

    @property
    def fig(self):
        """Matplotlib figure"""
        return self._fig

    @property
    def ax(self):
        """Matplotlib axes"""
        return self._ax

    @property
    def fontsize(self):
        """Default fontsize for text. Defaults to 14."""
        return self._fontsize

    @fontsize.setter
    def fontsize(self, value):
        """Set ``fontsize`` property as provided value."""
        self._fontsize = value

    def label(self, labels, text_options=None):
        """Label each wire.

        Args:
            labels (Iterable[str]): Iterable of labels for the wires

        Keyword Args:
            text_options (dict): any matplotlib keywords for a text object, such as font or size

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
        as a dictionary to the ``text_options`` keyword:

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=1)
            drawer.label(["a", "b"], text_options={"color": "indigo", "fontsize": "xx-large"})

        .. figure:: ../../_static/drawer/labels_formatted.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """
        if text_options is None:
            text_options = {"ha": "center", "va": "center", "fontsize": self.fontsize}

        for wire, ii_label in enumerate(labels):
            self._ax.text(-1.5, wire, ii_label, **text_options)

    def erase_wire(self, layer: int, wire: int, length: int) -> None:
        """Erases a portion of a wire by adding a rectangle that matches the background.

        Args:
            layer (int): starting x coordinate for erasing the wire
            wire (int): y location to erase the wire from
            length (float, int): horizontal distance from ``layer`` to erase the background.

        """

        rect = patches.Rectangle(
            (layer, wire - 0.1),
            length,
            0.2,
            facecolor=plt.rcParams["figure.facecolor"],
            edgecolor=plt.rcParams["figure.facecolor"],
            zorder=1.1,
        )
        self.ax.add_patch(rect)

    def box_gate(self, layer, wires, text="", box_options=None, text_options=None, **kwargs):
        """Draws a box and adds label text to its center.

        Args:
            layer (int): x coordinate for the box center
            wires (Union[int, Iterable[int]]): y locations to include inside the box. Only min and max
                of an Iterable affect the output
            text (str): string to print at the box's center

        Keyword Args:
            box_options=None (dict): any matplotlib keywords for the ``plt.Rectangle`` patch
            text_options=None (dict): any matplotlib keywords for the text
            extra_width (float): extra box width
            autosize (bool): whether to rotate and shrink text to fit within the box
            active_wire_notches (bool): whether or not to add notches indicating active wires.
                Defaults to ``True``.

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=1)

            drawer.box_gate(layer=0, wires=(0, 1), text="CY")

        .. figure:: ../../_static/drawer/box_gates.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        .. details::
            :title: Usage Details

        This method can accept two different sets of design keywords. ``box_options`` takes
        `Rectangle keywords <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html>`_
        , and ``text_options`` accepts
        `Matplotlib Text keywords <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html>`_ .

        .. code-block:: python

            box_options = {'facecolor': 'lightcoral', 'edgecolor': 'maroon', 'linewidth': 5}
            text_options = {'fontsize': 'xx-large', 'color': 'maroon'}

            drawer = MPLDrawer(n_wires=2, n_layers=1)

            drawer.box_gate(layer=0, wires=(0, 1), text="CY",
                box_options=box_options, text_options=text_options)

        .. figure:: ../../_static/drawer/box_gates_formatted.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        By default, text is rotated and/or shrunk to fit within the box. This behaviour can be turned off
        with the ``autosize=False`` keyword.

        .. code-block:: python

            drawer = MPLDrawer(n_layers=4, n_wires=2)

            drawer.box_gate(layer=0, wires=0, text="A longer label")
            drawer.box_gate(layer=0, wires=1, text="Label")

            drawer.box_gate(layer=1, wires=(0,1), text="long multigate label")

            drawer.box_gate(layer=3, wires=(0,1), text="Not autosized label", autosize=False)

        .. figure:: ../../_static/drawer/box_gates_autosized.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """
        extra_width = kwargs.get("extra_width", 0)
        autosize = kwargs.get("autosize", True)
        active_wire_notches = kwargs.get("active_wire_notches", True)

        if box_options is None:
            box_options = {}
        if "zorder" not in box_options:
            box_options["zorder"] = 2

        new_text_options = {"zorder": 3, "ha": "center", "va": "center", "fontsize": self.fontsize}
        if text_options is not None:
            new_text_options.update(text_options)

        wires = _to_tuple(wires)

        box_min = min(wires)
        box_max = max(wires)
        box_center = (box_max + box_min) / 2.0

        x_loc = layer - self._box_length / 2.0 - extra_width / 2.0 + self._pad
        y_loc = box_min - self._box_length / 2.0 + self._pad
        box_height = box_max - box_min + self._box_length - 2 * self._pad
        box_width = self._box_length + extra_width - 2 * self._pad

        box = patches.FancyBboxPatch(
            (x_loc, y_loc),
            box_width,
            box_height,
            boxstyle=self._boxstyle,
            **box_options,
        )
        self._ax.add_patch(box)

        text_obj = self._ax.text(
            layer,
            box_center,
            text,
            **new_text_options,
        )

        if active_wire_notches and (len(wires) != (box_max - box_min + 1)):
            notch_options = box_options.copy()
            notch_options["zorder"] += -1
            for wire in wires:
                self._add_notch(layer, wire, extra_width, notch_options)

        if autosize:
            margin = 0.1
            max_width = box_width - margin + 2 * self._pad
            # factor of 2 makes it look nicer
            max_height = box_height - 2 * margin + 2 * self._pad

            w, h = self._text_dims(text_obj)

            # rotate the text
            if (box_min != box_max) and (w > max_width) and (w > h):
                text_obj.set_rotation(90)
                w, h = self._text_dims(text_obj)

            # shrink by decreasing the font size
            current_fontsize = text_obj.get_fontsize()
            for s in range(int(current_fontsize), 1, -1):
                if (w < max_width) and (h < max_height):
                    break
                text_obj.set_fontsize(s)
                w, h = self._text_dims(text_obj)

    def _add_notch(self, layer, wire, extra_width, box_options):
        """Add a wire used marker to both sides of a box.

        Args:
            layer (int): x coordinate for the box center
            wire (int): y cordinate for the notches
            extra_width (float): extra box width
            box_options (dict): styling options
        """
        y = wire - self._notch_height / 2
        x1 = layer - self._box_length / 2.0 - extra_width / 2.0 - self._notch_width
        x2 = layer + self._box_length / 2.0 + extra_width / 2.0

        box1 = patches.FancyBboxPatch(
            (x1, y),
            self._notch_width,
            self._notch_height,
            boxstyle=self._notch_style,
            **box_options,
        )
        self._ax.add_patch(box1)
        box2 = patches.FancyBboxPatch(
            (x2, y),
            self._notch_width,
            self._notch_height,
            boxstyle=self._notch_style,
            **box_options,
        )
        self._ax.add_patch(box2)

    def _text_dims(self, text_obj):
        """Get width and height of text object in data coordinates.

        See `this tutorial <https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html>`_
        for details on matplotlib coordinate systems.

        If the renderered figure is resized, such as in a GUI display, rectangles and lines
        are resized, but text stays the same size.  Text objects rely on display coordinates, that wont shrink
        as the figure is modified.

        Args:
            text_obj (matplotlib.text.Text): the matplotlib text object

        Returns:
            width (float): the width of the text in data coordinates
            height (float): the height of the text in data coordinates
        """
        renderer = self._fig.canvas.get_renderer()

        # https://matplotlib.org/stable/api/_as_gen/matplotlib.artist.Artist.get_window_extent.html
        # Quote: "Be careful when using this function, the results will not update if the artist
        # window extent of the artist changes. "
        # But I haven't encountered any issues yet and don't see a better solution
        bbox = text_obj.get_window_extent(renderer)

        corners = self._ax.transData.inverted().transform(bbox)
        return abs(corners[1][0] - corners[0][0]), abs(corners[0][1] - corners[1][1])

    def ctrl(self, layer, wires, wires_target=None, control_values=None, options=None):
        """Add an arbitrary number of control wires

        Args:
            layer (int): the layer to draw the object in
            wires (Union[int, Iterable[int]]): set of wires to control on

        Keyword Args:
            wires_target=None (Union[int, Iterable[int]]): target wires. Used to determine min
                and max wires for the vertical line
            control_values=None (Union[bool, Iterable[bool]]): for each control wire, denotes whether to control
                on ``False=0`` or ``True=1``
            options=None (dict): Matplotlib keywords. The only supported keys are ``'color'``, ``'linewidth'``,
                and ``'zorder'``.

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=3)

            drawer.ctrl(layer=0, wires=0, wires_target=1)
            drawer.ctrl(layer=1, wires=(0, 1), control_values=[0, 1])

            options = {'color': "indigo", 'linewidth': 4}
            drawer.ctrl(layer=2, wires=(0, 1), control_values=[1, 0], options=options)

        .. figure:: ../../_static/drawer/ctrl.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """
        if options is None:
            options = {}

        wires_ctrl = _to_tuple(wires)
        wires_target = _to_tuple(wires_target)
        if control_values is not None:
            control_values = _to_tuple(control_values)

        wires_all = wires_ctrl + wires_target
        min_wire = min(wires_all)
        max_wire = max(wires_all)

        if len(wires_target) > 1:
            min_target, max_target = min(wires_target), max(wires_target)
            if any(min_target < w < max_target for w in wires_ctrl):
                warnings.warn(
                    "Some control indicators are hidden behind an operator. Consider re-ordering "
                    "your circuit wires to ensure all control indicators are visible.",
                    UserWarning,
                )

        line = plt.Line2D((layer, layer), (min_wire, max_wire), **options)
        self._ax.add_line(line)

        if control_values is None:
            for wire in wires_ctrl:
                self._ctrl_circ(layer, wire, options=options)
        else:
            if len(control_values) != len(wires_ctrl):
                raise ValueError("`control_values` must be the same length as `wires`")
            for wire, control_on in zip(wires_ctrl, control_values):
                if control_on:
                    self._ctrl_circ(layer, wire, options=options)
                else:
                    self._ctrlo_circ(layer, wire, options=options)

    def _ctrl_circ(self, layer, wires, options=None):
        """Draw a solid circle that indicates control on one.

        Acceptable keys in options dictionary:
          * zorder
          * color
          * linewidth
        """
        if options is None:
            options = {}
        if "color" not in options:
            options["color"] = plt.rcParams["lines.color"]
        if "zorder" not in options:
            options["zorder"] = 3

        circ_ctrl = plt.Circle((layer, wires), radius=self._ctrl_rad, **options)
        self._ax.add_patch(circ_ctrl)

    def _ctrlo_circ(self, layer, wires, options=None):
        """Draw an open circle that indicates control on zero.

        Acceptable keys in options dictionary:
          * zorder
          * color
          * linewidth
        """
        new_options = _open_circ_options_process(options)

        circ_ctrlo = plt.Circle((layer, wires), radius=(self._octrl_rad), **new_options)

        self._ax.add_patch(circ_ctrlo)

    def CNOT(self, layer, wires, control_values=None, options=None):
        """Draws a CNOT gate.

        Args:
            layer (int): layer to draw in
            control_values=None (Union[bool, Iterable[bool]]): for each control wire, denotes whether to control
                on ``False=0`` or ``True=1``
            wires (Union[int, Iterable[int]]): wires to use. Last wire is the target.

        Keyword Args:
            options=None: Matplotlib options. The only supported keys are ``'color'``, ``'linewidth'``,
                and ``'zorder'``.

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=2)

            drawer.CNOT(0, (0, 1))

            options = {'color': 'indigo', 'linewidth': 4}
            drawer.CNOT(1, (1, 0), options=options)

        .. figure:: ../../_static/drawer/cnot.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """

        self.ctrl(layer, wires[:-1], wires[-1], control_values=control_values, options=options)
        self._target_x(layer, wires[-1], options=options)

    def _target_x(self, layer, wires, options=None):
        """Draws the circle used to represent a CNOT's target

        Args:
            layer (int): layer to draw on
            wires (int): wire to draw on

        Keyword Args:
            options=None (dict): Matplotlib keywords. The only supported keys are ``'color'``, ``'linewidth'``,
                and ``'zorder'``.
        """
        if options is None:
            options = {}
        new_options = _open_circ_options_process(options)
        options["zorder"] = new_options["zorder"] + 1

        target_circ = plt.Circle((layer, wires), radius=self._circ_rad, **new_options)

        target_v = plt.Line2D(
            (layer, layer), (wires - self._circ_rad, wires + self._circ_rad), **options
        )
        target_h = plt.Line2D(
            (layer - self._circ_rad, layer + self._circ_rad), (wires, wires), **options
        )

        self._ax.add_patch(target_circ)
        self._ax.add_line(target_v)
        self._ax.add_line(target_h)

    def SWAP(self, layer, wires, options=None):
        """Draws a SWAP gate

        Args:
            layer (int): layer to draw on
            wires (Tuple[int, int]): two wires the SWAP acts on

        Keyword Args:
            options=None (dict): matplotlib keywords for ``Line2D`` objects

        **Example**

        The ``options`` keyword can accept any
        `Line2D compatible keywords <https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_
        in a dictionary.

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=2)

            drawer.SWAP(0, (0, 1))

            swap_options = {"linewidth": 2, "color": "indigo"}
            drawer.SWAP(1, (0, 1), options=swap_options)

        .. figure:: ../../_static/drawer/SWAP.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """
        if options is None:
            options = {}

        line = plt.Line2D((layer, layer), wires, **options)
        self._ax.add_line(line)

        for wire in wires:
            self._swap_x(layer, wire, options)

    def _swap_x(self, layer, wire, options=None):
        """Draw an x such as used in drawing a swap gate

        Args:
            layer (int): layer to draw on
            wires (int): wire to draw on

        Keyword Args:
            options=None (dict): matplotlib keywords for ``Line2D`` objects
        """
        if options is None:
            options = {}
        if "zorder" not in options:
            options["zorder"] = 2

        l1 = plt.Line2D(
            (layer - self._swap_dx, layer + self._swap_dx),
            (wire - self._swap_dx, wire + self._swap_dx),
            **options,
        )
        l2 = plt.Line2D(
            (layer - self._swap_dx, layer + self._swap_dx),
            (wire + self._swap_dx, wire - self._swap_dx),
            **options,
        )

        self._ax.add_line(l1)
        self._ax.add_line(l2)

    def measure(self, layer, wires, text=None, box_options=None, lines_options=None):
        """Draw a Measurement graphic at designated layer, wire combination.

        Args:
            layer (int): layer to draw on
            wires (int): wire to draw on

        Keyword Args:
            text=None (str): an annotation for the lower right corner.
            box_options=None (dict): dictionary to format a matplotlib rectangle
            lines_options=None (dict): dictionary to format matplotlib arc and arrow

        **Example**

        This method accepts two different formatting dictionaries. ``box_options`` edits the rectangle
        while ``lines_options`` edits the arc and arrow.

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=1)
            drawer.measure(layer=0, wires=0)

            measure_box = {'facecolor': 'white', 'edgecolor': 'indigo'}
            measure_lines = {'edgecolor': 'indigo', 'facecolor': 'plum', 'linewidth': 2}
            drawer.measure(layer=0, wires=1, box_options=measure_box, lines_options=measure_lines)

        .. figure:: ../../_static/drawer/measure.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """
        if box_options is None:
            box_options = {}
        if "zorder" not in box_options:
            box_options["zorder"] = 2

        if lines_options is None:
            lines_options = {}
        if "zorder" not in lines_options:
            lines_options["zorder"] = 3

        if not isinstance(wires, Sequence):
            wires = (wires,)

        wires = tuple(self._y(w) for w in wires)

        box_min = min(wires)
        box_max = max(wires)
        box_center = (box_max + box_min) / 2.0

        x_loc = layer - self._box_length / 2.0 + self._pad
        y_loc = box_min - self._box_length / 2.0 + self._pad

        box = patches.FancyBboxPatch(
            (x_loc, y_loc),
            self._box_length - 2 * self._pad,
            box_max - box_min + self._box_length - 2 * self._pad,
            boxstyle=self._boxstyle,
            **box_options,
        )
        self._ax.add_patch(box)

        arc = patches.Arc(
            (layer, box_center + 0.15 * self._box_length),
            0.6 * self._box_length,
            0.55 * self._box_length,
            theta1=180,
            theta2=0,
            **lines_options,
        )
        self._ax.add_patch(arc)

        # can experiment with the specific numbers to make it look decent
        arrow_start_x = layer - 0.15 * self._box_length
        arrow_start_y = box_center + 0.3 * self._box_length
        arrow_width = 0.3 * self._box_length
        arrow_height = -0.5 * self._box_length

        lines_options["zorder"] += 1
        self.ax.arrow(
            arrow_start_x,
            arrow_start_y,
            arrow_width,
            arrow_height,
            head_width=self._box_length / 8.0,
            **lines_options,
        )
        if text:
            self._ax.text(
                layer + 0.05 * self._box_length,
                box_center + 0.225,
                text,
                fontsize=(self.fontsize - 2),
            )

    def _y(self, wire):
        """Used for determining the correct y coordinate for classical wires.
        Classical wires should be enumerated starting at the number of quantum wires the drawer has.
        For example, if the drawer has ``3`` quantum wires, the first classical wire should be located at ``3``
        which corresponds to a ``y`` coordinate of ``2.9``.
        """
        if wire < self.n_wires:
            return wire
        return self.n_wires + self._cwire_scaling * (wire - self.n_wires)

    def classical_wire(self, layers, wires) -> None:
        """Draw a classical control line.

        Args:
            layers: a list of x coordinates for the classical wire
            wires: a list of y coordinates for the classical wire. Wire numbers
                greater than the number of quantum wires will be scaled as classical wires.

        """
        outer_stroke = path_effects.Stroke(
            linewidth=5 * plt.rcParams["lines.linewidth"], foreground=plt.rcParams["lines.color"]
        )

        inner_stroke = path_effects.Stroke(
            linewidth=3 * plt.rcParams["lines.linewidth"],
            foreground=plt.rcParams["figure.facecolor"],
        )

        line = plt.Line2D(
            layers, [self._y(w) for w in wires], path_effects=[outer_stroke, inner_stroke], zorder=1
        )
        self.ax.add_line(line)

    def cwire_join(self, layer, wire, erase_right=False):
        """Erase the horizontal edges of an intersection between classical wires. By default, erases
        only the left edge.

        Args:
            layer: the x-coordinate for the classical wire intersection
            wire: the classical wire y-coordinate for the intersection
            erase_right=False(bool):  whether or not to erase the right side of the intersection
                in addition to the left.

        """
        xs = (layer - 0.2, layer + 0.2) if erase_right else (layer - 0.2, layer)
        line = plt.Line2D(
            xs,
            (self._y(wire), self._y(wire)),
            zorder=2,
            color=plt.rcParams["figure.facecolor"],
            linewidth=3 * plt.rcParams["lines.linewidth"],  # match inner_stroke from classical_wire
        )
        self.ax.add_line(line)

    def cond(self, layer, measured_layer, wires, wires_target, options=None):
        """Add classical communication double-lines for conditional operations

        Args:
            layer (int): the layer to draw vertical lines in, containing the target operation
            measured_layer (int): the layer where the mid-circuit measurements are
            wires (Union[int, Iterable[int]]): set of wires to control on
            wires_target (Union[int, Iterable[int]]): target wires. Used to determine where to
                terminate the vertical double-line

        Keyword Args:
            options=None (dict): Matplotlib keywords passed to ``plt.Line2D``

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=3, n_layers=4)

            drawer.cond(layer=1, measured_layer=0, wires=[0], wires_target=[1])

            options = {'color': "indigo", 'linewidth': 1.5}
            drawer.cond(layer=3, measured_layer=2, wires=(1,), wires_target=(2,), options=options)

        .. figure:: ../../_static/drawer/cond.png
            :align: center
            :width: 60%
            :target: javascript:void(0);
        """
        if options is None:
            options = {}

        wires_ctrl = _to_tuple(sorted(wires))
        wires_target = _to_tuple(sorted(wires_target))
        start_x = measured_layer + self._box_length / 2.0
        lines = []

        if wires_ctrl[-1] < wires_target[0]:
            lines.extend(
                (
                    # draw from top-most measurement to double-elbow
                    plt.Line2D(
                        (start_x, layer + self._cond_shift),
                        (wires_ctrl[0] - self._cond_shift,) * 2,
                        **options,
                    ),
                    plt.Line2D(
                        (start_x, layer - self._cond_shift),
                        (wires_ctrl[0] + self._cond_shift,) * 2,
                        **options,
                    ),
                    # draw vertical lines that reach the target operation
                    plt.Line2D(
                        (layer + self._cond_shift,) * 2,
                        (wires_ctrl[0] - self._cond_shift, wires_target[0]),
                        **options,
                    ),
                    plt.Line2D(
                        (layer - self._cond_shift,) * 2,
                        (wires_ctrl[-1] + self._cond_shift, wires_target[0]),
                        **options,
                    ),
                )
            )
            for prev_idx, next_wire in enumerate(wires_ctrl[1:]):
                # draw ⅃ for every wire but the first one
                #      ‾
                lines.extend(
                    (
                        plt.Line2D(
                            (layer - self._cond_shift,) * 2,
                            (wires_ctrl[prev_idx] + self._cond_shift, next_wire - self._cond_shift),
                            **options,
                        ),
                        plt.Line2D(
                            (start_x, layer - self._cond_shift),
                            (next_wire - self._cond_shift,) * 2,
                            **options,
                        ),
                        plt.Line2D(
                            (start_x, layer - self._cond_shift),
                            (next_wire + self._cond_shift,) * 2,
                            **options,
                        ),
                    )
                )
        elif wires_target[-1] < wires_ctrl[0]:
            lines.extend(
                (
                    # draw from bottom-most measurement to double-elbow
                    plt.Line2D(
                        (start_x, layer + self._cond_shift),
                        (wires_ctrl[-1] + self._cond_shift,) * 2,
                        **options,
                    ),
                    plt.Line2D(
                        (start_x, layer - self._cond_shift),
                        (wires_ctrl[-1] - self._cond_shift,) * 2,
                        **options,
                    ),
                    # draw vertical lines that reach the target operation
                    plt.Line2D(
                        (layer + self._cond_shift,) * 2,
                        (wires_ctrl[-1] + self._cond_shift, wires_target[-1]),
                        **options,
                    ),
                    plt.Line2D(
                        (layer - self._cond_shift,) * 2,
                        (wires_ctrl[0] - self._cond_shift, wires_target[-1]),
                        **options,
                    ),
                )
            )
            for wire_idx, ctrl_wire in enumerate(wires_ctrl[:-1]):
                # draw _  for every wire but the first one
                #      ‾|
                lines.extend(
                    (
                        plt.Line2D(
                            (layer - self._cond_shift,) * 2,
                            (
                                ctrl_wire + self._cond_shift,
                                wires_ctrl[wire_idx + 1] - self._cond_shift,
                            ),
                            **options,
                        ),
                        plt.Line2D(
                            (start_x, layer - self._cond_shift),
                            (ctrl_wire - self._cond_shift,) * 2,
                            **options,
                        ),
                        plt.Line2D(
                            (start_x, layer - self._cond_shift),
                            (ctrl_wire + self._cond_shift,) * 2,
                            **options,
                        ),
                    )
                )
        else:
            raise ValueError(
                "Cannot draw interspersed mid-circuit measurements and conditional operations. "
                "Consider providing a wire order such that all measurement wires precede all "
                "wires for the operator being controlled, or vice versa."
            )

        for line in lines:
            self._ax.add_line(line)
