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

has_mpl = True
try:
    import matplotlib.pyplot as plt
    from matplotlib import patches
except (ModuleNotFoundError, ImportError) as e:
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
        wire_options=None (dict): matplotlib configuration options for drawing the wire lines
        figsize=None (Iterable): Allows users to specify the size of the figure manually. Defaults
            to scale with the size of the circuit via ``n_layers`` and ``n_wires``.

    **Example**

    .. code-block:: python

        drawer = MPLDrawer(n_wires=5, n_layers=5)

        drawer.label(["0", "a", r"$|\Psi\rangle$", r"$|\theta\rangle$", "aux"])

        drawer.box_gate(layer=0, wires=[0, 1, 2, 3, 4], text="Entangling Layers", text_options={'rotation': 'vertical'})
        drawer.box_gate(layer=1, wires=[0, 1], text="U(θ)")

        drawer.box_gate(layer=1, wires=4, text="Z")

        drawer.SWAP(layer=1, wires=(2, 3))
        drawer.CNOT(layer=2, wires=(0, 2))

        drawer.ctrl(layer=3, wires=[1, 3], control_values = [True, False])
        drawer.box_gate(layer=3, wires=2, text="H", box_options={'zorder': 4},
            text_options={'zorder': 5})

        drawer.ctrl(layer=4, wires=[1, 2])

        drawer.measure(layer=5, wires=0)

        drawer.fig.suptitle('My Circuit', fontsize='xx-large')

    .. figure:: ../../_static/drawer/example_basic.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    .. UsageDetails::

    **Matplotlib Integration**

    This class relies on matplotlib. As such, users can extend this class via interacting with the figure
    ``drawer.fig`` and axes ``drawer.ax`` objects manually. For instance, the example circuit manipulates the
    figure to set a title using ``drawer.fig.suptitle``. Users can save the image using ``plt.savefig`` or via
    the figure method ``drawer.fig.savefig``.

    As described in the next section, the figure supports both global styling and individual styling of
    elements with matplotlib styles, configuration, and keywords.

    **Formatting**

    You can globally control the style with ``plt.rcParams`` and styles, see the
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

        plt.style.use('Solarize_Light2')

    .. figure:: ../../_static/drawer/example_Solarize_Light2.png
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

    _box_dx = 0.4
    _circ_rad = 0.3
    _ctrl_rad = 0.1
    _octrl_rad = 0.1
    _swap_dx = 0.2

    def __init__(self, n_layers, n_wires, wire_options=None, figsize=None):

        if not has_mpl:
            raise ImportError(
                "Module matplotlib is required for ``MPLDrawer`` class. "
                "You can install matplotlib via \n\n   pip install matplotlib"
            )

        self.n_layers = n_layers
        self.n_wires = n_wires

        ## Creating figure and ax

        if figsize is None:
            figsize = (self.n_layers + 3, self.n_wires + 1)

        self._fig = plt.figure(figsize=figsize)
        self._ax = self._fig.add_axes(
            [0, 0, 1, 1],
            xlim=(-2, self.n_layers + 1),
            ylim=(-1, self.n_wires),
            xticks=[],
            yticks=[],
        )
        self._ax.axis("off")

        self._ax.invert_yaxis()

        if wire_options is None:
            wire_options = {}

        # adding wire lines
        for wire in range(self.n_wires):
            line = plt.Line2D((-1, self.n_layers), (wire, wire), zorder=1, **wire_options)
            self._ax.add_line(line)

    @property
    def fig(self):
        """Matplotlib figure"""
        return self._fig

    @property
    def ax(self):
        """Matplotlib axes"""
        return self._ax

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
            text_options = {}

        for wire, ii_label in enumerate(labels):
            self._ax.text(-1.5, wire, ii_label, **text_options)

    def box_gate(self, layer, wires, text="", extra_width=0, box_options=None, text_options=None):
        """Draws a box and adds label text to its center.

        Args:
            layer (int): x coordinate for the box center
            wires (Union[int, Iterable[int]]): y locations to include inside the box. Only min and max
                of an Iterable affect the output
            text (str): string to print at the box's center

        Keyword Args:
            extra_width (float): extra box width
            box_options=None (dict): any matplotlib keywords for the ``plt.Rectangle`` patch
            text_options=None (dict): any matplotlib keywords for the text

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=1)

            drawer.box_gate(layer=0, wires=(0, 1), text="CY")

        .. figure:: ../../_static/drawer/box_gates.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

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

        """
        if box_options is None:
            box_options = {}
        if "zorder" not in box_options:
            box_options["zorder"] = 2

        new_text_options = {"zorder": 3, "ha": "center", "va": "center", "fontsize": "x-large"}
        if text_options is not None:
            new_text_options.update(text_options)

        wires = _to_tuple(wires)

        box_min = min(wires)
        box_max = max(wires)
        box_len = box_max - box_min
        box_center = (box_max + box_min) / 2.0

        box = plt.Rectangle(
            (layer - self._box_dx - extra_width / 2, box_min - self._box_dx),
            2 * self._box_dx + extra_width,
            (box_len + 2 * self._box_dx),
            **box_options,
        )
        self._ax.add_patch(box)

        self._ax.text(
            layer,
            box_center,
            text,
            **new_text_options,
        )

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
            options["facecolor"] = plt.rcParams["lines.color"]
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

    def CNOT(self, layer, wires, options=None):
        """Draws a CNOT gate.

        Args:
            layer (int): layer to draw in
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

        self.ctrl(layer, wires[:-1], wires[-1], options=options)
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

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=1)

            drawer.SWAP(0, (0, 1))

        .. figure:: ../../_static/drawer/SWAP.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        The ``options`` keyword can accept any
        `Line2D compatible keywords <https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_
        in a dictionary.

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=1)

            swap_options = {"linewidth": 2, "color": "indigo"}
            drawer.SWAP(0, (0, 1), options=swap_options)

        .. figure:: ../../_static/drawer/SWAP_formatted.png
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

    def measure(self, layer, wires, box_options=None, lines_options=None):
        """Draw a Measurement graphic at designated layer, wire combination.

        Args:
            layer (int): layer to draw on
            wires (int): wire to draw on

        Keyword Args:
            box_options=None (dict): dictionary to format a matplotlib rectangle
            lines_options=None (dict): dictionary to format matplotlib arc and arrow

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=1, n_layers=1)
            drawer.measure(0, 0)

        .. figure:: ../../_static/drawer/measure.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        This method accepts two different formatting dictionaries. ``box_options`` edits the rectangle
        while ``lines_options`` edits the arc and arrow.

        .. code-block:: python

            drawer = MPLDrawer(n_wires=1, n_layers=1)

            measure_box = {'facecolor': 'white', 'edgecolor': 'indigo'}
            measure_lines = {'edgecolor': 'indigo', 'facecolor': 'plum', 'linewidth': 2}
            drawer.measure(0, 0, box_options=measure_box, lines_options=measure_lines)

        .. figure:: ../../_static/drawer/measure_formatted.png
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

        box = plt.Rectangle(
            (layer - self._box_dx, wires - self._box_dx),
            2 * self._box_dx,
            2 * self._box_dx,
            **box_options,
        )
        self._ax.add_patch(box)

        arc = patches.Arc(
            (layer, wires + self._box_dx / 8),
            1.2 * self._box_dx,
            1.1 * self._box_dx,
            theta1=180,
            theta2=0,
            **lines_options,
        )
        self._ax.add_patch(arc)

        # can experiment with the specific numbers to make it look decent
        arrow_start_x = layer - 0.33 * self._box_dx
        arrow_start_y = wires + 0.5 * self._box_dx
        arrow_width = 0.6 * self._box_dx
        arrow_height = -1.0 * self._box_dx

        lines_options["zorder"] += 1
        arrow = plt.arrow(
            arrow_start_x,
            arrow_start_y,
            arrow_width,
            arrow_height,
            head_width=self._box_dx / 4,
            **lines_options,
        )
        self._ax.add_line(arrow)
