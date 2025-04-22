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
Tests the MPLDrawer.

See section on "Testing Matplotlib based code" in the "Software Tests"
page in the developement guide.
"""
# pylint: disable=protected-access,wrong-import-position

import warnings

import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrow

from pennylane.drawer import MPLDrawer
from pennylane.math import allclose


class TestInitialization:
    """Tests drawer creation"""

    @pytest.mark.parametrize("wire_map", [{0: 0, 1: 1}, {"a": 0, "b": 1, "c": 2}])
    @pytest.mark.parametrize("n_layers", [2, 3])
    def test_figsize_wires(self, wire_map, n_layers):
        """Tests the figure is sized correctly."""

        n_wires = len(wire_map)

        drawer = MPLDrawer(wire_map=wire_map, n_layers=n_layers)

        assert drawer.fig.get_figwidth() == (n_layers + 3)
        assert drawer.fig.get_figheight() == (n_wires + 1)

        drawer = MPLDrawer(wire_map=wire_map, n_layers=n_layers)

        lines = drawer.ax.lines

        assert len(lines) == n_wires

        for wire, line in enumerate(lines):
            assert line.get_xdata() == (-1, n_layers)
            assert line.get_ydata() == (wire, wire)
        plt.close("all")

    def test_figsize_classical_wires(self):
        """Test the figsize is correct if classical wires are present."""
        n_wires = 4
        c_wires = 4
        n_layers = 1

        drawer = MPLDrawer(
            wire_map={i: i for i in range(n_wires)}, n_layers=n_layers, c_wires=c_wires
        )

        assert drawer.fig.get_figheight() == n_wires + 1 + 0.25 * 4 + 0.5
        assert drawer.fig.get_figwidth() == n_layers + 3

        assert drawer.ax.get_xlim() == (-2, 2)
        assert drawer.ax.get_ylim() == (n_wires + 0.25 * 4 + 0.5, -1)

    def test_customfigsize(self):
        """Tests a custom figsize alters the size"""

        drawer = MPLDrawer(1, {0: 0}, figsize=(5, 5))

        assert drawer.fig.get_figwidth() == 5
        assert drawer.fig.get_figheight() == 5
        plt.close("all")

    def test_customfigure(self):
        """Tests a custom figure is used"""

        fig = plt.figure()
        drawer = MPLDrawer(1, {0: 0}, fig=fig)

        assert drawer.fig == fig
        plt.close("all")

    def test_config_params_set(self):
        """Tests sizing hidden variables are set."""

        drawer = MPLDrawer(1, {0: 0})

        assert drawer._box_length == 0.75
        assert drawer._circ_rad == 0.3
        assert drawer._ctrl_rad == 0.1
        assert drawer._octrl_rad == 0.1
        assert drawer._swap_dx == 0.2
        assert drawer._fontsize == 14
        assert drawer._pad == 0.2
        assert drawer._boxstyle == "round, pad=0.2"
        assert drawer._notch_width == 0.04
        assert drawer._notch_height == 0.25
        assert drawer._notch_style == "round, pad=0.05"
        plt.close("all")

    def test_wires_formatting(self):
        """Tests wires formatting with options"""

        rgba_red = (1, 0, 0, 1)
        options = {"linewidth": 3, "color": rgba_red}
        drawer = MPLDrawer(wire_map={0: 0, 1: 1}, n_layers=2, wire_options=options)

        for line in drawer.ax.lines:
            assert line.get_linewidth() == 3
            assert line.get_color() == rgba_red

        plt.close("all")

    def test_fontsize(self):
        """Test fontsize can be get and set via property."""
        drawer = MPLDrawer(1, {0: 0})

        assert drawer._fontsize == drawer.fontsize
        assert drawer.fontsize == 14

        drawer.fontsize = 10

        assert drawer._fontsize == 10
        assert drawer.fontsize == 10
        plt.close("all")


class TestLabels:
    """Test wire labels work as expected."""

    def test_labels(self):
        """Tests labels are added"""

        wire_map = {"a": 0, "b": 1, "c": 2}

        drawer = MPLDrawer(1, wire_map)

        labels = wire_map.keys()
        drawer.label(labels)

        drawn_labels = drawer.ax.texts

        for wire, expected_label, actual_label in zip(range(3), labels, drawn_labels):
            assert actual_label.get_text() == expected_label

            assert actual_label.get_position() == (-1.5, wire)

        plt.close("all")

    def test_labels_formatting(self):
        """Test labels are formatted with text options."""

        wire_map = {0: 0, 1: 1, 2: 2}

        drawer = MPLDrawer(1, wire_map)

        rgba_red = (1, 0, 0, 1)
        labels = wire_map.keys()
        options = {"fontsize": 10, "color": rgba_red}
        drawer.label(labels, text_options=options)

        for text in drawer.ax.texts:
            assert text.get_fontsize() == 10
            assert text.get_color() == rgba_red

        plt.close("all")

    @pytest.mark.parametrize("n_layers", [1, 4])
    def test_crop_wire_labels(self, n_layers):
        """Test that cropping wire labels works."""

        wire_map = {"a": 0, "b": 1, "c": 2}

        drawer = MPLDrawer(n_layers, wire_map)

        labels = wire_map.keys()
        drawer.label(labels)
        old_width = drawer.fig.get_figwidth()
        assert old_width == n_layers + 3
        old_xlim = drawer.ax.get_xlim()
        assert old_xlim == (-2, n_layers + 1)
        drawer.crop_wire_labels()
        new_width = drawer.fig.get_figwidth()
        assert new_width == n_layers + 2
        new_xlim = drawer.ax.get_xlim()
        assert new_xlim == (-1, n_layers + 1)


def test_erase_wire():
    """Test the erase wire method."""

    drawer = MPLDrawer(5, {0: 0})
    drawer.erase_wire(1, 0, 3)

    assert len(drawer.ax.patches) == 1
    assert drawer.ax.patches[0].get_xy() == (1, -0.1)
    assert drawer.ax.patches[0].get_width() == 3
    assert drawer.ax.patches[0].get_height() == 0.2
    assert drawer.ax.patches[0].get_facecolor() == drawer.fig.get_facecolor()
    assert drawer.ax.patches[0].get_edgecolor() == drawer.fig.get_facecolor()
    assert drawer.ax.patches[0].zorder > drawer.ax.lines[0].zorder


class TestBoxGate:
    """Tests relating to box gate."""

    def test_simple_box(self):
        """tests basic functionality of box_gate."""

        drawer = MPLDrawer(1, {0: 0})

        drawer.box_gate(0, 0, "X")

        box = drawer.ax.patches[0]

        assert box.get_x() == -drawer._box_length / 2.0 + drawer._pad
        assert box.get_y() == -drawer._box_length / 2.0 + drawer._pad
        assert box.get_width() == drawer._box_length - 2 * drawer._pad
        assert box.get_height() == drawer._box_length - 2 * drawer._pad
        assert box.get_zorder() == 2

        text = drawer.ax.texts[0]

        assert text.get_text() == "X"
        assert text.get_position() == (0, 0)
        assert text.get_zorder() == 3
        plt.close("all")

    def test_multiwire_box(self):
        """tests a gate spanning multiple wires."""

        drawer = MPLDrawer(1, {0: 0, 1: 1, 2: 2})
        drawer.box_gate(0, (0, 2), text="Tall Gate")

        box = drawer.ax.patches[0]

        assert box.get_x() == -drawer._box_length / 2.0 + drawer._pad
        assert box.get_y() == -drawer._box_length / 2.0 + drawer._pad
        assert box.get_width() == drawer._box_length - 2 * drawer._pad
        assert box.get_height() == 2 + drawer._box_length - 2 * drawer._pad
        assert box.get_zorder() == 2

        text = drawer.ax.texts[0]

        assert text.get_text() == "Tall Gate"
        assert text.get_position() == (0, 1.0)
        assert text.get_zorder() == 3
        plt.close("all")

    def test_notch_standard_styling(self):
        """Test notch styling is correct"""

        drawer = MPLDrawer(n_layers=1, wire_map={i: i for i in range(3)})
        drawer.box_gate(0, (0, 2))

        xs = [-0.415, 0.375, -0.415, 0.375]
        ys = [-0.125, -0.125, 1.875, 1.875]

        # first patch is big box
        for x, y, notch in zip(xs, ys, drawer.ax.patches[1:]):
            assert notch.get_x() == x
            assert notch.get_y() == y
            assert notch.get_width() == drawer._notch_width
            assert notch.get_height() == drawer._notch_height
            assert notch.get_zorder() == 1
            assert notch.get_boxstyle().pad == 0.05
        plt.close("all")

    @pytest.mark.parametrize(
        "wires, n_notches", [((0, 1, 2, 3), 0), ((0,), 0), ((0, 2), 4), ((0, 1, 3), 6)]
    )
    def test_active_wire_notches_number(self, wires, n_notches):
        """Tests the number of active wires drawn is the expected amount."""

        drawer = MPLDrawer(n_layers=1, wire_map={i: i for i in range(4)})
        drawer.box_gate(layer=0, wires=wires)

        assert len(drawer.ax.patches) == (n_notches + 1)
        plt.close("all")

    def test_no_active_wire_notches(self):
        """Test active wire notches deactivated by keyword."""
        drawer = MPLDrawer(n_layers=1, wire_map={i: i for i in range(3)})
        drawer.box_gate(layer=0, wires=(0, 2), active_wire_notches=False)

        # only one patch for big box, no patches for notches
        assert len(drawer.ax.patches) == 1
        plt.close("all")

    def test_extra_width(self):
        """tests a box with added width."""

        drawer = MPLDrawer(1, {i: i for i in range(3)})
        drawer.box_gate(0, (0, 2), text="Wide Gate", extra_width=0.4)

        box = drawer.ax.patches[0]

        assert box.get_x() == -(drawer._box_length + 0.4) / 2.0 + drawer._pad
        assert box.get_y() == -drawer._box_length / 2.0 + drawer._pad
        assert box.get_width() == drawer._box_length + 0.4 - 2 * drawer._pad
        assert box.get_height() == 2 + drawer._box_length - 2 * drawer._pad

        text = drawer.ax.texts[0]

        assert text.get_text() == "Wide Gate"
        assert text.get_position() == (0, 1.0)

        xs = [-0.615, 0.575, -0.615, 0.575]
        for x, notch in zip(xs, drawer.ax.patches[1:]):
            assert notch.get_x() == x

        plt.close("all")

    def test_box_formatting(self):
        """Tests that box_options influences the rectangle"""

        drawer = MPLDrawer(1, {i: i for i in range(3)})
        rgba_red = (1, 0, 0, 1)
        rgba_green = (0, 1, 0, 1)
        options = {"facecolor": rgba_red, "edgecolor": rgba_green, "zorder": 5}
        drawer.box_gate(0, (0, 2), text="X", box_options=options)

        # notches get same styling options as box
        for p in drawer.ax.patches:
            assert p.get_facecolor() == rgba_red
            assert p.get_edgecolor() == rgba_green

        # except for zorder
        assert drawer.ax.patches[0].get_zorder() == 5
        for n in drawer.ax.patches[1:]:
            assert n.get_zorder() == 4
        plt.close("all")

    def test_text_formatting(self):
        """Tests rotated text"""

        drawer = MPLDrawer(1, {0: 0})
        rgba_red = (1, 0, 0, 1)
        options = {"color": rgba_red, "rotation": "vertical"}
        drawer.box_gate(0, 0, text="X", text_options=options)

        text = drawer.ax.texts[0]
        assert text.get_rotation() == 90.0
        assert text.get_color() == rgba_red
        plt.close("all")


class TestCTRL:
    """Tests ctrl, _target_x, and CNOT"""

    def test_ctrl_no_target(self):
        """Tests a single control with no target"""

        drawer = MPLDrawer(1, {0: 0})

        drawer.ctrl(0, 0)

        ctrl_line = drawer.ax.lines[1]

        assert ctrl_line.get_data() == ((0, 0), (0, 0))

        assert len(drawer.ax.patches) == 1

        circle = drawer.ax.patches[0]

        assert circle.width == 0.2
        assert circle.center == (0, 0)
        plt.close("all")

    def test_ctrl_multi_wires(self):
        """Tests two control wires with no target."""

        drawer = MPLDrawer(1, {i: i for i in range(3)})

        ctrl_wires = (0, 1)
        drawer.ctrl(0, ctrl_wires)

        ctrl_line = drawer.ax.lines[3]

        assert ctrl_line.get_data() == ((0, 0), ctrl_wires)

        circles = drawer.ax.patches

        assert len(circles) == 2

        for wire, circle in zip(ctrl_wires, circles):
            assert circle.width == 0.2
            assert circle.center == (0, wire)
        plt.close("all")

    def test_ctrl_on_zero(self):
        """Tests a control on zero circle is open"""

        drawer = MPLDrawer(1, {0: 0})

        drawer.ctrl(0, 0, control_values=False)

        circ = drawer.ax.patches[0]

        assert circ.get_facecolor() == to_rgba(plt.rcParams["axes.facecolor"])
        assert circ.get_edgecolor() == to_rgba(plt.rcParams["lines.color"])
        assert circ.get_linewidth() == plt.rcParams["lines.linewidth"]

        assert circ.center == (0, 0)
        assert circ.width == 0.2
        plt.close("all")

    def test_ctrl_control_values_error(self):
        """Tests a ValueError is raised if different number of wires and control_values."""

        drawer = MPLDrawer(1, {0: 0, 1: 1})

        with pytest.raises(ValueError, match="`control_values` must be the same length"):
            drawer.ctrl(0, (0, 1), control_values=True)

        plt.close("all")

    def test_ctrl_formatting(self):
        """Tests two control wires with no target."""

        drawer = MPLDrawer(1, {i: i for i in range(3)})

        ctrl_wires = (0, 1)
        rgba_red = (1, 0, 0, 1)
        options = {"color": rgba_red, "linewidth": 4}
        drawer.ctrl(0, ctrl_wires, control_values=[1, 0], options=options)

        ctrl_line = drawer.ax.lines[3]
        assert ctrl_line.get_color() == rgba_red
        assert ctrl_line.get_linewidth() == 4

        closed_circ = drawer.ax.patches[0]
        assert closed_circ.get_facecolor() == rgba_red

        open_circ = drawer.ax.patches[1]
        assert open_circ.get_edgecolor() == rgba_red
        assert open_circ.get_facecolor() == to_rgba(plt.rcParams["axes.facecolor"])
        assert open_circ.get_linewidth() == 4

        plt.close("all")

    def test_ctrl_circ(self):
        """Test only the ``_ctrl_circ`` private method."""

        drawer = MPLDrawer(1, {0: 0})
        drawer._ctrl_circ(0, 0)
        circ = drawer.ax.patches[0]

        assert circ.get_facecolor() == to_rgba(plt.rcParams["lines.color"])

        assert circ.center == (0, 0)
        assert circ.width == 0.2

        plt.close("all")

    def test_ctrlo_circ(self):
        """Test only the ``ctrlo_circ`` private method."""

        drawer = MPLDrawer(1, {0: 0})
        drawer._ctrlo_circ(0, 0)
        circ = drawer.ax.patches[0]

        assert circ.get_facecolor() == to_rgba(plt.rcParams["axes.facecolor"])
        assert circ.get_edgecolor() == to_rgba(plt.rcParams["lines.color"])
        assert circ.get_linewidth() == plt.rcParams["lines.linewidth"]
        plt.close("all")

    def test_ctrl_target(self):
        """Tests target impacts line extent"""

        drawer = MPLDrawer(1, {i: i for i in range(3)})

        drawer.ctrl(0, 0, 2)

        ctrl_line = drawer.ax.lines[3]

        assert ctrl_line.get_data() == ((0, 0), (0, 2))

        circles = drawer.ax.patches
        assert len(circles) == 1

        circle = drawer.ax.patches[0]

        assert circle.width == 0.2
        assert circle.center == (0, 0)
        plt.close("all")

    @pytest.mark.parametrize(
        "control_wires,target_wires",
        [
            ((1,), (0, 2)),
            ((0, 2), (1, 3)),
            ((1, 3), (0, 2)),
            ((0, 2, 4), (1, 3)),
        ],
    )
    def test_ctrl_raises_warning_with_overlap(self, control_wires, target_wires):
        """Tests that a warning is raised if some control indicators are not visible."""
        drawer = MPLDrawer(1, {i: i for i in range(4)})
        with pytest.warns(UserWarning, match="control indicators are hidden behind an operator"):
            drawer.ctrl(0, control_wires, target_wires)
        plt.close("all")

    @pytest.mark.parametrize("control_wires,target_wires", [((0,), (1, 2)), ((2,), (0, 1))])
    def test_ctrl_no_warning_without_overlap(self, control_wires, target_wires):
        drawer = MPLDrawer(1, {i: i for i in range(3)})
        with warnings.catch_warnings(record=True) as w:
            drawer.ctrl(0, control_wires, target_wires)
        assert len(w) == 0
        plt.close("all")

    def test_target_x(self):
        """Tests hidden target_x drawing method"""

        drawer = MPLDrawer(1, {i: i for i in range(3)})

        drawer._target_x(0, 0)

        center_line = drawer.ax.lines[3]
        assert center_line.get_data() == ((0, 0), (-0.3, 0.3))

        horizontal_line = drawer.ax.lines[4]
        assert horizontal_line.get_data() == ((-0.3, 0.3), (0, 0))

        circle = drawer.ax.patches[0]

        assert circle.center == (0, 0)
        assert circle.width == 0.6
        assert circle.get_facecolor() == to_rgba(plt.rcParams["axes.facecolor"])
        assert to_rgba(plt.rcParams["lines.color"]) == to_rgba(circle.get_edgecolor())
        plt.close("all")

    def test_target_x_color(self):
        """Test the color of target_x."""
        drawer = MPLDrawer(1, {i: i for i in range(3)})

        rgba_red = (1, 0, 0, 1)
        drawer._target_x(0, 0, options={"color": rgba_red})

        center_line = drawer.ax.lines[3]
        assert center_line.get_color() == rgba_red

        horizontal_line = drawer.ax.lines[4]
        assert horizontal_line.get_color() == rgba_red

        circle = drawer.ax.patches[0]
        assert circle.get_facecolor() == to_rgba(plt.rcParams["axes.facecolor"])
        assert circle.get_edgecolor() == rgba_red

        plt.close("all")

    def test_CNOT(self):
        """Tests the CNOT method"""

        drawer = MPLDrawer(1, {i: i for i in range(3)})

        drawer.CNOT(0, (0, 1))

        ctrl_line = drawer.ax.lines[3]
        assert ctrl_line.get_data() == ((0, 0), (0, 1))

        center_line = drawer.ax.lines[4]
        assert center_line.get_data() == ((0, 0), (0.7, 1.3))

        ctrl_circle = drawer.ax.patches[0]
        target_circle = drawer.ax.patches[1]

        assert ctrl_circle.center == (0, 0)
        assert ctrl_circle.width == 0.2

        assert target_circle.center == (0, 1)
        assert target_circle.width == 0.6
        assert target_circle.get_facecolor() == to_rgba(plt.rcParams["axes.facecolor"])
        assert to_rgba(plt.rcParams["lines.color"]) == to_rgba(target_circle.get_edgecolor())
        plt.close("all")

    def test_CNOT_control_values(self):
        """Tests the ``control_values`` keyword for CNOT."""

        drawer = MPLDrawer(1, {i: i for i in range(3)})

        drawer.CNOT(0, (0, 1, 2), control_values=[True, False])

        ctrl_circ1 = drawer.ax.patches[0]
        ctrl_circ2 = drawer.ax.patches[1]

        # first should be a closed in circle
        assert ctrl_circ1.get_facecolor() == to_rgba(plt.rcParams["lines.color"])
        # second facecolor should match the background
        assert ctrl_circ2.get_facecolor() == to_rgba(plt.rcParams["axes.facecolor"])
        plt.close("all")

    def test_CNOT_color(self):
        """Tests the color of CNOT."""
        drawer = MPLDrawer(1, {i: i for i in range(3)})
        rgba_red = (1, 0, 0, 1)
        drawer.CNOT(0, (0, 1), options={"color": rgba_red})

        ctrl_line = drawer.ax.lines[3]
        assert ctrl_line.get_color() == rgba_red

        center_line = drawer.ax.lines[4]
        assert center_line.get_color() == rgba_red

        ctrl_circle = drawer.ax.patches[0]
        assert ctrl_circle.get_facecolor() == rgba_red

        target_circle = drawer.ax.patches[1]
        assert target_circle.get_edgecolor() == rgba_red

        plt.close("all")


class TestSWAP:
    """Test the SWAP gate."""

    def test_swap_x(self):
        """Tests the ``_swap_x`` private method."""

        drawer = MPLDrawer(1, {0: 0})
        drawer._swap_x(0, 0)

        l1 = drawer.ax.lines[1]
        l2 = drawer.ax.lines[2]

        assert l1.get_data() == ((-0.2, 0.2), (-0.2, 0.2))
        assert l2.get_data() == ((-0.2, 0.2), (0.2, -0.2))
        plt.close("all")

    def test_SWAP(self):
        """Tests the SWAP method."""

        drawer = MPLDrawer(1, {i: i for i in range(3)})
        drawer.SWAP(0, (0, 2))

        connecting_line = drawer.ax.lines[3]
        assert connecting_line.get_data() == ((0, 0), (0, 2))

        x_lines = drawer.ax.lines[4:]
        assert x_lines[0].get_data() == ((-0.2, 0.2), (-0.2, 0.2))
        assert x_lines[1].get_data() == ((-0.2, 0.2), (0.2, -0.2))
        assert x_lines[2].get_data() == ((-0.2, 0.2), (1.8, 2.2))
        assert x_lines[3].get_data() == ((-0.2, 0.2), (2.2, 1.8))
        plt.close("all")

    def test_SWAP_options(self):
        """Tests that SWAP can be colored."""

        drawer = MPLDrawer(1, {i: i for i in range(3)})
        rgba_red = (1, 0, 0, 1)
        options = {"color": rgba_red, "linewidth": 3}
        drawer.SWAP(0, (0, 2), options=options)

        for line in drawer.ax.lines[3:]:
            assert line.get_color() == rgba_red
            assert line.get_linewidth() == 3

        plt.close("all")


class TestMeasure:
    """Tests the measure method."""

    def test_measure(self):
        """Tests the measure method."""

        drawer = MPLDrawer(1, {0: 0})
        drawer.measure(0, 0)

        box = drawer.ax.patches[0]
        assert box.get_x() == -drawer._box_length / 2.0 + drawer._pad
        assert box.get_y() == -drawer._box_length / 2.0 + drawer._pad
        assert box.get_width() == drawer._box_length - 2 * drawer._pad

        arc = drawer.ax.patches[1]
        assert arc.center == (0, 0.15 * drawer._box_length)
        assert arc.theta1 == 180
        assert arc.theta2 == 0
        assert allclose(arc.height, 0.55 * drawer._box_length)
        assert arc.width == 0.6 * drawer._box_length

        arrow = drawer.ax.patches[2]
        assert isinstance(arrow, FancyArrow)
        assert len(drawer.ax.texts) == 0
        plt.close("all")

    def test_measure_multiple_wires(self):
        """Tests the measure method when multiple wires are provided."""

        drawer = MPLDrawer(n_layers=1, wire_map={0: 0, 1: 1})
        drawer.measure(layer=0, wires=(0, 1))

        box = drawer.ax.patches[0]
        assert box.get_x() == -drawer._box_length / 2.0 + drawer._pad
        assert box.get_y() == -drawer._box_length / 2.0 + drawer._pad
        assert box.get_width() == drawer._box_length - 2 * drawer._pad
        assert box.get_height() == drawer._box_length - 2 * drawer._pad + 1

        arc = drawer.ax.patches[1]
        assert arc.center == (0, 0.5 + 0.15 * drawer._box_length)
        assert arc.theta1 == 180
        assert arc.theta2 == 0
        assert allclose(arc.height, 0.55 * drawer._box_length)
        assert arc.width == 0.6 * drawer._box_length

        arrow = drawer.ax.patches[2]
        assert isinstance(arrow, FancyArrow)
        assert len(drawer.ax.texts) == 0
        plt.close("all")

    def test_measure_classical_wires(self):
        """Tests the measure method when multiple wires are provided."""

        drawer = MPLDrawer(n_layers=1, wire_map={0: 0, 1: 1}, c_wires=2)
        drawer.measure(layer=0, wires=(2, 3))

        box = drawer.ax.patches[0]
        assert box.get_x() == -drawer._box_length / 2.0 + drawer._pad
        assert box.get_y() == 2 - drawer._box_length / 2.0 + drawer._pad
        assert box.get_width() == drawer._box_length - 2 * drawer._pad
        assert box.get_height() == drawer._box_length - 2 * drawer._pad + drawer._cwire_scaling

        arc = drawer.ax.patches[1]
        assert arc.center == (0, 2 + drawer._cwire_scaling / 2 + 0.15 * drawer._box_length)
        assert arc.theta1 == 180
        assert arc.theta2 == 0
        assert allclose(arc.height, 0.55 * drawer._box_length)
        assert arc.width == 0.6 * drawer._box_length

        arrow = drawer.ax.patches[2]
        assert isinstance(arrow, FancyArrow)
        assert len(drawer.ax.texts) == 0
        plt.close("all")

    def test_measure_text(self):
        """Test adding a postselection label to a measure box."""
        drawer = MPLDrawer(1, {0: 0})
        drawer.measure(0, 0, text="0")
        assert len(drawer.ax.texts) == 1
        assert drawer.ax.texts[0].get_text() == "0"
        assert drawer.ax.texts[0].get_position() == (0.05 * 0.75, 0.225)
        plt.close("all")

    def test_measure_formatted(self):
        """Tests you can color the measure box"""

        drawer = MPLDrawer(1, {0: 0})
        rgba_red = (1.0, 0, 0, 1.0)
        rgba_green = (0, 1, 0, 1)
        box_options = {"facecolor": rgba_red, "edgecolor": rgba_green}
        lines_options = {"color": rgba_green, "linewidth": 0.5}
        drawer.measure(0, 0, box_options=box_options, lines_options=lines_options)

        box = drawer.ax.patches[0]
        assert box.get_facecolor() == rgba_red
        assert box.get_edgecolor() == rgba_green

        arc = drawer.ax.patches[1]
        assert arc.get_edgecolor() == rgba_green
        assert arc.get_linewidth() == 0.5

        arrow = drawer.ax.patches[2]
        assert arrow.get_edgecolor() == rgba_green
        assert arrow.get_linewidth() == 0.5

        plt.close("all")


class TestAutosize:
    """Test the autosize keyword of the `box_gate` method"""

    def text_in_box(self, drawer):
        """This utility determines the last text drawn is inside the last patch drawn.
        This is done over and over in this test class, and so extracted for convenience.

        This is a complimentary approach to comparing sizing to that used in the drawer
        class `text_dims` method
        """

        text = drawer.ax.texts[-1]
        rect = drawer.ax.patches[-1]

        renderer = drawer.fig.canvas.get_renderer()

        # https://matplotlib.org/stable/api/_as_gen/matplotlib.artist.Artist.get_window_extent.html
        text_bbox = text.get_window_extent(renderer)
        rect_bbox = rect.get_window_extent(renderer)

        # check all text corners inside rectangle
        # https://matplotlib.org/stable/api/transformations.html
        return all(rect_bbox.contains(*p) for p in text_bbox.corners())

    def test_autosize_false(self):
        """Test that the text is unchanged if autosize is set to False."""

        drawer = MPLDrawer(n_layers=1, wire_map={0: 0, 1: 1})
        drawer.box_gate(0, (0, 1), text="very very long text", autosize=False)

        t = drawer.ax.texts[0]
        assert t.get_rotation() == 0
        assert t.get_fontsize() == drawer._fontsize

        plt.close("all")

    def test_autosize_one_wire(self):
        """Test case where the box is on only one wire.  The text should still
        be inside the box, but not rotated."""

        drawer = MPLDrawer(n_layers=1, wire_map={0: 0})
        drawer.box_gate(0, 0, text="very very long text", autosize=True)

        t = drawer.ax.texts[0]
        assert t.get_rotation() == 0

        assert self.text_in_box(drawer)

        plt.close("all")

    def test_autosize_multiwires(self):
        """Test case where the box is on multiple wires.  The text should
        be rotated 90deg, and still inside the box."""

        drawer = MPLDrawer(n_layers=1, wire_map={0: 0, 1: 1})
        drawer.box_gate(0, (0, 1), text="very very long text", autosize=True)

        t = drawer.ax.texts[0]
        assert t.get_rotation() == 90.0

        assert self.text_in_box(drawer)

        plt.close("all")

    def test_multiline_text_single_wire(self):
        """Test case where the box is on one wire and the text is skinny and tall.
        If the text is too tall, it should still be shrunk to fit inside the box."""

        drawer = MPLDrawer(n_layers=1, wire_map={0: 0})
        drawer.box_gate(0, 0, text="text\nwith\nall\nthe\nlines\nyep", autosize=True)

        t = drawer.ax.texts[0]
        assert t.get_rotation() == 0.0

        assert self.text_in_box(drawer)

        plt.close("all")

    def text_tall_multitline_text_multiwires(self):
        """Test case where the box is on mutiple wires and the text is skinny and tall.
        If the text is just too tall and the width fits inside the box, the text should not
        be rotated."""

        drawer = MPLDrawer(n_layers=1, wire_map={0: 0, 1: 1})
        drawer.box_gate(
            0,
            (0, 1),
            text="text\nwith\nall\nthe\nlines\nyep\ntoo\nmany\nlines\nway\ntoo\nmany",
            autosize=True,
        )

        t = drawer.ax.texts[0]
        assert t.get_rotation() == 0.0

        assert self.text_in_box(drawer)

        plt.close("all")

    def test_wide_multline_text_multiwires(self):
        """Test case where the box is on multiple wires and text is fat, tall,
        and fatter than it is tall. It should be rotated."""

        drawer = MPLDrawer(n_layers=1, wire_map={0: 0, 1: 1})
        drawer.box_gate(0, (0, 1), text="very very long text\nall\nthe\nlines\nyep", autosize=True)

        assert self.text_in_box(drawer)

        plt.close("all")


class TestClassicalWires:
    def test_classical_wire(self):
        """Test the addition of horiziontal classical wires."""
        drawer = MPLDrawer(wire_map={0: 0}, n_layers=4, c_wires=3)

        layers = [0, 0, 1, 1]
        wires = [0, 1, 1, 0]
        drawer.classical_wire(layers, wires)

        [_, cwire] = drawer.ax.lines
        assert cwire.get_xdata() == layers
        assert cwire.get_ydata() == [0, 1, 1, 0]  # cwires are scaledc

        [pe1, pe2] = cwire.get_path_effects()

        # probably not a good way to test this, but the best I can figure out
        assert pe1._gc == {
            "linewidth": 5 * plt.rcParams["lines.linewidth"],
            "foreground": plt.rcParams["lines.color"],
        }
        assert pe2._gc == {
            "linewidth": 3 * plt.rcParams["lines.linewidth"],
            "foreground": plt.rcParams["figure.facecolor"],
        }

        plt.close("all")

    def test_cwire_join(self):
        """Test the cwire join method."""
        drawer = MPLDrawer(wire_map={0: 0}, n_layers=4, c_wires=3)

        drawer.cwire_join(1, 2)

        [_, eraser] = drawer.ax.lines

        assert eraser.get_xdata() == (0.8, 1)
        assert eraser.get_ydata() == (1.25, 1.25)
        assert eraser.get_color() == plt.rcParams["figure.facecolor"]
        assert eraser.get_linewidth() == 3 * plt.rcParams["lines.linewidth"]
        plt.close("all")

    def test_cwire_join_erase_right(self):
        """Test the cwire join method."""
        drawer = MPLDrawer(wire_map={0: 0}, n_layers=4, c_wires=3)

        drawer.cwire_join(1, 1, erase_right=True)

        [_, eraser] = drawer.ax.lines

        assert eraser.get_xdata() == (0.8, 1.2)
        assert eraser.get_ydata() == (1, 1)
        assert eraser.get_color() == plt.rcParams["figure.facecolor"]
        assert eraser.get_linewidth() == 3 * plt.rcParams["lines.linewidth"]
        plt.close("all")


class TestCond:
    """Test the cond double-wire drawing function."""

    def test_cond_basic(self):
        """Tests cond from one wire to the next."""
        drawer = MPLDrawer(wire_map={0: 0, 1: 1}, n_layers=2)
        wire_data_before = [line.get_data() for line in drawer._wire_lines]

        drawer.cond(layer=1, measured_layer=0, wires=[0], wires_target=[1])
        actual_data = [line.get_data() for line in drawer.ax.lines]

        assert actual_data == [
            ((-1, 2), (0, 0)),
            ((-1, 2), (1, 1)),
            ((0.375, 1.03), (-0.03, -0.03)),
            ((0.375, 0.97), (0.03, 0.03)),
            ((1.03, 1.03), (-0.03, 1)),
            ((0.97, 0.97), (0.03, 1)),
        ]
        assert [line.get_data() for line in drawer._wire_lines] == wire_data_before
        plt.close("all")

    def test_cond_two_ctrl_wires(self):
        """Tests cond from two separated wires."""
        drawer = MPLDrawer(wire_map={i: i for i in range(4)}, n_layers=2)
        drawer.cond(layer=1, measured_layer=0, wires=[0, 2], wires_target=[3])
        actual_data = [line.get_data() for line in drawer.ax.lines]

        assert actual_data == [
            ((-1, 2), (0, 0)),
            ((-1, 2), (1, 1)),
            ((-1, 2), (2, 2)),
            ((-1, 2), (3, 3)),
            ((0.375, 1.03), (-0.03, -0.03)),
            ((0.375, 0.97), (0.03, 0.03)),
            ((1.03, 1.03), (-0.03, 3)),
            ((0.97, 0.97), (2.03, 3)),
            ((0.97, 0.97), (0.03, 1.97)),
            ((0.375, 0.97), (1.97, 1.97)),
            ((0.375, 0.97), (2.03, 2.03)),
        ]
        plt.close("all")

    def test_cond_two_ctrl_wires_upward(self):
        """Test cond when the conditional operation is above the control wires."""
        drawer = MPLDrawer(wire_map={i: i for i in range(3)}, n_layers=2)
        drawer.cond(layer=1, measured_layer=0, wires=[1, 2], wires_target=[0])
        actual_data = [line.get_data() for line in drawer.ax.lines]
        assert actual_data == [
            ((-1, 2), (0, 0)),
            ((-1, 2), (1, 1)),
            ((-1, 2), (2, 2)),
            ((0.375, 1.03), (2.03, 2.03)),
            ((0.375, 0.97), (1.97, 1.97)),
            ((1.03, 1.03), (2.03, 0)),
            ((0.97, 0.97), (0.97, 0)),
            ((0.97, 0.97), (1.03, 1.97)),
            ((0.375, 0.97), (0.97, 0.97)),
            ((0.375, 0.97), (1.03, 1.03)),
        ]
        plt.close("all")

    @pytest.mark.parametrize(
        "ctrl_wires, target_wires",
        [
            ((1,), (0, 2)),
            ((1, 3), (0, 2)),
            ((0, 2), (1,)),
            ((0, 2), (1, 3)),
        ],
    )
    def test_cond_fail_with_bad_order(self, ctrl_wires, target_wires):
        """Tests cond raises an error when the wires aren't neatly separated."""
        drawer = MPLDrawer(wire_map={i: i for i in range(4)}, n_layers=2)
        with pytest.raises(ValueError, match="Cannot draw interspersed mid-circuit measurements"):
            drawer.cond(layer=1, measured_layer=0, wires=ctrl_wires, wires_target=target_wires)
        plt.close("all")
