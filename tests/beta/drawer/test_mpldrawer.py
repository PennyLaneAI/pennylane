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
Tests the mpldrawer in the beta module.  
"""

from numpy import recfromtxt
import pytest

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from pennylane.beta.drawer import MPLDrawer
from pennylane.math import allclose


class TestInitialization:

    @pytest.mark.parametrize('n_wires', [2, 3])
    @pytest.mark.parametrize('n_layers', [2,3])
    def test_figsize_wires(self, n_wires, n_layers):
        """Tests the figure is sized correctly."""

        drawer = MPLDrawer(n_wires=n_wires, n_layers=n_layers)

        assert drawer.fig.get_figwidth() == (n_layers + 3)
        assert drawer.fig.get_figheight() == (n_wires + 1)

        drawer = MPLDrawer(n_wires=n_wires, n_layers=n_layers)

        lines = drawer.ax.lines

        assert len(lines) == n_wires

        for wire, line in enumerate(lines):
            assert line.get_xdata() == (-1, n_layers)
            assert line.get_ydata() == (wire, wire)

    def test_customfigsize(self):
        """Tests a custom figsize alters the size"""

        drawer = MPLDrawer(1,1, figsize=(5,5))

        assert drawer.fig.get_figwidth() == 5
        assert drawer.fig.get_figheight() == 5

    def test_config_params_set(self):
        """Tests sizing hidden variables are set."""

        drawer = MPLDrawer(1,1)

        assert drawer._box_dx == 0.4
        assert drawer._circ_rad == 0.3
        assert drawer._ctrl_rad == 0.1
        assert drawer._swap_dx == 0.2


def test_labels():
    """Tests labels are added"""

    drawer = MPLDrawer(1, 3)

    labels = ("a", "b", "c")
    drawer.label(labels)

    drawn_labels = drawer.ax.texts

    for wire, expected_label, actual_label in zip(range(3), labels, drawn_labels):

        assert actual_label.get_text() == expected_label

        assert actual_label.get_position() == (-1.5, wire)

class TestBoxGate():

    def test_simple_box(self):
        """tests basic functionality of box_gate."""

        drawer = MPLDrawer(1,1)

        drawer.box_gate(0, 0, "X")

        rect = drawer.ax.patches[0]

        assert rect.get_xy() == (-0.4, -0.4)
        assert rect.get_width() == 0.8
        assert rect.get_height() == 0.8

        text = drawer.ax.texts[0]

        assert text.get_text() == "X"
        assert text.get_position() == (0,0)

    def test_multiwire_box(self):
        """tests a gate spanning multiple wires."""

        drawer = MPLDrawer(1, 3)
        drawer.box_gate(0, (0,2), text="Tall Gate")

        rect = drawer.ax.patches[0]

        assert rect.get_xy() == (-0.4, -0.4)
        assert rect.get_width() == 0.8
        assert rect.get_height() == 2.8

        text = drawer.ax.texts[0]

        assert text.get_text() == "Tall Gate"
        assert text.get_position() == (0,1.0)

    def test_extra_width(self):
        """tests a box with added width."""

        drawer = MPLDrawer(1,1)
        drawer.box_gate(0,0, text="Wide Gate", extra_width=0.4)

        rect = drawer.ax.patches[0]

        assert allclose(rect.get_xy(), (-0.6, -0.4))
        assert rect.get_height() == 0.8
        assert allclose(rect.get_width(), 1.2)

        text = drawer.ax.texts[0]

        assert text.get_text() == "Wide Gate"
        assert text.get_position() == (0, 0)

    def test_rotate_text(self):
        """Tests rotated text"""

        drawer = MPLDrawer(1,1)
        drawer.box_gate(0,0, text="rotated text", rotate_text=True)

        text = drawer.ax.texts[0]
        assert text.get_rotation() == 90.0

class TestCTRL:
    def test_ctrl_no_target(self):
        """Tests a single control with no target"""

        drawer = MPLDrawer(1, 1)

        drawer.ctrl(0, 0)

        ctrl_line = drawer.ax.lines[1]

        assert ctrl_line.get_data() == ((0,0), (0,0))

        assert len(drawer.ax.patches) == 1

        circle = drawer.ax.patches[0]

        assert circle.width == 0.2
        assert circle.center == (0,0)

    def test_ctrl_multi_wires(self):
        """Tests two control wires with no target."""

        drawer = MPLDrawer(1, 3)

        ctrl_wires= (0,1)
        drawer.ctrl(0, ctrl_wires)

        ctrl_line = drawer.ax.lines[3]

        assert ctrl_line.get_data() == ((0,0), ctrl_wires)

        circles = drawer.ax.patches

        assert len(circles) == 2

        for wire, circle in zip(ctrl_wires, circles):
            assert circle.width == 0.2
            assert circle.center == (0, wire)

    def test_ctrl_target(self):
        """Tests target impacts line extent"""

        drawer = MPLDrawer(1,3)

        drawer.ctrl(0, 0, 2)

        ctrl_line = drawer.ax.lines[3]

        assert ctrl_line.get_data() == ((0,0), (0, 2))

        circles = drawer.ax.patches
        assert len(circles) == 1

        circle = drawer.ax.patches[0]

        assert circle.width == 0.2
        assert circle.center == (0,0)

    def test_target_x(self):
        """Tests hidden target_x drawing method"""

        drawer = MPLDrawer(1,3)
        
        drawer._target_x(0, 0)

        center_line = drawer.ax.lines[3]
        assert center_line.get_data() == ((0,0), (-0.3, 0.3))

        circle = drawer.ax.patches[0]

        assert circle.center == (0,0)
        assert circle.width == 0.6
        assert circle.fill == False
        assert to_rgba(plt.rcParams["lines.color"]) == to_rgba(circle.get_edgecolor())

    def test_CNOT(self,):
        """Tests the CNOT method"""

        drawer = MPLDrawer(1, 3)

        drawer.CNOT(0, (0,1))

        ctrl_line = drawer.ax.lines[3]
        assert ctrl_line.get_data() == ((0,0), (0,1))

        center_line = drawer.ax.lines[4]
        assert center_line.get_data() == ((0,0), (0.7, 1.3))

        ctrl_circle = drawer.ax.patches[0]
        target_circle = drawer.ax.patches[1]

        assert ctrl_circle.center == (0,0)
        assert ctrl_circle.width == 0.2

        assert target_circle.center == (0,1)
        assert target_circle.width == 0.6
        assert target_circle.fill == False
        assert to_rgba(plt.rcParams["lines.color"]) == to_rgba(target_circle.get_edgecolor())




