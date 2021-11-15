# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the draw_mpl transform.

See section on "Testing Matplotlib based code" in the "Software Tests"
page in the developement guide.
"""

import pytest

import pennylane as qml

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")

dev = qml.device("default.qubit", wires=(0, "a", 1.23))


@qml.qnode(dev)
def circuit1(x, y):
    qml.RX(x, wires=0)
    qml.CNOT(wires=(0, "a"))
    qml.RY(y, wires=1.23)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev)
def circuit2(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))


def test_default():
    """Tests standard usage produces expected figure and axes"""

    # not constructed before calling
    fig, ax = qml.draw_mpl(circuit1)(1.23, 2.34)

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes._axes.Axes)

    # proxy for whether correct things were drawn
    assert len(ax.patches) == 7
    assert len(ax.lines) == 7
    assert len(ax.texts) == 5

    assert ax.texts[0].get_text() == "0"
    assert ax.texts[1].get_text() == "a"
    assert ax.texts[2].get_text() == "1.23"

    # gates in same layer can be in any order

    texts = [t.get_text() for t in ax.texts[3:]]
    assert "RX" in texts
    assert "RY" in texts
    plt.close()


def test_decimals():
    """Test decimals changes operation labelling"""

    _, ax = qml.draw_mpl(circuit1, decimals=2)(1.23, 2.34)

    texts = [t.get_text() for t in ax.texts[3:]]
    assert "RX\n(1.23)" in texts
    assert "RY\n(2.34)" in texts
    plt.close()


def test_label_options():
    """Test label options modifies label style."""

    _, ax = qml.draw_mpl(circuit1, label_options={"color": "purple", "fontsize": 20})(1.23, 2.34)

    for l in ax.texts[0:3]:  # three labels
        assert l.get_color() == "purple"
        assert l.get_fontsize() == 20
    plt.close()


class TestWireBehaviour:
    """Tests that involve how wires are displayed"""

    def test_wire_order(self):
        """Test wire_order changes order of wires"""

        _, ax = qml.draw_mpl(circuit1, wire_order=(1.23, "a"))(1.23, 2.34)

        assert len(ax.texts) == 5

        assert ax.texts[0].get_text() == "1.23"
        assert ax.texts[1].get_text() == "a"
        assert ax.texts[2].get_text() == "0"
        plt.close()

    def test_empty_wires(self):
        """Test empty wires do not appear by default"""

        _, ax = qml.draw_mpl(circuit2)(1.23)

        assert len(ax.lines) == 2  # one wire label, one gate label
        assert ax.texts[0].get_text() == "0"
        assert ax.texts[1].get_text() == "RX"
        plt.close()

    def test_show_all_wires(self):
        """Test show_all_wires=True displays empty wires."""

        _, ax = qml.draw_mpl(circuit2, show_all_wires=True)(1.23)

        # three wires plus one gate label
        assert len(ax.lines) == 4
        assert ax.texts[0].get_text() == "0"
        assert ax.texts[1].get_text() == "a"
        assert ax.texts[2].get_text() == "1.23"
        plt.close()

    def test_wire_options(self):
        """Test wire options modifies wire styling"""

        _, ax = qml.draw_mpl(circuit1, wire_options={"color": "black", "linewidth": 4})(1.23, 2.34)

        for w in ax.lines[0:3]:  # three wires
            assert w.get_color() == "black"
            assert w.get_linewidth() == 4

        plt.close()


class TestMPLIntegration:
    def test_rcparams(self):
        """Test setting rcParams modifies style."""

        rgba_red = (1, 0, 0, 1)
        rgba_green = (0, 1, 0, 1)
        plt.rcParams["patch.facecolor"] = rgba_red
        plt.rcParams["lines.color"] = rgba_green

        _, ax = qml.draw_mpl(circuit1)(1.23, 2.34)

        assert ax.patches[0].get_facecolor() == rgba_red
        assert ax.patches[1].get_facecolor() == rgba_red

        for l in ax.lines[0:-1]:  # final is fancy arrow, has different styling
            assert l.get_color() == rgba_green

        plt.style.use("default")
        plt.close()

    def test_style(self):
        """Test matplotlib styles impact figure styling."""

        plt.style.use("fivethirtyeight")

        _, ax = qml.draw_mpl(circuit1)(1.23, 2.34)

        expected_facecolor = mpl.colors.to_rgba(plt.rcParams["patch.facecolor"])
        assert ax.patches[0].get_facecolor() == expected_facecolor
        assert ax.patches[1].get_facecolor() == expected_facecolor

        expected_linecolor = mpl.colors.to_rgba(plt.rcParmas["lines.color"])
        for l in ax.lines[0:-1]:  # final is fancy arrow, has different styling
            assert mpl.colors.to_rgba(l.get_color()) == expected_linecolor

        plt.style.use("default")
        plt.close()
