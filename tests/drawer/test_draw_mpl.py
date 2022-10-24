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


def test_standard_use():
    """Tests standard usage produces expected figure and axes"""

    # not constructed before calling
    fig, ax = qml.draw_mpl(circuit1)(1.23, 2.34)

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes._axes.Axes)

    # proxy for whether correct things were drawn
    assert len(ax.patches) == 7  # two boxes, 2 circles for CNOT, 3 patches for measure
    assert len(ax.lines) == 6  # three wires, three lines for CNOT
    assert len(ax.texts) == 5  # three wire labels, 2 box labels

    assert ax.texts[0].get_text() == "0"
    assert ax.texts[1].get_text() == "a"
    assert ax.texts[2].get_text() == "1.23"

    # gates in same layer can be in any order

    texts = [t.get_text() for t in ax.texts[3:]]
    assert "RX" in texts
    assert "RY" in texts
    plt.close()


@pytest.mark.parametrize(
    "strategy, initial_strategy, n_lines", [("gradient", "device", 3), ("device", "gradient", 13)]
)
def test_expansion_strategy(strategy, initial_strategy, n_lines):
    """Test that the expansion strategy keyword controls what operations are drawn."""

    @qml.qnode(qml.device("default.qubit", wires=3), expansion_strategy=initial_strategy)
    def circuit():
        qml.Permute([2, 0, 1], wires=(0, 1, 2))
        return qml.expval(qml.PauliZ(0))

    fig, ax = qml.draw_mpl(circuit, expansion_strategy=strategy)()

    assert len(ax.lines) == n_lines
    assert circuit.expansion_strategy == initial_strategy
    plt.close()


class TestKwargs:
    """Test various keywords arguments that can be passed to modify graphic."""

    def test_fontsize(self):
        """Test fontsize set by keyword argument."""

        _, ax = qml.draw_mpl(circuit1, fontsize=20)(1.234, 1.234)
        for t in ax.texts:
            assert t.get_fontsize() == 20
        plt.close()

    def test_decimals(self):
        """Test decimals changes operation labelling"""

        _, ax = qml.draw_mpl(circuit1, decimals=2)(1.23, 2.34)

        texts = [t.get_text() for t in ax.texts[3:]]
        assert "RX\n(1.23)" in texts
        assert "RY\n(2.34)" in texts
        plt.close()

    def test_label_options(self):
        """Test label options modifies label style."""

        _, ax = qml.draw_mpl(circuit1, label_options={"color": "purple", "fontsize": 20})(
            1.23, 2.34
        )

        for l in ax.texts[0:3]:  # three labels
            assert l.get_color() == "purple"
            assert l.get_fontsize() == 20
        plt.close()

    @pytest.mark.parametrize(
        "notches, n_patches",
        [
            (True, 8),
            (False, 4),
        ],  # 4 notches, 3 measurement, 1 box  # 1 box, 3 measurements
    )
    def test_active_wire_notches(self, notches, n_patches):
        """Test active wire notches can be toggled by keyword."""

        @qml.qnode(dev)
        def temp_circ():
            qml.QFT(wires=(0, 1.23))
            return qml.probs(0)

        _, ax = qml.draw_mpl(temp_circ, show_all_wires=True, active_wire_notches=notches)()

        assert len(ax.patches) == n_patches
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

        assert len(ax.lines) == 1  # one wire
        assert len(ax.texts) == 2  # one wire label and one gate label
        assert ax.texts[0].get_text() == "0"
        assert ax.texts[1].get_text() == "RX"
        plt.close()

    def test_show_all_wires(self):
        """Test show_all_wires=True displays empty wires."""

        _, ax = qml.draw_mpl(circuit2, show_all_wires=True)(1.23)

        assert len(ax.lines) == 3  # three wires

        assert len(ax.texts) == 4  # three wire labels and one gate label
        assert ax.texts[0].get_text() == "0"
        assert ax.texts[1].get_text() == "a"
        assert ax.texts[2].get_text() == "1.23"
        plt.close()

    def test_wire_order_not_on_device(self):
        """Test when ``wire_order`` priority by requesting ``show_all_wires`` with
        the ``wire_order`` containing wires not on the device. The output should have
        three wires, one active and two empty wires from the wire order."""

        _, ax = qml.draw_mpl(circuit2, wire_order=[2, "a"], show_all_wires=True)(1.23)

        assert len(ax.lines) == 3  # three wires

        assert len(ax.texts) == 4  # three wire labels and one gate label
        assert ax.texts[0].get_text() == "2"
        assert ax.texts[1].get_text() == "a"
        assert ax.texts[2].get_text() == "0"
        plt.close()

    def test_wire_options(self):
        """Test wire options modifies wire styling"""

        _, ax = qml.draw_mpl(circuit1, wire_options={"color": "black", "linewidth": 4})(1.23, 2.34)

        for w in ax.lines[0:3]:  # three wires
            assert w.get_color() == "black"
            assert w.get_linewidth() == 4

        plt.close()


class TestMPLIntegration:
    """Test using matplotlib styling to modify look of graphic."""

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

        expected_linecolor = mpl.colors.to_rgba(plt.rcParams["lines.color"])
        for l in ax.lines[0:-1]:  # final is fancy arrow, has different styling
            assert mpl.colors.to_rgba(l.get_color()) == expected_linecolor

        plt.style.use("default")
        plt.close()
