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
from pennylane import numpy as pnp

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")

dev = qml.device("default.qubit", wires=(0, "a", 1.23))


@qml.qnode(dev)
def circuit1(x, y):
    """Circuit on three qubits."""
    qml.RX(x, wires=0)
    qml.CNOT(wires=(0, "a"))
    qml.RY(y, wires=1.23)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev)
def circuit2(x):
    """Circuit on a single qubit."""
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))


def test_standard_use():
    """Tests standard usage produces expected figure and axes"""

    # not constructed before calling
    fig, ax = qml.draw_mpl(circuit1)(1.23, 2.34)

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes._axes.Axes)  # pylint:disable=protected-access

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
    plt.close("all")


def test_fig_argument():
    """Tests figure argument is used correcly"""

    fig = plt.figure()
    output_fig, ax = qml.draw_mpl(circuit1, fig=fig)(1.23, 2.34)
    assert ax.get_figure() == fig
    assert output_fig == fig
    plt.close("all")


class TestLevelExpansionStrategy:
    @pytest.fixture
    def transforms_circuit(self):
        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device("default.qubit"), diff_method="parameter-shift")
        def circ(weights, order):
            qml.RandomLayers(weights, wires=(0, 1))
            qml.Permute(order, wires=(0, 1, 2))
            qml.PauliX(0)
            qml.PauliX(0)
            qml.RX(0.1, wires=0)
            qml.RX(-0.1, wires=0)
            return qml.expval(qml.PauliX(0))

        return circ

    @pytest.mark.parametrize(
        "levels,expected_metadata",
        [
            ((0, "top"), (3, 9, 9)),
            ((2, "user"), (3, 5, 5)),
            ((3, "gradient"), (3, 6, 6)),
            ((8, "device"), (8, 5, 5)),
        ],
    )
    def test_equivalent_levels(self, transforms_circuit, levels, expected_metadata):
        """Test that the level keyword controls what operations are drawn."""
        var1, var2 = levels
        expected_lines, expected_patches, expected_texts = expected_metadata

        order = [2, 1, 0]
        weights = pnp.array([[1.0, 20]])

        _, ax1 = qml.draw_mpl(transforms_circuit, level=var1)(weights, order)
        _, ax2 = qml.draw_mpl(transforms_circuit, level=var2)(weights, order)

        assert len(ax1.lines) == len(ax2.lines) == expected_lines
        assert len(ax1.patches) == len(ax2.patches) == expected_patches
        assert len(ax1.texts) == len(ax2.texts) == expected_texts

        plt.close("all")

    def test_draw_at_level_1(self, transforms_circuit):
        """Test that at level one the first transform has been applied, cancelling inverses."""

        order = [2, 1, 0]
        weights = pnp.array([[1.0, 20]])

        _, ax = qml.draw_mpl(transforms_circuit, level=1)(weights, order)

        assert len(ax.lines) == 3
        assert len(ax.patches) == 7
        assert len(ax.texts) == 7

    def test_draw_with_qfunc_warns_with_level(self):
        """Test that draw warns the user about level being ignored."""

        def qfunc():
            qml.PauliZ(0)

        with pytest.warns(UserWarning, match="the level argument is ignored"):
            qml.draw_mpl(qfunc, level=None)

    def test_split_tapes_raises_warning(self):
        @qml.transforms.split_non_commuting
        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit():
            return [qml.expval(qml.X(0)), qml.expval(qml.Z(0))]

        with pytest.warns(UserWarning, match="Multiple tapes constructed"):
            qml.draw_mpl(circuit)()
        plt.close("all")


class TestKwargs:
    """Test various keywords arguments that can be passed to modify graphic."""

    def test_fontsize(self):
        """Test fontsize set by keyword argument."""

        _, ax = qml.draw_mpl(circuit1, fontsize=20)(1.234, 1.234)
        for t in ax.texts:
            assert t.get_fontsize() == 20
        plt.close("all")

    def test_decimals(self):
        """Test decimals changes operation labelling"""

        _, ax = qml.draw_mpl(circuit1, decimals=2)(1.23, 2.34)

        texts = [t.get_text() for t in ax.texts[3:]]
        assert "RX\n(1.23)" in texts
        assert "RY\n(2.34)" in texts
        plt.close("all")

    def test_label_options(self):
        """Test label options modifies label style."""

        _, ax = qml.draw_mpl(circuit1, label_options={"color": "purple", "fontsize": 20})(
            1.23, 2.34
        )

        for l in ax.texts[:3]:  # three labels
            assert l.get_color() == "purple"
            assert l.get_fontsize() == 20
        plt.close("all")

    def test_hide_wire_labels(self):
        """Test that wire labels are skipped with show_wire_labels=False."""
        fig, ax = qml.draw_mpl(circuit1, show_wire_labels=False)(1.23, 2.34)
        fig_with_labels, ax_with_labels = qml.draw_mpl(circuit1)(1.23, 2.34)

        # Only PauliX gate labels should be present
        assert len(ax.texts) == 2
        assert len(ax_with_labels.texts) == 2 + 3
        assert ax.texts[0].get_text() == "RX"
        assert ax.texts[1].get_text() == "RY"
        assert fig.get_figwidth() == 4
        assert fig_with_labels.get_figwidth() == 4 + 1

        plt.close("all")

    @pytest.mark.parametrize(
        "notches, n_patches",
        [
            (True, 8),  # 4 notches, 3 measurement, 1 box
            (False, 4),  # 1 box, 3 measurements
        ],
    )
    def test_active_wire_notches(self, notches, n_patches):
        """Test active wire notches can be toggled by keyword."""

        @qml.qnode(dev)
        def temp_circ():
            qml.QFT(wires=(0, 1.23))
            return qml.probs(0)

        _, ax = qml.draw_mpl(temp_circ, show_all_wires=True, active_wire_notches=notches)()

        assert len(ax.patches) == n_patches
        plt.close("all")

    def test_black_white_is_default_style(self):
        """Test that if no style is specified, the black_white style is the default for mpl_draw,
        rather than general matplotlib settings."""

        _, ax = qml.draw_mpl(circuit1)(1.234, 1.234)

        assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
        assert ax.patches[4].get_facecolor() == (1.0, 1.0, 1.0, 1.0)
        assert ax.patches[4].get_edgecolor() == (0.0, 0.0, 0.0, 1.0)

    def test_style(self):
        """Test style is set by keyword argument."""

        _, ax = qml.draw_mpl(circuit1, style="sketch")(1.234, 1.234)

        assert ax.get_facecolor() == (
            0.8392156862745098,
            0.9607843137254902,
            0.8862745098039215,
            1.0,
        )
        assert ax.patches[0].get_edgecolor() == (0.0, 0.0, 0.0, 1.0)
        assert ax.patches[0].get_facecolor() == (1.0, 0.9333333333333333, 0.8313725490196079, 1.0)
        assert ax.patches[2].get_facecolor() == (0.0, 0.0, 0.0, 1.0)
        assert ax.patches[3].get_facecolor() == (
            0.8392156862745098,
            0.9607843137254902,
            0.8862745098039215,
            1.0,
        )

    @pytest.mark.parametrize("as_qnode", (True, False))
    def test_max_length(self, as_qnode):
        """Test that long circuits can be broken up into multiple figures."""

        def c():
            for _ in range(15):
                qml.X(0)
            return qml.expval(qml.Z(0))

        if as_qnode:
            c = qml.QNode(c, qml.device("default.qubit"))

        figs_and_axes = qml.draw_mpl(c, max_length=5)()
        assert len(figs_and_axes) == 3
        for i in range(3):
            assert isinstance(figs_and_axes[i][0], plt.Figure)
            assert isinstance(figs_and_axes[i][1], plt.Axes)

        ax0 = figs_and_axes[0][1]
        ax1 = figs_and_axes[1][1]
        ax2 = figs_and_axes[2][1]
        assert len(ax0.patches) == 5
        assert len(ax1.patches) == 5
        assert len(ax2.patches) == 8  # three for measure box

        assert ax0.texts[-1].get_text() == "···"
        assert ax1.texts[-1].get_text() == "···"
        assert ax2.texts[-1].get_text() == "X"

        assert ax1.texts[1].get_text() == "···"
        assert ax2.texts[1].get_text() == "···"


class TestWireBehaviour:
    """Tests that involve how wires are displayed"""

    @pytest.mark.parametrize("use_qnode", (True, False))
    def test_wire_order(self, use_qnode):
        """Test wire_order changes order of wires"""

        def f(x, y):
            """Circuit on three qubits."""
            qml.RX(x, wires=0)
            qml.CNOT(wires=(0, "a"))
            qml.RY(y, wires=1.23)
            return qml.expval(qml.PauliZ(0))

        if use_qnode:
            f = qml.QNode(f, qml.device("default.qubit", wires=(0, "a", 1.23)))

        _, ax = qml.draw_mpl(f, wire_order=(1.23, "a"))(1.23, 2.34)

        assert len(ax.texts) == 5

        assert ax.texts[0].get_text() == "1.23"
        assert ax.texts[1].get_text() == "a"
        assert ax.texts[2].get_text() == "0"
        plt.close("all")

    def test_empty_wires(self):
        """Test empty wires do not appear by default"""

        _, ax = qml.draw_mpl(circuit2)(1.23)

        assert len(ax.lines) == 1  # one wire
        assert len(ax.texts) == 2  # one wire label and one gate label
        assert ax.texts[0].get_text() == "0"
        assert ax.texts[1].get_text() == "RX"
        plt.close("all")

    def test_show_all_wires(self):
        """Test show_all_wires=True displays empty wires."""

        _, ax = qml.draw_mpl(circuit2, show_all_wires=True)(1.23)

        assert len(ax.lines) == 3  # three wires

        assert len(ax.texts) == 4  # three wire labels and one gate label
        assert ax.texts[0].get_text() == "0"
        assert ax.texts[1].get_text() == "a"
        assert ax.texts[2].get_text() == "1.23"
        plt.close("all")

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
        plt.close("all")

    def test_uniform_wire_options(self):
        """Test wire options modifies wire styling"""

        _, ax = qml.draw_mpl(circuit1, wire_options={"color": "black", "linewidth": 4})(1.23, 2.34)

        for w in ax.lines[:3]:  # three wires
            assert w.get_color() == "black"
            assert w.get_linewidth() == 4

        plt.close("all")

    def test_individual_wire_options(self):
        """Test wire option styling when individual wires have their own options specified"""

        @qml.qnode(dev)
        def f_circ(x):
            """Circuit on ten qubits."""
            qml.RX(x, wires=0)
            for w in range(10):
                qml.Hadamard(w)
            return qml.expval(qml.PauliZ(0) @ qml.PauliY(1))

        # All wires are orange
        wire_options = {"color": "orange"}
        _, ax = qml.draw_mpl(f_circ, wire_options=wire_options)(0.52)

        for w in ax.lines:
            assert w.get_color() == "orange"

        # Wires are orange and cyan
        wire_options = {0: {"color": "orange"}, 1: {"color": "cyan"}}
        _, ax = qml.draw_mpl(f_circ, wire_options=wire_options)(0.52)

        assert ax.lines[0].get_color() == "orange"
        assert ax.lines[1].get_color() == "cyan"
        assert ax.lines[2].get_color() == "black"

        # Make all wires cyan and bold,
        # except for wires 2 and 6, which are dashed and another color
        wire_options = {
            "color": "cyan",
            "linewidth": 5,
            2: {"linestyle": "--", "color": "red"},
            6: {"linestyle": "--", "color": "orange", "linewidth": 1},
        }
        _, ax = qml.draw_mpl(f_circ, wire_options=wire_options)(0.52)

        for i, w in enumerate(ax.lines):
            if i == 2:
                assert w.get_color() == "red"
                assert w.get_linestyle() == "--"
                assert w.get_linewidth() == 5
            elif i == 6:
                assert w.get_color() == "orange"
                assert w.get_linestyle() == "--"
                assert w.get_linewidth() == 1
            else:
                assert w.get_color() == "cyan"
                assert w.get_linestyle() == "-"
                assert w.get_linewidth() == 5

        wire_options = {
            "linewidth": 5,
            2: {"linestyle": "--", "color": "red"},
            6: {"linestyle": "--", "color": "orange"},
        }

        _, ax = qml.draw_mpl(f_circ, wire_options=wire_options)(0.52)

        for i, w in enumerate(ax.lines):
            if i == 2:
                assert w.get_color() == "red"
                assert w.get_linestyle() == "--"
                assert w.get_linewidth() == 5
            elif i == 6:
                assert w.get_color() == "orange"
                assert w.get_linestyle() == "--"
                assert w.get_linewidth() == 5
            else:
                assert w.get_color() == "black"
                assert w.get_linestyle() == "-"
                assert w.get_linewidth() == 5

        plt.close("all")

    def test_individual_wire_options_with_string_labels(self):
        """Test that individual wire options work with string wire labels"""

        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.X("a")
            qml.Y("b")
            return qml.expval(qml.Z("a"))

        wire_options = {
            "color": "teal",
            "linewidth": 5,
            "b": {"color": "orange", "linestyle": "--"},
        }
        _, ax = qml.draw_mpl(circuit, wire_options=wire_options)()

        for i, w in enumerate(ax.lines):
            assert w.get_linewidth() == 5
            if i == 0:
                assert w.get_color() == "teal"
                assert w.get_linestyle() == "-"
            if i == 1:
                assert w.get_color() == "orange"
                assert w.get_linestyle() == "--"

    def test_wire_options_and_wire_order(self):
        """Test that individual wire options work with specifying a wire_order"""

        device = qml.device("default.qubit", wires=4)

        @qml.qnode(device)
        def circuit():
            for w in device.wires:
                qml.X(w)
            return qml.expval(qml.Z(0))

        wire_options = {
            "color": "teal",
            "linewidth": 5,
            3: {"color": "orange", "linestyle": "--"},  # wire 3 should be orange and dashed
        }
        _, ax = qml.draw_mpl(circuit, wire_order=[1, 3, 0, 2], wire_options=wire_options)()

        for i, w in enumerate(ax.lines):
            assert w.get_linewidth() == 5
            if i == 1:
                assert w.get_color() == "orange"
                assert w.get_linestyle() == "--"
            else:
                assert w.get_color() == "teal"
                assert w.get_linestyle() == "-"


class TestMPLIntegration:
    """Test using matplotlib styling to modify look of graphic."""

    def test_rcparams(self):
        """Test setting rcParams modifies style for draw_mpl(circuit, style="rcParams")."""

        rgba_red = (1, 0, 0, 1)
        rgba_green = (0, 1, 0, 1)
        plt.rcParams["patch.facecolor"] = rgba_red
        plt.rcParams["lines.color"] = rgba_green

        _, ax = qml.draw_mpl(circuit1, style="rcParams")(1.23, 2.34)

        assert ax.patches[0].get_facecolor() == rgba_red
        assert ax.patches[1].get_facecolor() == rgba_red

        for l in ax.lines[:-1]:  # final is fancy arrow, has different styling
            assert l.get_color() == rgba_green

        qml.drawer.use_style("black_white")
        plt.close("all")

    def test_style_with_matplotlib(self):
        """Test matplotlib styles impact figure styling for draw_mpl(circuit, style="rcParams")."""

        plt.style.use("fivethirtyeight")

        _, ax = qml.draw_mpl(circuit1, style="rcParams")(1.23, 2.34)

        expected_facecolor = mpl.colors.to_rgba(plt.rcParams["patch.facecolor"])
        assert ax.patches[0].get_facecolor() == expected_facecolor
        assert ax.patches[1].get_facecolor() == expected_facecolor

        expected_linecolor = mpl.colors.to_rgba(plt.rcParams["lines.color"])
        for l in ax.lines[:-1]:  # final is fancy arrow, has different styling
            assert mpl.colors.to_rgba(l.get_color()) == expected_linecolor

        qml.drawer.use_style("black_white")
        plt.close("all")

    def test_style_restores_settings(self):
        """Test that selecting style as draw_mpl(circuit, style=None) does not modify the users
        general matplotlib plotting settings"""

        initial_facecolor = mpl.rcParams["axes.facecolor"]
        initial_patch_facecolor = mpl.rcParams["patch.facecolor"]
        initial_patch_edgecolor = mpl.rcParams["patch.edgecolor"]

        # confirm settings were updated for the draw_mpl plot
        _, ax = qml.draw_mpl(circuit1, style="sketch")(1.234, 1.234)
        assert ax.get_facecolor() == (
            0.8392156862745098,
            0.9607843137254902,
            0.8862745098039215,
            1.0,
        )
        assert ax.patches[3].get_facecolor() == (
            0.8392156862745098,
            0.9607843137254902,
            0.8862745098039215,
            1.0,
        )
        assert ax.patches[3].get_edgecolor() == (0.0, 0.0, 0.0, 1.0)

        # confirm general matplotlib settings were reset after plotting
        assert mpl.rcParams["axes.facecolor"] == initial_facecolor
        assert mpl.rcParams["patch.facecolor"] == initial_patch_facecolor
        assert mpl.rcParams["patch.edgecolor"] == initial_patch_edgecolor


def test_draw_mpl_supports_qfuncs():
    """Test that draw_mpl works with non-QNode quantum functions."""

    def qfunc(x):
        qml.RX(x, 0)

    fig, ax = qml.draw_mpl(qfunc)(1.1)

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes._axes.Axes)  # pylint:disable=protected-access
    assert len(ax.patches) == 1
    assert len(ax.lines) == 1
    assert len(ax.texts) == 2
    assert ax.texts[0].get_text() == "0"
    assert ax.texts[1].get_text() == "RX"
    plt.close("all")


def test_draw_mpl_with_qfunc_warns_with_level():
    """Test that draw warns the user about level being ignored."""

    def qfunc():
        qml.PauliZ(0)

    with pytest.warns(UserWarning, match="the level argument is ignored"):
        qml.draw_mpl(qfunc, level=None)


def test_qnode_mid_circuit_measurement_not_deferred_device_api(mocker):
    """Test that a circuit containing mid-circuit measurements is not transformed by the drawer
    to use deferred measurements if the device uses the new device API."""

    @qml.qnode(qml.device("default.qubit"))
    def circ():
        qml.PauliX(0)
        qml.measure(0)
        return qml.probs(wires=0)

    draw_qnode = qml.draw_mpl(circ)
    spy = mocker.spy(qml.defer_measurements, "_transform")

    _ = draw_qnode()
    spy.assert_not_called()


def test_qnode_transform_program(mocker):
    """Test that qnode transforms are applied before drawing a circuit."""

    @qml.compile
    @qml.qnode(qml.device("default.qubit"))
    def circuit():
        qml.RX(1.1, 0)
        qml.RX(2.2, 0)
        return qml.state()

    draw_qnode = qml.draw_mpl(circuit, decimals=2)
    qnode_transform = circuit.transform_program[0]
    # pylint: disable=protected-access
    spy = mocker.spy(qnode_transform._transform_dispatcher, "_transform")

    _ = draw_qnode()
    spy.assert_called_once()


def test_draw_mpl_with_control_in_adjoint():
    def U(wires):
        qml.adjoint(qml.CNOT)(wires=wires)

    @qml.qnode(dev)
    def circuit():
        qml.ctrl(U, control=0)(wires=["a", 1.23])
        return qml.state()

    _, ax = qml.draw_mpl(circuit)()
    assert len(ax.lines) == 4  # three wires, one control
    assert len(ax.texts) == 4  # three wire labels, one gate label
    assert ax.texts[-1].get_text() == "X†"


def test_applied_transforms():
    """Test that any transforms applied to the qnode are included in the output."""

    @qml.transform
    def just_pauli_x(_):
        new_tape = qml.tape.QuantumScript([qml.PauliX(0)])
        return (new_tape,), lambda res: res[0]

    @just_pauli_x
    @qml.qnode(qml.device("default.qubit", wires=2))
    def my_circuit():
        qml.SWAP(wires=(0, 1))
        qml.CNOT(wires=(0, 1))
        return qml.probs(wires=(0, 1))

    _, ax = qml.draw_mpl(my_circuit)()

    assert len(ax.lines) == 1  # single wire used in tape
    assert len(ax.patches) == 1  # single pauli x gate
    assert len(ax.texts) == 2  # one wire label, one gate label

    plt.close("all")


@pytest.mark.parametrize("use_qnode", (True, False))
def test_wire_sorting_if_no_wire_order(use_qnode):
    """Test that wires are automatically sorted if the device and user
    dont provide a wire order."""

    def f():
        qml.X(4)
        qml.X(2)
        return qml.expval(qml.Z(0))

    if use_qnode:
        f = qml.QNode(f, qml.device("default.qubit"))

    _, ax = qml.draw_mpl(f)()

    assert ax.texts[0].get_text() == "0"
    assert ax.texts[1].get_text() == "2"
    assert ax.texts[2].get_text() == "4"

    plt.close()


@pytest.mark.parametrize("use_qnode", (True, False))
def test_wire_sorting_fallback_if_no_wire_order(use_qnode):
    """Test that wires are automatically sorted if the device and user
    dont provide a wire order."""

    def f():
        qml.X(4)
        qml.X("a")
        return qml.expval(qml.Z(0))

    if use_qnode:
        f = qml.QNode(f, qml.device("default.qubit"))

    _, ax = qml.draw_mpl(f)()

    assert ax.texts[0].get_text() == "4"
    assert ax.texts[1].get_text() == "a"
    assert ax.texts[2].get_text() == "0"
