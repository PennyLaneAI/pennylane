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
"""This file tests the ``qml.drawer.tape_mpl`` function.

See section on "Testing Matplotlib based code" in the "Software Tests"
page in the developement guide.
"""
# pylint: disable=protected-access, expression-not-assigned

import warnings

import numpy as np
import pytest

import pennylane as qml
from pennylane.drawer import tape_mpl
from pennylane.ops.op_math import Controlled
from pennylane.tape import QuantumScript

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")


def test_empty_tape():
    """Edge case where the tape is empty. Use this to test return types."""

    fig, ax = tape_mpl(QuantumScript())

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes._axes.Axes)

    assert fig.axes == [ax]
    plt.close()


with qml.queuing.AnnotatedQueue() as q1:
    qml.PauliX(0)
    qml.PauliX("a")
    qml.PauliX(1.234)


tape1 = QuantumScript.from_queue(q1)


def test_fontsize():
    """Test default fontsize set with keyword argument."""

    _, ax = tape_mpl(tape1, fontsize=20)
    for t in ax.texts:
        assert t.get_fontsize() == 20
    plt.close()


def test_fig_argument():
    """Test figure argument is used correctly"""

    fig = plt.figure()
    output_fig, ax = tape_mpl(tape1, fontsize=20, fig=fig)

    assert ax.get_figure() == fig
    assert output_fig == fig
    plt.close()


label_data = [
    ({}, ["0", "a", "1.234"]),  # default behaviour
    ({"wire_order": [1.234, "a", 0]}, ["1.234", "a", "0"]),  # provide standard wire order
    (
        {"wire_order": ["a", 1.234]},
        ["a", "1.234", "0"],
    ),  # wire order that doesn't include all active wires
    (
        {"wire_order": ["nope", "not there", 3]},
        ["0", "a", "1.234"],
    ),  # wire order includes unused wires
    (
        {"wire_order": ["aux", 0, "a", 1.234], "show_all_wires": True},
        [
            "aux",
            "0",
            "a",
        ],
    ),  # show_all_wires=True
]


class TestLabelling:
    """Test the labels produced by the circuit drawer for the wires."""

    @pytest.mark.parametrize("kwargs, labels", label_data)
    def test_labels(self, kwargs, labels):
        """Test the labels produced under different settings. Check both text value and position."""
        _, ax = tape_mpl(tape1, **kwargs)

        for wire, (text_obj, label) in enumerate(zip(ax.texts, labels)):
            assert text_obj.get_text() == label
            assert text_obj.get_position() == (-1.5, wire)

        plt.close()

    def test_label_options(self):
        """Test that providing the `label_options` argument alters the styling of the text."""

        _, ax = tape_mpl(tape1, label_options={"fontsize": 10})

        for text_obj in ax.texts[0:3]:
            assert text_obj.get_fontsize() == 10.0

        plt.close()

    @pytest.mark.parametrize("kwargs, _", label_data)
    def test_hide_wire_labels(self, kwargs, _):
        """Test that wire labels are skipped with show_wire_labels=False."""
        fig, ax = tape_mpl(tape1, show_wire_labels=False, **kwargs)

        # Only PauliX gate labels should be present
        assert len(ax.texts) == 3
        assert all(t.get_text() == "X" for t in ax.texts)
        assert fig.get_figwidth() == 3

        plt.close()


class TestWires:
    """Test that wire lines are produced correctly in different situations."""

    def test_empty_tape_wire_order(self):
        """Test situation with empty tape but specified wires and show_all_wires
        still draws wires."""

        _, ax = tape_mpl(QuantumScript(), wire_order=[0, 1, 2], show_all_wires=True)

        assert len(ax.lines) == 3
        for wire, line in enumerate(ax.lines):
            assert line.get_xdata() == (-1, 1)  # from -1 to number of layers
            assert line.get_ydata() == (wire, wire)

        plt.close()

    def test_single_layer(self):
        """Test a single layer with multiple wires. Check that the expected number
        of wires are drawn, and they are in the correct location."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.PauliX(0)
            qml.PauliY(1)
            qml.PauliZ(2)

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)

        assert len(ax.lines) == 3
        for wire, line in enumerate(ax.lines):
            assert line.get_xdata() == (-1, 1)  # from -1 to number of layers
            assert line.get_ydata() == (wire, wire)

        plt.close()

    def test_three_layers(self):
        """Test wire length when circuit has three layers."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.PauliX(0)
            qml.PauliX(0)
            qml.PauliX(0)

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)

        assert len(ax.lines) == 1
        assert ax.lines[0].get_xdata() == (-1, 3)  # from -1 to number of layers
        assert ax.lines[0].get_ydata() == (0, 0)

        plt.close()

    def test_wire_options(self):
        """Test wires are formatted by providing a wire_options dictionary."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.PauliX(0)
            qml.PauliX(1)

        tape = QuantumScript.from_queue(q_tape)
        rgba_red = (1, 0, 0, 1)
        _, ax = tape_mpl(tape, wire_options={"linewidth": 5, "color": rgba_red})

        for line in ax.lines:
            assert line.get_linewidth() == 5
            assert line.get_color() == rgba_red

        plt.close()


class TestSpecialGates:
    """Tests the gates with special drawing methods."""

    width = 0.75 - 2 * 0.2

    def test_SWAP(self):
        """Test SWAP gate special call"""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.SWAP(wires=(0, 1))

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)
        layer = 0

        # two wires produce two lines and SWAP contains 5 more lines
        assert len(ax.lines) == 7

        connecting_line = ax.lines[2]
        assert connecting_line.get_data() == ((layer, layer), [0, 1])

        dx = 0.2
        # check the coordinates of the swap lines
        x_lines = ax.lines[3:]
        for line in x_lines:
            assert line.get_xdata() == (layer - dx, layer + dx)

        assert x_lines[0].get_ydata() == (-dx, dx)
        assert x_lines[1].get_ydata() == (dx, -dx)

        assert x_lines[2].get_ydata() == (1 - dx, 1 + dx)
        assert x_lines[3].get_ydata() == (1 + dx, 1 - dx)

        plt.close()

    def test_CSWAP(self):
        """Test CSWAP special call"""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.CSWAP(wires=(0, 1, 2))

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)
        layer = 0

        # three wires, one control, 5 swap
        assert len(ax.lines) == 9

        control_line = ax.lines[3]
        assert control_line.get_data() == ((layer, layer), (0, 2))

        # control circle
        assert ax.patches[0].center == (layer, 0)

        # SWAP components
        connecting_line = ax.lines[4]
        assert connecting_line.get_data() == ((layer, layer), [1, 2])

        x_lines = ax.lines[5:]
        assert x_lines[0].get_data() == ((layer - 0.2, layer + 0.2), (0.8, 1.2))
        assert x_lines[1].get_data() == ((layer - 0.2, layer + 0.2), (1.2, 0.8))

        assert x_lines[2].get_data() == ((layer - 0.2, layer + 0.2), (1.8, 2.2))
        assert x_lines[3].get_data() == ((layer - 0.2, layer + 0.2), (2.2, 1.8))
        plt.close()

    def test_CNOT(self):
        """Test CNOT gets a special call"""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.CNOT(wires=(0, 1))

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)
        layer = 0

        assert len(ax.patches) == 2
        assert ax.patches[0].center == (layer, 0)
        assert ax.patches[1].center == (layer, 1)

        control_line = ax.lines[2]
        assert control_line.get_data() == ((layer, layer), (0, 1))

        assert len(ax.lines) == 5
        plt.close()

    def test_Toffoli(self):
        """Test Toffoli gets a special call."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.Toffoli(wires=(0, 1, 2))

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)
        layer = 0

        assert len(ax.patches) == 3
        assert ax.patches[0].center == (layer, 0)
        assert ax.patches[1].center == (layer, 1)
        assert ax.patches[2].center == (layer, 2)

        # three wires, one control line, two target lines
        assert len(ax.lines) == 6
        control_line = ax.lines[3]
        assert control_line.get_data() == ((layer, layer), (0, 2))

        plt.close()

    def test_MultiControlledX_no_control_values(self):
        """Test MultiControlledX gets a special call."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.MultiControlledX(wires=[0, 1, 2, 3, 4])

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)
        layer = 0

        assert len(ax.patches) == 5
        for wire, patch in enumerate(ax.patches):
            assert patch.center == (layer, wire)

        # five wires, one control line, two target lines
        assert len(ax.lines) == 8
        control_line = ax.lines[5]
        assert control_line.get_data() == ((layer, layer), (0, 4))

        plt.close()

    def test_MultiControlledX_control_values(self):
        """Test MultiControlledX special call with provided control values."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.MultiControlledX(wires=[0, 1, 2, 3, 4], control_values=[0, 1, 0, 1])

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)

        assert ax.patches[0].get_facecolor() == (1.0, 1.0, 1.0, 1.0)  # white
        assert ax.patches[1].get_facecolor() == (0.0, 0.0, 0.0, 1.0)  # black
        assert ax.patches[2].get_facecolor() == (1.0, 1.0, 1.0, 1.0)
        assert ax.patches[3].get_facecolor() == (0.0, 0.0, 0.0, 1.0)

        plt.close()

    def test_CZ(self):
        """Test CZ gets correct special call."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.CZ(wires=(0, 1))

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)
        layer = 0

        # two wires one control line
        assert len(ax.lines) == 3

        assert ax.lines[2].get_data() == ((layer, layer), (0, 1))

        # two control circles
        assert len(ax.patches) == 2
        assert ax.patches[0].center == (layer, 0)
        assert ax.patches[1].center == (layer, 1)

        plt.close()

    def test_CCZ(self):
        """Test that CCZ gets correct special call."""

        tape = QuantumScript([qml.CCZ(wires=(0, 1, 2))])
        _, ax = tape_mpl(tape)
        layer = 0

        # three wires and one control line
        assert len(ax.lines) == 4

        assert ax.lines[3].get_data() == ((layer, layer), (0, 2))

        # three control circles
        assert len(ax.patches) == 3
        for i in range(3):
            assert ax.patches[i].center == (layer, i)

        plt.close()

    def test_Barrier(self):
        """Test Barrier gets correct special call."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.Barrier(wires=(0, 1, 2))

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)

        assert len(ax.lines) == 3
        assert len(ax.collections) == 2
        assert np.allclose(ax.collections[0].get_color(), np.array([[0.0, 0.0, 0.0, 1.0]]))  # black
        assert np.allclose(ax.collections[0].get_color(), np.array([[0.0, 0.0, 0.0, 1.0]]))  # black

        plt.close()

    def test_WireCut(self):
        """Test WireCut gets correct special call."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.WireCut(wires=(0, 1))

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)

        assert len(ax.lines) == 2
        assert len(ax.texts) == 3
        assert len(ax.collections) == 1

        plt.close()

    def test_Prod(self):
        """Test Prod gets correct special call."""
        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.S(0) @ qml.T(0)

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)

        assert len(ax.lines) == 1
        assert len(ax.collections) == 0

        plt.close()

    def test_MidMeasureMP(self):
        """Tests MidMeasureMP has correct special handling."""
        m = qml.measure(0)
        tape = QuantumScript(m.measurements)
        _, ax = tape_mpl(tape)
        assert [l.get_data() for l in ax.lines] == [((-1, 1), (0, 0))]
        assert len(ax.patches) == 3

        assert ax.patches[0].get_x() == -0.175
        assert ax.patches[0].get_y() == -0.175
        plt.close()

    def test_MidMeasure_reset(self):
        """Test that a reset mid circuit measurement is correct."""
        m = qml.measure(0, reset=True)
        tape = QuantumScript(m.measurements)
        fig, ax = tape_mpl(tape)

        assert [l.get_data() for l in ax.lines] == [((-1, 2), (0, 0))]
        assert len(ax.patches) == 5

        # patches 0-2 are normal measure box
        assert ax.patches[0].get_x() == -0.175
        assert ax.patches[0].get_y() == -0.175
        assert isinstance(ax.patches[1], mpl.patches.Arc)
        assert isinstance(ax.patches[2], mpl.patches.FancyArrow)

        # patch 3 is wire erasing rectnagle
        assert ax.patches[3].get_xy() == (0, -0.1)
        assert ax.patches[3].get_width() == 1
        assert ax.patches[3].get_facecolor() == fig.get_facecolor()
        assert ax.patches[3].get_edgecolor() == fig.get_facecolor()

        # patch 4 is state prep box gate
        assert isinstance(ax.patches[0], mpl.patches.FancyBboxPatch)
        assert ax.patches[0].get_x() == -self.width / 2.0
        assert ax.patches[0].get_y() == -self.width / 2.0
        assert ax.patches[0].get_width() == self.width
        assert ax.patches[0].get_height() == self.width

        assert len(ax.texts) == 2
        assert ax.texts[1].get_text() == "|0⟩"
        plt.close()

    def test_MidMeasure_postselect(self):
        """Test that a mid circuit measurement with postselection gets a label."""
        m = qml.measure(0, postselect=True)
        tape = QuantumScript(m.measurements)
        _, ax = tape_mpl(tape)

        assert [l.get_data() for l in ax.lines] == [((-1, 1), (0, 0))]
        assert len(ax.patches) == 3

        assert len(ax.texts) == 2
        assert ax.texts[0].get_text() == "0"
        assert ax.texts[1].get_text() == "1"
        plt.close()


controlled_data = [
    (qml.CY(wires=(0, 1)), "Y"),
    (qml.CRX(1.2345, wires=(0, 1)), "RX"),
    (qml.CRot(1.2, 2.2, 3.3, wires=(0, 1)), "Rot"),
]


class TestControlledGates:
    """Tests generic controlled gates"""

    width = 0.75 - 2 * 0.2

    @pytest.mark.parametrize("op, label", controlled_data)
    def test_control_gates(self, op, label):
        """Test a variety of non-special gates. Checks control wires are drawn, and
        that a box is drawn over the target wires."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.apply(op)

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)
        layer = 0

        assert isinstance(ax.patches[0], mpl.patches.Circle)
        assert ax.patches[0].center == (layer, 0)

        control_line = ax.lines[2]
        assert control_line.get_data() == ((layer, layer), (0, 1))

        assert isinstance(ax.patches[1], mpl.patches.FancyBboxPatch)
        assert ax.patches[1].get_x() == layer - self.width / 2.0
        assert ax.patches[1].get_y() == 1 - self.width / 2.0

        # two wire labels, so [2] is box gate label
        assert ax.texts[2].get_text() == label

        # box and text must be raised above control wire
        # text raised over box
        assert ax.patches[1].get_zorder() > control_line.get_zorder()
        assert ax.texts[2].get_zorder() > ax.patches[1].get_zorder()

        plt.close()

    def test_CRX_decimals(self):
        """Test a controlled parametric operation with specified decimals."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.CRX(1.234, wires=(0, 1))

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape, decimals=2)

        # two wire labels, so CRX is third text object
        assert ax.texts[2].get_text() == "RX\n(1.23)"
        plt.close()

    def test_control_values_bool(self):
        """Test control_values get displayed correctly when they are provided as a list of bools."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            # pylint:disable=no-member
            qubit_unitary = qml.QubitUnitary(qml.RX.compute_matrix(0), wires=4)
            qml.ops.op_math.Controlled(qubit_unitary, (0, 1, 2, 3), [1, 0, 1, 0])

        tape = QuantumScript.from_queue(q_tape)
        self.check_tape_controlled_qubit_unitary(tape)

        plt.close()

    def test_nested_control_values_bool(self):
        """Test control_values get displayed correctly for nested controlled operations
        when they are provided as a list of bools."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            Controlled(
                qml.ctrl(qml.PauliY(wires=4), control=[2, 3], control_values=[1, 0]),
                control_wires=[0, 1],
                control_values=[1, 0],
            )

        tape = QuantumScript.from_queue(q_tape)
        self.check_tape_controlled_qubit_unitary(tape)

    def check_tape_controlled_qubit_unitary(self, tape):
        """Checks the control symbols for a tape with some version of a controlled qubit unitary."""
        _, ax = tape_mpl(tape, style="rcParams")  # use plt.rcParams values
        layer = 0

        # 5 wires -> 4 control, 1 target
        assert len(ax.patches) == 5
        # Circle for control values are positioned correctly
        # Circle color matched according to the control value
        assert ax.patches[0].center == (layer, 0)
        assert ax.patches[0].get_facecolor() == mpl.colors.to_rgba(plt.rcParams["lines.color"])
        assert ax.patches[1].center == (layer, 1)
        assert ax.patches[1].get_facecolor() == mpl.colors.to_rgba(plt.rcParams["axes.facecolor"])
        assert ax.patches[2].center == (layer, 2)
        assert ax.patches[2].get_facecolor() == mpl.colors.to_rgba(plt.rcParams["lines.color"])
        assert ax.patches[3].center == (layer, 3)
        assert ax.patches[3].get_facecolor() == mpl.colors.to_rgba(plt.rcParams["axes.facecolor"])

        # five wires, one control line
        assert len(ax.lines) == 6
        control_line = ax.lines[5]
        assert control_line.get_data() == ((layer, layer), (0, 4))

        plt.close()


general_op_data = [
    qml.RX(1.234, wires=0),
    qml.Hadamard(0),
    qml.S(wires=0),
    qml.IsingXX(1.234, wires=(0, 1)),
    qml.U3(1.234, 2.345, 3.456, wires=0),
    # State Prep
    qml.BasisState([0, 1, 0], wires=(0, 1, 2)),
    ### Templates
    qml.QFT(wires=range(3)),
    qml.Permute([4, 2, 0, 1, 3], wires=(0, 1, 2, 3, 4)),
    qml.GroverOperator(wires=(0, 1, 2, 3, 4, 5)),
    ### Continuous Variable
    qml.Kerr(1.234, wires=0),
    qml.Beamsplitter(1.234, 2.345, wires=(0, 1)),
    qml.Rotation(1.234, wires=0),
]


class TestGeneralOperations:
    """Tests general operations."""

    width = 0.75 - 2 * 0.2

    @pytest.mark.parametrize("op", general_op_data)
    def test_general_operations(self, op):
        """Test that a variety of operations produce a rectangle across relevant wires
        and a correct label text."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.apply(op)

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)

        num_wires = len(op.wires)
        assert ax.texts[num_wires].get_text() == op.label()

        assert isinstance(ax.patches[0], mpl.patches.FancyBboxPatch)
        assert ax.patches[0].get_x() == -self.width / 2.0
        assert ax.patches[0].get_y() == -self.width / 2.0
        assert ax.patches[0].get_width() == self.width
        assert ax.patches[0].get_height() == num_wires - 1 + self.width

        plt.close()

    def test_snapshot(self):
        """Test that `qml.Snapshot` works properly with `tape_mpl`."""

        # Test that empty figure is created when the only gate is `qml.Snapshot`
        tape = QuantumScript([qml.Snapshot()])
        fig, ax = tape_mpl(tape)

        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes._axes.Axes)

        assert fig.axes == [ax]
        assert len(ax.patches) == len(ax.texts) == 0

        # Test that `qml.Snapshot` works properly when other gates are present
        tape = QuantumScript([qml.Snapshot(), qml.Hadamard(0), qml.Hadamard(1), qml.Hadamard(2)])
        _, ax = tape_mpl(tape)

        assert isinstance(ax.patches[0], mpl.patches.FancyBboxPatch)
        assert ax.patches[0].get_x() == -self.width / 2.0
        assert ax.patches[0].get_y() == -self.width / 2.0
        assert ax.patches[0].get_width() == self.width
        assert ax.patches[0].get_height() == 2 + self.width

        plt.close()

    @pytest.mark.parametrize("input_wires", [tuple(), (0, 1), (0, 2, 1, 3)])
    @pytest.mark.parametrize("show_all_wires", [False, True])
    @pytest.mark.parametrize("cls", [qml.GlobalPhase, qml.Identity])
    def test_global_phase(self, input_wires, show_all_wires, cls):
        """Test that `GlobalPhase` and `Identity` works properly with `tape_mpl`."""

        data = [0.3625][: cls.num_params]
        tape = QuantumScript([cls(*data, wires=input_wires), qml.X(0)])
        fig, ax = tape_mpl(tape, show_all_wires=show_all_wires, wire_order=[0, 1, 2, 3, 4])

        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes._axes.Axes)

        assert fig.axes == [ax]

        if show_all_wires:
            num_wires = 5
        else:
            num_wires = max(len(input_wires), 1)
        assert isinstance(ax.patches[0], mpl.patches.FancyBboxPatch)
        assert ax.patches[0].get_x() == -self.width / 2.0
        assert ax.patches[0].get_y() == -self.width / 2.0
        assert ax.patches[0].get_width() == self.width
        assert ax.patches[0].get_height() == num_wires - 1 + self.width

        plt.close()

    @pytest.mark.parametrize("cls", [qml.GlobalPhase, qml.Identity])
    def test_multiple_global_ops(self, cls):
        """Test that global ops correctly reserve layers for themselves."""
        data = [0.3625][: cls.num_params]
        tape = QuantumScript([cls(*data, wires=wires) for wires in [[], [0], [1, 2]]])

        fig, ax = tape_mpl(tape)

        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes._axes.Axes)

        assert fig.axes == [ax]

        assert len(ax.patches) == 3  # Three boxes without notches
        for i, patch in enumerate(ax.patches):
            assert isinstance(patch, mpl.patches.FancyBboxPatch)
            assert patch.get_x() == i + -self.width / 2.0
            assert patch.get_y() == -self.width / 2.0
            assert patch.get_width() == self.width
            assert patch.get_height() == 2 + self.width

        plt.close()

    @pytest.mark.parametrize(
        "input_wires, control_wires",
        [
            (tuple(), (0,)),
            ((0, 1), (2, 3)),
            ((0, 3), (2,)),
            ((0, 2, 3), (1,)),
            ((0, 3), (1, 2)),
            ((0, 2), (1, 3)),
        ],
    )
    @pytest.mark.parametrize("show_all_wires", [False, True])
    @pytest.mark.parametrize("cls", [qml.GlobalPhase, qml.Identity])
    def test_ctrl_global_op(self, input_wires, control_wires, show_all_wires, cls):
        """Test that controlled `GlobalPhase` and `Identity` works properly with `tape_mpl`."""

        data = [0.3625][: cls.num_params]
        op = qml.ctrl(cls(*data, wires=input_wires), control=control_wires)
        tape = QuantumScript([op, qml.X(0), qml.X(1)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, ax = tape_mpl(tape, show_all_wires=show_all_wires, wire_order=[0, 1, 2, 3, 4])

        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes._axes.Axes)

        assert fig.axes == [ax]

        # Control node circular patches (may be invisible)
        for i, c in enumerate(control_wires):
            patch = ax.patches[i]
            assert isinstance(patch, mpl.patches.Circle)
            assert patch.center == (0, c)

        # Main box
        if show_all_wires:
            covered_wires = range(5)
        elif len(input_wires) == 0:
            covered_wires = (0, 1)  # Wires for qml.X operations transfer to GlobalPhase
        else:
            covered_wires = range(min(input_wires), max(input_wires) + 1)
        min_box_wire = min([w for w in covered_wires if w not in control_wires])
        max_box_wire = max([w for w in covered_wires if w not in control_wires])
        i = len(control_wires)
        main_patch = ax.patches[i]
        assert isinstance(main_patch, mpl.patches.FancyBboxPatch)

        assert main_patch.get_x() == -self.width / 2.0
        assert main_patch.get_y() == min_box_wire - self.width / 2.0
        assert main_patch.get_width() == self.width
        assert main_patch.get_height() == max_box_wire - min_box_wire + self.width
        for j in range(i):
            assert main_patch.zorder > ax.patches[j].zorder

        # Notches
        i += 1
        for w in control_wires:
            if min_box_wire < w < max_box_wire:
                notch_patch = ax.patches[i]
                assert isinstance(notch_patch, mpl.patches.FancyBboxPatch)
                assert notch_patch.zorder < main_patch.zorder

        plt.close()

    @pytest.mark.parametrize("cls", [qml.GlobalPhase, qml.Identity])
    def test_ctrl_global_op_without_target(self, cls):
        """Test that an error is raised if a controlled GlobalPhase is present that can
        not infer any target wires."""

        data = [0.251][: cls.num_params]
        op = qml.ctrl(cls(*data, wires=[]), control=(0, 4))
        tape = QuantumScript([op])
        with pytest.raises(ValueError, match="controlled global gate with unknown"):
            _ = tape_mpl(tape)

        # No error if wire_order provides additional wire(s)
        _ = tape_mpl(tape, wire_order=[0, 1, 2], show_all_wires=True)

        # Error if wire_order provides additional wire(s) but they are not drawn
        with pytest.raises(ValueError, match="controlled global gate with unknown"):
            _ = tape_mpl(tape, wire_order=[0, 1, 2], show_all_wires=False)

    @pytest.mark.parametrize("op", general_op_data)
    def test_general_operations_decimals(self, op):
        """Check that the decimals argument affects text strings when applicable."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.apply(op)

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape, decimals=2)

        num_wires = len(op.wires)
        assert ax.texts[num_wires].get_text() == op.label(decimals=2)

        plt.close()

    @pytest.mark.parametrize("wires, n", [((0,), 0), ((0, 1, 2), 0), ((0, 2), 4)])
    def test_notches(self, wires, n):
        """Test notches are included when non-active wires exist."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.QFT(wires=wires)

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape, show_all_wires=True, wire_order=[0, 1, 2])
        assert len(ax.patches) == (n + 1)
        plt.close()

    def test_active_wire_notches_False(self):
        """Test active wire notches are disable with active_wire_notches=False."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.QFT(wires=(0, 3))

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(
            tape, show_all_wires=True, wire_order=[0, 1, 2, 3], active_wire_notches=False
        )

        assert len(ax.patches) == 1
        plt.close()


measure_data = [
    ([qml.expval(qml.PauliX(0))], [0]),
    ([qml.probs(wires=(0, 1, 2))], [0, 1, 2]),
    ([qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0) @ qml.PauliY(1)), qml.state()], [0, 1]),
    ([qml.expval(qml.NumberOperator(wires=0))], [0]),
]


class TestMeasurements:
    """Tests measurements are drawn correctly"""

    width = 0.75 - 2 * 0.2

    @pytest.mark.parametrize("measurements, wires", measure_data)
    def test_measurements(self, measurements, wires):
        """Tests a variety of measurements draw measurement boxes on the correct wires."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            for m in measurements:
                qml.apply(m)

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)

        assert len(ax.patches) == 3 * len(wires)

        for ii, w in enumerate(wires):
            assert ax.patches[3 * ii].get_x() == 1 - self.width / 2.0
            assert ax.patches[3 * ii].get_y() == w - self.width / 2.0
            assert ax.patches[3 * ii + 1].center == (1, w + 0.75 * 0.15)  # arc
            assert isinstance(ax.patches[3 * ii + 2], mpl.patches.FancyArrow)  # fancy arrow

        plt.close()

    def test_state(self):
        """Test state produces measurements on all wires."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.state()

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape, wire_order=[0, 1, 2], show_all_wires=True)

        assert len(ax.patches) == 9  # three measure boxes with 3 patches each

        assert all(isinstance(box, mpl.patches.FancyBboxPatch) for box in ax.patches[::3])
        assert all(isinstance(arc, mpl.patches.Arc) for arc in ax.patches[1::3])
        assert all(isinstance(arrow, mpl.patches.FancyArrow) for arrow in ax.patches[2::3])

        for layer, box in enumerate(ax.patches[::3]):
            assert box.get_x() == 1 - self.width / 2.0
            assert box.get_y() == layer - self.width / 2.0
        plt.close()


class TestLayering:
    """Tests operations are placed into layers correctly."""

    width = 0.75 - 2 * 0.2

    def test_single_layer_multiple_wires(self):
        """Tests positions when multiple gates are all in the same layer."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.PauliX(0)
            qml.PauliX(1)
            qml.PauliX(2)

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)

        # As layers are stored in sets, we don't know the
        # order operations are added to ax.
        # So we check that each rectangle is in ``patches``
        # independent of order
        boxes_x = [p.get_x() for p in ax.patches]
        boxes_y = [p.get_y() for p in ax.patches]
        for wire in range(3):
            assert -self.width / 2.0 in boxes_x
            assert wire - self.width / 2.0 in boxes_y

        for t in ax.texts[3:]:
            assert t.get_text() == "X"
        plt.close()

    def test_three_layers_one_wire(self):
        """Tests the positions when multiple gates are all on the same wire."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.PauliX(0)
            qml.PauliX(0)
            qml.PauliX(0)

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape)

        for layer, box in enumerate(ax.patches):
            assert box.get_x() == layer - self.width / 2.0
            assert box.get_y() == -self.width / 2.0

        for t in ax.texts[1:]:
            assert t.get_text() == "X"
        plt.close()

    def test_blocking_IsingXX(self):
        """Tests the position of layers when a multiwire gate is blocking another gate on its empty wire."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.PauliX(0)
            qml.IsingXX(1.234, wires=(0, 2))
            qml.PauliX(1)

        tape = QuantumScript.from_queue(q_tape)
        _, ax = tape_mpl(tape, wire_order=[0, 1, 2], active_wire_notches=False)

        # layer=0, wire=0
        assert ax.patches[0].get_x() == -self.width / 2.0
        assert ax.patches[0].get_y() == -self.width / 2.0

        # layer=1, wire=0
        assert ax.patches[1].get_x() == 1.0 - self.width / 2.0
        assert ax.patches[1].get_y() == -self.width / 2.0

        # layer=2, wire=1
        assert ax.patches[2].get_x() == 2 - self.width / 2.0
        assert ax.patches[2].get_y() == 1 - self.width / 2.0

        assert ax.texts[3].get_text() == "X"
        assert ax.texts[4].get_text() == "IsingXX"
        assert ax.texts[5].get_text() == "X"
        plt.close()


class TestClassicalControl:
    """Tests involving mid circuit measurements and classical control."""

    def test_single_measure_multiple_conds(self):
        """Test a single mid circuit measurement with two conditional operators."""

        with qml.queuing.AnnotatedQueue() as q:
            m0 = qml.measure(0)
            qml.cond(m0, qml.PauliX)(0)
            qml.cond(m0, qml.PauliY)(0)

        tape = qml.tape.QuantumScript.from_queue(q)
        _, ax = qml.drawer.tape_mpl(tape, style="black_white")

        assert len(ax.patches) == 5  # three for measure, two for boxes

        [_, cwire] = ax.lines

        assert cwire.get_xdata() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert cwire.get_ydata() == [1, 0, 1, 1, 0, 1, 1, 0, 1]

        [pe1, pe2] = cwire.get_path_effects()

        # probably not a good way to test this, but the best I can figure out
        assert pe1._gc == {
            "linewidth": 5 * 1.5,  # hardcoded value to black_white linewidth
            "foreground": "black",  # lines.color for black white style
        }
        assert pe2._gc == {
            "linewidth": 3 * 1.5,  # hardcoded value to black_white linewidth
            "foreground": "white",  # figure.facecolor for black white sytle
        }
        plt.close()

    def test_combo_measurement(self):
        """Test a control that depends on two mid circuit measurements."""

        with qml.queuing.AnnotatedQueue() as q:
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            qml.cond(m0 & m1, qml.PauliY)(0)

        tape = qml.tape.QuantumScript.from_queue(q)
        _, ax = qml.drawer.tape_mpl(tape, style="black_white")

        assert len(ax.patches) == 7  # three for 2 measurements, one for box
        [_, _, cwire1, cwire2, eraser] = ax.lines

        assert cwire1.get_xdata() == [0, 0, 0, 2, 2, 2]
        assert cwire1.get_ydata() == [2, 0, 2, 2, 0, 2]

        assert cwire2.get_xdata() == [1, 1, 1, 2, 2, 2]
        assert cwire2.get_ydata() == [2.25, 1, 2.25, 2.25, 0, 2.25]

        for cwire in [cwire1, cwire2]:
            [pe1, pe2] = cwire.get_path_effects()

            # probably not a good way to test this, but the best I can figure out
            assert pe1._gc == {
                "linewidth": 5 * 1.5,  # hardcoded value to black_white linewidth
                "foreground": "black",  # lines.color for black white style
            }
            assert pe2._gc == {
                "linewidth": 3 * 1.5,  # hardcoded value to black_white linewidth
                "foreground": "white",  # figure.facecolor for black white sytle
            }

        assert eraser.get_xdata() == (1.8, 2)
        assert eraser.get_ydata() == (2, 2)
        assert eraser.get_color() == "white"  # hardcoded value to black_white color
        assert eraser.get_linewidth() == 3 * 1.5  # hardcoded value to black_white linewidth

        plt.close()

    def test_combo_measurement_non_terminal(self):
        """Test a combination measurement where the classical wires continue on.
        This covers the "erase_right=True" case.
        """
        with qml.queuing.AnnotatedQueue() as q:
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            qml.cond(m0 & m1, qml.PauliY)(0)
            qml.cond(m0, qml.S)(0)
            qml.cond(m1, qml.T)(1)

        tape = qml.tape.QuantumScript.from_queue(q)
        _, ax = qml.drawer.tape_mpl(tape, style="black_white")

        [_, _, cwire1, cwire2, eraser] = ax.lines

        assert cwire1.get_xdata() == [0, 0, 0, 2, 2, 2, 3, 3, 3]
        assert cwire1.get_ydata() == [2, 0, 2, 2, 0, 2, 2, 0, 2]

        assert cwire2.get_xdata() == [1, 1, 1, 2, 2, 2, 4, 4, 4]
        assert cwire2.get_ydata() == [2.25, 1, 2.25, 2.25, 0, 2.25, 2.25, 1, 2.25]

        for cwire in [cwire1, cwire2]:
            [pe1, pe2] = cwire.get_path_effects()

            # probably not a good way to test this, but the best I can figure out
            assert pe1._gc == {
                "linewidth": 5 * 1.5,  # hardcoded value to black_white linewidth
                "foreground": "black",  # lines.color for black white style
            }
            assert pe2._gc == {
                "linewidth": 3 * 1.5,  # hardcoded value to black_white linewidth
                "foreground": "white",  # figure.facecolor for black white sytle
            }

        assert eraser.get_xdata() == (1.8, 2.2)
        assert eraser.get_ydata() == (2, 2)
        assert eraser.get_color() == "white"  # hardcoded value to black_white color
        assert eraser.get_linewidth() == 3 * 1.5  # hardcoded value to black_white linewidth

        plt.close()

    def test_single_mcm_measure(self):
        """Test a final measurement of a mid circuit measurement."""

        with qml.queuing.AnnotatedQueue() as q:
            m0 = qml.measure(0)
            qml.expval(m0)
        _, ax = tape_mpl(qml.tape.QuantumScript.from_queue(q), style="black_white")

        assert len(ax.patches) == 6  # two measurement boxes
        assert ax.patches[3].get_x() == 1 - 0.75 / 2 + 0.2  # 1 - box_length/2 + pad
        assert qml.math.allclose(ax.patches[3].get_y(), 1 - 0.75 / 2 + 0.2)  # 1- box_length/2 + pad

        assert ax.patches[4].center == (
            1,
            1 + 0.15 * 0.75,
        )  # 1 +0.15 *box_length
        assert isinstance(ax.patches[5], mpl.patches.FancyArrow)

        [_, cwire] = ax.lines
        assert cwire.get_xdata() == [0, 0, 0, 1, 1, 1]
        assert cwire.get_ydata() == [1, 0, 1, 1, 1, 1]

        [pe1, pe2] = cwire.get_path_effects()

        # probably not a good way to test this, but the best I can figure out
        assert pe1._gc == {
            "linewidth": 5 * 1.5,  # hardcoded value to black_white linewidth
            "foreground": "black",  # lines.color for black white style
        }
        assert pe2._gc == {
            "linewidth": 3 * 1.5,  # hardcoded value to black_white linewidth
            "foreground": "white",  # figure.facecolor for black white sytle
        }

    def test_multiple_mcm_measure(self):
        """Test final measurements of multiple mid circuit measurements"""
        with qml.queuing.AnnotatedQueue() as q:
            m0 = qml.measure(0)
            m1 = qml.measure(0)
            _ = qml.measure(0)
            m2 = qml.measure(0)
            _ = qml.measure(0)
            qml.sample([m0, m1])
            qml.expval(m2)
        _, ax = qml.drawer.tape_mpl(qml.tape.QuantumScript.from_queue(q))

        [_, cwire0, cwire1, cwire2] = ax.lines
        assert cwire0.get_xdata() == [0, 0, 0, 5, 5, 5]
        assert cwire0.get_ydata() == [1, 0, 1, 1, 1, 1]
        assert cwire1.get_xdata() == [1, 1, 1, 5, 5, 5]
        assert cwire1.get_ydata() == [1.25, 0, 1.25, 1.25, 1.25, 1.25]
        assert cwire2.get_xdata() == [3, 3, 3, 5, 5, 5]
        assert cwire2.get_ydata() == [1.5, 0, 1.5, 1.5, 1.5, 1.5]

        assert len(ax.patches) == 18  # 6 * 3

        final_measure_box = ax.patches[15]
        assert final_measure_box.get_x() == 5 - 0.75 / 2 + 0.2  # 5 - box_length/2 + pad
        assert qml.math.allclose(
            final_measure_box.get_y(), 1 - 0.75 / 2 + 0.2
        )  # 1- box_length/2 + pad

        assert (
            final_measure_box.get_height() == 0.75 - 2 * 0.2 + 2 * 0.25
        )  # box_length - 2 * pad + 2 *cwire_scaling

        plt.close()
