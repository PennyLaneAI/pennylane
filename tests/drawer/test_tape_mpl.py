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


import pytest
from pytest_mock import mocker
import pennylane as qml

from pennylane.drawer import tape_mpl
from pennylane.tape import QuantumTape

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")


def test_empty_tape():
    """Edge case where the tape is empty. Use this to test return types."""

    fig, ax = tape_mpl(QuantumTape())

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes._axes.Axes)

    assert fig.axes == [ax]
    plt.close()


with QuantumTape() as tape1:
    qml.PauliX(0)
    qml.PauliX("a")
    qml.PauliX(1.234)


def test_fontsize():
    """Test default fontsize set with keyword argument."""

    _, ax = tape_mpl(tape1, fontsize=20)
    for t in ax.texts:
        assert t.get_fontsize() == 20
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


class TestWires:
    """Test that wire lines are produced correctly in different situations."""

    def test_empty_tape_wire_order(self):
        """Test situation with empty tape but specified wires and show_all_wires
        still draws wires."""

        _, ax = tape_mpl(QuantumTape(), wire_order=[0, 1, 2], show_all_wires=True)

        assert len(ax.lines) == 3
        for wire, line in enumerate(ax.lines):
            assert line.get_xdata() == (-1, 1)  # from -1 to number of layers
            assert line.get_ydata() == (wire, wire)

        plt.close()

    def test_single_layer(self):
        """Test a single layer with multiple wires. Check that the expected number
        of wires are drawn, and they are in the correct location."""

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.PauliY(1)
            qml.PauliZ(2)

        _, ax = tape_mpl(tape)

        assert len(ax.lines) == 3
        for wire, line in enumerate(ax.lines):
            assert line.get_xdata() == (-1, 1)  # from -1 to number of layers
            assert line.get_ydata() == (wire, wire)

        plt.close()

    def test_three_layers(self):
        """Test wire length when circuit has three layers."""

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.PauliX(0)
            qml.PauliX(0)

        _, ax = tape_mpl(tape)

        assert len(ax.lines) == 1
        assert ax.lines[0].get_xdata() == (-1, 3)  # from -1 to number of layers
        assert ax.lines[0].get_ydata() == (0, 0)

        plt.close()

    def test_wire_options(self):
        """Test wires are formatted by providing a wire_options dictionary."""

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.PauliX(1)

        rgba_red = (1, 0, 0, 1)
        _, ax = tape_mpl(tape, wire_options={"linewidth": 5, "color": rgba_red})

        for line in ax.lines:
            assert line.get_linewidth() == 5
            assert line.get_color() == rgba_red

        plt.close()


class TestSpecialGates:
    """Tests the gates with special drawing methods."""

    def test_SWAP(self):
        """Test SWAP gate special call"""

        with QuantumTape() as tape:
            qml.SWAP(wires=(0, 1))

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

        with QuantumTape() as tape:
            qml.CSWAP(wires=(0, 1, 2))

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

        with QuantumTape() as tape:
            qml.CNOT(wires=(0, 1))

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

        with QuantumTape() as tape:
            qml.Toffoli(wires=(0, 1, 2))

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

        with QuantumTape() as tape:
            qml.MultiControlledX(wires=[0, 1, 2, 3, 4])

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

        with QuantumTape() as tape:
            qml.MultiControlledX(wires=[0, 1, 2, 3, 4], control_values="0101")

        _, ax = tape_mpl(tape)

        assert ax.patches[0].get_facecolor() == (1.0, 1.0, 1.0, 1.0)  # white
        assert ax.patches[1].get_facecolor() == mpl.colors.to_rgba(plt.rcParams["lines.color"])
        assert ax.patches[2].get_facecolor() == (1.0, 1.0, 1.0, 1.0)
        assert ax.patches[3].get_facecolor() == mpl.colors.to_rgba(plt.rcParams["lines.color"])

        plt.close()

    def test_CZ(self):
        """Test CZ gets correct special call."""

        with QuantumTape() as tape:
            qml.CZ(wires=(0, 1))

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

    def test_Barrier(self):
        """Test Barrier gets correct special call."""

        with QuantumTape() as tape:
            qml.Barrier(wires=(0, 1, 2))

        _, ax = tape_mpl(tape)
        layer = 0

        assert len(ax.lines) == 3
        assert len(ax.collections) == 2

        plt.close()

    def test_WireCut(self):
        """Test WireCut gets correct special call."""

        with QuantumTape() as tape:
            qml.WireCut(wires=(0, 1))

        _, ax = tape_mpl(tape)
        layer = 0

        assert len(ax.lines) == 2
        assert len(ax.texts) == 1
        assert len(ax.collections) == 1

        plt.close()

    def test_Prod(self):
        with QuantumTape() as tape:
            qml.S(0) @ qml.T(0)

        _, ax = tape_mpl(tape)
        layer = 0

        assert len(ax.lines) == 1
        assert len(ax.collections) == 0

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

        with QuantumTape() as tape:
            qml.apply(op)

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

        with QuantumTape() as tape:
            qml.CRX(1.234, wires=(0, 1))

        _, ax = tape_mpl(tape, decimals=2)

        # two wire labels, so CRX is third text object
        assert ax.texts[2].get_text() == "RX\n(1.23)"
        plt.close()

    def test_control_values_str(self):
        """Test control values get displayed correctly when they are provided as a string."""

        with QuantumTape() as tape:
            qml.ControlledQubitUnitary(
                qml.matrix(qml.RX)(0, 0),
                control_wires=[0, 1, 2, 3],
                wires=[4],
                control_values="1010",
            )

        self.check_tape_controlled_qubit_unitary(tape)

    def test_control_values_bool(self):
        """Test control_values get displayed correctly when they are provided as a list of bools."""

        with QuantumTape() as tape:
            qubit_unitary = qml.QubitUnitary(qml.matrix(qml.RX)(0, 0), wires=4)
            qml.ops.op_math.Controlled(qubit_unitary, (0, 1, 2, 3), [1, 0, 1, 0])

        self.check_tape_controlled_qubit_unitary(tape)

    def check_tape_controlled_qubit_unitary(self, tape):
        """Checks the control symbols for a tape with some version of a controlled qubit unitary."""
        _, ax = tape_mpl(tape)
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

        with QuantumTape() as tape:
            qml.apply(op)

        _, ax = tape_mpl(tape)

        num_wires = len(op.wires)
        assert ax.texts[num_wires].get_text() == op.label()

        assert isinstance(ax.patches[0], mpl.patches.FancyBboxPatch)
        assert ax.patches[0].get_x() == -self.width / 2.0
        assert ax.patches[0].get_y() == -self.width / 2.0
        assert ax.patches[0].get_width() == self.width
        assert ax.patches[0].get_height() == num_wires - 1 + self.width

        plt.close()

    @pytest.mark.parametrize("op", general_op_data)
    def test_general_operations_decimals(self, op):
        """Check that the decimals argument affects text strings when applicable."""

        with QuantumTape() as tape:
            qml.apply(op)

        _, ax = tape_mpl(tape, decimals=2)

        num_wires = len(op.wires)
        assert ax.texts[num_wires].get_text() == op.label(decimals=2)

        plt.close()

    @pytest.mark.parametrize("wires, n", [((0,), 0), ((0, 1, 2), 0), ((0, 2), 4)])
    def test_notches(self, wires, n):
        """Test notches are included when non-active wires exist."""

        with QuantumTape() as tape:
            qml.QFT(wires=wires)

        _, ax = tape_mpl(tape, show_all_wires=True, wire_order=[0, 1, 2])
        assert len(ax.patches) == (n + 1)
        plt.close()

    def test_active_wire_notches_False(self):
        """Test active wire notches are disable with active_wire_notches=False."""

        with QuantumTape() as tape:
            qml.QFT(wires=(0, 3))

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

        with QuantumTape() as tape:
            for m in measurements:
                qml.apply(m)

        _, ax = tape_mpl(tape)

        assert len(ax.patches) == 3 * len(wires)

        for ii, w in enumerate(wires):
            assert ax.patches[3 * ii].get_x() == 1 - self.width / 2.0
            assert ax.patches[3 * ii].get_y() == w - self.width / 2.0
            assert ax.patches[3 * ii + 1].center == (1, w + 0.75 / 16)  # arc
            assert isinstance(ax.patches[3 * ii + 2], mpl.patches.FancyArrow)  # fancy arrow

        plt.close()

    def test_state(self):
        """Test state produces measurements on all wires."""

        with QuantumTape() as tape:
            qml.state()

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

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.PauliX(1)
            qml.PauliX(2)

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

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.PauliX(0)
            qml.PauliX(0)

        _, ax = tape_mpl(tape)

        for layer, box in enumerate(ax.patches):
            assert box.get_x() == layer - self.width / 2.0
            assert box.get_y() == -self.width / 2.0

        for t in ax.texts[1:]:
            assert t.get_text() == "X"
        plt.close()

    def test_blocking_IsingXX(self):
        """Tests the position of layers when a multiwire gate is blocking another gate on its empty wire."""

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.IsingXX(1.234, wires=(0, 2))
            qml.PauliX(1)

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
