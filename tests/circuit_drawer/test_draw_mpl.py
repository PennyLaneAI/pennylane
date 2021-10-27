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
"""This file tests the ``qml.circuit_drawer.draw_mpl`` function."""

import pytest
from pytest_mock import mocker
import pennylane as qml

from pennylane.circuit_drawer import draw_mpl
from pennylane.tape import QuantumTape

mpl = pytest.importorskip("matplotlib")

with QuantumTape() as tape1:
    qml.PauliX(0)
    qml.PauliX("a")
    qml.PauliX(1.234)


class TestLabelling:
    """Test the labels for the wires."""

    def test_tape_wires(self, mocker):
        """Test labels when determined from tape wires"""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")
        draw_mpl(tape1)

        mock_drawer().label.assert_called_with([0, "a", 1.234], text_options=None)

    def test_wire_order(self, mocker):
        """Test labels when full wire order provided"""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        draw_mpl(tape1, wire_order=[1.234, "a", 0])
        mock_drawer().label.assert_called_with([1.234, "a", 0], text_options=None)

    def test_partial_wire_order(self, mocker):
        """Test wire order that only contains some tape wires."""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        draw_mpl(tape1, wire_order=["a", 1.234])
        mock_drawer().label.assert_called_with(["a", 1.234, 0], text_options=None)

    def test_wire_order_unused_wire(self, mocker):
        """Test when wire order contains wire not in the tape"""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        draw_mpl(tape1, wire_order=["nope", "not there", 3])
        mock_drawer().label.assert_called_with([0, "a", 1.234], text_options=None)

    def test_show_all_wires_unused_wire(self, mocker):
        """Test empty wires in wire order added when ``show_all_wires=True``"""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        draw_mpl(tape1, wire_order=["aux", 0, "a", 1.234], show_all_wires=True)
        mock_drawer().label.assert_called_with(["aux", 0, "a", 1.234], text_options=None)

    def test_label_options(self, mocker):
        """Test text_options"""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        draw_mpl(tape1, label_options={"fontsize": 10})
        mock_drawer().label.assert_called_with([0, "a", 1.234], text_options={"fontsize": 10})


class TestWires:
    """Test wire lines are produced correctly in different situations"""

    def test_empty_tape_wire_order(self):
        """Test situation with empty tape but specified wires."""

        fig, ax = draw_mpl(QuantumTape(), wire_order=[0, 1, 2], show_all_wires=True)

        assert len(ax.lines) == 3
        for wire, line in enumerate(ax.lines):
            assert line.get_xdata() == (-1, 1)
            assert line.get_ydata() == (wire, wire)

    def test_single_layer(self):
        """Test a single layer with multiple wires."""

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.PauliY(1)
            qml.PauliZ(2)

        fig, ax = draw_mpl(tape)

        assert len(ax.lines) == 3
        for wire, line in enumerate(ax.lines):
            assert line.get_xdata() == (-1, 1)
            assert line.get_ydata() == (wire, wire)

    def test_three_layers(self):
        """Test wire length when circuit has three layers."""

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.PauliX(0)
            qml.PauliX(0)

        fig, ax = draw_mpl(tape)

        assert len(ax.lines) == 1
        assert ax.lines[0].get_xdata() == (-1, 3)
        assert ax.lines[0].get_ydata() == (0, 0)

    def test_wire_options(self):
        """Test wires are formatted by provided dictionary."""

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.PauliX(1)

        rgba_red = (1, 0, 0, 1)
        fig, ax = draw_mpl(tape, wire_options={"linewidth": 5, "color": rgba_red})

        for line in ax.lines:
            assert line.get_linewidth() == 5
            assert line.get_color() == rgba_red


class TestSpecialGates:
    """Tests the gates with special drawing methods."""

    def test_SWAP(self, mocker):
        """Test SWAP gate gets special call"""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.SWAP(wires=(0, 1))

        draw_mpl(tape)

        mock_drawer().SWAP.assert_called_with(0, [0, 1])

    def test_CSWAP(self, mocker):
        """Test CSWAP gets special call"""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.CSWAP(wires=(0, 1, 2))

        draw_mpl(tape)

        mock_drawer().ctrl.assert_called_with(0, wires=0, wires_target=[1, 2])
        mock_drawer().SWAP.assert_called_with(0, wires=[1, 2])

    def test_CNOT(self, mocker):
        """Test CNOT gets a special call"""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.CNOT(wires=(0, 1))

        draw_mpl(tape)

        mock_drawer().CNOT.assert_called_with(0, [0, 1])

    def test_Toffoli(self, mocker):
        """Test Toffoli gets a special call."""

        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.Toffoli(wires=(0, 1, 2))

        draw_mpl(tape)

        mock_drawer().CNOT.assert_called_with(0, [0, 1, 2])

    def test_MultiControlledX_no_control_values(self, mocker):
        """Test MultiControlledX gets a special call."""

        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.MultiControlledX(control_wires=[0, 1, 2, 3], wires=4)

        draw_mpl(tape)

        mock_drawer().ctrl.assert_called_with(
            0, [0, 1, 2, 3], 4, control_values=[True, True, True, True]
        )
        mock_drawer()._target_x.assert_called_with(0, 4)

    def test_MultiControlledX_control_values(self, mocker):
        """Test MultiControlledX with provided control values."""

        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.MultiControlledX(control_wires=[0, 1, 2, 3], wires=4, control_values="0101")

        draw_mpl(tape)

        mock_drawer().ctrl.assert_called_with(
            0, [0, 1, 2, 3], 4, control_values=[False, True, False, True]
        )
        mock_drawer()._target_x.assert_called_with(0, 4)

    def test_CZ(self, mocker):
        """Test CZ gets a special call."""

        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.CZ(wires=(0, 1))

        draw_mpl(tape)

        mock_drawer().ctrl.assert_called_with(0, [0, 1])


class TestControlledGates:
    """Tests generic controlled gates"""

    def test_CY(self, mocker):
        """Test a controlled non-parametric operation."""

        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.CY(wires=(0, 1))

        draw_mpl(tape)

        mock_drawer().ctrl.assert_called_with(0, [0], wires_target=[1])
        mock_drawer().box_gate.assert_called_with(
            0, [1], "Y", box_options={"zorder": 4}, text_options={"zorder": 5}
        )

    def test_CRX(self, mocker):
        """Test a controlled parametric operation."""

        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.CRX(1.234, wires=(0, 1))

        draw_mpl(tape)

        mock_drawer().ctrl.assert_called_with(0, [0], wires_target=[1])
        mock_drawer().box_gate.assert_called_with(
            0, [1], "RX", box_options={"zorder": 4}, text_options={"zorder": 5}
        )

    def test_CRX_decimals(self, mocker):
        """Test a controlled parametric operation with specified decimals."""

        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.CRX(1.234, wires=(0, 1))

        draw_mpl(tape, decimals=2)

        mock_drawer().ctrl.assert_called_with(0, [0], wires_target=[1])
        mock_drawer().box_gate.assert_called_with(
            0, [1], "RX\n(1.23)", box_options={"zorder": 4}, text_options={"zorder": 5}
        )


class TestGeneralOperations:
    """Tests general operations."""

    def test_RX(self, mocker):
        """Test RX gate"""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.RX(1.234, wires=0)

        draw_mpl(tape)

        mock_drawer().box_gate.assert_called_with(0, [0], "RX")

    def test_RX_decimals(self, mocker):
        """Test RX gate"""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.RX(1.234, wires=0)

        draw_mpl(tape, decimals=2)

        mock_drawer().box_gate.assert_called_with(0, [0], "RX\n(1.23)")

    def test_IsingXX(self, mocker):
        """Test a standard multiwire gate."""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.IsingXX(1.234, wires=(0, 1))

        draw_mpl(tape)

        mock_drawer().box_gate.assert_called_with(0, [0, 1], "IsingXX")

    def test_QFT(self, mocker):
        """Test a template operation"""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.QFT(wires=range(3))

        draw_mpl(tape)

        mock_drawer().box_gate.assert_called_with(0, [0, 1, 2], "QFT")


class TestMeasurements:
    """Tests measurements are drawn correctly"""

    def test_expval(self, mocker):
        """Test expval produce measure boxes"""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.expval(qml.PauliX(0))

        draw_mpl(tape)

        mock_drawer().measure.assert_called_with(1, 0)

    def test_state(self, mocker):
        """Test state produces measurements on all wires."""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.state()

        draw_mpl(tape, wire_order=[0, 1, 2], show_all_wires=True)

        call_list = [((1, 0),), ((1, 1),), ((1, 2),)]
        assert mock_drawer().measure.call_args_list == call_list

    def test_probs(self, mocker):
        """Test probs with wires."""

        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.probs(wires=(0, 1, 2))

        draw_mpl(tape)

        call_list = [((1, 0),), ((1, 1),), ((1, 2),)]
        assert mock_drawer().measure.call_args_list == call_list


class TestLayering:
    """Tests operations are placed into layers correctly."""

    def test_single_layer_multiple_wires(self, mocker):
        """Tests mulitple gates all in the same layer"""
        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.PauliX(1)
            qml.PauliX(2)

        draw_mpl(tape)

        # no order in set, so may be called in a different order
        mock_drawer().box_gate.assert_any_call(0, [0], "X")
        mock_drawer().box_gate.assert_any_call(0, [1], "X")
        mock_drawer().box_gate.assert_any_call(0, [2], "X")

    def test_three_layers_one_wire(self, mocker):
        """Tests multiple gates all on the same wire"""

        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.PauliX(0)
            qml.PauliX(0)

        draw_mpl(tape)

        mock_drawer().box_gate.assert_any_call(0, [0], "X")
        mock_drawer().box_gate.assert_any_call(1, [0], "X")
        mock_drawer().box_gate.assert_any_call(2, [0], "X")

    def test_blocking_IsingXX(self, mocker):
        """Tests a multiwire gate blocking another on its empty wire"""

        mock_drawer = mocker.patch("pennylane.circuit_drawer.draw.MPLDrawer")

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.IsingXX(1.234, wires=(0, 2))
            qml.PauliX(1)

        draw_mpl(tape, wire_order=[0, 1, 2])

        mock_drawer().box_gate.assert_any_call(0, [0], "X")
        mock_drawer().box_gate.assert_any_call(1, [0, 2], "IsingXX")
        mock_drawer().box_gate.assert_any_call(2, [1], "X")
