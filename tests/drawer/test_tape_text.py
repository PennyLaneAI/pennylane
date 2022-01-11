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
Unit tests for the `pennylane.draw_text` function.
"""

import pytest
import pennylane as qml

from pennylane.drawer import tape_text
from pennylane.drawer.tape_text import _add_grouping_symbols, _add_op, _add_measurement
from pennylane.tape import QuantumTape

default_wire_map = {0:0, 1:1, 2:2, 3:3}

with QuantumTape() as tape:
    qml.RX(1.23456, wires=0)
    qml.RY(2.3456, wires="a")
    qml.RZ(3.4567, wires=1.234)


class TestHelperFunctions:

    @pytest.mark.parametrize("op, out", [(qml.PauliX(0), ["", "", "", ""]),
        (qml.CNOT(wires=(0,2)), ['╭', '│', '╰', '']),
        (qml.CSWAP(wires=(0,2,3)), ['╭', '│', '├', '╰'])
    ])
    def test_add_grouping_symbols(self, op, out):
        assert out == _add_grouping_symbols(op,  ["", "", "", ""], default_wire_map)

    @pytest.mark.parametrize("op, out", [(qml.expval(qml.PauliX(0)), ['<X>', '', '', '']),
        (qml.probs(wires=(0,2)), ['╭Probs', '│', '╰Probs', '']),
        (qml.var(qml.PauliX(1)), ['', 'Var[X]', '', '']),
        (qml.state(), ['State', 'State', 'State', 'State']),
        (qml.sample(), ['Sample', 'Sample', 'Sample', 'Sample'])
    ])
    def test_add_measurements(self, op, out):
        """Test private _add_measurement function renders as expected."""
        assert out == _add_measurement(op, [""]*4, default_wire_map, None)

    @pytest.mark.parametrize("op, out", [(qml.PauliX(0), ["─X","─","─","─"]),
        (qml.CNOT(wires=(0,2)), ['╭C', '│', '╰X', '─']),
        (qml.Toffoli(wires=(0,1,3)), ['╭C', '├C', '│', '╰X']),
        (qml.IsingXX(1.23, wires=(0,2)), ['╭IsingXX', '│', '╰IsingXX', '─'])
    ])
    def test_add_op(self, op, out):
        """Test adding the first operation to array of strings"""
        assert out == _add_op(op, ["─"]*4, default_wire_map, None)

    @pytest.mark.parametrize("op, out", [
        (qml.PauliY(1), ['─X', '─Y', '─', '─']),
        (qml.CNOT(wires=(1,2)), ['─X', '╭C', '╰X', '─']),
        (qml.CRX(1.23, wires=(2,3)), ['─X', '─', '╭C', '╰RX'])
    ])
    def test_add_second_op(self, op, out):
        """Test adding a second operation to the array of strings"""
        start = _add_op(qml.PauliX(0),  ["─"]*4, default_wire_map, None)
        assert out == _add_op(op, start, default_wire_map, None)

class TestEmptyTapes:

    def test_empty_tape(self):
        """Test using an empty tape returns a blank string"""
        assert tape_text(QuantumTape()) == ''

    def test_empty_tape_wire_order(self):
        """Test wire order and show_all_wires shows wires with empty tape."""
        expected = 'a: ───┤  \nb: ───┤  '
        out = tape_text(QuantumTape(),wire_order=['a', 'b'], show_all_wires=True)
        assert expected == out


class TestLabeling:

    def test_any_wire_labels(self):
        """Test wire labels with different kinds of objects."""

        split_str = tape_text(tape).split("\n")
        assert split_str[0][0:6] == '    0:'
        assert split_str[1][0:6] == '    a:'
        assert split_str[2][0:6] == '1.234:'

    def test_wire_order(self):
        """Test wire_order keyword changes order of the wires"""

        split_str = tape_text(tape, wire_order=[1.234, "a", 0, "b"]).split("\n")
        assert split_str[2][0:6] == '    0:'
        assert split_str[1][0:6] == '    a:'
        assert split_str[0][0:6] == '1.234:'

    def test_show_all_wires(self):
        """Test wire_order constains unused wires, show_all_wires 
        forces them to display."""

        split_str = tape_text(tape, wire_order=["b"], show_all_wires=True).split("\n")

        assert split_str[0][0:6] == '    b:'
        assert split_str[1][0:6] == '    0:'
        assert split_str[2][0:6] == '    a:'
        assert split_str[3][0:6] == '1.234:'

class TestDecimals:

    def test_decimals(self):

        expected = ("    0: ──RX(1.23)─┤  \n"
                    "    a: ──RY(2.35)─┤  \n" 
                    "1.234: ──RZ(3.46)─┤  ")

        assert tape_text(tape, decimals=2) == expected

    def test_decimals_0(self):
        """Test decimals=0 rounds to integers"""

        expected = ("    0: ──RX(1)─┤  \n"
                    "    a: ──RY(2)─┤  \n"
                    "1.234: ──RZ(3)─┤  ")

        assert tape_text(tape, decimals=0) == expected

class TestMaxLength:

    def test_max_length_default(self):
        """Test max length defaults to 100."""
        with QuantumTape() as tape_ml:
            for _ in range(50):
                qml.PauliX(0)
                qml.PauliY(1)
                
            for _ in range(3):
                qml.sample()

        out = tape_text(tape)
        assert 95 <= max(len(s) for s in out.split("\n")) <= 100

    @pytest.mark.parametrize("ml", [10, 15, 20])
    def test_setting_max_length(self, ml):

        with QuantumTape() as tape_ml:
            for _ in range(10):
                qml.PauliX(0)
                qml.PauliY(1)
                
            for _ in range(3):
                qml.sample()

        out = tape_text(tape, max_length=ml)
        assert max(len(s) for s in out.split("\n")) <= ml