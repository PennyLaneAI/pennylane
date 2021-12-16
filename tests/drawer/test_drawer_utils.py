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
Unit tests for the pennylane.drawer.utils` module.
"""

import pytest
import pennylane as qml
from pennylane.drawer.utils import default_wire_map, convert_wire_order


class TestDefaultWireMap:
    """Tests ``_default_wire_map`` helper function."""

    def test_empty(self):
        """Test creating an empty wire map"""

        wire_map = default_wire_map([])
        assert wire_map == {}

    def test_simple(self):
        """Test creating a wire map with wires that do not have successive ordering"""

        ops = [qml.PauliX(0), qml.PauliX(2), qml.PauliX(1)]

        wire_map = default_wire_map(ops)
        assert wire_map == {0: 0, 2: 1, 1: 2}

    def test_string_wires(self):
        """Test wire map works with string labelled wires."""

        ops = [qml.PauliY("a"), qml.CNOT(wires=("b", "c"))]

        wire_map = default_wire_map(ops)
        assert wire_map == {"a": 0, "b": 1, "c": 2}


class TestConvertWireOrder:
    """Tests the ``convert_wire_order`` utility function."""

    def test_no_wire_order(self):
        """Test that a wire map is produced if no wire order is passed."""

        ops = [qml.PauliX(0), qml.PauliX(2), qml.PauliX(1)]

        wire_map = convert_wire_order(ops)

        assert wire_map == {0: 0, 2: 1, 1: 2}

    def test_wire_order_ints(self):
        """Tests wire map produced when initial wires are integers."""

        ops = [qml.PauliX(0), qml.PauliX(2), qml.PauliX(1)]
        wire_order = [2, 1, 0]

        wire_map = convert_wire_order(ops, wire_order)
        assert wire_map == {2: 0, 1: 1, 0: 2}

    def test_wire_order_str(self):
        """Test wire map produced when initial wires are strings."""

        ops = [qml.CNOT(wires=("a", "b")), qml.PauliX("c")]
        wire_order = ("c", "b", "a")

        wire_map = convert_wire_order(ops, wire_order)
        assert wire_map == {"c": 0, "b": 1, "a": 2}

    def test_show_all_wires_false(self):
        """Test when `show_all_wires` is set to `False` only used wires are in the map."""

        ops = [qml.PauliX("a"), qml.PauliY("c")]
        wire_order = ["a", "b", "c", "d"]

        wire_map = convert_wire_order(ops, wire_order, show_all_wires=False)
        assert wire_map == {"a": 0, "c": 1}

    def test_show_all_wires_true(self):
        """Test when `show_all_wires` is set to `True` everything in ``wire_order`` is included."""

        ops = [qml.PauliX("a"), qml.PauliY("c")]
        wire_order = ["a", "b", "c", "d"]

        wire_map = convert_wire_order(ops, wire_order, show_all_wires=True)
        assert wire_map == {"a": 0, "b": 1, "c": 2, "d": 3}
