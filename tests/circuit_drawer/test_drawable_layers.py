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
Unit tests for the pennylane.circuit_drawer.drawable_layers` module.
"""

import pytest

import pennylane as qml
from pennylane.circuit_drawer.drawable_layers import _default_wire_map, _recursive_find_layer, drawable_layers, drawable_grid

class TestWireMap:
    """Tests ``_default_wire_map`` helper function."""

    def test_empty(self):

        wire_map = _default_wire_map([])
        assert wire_map == {}

    def test_simple(self):
        """Test wires not sucessive ordering"""

        ops = [qml.PauliX(0), qml.PauliX(2), qml.PauliX(1)]

        wire_map = _default_wire_map(ops)
        assert wire_map == {0:0, 2:1, 1:2}

    def test_string_wires(self):
        """Test wire map works with string labelled wires."""

        ops = [qml.PauliY("a"), qml.CNOT(wires=("b", "c"))]

        wire_map = _default_wire_map(ops)
        assert wire_map == {"a": 0, "b": 1, "c": 2}