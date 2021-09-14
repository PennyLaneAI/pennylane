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
from pennylane.circuit_drawer.drawable_layers import _recursive_find_layer, drawable_layers, drawable_grid
from pennylane.circuit_drawer.grid import Grid

class TestRecursiveFinedLayer:
    """Tests for `_recursive_find_layer`"""

    def test_first_layer(self):
        out = _recursive_find_layer(0, {0}, [{1}])
        assert out == 0

    def test_blocked_layer(self):
        out = _recursive_find_layer(0, {0}, [{0}])
        assert out == 1

    def test_recursion_no_block(self):

        out = _recursive_find_layer(2, {0}, [{1}, {1}, {1}])
        assert out == 0

    def test_recursion_block(self):

        # gets blocked at layer 1, so placed in layer 2
        out = _recursive_find_layer(3, {0}, [{1}, {0}, {1}, {1}])
        assert out == 2

class TestDrawableLayers:
    """Tests for `drawable_layers`"""

    def test_single_wires_no_blocking(self):
        """Simple case where nothing blocks each other"""

        ops = [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)]

        layers = drawable_layers(ops)

        assert layers == [set(ops)]

    def test_single_wires_blocking(self):
        """Simple case where single wire gates block each other"""

        ops = [qml.PauliX(0), qml.PauliX(0), qml.PauliX(0)]

        layers = drawable_layers(ops)

        assert layers == [{ops[0]}, {ops[1]}, {ops[2]}]

    def test_multiwire_blocking(self):
        """Multi-wire gate blocks on unused wire"""

        wire_map = {0:0, 1:1, 2:2}
        ops = [qml.PauliZ(1), qml.CNOT(wires=(0,2)), qml.PauliX(1)]

        layers = drawable_layers(ops, wire_map=wire_map)

        assert layers == [{ops[0]}, {ops[1]}, {ops[2]}]

    @pytest.mark.parametrize("measurement", (qml.state(), qml.sample() ))
    def test_all_wires_measurement(self, measurement):
        """All wires gates block on all available wires."""

        ops = [qml.PauliX(0), measurement, qml.PauliY(1)]

        layers = drawable_layers(ops)

        assert layers == [{ops[0]}, {ops[1]}, {ops[2]}]

    
class TestDrawableGrid:
    """Unittests for `drawable_grid`."""

    def test_empty(self):
        """Tests when empty operations and no wire map"""
        grid = drawable_grid([])
        
        assert grid == [[]]

        Grid_obj = Grid(grid)
        assert Grid_obj.raw_grid.shape == (1,0)

    def test_empty_wire_map(self):
        """Test grids compatible when no operations present but wire map """

        wire_map = {1:1, 2:2}
        grid = drawable_grid([], wire_map)

        assert grid == [[], []]
        
        Grid_obj = Grid(grid)
        assert Grid_obj.raw_grid.shape == (2,0)

    def test_single_wires_no_blocking(self):
        """Simple case where nothing blocks each other"""
        
        ops = [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)]

        grid = drawable_grid(ops)

        assert grid == [[ops[0]], [ops[1]], [ops[2]]]

        Grid_obj = Grid(grid)
        assert Grid_obj.raw_grid.shape == (3,1)

    def test_single_wires_blocking(self):
        """Simple case where single wire gates block each other"""

        ops = [qml.PauliX(0), qml.PauliX(0), qml.PauliX(0)]

        grid = drawable_grid(ops)

        assert grid == [ops]

        Grid_obj = Grid(grid)
        assert Grid_obj.raw_grid.shape == (1,3)

    def test_multiwire_blocking(self):
        """Multi-wire gate blocks on unused wire."""

        wire_map = {0:0, 1:1, 2:2}
        ops = [qml.PauliZ(1), qml.CNOT(wires=(0,2)), qml.PauliX(1)]

        grid = drawable_grid(ops, wire_map)

        assert grid[0] == [None, ops[1], None]
        assert grid[1] == [ops[0], None, ops[2]]
        assert grid[2] == [None, ops[1], None]

        Grid_obj = Grid(grid)
        assert Grid_obj.raw_grid.shape == (3,3)