# Copyright 2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.circuit_drawer.grid` module.
"""
import pytest
import numpy as np

from pennylane.circuit_drawer import Grid
from pennylane.circuit_drawer.grid import _transpose


class TestFunctions:
    """Test the helper functions."""

    @pytest.mark.parametrize(
        "input,expected_output",
        [
            ([[0, 1], [2, 3]], [[0, 2], [1, 3]]),
            ([[0, 1, 2], [3, 4, 5]], [[0, 3], [1, 4], [2, 5]]),
            ([[0], [1], [2]], [[0, 1, 2]]),
        ],
    )
    def test_transpose(self, input, expected_output):
        """Test that transpose transposes a list of list."""
        assert _transpose(input) == expected_output

    @pytest.mark.parametrize(
        "input",
        [
            [[0, 1], [2, 3]],
            [[0, 2], [1, 3]],
            [[0, 1, 2], [3, 4, 5]],
            [[0, 3], [1, 4], [2, 5]],
            [[0], [1], [2]],
            [[0, 1, 2]],
        ],
    )
    def test_transpose_squared(self, input):
        """Test that transpose transposes a list of list."""
        assert _transpose(_transpose(input)) == input


class TestGrid:
    """Test the Grid helper class."""

    def test_empty_init(self):
        """Test that the Grid class is initialized correctly when no raw_grid is given."""
        grid = Grid()

        assert grid.num_layers == 0
        assert grid.num_wires == 0
        assert grid.raw_grid is None

    def test_init(self):
        """Test that the Grid class is initialized correctly."""

        raw_grid = [[0, 3], [1, 4], [2, 5]]
        grid = Grid(raw_grid)

        assert np.array_equal(grid.raw_grid, raw_grid)
        assert np.array_equal(grid.raw_grid.T, [[0, 1, 2], [3, 4, 5]])

    @pytest.mark.parametrize(
        "idx,expected_transposed_grid",
        [
            (0, [[6, 7, 8], [0, 1, 2], [3, 4, 5]]),
            (1, [[0, 1, 2], [6, 7, 8], [3, 4, 5]]),
            (2, [[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
        ],
    )
    def test_insert_layer(self, idx, expected_transposed_grid):
        """Test that layer insertion works properly."""

        raw_grid = [[0, 3], [1, 4], [2, 5]]
        grid = Grid(raw_grid)

        grid.insert_layer(idx, [6, 7, 8])

        assert np.array_equal(grid.raw_grid.T, expected_transposed_grid)
        assert np.array_equal(grid.raw_grid, _transpose(expected_transposed_grid))

    def test_append_layer(self):
        """Test that layer appending works properly."""

        raw_grid = [[0, 3], [1, 4], [2, 5]]
        grid = Grid(raw_grid)

        grid.append_layer([6, 7, 8])

        assert np.array_equal(grid.raw_grid, [[0, 3, 6], [1, 4, 7], [2, 5, 8]])
        assert np.array_equal(grid.raw_grid.T, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    @pytest.mark.parametrize(
        "idx,expected_transposed_grid", [(0, [[6, 7, 8], [3, 4, 5]]), (1, [[0, 1, 2], [6, 7, 8]]),]
    )
    def test_replace_layer(self, idx, expected_transposed_grid):
        """Test that layer replacement works properly."""

        raw_grid = [[0, 3], [1, 4], [2, 5]]
        grid = Grid(raw_grid)

        grid.replace_layer(idx, [6, 7, 8])

        assert np.array_equal(grid.raw_grid.T, expected_transposed_grid)
        assert np.array_equal(grid.raw_grid, _transpose(expected_transposed_grid))

    @pytest.mark.parametrize(
        "idx,expected_grid",
        [
            (0, [[6, 7], [0, 3], [1, 4], [2, 5]]),
            (1, [[0, 3], [6, 7], [1, 4], [2, 5]]),
            (2, [[0, 3], [1, 4], [6, 7], [2, 5]]),
            (3, [[0, 3], [1, 4], [2, 5], [6, 7]]),
        ],
    )
    def test_insert_wire(self, idx, expected_grid):
        """Test that wire insertion works properly."""

        raw_grid = [[0, 3], [1, 4], [2, 5]]
        grid = Grid(raw_grid)

        grid.insert_wire(idx, [6, 7])

        assert np.array_equal(grid.raw_grid, expected_grid)
        assert np.array_equal(grid.raw_grid.T, _transpose(expected_grid))

    def test_append_wire(self):
        """Test that wire appending works properly."""

        raw_grid = [[0, 3], [1, 4], [2, 5]]
        grid = Grid(raw_grid)

        grid.append_wire([6, 7])

        assert np.array_equal(grid.raw_grid, [[0, 3], [1, 4], [2, 5], [6, 7]])
        assert np.array_equal(grid.raw_grid.T, [[0, 1, 2, 6], [3, 4, 5, 7]])

    @pytest.mark.parametrize(
        "raw_grid,expected_num_layers",
        [
            ([[6, 7], [0, 3], [1, 4], [2, 5]], 2),
            ([[0, 1, 2, 6], [3, 4, 5, 7]], 4),
            ([[0, 2, 6], [3, 5, 7]], 3),
        ],
    )
    def test_num_layers(self, raw_grid, expected_num_layers):
        """Test that num_layers returns the correct number of layers."""
        grid = Grid(raw_grid)

        assert grid.num_layers == expected_num_layers

    @pytest.mark.parametrize(
        "raw_grid,expected_num_wires",
        [
            ([[6, 7], [0, 3], [1, 4], [2, 5]], 4),
            ([[0, 1, 2, 6], [3, 4, 5, 7]], 2),
            ([[0, 2, 6], [3, 5, 7]], 2),
        ],
    )
    def test_num_wires(self, raw_grid, expected_num_wires):
        """Test that num_layers returns the correct number of wires."""
        grid = Grid(raw_grid)

        assert grid.num_wires == expected_num_wires

    @pytest.mark.parametrize(
        "raw_transposed_grid",
        [
            ([[6, 7], [0, 3], [1, 4], [2, 5]]),
            ([[0, 1, 2, 6], [3, 4, 5, 7]]),
            ([[0, 2, 6], [3, 5, 7]]),
        ],
    )
    def test_layer(self, raw_transposed_grid):
        """Test that layer returns the correct layer."""
        grid = Grid(_transpose(raw_transposed_grid))

        for idx, layer in enumerate(raw_transposed_grid):
            assert np.array_equal(grid.layer(idx), layer)

    @pytest.mark.parametrize(
        "raw_grid",
        [
            ([[6, 7], [0, 3], [1, 4], [2, 5]]),
            ([[0, 1, 2, 6], [3, 4, 5, 7]]),
            ([[0, 2, 6], [3, 5, 7]]),
        ],
    )
    def test_wire(self, raw_grid):
        """Test that wire returns the correct wire."""
        grid = Grid(raw_grid)

        for idx, wire in enumerate(raw_grid):
            assert np.array_equal(grid.wire(idx), wire)

    def test_copy(self):
        """Test that copy copies the grid."""
        raw_grid = [[0, 3], [1, 4], [2, 5]]
        grid = Grid(raw_grid)

        other_grid = grid.copy()

        # Assert that everything is indeed copied
        assert other_grid is not grid
        assert other_grid.raw_grid is not grid.raw_grid

        # Assert the copy is correct
        assert np.array_equal(other_grid.raw_grid, grid.raw_grid)

    def test_append_grid_by_layers(self):
        """Test appending a grid to another by layers."""
        raw_grid_transpose1 = [[0, 3], [1, 4], [2, 5]]
        raw_grid_transpose2 = [[6, 7], [8, 9]]

        grid1 = Grid(_transpose(raw_grid_transpose1))
        grid2 = Grid(_transpose(raw_grid_transpose2))

        grid1.append_grid_by_layers(grid2)

        assert np.array_equal(grid1.raw_grid.T, [[0, 3], [1, 4], [2, 5], [6, 7], [8, 9]])
        assert np.array_equal(grid1.raw_grid, _transpose([[0, 3], [1, 4], [2, 5], [6, 7], [8, 9]]))

    def test_str(self):
        """Test string rendering of Grid."""
        raw_grid = [[0, 3], [1, 4], [2, 5]]
        grid = Grid(raw_grid)

        assert str(grid) == "[0 3]\n[1 4]\n[2 5]\n"

    def test_replace_error_message(self):
        """Test that an exception is raised when replacing layers in
        an uninitialized Grid is attempted."""
        grid = Grid()

        with pytest.raises(AttributeError, match="Can't replace layer. The Grid has not yet been initialized."):
            grid.replace_layer(1, [1, 2, 3])
