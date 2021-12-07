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
This module contains the Grid class which emulates a 2D array.
"""
import numpy as np


def _transpose(target_list):
    """Transpose the given list of lists.

    Args:
        target_list (list[list[object]]): List of list that will be transposed

    Returns:
        list[list[object]]: Transposed list of lists
    """
    return list(map(list, zip(*target_list)))


class Grid:
    """Helper class to manage Gates aligned in a grid.

    The rows of the Grid are referred to as "wires",
    whereas the columns of the Grid are reffered to as "layers".

    Args:
        raw_grid (list, optional): Raw grid from which the Grid instance is built.
    """

    def __init__(self, raw_grid=None):
        if raw_grid is None:
            # Let initialization pend until first data is entered
            self.raw_grid = None
        else:
            self.raw_grid = np.array(raw_grid, dtype=object)
            if len(self.raw_grid.shape) != 2:
                raise ValueError(
                    f"The entered raw grid was not parsed as two-dimensional array: {raw_grid}"
                )

    def insert_layer(self, idx, layer):
        """Insert a layer into the Grid at the specified index.

        If the Grid is empty, the index is ignored and the layer inserted.

        Args:
            idx (int): Index at which to insert the new layer
            layer (list): Layer that will be inserted
        """
        if self.raw_grid is not None:
            self.raw_grid = np.insert(self.raw_grid, idx, np.array([layer], dtype=object), axis=1)
        else:
            self.raw_grid = np.array([layer], dtype=object).T

        return self

    def append_layer(self, layer):
        """Append a layer to the Grid.

        Args:
            layer (list): Layer that will be appended
        """
        if self.raw_grid is not None:
            self.raw_grid = np.append(self.raw_grid, np.array([layer], dtype=object).T, axis=1)
        else:
            self.raw_grid = np.array([layer], dtype=object).T

        return self

    def replace_layer(self, idx, layer):
        """Replace a layer in the Grid at the specified index.

        Args:
            idx (int): Index of the layer to be replaced
            layer (list): Layer that replaces the old layer
        """
        if self.raw_grid is not None:
            self.raw_grid[:, idx] = np.array(layer, dtype=object)
        else:
            raise AttributeError("Can't replace layer. The Grid has not yet been initialized.")

        return self

    def insert_wire(self, idx, wire):
        """Insert a wire into the Grid at the specified index.

        If the Grid is empty, the index is ignored and the wire inserted.

        Args:
            idx (int): Index at which to insert the new wire
            wire (list): Wire that will be inserted
        """
        if self.raw_grid is not None:
            self.raw_grid = np.insert(self.raw_grid, idx, np.array([wire], dtype=object), axis=0)
        else:
            self.raw_grid = np.array([wire], dtype=object)

        return self

    def append_wire(self, wire):
        """Append a wire to the Grid.

        Args:
            wire (list): Wire that will be appended
        """
        if self.raw_grid is not None:
            self.raw_grid = np.append(self.raw_grid, np.array([wire], dtype=object), axis=0)
        else:
            self.raw_grid = np.array([wire], dtype=object)

        return self

    @property
    def num_layers(self):
        """Number of layers in the Grid.

        Returns:
            int: Number of layers in the Grid
        """
        if self.raw_grid is not None:
            return self.raw_grid.shape[1]

        return 0

    def layer(self, idx):
        """Return the layer at the specified index.

        Args:
            idx (int): Index of the layer to be retrieved

        Returns:
            list: The layer at the specified index
        """
        return self.raw_grid[:, idx]

    @property
    def num_wires(self):
        """Number of wires in the Grid.

        Returns:
            int: Number of wires in the Grid
        """
        if self.raw_grid is not None:
            return self.raw_grid.shape[0]

        return 0

    def wire(self, idx):
        """Return the wire at the specified index.

        Args:
            idx (int): Index of the wire to be retrieved

        Returns:
            list: The wire at the specified index
        """
        return self.raw_grid[idx]

    def copy(self):
        """Create a copy of the Grid.

        Returns:
            Grid: A copy of the Grid
        """
        return Grid(self.raw_grid.copy()) if self.raw_grid is not None else Grid()

    def append_grid_by_layers(self, other_grid):
        """Append the layers of another Grid to this Grid.

        Args:
            other_grid (pennylane.circuit_drawer.Grid): Grid whose layers will be appended
        """
        for i in range(other_grid.num_layers):
            self.append_layer(other_grid.layer(i))

        return self

    def __str__(self):
        """String representation"""
        ret = ""
        for wire in self.raw_grid:
            ret += str(wire)
            ret += "\n"

        return ret
