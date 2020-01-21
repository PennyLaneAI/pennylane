# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

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
This module contains the CircuitDrawer class which is used to draw CircuitGraph instances.
"""
import abc
import math
from collections import OrderedDict

import numpy as np

import pennylane as qml

from .charsets import UnicodeCharSet
from .representation_resolver import RepresentationResolver

# pylint: disable=too-many-branches,too-many-arguments,too-many-return-statements,too-many-statements,consider-using-enumerate,too-many-instance-attributes


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

    Simultaneous access to both layers and wires via indexing is provided
    by keeping both the raw grid (wires are indexed) and the transposed raw grid
    (layers are indexed).

    Args:
        raw_grid (list, optional): Raw grid from which the Grid instance is built.
    """

    def __init__(self, raw_grid=None):
        if raw_grid is None:
            raw_grid = []

        self.raw_grid = raw_grid
        self.raw_grid_transpose = _transpose(raw_grid)

    def insert_layer(self, idx, layer):
        """Insert a layer into the Grid at the specified index.

        Args:
            idx (int): Index at which to insert the new layer
            layer (list): Layer that will be inserted
        """
        self.raw_grid_transpose.insert(idx, layer)
        self.raw_grid = _transpose(self.raw_grid_transpose)

        return self

    def append_layer(self, layer):
        """Append a layer to the Grid.

        Args:
            layer (list): Layer that will be appended
        """
        self.raw_grid_transpose.append(layer)
        self.raw_grid = _transpose(self.raw_grid_transpose)

        return self

    def replace_layer(self, idx, layer):
        """Replace a layer in the Grid at the specified index.

        Args:
            idx (int): Index of the layer to be replaced
            layer (list): Layer that replaces the old layer
        """
        self.raw_grid_transpose[idx] = layer
        self.raw_grid = _transpose(self.raw_grid_transpose)

        return self

    def insert_wire(self, idx, wire):
        """Insert a wire into the Grid at the specified index.

        Args:
            idx (int): Index at which to insert the new wire
            wire (list): Wire that will be inserted
        """
        self.raw_grid.insert(idx, wire)
        self.raw_grid_transpose = _transpose(self.raw_grid)

        return self

    def append_wire(self, wire):
        """Append a wire to the Grid.

        Args:
            wire (list): Wire that will be appended
        """
        self.raw_grid.append(wire)
        self.raw_grid_transpose = _transpose(self.raw_grid)

        return self

    @property
    def num_layers(self):
        """Number of layers in the Grid.

        Returns:
            int: Number of layers in the Grid
        """
        return len(self.raw_grid_transpose)

    def layer(self, idx):
        """Return the layer at the specified index.

        Args:
            idx (int): Index of the layer to be retrieved

        Returns:
            list: The layer at the specified index
        """
        return self.raw_grid_transpose[idx]

    @property
    def num_wires(self):
        """Number of wires in the Grid.

        Returns:
            int: Number of wires in the Grid
        """
        return len(self.raw_grid)

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
        return Grid(self.raw_grid.copy())

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


def _remove_duplicates(input_list):
    """Remove duplicate entries from a list.

    This operation preserves the order of the list's elements.

    Args:
        input_list (list[Hashable]): The list whose duplicate entries shall be removed

    Returns:
        list[Hashable]: The input list without duplicate entries
    """
    return list(OrderedDict.fromkeys(input_list))


class CircuitDrawer:
    """Creates a circuit diagram from the operators of a CircuitGraph in grid form.

    Args:
        raw_operation_grid (list[list[~.Operation]]): The CircuitGraph's operations
        raw_observable_grid (list[list[qml.operation.Observable]]): The CircuitGraph's observables
        charset (pennylane.circuit_drawer.CharSet, optional): The CharSet that shall be used for drawing.
        show_variable_names (bool, optional): Show variable names instead of variable values.
    """

    def resolve_representation(self, grid, representation_grid):
        """Resolve the string representation of the given Grid.

        Args:
            grid (pennylane.circuit_drawer.Grid): Grid that holds the circuit information
            representation_grid (pennylane.circuit_drawer.Grid): Grid that is used to store the string representations
        """
        for i in range(grid.num_layers):
            representation_layer = [""] * grid.num_wires

            for wire, operator in enumerate(grid.layer(i)):
                representation_layer[wire] = self.representation_resolver.element_representation(
                    operator, wire
                )

            representation_grid.append_layer(representation_layer)

    def resolve_decorations(self, grid, representation_grid, separate):
        """Resolve the decorations of the given Grid.

        Args:
            grid (pennylane.circuit_drawer.Grid): Grid that holds the circuit information
            representation_grid (pennylane.circuit_drawer.Grid): Grid that holds the string representations and into
                which the decorations will be inserted
            separate (bool): Insert decorations into separate layers

        Returns:
            list[int]: List with indices of inserted decoration layers
        """
        j = 0
        inserted_indices = []

        for i in range(grid.num_layers):
            layer_operators = _remove_duplicates(grid.layer(i))

            if not separate:
                decoration_layer = [""] * grid.num_wires

            for op in layer_operators:
                if op is None:
                    continue

                if isinstance(op, qml.operation.Tensor):
                    # pylint: disable=protected-access
                    wires = list(qml.utils._flatten(op.wires))
                else:
                    wires = op.wires

                if len(wires) > 1:
                    if separate:
                        decoration_layer = [""] * grid.num_wires

                    sorted_wires = wires.copy()
                    sorted_wires.sort()

                    decoration_layer[sorted_wires[0]] = self.charset.TOP_MULTI_LINE_GATE_CONNECTOR

                    for k in range(sorted_wires[0] + 1, sorted_wires[-1]):
                        if k in sorted_wires:
                            decoration_layer[k] = self.charset.MIDDLE_MULTI_LINE_GATE_CONNECTOR
                        else:
                            decoration_layer[k] = self.charset.EMPTY_MULTI_LINE_GATE_CONNECTOR

                    decoration_layer[
                        sorted_wires[-1]
                    ] = self.charset.BOTTOM_MULTI_LINE_GATE_CONNECTOR

                    if separate:
                        representation_grid.insert_layer(i + j, decoration_layer)
                        inserted_indices.append(i + j)
                        j += 1

            if not separate:
                representation_grid.insert_layer(i + j, decoration_layer)
                inserted_indices.append(i + j)
                j += 1

        return inserted_indices

    @staticmethod
    def pad_representation(
        representation_grid, pad_str, prepend_str, suffix_str, skip_indices,
    ):
        """Pads the given representation so that width inside layers is constant.

        Args:
            representation_grid (pennylane.circuit_drawer.Grid): Grid that holds the string representations that will be padded
            pad_str (str): String that shall be used for padding
            prepend_str (str): String that is prepended to all representations that are not skipped
            suffix_str (str): String that is appended to all representations
            skip_indices (list[int]): Indices of layers that should be skipped
        """
        for i in range(representation_grid.num_layers):
            layer = representation_grid.layer(i)
            max_width = max(map(len, layer))

            if i in skip_indices:
                continue

            # pylint: disable=cell-var-from-loop
            representation_grid.replace_layer(
                i,
                list(
                    map(
                        lambda x: prepend_str + str.ljust(x, max_width, pad_str) + suffix_str,
                        layer,
                    )
                ),
            )

    @staticmethod
    def move_multi_wire_gates(operator_grid):
        """Move multi-wire gates so that there are no interlocking multi-wire gates in the same layer.

        Args:
            operator_grid (pennylane.circuit_drawer.Grid): Grid that holds the circuit information and that will be edited.
        """
        n = operator_grid.num_layers
        i = -1
        while i < n - 1:
            i += 1

            this_layer = operator_grid.layer(i)
            layer_ops = _remove_duplicates(this_layer)
            other_layer = [None] * operator_grid.num_wires

            for j in range(len(layer_ops)):
                op = layer_ops[j]

                if op is None:
                    continue

                if len(op.wires) > 1:
                    sorted_wires = op.wires.copy()
                    sorted_wires.sort()

                    blocked_wires = list(range(sorted_wires[0], sorted_wires[-1] + 1))

                    for k in range(j + 1, len(layer_ops)):
                        other_op = layer_ops[k]

                        if other_op is None:
                            continue

                        other_sorted_wires = other_op.wires.copy()
                        other_sorted_wires.sort()
                        other_blocked_wires = list(
                            range(other_sorted_wires[0], other_sorted_wires[-1] + 1)
                        )

                        if not set(other_blocked_wires).isdisjoint(set(blocked_wires)):
                            op_indices = [
                                idx for idx, layer_op in enumerate(this_layer) if layer_op == op
                            ]

                            for l in op_indices:
                                other_layer[l] = op
                                this_layer[l] = None

                            break

            if not all([item is None for item in other_layer]):
                operator_grid.insert_layer(i + 1, other_layer)
                n += 1

    def __init__(
        self,
        raw_operation_grid,
        raw_observable_grid,
        charset=UnicodeCharSet,
        show_variable_names=False,
    ):
        self.charset = charset
        self.show_variable_names = show_variable_names
        self.representation_resolver = RepresentationResolver(charset, show_variable_names)
        self.operation_grid = Grid(raw_operation_grid)
        self.observable_grid = Grid(raw_observable_grid)
        self.operation_representation_grid = Grid()
        self.observable_representation_grid = Grid()
        self.operation_decoration_indices = []
        self.observable_decoration_indices = []

        CircuitDrawer.move_multi_wire_gates(self.operation_grid)

        # Resolve operator names
        self.resolve_representation(self.operation_grid, self.operation_representation_grid)
        self.resolve_representation(self.observable_grid, self.observable_representation_grid)

        # Add multi-wire gate lines
        self.operation_decoration_indices = self.resolve_decorations(
            self.operation_grid, self.operation_representation_grid, False,
        )
        self.observable_decoration_indices = self.resolve_decorations(
            self.observable_grid, self.observable_representation_grid, True,
        )

        CircuitDrawer.pad_representation(
            self.operation_representation_grid,
            charset.WIRE,
            "",
            2 * charset.WIRE,
            self.operation_decoration_indices,
        )

        CircuitDrawer.pad_representation(
            self.operation_representation_grid,
            charset.WIRE,
            "",
            "",
            set(range(self.operation_grid.num_layers)) - set(self.operation_decoration_indices),
        )

        CircuitDrawer.pad_representation(
            self.observable_representation_grid,
            " ",
            charset.MEASUREMENT + " ",
            " ",
            self.observable_decoration_indices,
        )

        CircuitDrawer.pad_representation(
            self.observable_representation_grid,
            charset.WIRE,
            "",
            "",
            set(range(self.observable_grid.num_layers)) - set(self.observable_decoration_indices),
        )

        self.full_representation_grid = self.operation_representation_grid.copy()
        self.full_representation_grid.append_grid_by_layers(self.observable_representation_grid)

    def draw(self):
        """Draw the circuit diagram.

        Returns:
            str: The circuit diagram
        """
        rendered_string = ""

        for i in range(self.full_representation_grid.num_wires):
            wire = self.full_representation_grid.wire(i)

            rendered_string += "{:2d}: {}".format(i, 2 * self.charset.WIRE)

            for s in wire:
                rendered_string += s

            rendered_string += "\n"

        for symbol, cache in [
            ("U", self.representation_resolver.unitary_matrix_cache),
            ("H", self.representation_resolver.hermitian_matrix_cache),
            ("M", self.representation_resolver.matrix_cache),
        ]:
            for idx, matrix in enumerate(cache):
                rendered_string += "{}{} =\n{}\n".format(symbol, idx, matrix)

        return rendered_string
