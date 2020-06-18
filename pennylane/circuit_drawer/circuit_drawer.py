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
from collections import OrderedDict, Iterable

import pennylane as qml

from .charsets import UnicodeCharSet
from .representation_resolver import RepresentationResolver
from .grid import Grid

# pylint: disable=too-many-branches,too-many-arguments,too-many-return-statements,too-many-statements,consider-using-enumerate,too-many-instance-attributes


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

    def __init__(
        self,
        raw_operation_grid,
        raw_observable_grid,
        charset=UnicodeCharSet,
        show_variable_names=False,
    ):
        self.operation_grid = Grid(raw_operation_grid)
        self.observable_grid = Grid(raw_observable_grid)
        self.charset = charset
        self.show_variable_names = show_variable_names

        self.make_wire_conversion_dicts(raw_operation_grid, raw_observable_grid)
        self.representation_resolver = RepresentationResolver(charset, show_variable_names)
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
            self.operation_grid, self.operation_representation_grid
        )
        self.observable_decoration_indices = self.resolve_decorations(
            self.observable_grid, self.observable_representation_grid
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

    def make_wire_conversion_dicts(self, raw_operation_grid, raw_observable_grid):
        """Prepare the dictionaries used to convert between internal and device wires.

        This conversion is necessary due to the fact that the circuit drawer internally uses
        ascending wires that have to be matched to the actual wires of the Operations inside
        the circuit.

        Args:
            raw_operation_grid (Iterable[~.Operator]): The raw grid of operations
            raw_observable_grid (Iterable[~.Operator]): The raw  grid of observables
        """
        # pylint: disable=protected-access
        all_operators = list(qml.utils._flatten(raw_operation_grid)) + list(
            qml.utils._flatten(raw_observable_grid)
        )
        all_wires = [op.wires for op in all_operators if op is not None]
        circuit_wires = sorted(set(qml.utils._flatten(all_wires)))
        internal_wires = list(range(len(circuit_wires)))

        self._cicuit_wire_to_internal_wire = dict(zip(circuit_wires, internal_wires))
        self._internal_wire_to_circuit_wire = dict(zip(internal_wires, circuit_wires))

    def circuit_wires_to_internal_wires(self, wires):
        """Convert one or multiple device wires to internal wires.

        Args:
            wires (Union[Iterable[int],int]): One or multiple device wires

        Returns:
            Union[Iterable[int],int]: The corresponding internal wires
        """
        if isinstance(wires, Iterable):
            return [self._cicuit_wire_to_internal_wire[wire] for wire in wires]

        return self._cicuit_wire_to_internal_wire[wires]

    def internal_wires_to_circuit_wires(self, wires):
        """Convert one or multiple internal wires to device wires.

        Args:
            wires (Union[Iterable[int],int]): One or multiple internal wires

        Returns:
            Union[Iterable[int],int]: The corresponding device wires
        """
        if isinstance(wires, Iterable):
            return [self._internal_wire_to_circuit_wire[wire] for wire in wires]

        return self._internal_wire_to_circuit_wire[wires]

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
                    operator, self.internal_wires_to_circuit_wires(wire)
                )

            representation_grid.append_layer(representation_layer)

    def add_multi_wire_connectors_to_layer(self, internal_wires, decoration_layer):
        """Add multi wire connectors for the given wires to a layer.

        Args:
            internal_wires (list[int]): The internal wires that are to be connected
            decoration_layer (list[str]): The decoration layer to which the wires will be added
        """
        min_wire = min(internal_wires)
        max_wire = max(internal_wires)

        decoration_layer[min_wire] = self.charset.TOP_MULTI_LINE_GATE_CONNECTOR

        for k in range(min_wire + 1, max_wire):
            if k in internal_wires:
                decoration_layer[k] = self.charset.MIDDLE_MULTI_LINE_GATE_CONNECTOR
            else:
                decoration_layer[k] = self.charset.EMPTY_MULTI_LINE_GATE_CONNECTOR

        decoration_layer[max_wire] = self.charset.BOTTOM_MULTI_LINE_GATE_CONNECTOR

    def resolve_decorations(self, grid, representation_grid):
        """Resolve the decorations of the given Grid.

        If decorations are in conflict, they are automatically spread over multiple layers.

        Args:
            grid (pennylane.circuit_drawer.Grid): Grid that holds the circuit information
            representation_grid (pennylane.circuit_drawer.Grid): Grid that holds the string representations and into
                which the decorations will be inserted

        Returns:
            list[int]: List with indices of inserted decoration layers
        """
        j = 0
        inserted_indices = []

        for i in range(grid.num_layers):
            layer_operators = _remove_duplicates(grid.layer(i))

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
                    internal_wires = self.circuit_wires_to_internal_wires(wires)
                    min_wire = min(internal_wires)
                    max_wire = max(internal_wires)

                    # If there is a conflict between decorations, we start a new decoration_layer
                    if any(
                        [decoration_layer[wire] != "" for wire in range(min_wire, max_wire + 1)]
                    ):
                        representation_grid.insert_layer(i + j, decoration_layer)
                        inserted_indices.append(i + j)
                        j += 1

                        decoration_layer = [""] * grid.num_wires

                    self.add_multi_wire_connectors_to_layer(internal_wires, decoration_layer)

            representation_grid.insert_layer(i + j, decoration_layer)
            inserted_indices.append(i + j)
            j += 1

        return inserted_indices

    @staticmethod
    def pad_representation(representation_grid, pad_str, prepend_str, suffix_str, skip_indices):
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

            # Take the current layer and pad it with the pad_str
            # and also prepend with prepend_str and append the suffix_str
            # pylint: disable=cell-var-from-loop
            representation_grid.replace_layer(
                i,
                list(
                    map(
                        lambda x: prepend_str + str.ljust(x, max_width, pad_str) + suffix_str, layer
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

    def draw(self):
        """Draw the circuit diagram.

        Returns:
            str: The circuit diagram
        """
        rendered_string = ""

        for i in range(self.full_representation_grid.num_wires):
            wire = self.full_representation_grid.wire(i)

            rendered_string += "{:2d}: {}".format(
                self.internal_wires_to_circuit_wires(i), 2 * self.charset.WIRE
            )

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
