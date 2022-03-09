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
This module contains the CircuitDrawer class which is used to draw CircuitGraph instances.
"""
from collections import OrderedDict

from pennylane.wires import Wires
from pennylane.utils import _flatten
from .charsets import CHARSETS, CharSet
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
        wires (Wires): all wires on the device for which the circuit is drawn
        charset (str, pennylane.circuit_drawer.CharSet, optional): The CharSet that shall be used for drawing.
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        max_length (int, optional): Maximum string width (columns) when printing the circuit to the CLI.
        _label_offsets (dict[strin, int], optional): Offset the printed index of different symbol types in nested circuits.
    """

    def __init__(
        self,
        raw_operation_grid,
        raw_observable_grid,
        wires,
        charset=None,
        show_all_wires=False,
        max_length=None,
        _label_offsets=None,
    ):
        self.operation_grid = Grid(raw_operation_grid)
        self.observable_grid = Grid(raw_observable_grid)
        self.wires = wires
        self.active_wires = self.extract_active_wires(raw_operation_grid, raw_observable_grid)

        if charset is None:
            self.charset = CHARSETS["unicode"]
        elif isinstance(charset, type) and issubclass(charset, CharSet):
            self.charset = charset
        else:
            if charset not in CHARSETS:
                supported_char = ", ".join(CHARSETS.keys())
                raise ValueError(
                    f"Charset '{charset}' is not supported. Supported charsets: {supported_char}."
                )
            self.charset = CHARSETS[charset]

        if show_all_wires:
            # if the provided wires include empty wires, make sure they are included
            # as active wires
            self.active_wires = wires.all_wires([wires, self.active_wires])

        # We add a -2 character offset to account for some downstream formatting
        self.max_length = max_length - 2 if max_length is not None else None

        self.representation_resolver = RepresentationResolver(
            self.charset, label_offsets=_label_offsets
        )
        self.operation_representation_grid = Grid()
        self.observable_representation_grid = Grid()
        self.operation_decoration_indices = []
        self.observable_decoration_indices = []

        self.move_multi_wire_gates(self.operation_grid)

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
            self.charset.WIRE,
            "",
            2 * self.charset.WIRE,
            self.operation_decoration_indices,
        )

        CircuitDrawer.pad_representation(
            self.operation_representation_grid,
            self.charset.WIRE,
            "",
            "",
            set(range(self.operation_grid.num_layers)) - set(self.operation_decoration_indices),
        )

        CircuitDrawer.pad_representation(
            self.observable_representation_grid,
            " ",
            self.charset.MEASUREMENT + " ",
            " ",
            self.observable_decoration_indices,
        )

        CircuitDrawer.pad_representation(
            self.observable_representation_grid,
            self.charset.WIRE,
            "",
            "",
            set(range(self.observable_grid.num_layers)) - set(self.observable_decoration_indices),
        )

        self.full_representation_grid = self.operation_representation_grid.copy()
        self.full_representation_grid.append_grid_by_layers(self.observable_representation_grid)

    def extract_active_wires(self, raw_operation_grid, raw_observable_grid):
        """Get the subset of wires on the device that are used in the circuit.

        Args:
            raw_operation_grid (Iterable[~.Operator]): The raw grid of operations
            raw_observable_grid (Iterable[~.Operator]): The raw  grid of observables

        Return:
            Wires: active wires on the device
        """
        # pylint: disable=protected-access
        all_operators = list(_flatten(raw_operation_grid)) + list(_flatten(raw_observable_grid))
        all_wires_with_duplicates = [op.wires for op in all_operators if op is not None]
        # make Wires object containing all used wires
        all_wires = Wires.all_wires(all_wires_with_duplicates)
        # shared wires will observe the ordering of the device's wires
        shared_wires = Wires.shared_wires([self.wires, all_wires])
        return shared_wires

    def resolve_representation(self, grid, representation_grid):
        """Resolve the string representation of the given Grid.

        Args:
            grid (pennylane.circuit_drawer.Grid): Grid that holds the circuit information
            representation_grid (pennylane.circuit_drawer.Grid): Grid that is used to store the string representations
        """
        for i in range(grid.num_layers):
            representation_layer = [""] * grid.num_wires

            for wire_indices, operator in enumerate(grid.layer(i)):
                wire = self.active_wires[wire_indices]
                representation_layer[
                    wire_indices
                ] = self.representation_resolver.element_representation(operator, wire)

            representation_grid.append_layer(representation_layer)

    def add_multi_wire_connectors_to_layer(self, wire_indices, decoration_layer):
        """Add multi wire connectors for the given wires to a layer.

        Args:
            wire_indices (list[int]): The indices of wires that are to be connected
            decoration_layer (list[str]): The decoration layer to which the wires will be added
        """
        min_wire = min(wire_indices)
        max_wire = max(wire_indices)

        decoration_layer[min_wire] = self.charset.TOP_MULTI_LINE_GATE_CONNECTOR

        for k in range(min_wire + 1, max_wire):
            if k in wire_indices:
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

                wires = op.wires
                wire_indices = self.active_wires.indices(wires)

                if len(wire_indices) > 1:
                    min_wire = min(wire_indices)
                    max_wire = max(wire_indices)

                    # If there is a conflict between decorations, we start a new decoration_layer
                    if any(decoration_layer[wire] != "" for wire in range(min_wire, max_wire + 1)):
                        representation_grid.insert_layer(i + j, decoration_layer)
                        inserted_indices.append(i + j)
                        j += 1

                        decoration_layer = [""] * grid.num_wires

                    self.add_multi_wire_connectors_to_layer(wire_indices, decoration_layer)

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

    def move_multi_wire_gates(self, operator_grid):
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

                # translate wires to their indices on the device
                wire_indices = self.active_wires.indices(op.wires)

                if len(op.wires) > 1:

                    sorted_wires = wire_indices.copy()
                    sorted_wires.sort()

                    blocked_wires = list(range(sorted_wires[0], sorted_wires[-1] + 1))

                    for k in range(j + 1, len(layer_ops)):
                        other_op = layer_ops[k]

                        if other_op is None:
                            continue

                        # translate wires to their indices on the device
                        other_wire_indices = self.active_wires.indices(other_op.wires)
                        other_sorted_wire_indices = other_wire_indices.copy()
                        other_sorted_wire_indices.sort()
                        other_blocked_wires = list(
                            range(other_sorted_wire_indices[0], other_sorted_wire_indices[-1] + 1)
                        )

                        if not set(other_blocked_wires).isdisjoint(set(blocked_wires)):
                            op_indices = [
                                idx for idx, layer_op in enumerate(this_layer) if layer_op == op
                            ]

                            for l in op_indices:
                                other_layer[l] = op
                                this_layer[l] = None

                            break

            if not all(item is None for item in other_layer):
                operator_grid.insert_layer(i + 1, other_layer)
                n += 1

    def draw(self):
        """Draw the circuit diagram.

        Returns:
            str: The circuit diagram
        """
        rendered_string = ""

        # extract the wire labels as strings and get their maximum length
        wire_names = []
        padding = 0
        for i in range(self.full_representation_grid.num_wires):
            wire_name = str(self.active_wires.labels[i])
            padding = max(padding, len(wire_name))
            wire_names.append(wire_name)

        for i in range(self.full_representation_grid.num_wires):
            # format wire name nicely
            wire = self.full_representation_grid.wire(i)
            s = " {:>" + str(padding) + "}: {}"

            rendered_string += s.format(wire_names[i], 2 * self.charset.WIRE)
            for s in wire:
                rendered_string += s

            rendered_string += "\n"

        for symbol, cache, offset in [
            (
                "U",
                self.representation_resolver.unitary_matrix_cache,
                self.representation_resolver.label_offsets["unitary"],
            ),
            (
                "H",
                self.representation_resolver.hermitian_matrix_cache,
                self.representation_resolver.label_offsets["hermitian"],
            ),
            (
                "M",
                self.representation_resolver.matrix_cache,
                self.representation_resolver.label_offsets["matrix"],
            ),
            (
                "T",
                [
                    draw_fn(self.representation_resolver)
                    for draw_fn in self.representation_resolver.tape_cache.values()
                ],
                self.representation_resolver.label_offsets["tape"],
            ),
        ]:
            for idx, matrix in enumerate(cache):
                rendered_string += f"{symbol}{idx + offset} =\n{matrix}\n"

        # Restrict CLI print width to max_length
        if self.max_length is not None:
            wires = rendered_string.split("\n")
            n_wraps = (len(wires[0]) // self.max_length) + 1
            rendered_substrings = []
            for i in range(n_wraps):
                for wire in wires:
                    # Print body of circuit. We compute some index and whitespace offsets to ensure the rendered wire matches existing unit tests.
                    if (self.max_length * (i + 1)) < len(wire):
                        offset_white_space = (
                            int((i != 0) or (self.max_length * (i + 1) >= len(wire))) * " "
                        )
                        offset_index = int(i != 0)
                        rendered_substrings.append(
                            f"{offset_white_space}"
                            + f"{wire[(i * self.max_length) + offset_index: ((i + 1) * self.max_length) + 1]}"
                        )
                    # Last wrap, print the tail of the circuit. We trim the last character of whitespace at the end.
                    else:
                        rendered_substrings.append(f" {wire[i * self.max_length + 1:]}"[:-1])
            rendered_string = "\n".join(rendered_substrings)

        return rendered_string
