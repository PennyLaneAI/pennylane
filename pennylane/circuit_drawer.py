# Copyright 2019 Xanadu Quantum Technologies Inc.

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
import numpy as np
import pennylane as qml

class Grid:
    def __init__(self, raw_grid=[]):
        self.raw_grid = raw_grid
        self.raw_grid_transpose = list(map(list, zip(*raw_grid)))

    def insert_layer(self, idx, layer):
        self.raw_grid_transpose.insert(idx, layer)
        self.raw_grid = list(map(list, zip(*self.raw_grid_transpose)))

    def append_layer(self, layer):
        self.raw_grid_transpose.append(layer)
        self.raw_grid = list(map(list, zip(*self.raw_grid_transpose)))

    def replace_layer(self, idx, layer):
        self.raw_grid_transpose[idx] = layer
        self.raw_grid = list(map(list, zip(*self.raw_grid_transpose)))

    def insert_wire(self, idx, wire):
        self.raw_grid.insert(idx, wire)
        self.raw_grid_transpose = list(map(list, zip(*self.raw_grid)))

    def append_wire(self, wire):
        self.raw_grid.append(wire)
        self.raw_grid_transpose = list(map(list, zip(*self.raw_grid)))

    @property
    def num_layers(self):
        return len(self.raw_grid_transpose)

    def layer(self, idx):
        return self.raw_grid_transpose[idx]

    @property
    def num_wires(self):
        return len(self.raw_grid)

    def wire(self, idx):
        return self.raw_grid[idx]

    def copy(self):
        return Grid(self.raw_grid.copy())

    def append_grid_by_layers(self, other_grid):
        for i in range(other_grid.num_layers):
            self.append_layer(other_grid.layer(i))

    def __str__(self):
        ret = ""
        for wire in self.raw_grid:
            ret += str(wire)
            ret += "\n"

        return ret


class UnicodeCharSet:
    WIRE = "─"
    MEASUREMENT = "┤"
    TOP_MULTI_LINE_GATE_CONNECTOR = "╭"
    MIDDLE_MULTI_LINE_GATE_CONNECTOR = "├"
    BOTTOM_MULTI_LINE_GATE_CONNECTOR = "╰"
    EMPTY_MULTI_LINE_GATE_CONNECTOR = "│"
    CONTROL = "C"
    LANGLE = "⟨"
    RANGLE = "⟩"
    VERTICAL_LINE = "│"
    CROSSED_LINES = "╳"


class RepresentationResolver:
    # Symbol for uncontrolled wires
    resolution_dict = {
        "PauliX": "X",
        "CNOT": "X",
        "Toffoli": "X",
        "CSWAP": "SWAP",
        "PauliY": "Y",
        "PauliZ": "Z",
        "CZ": "Z",
        "Identity": "I",
        "Hadamard": "H",
        "CRX": "RX",
        "CRY": "RY",
        "CRZ": "RZ",
        "CRot": "Rot",
        "Beamsplitter": "BS",
        "Squeezing": "S",
        "TwoModeSqueezing": "S",
        "Displacement": "D",
        "NumberOperator": "n",
        "Rotation": "R",
        "ControlledAddition": "Add",
        "ControlledPhase": "R",
        "ThermalState": "Thermal",
        "GaussianState": "Gaussian",
        "QuadraticPhase": "QuadPhase",
    }

    # Indices of control wires
    control_dict = {
        "CNOT": [0],
        "Toffoli": [0, 1],
        "CSWAP": [0],
        "CRX": [0],
        "CRY": [0],
        "CRZ": [0],
        "CRot": [0],
        "CZ": [0],
        "ControlledAddition": [0],
        "ControlledPhase": [0],
    }

    def __init__(self, charset=UnicodeCharSet, show_variable_names=False):
        self.charset = charset
        self.show_variable_names = show_variable_names
        self.matrix_cache = []
        self.unitary_matrix_cache = []
        self.hermitian_matrix_cache = []

    def render_parameter(self, par):
        if isinstance(par, qml.variable.Variable):
            return par.render(self.show_variable_names)

        return str(par)

    @staticmethod
    def append_array_if_not_in_list(element, target_list):
        for idx, target in enumerate(target_list):
            if np.array_equal(target, element):
                return idx

        target_list.append(element)

        return len(target_list) - 1

    def operation_representation(self, op, wire):
        name = op.name

        if name in RepresentationResolver.resolution_dict:
            name = RepresentationResolver.resolution_dict[name]

        if op.name in self.control_dict and wire in [
            op.wires[control_idx] for control_idx in self.control_dict[op.name]
        ]:
            return self.charset.CONTROL

        if op.num_params == 0:
            return name

        if op.name in ["GaussianState"]:
            param_strings = []
            for param in op.params:
                if isinstance(param, np.ndarray):
                    idx = RepresentationResolver.append_array_if_not_in_list(
                        param, self.matrix_cache
                    )

                    param_strings.append("M{}".format(idx))
                else:
                    param_strings.append(self.render_parameter(param))

            return "{}({})".format(name, ", ".join(param_strings))

        if op.name == "QubitUnitary":
            mat = op.params[0]
            idx = RepresentationResolver.append_array_if_not_in_list(mat, self.unitary_matrix_cache)

            return "U{}".format(idx)

        if op.name == "Hermitian":
            mat = op.params[0]
            idx = RepresentationResolver.append_array_if_not_in_list(
                mat, self.hermitian_matrix_cache
            )

            return "H{}".format(idx)

        if op.name == "FockStateProjector":
            n_str = ",".join([str(n) for n in op.params[0]])

            return (
                self.charset.VERTICAL_LINE
                + n_str
                + self.charset.CROSSED_LINES
                + n_str
                + self.charset.VERTICAL_LINE
            )

        return "{}({})".format(name, ", ".join([self.render_parameter(par) for par in op.params]))

    def observable_representation(self, obs, wire):
        if obs.return_type == qml.operation.Expectation:
            return (
                self.charset.LANGLE
                + "{}".format(self.operation_representation(obs, wire))
                + self.charset.RANGLE
            )
        elif obs.return_type == qml.operation.Variance:
            return "Var[{}]".format(self.operation_representation(obs, wire))
        elif obs.return_type == qml.operation.Sample:
            return "Sample[{}]".format(self.operation_representation(obs, wire))

    def operator_representation(self, op, wire):
        if op is None:
            return ""
        elif isinstance(op, str):
            return op
        elif isinstance(op, qml.operation.Observable) and op.return_type is not None:
            return self.observable_representation(op, wire)
        else:
            return self.operation_representation(op, wire)


class CircuitDrawer:
    def resolve_representation(self, grid, representation_grid):
        for i in range(grid.num_layers):
            representation_layer = [""] * grid.num_wires

            for wire, operator in enumerate(grid.layer(i)):
                representation_layer[wire] = self.representation_resolver.operator_representation(
                    operator, wire
                )

            representation_grid.append_layer(representation_layer)

    def resolve_decorations(self, grid, representation_grid, decoration_indices):
        j = 0
        for i in range(grid.num_layers):
            layer_operators = set(grid.layer(i))

            decoration_layer = [""] * grid.num_wires

            for op in layer_operators:
                if op is None:
                    continue

                if len(op.wires) > 1:
                    sorted_wires = op.wires.copy()
                    sorted_wires.sort()

                    decoration_layer[sorted_wires[0]] = self.charset.TOP_MULTI_LINE_GATE_CONNECTOR
                    decoration_layer[
                        sorted_wires[-1]
                    ] = self.charset.BOTTOM_MULTI_LINE_GATE_CONNECTOR
                    for k in range(sorted_wires[0] + 1, sorted_wires[-1]):
                        if k in sorted_wires:
                            decoration_layer[k] = self.charset.MIDDLE_MULTI_LINE_GATE_CONNECTOR
                        else:
                            decoration_layer[k] = self.charset.EMPTY_MULTI_LINE_GATE_CONNECTOR

            representation_grid.insert_layer(i + j, decoration_layer)
            decoration_indices.append(i + j)
            j += 1

    def resolve_decorations_separate(self, grid, representation_grid, decoration_indices):
        j = 0
        for i in range(grid.num_layers):
            layer_operators = set(grid.layer(i))

            for op in layer_operators:
                if op is None:
                    continue

                if len(op.wires) > 1:
                    decoration_layer = [""] * grid.num_wires
                    sorted_wires = op.wires.copy()
                    sorted_wires.sort()

                    decoration_layer[sorted_wires[0]] = self.charset.TOP_MULTI_LINE_GATE_CONNECTOR
                    decoration_layer[
                        sorted_wires[-1]
                    ] = self.charset.BOTTOM_MULTI_LINE_GATE_CONNECTOR
                    for k in range(sorted_wires[0] + 1, sorted_wires[-1]):
                        if k in sorted_wires:
                            decoration_layer[k] = self.charset.MIDDLE_MULTI_LINE_GATE_CONNECTOR
                        else:
                            decoration_layer[k] = self.charset.EMPTY_MULTI_LINE_GATE_CONNECTOR

                    representation_grid.insert_layer(i + j, decoration_layer)
                    decoration_indices.append(i + j)
                    j += 1

    def justify_and_prepend(self, x, prepend_str, suffix_str, max_width, pad_str):
        return prepend_str + str.ljust(x, max_width, pad_str) + suffix_str

    def resolve_padding(
        self,
        representation_grid,
        pad_str,
        skip_prepend_pad_str,
        prepend_str,
        suffix_str,
        skip_prepend_idx,
    ):
        for i in range(representation_grid.num_layers):
            layer = representation_grid.layer(i)
            max_width = max(map(len, layer))

            if i in skip_prepend_idx:
                representation_grid.replace_layer(
                    i,
                    list(
                        map(
                            lambda x: self.justify_and_prepend(
                                x, "", "", max_width, skip_prepend_pad_str
                            ),
                            layer,
                        )
                    ),
                )
            else:
                representation_grid.replace_layer(
                    i,
                    list(
                        map(
                            lambda x: self.justify_and_prepend(
                                x, prepend_str, suffix_str, max_width, pad_str
                            ),
                            layer,
                        )
                    ),
                )

    def move_multi_wire_gates(self, operator_grid):  # Move intertwined multi-wire gates
        n = operator_grid.num_layers
        i = -1
        while i < n - 1:
            i += 1

            this_layer = operator_grid.layer(i)
            layer_ops = list(set(this_layer))
            other_layer = [None] * operator_grid.num_wires

            for j in range(len(layer_ops)):
                op = layer_ops[j]

                if op is None:
                    continue

                if len(op.wires) > 1:
                    sorted_wires = op.wires.copy()
                    sorted_wires.sort()

                    blocked_wires = list(range(sorted_wires[0] + 1, sorted_wires[-1]))

                    if not blocked_wires:
                        continue

                    for k in range(j + 1, len(layer_ops)):
                        other_op = layer_ops[k]

                        if other_op is None:
                            continue

                        other_sorted_wires = other_op.wires.copy()
                        other_sorted_wires.sort()
                        other_blocked_wires = list(
                            range(other_sorted_wires[0] + 1, other_sorted_wires[-1])
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
        raw_operator_grid,
        raw_observable_grid,
        charset=UnicodeCharSet,
        show_variable_names=False,
    ):
        self.charset = charset
        self.show_variable_names = show_variable_names
        self.representation_resolver = RepresentationResolver(charset, show_variable_names)
        self.operator_grid = Grid(raw_operator_grid)
        self.observable_grid = Grid(raw_observable_grid)
        self.operator_representation_grid = Grid()
        self.observable_representation_grid = Grid()
        self.operator_decoration_indices = []
        self.observable_decoration_indices = []

        self.move_multi_wire_gates(self.operator_grid)

        # Resolve operator names
        self.resolve_representation(self.operator_grid, self.operator_representation_grid)
        self.resolve_representation(self.observable_grid, self.observable_representation_grid)

        # Add multi-wire gate lines
        self.resolve_decorations(
            self.operator_grid, self.operator_representation_grid, self.operator_decoration_indices
        )
        self.resolve_decorations_separate(
            self.observable_grid,
            self.observable_representation_grid,
            self.observable_decoration_indices,
        )

        self.resolve_padding(
            self.operator_representation_grid,
            charset.WIRE,
            charset.WIRE,
            "",
            2 * charset.WIRE,
            self.operator_decoration_indices,
        )
        self.resolve_padding(
            self.observable_representation_grid,
            " ",
            charset.WIRE,
            charset.MEASUREMENT + " ",
            " ",
            self.observable_decoration_indices,
        )

        self.full_representation_grid = self.operator_representation_grid.copy()
        self.full_representation_grid.append_grid_by_layers(self.observable_representation_grid)

    def draw(self):
        rendered_string = ""

        for i in range(self.full_representation_grid.num_wires):
            wire = self.full_representation_grid.wire(i)

            rendered_string += "{:2d}: {}".format(i, 2 * self.charset.WIRE)

            for s in wire:
                rendered_string += s

            rendered_string += "\n"

        for symbol, cache in {
            "U": self.representation_resolver.unitary_matrix_cache,
            "H": self.representation_resolver.hermitian_matrix_cache,
            "M": self.representation_resolver.matrix_cache,
        }.items():
            for idx, matrix in enumerate(cache):
                rendered_string += "{}{} =\n{}\n".format(symbol, idx, matrix)

        return rendered_string


# TODO:
# * Move to QNode, enable printing of unevaluated QNodes
# * Rename layers and greedy_layers to something more appropriate
# * Write tests
# * Add changelog entry
# * Add doc
