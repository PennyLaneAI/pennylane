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
import abc
import math
from collections import OrderedDict

import numpy as np

import pennylane as qml

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
        raw_grid (list, optional): Raw grid from which the Grid instance is built. Defaults to [].
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
            other_grid (pennylane.circuit_drawer.Grid): Grid whos layers will be appended
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


class CharSet(abc.ABC):
    """Charset base class."""

    # pylint: disable=too-few-public-methods

    WIRE = None
    MEASUREMENT = None
    TOP_MULTI_LINE_GATE_CONNECTOR = None
    MIDDLE_MULTI_LINE_GATE_CONNECTOR = None
    BOTTOM_MULTI_LINE_GATE_CONNECTOR = None
    EMPTY_MULTI_LINE_GATE_CONNECTOR = None
    CONTROL = None
    LANGLE = None
    RANGLE = None
    VERTICAL_LINE = None
    CROSSED_LINES = None
    PIPE = None
    OTIMES = None

    @staticmethod
    def to_superscript(num):
        """Convert the given number to a superscripted string."""
        pass


class UnicodeCharSet(CharSet):
    """Charset for CircuitDrawing made of Unicode characters."""

    # pylint: disable=too-few-public-methods

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
    PIPE = "|"
    OTIMES = "⊗"

    @staticmethod
    def to_superscript(num):
        """Convert the given number to a superscripted string."""
        ret = str(num)
        for old, new in {
            "0": "⁰",
            "1": "¹",
            "2": "²",
            "3": "³",
            "4": "⁴",
            "5": "⁵",
            "6": "⁶",
            "7": "⁷",
            "8": "⁸",
            "9": "⁹",
        }.items():
            ret = ret.replace(old, new)

        return ret

    @staticmethod
    def to_subscript(num):
        """Convert the given number to a subscripted string."""
        ret = str(num)
        for old, new in {
            "0": "₀",
            "1": "₁",
            "2": "₂",
            "3": "₃",
            "4": "₅",
            "5": "⁵",
            "6": "₆",
            "7": "₇",
            "8": "₈",
            "9": "₉",
        }.items():
            ret = ret.replace(old, new)

        return ret


class AsciiCharSet(CharSet):
    """Charset for CircuitDrawing made of Unicode characters."""

    # pylint: disable=too-few-public-methods

    WIRE = "-"
    MEASUREMENT = "|"
    TOP_MULTI_LINE_GATE_CONNECTOR = "+"
    MIDDLE_MULTI_LINE_GATE_CONNECTOR = "+"
    BOTTOM_MULTI_LINE_GATE_CONNECTOR = "+"
    EMPTY_MULTI_LINE_GATE_CONNECTOR = "|"
    CONTROL = "C"
    LANGLE = "<"
    RANGLE = ">"
    VERTICAL_LINE = "|"
    CROSSED_LINES = "X"
    PIPE = "|"
    OTIMES = "@"

    @staticmethod
    def to_superscript(num):
        """Convert the given number to a superscripted string."""
        return "^" + str(num)

    @staticmethod
    def to_subscript(num):
        """Convert the given number to a subscripted string."""
        return "_" + str(num)


Charsets = {
    "unicode": UnicodeCharSet,
    "ascii": AsciiCharSet,
}
"""Dictionary with all available CharSets."""


class RepresentationResolver:
    """Resolves the string representation of PennyLane objects.

    Args:
        charset (CharSet, optional): The CharSet to be used for representation resolution. Defaults to UnicodeCharSet.
        show_variable_names (bool, optional): Show variable names instead of variable values. Defaults to False.
    """

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
        "PhaseShift": "P",
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
        "X": "x",
        "P": "p",
    }
    """Symbol used for uncontrolled wires."""

    control_wire_dict = {
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
    """Indices of control wires."""

    def __init__(self, charset=UnicodeCharSet, show_variable_names=False):
        self.charset = charset
        self.show_variable_names = show_variable_names
        self.matrix_cache = []
        self.unitary_matrix_cache = []
        self.hermitian_matrix_cache = []

    @staticmethod
    def index_of_array_or_append(target_element, target_list):
        """Returns the first index of an appearance of the target element in the target list.
        If the target element is not in the list it will be added to the list.

        Args:
            target_element (np.ndarray): The object whos index is to be returned
            target_list (list[np.ndarray]): The list which shall be searched

        Returns:
            int: Index of the target element in the list.
        """
        for idx, target in enumerate(target_list):
            if np.array_equal(target, target_element):
                return idx

        target_list.append(target_element)

        return len(target_list) - 1

    def single_parameter_representation(self, par):
        """Resolve the representation of an Operator's parameter.

        Args:
            par (Union[qml.variable.Variable, int, float]): The parameter to be rendered

        Returns:
            str: String representation of the parameter
        """
        if isinstance(par, qml.variable.Variable):
            return par.render(self.show_variable_names)

        return str(round(par, 3))

    @staticmethod
    def _format_poly_term(coefficient, variable):
        """Format a term of a polynomial."""
        if coefficient == 0:
            return ""

        return " {} {} {}".format(
            "+" if coefficient > 0 else "-", round(math.fabs(coefficient), 3), variable
        )

    def operator_representation(self, op, wire):
        """Resolve the representation of an Operator.

        Args:
            op (pennylane.operation.Operator): The Operator instance whos representation shall be resolved
            wire (int): The Operator's wire which is currently resolved

        Returns:
            str: String representation of the Operator
        """
        if isinstance(op, qml.operation.Tensor):
            constituent_representations = [self.operator_representation(tensor_obs, wire) for tensor_obs in op.obs]

            return (" " + self.charset.OTIMES + " ").join(constituent_representations)

        name = op.name

        if name in RepresentationResolver.resolution_dict:
            name = RepresentationResolver.resolution_dict[name]

        if op.name in self.control_wire_dict and wire in [
            op.wires[control_idx] for control_idx in self.control_wire_dict[op.name]
        ]:
            return self.charset.CONTROL

        if op.num_params == 0:
            return name

        if op.name == "QubitUnitary":
            mat = op.params[0]
            idx = RepresentationResolver.index_of_array_or_append(mat, self.unitary_matrix_cache)

            return "U{}".format(idx)

        if op.name == "Hermitian":
            mat = op.params[0]
            idx = RepresentationResolver.index_of_array_or_append(mat, self.hermitian_matrix_cache)

            return "H{}".format(idx)

        if op.name == "QuadOperator":
            par_rep = self.single_parameter_representation(op.params[0])

            return "cos({0})x + sin({0})p".format(par_rep)

        if op.name == "FockStateProjector":
            n_str = ", ".join([str(n) for n in op.params[0]])

            return (
                self.charset.PIPE + n_str + self.charset.CROSSED_LINES + n_str + self.charset.PIPE
            )

        if op.name == "PolyXP":
            coefficients = op.params[0]
            order = len(coefficients.shape)

            if order == 1:
                poly_str = ""
                if coefficients[0] != 0:
                    poly_str += str(coefficients[0])

                for idx in range(0, coefficients.shape[0] // 2):
                    x = 2 * idx + 1
                    y = 2 * idx + 2
                    poly_str += RepresentationResolver._format_poly_term(
                        coefficients[x], "x{}".format(self.charset.to_subscript(idx))
                    )
                    poly_str += RepresentationResolver._format_poly_term(
                        coefficients[y], "p{}".format(self.charset.to_subscript(idx))
                    )

                return poly_str

            if order == 2:
                poly_str = str(coefficients[0, 0])

                for idx in range(0, coefficients.shape[0] // 2):
                    x = 2 * idx + 1
                    p = 2 * idx + 2
                    poly_str += RepresentationResolver._format_poly_term(
                        coefficients[0, x] + coefficients[x, 0],
                        "x{}".format(self.charset.to_subscript(idx)),
                    )
                    poly_str += RepresentationResolver._format_poly_term(
                        coefficients[0, p] + coefficients[p, 0],
                        "p{}".format(self.charset.to_subscript(idx)),
                    )

                for idx1 in range(0, coefficients.shape[0] // 2):
                    for idx2 in range(idx1, coefficients.shape[0] // 2):
                        x1 = 2 * idx1 + 1
                        p1 = 2 * idx1 + 2
                        x2 = 2 * idx2 + 1
                        p2 = 2 * idx2 + 2

                        if idx1 == idx2:
                            poly_str += RepresentationResolver._format_poly_term(
                                coefficients[x1, x1],
                                "x{}{}".format(
                                    self.charset.to_subscript(idx1), self.charset.to_superscript(2)
                                ),
                            )
                            poly_str += RepresentationResolver._format_poly_term(
                                coefficients[p1, p1],
                                "p{}{}".format(
                                    self.charset.to_subscript(idx1), self.charset.to_superscript(2)
                                ),
                            )
                            poly_str += RepresentationResolver._format_poly_term(
                                coefficients[x1, p1] + coefficients[p1, x1],
                                "x{}p{}".format(
                                    self.charset.to_subscript(idx1), self.charset.to_subscript(idx1)
                                ),
                            )
                        else:
                            poly_str += RepresentationResolver._format_poly_term(
                                coefficients[x1, x2] + coefficients[x2, x1],
                                "x{}x{}".format(
                                    self.charset.to_subscript(idx1), self.charset.to_subscript(idx2)
                                ),
                            )

                            poly_str += RepresentationResolver._format_poly_term(
                                coefficients[p1, p2] + coefficients[p2, p1],
                                "p{}p{}".format(
                                    self.charset.to_subscript(idx1), self.charset.to_subscript(idx2)
                                ),
                            )

                            poly_str += RepresentationResolver._format_poly_term(
                                coefficients[x1, p2] + coefficients[p2, x1],
                                "x{}p{}".format(
                                    self.charset.to_subscript(idx1), self.charset.to_subscript(idx2)
                                ),
                            )

                            poly_str += RepresentationResolver._format_poly_term(
                                coefficients[p1, x2] + coefficients[x2, p1],
                                "x{}p{}".format(
                                    self.charset.to_subscript(idx2), self.charset.to_subscript(idx1)
                                ),
                            )

                return poly_str

        if op.name == "FockState":
            return self.charset.PIPE + str(op.params[0]) + self.charset.RANGLE

        if op.name in {"BasisState", "FockStateVector"}:
            return self.charset.PIPE + str(op.params[0][op.wires.index(wire)]) + self.charset.RANGLE

        # Operations that only have matrix arguments
        if op.name in {
            "GaussianState",
            "FockDensityMatrix",
            "FockStateVector",
            "QubitStateVector",
            "Interferometer",
        }:
            param_strings = []
            for param in op.params:
                idx = RepresentationResolver.index_of_array_or_append(param, self.matrix_cache)

                param_strings.append("M{}".format(idx))

            return "{}({})".format(name, ", ".join(param_strings))

        return "{}({})".format(
            name, ", ".join([self.single_parameter_representation(par) for par in op.params])
        )

    def output_representation(self, obs, wire):
        """Resolve the representation of a circuit's output.

        Args:
            obs (pennylane.ops.Observable): The Observable instance whos representation shall be resolved
            wire (int): The Observable's wire which is currently resolved

        Returns:
            str: String representation of the Observable
        """
        # pylint: disable=inconsistent-return-statements
        if obs.return_type == qml.operation.Expectation:
            return (
                self.charset.LANGLE
                + "{}".format(self.operator_representation(obs, wire))
                + self.charset.RANGLE
            )

        if obs.return_type == qml.operation.Variance:
            return "Var[{}]".format(self.operator_representation(obs, wire))

        if obs.return_type == qml.operation.Sample:
            return "Sample[{}]".format(self.operator_representation(obs, wire))

        # Unknown return_type
        return "{}[{}]".format(str(obs.return_type), self.operator_representation(obs, wire))

    def element_representation(self, element, wire):
        """Resolve the representation of an element in the circuit's Grid.

        Args:
            element (Union[NoneType,str,qml.operation.Operator]): The circuit element whos representation shall be resolved
            wire (int): The element's wire which is currently resolved

        Returns:
            str: String representation of the element
        """
        if element is None:
            return ""
        if isinstance(element, str):
            return element
        if isinstance(element, qml.operation.Observable) and element.return_type is not None:
            return self.output_representation(element, wire)

        return self.operator_representation(element, wire)


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
        raw_operation_grid (list[list[qml.operation.Operation]]): The CircuitGraph's operations
        raw_observable_grid (list[list[qml.operation.Observable]]): The CircuitGraph's observables
        charset (pennylane.circuit_drawer.CharSet, optional): The CharSet that shall be used for drawing. Defaults to UnicodeCharSet.
        show_variable_names (bool, optional): Show variable names instead of variable values. Defaults to False.
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
        representation_grid,
        pad_str,
        prepend_str,
        suffix_str,
        skip_indices,
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

                    if not blocked_wires:
                        continue

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

        for symbol, cache in {
            "U": self.representation_resolver.unitary_matrix_cache,
            "H": self.representation_resolver.hermitian_matrix_cache,
            "M": self.representation_resolver.matrix_cache,
        }.items():
            for idx, matrix in enumerate(cache):
                rendered_string += "{}{} =\n{}\n".format(symbol, idx, matrix)

        return rendered_string
