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
This module contains the RepresentationResolver class which is used to map
Operations to their string representations.
"""
import numpy as np
import pennylane as qml

from .charsets import UnicodeCharSet


class RepresentationResolver:
    """Resolves the string representation of PennyLane objects.

    Args:
        charset (CharSet, optional): The CharSet to be used for representation resolution.
        label_offsets (dict[string, int], optional): Offset the printed index of different symbol types in nested circuits.
    """

    def __init__(self, charset=UnicodeCharSet, label_offsets=None):
        self.charset = charset
        if not label_offsets:
            label_offsets = {"matrix": 0, "unitary": 0, "hermitian": 0, "tape": 0}

        self.label_offsets = label_offsets
        self.matrix_cache = []
        self.unitary_matrix_cache = []
        self.hermitian_matrix_cache = []
        self.tape_cache = {}

    # Symbol for uncontrolled wires
    resolution_dict = {
        "PauliX": "X",
        "CNOT": "X",
        "Toffoli": "X",
        "CSWAP": "SWAP",
        "PauliY": "Y",
        "PauliZ": "Z",
        "CY": "Y",
        "CZ": "Z",
        "Identity": "I",
        "Hadamard": "H",
        "MultiRZ": "RZ",
        "CRX": "RX",
        "CRY": "RY",
        "CRZ": "RZ",
        "CRot": "Rot",
        "PhaseShift": "Rϕ",
        "Beamsplitter": "BS",
        "Squeezing": "S",
        "TwoModeSqueezing": "S",
        "Displacement": "D",
        "NumberOperator": "n",
        "Rotation": "R",
        "ControlledAddition": "X",
        "ControlledPhase": "Z",
        "ThermalState": "Thermal",
        "GaussianState": "Gaussian",
        "QuadraticPhase": "P",
        "CubicPhase": "V",
        "X": "x",
        "P": "p",
        "Barrier": "||",
        "WireCut": "//",
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
        "CY": [0],
        "CZ": [0],
        "ControlledAddition": [0],
        "ControlledPhase": [0],
    }
    """Indices of control wires."""

    @staticmethod
    def index_of_array_or_append(target_element, target_list):
        """Returns the first index of an appearance of the target element in the target list.
        If the target element is not in the list it will be added to the list.

        Args:
            target_element (np.ndarray): The object whose index is to be returned
            target_list (list[np.ndarray]): The list which shall be searched

        Returns:
            int: Index of the target element in the list.
        """
        for idx, target in enumerate(target_list):
            if np.array_equal(target, target_element):
                return idx

        target_list.append(target_element)

        return len(target_list) - 1

    @staticmethod
    def single_parameter_representation(par):
        """Resolve the representation of an Operator's parameter.

        Args:
            par (Union[int, float, str]): The parameter to be rendered

        Returns:
            str: String representation of the parameter
        """
        if isinstance(par, str):
            return par

        return f"{1.0 * par:.3g}"

    @staticmethod
    def _format_matrix_operation(operation, symbol, cache, offset):
        """Format an operation that corresponds to a single matrix.

        Args:
            operation (~.Operation): Operation that shall be formatted
            symbol (str): The symbol that should be used to identify matrices
            cache (List[numpy.ndarray]): The cache of already known matrices
            offset (int): Offset the printed index to the symbol for nested circuits.

        Returns:
            str: The formatted operation
        """
        mat = operation.data[0]
        idx = RepresentationResolver.index_of_array_or_append(mat, cache) + offset

        return f"{symbol}{idx}"

    @staticmethod
    def _format_controlled_qubit_unitary(operation, symbol, cache, offset):
        """Format an operation that corresponds to a single matrix with controls.

        Args:
            operation (~.Operation): Operation that shall be formatted
            symbol (str): The symbol that should be used to identify matrices
            cache (List[numpy.ndarray]): The cache of already known matrices
            offset (int): Offset the printed index to the symbol for nested circuits.

        Returns:
            str: The formatted operation
        """
        mat = operation.parameters[0]
        idx = RepresentationResolver.index_of_array_or_append(mat, cache) + offset

        return f"{symbol}{idx}"

    @staticmethod
    def _format_matrix_arguments(params, symbol, cache, offset):
        """Format a sequence of matrix parameters.

        Args:
            params (List[numpy.ndarray]): List of matrix parameters
            symbol (str): The symbol that should be used to identify matrices
            cache (List[numpy.ndarray]): The cache of already known matrices
            offset (int): Offset the printed index to the symbol for nested circuits.

        Returns:
            str: The formatted matrix arguments
        """
        param_strings = []
        for param in params:
            idx = RepresentationResolver.index_of_array_or_append(param, cache) + offset

            param_strings.append(f"{symbol}{idx}")

        return "(" + ",".join(param_strings) + ")"

    @staticmethod
    def _format_poly_term(coefficient, variable):
        """Format a term in a polynomial.

        Args:
            coefficient (float): The polynomial term's coefficient
            variable (str): The polynomial term's variable

        Returns:
            str: The formatted term
        """
        if coefficient == 0:
            return ""

        if coefficient == 1.0:
            return str(variable)

        if coefficient == -1.0:
            return "-" + str(variable)

        return f"{coefficient:+.3g}{variable}"

    def _format_polyxp_order1(self, coefficients):
        """Format a first-order polynomial of x and p operators.

        Args:
            coefficients (array[float]): The polynomial coefficients as a vector

        Returns:
            str: A string representing the polynomial
        """
        poly_str = ""

        if coefficients[0] != 0:
            poly_str += f"{coefficients[0]:.3g}"

        for idx in range(0, coefficients.shape[0] // 2):
            x = 2 * idx + 1
            y = 2 * idx + 2
            poly_str += RepresentationResolver._format_poly_term(
                coefficients[x], f"x{self.charset.to_subscript(idx)}"
            )
            poly_str += RepresentationResolver._format_poly_term(
                coefficients[y], f"p{self.charset.to_subscript(idx)}"
            )

        return poly_str

    def _format_polyxp_order2(self, coefficients):
        """Format a second-order polynomial of x and p operators.

        Args:
            coefficients (array[float]): The polynomial coefficients as a matrix

        Returns:
            str: A string representing the polynomial
        """
        poly_str = str(coefficients[0, 0])

        for idx in range(0, coefficients.shape[0] // 2):
            x = 2 * idx + 1
            p = 2 * idx + 2
            poly_str += RepresentationResolver._format_poly_term(
                coefficients[0, x] + coefficients[x, 0],
                f"x{self.charset.to_subscript(idx)}",
            )
            poly_str += RepresentationResolver._format_poly_term(
                coefficients[0, p] + coefficients[p, 0],
                f"p{self.charset.to_subscript(idx)}",
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
                        f"x{self.charset.to_subscript(idx1)}{self.charset.to_superscript(2)}",
                    )
                    poly_str += RepresentationResolver._format_poly_term(
                        coefficients[p1, p1],
                        f"p{self.charset.to_subscript(idx1)}{self.charset.to_superscript(2)}",
                    )
                    poly_str += RepresentationResolver._format_poly_term(
                        coefficients[x1, p1] + coefficients[p1, x1],
                        f"x{self.charset.to_subscript(idx1)}p{self.charset.to_subscript(idx1)}",
                    )
                else:
                    poly_str += RepresentationResolver._format_poly_term(
                        coefficients[x1, x2] + coefficients[x2, x1],
                        f"x{self.charset.to_subscript(idx1)}x{self.charset.to_subscript(idx2)}",
                    )

                    poly_str += RepresentationResolver._format_poly_term(
                        coefficients[p1, p2] + coefficients[p2, p1],
                        f"p{self.charset.to_subscript(idx1)}p{self.charset.to_subscript(idx2)}",
                    )

                    poly_str += RepresentationResolver._format_poly_term(
                        coefficients[x1, p2] + coefficients[p2, x1],
                        f"x{self.charset.to_subscript(idx1)}p{self.charset.to_subscript(idx2)}",
                    )

                    poly_str += RepresentationResolver._format_poly_term(
                        coefficients[p1, x2] + coefficients[x2, p1],
                        f"x{self.charset.to_subscript(idx2)}p{self.charset.to_subscript(idx1)}",
                    )

        return poly_str

    def _format_polyxp(self, operation):
        """Format a polynomial of x and p operators.

        Theses operators appear as observables in CV quantum computing.

        Args:
            operation (~.PolyXP): The PolyXP observable that shall be formatted.

        Returns:
            str: A string representing the polynomial
        """
        coefficients = operation.data[0]
        order = len(coefficients.shape)

        if order == 1:
            return self._format_polyxp_order1(coefficients)

        return self._format_polyxp_order2(coefficients)

    # pylint: disable=too-many-branches, too-many-return-statements
    def operator_representation(self, op, wire):
        """Return the string representation of an Operator.

        Args:
            op (pennylane.operation.Operator): The Operator instance whose representation shall be returned
            wire (Wires): The Operator's wire for which the string representation shall be returned

        Returns:
            str: String representation of the Operator
        """
        if isinstance(op, qml.measure.MeasurementProcess):
            if op.obs is not None:
                op = op.obs
            else:
                return "basis"  # when no observable is provided we perform a raw measurement

        if isinstance(op, qml.operation.Tensor):
            constituent_representations = [
                self.operator_representation(tensor_obs, wire) for tensor_obs in op.obs
            ]

            return (" " + self.charset.OTIMES + " ").join(constituent_representations)

        if isinstance(op, qml.tape.QuantumTape):
            return self.draw_tape(op)

        representation = ""
        base_name = getattr(op, "base_name", op.name)
        name = base_name

        # Use a shorter name if applicable
        if name in RepresentationResolver.resolution_dict:
            name = RepresentationResolver.resolution_dict[name]

        # Display a control symbol for all controlling qubits of a controlled Operation
        if base_name in self.control_wire_dict and wire in [
            op.wires[control_idx] for control_idx in self.control_wire_dict[base_name]
        ]:
            # No need to add a -1 for inverse here
            return self.charset.CONTROL

        if base_name == "MultiControlledX":
            if wire in op.control_wires:
                return self.charset.CONTROL
            representation = "X"

        elif op.num_params == 0:
            representation = name

        elif base_name == "PauliRot":
            rot_data = op.data[1][op.wires.index(wire)]
            representation = f"R{rot_data}({self.single_parameter_representation(op.data[0])})"

        elif base_name == "QubitUnitary":
            representation = RepresentationResolver._format_matrix_operation(
                op, "U", self.unitary_matrix_cache, self.label_offsets["unitary"]
            )

        elif base_name == "ControlledQubitUnitary":
            if wire in op.control_wires:
                return self.charset.CONTROL

            representation = RepresentationResolver._format_controlled_qubit_unitary(
                op, "U", self.unitary_matrix_cache, self.label_offsets["unitary"]
            )

        elif base_name == "Hermitian":
            representation = RepresentationResolver._format_matrix_operation(
                op, "H", self.hermitian_matrix_cache, self.label_offsets["hermitian"]
            )

        elif base_name == "QuadOperator":
            par_rep = self.single_parameter_representation(op.data[0])

            representation = f"cos({par_rep})x+sin({par_rep})p"

        elif base_name == "FockStateProjector":
            n_str = ",".join([str(n) for n in op.data[0]])

            representation = (
                self.charset.PIPE + n_str + self.charset.CROSSED_LINES + n_str + self.charset.PIPE
            )

        elif base_name == "PolyXP":
            representation = self._format_polyxp(op)

        elif base_name == "FockState":
            representation = self.charset.PIPE + str(op.data[0]) + self.charset.RANGLE

        elif base_name in {"BasisState", "FockStateVector"}:
            representation = (
                self.charset.PIPE + str(op.data[0][op.wires.index(wire)]) + self.charset.RANGLE
            )

        # Operations that only have matrix arguments
        elif len(qml.math.shape(op.data[0])) != 0:
            representation = name + RepresentationResolver._format_matrix_arguments(
                op.data, "M", self.matrix_cache, self.label_offsets["matrix"]
            )

        else:
            r = ", ".join([self.single_parameter_representation(par) for par in op.data])
            representation = f"{name}({r})"

        if getattr(op, "inverse", False):
            representation += self.charset.to_superscript("-1")

        return representation

    def draw_tape(self, op):
        """Draws quantum tapes and nested quantum tapes.

        Args:
            op (~.QuantumTape): the tape to draw

        Returns:
            str: the tape string to display
        """

        if op in self.tape_cache:
            idx = list(self.tape_cache).index(op) + self.label_offsets["tape"]
        else:

            from .circuit_drawer import CircuitDrawer  # pylint: disable=import-outside-toplevel

            grid, obs = op.graph.greedy_layers()

            def draw_nested_tape_callback(resolver):
                offsets = {
                    "matrix": resolver.label_offsets["matrix"] + len(resolver.matrix_cache),
                    "unitary": resolver.label_offsets["unitary"]
                    + len(resolver.unitary_matrix_cache),
                    "hermitian": resolver.label_offsets["hermitian"]
                    + len(resolver.hermitian_matrix_cache),
                    "tape": resolver.label_offsets["tape"] + len(resolver.tape_cache),
                }
                return CircuitDrawer(
                    grid, obs, op.graph.wires, charset=self.charset, _label_offsets=offsets
                ).draw()

            idx = len(self.tape_cache) + self.label_offsets["tape"]
            self.tape_cache[op] = draw_nested_tape_callback

        return f"QuantumTape:T{idx}"

    def output_representation(self, obs, wire):
        """Return the string representation of a circuit's output.

        Args:
            obs (pennylane.ops.Observable): The Observable instance whose representation shall be returned
            wire (Wires): The Observable's wire for which the string representation shall be returned

        Returns:
            str: String representation of the Observable
        """
        # pylint: disable=inconsistent-return-statements
        if obs.return_type == qml.operation.Expectation:
            return (
                self.charset.LANGLE
                + f"{self.operator_representation(obs, wire)}"
                + self.charset.RANGLE
            )

        if obs.return_type == qml.operation.Variance:
            return f"Var[{self.operator_representation(obs, wire)}]"

        if obs.return_type == qml.operation.Sample:
            return f"Sample[{self.operator_representation(obs, wire)}]"

        if obs.return_type == qml.operation.Probability:
            return "Probs"

        if obs.return_type == qml.operation.State:
            return "State"

        # Unknown return_type
        return f"{str(obs.return_type)}[{self.operator_representation(obs, wire)}]"

    def element_representation(self, element, wire):
        """Return the string representation of an element in the circuit's Grid.

        Args:
            element (Union[NoneType,str,qml.operation.Operator]): The circuit element whose representation shall be returned
            wire (Wires): The element's wire for which the string representation shall be returned

        Returns:
            str: String representation of the element
        """
        if element is None:
            return ""
        if isinstance(element, str):
            return element
        if (
            isinstance(element, (qml.operation.Observable, qml.measure.MeasurementProcess))
            and element.return_type is not None
        ):
            return self.output_representation(element, wire)

        return self.operator_representation(element, wire)
