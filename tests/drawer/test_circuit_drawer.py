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
Unit tests for the :mod:`pennylane.drawer` module.
"""
from unittest.mock import Mock
import pytest
import numpy as np

import pennylane as qml
from pennylane.drawer import CircuitDrawer
from pennylane.drawer.circuit_drawer import _remove_duplicates
from pennylane.drawer.grid import Grid, _transpose
from pennylane.drawer.charsets import CHARSETS, UnicodeCharSet, AsciiCharSet
from pennylane.wires import Wires

from pennylane.measure import state


class TestFunctions:
    """Test the helper functions."""

    @pytest.mark.parametrize(
        "input,expected_output",
        [
            ([1, 1, 1, 2, 2, 2, 3, 2, 3], [1, 2, 3]),
            (["a", "b", "c", "c", "a", "d"], ["a", "b", "c", "d"]),
            ([1, 2, 3, 4], [1, 2, 3, 4]),
        ],
    )
    def test_remove_duplicates(self, input, expected_output):
        """Test the function _remove_duplicates."""
        assert _remove_duplicates(input) == expected_output


op_CNOT21 = qml.CNOT(wires=[2, 1])
op_SWAP03 = qml.SWAP(wires=[0, 3])
op_SWAP12 = qml.SWAP(wires=[1, 2])
op_X0 = qml.PauliX(0)
op_CRX20 = qml.CRX(2.3, wires=[2, 0])
op_Z3 = qml.PauliZ(3)

dummy_raw_operation_grid = [
    [None, op_SWAP03, op_X0, op_CRX20],
    [op_CNOT21, op_SWAP12, None, None],
    [op_CNOT21, op_SWAP12, None, op_CRX20],
    [op_Z3, op_SWAP03, None, None],
]

dummy_raw_observable_grid = [
    [qml.sample(qml.Hermitian(2 * np.eye(2), wires=[0]))],
    [None],
    [qml.expval(qml.PauliY(wires=[2]))],
    [qml.var(qml.Hadamard(wires=[3]))],
]


class TestInitialization:
    def test_charset_default(self):

        drawer_None = CircuitDrawer(
            dummy_raw_operation_grid, dummy_raw_observable_grid, Wires(range(6)), charset=None
        )

        assert drawer_None.charset is UnicodeCharSet

    @pytest.mark.parametrize("charset", ("unicode", "ascii"))
    def test_charset_string(self, charset):

        drawer_str = CircuitDrawer(
            dummy_raw_operation_grid, dummy_raw_observable_grid, Wires(range(6)), charset=charset
        )

        assert drawer_str.charset is CHARSETS[charset]

    @pytest.mark.parametrize("charset", (UnicodeCharSet, AsciiCharSet))
    def test_charset_class(self, charset):

        drawer_class = CircuitDrawer(
            dummy_raw_operation_grid, dummy_raw_observable_grid, Wires(range(6)), charset=charset
        )

        assert drawer_class.charset is charset

    def test_charset_error(self):

        with pytest.raises(ValueError, match=r"Charset 'nope' is not supported."):
            CircuitDrawer(
                dummy_raw_operation_grid, dummy_raw_observable_grid, Wires(range(6)), charset="nope"
            )


@pytest.fixture
def dummy_circuit_drawer():
    """A dummy CircuitDrawer instance."""
    return CircuitDrawer(dummy_raw_operation_grid, dummy_raw_observable_grid, Wires(range(6)))


def assert_nested_lists_equal(list1, list2):
    """Assert that two nested lists are equal.

    Args:
        list1 (list[list[Any]]): The first list to be compared
        list2 (list[list[Any]]): The second list to be compared
    """
    # pylint: disable=protected-access
    for (obj1, obj2) in zip(qml.utils._flatten(list1), qml.utils._flatten(list2)):
        assert obj1 == obj2


def to_layer(operation_list, num_wires):
    """Convert the given list of operations to a layer.

    Args:
        operation_list (list[~.Operation]): List of Operations in the layer
        num_wires (int): Number of wires in the system

    Returns:
        list[Union[~.Operation,None]]: The corresponding layer
    """
    layer = [None] * num_wires

    for op in operation_list:
        # TODO: Currently this function only works if the device's wires are consecutive integers
        for wire in op.wires.tolist():
            layer[wire] = op

    return layer


def to_grid(layer_list, num_wires):
    """Convert the given list of operations per layer to a Grid.

    Args:
        layer_list (list[list[~.Operation]]): List of layers in terms of Operations
        num_wires (int): Number of wires in the system

    Returns:
        ~.Grid: The corresponding Operation grid
    """
    grid = Grid(_transpose([to_layer(layer_list[0], num_wires)]))

    for i in range(1, len(layer_list)):
        grid.append_layer(to_layer(layer_list[i], num_wires))

    return grid


class TestCircuitDrawer:
    """Test the CircuitDrawer class."""

    def test_resolve_representation(self, dummy_circuit_drawer):
        """Test that resolve_representation calls the representation resolver with the proper arguments."""

        dummy_circuit_drawer.representation_resolver.element_representation = Mock(
            return_value="Test"
        )

        dummy_circuit_drawer.resolve_representation(Grid(dummy_raw_operation_grid), Grid())

        args_tuples = [
            call[0]
            for call in dummy_circuit_drawer.representation_resolver.element_representation.call_args_list
        ]

        for idx, wire in enumerate(dummy_raw_operation_grid):
            for op in wire:
                assert (op, idx) in args_tuples

    interlocking_multiwire_gate_grid = to_grid(
        [[qml.CNOT(wires=[0, 4]), qml.CNOT(wires=[1, 5]), qml.Toffoli(wires=[2, 3, 6])]], 7
    )
    interlocking_multiwire_gate_representation_grid = Grid(
        [
            ["╭", "", ""],
            ["│", "╭", ""],
            ["│", "│", "╭"],
            ["│", "│", "├"],
            ["╰", "│", "│"],
            ["", "╰", "│"],
            ["", "", "╰"],
        ]
    )

    multiwire_and_single_wire_gate_grid = to_grid(
        [[qml.Toffoli(wires=[0, 3, 4]), qml.PauliX(wires=[1]), qml.Hadamard(wires=[2])]], 5
    )
    multiwire_and_single_wire_gate_representation_grid = Grid([["╭"], ["│"], ["│"], ["├"], ["╰"]])

    all_wire_state_preparation_grid = to_grid(
        [[qml.BasisState(np.array([0, 1, 0, 0, 1, 1]), wires=[0, 1, 2, 3, 4, 5])]], 6
    )
    all_wire_state_preparation_representation_grid = Grid(
        [["╭"], ["├"], ["├"], ["├"], ["├"], ["╰"]]
    )

    multiwire_gate_grid = to_grid(
        [[qml.CNOT(wires=[0, 1]), qml.PauliX(2), qml.CNOT(wires=[3, 4])]], 5
    )

    multiwire_gate_representation_grid = Grid(
        [
            ["╭"],
            ["╰"],
            [""],
            ["╭"],
            ["╰"],
        ]
    )

    multi_and_single_wire_gate_grid = to_grid(
        [
            [
                qml.CNOT(wires=[0, 1]),
                qml.PauliX(2),
                qml.PauliX(4),
                qml.CNOT(wires=[3, 5]),
                qml.Hadamard(6),
            ]
        ],
        7,
    )

    multi_and_single_wire_gate_representation_grid = Grid(
        [
            ["╭"],
            ["╰"],
            [""],
            ["╭"],
            ["│"],
            ["╰"],
            [""],
        ]
    )

    @pytest.mark.parametrize(
        "grid,target_representation_grid",
        [
            (interlocking_multiwire_gate_grid, interlocking_multiwire_gate_representation_grid),
            (
                multiwire_and_single_wire_gate_grid,
                multiwire_and_single_wire_gate_representation_grid,
            ),
            (all_wire_state_preparation_grid, all_wire_state_preparation_representation_grid),
            (multiwire_gate_grid, multiwire_gate_representation_grid),
            (multi_and_single_wire_gate_grid, multi_and_single_wire_gate_representation_grid),
        ],
    )
    def test_resolve_decorations(self, grid, target_representation_grid):
        """Test that decorations are properly resolved."""
        representation_grid = Grid()

        raw_operator_grid = grid.raw_grid
        # make a dummy observable grid
        raw_observable_grid = [[None] for _ in range(len(raw_operator_grid))]

        drawer = CircuitDrawer(raw_operator_grid, raw_observable_grid, Wires(range(10)))

        drawer.resolve_decorations(grid, representation_grid)

        assert_nested_lists_equal(representation_grid.raw_grid, target_representation_grid.raw_grid)

    CNOT04 = qml.CNOT(wires=[0, 4])
    CNOT15 = qml.CNOT(wires=[1, 5])
    Toffoli236 = qml.Toffoli(wires=[2, 3, 6])

    interlocking_CNOT_grid = to_grid([[CNOT04, CNOT15, Toffoli236]], 7)
    moved_interlocking_CNOT_grid = to_grid([[Toffoli236], [CNOT15], [CNOT04]], 7)

    SWAP02 = qml.SWAP(wires=[0, 2])
    SWAP35 = qml.SWAP(wires=[3, 5])
    SWAP14 = qml.SWAP(wires=[1, 4])
    SWAP24 = qml.SWAP(wires=[2, 4])

    interlocking_SWAP_grid = to_grid([[SWAP02, SWAP35, SWAP14], [SWAP24]], 6)
    moved_interlocking_SWAP_grid = to_grid([[SWAP35], [SWAP14], [SWAP02], [SWAP24]], 6)

    @pytest.mark.parametrize(
        "grid,target_grid",
        [
            (interlocking_CNOT_grid, moved_interlocking_CNOT_grid),
            (interlocking_SWAP_grid, moved_interlocking_SWAP_grid),
        ],
    )
    def test_move_multi_wire_gates(self, grid, target_grid):
        """Test that decorations are properly resolved."""

        operator_grid = grid.copy()

        raw_operator_grid = operator_grid.raw_grid
        # make a dummy observable grid
        raw_observable_grid = [[None] for _ in range(len(raw_operator_grid))]

        drawer = CircuitDrawer(raw_operator_grid, raw_observable_grid, Wires(range(10)))
        drawer.move_multi_wire_gates(operator_grid)

        assert_nested_lists_equal(operator_grid.raw_grid, target_grid.raw_grid)


@pytest.fixture
def parameterized_qubit_tape():
    """A parametrized qubit ciruit."""
    a, b, c = 0.1, 0.2, 0.3
    angles = np.array([0.4, 0.5, 0.6])

    with qml.tape.QuantumTape() as tape:
        qml.RX(a, wires=0)
        qml.RX(b, wires=1)
        qml.PauliZ(1)
        qml.CNOT(wires=[0, 1]).inv()
        qml.CRY(b, wires=[3, 1])
        qml.RX(angles[0], wires=0)
        qml.RX(4 * angles[1], wires=1)
        qml.PhaseShift(17 / 9 * c, wires=2)
        qml.RZ(b, wires=3)
        qml.RX(angles[2], wires=2).inv()
        qml.CRY(0.3589, wires=[3, 1]).inv()
        qml.CSWAP(wires=[4, 2, 1]).inv()
        qml.QubitUnitary(np.eye(2), wires=[2])
        qml.ControlledQubitUnitary(np.eye(2), control_wires=[0, 1], wires=[2])
        qml.MultiControlledX(control_wires=[0, 1, 2], wires=[3])
        qml.Toffoli(wires=[0, 2, 1])
        qml.CNOT(wires=[0, 2])
        qml.PauliZ(wires=[1])
        qml.PauliZ(wires=[1]).inv()
        qml.CZ(wires=[0, 1])
        qml.CZ(wires=[0, 2]).inv()
        qml.CY(wires=[1, 2])
        qml.CY(wires=[2, 0]).inv()
        qml.CNOT(wires=[2, 1])
        qml.CNOT(wires=[0, 2])
        qml.SWAP(wires=[0, 2]).inv()
        qml.CNOT(wires=[1, 3])
        qml.RZ(b, wires=3)
        qml.CSWAP(wires=[4, 0, 1])

        qml.expval(qml.PauliY(0)),
        qml.var(qml.Hadamard(wires=1)),
        qml.sample(qml.PauliX(2)),
        qml.expval(qml.Hermitian(np.eye(4), wires=[3, 4])),

    return tape


@pytest.fixture
def drawn_parameterized_qubit_circuit_with_values():
    """The rendered circuit representation of the above qubit circuit with variable values."""
    return (
        " 0: ──RX(0.1)───────────────╭C─────RX(0.4)──────────────────────────────────────╭C───╭C──╭C─────╭C───────╭C──╭C────────╭Y⁻¹──────────╭C──╭SWAP⁻¹───╭SWAP───┤ ⟨Y⟩       \n"
        + " 1: ──RX(0.2)────Z──────────╰X⁻¹──╭RY(0.2)──RX(2)────╭RY(0.359)⁻¹──╭SWAP⁻¹──────├C───├C──├X──Z──│───Z⁻¹──╰Z──│─────╭C──│─────╭X──╭C──│───│─────────├SWAP───┤ Var[H]    \n"
        + " 2: ──Rϕ(0.567)──RX(0.6)⁻¹────────│──────────────────│─────────────├SWAP⁻¹──U0──╰U0──├C──╰C─────╰X───────────╰Z⁻¹──╰Y──╰C────╰C──│───╰X──╰SWAP⁻¹───│───────┤ Sample[X] \n"
        + " 3: ──────────────────────────────╰C────────RZ(0.2)──╰C────────────│─────────────────╰X──────────────────────────────────────────╰X───────RZ(0.2)──│──────╭┤ ⟨H0⟩      \n"
        + " 4: ───────────────────────────────────────────────────────────────╰C──────────────────────────────────────────────────────────────────────────────╰C─────╰┤ ⟨H0⟩      \n"
        + "U0 =\n"
        + "[[1. 0.]\n"
        + " [0. 1.]]\n"
        + "H0 =\n"
        + "[[1. 0. 0. 0.]\n"
        + " [0. 1. 0. 0.]\n"
        + " [0. 0. 1. 0.]\n"
        + " [0. 0. 0. 1.]]\n"
    )


@pytest.fixture
def parameterized_wide_qubit_tape():
    """A wide parametrized qubit circuit."""
    a, b, c, d, e, f = 0.1, 0.2, 0.3, 47 / 17, 0.5, 0.6

    with qml.tape.QuantumTape() as tape:
        qml.RX(a, wires=0)
        qml.RX(b, wires=1)
        [qml.CNOT(wires=[2 * i, 2 * i + 1]) for i in range(4)]
        [qml.CNOT(wires=[i, i + 4]) for i in range(4)]
        [qml.PauliY(wires=[2 * i]) for i in range(4)]
        [qml.CSWAP(wires=[i + 2, i, i + 4]) for i in range(4)]
        qml.RX(a, wires=0)
        qml.RX(b, wires=1)

        [qml.expval(qml.Hermitian(np.eye(4), wires=[i, i + 4])) for i in range(4)]

    return tape


@pytest.fixture
def drawn_parameterized_wide_qubit_tape_with_variable_names():
    """The rendered circuit representation of the above wide qubit circuit with variable names."""
    return (
        " 0: ───RX(a)──╭C─────────────╭C──Y─────────────────╭SWAP───RX(a)──╭───┤ ⟨H0⟩ \n"
        + " 1: ───RX(b)──╰X─────────╭C──│──────╭SWAP───RX(b)──│──────────────│╭──┤ ⟨H0⟩ \n"
        + " 2: ──╭C──────────╭C──Y──│───│──────│──────────────├C─────╭SWAP───││╭─┤ ⟨H0⟩ \n"
        + " 3: ──╰X──────╭C──│──────│───│──────├C─────╭SWAP───│──────│───────│││╭┤ ⟨H0⟩ \n"
        + " 4: ──╭C──────│───│──────│───╰X──Y──│──────│───────╰SWAP──├C──────╰│││┤ ⟨H0⟩ \n"
        + " 5: ──╰X──────│───│──────╰X─────────╰SWAP──├C─────────────│────────╰││┤ ⟨H0⟩ \n"
        + " 6: ──╭C──────│───╰X──Y────────────────────│──────────────╰SWAP─────╰│┤ ⟨H0⟩ \n"
        + " 7: ──╰X──────╰X───────────────────────────╰SWAP─────────────────────╰┤ ⟨H0⟩ \n"
        + "H0 =\n"
        + "[[1. 0. 0. 0.]\n"
        + " [0. 1. 0. 0.]\n"
        + " [0. 0. 1. 0.]\n"
        + " [0. 0. 0. 1.]]\n"
    )


@pytest.fixture
def drawn_parameterized_wide_qubit_tape_with_values():
    """The rendered circuit representation of the above wide qubit circuit with variable values."""
    return (
        " 0: ───RX(0.1)──╭C─────────────╭C──Y───────────────────╭SWAP───RX(0.1)──╭───┤ ⟨H0⟩ \n"
        + " 1: ───RX(0.2)──╰X─────────╭C──│──────╭SWAP───RX(0.2)──│────────────────│╭──┤ ⟨H0⟩ \n"
        + " 2: ──╭C────────────╭C──Y──│───│──────│────────────────├C─────╭SWAP─────││╭─┤ ⟨H0⟩ \n"
        + " 3: ──╰X────────╭C──│──────│───│──────├C─────╭SWAP─────│──────│─────────│││╭┤ ⟨H0⟩ \n"
        + " 4: ──╭C────────│───│──────│───╰X──Y──│──────│─────────╰SWAP──├C────────╰│││┤ ⟨H0⟩ \n"
        + " 5: ──╰X────────│───│──────╰X─────────╰SWAP──├C───────────────│──────────╰││┤ ⟨H0⟩ \n"
        + " 6: ──╭C────────│───╰X──Y────────────────────│────────────────╰SWAP───────╰│┤ ⟨H0⟩ \n"
        + " 7: ──╰X────────╰X───────────────────────────╰SWAP─────────────────────────╰┤ ⟨H0⟩ \n"
        + "H0 =\n"
        + "[[1. 0. 0. 0.]\n"
        + " [0. 1. 0. 0.]\n"
        + " [0. 0. 1. 0.]\n"
        + " [0. 0. 0. 1.]]\n"
    )


@pytest.fixture
def wide_cv_tape():
    """A wide unparametrized CV circuit."""

    with qml.tape.QuantumTape() as tape:
        qml.GaussianState(
            2 * np.eye(16), np.array([(2 * i + 2) // 2 for i in range(16)]), wires=list(range(8))
        )
        [qml.Beamsplitter(0.4, 0, wires=[2 * i, 2 * i + 1]) for i in range(4)]
        [qml.Beamsplitter(0.25475, 0.2312344, wires=[i, i + 4]) for i in range(4)]

        [qml.expval(qml.FockStateProjector(np.array([1, 1]), wires=[i, i + 4])) for i in range(4)]

    return tape


@pytest.fixture
def drawn_wide_cv_tape():
    """The rendered circuit representation of the above wide CV circuit."""
    return (
        " 0: ──╭Gaussian(M0,M1)──╭BS(0.4, 0)───────────────────────────────────────────────────────────╭BS(0.255, 0.231)──╭───┤ ⟨|1,1╳1,1|⟩ \n"
        + " 1: ──├Gaussian(M0,M1)──╰BS(0.4, 0)────────────────────────────────────────╭BS(0.255, 0.231)──│──────────────────│╭──┤ ⟨|1,1╳1,1|⟩ \n"
        + " 2: ──├Gaussian(M0,M1)──╭BS(0.4, 0)─────────────────────╭BS(0.255, 0.231)──│──────────────────│──────────────────││╭─┤ ⟨|1,1╳1,1|⟩ \n"
        + " 3: ──├Gaussian(M0,M1)──╰BS(0.4, 0)──╭BS(0.255, 0.231)──│──────────────────│──────────────────│──────────────────│││╭┤ ⟨|1,1╳1,1|⟩ \n"
        + " 4: ──├Gaussian(M0,M1)──╭BS(0.4, 0)──│──────────────────│──────────────────│──────────────────╰BS(0.255, 0.231)──╰│││┤ ⟨|1,1╳1,1|⟩ \n"
        + " 5: ──├Gaussian(M0,M1)──╰BS(0.4, 0)──│──────────────────│──────────────────╰BS(0.255, 0.231)──────────────────────╰││┤ ⟨|1,1╳1,1|⟩ \n"
        + " 6: ──├Gaussian(M0,M1)──╭BS(0.4, 0)──│──────────────────╰BS(0.255, 0.231)──────────────────────────────────────────╰│┤ ⟨|1,1╳1,1|⟩ \n"
        + " 7: ──╰Gaussian(M0,M1)──╰BS(0.4, 0)──╰BS(0.255, 0.231)──────────────────────────────────────────────────────────────╰┤ ⟨|1,1╳1,1|⟩ \n"
        + "M0 =\n"
        + "[[2. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n"
        + " [0. 2. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n"
        + " [0. 0. 2. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n"
        + " [0. 0. 0. 2. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n"
        + " [0. 0. 0. 0. 2. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n"
        + " [0. 0. 0. 0. 0. 2. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n"
        + " [0. 0. 0. 0. 0. 0. 2. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n"
        + " [0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 0. 0. 0. 0. 0. 0.]\n"
        + " [0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 0. 0. 0. 0. 0.]\n"
        + " [0. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 0. 0. 0. 0.]\n"
        + " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 0. 0. 0.]\n"
        + " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 0. 0.]\n"
        + " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 0.]\n"
        + " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0.]\n"
        + " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0.]\n"
        + " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 2.]]\n"
        + "M1 =\n"
        + "[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]\n"
    )


@pytest.fixture
def parameterized_cv_tape():
    """A parametrized CV circuit."""
    a, b, c, d, e, f = 0.1, 0.2, 0.3, 47 / 17, 0.5, 0.6

    with qml.tape.QuantumTape() as tape:
        qml.ThermalState(3, wires=[1])
        qml.GaussianState(2 * np.eye(8), np.array([1, 1, 1, 2, 2, 3, 3, 3]), wires=[0, 1, 2, 3])
        qml.Rotation(a, wires=0)
        qml.Rotation(b, wires=1)
        qml.Beamsplitter(d, 1, wires=[0, 1])
        qml.Beamsplitter(e, 1, wires=[1, 2])
        qml.Displacement(f, 0, wires=[3])
        qml.Squeezing(2.3, 0, wires=[0])
        qml.Squeezing(2.3, 0, wires=[2])
        qml.Beamsplitter(d, 1, wires=[1, 2])
        qml.Beamsplitter(e, 1, wires=[2, 3])
        qml.TwoModeSqueezing(2, 2, wires=[3, 1])
        qml.ControlledPhase(2.3, wires=[2, 1])
        qml.ControlledAddition(2, wires=[0, 3])
        qml.QuadraticPhase(4, wires=[0])

        qml.expval(qml.ops.PolyXP(np.array([0, 1, 2]), wires=0))
        qml.expval(qml.ops.QuadOperator(4, wires=1))
        qml.expval(qml.ops.FockStateProjector(np.array([1, 5]), wires=[2, 3]))

    return tape


@pytest.fixture
def drawn_parameterized_cv_tape_with_variable_names():
    """The rendered circuit representation of the above CV circuit with variable names."""
    return (
        " 0: ──────────────╭Gaussian(M0,M1)──R(a)─────╭BS(d, 1)───S(2.3, 0)──────────────────────────────────────────────────────╭C─────P(4)───┤ ⟨x₀+2p₀⟩          \n"
        + " 1: ──Thermal(3)──├Gaussian(M0,M1)──R(b)─────╰BS(d, 1)──╭BS(e, 1)──────────────╭BS(d, 1)─────────────╭S(2, 2)──╭Z(2.3)──│─────────────┤ ⟨cos(4)x+sin(4)p⟩ \n"
        + " 2: ──────────────├Gaussian(M0,M1)──────────────────────╰BS(e, 1)───S(2.3, 0)──╰BS(d, 1)──╭BS(e, 1)──│─────────╰C───────│────────────╭┤ ⟨|1,5╳1,5|⟩       \n"
        + " 3: ──────────────╰Gaussian(M0,M1)──D(f, 0)───────────────────────────────────────────────╰BS(e, 1)──╰S(2, 2)───────────╰X(2)────────╰┤ ⟨|1,5╳1,5|⟩       \n"
        + "M0 =\n"
        + "[1 1 1 2 2 3 3 3]\n"
        + "M1 =\n"
        + "[[2. 0. 0. 0. 0. 0. 0. 0.]\n"
        + " [0. 2. 0. 0. 0. 0. 0. 0.]\n"
        + " [0. 0. 2. 0. 0. 0. 0. 0.]\n"
        + " [0. 0. 0. 2. 0. 0. 0. 0.]\n"
        + " [0. 0. 0. 0. 2. 0. 0. 0.]\n"
        + " [0. 0. 0. 0. 0. 2. 0. 0.]\n"
        + " [0. 0. 0. 0. 0. 0. 2. 0.]\n"
        + " [0. 0. 0. 0. 0. 0. 0. 2.]]\n"
    )


@pytest.fixture
def drawn_parameterized_cv_tape_with_values():
    """The rendered circuit representation of the above CV circuit with variable values."""
    return (
        " 0: ──────────────╭Gaussian(M0,M1)──R(0.1)─────╭BS(2.76, 1)───S(2.3, 0)────────────────────────────────────────────────────────────╭C─────P(4)───┤ ⟨x₀+2p₀⟩          \n"
        + " 1: ──Thermal(3)──├Gaussian(M0,M1)──R(0.2)─────╰BS(2.76, 1)──╭BS(0.5, 1)─────────────╭BS(2.76, 1)───────────────╭S(2, 2)──╭Z(2.3)──│─────────────┤ ⟨cos(4)x+sin(4)p⟩ \n"
        + " 2: ──────────────├Gaussian(M0,M1)───────────────────────────╰BS(0.5, 1)──S(2.3, 0)──╰BS(2.76, 1)──╭BS(0.5, 1)──│─────────╰C───────│────────────╭┤ ⟨|1,5╳1,5|⟩       \n"
        + " 3: ──────────────╰Gaussian(M0,M1)──D(0.6, 0)──────────────────────────────────────────────────────╰BS(0.5, 1)──╰S(2, 2)───────────╰X(2)────────╰┤ ⟨|1,5╳1,5|⟩       \n"
        + "M0 =\n"
        + "[[2. 0. 0. 0. 0. 0. 0. 0.]\n"
        + " [0. 2. 0. 0. 0. 0. 0. 0.]\n"
        + " [0. 0. 2. 0. 0. 0. 0. 0.]\n"
        + " [0. 0. 0. 2. 0. 0. 0. 0.]\n"
        + " [0. 0. 0. 0. 2. 0. 0. 0.]\n"
        + " [0. 0. 0. 0. 0. 2. 0. 0.]\n"
        + " [0. 0. 0. 0. 0. 0. 2. 0.]\n"
        + " [0. 0. 0. 0. 0. 0. 0. 2.]]\n"
        + "M1 =\n"
        + "[1 1 1 2 2 3 3 3]\n"
    )


@pytest.fixture
def qubit_circuit_with_unused_wires():
    """A qubit ciruit with unused wires."""
    with qml.tape.QuantumTape() as tape:
        qml.PauliX(0)
        qml.PauliX(5)
        qml.Toffoli(wires=[5, 1, 0])
        qml.expval(qml.PauliY(0))
        qml.expval(qml.PauliY(1))
        qml.expval(qml.PauliY(5))

    return tape


@pytest.fixture
def drawn_qubit_circuit_with_unused_wires():
    """The rendered circuit representation of the above qubit circuit."""
    return " 0: ──X──╭X──┤ ⟨Y⟩ \n" + " 1: ─────├C──┤ ⟨Y⟩ \n" + " 5: ──X──╰C──┤ ⟨Y⟩ \n"


@pytest.fixture
def qubit_circuit_with_probs():
    """A qubit ciruit with probs."""
    with qml.tape.QuantumTape() as tape:
        qml.PauliX(0)
        qml.PauliX(5)
        qml.Toffoli(wires=[5, 1, 0])
        qml.expval(qml.PauliY(0))
        qml.probs(wires=[1, 2, 4])

    return tape


@pytest.fixture
def qubit_circuit_with_state():
    """A qubit ciruit with a returned state."""
    with qml.tape.QuantumTape() as tape:
        qml.PauliX(0)
        qml.PauliX(5)
        qml.Toffoli(wires=[5, 1, 0])
        state()

    return tape


@pytest.fixture
def drawn_qubit_circuit_with_probs():
    """The rendered circuit representation of the above qubit circuit."""
    return (
        " 0: ──X──╭X───┤ ⟨Y⟩   \n"
        + " 1: ─────├C──╭┤ Probs \n"
        + " 2: ─────│───├┤ Probs \n"
        + " 4: ─────│───╰┤ Probs \n"
        + " 5: ──X──╰C───┤       \n"
    )


@pytest.fixture
def drawn_qubit_circuit_with_state():
    """The rendered circuit representation of the above qubit circuit."""
    return " 0: ──X──╭X──╭┤ State \n" + " 1: ─────├C──├┤ State \n" + " 5: ──X──╰C──╰┤ State \n"


@pytest.fixture
def qubit_circuit_with_interesting_wires():
    """A qubit ciruit with mixed-type wire labels."""

    with qml.tape.QuantumTape() as tape:
        qml.PauliX(0)
        qml.PauliX("b")
        qml.PauliX(-1)
        qml.Toffoli(wires=["b", "q2", 0])
        qml.expval(qml.PauliY(0))

    return tape


@pytest.fixture
def drawn_qubit_circuit_with_interesting_wires():
    """The rendered circuit representation of the above qubit circuit."""
    return (
        "  0: ──X──╭X──┤ ⟨Y⟩ \n"
        + " q2: ─────├C──┤     \n"
        + " -1: ──X──│───┤     \n"
        + "  b: ──X──╰C──┤     \n"
    )


class TestCircuitDrawerIntegration:
    """Test that tapes are properly drawn."""

    def test_qubit_circuit_with_values(
        self, parameterized_qubit_tape, drawn_parameterized_qubit_circuit_with_values
    ):
        """Test that a parametrized qubit circuit renders correctly with values."""
        output = parameterized_qubit_tape.draw(wire_order=qml.wires.Wires(range(5)))
        assert output == drawn_parameterized_qubit_circuit_with_values

    def test_wide_qubit_circuit_with_values(
        self, parameterized_wide_qubit_tape, drawn_parameterized_wide_qubit_tape_with_values
    ):
        """Test that a wide parametrized qubit circuit renders correctly with values."""
        output = parameterized_wide_qubit_tape.draw()

        assert output == drawn_parameterized_wide_qubit_tape_with_values

    def test_qubit_circuit_with_interesting_wires(
        self, qubit_circuit_with_interesting_wires, drawn_qubit_circuit_with_interesting_wires
    ):
        """Test that non-consecutive wires show correctly."""
        output = qubit_circuit_with_interesting_wires.draw(
            wire_order=qml.wires.Wires([0, "q2", -1, "b"])
        )

        assert output == drawn_qubit_circuit_with_interesting_wires

    def test_wide_cv_circuit(self, wide_cv_tape, drawn_wide_cv_tape):
        """Test that a wide CV circuit renders correctly."""
        output = wide_cv_tape.draw()

        assert output == drawn_wide_cv_tape

    @pytest.mark.slow
    def test_cv_circuit_with_values(
        self, parameterized_cv_tape, drawn_parameterized_cv_tape_with_values
    ):
        """Test that a parametrized CV circuit renders correctly with values."""
        output = parameterized_cv_tape.draw(wire_order=qml.wires.Wires(range(4)))
        assert output == drawn_parameterized_cv_tape_with_values

    def test_qubit_circuit_with_unused_wires(
        self, qubit_circuit_with_unused_wires, drawn_qubit_circuit_with_unused_wires
    ):
        """Test that a qubit circuit with unused wires renders correctly."""
        output = qubit_circuit_with_unused_wires.draw(wire_order=qml.wires.Wires([0, 1, 5]))

        assert output == drawn_qubit_circuit_with_unused_wires

    def test_qubit_circuit_with_probs(
        self, qubit_circuit_with_probs, drawn_qubit_circuit_with_probs
    ):
        """Test that a qubit circuit with probability output renders correctly."""
        output = qubit_circuit_with_probs.draw(wire_order=qml.wires.Wires(range(6)))
        assert output == drawn_qubit_circuit_with_probs

    def test_qubit_circuit_with_state(
        self, qubit_circuit_with_state, drawn_qubit_circuit_with_state
    ):
        """Test that a qubit circuit with unused wires renders correctly."""
        output = qubit_circuit_with_state.draw(wire_order=qml.wires.Wires(range(6)))

        assert output == drawn_qubit_circuit_with_state

    def test_direct_tape_integration(self):
        """Test that a regular tape renders correctly."""
        a, w = 2.3, [1.2, 3.2, 0.7]

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(0)
            qml.CRX(a, wires=[0, 1])
            qml.Rot(w[0], w[1], w[2], wires=[1])
            qml.CRX(-a, wires=[0, 1])

            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert tape.draw() == (
            " 0: ──H──╭C────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩ \n"
            + " 1: ─────╰RX(2.3)──Rot(1.2, 3.2, 0.7)──╰RX(-2.3)──╰┤ ⟨Z ⊗ Z⟩ \n"
        )

        assert tape.draw(charset="ascii") == (
            " 0: --H--+C----------------------------+C---------+| <Z @ Z> \n"
            + " 1: -----+RX(2.3)--Rot(1.2, 3.2, 0.7)--+RX(-2.3)--+| <Z @ Z> \n"
        )

    def test_same_wire_multiple_measurements(self):
        """Test that drawing a tape with multiple measurements on certain wires works correctly."""
        with qml.tape.QuantumTape() as tape:
            qml.RY(1.0, wires=0)
            qml.Hadamard(0)
            qml.RZ(2.0, wires=0)
            qml.expval(qml.PauliX(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliX(wires=[2]))
            qml.expval(qml.PauliX(wires=[0]) @ qml.PauliX(wires=[3]))

        expected = (
            " 0: ──RY(1)──H──RZ(2)──╭┤ ⟨X ⊗ X ⊗ X⟩ ╭┤ ⟨X ⊗ X⟩ \n"
            + " 1: ───────────────────├┤ ⟨X ⊗ X ⊗ X⟩ │┤         \n"
            + " 2: ───────────────────╰┤ ⟨X ⊗ X ⊗ X⟩ │┤         \n"
            + " 3: ────────────────────┤             ╰┤ ⟨X ⊗ X⟩ \n"
        )
        assert tape.draw() == expected

    def test_same_wire_multiple_measurements_many_obs(self):
        """Test that drawing a tape with multiple measurements on certain
        wires works correctly when there are more observables than the number of
        observables for any wire.
        """
        with qml.tape.QuantumTape() as tape:
            qml.RY(0.3, wires=0)
            qml.Hadamard(0)
            qml.RZ(0.2, wires=0)
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliZ(1))
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        expected = (
            " 0: ──RY(0.3)──H──RZ(0.2)──┤ ⟨Z⟩ ┤     ╭┤ ⟨Z ⊗ Z⟩ \n"
            + " 1: ───────────────────────┤     ┤ ⟨Z⟩ ╰┤ ⟨Z ⊗ Z⟩ \n"
        )
        assert tape.draw() == expected

    def test_qubit_circuit_with_max_length_kwdarg(self):
        """Test that a qubit circuit with max_length set to 30 renders correctly."""
        with qml.tape.QuantumTape() as tape:
            for i in range(3):
                qml.Hadamard(wires=i)
                qml.RX(i * 0.1, wires=i)
                qml.RY(i * 0.1, wires=i)
                qml.RZ(i * 0.1, wires=i)
            qml.expval(qml.PauliZ(0))

        expected = (
            " 0: ──H──RX(0)────RY(0)────RZ\n"
            + " 1: ──H──RX(0.1)──RY(0.1)──RZ\n"
            + " 2: ──H──RX(0.2)──RY(0.2)──RZ\n"
            + "\n"
            + " (0)────┤ ⟨Z⟩\n"
            + " (0.1)──┤    \n"
            + " (0.2)──┤    \n"
        )
        assert tape.draw(max_length=30) == expected

    def test_qubit_circuit_length_under_max_length_kwdarg(self):
        """Test that a qubit circuit with a circuit length less than the max_length renders correctly."""
        with qml.tape.QuantumTape() as tape:
            for i in range(3):
                qml.Hadamard(wires=i)
                qml.RX(i * 0.1, wires=i)
                qml.RY(i * 0.1, wires=i)
                qml.RZ(i * 0.1, wires=i)
            qml.expval(qml.PauliZ(0))

        expected = (
            " 0: ──H──RX(0)────RY(0)────RZ(0)────┤ ⟨Z⟩\n"
            + " 1: ──H──RX(0.1)──RY(0.1)──RZ(0.1)──┤    \n"
            + " 2: ──H──RX(0.2)──RY(0.2)──RZ(0.2)──┤    \n"
        )
        assert tape.draw(max_length=60) == expected

    def test_nested_tapes(self):

        with qml.tape.QuantumTape() as tape:
            with qml.tape.QuantumTape():
                qml.PauliX(0)
                qml.CNOT(wires=[0, 2])
                with qml.tape.QuantumTape():
                    qml.QuantumPhaseEstimation(
                        qml.PauliY.compute_matrix(), target_wires=[1], estimation_wires=[2]
                    )
                    qml.CNOT(wires=[1, 2])
            qml.Hadamard(1)
            with qml.tape.QuantumTape():
                qml.SWAP(wires=[0, 1])
            qml.state()

        expected = (
            " 0: ──╭QuantumTape:T0─────╭QuantumTape:T1──╭┤ State \n"
            + " 1: ──├QuantumTape:T0──H──╰QuantumTape:T1──├┤ State \n"
            + " 2: ──╰QuantumTape:T0──────────────────────╰┤ State \n"
            + "T0 =\n"
            + " 0: ──X──╭C───────────────────┤  \n"
            + " 2: ─────╰X──╭QuantumTape:T2──┤  \n"
            + " 1: ─────────╰QuantumTape:T2──┤  \n"
            + "T2 =\n"
            + " 1: ──╭QuantumPhaseEstimation(M0)──╭C──┤  \n"
            + " 2: ──╰QuantumPhaseEstimation(M0)──╰X──┤  \n"
            + "M0 =\n"
            + "[[ 0.+0.j -0.-1.j]\n"
            + " [ 0.+1.j  0.+0.j]]\n"
            + "\n"
            + "\n"
            + "T1 =\n"
            + " 0: ──╭SWAP──┤  \n"
            + " 1: ──╰SWAP──┤  \n"
            + "\n"
        )

        assert tape.draw(wire_order=qml.wires.Wires([0, 1, 2])) == expected


class TestWireOrdering:
    """Tests for wire ordering functionality"""

    def test_default_ordering(self):
        """Test that the default wire ordering matches the order of operations
        on the tape."""

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=-1)
            qml.CNOT(wires=["a", "q2"])
            qml.RX(0.2, wires="a")
            qml.expval(qml.PauliX(wires="q2"))

        res = tape.draw()
        expected = [
            " -1: ───H───────────┤     ",
            "  a: ──╭C──RX(0.2)──┤     ",
            " q2: ──╰X───────────┤ ⟨X⟩ \n",
        ]

        assert res == "\n".join(expected)

    def test_wire_reordering(self):
        """Test that wires are correctly reordered"""

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=-1)
            qml.CNOT(wires=["a", "q2"])
            qml.RX(0.2, wires="a")
            qml.expval(qml.PauliX(wires="q2"))

        res = tape.draw(wire_order=qml.wires.Wires(["q2", "a", -1]))
        expected = [
            " q2: ──╭X───────────┤ ⟨X⟩ ",
            "  a: ──╰C──RX(0.2)──┤     ",
            " -1: ───H───────────┤     \n",
        ]

        assert res == "\n".join(expected)

    def test_include_empty_wires(self):
        """Test that empty wires are correctly included"""

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=-1)
            qml.CNOT(wires=[-1, "q2"])
            qml.expval(qml.PauliX(wires="q2"))

        res = tape.draw(show_all_wires=True, wire_order=qml.wires.Wires([-1, "a", "q2", 0]))
        expected = [
            " -1: ──H──╭C──┤     ",
            "  a: ─────│───┤     ",
            " q2: ─────╰X──┤ ⟨X⟩ ",
            "  0: ─────────┤     \n",
        ]

        assert res == "\n".join(expected)

    def test_missing_wire(self):
        """Test that wires not specifically mentioned in the wire
        reordering are appended at the bottom of the circuit drawing"""

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=-1)
            qml.CNOT(wires=["a", "q2"])
            qml.RX(0.2, wires="a")
            qml.expval(qml.PauliX(wires="q2"))

        # test one missing wire
        res = tape.draw(wire_order=qml.wires.Wires(["q2", "a"]))
        expected = [
            " q2: ──╭X───────────┤ ⟨X⟩ ",
            "  a: ──╰C──RX(0.2)──┤     ",
            " -1: ───H───────────┤     \n",
        ]

        assert res == "\n".join(expected)

        # test one missing wire
        res = tape.draw(wire_order=qml.wires.Wires(["q2", -1]))
        expected = [
            " q2: ─────╭X───────────┤ ⟨X⟩ ",
            " -1: ──H──│────────────┤     ",
            "  a: ─────╰C──RX(0.2)──┤     \n",
        ]

        assert res == "\n".join(expected)

        # test multiple missing wires
        res = tape.draw(wire_order=qml.wires.Wires(["q2"]))
        expected = [
            " q2: ─────╭X───────────┤ ⟨X⟩ ",
            " -1: ──H──│────────────┤     ",
            "  a: ─────╰C──RX(0.2)──┤     \n",
        ]

        assert res == "\n".join(expected)

    def test_no_ops_draws(self):
        """Test that a tape with no operations still draws correctly"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliX(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliX(wires=[2]))

        res = tape.draw()
        expected = [
            " 0: ──╭┤ ⟨X ⊗ X ⊗ X⟩ \n",
            " 1: ──├┤ ⟨X ⊗ X ⊗ X⟩ \n",
            " 2: ──╰┤ ⟨X ⊗ X ⊗ X⟩ \n",
        ]

        assert res == "".join(expected)
