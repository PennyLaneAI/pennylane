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
Unit tests for the :mod:`pennylane.circuit_drawer` module.
"""
from unittest.mock import Mock
import pytest
import numpy as np

import pennylane as qml
from pennylane.circuit_drawer import CircuitDrawer
from pennylane.circuit_drawer.circuit_drawer import _remove_duplicates
from pennylane.circuit_drawer.grid import Grid, _transpose


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


@pytest.fixture
def dummy_circuit_drawer():
    """A dummy CircuitDrawer instance."""
    return CircuitDrawer(dummy_raw_operation_grid, dummy_raw_observable_grid)


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
        for wire in op.wires:
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
        [[qml.Toffoli(wires=[0, 4, 5]), qml.PauliX(wires=[2]), qml.Hadamard(wires=[3])]], 6
    )
    multiwire_and_single_wire_gate_representation_grid = Grid(
        [["╭"], ["│"], ["│"], ["│"], ["├"], ["╰"]]
    )

    all_wire_state_preparation_grid = to_grid(
        [[qml.BasisState(np.array([0, 1, 0, 0, 1]), wires=[0, 1, 2, 3, 4, 5])]], 6
    )
    all_wire_state_preparation_representation_grid = Grid(
        [["╭"], ["├"], ["├"], ["├"], ["├"], ["╰"]]
    )

    multiwire_gate_grid = to_grid([[qml.CNOT(wires=[0, 1]), qml.CNOT(wires=[3, 4])]], 5)

    multiwire_gate_representation_grid = Grid([["╭"], ["╰"], [""], ["╭"], ["╰"],])

    multi_and_single_wire_gate_grid = to_grid(
        [[qml.CNOT(wires=[0, 1]), qml.PauliX(2), qml.CNOT(wires=[3, 5]), qml.Hadamard(6)]], 7
    )

    multi_and_single_wire_gate_representation_grid = Grid(
        [["╭"], ["╰"], [""], ["╭"], ["│"], ["╰"], [""],]
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
    def test_resolve_decorations(self, dummy_circuit_drawer, grid, target_representation_grid):
        """Test that decorations are properly resolved."""

        representation_grid = Grid()
        dummy_circuit_drawer.resolve_decorations(grid, representation_grid)

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
    def test_move_multi_wire_gates(self, dummy_circuit_drawer, grid, target_grid):
        """Test that decorations are properly resolved."""

        operator_grid = grid.copy()
        dummy_circuit_drawer.move_multi_wire_gates(operator_grid)

        print(operator_grid)
        print(target_grid)

        assert_nested_lists_equal(operator_grid.raw_grid, target_grid.raw_grid)


@pytest.fixture
def parameterized_qubit_qnode():
    """A parametrized qubit ciruit."""

    def qfunc(a, b, c, angles):
        qml.RX(a, wires=0)
        qml.RX(b, wires=1)
        qml.PauliZ(1)
        qml.CNOT(wires=[0, 1])
        qml.CRY(b, wires=[3, 1])
        qml.RX(angles[0], wires=0)
        qml.RX(4 * angles[1], wires=1)
        qml.PhaseShift(17 / 9 * c, wires=2)
        qml.RZ(b, wires=3)
        qml.RX(angles[2], wires=2)
        qml.CRY(0.3589, wires=[3, 1])
        qml.QubitUnitary(np.eye(2), wires=[2])
        qml.Toffoli(wires=[0, 2, 1])
        qml.CNOT(wires=[0, 2])
        qml.PauliZ(wires=[1])
        qml.PauliZ(wires=[1])
        qml.CZ(wires=[0, 1])
        qml.CZ(wires=[0, 2])
        qml.CNOT(wires=[2, 1])
        qml.CNOT(wires=[0, 2])
        qml.SWAP(wires=[0, 2])
        qml.CNOT(wires=[1, 3])
        qml.RZ(b, wires=3)
        qml.CSWAP(wires=[4, 0, 1])

        return [
            qml.expval(qml.PauliY(0)),
            qml.var(qml.Hadamard(wires=1)),
            qml.sample(qml.PauliX(2)),
            qml.expval(qml.Hermitian(np.eye(4), wires=[3, 4])),
        ]

    dev = qml.device("default.qubit", wires=5)

    qnode = qml.QNode(qfunc, dev)
    qnode._construct((0.1, 0.2, 0.3, np.array([0.4, 0.5, 0.6])), {})
    qnode.evaluate((0.1, 0.2, 0.3, np.array([0.4, 0.5, 0.6])), {})

    return qnode


@pytest.fixture
def drawn_parameterized_qubit_circuit_with_variable_names():
    """The rendered circuit representation of the above qubit circuit with variable names."""
    return (
        " 0: ──RX(a)───────────────────────╭C────RX(angles[0])───────────────────────────────╭C─────╭C─────╭C──╭C──────────╭C──╭SWAP───╭SWAP───┤ ⟨Y⟩       \n"
        + " 1: ──RX(b)────────Z──────────────╰X───╭RY(b)──────────RX(4*angles[1])──╭RY(0.359)──├X──Z──│───Z──╰Z──│───╭X──╭C──│───│───────├SWAP───┤ Var[H]    \n"
        + " 2: ──Rϕ(1.889*c)──RX(angles[2])───U0──│────────────────────────────────│───────────╰C─────╰X─────────╰Z──╰C──│───╰X──╰SWAP───│───────┤ Sample[X] \n"
        + " 3: ───────────────────────────────────╰C──────────────RZ(b)────────────╰C────────────────────────────────────╰X───────RZ(b)──│──────╭┤ ⟨H0⟩      \n"
        + " 4: ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╰C─────╰┤ ⟨H0⟩      \n"
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
def drawn_parameterized_qubit_circuit_with_values():
    """The rendered circuit representation of the above qubit circuit with variable values."""
    return (
        " 0: ──RX(0.1)─────────────╭C────RX(0.4)───────────────────────╭C─────╭C─────╭C──╭C──────────╭C──╭SWAP─────╭SWAP───┤ ⟨Y⟩       \n"
        + " 1: ──RX(0.2)────Z────────╰X───╭RY(0.2)──RX(2.0)──╭RY(0.359)──├X──Z──│───Z──╰Z──│───╭X──╭C──│───│─────────├SWAP───┤ Var[H]    \n"
        + " 2: ──Rϕ(0.567)──RX(0.6)───U0──│──────────────────│───────────╰C─────╰X─────────╰Z──╰C──│───╰X──╰SWAP─────│───────┤ Sample[X] \n"
        + " 3: ───────────────────────────╰C────────RZ(0.2)──╰C────────────────────────────────────╰X───────RZ(0.2)──│──────╭┤ ⟨H0⟩      \n"
        + " 4: ──────────────────────────────────────────────────────────────────────────────────────────────────────╰C─────╰┤ ⟨H0⟩      \n"
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
def parameterized_wide_qubit_qnode():
    """A wide parametrized qubit circuit."""

    def qfunc(a, b, c, d, e, f):
        qml.RX(a, wires=0)
        qml.RX(b, wires=1)
        [qml.CNOT(wires=[2 * i, 2 * i + 1]) for i in range(4)]
        [qml.CNOT(wires=[i, i + 4]) for i in range(4)]
        [qml.PauliY(wires=[2 * i]) for i in range(4)]
        [qml.CSWAP(wires=[i + 2, i, i + 4]) for i in range(4)]
        qml.RX(a, wires=0)
        qml.RX(b, wires=1)

        return [qml.expval(qml.Hermitian(np.eye(4), wires=[i, i + 4])) for i in range(4)]

    dev = qml.device("default.qubit", wires=8)
    qnode = qml.QNode(qfunc, dev)
    qnode._construct((0.1, 0.2, 0.3, 47 / 17, 0.5, 0.6), {})
    qnode.evaluate((0.1, 0.2, 0.3, 47 / 17, 0.5, 0.6), {})

    return qnode


@pytest.fixture
def drawn_parameterized_wide_qubit_qnode_with_variable_names():
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
def drawn_parameterized_wide_qubit_qnode_with_values():
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
def wide_cv_qnode():
    """A wide unparametrized CV circuit."""

    def qfunc():
        qml.GaussianState(
            np.array([(2 * i + 2) // 2 for i in range(16)]), 2 * np.eye(16), wires=list(range(8))
        )
        [qml.Beamsplitter(0.4, 0, wires=[2 * i, 2 * i + 1]) for i in range(4)]
        [qml.Beamsplitter(0.25475, 0.2312344, wires=[i, i + 4]) for i in range(4)]

        return [
            qml.expval(qml.FockStateProjector(np.array([1, 1]), wires=[i, i + 4])) for i in range(4)
        ]

    dev = qml.device("default.gaussian", wires=8)
    qnode = qml.QNode(qfunc, dev)
    qnode._construct((), {})
    qnode.evaluate((), {})

    return qnode


@pytest.fixture
def drawn_wide_cv_qnode():
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
        + "[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]\n"
        + "M1 =\n"
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
    )


@pytest.fixture
def parameterized_cv_qnode():
    """A parametrized CV circuit."""

    def qfunc(a, b, c, d, e, f):
        qml.ThermalState(3, wires=[1])
        qml.GaussianState(np.array([1, 1, 1, 2, 2, 3, 3, 3]), 2 * np.eye(8), wires=[0, 1, 2, 3])
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

        return [
            qml.expval(qml.ops.PolyXP(np.array([0, 1, 2]), wires=0)),
            qml.expval(qml.ops.QuadOperator(4, wires=1)),
            qml.expval(qml.ops.FockStateProjector(np.array([1, 5]), wires=[2, 3])),
        ]

    dev = qml.device("default.gaussian", wires=4)

    qnode = qml.QNode(qfunc, dev)
    qnode._construct((0.1, 0.2, 0.3, 47 / 17, 0.5, 0.6), {})
    qnode.evaluate((0.1, 0.2, 0.3, 47 / 17, 0.5, 0.6), {})

    return qnode


@pytest.fixture
def drawn_parameterized_cv_qnode_with_variable_names():
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
def drawn_parameterized_cv_qnode_with_values():
    """The rendered circuit representation of the above CV circuit with variable values."""
    return (
        " 0: ──────────────╭Gaussian(M0,M1)──R(0.1)─────╭BS(2.765, 1)───S(2.3, 0)─────────────────────────────────────────────────────────────╭C─────P(4)───┤ ⟨x₀+2p₀⟩          \n"
        + " 1: ──Thermal(3)──├Gaussian(M0,M1)──R(0.2)─────╰BS(2.765, 1)──╭BS(0.5, 1)─────────────╭BS(2.765, 1)───────────────╭S(2, 2)──╭Z(2.3)──│─────────────┤ ⟨cos(4)x+sin(4)p⟩ \n"
        + " 2: ──────────────├Gaussian(M0,M1)────────────────────────────╰BS(0.5, 1)──S(2.3, 0)──╰BS(2.765, 1)──╭BS(0.5, 1)──│─────────╰C───────│────────────╭┤ ⟨|1,5╳1,5|⟩       \n"
        + " 3: ──────────────╰Gaussian(M0,M1)──D(0.6, 0)────────────────────────────────────────────────────────╰BS(0.5, 1)──╰S(2, 2)───────────╰X(2)────────╰┤ ⟨|1,5╳1,5|⟩       \n"
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
def qubit_circuit_with_unused_wires():
    """A qubit ciruit with unused wires."""

    def qfunc():
        qml.PauliX(0)
        qml.PauliX(5)
        qml.Toffoli(wires=[5, 1, 0])

        return [
            qml.expval(qml.PauliY(0)),
            qml.expval(qml.PauliY(1)),
            qml.expval(qml.PauliY(5)),
        ]

    dev = qml.device("default.qubit", wires=6)

    qnode = qml.QNode(qfunc, dev)
    qnode._construct((), {})
    qnode.evaluate((), {})

    return qnode


@pytest.fixture
def drawn_qubit_circuit_with_unused_wires():
    """The rendered circuit representation of the above qubit circuit."""
    return (
          " 0: ──X──╭X──┤ ⟨Y⟩ \n" +
          " 1: ─────├C──┤ ⟨Y⟩ \n" +
          " 5: ──X──╰C──┤ ⟨Y⟩ \n"
    )


class TestCircuitDrawerIntegration:
    """Test that QNodes are properly drawn."""

    def test_qubit_circuit_with_variable_names(
        self, parameterized_qubit_qnode, drawn_parameterized_qubit_circuit_with_variable_names
    ):
        """Test that a parametrized qubit circuit renders correctly with variable names."""
        output = parameterized_qubit_qnode.circuit.draw(show_variable_names=True)

        assert output == drawn_parameterized_qubit_circuit_with_variable_names

    def test_qubit_circuit_with_values(
        self, parameterized_qubit_qnode, drawn_parameterized_qubit_circuit_with_values
    ):
        """Test that a parametrized qubit circuit renders correctly with values."""
        output = parameterized_qubit_qnode.circuit.draw(show_variable_names=False)

        assert output == drawn_parameterized_qubit_circuit_with_values

    def test_wide_qubit_circuit_with_variable_names(
        self,
        parameterized_wide_qubit_qnode,
        drawn_parameterized_wide_qubit_qnode_with_variable_names,
    ):
        """Test that a wide parametrized qubit circuit renders correctly with variable names."""
        output = parameterized_wide_qubit_qnode.draw(show_variable_names=True)

        assert output == drawn_parameterized_wide_qubit_qnode_with_variable_names

    def test_wide_qubit_circuit_with_values(
        self, parameterized_wide_qubit_qnode, drawn_parameterized_wide_qubit_qnode_with_values
    ):
        """Test that a wide parametrized qubit circuit renders correctly with values."""
        output = parameterized_wide_qubit_qnode.draw(show_variable_names=False)

        assert output == drawn_parameterized_wide_qubit_qnode_with_values

    def test_wide_cv_circuit(self, wide_cv_qnode, drawn_wide_cv_qnode):
        """Test that a wide CV circuit renders correctly."""
        output = wide_cv_qnode.draw()

        assert output == drawn_wide_cv_qnode

    def test_cv_circuit_with_variable_names(
        self, parameterized_cv_qnode, drawn_parameterized_cv_qnode_with_variable_names
    ):
        """Test that a parametrized CV circuit renders correctly with variable names."""
        output = parameterized_cv_qnode.draw(show_variable_names=True)

        assert output == drawn_parameterized_cv_qnode_with_variable_names

    def test_cv_circuit_with_values(
        self, parameterized_cv_qnode, drawn_parameterized_cv_qnode_with_values
    ):
        """Test that a parametrized CV circuit renders correctly with values."""
        output = parameterized_cv_qnode.draw(show_variable_names=False)

        assert output == drawn_parameterized_cv_qnode_with_values

    def test_qubit_circuit_with_unused_wires(self, qubit_circuit_with_unused_wires, drawn_qubit_circuit_with_unused_wires):
        """Test that a qubit circuit with unused wires renders correctly."""
        output = qubit_circuit_with_unused_wires.draw()

        assert output == drawn_qubit_circuit_with_unused_wires

    def test_direct_qnode_integration(self):
        """Test that a regular QNode renders correctly."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def qfunc(a, w):
            qml.Hadamard(0)
            qml.CRX(a, wires=[0, 1])
            qml.Rot(w[0], w[1], w[2], wires=[1])
            qml.CRX(-a, wires=[0, 1])

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        res = qfunc(2.3, [1.2, 3.2, 0.7])

        assert qfunc.draw() == (
            " 0: ──H──╭C────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩ \n"
            + " 1: ─────╰RX(2.3)──Rot(1.2, 3.2, 0.7)──╰RX(-2.3)──╰┤ ⟨Z ⊗ Z⟩ \n"
        )

        assert qfunc.draw(charset="ascii") == (
            " 0: --H--+C----------------------------+C---------+| <Z @ Z> \n"
            + " 1: -----+RX(2.3)--Rot(1.2, 3.2, 0.7)--+RX(-2.3)--+| <Z @ Z> \n"
        )

        assert qfunc.draw(show_variable_names=True) == (
            " 0: ──H──╭C─────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩ \n"
            + " 1: ─────╰RX(a)──Rot(w[0], w[1], w[2])──╰RX(-1*a)──╰┤ ⟨Z ⊗ Z⟩ \n"
        )
