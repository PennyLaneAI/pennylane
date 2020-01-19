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

import itertools
import pytest
import numpy as np
from unittest.mock import Mock

import pennylane as qml
from pennylane.circuit_drawer import (
    _remove_duplicates,
    _transpose,
    Grid,
    RepresentationResolver,
    CircuitDrawer,
)
from pennylane.variable import Variable


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


class TestGrid:
    """Test the Grid helper class."""

    def test_init(self):
        """Test that the Grid class is initialized correctly."""

        raw_grid = [[0, 3], [1, 4], [2, 5]]
        grid = Grid(raw_grid)

        assert grid.raw_grid == raw_grid
        assert grid.raw_grid_transpose == [[0, 1, 2], [3, 4, 5]]

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

        assert grid.raw_grid_transpose == expected_transposed_grid
        assert grid.raw_grid == _transpose(expected_transposed_grid)

    def test_append_layer(self):
        """Test that layer appending works properly."""

        raw_grid = [[0, 3], [1, 4], [2, 5]]
        grid = Grid(raw_grid)

        grid.append_layer([6, 7, 8])

        assert grid.raw_grid == [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
        assert grid.raw_grid_transpose == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    @pytest.mark.parametrize(
        "idx,expected_transposed_grid", [(0, [[6, 7, 8], [3, 4, 5]]), (1, [[0, 1, 2], [6, 7, 8]]),]
    )
    def test_replace_layer(self, idx, expected_transposed_grid):
        """Test that layer replacement works properly."""

        raw_grid = [[0, 3], [1, 4], [2, 5]]
        grid = Grid(raw_grid)

        grid.replace_layer(idx, [6, 7, 8])

        assert grid.raw_grid_transpose == expected_transposed_grid
        assert grid.raw_grid == _transpose(expected_transposed_grid)

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

        assert grid.raw_grid == expected_grid
        assert grid.raw_grid_transpose == _transpose(expected_grid)

    def test_append_wire(self):
        """Test that wire appending works properly."""

        raw_grid = [[0, 3], [1, 4], [2, 5]]
        grid = Grid(raw_grid)

        grid.append_wire([6, 7])

        assert grid.raw_grid == [[0, 3], [1, 4], [2, 5], [6, 7]]
        assert grid.raw_grid_transpose == [[0, 1, 2, 6], [3, 4, 5, 7]]

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
            assert grid.layer(idx) == layer

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
            assert grid.wire(idx) == wire

    def test_copy(self):
        """Test that copy copies the grid."""
        raw_grid = [[0, 3], [1, 4], [2, 5]]
        grid = Grid(raw_grid)

        other_grid = grid.copy()

        # Assert that everything is indeed copied
        assert other_grid is not grid
        assert other_grid.raw_grid is not grid.raw_grid
        assert other_grid.raw_grid_transpose is not grid.raw_grid_transpose

        # Assert the copy is correct
        assert other_grid.raw_grid == grid.raw_grid
        assert other_grid.raw_grid_transpose == grid.raw_grid_transpose

    def test_append_grid_by_layers(self):
        """Test appending a grid to another by layers."""
        raw_grid_transpose1 = [[0, 3], [1, 4], [2, 5]]
        raw_grid_transpose2 = [[6, 7], [8, 9]]

        grid1 = Grid(_transpose(raw_grid_transpose1))
        grid2 = Grid(_transpose(raw_grid_transpose2))

        grid1.append_grid_by_layers(grid2)

        assert grid1.raw_grid_transpose == [[0, 3], [1, 4], [2, 5], [6, 7], [8, 9]]
        assert grid1.raw_grid == _transpose([[0, 3], [1, 4], [2, 5], [6, 7], [8, 9]])

    def test_str(self):
        """Test string rendering of Grid."""
        raw_grid = [[0, 3], [1, 4], [2, 5]]
        grid = Grid(raw_grid)

        assert str(grid) == "[0, 3]\n[1, 4]\n[2, 5]\n"


@pytest.fixture
def unicode_representation_resolver():
    """An instance of a RepresentationResolver with unicode charset."""
    return RepresentationResolver()


@pytest.fixture
def ascii_representation_resolver():
    """An instance of a RepresentationResolver with unicode charset."""
    return RepresentationResolver(charset=qml.circuit_drawer.AsciiCharSet)


@pytest.fixture
def unicode_representation_resolver_varnames():
    """An instance of a RepresentationResolver with unicode charset and show_variable_names=True."""
    return RepresentationResolver(show_variable_names=True)


@pytest.fixture
def variable(monkeypatch):
    """A mocked Variable instance for a non-keyword variable."""
    monkeypatch.setattr(Variable, "free_param_values", [0, 1, 2, 3])
    yield Variable(2, "test")


@pytest.fixture
def kwarg_variable(monkeypatch):
    """A mocked Variable instance for a keyword variable."""
    monkeypatch.setattr(Variable, "kwarg_values", {"kwarg_test": [0, 1, 2, 3]})
    yield Variable(1, "kwarg_test", True)


class TestRepresentationResolver:
    """Test the RepresentationResolver class."""

    @pytest.mark.parametrize(
        "list,element,index,list_after",
        [
            ([1, 2, 3], 2, 1, [1, 2, 3]),
            ([1, 2, 2, 3], 2, 1, [1, 2, 2, 3]),
            ([1, 2, 3], 4, 3, [1, 2, 3, 4]),
        ],
    )
    def test_index_of_array_or_append(self, list, element, index, list_after):
        """Test the method index_of_array_or_append."""

        assert RepresentationResolver.index_of_array_or_append(element, list) == index
        assert list == list_after

    @pytest.mark.parametrize("par,expected", [(3, "3"), (5.236422, "5.236"),])
    def test_single_parameter_representation(self, unicode_representation_resolver, par, expected):
        """Test that single parameters are properly resolved."""
        assert unicode_representation_resolver.single_parameter_representation(par) == expected

    def test_single_parameter_representation_variable(
        self, unicode_representation_resolver, variable
    ):
        """Test that variables are properly resolved."""

        assert unicode_representation_resolver.single_parameter_representation(variable) == "2"

    def test_single_parameter_representation_kwarg_variable(
        self, unicode_representation_resolver, kwarg_variable
    ):
        """Test that kwarg variables are properly resolved."""

        assert (
            unicode_representation_resolver.single_parameter_representation(kwarg_variable) == "1"
        )

    @pytest.mark.parametrize("par,expected", [(3, "3"), (5.236422, "5.236"),])
    def test_single_parameter_representation_varnames(
        self, unicode_representation_resolver_varnames, par, expected
    ):
        """Test that single parameters are properly resolved when show_variable_names is True."""
        assert (
            unicode_representation_resolver_varnames.single_parameter_representation(par)
            == expected
        )

    def test_single_parameter_representation_variable_varnames(
        self, unicode_representation_resolver_varnames, variable
    ):
        """Test that variables are properly resolved when show_variable_names is True."""

        assert (
            unicode_representation_resolver_varnames.single_parameter_representation(variable)
            == "test"
        )

    def test_single_parameter_representation_kwarg_variable_varnames(
        self, unicode_representation_resolver_varnames, kwarg_variable
    ):
        """Test that kwarg variables are properly resolved when show_variable_names is True."""

        assert (
            unicode_representation_resolver_varnames.single_parameter_representation(kwarg_variable)
            == "kwarg_test"
        )

    @pytest.mark.parametrize(
        "op,wire,target",
        [
            (qml.PauliX(wires=[1]), 1, "X"),
            (qml.CNOT(wires=[0, 1]), 1, "X"),
            (qml.CNOT(wires=[0, 1]), 0, "C"),
            (qml.Toffoli(wires=[0, 2, 1]), 1, "X"),
            (qml.Toffoli(wires=[0, 2, 1]), 0, "C"),
            (qml.Toffoli(wires=[0, 2, 1]), 2, "C"),
            (qml.CSWAP(wires=[0, 2, 1]), 1, "SWAP"),
            (qml.CSWAP(wires=[0, 2, 1]), 2, "SWAP"),
            (qml.CSWAP(wires=[0, 2, 1]), 0, "C"),
            (qml.PauliY(wires=[1]), 1, "Y"),
            (qml.PauliZ(wires=[1]), 1, "Z"),
            (qml.CZ(wires=[0, 1]), 1, "Z"),
            (qml.CZ(wires=[0, 1]), 0, "C"),
            (qml.Identity(wires=[1]), 1, "I"),
            (qml.Hadamard(wires=[1]), 1, "H"),
            (qml.CRX(3.14, wires=[0, 1]), 1, "RX(3.14)"),
            (qml.CRX(3.14, wires=[0, 1]), 0, "C"),
            (qml.CRY(3.14, wires=[0, 1]), 1, "RY(3.14)"),
            (qml.CRY(3.14, wires=[0, 1]), 0, "C"),
            (qml.CRZ(3.14, wires=[0, 1]), 1, "RZ(3.14)"),
            (qml.CRZ(3.14, wires=[0, 1]), 0, "C"),
            (qml.CRot(3.14, 2.14, 1.14, wires=[0, 1]), 1, "Rot(3.14, 2.14, 1.14)"),
            (qml.CRot(3.14, 2.14, 1.14, wires=[0, 1]), 0, "C"),
            (qml.PhaseShift(3.14, wires=[0]), 0, "P(3.14)"),
            (qml.Beamsplitter(1, 2, wires=[0, 1]), 1, "BS(1, 2)"),
            (qml.Beamsplitter(1, 2, wires=[0, 1]), 0, "BS(1, 2)"),
            (qml.Squeezing(1, 2, wires=[1]), 1, "S(1, 2)"),
            (qml.TwoModeSqueezing(1, 2, wires=[0, 1]), 1, "S(1, 2)"),
            (qml.TwoModeSqueezing(1, 2, wires=[0, 1]), 0, "S(1, 2)"),
            (qml.Displacement(1, 2, wires=[1]), 1, "D(1, 2)"),
            (qml.NumberOperator(wires=[1]), 1, "n"),
            (qml.Rotation(3.14, wires=[1]), 1, "R(3.14)"),
            (qml.ControlledAddition(3.14, wires=[0, 1]), 1, "Add(3.14)"),
            (qml.ControlledAddition(3.14, wires=[0, 1]), 0, "C"),
            (qml.ControlledPhase(3.14, wires=[0, 1]), 1, "R(3.14)"),
            (qml.ControlledPhase(3.14, wires=[0, 1]), 0, "C"),
            (qml.ThermalState(3, wires=[1]), 1, "Thermal(3)"),
            (
                qml.GaussianState(np.array([1, 2]), np.array([[2, 0], [0, 2]]), wires=[1]),
                1,
                "Gaussian(M0, M1)",
            ),
            (qml.QuadraticPhase(3.14, wires=[1]), 1, "QuadPhase(3.14)"),
            (qml.RX(3.14, wires=[1]), 1, "RX(3.14)"),
            (qml.S(wires=[2]), 2, "S"),
            (qml.T(wires=[2]), 2, "T"),
            (qml.RX(3.14, wires=[1]), 1, "RX(3.14)"),
            (qml.RY(3.14, wires=[1]), 1, "RY(3.14)"),
            (qml.RZ(3.14, wires=[1]), 1, "RZ(3.14)"),
            (qml.Rot(3.14, 2.14, 1.14, wires=[1]), 1, "Rot(3.14, 2.14, 1.14)"),
            (qml.U1(3.14, wires=[1]), 1, "U1(3.14)"),
            (qml.U2(3.14, 2.14, wires=[1]), 1, "U2(3.14, 2.14)"),
            (qml.U3(3.14, 2.14, 1.14, wires=[1]), 1, "U3(3.14, 2.14, 1.14)"),
            (qml.BasisState(np.array([0, 1, 0]), wires=[1, 2, 3]), 1, "|0⟩"),
            (qml.BasisState(np.array([0, 1, 0]), wires=[1, 2, 3]), 2, "|1⟩"),
            (qml.BasisState(np.array([0, 1, 0]), wires=[1, 2, 3]), 3, "|0⟩"),
            (qml.QubitStateVector(np.array([0, 1, 0, 0]), wires=[1, 2]), 1, "QubitStateVector(M0)"),
            (qml.QubitStateVector(np.array([0, 1, 0, 0]), wires=[1, 2]), 2, "QubitStateVector(M0)"),
            (qml.QubitUnitary(np.eye(2), wires=[1]), 1, "U0"),
            (qml.QubitUnitary(np.eye(4), wires=[1, 2]), 2, "U0"),
            (qml.Kerr(3.14, wires=[1]), 1, "Kerr(3.14)"),
            (qml.CrossKerr(3.14, wires=[1, 2]), 1, "CrossKerr(3.14)"),
            (qml.CrossKerr(3.14, wires=[1, 2]), 2, "CrossKerr(3.14)"),
            (qml.CubicPhase(3.14, wires=[1]), 1, "CubicPhase(3.14)"),
            (qml.Interferometer(np.eye(4), wires=[1, 3]), 1, "Interferometer(M0)"),
            (qml.Interferometer(np.eye(4), wires=[1, 3]), 3, "Interferometer(M0)"),
            (qml.CatState(3.14, 2.14, 1, wires=[1]), 1, "CatState(3.14, 2.14, 1)"),
            (qml.CoherentState(3.14, 2.14, wires=[1]), 1, "CoherentState(3.14, 2.14)"),
            (
                qml.FockDensityMatrix(np.kron(np.eye(4), np.eye(4)), wires=[1, 2]),
                1,
                "FockDensityMatrix(M0)",
            ),
            (
                qml.FockDensityMatrix(np.kron(np.eye(4), np.eye(4)), wires=[1, 2]),
                2,
                "FockDensityMatrix(M0)",
            ),
            (
                qml.DisplacedSqueezedState(3.14, 2.14, 1.14, 0.14, wires=[1]),
                1,
                "DisplacedSqueezedState(3.14, 2.14, 1.14, 0.14)",
            ),
            (qml.FockState(7, wires=[1]), 1, "|7⟩"),
            (qml.FockStateVector(np.array([4, 5, 7]), wires=[1, 2, 3]), 1, "|4⟩"),
            (qml.FockStateVector(np.array([4, 5, 7]), wires=[1, 2, 3]), 2, "|5⟩"),
            (qml.FockStateVector(np.array([4, 5, 7]), wires=[1, 2, 3]), 3, "|7⟩"),
            (qml.SqueezedState(3.14, 2.14, wires=[1]), 1, "SqueezedState(3.14, 2.14)"),
            (qml.Hermitian(np.eye(4), wires=[1, 2]), 1, "H0"),
            (qml.Hermitian(np.eye(4), wires=[1, 2]), 2, "H0"),
            (qml.X(wires=[1]), 1, "x"),
            (qml.P(wires=[1]), 1, "p"),
            (qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]), 1, "|4, 5, 7╳4, 5, 7|"),
            (
                qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1]),
                2,
                "1.0 + 2.0 x₀ - 1.3 x₁ + 6.0 p₁",
            ),
            (
                qml.PolyXP(
                    np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                ),
                1,
                "1.2 + 1.1 x₀ + 3.2 p₀ + 1.2 x₀² + 2.3 p₀² + 3.0 x₀p₀",
            ),
            (qml.QuadOperator(3.14, wires=[1]), 1, "cos(3.14)x + sin(3.14)p"),
        ],
    )
    def test_operator_representation_unicode(self, unicode_representation_resolver, op, wire, target):
        """Test that an Operator instance is properly resolved."""
        assert unicode_representation_resolver.operator_representation(op, wire) == target
        
    @pytest.mark.parametrize(
        "op,wire,target",
        [
            (qml.PauliX(wires=[1]), 1, "X"),
            (qml.CNOT(wires=[0, 1]), 1, "X"),
            (qml.CNOT(wires=[0, 1]), 0, "C"),
            (qml.Toffoli(wires=[0, 2, 1]), 1, "X"),
            (qml.Toffoli(wires=[0, 2, 1]), 0, "C"),
            (qml.Toffoli(wires=[0, 2, 1]), 2, "C"),
            (qml.CSWAP(wires=[0, 2, 1]), 1, "SWAP"),
            (qml.CSWAP(wires=[0, 2, 1]), 2, "SWAP"),
            (qml.CSWAP(wires=[0, 2, 1]), 0, "C"),
            (qml.PauliY(wires=[1]), 1, "Y"),
            (qml.PauliZ(wires=[1]), 1, "Z"),
            (qml.CZ(wires=[0, 1]), 1, "Z"),
            (qml.CZ(wires=[0, 1]), 0, "C"),
            (qml.Identity(wires=[1]), 1, "I"),
            (qml.Hadamard(wires=[1]), 1, "H"),
            (qml.CRX(3.14, wires=[0, 1]), 1, "RX(3.14)"),
            (qml.CRX(3.14, wires=[0, 1]), 0, "C"),
            (qml.CRY(3.14, wires=[0, 1]), 1, "RY(3.14)"),
            (qml.CRY(3.14, wires=[0, 1]), 0, "C"),
            (qml.CRZ(3.14, wires=[0, 1]), 1, "RZ(3.14)"),
            (qml.CRZ(3.14, wires=[0, 1]), 0, "C"),
            (qml.CRot(3.14, 2.14, 1.14, wires=[0, 1]), 1, "Rot(3.14, 2.14, 1.14)"),
            (qml.CRot(3.14, 2.14, 1.14, wires=[0, 1]), 0, "C"),
            (qml.PhaseShift(3.14, wires=[0]), 0, "P(3.14)"),
            (qml.Beamsplitter(1, 2, wires=[0, 1]), 1, "BS(1, 2)"),
            (qml.Beamsplitter(1, 2, wires=[0, 1]), 0, "BS(1, 2)"),
            (qml.Squeezing(1, 2, wires=[1]), 1, "S(1, 2)"),
            (qml.TwoModeSqueezing(1, 2, wires=[0, 1]), 1, "S(1, 2)"),
            (qml.TwoModeSqueezing(1, 2, wires=[0, 1]), 0, "S(1, 2)"),
            (qml.Displacement(1, 2, wires=[1]), 1, "D(1, 2)"),
            (qml.NumberOperator(wires=[1]), 1, "n"),
            (qml.Rotation(3.14, wires=[1]), 1, "R(3.14)"),
            (qml.ControlledAddition(3.14, wires=[0, 1]), 1, "Add(3.14)"),
            (qml.ControlledAddition(3.14, wires=[0, 1]), 0, "C"),
            (qml.ControlledPhase(3.14, wires=[0, 1]), 1, "R(3.14)"),
            (qml.ControlledPhase(3.14, wires=[0, 1]), 0, "C"),
            (qml.ThermalState(3, wires=[1]), 1, "Thermal(3)"),
            (
                qml.GaussianState(np.array([1, 2]), np.array([[2, 0], [0, 2]]), wires=[1]),
                1,
                "Gaussian(M0, M1)",
            ),
            (qml.QuadraticPhase(3.14, wires=[1]), 1, "QuadPhase(3.14)"),
            (qml.RX(3.14, wires=[1]), 1, "RX(3.14)"),
            (qml.S(wires=[2]), 2, "S"),
            (qml.T(wires=[2]), 2, "T"),
            (qml.RX(3.14, wires=[1]), 1, "RX(3.14)"),
            (qml.RY(3.14, wires=[1]), 1, "RY(3.14)"),
            (qml.RZ(3.14, wires=[1]), 1, "RZ(3.14)"),
            (qml.Rot(3.14, 2.14, 1.14, wires=[1]), 1, "Rot(3.14, 2.14, 1.14)"),
            (qml.U1(3.14, wires=[1]), 1, "U1(3.14)"),
            (qml.U2(3.14, 2.14, wires=[1]), 1, "U2(3.14, 2.14)"),
            (qml.U3(3.14, 2.14, 1.14, wires=[1]), 1, "U3(3.14, 2.14, 1.14)"),
            (qml.BasisState(np.array([0, 1, 0]), wires=[1, 2, 3]), 1, "|0>"),
            (qml.BasisState(np.array([0, 1, 0]), wires=[1, 2, 3]), 2, "|1>"),
            (qml.BasisState(np.array([0, 1, 0]), wires=[1, 2, 3]), 3, "|0>"),
            (qml.QubitStateVector(np.array([0, 1, 0, 0]), wires=[1, 2]), 1, "QubitStateVector(M0)"),
            (qml.QubitStateVector(np.array([0, 1, 0, 0]), wires=[1, 2]), 2, "QubitStateVector(M0)"),
            (qml.QubitUnitary(np.eye(2), wires=[1]), 1, "U0"),
            (qml.QubitUnitary(np.eye(4), wires=[1, 2]), 2, "U0"),
            (qml.Kerr(3.14, wires=[1]), 1, "Kerr(3.14)"),
            (qml.CrossKerr(3.14, wires=[1, 2]), 1, "CrossKerr(3.14)"),
            (qml.CrossKerr(3.14, wires=[1, 2]), 2, "CrossKerr(3.14)"),
            (qml.CubicPhase(3.14, wires=[1]), 1, "CubicPhase(3.14)"),
            (qml.Interferometer(np.eye(4), wires=[1, 3]), 1, "Interferometer(M0)"),
            (qml.Interferometer(np.eye(4), wires=[1, 3]), 3, "Interferometer(M0)"),
            (qml.CatState(3.14, 2.14, 1, wires=[1]), 1, "CatState(3.14, 2.14, 1)"),
            (qml.CoherentState(3.14, 2.14, wires=[1]), 1, "CoherentState(3.14, 2.14)"),
            (
                qml.FockDensityMatrix(np.kron(np.eye(4), np.eye(4)), wires=[1, 2]),
                1,
                "FockDensityMatrix(M0)",
            ),
            (
                qml.FockDensityMatrix(np.kron(np.eye(4), np.eye(4)), wires=[1, 2]),
                2,
                "FockDensityMatrix(M0)",
            ),
            (
                qml.DisplacedSqueezedState(3.14, 2.14, 1.14, 0.14, wires=[1]),
                1,
                "DisplacedSqueezedState(3.14, 2.14, 1.14, 0.14)",
            ),
            (qml.FockState(7, wires=[1]), 1, "|7>"),
            (qml.FockStateVector(np.array([4, 5, 7]), wires=[1, 2, 3]), 1, "|4>"),
            (qml.FockStateVector(np.array([4, 5, 7]), wires=[1, 2, 3]), 2, "|5>"),
            (qml.FockStateVector(np.array([4, 5, 7]), wires=[1, 2, 3]), 3, "|7>"),
            (qml.SqueezedState(3.14, 2.14, wires=[1]), 1, "SqueezedState(3.14, 2.14)"),
            (qml.Hermitian(np.eye(4), wires=[1, 2]), 1, "H0"),
            (qml.Hermitian(np.eye(4), wires=[1, 2]), 2, "H0"),
            (qml.X(wires=[1]), 1, "x"),
            (qml.P(wires=[1]), 1, "p"),
            (qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]), 1, "|4, 5, 7X4, 5, 7|"),
            (
                qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1]),
                2,
                "1.0 + 2.0 x_0 - 1.3 x_1 + 6.0 p_1",
            ),
            (
                qml.PolyXP(
                    np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                ),
                1,
                "1.2 + 1.1 x_0 + 3.2 p_0 + 1.2 x_0^2 + 2.3 p_0^2 + 3.0 x_0p_0",
            ),
            (qml.QuadOperator(3.14, wires=[1]), 1, "cos(3.14)x + sin(3.14)p"),
        ],
    )
    def test_operator_representation_ascii(self, ascii_representation_resolver, op, wire, target):
        """Test that an Operator instance is properly resolved."""
        assert ascii_representation_resolver.operator_representation(op, wire) == target

    @pytest.mark.parametrize(
        "obs,wire,target",
        [
            (qml.expval(qml.PauliX(wires=[1])), 1, "⟨X⟩"),
            (qml.expval(qml.PauliY(wires=[1])), 1, "⟨Y⟩"),
            (qml.expval(qml.PauliZ(wires=[1])), 1, "⟨Z⟩"),
            (qml.expval(qml.Hadamard(wires=[1])), 1, "⟨H⟩"),
            (qml.expval(qml.Hermitian(np.eye(4), wires=[1, 2])), 1, "⟨H0⟩"),
            (qml.expval(qml.Hermitian(np.eye(4), wires=[1, 2])), 2, "⟨H0⟩"),
            (qml.expval(qml.NumberOperator(wires=[1])), 1, "⟨n⟩"),
            (qml.expval(qml.X(wires=[1])), 1, "⟨x⟩"),
            (qml.expval(qml.P(wires=[1])), 1, "⟨p⟩"),
            (
                qml.expval(qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3])),
                1,
                "⟨|4, 5, 7╳4, 5, 7|⟩",
            ),
            (
                qml.expval(qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1])),
                2,
                "⟨1.0 + 2.0 x₀ - 1.3 x₁ + 6.0 p₁⟩",
            ),
            (
                qml.expval(
                    qml.PolyXP(
                        np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                    )
                ),
                1,
                "⟨1.2 + 1.1 x₀ + 3.2 p₀ + 1.2 x₀² + 2.3 p₀² + 3.0 x₀p₀⟩",
            ),
            (qml.expval(qml.QuadOperator(3.14, wires=[1])), 1, "⟨cos(3.14)x + sin(3.14)p⟩"),
            (qml.var(qml.PauliX(wires=[1])), 1, "Var[X]"),
            (qml.var(qml.PauliY(wires=[1])), 1, "Var[Y]"),
            (qml.var(qml.PauliZ(wires=[1])), 1, "Var[Z]"),
            (qml.var(qml.Hadamard(wires=[1])), 1, "Var[H]"),
            (qml.var(qml.Hermitian(np.eye(4), wires=[1, 2])), 1, "Var[H0]"),
            (qml.var(qml.Hermitian(np.eye(4), wires=[1, 2])), 2, "Var[H0]"),
            (qml.var(qml.NumberOperator(wires=[1])), 1, "Var[n]"),
            (qml.var(qml.X(wires=[1])), 1, "Var[x]"),
            (qml.var(qml.P(wires=[1])), 1, "Var[p]"),
            (
                qml.var(qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3])),
                1,
                "Var[|4, 5, 7╳4, 5, 7|]",
            ),
            (
                qml.var(qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1])),
                2,
                "Var[1.0 + 2.0 x₀ - 1.3 x₁ + 6.0 p₁]",
            ),
            (
                qml.var(
                    qml.PolyXP(
                        np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                    )
                ),
                1,
                "Var[1.2 + 1.1 x₀ + 3.2 p₀ + 1.2 x₀² + 2.3 p₀² + 3.0 x₀p₀]",
            ),
            (qml.var(qml.QuadOperator(3.14, wires=[1])), 1, "Var[cos(3.14)x + sin(3.14)p]"),
            (qml.sample(qml.PauliX(wires=[1])), 1, "Sample[X]"),
            (qml.sample(qml.PauliY(wires=[1])), 1, "Sample[Y]"),
            (qml.sample(qml.PauliZ(wires=[1])), 1, "Sample[Z]"),
            (qml.sample(qml.Hadamard(wires=[1])), 1, "Sample[H]"),
            (qml.sample(qml.Hermitian(np.eye(4), wires=[1, 2])), 1, "Sample[H0]"),
            (qml.sample(qml.Hermitian(np.eye(4), wires=[1, 2])), 2, "Sample[H0]"),
            (qml.sample(qml.NumberOperator(wires=[1])), 1, "Sample[n]"),
            (qml.sample(qml.X(wires=[1])), 1, "Sample[x]"),
            (qml.sample(qml.P(wires=[1])), 1, "Sample[p]"),
            (
                qml.sample(qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3])),
                1,
                "Sample[|4, 5, 7╳4, 5, 7|]",
            ),
            (
                qml.sample(qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1])),
                2,
                "Sample[1.0 + 2.0 x₀ - 1.3 x₁ + 6.0 p₁]",
            ),
            (
                qml.sample(
                    qml.PolyXP(
                        np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                    )
                ),
                1,
                "Sample[1.2 + 1.1 x₀ + 3.2 p₀ + 1.2 x₀² + 2.3 p₀² + 3.0 x₀p₀]",
            ),
            (qml.sample(qml.QuadOperator(3.14, wires=[1])), 1, "Sample[cos(3.14)x + sin(3.14)p]"),
            (
                qml.expval(qml.PauliX(wires=[1]) @ qml.PauliY(wires=[2]) @ qml.PauliZ(wires=[3])),
                1,
                "⟨X ⊗ Y ⊗ Z⟩",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                1,
                "⟨|4, 5, 7╳4, 5, 7| ⊗ x⟩",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                2,
                "⟨|4, 5, 7╳4, 5, 7| ⊗ x⟩",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                3,
                "⟨|4, 5, 7╳4, 5, 7| ⊗ x⟩",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                4,
                "⟨|4, 5, 7╳4, 5, 7| ⊗ x⟩",
            ),
            (
                qml.sample(
                    qml.Hermitian(np.eye(4), wires=[1, 2]) @ qml.Hermitian(np.eye(4), wires=[0, 3])
                ),
                0,
                "Sample[H0 ⊗ H0]",
            ),
            (
                qml.sample(
                    qml.Hermitian(np.eye(4), wires=[1, 2])
                    @ qml.Hermitian(2 * np.eye(4), wires=[0, 3])
                ),
                0,
                "Sample[H0 ⊗ H1]",
            ),
        ],
    )
    def test_output_representation_unicode(self, unicode_representation_resolver, obs, wire, target):
        """Test that an Observable instance with return type is properly resolved."""
        assert unicode_representation_resolver.output_representation(obs, wire) == target


    @pytest.mark.parametrize(
        "obs,wire,target",
        [
            (qml.expval(qml.PauliX(wires=[1])), 1, "<X>"),
            (qml.expval(qml.PauliY(wires=[1])), 1, "<Y>"),
            (qml.expval(qml.PauliZ(wires=[1])), 1, "<Z>"),
            (qml.expval(qml.Hadamard(wires=[1])), 1, "<H>"),
            (qml.expval(qml.Hermitian(np.eye(4), wires=[1, 2])), 1, "<H0>"),
            (qml.expval(qml.Hermitian(np.eye(4), wires=[1, 2])), 2, "<H0>"),
            (qml.expval(qml.NumberOperator(wires=[1])), 1, "<n>"),
            (qml.expval(qml.X(wires=[1])), 1, "<x>"),
            (qml.expval(qml.P(wires=[1])), 1, "<p>"),
            (
                qml.expval(qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3])),
                1,
                "<|4, 5, 7X4, 5, 7|>",
            ),
            (
                qml.expval(qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1])),
                2,
                "<1.0 + 2.0 x_0 - 1.3 x_1 + 6.0 p_1>",
            ),
            (
                qml.expval(
                    qml.PolyXP(
                        np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                    )
                ),
                1,
                "<1.2 + 1.1 x_0 + 3.2 p_0 + 1.2 x_0^2 + 2.3 p_0^2 + 3.0 x_0p_0>",
            ),
            (qml.expval(qml.QuadOperator(3.14, wires=[1])), 1, "<cos(3.14)x + sin(3.14)p>"),
            (qml.var(qml.PauliX(wires=[1])), 1, "Var[X]"),
            (qml.var(qml.PauliY(wires=[1])), 1, "Var[Y]"),
            (qml.var(qml.PauliZ(wires=[1])), 1, "Var[Z]"),
            (qml.var(qml.Hadamard(wires=[1])), 1, "Var[H]"),
            (qml.var(qml.Hermitian(np.eye(4), wires=[1, 2])), 1, "Var[H0]"),
            (qml.var(qml.Hermitian(np.eye(4), wires=[1, 2])), 2, "Var[H0]"),
            (qml.var(qml.NumberOperator(wires=[1])), 1, "Var[n]"),
            (qml.var(qml.X(wires=[1])), 1, "Var[x]"),
            (qml.var(qml.P(wires=[1])), 1, "Var[p]"),
            (
                qml.var(qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3])),
                1,
                "Var[|4, 5, 7X4, 5, 7|]",
            ),
            (
                qml.var(qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1])),
                2,
                "Var[1.0 + 2.0 x_0 - 1.3 x_1 + 6.0 p_1]",
            ),
            (
                qml.var(
                    qml.PolyXP(
                        np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                    )
                ),
                1,
                "Var[1.2 + 1.1 x_0 + 3.2 p_0 + 1.2 x_0^2 + 2.3 p_0^2 + 3.0 x_0p_0]",
            ),
            (qml.var(qml.QuadOperator(3.14, wires=[1])), 1, "Var[cos(3.14)x + sin(3.14)p]"),
            (qml.sample(qml.PauliX(wires=[1])), 1, "Sample[X]"),
            (qml.sample(qml.PauliY(wires=[1])), 1, "Sample[Y]"),
            (qml.sample(qml.PauliZ(wires=[1])), 1, "Sample[Z]"),
            (qml.sample(qml.Hadamard(wires=[1])), 1, "Sample[H]"),
            (qml.sample(qml.Hermitian(np.eye(4), wires=[1, 2])), 1, "Sample[H0]"),
            (qml.sample(qml.Hermitian(np.eye(4), wires=[1, 2])), 2, "Sample[H0]"),
            (qml.sample(qml.NumberOperator(wires=[1])), 1, "Sample[n]"),
            (qml.sample(qml.X(wires=[1])), 1, "Sample[x]"),
            (qml.sample(qml.P(wires=[1])), 1, "Sample[p]"),
            (
                qml.sample(qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3])),
                1,
                "Sample[|4, 5, 7X4, 5, 7|]",
            ),
            (
                qml.sample(qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1])),
                2,
                "Sample[1.0 + 2.0 x_0 - 1.3 x_1 + 6.0 p_1]",
            ),
            (
                qml.sample(
                    qml.PolyXP(
                        np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                    )
                ),
                1,
                "Sample[1.2 + 1.1 x_0 + 3.2 p_0 + 1.2 x_0^2 + 2.3 p_0^2 + 3.0 x_0p_0]",
            ),
            (qml.sample(qml.QuadOperator(3.14, wires=[1])), 1, "Sample[cos(3.14)x + sin(3.14)p]"),
            (
                qml.expval(qml.PauliX(wires=[1]) @ qml.PauliY(wires=[2]) @ qml.PauliZ(wires=[3])),
                1,
                "<X @ Y @ Z>",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                1,
                "<|4, 5, 7X4, 5, 7| @ x>",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                2,
                "<|4, 5, 7X4, 5, 7| @ x>",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                3,
                "<|4, 5, 7X4, 5, 7| @ x>",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                4,
                "<|4, 5, 7X4, 5, 7| @ x>",
            ),
            (
                qml.sample(
                    qml.Hermitian(np.eye(4), wires=[1, 2]) @ qml.Hermitian(np.eye(4), wires=[0, 3])
                ),
                0,
                "Sample[H0 @ H0]",
            ),
            (
                qml.sample(
                    qml.Hermitian(np.eye(4), wires=[1, 2])
                    @ qml.Hermitian(2 * np.eye(4), wires=[0, 3])
                ),
                0,
                "Sample[H0 @ H1]",
            ),
        ],
    )
    def test_output_representation_ascii(self, ascii_representation_resolver, obs, wire, target):
        """Test that an Observable instance with return type is properly resolved."""
        assert ascii_representation_resolver.output_representation(obs, wire) == target

    def test_element_representation_none(self, unicode_representation_resolver):
        """Test that element_representation properly handles None."""
        assert unicode_representation_resolver.element_representation(None, 0) == ""

    def test_element_representation_str(self, unicode_representation_resolver):
        """Test that element_representation properly handles strings."""
        assert unicode_representation_resolver.element_representation("Test", 0) == "Test"

    def test_element_representation_calls_output(self, unicode_representation_resolver):
        """Test that element_representation calls output_representation for returned observables."""

        unicode_representation_resolver.output_representation = Mock()

        obs = qml.sample(qml.PauliX(3))
        wire = 3

        unicode_representation_resolver.element_representation(obs, wire)

        assert unicode_representation_resolver.output_representation.call_args[0] == (obs, wire)

    def test_element_representation_calls_operator(self, unicode_representation_resolver):
        """Test that element_representation calls operator_representation for all operators that are not returned."""

        unicode_representation_resolver.operator_representation = Mock()

        op = qml.PauliX(3)
        wire = 3

        unicode_representation_resolver.element_representation(op, wire)

        print()

        assert unicode_representation_resolver.operator_representation.call_args[0] == (op, wire)


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

    @pytest.mark.parametrize(
        "grid,target_representation_grid",
        [
            (interlocking_multiwire_gate_grid, interlocking_multiwire_gate_representation_grid),
            (
                multiwire_and_single_wire_gate_grid,
                multiwire_and_single_wire_gate_representation_grid,
            ),
            (all_wire_state_preparation_grid, all_wire_state_preparation_representation_grid),
        ],
    )
    def test_resolve_decorations_separate(
        self, dummy_circuit_drawer, grid, target_representation_grid
    ):
        """Test that decorations are properly resolved."""

        representation_grid = Grid()
        dummy_circuit_drawer.resolve_decorations(grid, representation_grid, True)

        assert_nested_lists_equal(representation_grid.raw_grid, target_representation_grid.raw_grid)

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
            (multiwire_gate_grid, multiwire_gate_representation_grid),
            (multi_and_single_wire_gate_grid, multi_and_single_wire_gate_representation_grid),
        ],
    )
    def test_resolve_decorations_not_separate(
        self, dummy_circuit_drawer, grid, target_representation_grid
    ):
        """Test that decorations are properly resolved."""

        representation_grid = Grid()
        dummy_circuit_drawer.resolve_decorations(grid, representation_grid, False)

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
        qml.RY(17 / 9 * c, wires=2)
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
        + " 2: ──RY(1.889*c)──RX(angles[2])───U0──│────────────────────────────────│───────────╰C─────╰X─────────╰Z──╰C──│───╰X──╰SWAP───│───────┤ Sample[X] \n"
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
        + " 2: ──RY(0.567)──RX(0.6)───U0──│──────────────────│───────────╰C─────╰X─────────╰Z──╰C──│───╰X──╰SWAP─────│───────┤ Sample[X] \n"
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
        " 0: ──╭Gaussian(M0, M1)──╭BS(0.4, 0)───────────────────────────────────────────────────────────╭BS(0.255, 0.231)──╭───┤ ⟨|1, 1╳1, 1|⟩ \n"
        + " 1: ──├Gaussian(M0, M1)──╰BS(0.4, 0)────────────────────────────────────────╭BS(0.255, 0.231)──│──────────────────│╭──┤ ⟨|1, 1╳1, 1|⟩ \n"
        + " 2: ──├Gaussian(M0, M1)──╭BS(0.4, 0)─────────────────────╭BS(0.255, 0.231)──│──────────────────│──────────────────││╭─┤ ⟨|1, 1╳1, 1|⟩ \n"
        + " 3: ──├Gaussian(M0, M1)──╰BS(0.4, 0)──╭BS(0.255, 0.231)──│──────────────────│──────────────────│──────────────────│││╭┤ ⟨|1, 1╳1, 1|⟩ \n"
        + " 4: ──├Gaussian(M0, M1)──╭BS(0.4, 0)──│──────────────────│──────────────────│──────────────────╰BS(0.255, 0.231)──╰│││┤ ⟨|1, 1╳1, 1|⟩ \n"
        + " 5: ──├Gaussian(M0, M1)──╰BS(0.4, 0)──│──────────────────│──────────────────╰BS(0.255, 0.231)──────────────────────╰││┤ ⟨|1, 1╳1, 1|⟩ \n"
        + " 6: ──├Gaussian(M0, M1)──╭BS(0.4, 0)──│──────────────────╰BS(0.255, 0.231)──────────────────────────────────────────╰│┤ ⟨|1, 1╳1, 1|⟩ \n"
        + " 7: ──╰Gaussian(M0, M1)──╰BS(0.4, 0)──╰BS(0.255, 0.231)──────────────────────────────────────────────────────────────╰┤ ⟨|1, 1╳1, 1|⟩ \n"
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
        " 0: ──────────────╭Gaussian(M0, M1)──R(a)─────╭BS(d, 1)───S(2.3, 0)──────────────────────────────────────────────────────╭C───────QuadPhase(4)───┤ ⟨ + 1.0 x₀ + 2.0 p₀⟩ \n"
        + " 1: ──Thermal(3)──├Gaussian(M0, M1)──R(b)─────╰BS(d, 1)──╭BS(e, 1)──────────────╭BS(d, 1)─────────────╭S(2, 2)──╭R(2.3)──│───────────────────────┤ ⟨cos(4)x + sin(4)p⟩  \n"
        + " 2: ──────────────├Gaussian(M0, M1)──────────────────────╰BS(e, 1)───S(2.3, 0)──╰BS(d, 1)──╭BS(e, 1)──│─────────╰C───────│──────────────────────╭┤ ⟨|1, 5╳1, 5|⟩        \n"
        + " 3: ──────────────╰Gaussian(M0, M1)──D(f, 0)───────────────────────────────────────────────╰BS(e, 1)──╰S(2, 2)───────────╰Add(2)────────────────╰┤ ⟨|1, 5╳1, 5|⟩        \n"
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
        " 0: ──────────────╭Gaussian(M0, M1)──R(0.1)─────╭BS(2.765, 1)───S(2.3, 0)─────────────────────────────────────────────────────────────╭C───────QuadPhase(4)───┤ ⟨ + 1.0 x₀ + 2.0 p₀⟩ \n"
        + " 1: ──Thermal(3)──├Gaussian(M0, M1)──R(0.2)─────╰BS(2.765, 1)──╭BS(0.5, 1)─────────────╭BS(2.765, 1)───────────────╭S(2, 2)──╭R(2.3)──│───────────────────────┤ ⟨cos(4)x + sin(4)p⟩  \n"
        + " 2: ──────────────├Gaussian(M0, M1)────────────────────────────╰BS(0.5, 1)──S(2.3, 0)──╰BS(2.765, 1)──╭BS(0.5, 1)──│─────────╰C───────│──────────────────────╭┤ ⟨|1, 5╳1, 5|⟩        \n"
        + " 3: ──────────────╰Gaussian(M0, M1)──D(0.6, 0)────────────────────────────────────────────────────────╰BS(0.5, 1)──╰S(2, 2)───────────╰Add(2)────────────────╰┤ ⟨|1, 5╳1, 5|⟩        \n"
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
