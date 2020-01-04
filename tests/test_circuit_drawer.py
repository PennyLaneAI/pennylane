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

import pytest
import numpy as np

import pennylane as qml
from pennylane.circuit_drawer import _transpose, Grid, RepresentationResolver, CircuitDrawer


@pytest.fixture
def parameterized_qubit_circuit():
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

    return qfunc


@pytest.fixture
def parameterized_wide_qubit_circuit():
    def qfunc(a, b, c, d, e, f):
        qml.RX(a, wires=0)
        qml.RX(b, wires=1)
        [qml.CNOT(wires=[2 * i, 2 * i + 1]) for i in range(4)]
        [qml.CNOT(wires=[i, i + 4]) for i in range(4)]
        [qml.CSWAP(wires=[i + 2, i, i + 4]) for i in range(4)]
        qml.RX(a, wires=0)
        qml.RX(b, wires=1)

        return [qml.expval(qml.Hermitian(np.eye(4), wires=[i, i + 4])) for i in range(4)]

    return qfunc


@pytest.fixture
def parameterized_wide_cv_circuit():
    def qfunc(a, b, c, d, e, f):
        qml.GaussianState(
            np.array([(2 * i + 2) // 2 for i in range(16)]), 2 * np.eye(16), wires=list(range(8))
        )
        [qml.Beamsplitter(0.4, 0, wires=[2 * i, 2 * i + 1]) for i in range(4)]
        [qml.Beamsplitter(0.25475, 0.2312344, wires=[i, i + 4]) for i in range(4)]

        return [
            qml.expval(qml.FockStateProjector(np.array([1, 1]), wires=[i, i + 4])) for i in range(4)
        ]

    return qfunc


@pytest.fixture
def parameterized_cv_circuit():
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
        # qml.Kerr(2, wires=[1])
        # qml.CubicPhase(2, wires=[2])
        # qml.CrossKerr(2, wires=[3, 1])

        return [
            qml.expval(qml.ops.PolyXP(np.array([0, 1, 2]), wires=0)),
            qml.expval(qml.ops.QuadOperator(4, wires=1)),
            qml.expval(qml.ops.FockStateProjector(np.array([1, 5]), wires=[2, 3])),
        ]

    return qfunc


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
    return RepresentationResolver()

class TestRepresentationResolver:
    """Test the RepresentationResolver class."""

    @pytest.mark.parametrize("par,expected", [
        (3, "3"),
        (5.236422, "5.236"),
    ])
    def test_single_parameter_representation(self, unicode_representation_resolver, par, expected):
        """Test that single parameters are properly resolved."""
        assert unicode_representation_resolver.single_parameter_representation(par) == expected


class TestCircuitGraphDrawing:
    def test_simple_circuit(self, parameterized_qubit_circuit):
        """A test of the different layers, their successors and ancestors using a simple circuit"""

        dev = qml.device("default.qubit", wires=5)
        qnode = qml.QNode(parameterized_qubit_circuit, dev)
        qnode._construct((0.1, 0.2, 0.3, np.array([0.4, 0.5, 0.6])), {})
        print(qnode.circuit.draw(show_variable_names=True))
        qnode.evaluate((0.1, 0.2, 0.3, np.array([47 / 17, 0.5, 0.6])), {})
        print(qnode.draw())

        raise Exception()

    def test_wide_circuit(self, parameterized_wide_qubit_circuit):
        """A test of the different layers, their successors and ancestors using a simple circuit"""

        dev = qml.device("default.qubit", wires=8)
        qnode = qml.QNode(parameterized_wide_qubit_circuit, dev)
        qnode._construct((0.1, 0.2, 0.3, 0.4, 0.5, 0.6), {})
        print(qnode.circuit.draw(show_variable_names=True))
        qnode.evaluate((0.1, 0.2, 0.3, 47 / 17, 0.5, 0.6), {})
        print(qnode.circuit.draw())

        raise Exception()

    def test_simple_cv_circuit(self, parameterized_cv_circuit):
        """A test of the different layers, their successors and ancestors using a simple circuit"""

        dev = qml.device("default.gaussian", wires=4)
        qnode = qml.QNode(parameterized_cv_circuit, dev)
        qnode._construct((0.1, 0.2, 0.3, 0.4, 0.5, 0.6), {})
        print(qnode.circuit.draw())
        qnode.evaluate((0.1, 0.2, 0.3, 47 / 17, 0.5, 0.6), {})
        print(qnode.circuit.draw())

        raise Exception()

    def test_wide_cv_circuit(self, parameterized_wide_cv_circuit):
        """A test of the different layers, their successors and ancestors using a simple circuit"""

        dev = qml.device("default.gaussian", wires=8)
        qnode = qml.QNode(parameterized_wide_cv_circuit, dev)
        qnode._construct((0.1, 0.2, 0.3, 0.4, 0.5, 0.6), {})
        print(qnode.circuit.draw())
        qnode.evaluate((0.1, 0.2, 0.3, 47 / 17, 0.5, 0.6), {})
        print(qnode.circuit.draw())

    def test_template(self, parameterized_wide_cv_circuit):
        """A test of the different layers, their successors and ancestors using a simple circuit"""

        dev = qml.device("default.qubit", wires=8)

        @qml.qnode(dev)
        def circuit(a, b, weights, c, d, other_weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=range(8))
            qml.RX(a, wires=[0])
            qml.RX(b, wires=[1])
            qml.RX(c, wires=[2])
            qml.RX(d, wires=[3])
            [qml.RX(other_weights[i], wires=[i]) for i, weight in enumerate(other_weights)]

            return [qml.var(qml.PauliX(i)) for i in range(8)]

        weights = qml.init.strong_ent_layers_uniform(3, 8)

        circuit._construct((2, 3, weights, 1, 33, np.array([1, 3, 4, 2, 2, 2, 3, 4])), {})

        print(circuit.draw(show_variable_names=True))

        circuit(2, 3, weights, 1, 33, np.array([1, 3, 4, 2, 2, 2, 3, 4]))

        print(circuit.draw())
        circuit.print_applied()

        raise Exception()
