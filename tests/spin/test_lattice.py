# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Unit tests for functions needed for computing the lattice.
"""
import pytest

from pennylane import numpy as np
from pennylane.spin import Lattice

# pylint: disable=too-many-arguments, too-many-instance-attributes


def test_boundary_condition_dimension_error():
    r"""Test that an error is raised if a wrong dimensions are entered for boundary_condition."""
    unit_cell = [[1]]
    L = [10]
    with pytest.raises(ValueError, match="Argument 'boundary_condition' must be a bool"):
        Lattice(L=L, unit_cell=unit_cell, boundary_condition=[True, True])


def test_boundary_condition_type_error():
    r"""Test that an error is raised if a wrong type is entered for boundary_condition."""
    unit_cell = [[1]]
    L = [10]
    with pytest.raises(ValueError, match="Argument 'boundary_condition' must be a bool"):
        Lattice(L=L, unit_cell=unit_cell, boundary_condition=[4])


def test_unit_cell_error():
    r"""Test that an error is raised if a wrong dimension is entered for unit_cell."""
    unit_cell = [0, 1]
    L = [2, 2]
    with pytest.raises(
        ValueError, match="'unit_cell' must have ndim==2, as array of primitive vectors."
    ):
        Lattice(L=L, unit_cell=unit_cell)


def test_basis_error():
    r"""Test that an error is raised if a wrong dimension is entered for basis."""
    unit_cell = [[0, 1], [1, 0]]
    L = [2, 2]
    basis = [0, 0]
    with pytest.raises(
        ValueError, match="'basis' must have ndim==2, as array of initial coordinates."
    ):
        Lattice(L=L, unit_cell=unit_cell, basis=basis)


def test_unit_cell_shape_error():
    r"""Test that an error is raised if a wrong dimension is entered for unit_cell."""
    unit_cell = [[0, 1, 2], [0, 1, 1]]
    L = [2, 2]
    with pytest.raises(ValueError, match="The number of primitive vectors must match their length"):
        Lattice(L=L, unit_cell=unit_cell)


def test_L_error():
    r"""Test that an error is raised if length of unit_cell is provided in negative."""

    unit_cell = [[0, 1], [1, 0]]
    L = [2, -2]
    with pytest.raises(TypeError, match="Argument `L` must be a list of positive integers"):
        Lattice(L=L, unit_cell=unit_cell)


def test_L_type_error():
    r"""Test that an error is raised if length of unit_cell is provided not as an int."""

    unit_cell = [[0, 1], [1, 0]]
    L = [2, 2.4]
    with pytest.raises(TypeError, match="Argument `L` must be a list of positive integers"):
        Lattice(L=L, unit_cell=unit_cell)


@pytest.mark.parametrize(
    ("unit_cell", "basis", "L"),
    [
        ([[0, 1], [1, 0]], [[1.5, 1.5]], [3, 3]),
        ([[0, 1], [1, 0]], [[-1, -1]], [3, 3]),
        ([[0, 1], [1, 0]], [[10, 10]], [3, 3]),
        ([[1, 0], [0.5, np.sqrt(3) / 2]], [[0.5, 0.5 / 3**0.5], [1, 1 / 3**0.5]], [2, 2]),
    ],
)
def test_basis(unit_cell, basis, L):
    r"""Test that the lattice points start from the coordinates provided in the basis"""

    lattice = Lattice(L=L, unit_cell=unit_cell, basis=basis)
    for i, b in enumerate(basis):
        assert np.allclose(b, lattice.lattice_points[i])


@pytest.mark.parametrize(
    ("unit_cell", "basis", "L", "expected_number"),
    # expected_number here was obtained manually
    [
        ([[0, 1], [1, 0]], [[0, 0]], [3, 3], 9),
        ([[0, 1], [1, 0]], [[0, 0]], [6, 7], 42),
        ([[1, 0], [0.5, np.sqrt(3) / 2]], [[0.5, 0.5 / 3**0.5], [1, 1 / 3**0.5]], [2, 2], 8),
        (np.eye(3), None, [3, 3, 4], 36),
    ],
)
def test_lattice_points(unit_cell, basis, L, expected_number):
    r"""Test that the correct number of lattice points are generated for the given attributes"""
    lattice = Lattice(L=L, unit_cell=unit_cell, basis=basis)
    assert len(lattice.lattice_points == expected_number)


@pytest.mark.parametrize(
    # expected_edges here were obtained with netket.
    ("unit_cell", "basis", "L", "boundary_condition", "expected_edges"),
    [
        (
            [[0, 1], [1, 0]],
            [[0, 0]],
            [3, 3],
            [True, True],
            [
                (0, 1, 0),
                (1, 2, 0),
                (3, 4, 0),
                (5, 8, 0),
                (6, 8, 0),
                (0, 3, 0),
                (1, 4, 0),
                (0, 6, 0),
                (4, 7, 0),
                (6, 7, 0),
                (1, 7, 0),
                (0, 2, 0),
                (4, 5, 0),
                (3, 6, 0),
                (2, 5, 0),
                (3, 5, 0),
                (7, 8, 0),
                (2, 8, 0),
            ],
        ),
        (
            [[0, 1], [1, 0]],
            [[0, 0]],
            [3, 4],
            [True, False],
            [
                (3, 7, 0),
                (8, 9, 0),
                (0, 8, 0),
                (1, 9, 0),
                (4, 5, 0),
                (5, 6, 0),
                (4, 8, 0),
                (5, 9, 0),
                (9, 10, 0),
                (0, 1, 0),
                (10, 11, 0),
                (0, 4, 0),
                (1, 2, 0),
                (1, 5, 0),
                (2, 10, 0),
                (6, 7, 0),
                (6, 10, 0),
                (3, 11, 0),
                (2, 3, 0),
                (2, 6, 0),
                (7, 11, 0),
            ],
        ),
        (
            [[1, 0], [0.5, np.sqrt(3) / 2]],
            [[0.5, 0.5 / 3**0.5], [1, 1 / 3**0.5]],
            [2, 2],
            True,
            [
                (0, 1, 0),
                (1, 2, 0),
                (2, 7, 0),
                (0, 3, 0),
                (1, 4, 0),
                (2, 3, 0),
                (6, 7, 0),
                (4, 5, 0),
                (5, 6, 0),
                (0, 5, 0),
                (3, 6, 0),
                (4, 7, 0),
            ],
        ),
    ],
)
def test_boundary_condition(unit_cell, basis, L, boundary_condition, expected_edges):
    r"""Test that the correct edges are obtained for given boundary conditions"""
    lattice = Lattice(L=L, unit_cell=unit_cell, basis=basis, boundary_condition=boundary_condition)
    assert sorted(lattice.edges) == sorted(expected_edges)


@pytest.mark.parametrize(
    # expected_edges here were obtained with netket.
    ("unit_cell", "basis", "L", "neighbour_order", "expected_edges"),
    [
        (
            [[0, 1], [1, 0]],
            [[0, 0]],
            [3, 3],
            2,
            [
                (0, 1, 0),
                (1, 2, 0),
                (3, 4, 0),
                (5, 8, 0),
                (0, 3, 0),
                (1, 4, 0),
                (6, 7, 0),
                (4, 5, 0),
                (3, 6, 0),
                (2, 5, 0),
                (4, 7, 0),
                (7, 8, 0),
                (2, 4, 1),
                (0, 4, 1),
                (1, 5, 1),
                (3, 7, 1),
                (4, 6, 1),
                (5, 7, 1),
                (4, 8, 1),
                (1, 3, 1),
            ],
        ),
        (
            [[0, 1], [1, 0]],
            [[0, 0]],
            [3, 4],
            3,
            [
                (0, 1, 0),
                (9, 10, 0),
                (1, 2, 0),
                (0, 4, 0),
                (10, 11, 0),
                (1, 5, 0),
                (3, 7, 0),
                (2, 3, 0),
                (6, 7, 0),
                (4, 5, 0),
                (8, 9, 0),
                (2, 6, 0),
                (5, 6, 0),
                (4, 8, 0),
                (6, 10, 0),
                (5, 9, 0),
                (7, 11, 0),
                (2, 7, 1),
                (5, 8, 1),
                (4, 9, 1),
                (6, 11, 1),
                (7, 10, 1),
                (1, 4, 1),
                (5, 10, 1),
                (0, 5, 1),
                (3, 6, 1),
                (1, 6, 1),
                (2, 5, 1),
                (6, 9, 1),
                (2, 10, 2),
                (4, 6, 2),
                (8, 10, 2),
                (5, 7, 2),
                (0, 2, 2),
                (9, 11, 2),
                (0, 8, 2),
                (1, 3, 2),
                (1, 9, 2),
                (3, 11, 2),
            ],
        ),
        ([[1, 0], [0.5, np.sqrt(3) / 2]], [[0.5, 0.5 / 3**0.5], [1, 1 / 3**0.5]], [2, 2], 0, []),
    ],
)
def test_neighbour_order(unit_cell, basis, L, neighbour_order, expected_edges):
    r"""Test that the correct edges are obtained for given neighbour order"""
    lattice = Lattice(L=L, unit_cell=unit_cell, basis=basis, neighbour_order=neighbour_order)
    assert sorted(lattice.edges) == sorted(expected_edges)


@pytest.mark.parametrize(
    ("unit_cell", "basis", "L", "boundary_condition", "n_dim", "expected_bc"),
    [
        ([[0, 1], [1, 0]], [[1.5, 1.5]], [3, 3], True, 2, [True, True]),
        ([[0, 1], [1, 0]], [[-1, -1]], [3, 3], False, 2, [False, False]),
        ([[0, 1], [1, 0]], [[10, 10]], [3, 3], [True, False], 2, [True, False]),
        (
            [[1, 0], [0.5, np.sqrt(3) / 2]],
            [[0.5, 0.5 / 3**0.5], [1, 1 / 3**0.5]],
            [2, 2],
            [True, True],
            2,
            [True, True],
        ),
        (np.eye(3), [[0, 0, 0]], [3, 3, 4], True, 3, [True, True, True]),
    ],
)
def test_attributes(unit_cell, basis, L, boundary_condition, n_dim, expected_bc):
    r"""Test that the methods and attributes return correct values"""
    lattice = Lattice(L=L, unit_cell=unit_cell, basis=basis, boundary_condition=boundary_condition)

    assert np.all(lattice.unit_cell == unit_cell)
    assert np.all(lattice.basis == basis)
    assert lattice.n_dim == n_dim
    assert np.all(lattice.boundary_condition == expected_bc)


def test_add_edge_error():
    r"""Test that an error is raised if the added edge is already present for a lattice"""
    edge_indices = [[4, 5]]
    unit_cell = [[0, 1], [1, 0]]
    L = [3, 3]
    lattice = Lattice(L=L, unit_cell=unit_cell)

    with pytest.raises(ValueError, match="Edge is already present"):
        lattice.add_edge(edge_indices)


def test_add_edge():
    r"""Test that edges are added per their index to a lattice"""
    edge_indices = [[1, 3], [4, 6]]
    unit_cell = [[0, 1], [1, 0]]
    L = [3, 3]
    lattice = Lattice(L=L, unit_cell=unit_cell)

    lattice.add_edge(edge_indices)
    assert np.all(np.isin(edge_indices, lattice.edges))
