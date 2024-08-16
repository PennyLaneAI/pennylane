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
This module contains functions to create 
:class:`~pennylane.spin.lattice` objects of different shapes.
"""

from pennylane import numpy as np
from .lattice import Lattice


def Chain(n_cells, boundary_condition=False, neighbour_order=1):
    r"""Generates a chain lattice"""
    unit_cell = [[1]]
    L = n_cells[0:1]
    lattice_chain = Lattice(
        L,
        unit_cell=unit_cell,
        neighbour_order=neighbour_order,
        boundary_condition=boundary_condition,
    )
    return lattice_chain


def Square(n_cells, boundary_condition=False, neighbour_order=1):
    r"""Generates a square lattice"""
    unit_cell = [[1, 0], [0, 1]]
    basis = [[0, 0]]
    L = n_cells[0:2]
    lattice_square = Lattice(
        L=L,
        unit_cell=unit_cell,
        basis=basis,
        neighbour_order=neighbour_order,
        boundary_condition=boundary_condition,
    )

    return lattice_square


def Rectangle(n_cells, boundary_condition=False, neighbour_order=1):
    r"""Generates a rectangle lattice"""
    unit_cell = [[1, 0], [0, 1]]
    basis = [[0, 0]]

    L = n_cells[0:2]
    lattice_rec = Lattice(
        L=L,
        unit_cell=unit_cell,
        basis=basis,
        neighbour_order=neighbour_order,
        boundary_condition=boundary_condition,
    )

    return lattice_rec


def Honeycomb(n_cells, boundary_condition=False, neighbour_order=1):
    r"""Generates a honeycomb lattice"""
    unit_cell = [[1, 0], [0.5, np.sqrt(3) / 2]]
    basis = [[0.5, 0.5 / 3**0.5], [1, 1 / 3**0.5]]

    L = n_cells[0:2]
    lattice_honeycomb = Lattice(
        L=L,
        unit_cell=unit_cell,
        basis=basis,
        neighbour_order=neighbour_order,
        boundary_condition=boundary_condition,
    )

    return lattice_honeycomb


def Triangle(n_cells, boundary_condition=False, neighbour_order=1):
    r"""Generates a triangular lattice"""
    unit_cell = [[1, 0], [0.5, np.sqrt(3) / 2]]
    basis = [[0, 0]]

    L = n_cells[0:2]
    lattice_triangle = Lattice(
        L=L,
        unit_cell=unit_cell,
        basis=basis,
        neighbour_order=neighbour_order,
        boundary_condition=boundary_condition,
    )

    return lattice_triangle
