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
This module contains functions to create different templates of spin Hamiltonians.
"""

from pennylane import X, Z
from pennylane import numpy as np

from .lattice_shapes import Chain, Honeycomb, Rectangle, Square, Triangle

# pylint: disable=too-many-arguments


def generate_lattice(lattice, n_cells, boundary_condition=False, neighbour_order=1):
    r"""Generates the lattice object for given attributes."""

    lattice_shape = lattice.strip().lower()

    if lattice_shape not in ["chain", "square", "rectangle", "honeycomb", "triangle"]:
        raise ValueError(
            f"Lattice shape, '{lattice}' is not supported."
            f"Please set lattice to: chain, square, rectangle, honeycomb, or triangle"
        )

    if lattice_shape == "chain":
        lattice = Chain(n_cells, boundary_condition, neighbour_order)
    elif lattice_shape == "square":
        lattice = Square(n_cells, boundary_condition, neighbour_order)
    elif lattice_shape == "rectangle":
        lattice = Rectangle(n_cells, boundary_condition, neighbour_order)
    elif lattice_shape == "honeycomb":
        lattice = Honeycomb(n_cells, boundary_condition, neighbour_order)
    elif lattice_shape == "triangle":
        lattice = Triangle(n_cells, boundary_condition, neighbour_order)

    return lattice


def transverse_ising(
    lattice, n_cells, coupling, h=1.0, boundary_condition=False, neighbour_order=1
):
    r"""Generates the transverse field Ising model on a lattice.
    The Hamiltonian is represented as:
    .. math::

        \hat{H} =  -J \sum_{<i,j>} \sigma_i^{z} \sigma_j^{z} - h\sum{i} \sigma_{i}^{x}

    where J is the coupling defined for the Hamiltonian, h is the strength of transverse
    magnetic field and i,j represent the indices for neighbouring spins.

    Args:
       lattice: Shape of the lattice. Input Values can be ``'Chain'``, ``'Square'``, ``'Rectangle'``, ``'Honeycomb'``, or ``'Triangle'``.
       n_cells: A list containing umber of unit cells in each direction.
       coupling: Coupling between spins, it can be a constant or a 2D array of shape number of spins * number of spins.
       h: Value of external magnetic field.
       boundary_condition: defines boundary conditions, boolean or series of bools with dimensions same as L.
       neighbour_order: Range of neighbours a spin interacts with.

    Returns:
       pennylane.LinearCombination: Hamiltonian for the transverse-field ising model.
    """
    lattice = generate_lattice(lattice, n_cells, boundary_condition, neighbour_order)
    coupling = np.asarray(coupling)
    hamiltonian = 0.0
    print(coupling.shape)
    if coupling.shape not in [(neighbour_order,), (lattice.n_sites, lattice.n_sites)]:
        raise ValueError(
            f"Coupling shape should be equal to {neighbour_order} or {lattice.n_sites}x{lattice.n_sites}"
        )

    if coupling.shape == (neighbour_order,):
        for edge in lattice.edges:
            i, j, order = edge[0], edge[1], edge[2]
            hamiltonian += -coupling[order] * (Z(i) @ Z(j))
    else:
        for edge in lattice.edges:
            i, j = edge[0], edge[1]
            hamiltonian += -coupling[i][j] * (Z(i) @ Z(j))

    for vertex in range(lattice.n_sites):
        hamiltonian += -h * X(vertex)

    return hamiltonian
