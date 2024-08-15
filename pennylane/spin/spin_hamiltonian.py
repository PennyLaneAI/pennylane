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

import pennylane as qml
from pennylane import X, Y, Z
from pennylane.fermi import FermiWord

from .lattice_shapes import Chain, Square, Rectangle, Honeycomb, Triangle

# pylint: disable=too-many-arguments


def generate_lattice(lattice, n_cells, boundary_condition, neighbour_order):
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
        lattice_shape = Honeycomb(n_cells, boundary_condition, neighbour_order)
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
    hamiltonian = 0.0
    if isinstance(coupling, (int, float, complex)):
        for edge in lattice.edges:
            i, j = edge[0], edge[1]
            hamiltonian += -coupling * (Z(i) @ Z(j))
    else:
        for edge in lattice.edges:
            i, j = edge[0], edge[1]
            hamiltonian += -coupling[i][j] * (Z(i) @ Z(j))

    for vertex in range(lattice.n_sites):
        hamiltonian += -h * X(vertex)

    return hamiltonian


def heisenberg(lattice, n_cells, coupling, boundary_condition=False, neighbour_order=1):
    r"""Generates the Heisenberg model on a lattice.
    The Hamiltonian is represented as:
    .. math::

         \hat{H} = J\sum_{<i,j>}(\sigma_i^x\sigma_j^x + \sigma_i^y\sigma_j^y + \sigma_i^z\sigma_j^z)

    where J is the coupling constant defined for the Hamiltonian, and i,j represent the indices for neighbouring spins.

    Args:
       lattice: Shape of the lattice. Input Values can be ``'Chain'``, ``'Square'``, ``'Rectangle'``, ``'Honeycomb'``, or ``'Triangle'``.
       n_cells: A list containing umber of unit cells in each direction.
       coupling: Coupling between spins, it can be a 1D array of length 3  or 3D array of shape 3 * number of spins * number of spins.
       boundary_condition: defines boundary conditions, boolean or series of bools with dimensions same as L.
       neighbour_order: Range of neighbours a spin interacts with.

    Returns:
       pennylane.LinearCombination: Hamiltonian for the heisenberg model.
    """

    lattice = generate_lattice(lattice, n_cells, boundary_condition, neighbour_order)

    hamiltonian = 0.0
    if isinstance(coupling[0], (int, float, complex)):
        for edge in lattice.edges:
            i, j = edge[0], edge[1]
            hamiltonian += (
                coupling[0] * X(i) @ X(j) + coupling[1] * Y(i) @ Y(j) + coupling[2] * Z(i) @ Z(j)
            )
    else:
        for edge in lattice.edges:
            i, j = edge[0], edge[1]
            hamiltonian += (
                coupling[0][i][j] * X(i) @ X(j)
                + coupling[1][i][j] * Y(i) @ Y(j)
                + coupling[2][i][j] * Z(i) @ Z(j)
            )

    return hamiltonian


def fermihubbard(
    lattice,
    n_cells,
    hopping,
    interaction,
    boundary_condition=False,
    neighbour_order=1,
    mapping="jordan_wigner",
):
    r"""Generates the Hubbard model on a lattice.
    The Hamiltonian is represented as:
    .. math::

        \hat{H} = -t\sum_{<i,j>, \sigma}(c_{i\sigma}^{\dagger}c_{j\sigma}) + U\sum_{i}n_{i \uparrow} n_{i\downarrow}

    where t is the hopping term representing the kinetic energy of electrons, and U is the on-site Coulomb interaction, representing the repulsion between electrons.

    Args:
       lattice: Shape of the lattice. Input Values can be ``'Chain'``, ``'Square'``, ``'Rectangle'``, ``'Honeycomb'``, or ``'Triangle'``.
       n_cells: A list containing umber of unit cells in each direction.
       hopping: hopping between spins, it can be a constant or a 2D array of shape number of spins * number of spins.
       interaction: Coulomb interaction between spins, it can be constant or a 2D array of shape number of spins*number of spins.
       boundary_condition: defines boundary conditions, boolean or series of bools with dimensions same as L.
       neighbour_order: Range of neighbours a spin interacts with.

    Returns:
       pennylane.operator: Hamiltonian for the Fermi-Hubbard model.

    """
    lattice = generate_lattice(lattice, n_cells, boundary_condition, neighbour_order)

    hamiltonian = 0.0
    if isinstance(hopping, (int, float, complex)):
        for edge in lattice.edges:
            i, j = edge[0], edge[1]
            hopping_term = -hopping * (
                FermiWord({(0, i): "+", (1, j): "-"}) + FermiWord({(0, j): "+", (1, i): "-"})
            )
            int_term = interaction * FermiWord({(0, i): "+", (1, i): "-", (2, j): "+", (3, j): "-"})
            hamiltonian += hopping_term + int_term
    else:
        for edge in lattice.edges:
            i, j = edge[0], edge[1]
            hopping_term = -hopping[i][j] * (
                FermiWord({(0, i): "+", (1, j): "-"}) + FermiWord({(0, j): "+", (1, i): "-"})
            )
            int_term = interaction[i][j] * FermiWord(
                {(0, i): "+", (1, i): "-", (2, j): "+", (3, j): "-"}
            )
            hamiltonian += hopping_term + int_term

    if mapping not in ["jordan_wigner", "parity", "bravyi_kitaev"]:
        raise ValueError(
            f"The '{mapping}' transformation is not available."
            f"Please set mapping to 'jordan_wigner', 'parity', or 'bravyi_kitaev'"
        )
    qubit_ham = qml.qchem.qubit_observable(hamiltonian, mapping=mapping)

    return qubit_ham.simplify()
