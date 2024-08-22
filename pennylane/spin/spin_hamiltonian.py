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
This file contains functions to create spin Hamiltonians.
"""

from pennylane import X, Z, math

from .lattice import _generate_lattice

# pylint: disable=too-many-arguments


def transverse_ising(
    lattice, n_cells, coupling=None, h=1.0, boundary_condition=False, neighbour_order=1
):
    r"""Generates the transverse-field Ising model on a lattice.

    The Hamiltonian is represented as:

    .. math::

        \hat{H} =  -J \sum_{<i,j>} \sigma_i^{z} \sigma_j^{z} - h\sum_{i} \sigma_{i}^{x}

    where ``J`` is the coupling parameter defined for the Hamiltonian, ``h`` is the strength of the
    transverse magnetic field and ``i,j`` represent the indices for neighbouring spins.

    Args:
       lattice (str): Shape of the lattice. Input Values can be ``'chain'``, ``'square'``,
           ``'rectangle'``, ``'honeycomb'``, ``'triangle'``, or ``'kagome'``.
       n_cells (list[int]): Number of cells in each direction of the grid.
       coupling (float or List[float] or List[math.array[float]]): Coupling between spins, it can be a
           list of length equal to ``neighbour_order`` or a square matrix of size
           ``(num_spins,  num_spins)``. Default value is [1.0].
       h (float): Value of external magnetic field. Default is 1.0.
       boundary_condition (bool or list[bool]): Defines boundary conditions different lattice axes,
           default is ``False`` indicating open boundary condition.
       neighbour_order (int): Specifies the interaction level for neighbors within the lattice.
           Default is 1 (nearest neighbour).

    Returns:
       pennylane.LinearCombination: Hamiltonian for the transverse-field ising model.

    **Example**

    >>> n_cells = [2,2]
    >>> J = 0.5
    >>> h = 0.1
    >>> spin_ham = transverse_ising("Square", n_cells, coupling=J, h=h)
    >>> spin_ham
    -0.5 * (Z(0) @ Z(1))
    + -0.5 * (Z(0) @ Z(2))
    + -0.5 * (Z(1) @ Z(3))
    + -0.5 * (Z(2) @ Z(3))
    + -0.1 * X(0) + -0.1 * X(1)
    + -0.1 * X(2) + -0.1 * X(3)

    """
    lattice = _generate_lattice(lattice, n_cells, boundary_condition, neighbour_order)
    if coupling is None:
        coupling = [1.0]
    elif isinstance(coupling, (int, float, complex)):
        coupling = [coupling]
    coupling = math.asarray(coupling)

    hamiltonian = 0.0

    if coupling.shape not in [(neighbour_order,), (lattice.n_sites, lattice.n_sites)]:
        raise ValueError(
            f"Coupling should be a number or an array of shape {neighbour_order}x1 or {lattice.n_sites}x{lattice.n_sites}"
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
