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

import pennylane as qml
from pennylane import X, Y, Z, math
from pennylane.fermi import FermiWord

from .lattice import _generate_lattice

# pylint: disable=too-many-arguments


def transverse_ising(
    lattice, n_cells, coupling=1.0, h=1.0, boundary_condition=False, neighbour_order=1
):
    r"""Generates the Hamiltonian for the transverse-field Ising model on a lattice.

    The Hamiltonian is represented as:

    .. math::

        \hat{H} =  -J \sum_{<i,j>} \sigma_i^{z} \sigma_j^{z} - h\sum_{i} \sigma_{i}^{x}

    where ``J`` is the coupling parameter defined for the Hamiltonian, ``h`` is the strength of the
    transverse magnetic field and ``i,j`` represent the indices for neighbouring spins.

    Args:
        lattice (str): Shape of the lattice. Input values can be ``'chain'``, ``'square'``,
            ``'rectangle'``, ``'honeycomb'``, ``'triangle'``, or ``'kagome'``.
        n_cells (List[int]): Number of cells in each direction of the grid.
        coupling (float or List[float] or List[math.array[float]]): Coupling between spins. It can
            be a number, a list of length equal to ``neighbour_order`` or a square matrix of shape
            ``(num_spins,  num_spins)``, where ``num_spins`` is the total number of spins. Default
            value is 1.0.
        h (float): Value of external magnetic field. Default is 1.0.
        boundary_condition (bool or list[bool]): Defines boundary conditions for different lattice
            axes, default is ``False`` indicating open boundary condition.
        neighbour_order (int): Specifies the interaction level for neighbors within the lattice.
            Default is 1, indicating nearest neighbours.

    Returns:
        ~ops.op_math.Sum: Hamiltonian for the transverse-field Ising model.

    **Example**

    >>> n_cells = [2,2]
    >>> j = 0.5
    >>> h = 0.1
    >>> spin_ham = qml.spin.transverse_ising("square", n_cells, coupling=j, h=h)
    >>> spin_ham
    (
    -0.5 * (Z(0) @ Z(1))
    + -0.5 * (Z(0) @ Z(2))
    + -0.5 * (Z(1) @ Z(3))
    + -0.5 * (Z(2) @ Z(3))
    + -0.1 * X(0)
    + -0.1 * X(1)
    + -0.1 * X(2)
    + -0.1 * X(3)
    )

    """
    lattice = _generate_lattice(lattice, n_cells, boundary_condition, neighbour_order)

    if isinstance(coupling, (int, float, complex)):
        coupling = [coupling]
    coupling = math.asarray(coupling)

    hamiltonian = 0.0 * qml.I(0)

    if coupling.shape not in [(neighbour_order,), (lattice.n_sites, lattice.n_sites)]:
        raise ValueError(
            f"The coupling parameter should be a number or an array of shape ({neighbour_order},) or ({lattice.n_sites},{lattice.n_sites})"
        )

    if coupling.shape == (neighbour_order,):
        for edge in lattice.edges:
            i, j, order = edge
            hamiltonian += -coupling[order] * (Z(i) @ Z(j))
    else:
        for edge in lattice.edges:
            i, j = edge[0:2]
            hamiltonian += -coupling[i][j] * (Z(i) @ Z(j))

    for vertex in range(lattice.n_sites):
        hamiltonian += -h * X(vertex)

    return hamiltonian.simplify()


def heisenberg(lattice, n_cells, coupling=None, boundary_condition=False, neighbour_order=1):
    r"""Generates the Hamiltonian for the Heisenberg model on a lattice.

    The Hamiltonian is represented as:

    .. math::

         \hat{H} = J\sum_{<i,j>}(\sigma_i^x\sigma_j^x + \sigma_i^y\sigma_j^y + \sigma_i^z\sigma_j^z)

    where ``J`` is the coupling constant defined for the Hamiltonian, and ``i,j`` represent the
    indices for neighbouring spins.

    Args:
        lattice (str): Shape of the lattice. Input values can be ``'chain'``, ``'square'``,
            ``'rectangle'``, ``'honeycomb'``, ``'triangle'``, or ``'kagome'``.
        n_cells (List[int]): Number of cells in each direction of the grid.
        coupling (List[List[float]] or List[math.array[float]]): Coupling between spins. It can be a
            2D array of shape ``(neighbour_order, 3)`` or a 3D array of shape
            ``(3, num_spins, num_spins)``, where ``num_spins`` is the total number of spins.
        boundary_condition (bool or list[bool]): Defines boundary conditions for different lattice
            axes, default is ``False`` indicating open boundary condition.
        neighbour_order (int): Specifies the interaction level for neighbors within the lattice.
            Default is 1, indicating nearest neighbours.

    Returns:
        ~ops.op_math.Sum: Hamiltonian for the heisenberg model.

    **Example**

    >>> n_cells = [2,2]
    >>> j = [[0.5, 0.5, 0.5]]
    >>> spin_ham = qml.spin.heisenberg("square", n_cells, coupling=j)
    >>> spin_ham
    (
    0.5 * (X(0) @ X(1))
    + 0.5 * (Y(0) @ Y(1))
    + 0.5 * (Z(0) @ Z(1))
    + 0.5 * (X(0) @ X(2))
    + 0.5 * (Y(0) @ Y(2))
    + 0.5 * (Z(0) @ Z(2))
    + 0.5 * (X(1) @ X(3))
    + 0.5 * (Y(1) @ Y(3))
    + 0.5 * (Z(1) @ Z(3))
    + 0.5 * (X(2) @ X(3))
    + 0.5 * (Y(2) @ Y(3))
    + 0.5 * (Z(2) @ Z(3))
    )
    """

    lattice = _generate_lattice(lattice, n_cells, boundary_condition, neighbour_order)

    if coupling is None:
        coupling = [[1.0, 1.0, 1.0]]
    coupling = math.asarray(coupling)
    if coupling.ndim == 1:
        coupling = math.asarray([coupling])

    if coupling.shape not in [(neighbour_order, 3), (3, lattice.n_sites, lattice.n_sites)]:
        raise ValueError(
            f"The coupling parameter shape should be equal to ({neighbour_order},3) or (3,{lattice.n_sites},{lattice.n_sites})"
        )

    hamiltonian = 0.0 * qml.I(0)
    if coupling.shape == (neighbour_order, 3):
        for edge in lattice.edges:
            i, j, order = edge
            hamiltonian += (
                coupling[order][0] * (X(i) @ X(j))
                + coupling[order][1] * (Y(i) @ Y(j))
                + coupling[order][2] * (Z(i) @ Z(j))
            )
    else:
        for edge in lattice.edges:
            i, j = edge[0:2]
            hamiltonian += (
                coupling[0][i][j] * X(i) @ X(j)
                + coupling[1][i][j] * Y(i) @ Y(j)
                + coupling[2][i][j] * Z(i) @ Z(j)
            )

    return hamiltonian.simplify()


def fermi_hubbard(
    lattice,
    n_cells,
    hopping=1.0,
    coulomb=1.0,
    boundary_condition=False,
    neighbour_order=1,
    mapping="jordan_wigner",
):
    r"""Generates the Hamiltonian for the Fermi-Hubbard model on a lattice.

    The Hamiltonian is represented as:

    .. math::

        \hat{H} = -t\sum_{<i,j>, \sigma}(c_{i\sigma}^{\dagger}c_{j\sigma}) + U\sum_{i}n_{i \uparrow} n_{i\downarrow}

    where ``t`` is the hopping term representing the kinetic energy of electrons, ``U`` is the
    on-site Coulomb interaction, representing the repulsion between electrons, ``i,j`` represent the
    indices for neighbouring spins, :math:`\sigma` is the spin degree of freedom, and
    :math:`n_{i \uparrow}, n_{i \downarrow}` are number operators for spin-up and spin-down fermions
    at site ``i``. This function assumes there are two fermions with opposite spins on each lattice
    site.

    Args:
        lattice (str): Shape of the lattice. Input values can be ``'chain'``, ``'square'``,
            ``'rectangle'``, ``'honeycomb'``, ``'triangle'``, or ``'kagome'``.
        n_cells (List[int]): Number of cells in each direction of the grid.
        hopping (float or List[float] or List[math.array(float)]): Hopping strength between
            neighbouring sites, it can be a number, a list of length equal to ``neighbour_order`` or
            a square matrix of size ``(num_spins, num_spins)``, where ``num_spins`` is the total
            number of spins. Default value is 1.0.
        coulomb (float or List[float]): Coulomb interaction between spins. It can be a constant or a
            list of length equal to number of spins.
        boundary_condition (bool or list[bool]): Defines boundary conditions for different lattice
            axes, default is ``False`` indicating open boundary condition.
        neighbour_order (int): Specifies the interaction level for neighbors within the lattice.
            Default is 1, indicating nearest neighbours.
        mapping (str): Specifies the fermion-to-qubit mapping. Input values can be
            ``'jordan_wigner'``, ``'parity'`` or ``'bravyi_kitaev'``.

    Returns:
       ~ops.op_math.Sum: Hamiltonian for the Fermi-Hubbard model.

    **Example**

    >>> n_cells = [2]
    >>> h = [0.5]
    >>> u = 1.0
    >>> spin_ham = qml.spin.fermi_hubbard("chain", n_cells, hopping=h, coulomb=u)
    >>> spin_ham
    (
    -0.25 * (Y(0) @ Z(1) @ Y(2))
    + -0.25 * (X(0) @ Z(1) @ X(2))
    + 0.5 * I(0)
    + -0.25 * (Y(1) @ Z(2) @ Y(3))
    + -0.25 * (X(1) @ Z(2) @ X(3))
    + -0.25 * Z(1)
    + -0.25 * Z(0)
    + 0.25 * (Z(0) @ Z(1))
    + -0.25 * Z(3)
    + -0.25 * Z(2)
    + 0.25 * (Z(2) @ Z(3))
    )
    """

    lattice = _generate_lattice(lattice, n_cells, boundary_condition, neighbour_order)

    if isinstance(hopping, (int, float, complex)):
        hopping = [hopping]

    hopping = math.asarray(hopping)

    if hopping.shape not in [(neighbour_order,), (lattice.n_sites, lattice.n_sites)]:
        raise ValueError(
            f"The hopping parameter should be a number or an array of shape ({neighbour_order},) or ({lattice.n_sites},{lattice.n_sites})"
        )

    spin = 2
    hopping_ham = 0.0 * FermiWord({})
    if hopping.shape == (neighbour_order,):
        for edge in lattice.edges:
            for s in range(spin):
                i, j, order = edge
                s1 = i * spin + s
                s2 = j * spin + s
                hopping_term = -hopping[order] * (
                    FermiWord({(0, s1): "+", (1, s2): "-"})
                    + FermiWord({(0, s2): "+", (1, s1): "-"})
                )
                hopping_ham += hopping_term
    else:
        for edge in lattice.edges:
            for s in range(spin):
                i, j = edge[0:2]
                s1 = i * spin + s
                s2 = j * spin + s
                hopping_term = -hopping[i][j] * (
                    FermiWord({(0, s1): "+", (1, s2): "-"})
                    + FermiWord({(0, s2): "+", (1, s1): "-"})
                )
                hopping_ham += hopping_term

    int_term = 0.0 * FermiWord({})
    if isinstance(coulomb, (int, float, complex)):
        coulomb = math.ones(lattice.n_sites) * coulomb

    for i in range(lattice.n_sites):
        up_spin = i * spin
        down_spin = i * spin + 1
        int_term += coulomb[i] * FermiWord(
            {(0, up_spin): "+", (1, up_spin): "-", (2, down_spin): "+", (3, down_spin): "-"}
        )

    hamiltonian = hopping_ham + int_term

    if mapping not in ["jordan_wigner", "parity", "bravyi_kitaev"]:
        raise ValueError(
            f"The '{mapping}' transformation is not available."
            f"Please set mapping to 'jordan_wigner', 'parity', or 'bravyi_kitaev'"
        )
    qubit_ham = qml.qchem.qubit_observable(hamiltonian, mapping=mapping)

    return qubit_ham.simplify()
