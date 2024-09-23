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

from .lattice import Lattice, _generate_lattice

# pylint: disable=too-many-arguments, too-many-branches


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


def emery(
    lattice,
    n_cells,
    hopping=1.0,
    coulomb=1.0,
    intersite_coupling=1.0,
    boundary_condition=False,
    neighbour_order=1,
    mapping="jordan_wigner",
):
    r"""Generates the Hamiltonian for the Emery model on a lattice.

    The `Hamiltonian <https://arxiv.org/pdf/2309.11786>`_ for the `Emery model <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.58.2794>`_ is represented as:

    .. math::
        \begin{align*}
          \hat{H} & = -\sum_{\langle i,j \rangle, \sigma} t_{ij}(c_{i\sigma}^{\dagger}c_{j\sigma})
          + \sum_{i}U_{i}n_{i \uparrow} n_{i\downarrow} + \sum_{<i,j>}V_{ij}(n_{i \uparrow} +
          n_{i \downarrow})(n_{j \uparrow} + n_{j \downarrow})\ ,
        \end{align*}

    where :math:`t_{ij}` is the hopping term representing the kinetic energy of electrons,
    :math:`U_{i}` is the on-site Coulomb interaction, representing the repulsion between electrons,
    :math:`V_{ij}` is the intersite coupling, :math:`i, j` are the indices for neighbouring spins,
    :math:`\sigma` is the spin degree of freedom, and :math:`n_{k \uparrow}`, :math:`n_{k \downarrow}`
    are number operators for spin-up and spin-down fermions at site :math:`k`.
    This function assumes there are two fermions with opposite spins on each lattice site.

    Args:
        lattice (str): Shape of the lattice. Input values can be ``'chain'``, ``'square'``,
            ``'rectangle'``, ``'honeycomb'``, ``'triangle'``, or ``'kagome'``.
        n_cells (list[int]): Number of cells in each direction of the grid.
        hopping (float or list[float] or tensor_like[float]): Hopping strength between
            neighbouring sites. It can be a number, a list of length equal to ``neighbour_order`` or
            a square matrix of shape ``(n_sites, n_sites)``, where ``n_sites`` is the total
            number of sites. Default value is 1.0.
        coulomb (float or list[float]): Coulomb interaction between spins. It can be a constant or a
            list of length equal to number of spins.
        intersite_coupling (float or list[float] or tensor_like[float]): Interaction strength between spins on
            neighbouring sites. It can be a number, a list with length equal to ``neighbour_order`` or
            a square matrix of size ``(n_sites, n_sites)``, where ``n_sites`` is the total
            number of sites. Default value is 1.0.
        boundary_condition (bool or list[bool]): Defines boundary conditions for different lattice
            axes. Default is ``False`` indicating open boundary condition.
        neighbour_order (int): Specifies the interaction level for neighbors within the lattice.
            Default is 1, indicating nearest neighbours.
        mapping (str): Specifies the fermion-to-qubit mapping. Input values can be
            ``'jordan_wigner'``, ``'parity'`` or ``'bravyi_kitaev'``.

    Raises:
       ValueError:
          If ``hopping``, ``coulomb``, or ``intersite_coupling`` doesn't have correct dimensions,
          or if ``mapping`` is not available.

    Returns:
       ~ops.op_math.Sum: Hamiltonian for the Emery model.

    **Example**

    >>> n_cells = [2]
    >>> h = [0.5]
    >>> u = 1.0
    >>> v = 0.2
    >>> spin_ham = qml.spin.emery("chain", n_cells, hopping=h, coulomb=u,
                   intersite_coupling=v)
    >>> spin_ham
    (
      -0.25 * (Y(0) @ Z(1) @ Y(2))
      + -0.25 * (X(0) @ Z(1) @ X(2))
      + 0.7000000000000002 * I(0)
      + -0.25 * (Y(1) @ Z(2) @ Y(3))
      + -0.25 * (X(1) @ Z(2) @ X(3))
      + -0.35 * Z(1)
      + -0.35 * Z(0)
      + 0.25 * (Z(0) @ Z(1))
      + -0.35 * Z(3)
      + -0.35 * Z(2)
      + 0.25 * (Z(2) @ Z(3))
      + 0.05 * (Z(0) @ Z(2))
      + 0.05 * (Z(0) @ Z(3))
      + 0.05 * (Z(1) @ Z(2))
      + 0.05 * (Z(1) @ Z(3))
    )

    """

    lattice = _generate_lattice(lattice, n_cells, boundary_condition, neighbour_order)

    hopping = (
        math.asarray([hopping])
        if isinstance(hopping, (int, float, complex))
        else math.asarray(hopping)
    )
    intersite_coupling = (
        math.asarray([intersite_coupling])
        if isinstance(intersite_coupling, (int, float, complex))
        else math.asarray(intersite_coupling)
    )

    if hopping.shape not in [(neighbour_order,), (lattice.n_sites, lattice.n_sites)]:
        raise ValueError(
            f"The hopping parameter should be a number or an "
            f"array of shape ({neighbour_order},) or ({lattice.n_sites},{lattice.n_sites})."
        )

    if intersite_coupling.shape not in [(neighbour_order,), (lattice.n_sites, lattice.n_sites)]:
        raise ValueError(
            f"The intersite_coupling parameter should be a number or "
            f"an array of shape ({neighbour_order},) or ({lattice.n_sites},{lattice.n_sites})."
        )

    spin = 2
    hopping_term = 0.0 * FermiWord({})
    intersite_term = 0.0 * FermiWord({})
    for i, j, order in lattice.edges:
        hop = hopping[order] if hopping.shape == (neighbour_order,) else hopping[i][j]

        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            hopping_term -= hop * (
                FermiWord({(0, s1): "+", (1, s2): "-"}) + FermiWord({(0, s2): "+", (1, s1): "-"})
            )

        intersite = (
            intersite_coupling[order]
            if intersite_coupling.shape == (neighbour_order,)
            else intersite_coupling[i][j]
        )
        intersite_term += (
            intersite
            * (
                FermiWord({(0, i * spin): "+", (1, i * spin): "-"})
                + FermiWord({(0, i * spin + 1): "+", (1, i * spin + 1): "-"})
            )
            * (
                FermiWord({(0, j * spin): "+", (1, j * spin): "-"})
                + FermiWord({(0, j * spin + 1): "+", (1, j * spin + 1): "-"})
            )
        )

    if isinstance(coulomb, (int, float, complex)):
        coulomb = math.ones(lattice.n_sites) * coulomb

    coulomb_term = 0.0 * FermiWord({})
    for i in range(lattice.n_sites):
        up_spin = i * spin
        down_spin = i * spin + 1
        coulomb_term += coulomb[i] * FermiWord(
            {(0, up_spin): "+", (1, up_spin): "-", (2, down_spin): "+", (3, down_spin): "-"}
        )

    hamiltonian = hopping_term + coulomb_term + intersite_term

    if mapping not in ["jordan_wigner", "parity", "bravyi_kitaev"]:
        raise ValueError(
            f"The '{mapping}' transformation is not available."
            f"Please set mapping to 'jordan_wigner', 'parity', or 'bravyi_kitaev'."
        )
    qubit_ham = qml.qchem.qubit_observable(hamiltonian, mapping=mapping)

    return qubit_ham.simplify()


def haldane(
    lattice,
    n_cells,
    hopping=1.0,
    hopping_next=1.0,
    phi=1.0,
    boundary_condition=False,
    mapping="jordan_wigner",
):
    r"""Generates the Hamiltonian for the Haldane model on a lattice.

    The `Hamiltonian <https://arxiv.org/pdf/2211.13615>`_ for the  `Haldane model <https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.61.2015>`_  is represented as:

    .. math::

        \begin{align*}
          \hat{H} & = -\sum_{\langle i,j \rangle}t_{ij}^{1}
          (c_{i\sigma}^\dagger c_{j\sigma} + c_{j\sigma}^\dagger c_{i\sigma})
          - \sum_{\langle\langle i,j \rangle\rangle, \sigma} t_{ij}^{2}
          \left( e^{i\phi_{ij}} c_{i\sigma}^\dagger c_{j\sigma} + e^{-i\phi_{ij}} c_{j\sigma}^\dagger c_{i\sigma} \right)
        \end{align*}

    where :math:`t^{1}_{ij}` is the hopping term representing the hopping amplitude between neighbouring
    sites :math:`\langle i,j \rangle`, :math:`t^{2}_{ij}` is the hopping amplitude between next-nearest neighbours :math:`\langle \langle i,j \rangle \rangle`, :math:`\phi_{ij}` is the phase
    factor that breaks time-reversal symmetry in the system, and :math:`\sigma` is the spin degree of freedom.
    This function assumes there are two fermions with opposite spins on each lattice site.

    Args:
        lattice (str): Shape of the lattice. Input values can be ``'chain'``, ``'square'``,
            ``'rectangle'``, ``'honeycomb'``, ``'triangle'``, or ``'kagome'``.
        n_cells (list[int]): Number of cells in each direction of the grid.
        hopping (float or tensor_like[float]): Hopping strength between
            nearest neighbouring sites. It can be a number, or
            a square matrix of size ``(n_sites, n_sites)``, where ``n_sites`` is the total
            number of sites. Default value is 1.0.
        hopping_next (float or tensor_like[float]): Hopping strength between next-nearest
            neighbouring sites. It can be a number, or
            a square matrix of size ``(n_sites, n_sites)``, where ``n_sites`` is the total
            number of sites. Default value is 1.0.
        phi (float or tensor_like[float]): Phase factor in the system. It can be a number, or
            a square matrix of size ``(n_sites, n_sites)``, where ``n_sites`` is the total
            number of sites. Default value is 1.0.
        boundary_condition (bool or list[bool]): Defines boundary conditions for different lattice
            axes. Default is ``False`` indicating open boundary condition.
        mapping (str): Specifies the fermion-to-qubit mapping. Input values can be
            ``'jordan_wigner'``, ``'parity'`` or ``'bravyi_kitaev'``.

    Raises:
       ValueError:
          If ``hopping``, ``hopping_next``, or ``phi`` doesn't have correct dimensions,
          or if ``mapping`` is not available.

    Returns:
       ~ops.op_math.Sum: Hamiltonian for the Haldane model.

    **Example**

    >>> n_cells = [2]
    >>> h1 = 0.5
    >>> h2 = 1.0
    >>> phi = 0.1
    >>> spin_ham = qml.spin.haldane("chain", n_cells, hopping=h1, hopping_next=h2, phi=phi)
    >>> spin_ham
    (
      -0.25 * (Y(0) @ Z(1) @ Y(2))
      + -0.25 * (X(0) @ Z(1) @ X(2))
      + -0.25 * (Y(1) @ Z(2) @ Y(3))
      + -0.25 * (X(1) @ Z(2) @ X(3))
    )

    """

    lattice = _generate_lattice(lattice, n_cells, boundary_condition, neighbour_order=2)

    hopping = (
        math.asarray([hopping])
        if isinstance(hopping, (int, float, complex))
        else math.asarray(hopping)
    )
    hopping_next = (
        math.asarray([hopping_next])
        if isinstance(hopping_next, (int, float, complex))
        else math.asarray(hopping_next)
    )
    phi = math.asarray([phi]) if isinstance(phi, (int, float, complex)) else math.asarray(phi)

    if hopping.shape not in [(1,), (lattice.n_sites, lattice.n_sites)]:
        raise ValueError(
            f"The hopping parameter should be a constant or an array of shape ({lattice.n_sites},{lattice.n_sites})."
        )

    if hopping_next.shape not in [(1,), (lattice.n_sites, lattice.n_sites)]:
        raise ValueError(
            f"The hopping_next parameter should be a constant or an array of shape ({lattice.n_sites},{lattice.n_sites})."
        )

    if phi.shape not in [(1,), (lattice.n_sites, lattice.n_sites)]:
        raise ValueError(
            f"The phi parameter should be a constant or an array of shape ({lattice.n_sites},{lattice.n_sites})."
        )

    spin = 2
    hamiltonian = 0.0 * FermiWord({})
    for i, j, order in lattice.edges:

        hop1 = hopping[0] if hopping.shape == (1,) else hopping[i][j]
        hop2 = hopping_next[0] if hopping_next.shape == (1,) else hopping_next[i][j]
        phi_term = phi[0] if phi.shape == (1,) else phi[i][j]

        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            if order == 0:
                hamiltonian -= hop1 * (
                    FermiWord({(0, s1): "+", (1, s2): "-"})
                    + FermiWord({(0, s2): "+", (1, s1): "-"})
                )
            else:
                hamiltonian -= hop2 * (
                    math.exp(1j * phi_term) * FermiWord({(0, s1): "+", (1, s2): "-"})
                )
                hamiltonian -= hop2 * (
                    math.exp(-1j * phi_term) * FermiWord({(0, s2): "+", (1, s1): "-"})
                )

    if mapping not in ["jordan_wigner", "parity", "bravyi_kitaev"]:
        raise ValueError(
            f"The '{mapping}' transformation is not available."
            f"Please set mapping to 'jordan_wigner', 'parity', or 'bravyi_kitaev'."
        )
    qubit_ham = qml.qchem.qubit_observable(hamiltonian, mapping=mapping)

    return qubit_ham.simplify()


def kitaev(n_cells, coupling=None, boundary_condition=False):
    r"""Generates the Hamiltonian for the Kitaev model on the Honeycomb lattice.

    The `Kitaev <https://arxiv.org/abs/cond-mat/0506438>`_ model Hamiltonian is represented as:

    .. math::
        \begin{align*}
          \hat{H} = K_x.\sum_{\langle i,j \rangle \in X}\sigma_i^x\sigma_j^x +
          \:\: K_y.\sum_{\langle i,j \rangle \in Y}\sigma_i^y\sigma_j^y +
          \:\: K_z.\sum_{\langle i,j \rangle \in Z}\sigma_i^z\sigma_j^z
        \end{align*}

    where :math:`K_x`, :math:`K_y`, :math:`K_z` are the coupling constants defined for the Hamiltonian,
    and :math:`X`, :math:`Y`, :math:`Z` represent the set of edges in the Honeycomb lattice between spins
    :math:`i` and :math:`j` with real-space bond directions :math:`[0, 1], [\frac{\sqrt{3}}{2}, \frac{1}{2}],
    \frac{\sqrt{3}}{2}, -\frac{1}{2}]`, respectively.

    Args:
       n_cells (list[int]): Number of cells in each direction of the grid.
       coupling (Optional[list[float] or tensor_like(float)]): Coupling between spins, it is a list of length 3.
                            Default value is [1.0, 1.0, 1.0].
       boundary_condition (bool or list[bool]): Defines boundary conditions for different lattice axes.
           The default is ``False``, indicating open boundary conditions for all.

    Raises:
       ValueError: if ``coupling`` doesn't have correct dimensions.

    Returns:
       ~ops.op_math.Sum: Hamiltonian for the Kitaev model.

    **Example**

    >>> n_cells = [2,2]
    >>> k = [0.5, 0.6, 0.7]
    >>> spin_ham = qml.spin.kitaev(n_cells, coupling=k)
    >>> spin_ham
    (
      0.5 * (X(0) @ X(1))
      + 0.5 * (X(2) @ X(3))
      + 0.5 * (X(4) @ X(5))
      + 0.5 * (X(6) @ X(7))
      + 0.6 * (Y(1) @ Y(2))
      + 0.6 * (Y(5) @ Y(6))
      + 0.7 * (Z(1) @ Z(4))
      + 0.7 * (Z(3) @ Z(6))
    )

    """

    if coupling is None:
        coupling = [1.0, 1.0, 1.0]

    if len(coupling) != 3:
        raise ValueError("The coupling parameter should be a list of length 3.")

    vectors = [[1, 0], [0.5, 0.75**0.5]]
    positions = [[0, 0], [0.5, 0.5 / 3**0.5]]
    custom_edges = [
        [(0, 1), ("XX", coupling[0])],
        [(1, 2), ("YY", coupling[1])],
        [(1, n_cells[1] * 2), ("ZZ", coupling[2])],
    ]

    lattice = Lattice(
        n_cells=n_cells[0:2],
        vectors=vectors,
        positions=positions,
        boundary_condition=boundary_condition,
        custom_edges=custom_edges,
    )
    opmap = {"X": X, "Y": Y, "Z": Z}
    hamiltonian = 0.0 * qml.I(0)
    for edge in lattice.edges:
        v1, v2 = edge[0:2]
        op1, op2 = edge[2][0]
        coeff = edge[2][1]
        hamiltonian += coeff * (opmap[op1](v1) @ opmap[op2](v2))

    return hamiltonian.simplify()
