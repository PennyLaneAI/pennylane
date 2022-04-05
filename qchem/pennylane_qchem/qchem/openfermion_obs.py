# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains functions to construct many-body observables whose expectation
values can be used to simulate molecular properties.
"""
# pylint: disable=too-many-arguments, too-few-public-methods
import os
import subprocess
from shutil import copyfile

import numpy as np

import pennylane as qml

from . import openfermion

# Bohr-Angstrom correlation coefficient (https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0)
bohr_angs = 0.529177210903


def observable(fermion_ops, init_term=0, mapping="jordan_wigner", wires=None):

    r"""Builds the fermionic many-body observable whose expectation value can be
    measured in PennyLane.

    The second-quantized operator of the fermionic many-body system can combine one-particle
    and two-particle operators as in the case of electronic Hamiltonians :math:`\hat{H}`:

    .. math::

        \hat{H} = \sum_{\alpha, \beta} \langle \alpha \vert \hat{t}^{(1)} +
        \cdots + \hat{t}^{(n)} \vert \beta \rangle ~ \hat{c}_\alpha^\dagger \hat{c}_\beta
        + \frac{1}{2} \sum_{\alpha, \beta, \gamma, \delta}
        \langle \alpha, \beta \vert \hat{v}^{(1)} + \cdots + \hat{v}^{(n)}
        \vert \gamma, \delta \rangle ~ \hat{c}_\alpha^\dagger \hat{c}_\beta^\dagger
        \hat{c}_\gamma \hat{c}_\delta

    In the latter equations the indices :math:`\alpha, \beta, \gamma, \delta` run over the
    basis of single-particle states. The operators :math:`\hat{c}^\dagger` and :math:`\hat{c}`
    are the particle creation and annihilation operators, respectively.
    :math:`\langle \alpha \vert \hat{t} \vert \beta \rangle` denotes the matrix element of
    the single-particle operator :math:`\hat{t}` entering the observable. For example,
    in electronic structure calculations, this is the case for the kinetic energy operator,
    the nuclei Coulomb potential, or any other external fields included in the Hamiltonian.
    On the other hand, :math:`\langle \alpha, \beta \vert \hat{v} \vert \gamma, \delta \rangle`
    denotes the matrix element of the two-particle operator :math:`\hat{v}`, for example, the
    Coulomb interaction between the electrons.

    - The observable is built by adding the operators
      :math:`\sum_{\alpha, \beta} t_{\alpha\beta}^{(i)}
      \hat{c}_\alpha^\dagger \hat{c}_\beta` and
      :math:`\frac{1}{2} \sum_{\alpha, \beta, \gamma, \delta}
      v_{\alpha\beta\gamma\delta}^{(i)}
      \hat{c}_\alpha^\dagger \hat{c}_\beta^\dagger \hat{c}_\gamma \hat{c}_\delta`.

    - Second-quantized operators contributing to the
      many-body observable must be represented using the `FermionOperator
      <https://github.com/quantumlib/OpenFermion/blob/master/docs/
      tutorials/intro_to_openfermion.ipynb>`_ data structure as implemented in OpenFermion.
      See the functions :func:`~.one_particle` and :func:`~.two_particle` to build the
      FermionOperator representations of one-particle and two-particle operators.

    - The function uses tools of `OpenFermion <https://github.com/quantumlib/OpenFermion>`_
      to map the resulting fermionic Hamiltonian to the basis of Pauli matrices via the
      Jordan-Wigner or Bravyi-Kitaev transformation. Finally, the qubit operator is converted
      to a PennyLane observable by the function :func:`~.convert_observable`.

    Args:
        fermion_ops (list[FermionOperator]): list containing the FermionOperator data structures
            representing the one-particle and/or two-particle operators entering the many-body
            observable
        init_term (float): Any quantity required to initialize the many-body observable. For
            example, this can be used to pass the nuclear-nuclear repulsion energy :math:`V_{nn}`
            which is typically included in the electronic Hamiltonian of molecules.
        mapping (str): Specifies the fermion-to-qubit mapping. Input values can
            be ``'jordan_wigner'`` or ``'bravyi_kitaev'``.
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert the qubit operator
            to an observable measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).

    Returns:
        pennylane.Hamiltonian: the fermionic-to-qubit transformed observable

    **Example**

    >>> t = FermionOperator("0^ 0", 0.5) + FermionOperator("1^ 1", 0.25)
    >>> v = FermionOperator("1^ 0^ 0 1", -0.15) + FermionOperator("2^ 0^ 2 0", 0.3)
    >>> print(observable([t, v], mapping="jordan_wigner"))
    (0.2625) [I0]
    + (-0.1375) [Z0]
    + (-0.0875) [Z1]
    + (-0.0375) [Z0 Z1]
    + (0.075) [Z2]
    + (-0.075) [Z0 Z2]
    """

    if mapping.strip().lower() not in ("jordan_wigner", "bravyi_kitaev"):
        raise TypeError(
            f"The '{mapping}' transformation is not available. \n "
            f"Please set 'mapping' to 'jordan_wigner' or 'bravyi_kitaev'."
        )

    # Initialize the FermionOperator
    mb_obs = openfermion.ops.FermionOperator("") * init_term
    for ops in fermion_ops:
        if not isinstance(ops, openfermion.ops.FermionOperator):
            raise TypeError(
                f"Elements in the lists are expected to be of type 'FermionOperator'; got {type(ops)}"
            )
        mb_obs += ops

    # Map the fermionic operator to a qubit operator
    if mapping.strip().lower() == "bravyi_kitaev":
        return qml.qchem.convert.import_operator(
            openfermion.transforms.bravyi_kitaev(mb_obs), wires=wires
        )

    return qml.qchem.convert.import_operator(
        openfermion.transforms.jordan_wigner(mb_obs), wires=wires
    )


def one_particle(matrix_elements, core=None, active=None, cutoff=1.0e-12):
    r"""Generates the `FermionOperator <https://github.com/quantumlib/OpenFermion/blob/master/docs/
    tutorials/intro_to_openfermion.ipynb>`_ representing a given one-particle operator
    required to build many-body qubit observables.

    Second quantized one-particle operators are expanded in the basis of single-particle
    states as

    .. math::

        \hat{T} = \sum_{\alpha, \beta} \langle \alpha \vert \hat{t} \vert \beta \rangle
        [\hat{c}_{\alpha\uparrow}^\dagger \hat{c}_{\beta\uparrow} +
        \hat{c}_{\alpha\downarrow}^\dagger \hat{c}_{\beta\downarrow}].

    In the equation above the indices :math:`\alpha, \beta` run over the basis of spatial
    orbitals :math:`\phi_\alpha(r)`. Since the operator :math:`\hat{t}` acts only on the
    spatial coordinates, the spin quantum numbers are indicated explicitly with the up/down arrows.
    The operators :math:`\hat{c}^\dagger` and :math:`\hat{c}` are the particle creation
    and annihilation operators, respectively, and
    :math:`\langle \alpha \vert \hat{t} \vert \beta \rangle` denotes the matrix elements of
    the operator :math:`\hat{t}`

    .. math::

        \langle \alpha \vert \hat{t} \vert \beta \rangle = \int dr ~ \phi_\alpha^*(r)
        \hat{t}(r) \phi_\beta(r).

    If an active space is defined (see :func:`~.active_space`), the summation indices
    run over the active orbitals and the contribution due to core orbitals is computed as
    :math:`t_\mathrm{core} = 2 \sum_{\alpha\in \mathrm{core}}
    \langle \alpha \vert \hat{t} \vert \alpha \rangle`.

    Args:
        matrix_elements (array[float]): 2D NumPy array with the matrix elements
            :math:`\langle \alpha \vert \hat{t} \vert \beta \rangle`
        core (list): indices of core orbitals, i.e., the orbitals that are
            not correlated in the many-body wave function
        active (list): indices of active orbitals, i.e., the orbitals used to
            build the correlated many-body wave function
        cutoff (float): Cutoff value for including matrix elements. The
            matrix elements with absolute value less than ``cutoff`` are neglected.

    Returns:
        FermionOperator: an instance of OpenFermion's ``FermionOperator`` representing the
        one-particle operator :math:`\hat{T}`.

    **Example**

    >>> matrix_elements = np.array([[-1.27785301e+00,  0.00000000e+00],
    ...                             [ 1.52655666e-16, -4.48299696e-01]])
    >>> t_op = one_particle(matrix_elements)
    >>> print(t_op)
    -1.277853006156875 [0^ 0] +
    -1.277853006156875 [1^ 1] +
    -0.44829969610163756 [2^ 2] +
    -0.44829969610163756 [3^ 3]
    """

    orbitals = matrix_elements.shape[0]

    if matrix_elements.ndim != 2:
        raise ValueError(
            f"'matrix_elements' must be a 2D array; got matrix_elements.ndim = {matrix_elements.ndim}"
        )

    if not core:
        t_core = 0
    else:
        if any(i > orbitals - 1 or i < 0 for i in core):
            raise ValueError(
                f"Indices of core orbitals must be between 0 and {orbitals - 1}; got core = {core}"
            )

        # Compute contribution due to core orbitals
        t_core = 2 * np.sum(matrix_elements[np.array(core), np.array(core)])

    if active is None:
        if not core:
            active = list(range(orbitals))
        else:
            active = [i for i in range(orbitals) if i not in core]

    if any(i > orbitals - 1 or i < 0 for i in active):
        raise ValueError(
            f"Indices of active orbitals must be between 0 and {orbitals - 1}; got active = {active}"
        )

    # Indices of the matrix elements with absolute values >= cutoff
    indices = np.nonzero(np.abs(matrix_elements) >= cutoff)

    # Single out the indices of active orbitals
    num_indices = len(indices[0])
    pairs = [
        [indices[0][i], indices[1][i]]
        for i in range(num_indices)
        if all(indices[j][i] in active for j in range(len(indices)))
    ]

    # Build the FermionOperator representing T
    t_op = openfermion.ops.FermionOperator("") * t_core
    for pair in pairs:
        alpha, beta = pair
        element = matrix_elements[alpha, beta]

        # spin-up term
        a = 2 * active.index(alpha)
        b = 2 * active.index(beta)
        t_op += openfermion.ops.FermionOperator(((a, 1), (b, 0)), element)

        # spin-down term
        t_op += openfermion.ops.FermionOperator(((a + 1, 1), (b + 1, 0)), element)

    return t_op


def two_particle(matrix_elements, core=None, active=None, cutoff=1.0e-12):
    r"""Generates the `FermionOperator <https://github.com/quantumlib/OpenFermion/blob/master/docs/
    tutorials/intro_to_openfermion.ipynb>`_ representing a given two-particle operator
    required to build many-body qubit observables.

    Second quantized two-particle operators are expanded in the basis of single-particle
    states as

    .. math::

        \hat{V} = \frac{1}{2} \sum_{\alpha, \beta, \gamma, \delta}
        \langle \alpha, \beta \vert \hat{v} \vert \gamma, \delta \rangle
        ~ &[& \hat{c}_{\alpha\uparrow}^\dagger \hat{c}_{\beta\uparrow}^\dagger
        \hat{c}_{\gamma\uparrow} \hat{c}_{\delta\uparrow} + \hat{c}_{\alpha\uparrow}^\dagger
        \hat{c}_{\beta\downarrow}^\dagger \hat{c}_{\gamma\downarrow} \hat{c}_{\delta\uparrow} \\
        &+& \hat{c}_{\alpha\downarrow}^\dagger \hat{c}_{\beta\uparrow}^\dagger
        \hat{c}_{\gamma\uparrow} \hat{c}_{\delta\downarrow} + \hat{c}_{\alpha\downarrow}^\dagger
        \hat{c}_{\beta\downarrow}^\dagger \hat{c}_{\gamma\downarrow} \hat{c}_{\delta\downarrow}~].

    In the equation above the indices :math:`\alpha, \beta, \gamma, \delta` run over the basis
    of spatial orbitals :math:`\phi_\alpha(r)`. Since the operator :math:`v` acts only on the
    spatial coordinates the spin quantum numbers are indicated explicitly with the up/down arrows.
    The operators :math:`\hat{c}^\dagger` and :math:`\hat{c}` are the particle creation and
    annihilation operators, respectively, and
    :math:`\langle \alpha, \beta \vert \hat{v} \vert \gamma, \delta \rangle` denotes the
    matrix elements of the operator :math:`\hat{v}`

    .. math::

        \langle \alpha, \beta \vert \hat{v} \vert \gamma, \delta \rangle =
        \int dr_1 \int dr_2 ~ \phi_\alpha^*(r_1) \phi_\beta^*(r_2) ~\hat{v}(r_1, r_2)~
        \phi_\gamma(r_2) \phi_\delta(r_1).

    If an active space is defined (see :func:`~.active_space`), the summation indices
    run over the active orbitals and the contribution due to core orbitals is computed as

    .. math::

        && \hat{V}_\mathrm{core} = v_\mathrm{core} +
        \sum_{\alpha, \beta \in \mathrm{active}} \sum_{i \in \mathrm{core}}
        (2 \langle i, \alpha \vert \hat{v} \vert \beta, i \rangle -
        \langle i, \alpha \vert \hat{v} \vert i, \beta \rangle)~
        [\hat{c}_{\alpha\uparrow}^\dagger \hat{c}_{\beta\uparrow} +
        \hat{c}_{\alpha\downarrow}^\dagger \hat{c}_{\beta\downarrow}] \\
        && v_\mathrm{core} = \sum_{\alpha,\beta \in \mathrm{core}}
        [2 \langle \alpha, \beta \vert \hat{v} \vert \beta, \alpha \rangle -
        \langle \alpha, \beta \vert \hat{v} \vert \alpha, \beta \rangle].

    Args:
        matrix_elements (array[float]): 4D NumPy array with the matrix elements
            :math:`\langle \alpha, \beta \vert \hat{v} \vert \gamma, \delta \rangle`
        core (list): indices of core orbitals, i.e., the orbitals that are
            not correlated in the many-body wave function
        active (list): indices of active orbitals, i.e., the orbitals used to
            build the correlated many-body wave function
        cutoff (float): Cutoff value for including matrix elements. The
            matrix elements with absolute value less than ``cutoff`` are neglected.

    Returns:
        FermionOperator: an instance of OpenFermion's ``FermionOperator`` representing the
        two-particle operator :math:`\hat{V}`.

    **Example**

    >>> matrix_elements = np.array([[[[ 6.82389533e-01, -1.45716772e-16],
    ...                               [-2.77555756e-17,  1.79000576e-01]],
    ...                              [[-2.77555756e-17,  1.79000576e-16],
    ...                               [ 6.70732778e-01, 0.00000000e+00]]],
    ...                             [[[-1.45716772e-16,  6.70732778e-16],
    ...                               [ 1.79000576e-01, -8.32667268e-17]],
    ...                              [[ 1.79000576e-16, -8.32667268e-17],
    ...                               [ 0.00000000e+00,  7.05105632e-01]]]])
    >>> v_op = two_particle(matrix_elements)
    >>> print(v_op)
    0.3411947665760211 [0^ 0^ 0 0]
    + 0.08950028803070323 [0^ 0^ 2 2]
    + 0.3411947665760211 [0^ 1^ 1 0]
    + 0.08950028803070323 [0^ 1^ 3 2]
    + 0.08950028803070323 [0^ 2^ 0 2]
    + 0.3353663891543792 [0^ 2^ 2 0]
    + 0.08950028803070323 [0^ 3^ 1 2]
    + 0.3353663891543792 [0^ 3^ 3 0]
    + 0.3411947665760211 [1^ 0^ 0 1]
    + 0.08950028803070323 [1^ 0^ 2 3]
    + 0.3411947665760211 [1^ 1^ 1 1]
    + 0.08950028803070323 [1^ 1^ 3 3]
    + 0.08950028803070323 [1^ 2^ 0 3]
    + 0.3353663891543792 [1^ 2^ 2 1]
    + 0.08950028803070323 [1^ 3^ 1 3]
    + 0.3353663891543792 [1^ 3^ 3 1]
    + 0.3353663891543792 [2^ 0^ 0 2]
    + 0.08950028803070323 [2^ 0^ 2 0]
    + 0.3353663891543792 [2^ 1^ 1 2]
    + 0.08950028803070323 [2^ 1^ 3 0]
    + 0.08950028803070323 [2^ 2^ 0 0]
    + 0.352552816086392 [2^ 2^ 2 2]
    + 0.08950028803070323 [2^ 3^ 1 0]
    + 0.352552816086392 [2^ 3^ 3 2]
    + 0.3353663891543792 [3^ 0^ 0 3]
    + 0.08950028803070323 [3^ 0^ 2 1]
    + 0.3353663891543792 [3^ 1^ 1 3]
    + 0.08950028803070323 [3^ 1^ 3 1]
    + 0.08950028803070323 [3^ 2^ 0 1]
    + 0.352552816086392 [3^ 2^ 2 3]
    + 0.08950028803070323 [3^ 3^ 1 1]
    + 0.352552816086392 [3^ 3^ 3 3]
    """

    orbitals = matrix_elements.shape[0]

    if matrix_elements.ndim != 4:
        raise ValueError(
            f"'matrix_elements' must be a 4D array; got 'matrix_elements.ndim = ' {matrix_elements.ndim}"
        )

    if not core:
        v_core = 0
    else:
        if any(i > orbitals - 1 or i < 0 for i in core):
            raise ValueError(
                f"Indices of core orbitals must be between 0 and {orbitals - 1}; got core = {core}"
            )

        # Compute the contribution of core orbitals
        v_core = sum(
            [
                2 * matrix_elements[alpha, beta, beta, alpha]
                - matrix_elements[alpha, beta, alpha, beta]
                for alpha in core
                for beta in core
            ]
        )

    if active is None:
        if not core:
            active = list(range(orbitals))
        else:
            active = [i for i in range(orbitals) if i not in core]

    if any(i > orbitals - 1 or i < 0 for i in active):
        raise ValueError(
            f"Indices of active orbitals must be between 0 and {orbitals - 1}; got active = {active}"
        )

    # Indices of the matrix elements with absolute values >= cutoff
    indices = np.nonzero(np.abs(matrix_elements) >= cutoff)

    # Single out the indices of active orbitals
    num_indices = len(indices[0])
    quads = [
        [indices[0][i], indices[1][i], indices[2][i], indices[3][i]]
        for i in range(num_indices)
        if all(indices[j][i] in active for j in range(len(indices)))
    ]

    # Build the FermionOperator representing V
    v_op = openfermion.ops.FermionOperator("") * v_core

    # add renormalized (due to core orbitals) "one-particle" operators
    if core:
        for alpha in active:
            for beta in active:

                element = 2 * np.sum(
                    matrix_elements[np.array(core), alpha, beta, np.array(core)]
                ) - np.sum(matrix_elements[np.array(core), alpha, np.array(core), beta])

                # up-up term
                a = 2 * active.index(alpha)
                b = 2 * active.index(beta)
                v_op += openfermion.ops.FermionOperator(((a, 1), (b, 0)), element)

                # down-down term
                v_op += openfermion.ops.FermionOperator(((a + 1, 1), (b + 1, 0)), element)

    # add two-particle operators
    for quad in quads:
        alpha, beta, gamma, delta = quad
        element = matrix_elements[alpha, beta, gamma, delta]

        # up-up-up-up term
        a = 2 * active.index(alpha)
        b = 2 * active.index(beta)
        g = 2 * active.index(gamma)
        d = 2 * active.index(delta)
        v_op += openfermion.ops.FermionOperator(((a, 1), (b, 1), (g, 0), (d, 0)), 0.5 * element)

        # up-down-down-up term
        b = 2 * active.index(beta) + 1
        g = 2 * active.index(gamma) + 1
        v_op += openfermion.ops.FermionOperator(((a, 1), (b, 1), (g, 0), (d, 0)), 0.5 * element)

        # down-up-up-down term
        a = 2 * active.index(alpha) + 1
        b = 2 * active.index(beta)
        g = 2 * active.index(gamma)
        d = 2 * active.index(delta) + 1
        v_op += openfermion.ops.FermionOperator(((a, 1), (b, 1), (g, 0), (d, 0)), 0.5 * element)

        # down-down-down-down term
        b = 2 * active.index(beta) + 1
        g = 2 * active.index(gamma) + 1
        v_op += openfermion.ops.FermionOperator(((a, 1), (b, 1), (g, 0), (d, 0)), 0.5 * element)

    return v_op


def dipole_of(
    symbols,
    coordinates,
    name="molecule",
    charge=0,
    mult=1,
    basis="sto-3g",
    package="pyscf",
    core=None,
    active=None,
    mapping="jordan_wigner",
    cutoff=1.0e-12,
    outpath=".",
    wires=None,
):
    r"""Computes the electric dipole moment operator in the Pauli basis.

    The second quantized dipole moment operator :math:`\hat{D}` of a molecule is given by

    .. math::

        \hat{D} = -\sum_{\alpha, \beta} \langle \alpha \vert \hat{{\bf r}} \vert \beta \rangle
        [\hat{c}_{\alpha\uparrow}^\dagger \hat{c}_{\beta\uparrow} +
        \hat{c}_{\alpha\downarrow}^\dagger \hat{c}_{\beta\downarrow}] + \hat{D}_\mathrm{n}.

    In the equation above, the indices :math:`\alpha, \beta` run over the basis of Hartree-Fock
    molecular orbitals and the operators :math:`\hat{c}^\dagger` and :math:`\hat{c}` are the
    electron creation and annihilation operators, respectively. The matrix elements of the
    position operator :math:`\hat{{\bf r}}` are computed as

    .. math::

        \langle \alpha \vert \hat{{\bf r}} \vert \beta \rangle = \sum_{i, j}
         C_{\alpha i}^*C_{\beta j} \langle i \vert \hat{{\bf r}} \vert j \rangle,

    where :math:`\vert i \rangle` is the wave function of the atomic orbital,
    :math:`C_{\alpha i}` are the coefficients defining the molecular orbitals,
    and :math:`\langle i \vert \hat{{\bf r}} \vert j \rangle`
    is the representation of operator :math:`\hat{{\bf r}}` in the atomic basis.

    The contribution of the nuclei to the dipole operator is given by

    .. math::

        \hat{D}_\mathrm{n} = \sum_{i=1}^{N_\mathrm{atoms}} Z_i {\bf R}_i \hat{I},


    where :math:`Z_i` and :math:`{\bf R}_i` denote, respectively, the atomic number and the
    nuclear coordinates of the :math:`i`-th atom of the molecule.

    Args:
        symbols (list[str]): symbols of the atomic species in the molecule
        coordinates (array[float]): 1D array with the atomic positions in Cartesian
            coordinates. The coordinates must be given in atomic units and the size of the array
            should be ``3*N`` where ``N`` is the number of atoms.
        name (str): name of the molecule
        charge (int): charge of the molecule
        mult (int): spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1` of the
            Hartree-Fock (HF) state based on the number of unpaired electrons occupying the
            HF orbitals
        basis (str): Atomic basis set used to represent the molecular orbitals. Basis set
            availability per element can be found
            `here <www.psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement>`_
        package (str): quantum chemistry package (pyscf or psi4) used to solve the
            mean field electronic structure problem
        core (list): indices of core orbitals
        active (list): indices of active orbitals
        mapping (str): transformation (``'jordan_wigner'`` or ``'bravyi_kitaev'``) used to
            map the fermionic operator to the Pauli basis
        cutoff (float): Cutoff value for including the matrix elements
            :math:`\langle \alpha \vert \hat{{\bf r}} \vert \beta \rangle`. The matrix elements
            with absolute value less than ``cutoff`` are neglected.
        outpath (str): path to the directory containing output files
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert the qubit operator
            to an observable measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).

    Returns:
        list[pennylane.Hamiltonian]: the qubit observables corresponding to the components
        :math:`\hat{D}_x`, :math:`\hat{D}_y` and :math:`\hat{D}_z` of the dipole operator in
        atomic units.

    **Example**

    >>> symbols = ["H", "H", "H"]
    >>> coordinates = np.array([0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0])
    >>> dip_obs = dipole(symbols, coordinates, charge=1)
    >>> print(dipole_obs)
    [<Hamiltonian: terms=19, wires=[0, 1, 2, 3, 4, 5]>,
    <Hamiltonian: terms=19, wires=[0, 1, 2, 3, 4, 5]>,
    <Hamiltonian: terms=1, wires=[0]>]

    >>> print(dip_obs[0]) # x-component of D
    (0.24190977644628117) [Z4]
    + (0.24190977644628117) [Z5]
    + (0.4781123173263878) [Z0]
    + (0.4781123173263878) [Z1]
    + (0.714477906181248) [Z2]
    + (0.714477906181248) [Z3]
    + (-0.3913638489487808) [Y0 Z1 Y2]
    + (-0.3913638489487808) [X0 Z1 X2]
    + (-0.3913638489487808) [Y1 Z2 Y3]
    + (-0.3913638489487808) [X1 Z2 X3]
    + (-0.1173495878099553) [Y2 Z3 Y4]
    + (-0.1173495878099553) [X2 Z3 X4]
    + (-0.1173495878099553) [Y3 Z4 Y5]
    + (-0.1173495878099553) [X3 Z4 X5]
    + (0.26611147045300276) [Y0 Z1 Z2 Z3 Y4]
    + (0.26611147045300276) [X0 Z1 Z2 Z3 X4]
    + (0.26611147045300276) [Y1 Z2 Z3 Z4 Y5]
    + (0.26611147045300276) [X1 Z2 Z3 Z4 X5]
    """

    atomic_numbers = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
    }

    if mult != 1:
        raise ValueError(
            f"Currently, this functionality is constrained to Hartree-Fock states with spin multiplicity = 1;"
            f" got multiplicity 2S+1 =  {mult}"
        )

    for i in symbols:
        if i not in atomic_numbers:
            raise ValueError(
                f"Currently, only first- or second-row elements of the periodic table are supported;"
                f" got element {i}"
            )

    hf_file = qml.qchem.meanfield(symbols, coordinates, name, charge, mult, basis, package, outpath)

    hf = openfermion.MolecularData(filename=hf_file.strip())

    # Load dipole matrix elements in the atomic basis
    # pylint: disable=import-outside-toplevel
    from pyscf import gto

    mol = gto.M(
        atom=hf.geometry, basis=hf.basis, charge=hf.charge, spin=0.5 * (hf.multiplicity - 1)
    )
    dip_ao = mol.intor_symmetric("int1e_r", comp=3).real

    # Transform dipole matrix elements to the MO basis
    n_orbs = hf.n_orbitals
    c_hf = hf.canonical_orbitals

    dip_mo = np.zeros((3, n_orbs, n_orbs))
    for comp in range(3):
        for alpha in range(n_orbs):
            for beta in range(alpha + 1):
                dip_mo[comp, alpha, beta] = c_hf[:, alpha] @ dip_ao[comp] @ c_hf[:, beta]

        dip_mo[comp] += dip_mo[comp].T - np.diag(np.diag(dip_mo[comp]))

    # Compute the nuclear contribution
    dip_n = np.zeros(3)
    for comp in range(3):
        for i, symb in enumerate(symbols):
            dip_n[comp] += atomic_numbers[symb] * coordinates[3 * i + comp]

    # Build the observable
    dip = []
    for i in range(3):
        fermion_obs = one_particle(dip_mo[i], core=core, active=active, cutoff=cutoff)
        dip.append(observable([-fermion_obs], init_term=dip_n[i], mapping=mapping, wires=wires))

    return dip


def _exec_exists(prog):
    r"""Checks whether the executable program ``prog`` exists in any of the directories
    set in the ``PATH`` environment variable.

    Args:
        prog (str): name of the executable program

    Returns:
        boolean: ``True`` if the executable ``prog`` is found; ``False`` otherwise
    """
    for dir_in_path in os.environ["PATH"].split(os.pathsep):
        path_to_prog = os.path.join(dir_in_path, prog)
        if os.path.exists(path_to_prog):
            try:
                subprocess.call([path_to_prog], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except OSError:
                return False
            return True
    return False


def read_structure(filepath, outpath="."):
    r"""Reads the structure of the polyatomic system from a file and returns
    a list with the symbols of the atoms in the molecule and a 1D array
    with their positions :math:`[x_1, y_1, z_1, x_2, y_2, z_2, \dots]` in
    atomic units (Bohr radius = 1).

    The atomic coordinates in the file must be in Angstroms.
    The `xyz <https://en.wikipedia.org/wiki/XYZ_file_format>`_ format is supported out of the box.
    If `Open Babel <https://openbabel.org/>`_ is installed,
    `any format recognized by Open Babel <https://openbabel.org/wiki/Category:Formats>`_
    is also supported. Additionally, the new file ``structure.xyz``,
    containing the input geometry, is created in a directory with path given by ``outpath``.

    Open Babel can be installed using ``apt`` if on Ubuntu:

    .. code-block:: bash

        sudo apt install openbabel

    or using Anaconda:

    .. code-block:: bash

        conda install -c conda-forge openbabel

    See the Open Babel documentation for more details on installation.

    Args:
        filepath (str): name of the molecular structure file in the working directory
            or the absolute path to the file if it is located in a different folder
        outpath (str): path to the output directory

    Returns:
        tuple[list, array]: symbols of the atoms in the molecule and a 1D array with their
        positions in atomic units.

    **Example**

    >>> symbols, coordinates = read_structure('h2.xyz')
    >>> print(symbols, coordinates)
    ['H', 'H'] [0.    0.   -0.66140414    0.    0.    0.66140414]
    """

    obabel_error_message = (
        "Open Babel converter not found:\n"
        "If using Ubuntu or Debian, try: 'sudo apt install openbabel' \n"
        "If using openSUSE, try: 'sudo zypper install openbabel' \n"
        "If using CentOS or Fedora, try: 'sudo snap install openbabel' "
        "Open Babel can also be downloaded from http://openbabel.org/wiki/Main_Page, \n"
        "make sure you add it to the PATH environment variable. \n"
        "If Anaconda is installed, try: 'conda install -c conda-forge openbabel'"
    )

    extension = filepath.split(".")[-1].strip().lower()

    file_in = filepath.strip()
    file_out = os.path.join(outpath, "structure.xyz")

    if extension != "xyz":
        if not _exec_exists("obabel"):
            raise TypeError(obabel_error_message)
        try:
            subprocess.run(
                ["obabel", "-i" + extension, file_in, "-oxyz", "-O", file_out], check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Open Babel error. See the following Open Babel "
                f"output for details:\n\n {e.stdout}\n{e.stderr}"
            ) from e
    else:
        copyfile(file_in, file_out)

    symbols = []
    coordinates = []
    with open(file_out, encoding="utf-8") as f:
        for line in f.readlines()[2:]:
            symbol, x, y, z = line.split()
            symbols.append(symbol)
            coordinates.append(float(x))
            coordinates.append(float(y))
            coordinates.append(float(z))

    return symbols, np.array(coordinates) / bohr_angs


def meanfield(
    symbols,
    coordinates,
    name="molecule",
    charge=0,
    mult=1,
    basis="sto-3g",
    package="pyscf",
    outpath=".",
):  # pylint: disable=too-many-arguments
    r"""Generates a file from which the mean field electronic structure
    of the molecule can be retrieved.

    This function uses OpenFermion-PySCF and OpenFermion-Psi4 plugins to
    perform the Hartree-Fock (HF) calculation for the polyatomic system using the quantum
    chemistry packages ``PySCF`` and ``Psi4``, respectively. The mean field electronic
    structure is saved in an hdf5-formatted file.

    The charge of the molecule can be given to simulate cationic/anionic systems.
    Also, the spin multiplicity can be input to determine the number of unpaired electrons
    occupying the HF orbitals as illustrated in the figure below.

    |

    .. figure:: ../../_static/qchem/hf_references.png
        :align: center
        :width: 50%

    |

    Args:
        symbols (list[str]): symbols of the atomic species in the molecule
        coordinates (array[float]): 1D array with the atomic positions in Cartesian
            coordinates. The coordinates must be given in atomic units and the size of the array
            should be ``3*N`` where ``N`` is the number of atoms.
        name (str): molecule label
        charge (int): net charge of the system
        mult (int): Spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1` for
            :math:`N_\mathrm{unpaired}` unpaired electrons occupying the HF orbitals.
            Possible values for ``mult`` are :math:`1, 2, 3, \ldots`. If not specified,
            a closed-shell HF state is assumed.
        basis (str): Atomic basis set used to represent the HF orbitals. Basis set
            availability per element can be found
            `here <www.psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement>`_
        package (str): Quantum chemistry package used to solve the Hartree-Fock equations.
            Either ``'pyscf'`` or ``'psi4'`` can be used.
        outpath (str): path to output directory

    Returns:
        str: absolute path to the file containing the mean field electronic structure

    **Example**

    >>> symbols, coordinates = (['H', 'H'], np.array([0., 0., -0.66140414, 0., 0., 0.66140414]))
    >>> meanfield(symbols, coordinates, name="h2")
    ./h2_pyscf_sto-3g
    """

    if coordinates.size != 3 * len(symbols):
        raise ValueError(
            f"The size of the array 'coordinates' has to be 3*len(symbols) = {3 * len(symbols)};"
            f" got 'coordinates.size' = {coordinates.size}"
        )

    package = package.strip().lower()

    if package not in ("psi4", "pyscf"):
        error_message = (
            f"Integration with quantum chemistry package '{package}' is not available. \n Please set"
            f" 'package' to 'pyscf' or 'psi4'."
        )
        raise TypeError(error_message)

    filename = name + "_" + package.lower() + "_" + basis.strip()
    path_to_file = os.path.join(outpath.strip(), filename)

    geometry = [
        [symbol, tuple(np.array(coordinates)[3 * i : 3 * i + 3] * bohr_angs)]
        for i, symbol in enumerate(symbols)
    ]

    molecule = openfermion.MolecularData(geometry, basis, mult, charge, filename=path_to_file)

    if package == "psi4":
        # pylint: disable=import-outside-toplevel
        from openfermionpsi4 import run_psi4

        run_psi4(molecule, run_scf=1, verbose=0, tolerate_error=1)

    if package == "pyscf":
        # pylint: disable=import-outside-toplevel
        from openfermionpyscf import run_pyscf

        run_pyscf(molecule, run_scf=1, verbose=0)

    return path_to_file


def active_space(electrons, orbitals, mult=1, active_electrons=None, active_orbitals=None):
    r"""Builds the active space for a given number of active electrons and active orbitals.

    Post-Hartree-Fock (HF) electron correlation methods expand the many-body wave function
    as a linear combination of Slater determinants, commonly referred to as configurations.
    This configurations are generated by exciting electrons from the occupied to the
    unoccupied HF orbitals as sketched in the figure below. Since the number of configurations
    increases combinatorially with the number of electrons and orbitals this expansion can be
    truncated by defining an active space.

    The active space is created by classifying the HF orbitals as core, active and
    external orbitals:

    - Core orbitals are always occupied by two electrons
    - Active orbitals can be occupied by zero, one, or two electrons
    - The external orbitals are never occupied

    |

    .. figure:: ../../_static/qchem/sketch_active_space.png
        :align: center
        :width: 50%

    |

    .. note::
        The number of active *spin*-orbitals ``2*active_orbitals`` determines the number of
        qubits required to perform the quantum simulations of the electronic structure
        of the many-electron system.

    Args:
        electrons (int): total number of electrons
        orbitals (int): total number of orbitals
        mult (int): Spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1` for
            :math:`N_\mathrm{unpaired}` unpaired electrons occupying the HF orbitals.
            Possible values for ``mult`` are :math:`1, 2, 3, \ldots`. If not specified,
            a closed-shell HF state is assumed.
        active_electrons (int): Number of active electrons. If not specified, all electrons
            are treated as active.
        active_orbitals (int): Number of active orbitals. If not specified, all orbitals
            are treated as active.

    Returns:
        tuple: lists of indices for core and active orbitals

    **Example**

    >>> electrons = 4
    >>> orbitals = 4
    >>> core, active = active_space(electrons, orbitals, active_electrons=2, active_orbitals=2)
    >>> print(core) # core orbitals
    [0]
    >>> print(active) # active orbitals
    [1, 2]
    """
    # pylint: disable=too-many-branches

    if active_electrons is None:
        ncore_orbs = 0
        core = []
    else:
        if active_electrons <= 0:
            raise ValueError(
                f"The number of active electrons ({active_electrons}) " f"has to be greater than 0."
            )

        if active_electrons > electrons:
            raise ValueError(
                f"The number of active electrons ({active_electrons}) "
                f"can not be greater than the total "
                f"number of electrons ({electrons})."
            )

        if active_electrons < mult - 1:
            raise ValueError(
                f"For a reference state with multiplicity {mult}, "
                f"the number of active electrons ({active_electrons}) should be "
                f"greater than or equal to {mult - 1}."
            )

        if mult % 2 == 1:
            if active_electrons % 2 != 0:
                raise ValueError(
                    f"For a reference state with multiplicity {mult}, "
                    f"the number of active electrons ({active_electrons}) should be even."
                )
        else:
            if active_electrons % 2 != 1:
                raise ValueError(
                    f"For a reference state with multiplicity {mult}, "
                    f"the number of active electrons ({active_electrons}) should be odd."
                )

        ncore_orbs = (electrons - active_electrons) // 2
        core = list(range(ncore_orbs))

    if active_orbitals is None:
        active = list(range(ncore_orbs, orbitals))
    else:
        if active_orbitals <= 0:
            raise ValueError(
                f"The number of active orbitals ({active_orbitals}) " f"has to be greater than 0."
            )

        if ncore_orbs + active_orbitals > orbitals:
            raise ValueError(
                f"The number of core ({ncore_orbs}) + active orbitals ({active_orbitals}) can not be "
                f"greater than the total number of orbitals ({orbitals})"
            )

        homo = (electrons + mult - 1) / 2
        if ncore_orbs + active_orbitals <= homo:
            raise ValueError(
                f"For n_active_orbitals={active_orbitals}, there are no virtual orbitals "
                f"in the active space."
            )

        active = list(range(ncore_orbs, ncore_orbs + active_orbitals))

    return core, active


def decompose(hf_file, mapping="jordan_wigner", core=None, active=None):
    r"""Decomposes the molecular Hamiltonian into a linear combination of Pauli operators using
    OpenFermion tools.

    This function uses OpenFermion functions to build the second-quantized electronic Hamiltonian
    of the molecule and map it to the Pauli basis using the Jordan-Wigner or Bravyi-Kitaev
    transformation.

    Args:
        hf_file (str): absolute path to the hdf5-formatted file with the
            Hartree-Fock electronic structure
        mapping (str): Specifies the transformation to map the fermionic Hamiltonian to the
            Pauli basis. Input values can be ``'jordan_wigner'`` or ``'bravyi_kitaev'``.
        core (list): indices of core orbitals, i.e., the orbitals that are
            not correlated in the many-body wave function
        active (list): indices of active orbitals, i.e., the orbitals used to
            build the correlated many-body wave function

    Returns:
        QubitOperator: an instance of OpenFermion's ``QubitOperator``

    **Example**

    >>> decompose('./pyscf/sto-3g/h2', mapping='bravyi_kitaev')
    (-0.04207897696293986+0j) [] + (0.04475014401986122+0j) [X0 Z1 X2] +
    (0.04475014401986122+0j) [X0 Z1 X2 Z3] +(0.04475014401986122+0j) [Y0 Z1 Y2] +
    (0.04475014401986122+0j) [Y0 Z1 Y2 Z3] +(0.17771287459806262+0j) [Z0] +
    (0.17771287459806265+0j) [Z0 Z1] +(0.1676831945625423+0j) [Z0 Z1 Z2] +
    (0.1676831945625423+0j) [Z0 Z1 Z2 Z3] +(0.12293305054268105+0j) [Z0 Z2] +
    (0.12293305054268105+0j) [Z0 Z2 Z3] +(0.1705973832722409+0j) [Z1] +
    (-0.2427428049645989+0j) [Z1 Z2 Z3] +(0.1762764080276107+0j) [Z1 Z3] +
    (-0.2427428049645989+0j) [Z2]
    """

    # loading HF data from the hdf5 file
    molecule = openfermion.MolecularData(filename=hf_file.strip())

    # getting the terms entering the second-quantized Hamiltonian
    terms_molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=core, active_indices=active
    )

    # generating the fermionic Hamiltonian
    fermionic_hamiltonian = openfermion.transforms.get_fermion_operator(terms_molecular_hamiltonian)

    mapping = mapping.strip().lower()

    if mapping not in ("jordan_wigner", "bravyi_kitaev"):
        raise TypeError(
            f"The '{mapping}' transformation is not available. \n "
            f"Please set 'mapping' to 'jordan_wigner' or 'bravyi_kitaev'."
        )

    # fermionic-to-qubit transformation of the Hamiltonian
    if mapping == "bravyi_kitaev":
        return openfermion.transforms.bravyi_kitaev(fermionic_hamiltonian)

    return openfermion.transforms.jordan_wigner(fermionic_hamiltonian)


def molecular_hamiltonian(
    symbols,
    coordinates,
    name="molecule",
    charge=0,
    mult=1,
    basis="sto-3g",
    package="pyscf",
    active_electrons=None,
    active_orbitals=None,
    mapping="jordan_wigner",
    outpath=".",
    wires=None,
):  # pylint:disable=too-many-arguments
    r"""Generates the qubit Hamiltonian of a molecule.

    This function drives the construction of the second-quantized electronic Hamiltonian
    of a molecule and its transformation to the basis of Pauli matrices.

    #. OpenFermion-PySCF or OpenFermion-Psi4 plugins are used to launch
       the Hartree-Fock (HF) calculation for the polyatomic system using the quantum
       chemistry package ``PySCF`` or ``Psi4``, respectively.

       - The net charge of the molecule can be given to simulate
         cationic/anionic systems. Also, the spin multiplicity can be input
         to determine the number of unpaired electrons occupying the HF orbitals
         as illustrated in the left panel of the figure below.

       - The basis of Gaussian-type *atomic* orbitals used to represent the *molecular* orbitals
         can be specified to go beyond the minimum basis approximation. Basis set availability
         per element can be found
         `here <www.psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement>`_

    #. An active space can be defined for a given number of *active electrons*
       occupying a reduced set of *active orbitals* in the vicinity of the frontier
       orbitals as sketched in the right panel of the figure below.

    #. Finally, the second-quantized Hamiltonian is mapped to the Pauli basis and
       converted to a PennyLane observable.

    |

    .. figure:: ../../_static/qchem/fig_mult_active_space.png
        :align: center
        :width: 90%

    |

    Args:
        symbols (list[str]): symbols of the atomic species in the molecule
        coordinates (array[float]): 1D array with the atomic positions in Cartesian
            coordinates. The coordinates must be given in atomic units and the size of the array
            should be ``3*N`` where ``N`` is the number of atoms.
        name (str): name of the molecule
        charge (int): Net charge of the molecule. If not specified a a neutral system is assumed.
        mult (int): Spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1`
            for :math:`N_\mathrm{unpaired}` unpaired electrons occupying the HF orbitals.
            Possible values of ``mult`` are :math:`1, 2, 3, \ldots`. If not specified,
            a closed-shell HF state is assumed.
        basis (str): Atomic basis set used to represent the molecular orbitals. Basis set
            availability per element can be found
            `here <www.psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement>`_
        package (str): quantum chemistry package (pyscf or psi4) used to solve the
            mean field electronic structure problem
        active_electrons (int): Number of active electrons. If not specified, all electrons
            are considered to be active.
        active_orbitals (int): Number of active orbitals. If not specified, all orbitals
            are considered to be active.
        mapping (str): transformation (``'jordan_wigner'`` or ``'bravyi_kitaev'``) used to
            map the fermionic Hamiltonian to the qubit Hamiltonian
        outpath (str): path to the directory containing output files
        wires (Wires, list, tuple, dict): Custom wire mapping for connecting to Pennylane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted for
            partial mapping. If None, will use identity map.

    Returns:
        tuple[pennylane.Hamiltonian, int]: the fermionic-to-qubit transformed Hamiltonian
        and the number of qubits

    **Example**

    >>> symbols, coordinates = (['H', 'H'], np.array([0., 0., -0.66140414, 0., 0., 0.66140414]))
    >>> H, qubits = molecular_hamiltonian(symbols, coordinates)
    >>> print(qubits)
    4
    >>> print(H)
    (-0.04207897647782188) [I0]
    + (0.17771287465139934) [Z0]
    + (0.1777128746513993) [Z1]
    + (-0.24274280513140484) [Z2]
    + (-0.24274280513140484) [Z3]
    + (0.17059738328801055) [Z0 Z1]
    + (0.04475014401535161) [Y0 X1 X2 Y3]
    + (-0.04475014401535161) [Y0 Y1 X2 X3]
    + (-0.04475014401535161) [X0 X1 Y2 Y3]
    + (0.04475014401535161) [X0 Y1 Y2 X3]
    + (0.12293305056183801) [Z0 Z2]
    + (0.1676831945771896) [Z0 Z3]
    + (0.1676831945771896) [Z1 Z2]
    + (0.12293305056183801) [Z1 Z3]
    + (0.176276408043196) [Z2 Z3]
    """

    hf_file = meanfield(symbols, coordinates, name, charge, mult, basis, package, outpath)

    molecule = openfermion.MolecularData(filename=hf_file)

    core, active = active_space(
        molecule.n_electrons, molecule.n_orbitals, mult, active_electrons, active_orbitals
    )

    h_of, qubits = (decompose(hf_file, mapping, core, active), 2 * len(active))

    return qml.qchem.convert.import_operator(h_of, wires=wires), qubits


def excitations(electrons, orbitals, delta_sz=0):
    r"""Generates single and double excitations from a Hartree-Fock reference state.

    Single and double excitations can be generated by acting with the operators
    :math:`\hat T_1` and :math:`\hat T_2` on the Hartree-Fock reference state:

    .. math::

        && \hat{T}_1 = \sum_{r \in \mathrm{occ} \\ p \in \mathrm{unocc}}
        \hat{c}_p^\dagger \hat{c}_r \\
        && \hat{T}_2 = \sum_{r>s \in \mathrm{occ} \\ p>q \in
        \mathrm{unocc}} \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s.


    In the equations above the indices :math:`r, s` and :math:`p, q` run over the
    occupied (occ) and unoccupied (unocc) *spin* orbitals and :math:`\hat c` and
    :math:`\hat c^\dagger` are the electron annihilation and creation operators,
    respectively.

    |

    .. figure:: ../../_static/qchem/sd_excitations.png
        :align: center
        :width: 80%

    |

    Args:
        electrons (int): Number of electrons. If an active space is defined, this
            is the number of active electrons.
        orbitals (int): Number of *spin* orbitals. If an active space is defined,
            this is the number of active spin-orbitals.
        delta_sz (int): Specifies the selection rules ``sz[p] - sz[r] = delta_sz`` and
            ``sz[p] + sz[p] - sz[r] - sz[s] = delta_sz`` for the spin-projection ``sz`` of
            the orbitals involved in the single and double excitations, respectively.
            ``delta_sz`` can take the values :math:`0`, :math:`\pm 1` and :math:`\pm 2`.

    Returns:
        tuple(list, list): lists with the indices of the spin orbitals involved in the
        single and double excitations

    **Example**

    >>> electrons = 2
    >>> orbitals = 4
    >>> singles, doubles = excitations(electrons, orbitals)
    >>> print(singles)
    [[0, 2], [1, 3]]
    >>> print(doubles)
    [[0, 1, 2, 3]]
    """

    if not electrons > 0:
        raise ValueError(
            f"The number of active electrons has to be greater than 0 \n"
            f"Got n_electrons = {electrons}"
        )

    if orbitals <= electrons:
        raise ValueError(
            f"The number of active spin-orbitals ({orbitals}) "
            f"has to be greater than the number of active electrons ({electrons})."
        )

    if delta_sz not in (0, 1, -1, 2, -2):
        raise ValueError(
            f"Expected values for 'delta_sz' are 0, +/- 1 and +/- 2 but got ({delta_sz})."
        )

    # define the spin projection 'sz' of the single-particle states
    sz = np.array([0.5 if (i % 2 == 0) else -0.5 for i in range(orbitals)])

    singles = [
        [r, p]
        for r in range(electrons)
        for p in range(electrons, orbitals)
        if sz[p] - sz[r] == delta_sz
    ]

    doubles = [
        [s, r, q, p]
        for s in range(electrons - 1)
        for r in range(s + 1, electrons)
        for q in range(electrons, orbitals - 1)
        for p in range(q + 1, orbitals)
        if (sz[p] + sz[q] - sz[r] - sz[s]) == delta_sz
    ]

    return singles, doubles


def hf_state(electrons, orbitals):
    r"""Generates the occupation-number vector representing the Hartree-Fock state.

    The many-particle wave function in the Hartree-Fock (HF) approximation is a `Slater determinant
    <https://en.wikipedia.org/wiki/Slater_determinant>`_. In Fock space, a Slater determinant
    for :math:`N` electrons is represented by the occupation-number vector:

    .. math::

        \vert {\bf n} \rangle = \vert n_1, n_2, \dots, n_\mathrm{orbs} \rangle,
        n_i = \left\lbrace \begin{array}{ll} 1 & i \leq N \\ 0 & i > N \end{array} \right.,

    where :math:`n_i` indicates the occupation of the :math:`i`-th orbital.

    Args:
        electrons (int): Number of electrons. If an active space is defined, this
            is the number of active electrons.
        orbitals (int): Number of *spin* orbitals. If an active space is defined,
            this is the number of active spin-orbitals.

    Returns:
        array: NumPy array containing the vector :math:`\vert {\bf n} \rangle`

    **Example**

    >>> state = hf_state(2, 6)
    >>> print(state)
    [1 1 0 0 0 0]
    """

    if electrons <= 0:
        raise ValueError(
            f"The number of active electrons has to be larger than zero; got 'electrons' = {electrons}"
        )

    if electrons > orbitals:
        raise ValueError(
            f"The number of active orbitals cannot be smaller than the number of active electrons;"
            f" got 'orbitals'={orbitals} < 'electrons'={electrons}"
        )

    state = np.where(np.arange(orbitals) < electrons, 1, 0)

    return np.array(state)


def excitations_to_wires(singles, doubles, wires=None):
    r"""Map the indices representing the single and double excitations
    generated with the function :func:`~.excitations` to the wires that
    the Unitary Coupled-Cluster (UCCSD) template will act on.

    Args:
        singles (list[list[int]]): list with the indices ``r``, ``p`` of the two qubits
            representing the single excitation
            :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF}\rangle`
        doubles (list[list[int]]): list with the indices ``s``, ``r``, ``q``, ``p`` of the four
            qubits representing the double excitation
            :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger
            \hat{c}_r \hat{c}_s \vert \mathrm{HF}\rangle`
        wires (Iterable[Any]): Wires of the quantum device. If None, will use consecutive wires.

    The indices :math:`r, s` and :math:`p, q` in these lists correspond, respectively, to the
    occupied and virtual orbitals involved in the generated single and double excitations.

    Returns:
        tuple[list[list[Any]], list[list[list[Any]]]]: lists with the sequence of wires,
        resulting from the single and double excitations, that the Unitary Coupled-Cluster
        (UCCSD) template will act on.

    **Example**

    >>> singles = [[0, 2], [1, 3]]
    >>> doubles = [[0, 1, 2, 3]]
    >>> singles_wires, doubles_wires = excitations_to_wires(singles, doubles)
    >>> print(single_wires)
    [[0, 1, 2], [1, 2, 3]]
    >>> print(doubles_wires)
    [[[0, 1], [2, 3]]]

    >>> wires=['a0', 'b1', 'c2', 'd3']
    >>> singles_wires, doubles_wires = excitations_to_wires(singles, doubles, wires=wires)
    >>> print(singles_wires)
    [['a0', 'b1', 'c2'], ['b1', 'c2', 'd3']]
    >>> print(doubles_wires)
    [[['a0', 'b1'], ['c2', 'd3']]]
    """

    if (not singles) and (not doubles):
        raise ValueError(
            f"'singles' and 'doubles' lists can not be both empty; "
            f"got singles = {singles}, doubles = {doubles}"
        )

    expected_shape = (2,)
    for single_ in singles:
        if np.array(single_).shape != expected_shape:
            raise ValueError(
                f"Expected entries of 'singles' to be of shape (2,); got {np.array(single_).shape}"
            )

    expected_shape = (4,)
    for double_ in doubles:
        if np.array(double_).shape != expected_shape:
            raise ValueError(
                f"Expected entries of 'doubles' to be of shape (4,); got {np.array(double_).shape}"
            )

    max_idx = 0
    if singles:
        max_idx = np.max(singles)
    if doubles:
        max_idx = max(np.max(doubles), max_idx)

    if wires is None:
        wires = range(max_idx + 1)
    elif len(wires) != max_idx + 1:
        raise ValueError(f"Expected number of wires is {max_idx + 1}; got {len(wires)}")

    singles_wires = []
    for r, p in singles:
        s_wires = [wires[i] for i in range(r, p + 1)]
        singles_wires.append(s_wires)

    doubles_wires = []
    for s, r, q, p in doubles:
        d1_wires = [wires[i] for i in range(s, r + 1)]
        d2_wires = [wires[i] for i in range(q, p + 1)]
        doubles_wires.append([d1_wires, d2_wires])

    return singles_wires, doubles_wires
