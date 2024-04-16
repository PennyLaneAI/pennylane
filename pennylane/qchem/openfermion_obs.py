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
"""This module contains functions to construct many-body observables with ``OpenFermion-PySCF``.
"""
# pylint: disable=too-many-arguments, too-few-public-methods, too-many-branches, unused-variable
# pylint: disable=consider-using-generator, protected-access
import os

import numpy as np

import pennylane as qml
from pennylane.operation import active_new_opmath
from pennylane.pauli.utils import simplify


# Bohr-Angstrom correlation coefficient (https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0)
bohr_angs = 0.529177210903


def _import_of():
    """Import openfermion and openfermionpyscf."""
    try:
        # pylint: disable=import-outside-toplevel, unused-import, multiple-imports
        import openfermion, openfermionpyscf
    except ImportError as Error:
        raise ImportError(
            "This feature requires openfermionpyscf. "
            "It can be installed with: pip install openfermionpyscf."
        ) from Error

    return openfermion, openfermionpyscf


def _import_pyscf():
    """Import pyscf."""
    try:
        # pylint: disable=import-outside-toplevel, unused-import, multiple-imports
        import pyscf
    except ImportError as Error:
        raise ImportError(
            "This feature requires pyscf. It can be installed with: pip install pyscf."
        ) from Error

    return pyscf


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
    >>> observable([t, v], mapping="jordan_wigner")
    (
        0.2625 * I(0)
      + -0.1375 * Z(0)
      + -0.0875 * Z(1)
      + -0.0375 * (Z(0) @ Z(1))
      + 0.075 * Z(2)
      + -0.075 * (Z(0) @ Z(2))
    )
    """
    openfermion, _ = _import_of()

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

    >>> import numpy as np
    >>> matrix_elements = np.array([[-1.27785301e+00,  0.00000000e+00],
    ...                             [ 1.52655666e-16, -4.48299696e-01]])
    >>> t_op = one_particle(matrix_elements)
    >>> print(t_op)
    -1.277853006156875 [0^ 0] +
    -1.277853006156875 [1^ 1] +
    -0.44829969610163756 [2^ 2] +
    -0.44829969610163756 [3^ 3]
    """
    openfermion, _ = _import_of()

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

    >>> import numpy as np
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
    0.3411947665 [0^ 0^ 0 0] +
    0.089500288 [0^ 0^ 2 2] +
    0.3411947665 [0^ 1^ 1 0] +
    0.089500288 [0^ 1^ 3 2] +
    0.335366389 [0^ 2^ 2 0] +
    0.335366389 [0^ 3^ 3 0] +
    0.3411947665 [1^ 0^ 0 1] +
    0.089500288 [1^ 0^ 2 3] +
    0.3411947665 [1^ 1^ 1 1] +
    0.089500288 [1^ 1^ 3 3] +
    0.335366389 [1^ 2^ 2 1] +
    0.335366389 [1^ 3^ 3 1] +
    0.089500288 [2^ 0^ 2 0] +
    0.089500288 [2^ 1^ 3 0] +
    0.352552816 [2^ 2^ 2 2] +
    0.352552816 [2^ 3^ 3 2] +
    0.089500288 [3^ 0^ 2 1] +
    0.089500288 [3^ 1^ 3 1] +
    0.352552816 [3^ 2^ 2 3] +
    0.352552816 [3^ 3^ 3 3]
    """
    openfermion, _ = _import_of()

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
        package (str): quantum chemistry package (pyscf) used to solve the
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
    >>> dipole_obs = dipole_of(symbols, coordinates, charge=1)
    >>> print([(h.wires) for h in dipole_obs])
    [<Wires = [0, 1, 2, 3, 4, 5]>, <Wires = [0, 1, 2, 3, 4, 5]>, <Wires = [0]>]

    >>> dipole_obs[0] # x-component of D
    (
        0.4781123173263876 * Z(0)
      + 0.4781123173263876 * Z(1)
      + -0.3913638489489803 * (Y(0) @ Z(1) @ Y(2))
      + -0.3913638489489803 * (X(0) @ Z(1) @ X(2))
      + -0.3913638489489803 * (Y(1) @ Z(2) @ Y(3))
      + -0.3913638489489803 * (X(1) @ Z(2) @ X(3))
      + 0.2661114704527088 * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Y(4))
      + 0.2661114704527088 * (X(0) @ Z(1) @ Z(2) @ Z(3) @ X(4))
      + 0.2661114704527088 * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Y(5))
      + 0.2661114704527088 * (X(1) @ Z(2) @ Z(3) @ Z(4) @ X(5))
      + 0.7144779061810713 * Z(2)
      + 0.7144779061810713 * Z(3)
      + -0.11734958781031017 * (Y(2) @ Z(3) @ Y(4))
      + -0.11734958781031017 * (X(2) @ Z(3) @ X(4))
      + -0.11734958781031017 * (Y(3) @ Z(4) @ Y(5))
      + -0.11734958781031017 * (X(3) @ Z(4) @ X(5))
      + 0.24190977644645698 * Z(4)
      + 0.24190977644645698 * Z(5)
    )
    """
    openfermion, _ = _import_of()

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

    This function uses OpenFermion-PySCF plugins to
    perform the Hartree-Fock (HF) calculation for the polyatomic system using the quantum
    chemistry packages ``PySCF``. The mean field electronic
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
        outpath (str): path to output directory

    Returns:
        str: absolute path to the file containing the mean field electronic structure

    **Example**

    >>> symbols, coordinates = (['H', 'H'], np.array([0., 0., -0.66140414, 0., 0., 0.66140414]))
    >>> meanfield(symbols, coordinates, name="h2")
    ./h2_pyscf_sto-3g
    """
    openfermion, openfermionpyscf = _import_of()

    if coordinates.size != 3 * len(symbols):
        raise ValueError(
            f"The size of the array 'coordinates' has to be 3*len(symbols) = {3 * len(symbols)};"
            f" got 'coordinates.size' = {coordinates.size}"
        )

    package = package.strip().lower()

    if package not in "pyscf":
        error_message = (
            f"Integration with quantum chemistry package '{package}' is not available. \n Please set"
            f" 'package' to 'pyscf'."
        )
        raise TypeError(error_message)

    filename = name + "_" + package.lower() + "_" + basis.strip()
    path_to_file = os.path.join(outpath.strip(), filename)

    geometry = [
        [symbol, tuple(np.array(coordinates)[3 * i : 3 * i + 3] * bohr_angs)]
        for i, symbol in enumerate(symbols)
    ]

    molecule = openfermion.MolecularData(geometry, basis, mult, charge, filename=path_to_file)

    if package == "pyscf":
        # pylint: disable=import-outside-toplevel
        from openfermionpyscf import run_pyscf

        run_pyscf(molecule, run_scf=1, verbose=0)

    return path_to_file


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
    openfermion, _ = _import_of()

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
    method="dhf",
    active_electrons=None,
    active_orbitals=None,
    mapping="jordan_wigner",
    outpath=".",
    wires=None,
    alpha=None,
    coeff=None,
    args=None,
    load_data=False,
    convert_tol=1e012,
):  # pylint:disable=too-many-arguments, too-many-statements
    r"""Generate the qubit Hamiltonian of a molecule.

    This function drives the construction of the second-quantized electronic Hamiltonian
    of a molecule and its transformation to the basis of Pauli matrices.

    The net charge of the molecule can be given to simulate cationic/anionic systems. Also, the
    spin multiplicity can be input to determine the number of unpaired electrons occupying the HF
    orbitals as illustrated in the left panel of the figure below.

    The basis of Gaussian-type *atomic* orbitals used to represent the *molecular* orbitals can be
    specified to go beyond the minimum basis approximation.

    An active space can be defined for a given number of *active electrons* occupying a reduced set
    of *active orbitals* as sketched in the right panel of the figure below.

    |

    .. figure:: ../../_static/qchem/fig_mult_active_space.png
        :align: center
        :width: 90%

    |

    Args:
        symbols (list[str]): symbols of the atomic species in the molecule
        coordinates (array[float]): atomic positions in Cartesian coordinates.
            The atomic coordinates must be in atomic units and can be given as either a 1D array of
            size ``3*N``, or a 2D array of shape ``(N, 3)`` where ``N`` is the number of atoms.
        name (str): name of the molecule
        charge (int): Net charge of the molecule. If not specified a neutral system is assumed.
        mult (int): Spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1`
            for :math:`N_\mathrm{unpaired}` unpaired electrons occupying the HF orbitals.
            Possible values of ``mult`` are :math:`1, 2, 3, \ldots`. If not specified,
            a closed-shell HF state is assumed.
        basis (str): atomic basis set used to represent the molecular orbitals
        method (str): Quantum chemistry method used to solve the
            mean field electronic structure problem. Available options are ``method="dhf"``
            to specify the built-in differentiable Hartree-Fock solver, ``method="pyscf"`` to use
            the PySCF package (requires ``pyscf`` to be installed), or ``method="openfermion"`` to
            use the OpenFermion-PySCF plugin (this requires ``openfermionpyscf`` to be installed).
        active_electrons (int): Number of active electrons. If not specified, all electrons
            are considered to be active.
        active_orbitals (int): Number of active orbitals. If not specified, all orbitals
            are considered to be active.
        mapping (str): transformation used to map the fermionic Hamiltonian to the qubit Hamiltonian
        outpath (str): path to the directory containing output files
        wires (Wires, list, tuple, dict): Custom wire mapping for connecting to Pennylane ansatz.
            For types ``Wires``/``list``/``tuple``, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted for
            partial mapping. If None, will use identity map.
        alpha (array[float]): exponents of the primitive Gaussian functions
        coeff (array[float]): coefficients of the contracted Gaussian functions
        args (array[array[float]]): initial values of the differentiable parameters
        load_data (bool): flag to load data from the basis-set-exchange library
        convert_tol (float): Tolerance in `machine epsilon <https://numpy.org/doc/stable/reference/generated/numpy.real_if_close.html>`_
            for the imaginary part of the Hamiltonian coefficients created by openfermion.
            Coefficients with imaginary part less than 2.22e-16*tol are considered to be real.

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

    if method not in ["dhf", "pyscf", "openfermion"]:
        raise ValueError("Only 'dhf', 'pyscf' and 'openfermion' backends are supported.")

    if len(coordinates) == len(symbols) * 3:
        geometry_dhf = qml.numpy.array(coordinates.reshape(len(symbols), 3))
        geometry_hf = coordinates
    elif len(coordinates) == len(symbols):
        geometry_dhf = qml.numpy.array(coordinates)
        geometry_hf = coordinates.flatten()

    wires_map = None

    if wires:
        wires_new = qml.qchem.convert._process_wires(wires)
        wires_map = dict(zip(range(len(wires_new)), list(wires_new.labels)))

    if method == "dhf":

        if mapping != "jordan_wigner":
            raise ValueError(
                "Only 'jordan_wigner' mapping is supported for the differentiable workflow."
            )
        if mult != 1:
            raise ValueError(
                "Openshell systems are not supported for the differentiable workflow. Use "
                "`method = 'pyscf'` or change the charge or spin multiplicity of the molecule."
            )
        if args is None and isinstance(geometry_dhf, qml.numpy.tensor):
            geometry_dhf.requires_grad = False
        mol = qml.qchem.Molecule(
            symbols,
            geometry_dhf,
            charge=charge,
            mult=mult,
            basis_name=basis,
            load_data=load_data,
            alpha=alpha,
            coeff=coeff,
        )
        core, active = qml.qchem.active_space(
            mol.n_electrons, mol.n_orbitals, mult, active_electrons, active_orbitals
        )

        requires_grad = args is not None
        h = (
            qml.qchem.diff_hamiltonian(mol, core=core, active=active)(*args)
            if requires_grad
            else qml.qchem.diff_hamiltonian(mol, core=core, active=active)()
        )

        if active_new_opmath():
            h_as_ps = qml.pauli.pauli_sentence(h)
            coeffs = qml.numpy.real(list(h_as_ps.values()), requires_grad=requires_grad)

            h_as_ps = qml.pauli.PauliSentence(dict(zip(h_as_ps.keys(), coeffs)))
            h = (
                qml.s_prod(0, qml.Identity(h.wires[0]))
                if len(h_as_ps) == 0
                else h_as_ps.operation()
            )
        else:
            coeffs = qml.numpy.real(h.coeffs, requires_grad=requires_grad)
            h = qml.Hamiltonian(coeffs, h.ops)

        if wires:
            h = qml.map_wires(h, wires_map)
        return h, 2 * len(active)

    if method == "pyscf" and mapping.strip().lower() == "jordan_wigner":
        core_constant, one_mo, two_mo = _pyscf_integrals(
            symbols, geometry_hf, charge, mult, basis, active_electrons, active_orbitals
        )

        hf = qml.qchem.fermionic_observable(core_constant, one_mo, two_mo)

        if active_new_opmath():
            h_pl = qml.jordan_wigner(hf, wire_map=wires_map, tol=1.0e-10).simplify()

        else:
            h_pl = qml.jordan_wigner(hf, ps=True, wire_map=wires_map, tol=1.0e-10).hamiltonian()
            h_pl = simplify(h_pl)

        return h_pl, len(h_pl.wires)

    openfermion, _ = _import_of()

    hf_file = meanfield(symbols, geometry_hf, name, charge, mult, basis, "pyscf", outpath)

    molecule = openfermion.MolecularData(filename=hf_file)

    core, active = qml.qchem.active_space(
        molecule.n_electrons, molecule.n_orbitals, mult, active_electrons, active_orbitals
    )

    h_of, qubits = (decompose(hf_file, mapping, core, active), 2 * len(active))

    h_pl = qml.qchem.convert.import_operator(h_of, wires=wires, tol=convert_tol)

    return h_pl, len(h_pl.wires)


def _pyscf_integrals(
    symbols,
    coordinates,
    charge=0,
    mult=1,
    basis="sto-3g",
    active_electrons=None,
    active_orbitals=None,
):
    r"""Compute pyscf integrals."""

    pyscf = _import_pyscf()

    geometry = [
        [symbol, tuple(np.array(coordinates)[3 * i : 3 * i + 3] * bohr_angs)]
        for i, symbol in enumerate(symbols)
    ]

    # create the Mole object
    mol = pyscf.gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = charge
    mol.spin = mult - 1
    mol.verbose = 0

    # initialize the molecule
    mol_pyscf = mol.build()

    # run HF calculations
    if mult == 1:
        molhf = pyscf.scf.RHF(mol_pyscf)
    else:
        molhf = pyscf.scf.ROHF(mol_pyscf)
    _ = molhf.kernel()

    # compute atomic and molecular orbitals
    one_ao = mol_pyscf.intor_symmetric("int1e_kin") + mol_pyscf.intor_symmetric("int1e_nuc")
    two_ao = mol_pyscf.intor("int2e_sph")
    one_mo = np.einsum("pi,pq,qj->ij", molhf.mo_coeff, one_ao, molhf.mo_coeff, optimize=True)
    two_mo = pyscf.ao2mo.incore.full(two_ao, molhf.mo_coeff)

    core_constant = np.array([molhf.energy_nuc()])

    # convert the two-electron integral tensor to the physicists’ notation
    two_mo = np.swapaxes(two_mo, 1, 3)

    # define the active space and recompute the integrals
    core, active = qml.qchem.active_space(
        mol.nelectron, mol.nao, mult, active_electrons, active_orbitals
    )

    if core and active:
        for i in core:
            core_constant = core_constant + 2 * one_mo[i][i]
            for j in core:
                core_constant = core_constant + 2 * two_mo[i][j][j][i] - two_mo[i][j][i][j]

        for p in active:
            for q in active:
                for i in core:
                    one_mo[p, q] = one_mo[p, q] + (2 * two_mo[i][p][q][i] - two_mo[i][p][i][q])

        one_mo = one_mo[qml.math.ix_(active, active)]
        two_mo = two_mo[qml.math.ix_(active, active, active, active)]

    return core_constant, one_mo, two_mo
