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
import pennylane as qml
import numpy as np

from . import openfermion, structure


def _spin2_matrix_elements(sz):
    r"""Builds the table of matrix elements
    :math:`\langle \bm{\alpha}, \bm{\beta} \vert \hat{s}_1 \cdot \hat{s}_2 \vert
    \bm{\gamma}, \bm{\delta} \rangle` of the two-particle spin operator
    :math:`\hat{s}_1 \cdot \hat{s}_2`.

    The matrix elements are evaluated using the expression

    .. math::

        \langle ~ (\alpha, s_{z_\alpha});~ (\beta, s_{z_\beta}) ~ \vert \hat{s}_1 &&
        \cdot \hat{s}_2 \vert ~ (\gamma, s_{z_\gamma}); ~ (\delta, s_{z_\gamma}) ~ \rangle =
        \delta_{\alpha,\delta} \delta_{\beta,\gamma} \\
        && \times \left( \frac{1}{2} \delta_{s_{z_\alpha}, s_{z_\delta}+1}
        \delta_{s_{z_\beta}, s_{z_\gamma}-1} + \frac{1}{2} \delta_{s_{z_\alpha}, s_{z_\delta}-1}
        \delta_{s_{z_\beta}, s_{z_\gamma}+1} + s_{z_\alpha} s_{z_\beta}
        \delta_{s_{z_\alpha}, s_{z_\delta}} \delta_{s_{z_\beta}, s_{z_\gamma}} \right),

    where :math:`\alpha` and :math:`s_{z_\alpha}` refer to the quantum numbers of the spatial
    function and the spin projection, respectively, of the single-particle state
    :math:`\vert \bm{\alpha} \rangle \equiv \vert \alpha, s_{z_\alpha} \rangle`.

    Args:
        sz (array[float]): spin-projection of the single-particle states

    Returns:
        array: NumPy array with the table of matrix elements. The first four columns
        contain the indices :math:`\bm{\alpha}`, :math:`\bm{\beta}`, :math:`\bm{\gamma}`,
        :math:`\bm{\delta}` and the fifth column stores the computed matrix element.

    **Example**

    >>> sz = np.array([0.5, -0.5])
    >>> print(_spin2_matrix_elements(sz))
    [[ 0.    0.    0.    0.    0.25]
     [ 0.    1.    1.    0.   -0.25]
     [ 1.    0.    0.    1.   -0.25]
     [ 1.    1.    1.    1.    0.25]
     [ 0.    1.    0.    1.    0.5 ]
     [ 1.    0.    1.    0.    0.5 ]]
    """

    n = np.arange(sz.size)

    alpha = n.reshape(-1, 1, 1, 1)
    beta = n.reshape(1, -1, 1, 1)
    gamma = n.reshape(1, 1, -1, 1)
    delta = n.reshape(1, 1, 1, -1)

    # we only care about indices satisfying the following boolean mask
    mask = np.logical_and(alpha // 2 == delta // 2, beta // 2 == gamma // 2)

    # diagonal elements
    diag_mask = np.logical_and(sz[alpha] == sz[delta], sz[beta] == sz[gamma])
    diag_indices = np.argwhere(np.logical_and(mask, diag_mask))
    diag_values = (sz[alpha] * sz[beta]).flatten()

    diag = np.vstack([diag_indices.T, diag_values]).T

    # off-diagonal elements
    m1 = np.logical_and(sz[alpha] == sz[delta] + 1, sz[beta] == sz[gamma] - 1)
    m2 = np.logical_and(sz[alpha] == sz[delta] - 1, sz[beta] == sz[gamma] + 1)

    off_diag_mask = np.logical_and(mask, np.logical_or(m1, m2))
    off_diag_indices = np.argwhere(off_diag_mask)
    off_diag_values = np.full([len(off_diag_indices)], 0.5)

    off_diag = np.vstack([off_diag_indices.T, off_diag_values]).T

    # combine the off diagonal and diagonal tables into a single table
    return np.vstack([diag, off_diag])


def spin2(electrons, orbitals, mapping="jordan_wigner", wires=None):
    r"""Computes the total spin operator :math:`\hat{S}^2`.

    The total spin operator :math:`\hat{S}^2` is given by

    .. math::

        \hat{S}^2 = \frac{3}{4}N + \sum_{ \bm{\alpha}, \bm{\beta}, \bm{\gamma}, \bm{\delta} }
        \langle \bm{\alpha}, \bm{\beta} \vert \hat{s}_1 \cdot \hat{s}_2
        \vert \bm{\gamma}, \bm{\delta} \rangle ~
        \hat{c}_\bm{\alpha}^\dagger \hat{c}_\bm{\beta}^\dagger
        \hat{c}_\bm{\gamma} \hat{c}_\bm{\delta},

    where the two-particle matrix elements are computed as,

    .. math::

        \langle \bm{\alpha}, \bm{\beta} \vert \hat{s}_1 \cdot \hat{s}_2
        \vert \bm{\gamma}, \bm{\delta} \rangle = && \delta_{\alpha,\delta} \delta_{\beta,\gamma} \\
        && \times \left( \frac{1}{2} \delta_{s_{z_\alpha}, s_{z_\delta}+1}
        \delta_{s_{z_\beta}, s_{z_\gamma}-1} + \frac{1}{2} \delta_{s_{z_\alpha}, s_{z_\delta}-1}
        \delta_{s_{z_\beta}, s_{z_\gamma}+1} + s_{z_\alpha} s_{z_\beta}
        \delta_{s_{z_\alpha}, s_{z_\delta}} \delta_{s_{z_\beta}, s_{z_\gamma}} \right).

    In the equations above :math:`N` is the number of electrons, :math:`\alpha` refer to the
    quantum numbers of the spatial wave function and :math:`s_{z_\alpha}` is
    the spin projection of the single-particle state
    :math:`\vert \bm{\alpha} \rangle \equiv \vert \alpha, s_{z_\alpha} \rangle`.
    The operators :math:`\hat{c}^\dagger` and :math:`\hat{c}` are the particle creation
    and annihilation operators, respectively.

    Args:
        electrons (int): Number of electrons. If an active space is defined, this is
            the number of active electrons.
        orbitals (int): Number of *spin* orbitals. If an active space is defined,  this is
            the number of active spin-orbitals.
        mapping (str): Specifies the transformation to map the fermionic operator to the
            Pauli basis. Input values can be ``'jordan_wigner'`` or ``'bravyi_kitaev'``.
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert the qubit operator
            to an observable measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).

    Returns:
        pennylane.Hamiltonian: the total spin observable :math:`\hat{S}^2`

    **Example**

    >>> electrons = 2
    >>> orbitals = 4
    >>> S2 = spin2(electrons, orbitals, mapping="jordan_wigner")
    >>> print(S2)
    (0.75) [I0]
    + (0.375) [Z1]
    + (-0.375) [Z0 Z1]
    + (0.125) [Z0 Z2]
    + (0.375) [Z0]
    + (-0.125) [Z0 Z3]
    + (-0.125) [Z1 Z2]
    + (0.125) [Z1 Z3]
    + (0.375) [Z2]
    + (0.375) [Z3]
    + (-0.375) [Z2 Z3]
    + (0.125) [Y0 X1 Y2 X3]
    + (0.125) [Y0 Y1 X2 X3]
    + (0.125) [Y0 Y1 Y2 Y3]
    + (-0.125) [Y0 X1 X2 Y3]
    + (-0.125) [X0 Y1 Y2 X3]
    + (0.125) [X0 X1 X2 X3]
    + (0.125) [X0 X1 Y2 Y3]
    + (0.125) [X0 Y1 X2 Y3]

    >>> S2 = spin2(electrons, orbitals, mapping="jordan_wigner", wires=['w0','w1','w2','w3'])
    >>> print(S2)
    (0.75) [Iw0]
    + (0.375) [Zw1]
    + (-0.375) [Zw0 Zw1]
    + (0.125) [Zw0 Zw2]
    + (0.375) [Zw0]
    + (-0.125) [Zw0 Zw3]
    + (-0.125) [Zw1 Zw2]
    + (0.125) [Zw1 Zw3]
    + (0.375) [Zw2]
    + (0.375) [Zw3]
    + (-0.375) [Zw2 Zw3]
    + (0.125) [Yw0 Xw1 Yw2 Xw3]
    + (0.125) [Yw0 Yw1 Xw2 Xw3]
    + (0.125) [Yw0 Yw1 Yw2 Yw3]
    + (-0.125) [Yw0 Xw1 Xw2 Yw3]
    + (-0.125) [Xw0 Yw1 Yw2 Xw3]
    + (0.125) [Xw0 Xw1 Xw2 Xw3]
    + (0.125) [Xw0 Xw1 Yw2 Yw3]
    + (0.125) [Xw0 Yw1 Xw2 Yw3]
    """

    if electrons <= 0:
        raise ValueError(f"'electrons' must be greater than 0; got for 'electrons' {electrons}")

    if orbitals <= 0:
        raise ValueError(f"'orbitals' must be greater than 0; got for 'orbitals' {orbitals}")

    sz = np.where(np.arange(orbitals) % 2 == 0, 0.5, -0.5)

    table = _spin2_matrix_elements(sz)

    # create the list of ``FermionOperator`` objects
    s2_op = openfermion.ops.FermionOperator("") * 3 / 4 * electrons
    for i in table:
        s2_op += openfermion.ops.FermionOperator(
            ((int(i[0]), 1), (int(i[1]), 1), (int(i[2]), 0), (int(i[3]), 0)), i[4]
        )

    return observable([s2_op], mapping=mapping, wires=wires)


def observable(fermion_ops, init_term=0, mapping="jordan_wigner", wires=None):

    r"""Builds the Fermion many-body observable whose expectation value can be
    measured in PennyLane.

    The second-quantized operator of the Fermion many-body system can combine one-particle
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
    in electronic structure calculations, this is the case for: the kinetic energy operator,
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
        return structure.convert_observable(
            openfermion.transforms.bravyi_kitaev(mb_obs), wires=wires
        )

    return structure.convert_observable(openfermion.transforms.jordan_wigner(mb_obs), wires=wires)


def spin_z(orbitals, mapping="jordan_wigner", wires=None):
    r"""Computes the total spin projection operator :math:`\hat{S}_z` in the Pauli basis.

    The total spin projection operator :math:`\hat{S}_z` is given by

    .. math::

        \hat{S}_z = \sum_{\alpha, \beta} \langle \alpha \vert \hat{s}_z \vert \beta \rangle
        ~ \hat{c}_\alpha^\dagger \hat{c}_\beta, ~~ \langle \alpha \vert \hat{s}_z
        \vert \beta \rangle = s_{z_\alpha} \delta_{\alpha,\beta},

    where :math:`s_{z_\alpha} = \pm 1/2` is the spin-projection of the single-particle state
    :math:`\vert \alpha \rangle`. The operators :math:`\hat{c}^\dagger` and :math:`\hat{c}`
    are the particle creation and annihilation operators, respectively.

    Args:
        orbitals (str): Number of *spin* orbitals. If an active space is defined, this is
            the number of active spin-orbitals.
        mapping (str): Specifies the transformation to map the fermionic operator to the
            Pauli basis. Input values can be ``'jordan_wigner'`` or ``'bravyi_kitaev'``.
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert the qubit operator
            to an observable measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).

    Returns:
        pennylane.Hamiltonian: the total spin projection observable :math:`\hat{S}_z`

    **Example**

    >>> orbitals = 4
    >>> Sz = spin_z(orbitals, mapping="jordan_wigner")
    >>> print(Sz)
    (-0.25) [Z0]
    + (0.25) [Z1]
    + (-0.25) [Z2]
    + (0.25) [Z3]
    """

    if orbitals <= 0:
        raise ValueError(f"'orbitals' must be greater than 0; got for 'orbitals' {orbitals}")

    r = np.arange(orbitals)
    sz_orb = np.where(np.arange(orbitals) % 2 == 0, 0.5, -0.5)
    table = np.vstack([r, r, sz_orb]).T

    sz_op = openfermion.ops.FermionOperator()
    for i in table:
        sz_op += openfermion.ops.FermionOperator(((int(i[0]), 1), (int(i[1]), 0)), i[2])

    return observable([sz_op], mapping=mapping, wires=wires)


def particle_number(orbitals, mapping="jordan_wigner", wires=None):
    r"""Computes the particle number operator :math:`\hat{N}=\sum_\alpha \hat{n}_\alpha`
    in the Pauli basis.

    The particle number operator is given by

    .. math::

        \hat{N} = \sum_\alpha \hat{c}_\alpha^\dagger \hat{c}_\alpha,

    where the index :math:`\alpha` runs over the basis of single-particle states
    :math:`\vert \alpha \rangle`, and the operators :math:`\hat{c}^\dagger` and :math:`\hat{c}` are
    the particle creation and annihilation operators, respectively.

    Args:
        orbitals (int): Number of *spin* orbitals. If an active space is defined, this is
            the number of active spin-orbitals.
        mapping (str): Specifies the transformation to map the fermionic operator to the
            Pauli basis. Input values can be ``'jordan_wigner'`` or ``'bravyi_kitaev'``.
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert the qubit operator
            to an observable measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).

    Returns:
        pennylane.Hamiltonian: the fermionic-to-qubit transformed observable

    **Example**

    >>> orbitals = 4
    >>> N = particle_number(orbitals, mapping="jordan_wigner")
    >>> print(N)
    (2.0) [I0]
    + (-0.5) [Z0]
    + (-0.5) [Z1]
    + (-0.5) [Z2]
    + (-0.5) [Z3]
    >>> N = particle_number(orbitals, mapping="jordan_wigner", wires=['w0','w1','w2','w3'])
    >>> print(N)
    (2.0) [Iw0]
    + (-0.5) [Zw0]
    + (-0.5) [Zw1]
    + (-0.5) [Zw2]
    + (-0.5) [Zw3]
    """

    if orbitals <= 0:
        raise ValueError(f"'orbitals' must be greater than 0; got for 'orbitals' {orbitals}")

    r = np.arange(orbitals)
    table = np.vstack([r, r, np.ones([orbitals])]).T

    N_obs = openfermion.ops.FermionOperator()
    for i in table:
        N_obs += openfermion.ops.FermionOperator(((int(i[0]), 1), (int(i[1]), 0)), i[2])

    return observable([N_obs], mapping=mapping, wires=wires)


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


def dipole(
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


__all__ = [
    "observable",
    "particle_number",
    "spin_z",
    "spin2",
    "dipole",
    "one_particle",
    "two_particle",
    "_spin2_matrix_elements",
]
