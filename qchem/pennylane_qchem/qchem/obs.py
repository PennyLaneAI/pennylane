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
import numpy as np
from openfermion.ops import FermionOperator
from openfermion.transforms import bravyi_kitaev, jordan_wigner

from . import structure


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
        raise ValueError(
            "'electrons' must be greater than 0; got for 'electrons' {}".format(electrons)
        )

    if orbitals <= 0:
        raise ValueError(
            "'orbitals' must be greater than 0; got for 'orbitals' {}".format(orbitals)
        )

    sz = np.where(np.arange(orbitals) % 2 == 0, 0.5, -0.5)

    table = _spin2_matrix_elements(sz)
    # To be consistent with the 'observable' function
    table[:, 4] = 2 * table[:, 4]

    return observable([table], init_term=3 / 4 * electrons, mapping=mapping, wires=wires)


def observable(matrix_elements, init_term=0, mapping="jordan_wigner", wires=None):

    r"""Builds the many-body observable whose expectation value can be
    measured in PennyLane.

    This function can be used to build second-quantized operators in the basis
    of single-particle states (e.g., HF states) and to transform them into
    PennyLane observables. In general, the many-body observable :math:`\hat{O}` can combine
    one-particle and two-particle operators, as in the case for electronic Hamiltonians:

    .. math::

        \hat{O} = \sum_{\alpha, \beta} \langle \alpha \vert \hat{t}^{(1)} +
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
    the nuclei Coulomb potential, or any other external fields included in the model Hamiltonian.
    On the other hand, :math:`\langle \alpha, \beta \vert \hat{v} \vert \gamma, \delta \rangle`
    denotes the matrix element of the two-particle operator :math:`\hat{v}`, for example, the
    Coulomb interaction between the electrons.

    If an `active space <https://en.wikipedia.org/wiki/Complete_active_space>`_
    (see :func:`~.active_space`) is defined, the observable is expanded over the truncated
    basis of active orbitals. The contribution of core orbitals can be passed to the
    function using the keyword argument ``init_term``.

    The function uses tools of `OpenFermion <https://github.com/quantumlib/OpenFermion>`_
    to build the second-quantized operator and map it to basis of Pauli matrices via the
    Jordan-Wigner or Bravyi-Kitaev transformation. Finally, the qubit operator is
    converted to a a PennyLane observable by the function :func:`~.convert_observable`.

    Args:
        matrix_elements (list(array[float])): list containing 2D numpy arrays with the matrix
            elements of the operators :math:`\hat{t}` and :math:`\hat{v}`.
            For single-particle operators the :math:`i`-th array in the list will have shape
            ``(matrix_elements[i].shape[0], 3)`` with each row containing the indices
            :math:`\alpha`, :math:`\beta` and the matrix element
            :math:`\langle \alpha \vert \hat{t}^{(i)}\vert \beta \rangle`.
            For two-particle operators the :math:`j`-th array in the list
            will have shape ``(matrix_elements[j].shape[0], 5)`` with each row containing
            the indices :math:`\alpha`, :math:`\beta`, :math:`\gamma`, :math:`\delta` and
            the matrix element
            :math:`\langle \alpha, \beta \vert \hat{v}^{(j)}\vert \gamma, \delta \rangle`.
        init_term (float): the contribution of core orbitals, if any, or other quantity
            required to initialize the many-body observable.
        mapping (str): specifies the fermion-to-qubit mapping. Input values can
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

    >>> t = np.array([[0., 0., 0.5], [1.0, 1.0, -0.5], [1.0, 0., 0.]])
    >>> v = np.array([[ 0., 0., 0., 0., 0.25], [ 0., 1., 1., 0., -0.25], [ 1., 0., 0., 1., -0.5]])
    >>> matrix_elements = [t, v]
    >>> print(observable(matrix_elements, init_term=1/4, mapping="bravyi_kitaev"))
    (0.15625) [I0]
    + (-0.15625) [Z0]
    + (0.34375) [Z0 Z1]
    + (-0.09375) [Z1]
    """

    if mapping.strip().lower() not in ("jordan_wigner", "bravyi_kitaev"):
        raise TypeError(
            "The '{}' transformation is not available. \n "
            "Please set 'mapping' to 'jordan_wigner' or 'bravyi_kitaev'.".format(mapping)
        )

    # Initialize the FermionOperator
    mb_obs = FermionOperator() + FermionOperator("") * init_term

    for table in matrix_elements:

        if len(table.shape) != 2:
            raise ValueError(
                "Expected dimension for arrays in 'matrix_elements' is 2; got {}".format(
                    table.shape
                )
            )

        if table.shape[1] not in (3, 5):
            raise ValueError(
                "Expected entries of matrix element tables to be of shape (3,) or (5,); got {}".format(
                    table.shape[1]
                )
            )

        if table.shape[1] == 5:
            # two-particle operator
            for i in table:
                mb_obs += FermionOperator(
                    ((int(i[0]), 1), (int(i[1]), 1), (int(i[2]), 0), (int(i[3]), 0)), i[4] / 2
                )
        else:
            # single-particle operator
            for i in table:
                mb_obs += FermionOperator(((int(i[0]), 1), (int(i[1]), 0)), i[2])

    # Map the fermionic operator to a qubit operator
    if mapping.strip().lower() == "bravyi_kitaev":
        return structure.convert_observable(bravyi_kitaev(mb_obs), wires=wires)

    return structure.convert_observable(jordan_wigner(mb_obs), wires=wires)


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
        raise ValueError(
            "'orbitals' must be greater than 0; got for 'orbitals' {}".format(orbitals)
        )

    r = np.arange(orbitals)
    sz_orb = np.where(np.arange(orbitals) % 2 == 0, 0.5, -0.5)
    table = np.vstack([r, r, sz_orb]).T

    return observable([table], mapping=mapping, wires=wires)


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
        raise ValueError(
            "'orbitals' must be greater than 0; got for 'orbitals' {}".format(orbitals)
        )

    r = np.arange(orbitals)
    table = np.vstack([r, r, np.ones([orbitals])]).T

    return observable([table], mapping=mapping, wires=wires)


__all__ = [
    "observable",
    "particle_number",
    "spin_z",
    "spin2",
    "_spin2_matrix_elements",
]
