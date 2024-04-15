# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains the functions needed for computing the spin observables.
"""
import numpy as np
from pennylane.fermi import FermiSentence, FermiWord

from .observable_hf import qubit_observable


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


def spin2(electrons, orbitals):
    r"""Compute the total spin observable :math:`\hat{S}^2`.

    The total spin observable :math:`\hat{S}^2` is given by

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

    Returns:
        pennylane.Hamiltonian: the total spin observable :math:`\hat{S}^2`

    Raises:
        ValueError: If electrons or orbitals is less than or equal to 0

    **Example**

    >>> electrons = 2
    >>> orbitals = 4
    >>> spin2(electrons, orbitals)
    (
        0.75 * I(0)
      + 0.375 * Z(0)
      + 0.375 * Z(1)
      + -0.375 * (Z(0) @ Z(1))
      + 0.375 * Z(2)
      + 0.125 * (Z(0) @ Z(2))
      + 0.375 * Z(3)
      + -0.125 * (Z(0) @ Z(3))
      + -0.125 * (Z(1) @ Z(2))
      + 0.125 * (Z(1) @ Z(3))
      + -0.375 * (Z(2) @ Z(3))
      + 0.125 * (Y(0) @ Y(2) @ X(3) @ X(1))
      + 0.125 * (Y(0) @ X(2) @ X(3) @ Y(1))
      + 0.125 * (Y(0) @ Y(2) @ Y(3) @ Y(1))
      + -0.125 * (Y(0) @ X(2) @ Y(3) @ X(1))
      + -0.125 * (X(0) @ Y(2) @ X(3) @ Y(1))
      + 0.125 * (X(0) @ X(2) @ X(3) @ X(1))
      + 0.125 * (X(0) @ Y(2) @ Y(3) @ X(1))
      + 0.125 * (X(0) @ X(2) @ Y(3) @ Y(1))
    )
    """

    if electrons <= 0:
        raise ValueError(f"'electrons' must be greater than 0; got for 'electrons' {electrons}")

    if orbitals <= 0:
        raise ValueError(f"'orbitals' must be greater than 0; got for 'orbitals' {orbitals}")

    sz = np.where(np.arange(orbitals) % 2 == 0, 0.5, -0.5)

    table = _spin2_matrix_elements(sz)

    sentence = FermiSentence({FermiWord({}): 3 / 4 * electrons})

    for i in table:
        sentence.update(
            {
                FermiWord(
                    {
                        (0, int(i[0])): "+",
                        (1, int(i[1])): "+",
                        (2, int(i[2])): "-",
                        (3, int(i[3])): "-",
                    }
                ): i[4]
            }
        )
    sentence.simplify()

    return qubit_observable(sentence)


def spinz(orbitals):
    r"""Computes the total spin projection observable :math:`\hat{S}_z`.

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

    Returns:
        pennylane.Hamiltonian: the total spin projection observable :math:`\hat{S}_z`

    Raises:
        ValueError: If orbitals is less than or equal to 0

    **Example**

    >>> orbitals = 4
    >>> print(spinz(orbitals))
    (
        -0.25 * Z(0)
      + 0.25 * Z(1)
      + -0.25 * Z(2)
      + 0.25 * Z(3)
    )
    """

    if orbitals <= 0:
        raise ValueError(f"'orbitals' must be greater than 0; got for 'orbitals' {orbitals}")

    r = np.arange(orbitals)
    sz_orb = np.where(r % 2 == 0, 0.5, -0.5)
    table = np.vstack([r, r, sz_orb]).T

    sentence = FermiSentence({})

    for i in table:
        sentence.update(
            {
                FermiWord(
                    {
                        (0, int(i[0])): "+",
                        (1, int(i[1])): "-",
                    }
                ): i[2]
            }
        )
    sentence.simplify()

    return qubit_observable(sentence)
