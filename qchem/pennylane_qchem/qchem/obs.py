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
import numpy as np
from openfermion.hamiltonians import MolecularData
from openfermion.ops import FermionOperator
from openfermion.transforms import bravyi_kitaev, jordan_wigner

from . import structure


def _spin2_matrix_elements(sz, n_spin_orbs):
    r"""Generates the table of matrix elements
    :math:`\langle \alpha, \beta \vert \hat{s}_1 \cdot \hat{s}_2 \vert \gamma, \delta \rangle`
    of the two-particle spin operator :math:`\hat{s}_1 \cdot \hat{s}_2`.

    The matrix elements are evaluated using the expression,

    .. math::

        \langle \alpha, \beta \vert \hat{s}_1 \cdot \hat{s}_2
        \vert \gamma, \delta \rangle = \delta_{\alpha,\delta} \delta_{\beta,\gamma}
        \left( \frac{1}{2} \delta_{m_\alpha, m_\delta+1} \delta_{m_\beta, m_\gamma-1}
        + \frac{1}{2} \delta_{m_\alpha, m_\delta-1} \delta_{m_\beta, m_\gamma+1}
        + m_\alpha m_\beta \delta_{m_\alpha, m_\delta} \delta_{m_\beta, m_\gamma} \right),

    where :math:`\alpha` and :math:`m_\alpha` refer to the quantum numbers of the spatial
    :math:`\varphi_\alpha({\bf r})` and spin :math:`\chi_{m_\alpha}(s_z)` wave functions,
    respectively, of the single-particle state :math:`\vert \alpha \rangle`.

    Args:
        sz (array[float]): spin-projection quantum number of the spin-orbitals
        n_spin_orbs (int): number of spin orbitals

    Returns:
        array: NumPy array with the table of matrix elements. The first four columns
        contain the indices :math:`\alpha`, :math:`\beta`, :math:`\gamma`, :math:`\delta`
        and the fifth column stores the computed matrix element.

    **Example**

    >>> n_spin_orbs = 2
    >>> sz = np.array([0.5, -0.5])
    >>> print(_spin2_matrix_elements(sz, n_spin_orbs))
    [[ 0.    0.    0.    0.    0.25]
     [ 0.    1.    1.    0.   -0.25]
     [ 1.    0.    0.    1.   -0.25]
     [ 1.    1.    1.    1.    0.25]
     [ 0.    1.    0.    1.    0.5 ]
     [ 1.    0.    1.    0.    0.5 ]]
    """

    if sz.size != n_spin_orbs:
        raise ValueError(
            "Size of 'sz' must be equal to 'n_spin_orbs'; size got for 'sz' {}".format(sz.size)
        )

    n = np.arange(n_spin_orbs)

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


def get_spin2_matrix_elements(mol_name, hf_data, n_active_electrons=None, n_active_orbitals=None):
    r"""Reads the Hartree-Fock (HF) electronic structure data file, defines an active space and
    generates the table of matrix elements required to build the total-spin
    operator :math:`\hat{S}^2`.

    The second-quantized expression for the operator :math:`\hat{S}^2` reads,

    .. math::

        \hat{S}^2 = \frac{3}{4}N + \sum_{\alpha, \beta, \gamma, \delta}
        \langle \alpha, \beta \vert \hat{s}_1 \cdot \hat{s}_2
        \vert \gamma, \delta \rangle ~ \hat{c}_\alpha^\dagger \hat{c}_\beta^\dagger
        \hat{c}_\gamma \hat{c}_\delta,
    
    where the two-particle matrix elements are computedas,

    .. math::

        \langle \alpha, \beta \vert \hat{s}_1 \cdot \hat{s}_2
        \vert \gamma, \delta \rangle = \delta_{\alpha,\delta} \delta_{\beta,\gamma}
        \left( \frac{1}{2} \delta_{m_\alpha, m_\delta+1} \delta_{m_\beta, m_\gamma-1}
        + \frac{1}{2} \delta_{m_\alpha, m_\delta-1} \delta_{m_\beta, m_\gamma+1}
        + m_\alpha m_\beta \delta_{m_\alpha, m_\delta} \delta_{m_\beta, m_\gamma} \right).

    In the equations above :math:`N` is the number of particles, :math:`m_\alpha` refers to
    the quantum number of the spin wave function :math:`\chi_{m_\alpha}(s_z)` of the
    spin-orbital :math:`\vert \alpha \rangle` and :math:`\hat{c}_\alpha^\dagger`
    (:math:`\hat{c}_\alpha`) is the creation (annihilation) particle operator acting
    on the :math:`\alpha`-th active orbital.

    Args:
        mol_name (str): name of the molecule
        hf_data (str): path to the directory with the HF electronic structure data
        n_active_electrons (int): number of active electrons
        n_active_orbitals (int): number of active orbitals

    Returns:
        tuple: the table of the two-particle matrix elements and the single-particle
        contribution :math:`\frac{3N}{4}`. The first four columns of the table contains the
        indices :math:`\alpha`, :math:`\beta`, :math:`\gamma`, :math:`\delta` and the
        fifth column the value of the matrix elements.

    **Example**

    >>> get_spin2_matrix_elements(
        'h2',
        './pyscf/sto-3g',
        n_active_electrons=2,
        n_active_orbitals=2)
    [[ 0.    0.    0.    0.    0.25]
     [ 0.    1.    1.    0.   -0.25]
     [ 0.    2.    2.    0.    0.25]
     [ 0.    3.    3.    0.   -0.25]
     [ 1.    0.    0.    1.   -0.25]
     [ 1.    1.    1.    1.    0.25]
     [ 1.    2.    2.    1.   -0.25]
     [ 1.    3.    3.    1.    0.25]
     [ 2.    0.    0.    2.    0.25]
     [ 2.    1.    1.    2.   -0.25]
     [ 2.    2.    2.    2.    0.25]
     [ 2.    3.    3.    2.   -0.25]
     [ 3.    0.    0.    3.   -0.25]
     [ 3.    1.    1.    3.    0.25]
     [ 3.    2.    2.    3.   -0.25]
     [ 3.    3.    3.    3.    0.25]
     [ 0.    1.    0.    1.    0.5 ]
     [ 0.    3.    2.    1.    0.5 ]
     [ 1.    0.    1.    0.    0.5 ]
     [ 1.    2.    3.    0.    0.5 ]
     [ 2.    1.    0.    3.    0.5 ]
     [ 2.    3.    2.    3.    0.5 ]
     [ 3.    0.    1.    2.    0.5 ]
     [ 3.    2.    3.    2.    0.5 ]] 1.5
    """

    active_indices = structure.active_space(
        mol_name,
        hf_data,
        n_active_electrons=n_active_electrons,
        n_active_orbitals=n_active_orbitals,
    )[1]

    if n_active_electrons is None:
        hf_elect_struct = MolecularData(filename=os.path.join(hf_data.strip(), mol_name.strip()))
        n_electrons = hf_elect_struct.n_electrons
    else:
        n_electrons = n_active_electrons

    n_spin_orbs = 2 * len(active_indices)
    sz = np.where(np.arange(n_spin_orbs) % 2 == 0, 0.5, -0.5)

    return _spin2_matrix_elements(sz, n_spin_orbs), 3 / 4 * n_electrons


def observable(me_table, init_term=0, mapping="jordan_wigner"):

    r"""Builds the many-body observable whose expectation value can be
    measured in PennyLane.

    This function can be used to build second-quantized operators in the basis
    of single-particle states (e.g., HF states) and to transform them into
    PennyLane observables. In general, single- and two-particle operators can be
    expanded in a truncated set of orbitals that define an active space,

    .. math::

        &&\hat A = \sum_{\alpha \leq 2n_\mathrm{docc}} \langle \alpha \vert \hat{\mathcal{A}}
        \vert \alpha \rangle ~ \hat{n}_\alpha +
        \sum_{\alpha, \beta ~ \in ~ \mathrm{active~space}} \langle \alpha \vert \hat{\mathcal{A}}
        \vert \beta \rangle ~ \hat{c}_\alpha^\dagger\hat{c}_\beta \\
        &&\hat B = \frac{1}{2} \left\{ \sum_{\alpha, \beta \leq 2n_\mathrm{docc}}
        \langle \alpha, \beta \vert \hat{\mathcal{B}} \vert \beta, \alpha \rangle
        ~ \hat{n}_\alpha \hat{n}_\beta + \sum_{\alpha, \beta, \gamma, \delta ~
        \in ~ \mathrm{active~space}} \langle \alpha, \beta \vert \hat{\mathcal{B}}
        \vert \gamma, \delta \rangle ~ \hat{c}_{\alpha}^\dagger \hat{c}_{\beta}^\dagger
        \hat{c}_{\gamma} \hat{c}_{\delta} \right\}.

    In the latter equations :math:`n_\mathrm{docc}` denotes the doubly-occupied orbitals,
    if any, not included in the active space and
    :math:`\langle \alpha \vert \hat{\mathcal{A}} \vert \beta \rangle` and
    :math:`\langle \alpha, \beta \vert\hat{\mathcal{B}} \vert \gamma, \delta \rangle`
    are the matrix elements of the one- and two-particle operators
    :math:`\hat{\mathcal{A}}` and :math:`\hat{\mathcal{B}}`, respectively.

    The function utilizes tools of `OpenFermion <https://github.com/quantumlib/OpenFermion>`_
    to build the second-quantized operator and map it to basis of Pauli matrices via the
    Jordan-Wigner or Bravyi-Kitaev transformation. Finally, the qubit operator is
    converted to a a PennyLane observable by the function :func:`~.convert_observable`.

    Args:
        me_table (array[float]): Numpy array with the table of matrix elements.
            For a single-particle operator this array will have shape
            ``(me_table.shape[0], 3)`` with each row containing the indices
            :math:`\alpha`, :math:`\beta` and the matrix element :math:`\langle \alpha \vert
            \hat{\mathcal{A}}\vert \beta \rangle`. For a two-particle operator this
            array will have shape ``(me_table.shape[0], 5)`` with each row containing
            the indices :math:`\alpha`, :math:`\beta`, :math:`\gamma`, :math:`\delta` and
            the matrix elements :math:`\langle \alpha, \beta \vert \hat{\mathcal{B}}
            \vert \gamma, \delta \rangle`.
        init_term: the contribution of doubly-occupied orbitals, if any, or other quantity
            required to initialize the many-body observable.
        mapping (str): specifies the fermion-to-qubit mapping. Input values can
            be ``'jordan_wigner'`` or ``'bravyi_kitaev'``.

    Returns:
        pennylane.Hamiltonian: the fermionic-to-qubit transformed observable

    **Example**

    >>> s2_matrix_elements, init_term = get_spin2_matrix_elements('h2', './pyscf/sto-3g')
    >>> s2_obs = observable(s2_matrix_elements, init_term=init_term)
    >>> print(s2_obs)
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
    """

    if mapping.strip().lower() not in ("jordan_wigner", "bravyi_kitaev"):
        raise TypeError(
            "The '{}' transformation is not available. \n "
            "Please set 'mapping' to 'jordan_wigner' or 'bravyi_kitaev'.".format(mapping)
        )

    sp_op_shape = (3,)
    tp_op_shape = (5,)
    for i_table in me_table:
        if np.array(i_table).shape not in (sp_op_shape, tp_op_shape):
            raise ValueError(
                "expected entries of 'me_table' to be of shape (3,) or (5,) ; got {}".format(
                    np.array(i_table).shape
                )
            )

    # Initialize the FermionOperator
    mb_obs = FermionOperator() + FermionOperator("") * init_term

    for i in me_table:

        if i.shape == (5,):
            # two-particle operator
            mb_obs += FermionOperator(
                ((int(i[0]), 1), (int(i[1]), 1), (int(i[2]), 0), (int(i[3]), 0)), i[4]
            )
        elif i.shape == (3,):
            # single-particle operator
            mb_obs += FermionOperator(((int(i[0]), 1), (int(i[1]), 0)), i[2])

    # Map the fermionic operator to a qubit operator
    if mapping.strip().lower() == "bravyi_kitaev":
        return structure.convert_observable(bravyi_kitaev(mb_obs))

    return structure.convert_observable(jordan_wigner(mb_obs))


def get_spinZ_matrix_elements(mol_name, hf_data, n_active_electrons=None, n_active_orbitals=None):
    r"""Reads the Hartree-Fock (HF) electronic structure data file, defines an active
    space and generates the table of matrix elements required to build the second-quantized
    spin-projection operator :math:`\hat{S}_z`.

    The second-quantized operator :math:`\hat{S}_z` reads,

    .. math::

        \hat{S}_z = \sum_{\alpha, \beta} \langle \alpha \vert \hat{s}_z \vert \beta \rangle
        ~ \hat{c}_\alpha^\dagger\hat{c}_\beta,

        \langle \alpha \vert \hat{s}_z \vert \beta \rangle = m_\alpha \delta_{\alpha,\beta},

    where :math:`m_\alpha` refers to the quantum number of the spin wave function
    :math:`\chi_{m_\alpha}(s_z)` of the spin-orbital :math:`\vert \alpha \rangle`
    and :math:`\hat{c}_\alpha^\dagger` (:math:`\hat{c}_\alpha`) is the creation (annihilation)
    particle operator acting on the :math:`\alpha`-th active orbital.

    Args:
        mol_name (str): name of the molecule
        hf_data (str): path to the directory with the HF electronic structure data
        n_active_electrons (int): number of active electrons
        n_active_orbitals (int): number of active orbitals

    Returns:
        array: NumPy array with the table of matrix elements. Since :math:`\hat{S}_z` is
        diagonal in the basis of HF orbitals the first two columns contains the index
        :math:`\alpha` and the third column stores the matrix element.

    **Example**

    >>> get_spinZ_matrix_elements('h2', './pyscf/sto-3g', n_active_electrons=2, n_active_orbitals=2)
    [[ 0.   0.   0.5]
    [ 1.   1.  -0.5]
    [ 2.   2.   0.5]
    [ 3.   3.  -0.5]]
    """

    active_indices = structure.active_space(
        mol_name,
        hf_data,
        n_active_electrons=n_active_electrons,
        n_active_orbitals=n_active_orbitals,
    )[1]

    n_spin_orbs = 2 * len(active_indices)
    sz = np.where(np.arange(n_spin_orbs) % 2 == 0, 0.5, -0.5)

    spinz_matrix_elements = np.zeros((n_spin_orbs, 3))
    for alpha in range(n_spin_orbs):
        spinz_matrix_elements[alpha] = np.array([alpha, alpha, sz[alpha]])

    return spinz_matrix_elements


__all__ = [
    "_spin2_matrix_elements",
    "get_spin2_matrix_elements",
    "observable",
    "get_spinZ_matrix_elements",
]
