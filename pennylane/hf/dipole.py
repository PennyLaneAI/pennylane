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
This module contains the functions needed for computing the dipole moment.
"""
# pylint: disable= too-many-branches, too-many-arguments, too-many-locals, too-many-nested-blocks
import autograd.numpy as anp
import pennylane as qml
from pennylane import numpy as np
from pennylane.hf.hartree_fock import generate_scf, nuclear_energy
from pennylane.hf.matrices import generate_moment_matrix


def generate_dipole_integrals(mol, core=None, active=None):
    r"""Return a function that computes the dipole moment integrals in the molecular orbital basis.

    These integrals are required to construct the dipole operator in the second-quantized form

    .. math::

        D = \sum_{pq} d_{pq} c_p^{\dagger} c_q + D_n,

    where :math:`d_{pq}` is the moment integral, :math:`c^{\dagger}` and :math:`c` are the creation
    and annihilation operators, respectively, and :math:`D_n` is the contribution of the nuclei to
    the dipole operator.

    The integrals can be computed by integrating over molecular orbitals :math:`\phi` as

    .. math::

        d_{pq} = \int \phi_p(r)^* r \phi_q(r) dr,,

    The molecular orbitals are constructed as a linear combination of atomic orbitals as

    .. math::

        \phi_i = \sum_{\nu}c_{\nu}^i \chi_{\nu}.

    The dipole moment integral can be written in the molecular orbital basis as

    .. math::

        d_{pq} = \sum_{\mu \nu} C_{p \mu} d_{\mu \nu} C_{\nu q},

    where :math:`h_{\mu \nu}` refers to the dipole moment integral in the atomic orbital basis and
    :math:`C` is the molecular orbital expansion coefficient matrix.

    Args:
        mol (Molecule): the molecule object
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that computes the dipole moment integrals in the molecular orbital basis

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.hf.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> generate_dipole_integrals(mol)(*args)

    """

    def dipole_integrals(*args):
        r"""Compute the dipole moment integrals in the molecular orbital basis.

        Args:
            args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            tuple[array[float]]: 1D tuple containing dipole moment integrals
        """
        _, coeffs, _, _, _ = generate_scf(mol)(*args)

        dx = anp.einsum(
            "qr,rs,st->qt", coeffs.T, generate_moment_matrix(mol.basis_set, 1, 0)(*args), coeffs
        )
        dy = anp.einsum(
            "qr,rs,st->qt", coeffs.T, generate_moment_matrix(mol.basis_set, 1, 1)(*args), coeffs
        )
        dz = anp.einsum(
            "qr,rs,st->qt", coeffs.T, generate_moment_matrix(mol.basis_set, 1, 2)(*args), coeffs
        )

        core_constant = anp.array([0])

        if core is None and active is None:
            return core_constant, dx, dy, dz

        for i in core:
            core_constant = core_constant + 2 * (dx[i][i] + dy[i][i] + dz[i][i])

        dx = dx[anp.ix_(active, active)]
        dy = dy[anp.ix_(active, active)]
        dz = dz[anp.ix_(active, active)]

        return core_constant, dx, dy, dz

    return dipole_integrals
