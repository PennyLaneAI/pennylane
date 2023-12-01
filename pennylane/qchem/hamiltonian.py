# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains the functions needed for computing the molecular Hamiltonian.
"""
# pylint: disable= too-many-branches, too-many-arguments, too-many-locals, too-many-nested-blocks
import pennylane as qml

from .hartree_fock import nuclear_energy, scf
from .observable_hf import fermionic_observable, qubit_observable


def electron_integrals(mol, core=None, active=None):
    r"""Return a function that computes the one- and two-electron integrals in the molecular orbital
    basis.

    The one- and two-electron integrals are required to construct a molecular Hamiltonian in the
    second-quantized form

    .. math::

        H = \sum_{pq} h_{pq} c_p^{\dagger} c_q + \frac{1}{2} \sum_{pqrs} h_{pqrs} c_p^{\dagger} c_q^{\dagger} c_r c_s,

    where :math:`c^{\dagger}` and :math:`c` are the creation and annihilation operators,
    respectively, and :math:`h_{pq}` and :math:`h_{pqrs}` are the one- and two-electron integrals.
    These integrals can be computed by integrating over molecular orbitals :math:`\phi` as

    .. math::

        h_{pq} = \int \phi_p(r)^* \left ( -\frac{\nabla_r^2}{2} - \sum_i \frac{Z_i}{|r-R_i|} \right )  \phi_q(r) dr,

    and

    .. math::

        h_{pqrs} = \int \frac{\phi_p(r_1)^* \phi_q(r_2)^* \phi_r(r_2) \phi_s(r_1)}{|r_1 - r_2|} dr_1 dr_2.

    The molecular orbitals are constructed as a linear combination of atomic orbitals as

    .. math::

        \phi_i = \sum_{\nu}c_{\nu}^i \chi_{\nu}.

    The one- and two-electron integrals can be written in the molecular orbital basis as

    .. math::

        h_{pq} = \sum_{\mu \nu} C_{p \mu} h_{\mu \nu} C_{\nu q},

    and

    .. math::

        h_{pqrs} = \sum_{\mu \nu \rho \sigma} C_{p \mu} C_{q \nu} h_{\mu \nu \rho \sigma} C_{\rho r} C_{\sigma s}.

    The :math:`h_{\mu \nu}` and :math:`h_{\mu \nu \rho \sigma}` terms refer to the elements of the
    core matrix and the electron repulsion tensor, respectively, and :math:`C` is the molecular
    orbital expansion coefficient matrix.

    Args:
        mol (~qchem.molecule.Molecule): the molecule object
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that computes the core constant and the one- and two-electron integrals

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> electron_integrals(mol)(*args)
    (1.0,
     array([[-1.3902192695e+00,  0.0000000000e+00],
            [-4.4408920985e-16, -2.9165331336e-01]]),
     array([[[[ 7.1443907755e-01, -2.7755575616e-17],
              [ 5.5511151231e-17,  1.7024144301e-01]],
             [[ 5.5511151231e-17,  1.7024144301e-01],
              [ 7.0185315353e-01,  6.6613381478e-16]]],
            [[[-1.3877787808e-16,  7.0185315353e-01],
              [ 1.7024144301e-01,  2.2204460493e-16]],
             [[ 1.7024144301e-01, -4.4408920985e-16],
              [ 6.6613381478e-16,  7.3883668974e-01]]]]))
    """

    def _electron_integrals(*args):
        r"""Compute the one- and two-electron integrals in the molecular orbital basis.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            tuple[array[float]]: 1D tuple containing core constant, one- and two-electron integrals
        """
        _, coeffs, _, h_core, repulsion_tensor = scf(mol)(*args)
        one = qml.math.einsum("qr,rs,st->qt", coeffs.T, h_core, coeffs)
        two = qml.math.swapaxes(
            qml.math.einsum(
                "ab,cd,bdeg,ef,gh->acfh", coeffs.T, coeffs.T, repulsion_tensor, coeffs, coeffs
            ),
            1,
            3,
        )
        core_constant = nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)

        if core is None and active is None:
            return core_constant, one, two

        for i in core:
            core_constant = core_constant + 2 * one[i][i]
            for j in core:
                core_constant = core_constant + 2 * two[i][j][j][i] - two[i][j][i][j]

        for p in active:
            for q in active:
                for i in core:
                    o = qml.math.zeros(one.shape)
                    o[p, q] = 1.0
                    one = one + (2 * two[i][p][q][i] - two[i][p][i][q]) * o

        one = one[qml.math.ix_(active, active)]
        two = two[qml.math.ix_(active, active, active, active)]

        return core_constant, one, two

    return _electron_integrals


def fermionic_hamiltonian(mol, cutoff=1.0e-12, core=None, active=None):
    r"""Return a function that computes the fermionic Hamiltonian.

    Args:
        mol (~qchem.molecule.Molecule): the molecule object
        cutoff (float): cutoff value for discarding the negligible electronic integrals
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that computes the fermionic hamiltonian

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> h = fermionic_hamiltonian(mol)(*args)
    """

    def _fermionic_hamiltonian(*args):
        r"""Compute the fermionic hamiltonian.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            FermiSentence: fermionic Hamiltonian
        """

        core_constant, one, two = electron_integrals(mol, core, active)(*args)

        return fermionic_observable(core_constant, one, two, cutoff)

    return _fermionic_hamiltonian


def diff_hamiltonian(mol, cutoff=1.0e-12, core=None, active=None):
    r"""Return a function that computes the qubit Hamiltonian.

    Args:
        mol (~qchem.molecule.Molecule): the molecule object
        cutoff (float): cutoff value for discarding the negligible electronic integrals
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that computes the qubit hamiltonian

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> h = diff_hamiltonian(mol)(*args)
    >>> h.coeffs
    array([ 0.29817879+0.j,  0.20813365+0.j,  0.20813365+0.j,
             0.17860977+0.j,  0.04256036+0.j, -0.04256036+0.j,
            -0.04256036+0.j,  0.04256036+0.j, -0.34724873+0.j,
             0.13290293+0.j, -0.34724873+0.j,  0.17546329+0.j,
             0.17546329+0.j,  0.13290293+0.j,  0.18470917+0.j])
    """

    def _molecular_hamiltonian(*args):
        r"""Compute the qubit hamiltonian.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            Hamiltonian: the qubit Hamiltonian
        """

        h_ferm = fermionic_hamiltonian(mol, cutoff, core, active)(*args)

        return qubit_observable(h_ferm)

    return _molecular_hamiltonian
