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
from pennylane.hf.basis_data import atomic_numbers
from pennylane.hf.matrices import generate_moment_matrix
from pennylane.hf.hamiltonian import simplify, _generate_qubit_operator, _return_pauli

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
            tuple[array[float]]: tuple containing dipole moment integrals and core/nuclear constants
        """
        _, coeffs, _, _, _ = qml.hf.generate_scf(mol)(*args)

        dx = anp.einsum(
            "qr,rs,st->qt", coeffs.T, generate_moment_matrix(mol.basis_set, 1, 0)(*args), coeffs
        )
        dy = anp.einsum(
            "qr,rs,st->qt", coeffs.T, generate_moment_matrix(mol.basis_set, 1, 1)(*args), coeffs
        )
        dz = anp.einsum(
            "qr,rs,st->qt", coeffs.T, generate_moment_matrix(mol.basis_set, 1, 2)(*args), coeffs
        )

        core_x, core_y, core_z = anp.array([0]), anp.array([0]), anp.array([0])

        for i in range(len(mol.symbols)): # nuclear contributions
            core_x = core_x + atomic_numbers[mol.symbols[i]] * mol.coordinates[i][0]
            core_y = core_y + atomic_numbers[mol.symbols[i]] * mol.coordinates[i][1]
            core_z = core_z + atomic_numbers[mol.symbols[i]] * mol.coordinates[i][2]

        if core is None and active is None:
            return (core_x, core_y, core_z), (dx, dy, dz)

        for i in core:
            core_x = core_x + 2 * dx[i][i]
            core_y = core_y + 2 * dy[i][i]
            core_z = core_z + 2 * dz[i][i]

        dx = dx[anp.ix_(active, active)]
        dy = dy[anp.ix_(active, active)]
        dz = dz[anp.ix_(active, active)]

        return (core_x, core_y, core_z), (dx, dy, dz)

    return dipole_integrals


def generate_fermionic_dipole(mol, cutoff=1.0e-12, core=None, active=None):
    r"""Return a function that computes the fermionic dipole.

    Args:
        mol (Molecule): the molecule object
        cutoff (float): cutoff value for discarding the negligible electronic integrals

    Returns:
        function: function that computes the fermionic dipole

    **Example**

    """

    def fermionic_dipole(*args):
        r"""Compute the fermionic dipole.

        Args:
            args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            tuple(array[float], list[list[int]]): the dipole coefficients and operators
        """
        f = []
        constants, integrals = generate_dipole_integrals(mol, core, active)(*args)
        constants = anp.negative(constants)

        for i in range(3):
            f.append(
                one_particle(constants[i], integrals[i], cutoff=cutoff, core=core, active=active))

        return f

    return fermionic_dipole


def generate_dipole(mol, cutoff=1.0e-12, core=None, active=None):
    r"""Return a function that computes the qubit dipole.

    Args:
        mol (Molecule): the molecule object
        cutoff (float): cutoff value for discarding the negligible electronic integrals

    Returns:
        function: function that computes the qubit dipole

    **Example**

    """
    def dipole(*args):
        r"""Compute the qubit dipole.

        Args:
            args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            (list[Hamiltonian]): x, y and z components of the dipole observable
        """

        d = []
        d_ferm = generate_fermionic_dipole(mol, cutoff, core, active)(*args)
        for i in d_ferm:
            d.append(qubit_operator(i, cutoff=cutoff))

        return d

    return dipole


def one_particle(core_constant, integral, cutoff=1.0e-12):
    r"""Create a fermionic operator from one-particle molecular orbital integrals.

    Args:
        core_constant (array[float]): the contribution of the core orbitalss and nucleai
        integral (array[float]): the one-particle molecular orbital integrals
        cutoff (float): cutoff value for discarding the negligible terms

    Returns:
        tuple(array[float], list[int]): fermionic coefficients and operators

    **Example**

    """
    coeffs = anp.array([])

    if core_constant != 0:
        coeffs = anp.concatenate((coeffs, core_constant))
        operators = [[]]
    else:
        operators = []

    for d in [integral]:
        i = anp.argwhere(abs(d) >= cutoff)
        operators = operators + (i * 2).tolist() + (i * 2 + 1).tolist()  # up-up + down-down terms
        coeffs = anp.concatenate((coeffs, anp.tile(d[abs(d) >= cutoff], 2)))

    indices_sort = [operators.index(i) for i in sorted(operators)]

    return coeffs[indices_sort], sorted(operators)


def qubit_operator(d_ferm, cutoff=1.0e-12):
    r"""Convert a fermionic operator to a PennyLane qubit operator.

    Args:
        d_ferm tuple(array[float], list[int]): fermionic operator
        cutoff (float): cutoff value for discarding the negligible terms

    Returns:
        Hamiltonian: Simplified PennyLane Hamiltonian

    **Example**

    """
    if len(d_ferm[0]) == 0 and len(d_ferm[1]) == 0:
        return qml.Hamiltonian([d_ferm[0]], [qml.Identity(0)])

    ops = []
    coeffs = anp.array([])

    for n, t in enumerate(d_ferm[1]):

        if len(t) == 0:
            coeffs = anp.array([0.0])
            coeffs = coeffs + np.array([d_ferm[0][n]])
            ops = ops + [qml.Identity(0)]

        else:
            op = _generate_qubit_operator(t)
            if op != 0:
                for i, o in enumerate(op[1]):
                    if len(o) == 0:
                        op[1][i] = qml.Identity(0)
                    if len(o) == 1:
                        op[1][i] = _return_pauli(o[0][1])(o[0][0])
                    if len(o) > 1:
                        k = qml.Identity(0)
                        for o_ in o:
                            k = k @ _return_pauli(o_[1])(o_[0])
                        op[1][i] = k
                coeffs = np.concatenate([coeffs, np.array(op[0]) * d_ferm[0][n]])
                ops = ops + op[1]

    d = simplify(qml.Hamiltonian(coeffs, ops), cutoff=cutoff) * (-1)

    return d