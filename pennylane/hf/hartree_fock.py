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
This module contains the functions needed for performing the self-consistent-field iterations.
"""

import autograd.numpy as anp
from pennylane.hf.matrices import (
    core_matrix,
    molecular_density_matrix,
    overlap_matrix,
    repulsion_tensor,
)


def generate_hartree_fock(mol, n_steps=50, tol=1e-8):
    r"""Return a function that performs the self-consistent-field iterations.

    Args:
￼       mol (Molecule): the molecule object
        n_steps (int): the number of iterations
        tol (float): convergence tolerance

    Returns:
         function: function that performs the self-consistent-field iterations

    **Example**
￼
￼   >>> symbols  = ['H', 'H']
￼   >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
￼   >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
￼   >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True),
￼   >>> mol = Molecule(symbols, geometry, alpha=alpha)
￼   >>> args = [alpha]
￼   >>> v_fock, coeffs, fock_matrix, h_core = generate_hartree_fock(mol)(*args)
    >>> v_fock
￼   array([-0.67578019,  0.94181155])
    """

    def hartree_fock(*args):
        r"""Perform the self-consistent-field iterations.

        Args:
￼            args (array[float]): initial values of the differentiable parameters

        Returns:
￼            tuple(array[float]): eigenvalues of the Fock matrix, molecular orbital coefficients,
             Fock matrix, core matrix
        """
        basis_functions = mol.basis_set
        charges = mol.nuclear_charges
        r = mol.coordinates
        n_electron = mol.n_electrons

        if r.requires_grad:
            e_repulsion = repulsion_tensor(basis_functions)(*args[1:])
            s = overlap_matrix(basis_functions)(*args[1:])
        else:
            e_repulsion = repulsion_tensor(basis_functions)(*args)
            s = overlap_matrix(basis_functions)(*args)
        h_core = core_matrix(basis_functions, charges, r)(*args)

        w, v = anp.linalg.eigh(s)
        x = v @ anp.diag(anp.array([1 / anp.sqrt(i) for i in w])) @ v.T

        v_fock, w_fock = anp.linalg.eigh(x.T @ h_core @ x)
        coeffs = x @ w_fock

        p = molecular_density_matrix(n_electron, coeffs)

        for _ in range(n_steps):

            j = anp.einsum("pqrs,rs->pq", e_repulsion, p)
            k = anp.einsum("psqr,rs->pq", e_repulsion, p)

            fock_matrix = h_core + 2 * j - k

            v_fock, w_fock = anp.linalg.eigh(x.T @ fock_matrix @ x)

            coeffs = x @ w_fock

            p_update = molecular_density_matrix(n_electron, coeffs)

            if anp.linalg.norm(p_update - p) <= tol:
                break

            p = p_update

        return v_fock, coeffs, fock_matrix, h_core

    return hartree_fock


def nuclear_energy(charges, r):
    r"""Return a function that computes the nuclear repulsion energy.

    The nuclear energy computed as

    .. math::
￼
￼        \sum_{i>j}^n \frac{q_i q_j}{r_{ij}},

    where :math:`q`, :math:`r` and :math:`n` represent the nuclear charges, nuclear positions and
    the number of nuclei, respectively.

    Args:
        charges (list[int]): nuclear charges
        r (array[float]): nuclear positions

    Returns:
        function: function that computes the nuclear repulsion energy

    **Example**

 ￼  >>> symbols  = ['H', 'F']
￼   >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], requires_grad = True)
￼   >>> mol = Molecule(symbols, geometry)
    >>> args = [mol.coordinates]
￼   >>> e = nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)
    >>> print(e)
    4.5
    """

    def nuclear(*args):
        r"""Compute the nuclear repulsion energy.

        Args:
            args (array[float]): initial values of the differentiable parameters

        Returns:
￼            array[float]: nuclear repulsion energy
        """
        if r.requires_grad:
            coor = args[0]
        else:
            coor = r
        e = 0
        for i, r1 in enumerate(coor):
            for j, r2 in enumerate(coor):
                if i > j:
                    e = e + (charges[i] * charges[j] / anp.sqrt(((r1 - r2) ** 2).sum()))
        return e

    return nuclear


def hf_energy(mol):
    r"""Return a function that computes the Hartree-Fock energy.

    Args:
￼       mol (Molecule): the molecule object

    Returns:
         function: function that computes the Hartree-Fock energy

    **Example**
￼
￼   >>> symbols  = ['H', 'H']
￼   >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
￼   >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
￼   >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True),
￼   >>> mol = Molecule(symbols, geometry, alpha=alpha)
￼   >>> args = [alpha]
￼   >>> hf_energy(mol)(*args)
￼   -1.065999461545263
    """

    def energy(*args):
        r"""Compute the Hartree-Fock energy.

        Args:
            args (array[float]): initial values of the differentiable parameters

        Returns:
￼            float: the Hartree-Fock energy
        """
        v_fock, coeffs, fock_matrix, h_core = generate_hartree_fock(mol)(*args)
        e_rep = nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)
        e_elec = anp.einsum(
            "pq,qp", fock_matrix + h_core, molecular_density_matrix(mol.n_electrons, coeffs)
        )
        return e_elec + e_rep

    return energy
