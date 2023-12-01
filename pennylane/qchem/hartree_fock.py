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
This module contains the functions needed for performing the self-consistent-field calculations.
"""

import itertools

import pennylane as qml

from .matrices import core_matrix, mol_density_matrix, overlap_matrix, repulsion_tensor


def scf(mol, n_steps=50, tol=1e-8):
    r"""Return a function that performs the self-consistent-field calculations.

    In the Hartree-Fock method, molecular orbitals are typically constructed as a linear combination
    of atomic orbitals

    .. math::

        \phi_i(r) = \sum_{\mu} C_{\mu i} \chi_{\mu}(r),

    with coefficients :math:`C_{\mu i}` that are initially unknown. The self-consistent-field
    iterations are performed to find a converged set of molecular orbital coefficients that minimize
    the total energy of the molecular system. This optimization problem can be reduced to solving a
    linear system of equations which are usually written as

    .. math::

        FC = SCE,

    where :math:`E` is a diagonal matrix of eigenvalues, representing the molecular orbital
    energies, :math:`C` is the matrix of molecular orbital coefficients, :math:`S` is the overlap
    matrix and :math:`F` is the Fock matrix, which also depends on the coefficients. Fixing an
    initial guess :math:`C_0`, the corresponding :math:`F_0` is built and the system
    :math:`F_0C_0 = SC_0E` is solved to obtain a solution :math:`C_1`. This process is iteratively
    repeated until the coefficients are converged.

    The key step in in this process is constructing the Fock matrix which is defined as

    .. math::

        F = H + \frac{1}{2} J - K,

    where :math:`H`, :math:`J` and :math:`K` are the core Hamiltonian matrix, Coulomb matrix and
    exchange matrix, respectively. The entries of :math:`H` are computed from the electronic kinetic
    energy and the electron-nuclear attraction integrals, which are integrals over atomic basis
    functions. The elements of the :math:`J` and :math:`K` matrices are obtained from the Coulomb
    and exchange integrals over the basis functions.

    Following the procedure in
    [`Lehtola et al. Molecules 2020, 25, 1218 <https://www.mdpi.com/1420-3049/25/5/1218>`_], we
    express the molecular orbital coefficients in terms of a matrix :math:`X` as
    :math:`C = X \tilde{C}` which gives the following transformed equation

    .. math::

         \tilde{F} \tilde{C} = \tilde{S} \tilde{C} E,

    where :math:`\tilde{F} = X^T F X`, :math:`\tilde{S} = X^T S X` and :math:`S` is the overlap
    matrix. We chose :math:`X` such that :math:`\tilde{S} = 1` as

    .. math::

        X = V \Lambda^{-1/2} V^T,

    where :math:`V` and :math:`\Lambda` are the eigenvectors and eigenvalues of :math:`S`,
    respectively. This gives the eigenvalue equation

    .. math::

         \tilde{F}\tilde{C} = \tilde{C}E,

    which is solved with conventional methods iteratively.

    Args:
        mol (~qchem.molecule.Molecule): the molecule object
        n_steps (int): the number of iterations
        tol (float): convergence tolerance

    Returns:
        function: function that performs the self-consistent-field calculations

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> v_fock, coeffs, fock_matrix, h_core, rep_tensor = scf(mol)(*args)
    >>> v_fock
    array([-0.67578019,  0.94181155])
    """

    def _scf(*args):
        r"""Perform the self-consistent-field iterations.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            tuple(array[float]): eigenvalues of the Fock matrix, molecular orbital coefficients,
            Fock matrix, core matrix
        """
        basis_functions = mol.basis_set
        charges = mol.nuclear_charges
        r = mol.coordinates
        n_electron = mol.n_electrons

        if r.requires_grad:
            args_r = [[args[0][i]] * mol.n_basis[i] for i in range(len(mol.n_basis))]
            args_ = [*args] + [qml.math.vstack(list(itertools.chain(*args_r)))]
            rep_tensor = repulsion_tensor(basis_functions)(*args_[1:])
            s = overlap_matrix(basis_functions)(*args_[1:])
            h_core = core_matrix(basis_functions, charges, r)(*args_)
        else:
            rep_tensor = repulsion_tensor(basis_functions)(*args)
            s = overlap_matrix(basis_functions)(*args)
            h_core = core_matrix(basis_functions, charges, r)(*args)

        rng = qml.math.random.default_rng(2030)
        s = s + qml.math.diag(rng.random(len(s)) * 1.0e-12)

        w, v = qml.math.linalg.eigh(s)
        x = v @ qml.math.diag(1.0 / qml.math.sqrt(w)) @ v.T

        eigvals, w_fock = qml.math.linalg.eigh(
            x.T @ h_core @ x
        )  # initial guess for the scf problem
        coeffs = x @ w_fock

        p = mol_density_matrix(n_electron, coeffs)

        for _ in range(n_steps):
            j = qml.math.einsum("pqrs,rs->pq", rep_tensor, p)
            k = qml.math.einsum("psqr,rs->pq", rep_tensor, p)

            fock_matrix = h_core + 2 * j - k

            eigvals, w_fock = qml.math.linalg.eigh(x.T @ fock_matrix @ x)

            coeffs = x @ w_fock

            p_update = mol_density_matrix(n_electron, coeffs)

            if qml.math.linalg.norm(p_update - p) <= tol:
                break

            p = p_update

        mol.mo_coefficients = coeffs

        return eigvals, coeffs, fock_matrix, h_core, rep_tensor

    return _scf


def nuclear_energy(charges, r):
    r"""Return a function that computes the nuclear-repulsion energy.

    The nuclear-repulsion energy is computed as

    .. math::

        \sum_{i>j}^n \frac{q_i q_j}{r_{ij}},

    where :math:`q`, :math:`r` and :math:`n` denote the nuclear charges (atomic numbers), nuclear
    positions and the number of nuclei, respectively.

    Args:
        charges (list[int]): nuclear charges in atomic units
        r (array[float]): nuclear positions

    Returns:
        function: function that computes the nuclear-repulsion energy

    **Example**

    >>> symbols  = ['H', 'F']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], requires_grad = True)
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> args = [mol.coordinates]
    >>> e = nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)
    >>> print(e)
    4.5
    """

    def _nuclear_energy(*args):
        r"""Compute the nuclear-repulsion energy.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[float]: nuclear-repulsion energy
        """
        if r.requires_grad:
            coor = args[0]
        else:
            coor = r
        e = qml.math.array([0.0])
        for i, r1 in enumerate(coor):
            for j, r2 in enumerate(coor[i + 1 :]):
                e = e + (charges[i] * charges[i + j + 1] / qml.math.linalg.norm(r1 - r2))
        return e

    return _nuclear_energy


def hf_energy(mol):
    r"""Return a function that computes the Hartree-Fock energy.

    Args:
        mol (~qchem.molecule.Molecule): the molecule object

    Returns:
        function: function that computes the Hartree-Fock energy

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> hf_energy(mol)(*args)
    -1.065999461545263
    """

    def _hf_energy(*args):
        r"""Compute the Hartree-Fock energy.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            float: the Hartree-Fock energy
        """
        _, coeffs, fock_matrix, h_core, _ = scf(mol)(*args)
        e_rep = nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)
        e_elec = qml.math.einsum(
            "pq,qp", fock_matrix + h_core, mol_density_matrix(mol.n_electrons, coeffs)
        )
        energy = e_elec + e_rep
        return energy.reshape(())

    return _hf_energy
