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
This module contains the functions needed for computing matrices.
"""

import autograd.numpy as anp
from pennylane.hf.matrices import (
    core_matrix,
    repulsion_tensor,
    overlap_matrix,
    molecular_density_matrix,
)


def generate_hartree_fock(mol, n_steps=50, tol=1e-8):
    r"""Return a function that performs the self-consistent-field iterations."""

    def hartree_fock(*args):
        r"""Perform the self-consistent-field iterations."""
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
    """
    Generates the repulsion between nuclei of the atoms in a molecule
    """

    def nuclear(*args):
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
    """ """

    def energy(*args):
        v_fock, coeffs, fock_matrix, h_core = generate_hartree_fock(mol)(*args)
        e_rep = nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)
        e_elec = anp.einsum(
            "pq,qp", fock_matrix + h_core, molecular_density_matrix(mol.n_electrons, coeffs)
        )
        return e_elec + e_rep

    return energy
