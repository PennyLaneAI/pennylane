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
import autograd.numpy as anp


def generate_electron_integrals(mol, occupied=None, active=None):
    """Return a function that computes the one and two electron integrals."""

    def electron_integrals(*args):
        v_fock, coeffs, fock_matrix, h_core, e_tensor = scf(mol)(*args)
        one = anp.einsum("qr,rs,st->qt", coeffs.T, h_core, coeffs)
        two = anp.swapaxes(
            anp.einsum("ab,cd,bdeg,ef,gh->acfh", coeffs.T, coeffs.T, e_tensor, coeffs, coeffs), 1, 3
        )
        core, one_elec, two_elec = get_active(one, two, occupied=occupied, active=active)
        return anp.concatenate((anp.array([core]), one_elec.flatten(), two_elec.flatten()))

    return electron_integrals


def get_active(one_body_integrals, two_body_integrals, occupied=None, active=None):
    """
    Gets integrals in some active space
    """
    # Fix data type for a few edge cases
    occupied = [] if occupied is None else occupied

    # Determine core constant
    core_constant = 0.0
    for i in occupied:
        core_constant = core_constant + 2 * one_body_integrals[i][i]
        for j in occupied:
            core_constant = core_constant + (
                2 * two_body_integrals[i][j][j][i] - two_body_integrals[i][j][i][j]
            )

    # Modified one electron integrals
    one_body_integrals_new = anp.zeros(one_body_integrals.shape)
    for u in active:
        for v in active:
            for i in occupied:
                c = 2 * two_body_integrals[i][u][v][i] - two_body_integrals[i][u][i][v]
                one_body_integrals_new = one_body_integrals_new + c * build_arr(
                    one_body_integrals.shape, (u, v)
                )

    one_body_integrals_new = one_body_integrals_new + one_body_integrals

    # Restrict integral ranges and change M appropriately
    return (
        core_constant,
        one_body_integrals_new[anp.ix_(active, active)],
        two_body_integrals[anp.ix_(active, active, active, active)],
    )
