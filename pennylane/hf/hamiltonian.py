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
from pennylane.hf.hartree_fock import generate_scf


def generate_electron_integrals(mol, core=None, active=None):
    r"""Return a function that computes the one- and two-electron integrals in the atomic orbital
    basis.

    The one- and two-electron integrals in the molecular orbital basis can be written in terms of
    the integrals in the atomic orbital basis, by recalling that
    :math:`\phi_i = \sum_{\nu}c_{\nu}^i \chi_{\nu}`, as

    .. math::

        h_{pq} = \sum_{\mu \nu} C_{p \mu} h_{\mu \nu} C_{\nu q},

    and

    .. math::

        h_{pqrs} = \sum_{\mu \nu \rho \sigma} C_{p \mu} C_{q \nu} h_{\mu \nu \rho \sigma} C_{\rho r} C_{\sigma s}.


    The :math:`h_{\mu \nu}` and :math:`h_{\mu \nu \rho \sigma}` terms refer to the elements of the
    core matrix and the electron repulsion tensor, respectively.

    Args:
        mol (Molecule): the molecule object
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that computes the core energy, the one- and two-electron integrals

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> generate_electron_integrals(mol)(*args)
    array([ 0.00000000e+00, -1.39021927e+00,  0.00000000e+00,  0.00000000e+00,
           -2.91653313e-01,  7.14439078e-01, -2.77555756e-17,  5.55111512e-17,
            1.70241443e-01,  5.55111512e-17,  1.70241443e-01,  7.01853154e-01,
            6.66133815e-16, -1.38777878e-16,  7.01853154e-01,  1.70241443e-01,
            2.22044605e-16,  1.70241443e-01, -4.44089210e-16,  6.66133815e-16,
            7.38836690e-01])
    """

    def electron_integrals(*args):
        r"""Compute the one- and two-electron integrals in the atomic orbital basis.

        Args:
            args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[float]: 1D array containing the core energy, the one- and two-electron integrals
        """
        v_fock, coeffs, fock_matrix, h_core, repulsion_tensor = generate_scf(mol)(*args)
        one = anp.einsum("qr,rs,st->qt", coeffs.T, h_core, coeffs)
        two = anp.swapaxes(
            anp.einsum(
                "ab,cd,bdeg,ef,gh->acfh", coeffs.T, coeffs.T, repulsion_tensor, coeffs, coeffs
            ),
            1,
            3,
        )
        e_core = anp.array([0.0])

        if core is None and active is None:
            return anp.concatenate((e_core, one.flatten(), two.flatten()))

        else:
            for i in core:
                e_core = e_core + 2 * one[i][i]
                for j in core:
                    e_core = e_core + 2 * two[i][j][j][i] - two[i][j][i][j]

            for p in active:
                for q in active:
                    for i in core:
                        o = anp.zeros(one.shape)
                        o[p, q] = 1.0
                        one = one + (2 * two[i][p][q][i] - two[i][p][i][q]) * o

            two = two[anp.ix_(active, active, active, active)]

            return anp.concatenate((e_core, one.flatten(), two.flatten()))

    return electron_integrals
