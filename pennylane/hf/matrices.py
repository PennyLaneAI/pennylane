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
from pennylane.hf.integrals import generate_overlap


def molecular_density_matrix(n_electron, c):
    """Construct the density matrix.

    The density matrix, :math:`P`, is computed from the molecular orbital coefficients :math:`C` as

    .. math::

        P_{\mu \nu} = \sum_{i=1}^{N} C_{\mu i} C_{\nu i},

    where :math:`N = N_{electrons} / 2` is the number of occupied orbitals. Note that the total
    density matrix is the sum of the :math:`\sigma` and :math:`\betta` dennsity
    matrices, :math:`P = P^{\sigma} + P^{\betta}`.

    Args:
        n_electron (integer): number of electrons
        c (array[float]): molecular orbital coefficients

    Returns:
        array[float]: total density matrix

    **Example**

    >>> c = np.array([[-0.54828771,  1.21848441], [-0.54828771, -1.21848441]])
    >>> n_electron = 2
    >>> density_matrix(n_electron, c)
    array([[0.30061941, 0.30061941], [0.30061941, 0.30061941]])
    """
    p = anp.dot(c[:, : n_electron // 2], anp.conjugate(c[:, : n_electron // 2]).T)
    return p


def overlap_matrix(basis_functions):
    r"""Return a function that constructs the overlap matrix for a given set of basis functions.

    Args:
        basis_functions (list[BasisFunction]): basis functions

    Returns:
        function: function that constructs the overlap matrix

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> mol = Molecule(symbols, geometry)
    """

    def overlap(*args):
        r"""Construct the overlap matrix for a given set of basis functions."""
        n = len(basis_functions)
        s = anp.eye(len(basis_functions))
        for i, a in enumerate(basis_functions):
            for j, b in enumerate(basis_functions):
                if i < j:
                    if args:
                        overlap_integral = generate_overlap(a, b)([args[0][i], args[0][j]])
                    else:
                        overlap_integral = generate_overlap(a, b)()
                    o = anp.zeros((n, n))
                    o[i, j] = o[j, i] = 1.0
                    s = s + overlap_integral * o
        return s

    return overlap
