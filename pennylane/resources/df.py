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
This module contains the functions needed for resource estimation with double factorization method.
"""

from pennylane import numpy as np


def rank(factors, eigvals, tol=1e-5):
    r"""Return the double-factorization ranks for a two-electron integral tensor.

    The double factorization of a two-electron integral tensor :math:`V`, in the chemist notation,
    consists of two factorization steps. First, the tensor is factorized such that

    .. math::

        V_{ijkl} = \sum_r^R L_{ij}^{(r)} L_{kl}^{(r) T},

    where :math:`R` is the rank of the first factorization step. Then, the matrices :math:`L^{(r)}`
    are diagonalized to obtain a set of eigenvalues and eigenvectors for each matrix. These
    eigenvalues and eigenvectors are truncated with a rank :math:`M^{(r)}` for each matrix. An
    average rank :math:`M` is then computed by averaging over all :math:`M^{(r)}` ranks.

    This function computes the rank :math:`R` and the average rank :math:`M`.

    Args:
        factors (array[array[float]]): matrices (factors) obtained from factorizing the
            two-electron integral tensor
        eigvals (array[float]): eigenvalues of the matrices obtained from factorizing the
            two-electron integral tensor
        tol (float): cutoff value for discarding the negligible eigenvalues

    Returns:
        tuple(array[float]): the ranks of double-factorized two-electron integral tensor

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], requires_grad = False) / 0.5291772
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> core, one, two = qml.qchem.electron_integrals(mol)()
    >>> two = np.swapaxes(two, 1, 3) # convert to chemist's notation
    >>> l, w, v = factorize(two, 1e-5)
    >>> print(rank(l, w))
    (3, 2)
    """
    rank_r = len(factors)

    vals_flat = np.vstack(eigvals).flatten()

    vals_nonzero = [val for val in vals_flat if abs(val) > tol]

    rank_m = int(len(vals_nonzero) / len(factors))

    return rank_r, rank_m


def eta(k):
    r"""Return the value of :math:`\eta` for a given rank.

    This function computes the maximum value of :math:`\eta` such that a given rank is divisible by
    :math:`2^\eta` [`arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_].

    Args:
        k (int): the rank for which :math:`\eta` is computed

    Returns:
        int: the value of :math:`\eta` for a given rank

    **Example**

    >>> eta(26)
    1
    """
    factors = [n for n in range(1, k + 1) if k % n == 0]
    etas = [int(np.log2(f)) for f in factors if np.log2(f) % 1 == 0]
    return max(etas)
