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
    r"""Return the ranks of double-factorization for a two-electron integral tensor.

    The double factorization of a two-electron integral tensor consists of two factorization steps.
    First, the tensor is factorized such that

    .. math::

        h_{ijkl} = \sum_r^R L_{ij}^{(r)} L_{kl}^{(r) T},

    where :math:`R` is the rank of the first factorization step. Then, the matrices :math:`L^{(r)}`
    are diagonalized to obtain a set of eigenvalues and eigenvectors for each matrix which can be
    truncated with a rank :math:`M^{(r)}` for each matrix.

    This function computes the rank :math:`R` and the average rank :math:`M` which is averaged over
    :math:`M^{(r)}` ranks.

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

    rank_m = int(len(vals_nonzero) / len(eigvals))

    return rank_r, rank_m


def expansion_factor(constants):
    r"""Return expansion factors that minimize the cost.

    The expansion factors are parameters, selected as integer powers of 2 as :math:`k = 2^n`, that
    determine the complexity of applying QROMs. These parameters are optimized to minimize the
    cost. In the most general form, the complexity of a QROM computation depends on the expansion
    factor as [`arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_]

    .. math::

        cost = \left \lceil \frac{a + b}{k} \right \rceil + \left \lceil \frac{c}{k} \right \rceil +
        d \left ( k + e \right ),

    where :math:`a, b, c, d, e` are constants that depend on the nature of the QROM implementation.

    To obtain the optimum values of :math:`k`, we first assume that the cost function is continues
    and use differentiation to obtain the value of :math:`k` that minimizes the cost. This value of
    :math:`k` is not necessarily an integer power of 2. We then obtain the value of :math:`n` as
    :math:`n = \log_2(k)` and compute the cost for
    :math:`n_{int}= \left \{\left \lceil n \right \rceil, \left \lfloor n \right \rfloor \right \}`.
    The value of :math:`n_{int}` that gives the smaller cost is used to compute :math:`k_{opt}`.

    Args:
        constants (tuple[float]): constants determining the QROM complexity

    Returns:
        int: the expansion factor

    **Example**
    """
    a, b, c, d, e = constants
    n = np.log2(((a + b + c) / d) ** 0.5)
    k = np.array([2 ** np.floor(n), 2 ** np.ceil(n)])
    cost = np.ceil((a + b) / k) + np.ceil(c / k) + d * (k + e)
    k_opt = int(k[np.argmin(cost)])

    return k_opt
