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

    def near_k(self, n_opt):
        return np.array([2 ** np.floor(n_opt), 2 ** np.ceil(n_opt)])

    def expansion_factor(self, n, l, bp1, bo, bp2, xi, beth):
        r"""Return expansion factors that minimize the cost.

        The expansion factors are parameters chosen as powers of 2 that determine the complexity of
        applying QROMs.

        k1: QROM for state preparation on the first register
        k2: QROM for outputing data from the l register

        k3: inverse the k2 QROM
        k4: inverse the k1 QROM

        k5: QROM for the rotation

        k6:

        """
        # kp1
        n1 = np.log2(((l + 1) / bp1) ** 0.5)
        k1 = np.array([2 ** np.floor(n1), 2 ** np.ceil(n1)])
        cost = np.ceil((l + 1) / k1) + bp1 * (k1 - 1)
        k1_opt = int(k1[np.argmin(cost)])

        # ko
        n2 = np.log2(((l + 1) / bo) ** 0.5)
        k2 = np.array([2 ** np.floor(n2), 2 ** np.ceil(n2)])
        cost = np.ceil((l + 1) / k2) + bo * (k2 - 1)
        k2_opt = int(k2[np.argmin(cost)])

        # kpp1
        n3 = np.log2((l + 1) ** 0.5)
        k3 = np.array([2 ** np.floor(n3), 2 ** np.ceil(n3)])
        cost = np.ceil((l + 1) / k3) + k3
        k3_opt = int(k3[np.argmin(cost)])

        # kppo
        k4_opt = k3_opt

        # kr
        n5 = np.log2(((2 * l * xi - n / 2) / (n * beth)) ** 0.5)
        k5 = np.array([2 ** np.floor(n5), 2 ** np.ceil(n5)])
        cost = np.ceil((l * xi + n / 2) / k5) + np.ceil((l * xi) / k5) + n * beth * k5
        k5_opt = int(k5[np.argmin(cost)])

        # kpr
        n6 = np.log2((l * xi - n / 4) ** 0.5)
        k6 = np.array([2 ** np.floor(n6), 2 ** np.ceil(n6)])
        cost = np.ceil((l * xi + n / 2) / k6) + np.ceil((l * xi) / k6) + 2 * k6
        k6_opt = int(k6[np.argmin(cost)])

        # kp2
        n7 = np.log2(((2 * l * xi - n / 2) / (2 * bp2)) ** 0.5)
        k7 = np.array([2 ** np.floor(n7), 2 ** np.ceil(n7)])
        cost = np.ceil((l * xi + n / 2) / k7) + np.ceil((l * xi) / k7) + 2 * bp2 * (k7 - 1)
        k7_opt = int(k7[np.argmin(cost)])

        # kpp2
        k8_opt = k6_opt

        return k1_opt, k2_opt, k3_opt, k4_opt, k5_opt, k6_opt, k7_opt, k8_opt
