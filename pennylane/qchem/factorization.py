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
This module contains the functions needed for two-electron tensor factorization.
"""
from pennylane import numpy as np


def factorize(two, tol):
    r"""Return a rank-$r$ factorization of a two-electron tensor.

    The second quantized electronic Hamiltonian is constructed in terms of fermionic creation,
    $a^{\dagger}$ , and annihilation, $a$, operators as

    .. math::

        H = \sum_{\alpha \in \{\uparrow, \downarrow \} } \sum_{pq} h_{pq} a_{p,\alpha}^{\dagger}
        a_{q, \alpha} + \frac{1}{2} \sum_{\alpha, \beta \in \{\uparrow, \downarrow \} } \sum_{pqrs}
        h_{pqrs} a_{p, \alpha}^{\dagger} a_{q, \beta}^{\dagger} a_{r, \beta} a_{s, \alpha},

    where :math:`h_{pq}` and :math:`h_{pqrs}` are the one- and two-electron integrals computed as

    .. math::

        h_{pq} = \int \phi_p(r)^* \left ( -\frac{\nabla_r^2}{2} - \sum_i \frac{Z_i}{|r-R_i|} \right)
        \phi_q(r) dr,

    and

    .. math::

        h_{pqrs} = \int \frac{\phi_p(r_1)^* \phi_q(r_2)^* \phi_r(r_2) \phi_s(r_1)}{|r_1 - r_2|}
        dr_1 dr_2.

    Rearranging the integrals in the chemist notation gives

    .. math::

        H = \sum_{\alpha \in \{\uparrow, \downarrow \} } \sum_{pq} T_{pq} a_{p,\alpha}^{\dagger}
        a_{q, \alpha} + \frac{1}{2} \sum_{\alpha, \beta \in \{\uparrow, \downarrow \} } \sum_{pqrs}
        V_{pqrs} a_{p, \alpha}^{\dagger} a_{q, \alpha} a_{r, \beta}^{\dagger} a_{s, \beta}.

    with

    .. math::

        T_{pq} = h_{ij} - \frac{1}{2} \sum_s h_{pssq}.


    and $V$ is the two-electron tensor in chemist notation.

    The objective of the factorization is to find a set of symmetric matrices, $L$, such that

    .. math::

           V_{ijkl} = \sum_r L_{ij}^{(r)} L_{kl}^{(r) T}.

    and the rank $r \in \mathcal{O}(n)$.

    The algorithm has the following steps.

    1. Matricize the $n \times n \times n \times n$ two-electron tensor to a $n^2 \times n^2$ matrix
    where n is the number of orbitals.

    2. Diagonalize the resulting matrix and keep the $r$ eigenvectors which their corresponding
    eigenvalues are larger than a threshold.

    3. Reshape the selected eigenvectors to $n \times n$ matrices and return them.

    Args:
        two (array[array[float]]): the two-electron repulsion tensor in the molecular orbital basis
        tol (float): cutoff value for discarding the negligible factors

    Returns:
        array[array[float]]: array of $r$ symmetric matrices approximating the two-electron tensor

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]) / 0.5291772
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> core, one, two = qml.qchem.electron_integrals(mol)()
    >>> two = np.swapaxes(two, 1, 3) # convert to chemist's notation
    >>> factorize_first(two, 1e-5)
    tensor([[[ 1.06723431e-01,  3.14003079e-15],
             [ 4.22991573e-15, -1.04898524e-01]],

            [[-8.86414958e-14, -4.25688240e-01],
             [-4.25688240e-01, -1.20077117e-13]],

            [[-8.14472824e-01,  2.01622177e-13],
             [ 2.01733446e-13, -8.28642110e-01]]], requires_grad=True)
    """
    n = two.shape[0]
    two = two.reshape(n * n, n * n)

    eigvals, eigvecs = np.linalg.eigh(two)
    eigvals = np.array([val for val in eigvals if abs(val) > tol])
    eigvecs = eigvecs[:, -len(eigvals):]

    l = eigvecs @ np.diag(np.sqrt(abs(eigvals)))

    factors = np.array([l.reshape(n, n, len(eigvals))[:, :, r] for r in range(len(eigvals))])

    eigvals, eigvecs = np.linalg.eigh(factors)
    eigvals = np.array([val for val in eigvals if np.sum(abs(eigvals)) > tol])
    eigvals = eigvals[:, -len(eigvals):]

    return factors, eigvals, eigvals
