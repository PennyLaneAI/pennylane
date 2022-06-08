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
    r"""Return the double-factorized form of a two-electron tensor.

    The second quantized electronic Hamiltonian is constructed in terms of fermionic creation,
    :math:`a^{\dagger}` , and annihilation, :math:`a`, operators as
    [`arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`_]

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

    Rearranging the integrals in the chemist notation, [11|22], gives

    .. math::

        H = \sum_{\alpha \in \{\uparrow, \downarrow \} } \sum_{pq} T_{pq} a_{p,\alpha}^{\dagger}
        a_{q, \alpha} + \frac{1}{2} \sum_{\alpha, \beta \in \{\uparrow, \downarrow \} } \sum_{pqrs}
        V_{pqrs} a_{p, \alpha}^{\dagger} a_{q, \alpha} a_{r, \beta}^{\dagger} a_{s, \beta}.

    with

    .. math::

        T_{pq} = h_{pq} - \frac{1}{2} \sum_s h_{pssq}.


    and :math:`V` is the two-electron tensor in chemist notation.

    The objective of the factorization is to find a set of symmetric matrices, :math:`L`, such that

    .. math::

           V_{ijkl} = \sum_r^R L_{ij}^{(r)} L_{kl}^{(r) T},

    with the rank :math:`R \leq n^2`. The matrices :math:`L` are further diagonalized
    and truncated in a second level of factorization.

    The algorithm has the following steps [`arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`_]:

        1. Reshape the :math:`n \times n \times n \times n` two-electron tensor to a
            :math:`n^2 \times n^2` matrix where :math:`n` is the number of orbitals.

        2. Diagonalize the resulting matrix and keep the :math:`r` eigenvectors that have
            corresponding eigenvalues larger than a threshold.

        3. Reshape the selected eigenvectors to :math:`n \times n` matrices.

        4. Diagonalize the :math:`n \times n` matrices and keep those matrices that the norm of
            their eigenvalues is larger than a threshold.

    Args:
        two (array[array[float]]): the two-electron repulsion tensor in the molecular orbital basis
            arranged in chemist notation [11|22]
        tol (float): threshold error value for discarding the negligible factors

    Returns:
        tuple(array[float]): array of symmetric matrices (factors) approximating the two-electron
        tensor, eigenvalues of the generated factors, and eigenvectors of the generated factors

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [1.398397361, 0.0, 0.0]], requires_grad = False)
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> core, one, two = qml.qchem.electron_integrals(mol)()
    >>> two = np.swapaxes(two, 1, 3) # convert to chemist's notation
    >>> factors, eigvals, eigvecs = factorize(two, 1e-5)
    >>> print(factors)
    [[[ 1.06723440e-01  9.73575768e-15]
      [ 8.36288956e-15 -1.04898533e-01]]
     [[-2.20945401e-13 -4.25688222e-01]
      [-4.25688222e-01 -2.98228790e-13]]
     [[-8.14472856e-01  5.01669019e-13]
      [ 5.01689072e-13 -8.28642140e-01]]]
    """
    n = two.shape[0]
    two = two.reshape(n * n, n * n)

    eigvals_r, eigvecs_r = np.linalg.eigh(two)
    eigvals_r = np.array([val for val in eigvals_r if abs(val) > tol])
    eigvecs_r = eigvecs_r[:, -len(eigvals_r) :]

    vectors = eigvecs_r @ np.diag(np.sqrt(abs(eigvals_r)))

    r = len(eigvals_r)
    factors = np.array([vectors.reshape(n, n, r)[:, :, k] for k in range(r)])

    eigvals_m, eigvecs_m = np.linalg.eigh(factors)
    eigvals_m = np.array([val for val in eigvals_m if np.sum(abs(eigvals_m)) > tol])
    eigvecs_m = eigvecs_m[:, -len(eigvals_m) :]

    return factors, eigvals_m, eigvecs_m
