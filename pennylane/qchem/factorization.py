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
import pennylane as qml


def factorize(two_electron, tol_factor=1.0e-5, tol_eigval=1.0e-5):
    r"""Return the double-factorized form of a two-electron integral tensor.

    The two-electron tensor :math:`V`, in
    `chemist notation <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`_, is first
    factorized in terms of symmetric matrices :math:`L^{(r)}` such that
    :math:`V_{ijkl} = \sum_r^R L_{ij}^{(r)} L_{kl}^{(r) T}`. The rank :math:`R` is determined by a
    threshold error. Then, each matrix :math:`L^{(r)}` is diagonalized and its eigenvalues (and
    corresponding eigenvectors) are truncated at a threshold error.

    Args:
        two_electron (array[array[float]]): two-electron integral tensor in the molecular orbital
            basis arranged in chemist notation
        tol_factor (float): threshold error value for discarding the negligible factors
        tol_eigval (float): threshold error value for discarding the negligible factor eigenvalues

    Returns:
        tuple(array[array[float]], list[array[float]], list[array[float]]): tuple containing
        symmetric matrices (factors) approximating the two-electron integral tensor, truncated
        eigenvalues of the generated factors, and truncated eigenvectors of the generated factors

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [1.398397361, 0.0, 0.0]], requires_grad = False)
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> core, one, two = qml.qchem.electron_integrals(mol)()
    >>> two = np.swapaxes(two, 1, 3) # convert to chemist notation
    >>> factors, eigvals, eigvecs = factorize(two, 1e-5, 1e-5)
    >>> print(factors)
    [[[ 1.06723440e-01  9.73575768e-15]
      [ 8.36288956e-15 -1.04898533e-01]]
     [[-2.20945401e-13 -4.25688222e-01]
      [-4.25688222e-01 -2.98228790e-13]]
     [[-8.14472856e-01  5.01669019e-13]
      [ 5.01689072e-13 -8.28642140e-01]]]

    .. details::
        :title: Theory

        The second quantized electronic Hamiltonian is constructed in terms of fermionic creation,
        :math:`a^{\dagger}` , and annihilation, :math:`a`, operators as
        [`arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`_]

        .. math::

            H = \sum_{\alpha \in \{\uparrow, \downarrow \} } \sum_{pq} h_{pq} a_{p,\alpha}^{\dagger}
            a_{q, \alpha} + \frac{1}{2} \sum_{\alpha, \beta \in \{\uparrow, \downarrow \} } \sum_{pqrs}
            h_{pqrs} a_{p, \alpha}^{\dagger} a_{q, \beta}^{\dagger} a_{r, \beta} a_{s, \alpha},

        where :math:`h_{pq}` and :math:`h_{pqrs}` are the one- and two-electron integrals computed
        as

        .. math::

            h_{pq} = \int \phi_p(r)^* \left ( -\frac{\nabla_r^2}{2} - \sum_i \frac{Z_i}{|r-R_i|} \right)
            \phi_q(r) dr,

        and

        .. math::

            h_{pqrs} = \int \frac{\phi_p(r_1)^* \phi_q(r_2)^* \phi_r(r_2) \phi_s(r_1)}{|r_1 - r_2|}
            dr_1 dr_2.

        The two-electron integrals can be rearranged in the so-called chemist notation which gives

        .. math::

            V_{pqrs} = \int \frac{\phi_p(r_1)^* \phi_q(r_1)^* \phi_r(r_2) \phi_s(r_2)}{|r_1 - r_2|}
            dr_1 dr_2,

        and the molecular Hamiltonian can be rewritten as

        .. math::

            H = \sum_{\alpha \in \{\uparrow, \downarrow \} } \sum_{pq} T_{pq} a_{p,\alpha}^{\dagger}
            a_{q, \alpha} + \frac{1}{2} \sum_{\alpha, \beta \in \{\uparrow, \downarrow \} } \sum_{pqrs}
            V_{pqrs} a_{p, \alpha}^{\dagger} a_{q, \alpha} a_{r, \beta}^{\dagger} a_{s, \beta},

        with

        .. math::

            T_{pq} = h_{pq} - \frac{1}{2} \sum_s h_{pssq}.


        This notation allows a low-rank factorization of the two-electron integral. The objective of
        the factorization is to find a set of symmetric matrices, :math:`L^{(r)}`, such that

        .. math::

               V_{ijkl} = \sum_r^R L_{ij}^{(r)} L_{kl}^{(r) T},

        with the rank :math:`R \leq n^2` where :math:`n` is the number of molecular orbitals. The
        matrices :math:`L^{(r)}` are diagonalized and for each matrix the eigenvalues that are
        smaller than a given threshold (and their corresponding eigenvectors) are discarded.

        The factorization algorithm has the following steps
        [`arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`_]:

        - Reshape the :math:`n \times n \times n \times n` two-electron tensor to a
          :math:`n^2 \times n^2` matrix where :math:`n` is the number of orbitals.

        - Diagonalize the resulting matrix and keep the :math:`r` eigenvectors that have
          corresponding eigenvalues larger than a threshold.

        - Multiply the eigenvectors by the square root of the eigenvalues to obtain
          matrices :math:`L^{(r)}`.

        - Reshape the selected eigenvectors to :math:`n \times n` matrices.

        - Diagonalize the :math:`n \times n` matrices and for each matrix keep the eigenvalues (and
          their corresponding eigenvectors) that are larger than a threshold.
    """
    shape = two_electron.shape

    if len(shape) != 4 or len(set(shape)) != 1:
        raise ValueError("The two-electron repulsion tensor must have a (N x N x N x N) shape.")

    n = shape[0]
    two = two_electron.reshape(n * n, n * n)

    eigvals_r, eigvecs_r = np.linalg.eigh(two)
    eigvals_r = np.array([val for val in eigvals_r if abs(val) > tol_factor])

    eigvecs_r = eigvecs_r[:, -len(eigvals_r) :]

    if eigvals_r.size == 0:
        raise ValueError(
            "All factors are discarded. Consider decreasing the first threshold error."
        )

    vectors = eigvecs_r @ np.diag(np.sqrt(eigvals_r))

    r = len(eigvals_r)
    factors = np.array([vectors.reshape(n, n, r)[:, :, k] for k in range(r)])

    eigvals, eigvecs = np.linalg.eigh(factors)
    eigvals_m = []
    eigvecs_m = []
    for n, eigval in enumerate(eigvals):
        idx = [i for i, v in enumerate(eigval) if abs(v) > tol_eigval]
        eigvals_m.append(eigval[idx])
        eigvecs_m.append(eigvecs[n][idx])

    if np.sum([len(v) for v in eigvecs_m]) == 0:
        raise ValueError(
            "All eigenvectors are discarded. Consider decreasing the second threshold error."
        )

    return factors, eigvals_m, eigvecs_m


def basis_rotation(one_electron, two_electron, tol_factor, tol_eigval, error):
    r"""Return Hamiltonian coefficients and diagonalizing gates obtained with the basis rotation
    grouping method.

    .. details::
        :title: Theory

        A second-quantized molecular Hamiltonian can be constructed in the chemist notation format
        as

        .. math::

            H = \sum_{\alpha \in \{\uparrow, \downarrow \} } \sum_{pq} T_{pq} a_{p,\alpha}^{\dagger}
            a_{q, \alpha} + \frac{1}{2} \sum_{\alpha, \beta \in \{\uparrow, \downarrow \} } \sum_{pqrs}
            V_{pqrs} a_{p, \alpha}^{\dagger} a_{q, \alpha} a_{r, \beta}^{\dagger} a_{s, \beta},

        where :math:`V_{pqrs}` denotes a two-electron integral in the chemist notation and
        :math:`T_{pq}` is obtained from the one- and two electron integrals, :math:`h_{pq}` and
        :math:`h_{pssq}`, as

        .. math::

            T_{pq} = h_{pq} - \frac{1}{2} \sum_s h_{pssq}.

        The tensor :math:`V` can be converted to a matrix which is indexed by the indices :math:`pq`
        and :math:`rs` and eigendecomposed up to a rank :math:`R` to give

        .. math::

            V_{pqrs} = \sum_r^R w^{(r)} L_{pq}^{(r)} L_{rs}^{(r) T},

        where :math:`w` and :math:`L` denote the eigenvalues and eigenvectors of the matrix,
        respectively. The molecular Hamiltonian can then be rewritten as

        .. math::

            H = \sum_{\alpha \in \{\uparrow, \downarrow \} } \sum_{pq} T_{pq} a_{p,\alpha}^{\dagger}
            a_{q, \alpha} + \frac{1}{2} w^{(r)} \sum_r^R \left ( \sum_{\alpha, \beta \in \{\uparrow, \downarrow \} } \sum_{pq}
            L_{pq}^{(r)} a_{p, \alpha}^{\dagger} a_{q, \alpha} \right )^2.

        The orbital basis can be rotated such that each :math:`T` and :math:`L^{(r)}` matrix is
        diagonal. The Hamiltonian can then be written as

        .. math::

            H = U_0 \left ( \sum_p d_p n_p \right ) U_0^{\dagger} + \sum_l^L U_l \left ( \sum_{pq}
            d_{pq}^{(l)} n_p n_q \right ) U_l^{\dagger}

        where the coefficients :math:`d` are obtained by diagonalizing the :math:`T` and
        :math:`L^{(r)}` matrices. This function returns the coefficients :math:`d` and the
        eignvectors of the :math:`T` and :math:`L^{(r)}` matrices.
    """
    two_electron = np.swapaxes(two_electron, 1, 3)

    factors = qml.qchem.factorize(two_electron, tol_factor, tol_eigval)[0]
    eigvals, eigvecs = np.linalg.eigh(factors)

    t_matrix = one_electron - 0.5 * np.einsum("illj", two_electron)
    t_eigvals, t_eigvecs = np.linalg.eigh(t_matrix)

    d_eigvals = []
    d_eigvecs = []

    for i in range(len(eigvals)):
        for j, eigval in enumerate(eigvals[i]):
            eigvecs[i][j] = eigvecs[i][j]
        d_eigval, d_eigvec = np.linalg.eigh(eigvecs[i])
        d_eigvals.append(np.array(d_eigval) * (0.5 * eigval))
        d_eigvecs.append(d_eigvec * (0.5 * eigval))

    coeffs = [np.array(t_eigvals)] + [np.outer(x, x).flatten() for x in d_eigvals]
    eigvec = [t_eigvecs] + d_eigvecs

    return coeffs, eigvec
