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

from functools import partial

import numpy as np
import scipy as sp

import pennylane as qml


def factorize(two_electron, tol_factor=1.0e-5, tol_eigval=1.0e-5, cholesky=False):
    r"""Return the double-factorized form of a two-electron integral tensor in spatial basis.

    The two-electron tensor :math:`V`, in
    `chemist notation <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`_,
    is first factorized in terms of symmetric matrices :math:`L^{(r)}` such that
    :math:`V_{ijkl} = \sum_r^R L_{ij}^{(r)} L_{kl}^{(r) T}`. The rank :math:`R` is
    determined by a threshold error. Then, each matrix :math:`L^{(r)}` is diagonalized
    and its eigenvalues (and corresponding eigenvectors) are truncated at a threshold error.

    Args:
        two_electron (array[array[float]]): two-electron integral tensor in the molecular orbital
            basis arranged in chemist notation
        tol_factor (float): threshold error value for discarding the negligible factors
        tol_eigval (float): threshold error value for discarding the negligible factor eigenvalues
        cholesky (bool): use Cholesky decomposition for obtaining the symmetric matrices
            :math:`L^{(r)}` instead of eigendecomposition.

    Returns:
        tuple(array[array[float]], list[array[float]], list[array[float]]): tuple containing
        symmetric matrices (factors) approximating the two-electron integral tensor, truncated
        eigenvalues of the generated factors, and truncated eigenvectors of the generated factors

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0],
    ...                      [1.398397361, 0.0, 0.0]], requires_grad=False)
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

        - Decompose the resulting matrix either via Cholesky decomposition or
          via eigenvalue decomposition.

        - In the Cholesky decomposition, we build matrices :math:`L^{(r)}` as
          :math:`v_iv_i^{\dagger}`, where each vector :math:`v_i` is constructed
          to explain the part of the matrix that hasn't been explained by previous
          vectors and reduce the error term. We build :math:`r` such matrices that
          result in an overall approximation error less than the threshold.

        - In the eigenvalue decomposition, we keep the :math:`r` eigenvectors that
          have corresponding eigenvalues larger than a threshold and multiply these
          eigenvectors by the square root of their eigenvalues to obtain
          matrices :math:`L^{(r)}`.

        - Diagonalize the :math:`L^{(r)}` (:math:`n \times n`) matrices and for each
          matrix keep the eigenvalues (and their corresponding eigenvectors) that are
          larger than a threshold.
    """
    shape = qml.math.shape(two_electron)

    if len(shape) != 4 or len(set(shape)) != 1:
        raise ValueError("The two-electron repulsion tensor must have a (N x N x N x N) shape.")

    two_body_tensor = qml.math.reshape(two_electron, (shape[0] * shape[1], -1))
    interface = qml.math.get_interface(two_body_tensor)

    if cholesky:
        factors = _double_factorization_cholesky(two_body_tensor, tol_factor, shape, interface)
    else:
        factors = _double_factorization_eigen(two_body_tensor, tol_factor, shape, interface)

    eigvals, eigvecs = qml.math.linalg.eigh(factors)
    eigvals_m, eigvecs_m = [], []
    for n, eigval in enumerate(eigvals):
        idx = [i for i, v in enumerate(eigval) if abs(v) > tol_eigval]
        eigvals_m.append(eigval[idx])
        eigvecs_m.append(eigvecs[n][idx])

    if np.sum([len(v) for v in eigvecs_m]) == 0:
        raise ValueError(
            "All eigenvectors are discarded. Consider decreasing the second threshold error."
        )

    return factors, eigvals_m, eigvecs_m


def _double_factorization_eigen(two, tol_factor=1.0e-10, shape=None, interface=None):
    """Double factorization via generalized eigen decomposition"""
    eigvals_r, eigvecs_r = qml.math.linalg.eigh(two)
    eigvals_r = qml.math.array([val for val in eigvals_r if abs(val) > tol_factor])

    eigvecs_r = eigvecs_r[:, -len(eigvals_r) :]
    if eigvals_r.size == 0:
        raise ValueError(
            "All factors are discarded. Consider decreasing the first threshold error."
        )
    vectors = eigvecs_r @ qml.math.diag(qml.math.sqrt(eigvals_r))

    n, r = shape[0], len(eigvals_r)
    factors = qml.math.array([vectors.reshape(n, n, r)[:, :, k] for k in range(r)], like=interface)
    return factors


def _double_factorization_cholesky(two, tol_factor=1.0e-10, shape=None, interface=None):
    """Double factorization via `Cholesky decomposition <https://en.wikipedia.org/wiki/Cholesky_decomposition>`_"""
    n2 = shape[0] * shape[1]
    cholesky_vecs = qml.math.zeros((n2, n2 + 1), like=interface)
    cholesky_diag = qml.math.array(qml.math.diagonal(two).real, like=interface)

    for idx in range(n2 + 1):
        if (max_err := qml.math.max(cholesky_diag)) < tol_factor:
            cholesky_vecs = cholesky_vecs[:, :idx]
            break

        max_idx = qml.math.argmax(cholesky_diag)
        cholesky_mat = cholesky_vecs[:, :idx]
        cholesky_vec = (
            two[:, max_idx] - cholesky_mat @ cholesky_mat[max_idx].conj()
        ) / qml.math.sqrt(max_err)

        cholesky_vecs[:, idx] = cholesky_vec
        cholesky_diag -= qml.math.abs(cholesky_vec) ** 2

    factors = cholesky_vecs.T.reshape(-1, shape[0], shape[0])
    return factors


def basis_rotation(one_electron, two_electron, tol_factor=1.0e-5):
    r"""Return the grouped coefficients and observables of a molecular Hamiltonian and the basis
    rotation unitaries obtained with the basis rotation grouping method.

    Args:
        one_electron (array[float]): one-electron integral matrix in the molecular orbital basis
        two_electron (array[array[float]]): two-electron integral tensor in the molecular orbital
            basis arranged in chemist notation
        tol_factor (float): threshold error value for discarding the negligible factors

    Returns:
        tuple(list[array[float]], list[list[Observable]], list[array[float]]): tuple containing
        grouped coefficients, grouped observables and basis rotation transformation matrices

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0],
    ...                      [1.398397361, 0.0, 0.0]], requires_grad=False)
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> core, one, two = qml.qchem.electron_integrals(mol)()
    >>> coeffs, ops, unitaries = basis_rotation(one, two, tol_factor=1.0e-5)
    >>> print(coeffs)
    [array([ 0.84064649, -2.59579282,  0.84064649,  0.45724992,  0.45724992]),
     array([ 9.57150297e-05,  5.60006390e-03,  9.57150297e-05,  2.75092558e-03,
            -9.73801723e-05, -2.79878310e-03, -9.73801723e-05, -2.79878310e-03,
            -2.79878310e-03, -2.79878310e-03,  2.84747318e-03]),
     array([ 0.04530262, -0.04530262, -0.04530262, -0.04530262, -0.04530262,
            0.09060523,  0.04530262]),
     array([-0.66913628,  1.6874169 , -0.66913628,  0.16584151, -0.68077716,
            0.16872663, -0.68077716,  0.16872663,  0.16872663,  0.16872663,
            0.17166195])]

    .. details::
        :title: Theory

        A second-quantized molecular Hamiltonian can be constructed in the
        `chemist notation <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`_ format
        following Eq. (1) of
        [`PRX Quantum 2, 030305, 2021 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030305>`_]
        as

        .. math::

            H = \sum_{\alpha \in \{\uparrow, \downarrow \} } \sum_{pq} T_{pq} a_{p,\alpha}^{\dagger}
            a_{q, \alpha} + \frac{1}{2} \sum_{\alpha, \beta \in \{\uparrow, \downarrow \} } \sum_{pqrs}
            V_{pqrs} a_{p, \alpha}^{\dagger} a_{q, \alpha} a_{r, \beta}^{\dagger} a_{s, \beta},

        where :math:`V_{pqrs}` denotes a two-electron integral in the chemist notation and
        :math:`T_{pq}` is obtained from the one- and two-electron integrals, :math:`h_{pq}` and
        :math:`h_{pqrs}`, as

        .. math::

            T_{pq} = h_{pq} - \frac{1}{2} \sum_s h_{pssq}.

        The tensor :math:`V` can be converted to a matrix which is indexed by the indices :math:`pq`
        and :math:`rs` and eigendecomposed up to a rank :math:`R` to give

        .. math::

            V_{pqrs} = \sum_r^R L_{pq}^{(r)} L_{rs}^{(r) T},

        where :math:`L` denotes the matrix of eigenvectors of the matrix :math:`V`. The molecular
        Hamiltonian can then be rewritten following Eq. (7) of
        [`Phys. Rev. Research 3, 033055, 2021 <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.033055>`_]
        as

        .. math::

            H = \sum_{\alpha \in \{\uparrow, \downarrow \} } \sum_{pq} T_{pq} a_{p,\alpha}^{\dagger}
            a_{q, \alpha} + \frac{1}{2} \sum_r^R \left ( \sum_{\alpha \in \{\uparrow, \downarrow \} } \sum_{pq}
            L_{pq}^{(r)} a_{p, \alpha}^{\dagger} a_{q, \alpha} \right )^2.

        The orbital basis can be rotated such that each :math:`T` and :math:`L^{(r)}` matrix is
        diagonal. The Hamiltonian can then be written following Eq. (2) of
        [`npj Quantum Information, 7, 23 (2021) <https://www.nature.com/articles/s41534-020-00341-7>`_]
        as

        .. math::

            H = U_0 \left ( \sum_p d_p n_p \right ) U_0^{\dagger} + \sum_r^R U_r \left ( \sum_{pq}
            d_{pq}^{(r)} n_p n_q \right ) U_r^{\dagger},

        where the coefficients :math:`d` are obtained by diagonalizing the :math:`T` and
        :math:`L^{(r)}` matrices. The number operators :math:`n_p = a_p^{\dagger} a_p` can be
        converted to qubit operators using

        .. math::

            n_p = \frac{1-Z_p}{2},

        where :math:`Z_p` is the Pauli :math:`Z` operator applied to qubit :math:`p`. This gives
        the qubit Hamiltonian

        .. math::

           H = U_0 \left ( \sum_p O_p^{(0)} \right ) U_0^{\dagger} + \sum_r^R U_r \left ( \sum_{q} O_q^{(r)} \right ) U_r^{\dagger},

        where :math:`O = \sum_i c_i P_i` is a linear combination of Pauli words :math:`P_i` that are
        a tensor product of Pauli :math:`Z` and Identity operators. This allows all the Pauli words
        in each of the :math:`O` terms to be measured simultaneously. This function returns the
        coefficients and the Pauli words grouped for each of the :math:`O` terms as well as the
        basis rotation transformation matrices that are constructed from the eigenvectors of the
        :math:`T` and :math:`L^{(r)}` matrices. Each column of the transformation matrix is an
        eigenvector of the corresponding :math:`T` or :math:`L^{(r)}` matrix.
    """

    num_orbitals = one_electron.shape[0] * 2
    one_body_tensor, chemist_two_body_tensor = _chemist_transform(one_electron, two_electron)
    chemist_one_body_tensor = np.kron(one_body_tensor, np.eye(2))  # account for spin
    t_eigvals, t_eigvecs = np.linalg.eigh(chemist_one_body_tensor)

    factors, _, _ = factorize(chemist_two_body_tensor, tol_factor=tol_factor)
    factors = [np.kron(factor, np.eye(2)) for factor in factors]  # account for spin

    v_coeffs, v_unitaries = np.linalg.eigh(factors)
    indices = [np.argsort(v_coeff)[::-1] for v_coeff in v_coeffs]
    v_coeffs = [v_coeff[indices[idx]] for idx, v_coeff in enumerate(v_coeffs)]
    v_unitaries = [v_unitary[:, indices[idx]] for idx, v_unitary in enumerate(v_unitaries)]

    ops_t = 0.0
    for p in range(num_orbitals):
        ops_t += 0.5 * t_eigvals[p] * (qml.Identity(p) - qml.Z(p))

    ops_l = []
    for idx in range(len(factors)):
        ops_l_ = 0.0
        for p in range(num_orbitals):
            for q in range(num_orbitals):
                ops_l_ += (
                    v_coeffs[idx][p]
                    * v_coeffs[idx][q]
                    * 0.25
                    * (
                        qml.Identity(p)
                        - qml.Z(p)
                        - qml.Z(q)
                        + (qml.Identity(p) if p == q else (qml.Z(p) @ qml.Z(q)))
                    )
                )
        ops_l.append(ops_l_)

    ops = [ops_t] + ops_l

    c_group, o_group = [], []
    for op in ops:
        c_g, o_g = op.simplify().terms()
        c_group.append(c_g)
        o_group.append(o_g)

    u_transform = list([t_eigvecs] + list(v_unitaries))  # Inverse of diagonalizing unitaries

    return c_group, o_group, u_transform


def _chemist_transform(one_body_tensor=None, two_body_tensor=None, spatial_basis=True):
    r"""Transforms one- and two-body terms in physicists' notation to `chemists' notation <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`_\ .

    This converts the input two-body tensor :math:`h_{pqrs}` that constructs :math:`\sum_{pqrs} h_{pqrs} a^\dagger_p a^\dagger_q a_r a_s`
    to a transformed two-body tensor :math:`V_{pqrs}` that follows the chemists' convention to construct :math:`\sum_{pqrs} V_{pqrs} a^\dagger_p a_q a^\dagger_r a_s`
    in the spatial basis. During the tranformation, some extra one-body terms come out. These are returned as a one-body tensor :math:`T_{pq}` in the
    chemists' notation either as is or after summation with the input one-body tensor :math:`h_{pq}`, if provided.

    Args:
        one_body_tensor (array[float]): a one-electron integral tensor giving the :math:`h_{pq}`.
        two_body_tensor (array[float]): a two-electron integral tensor giving the :math:`h_{pqrs}`.
        spatial_basis (bool): True if the integral tensor are passed in spatial-orbital basis. False if they are in spin basis.

    Returns:
        tuple(array[float], array[float]) or tuple(array[float],): transformed one-body tensor :math:`T_{pq}` and two-body tensor :math:`V_{pqrs}` for the provided terms.

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0],
    ...                      [1.398397361, 0.0, 0.0]], requires_grad=False)
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> core, one, two = qml.qchem.electron_integrals(mol)()
    >>> qml.qchem.factorization._chemist_transform(two_body_tensor=two, spatial_basis=True)
    (tensor([[-0.427983, -0.      ],
             [-0.      , -0.439431]], requires_grad=True),
    tensor([[[[0.337378, 0.      ],
             [0.       , 0.331856]],
             [[0.      , 0.090605],
             [0.090605 , 0.      ]]],
            [[[0.      , 0.090605],
             [0.090605 , 0.      ]],
             [[0.331856, 0.      ],
             [0.       , 0.348826]]]], requires_grad=True))

    .. details::
        :title: Theory

        The two-electron integral in physicists' notation is defined as:

        .. math::

            \langle pq \vert rs \rangle = h_{pqrs} = \int \frac{\chi^*_{p}(x_1) \chi^*_{q}(x_2) \chi_{r}(x_1) \chi_{s}(x_2)}{|r_1 - r_2|} dx_1 dx_2,

        while in chemists' notation it is written as:

        .. math::

            [pq \vert rs] = V_{pqrs} = \int \frac{\chi^*_{p}(x_1) \chi_{q}(x_1) \chi^*_{r}(x_2) \chi_{s}(x_2)}{|r_1 - r_2|} dx_1 dx_2.

        In the spin basis, this index reordering :math:`pqrs \rightarrow psrq` leads to formation of one-body terms :math:`h_{prrs}` that come out during
        the coversion:

        .. math::

            h_{prrs} = \int \frac{\chi^*_{p}(x_1) \chi^*_{r}(x_2) \chi_{r}(x_1) \chi_{s}(x_2)}{|x_1 - x_2|} dx_1 dx_2,

        where both :math:`\chi_{r}(x_1)` and :math:`\chi_{r}(x_2)` will have same spin functions, i.e.,
        :math:`\chi_{r}(x_i) = \phi(r_i)\alpha(\omega)` or :math:`\chi_{r}(x_i) = \phi(r_i)\beta(\omega)`\ . These are added to the one-electron
        integral tensor :math:`h_{pq}` to compute :math:`T_{pq}`\ .

    """

    chemist_two_body_coeffs, chemist_one_body_coeffs = None, None

    if one_body_tensor is not None:
        chemist_one_body_coeffs = one_body_tensor.copy()

    if two_body_tensor is not None:
        chemist_two_body_coeffs = np.swapaxes(two_body_tensor, 1, 3)
        # pylint:disable=invalid-unary-operand-type
        one_body_coeffs = -np.einsum("prrs", chemist_two_body_coeffs)

        if chemist_one_body_coeffs is None:
            chemist_one_body_coeffs = np.zeros_like(one_body_coeffs)

        if spatial_basis:
            chemist_two_body_coeffs = 0.5 * chemist_two_body_coeffs
            one_body_coeffs = 0.5 * one_body_coeffs

        chemist_one_body_coeffs += one_body_coeffs

    return (x for x in [chemist_one_body_coeffs, chemist_two_body_coeffs] if x is not None)


def symmetry_shift(core, one_electron, two_electron, n_elec, method="L-BFGS-B"):
    r"""Performs a `block-invariant symmetry shift <https://arxiv.org/pdf/2304.13772>`_ (BLISS) on the Hamiltonian.

    This decreases the 1-norm of the one-body :math:`T_{pq}` and two-body
    components :math:`V_{pqrs}` of the given Hamiltonian :math:`\hat{H}` and also
    its spectral range, based on the number of electron operator :math:`\hat{N}_e`.
    The shifted Hamiltonian (:math:`\hat{H}^{\prime}`) will be given by:

    .. math::

        H^{\prime}(k_1, k_2, \vec{\xi}) = \hat{H} - k_1 (\hat{N}_e - N_e) - k_2 (\hat{N}_e^2 - \hat{N}_e^2) + \sum_{ij}\xi_{ij} T_{ij} (\hat{N}_e - N_e),

    where :math:`N_e` is the target number of electrons and :math:`\vec{\xi}` is
    a real vector with symmetric elements.

    Args:
        one_electron (array[float]): a one-electron integral tensor giving the :math:`T_{pq}`.
        two_electron (array[float]): a two-electron integral tensor in the chemist notation giving the :math:`V_{pqrs}`.
        n_elec (bool): target number of electrons for selecting the target eigenstates.
        method (str | callable): type of solver used by ``scipy.optimize.minimize``. Please refer to its
            `documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_
            for the list of all available solvers. Default solver is `"L-BFGS-B"`

    Returns:
        tuple(array[float], array[float], array[float]): symmetry shifted core, one-body tensor and two-body tensor for the provided terms.

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = qml.numpy.array([[0.0, 0.0, 0.0], [1.398397361, 0.0, 0.0]], requires_grad=False)
    >>> mol = qml.qchem.Molecule(symbols, geometry, basis_name="STO-3G")
    >>> core, one, two = qml.qchem.electron_integrals(mol)()
    >>> ctwo = np.swapaxes(two, 1, 3)
    >>> shifted_core, shifted_one, shifted_two = symmetry_shift(core, one, ctwo, n_elec=mol.n_electrons)
    >>> print(shifted_two)
    [[[[ 1.12461110e-02 -1.70030746e-09]
      [-1.70030746e-09 -1.12461660e-02]]
     [[-1.70030746e-09  1.81210462e-01]
      [ 1.81210462e-01 -1.70032620e-09]]]
     [[[-1.70030763e-09  1.81210462e-01]
      [ 1.81210462e-01 -1.70032598e-09]]
     [[-1.12461660e-02 -1.70032620e-09]
      [-1.70032620e-09  1.12461854e-02]]]]
    """
    norb = one_electron.shape[0]
    ki_vec = np.array([0.0, 0.0])
    xi_mat = np.zeros((norb, norb))
    xi_idx = np.tril_indices_from(xi_mat)
    xi_vec = xi_mat[xi_idx]

    params = np.hstack((ki_vec, xi_vec))
    cost_func = partial(
        _symmetry_shift_two_body_loss, two=two_electron, xi_idx=xi_idx
    )  # Step 1: Reduce norm of two-body term

    res_two = sp.optimize.minimize(cost_func, params, method=method)
    _, k2, xi, _, N2, F_ = _symmetry_shift_terms(res_two.x, xi_idx, norb)
    new_two = two_electron - k2 * N2 - F_ / 4

    params = np.hstack(([0.0, k2], xi[xi_idx]))
    cost_func = partial(
        _symmetry_shift_one_body_loss, one=one_electron, n_elec=n_elec, xi_idx=xi_idx
    )  # Step 2: Reduce norm of one-body term

    res_one = sp.optimize.minimize(cost_func, params, method=method)
    k1, _, _, N1, _, _ = _symmetry_shift_terms(res_one.x, xi_idx, norb)
    new_core = core + k1 * n_elec + k2 * n_elec**2
    new_one = one_electron - k1 * N1 + n_elec * xi / 2

    return new_core, new_one, new_two


def _symmetry_shift_terms(params, xi_idx, norb):
    """Computes the terms for symmetry shift"""
    (k1, k2), xi_vec = params[:2], params[2:]
    if not xi_vec.size:  # pragma: no cover
        xi_vec = np.zeros_like(xi_idx[0])
    xi = np.zeros((norb, norb))
    xi[xi_idx], xi[xi_idx[::-1]] = xi_vec, xi_vec

    N1 = np.eye(norb)
    N2 = np.einsum("pq,rs->pqrs", N1, N1)
    T_ = np.einsum("pq,rs->pqrs", xi, N1)
    F_ = T_ + np.transpose(T_, (2, 3, 0, 1))

    return k1, k2, xi, N1, N2, F_


def _symmetry_shift_two_body_loss(params, two, xi_idx):
    """Two body loss term for symmetry shift"""
    _, k2, _, _, N2, F_ = _symmetry_shift_terms(params, xi_idx, two.shape[0])
    new_two = two - k2 * N2 - F_ / 4
    return np.linalg.norm(new_two)


def _symmetry_shift_one_body_loss(params, one, n_elec, xi_idx):
    """One body loss term for symmetry shift"""
    k1, _, xi, N1, _, _ = _symmetry_shift_terms(params, xi_idx, one.shape[0])
    new_one = one - k1 * N1 + n_elec * xi / 2
    return np.linalg.norm(new_one)
