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

has_jax_optax = True
try:  # pragma: no cover

    import optax
    from jax import jit
    from jax import numpy as jnp
    from jax import scipy as jsp
    from jax import value_and_grad
except (ModuleNotFoundError, ImportError) as e:  # pragma: no cover
    has_jax_optax = False

# pylint: disable=too-many-arguments, too-many-positional-arguments


def factorize(
    two_electron,
    tol_factor=1.0e-5,
    tol_eigval=1.0e-5,
    cholesky=False,
    compressed=False,
    regularization=None,
    **compression_kwargs,
):
    r"""Return the double-factorized form of a two-electron integral tensor in spatial basis.

    The two-electron tensor :math:`V`, in the
    `chemist notation <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`_,
    can be decomposed in terms of orthonormal matrices :math:`U` (leaf tensors) and
    symmetric matrices :math:`Z` (core tensors) such that
    :math:`V_{ijkl} \approx \sum_r^R \sum_{pq} U_{ip}^{(r)} U_{jp}^{(r)} Z_{pq}^{(r)} U_{kq}^{(r)} U_{lq}^{(r)}`,
    where the rank :math:`R` is determined by a threshold error.

    For explicit double factorization, i.e., when ``compressed=False``, the above decomposition
    is done using an eigenvalue or Cholesky decomposition to obtain symmetric matrices
    :math:`L^{(r)}` such that :math:`V_{ijkl} = \sum_r^R L_{ij}^{(r)} L_{kl}^{(r) T}`,
    where core and leaf tensors are obtained by further diagonalizing each matrix :math:`L^{(r)}`
    and truncating its eigenvalues (and the corresponding eigenvectors) at a threshold error.
    See theory section for more details.

    For compressed double factorization (CDF), i.e., when ``compressed=True``, the above
    decomposition is done by optimizing the following cost function :math:`\mathcal{L}`
    in a greedy layered-wise manner:

    .. math::

       \mathcal{L}(U, Z) = \frac{1}{2} \bigg|V_{ijkl} - \sum_r^R \sum_{pq} U_{ip}^{(r)} U_{jp}^{(r)} Z_{pq}^{(r)} U_{kq}^{(r)} U_{lq}^{(r)}\bigg|_{\text{F}} + \rho \sum_r^R \sum_{pq} \bigg|Z_{pq}^{(r)}\bigg|^{\gamma},

    where leaf tensors :math:`U` are defined by the antisymmetric orbital rotations :math:`X` such
    that :math:`U^{(r)} = \exp{(X^{(r)})}`, :math:`|\cdot|_{\text{F}}` computes the Frobenius norm,
    :math:`\rho` is a constant scaling factor, and :math:`|\cdot|^\gamma` specifies the optional L1
    and L2 regularization. See references `arXiv:2104.08957 <https://arxiv.org/abs/2104.08957>`__
    and `arxiv:2212.07957 <https://arxiv.org/pdf/2212.07957>`__ for more details.

    .. note::

        Packages JAX and Optax are required when performing CDF with ``compressed=True``.
        Install them using ``pip install jax~=0.6.0 optax``.

    Args:
        two_electron (array[array[float]]): Two-electron integral tensor in the molecular orbital
            basis arranged in chemist notation.
        tol_factor (float): Threshold error value for discarding the negligible factors.
            This will be used only when ``compressed=False``.
        tol_eigval (float): Threshold error value for discarding the negligible factor eigenvalues.
            This will be used only when ``compressed=False``.
        cholesky (bool): Use Cholesky decomposition for obtaining the symmetric matrices
            :math:`L^{(r)}` instead of eigendecomposition. Default is ``False``.
        compressed (bool): Use compressed double factorization to optimize the factors returned
            in the decomposition. Look at the keyword arguments (``compression_kwargs``) for
            the available options which must be provided only when ``compressed=True``.
        regularization (string | None): Type of regularization (``"L1"`` or ``"L2"``) to be
            used for optimizing the factors. Default is to not include any regularization term.

    Keyword Args:
        num_factors (int): Maximum number of factors that should be optimized for compressed
            double factorization. Default is :math:`2\times N`, where `N` is the number of
            dimensions of two-electron tensor.
        num_steps (int): Maximum number of epochs for optimizing each factor. Default is ``1000``.
        optimizer (optax.optimizer): An optax optimizer instance. If not provided, `Adam
            <https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adam>`_ is
            used with ``0.001`` learning rate.
        init_params (dict[str, TensorLike] | None): Intial values of the orbital rotations
            (:math:`X`) and core tensors (:math:`Z`) of shape ``(num_factors, N, N)`` given as
            a dictionary with keys ``"X"`` and ``"Z"``, where `N` is the number of dimension of
            two-electron tensor. If not given, zero matrices will be used if ``cholesky=False``
            and the core and leaf tensors corresponding to the first ``num_factors`` will be
            used if ``cholesky=True``.
        norm_prefactor (float): Prefactor for scaling the regularization term. Default is ``1e-5``.

    Returns:
        tuple(TensorLike, TensorLike, TensorLike): Tuple containing symmetric matrices (factors)
        approximating the two-electron integral tensor and core tensors and leaf tensors of
        the generated factors. In the explicit case where the core and leaf tensors could be
        truncated, they will be returned as a list.

    Raises:
        ValueError: If the specified regularization type is not supported.
        ImportError: If the specified packages are not installed when ``compressed=True``.

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0],
    ...                      [1.398397361, 0.0, 0.0]], requires_grad=False)
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> core, one, two = qml.qchem.electron_integrals(mol)()
    >>> two = np.swapaxes(two, 1, 3) # convert to chemist notation
    >>> factors, cores, leaves = qml.qchem.factorize(two, 1e-5, 1e-5)
    >>> print(factors)
    [[[-1.06723440e-01  6.42958741e-15]
      [ 7.71977824e-15  1.04898533e-01]]
     [[ 1.71099288e-13 -4.25688222e-01]
      [-4.25688222e-01  2.31561666e-13]]
     [[-8.14472856e-01 -3.89054708e-13]
      [-3.88994463e-13 -8.28642140e-01]]]

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

        with the rank :math:`R \leq n^2` where :math:`n` is the number of molecular orbitals.
        The matrices :math:`L^{(r)}` are diagonalized and for each matrix the eigenvalues that
        are smaller than a given threshold (and their corresponding eigenvectors) are discarded.
        These can be used to further decompose :math:`V_{ijkl}` in terms of orthonormal matrices
        :math:`U` (leaf tensors) and symmetric matrices :math:`Z` (core tensors), such that

        .. math::

            V_{ijkl} = \sum_r^R \sum_{pq} U_{ip}^{(r)} U_{jp}^{(r)} Z_{pq}^{(r)} U_{kq}^{(r)} U_{lq}^{(r)},

        where :math:`U^{(r)}` are the eigenvectors of :math:`L^{(r)}` and
        :math:`Z^{(r)}` are the outer product of the eigenvalues of :math:`L^{(r)}`.

        The factorization algorithm has the following steps
        [`arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`_]:

        - Reshape the :math:`n \times n \times n \times n` two-electron tensor to a
          :math:`n^2 \times n^2` matrix where :math:`n` is the number of orbitals.

        - Decompose the resulting matrix either via eigendecomposition or
          Cholesky decomposition.

        - For the eigendecomposition, keep the :math:`r` eigenvectors with
          corresponding eigenvalues larger than the threshold. Multiply these
          eigenvectors by the square root of the eigenvalues and reshape them
          to :math:`r \times n \times n` matrices to obtain :math:`L^{(r)}`.

        - Whereas for the Cholesky decomposition, keep the first :math:`r` Cholesky
          vectors that result in an residual error below the threshold and reshape
          them to :math:`r \times n \times n` matrices to obtain :math:`L^{(r)}`.

        - Diagonalize the :math:`L^{(r)}` (:math:`n \times n`) matrices and for
          each matrix keep the eigenvalues (and their corresponding eigenvectors)
          that are larger than a threshold.

        - Compute the orthonormal matrices :math:`U` and the symmetric matrices :math:`Z`
          from the retained eigenvalues and eigenvectors to get the core and leaf tensors.
    """
    shape = qml.math.shape(two_electron)

    if len(shape) != 4 or len(set(shape)) != 1:
        raise ValueError("The two-electron repulsion tensor must have a (N x N x N x N) shape.")

    two = qml.math.reshape(two_electron, (shape[0] * shape[1], -1))
    interface = qml.math.get_interface(two_electron)

    if not compressed:
        _explicit_df_func = (
            _double_factorization_cholesky if cholesky else _double_factorization_eigen
        )
        factors, f_eigvals, f_eigvecs = _explicit_df_func(two, tol_factor, shape, interface)

        # compute the core tensors and leaf tensors from the factors' eigendecomposition
        core_tensors, leaf_tensors = [], []
        for f_eigval, f_eigvec in zip(f_eigvals, f_eigvecs, strict=True):
            fidx = qml.math.where(qml.math.abs(f_eigval) > tol_eigval)[0]
            core_tensors.append(qml.math.einsum("i,j->ij", f_eigval[fidx], f_eigval[fidx]))
            leaf_tensors.append(f_eigvec[:, fidx])

        if np.sum([len(v) for v in core_tensors]) == 0:
            raise ValueError(
                "All eigenvectors are discarded. Consider decreasing the tol_eigval threshold."
            )

    else:
        if not has_jax_optax:
            raise ImportError(
                "Jax and Optax libraries are required for optimizing the factors. Install them via "
                "pip install jax~=0.6.0 optax"
            )  # pragma: no cover

        norm_order = {None: None, "L1": 1, "L2": 2}.get(regularization, "LX")
        if norm_order == "LX":
            raise ValueError(
                f"Supported regularization types include 'L1' and 'L2', got {regularization}."
            )
        optimizer = compression_kwargs.get("optimizer", optax.adam(learning_rate=0.001))
        num_steps = compression_kwargs.get("num_steps", 1000)
        num_factors = compression_kwargs.get("num_factors", 2 * shape[0])
        prefactor = compression_kwargs.get("norm_prefactor", 1e-5)
        init_params = compression_kwargs.get("init_params", None)

        if cholesky and init_params is None:
            # compute the factors via cholesky decomposition routine
            factors, f_eigvals, f_eigvecs = _double_factorization_cholesky(
                two, tol_factor, shape, interface, num_factors
            )
            # compute the core and orbital rotation tensors from the factors
            core_matrices = qml.math.einsum("ti,tj->tij", f_eigvals, f_eigvals)
            asym_matrices = [sp.linalg.logm(f_eigvec).real for f_eigvec in f_eigvecs]
            init_params = {"X": asym_matrices, "Z": core_matrices}
            num_factors = qml.math.shape(core_matrices)[0]

        if init_params is not None:
            init_params = {
                "X": qml.math.array(init_params["X"], like="jax"),
                "Z": qml.math.array(init_params["Z"], like="jax"),
            }

        core_tensors, leaf_tensors = _double_factorization_compressed(
            two_electron, optimizer, num_factors, num_steps, init_params, prefactor, norm_order
        )

        # Since core_tensors are symmetric but not constrained to be rank-one,
        # factors are computed by using their Schur decompositions.
        upr_tri, unitary = jsp.linalg.schur(core_tensors)
        factors = qml.math.array(
            jnp.einsum(
                "tpk,tqk,tki->tpqi",
                leaf_tensors,
                leaf_tensors,
                unitary @ jnp.sqrt(upr_tri.astype(jnp.complex64)),
            ),  # einsum contraction for them: "tpqi, trsi -> pqrs"
            like=interface,
        )
        core_tensors = qml.math.array(core_tensors, like=interface)
        leaf_tensors = qml.math.array(leaf_tensors, like=interface)

    return factors, core_tensors, leaf_tensors


def _double_factorization_eigen(two, tol_factor=1.0e-10, shape=None, interface=None):
    """Explicit double factorization using eigen decomposition
    of the two-electron integral tensor described in
    `npj Quantum Information 7, 83 (2021) <https://doi.org/10.1038/s41534-021-00416-z>`_.

    Args:
        two (array[array[float]]): two-electron integral tensor in the molecular orbital
            basis arranged in chemist notation
        tol_factor (float): threshold error value for discarding the negligible factors
        shape (tuple[int, int]): shape for the provided two_electron
        interface (string): interface for two_electron tensor
        num_factors (int): number of factors to be computed.

    Returns:
        tuple(array[array[float]], array[array[float]], array[array[float]]): tuple containing
        symmetric matrices (factors) approximating the two-electron integral tensor, truncated
        eigenvalues of the generated factors, and truncated eigenvectors of the generated factors
    """
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
    feigvals, feigvecs = qml.math.linalg.eigh(factors)
    return factors, feigvals, feigvecs


def _double_factorization_cholesky(
    two, tol_factor=1.0e-10, shape=None, interface=None, num_factors=None
):
    """Explicit double factorization using Cholesky decomposition
    of the two-electron integral tensor described in
    `J. Chem. Phys. 118, 9481-9484 (2003) <https://doi.org/10.1063/1.1578621>`_.

    Args:
        two (array[array[float]]): two-electron integral tensor in the molecular orbital
            basis arranged in chemist notation
        tol_factor (float): threshold error value for discarding the negligible factors
        shape (tuple[int, int]): shape for the provided two_electron
        interface (string): interface for two_electron tensor
        num_factors (int): number of factors to be computed.

    Returns:
        tuple(array[array[float]], array[array[float]], array[array[float]]): tuple containing
        symmetric matrices (factors) approximating the two-electron integral tensor, truncated
        eigenvalues of the generated factors, and truncated eigenvectors of the generated factors
    """
    n2 = shape[0] * shape[1]
    if num_factors is None:
        num_factors = n2

    cholesky_vecs = qml.math.zeros((n2, num_factors), like=interface)
    cholesky_diag = qml.math.array(qml.math.diagonal(two).real, like=interface)

    for idx in range(num_factors):
        if (max_err := qml.math.max(cholesky_diag)) < tol_factor:
            cholesky_vecs = cholesky_vecs[:, :idx]
            break

        max_idx = qml.math.argmax(cholesky_diag)
        cholesky_mat = cholesky_vecs[:, :idx]
        cholesky_vec = (
            two[:, max_idx] - cholesky_mat @ qml.math.conjugate(cholesky_mat[max_idx])
        ) / qml.math.sqrt(max_err)

        cholesky_vecs[:, idx] = cholesky_vec
        cholesky_diag -= qml.math.abs(cholesky_vec) ** 2

    factors = cholesky_vecs.T.reshape(-1, shape[0], shape[0])
    feigvals, feigvecs = qml.math.linalg.eigh(factors)
    return factors, feigvals, feigvecs


def _double_factorization_compressed(
    two, optimizer, num_factors, num_steps=1000, init_params=None, prefactor=1e-5, norm_order=None
):
    r"""Compressed double factorization with optional regularization based on
    `arXiv:2104.08957 <https://arxiv.org/abs/2104.08957>`__ and
    `arxiv:2212.07957 <https://arxiv.org/pdf/2212.07957>`__.

    Here we decompose the two-electron tensor :math:`V` in terms of ``R=num_factors`` orthonormal
    matrices :math:`U` (leaf tensors) and symmetric matrices :math:`Z` (core tensors) such that:

    .. math::

        V_{ijkl} \approx \sum_r^R \sum_{pq} U_{ip}^{(r)} U_{jp}^{(r)} Z_{pq}^{(r)} U_{kq}^{(r)} U_{lq}^{(r)}.

    This is done by optimizing the following cost function :math:`\mathcal{L}` via a greedy
    approach, i.e., we optimize the leaf-tensor pairs layer-by-layer (or one-by-one) instead
    of optimizing them all at once as the latter gives unfavorable performance (and scaling):

    .. math::

       \mathcal{L}(U, Z) = \frac{1}{2} \bigg|V_{ijkl} - \sum_r^R \sum_{pq} U_{ip}^{(r)} U_{jp}^{(r)} Z_{pq}^{(r)} U_{kq}^{(r)} U_{lq}^{(r)}\bigg|_{\text{F}} + \rho \sum_r^R \sum_{pq} \bigg|Z_{pq}^{(r)}\bigg|^{\gamma}.

    First, leaf tensors :math:`U^{(r)} = \exp{(X^{(r)})}` are parameterized by the
    anti-symmetric orbital rotations :math:`X^{(r)}` to compute the Frobenius norm
    of the difference between the :math:`V` and the above approximation. A further
    regularization term penalizing large terms in :math:`|Z^{(r)}|` is added to
    this after scaling it with a constant factor :math:`\rho` to compute the loss.

    Args:
        two (array[array[float]]): Two-electron integral tensor in the molecular orbital
            basis arranged in chemist notation.
        optimizer (optax.optimizer): An optax optimizer instance. If not provided, `Adam
            <https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adam>`_ is
            used with ``0.001`` learning rate.
        num_factors (int): Maximum number of factors that should be optimized for compressed
            double factorization. Default is :math:`2\times N`, where `N` is the number of
            dimensions of two-electron tensor.
        num_steps (int): Maximum number of epochs for optimizing each factor. Default is ``1000``.
        init_params (dict[str, TensorLike] | None): Intial values of the orbital rotations
            (:math:`X`) and core tensors (:math:`Z`) of shape ``(num_factors, N, N)`` given as
            a dictionary with keys ``"X"`` and ``"Z"``, where `N` is the number of dimension of
            two-electron tensor. If not given, zero matrices will be used.
        prefactor (float): Prefactor for scaling the regularization term. Default is ``1e-5``.
        norm_order (int): Type of regularization (``0``: None, ``1``: L1, and ``2``: L2) used
            for optimizing. Default is to not include any regularization term.

    Returns:
        tuple(TensorLike, TensorLike): Tuple containing core tensors and leaf tensors
        approximating the two-electron integral tensor.
    """
    norb = two.shape[0]
    asym_tensors, core_tensors = jnp.zeros((2, 0, norb, norb))

    cost_func = value_and_grad(
        partial(_compressed_cost_fn, two=two, norm_order=norm_order, prefactor=prefactor)
    )

    @jit
    def _step(params, opt_state, asym_tensors, core_tensors):
        """An optimization step for computing the loss and updating the parameters"""
        cost, grads = cost_func(params, asym_tensors=asym_tensors, core_tensors=core_tensors)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, cost

    for fidx in range(num_factors):
        params = (
            {"X": jnp.zeros((1, norb, norb)), "Z": jnp.zeros((1, norb, norb))}
            if init_params is None
            else {"X": init_params["X"][fidx][None, :], "Z": init_params["X"][fidx][None, :]}
        )
        opt_state = optimizer.init(params)

        for _ in range(num_steps):
            params, opt_state, _ = _step(params, opt_state, asym_tensors, core_tensors)

        Xs, Zs = params["X"], params["Z"]
        asym_tensors = jnp.concatenate(
            (asym_tensors, 0.5 * (Xs - jnp.transpose(Xs, (0, 2, 1)))), axis=0
        )
        core_tensors = jnp.concatenate(
            (core_tensors, 0.5 * (Zs + jnp.transpose(Zs, (0, 2, 1)))), axis=0
        )

    leaf_tensors = jsp.linalg.expm(asym_tensors)
    return core_tensors, leaf_tensors


def _compressed_cost_fn(params, two, asym_tensors, core_tensors, norm_order, prefactor):
    """Loss function for the compressed double factorization.

    The loss is computed based on evaluating Frobenius norm of the two-body tensor
    approximated from leaf and core tensors against the original two-body tensor
    and optional evaluation of regularization of the core tensors.
    """
    Xs, Zs = params["X"], params["Z"]

    Xs = jnp.concatenate((asym_tensors, Xs), axis=0)
    Xs = 0.5 * (Xs - jnp.transpose(Xs, (0, 2, 1)))
    Us = jsp.linalg.expm(Xs)

    Zs = jnp.concatenate((core_tensors, Zs), axis=0)
    Zs = 0.5 * (Zs + jnp.transpose(Zs, (0, 2, 1)))

    cdf_two = jnp.einsum("tpk,tqk,tkl,trl,tsl->pqrs", Us, Us, Zs, Us, Us)

    cost = jnp.linalg.norm(cdf_two - two) ** 2
    if norm_order is not None:
        cost += prefactor * jnp.linalg.norm(Zs, ord=norm_order, axis=(1, 2))[0]

    return cost


def basis_rotation(one_electron, two_electron, tol_factor=1.0e-5, **factorization_kwargs):
    r"""Return the grouped coefficients and observables of a molecular Hamiltonian and the basis
    rotation unitaries obtained with the basis rotation grouping method.

    Args:
        one_electron (array[float]): One-electron integral matrix in the molecular orbital basis.
        two_electron (array[array[float]]): Two-electron integral tensor in the molecular orbital
            basis arranged in chemist notation.
        tol_factor (float): Threshold error value for discarding the negligible factors.

    Keyword Args:
        tol_eigval (float): Threshold error value for discarding the negligible factor
            eigenvalues. This can be used only when ``compressed=False``.
        cholesky (bool): Use Cholesky decomposition for the ``two_electron`` instead of
            eigendecomposition. Default is ``False``.
        compressed (bool): Use compressed double factorization for decomposing the ``two_electron``.
        regularization (string | None): Type of regularization (``"L1"`` or ``"L2"``) to be
            used for optimizing the factors. Default is to not include any regularization term.
        **compression_kwargs: Look at the keyword arguments (``compression_kwargs``) in the
            :func:`~.factorize` method for all the available options with ``compressed=True``.

    Returns:
        tuple(list[array[float]], list[list[Operator]], list[array[float]]): Tuple containing
        grouped coefficients, grouped observables and basis rotation transformation matrices.

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0],
    ...                      [1.398397361, 0.0, 0.0]], requires_grad=False)
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> core, one, two = qml.qchem.electron_integrals(mol)()
    >>> coeffs, ops, unitaries = basis_rotation(one, two, tol_factor=1.0e-5)
    >>> print(coeffs)
    [array([-2.59579282,  0.84064649,  0.84064649,  0.45724992,  0.45724992]),
     array([ 5.60006390e-03, -9.73801723e-05, -9.73801723e-05,  2.84747318e-03,
             9.57150297e-05, -2.79878310e-03,  9.57150297e-05, -2.79878310e-03,
            -2.79878310e-03, -2.79878310e-03,  2.75092558e-03]),
     array([ 0.09060523,  0.04530262, -0.04530262, -0.04530262, -0.04530262,
            -0.04530262,  0.04530262]),
     array([ 1.6874169 , -0.68077716, -0.68077716,  0.17166195, -0.66913628,
             0.16872663, -0.66913628,  0.16872663,  0.16872663,  0.16872663,
             0.16584151])]

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

    factorization_kwargs["tol_factor"] = tol_factor
    factors, core_tensors, leaf_tensors = factorize(chemist_two_body_tensor, **factorization_kwargs)
    v_unitaries = [
        np.kron(leaf_tensor, np.eye(2)) for leaf_tensor in leaf_tensors
    ]  # account for spin

    ops_t = 0.0
    for p in range(num_orbitals):
        ops_t += 0.5 * t_eigvals[p] * (qml.Identity(p) - qml.Z(p))

    ops_l = []
    for idx in range(len(factors)):
        ops_l_ = 0.0
        for p in range(num_orbitals):
            for q in range(num_orbitals):
                ops_l_ += (
                    core_tensors[idx][p // 2, q // 2]
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
        c_group.append(np.array(c_g))
        o_group.append(o_g)

    u_transform = list([t_eigvecs] + list(v_unitaries))  # Inverse of diagonalizing unitaries

    return c_group, o_group, u_transform


def _chemist_transform(one_body_tensor=None, two_body_tensor=None, spatial_basis=True):
    r"""Transforms one- and two-body terms in physicists' notation to `chemists' notation <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`_\ .

    This converts the input two-body tensor :math:`h_{pqrs}` that constructs :math:`\sum_{pqrs} h_{pqrs} a^\dagger_p a^\dagger_q a_r a_s`
    to a transformed two-body tensor :math:`V_{pqrs}` that follows the chemists' convention to construct :math:`\sum_{pqrs} V_{pqrs} a^\dagger_p a_q a^\dagger_r a_s`
    in the spatial basis. During the transformation, some extra one-body terms come out. These are returned as a one-body tensor :math:`T_{pq}` in the
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
        one_body_coeffs = -np.einsum("prrs", chemist_two_body_coeffs)

        if chemist_one_body_coeffs is None:
            chemist_one_body_coeffs = np.zeros_like(one_body_coeffs)

        if spatial_basis:
            chemist_two_body_coeffs = 0.5 * chemist_two_body_coeffs
            one_body_coeffs = 0.5 * one_body_coeffs

        chemist_one_body_coeffs += one_body_coeffs

    return (x for x in [chemist_one_body_coeffs, chemist_two_body_coeffs] if x is not None)


def symmetry_shift(core, one_electron, two_electron, n_elec, method="L-BFGS-B", **method_kwargs):
    r"""Performs a block-invariant symmetry shift on the electronic integrals.

    The block-invariant symmetry shift (BLISS) method [`arXiv:2304.13772
    <https://arxiv.org/pdf/2304.13772>`_] decreases the one-norm and the
    spectral range of a molecular Hamiltonian :math:`\hat{H}` defined by
    its one-body :math:`T_{pq}` and two-body components. It constructs
    a shifted Hamiltonian (:math:`\hat{H}^{\prime}`), such that:

    .. math::

        H^{\prime}(k_1, k_2, \vec{\xi}) = \hat{H} - k_1 (\hat{N}_e - N_e) - k_2 (\hat{N}_e^2 - \hat{N}_e^2) + \sum_{ij}\xi_{ij} T_{ij} (\hat{N}_e - N_e),

    where :math:`\hat{N}_e` is the electron number operator, :math:`N_e` is the
    number of electrons of the molecule and :math:`k_u, \xi_{ij} \in \mathbb{R}` are
    the parameters that are optimized with the constraint :math:`\xi_{ij} = \xi_{ji}`
    to minimize the overall one-norm of the :math:`\hat{H}^{\prime}`.

    Args:
        core (array[float]): the contribution of the core orbitals and nuclei
        one_electron (array[float]): a one-electron integral tensor
        two_electron (array[float]): a two-electron integral tensor in the chemist notation
        n_elec (bool): number of electrons in the molecule
        method (str | callable): solver method used by ``scipy.optimize.minimize``
            to optimize the parameters. Please refer to its `documentation
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_
            for the list of all available solvers. Default solver is ``"L-BFGS-B"``.
        **method_kwargs: keyword arguments to pass when calling ``scipy.optimize.minimize`` with ``method=method``

    Returns:
        tuple(array[float], array[float], array[float]): symmetry shifted core, one-body tensor and two-body tensor for the provided terms

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = qml.numpy.array([[0.0, 0.0, 0.0],
    ...                             [1.398397361, 0.0, 0.0]], requires_grad=False)
    >>> mol = qml.qchem.Molecule(symbols, geometry, basis_name="STO-3G")
    >>> core, one, two = qml.qchem.electron_integrals(mol)()
    >>> ctwo = np.swapaxes(two, 1, 3)
    >>> s_core, s_one, s_two = symmetry_shift(core, one, ctwo, n_elec=mol.n_electrons)
    >>> print(s_two)
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

    res_two = sp.optimize.minimize(cost_func, params, method=method, **method_kwargs)
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
    """Computes the terms required for performing symmetry shift
    (Eq. 8-9, `arXiv:2304.13772 <https://arxiv.org/abs/2304.13772>`_)
    from the flattened solution parameter array obtained from scipy optimizer's result."""
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
    """Two body loss term for symmetry shift."""
    _, k2, _, _, N2, F_ = _symmetry_shift_terms(params, xi_idx, two.shape[0])
    new_two = two - k2 * N2 - F_ / 4
    return np.linalg.norm(new_two)


def _symmetry_shift_one_body_loss(params, one, n_elec, xi_idx):
    """One body loss term for symmetry shift."""
    k1, _, xi, N1, _, _ = _symmetry_shift_terms(params, xi_idx, one.shape[0])
    new_one = one - k1 * N1 + n_elec * xi / 2
    return np.linalg.norm(new_one)
