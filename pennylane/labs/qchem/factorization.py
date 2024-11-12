# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functionality for symmetry shift and compressed double factorization."""

from functools import partial

import numpy as np
import scipy as sp

import pennylane as qml


def factorize(two_electron, tol_factor=1.0e-5, tol_eigval=1.0e-5, cholesky=False):
    r"""Return the double-factorized form of a two-electron integral tensor in spatial basis"""
    shape = qml.math.shape(two_electron)

    if len(shape) != 4 or len(set(shape)) != 1:
        raise ValueError("The two-electron repulsion tensor must have a (N x N x N x N) shape.")

    n, n2 = shape[0], shape[0] * shape[1]
    two = qml.math.reshape(two_electron, (n2, n2))
    interface = qml.math.get_interface(two_electron)

    if cholesky:
        cholesky_vecs = qml.math.zeros((n2, n2 + 1), like=interface)
        cholesky_diag = qml.math.array(qml.math.diagonal(two).real, like=interface)

        for idx in range(n2 + 1):
            if (max_err := qml.math.max(cholesky_diag)) < tol_factor:
                break

            max_idx = qml.math.argmax(cholesky_diag)
            cholesky_mat = cholesky_vecs[:, :idx]
            cholesky_vec = (
                two[:, max_idx] - cholesky_mat @ cholesky_mat[max_idx].conj()
            ) / qml.math.sqrt(max_err)

            cholesky_vecs[:, idx] = cholesky_vec
            cholesky_diag -= qml.math.abs(cholesky_vec) ** 2

        factors = cholesky_vecs.T.reshape(-1, n, n)

    else:
        eigvals_r, eigvecs_r = qml.math.linalg.eigh(two)
        eigvals_r = qml.math.array([val for val in eigvals_r if abs(val) > tol_factor])

        eigvecs_r = eigvecs_r[:, -len(eigvals_r) :]

        if eigvals_r.size == 0:
            raise ValueError(
                "All factors are discarded. Consider decreasing the first threshold error."
            )

        vectors = eigvecs_r @ qml.math.diag(qml.math.sqrt(eigvals_r))

        r = len(eigvals_r)
        factors = qml.math.array(
            [vectors.reshape(n, n, r)[:, :, k] for k in range(r)], like=interface
        )

    eigvals, eigvecs = qml.math.linalg.eigh(factors)
    eigvals = qml.math.asarray(eigvals, like=interface)

    eigvals_m, eigvecs_m = [], []
    for eidx, eigval in enumerate(eigvals):
        tidx = qml.math.where(qml.math.abs(eigval) > tol_eigval)[0]
        eigvals_m.append(eigval[tidx])
        eigvecs_m.append(eigvecs[eidx, tidx])

    if qml.math.sum([len(v) for v in eigvecs_m]) == 0:
        raise ValueError(
            "All eigenvectors are discarded. Consider decreasing the second threshold error."
        )

    return factors, eigvals_m, eigvecs_m


def symmetry_shift(core, one, two, n_elec=None, verbose=False):
    """Performs a symmetry-shift on the two-electron integral"""
    norb = one.shape[0]
    ki_vec = np.array([0.0, 0.0])
    xi_mat = np.random.RandomState(42).rand(norb, norb)
    xi_idx = np.tril_indices_from(xi_mat)
    xi_vec = xi_mat[xi_idx]

    prior_cost = np.linalg.norm(two) + np.linalg.norm(one, ord="fro")

    params = np.hstack((ki_vec, xi_vec))
    cost_func = partial(_symmetry_shift_loss, one=one, two=two, n_elec=n_elec, xi_idx=xi_idx)

    res = sp.optimize.minimize(cost_func, params, method="L-BFGS-B")
    if verbose:
        print(f"Converged? : {res.success}")
        print(f"Parameters : {res.x}")

    new_params = res.x
    k1, k2, xi, N1, N2, T_ = _symmetry_shift_terms(new_params, xi_idx, norb)

    new_core = core + k1 * n_elec + k2 * n_elec**2
    new_one = one - k1 * N1 + n_elec * xi / 2
    new_two = two - k2 * N2 - (T_ + np.transpose(T_, (2, 3, 0, 1))) / 4

    cost = np.linalg.norm(new_two) + np.linalg.norm(new_one, ord="fro")
    if verbose:
        print(f"Before BLISS: {prior_cost}")
        print(f"After BLISS:  {cost}")
        print(f"Reduction:    {(prior_cost-cost)/prior_cost}")

    return new_core, new_one, new_two


def _symmetry_shift_loss(params, one, two, n_elec, xi_idx):
    """Computes BLISS-based loss for the symmetry shift"""
    k1, k2, xi, N1, N2, T_ = _symmetry_shift_terms(params, xi_idx, one.shape[0])

    new_one = one - k1 * N1 + n_elec * xi / 2
    new_two = two - k2 * N2 - (T_ + np.transpose(T_, (2, 3, 0, 1))) / 4

    return compute_one_norm(new_one, new_two)


def _symmetry_shift_terms(params, xi_idx, norb):
    """Computes the terms for symmetry shift"""
    (k1, k2), xi_vec = params[:2], params[2:]
    if not xi_vec.size:
        xi_vec = np.zeros_like(xi_idx[0])
    xi = np.zeros((norb, norb))
    xi[xi_idx], xi[xi_idx[::-1]] = xi_vec, xi_vec

    N1 = np.eye(norb)
    N2 = np.einsum("pq,rs->pqrs", N1, N1)
    T_ = np.einsum("pq,rs->pqrs", xi, N1)

    return k1, k2, xi, N1, N2, T_


def compute_one_norm(one, two):
    """Computes 1-norm for the given 1-body and 2-body integral pair"""
    i, j, k, l = np.indices(two.shape)
    mask_ik_jl = ((i > k) & (j > l)).astype(int)

    term1 = np.sum(np.abs(one + 2 * np.einsum("ijkk->ij", two)))
    term2 = np.sum(np.abs((two - np.transpose(two, (0, 3, 2, 1))) * mask_ik_jl))
    term3 = 0.5 * np.sum(np.abs(two))

    return term1 + term2 + term3
