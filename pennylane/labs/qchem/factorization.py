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
